"""
DiscoGPTransformer: maskable transformer for sheaf discovery / pruning.

This module wraps a (HookedTransformer-compatible) architecture with two kinds of
learnable masks:
- **Weight masks**: continuous logits per-parameter that are sampled via a
  straight-through Gumbel-sigmoid to 0/1 during forward passes.
- **Edge masks**: continuous logits per edge/node whose samples gate residual
  contributions across the computational graph (heads/MLP/output).

It provides:
- Utilities to turn masks on/off (optionally deterministically or reversed),
- Sparsity/overlap losses for regularization,
- Evaluation helpers that compare masked vs. original logits,
- A simple pruning loop that optimizes mask logits w.r.t. faithfulness and
  completeness objectives.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast, overload
from typing_extensions import Literal

import os
import gc
from pathlib import Path

from tqdm.auto import tqdm
from pprint import pprint
import wandb

import numpy as np
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DistributedSampler

import einops
from fancy_einsum import einsum

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens.components import (
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    LayerNorm,
    LayerNormPre,
    Unembed,
    GatedMLP,
    MLP,
    MoE,
    Embed,
)
from transformer_lens import HookedTransformer

from .modules.transformer_block import DiscoGPTransformerBlock

from .data import setup_task
from .evaluation import (
    compute_complete_loss_binary_label,
    compute_faith_loss_binary_label,
    compute_complete_loss_multi_label,
    compute_faith_loss_multi_label,
)
from .utils import schedule_epoch_lambda
from .configs import Config

def gumbel_sigmoid(logits, gs_temp: float = 1.0, eps: float = 1e-10):
    """Sample a Bernoulli-like gate using a straight-through Gumbel-sigmoid.

    - Adds Gumbel noise to `logits`, divides by temperature `gs_temp`, and applies
      `sigmoid`.
    - Uses a straight-through estimator (round at 0.5 but keep gradients).
    """
    # Draw two uniforms to create a (differenced) Gumbel noise sample
    uniform = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()

    # Relaxed sample
    res = torch.sigmoid((logits + noise) / gs_temp)

    # Straight-through: hard threshold in forward, identity in backward
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res


def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def rank0_print(*args, **kwargs):
    """Print only from rank 0 (works even if dist not initialized)."""
    if is_main_process():
        print(*args, **kwargs)


class DDPDiscoGPTransformer(nn.Module):
    """Transformer with learnable weight and edge masks for sheaf (circuit) discovery."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token + positional embeddings (if not rotary)
        self.embed = Embed(self.cfg)
        if self.cfg.positional_embedding_type != "rotary":
            self.pos_embed = PosEmbed(self.cfg)

        # Stack of custom blocks that apply/propagate masks internally
        self.blocks = nn.ModuleList(
            [DiscoGPTransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)]
        )

        # Output edge mask logits cover all nodes (heads + mlp per layer + final output)
        total_nodes = (cfg.n_heads + 1) * cfg.n_layers + 1
        self.edge_mask_output_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((total_nodes,)), mean=self.cfg.logits_e_init, std=0.01),
            requires_grad=True,
        )

        # Final normalization choice
        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.cfg.final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)

        self.unembed = Unembed(self.cfg)

        if self.cfg.init_weights:
            self.init_weights()

        self.mask_logits_paramdict = None

    def setup_mask(self):
        """Initialize mask-related state and (optionally) mask logits parameters.

        Weight masks are registered only if `cfg.use_weight_masks` to save memory.
        Edge masks are taken from existing parameters with 'edge' in their name.
        """
        # Temperatures for Gumbel-sigmoid sampling
        self.gs_temp_weight = self.cfg.gs_temp_weight
        self.gs_temp_edge = self.cfg.gs_temp_edge

        # Containers for mask bookkeeping
        self.unmasked_params = {}
        self.mask_logits_dict_weight = {}
        self.mask_logits_dict_edge = {}
        self.use_weight_masks = self.cfg.use_weight_masks
        self.use_edge_masks = self.cfg.use_edge_masks
        self.reference_weight_mask = {}
        self.overlap_weight_or_circuit = {}
        self.mask_logits_weight_paramnames = []
        self.mask_name_to_key = {}

        # ---- Weight mask logits initialization ----
        # Register mask logits per-parameter (excluding emb/ln/edge) and freeze real weights.
        if self.use_weight_masks:
            self.N_weight = 0
            cnt = 0
            # use a ParameterDict so registration and ordering is explicit & safe
            self.mask_logits_paramdict = nn.ParameterDict()
            named_params_snapshot = list(self.named_parameters())
            for name, p in named_params_snapshot:
                # Skip embeddings/unembeddings, LayerNorms, and any edge-mask parameters
                if 'emb' not in name and 'edge' not in name and 'ln' not in name:
                    p.grad = None
                    p.requires_grad = False
                    self.unmasked_params[name] = p.clone()

                    # create mask logits Parameter with same shape
                    init_tensor = torch.normal(mean=self.cfg.logits_w_init, std=0.01, size=p.shape, device=self.cfg.device)
                    masks_logits = nn.Parameter(init_tensor, requires_grad=True)

                    # store into ParameterDict under the original parameter's name (or an alternate)
                    param_key = f"mask_logits_weight__{name}"
                    # ParameterDict requires keys be strings with valid characters; replace '.' with '__'
                    param_key_safe = param_key.replace('.', '__')
                    self.mask_logits_paramdict[param_key_safe] = masks_logits

                    # also keep the mapping from original weight name to mask key
                    # self.mask_logits_dict_weight[name] = self.mask_logits_paramdict[param_key_safe]
                    self.mask_name_to_key[name] = param_key_safe
                    self.mask_logits_weight_paramnames.append(param_key_safe)

                    cnt += 1
                    with torch.no_grad():
                        self.N_weight += p.numel()

            # convert N_weight to float
            self.N_weight = float(self.N_weight)

        # ---- Edge mask logits initialization ----
        # Collect parameters that are already defined as edge-mask logits.
        if self.use_edge_masks:
            self.N_edge = 0
            for name, p in self.named_parameters():
                if 'edge' in name:
                    self.mask_logits_dict_edge[name] = p
                    with torch.no_grad():
                        self.N_edge += torch.ones_like(p.view(-1)).sum().cpu()

        self.N_edge   = float(self.N_edge.item())   if torch.is_tensor(self.N_edge)   else float(self.N_edge)

    def forward(self, tokens, return_states: bool = False):
        """Forward pass with optional state return (pre-unembed).

        Input
        ---
        tokens: LongTensor [batch, position]
        return_states: if True, return the residual stream per-node before final ln+unembed.
        """
        if self.cfg.positional_embedding_type == "standard":
            # Standard absolute position embeddings
            embed = self.embed(tokens)
            pos_embed = self.pos_embed(tokens)
            residual = embed + pos_embed
        elif self.cfg.positional_embedding_type == "rotary":
            # Rotary embeddings applied inside attention layers
            residual = self.embed(tokens)
        else:
            raise ValueError(
                f"Invalid positional_embedding_type passed in {self.cfg.positional_embedding_type}"
            )

        # Add explicit prev_head_idx dimension for gated aggregation across nodes
        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")

        # Pass through transformer blocks; each block may expand the prev_head_idx dim
        for i, block in enumerate(self.blocks):
            residual = block(residual)

        if return_states:
            return residual

        # Sample or threshold the *output* edge mask that combines node streams
        if self.cfg.use_edge_masks:
            if self.cfg.use_deterministic_masks:
                sampled_output_mask = torch.where(self.edge_mask_output_logits > 0., 1., 0.).to(self.cfg.device, dtype=self.cfg.dtype)
            else:
                sampled_output_mask = gumbel_sigmoid(self.edge_mask_output_logits, gs_temp=self.cfg.gs_temp_edge)
            if self.cfg.use_reverse_masks:
                sampled_output_mask = 1. - sampled_output_mask
        else:
            sampled_output_mask = torch.ones(self.edge_mask_output_logits.shape).to(self.cfg.device, dtype=self.cfg.dtype)

        # Collapse the prev_head_idx dimension via a weighted sum by the sampled mask
        residual = einsum(
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model",
            residual,
            sampled_output_mask,
        )

        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(normalized_resid_final)
        return [logits]

    # --- Runtime mask toggles -------------------------------------------------

    def turn_on_edge_masks(self, deterministic: bool = False, reverse: bool = False):
        """Enable edge masks for this model and all blocks.

        - `deterministic=True` means use (logits>0) thresholding instead of sampling.
        - `reverse=True` flips 0/1 decisions to evaluate completeness.
        """
        self.cfg.use_edge_masks = True
        self.cfg.use_deterministic_masks = deterministic
        self.cfg.use_reverse_masks = reverse
        for block in self.blocks:
            block.cfg.use_edge_masks = True
            block.cfg.use_deterministic_masks = deterministic
            block.cfg.use_reverse_masks = reverse

    def turn_off_edge_masks(self):
        """Disable edge masks for this model and all blocks."""
        self.cfg.use_edge_masks = False
        for block in self.blocks:
            block.cfg.use_edge_masks = False

    def turn_on_weight_masks(self, deterministic: bool = False, reverse: bool = False):
        """Materialize masked weights in-place from logits using sampling/thresholding.

        Uses ParameterDict (self.mask_logits_paramdict) mapped to original weight names
        via self.mask_name_to_key for deterministic, registered handling.
        Caller should ensure RNG sync across ranks BEFORE calling this when using stochastic masks.
        """
        if not self.use_weight_masks:
            return

        # get current named parameters mapping for live params on this device
        named_params = dict(self.named_parameters())

        # iterate original param names -> paramdict key
        for orig_name, key in self.mask_name_to_key.items():
            # get the live parameter and the unmasked copy (on correct device)
            param = named_params.get(orig_name)
            unmasked_m = self.unmasked_params[orig_name].to(param.device, dtype=param.dtype)

            # get the registered mask logits (ParameterDict ensures device is correct after model.to())
            mask_logits = self.mask_logits_paramdict[key]
            # if needed ensure dtype/device (should be unnecessary after model.to())
            if mask_logits.device != param.device or mask_logits.dtype != param.dtype:
                mask_logits_local = mask_logits.to(device=param.device, dtype=param.dtype)
            else:
                mask_logits_local = mask_logits

            # sample mask: differentiable unless deterministic is True
            if deterministic:
                with torch.no_grad():
                    sampled_masks = torch.where(mask_logits_local > 0.0, 1.0, 0.0).to(param.dtype)
            else:
                sampled_masks = gumbel_sigmoid(mask_logits_local, gs_temp=self.gs_temp_weight)
                # ensure dtype alignment
                if sampled_masks.dtype != param.dtype:
                    sampled_masks = sampled_masks.to(param.dtype)

            if reverse:
                sampled_masks = 1.0 - sampled_masks

            # apply in-place to the parameter
            param.copy_(sampled_masks * unmasked_m)


    def turn_off_weight_masks(self):
        """Restore original (unmasked) weights in-place and detach."""
        if not self.use_weight_masks:
            return

        named_params = dict(self.named_parameters())

        for orig_name, key in self.mask_name_to_key.items():
            param = named_params.get(orig_name)
            unmasked_m = self.unmasked_params[orig_name].to(param.device, dtype=param.dtype)
            param.copy_(unmasked_m)
            param.detach_()

    # --- Mask-related losses --------------------------------------------------

    def weight_experiment_loss(self):
        """Penalty that pushes mask_logits > 0 to agree with a reference mask (if given)."""
        if not bool(self.reference_weight_mask):
            # return zero tensor on the right device/dtype
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        # accumulate as a differentiable tensor
        experiment_loss = None
        # iterate over the same masks as training uses
        if hasattr(self, "mask_logits_paramdict") and self.mask_logits_paramdict is not None:
            mask_iter = self.mask_logits_paramdict.items()
        else:
            mask_iter = ((k, getattr(self, k)) for k in self.mask_logits_weight_paramnames)

        for key, mask_logits in mask_iter:
            # reference_value should be same shape as mask_logits or broadcastable
            ref = self.reference_weight_mask.get(key, None)
            if ref is None:
                # If your reference map uses original param names, you may need to adapt keys
                continue
            # ensure tensor, to device and dtype
            if not torch.is_tensor(ref):
                ref_t = torch.as_tensor(ref, device=mask_logits.device, dtype=mask_logits.dtype)
            else:
                ref_t = ref.to(device=mask_logits.device, dtype=mask_logits.dtype)

            # compute per-element condition mask (ref==1) & (mask_logits > 0)
            # keep it differentiable only on mask_logits: we will sum mask_logits entries where condition holds
            cond = (ref_t == 1) & (mask_logits > 0.)
            # cast to same dtype and multiply by mask_logits (only mask_logits contributes grad)
            selected = mask_logits[cond]
            if selected.numel() == 0:
                part = torch.tensor(0.0, device=mask_logits.device, dtype=mask_logits.dtype)
            else:
                part = 2.0 * selected.sum()  # same as your original expression
            experiment_loss = part if experiment_loss is None else (experiment_loss + part)

        if experiment_loss is None:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        # divide by N_weight (python float) to keep same normalization
        if float(self.N_weight) == 0.0:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)
        return experiment_loss / float(self.N_weight)


    def weight_sparseness_loss(self):
        """L1-like sparsity on weight mask probabilities (via sigmoid).

        DDP-safe: compute the local sum over mask logits (a tensor depending on the local
        mask parameters). Do NOT all_reduce here — let DDP average gradients in backward.
        """
        if not self.use_weight_masks:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        # accumulate as a tensor (keeps grad path to mask_logits)
        sparse_loss = None

        # prefer ParameterDict if present
        if hasattr(self, "mask_logits_paramdict") and self.mask_logits_paramdict is not None:
            masks_iter = self.mask_logits_paramdict.values()
        else:
            # fallback to your previous dict/list approach
            masks_iter = (getattr(self, name) for name in self.mask_logits_weight_paramnames)

        for mask_logits in masks_iter:
            # ensure on device
            if mask_logits.device != self.cfg.device:
                mask_logits = mask_logits.to(self.cfg.device)
            s = torch.sigmoid(mask_logits).sum()  # tensor that depends on mask_logits
            sparse_loss = s if sparse_loss is None else (sparse_loss + s)

        if sparse_loss is None:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        # normalize by single-model N_weight (python float)
        if float(self.N_weight) == 0.0:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)
        return sparse_loss / float(self.N_weight)


    def edge_sparseness_loss(self):
        """L1-like sparsity on edge mask probabilities (via sigmoid)."""
        if not self.use_edge_masks:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        sparse_loss = None
        # mask logits for edges are often registered by name; handle similarly
        # using mask_logits_dict_edge which should hold nn.Parameters
        for _, mask_logits in self.mask_logits_dict_edge.items():
            if mask_logits.device != self.cfg.device:
                mask_logits = mask_logits.to(self.cfg.device)
            s = torch.sigmoid(mask_logits).sum()
            sparse_loss = s if sparse_loss is None else (sparse_loss + s)

        if sparse_loss is None:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        if float(self.N_edge) == 0.0:
            return torch.tensor(0.0, device=self.cfg.device, dtype=self.cfg.dtype)

        return sparse_loss / float(self.N_edge)

    # --- Introspection helpers ------------------------------------------------

    def get_edge_masks(self):
        """Return thresholded (0/1) edge masks for attention Q/K/V, MLP, and output."""
        edge_mask_dict = {
            'attn_q': [],
            'attn_k': [],
            'attn_v': [],
            'mlp': [],
            'output': [],
        }
        with torch.no_grad():
            edge_mask_dict['output'] = torch.where(self.edge_mask_output_logits > 0., 1., 0.).cpu()
            for i in range(self.cfg.n_layers):
                block_i = self.blocks[i]
                edge_mask_attn_q_i = torch.where(block_i.edge_mask_attention_q_logits > 0., 1., 0.).cpu()
                edge_mask_attn_k_i = torch.where(block_i.edge_mask_attention_k_logits > 0., 1., 0.).cpu()
                edge_mask_attn_v_i = torch.where(block_i.edge_mask_attention_v_logits > 0., 1., 0.).cpu()
                edge_mask_mlps_i = torch.where(block_i.edge_mask_mlp_logits > 0., 1., 0.).cpu()
                edge_mask_dict['attn_q'].append(edge_mask_attn_q_i)
                edge_mask_dict['attn_k'].append(edge_mask_attn_k_i)
                edge_mask_dict['attn_v'].append(edge_mask_attn_v_i)
                edge_mask_dict['mlp'].append(edge_mask_mlps_i)

        return edge_mask_dict

    def get_weight_density(self):
        """Return (#weights, #preserved, density) for thresholded (>0) weight masks (DDP-safe).

        Behavior:
        - Each rank counts its local preserved masks (mask_logits > 0).
        - We SUM across ranks then divide by world_size to get the per-model preserved count
            (because each rank holds a full copy of the model).
        - Uses self.N_weight as the single-model total weight count (float or int).
        """
        if not self.use_weight_masks:
            return -1, -1, 1.0

        try:
            is_dist = dist.is_available() and dist.is_initialized()
            world_size = dist.get_world_size() if is_dist else 1
            device = self.cfg.device if isinstance(self.cfg.device, torch.device) else torch.device(self.cfg.device)

            # local preserved count (use a tensor on the right device / dtype)
            local_preserved = torch.tensor(0, dtype=torch.long, device=device)

            with torch.no_grad():
                # iterate over registered mask parameters
                # prefer ParameterDict if you migrated to it; fall back to the paramname list otherwise
                if hasattr(self, "mask_logits_paramdict") and self.mask_logits_paramdict is not None:
                    mask_iter = self.mask_logits_paramdict.values()
                else:
                    # older code path: mask names stored in mask_logits_weight_paramnames as keys for getattr
                    mask_iter = (getattr(self, name) for name in self.mask_logits_weight_paramnames)

                for mask_logits in mask_iter:
                    # make sure mask_logits is on device
                    if mask_logits.device != device:
                        ml = mask_logits.to(device)
                    else:
                        ml = mask_logits
                    local_preserved += (ml > 0.).sum().to(local_preserved.device)

            # aggregate across ranks
            if is_dist:
                # sum across ranks
                dist.all_reduce(local_preserved, op=dist.ReduceOp.SUM)
                # get per-model preserved by averaging across replicas
                total_preserved_per_model = local_preserved.float() / float(world_size)
            else:
                total_preserved_per_model = local_preserved.float()

            # N_weight should be the single-model total (float)
            N_weight = float(self.N_weight)

            # safety: avoid division by zero
            if N_weight == 0:
                return 0, int(total_preserved_per_model.item()), 0.0

            density = float((total_preserved_per_model / N_weight).item())

            return int(N_weight), int(round(total_preserved_per_model.item())), density

        except Exception as e:
            # masks missing / uninitialized or other error
            return -1, -1, 1.0

    def get_edge_density(self):
        """Return (#edges, #preserved, density) for thresholded (>0) edge masks."""
        N_edge_preserved = 0
        with torch.no_grad():
            for _, mask in self.mask_logits_dict_edge.items():
                N_edge_preserved += torch.where(mask > 0., 1, 0).sum()

        edge_den = N_edge_preserved / self.N_edge
        return self.N_edge.item(), N_edge_preserved.item(), edge_den.item()

    # --- Loading/stubs --------------------------------------------------------

    def load_weight_mask_paramdict(self, path, map_location=None, strict=True):
        """
        Load weight masks saved with:
            torch.save(self.mask_logits_paramdict.state_dict(), path)

        Simple and exact: load the state_dict into the ParameterDict.
        """
        map_location = map_location or self.cfg.device
        ckpt = torch.load(path, map_location=map_location)
        # load into ParameterDict (will copy into the registered nn.Parameters)
        self.mask_logits_paramdict.load_state_dict(ckpt, strict=strict)
        # ensure they're trainable
        for p in self.mask_logits_paramdict.parameters():
            p.requires_grad_(True)


    def load_edge_mask_dict(self, path, map_location=None, strict=True):
        """
        Load edge masks saved with:
            torch.save(self.mask_logits_dict_edge, path)

        Expects ckpt to be a dict mapping edge_param_name -> tensor (or Parameter).
        """
        map_location = map_location or self.cfg.device
        ckpt = torch.load(path, map_location=map_location)

        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Expected dict checkpoint for edge masks, got {type(ckpt)}")

        with torch.no_grad():
            for saved_name, saved_tensor in ckpt.items():
                if saved_name not in self.mask_logits_dict_edge:
                    if strict:
                        raise RuntimeError(f"Saved edge mask '{saved_name}' not present in model")
                    else:
                        continue

                tgt = self.mask_logits_dict_edge[saved_name]
                tgt_data = tgt.data if isinstance(tgt, torch.nn.Parameter) else tgt

                if tuple(saved_tensor.shape) != tuple(tgt_data.shape):
                    if strict:
                        raise RuntimeError(
                            f"Shape mismatch for '{saved_name}': saved {tuple(saved_tensor.shape)} vs target {tuple(tgt_data.shape)}"
                        )
                    else:
                        continue

                tgt_data.copy_(saved_tensor.to(tgt_data.device, dtype=tgt_data.dtype))

        # ensure edge mask params remain trainable if they are Parameters
        for p in self.mask_logits_dict_edge.values():
            if isinstance(p, torch.nn.Parameter):
                p.requires_grad_(True)

    @classmethod
    def from_pretrained(cls, cfg):
        """Instantiate, load base weights from HookedTransformer, and set up masks.

        - Copies the state_dict from a pretrained `HookedTransformer`.
        - Calls `hook_state_dict` first to adapt K/V for GQA before loading.
        - Prepares tokenizer and data loaders via `setup_task`.
        """
        model = cls(cfg)
        print("cfg name:", cfg.full_model_name)
        state_dict = HookedTransformer.from_pretrained(cfg.full_model_name).state_dict()
        model.hook_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        model.setup_mask()

        # Keep an untouched copy of original weights for in-place masking
        for n, p in state_dict.items():
            if n in model.unmasked_params:
                model.unmasked_params[n] = p.clone()
        del state_dict
        torch.cuda.empty_cache()

        # Tokenizer setup (ensure padding token exists)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(model.cfg.device, dtype=model.cfg.dtype)

        model.tokenizer = tokenizer
        model.dls = setup_task(model)

        return model
    
    def setup_experiment(self):
        """Prepare cached original outputs for all splits and optionally start a Weights & Biases run.

        Caching original (unmasked) logits is required so later evaluations can compare
        masked outputs against the unmodified model outputs (KL/faithfulness metrics).
        """
        # Precompute and store original logits for train/eval/test dataloaders.

        # In Parallelism, origin input is no longer supported
        # self.prepare_origin_output(self.dls.train)
        # self.prepare_origin_output(self.dls.eval)
        # self.prepare_origin_output(self.dls.test)

        # Initialize wandb if requested in the config to track experiment metrics.
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        self.wandb_run = None

        # Only rank 0 initializes wandb
        if self.cfg.get('use_wandb', False) and rank == 0:
            run = wandb.init(
                project=self.cfg.wandb_project_name,
                entity=self.cfg.wandb_entity,
                config=self.cfg.to_dict(collapse=True),
            )
            self.wandb_run = run
        else:
            # Disable wandb completely on non-zero ranks
            wandb.init(mode="disabled")

        # Optional but recommended: sync all ranks
        if is_dist:
            dist.barrier()

    def teardown_experiment(self):
        """Cleanly finish any external experiment tracking (e.g., wandb)."""
        if self.cfg.get('use_wandb', False):
            # Ensure the wandb run is finalized to flush logs and release resources.
            self.wandb_run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def log_result(self, result):
        """Log a single dictionary of results to the configured tracking backend.

        Currently supports wandb when enabled in the configuration.
        """
        if self.cfg.get('use_wandb', False):
            # Forward the result dict to wandb for visualization/monitoring.
            self.wandb_run.log(result)

        if self.cfg.get('print_results', True):
            # Print to console for immediate feedback.
            pprint(result)

    def hook_state_dict(self, state_dict):
        """Preprocess state_dict before loading (e.g., expand K/V for GQA models).

        If `n_key_value_heads` < `n_heads`, repeat K/V heads to match Q head count
        so that edge masks align with the actual compute graph.
        """
        if self.cfg.n_key_value_heads is not None:
            repeat_kv_heads = self.cfg.n_heads // self.cfg.n_key_value_heads

            for layer in range(self.cfg.n_layers):
                prefix = f"blocks.{layer}.attn."
                state_dict[prefix + "_W_K"] = torch.repeat_interleave(state_dict[prefix + "_W_K"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_b_K"] = torch.repeat_interleave(state_dict[prefix + "_b_K"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_W_V"] = torch.repeat_interleave(state_dict[prefix + "_W_V"], dim=0, repeats=repeat_kv_heads)
                state_dict[prefix + "_b_V"] = torch.repeat_interleave(state_dict[prefix + "_b_V"], dim=0, repeats=repeat_kv_heads)

    # --- Evaluation / bookkeeping --------------------------------------------

    @torch.no_grad
    def evaluate(self, dl=None, reverse: bool = False):
        """Evaluate accuracy/KL/faith on a given dataloader with deterministic masks.

        - If `reverse=True`, measures *completeness* by flipping masks.
        - Requires `prepare_origin_output` to have been called to compare KL/faith.
        """
        if dl is None:
            dl = self.dls.eval

        self.eval()

        # Use deterministic masks for evaluation
        self.turn_on_weight_masks(deterministic=True, reverse=reverse)
        if self.use_edge_masks:
            self.turn_on_edge_masks(deterministic=True, reverse=reverse)

        # Densities for reporting
        if self.cfg.use_weight_masks:
            _, _, weight_density = self.get_weight_density()
        else:
            weight_density = 'na'
        if self.cfg.use_edge_masks:
            _, _, edge_density = self.get_edge_density()
        else:
            edge_density = 'na'

        # Respect config toggles
        if not self.cfg.use_weight_masks:
            self.turn_off_weight_masks()
        if not self.cfg.use_edge_masks:
            self.turn_off_edge_masks()

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1
        device = self.cfg.device if isinstance(self.cfg.device, torch.device) else torch.device(self.cfg.device)

        # local accumulators (tensors so we can all_reduce)
        local_correct = torch.tensor(0, dtype=torch.long, device=device)
        local_seen = torch.tensor(0, dtype=torch.long, device=device)
        local_faith_sum = torch.tensor(0.0, dtype=torch.float32, device=device)

        for i, batch_inputs in enumerate(dl):
            # Original (unmasked) logits must be precomputed and cached on the loader
            # original_logits = dl.original_output[i]

            batch_logits_masked = self(batch_inputs['input_ids'].to(self.cfg.device))[0]
            eval_results = self.compute_loss(batch_logits_masked, batch_inputs)

            # faith is average over batch in current compute_faith_loss implementation; convert to sum
            # (handles tensor or python float)
            faith_val = eval_results['faith']
            if isinstance(faith_val, torch.Tensor):
                faith_val = faith_val.detach().to(device)
            else:
                faith_val = torch.tensor(float(faith_val), dtype=torch.float32, device=device)

            local_faith_sum += faith_val * float(batch_size)

            # correct count
            n_corr = int(eval_results.get('n_correct', 0))
            local_correct += torch.tensor(n_corr, dtype=torch.long, device=device)
            local_seen += torch.tensor(batch_size, dtype=torch.long, device=device)


        self.turn_off_weight_masks()
        self.turn_off_edge_masks()

        # aggregate across ranks
        if is_dist:
            dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_seen, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_faith_sum, op=dist.ReduceOp.SUM)

        total = int(local_seen.item())
        correct = int(local_correct.item())

        # compute global mean faith loss
        if total > 0:
            faith_loss_mean = float((local_faith_sum / local_seen).item())
        else:
            faith_loss_mean = float('nan')

        # aggregate densities defensively (average across ranks) if they are numeric
        def _avg_if_numeric(x):
            if x == 'na':
                return 'na'
            try:
                tensor_x = torch.tensor(float(x), device=device)
                if is_dist:
                    dist.all_reduce(tensor_x, op=dist.ReduceOp.SUM)
                    return float((tensor_x / world_size).item())
                else:
                    return float(tensor_x.item())
            except Exception:
                return x

        weight_density = _avg_if_numeric(weight_density)
        edge_density = _avg_if_numeric(edge_density)

        results = {
            'acc': (correct / total) if total > 0 else float('nan'),
            'faith_loss': faith_loss_mean,
            'weight_density': weight_density,
            'edge_density': edge_density,
            'n_correct': correct,
            'total': total,
        }
        return results

    @torch.no_grad
    def prepare_origin_output(self, dl=None):
        """Cache original (unmasked) logits per batch index onto the dataloader.

        This must be called prior to `evaluate` so that KL/faith can compare
        masked vs. original outputs fairly.
        """
        self.turn_off_weight_masks()
        self.turn_off_edge_masks()

        self.eval()

        if dl is None:
            dl = self.dls.eval

        record = {}

        for i, batch_inputs in enumerate(dl):
            batch_logits_orig = self(batch_inputs['input_ids'].to(self.cfg.device))[0]

            if self.cfg.task_type in ['ioi', 'blimp']:
                batch_seq_lens = batch_inputs['seq_lens']
                batch_size = batch_logits_orig.shape[0]
                logits_target_good_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target good']]
                logits_target_bad_orig = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1, batch_inputs['target bad']]
                logits_gb_orig = torch.stack([logits_target_good_orig, logits_target_bad_orig], -1)  # (B, 2)
                record[i] = logits_gb_orig.cpu()

            elif self.cfg.task_type in ['pararel']:
                batch_seq_lens = batch_inputs['seq_lens']
                batch_size = batch_logits_orig.shape[0]
                full_logit = batch_logits_orig[torch.arange(batch_size), batch_seq_lens - 1]  # (B, answer_idx_vocab_size)
                answer_token_logits = full_logit[:, self.answer_idx_vocab]  # (B, answer_idx_vocab_size)
                record[i] = answer_token_logits.cpu()
            else:
                raise NotImplementedError(f"Original output not implemented for task type {self.cfg.task_type}")

        # Attach to the DataLoader instance (okay in Python, albeit a bit unconventional)
        dl.original_output = record

    # --- Loss dispatchers -----------------------------------------------------

    def compute_faith_loss(self, batch_logits_masked, batch_inputs, original_logits=None):
        """Compute faithfulness loss for current task type.

        For PARArel, restrict logits to the known answer vocab.
        """
        if self.cfg.task_type in ['ioi', 'blimp']:
            return compute_faith_loss_binary_label(batch_logits_masked, batch_inputs, original_logits)
        elif self.cfg.task_type in ['pararel']:
            batch_logits_masked = batch_logits_masked[:, :, self.answer_idx_vocab]
            return compute_faith_loss_multi_label(batch_logits_masked, batch_inputs, original_logits)
        else:
            raise NotImplementedError(f"Faith loss not implemented for task type {self.cfg.task_type}")

    def compute_complete_loss(self, batch_logits_masked, batch_inputs):
        """Compute completeness loss for the current task type."""
        if self.cfg.task_type in ['ioi', 'blimp']:
            return compute_complete_loss_binary_label(batch_logits_masked, batch_inputs)
        elif self.cfg.task_type in ['pararel']:
            batch_logits_masked = batch_logits_masked[:, :, self.answer_idx_vocab]
            return compute_complete_loss_multi_label(batch_logits_masked, batch_inputs)
        else:
            raise NotImplementedError(f"Complete loss not implemented for task type {self.cfg.task_type}")

    def compute_loss(self, batch_logits_masked, batch_inputs, original_logits=None):
        """Wrapper used in `evaluate`; currently returns just faithfulness results."""
        results = {}
        faith_results = self.compute_faith_loss(batch_logits_masked, batch_inputs, original_logits)
        results.update(faith_results)
        return results

    # --- Pruning loop ---------------------------------------------------------

    def search(self, modes='we'):
        """Run pruning for weights ('w'), edges ('e'), or both (default 'we')."""
        if 'w' in modes:
            self.run_prune(mode='w')

        gc.collect()
        torch.cuda.empty_cache()

        if 'e' in modes:
            return self.run_prune(mode='e')

    def evaluate_and_report(self, epoch=None, mode=None, meta={}):
        """Evaluate on train/eval/test splits and pretty-print a summary."""
        full_results = {}
        full_results.update(meta)

        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0

        splits = {'train': self.dls.train, 'eval': self.dls.eval, 'test': self.dls.test}
        for split_name, dl in splits.items():
            comp = self.evaluate(dl=dl, reverse=True)
            results = self.evaluate(dl=dl)
            results['comp'] = comp['acc']
            results['prune_mode'] = mode
            results['epoch'] = epoch
            full_results[split_name] = results

        if not is_dist or rank == 0:
            self.log_result(full_results)

    def run_prune(self, mode):
        """Optimize mask logits for sparsity + (faithfulness [+ completeness]).

        - For weights ('w'): minimize faith + λ_sparse * sparsity.
        - For edges   ('e'): additionally include completeness via reversed masks.
        """
         # ---------------------- distributed bookkeeping -----------------------
        is_distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_distributed else 0
        world_size = dist.get_world_size() if is_distributed else 1

        # ---------------------- choose mask params ----------------------------
        if mode == 'w':
            mask_params = list(self.mask_logits_paramdict.parameters())
            hparams = self.cfg.weight
        elif mode == 'e':
            mask_params = list(self.mask_logits_dict_edge.values())
            hparams = self.cfg.edge
        else:
            raise ValueError(f"Unknown mode {mode}")

        # ---------------------- BROADCAST (CRITICAL) --------------------------
        if is_distributed:
            # ensure identical starting logits on all ranks
            for p in mask_params:
                dist.broadcast(p.data, src=0)

        # ---------------------- optimizer -------------------------------------
        # if mode == 'w':
        #     optimizer = torch.optim.AdamW(
        #         [{"params": mask_params, "lr": hparams.lr}]
        #     )
        # else:  # mode == 'e'
        #     optimizer = torch.optim.AdamW(mask_params, lr=hparams.lr)
        optimizer = None
        if rank == 0:
            optimizer = torch.optim.AdamW(
                [{"params": mask_params, "lr": hparams.lr}]
            )
        # Ensure we don't double-mask: disable edges when pruning weights, but keep weights fixed when pruning edges
        if mode == 'w':
            self.turn_off_edge_masks()
        elif mode == 'e':
            self.turn_on_weight_masks(deterministic=True)

        disable_tqdm = not is_main_process() or self.cfg.get('disable_tqdm', False)
        epoch_loop = tqdm(
            range(hparams.train_epochs),
            desc='Number of Epochs', leave=True, dynamic_ncols=True,
            disable=disable_tqdm)

        base_seed = getattr(self.cfg, "seed", 42)
        if base_seed is None:
            base_seed = 42
        sync_mask_rng = True  # config toggle: sync mask RNG across ranks?


        for i, epoch in enumerate(epoch_loop):
            # Lambda scheduling (warmup/cooldown, etc.)
            lambda_sparse = schedule_epoch_lambda(
                epoch,
                lambda_0=hparams.lambda_sparse_init,
                max_times=hparams.max_times_lambda_sparse,
                min_times=hparams.min_times_lambda_sparse,
                n_epoch_warmup=hparams.n_epoch_warmup_lambda_sparse,
                n_epoch_cooldown=hparams.n_epoch_cooldown_lambda_sparse,
            )
            lambda_complete = schedule_epoch_lambda(epoch, hparams.lambda_complete_init)

            epoch_loop.set_description(f"Epoch {epoch} {mode} λ_s={lambda_sparse:.3f} λ_c={lambda_complete:.3f}")

            train_sampler = getattr(self.dls.train, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            for batch_idx, batch_inputs in enumerate(self.dls.train):
                batch_input_ids = batch_inputs['input_ids'].to(self.cfg.device)

                if rank == 0:
                    optimizer.zero_grad()

                # Optionally synchronize RNG for mask sampling so every rank samples identical Gumbel noise
                # This is helpful if you want identical sampled masks across ranks (recommended for determinism).
                if is_distributed and sync_mask_rng:
                    # create a deterministic per-epoch+batch seed shared across ranks
                    per_iter_seed = int(base_seed + epoch * 1_000_000 + batch_idx)
                    # set both CPU and CUDA RNGs (call on every rank)
                    torch.manual_seed(per_iter_seed)
                    torch.cuda.manual_seed_all(per_iter_seed)

                # Sample current masks and compute sparsity loss
                if mode == 'w':
                    self.turn_on_weight_masks(deterministic=False, reverse=False)
                    sparse_loss = self.weight_sparseness_loss()
                elif mode == 'e':
                    self.turn_on_edge_masks(deterministic=False, reverse=False)
                    sparse_loss = self.edge_sparseness_loss()

                # Faithfulness on masked model
                batch_logits_masked = self(batch_input_ids)[0]  # (B, seq_len, vocab_size)
                eval_results = self.compute_faith_loss(batch_logits_masked, batch_inputs)
                faith_loss = eval_results['faith']

                # Completeness (evaluate with reversed masks) — only for edge mode during this step
                if mode == 'e' and lambda_complete > 0:
                    self.turn_on_edge_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_input_ids)[0]
                    complete_loss, _ = self.compute_complete_loss(batch_logits, batch_inputs)
                else:
                    complete_loss = 0.0

                if mode == 'w':
                    loss = faith_loss + sparse_loss * lambda_sparse
                elif mode == 'e':
                    loss = faith_loss + sparse_loss * lambda_sparse + complete_loss * lambda_complete

                loss.backward()
                if is_distributed:
                    for p in mask_params:
                        if p.grad is not None:
                            dist.reduce(p.grad, dst=0, op=dist.ReduceOp.SUM)

                if rank == 0:
                    p.grad.div_(world_size)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # sync updated masks
                if is_distributed:
                    for p in mask_params:
                        dist.broadcast(p.data, src=0)
                for p in mask_params:
                    if rank != 0:
                        p.grad = None

                if mode == 'w':
                    self.turn_off_weight_masks()

                # For weights, compute completeness in a second pass with reversed masks
                if mode == 'w' and lambda_complete > 0:
                    self.turn_on_weight_masks(deterministic=False, reverse=True)
                    batch_logits = self(batch_input_ids)[0]
                    complete_loss, _ = self.compute_complete_loss(batch_logits, batch_inputs)
                    loss = complete_loss * lambda_complete
                    loss.backward()
                    if is_distributed:
                        for p in mask_params:
                            if p.grad is not None:
                                dist.reduce(p.grad, dst=0, op=dist.ReduceOp.SUM)

                    if rank == 0:
                        p.grad.div_(world_size)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    # sync updated masks
                    if is_distributed:
                        for p in mask_params:
                            dist.broadcast(p.data, src=0)
                    for p in mask_params:
                        if rank != 0:
                            p.grad = None

                    self.turn_off_weight_masks()

            # Periodic evaluation
            if i % self.cfg.evaluate_every == self.cfg.evaluate_every - 1:
                self.evaluate_and_report(
                    epoch=epoch, mode=mode,
                    meta={
                        'lambda_sparse': lambda_sparse,
                        'lambda_complete': lambda_complete
                    })

            if self.cfg.has('save_every', 'output_dir_path') and self.cfg.save_every and i % self.cfg.save_every == self.cfg.save_every - 1:
                if not is_distributed or rank == 0:
                    output_dir = Path(self.cfg.output_dir_path) / self.cfg.exp_name
                    output_dir.mkdir(parents=True, exist_ok=True)

                    if mode == 'w':
                        torch.save(self.mask_logits_paramdict.state_dict(), output_dir / f'weight_mask_{mode}_epoch{epoch}.pt')
                    if mode == 'e':
                        torch.save(self.mask_logits_dict_edge, output_dir / f'edge_mask_{mode}_epoch{epoch}.pt')

        del mask_params
        if rank == 0:
            del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
