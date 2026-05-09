"""
run.py — minimal, commented quickstart for DiscoGP
---
This tiny script shows the **whole workflow** without any CLI plumbing:
1) set pruning hyperparameters,
2) pick a task (PARArel / IOI / BLiMP),
3) choose a base model,
4) load everything,
5) evaluate → prune → evaluate.

Tips
---
- **Swap models:** change `model_cfg = Config.from_tl(...)` below.
  Works with (examples):
    - "gpt2"  --- classic small baseline
    - "Qwen/Qwen3-0.6B"
    - "Qwen/Qwen3-1.7B"
    - "meta-llama/Llama-3.2-1B-Instruct"
- **Change the task:** uncomment the task block you want (PARArel / IOI / BLiMP)
  and fill the few obvious knobs (e.g., PARArel JSON path, IOI size, BLiMP paradigm).
- **Dtype:** bfloat16 is a good default on recent GPUs; change if needed.
- **Data splits:** we split into train / (dev+test) and then into dev / test.

That's it — keep it simple and tinker via comments.
"""

from pprint import pprint
import gc
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from disco_gp import DDPDiscoGPTransformer, Config, set_seed

def make_ddp_dataloader_from_existing(orig_dl, per_gpu_bs, rank, world_size, train_split=False):
    """Rebuild a DataLoader for DDP by installing a DistributedSampler using the same dataset/collate/etc."""
    dataset = orig_dl.dataset
    collate_fn = getattr(orig_dl, "collate_fn", None)
    num_workers = getattr(orig_dl, "num_workers", 0)
    pin_memory = getattr(orig_dl, "pin_memory", False)
    drop_last = getattr(orig_dl, "drop_last", False)
    persistent_workers = getattr(orig_dl, "persistent_workers", False)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=train_split)

    dl = DataLoader(
        dataset,
        batch_size=per_gpu_bs,
        sampler=sampler,
        shuffle=False,  # sampler handles shuffling
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(train_split and drop_last),
        persistent_workers=persistent_workers,
    )
    return dl

def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

def rank0_print(*args, **kwargs):
    """Print only from rank 0 (works even if dist not initialized)."""
    if is_main_process():
        print(*args, **kwargs)

if __name__ == "__main__":
    if not is_main_process():
        logging.getLogger().setLevel(logging.ERROR)
        # a few common noisy loggers you might want to quiet on worker ranks:
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()


    # Reproducibility
    set_seed(42)


    # ---------------------------------------------------------------
    # 0) Weight and Bias initialization (optional)
    # ---------------------------------------------------------------
    wandb_cfg = Config(
        use_wandb=False,
        wandb_project_name="DiscoGP", # Set to your project name if using W&B
        wandb_entity='WANDB_ENTITY',  # set to your username or team name if using W&B
    )

    # ---------------------------------------------------------------
    # 1) Pruning hyperparameters
    # ---------------------------------------------------------------
    weight_hparams = Config(
        use_weight_masks=True,
        gs_temp_weight=0.01,
        logits_w_init=1.0,
        lr=0.1,
        lambda_sparse_init=1.0,
        lambda_complete_init=1.0,
        min_times_lambda_sparse=1.0,
        max_times_lambda_sparse=1000.0,
        train_epochs=500,
        n_epoch_warmup_lambda_sparse=500,
        n_epoch_cooldown_lambda_sparse=1,
    )

    edge_hparams = Config(
        use_edge_masks=True,
        gs_temp_edge=1.0,
        logits_e_init=1.0,
        lr=0.1,
        lambda_sparse_init=1.0,
        lambda_complete_init=0.0,
        min_times_lambda_sparse=0.01,
        max_times_lambda_sparse=100.0,
        train_epochs=100,
        n_epoch_warmup_lambda_sparse=20,
        n_epoch_cooldown_lambda_sparse=20,
    )

    # ---------------------------------------------------------------
    # 2) Pick a task (choose ONE block)
    #    PARArel (relational probing), IOI, or BLiMP minimal pairs.
    # ---------------------------------------------------------------
    # --- PARArel (default) ---
    # task_cfg = Config(
    #     task_type="pararel",
    #     pararel_rel_ids="P36",                    # space-separated (e.g., "P36 P1376")
    #     pararel_data_path="./data/pararel_data_all.json",
    #     batch_size=64,
    #     ds_split_ratios=(0.8, 0.1, 0.1),
    # )

    # --- IOI (Indirect Object Identification) ---
    # task_cfg = Config(
    #     task_type="ioi",
    #     n_ioi_data=1000,
    #     batch_size=64,
    #     ds_split_ratios=(0.8, 0.1, 0.1),
    # )

    # --- BLiMP (choose a paradigm) ---
    task_cfg = Config(
        task_type="blimp",
        paradigm="anaphor_number_agreement",     # e.g., "anaphor_gender_agreement", etc.
        batch_size=4,
        ds_split_ratios=(0.8, 0.1, 0.1),
    )

    # ---------------------------------------------------------------
    # 3) Experiment meta (how often to print/eval/save)
    # ---------------------------------------------------------------
    exp_cfg = Config(
        evaluate_every=1,            # evaluate & print every N epochs
        # save_every=1,                # if specified, save masks every N epochs
        output_dir_path="./outputs",
        exp_name="quickstart",
    )

    # ---------------------------------------------------------------
    # 4) Choose a base model (swap this line to try others)
    #    Works with: "gpt2", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B",
    #    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"
    # ---------------------------------------------------------------
    model_cfg = Config.from_tl("gpt2", dtype=torch.bfloat16)

    # model_cfg = Config.from_tl("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
    # model_cfg = Config.from_tl("Qwen/Qwen3-1.7B", dtype=torch.bfloat16)
    # model_cfg = Config.from_tl("meta-llama/Llama-3.2-1B", dtype=torch.bfloat16)
    # model_cfg = Config.from_tl("meta-llama/Llama-3.2-1B-Instruct", dtype=torch.bfloat16)

    # Merge everything into a single config the model understands.
    cfg = Config.from_configs(
        wandb=wandb_cfg,
        weight=weight_hparams,
        edge=edge_hparams,
        task=task_cfg,
        model=model_cfg,
        exp=exp_cfg,
    )

    # ---------------------------------------------------------------
    # 5) Load the model/tokenizer and task dataloaders
    # ---------------------------------------------------------------
    print("[Step] Loading model + data…")
    model = DDPDiscoGPTransformer.from_pretrained(cfg)
    model.to(cfg.device, dtype=cfg.dtype)

    global_bs = int(cfg.task.batch_size)
    per_gpu_bs = max(1, global_bs // max(1, world_size))

    orig_train_dl = model.dls.train
    orig_eval_dl = model.dls.eval
    orig_test_dl = model.dls.test

    ddp_train = make_ddp_dataloader_from_existing(orig_train_dl, per_gpu_bs, rank, world_size, train_split=True)
    ddp_eval = make_ddp_dataloader_from_existing(orig_eval_dl, per_gpu_bs, rank, world_size, train_split=False)
    ddp_test = make_ddp_dataloader_from_existing(orig_test_dl, per_gpu_bs, rank, world_size, train_split=False)

    model.dls = SimpleNamespace(train=ddp_train, eval=ddp_eval, test=ddp_test)

    # ---------------------------------------------------------------
    # 6) Run experiment setup (cache original model outputs, wandb, etc.)
    # ---------------------------------------------------------------
    if rank == 0:
        print("[rank 0] Setup the experiment…")
    model.setup_experiment()

    # ---------------------------------------------------------------
    # 7) Baseline evaluation (before any pruning)
    # ---------------------------------------------------------------
    if rank == 0:
        print("[rank 0] Running baseline evaluation")
    model.evaluate_and_report(epoch=0, mode="baseline")

    # ---------------------------------------------------------------
    # 8) Discover sheaves: prune weights and/or edges
    #    Internally, this optimizes mask logits for faithfulness/completeness
    #    with sparsity regularization.
    # ---------------------------------------------------------------
    if rank == 0:
        print("[rank 0] Pruning (weights + edges)…")
    model.search()   # modes='we' by default; use 'w' or 'e' to restrict

    # ---------------------------------------------------------------
    # 9) Final evaluation (after pruning)
    # ---------------------------------------------------------------
    if rank == 0:
        print("[rank 0] Final evaluation:")
    model.evaluate_and_report(epoch="final", mode="pruned")

    # ---------------------------------------------------------------
    # 10) Teardown experiment (close wandb run, save final masks, etc
    # ---------------------------------------------------------------
    model.teardown_experiment()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\n[rank 0] Done. Swap models/tasks above to explore further!")
