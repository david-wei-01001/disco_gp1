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

import torch
from disco_gp import DiscoGPTransformer, Config, set_seed

if __name__ == "__main__":


    # Reproducibility
    set_seed(42)


    # ---------------------------------------------------------------
    # 0) Weight and Bias initialization (optional)
    # ---------------------------------------------------------------
    wandb_cfg = Config(
        use_wandb=True,
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
        batch_size=64,
        ds_split_ratios=(0.8, 0.1, 0.1),
    )

    # ---------------------------------------------------------------
    # 3) Experiment meta (how often to print/eval/save)
    # ---------------------------------------------------------------
    exp_cfg = Config(
        evaluate_every=1,            # evaluate & print every N epochs
        # save_every=1,              # if specified, save weight masks every N epochs
        # edge_save_every=1,         # if specified, save edge masks every N epochs
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
    model = DiscoGPTransformer.from_pretrained(cfg)

    # ---------------------------------------------------------------
    # 6) Run experiment setup (cache original model outputs, wandb, etc.)
    # ---------------------------------------------------------------
    print("[Step] Setup the experiment…")
    model.setup_experiment()

    # ---------------------------------------------------------------
    # 7) Baseline evaluation (before any pruning)
    # ---------------------------------------------------------------
    print("[Step] Baseline evaluation:")
    model.evaluate_and_report(epoch=0, mode="baseline")

    # ---------------------------------------------------------------
    # 8) Discover sheaves: prune weights and/or edges
    #    Internally, this optimizes mask logits for faithfulness/completeness
    #    with sparsity regularization.
    # ---------------------------------------------------------------
    print("[Step] Pruning (weights + edges)…")
    model.search()   # modes='we' by default; use 'w' or 'e' to restrict

    # ---------------------------------------------------------------
    # 9) Final evaluation (after pruning)
    # ---------------------------------------------------------------
    print("[Step] Final evaluation:")
    model.evaluate_and_report(epoch="final", mode="pruned")

    # ---------------------------------------------------------------
    # 10) Teardown experiment (close wandb run, save final masks, etc
    # ---------------------------------------------------------------
    model.teardown_experiment()
    print("\nDone. Swap models/tasks above to explore further!")
