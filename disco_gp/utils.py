import os
import random
import numpy as np
import torch
import json
import tempfile
from typing import Any, Dict, List, Optional
import math

def set_seed(seed: int):
    """Set seed for reproducibility across random, NumPy, and PyTorch (CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def schedule_epoch_lambda(epoch, lambda_0, max_times=1., min_times=1., 
                          n_epoch_warmup=0, n_epoch_cooldown=0):

    if epoch < n_epoch_warmup:
        return lambda_0  + lambda_0 * (max_times - 1) * epoch / n_epoch_warmup
        
    elif epoch < n_epoch_warmup + n_epoch_cooldown:
        return lambda_0 * max_times - lambda_0 * (max_times - min_times) * (epoch - n_epoch_warmup) / n_epoch_cooldown
        
    else:
        return lambda_0 * min_times

def ensure_dir_for_file(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_jsonl(store_path: str, obj: Dict[str, Any]) -> None:
    """
    Append a JSON object as one line to store_path (JSON Lines).
    Uses flush + fsync to reduce risk of lost writes on crash.
    """
    ensure_dir_for_file(store_path)
    line = json.dumps(obj, ensure_ascii=False)
    # Open in append mode, write line + newline, flush & fsync for durability
    with open(store_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some environments (e.g., certain network filesystems) may not support fsync.
            pass

def read_jsonl(store_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON Lines file and return list of dicts.
    If the last line is partial/corrupt, it will be skipped rather than causing an exception.
    """
    results = []
    if not os.path.exists(store_path):
        return results
    with open(store_path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            try:
                results.append(json.loads(raw))
            except json.JSONDecodeError:
                # Skip malformed line (likely partial write from crash).
                # You could log a warning here if you have logging available.
                # Optionally attempt to salvage by trimming trailing characters, but safer to skip.
                continue
    return results

def atomic_append_json_array(store_path: str, obj: Dict[str, Any]) -> None:
    """
    Alternative: keep the file as a single JSON array.
    This reads the whole file, appends, then writes atomically (temp file + os.replace).
    Safer for correctness (single valid JSON) but less efficient and less append-friendly.
    """
    ensure_dir_for_file(store_path)
    data: List[Any] = []
    if os.path.exists(store_path):
        try:
            with open(store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    # unexpected format -- back up and start a fresh list
                    data = []
        except Exception:
            # if corrupted, back up current file and start fresh array
            try:
                backup = store_path + ".broken"
                os.replace(store_path, backup)
            except Exception:
                pass
            data = []

    data.append(obj)

    dirpath = os.path.dirname(os.path.abspath(store_path)) or "."
    fd, tmpname = tempfile.mkstemp(prefix=".tmp_json_", dir=dirpath)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            json.dump(data, tf, ensure_ascii=False, indent=2)
            tf.flush()
            try:
                os.fsync(tf.fileno())
            except OSError:
                pass
        # Atomic replace
        os.replace(tmpname, store_path)
    finally:
        # If something went wrong and tmp still exists, remove it
        if os.path.exists(tmpname):
            try:
                os.remove(tmpname)
            except OSError:
                pass
