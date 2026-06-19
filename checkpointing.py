"""
Checkpoint helpers for restartable OmniVec2 Stage 1 training.
"""
import os
import random
import warnings
from typing import Any, Dict

import numpy as np
import torch


def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_byte_tensor(state_value):
    if state_value is None:
        return None
    if isinstance(state_value, torch.Tensor) and state_value.dtype == torch.uint8 and state_value.device.type == "cpu":
        return state_value
    if isinstance(state_value, torch.Tensor):
        return state_value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state_value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        try:
            torch.set_rng_state(_coerce_byte_tensor(state["torch"]))
        except (TypeError, RuntimeError) as exc:
            warnings.warn(f"Skipping CPU RNG restore due to incompatible checkpoint format: {exc}")
    if torch.cuda.is_available() and "cuda" in state:
        try:
            cuda_state = [_coerce_byte_tensor(s) for s in state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_state)
        except (TypeError, RuntimeError) as exc:
            warnings.warn(f"Skipping CUDA RNG restore due to incompatible checkpoint format: {exc}")


def make_checkpoint_state(model, optimizer, scheduler, epoch, best_val_loss, history, args, extra=None):
    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "args": vars(args),
        "rng_state": _rng_state(),
    }
    if extra:
        state.update(extra)
    return state


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def rotate_named_checkpoints(checkpoint_dir: str, prefix: str, keep_last_n: int) -> None:
    for idx in range(keep_last_n, 0, -1):
        src = os.path.join(checkpoint_dir, f"{prefix}_checkpoint{idx}.pth")
        if not os.path.exists(src):
            continue
        if idx == keep_last_n:
            os.remove(src)
        else:
            dst = os.path.join(checkpoint_dir, f"{prefix}_checkpoint{idx + 1}.pth")
            os.replace(src, dst)


def save_stage_checkpoint_bundle(checkpoint_dir: str, prefix: str, state: Dict[str, Any], is_best: bool,
                                 keep_last_n: int) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(os.path.join(checkpoint_dir, f"{prefix}_last.pth"), state)
    rotate_named_checkpoints(checkpoint_dir, prefix, keep_last_n)
    save_checkpoint(os.path.join(checkpoint_dir, f"{prefix}_checkpoint1.pth"), state)
    if is_best:
        save_checkpoint(os.path.join(checkpoint_dir, f"{prefix}_best.pth"), state)


def load_training_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return checkpoint
