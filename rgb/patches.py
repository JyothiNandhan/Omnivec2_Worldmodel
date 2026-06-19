"""
RGB patch helpers — patchify, unpatchify, and per-patch normalization.
"""
import torch

try:
    from ..config import IMG_SIZE, PATCH_SIZE
except ImportError:
    from config import IMG_SIZE, PATCH_SIZE


def patchify(imgs: torch.Tensor) -> torch.Tensor:
    """Convert images to patch sequences.
    (B, 3, H, W) → (B, N, 3·P²)
    """
    p = PATCH_SIZE
    B, C, H, W = imgs.shape
    h = w = H // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h * w, C * p * p)
    return x


def unpatchify(patches: torch.Tensor) -> torch.Tensor:
    """Convert patch sequences back to images.
    (B, N, 3·P²) → (B, 3, H, W)
    """
    p = PATCH_SIZE
    h = w = IMG_SIZE // p
    x = patches.reshape(patches.shape[0], h, w, 3, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(patches.shape[0], 3, h * p, w * p)
    return x


def normalize_patches(patches: torch.Tensor):
    """Per-patch zero-mean unit-variance normalization.
    Returns: (normalized, mean, var)
    """
    mean = patches.mean(dim=-1, keepdim=True)
    var  = patches.var(dim=-1, keepdim=True)
    return (patches - mean) / (var + 1e-6).sqrt(), mean, var
