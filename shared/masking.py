"""
Random masking (MAE-style) — shared across both modalities.
"""
import torch


def random_mask(tokens: torch.Tensor, mask_ratio: float):
    """Randomly mask a fraction of tokens.

    Returns: x_visible, ids_restore, ids_keep, mask (1=masked, 0=visible)
    """
    B, N, D = tokens.shape
    len_keep = int(N * (1 - mask_ratio))
    noise       = torch.rand(B, N, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep    = ids_shuffle[:, :len_keep]
    x_visible   = torch.gather(
        tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.ones(B, N, device=tokens.device)
    mask.scatter_(1, ids_keep, 0.0)
    return x_visible, ids_restore, ids_keep, mask
