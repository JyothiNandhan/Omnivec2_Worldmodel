"""
Loss functions and metrics (paper §3.2.2) — shared across both modalities.
"""
import torch


def masked_mse_loss(pred, target, mask):
    """MSE loss on masked patches only."""
    loss_per_patch = ((pred - target) ** 2).mean(dim=-1)
    return (loss_per_patch * mask).sum() / mask.sum()


@torch.no_grad()
def reconstruction_psnr(pred, target, mask):
    """PSNR (dB) on masked patches — higher is better."""
    mse = ((pred - target) ** 2).mean(dim=-1)
    masked_mse = (mse * mask).sum() / mask.sum()
    return (10.0 * torch.log10(1.0 / (masked_mse + 1e-8))).item()
