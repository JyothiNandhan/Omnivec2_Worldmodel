"""
Positional encodings — shared utilities.
- 2D sinusoidal (fixed) for RGB
- Learned 3D MLP for LiDAR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..config import EMBED_DIM
except ImportError:
    from config import EMBED_DIM


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sine-cosine positional embeddings → (1, grid_size², embed_dim)."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 2)
    half_dim = embed_dim // 2
    omega = 1.0 / (10000 ** (
        torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim))
    out_h = grid[:, 0:1] * omega.unsqueeze(0)
    out_w = grid[:, 1:2] * omega.unsqueeze(0)
    pe = torch.cat([torch.sin(out_h), torch.cos(out_h),
                    torch.sin(out_w), torch.cos(out_w)], dim=-1)
    if pe.shape[-1] < embed_dim:
        pe = F.pad(pe, (0, embed_dim - pe.shape[-1]))
    return pe[:, :embed_dim].unsqueeze(0)


class LidarPositionalEmbedding(nn.Module):
    """Learned 3D positional embedding from patch center coordinates."""
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(),
            nn.Linear(128, embed_dim))

    def forward(self, centers):
        return self.mlp(centers)
