"""
RGB tokenizer — Conv2d patch embedding (paper §3.1).
Converts an image into a sequence of EMBED_DIM-dimensional tokens.
"""
import torch.nn as nn

try:
    from ..config import EMBED_DIM, IMG_SIZE, PATCH_SIZE
    from ..shared.positional import get_2d_sincos_pos_embed
except ImportError:
    from config import EMBED_DIM, IMG_SIZE, PATCH_SIZE
    from shared.positional import get_2d_sincos_pos_embed


class RGBPatchEmbedding(nn.Module):
    """(B, 3, H, W) → (B, num_patches, EMBED_DIM)"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM,
                              kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
        grid_size = IMG_SIZE // PATCH_SIZE
        self.pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(EMBED_DIM, grid_size),
            requires_grad=False)   # fixed sin-cos

    def forward(self, imgs):
        x = self.proj(imgs).flatten(2).transpose(1, 2)   # (B, N, D)
        return x + self.pos_embed
