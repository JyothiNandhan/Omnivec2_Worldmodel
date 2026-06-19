"""
LiDAR tokenizer — PointNet-style patch encoder (from Point-BERT).
Converts 3D point patches into EMBED_DIM-dimensional tokens.
"""
import torch
import torch.nn as nn

try:
    from ..config import EMBED_DIM
except ImportError:
    from config import EMBED_DIM


class LidarPatchEncoder(nn.Module):
    """PointNet-style encoder for 3D point patches.

    (B, G, K, 3) → (B, G, EMBED_DIM)

    Architecture:
        Conv1d(3→128) → BN → ReLU → Conv1d(128→256) → MaxPool
        → cat(global, local)
        → Conv1d(512→512) → BN → ReLU → Conv1d(512→D) → MaxPool
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1))
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, embed_dim, 1))

    def forward(self, patches):
        B, G, K, _ = patches.shape
        x = patches.reshape(B * G, K, 3).transpose(2, 1)        # (BG, 3, K)
        x = self.first_conv(x)                                    # (BG, 256, K)
        x_global = x.max(dim=2, keepdim=True)[0]                 # (BG, 256, 1)
        x = torch.cat([x_global.expand(-1, -1, K), x], dim=1)   # (BG, 512, K)
        x = self.second_conv(x)                                   # (BG, D, K)
        x = x.max(dim=2)[0]                                       # (BG, D)
        return x.reshape(B, G, self.embed_dim)
