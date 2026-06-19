"""
LiDAR decoder — lightweight transformer for 3D point patch reconstruction.
Discarded after Stage 1 pretraining.
"""
import torch
import torch.nn as nn

try:
    from ..config import FC_HIDDEN, DEC_EMBED_DIM, DEC_HEADS, DEC_LAYERS_LIDAR
    from ..config import GROUP_SIZE
    from ..shared.positional import LidarPositionalEmbedding
except ImportError:
    from config import FC_HIDDEN, DEC_EMBED_DIM, DEC_HEADS, DEC_LAYERS_LIDAR
    from config import GROUP_SIZE
    from shared.positional import LidarPositionalEmbedding


class LidarDecoder(nn.Module):
    """Reconstructs 3D point patches from shared encoder output.

    Input:  encoded (B, N_vis, FC_HIDDEN),  ids_restore (B, G),  centers (B, G, 3)
    Output: (B, G, K×3) — predicted point patches (flattened)
    """

    def __init__(self):
        super().__init__()
        self.lidar_patch_dim = GROUP_SIZE * 3

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_EMBED_DIM))
        nn.init.normal_(self.mask_token, std=0.02)

        # Project encoder→decoder dim
        self.enc_to_dec = nn.Linear(FC_HIDDEN, DEC_EMBED_DIM)

        # Learned 3D positional embedding
        self.dec_pos = LidarPositionalEmbedding(DEC_EMBED_DIM)

        # Transformer (lighter than RGB decoder)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=DEC_EMBED_DIM, nhead=DEC_HEADS,
            dim_feedforward=DEC_EMBED_DIM * 4, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            dec_layer, num_layers=DEC_LAYERS_LIDAR)
        self.norm = nn.LayerNorm(DEC_EMBED_DIM)

        # Final projection → point patches
        self.head = nn.Linear(DEC_EMBED_DIM, self.lidar_patch_dim)

    def forward(self, encoded, ids_restore, centers):
        x = self.enc_to_dec(encoded)
        B, N_vis, D = x.shape
        N = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.dec_pos(centers)
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)
