"""
RGB decoder — lightweight transformer for pixel patch reconstruction.
Discarded after Stage 1 pretraining.
"""
import torch
import torch.nn as nn

try:
    from ..config import FC_HIDDEN, DEC_EMBED_DIM, DEC_HEADS, DEC_LAYERS_RGB
    from ..config import IMG_SIZE, PATCH_SIZE
    from ..shared.positional import get_2d_sincos_pos_embed
except ImportError:
    from config import FC_HIDDEN, DEC_EMBED_DIM, DEC_HEADS, DEC_LAYERS_RGB
    from config import IMG_SIZE, PATCH_SIZE
    from shared.positional import get_2d_sincos_pos_embed


class RGBDecoder(nn.Module):
    """Reconstructs pixel patches from shared encoder output.

    Input:  encoded (B, N_vis, FC_HIDDEN),  ids_restore (B, N)
    Output: (B, N, 3·P²) — predicted pixel patches
    """

    def __init__(self):
        super().__init__()
        self.patch_dim = 3 * PATCH_SIZE ** 2

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DEC_EMBED_DIM))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embedding
        grid_size = IMG_SIZE // PATCH_SIZE
        self.dec_pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(DEC_EMBED_DIM, grid_size),
            requires_grad=False)

        # Project encoder→decoder dim
        self.enc_to_dec = nn.Linear(FC_HIDDEN, DEC_EMBED_DIM)

        # Transformer
        dec_layer = nn.TransformerEncoderLayer(
            d_model=DEC_EMBED_DIM, nhead=DEC_HEADS,
            dim_feedforward=DEC_EMBED_DIM * 4, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            dec_layer, num_layers=DEC_LAYERS_RGB)
        self.norm = nn.LayerNorm(DEC_EMBED_DIM)

        # Final projection → pixel patches
        self.head = nn.Linear(DEC_EMBED_DIM, self.patch_dim)

    def forward(self, encoded, ids_restore):
        x = self.enc_to_dec(encoded)
        B, N_vis, D = x.shape
        N = ids_restore.shape[1]
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = x + self.dec_pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x)
