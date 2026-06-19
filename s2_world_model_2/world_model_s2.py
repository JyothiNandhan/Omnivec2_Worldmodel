"""
OmniVec2 Stage 2 World Model Architecture.

Data flow per frame:
  RGB + LiDAR
      → Stage 1 (tokenizer + f(.))              [frozen]
      → Stage 2 (cross-attention + g(.))         [frozen]
      → g(.) tokens  ← this is the feature space the World Model operates in

The TemporalTokenPredictor sees a history of g(.) tokens
[T-H, ..., T-1, T] and predicts the g(.) tokens at T+1.

Architecture (v5):
  inner_dim=256, 6 transformer layers, 8 heads, FFN=1024 (~5.3M trainable params)
  Input/output projections bridge Stage-2 token dim (128) to internal dim (256).
  Pixel head is connected (no detach) so pixel reconstruction loss co-supervises
  the temporal encoder, not just the pixel MLP.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FC_HIDDEN, PATCH_SIZE  # noqa: E402
from rgb.patches import patchify  # noqa: E402


class CrossPatchTransformer(nn.Module):
    """
    Refines predicted RGB tokens with spatial attention before pixel decoding.

    Each token attends to all other patch tokens so neighboring patches agree
    on texture, color, and depth discontinuities before the MLP maps them to
    pixels. Runs entirely in token space — no 2D operations, geometry intact.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.encoder(x))


class TemporalTokenPredictor(nn.Module):
    """
    Temporal Transformer that predicts future g(.) tokens from a history.

    Input shape:  (B, T, N, C)  — batch, time, tokens, channels
    Output shape: (B, N, C)     — predicted next-step tokens

    The model works at an internal dimension (inner_dim) that is wider than
    the Stage-2 token dim (embed_dim=128). Input/output linear projections
    bridge the two spaces so the interface with Stage 2 is unchanged.
    """

    def __init__(
        self,
        embed_dim: int = FC_HIDDEN,   # 128 — I/O dim matching Stage 2 token dim
        inner_dim: int = 256,          # internal transformer dim (wider for capacity)
        num_heads: int = 8,
        num_layers: int = 6,
        max_history: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.inner_dim = inner_dim

        # Bridge Stage-2 token dim ↔ internal transformer dim
        self.input_proj  = nn.Linear(embed_dim, inner_dim)
        self.output_proj = nn.Linear(inner_dim,  embed_dim)

        # Learnable modality and time embeddings live in inner_dim space
        self.rgb_modality   = nn.Parameter(torch.zeros(1, 1, 1, inner_dim))
        self.lidar_modality = nn.Parameter(torch.zeros(1, 1, 1, inner_dim))
        self.ego_modality   = nn.Parameter(torch.zeros(1, 1, 1, inner_dim))
        self.time_embed     = nn.Parameter(torch.zeros(1, max_history, 1, inner_dim))
        nn.init.normal_(self.rgb_modality,   std=0.02)
        nn.init.normal_(self.lidar_modality, std=0.02)
        nn.init.normal_(self.ego_modality,   std=0.02)
        nn.init.normal_(self.time_embed,     std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=num_heads,
            dim_feedforward=inner_dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Separate output norms (inner_dim) before delta prediction
        self.rgb_norm   = nn.LayerNorm(inner_dim)
        self.lidar_norm = nn.LayerNorm(inner_dim)

        # Ego-motion projector: 7D → inner_dim Action Token
        self.ego_mlp = nn.Sequential(
            nn.Linear(7, inner_dim),
            nn.GELU(),
            nn.LayerNorm(inner_dim),
        )

        # Delta predictors: inner_dim → embed_dim → embed_dim
        # Zero-initialised final layer → copy-last at init (safe starting point)
        self.rgb_delta_predictor = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lidar_delta_predictor = nn.Sequential(
            nn.Linear(inner_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Pixel prediction heads (operate on embed_dim predicted tokens)
        self.rgb_mean_predictor = nn.Linear(embed_dim, 1)
        self.rgb_var_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softplus(),
        )
        self.rgb_patch_delta_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3 * PATCH_SIZE ** 2),
            nn.Tanh(),
        )
        self.cross_patch_transformer = CrossPatchTransformer(embed_dim, num_layers=4)

        # Zero-init final layers of delta predictors → copy-last at init
        nn.init.zeros_(self.rgb_delta_predictor[-1].weight)
        nn.init.zeros_(self.rgb_delta_predictor[-1].bias)
        nn.init.zeros_(self.lidar_delta_predictor[-1].weight)
        nn.init.zeros_(self.lidar_delta_predictor[-1].bias)
        nn.init.zeros_(self.rgb_patch_delta_predictor[-2].weight)
        nn.init.zeros_(self.rgb_patch_delta_predictor[-2].bias)

    def forward(self, rgb_token_sequence, lidar_token_sequence, ego_token_sequence):
        B, T, N_rgb, C = rgb_token_sequence.shape
        _, _, N_lid, _ = lidar_token_sequence.shape

        if T > self.time_embed.shape[1]:
            raise ValueError(f"History {T} > max_history {self.time_embed.shape[1]}")

        # 1. Project ego-motion → Action Token: (B, T, 7) → (B, T, 1, inner_dim)
        ego_x = self.ego_mlp(ego_token_sequence).unsqueeze(2)

        # 2. Project tokens to inner_dim and add time + modality embeddings
        rgb_x = self.input_proj(rgb_token_sequence) + self.time_embed[:, :T] + self.rgb_modality
        lid_x = self.input_proj(lidar_token_sequence) + self.time_embed[:, :T] + self.lidar_modality
        ego_x = ego_x + self.time_embed[:, :T] + self.ego_modality

        # 3. Spatial concatenation: (B, T, N_rgb + N_lid + 1, inner_dim)
        combined = torch.cat([rgb_x, lid_x, ego_x], dim=2)
        N_total = N_rgb + N_lid + 1

        # 4. Flatten to (B, T*N_total, inner_dim) for joint spatio-temporal attention
        x = combined.reshape(B, T * N_total, self.inner_dim)

        # 5. Apply joint transformer
        x = self.temporal_encoder(x)

        # 6. Extract tokens for the last frame T
        x_last = x[:, -N_total:, :]   # (B, N_total, inner_dim)

        # 7. Split back into RGB and LiDAR (ego token discarded)
        rgb_out = x_last[:, :N_rgb, :]            # (B, N_rgb, inner_dim)
        lid_out = x_last[:, N_rgb:N_rgb+N_lid, :] # (B, N_lid, inner_dim)

        # 8. Predict residual deltas (inner_dim → embed_dim)
        pred_rgb_delta   = self.rgb_delta_predictor(self.rgb_norm(rgb_out))
        pred_lidar_delta = self.lidar_delta_predictor(self.lidar_norm(lid_out))
        pred_rgb   = rgb_token_sequence[:, -1]   + pred_rgb_delta
        pred_lidar = lidar_token_sequence[:, -1] + pred_lidar_delta

        # 9. Pixel head: refine predicted tokens with cross-patch attention then decode.
        # No detach — pixel reconstruction loss co-supervises the temporal encoder.
        # pred_rgb carries full gradient through delta_predictor → temporal_encoder.
        rgb_patch_ctx = self.cross_patch_transformer(pred_rgb)

        pred_rgb_mean        = self.rgb_mean_predictor(rgb_patch_ctx)
        pred_rgb_var         = self.rgb_var_predictor(rgb_patch_ctx)
        pred_rgb_patch_delta = 0.5 * self.rgb_patch_delta_predictor(rgb_patch_ctx)

        return {
            "pred_rgb_tokens":       pred_rgb,
            "pred_lidar_tokens":     pred_lidar,
            "pred_rgb_token_delta":  pred_rgb_delta,
            "pred_lidar_token_delta":pred_lidar_delta,
            "pred_rgb_mean":         pred_rgb_mean,
            "pred_rgb_var":          pred_rgb_var,
            "pred_rgb_patch_delta":  pred_rgb_patch_delta,
        }


class OmniVec2Stage2WorldModel(nn.Module):
    """
    Wraps a trained OmniVec2Stage2 backbone and a TemporalTokenPredictor.

    Stage 2 is completely frozen.
    Only the TemporalTokenPredictor is trained.
    """

    def __init__(self, stage2_model, temporal_model=None, freeze_stage2: bool = True):
        super().__init__()
        self.stage2  = stage2_model
        self.temporal = temporal_model or TemporalTokenPredictor()
        if freeze_stage2:
            self.freeze_stage2()

    def freeze_stage2(self):
        for p in self.stage2.parameters():
            p.requires_grad = False

    def unfreeze_stage2(self):
        for p in self.stage2.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _encode_frame(self, imgs, points):
        """
        Encode one frame through frozen Stage 2 → return g(.) tokens.

        mask_ratio = 0 → no masking → full token sequence.
        """
        out = self.stage2(imgs, points, mask_ratio_rgb=0.0, mask_ratio_lidar=0.0)
        return out["rgb_g"], out["lidar_g"]   # (B, N, C), (B, M, C)

    def encode_sequence(self, rgb_sequence, lidar_sequence):
        """
        Encode a history of T frames into Stage 2 g(.) token sequences.

        Args:
            rgb_sequence:   (B, T, 3, H, W)
            lidar_sequence: (B, T, N_POINTS, 3)

        Returns:
            rgb_g_seq:   (B, T, N, C)
            lidar_g_seq: (B, T, M, C)
        """
        B, T = rgb_sequence.shape[:2]
        rgb_flat   = rgb_sequence.reshape(B * T, *rgb_sequence.shape[2:])
        lidar_flat = lidar_sequence.reshape(B * T, *lidar_sequence.shape[2:])

        rgb_g_flat, lidar_g_flat = self._encode_frame(rgb_flat, lidar_flat)

        N, M, C = rgb_g_flat.shape[1], lidar_g_flat.shape[1], rgb_g_flat.shape[2]
        return rgb_g_flat.reshape(B, T, N, C), lidar_g_flat.reshape(B, T, M, C)

    @torch.no_grad()
    def encode_target(self, rgb_target, lidar_target):
        """Encode the ground-truth future frame → g(.) tokens (prediction target)."""
        return self._encode_frame(rgb_target, lidar_target)

    def forward(self, rgb_sequence, lidar_sequence, ego_sequence):
        """
        Args:
            rgb_sequence:   (B, T, 3, H, W)
            lidar_sequence: (B, T, N_POINTS, 3)
            ego_sequence:   (B, T, 7)

        Returns:
            dict: pred_rgb_tokens (B, N, C), pred_lidar_tokens (B, M, C)
        """
        rgb_g_seq, lidar_g_seq = self.encode_sequence(rgb_sequence, lidar_sequence)
        out = self.temporal(rgb_g_seq, lidar_g_seq, ego_sequence)
        last_rgb_patches = patchify(rgb_sequence[:, -1])
        out["pred_rgb_patches"] = (
            last_rgb_patches + out["pred_rgb_patch_delta"]
        ).clamp(0.0, 1.0)
        return out
