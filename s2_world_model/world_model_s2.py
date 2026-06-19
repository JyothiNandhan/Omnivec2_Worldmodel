"""
OmniVec2 Stage 2 World Model Architecture.

Data flow per frame:
  RGB + LiDAR
      → Stage 1 (tokenizer + f(.))              [frozen]
      → Stage 2 (cross-attention + g(.))         [frozen]
      → g(.) tokens  ← this is the feature space the World Model operates in

The TemporalTokenPredictor sees a history of g(.) tokens
[T-H, ..., T-1, T] and predicts the g(.) tokens at T+1.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FC_HIDDEN  # noqa: E402


class TemporalTokenPredictor(nn.Module):
    """
    Temporal Transformer that predicts future g(.) tokens from a history.

    Input shape:  (B, T, N, C)  — batch, time, tokens, channels
    Output shape: (B, N, C)     — predicted next-step tokens
    """

    def __init__(
        self,
        embed_dim: int = FC_HIDDEN,
        num_heads: int = 4,
        num_layers: int = 4,
        max_history: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Learnable modality and time embeddings
        self.rgb_modality   = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.lidar_modality = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.time_embed     = nn.Parameter(torch.zeros(1, max_history, 1, embed_dim))
        nn.init.normal_(self.rgb_modality,   std=0.02)
        nn.init.normal_(self.lidar_modality, std=0.02)
        nn.init.normal_(self.time_embed,     std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Separate output norms and predictors for the two modalities
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        
        self.rgb_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.lidar_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # New heads to predict the future patch color statistics (for visualization)
        # Predicting 1 value per patch for mean and variance (computed across the 192 patch features)
        self.rgb_mean_predictor = nn.Linear(embed_dim, 1)
        self.rgb_var_predictor = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softplus() # Variance must be positive
        )

    def forward(self, rgb_token_sequence, lidar_token_sequence):
        B, T, N_rgb, C = rgb_token_sequence.shape
        _, _, N_lid, _ = lidar_token_sequence.shape
        
        if T > self.time_embed.shape[1]:
            raise ValueError(f"History {T} > max_history {self.time_embed.shape[1]}")
            
        # 1. Add time and modality embeddings
        rgb_x = rgb_token_sequence + self.time_embed[:, :T] + self.rgb_modality
        lid_x = lidar_token_sequence + self.time_embed[:, :T] + self.lidar_modality
        
        # 2. Spatial Concatenation: (B, T, N_rgb + N_lid, C)
        combined = torch.cat([rgb_x, lid_x], dim=2)
        N_total = N_rgb + N_lid
        
        # 3. Flatten to (B, T * N_total, C) for Deep Joint Spatio-Temporal Attention
        #    This allows RGB to directly cross-attend to LiDAR point clouds over time!
        x = combined.reshape(B, T * N_total, C)
        
        # 4. Apply joint transformer
        x = self.temporal_encoder(x)
        
        # 5. Extract the tokens corresponding to the last frame 'T'
        x_last = x[:, -N_total:, :]  # shape: (B, N_total, C)
        
        # 6. Split back into RGB and LiDAR
        rgb_out = x_last[:, :N_rgb, :]
        lid_out = x_last[:, N_rgb:, :]
        
        # 7. Project to the future target frame
        pred_rgb = self.rgb_predictor(self.rgb_norm(rgb_out))
        pred_lidar = self.lidar_predictor(self.lidar_norm(lid_out))
        
        # 8. Predict the future color statistics for RGB visualization
        pred_rgb_mean = self.rgb_mean_predictor(rgb_out)
        pred_rgb_var = self.rgb_var_predictor(rgb_out)
        
        return {
            "pred_rgb_tokens": pred_rgb,
            "pred_lidar_tokens": pred_lidar,
            "pred_rgb_mean": pred_rgb_mean,
            "pred_rgb_var": pred_rgb_var,
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

    def forward(self, rgb_sequence, lidar_sequence):
        """
        Args:
            rgb_sequence:   (B, T, 3, H, W)
            lidar_sequence: (B, T, N_POINTS, 3)

        Returns:
            dict: pred_rgb_tokens (B, N, C), pred_lidar_tokens (B, M, C)
        """
        rgb_g_seq, lidar_g_seq = self.encode_sequence(rgb_sequence, lidar_sequence)
        return self.temporal(rgb_g_seq, lidar_g_seq)
