"""
Shared Encoder f(·) — the core of OmniVec2 (paper §3.1).

BERT-style transformer + 3 FC layers with ReLU.
Processes BOTH RGB and LiDAR tokens through the same weights.
This is what gets kept after Stage 1 pretraining.
"""
import torch.nn as nn

try:
    from ..config import EMBED_DIM, NUM_HEADS, ENC_LAYERS, FC_HIDDEN
except ImportError:
    from config import EMBED_DIM, NUM_HEADS, ENC_LAYERS, FC_HIDDEN


class SharedEncoder(nn.Module):
    """Feature transformation network f(·) — shared across all modalities.

    Input:  (B, N_visible, EMBED_DIM)
    Output: (B, N_visible, FC_HIDDEN)
    """
    def __init__(self):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 4, activation="gelu",
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM, FC_HIDDEN), nn.ReLU(),
            nn.Linear(FC_HIDDEN, FC_HIDDEN), nn.ReLU(),
            nn.Linear(FC_HIDDEN, FC_HIDDEN), nn.ReLU())

    def forward(self, x):
        x = self.transformer(x)
        x = self.norm(x)
        return self.fc(x)
