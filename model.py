"""
OmniVec2 Stage 1 — Combined Multimodal Model.

    RGB:   rgb.tokenizer → shared.encoder → rgb.decoder
    LiDAR: lidar.tokenizer → shared.encoder → lidar.decoder
"""
import torch
import torch.nn as nn

try:
    from .config import NUM_GROUP, GROUP_SIZE
    from .shared.encoder import SharedEncoder
    from .shared.masking import random_mask
    from .shared.positional import LidarPositionalEmbedding
    from .rgb.tokenizer import RGBPatchEmbedding
    from .rgb.decoder import RGBDecoder
    from .rgb.patches import patchify, normalize_patches
    from .lidar.tokenizer import LidarPatchEncoder
    from .lidar.decoder import LidarDecoder
    from .data.lidar_helpers import make_patches_batch
except ImportError:
    from config import NUM_GROUP, GROUP_SIZE
    from shared.encoder import SharedEncoder
    from shared.masking import random_mask
    from shared.positional import LidarPositionalEmbedding
    from rgb.tokenizer import RGBPatchEmbedding
    from rgb.decoder import RGBDecoder
    from rgb.patches import patchify, normalize_patches
    from lidar.tokenizer import LidarPatchEncoder
    from lidar.decoder import LidarDecoder
    from data.lidar_helpers import make_patches_batch


class OmniVec2Stage1(nn.Module):
    """Full Stage 1 masked autoencoder — shared encoder, two modalities."""

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.rgb_tokenizer       = RGBPatchEmbedding()
        self.rgb_decoder         = RGBDecoder()
        self.lidar_patch_encoder = LidarPatchEncoder()
        self.lidar_pos_embed     = LidarPositionalEmbedding()
        self.lidar_decoder       = LidarDecoder()

    def forward_rgb(self, imgs, mask_ratio):
        tokens = self.rgb_tokenizer(imgs)
        x_vis, ids_restore, _, mask = random_mask(tokens, mask_ratio)
        latent = self.encoder(x_vis)
        pred   = self.rgb_decoder(latent, ids_restore)
        with torch.no_grad():
            raw_patches = patchify(imgs)
            target, _, _ = normalize_patches(raw_patches)
        return pred, target, mask

    def forward_lidar(self, points, mask_ratio):
        patches, centers = make_patches_batch(points, NUM_GROUP, GROUP_SIZE)
        tokens = (self.lidar_patch_encoder(patches)
                  + self.lidar_pos_embed(centers))
        x_vis, ids_restore, _, mask = random_mask(tokens, mask_ratio)
        latent = self.encoder(x_vis)
        pred   = self.lidar_decoder(latent, ids_restore, centers)
        with torch.no_grad():
            target = patches.reshape(patches.shape[0], NUM_GROUP, -1)
            target, _, _ = normalize_patches(target)
        return pred, target, mask, patches, centers

    def export_stage1_backbone(self):
        """Paper-faithful Stage 1 export: tokenizers + shared encoder only."""
        return {
            "encoder":             self.encoder.state_dict(),
            "rgb_tokenizer":       self.rgb_tokenizer.state_dict(),
            "lidar_patch_encoder": self.lidar_patch_encoder.state_dict(),
            "lidar_pos_embed":     self.lidar_pos_embed.state_dict(),
        }

    def save_pretrained(self, path):
        state = self.export_stage1_backbone()
        torch.save(state, path)
        print(f"Pretrained weights saved → {path}")
