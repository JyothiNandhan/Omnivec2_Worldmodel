"""
NuScenes RGB camera dataset.
Loads images from all 6 cameras, resized to IMG_SIZE × IMG_SIZE.
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    from ..config import CAMERAS, IMG_SIZE
except ImportError:
    from config import CAMERAS, IMG_SIZE


class NuScenesRGBDataset(Dataset):
    """RGB camera images from NuScenes (all 6 cameras per sample)."""

    def __init__(self, nusc, scene_tokens: set, dataroot: str):
        super().__init__()
        self.nusc = nusc
        self.dataroot = dataroot
        self.items = []   # (cam_data_token, cam_name)
        for scene in nusc.scene:
            if scene["token"] not in scene_tokens:
                continue
            tok = scene["first_sample_token"]
            while tok:
                sample = nusc.get("sample", tok)
                for cam in CAMERAS:
                    self.items.append((sample["data"][cam], cam))
                tok = sample["next"] if sample["next"] != "" else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cam_data_token, _ = self.items[idx]
        cam_data = self.nusc.get("sample_data", cam_data_token)
        img_path = os.path.join(self.dataroot, cam_data["filename"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)   # (3, H, W)
