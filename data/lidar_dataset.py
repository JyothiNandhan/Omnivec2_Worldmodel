"""
NuScenes LiDAR point cloud dataset.
Returns raw point clouds (N_POINTS, 3) from the LIDAR_TOP sensor.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from nuscenes.utils.data_classes import LidarPointCloud

try:
    from ..config import N_POINTS
except ImportError:
    from config import N_POINTS


class NuScenesLidarDataset(Dataset):
    """LiDAR point clouds from NuScenes (LIDAR_TOP sensor)."""

    def __init__(self, nusc, scene_tokens: set, dataroot: str):
        super().__init__()
        self.nusc = nusc
        self.dataroot = dataroot
        self.items = []
        for scene in nusc.scene:
            if scene["token"] not in scene_tokens:
                continue
            tok = scene["first_sample_token"]
            while tok:
                sample = nusc.get("sample", tok)
                self.items.append(sample["data"]["LIDAR_TOP"])
                tok = sample["next"] if sample["next"] != "" else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sd = self.nusc.get("sample_data", self.items[idx])
        lidar_path = os.path.join(self.dataroot, sd["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points[:3].T.astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        replace = pts.shape[0] < N_POINTS
        choice = np.random.choice(pts.shape[0], N_POINTS, replace=replace)
        return torch.from_numpy(pts[choice])   # (N_POINTS, 3)
