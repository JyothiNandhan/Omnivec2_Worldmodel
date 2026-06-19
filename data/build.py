"""
Dataloader builder — scene-based split, 4 loaders (train/val × RGB/LiDAR).
"""
import random

from torch.utils.data import DataLoader

from .rgb_dataset import NuScenesRGBDataset
from .lidar_dataset import NuScenesLidarDataset
def build_dataloaders(nusc, dataroot, batch_size, num_workers,
                      split_ratio=0.8, scene_limit=0, seed=42):
    """Scene-based split → 4 DataLoaders."""
    all_scenes = list(nusc.scene)
    if scene_limit and scene_limit > 0:
        all_scenes = all_scenes[:scene_limit]

    random.Random(seed).shuffle(all_scenes)

    if len(all_scenes) < 2:
        raise ValueError("Need at least 2 scenes after applying scene_limit.")

    n_train = int(len(all_scenes) * split_ratio)
    n_train = max(1, min(n_train, len(all_scenes) - 1))
    train_scenes = all_scenes[:n_train]
    val_scenes = all_scenes[n_train:]
    train_tokens = {s["token"] for s in train_scenes}
    val_tokens   = {s["token"] for s in val_scenes}

    train_rgb   = NuScenesRGBDataset(nusc, train_tokens, dataroot)
    val_rgb     = NuScenesRGBDataset(nusc, val_tokens, dataroot)
    train_lidar = NuScenesLidarDataset(nusc, train_tokens, dataroot)
    val_lidar   = NuScenesLidarDataset(nusc, val_tokens, dataroot)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_rgb_dl   = DataLoader(train_rgb,   shuffle=True,  drop_last=True, **kw)
    val_rgb_dl     = DataLoader(val_rgb,     shuffle=False, **kw)
    train_lidar_dl = DataLoader(train_lidar, shuffle=True,  drop_last=True, **kw)
    val_lidar_dl   = DataLoader(val_lidar,   shuffle=False, **kw)

    print(f"Selected scenes: {len(all_scenes)}")
    print(f"Train scenes: {len(train_tokens)} | Val scenes: {len(val_tokens)}")
    print(f"Selected scenes shuffled with seed {seed}")
    print(f"Train: {len(train_rgb)} RGB imgs, {len(train_lidar)} LiDAR clouds")
    print(f"Val:   {len(val_rgb)} RGB imgs, {len(val_lidar)} LiDAR clouds")
    return train_rgb_dl, val_rgb_dl, train_lidar_dl, val_lidar_dl
