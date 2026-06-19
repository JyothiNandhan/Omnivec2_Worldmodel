"""nuScenes temporal samples for OmniVec2 world-model training."""
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nuscenes.utils.data_classes import LidarPointCloud

try:
    from ..config import IMG_SIZE, N_POINTS
except ImportError:
    from config import IMG_SIZE, N_POINTS


class NuScenesWorldModelDataset(Dataset):
    """
    Returns consecutive frames from a single nuScenes scene.

    Output:
        rgb_sequence:   (history, 3, H, W)
        lidar_sequence: (history, N_POINTS, 3)
        rgb_target:     (3, H, W)
        lidar_target:   (N_POINTS, 3)
    """

    def __init__(self, nusc, scene_tokens: set, dataroot: str, history: int = 4, steps_ahead: int = 1):
        super().__init__()
        if history < 1:
            raise ValueError("history must be >= 1")
        if steps_ahead < 1:
            raise ValueError("steps_ahead must be >= 1")

        self.nusc = nusc
        self.dataroot = dataroot
        self.history = history
        self.steps_ahead = steps_ahead
        self.items = []

        # ── Step 1: collect every sample token across selected scenes ─────────
        # Build scene→sample_tokens map and a flat set of all unique tokens.
        scene_sample_map = {}
        all_tokens = set()
        for scene in nusc.scene:
            if scene["token"] not in scene_tokens:
                continue
            toks = []
            tok = scene["first_sample_token"]
            while tok:
                toks.append(tok)
                all_tokens.add(tok)
                s = nusc.get("sample", tok)
                tok = s["next"] if s["next"] != "" else None
            scene_sample_map[scene["token"]] = toks

        # ── Step 2: Build windows directly (Skipping slow startup pre-check) ──
        # We rely entirely on the runtime try/except fallback in __getitem__ 
        # to handle missing NFS files, which allows the script to start instantly.
        window = history + steps_ahead
        for sample_tokens in scene_sample_map.values():
            for start in range(0, max(0, len(sample_tokens) - window + 1)):
                window_tokens = sample_tokens[start:start + window]
                self.items.append(window_tokens)

        print(
            f"[WorldModelDataset] Instantiated with {len(self.items)} temporal windows. "
            f"(Runtime fallback active for missing files)."
        )



    def __len__(self):
        return len(self.items)

    def _load_rgb(self, sample_token):
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        img_path = os.path.join(self.dataroot, cam_data["filename"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)

    def _load_lidar(self, sample_token):
        sample = self.nusc.get("sample", sample_token)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_path = os.path.join(self.dataroot, lidar_data["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points[:3].T.astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        replace = pts.shape[0] < N_POINTS
        choice = np.random.choice(pts.shape[0], N_POINTS, replace=replace)
        return torch.from_numpy(pts[choice])

    def _load_ego_state(self, sample_token):
        sample = self.nusc.get("sample", sample_token)
        cam_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
        
        # translation is [x, y, z] in meters
        trans = torch.tensor(ego_pose["translation"], dtype=torch.float32)
        # rotation is [w, x, y, z] quaternion
        rot = torch.tensor(ego_pose["rotation"], dtype=torch.float32)
        return torch.cat([trans, rot], dim=0) # Shape: (7,)

    def __getitem__(self, idx):
        # Safety net: /orange NFS storage can drop files after startup pre-check.
        # On failure, pick a random different index and retry (max 10 attempts).
        for attempt in range(10):
            try:
                tokens       = self.items[idx]
                input_tokens = tokens[:self.history]
                target_token = tokens[-1]

                rgb_sequence   = torch.stack([self._load_rgb(tok)   for tok in input_tokens])
                lidar_sequence = torch.stack([self._load_lidar(tok) for tok in input_tokens])
                
                # Extract Ego-Motion sequence (T, 7)
                ego_sequence = torch.stack([self._load_ego_state(tok) for tok in input_tokens])
                # CRITICAL: Zero-center the trajectory to stabilize gradients
                # We subtract the (x,y,z) of T=0 from all frames in the window
                ego_sequence[:, :3] = ego_sequence[:, :3] - ego_sequence[0, :3]

                return {
                    "rgb_sequence":        rgb_sequence,
                    "lidar_sequence":      lidar_sequence,
                    "ego_sequence":        ego_sequence,
                    "rgb_target":          self._load_rgb(target_token),
                    "lidar_target":        self._load_lidar(target_token),
                    "target_sample_token": target_token,
                }
            except (FileNotFoundError, OSError):
                # NFS file unavailable — silently fall back to a random item
                idx = int(torch.randint(len(self.items), (1,)).item())

        raise RuntimeError(
            f"[WorldModelDataset] Failed to load a valid sample after 10 attempts. "
            "Check that /orange storage is mounted and accessible."
        )


def build_world_model_dataloaders(
    nusc,
    dataroot: str,
    batch_size: int,
    num_workers: int,
    split_ratio: float = 0.8,
    scene_limit: int = 0,
    seed: int = 42,
    history: int = 4,
    steps_ahead: int = 1,
):
    """Scene-based split for temporal world-model training."""
    all_scenes = list(nusc.scene)
    if scene_limit and scene_limit > 0:
        all_scenes = all_scenes[:scene_limit]

    random.Random(seed).shuffle(all_scenes)
    if len(all_scenes) < 2:
        raise ValueError("Need at least 2 scenes after applying scene_limit.")

    n_train = max(1, min(int(len(all_scenes) * split_ratio), len(all_scenes) - 1))
    train_tokens = {s["token"] for s in all_scenes[:n_train]}
    val_tokens   = {s["token"] for s in all_scenes[n_train:]}

    train_ds = NuScenesWorldModelDataset(nusc, train_tokens, dataroot, history, steps_ahead)
    val_ds   = NuScenesWorldModelDataset(nusc, val_tokens,   dataroot, history, steps_ahead)

    # Two DataLoaders share the node's CPU slots — split workers evenly so
    # PyTorch's total worker count stays within the system limit.
    workers_per_loader = max(1, num_workers // 2)
    kw = dict(
        batch_size=batch_size,
        num_workers=workers_per_loader,
        pin_memory=True,
        persistent_workers=workers_per_loader > 0,
    )
    train_dl = DataLoader(train_ds, shuffle=True,  drop_last=True, **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False,                  **kw)

    print(
        f"[WorldModel] Scenes: {len(all_scenes)} total "
        f"({len(train_tokens)} train / {len(val_tokens)} val)"
    )
    print(f"[WorldModel] Samples: {len(train_ds)} train / {len(val_ds)} val")
    return train_dl, val_dl
