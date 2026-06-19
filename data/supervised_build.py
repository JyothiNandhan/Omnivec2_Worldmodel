import random
import numpy as np
from torch.utils.data import DataLoader
from .supervised_dataset import (
    BACKGROUND_CLASS_ID,
    MAIN_CATEGORIES,
    NuScenesRGBSupervisedDataset,
    NuScenesLidarSupervisedDataset,
    coarse_category_id,
)


def _annotation_summary(nusc, scene_tokens):
    counts = {name: 0 for name in MAIN_CATEGORIES}
    counts["background"] = 0
    for scene in nusc.scene:
        if scene["token"] not in scene_tokens:
            continue
        tok = scene["first_sample_token"]
        while tok:
            sample = nusc.get("sample", tok)
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                cls_id = coarse_category_id(ann["category_name"])
                if cls_id == BACKGROUND_CLASS_ID:
                    counts["background"] += 1
                else:
                    counts[MAIN_CATEGORIES[cls_id]] += 1
            tok = sample["next"] if sample["next"] != "" else None
    return counts


def _format_nonzero_counts(counts):
    pairs = [(name, count) for name, count in counts.items() if count > 0]
    if not pairs:
        return "none"
    return ", ".join(f"{name}:{count}" for name, count in pairs)


def _segmentation_pixel_summary(dataset, max_items=64):
    counts = np.zeros(len(MAIN_CATEGORIES) + 1, dtype=np.int64)
    total_items = min(len(dataset), max_items)
    for idx in range(total_items):
        labels = dataset[idx]["segmentation"].numpy().reshape(-1)
        counts += np.bincount(labels, minlength=len(counts))
    total = int(counts.sum())
    foreground = int(counts[:BACKGROUND_CLASS_ID].sum())
    background = int(counts[BACKGROUND_CLASS_ID])
    if total == 0:
        return "no sampled pixels"

    foreground_parts = []
    for class_idx, class_name in enumerate(MAIN_CATEGORIES):
        if counts[class_idx] > 0:
            pct = 100.0 * counts[class_idx] / total
            foreground_parts.append(f"{class_name}:{int(counts[class_idx])} ({pct:.2f}%)")
    fg_text = ", ".join(foreground_parts) if foreground_parts else "none"
    return (
        f"sampled_items:{total_items}, foreground:{foreground} ({100.0 * foreground / total:.2f}%), "
        f"background:{background} ({100.0 * background / total:.2f}%), classes: {fg_text}"
    )


def build_supervised_dataloaders(nusc, dataroot, batch_size, num_workers,
                                 split_ratio=0.8, scene_limit=0, seed=42,
                                 scene_tokens=None):
    """Scene-based split → 4 Supervised DataLoaders returning (Tensor, Label).

    If `scene_tokens` is provided, those scenes become the selected pool and are
    split into train/val using the same shuffle + split_ratio behavior as Stage 1/2.
    """
    if scene_tokens is not None:
        selected_tokens = set(scene_tokens)
        all_scenes = [s for s in nusc.scene if s["token"] in selected_tokens]

        missing_tokens = selected_tokens - {s["token"] for s in all_scenes}
        if missing_tokens:
            raise ValueError(f"{len(missing_tokens)} scene_tokens were not found in nuScenes metadata.")
    else:
        all_scenes = list(nusc.scene)
        if scene_limit and scene_limit > 0:
            all_scenes = all_scenes[:scene_limit]

    random.Random(seed).shuffle(all_scenes)

    if len(all_scenes) < 2:
        raise ValueError("Need at least 2 scenes after applying scene selection.")

    n_train = int(len(all_scenes) * split_ratio)
    n_train = max(1, min(n_train, len(all_scenes) - 1))
    train_scenes = all_scenes[:n_train]
    val_scenes = all_scenes[n_train:]
    train_tokens = {s["token"] for s in train_scenes}
    val_tokens = {s["token"] for s in val_scenes}

    train_rgb = NuScenesRGBSupervisedDataset(nusc, train_tokens, dataroot)
    val_rgb = NuScenesRGBSupervisedDataset(nusc, val_tokens, dataroot)
    train_lidar = NuScenesLidarSupervisedDataset(nusc, train_tokens, dataroot)
    val_lidar = NuScenesLidarSupervisedDataset(nusc, val_tokens, dataroot)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
              
    train_rgb_dl = DataLoader(train_rgb, shuffle=True, drop_last=True, **kw)
    val_rgb_dl = DataLoader(val_rgb, shuffle=False, **kw)
    train_lidar_dl = DataLoader(train_lidar, shuffle=True, drop_last=True, **kw)
    val_lidar_dl = DataLoader(val_lidar, shuffle=False, **kw)

    print(f"Selected scenes: {len(all_scenes)}")
    print(f"Train scenes: {len(train_tokens)} | Val scenes: {len(val_tokens)}")
    print(f"Selected scenes shuffled with seed {seed}")
    print(f"Train: {len(train_rgb)} RGB imgs, {len(train_lidar)} LiDAR clouds")
    print(f"Val:   {len(val_rgb)} RGB imgs, {len(val_lidar)} LiDAR clouds")
    print(f"Train annotations: {_format_nonzero_counts(_annotation_summary(nusc, train_tokens))}")
    print(f"Val annotations:   {_format_nonzero_counts(_annotation_summary(nusc, val_tokens))}")
    print(f"Train seg pixels: {_segmentation_pixel_summary(train_rgb)}")
    print(f"Val seg pixels:   {_segmentation_pixel_summary(val_rgb)}")

    return train_rgb_dl, val_rgb_dl, train_lidar_dl, val_lidar_dl
