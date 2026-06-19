import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

try:
    from ..config import CAMERAS, IMG_SIZE, N_POINTS, PATCH_SIZE
except ImportError:
    from config import CAMERAS, IMG_SIZE, N_POINTS, PATCH_SIZE


# Use the nuScenes detection class roots (10 classes) as the semantic categories.
# These are the common detection categories used by nuScenes:
# car, truck, bus, trailer, construction_vehicle, pedestrian,
# motorcycle, bicycle, traffic_cone, barrier
MAIN_CATEGORIES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]
BACKGROUND_CLASS_ID = len(MAIN_CATEGORIES)
NUM_SEG_CLASSES = BACKGROUND_CLASS_ID + 1

NUSCENES_CATEGORY_TO_MAIN = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "human.pedestrian": "pedestrian",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}


def coarse_category_id(category_name: str) -> int:
    """Map a nuScenes `category_name` (e.g. 'vehicle.car') to a coarse integer id.

    If the root category is not in the MAIN_CATEGORIES list, return the background id.
    """
    if category_name in MAIN_CATEGORIES:
        return MAIN_CATEGORIES.index(category_name)

    for prefix, main_category in NUSCENES_CATEGORY_TO_MAIN.items():
        if category_name == prefix or category_name.startswith(prefix + "."):
            return MAIN_CATEGORIES.index(main_category)

    return BACKGROUND_CLASS_ID


def get_dominant_category(nusc, sample_token):
    """Extracts annotations for a sample and returns the dominant category ID.

    Returns an integer in [0, len(MAIN_CATEGORIES)] where the last id is background/none.
    """
    sample = nusc.get("sample", sample_token)
    counts = {k: 0 for k in MAIN_CATEGORIES}
    counts["background"] = 0

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        cls_id = coarse_category_id(ann["category_name"])
        if cls_id != BACKGROUND_CLASS_ID:
            counts[MAIN_CATEGORIES[cls_id]] += 1

    max_cat = max(counts.keys(), key=lambda k: counts[k])
    if counts[max_cat] == 0 or max_cat == "background":
        return BACKGROUND_CLASS_ID
    return MAIN_CATEGORIES.index(max_cat)


def make_patch_labels_from_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a dense semantic mask into one semantic label per RGB patch."""
    patch = PATCH_SIZE
    grid = IMG_SIZE // patch
    labels = np.zeros(grid * grid, dtype=np.int64)
    idx = 0
    for row in range(grid):
        for col in range(grid):
            patch_mask = mask[row * patch:(row + 1) * patch, col * patch:(col + 1) * patch]
            counts = np.bincount(patch_mask.reshape(-1), minlength=NUM_SEG_CLASSES)
            foreground_counts = counts[:BACKGROUND_CLASS_ID]
            if foreground_counts.sum() > 0:
                labels[idx] = int(np.argmax(foreground_counts))
            else:
                labels[idx] = BACKGROUND_CLASS_ID
            idx += 1
    return labels


def weak_camera_segmentation_mask(nusc, cam_data_token: str, dataroot: str) -> np.ndarray:
    """
    Build a weak semantic mask by projecting 3D boxes into the camera plane and filling
    their 2D extents with coarse category IDs.
    """
    cam_data = nusc.get("sample_data", cam_data_token)
    _, boxes, camera_intrinsic = nusc.get_sample_data(cam_data_token)
    width, height = int(cam_data["width"]), int(cam_data["height"])
    mask = np.full((height, width), BACKGROUND_CLASS_ID, dtype=np.int64)

    # Draw far boxes first so nearer boxes overwrite them.
    boxes = sorted(boxes, key=lambda box: float(box.center[2]), reverse=True)
    intrinsic = np.asarray(camera_intrinsic)
    for box in boxes:
        corners = box.corners()
        in_front = corners[2, :] > 1e-3
        if not np.any(in_front):
            continue

        projected = view_points(corners[:, in_front], intrinsic, normalize=True)[:2, :]
        if projected.shape[1] == 0:
            continue

        x1 = int(np.floor(np.min(projected[0])))
        y1 = int(np.floor(np.min(projected[1])))
        x2 = int(np.ceil(np.max(projected[0])))
        y2 = int(np.ceil(np.max(projected[1])))

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        label = coarse_category_id(box.name)
        mask[y1:y2 + 1, x1:x2 + 1] = label

    resized = Image.fromarray(mask.astype(np.uint8), mode="L").resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST
    )
    return np.array(resized, dtype=np.int64)


class NuScenesRGBSupervisedDataset(Dataset):
    """RGB dataset returning image, classification label, and dense weak segmentation mask."""

    def __init__(self, nusc, scene_tokens: set, dataroot: str):
        self.nusc = nusc
        self.dataroot = dataroot
        self.items = []
        for scene in nusc.scene:
            if scene["token"] not in scene_tokens:
                continue
            tok = scene["first_sample_token"]
            while tok:
                sample = nusc.get("sample", tok)
                cls_label = get_dominant_category(nusc, tok)
                for cam in CAMERAS:
                    self.items.append(
                        {
                            "sample_token": tok,
                            "cam_data_token": sample["data"][cam],
                            "cam_name": cam,
                            "class_label": cls_label,
                        }
                    )
                tok = sample["next"] if sample["next"] != "" else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        cam_data = self.nusc.get("sample_data", item["cam_data_token"])
        img_path = os.path.join(self.dataroot, cam_data["filename"])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        weak_mask = weak_camera_segmentation_mask(self.nusc, item["cam_data_token"], self.dataroot)
        return {
            "image": torch.from_numpy(img_np).permute(2, 0, 1),
            "class_label": torch.tensor(item["class_label"], dtype=torch.long),
            "segmentation": torch.from_numpy(weak_mask).long(),
        }


class NuScenesLidarSupervisedDataset(Dataset):
    """LiDAR dataset that returns point cloud and classification label for Stage 3 fusion."""

    def __init__(self, nusc, scene_tokens: set, dataroot: str):
        self.nusc = nusc
        self.dataroot = dataroot
        self.items = []
        for scene in nusc.scene:
            if scene["token"] not in scene_tokens:
                continue
            tok = scene["first_sample_token"]
            while tok:
                sample = nusc.get("sample", tok)
                label = get_dominant_category(nusc, tok)
                self.items.append({"lidar_token": sample["data"]["LIDAR_TOP"], "class_label": label})
                tok = sample["next"] if sample["next"] != "" else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sd = self.nusc.get("sample_data", item["lidar_token"])
        lidar_path = os.path.join(self.dataroot, sd["filename"])
        pc = LidarPointCloud.from_file(lidar_path)
        pts = pc.points[:3].T.astype(np.float32)
        pts = pts[np.isfinite(pts).all(axis=1)]
        replace = pts.shape[0] < N_POINTS
        choice = np.random.choice(pts.shape[0], N_POINTS, replace=replace)
        return {
            "points": torch.from_numpy(pts[choice]),
            "class_label": torch.tensor(item["class_label"], dtype=torch.long),
        }
