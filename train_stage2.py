"""
Dedicated Stage 2 entrypoint:
load Stage 1 weights, then train Stage 2 up to g(.).
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from .checkpointing import load_training_checkpoint
    from .config import parse_args
    from .data.build import build_dataloaders
    from .model import OmniVec2Stage1
    from .stage2 import OmniVec2Stage2, train_stage2
except ImportError:
    from checkpointing import load_training_checkpoint
    from config import parse_args
    from data.build import build_dataloaders
    from model import OmniVec2Stage1
    from stage2 import OmniVec2Stage2, train_stage2

from nuscenes.nuscenes import NuScenes


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_stage1_init_path(args):
    if args.stage1_checkpoint:
        return args.stage1_checkpoint
    return os.path.join(args.checkpoints_dir, "stage1_best.pth")


def load_stage1_into_model(stage1_model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {path}")

    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not expose the weights_only argument.
        state = torch.load(path, map_location=DEVICE)

    if "model_state_dict" in state:
        stage1_model.load_state_dict(state["model_state_dict"])
        return

    expected_keys = ["encoder", "rgb_tokenizer", "lidar_patch_encoder", "lidar_pos_embed"]
    if all(key in state for key in expected_keys):
        stage1_model.encoder.load_state_dict(state["encoder"])
        stage1_model.rgb_tokenizer.load_state_dict(state["rgb_tokenizer"])
        stage1_model.lidar_patch_encoder.load_state_dict(state["lidar_patch_encoder"])
        stage1_model.lidar_pos_embed.load_state_dict(state["lidar_pos_embed"])
        return

    raise RuntimeError(
        f"Unsupported Stage 1 checkpoint format at {path}. "
        "Expected a training checkpoint or exported Stage 1 backbone."
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.exports_dir, exist_ok=True)

    print(f"Device      : {DEVICE}")
    print(f"NuScenes    : {args.version} @ {args.dataroot}")
    print(f"Output      : {args.output_dir}")
    print(f"Checkpoints : {args.checkpoints_dir}")
    print(f"Exports     : {args.exports_dir}")

    stage1_init = default_stage1_init_path(args)
    print(f"Stage 1 init: {stage1_init}")

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    train_rgb_dl, val_rgb_dl, train_lidar_dl, val_lidar_dl = build_dataloaders(
        nusc,
        args.dataroot,
        args.batch_size,
        args.num_workers,
        split_ratio=args.train_split_ratio,
        scene_limit=args.scene_limit,
        seed=args.seed,
    )

    stage1_model = OmniVec2Stage1().to(DEVICE)
    load_stage1_into_model(stage1_model, stage1_init)
    print("Loaded Stage 1 weights successfully.")

    if args.stage2_epochs is not None:
        args.epochs = args.stage2_epochs
    if args.stage2_lr is not None:
        args.lr = args.stage2_lr

    model = OmniVec2Stage2(stage1_model).to(DEVICE)
    train_stage2(
        model,
        train_rgb_dl,
        val_rgb_dl,
        train_lidar_dl,
        val_lidar_dl,
        args,
    )

    torch.save(
        model.export_pretrained_fg(),
        os.path.join(args.exports_dir, "omnivec2_stage2_fg.pth"),
    )
    torch.save(
        model.state_dict(),
        os.path.join(args.exports_dir, "omnivec2_stage2_full.pth"),
    )
    print("\nStage 2 training complete.")


if __name__ == "__main__":
    main()
