"""
Configuration & CLI arguments for OmniVec2 Stage 1.
All hyperparameters live here — one place to tweak everything.
"""
import argparse
import os

# ═══════════════════ Data ═══════════════════
NUSCENES_DATAROOT = "/orange/iruchkin/isen/nsfull"
NUSCENES_VERSION  = "v1.0-trainval"
SCENE_LIMIT = 100
TRAIN_SPLIT_RATIO = 0.8
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]

# ═══════════════════ RGB ════════════════════
IMG_SIZE   = 224
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196

# ═══════════════════ LiDAR ══════════════════
N_POINTS   = 1024
NUM_GROUP  = 64
GROUP_SIZE = 32

# ═══════════════════ Shared Encoder ═════════
EMBED_DIM  = 256
NUM_HEADS  = 8
ENC_LAYERS = 6
FC_HIDDEN  = 128

# ═══════════════════ Stage 2 g(.) ═══════════
G_NUM_HEADS = 4
G_LAYERS    = 6
CA_HEADS    = 4

# ═══════════════════ Decoders ═══════════════
DEC_EMBED_DIM    = 128
DEC_HEADS        = 4
DEC_LAYERS_RGB   = 4
DEC_LAYERS_LIDAR = 2

# ═══════════════════ Training ═══════════════
MASK_RATIO_RGB   = 0.75
MASK_RATIO_LIDAR = 0.75
EPOCHS     = 30
STAGE1_EPOCHS = None
STAGE2_EPOCHS = None
STAGE3_EPOCHS = 35
LR         = 1e-4
STAGE2_LR  = None
STAGE3_EARLY_STOPPING_PATIENCE = None
STAGE3_TASK_MODE = "segmentation"
STAGE3_SEG_NUM_CLASSES = 11
STAGE3_SEG_LOSS_WEIGHT = 1.0
BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0
KEEP_LAST_N_CHECKPOINTS = 4
SAVE_EVERY = 1
CHECKPOINT_DIRNAME = "checkpoints"
EXPORT_DIRNAME = "exports"


def parse_args():
    """CLI arguments — override any default above at runtime."""
    p = argparse.ArgumentParser(
        description="OmniVec2 Stage 1 — RGB + LiDAR Masked Pretraining")
    # Data
    p.add_argument("--dataroot",   type=str, default=NUSCENES_DATAROOT)
    p.add_argument("--version",    type=str, default=NUSCENES_VERSION)
    p.add_argument("--output_dir", type=str, default="./results_stage1")
    p.add_argument("--scene_limit", type=int, default=SCENE_LIMIT,
                   help="Use only the first N scenes in dataset order. 0 uses all scenes.")
    p.add_argument("--train_split_ratio", type=float, default=TRAIN_SPLIT_RATIO,
                   help="Fraction of selected scenes used for training.")
    p.add_argument("--checkpoints_dir", type=str, default=None,
                   help="Directory for resume checkpoints. Defaults to <output_dir>/checkpoints.")
    p.add_argument("--exports_dir", type=str, default=None,
                   help="Directory for exported pretrained weights. Defaults to <output_dir>/exports.")
    # Training
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--stage1_epochs", type=int, default=STAGE1_EPOCHS,
                   help="Override epochs for Stage 1 when using a combined pipeline.")
    p.add_argument("--stage2_epochs", type=int, default=STAGE2_EPOCHS,
                   help="Override epochs for Stage 2 when using a combined pipeline.")
    p.add_argument("--stage3_epochs", type=int, default=STAGE3_EPOCHS,
                   help="Override epochs for Stage 3 when using a dedicated Stage 3 run.")
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--stage2_lr",  type=float, default=STAGE2_LR,
                   help="Override learning rate for Stage 2 when using a combined pipeline.")
    p.add_argument("--stage3_early_stopping_patience", type=int, default=STAGE3_EARLY_STOPPING_PATIENCE,
                   help="Stage 3 only: stop if validation loss does not improve for N epochs.")
    p.add_argument("--stage3_task_mode", type=str, default=STAGE3_TASK_MODE,
                   choices=["classification", "segmentation", "multitask"],
                   help="Stage 3 task mode. Default runs segmentation only while retaining classification code.")
    p.add_argument("--stage3_seg_num_classes", type=int, default=STAGE3_SEG_NUM_CLASSES,
                   help="Number of RGB segmentation classes used by Stage 3 segmentation mode.")
    p.add_argument("--stage3_seg_loss_weight", type=float, default=STAGE3_SEG_LOSS_WEIGHT,
                   help="Weight applied to Stage 3 RGB segmentation loss in multitask mode.")
    p.add_argument("--bbox_side", type=str, default="all", choices=["left", "right", "all"],
                   help="When visualizing bounding boxes, filter boxes on the left or right half (or show all).")
    p.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    p.add_argument("--scene_tokens_file", type=str, default="",
                   help="Path to a newline-separated file with nuScenes scene tokens to use for the run.")
    p.add_argument("--num_workers",type=int,   default=NUM_WORKERS)
    p.add_argument("--seed",       type=int,   default=SEED)
    p.add_argument("--mask_ratio_rgb",   type=float, default=MASK_RATIO_RGB)
    p.add_argument("--mask_ratio_lidar", type=float, default=MASK_RATIO_LIDAR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=KEEP_LAST_N_CHECKPOINTS)
    p.add_argument("--save_every", type=int, default=SAVE_EVERY,
                   help="Save rolling/last checkpoints every N epochs.")
    p.add_argument("--resume", type=str, default="",
                   help="Checkpoint path to resume from.")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Checkpoint path to load for visualization/evaluation.")
    p.add_argument("--stage1_checkpoint", type=str, default="",
                   help="Stage 1 checkpoint/export to initialize Stage 2 from.")
    p.add_argument("--stage2_checkpoint", type=str, default="",
                   help="Stage 2 checkpoint/export to initialize Stage 3 from.")
    p.add_argument("--auto_resume", action="store_true",
                   help="Resume automatically from <checkpoints_dir>/stage1_last.pth if present.")
    p.add_argument("--skip_pretraining_visuals", action="store_true",
                   help="Skip tokenizer visualizations before training.")
    p.add_argument("--skip_posttraining_visuals", action="store_true",
                   help="Skip reconstruction/curve visualizations after training.")
    args = p.parse_args()

    if not 0.0 < args.train_split_ratio < 1.0:
        raise ValueError("--train_split_ratio must be between 0 and 1.")
    if not 0.0 <= args.mask_ratio_rgb < 1.0:
        raise ValueError("--mask_ratio_rgb must be in [0, 1).")
    if not 0.0 <= args.mask_ratio_lidar < 1.0:
        raise ValueError("--mask_ratio_lidar must be in [0, 1).")
    if args.keep_last_n_checkpoints < 1:
        raise ValueError("--keep_last_n_checkpoints must be at least 1.")
    if args.save_every < 1:
        raise ValueError("--save_every must be at least 1.")
    if args.stage3_early_stopping_patience is not None and args.stage3_early_stopping_patience < 1:
        raise ValueError("--stage3_early_stopping_patience must be at least 1 when provided.")
    if args.stage3_seg_num_classes < 2:
        raise ValueError("--stage3_seg_num_classes must be at least 2.")
    if args.stage3_seg_loss_weight <= 0.0:
        raise ValueError("--stage3_seg_loss_weight must be positive.")

    if args.checkpoints_dir is None:
        args.checkpoints_dir = os.path.join(args.output_dir, CHECKPOINT_DIRNAME)
    if args.exports_dir is None:
        args.exports_dir = os.path.join(args.output_dir, EXPORT_DIRNAME)
    return args
