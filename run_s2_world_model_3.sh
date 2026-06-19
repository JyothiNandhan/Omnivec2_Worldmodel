#!/bin/bash
#SBATCH --job-name=omnivec2_s2_wm3
#SBATCH --output=omnivec2_s2_wm3_%j.log
#SBATCH --error=omnivec2_s2_wm3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrepaka@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --account=iruchkin
#SBATCH --qos=iruchkin

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-omnivec2}}"
mkdir -p "$MPLCONFIGDIR"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"

unset PYTHONHOME
unset PYTHONPATH
module load conda
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-omnivec2_fix}}"
conda activate "$CONDA_ENV_NAME"
hash -r
PYTHON_BIN="$(command -v python)"

DATAROOT="${DATAROOT:-/orange/iruchkin/isen/nsfull}"
VERSION="${VERSION:-v1.0-trainval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_first500"
OUTPUT_DIR="${OUTPUT_ROOT}/s2_world_model_4"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${STAGE2_DIR}/checkpoints/stage2_best.pth}"

WM3_RESUME="${OUTPUT_DIR}/checkpoints/wm_s2_last.pth"
WM2_RESUME="${OUTPUT_ROOT}/s2_world_model_2/checkpoints/wm_s2_last.pth"
WARM_START_WM2="${WARM_START_WM2:-0}"

SCENE_LIMIT="${SCENE_LIMIT:-500}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
HISTORY="${HISTORY:-4}"
STEPS_AHEAD="${STEPS_AHEAD:-2}"

# v5 architecture (3× wider internal dim, deeper)
TEMPORAL_LAYERS="${TEMPORAL_LAYERS:-6}"
TEMPORAL_HEADS="${TEMPORAL_HEADS:-8}"
TEMPORAL_INNER_DIM="${TEMPORAL_INNER_DIM:-256}"

CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
RGB_PATCH_WEIGHT="${RGB_PATCH_WEIGHT:-0.5}"   # reduced: pixel loss now has direct gradient path
SSIM_WEIGHT="${SSIM_WEIGHT:-0.0}"
LIDAR_WEIGHT="${LIDAR_WEIGHT:-1.5}"
VARIANCE_WEIGHT="${VARIANCE_WEIGHT:-0.2}"

if [[ ! -f "$STAGE2_CHECKPOINT" ]]; then
    echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CHECKPOINT"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/checkpoints"

echo "Stage 2 -> World Model (v5: bigger predictor, pixel head connected)"
echo "Arch        : inner_dim=$TEMPORAL_INNER_DIM, layers=$TEMPORAL_LAYERS, heads=$TEMPORAL_HEADS"
echo "Folder      : $SCRIPT_DIR"
echo "Data root   : $DATAROOT"
echo "Output      : $OUTPUT_DIR"
echo "Stage2 ckpt : $STAGE2_CHECKPOINT"
echo "WM resume   : $WM3_RESUME (fresh start if missing)"
echo "Epochs      : $EPOCHS"
echo "Batch size  : $BATCH_SIZE"
echo "Contrastive : $CONTRASTIVE_WEIGHT"
echo "RGB patch wt: $RGB_PATCH_WEIGHT"
echo "SSIM weight : $SSIM_WEIGHT"
echo "LiDAR weight: $LIDAR_WEIGHT"
echo "Variance wt : $VARIANCE_WEIGHT"
echo "Python      : $PYTHON_BIN"

RESUME_FLAG=""
if [[ -f "$WM3_RESUME" ]]; then
    echo "Auto-resuming from WM3 checkpoint: $WM3_RESUME"
    RESUME_FLAG="--auto_resume"
elif [[ "$WARM_START_WM2" == "1" && -f "$WM2_RESUME" ]]; then
    echo "Warm-starting from WM2 checkpoint: $WM2_RESUME"
    RESUME_FLAG="--resume $WM2_RESUME"
fi

"$PYTHON_BIN" "${SCRIPT_DIR}/train_s2_world_model_3.py" \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --output_dir "$OUTPUT_DIR" \
    --scene_limit "$SCENE_LIMIT" \
    --stage2_checkpoint "$STAGE2_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --history "$HISTORY" \
    --steps_ahead "$STEPS_AHEAD" \
    --temporal_layers "$TEMPORAL_LAYERS" \
    --temporal_heads "$TEMPORAL_HEADS" \
    --temporal_inner_dim "$TEMPORAL_INNER_DIM" \
    --contrastive_weight "$CONTRASTIVE_WEIGHT" \
    --rgb_patch_weight "$RGB_PATCH_WEIGHT" \
    --ssim_weight "$SSIM_WEIGHT" \
    --lidar_weight "$LIDAR_WEIGHT" \
    --variance_weight "$VARIANCE_WEIGHT" \
    $RESUME_FLAG
