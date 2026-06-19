#!/bin/bash
#SBATCH --job-name=omnivec2full_s2
#SBATCH --output=omnivec2full_s2_%j.log
#SBATCH --error=omnivec2full_s2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrepaka@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --account=iruchkin
#SBATCH --qos=iruchkin

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"

unset PYTHONHOME
unset PYTHONPATH
module load conda
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-omnivec2}}"
conda activate "$CONDA_ENV_NAME"
hash -r
PYTHON_BIN="$(command -v python)"

DATAROOT="${DATAROOT:-/orange/iruchkin/isen/nsfull}"
VERSION="${VERSION:-v1.0-trainval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
STAGE1_DIR="${OUTPUT_ROOT}/stage1_first500"
OUTPUT_DIR="${OUTPUT_ROOT}/stage2_first500"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${STAGE1_DIR}/checkpoints/stage1_best.pth}"
SCENE_LIMIT="${SCENE_LIMIT:-500}"
EPOCHS="${STAGE2_EPOCHS:-${EPOCHS:-50}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CHECKPOINT"
    echo "Run run_full_stage1_first500.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/checkpoints" "${OUTPUT_DIR}/exports"

echo "Stage 2 first ${SCENE_LIMIT} scenes"
echo "Folder     : $SCRIPT_DIR"
echo "Data root  : $DATAROOT"
echo "Version    : $VERSION"
echo "Output     : $OUTPUT_DIR"
echo "Stage1 ckpt: $STAGE1_CHECKPOINT"
echo "Epochs     : $EPOCHS"
echo "Batch size : $BATCH_SIZE"
echo "Workers    : $NUM_WORKERS"
echo "Python     : $PYTHON_BIN"

"$PYTHON_BIN" "${SCRIPT_DIR}/train_stage2.py" \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --output_dir "$OUTPUT_DIR" \
    --scene_limit "$SCENE_LIMIT" \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --auto_resume
