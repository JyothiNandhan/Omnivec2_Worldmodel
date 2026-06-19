#!/bin/bash
#SBATCH --job-name=omnivec2_s2_wm
#SBATCH --output=omnivec2_s2_wm_%j.log
#SBATCH --error=omnivec2_s2_wm_%j.err
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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-omnivec2_fix}}"
conda activate "$CONDA_ENV_NAME"
hash -r
PYTHON_BIN="$(command -v python)"

DATAROOT="${DATAROOT:-/orange/iruchkin/isen/nsfull}"
VERSION="${VERSION:-v1.0-trainval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
STAGE2_DIR="${OUTPUT_ROOT}/stage2_first500"
OUTPUT_DIR="${OUTPUT_ROOT}/s2_world_model_2"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-${STAGE2_DIR}/checkpoints/stage2_best.pth}"

SCENE_LIMIT="${SCENE_LIMIT:-500}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
HISTORY="${HISTORY:-4}"
STEPS_AHEAD="${STEPS_AHEAD:-2}"  # 2 steps from T-1 = T+1 = 0.5s ahead of current frame

if [[ ! -f "$STAGE2_CHECKPOINT" ]]; then
    echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CHECKPOINT"
    echo "Wait for run_full_stage2_first500.sh to finish."
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/checkpoints"

echo "Stage 2 -> World Model"
echo "Folder     : $SCRIPT_DIR"
echo "Data root  : $DATAROOT"
echo "Output     : $OUTPUT_DIR"
echo "Stage2 ckpt: $STAGE2_CHECKPOINT"
echo "Epochs     : $EPOCHS"
echo "Batch size : $BATCH_SIZE"
echo "History    : $HISTORY frames"
echo "Predicts   : $STEPS_AHEAD steps ahead"
echo "Python     : $PYTHON_BIN"

"$PYTHON_BIN" "${SCRIPT_DIR}/train_s2_world_model_2.py" \
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
    --auto_resume
