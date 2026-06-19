#!/bin/bash
# SBATCH script to generate bounding box visualizations from Stage-3 Without re-training

#SBATCH --job-name=vis_stage3
#SBATCH --output=vis_stage3_%j.log
#SBATCH --error=vis_stage3_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --account=iruchkin
#SBATCH --qos=iruchkin

set -euo pipefail

echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-local}"
echo "Start:     $(date)"

unset PYTHONHOME
unset PYTHONPATH
module load conda
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-omnivec2}}"
conda activate "${CONDA_ENV_NAME}"
PYTHON_BIN="$(command -v python)"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SCRIPT_DIR}"

NUSCENES_ROOT="${NUSCENES_ROOT:-/orange/iruchkin/isen/nsfull}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/stage3_101_150}"
SCENE_TOKENS="${SCRIPT_DIR}/scene_tokens_101_150.txt"

# The checkpoint is found inside exports/
CHECKPOINT="${OUTPUT_DIR}/exports/omnivec2_stage3_full.pth"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: Stage 3 checkpoint not found at ${CHECKPOINT}"
  echo "Please let your training finish first!"
  exit 1
fi

echo "Running visualize_stage3.py to generate Bounding Boxes..."
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/visualize_stage3.py" \
  --dataroot "${NUSCENES_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --scene_tokens_file "${SCENE_TOKENS}" \
  --stage3_seg_num_classes 11 \
  --batch_size 8 \
  --num_workers 4

echo "Visualization Completed! Check your runs/stage3_101_150/ folder for the image files."
