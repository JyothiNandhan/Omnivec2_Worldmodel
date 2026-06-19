#!/bin/bash
#SBATCH --job-name=omnivec2_s1
#SBATCH --output=omnivec2_s1_%j.log
#SBATCH --error=omnivec2_s1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrepaka@ufl.edu
#
# ── GPU Resources ──
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#
# ── Account ──
#SBATCH --account=iruchkin
#SBATCH --qos=iruchkin

set -euo pipefail

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:     $(date)"
echo "═══════════════════════════════════════════════════"

# ── Load modules / activate env ──
unset PYTHONHOME
unset PYTHONPATH
module load conda
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-${CONDA_DEFAULT_ENV:-omnivec2}}"
conda activate "$CONDA_ENV_NAME"
hash -r
PYTHON_BIN="$(command -v python)"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Could not resolve python executable after activating ${CONDA_ENV_NAME}."
    exit 1
fi

# ── Paths (edit these if needed) ──
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SCRIPT_DIR"
DATAROOT="${DATAROOT:-/orange/iruchkin/isen/nsfull}"
VERSION="${VERSION:-v1.0-trainval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/runs}"
RUN_NAME="${RUN_NAME:-stage1_nuscenes}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/checkpoints"
echo "Script dir: $SCRIPT_DIR"
echo "Data root : $DATAROOT"
echo "Version   : $VERSION"
echo "Output    : $OUTPUT_DIR"
echo "Conda env : $CONDA_ENV_NAME"
echo "Python    : $(which python)"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "PYTHON_BIN : $PYTHON_BIN"
echo "═══════════════════════════════════════════════════"

"$PYTHON_BIN" - <<'PY'
import encodings
import sys
print("Python preflight OK")
print("Executable:", sys.executable)
print("Prefix:", sys.prefix)
PY

# ── Run ──
"$PYTHON_BIN" "${SCRIPT_DIR}/train.py" \
    --dataroot   "$DATAROOT" \
    --version    "$VERSION" \
    --output_dir "$OUTPUT_DIR" \
    --auto_resume

echo "═══════════════════════════════════════════════════"
echo "End:       $(date)"
echo "Results:   $OUTPUT_DIR"
