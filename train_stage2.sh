#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=thesis-stage2
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email-removed>

set -e          # exit immediately if any command fails
set -o pipefail # catch errors inside pipes

# ── Environment ──────────────────────────────────────────────
module load miniforge3/24.3.0-0
eval "$(conda shell.bash hook)"   # initialise conda shell functions without needing `conda init`
conda activate thesis-env
cd ~/thesis

# Compute nodes have no internet — force HuggingFace to use local cache only.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Pre-flight checks ────────────────────────────────────────
echo "=== Pre-flight checks ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Python:  $(which python)"
python --version

echo "Checking Stage 1 artifact..."
ls -lh models/stage1_xgboost.joblib

echo "Checking local Clinical-Longformer model..."
ls -lh models/clinical_longformer/config.json

echo "Checking MIMIC-IV-Note path..."
python -c "
from src.config import load_config
from pathlib import Path
cfg = load_config()
note_dir = Path(cfg.data.mimic_iv_note_dir)
assert note_dir.exists(), f'MIMIC-IV-Note not found: {note_dir}'
files = list(note_dir.iterdir())
print(f'  OK — {note_dir}  ({len(files)} files)')
"

echo "Checking CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  OK — {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"

echo "=== All checks passed — starting training ==="

# ── Run ──────────────────────────────────────────────────────
python setup_stage2.py --mode full

echo "Finished: $(date)"
