#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=75:00:00
# Bumped from 48h 2026-09-06: a smoke test measured 2.42s/optimizer-step,
# which alone extrapolates to ~60h for the full 10 epochs (88,600 steps) --
# and that doesn't even include the periodic full-validation-set eval pass
# every save_steps=500 steps, whose real cost is still unmeasured. 75h is
# a deliberately generous ceiling, not a measured requirement. If eval
# overhead turns out to be large, even this may not be enough for all 10
# epochs -- early_stopping_patience=3 may end it sooner regardless. Watch
# the first real eval pass (shortly after step 500) to get an actual number.
#SBATCH --job-name=thesis-stage2
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# Email notifications: pass the address at submit time, e.g.  sbatch --mail-user=<your-email> train_stage2.sh ...

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

# dataloader_num_workers=8 (config.yaml) forks 8 worker processes that each
# touch the fast (Rust-backed) tokenizer -- a known combination that warns
# or, on some versions/platforms, deadlocks under fork-based multiprocessing
# (the default on Linux). Disabling is the standard fix and has no downside
# here (tokenization is CPU-light relative to the model forward pass).
export TOKENIZERS_PARALLELISM=false

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
# config.json is git-tracked and always exists after a pull -- it proves
# nothing about whether the actual weights were downloaded. Compute nodes
# have no internet, so a missing weights file here would only surface as a
# crash deep into the job. Check for the real weight file explicitly.
if ! compgen -G "models/clinical_longformer/model.safetensors" > /dev/null \
   && ! compgen -G "models/clinical_longformer/pytorch_model.bin" > /dev/null; then
    echo "ERROR: no model weights found in models/clinical_longformer/"
    echo "  (only tokenizer/config files are git-tracked -- the weights are not)."
    echo "  Run 'bash download_model.sh' on the login node first."
    exit 1
fi
echo "  OK — weight file present: $(ls -1 models/clinical_longformer/*.safetensors models/clinical_longformer/*.bin 2>/dev/null)"

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

# setup_stage2.py's own predict step (above) only scores Stage 1-flagged
# admissions (the deployment/cascade population) -- that's biased toward an
# already-high-risk subset and not valid for comparing Stage 2 against
# Stage 1 on equal footing (RQ1). predict_stage2_all scores every
# test-partition admission with a note, independent of Stage 1's flag, per
# docs/ARCHITECTURE.md's deferred-next-step list. Reuses the model +
# calibration setup_stage2.py just produced -- same GPU allocation, no
# second job needed.
echo "[stage2] Scoring population-wide (independent of Stage 1's flag, for RQ1) ..."
python -m src.stage2.predict --all

echo "Finished: $(date)"
