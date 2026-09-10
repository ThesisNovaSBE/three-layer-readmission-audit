#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# UNMEASURED GUESS as of 2026-09-10 -- no real per-call latency data exists
# yet for MedGemma-27B + guided decoding at this context length. This is
# safe to guess wrong in either direction, unlike Stage 2's calibrate/
# predict steps: batch.py writes each admission's result to the output CSV
# incrementally as it goes (not one save at the end), and --resume skips
# hadm_ids already written -- a timeout here loses at most the one
# in-flight call, not the whole run. Smoke-test with --limit first (see
# scripts/README or session log) to get a real number before trusting this.
#SBATCH --job-name=thesis-stage3-batch
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_batch_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_batch_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email-removed>

set -eo pipefail
# No -u (nounset): thesis-env's MKL conda activation hook references an
# unset variable and dies under -u -- see scripts/slurm_stage1_tune.sh's
# 2026-09-02 fix for the same issue.

module load miniforge3/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate thesis-env
cd ~/thesis

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "[slurm] Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "[slurm] Git: $(git rev-parse --short HEAD)"

# ── Pre-flight checks ────────────────────────────────────────
echo "=== Pre-flight checks ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Checking Stage 1 artifact..."
ls -lh models/stage1_xgboost.joblib

echo "Checking Stage 2 results..."
ls -lh models/stage2_results.csv

echo "Checking MedGemma weights (real files, not just a directory)..."
if ! compgen -G "models/medgemma-27b-text-it/*.safetensors" > /dev/null; then
    echo "ERROR: no .safetensors weights found in models/medgemma-27b-text-it/"
    echo "  Run 'bash download_stage3_model.sh' on the login node first."
    exit 1
fi
echo "  OK -- $(ls models/medgemma-27b-text-it/*.safetensors | wc -l) weight shard(s) present"

echo "Checking CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'  OK — {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"

echo "=== All checks passed — starting batch audit ==="

# ── Run ──────────────────────────────────────────────────────
# --resume: safe to always pass. On a fresh run the output CSV doesn't
# exist yet, so it behaves identically to a plain run; on a resubmission
# after a timeout/crash, it picks up where it left off instead of
# redoing (and re-billing) work already done.
python -m src.stage3.batch --resume

echo "[slurm] Done at $(date)"
echo "[slurm] Results saved to models/stage3_batch_results.csv"
