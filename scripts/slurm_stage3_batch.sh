#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Real measured throughput as of 2026-09-21 (job 16119168, --limit 50
# --batch-size 4): 34m37s for 50 patients = ~41.5s/admission. At ~9,899
# target admissions that is ~114 hours (~4.75 days) of pure generation
# time, i.e. this script WILL hit its 24h limit and need ~5 resubmissions
# minimum, not counting queue wait between them (30+ hours once this
# project, when kisski was congested). This is safe to run into: batch.py
# writes each admission's result to the output CSV incrementally (not one
# save at the end), and --resume skips hadm_ids already written -- a
# timeout here loses at most the one in-flight chunk (<= generation_batch_
# size admissions), not the whole run.
#SBATCH --job-name=thesis-stage3-batch
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_batch_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_batch_%j.err
# TIME_LIMIT added 2026-09-27: given the above, this job WILL end via
# timeout, not completion, on every segment but the last. Some SLURM
# configurations do not fire END for a TIMEOUT state (only COMPLETED) --
# without TIME_LIMIT explicitly listed, a timed-out segment could go
# unnoticed for hours while it sits idle waiting to be resubmitted.
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=lennartstenzel@gmail.com

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
# Read the actual configured path from config.yaml rather than hardcoding
# it here a second time -- this hardcoded to the old repo-relative
# "models/medgemma-27b-text-it" path until 2026-09-27, which stopped
# matching config.yaml's stage3.model_name after the 2026-09-18 migration
# off personal HOME quota to KISSKI project storage. A stale relative path
# here would fail this preflight check every single job attempt (exit 1
# before any GPU work starts), silently burning a queue-wait cycle (which
# has taken 30+ hours during real congestion this project) each time,
# without ever explaining that the actual model files are fine.
MEDGEMMA_DIR=$(python -c "from src.config import load_config; print(load_config().stage3.model_name)")
if ! compgen -G "${MEDGEMMA_DIR}/*.safetensors" > /dev/null; then
    echo "ERROR: no .safetensors weights found in ${MEDGEMMA_DIR}"
    echo "  Run 'bash download_stage3_model.sh' on the login node first."
    exit 1
fi
echo "  OK -- $(ls ${MEDGEMMA_DIR}/*.safetensors | wc -l) weight shard(s) present at ${MEDGEMMA_DIR}"

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
