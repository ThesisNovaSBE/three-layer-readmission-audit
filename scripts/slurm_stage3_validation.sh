#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# D1/D2/D3 validation controls (session 19's run_blind_note_control /
# run_no_stage2_control / check_self_agreement, never run at scale before
# this script existed) on a deliberate sample -- NOT a second full batch
# run. Calls explain_patient ONE admission at a time per control (not
# batched like the main batch run), so per-admission latency here is
# almost certainly higher than slurm_stage3_batch.sh's measured
# ~41.5s/admission (that number was for batch-size-4 generation). No
# verified single-call throughput exists yet -- the 24h budget below is
# copied from the main batch job as a safe starting point, not a measured
# estimate. Safe to resubmit on timeout: scripts/run_validation_controls.py
# writes each control's CSV incrementally and skips already-done hadm_ids
# automatically on rerun, no --resume flag needed.
#SBATCH --job-name=thesis-stage3-validation
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_validation_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_validation_%j.err
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

echo "Checking the completed Stage 3 batch run (this script samples from it)..."
ls -lh models/stage3_batch_results.csv

echo "Checking MedGemma weights (real files, not just a directory)..."
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

echo "=== All checks passed — starting validation controls ==="

# ── Run ──────────────────────────────────────────────────────
# --n 150: a deliberate sample (per run_blind_note_control's own
# docstring: "tens to low hundreds"), not a second full audit. Override
# with `sbatch --export=N=50 scripts/slurm_stage3_validation.sh` or just
# edit the number below before submitting.
python -m scripts.run_validation_controls --n 150

echo "[slurm] Done at $(date)"
echo "[slurm] Results saved to models/stage3_control_*.csv and"
echo "        models/stage3_validation_controls_summary.json"
