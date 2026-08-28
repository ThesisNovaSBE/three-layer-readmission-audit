#!/usr/bin/env bash
# GWDG Grete SLURM submission — Stage 1 Optuna tune → train → evaluate
#
# Usage (from repo root on the login node):
#   sbatch scripts/slurm_stage1_tune.sh
#
# Estimated wall time: previously ~4 h using 16 CPU cores only -- the script
# requested this A100 node but XGBoost never actually used it (tree_method
# was CPU-only). Now passes --device cuda so the GPU is genuinely engaged,
# and --tune-n-jobs 8 runs 8 trials concurrently against it (a single fit on
# this dataset is small relative to an A100 -- one trial at a time leaves
# most of the GPU idle). 8 is a starting point sized to the 16 CPUs below,
# not a tuned value; watch `nvidia-smi` during a run and adjust. No measured
# wall-time for either change yet, so --time is left with headroom until a
# real run establishes one. Adjust --time down once you have that number.

#SBATCH --job-name=thesis_stage1_tune
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j_stage1_tune.out
#SBATCH --error=logs/slurm_%j_stage1_tune.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email-removed>

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3
conda activate thesis 2>/dev/null || source activate thesis

export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

echo "[slurm] Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "[slurm] Repo: ${SLURM_SUBMIT_DIR}"
echo "[slurm] Git: $(git rev-parse --short HEAD)"

# ── Step 1: rebuild feature matrix (idempotent; skips if up-to-date) ─────────
echo "[slurm] Building feature matrix …"
python -m src.data.features --mode full

# ── Step 2: Optuna hyperparameter search (400 trials) ────────────────────────
# Stored in models/optuna_stage1_xgboost_journal.log — resumable if job is
# preempted (resubmit unchanged; run_study() picks up recorded trials
# automatically). File-backed, not sqlite -- sqlite's locking is unreliable
# on this filesystem, especially with concurrent trials (see tune.py).
echo "[slurm] Running Optuna tune (400 trials, GPU, 8 concurrent) …"
python -m src.model.tune --mode full --device cuda --tune-n-jobs 8

# ── Step 3: Full retrain with best params ────────────────────────────────────
echo "[slurm] Training with best params (GPU) …"
python -m src.model.train --mode full --device cuda

# ── Step 4: Standard Stage 1 evaluation (val set) ────────────────────────────
echo "[slurm] Running Stage 1 evaluation …"
python -m src.model.evaluate --mode full

# ── Step 5: End-to-end blind pipeline evaluation ─────────────────────────────
echo "[slurm] Running end-to-end pipeline evaluation …"
python -m src.model.evaluate_pipeline

echo "[slurm] All steps complete at $(date)"
echo "[slurm] Results saved to models/"
