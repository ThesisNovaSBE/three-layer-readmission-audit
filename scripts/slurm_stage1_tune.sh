#!/usr/bin/env bash
# GWDG KISSKI SLURM submission — Stage 1 Optuna tune → train → evaluate
#
# Usage (from ~/thesis on the login node):
#   sbatch scripts/slurm_stage1_tune.sh
#
# Partition/module/env match the confirmed-working train_stage2.sh and
# diagnose_cluster.sh (2026-08-29 fix) -- this script previously guessed at
# a separate "Grete"/a100 partition and anaconda3/thesis env that don't
# match this project's actual cluster setup (there is only the kisski
# partition; session 11's cluster log confirms module load miniforge3 +
# conda activate thesis-env). That mismatch was flagged but left unfixed
# in session 21 pending confirmation -- confirmed now via the other two
# scripts, so fixed here rather than risking a wasted submission.
#
# Measured wall time (job 15700645, 2026-09-02): 140/400 trials in ~7.5h with
# --tune-n-jobs 8 (39% GPU util, confirmed via nvidia-smi -- device=cuda is
# genuinely engaged) -> ~19 trials/hour -> the full 400 needs ~21h total, not
# 8h. That job hit the time limit and was killed; --time bumped to 24h below
# to cover the remaining ~260 trials plus retrain/eval with margin. This is
# resumable regardless (JournalFileBackend + load_if_exists in tune.py) --
# resubmitting unchanged picks up from whatever trial count is already
# recorded in models/optuna_stage1_xgboost_journal.log rather than
# restarting, so a too-short --time only costs the wait, never the progress.

#SBATCH --job-name=thesis_stage1_tune
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage1_tune_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage1_tune_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email-removed>

set -eo pipefail
# Deliberately no -u (nounset): thesis-env's MKL conda activation hook
# (libblas_mkl_activate.sh) references $MKL_INTERFACE_LAYER without a
# default, which -u treats as a fatal error and kills the job before it
# even reaches this script's own commands (confirmed 2026-09-02 -- a real
# job died in 25s with exactly this error, no script output at all).

# ── Environment ──────────────────────────────────────────────────────────────
module load miniforge3/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate thesis-env

cd ~/thesis
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "[slurm] Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "[slurm] Repo: $(pwd)"
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
