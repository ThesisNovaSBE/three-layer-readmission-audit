#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# D1/D2/D3 validation controls (session 19's run_blind_note_control /
# run_no_stage2_control / check_self_agreement, never run at scale before
# this script existed) on a deliberate sample -- NOT a second full batch
# run.
#
# 2026-11: rewritten to run through the main run's own batched chunk
# processor (cfg.stage3.generation_batch_size, left-padded), not one
# admission at a time -- confirmed that single-sequence vs. batched
# bfloat16 generation can itself produce different decisions, which was
# confounding the no-Stage-2 control's agreement numbers with "batched vs
# single" on top of the thing it's meant to isolate ("Stage 2 removed").
# This makes the 2026-10-30 timing note below (one-at-a-time, 70-210s/
# admission) stale -- batched throughput should track the main run's
# ~41.5s/admission at generation_batch_size=4, i.e. noticeably faster for
# the same --n. Re-time on the first real run below and update this
# comment with the real number once observed.
#
# --n 150 for all four arms (reference, repeated reference, blind-note,
# no-Stage-2): the old 150/40/30 mix was an accident of the interrupted
# 2026-10-30 job, not a deliberate choice, and RQ2 leans on these
# controls' agreement numbers -- the no-Stage-2 control's 7/40 has a ~9-32%
# 95% CI, no narrower at n=30. Rough cost at the main run's measured
# ~50s/admission (136 GPU-hours / 9,899 admissions): 150*4 = 600 audits
# ~= 8 hours of pure generation -- --time above gives real margin over
# that bare estimate, not a tight fit.
#
# The --arm-test step below is a mandatory gate (see
# scripts/run_validation_controls.py): it reruns the first 8 admissions
# in the main run's own row order (same batch grouping -- a random
# sample would pair different neighbours and risk a bf16 false failure)
# through the unmanipulated batched path and exits non-zero if the
# decisions don't exactly match the main run. This SLURM script's
# gpu:A100:1 / 80gb_vram request already matches the main run's GPU type
# -- do not resubmit this specific script on the kisski-h100 partition,
# since a different GPU can also change bf16 rounding and would make the
# arm-test fail for a reason unrelated to the code being tested. The
# full controls only run if the arm-test passes -- otherwise this job
# would spend GPU hours producing agreement numbers built on a broken
# reimplementation of the batched path.
#
# 2026-10-30 (pre-rewrite) history, kept for context: an earlier --n 150
# one-at-a-time attempt (job 16873264) got stuck for ~4 hours on the
# first D1 blind-note patient and was cancelled (607.5 core-hours, zero
# patients written) -- a follow-up isolated single-patient timing test
# (job 16947546) ran all three conditions cleanly and fast (212.1s/109.4s/
# 69.5s), so the stall did not reproduce and was most likely a one-off,
# not a systemic bug. Safe to resubmit on timeout either way:
# scripts/run_validation_controls.py writes each control's CSV
# incrementally and skips already-done hadm_ids automatically on rerun,
# no --resume flag needed.
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
echo "[slurm] Acceptance test: batched reference arm vs. the main run (8 admissions) ..."
python -m scripts.run_validation_controls --arm-test 8

# --n 150: one sample size across all four arms (see comment above) --
# per run_blind_note_control's own docstring this is still "a deliberate
# sample, tens to low hundreds," not a second full audit.
echo "[slurm] Arm test passed -- running the full D1/D2/D3 controls (n=150) ..."
python -m scripts.run_validation_controls --n 150

echo "[slurm] Done at $(date)"
echo "[slurm] Results saved to models/stage3_control_*.csv and"
echo "        models/stage3_validation_controls_summary.json"
