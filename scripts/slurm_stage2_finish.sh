#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# 6h: measured on a previous run ~3.3h (calibration ~1h19m, cascade predict ~17m,
# population-wide predict ~2h); limit kept short so the job can backfill.
#SBATCH --job-name=thesis-stage2-finish
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_finish_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_finish_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lennartstenzel@gmail.com

# Runs only the post-training steps against the ALREADY fine-tuned model at
# models/stage2_longformer_best -- deliberately does NOT call train_stage2()
# / setup_stage2.py, since resubmitting the full pipeline would resume from
# the last checkpoint and continue training past where early stopping
# already correctly stopped it (confirmed 2026-09-06: it stopped ~epoch 1.02
# because validation AUPRC had already peaked and started declining --
# resuming would just keep degrading it, not improve it).
#
# This exists because calibrate.py was found reusing a stale calibration
# file from before the model was retrained (fixed 2026-09-08, see
# src/stage2/calibrate.py's fingerprint check) -- --force here guarantees a
# fresh calibration regardless of that fingerprint file's state.

set -eo pipefail

module load miniforge3/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate thesis-env
cd ~/thesis

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[slurm] Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "[slurm] Git: $(git rev-parse --short HEAD)"

echo "[slurm] Recomputing calibration against the current model ..."
python -m src.stage2.calibrate --force

echo "[slurm] Scoring Stage 1-flagged patients (cascade population) ..."
python -m src.stage2.predict

echo "[slurm] Scoring population-wide (independent of Stage 1's flag, for RQ1) ..."
python -m src.stage2.predict --all

echo "[slurm] Running Stage 2 per-age-group evaluation ..."
python -m src.stage2.evaluate

echo "[slurm] Refreshing Stage 1+2 end-to-end pipeline evaluation ..."
echo "[slurm] NOTE: if models/stage3_batch_results.csv is still the old"
echo "[slurm]   (pre-fix) Stage 3 run, this evaluate_pipeline output mixes"
echo "[slurm]   corrected Stage 2 with uncorrected Stage 3 -- interim only,"
echo "[slurm]   rerun this step after the real Stage 3 rerun completes."
python -m src.model.evaluate_pipeline

echo "[slurm] Running RQ1 comparison (Stage 1 vs Stage 2, notes-covered population) ..."
python -m src.model.compare_stages

echo "[slurm] All steps complete at $(date)"
echo "[slurm] Results saved to models/"
