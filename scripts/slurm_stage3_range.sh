#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=32:00:00
# Default wall time for one parallel range job. Override on the sbatch command
# line as needed, e.g. a short gap job:
#     sbatch --time=12:00:00 scripts/slurm_stage3_range.sh 0 5200 models/stage3_batch_results.csv
#SBATCH --job-name=thesis-stage3-range
# Bootstrap log (%j): only the few lines before the exec-redirect below land
# here. #SBATCH directives are parsed by sbatch at submission and cannot see
# this script's positional args ($1/$2), so the real per-range log -- whose
# name contains S and E as required -- is opened by the script itself once the
# args are known (see the exec line under "Resolve arguments").
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_range_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage3_range_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=<email-removed>
#
# Parallel range job: audit targets[S:E] of the ordered target list (exactly
# the list the full sequential run uses), so several segments can run at once
# and be merged afterwards with scripts/merge_stage3_parts.py. S and E must be
# multiples of the generation batch size (except E may equal the total target
# count) -- src.stage3.batch validates this -- so each range's chunk
# composition is identical to a single sequential run over the same slice.
#
# Usage:
#     sbatch scripts/slurm_stage3_range.sh S E [OUT]
# With no OUT, the output defaults to
#     models/stage3_batch_results.range{S}-{E}.csv
# (plus its .raw.jsonl sidecar). The gap job that fills what the main
# sequential run did not reach writes straight into the main file instead:
#     sbatch --time=12:00:00 scripts/slurm_stage3_range.sh 0 5200 models/stage3_batch_results.csv
# with --resume it only audits admissions missing from that file.

set -eo pipefail
# No -u (nounset): thesis-env's MKL conda activation hook references an
# unset variable and dies under -u -- see scripts/slurm_stage1_tune.sh's
# 2026-09-02 fix for the same issue.

# ── Resolve arguments ────────────────────────────────────────
S="$1"
E="$2"
OUT="$3"  # optional
if [ -z "$S" ] || [ -z "$E" ]; then
    echo "ERROR: usage: sbatch scripts/slurm_stage3_range.sh S E [OUT]"
    exit 1
fi
LOG_DIR="/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs"
RANGE_LOG="${LOG_DIR}/stage3_range_${S}-${E}_${SLURM_JOB_ID}.log"
# Everything from here on goes to a log whose name carries S and E. The %j
# bootstrap log above keeps only the lines above this redirect.
exec > "${RANGE_LOG}" 2>&1

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
echo "[slurm] Range [${S}, ${E})${OUT:+, out=${OUT}}"

# ── Pre-flight checks ────────────────────────────────────────
echo "=== Pre-flight checks ==="
# Fail-fast interlock (N6): a previous segment (sequential OR range) aborted
# with >20% failures and left this shared marker. Refuse to burn another slot
# until it is investigated and the file is deleted.
if [ -f models/stage3_FAILFAST ]; then
    echo "ERROR: models/stage3_FAILFAST exists -- a prior run aborted on a high"
    echo "  failure rate. Investigate, then delete the file to resubmit:"
    cat models/stage3_FAILFAST
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Checking Stage 1 artifact..."
ls -lh models/stage1_xgboost.joblib

echo "Checking Stage 2 results..."
ls -lh models/stage2_results.csv

echo "Checking MedGemma weights (real files, not just a directory)..."
# Read the actual configured path from config.yaml rather than hardcoding it
# -- same reasoning as scripts/slurm_stage3_batch.sh.
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

echo "=== All checks passed — starting range audit [${S}, ${E}) ==="

# ── Run ──────────────────────────────────────────────────────
# --resume: safe across this range job's own resubmission chain (timeout/crash
# -> skips admissions already in the output file). The merge step dedups any
# overlap between ranges / the main file, so overlapping boundaries are fine.
if [ -n "${OUT}" ]; then
    python -m src.stage3.batch --resume --range-start "${S}" --range-end "${E}" --out "${OUT}"
else
    python -m src.stage3.batch --resume --range-start "${S}" --range-end "${E}"
fi

echo "[slurm] Done at $(date)"
if [ -n "${OUT}" ]; then
    echo "[slurm] Results saved to ${OUT}"
else
    echo "[slurm] Results saved to models/stage3_batch_results.range${S}-${E}.csv"
fi
