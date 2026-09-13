#!/bin/bash
# Run this ONCE on the KISSKI login node (which has internet access).
# Saves google/medgemma-27b-text-it to models/medgemma-27b-text-it/ so
# compute nodes (no internet) can load it offline via vLLM.
#
# MedGemma is a GATED model -- you must have already accepted the Health AI
# Developer Foundations terms for it on your Hugging Face account
# (https://huggingface.co/google/medgemma-27b-text-it), and have a Hugging
# Face access token available as $HF_TOKEN. Get a token at
# https://huggingface.co/settings/tokens (read access is enough).
#
# This is a real download (~27B params in bf16, roughly 54-55 GB) -- expect
# it to take a while even on a fast connection. Not yet run/tested as of
# 2026-09-10; if it fails partway through, re-running is safe (snapshot_
# download resumes/skips files it already has).
#
# Usage (on login node, from ~/thesis):
#   export HF_TOKEN=hf_...
#   bash download_stage3_model.sh

set -e
cd ~/thesis

MODEL_ID="google/medgemma-27b-text-it"
SAVE_DIR="models/medgemma-27b-text-it"

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: \$HF_TOKEN is not set."
    echo "  1. Accept the Health AI Developer Foundations terms at:"
    echo "     https://huggingface.co/${MODEL_ID}"
    echo "  2. Create a read-access token at https://huggingface.co/settings/tokens"
    echo "  3. Run: export HF_TOKEN=hf_..."
    echo "  Then re-run this script."
    exit 1
fi

echo "=== Downloading ${MODEL_ID} to ${SAVE_DIR}/ ==="
echo "This is ~27B params in bf16 (~54-55 GB) -- expect a real wait."
df -h ~/thesis

module load miniforge3/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate thesis-env

python - <<EOF
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
save_dir = "${SAVE_DIR}"

Path(save_dir).mkdir(parents=True, exist_ok=True)

print(f"Downloading '{model_id}' -> {save_dir} ...")
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    # Skip original (often fp32/pt) weight duplicates when safetensors are
    # available -- vLLM reads safetensors directly, no need for both.
    ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "original/*"],
)

files = sorted(Path(save_dir).rglob("*"))
total_bytes = sum(f.stat().st_size for f in files if f.is_file())
print(f"\nDone -- {sum(1 for f in files if f.is_file())} files, "
      f"{total_bytes / 1024**3:.1f} GB total, in {save_dir}")
EOF

echo ""
echo "=== Model ready. Set config.yaml's stage3.model_name to '${SAVE_DIR}' ==="
echo "=== (already the default as of 2026-09-10) ==="
