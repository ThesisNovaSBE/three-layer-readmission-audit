#!/bin/bash
# Run this ONCE on the KISSKI login node (which has internet access).
# Saves yikuan8/Clinical-Longformer to models/clinical_longformer/
# so compute nodes can load it offline.
#
# Usage (on login node, from ~/thesis):
#   bash download_model.sh

set -e
cd ~/thesis

echo "=== Downloading yikuan8/Clinical-Longformer to models/clinical_longformer/ ==="

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate thesis-env

python - <<'EOF'
from pathlib import Path
from transformers import AutoTokenizer, LongformerForSequenceClassification

model_id = "yikuan8/Clinical-Longformer"
save_dir = "models/clinical_longformer"

Path(save_dir).mkdir(parents=True, exist_ok=True)

print(f"Downloading tokenizer from '{model_id}' ...")
tok = AutoTokenizer.from_pretrained(model_id)
tok.save_pretrained(save_dir)
print("  Tokenizer saved.")

print(f"Downloading model weights (~500 MB) ...")
model = LongformerForSequenceClassification.from_pretrained(model_id)
model.save_pretrained(save_dir)
print("  Model saved.")

files = list(Path(save_dir).iterdir())
print(f"\nDone — {len(files)} files in {save_dir}:")
for f in sorted(files):
    size = f.stat().st_size / 1024**2
    print(f"  {f.name}  ({size:.1f} MB)")
EOF

echo ""
echo "=== Model ready. You can now sbatch train_stage2.sh ==="
