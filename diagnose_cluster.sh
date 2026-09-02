#!/bin/bash
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH -C 80gb_vram
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=thesis-diag
#SBATCH --output=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/diag_%j.log
#SBATCH --error=/projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/diag_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email-removed>

# NO set -e — all checks run even if some fail, so we see every problem at once

PASS="[PASS]"
FAIL="[FAIL]"

echo "======================================================"
echo "  CLUSTER DIAGNOSTIC — $(date)"
echo "  Node: $(hostname)"
echo "  Job:  $SLURM_JOB_ID"
echo "======================================================"
echo ""

# ── 1. Module load ─────────────────────────────────────────
echo "--- 1. module load miniforge3/24.3.0-0 ---"
if module load miniforge3/24.3.0-0 2>&1; then
    echo "$PASS module load miniforge3/24.3.0-0"
else
    echo "$FAIL module load miniforge3/24.3.0-0"
    echo "  Available modules matching conda/python/mini:"
    module avail 2>&1 | grep -i -E "conda|mini|python|anaconda" || echo "  (none found)"
fi
echo ""

# ── 2. Conda activation ────────────────────────────────────
echo "--- 2. conda activate thesis-env ---"
eval "$(conda shell.bash hook)" 2>&1
if conda activate thesis-env 2>&1; then
    echo "$PASS conda activate thesis-env"
    echo "  Python:  $(which python)"
    echo "  Version: $(python --version 2>&1)"
else
    echo "$FAIL conda activate thesis-env"
    echo "  Available environments:"
    conda env list 2>&1
fi
echo ""

# ── 3. GPU ─────────────────────────────────────────────────
echo "--- 3. nvidia-smi ---"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1; then
    echo "$PASS nvidia-smi"
else
    echo "$FAIL nvidia-smi (exit $?)"
fi
echo ""

# ── 4. PyTorch + CUDA ──────────────────────────────────────
echo "--- 4. PyTorch CUDA ---"
python - <<'EOF'
import sys
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"  Device:  {name}, {vram} GB VRAM")
        print("[PASS] PyTorch CUDA")
    else:
        print("[FAIL] CUDA not available — check conda env has pytorch+cuda build")
        print(f"  torch.version.cuda = {torch.version.cuda}")
except ImportError as e:
    print(f"[FAIL] cannot import torch: {e}")
    sys.exit(1)
EOF
echo ""

# ── 5. Transformers / HuggingFace ──────────────────────────
echo "--- 5. transformers import ---"
python - <<'EOF'
try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
    print("[PASS] transformers")
except ImportError as e:
    print(f"[FAIL] cannot import transformers: {e}")
EOF
echo ""

# ── 6. Stage 1 artifact ────────────────────────────────────
echo "--- 6. Stage 1 artifact ---"
if ls -lh ~/thesis/models/stage1_xgboost.joblib 2>&1; then
    echo "$PASS Stage 1 artifact found"
else
    echo "$FAIL Stage 1 artifact NOT found"
    echo "  Contents of ~/thesis/models/ :"
    ls ~/thesis/models/ 2>&1 || echo "  (directory missing)"
fi
echo ""

# ── 7. .env file ───────────────────────────────────────────
echo "--- 7. .env file ---"
ENV_FILE=~/thesis/.env
if [ -f "$ENV_FILE" ]; then
    echo "$PASS .env exists at $ENV_FILE"
    echo "  Contents:"
    cat "$ENV_FILE"
else
    echo "$FAIL .env NOT found at $ENV_FILE"
fi
echo ""

# ── 8. Config loads + MIMIC path ───────────────────────────
echo "--- 8. config + MIMIC-IV-Note path ---"
cd ~/thesis
python - <<'EOF'
import sys
sys.path.insert(0, ".")
try:
    from src.config import load_config
    from pathlib import Path
    cfg = load_config()
    note_dir = Path(cfg.data.mimic_iv_note_dir)
    print(f"  Configured MIMIC-IV-Note dir: {note_dir}")
    print(f"  Exists:  {note_dir.exists()}")
    if note_dir.exists():
        files = list(note_dir.iterdir())
        print(f"  Files:   {len(files)}")
        print("[PASS] MIMIC-IV-Note path")
    else:
        print("[FAIL] MIMIC-IV-Note path does not exist on this node")
        # Check if $PROJECT_DIR path exists
        p2 = Path("/projects/extern/kisski/kisski-nova-rpcl/dir.project/mimic-iv-note")
        print(f"  Fallback check {p2}: exists={p2.exists()}")
except Exception as e:
    print(f"[FAIL] config error: {e}")
    import traceback; traceback.print_exc()
EOF
echo ""

# ── 9. MIMIC-IV structured path ────────────────────────────
echo "--- 9. MIMIC-IV structured path ---"
cd ~/thesis
python - <<'EOF'
import sys
sys.path.insert(0, ".")
try:
    from src.config import load_config
    from pathlib import Path
    cfg = load_config()
    iv_dir = Path(cfg.data.mimic_iv_dir)
    print(f"  Configured MIMIC-IV dir: {iv_dir}")
    print(f"  Exists: {iv_dir.exists()}")
    if iv_dir.exists():
        print("[PASS] MIMIC-IV path")
    else:
        print("[FAIL] MIMIC-IV path does not exist")
except Exception as e:
    print(f"[FAIL] config error: {e}")
EOF
echo ""

# ── 10. Disk space ─────────────────────────────────────────
echo "--- 10. disk space ---"
df -h ~/thesis /projects/extern/kisski/kisski-nova-rpcl/dir.project 2>&1
echo ""

echo "======================================================"
echo "  DIAGNOSTIC COMPLETE — $(date)"
echo "======================================================"
