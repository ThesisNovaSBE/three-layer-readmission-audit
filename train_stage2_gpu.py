"""GPU training script for Stage 2 — optimised for Colab / HPC environments.

This script is the recommended entry point when running on a GPU server.
It handles the macOS joblib/torch ordering constraint (load artifact before
importing torch) and sets the correct CUDA environment variables.

Runs Steps 0–2 of the Stage 2 pipeline:
  Step 0: Build patient-level splits (fast, CPU)
  Step 1: Fine-tune Clinical-Longformer (GPU required)
  Step 2: Calibrate per age group (GPU required)

After this script completes, run predict + evaluate on any machine:
  python -m src.stage2.predict
  python -m src.stage2.evaluate

Hardware presets (edit config.yaml to match your GPU):
  A100 80GB (Colab Pro+):  bf16: true,  fp16: false, gradient_checkpointing: false
  V100 32GB:               bf16: false, fp16: true,  gradient_checkpointing: true
  RTX 4090 24GB:           bf16: false, fp16: true,  gradient_checkpointing: true
  CPU / macOS (fallback):  bf16: false, fp16: false, gradient_checkpointing: false

Usage:
  # On Colab — paste in a cell:
  !python train_stage2_gpu.py --mode full

  # On HPC (SLURM):
  srun python train_stage2_gpu.py --mode full

  # Quick smoke test (synthetic data not supported; needs MIMIC notes):
  python train_stage2_gpu.py --mode quick
"""

import argparse
import os
import sys
from pathlib import Path

# ── CRITICAL: load Stage 1 artifact BEFORE importing torch ────────────────────
# joblib.load of a pickled XGBoost model segfaults on macOS (and sometimes Linux)
# if called after torch has been imported. Load first, then import everything else.
import joblib
from src.config import load_config, get_model_dir

_cfg_early = load_config()
_stage1_name = _cfg_early["stage1"]["model"]
_artifact_path = get_model_dir() / f"stage1_{_stage1_name}.joblib"

if not _artifact_path.exists():
    print(f"[train_stage2_gpu] ERROR: Stage 1 artifact not found at {_artifact_path}")
    print("  Run setup_demo.py then python -m src.model.train first.")
    sys.exit(1)

_preloaded_artifact = joblib.load(_artifact_path)
print(f"[train_stage2_gpu] Pre-loaded Stage 1 artifact ({_stage1_name})")

# ── Now safe to import torch ──────────────────────────────────────────────────
import torch  # noqa: E402

from src.stage2.splits import build_splits       # noqa: E402
from src.stage2.train import train_stage2        # noqa: E402
from src.stage2.calibrate import calibrate       # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="GPU training for Stage 2")
    parser.add_argument("--mode", choices=["quick", "full"], default=None,
                        help="Override run.mode from config.yaml")
    parser.add_argument("--skip-splits", action="store_true",
                        help="Skip Step 0 if splits already exist")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip Step 1 if model already exists (go straight to calibrate)")
    args = parser.parse_args()

    cfg = _cfg_early
    if args.mode:
        cfg["run"]["mode"] = args.mode

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print("  Stage 2 GPU Training Script")
    print("=" * 64)
    print(f"  Mode:    {cfg['run']['mode']}")
    print(f"  Model:   {cfg['stage2']['model_name']}")
    print(f"  Seq len: {cfg['stage2']['max_seq_length']} tokens")
    print(f"  Device:  {device}")
    if device == "cuda":
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: No GPU detected. Training will be extremely slow.")
        print("           Set max_seq_length: 512 in config.yaml for CPU fallback.")
    print()

    # Step 0: Build splits
    if not args.skip_splits:
        print("[0/2] Building patient-level splits ...")
        build_splits(cfg, artifact=_preloaded_artifact)
    else:
        print("[0/2] Skipping splits (--skip-splits)")

    # Step 1: Fine-tune
    if not args.skip_train:
        print("\n[1/2] Fine-tuning Clinical-Longformer ...")
        train_stage2(cfg, artifact=_preloaded_artifact)
    else:
        print("[1/2] Skipping training (--skip-train)")

    # Step 2: Calibrate
    print("\n[2/2] Calibrating per age group ...")
    calibrate(cfg, artifact=_preloaded_artifact)

    model_dir = get_model_dir()
    print("\n" + "=" * 64)
    print("  Training complete.")
    print("=" * 64)
    print(f"  Best model:   {model_dir / 'stage2_longformer_best'}")
    print(f"  Calibration:  {model_dir / 'stage2_calibration.json'}")
    print()
    print("  Next steps (run on any machine — no GPU needed):")
    print("    python -m src.stage2.predict")
    print("    python -m src.stage2.evaluate")
    print("=" * 64)


if __name__ == "__main__":
    main()
