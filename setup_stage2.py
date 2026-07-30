"""One-command Stage 2 runner: fine-tune Clinical-Longformer + score flagged patients.

Runs the full Stage 2 pipeline in sequence:
  1. Fine-tune Clinical-Longformer on discharge notes (train split)
  2. Score Stage 1 flagged patients on the test split
  3. Print a summary of how many flags were confirmed / pruned

Requires:
  - Stage 1 artifact:  models/stage1_xgboost.joblib
                       (run setup_demo.py then python -m src.model.train first)
  - MIMIC-IV-Note:     MIMIC_IV_NOTE_DIR set in .env
                       (Stage 2 cannot run on synthetic data — no synthetic notes exist)
  - HuggingFace model: yikuan8/Clinical-Longformer
                       (downloads automatically on first run, ~500 MB)

Usage:
    python setup_stage2.py
    python setup_stage2.py --mode full
"""

import argparse
import os
import sys
from pathlib import Path

# ── Step 1: load the Stage 1 artifact with joblib BEFORE torch is imported.
# Importing torch first and then calling joblib.load() on a pickled XGBoost
# model causes a C-extension conflict and segfaults on macOS.
import joblib  # noqa: E402 — must come before torch
from src.config import load_config, get_model_dir  # noqa: E402

_cfg_early = load_config()
_stage1_name = _cfg_early.stage1.model
_artifact_path = get_model_dir() / f"stage1_{_stage1_name}.joblib"
if not _artifact_path.exists():
    print(f"[setup_stage2] ERROR: Stage 1 artifact not found at {_artifact_path}")
    print("  Run setup_demo.py then python -m src.model.train first.")
    sys.exit(1)
_preloaded_artifact = joblib.load(_artifact_path)
print(f"[setup_stage2] Pre-loaded Stage 1 artifact from {_artifact_path}")

# ── Step 2: NOW set env vars and import torch / transformers.
# On macOS only: disable MPS/CUDA to prevent the C-extension segfault that
# occurs when torch is imported after joblib on Apple Silicon.
# On Linux (cluster): leave CUDA untouched so the GPU is used.
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Pre-load all heavy libraries here so they are fully initialised before any
# function call. Lazy imports inside functions trigger the segfault on macOS.
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from src.stage2.splits import build_splits       # noqa: E402
from src.stage2.train import train_stage2       # noqa: E402
from src.stage2.predict import predict_stage2   # noqa: E402
from src.stage2.calibrate import calibrate      # noqa: E402
from src.stage2.evaluate import evaluate        # noqa: E402



def check_prerequisites(cfg) -> None:
    model_dir = get_model_dir()
    stage1_name = cfg.stage1.model
    artifact = model_dir / f"stage1_{stage1_name}.joblib"
    if not artifact.exists():
        print(f"[setup_stage2] ERROR: Stage 1 artifact not found at {artifact}")
        print("  Run setup_demo.py then python -m src.model.train first.")
        sys.exit(1)

    note_dir = cfg.data.mimic_iv_note_dir
    if not note_dir or not Path(note_dir).exists():
        print("[setup_stage2] ERROR: MIMIC-IV-Note path not configured or not found.")
        print("  Set MIMIC_IV_NOTE_DIR in your .env file.")
        print("  (Stage 2 requires real discharge notes — synthetic data has none.)")
        sys.exit(1)

    print(f"[setup_stage2] Stage 1 artifact: {artifact}")
    print(f"[setup_stage2] MIMIC-IV-Note:    {note_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 end-to-end")
    parser.add_argument("--mode", choices=["quick", "full"], default=None,
                        help="Override run.mode from config.yaml")
    args = parser.parse_args()

    cfg = _cfg_early  # already loaded at module level (before torch)
    if args.mode:
        cfg.run.mode = args.mode

    print("=" * 64)
    print("  Readmission Prediction — Stage 2 (Clinical-Longformer)")
    print("=" * 64)
    print(f"  Mode:  {cfg.run.mode}")
    print(f"  Model: {cfg.stage2.model_name}")
    print(f"  Max sequence length: {cfg.stage2.max_seq_length} tokens")
    print()

    check_prerequisites(cfg)

    # ── Step 0: Patient-level splits ──
    print("\n[0/4] Building patient-level train / val / calibration splits ...")
    build_splits(cfg, artifact=_preloaded_artifact)

    # ── Step 1: Fine-tune ──
    print("\n[1/4] Fine-tuning Clinical-Longformer on discharge notes...")
    print("      (model downloads automatically from HuggingFace on first run)\n")
    # Pass the pre-loaded artifact so train_stage2 does NOT call joblib.load()
    # after torch is in memory (which segfaults on macOS).
    train_stage2(cfg, artifact=_preloaded_artifact)

    # ── Step 2: Calibrate ──
    print("\n[2/4] Running Platt scaling + per-group threshold selection...\n")
    calibrate(cfg, artifact=_preloaded_artifact)

    # ── Step 3: Predict ──
    print("\n[3/4] Scoring Stage 1 flagged patients with fine-tuned model...\n")
    results = predict_stage2(cfg, artifact=_preloaded_artifact)

    # ── Step 4: Evaluate ──
    print("\n[4/4] Computing per-age-group evaluation metrics...\n")
    eval_results = evaluate(cfg, artifact=_preloaded_artifact)

    # ── Summary ──
    total = len(results)
    confirmed = int(results["stage2_confirmed"].sum())
    pruned = total - confirmed
    overall = eval_results.get("overall", {})
    gaps    = eval_results.get("fairness_gaps", {})

    print("\n" + "=" * 64)
    print("  Stage 2 complete")
    print("=" * 64)
    print(f"  Stage 1 flagged:      {total:,}")
    print(f"  Stage 2 confirmed:    {confirmed:,}  ({confirmed / max(total, 1):.1%} retained)")
    print(f"  Stage 2 pruned:       {pruned:,}  ({pruned / max(total, 1):.1%} removed)")
    if overall:
        print(f"  Overall AUROC:        {overall.get('auroc', float('nan')):.3f}")
        print(f"  Overall Recall:       {overall.get('recall', float('nan')):.3f}")
        print(f"  Overall Precision:    {overall.get('precision', float('nan')):.3f}")
        print(f"  ECE:                  {overall.get('ece', float('nan')):.4f}")
    if gaps:
        print(f"  Recall gap (age):     {gaps.get('recall_gap', float('nan')):.3f}")
        print(f"  Precision gap (age):  {gaps.get('precision_gap', float('nan')):.3f}")
    print(f"  Results saved to:     models/stage2_results.csv")
    print(f"  Evaluation saved to:  models/stage2_evaluation.json")
    print()
    print("  Next: python setup_stage3.py  (requires Ollama + phi4-mini)")
    print("=" * 64)


if __name__ == "__main__":
    main()
