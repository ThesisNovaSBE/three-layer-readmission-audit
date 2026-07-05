"""Post-training calibration for Stage 2 Clinical-Longformer.

Runs Platt scaling (logistic regression on raw logits) per age group on the
calibration split, then selects per-group decision thresholds that satisfy a
recall floor while maximising F2.

Output: models/stage2_calibration.json
    {
      "calibrators": {"18-40": [coef, intercept], ...},
      "thresholds":  {"18-40": 0.31, "41-55": 0.28, ...},
      "strategy":    "recall_floor",
      "recall_floor": 0.65,
      "calibration_metrics": {...}
    }

Usage:
    python -m src.stage2.calibrate
    python -m src.stage2.calibrate --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LongformerForSequenceClassification

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix
from src.stage2.dataset import load_notes, build_notes_dataframe, ClinicalNotesDataset
from src.stage2.splits import build_splits
from src.schemas import TARGET_COL


# ── Age band key helpers ──────────────────────────────────────────────────────

_BAND_MAP = {
    "(17, 40]":  "18-40",
    "(40, 55]":  "41-55",
    "(55, 70]":  "56-70",
    "(70, 120]": "70+",
}


def _band_key(age_band) -> str:
    return _BAND_MAP.get(str(age_band), str(age_band))


# ── Inference helpers ─────────────────────────────────────────────────────────

def _get_logits(model, dataset: ClinicalNotesDataset, batch_size: int, device) -> np.ndarray:
    """Run inference and return raw logits (class-1 column)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[calibrate] scoring"):
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in {"labels", "group_weight"}}
            logits = model(**inputs).logits          # (B, 2)
            all_logits.extend(logits[:, 1].cpu().numpy().tolist())
    return np.array(all_logits)


# ── Threshold selection ───────────────────────────────────────────────────────

def _recall_floor_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    recall_floor: float,
    n_steps: int = 200,
) -> float:
    """Return the threshold that satisfies recall >= recall_floor and maximises F2.

    If no threshold achieves the recall floor, return the one with highest recall.
    """
    best_thr   = 0.5
    best_f2    = -1.0
    best_recall = -1.0

    for thr in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= thr).astype(int)
        rec = recall_score(labels, preds, zero_division=0)
        if rec < recall_floor:
            continue
        pre = precision_score(labels, preds, zero_division=0)
        f2  = (5 * pre * rec) / (4 * pre + rec + 1e-9)
        if f2 > best_f2:
            best_f2  = f2
            best_thr = float(thr)
            best_recall = rec

    if best_f2 < 0:
        # Recall floor not achievable — pick threshold with highest recall
        for thr in np.linspace(0.01, 0.99, n_steps):
            preds = (probs >= thr).astype(int)
            rec = recall_score(labels, preds, zero_division=0)
            if rec > best_recall:
                best_recall = rec
                best_thr    = float(thr)

    return best_thr


# ── Main calibration routine ──────────────────────────────────────────────────

def calibrate(cfg: dict, artifact: dict | None = None, force: bool = False) -> dict:
    """Run Platt scaling + per-group threshold selection on the calibration split.

    Args:
        cfg:      loaded config dict.
        artifact: pre-loaded Stage 1 artifact; loaded from disk if None.
        force:    recompute even if calibration JSON already exists.

    Returns:
        Calibration dict (same structure written to JSON).
    """
    model_dir  = get_model_dir()
    cal_path   = model_dir / "stage2_calibration.json"

    if not force and cal_path.exists():
        print("[calibrate] Loading existing calibration from", cal_path)
        return json.loads(cal_path.read_text())

    stage2_path = model_dir / "stage2_longformer_best"
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at {stage2_path}. "
            "Run `python -m src.stage2.train` first."
        )

    s2             = cfg["stage2"]
    batch_size     = s2["batch_size"] * 2
    max_length     = s2["max_seq_length"]
    recall_floor   = s2.get("recall_floor", 0.65)
    strategy       = s2.get("threshold_strategy", "recall_floor")

    # Load Stage 1 artifact for train_idx / mode
    if artifact is None:
        stage1_name = cfg["stage1"]["model"]
        artifact = joblib.load(model_dir / f"stage1_{stage1_name}.joblib")

    mode = artifact["mode"]

    # Patient-level splits → calibration hadm_ids
    splits = build_splits(cfg, artifact=artifact)
    cal_hadm_ids = set(splits["cal"]["hadm_id"].astype(int))

    # Feature matrix for age_band
    matrix    = load_feature_matrix(cfg, mode)
    labels_df = matrix[["hadm_id", "subject_id", "age_band", TARGET_COL]]

    # Load notes for calibration split only
    print(f"[calibrate] Loading notes for {len(cal_hadm_ids):,} calibration admissions ...")
    notes  = load_notes(cfg, hadm_ids=cal_hadm_ids)
    sub_df = labels_df[labels_df["hadm_id"].isin(cal_hadm_ids)]
    cal_df = build_notes_dataframe(notes, sub_df[["hadm_id", "subject_id", TARGET_COL]])
    cal_df = cal_df.merge(labels_df[["hadm_id", "age_band"]], on="hadm_id", how="left")
    cal_df["age_band_key"] = cal_df["age_band"].apply(_band_key)

    print(f"[calibrate] Calibration notes: {len(cal_df):,}  "
          f"(readmission rate: {cal_df[TARGET_COL].mean():.1%})")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] Device: {device}")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    model     = LongformerForSequenceClassification.from_pretrained(str(stage2_path))
    model.to(device)

    dataset = ClinicalNotesDataset(
        cal_df["text"].tolist(),
        cal_df[TARGET_COL].tolist(),
        tokenizer,
        max_length,
    )
    raw_logits = _get_logits(model, dataset, batch_size, device)
    labels_arr = cal_df[TARGET_COL].values

    # ── Per-group Platt scaling ──────────────────────────────────────────────
    bands = cal_df["age_band_key"].values
    calibrators: dict[str, list[float]] = {}
    thresholds:  dict[str, float]       = {}
    cal_metrics: dict[str, dict]        = {}

    unique_bands = sorted(set(bands))
    for band in unique_bands:
        mask = bands == band
        X_b  = raw_logits[mask].reshape(-1, 1)
        y_b  = labels_arr[mask]

        if y_b.sum() < 5 or (y_b == 0).sum() < 5:
            print(f"  [calibrate] {band}: too few samples ({mask.sum()}) — "
                  "skipping per-group calibration, using global fallback")
            calibrators[band] = [1.0, 0.0]   # identity (logit passthrough)
        else:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_b, y_b)
            calibrators[band] = [float(lr.coef_[0][0]), float(lr.intercept_[0])]

        # Calibrated probabilities
        coef, intercept = calibrators[band]
        cal_logits = coef * raw_logits[mask] + intercept
        cal_probs  = 1.0 / (1.0 + np.exp(-cal_logits))

        # Threshold selection
        if strategy == "recall_floor":
            thr = _recall_floor_threshold(cal_probs, y_b, recall_floor)
        else:
            thr = 0.5
        thresholds[band] = thr

        # Report
        preds  = (cal_probs >= thr).astype(int)
        rec    = recall_score(y_b, preds, zero_division=0)
        pre    = precision_score(y_b, preds, zero_division=0)
        try:
            auroc = float(roc_auc_score(y_b, cal_probs))
        except ValueError:
            auroc = float("nan")
        f2 = (5 * pre * rec) / (4 * pre + rec + 1e-9)

        cal_metrics[band] = {
            "n": int(mask.sum()),
            "pos_rate": float(y_b.mean()),
            "threshold": thr,
            "recall": float(rec),
            "precision": float(pre),
            "f2": float(f2),
            "auroc": auroc,
        }
        print(f"  {band:<8}  n={mask.sum():>5,}  thr={thr:.3f}  "
              f"rec={rec:.3f}  pre={pre:.3f}  F2={f2:.3f}  AUROC={auroc:.3f}")

    result = {
        "calibrators": calibrators,
        "thresholds":  thresholds,
        "strategy":    strategy,
        "recall_floor": recall_floor,
        "calibration_metrics": cal_metrics,
    }

    cal_path.write_text(json.dumps(result, indent=2))
    print(f"\n[calibrate] Saved calibration -> {cal_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Stage 2 per age group")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if calibration JSON exists")
    args = parser.parse_args()
    cfg  = load_config()
    calibrate(cfg, force=args.force)
