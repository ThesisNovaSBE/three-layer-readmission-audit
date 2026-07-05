"""Stage 2 evaluation: per-age-group metrics, fairness gap, and calibration error.

Loads models/stage2_results.csv (written by predict.py) and the feature matrix
(for age_band), then computes:
  - Per-group AUROC, AUPRC, precision, recall, F2, ECE
  - Fairness gaps: max - min across age bands
  - Overall metrics on the notes cohort

Output: models/stage2_evaluation.json

Usage:
    python -m src.stage2.evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL

import joblib


_BAND_MAP = {
    "(17, 40]":  "18-40",
    "(40, 55]":  "41-55",
    "(55, 70]":  "56-70",
    "(70, 120]": "70+",
}


def _band_key(age_band) -> str:
    return _BAND_MAP.get(str(age_band), str(age_band))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    total = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        frac_pos = labels[mask].mean()
        mean_prob = probs[mask].mean()
        ece += mask.sum() / total * abs(mean_prob - frac_pos)
    return float(ece)


def _f2(pre: float, rec: float) -> float:
    return (5 * pre * rec) / (4 * pre + rec + 1e-9)


def _group_metrics(
    df: pd.DataFrame,
    score_col: str,
    confirmed_col: str,
) -> dict:
    y     = df[TARGET_COL].values
    probs = df[score_col].values
    preds = df[confirmed_col].values

    rec = recall_score(y, preds, zero_division=0)
    pre = precision_score(y, preds, zero_division=0)
    try:
        auroc = float(roc_auc_score(y, probs))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, probs))
    except ValueError:
        auprc = float("nan")

    return {
        "n": int(len(df)),
        "pos_rate": float(y.mean()),
        "auroc":   auroc,
        "auprc":   auprc,
        "recall":  float(rec),
        "precision": float(pre),
        "f2":      float(_f2(pre, rec)),
        "ece":     _ece(probs, y),
    }


def evaluate(cfg: dict, artifact: dict | None = None) -> dict:
    """Compute per-age-group evaluation metrics on the test set.

    Args:
        cfg:      loaded config dict.
        artifact: pre-loaded Stage 1 artifact; loaded from disk if None.

    Returns:
        Evaluation dict (also written to models/stage2_evaluation.json).
    """
    model_dir   = get_model_dir()
    results_csv = model_dir / "stage2_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Stage 2 results not found at {results_csv}. "
            "Run `python -m src.stage2.predict` first."
        )

    # Load Stage 1 artifact to get mode
    if artifact is None:
        stage1_name = cfg["stage1"]["model"]
        artifact = joblib.load(model_dir / f"stage1_{stage1_name}.joblib")
    mode = artifact["mode"]

    results = pd.read_csv(results_csv)

    # Join with age_band from feature matrix
    matrix = load_feature_matrix(cfg, mode)
    age_df = matrix[["hadm_id", "age_band"]].copy()
    age_df["age_band_key"] = age_df["age_band"].apply(_band_key)

    results = results.merge(age_df[["hadm_id", "age_band_key"]], on="hadm_id", how="left")
    missing_band = results["age_band_key"].isna().sum()
    if missing_band > 0:
        print(f"[evaluate] WARNING: {missing_band} rows missing age_band — excluded")
    results = results.dropna(subset=["age_band_key"])

    score_col     = "stage2_score"
    confirmed_col = "stage2_confirmed"

    # ── Overall (notes cohort) ──
    overall = _group_metrics(results, score_col, confirmed_col)
    print(f"\n[evaluate] Overall (notes cohort, n={overall['n']:,})")
    print(f"  AUROC={overall['auroc']:.3f}  AUPRC={overall['auprc']:.3f}  "
          f"Recall={overall['recall']:.3f}  Precision={overall['precision']:.3f}  "
          f"F2={overall['f2']:.3f}  ECE={overall['ece']:.4f}")

    # ── Per-group ──
    group_results: dict[str, dict] = {}
    print("\n[evaluate] Per-age-group metrics:")
    print(f"  {'Band':<8} {'N':>6} {'AUROC':>7} {'Recall':>8} {'Precision':>10} "
          f"{'F2':>6} {'ECE':>7}")
    print("  " + "-" * 55)

    bands = sorted(results["age_band_key"].unique())
    for band in bands:
        sub = results[results["age_band_key"] == band]
        m   = _group_metrics(sub, score_col, confirmed_col)
        group_results[band] = m
        print(f"  {band:<8} {m['n']:>6,} {m['auroc']:>7.3f} {m['recall']:>8.3f} "
              f"{m['precision']:>10.3f} {m['f2']:>6.3f} {m['ece']:>7.4f}")

    # ── Fairness gaps ──
    recalls    = [m["recall"]    for m in group_results.values()]
    precisions = [m["precision"] for m in group_results.values()]
    aurocs     = [m["auroc"]     for m in group_results.values() if not np.isnan(m["auroc"])]

    fairness = {
        "recall_gap":    float(max(recalls)    - min(recalls)),
        "precision_gap": float(max(precisions) - min(precisions)),
        "auroc_gap":     float(max(aurocs)     - min(aurocs)) if aurocs else float("nan"),
        "worst_recall_band":    bands[recalls.index(min(recalls))],
        "worst_precision_band": bands[precisions.index(min(precisions))],
    }
    print(f"\n[evaluate] Fairness gaps:")
    print(f"  Recall gap:    {fairness['recall_gap']:.3f}  "
          f"(worst: {fairness['worst_recall_band']})")
    print(f"  Precision gap: {fairness['precision_gap']:.3f}  "
          f"(worst: {fairness['worst_precision_band']})")
    print(f"  AUROC gap:     {fairness['auroc_gap']:.3f}")

    output = {
        "overall":       overall,
        "by_age_band":   group_results,
        "fairness_gaps": fairness,
    }

    out_path = model_dir / "stage2_evaluation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[evaluate] Saved -> {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 per age group")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()
    cfg  = load_config()
    if args.mode:
        cfg["run"]["mode"] = args.mode
    evaluate(cfg)
