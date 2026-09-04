"""Stage 2 evaluation: per-age-group metrics, fairness gaps, and calibration error.

Loads ``models/stage2_results.csv`` (written by predict.py) and the feature
matrix (for age_band), then computes:

- Per-group AUROC, AUPRC, precision, recall, F2, ECE
- Fairness gaps: max − min across age bands
- Overall metrics on the notes cohort

Output: ``models/stage2_evaluation.json``

Usage::

    python -m src.stage2.evaluate
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix
# Aliased: Stage 2 must evaluate against the same target Stage 1 uses
# (MODEL_TARGET_COL, unplanned readmission per src/schemas.py).
from src.schemas import MODEL_TARGET_COL as TARGET_COL
from src.stage2._utils import band_key


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        frac_pos = labels[mask].mean()
        mean_prob = probs[mask].mean()
        ece += mask.sum() / total * abs(mean_prob - frac_pos)
    return float(ece)


def _f2_score(pre: float, rec: float) -> float:
    """Compute the F2 score from precision and recall."""
    return (5 * pre * rec) / (4 * pre + rec + 1e-9)


def _group_metrics(
    df: pd.DataFrame,
    score_col: str,
    confirmed_col: str,
) -> dict:
    """Compute all evaluation metrics for one subgroup DataFrame.

    Args:
        df:            subset of the results DataFrame for this group.
        score_col:     column name for predicted probabilities.
        confirmed_col: column name for binary predictions.

    Returns:
        Metrics dict with keys: n, pos_rate, auroc, auprc, recall,
        precision, f2, ece.
    """
    y_true = df[TARGET_COL].values
    probs = df[score_col].values
    preds = df[confirmed_col].values

    rec = recall_score(y_true, preds, zero_division=0)
    pre = precision_score(y_true, preds, zero_division=0)
    try:
        auroc_val = float(roc_auc_score(y_true, probs))
    except ValueError:
        auroc_val = float("nan")
    try:
        auprc_val = float(average_precision_score(y_true, probs))
    except ValueError:
        auprc_val = float("nan")

    return {
        "n": int(len(df)),
        "pos_rate": float(y_true.mean()),
        "auroc": auroc_val,
        "auprc": auprc_val,
        "recall": float(rec),
        "precision": float(pre),
        "f2": float(_f2_score(pre, rec)),
        "ece": _ece(probs, y_true),
    }


def evaluate(cfg: AppConfig, artifact: dict | None = None) -> dict:
    """Compute per-age-group evaluation metrics on the test set.

    Args:
        cfg:      validated project config.
        artifact: pre-loaded Stage 1 artifact; loaded from disk if ``None``.

    Returns:
        Evaluation dict (also written to ``models/stage2_evaluation.json``).

    Raises:
        FileNotFoundError: if Stage 2 results CSV does not exist.
    """
    model_dir = get_model_dir()
    results_csv = model_dir / "stage2_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Stage 2 results not found at {results_csv}. "
            "Run `python -m src.stage2.predict` first."
        )

    if artifact is None:
        artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")
    mode = artifact["mode"]

    results = pd.read_csv(results_csv)

    matrix = load_feature_matrix(cfg, mode)
    age_df = matrix[["hadm_id", "age_band"]].copy()
    age_df["age_band_key"] = age_df["age_band"].apply(band_key)

    results = results.merge(age_df[["hadm_id", "age_band_key"]], on="hadm_id", how="left")
    missing_band = results["age_band_key"].isna().sum()
    if missing_band > 0:
        print(f"[evaluate] WARNING: {missing_band} rows missing age_band — excluded")
    results = results.dropna(subset=["age_band_key"])

    score_col = "stage2_score"
    confirmed_col = "stage2_confirmed"

    overall = _group_metrics(results, score_col, confirmed_col)
    print(f"\n[evaluate] Overall (notes cohort, n={overall['n']:,})")
    print(
        f"  AUROC={overall['auroc']:.3f}  AUPRC={overall['auprc']:.3f}  "
        f"Recall={overall['recall']:.3f}  Precision={overall['precision']:.3f}  "
        f"F2={overall['f2']:.3f}  ECE={overall['ece']:.4f}"
    )

    group_results: dict[str, dict] = {}
    bands = sorted(results["age_band_key"].unique())

    print("\n[evaluate] Per-age-group metrics:")
    print(
        f"  {'Band':<8} {'N':>6} {'AUROC':>7} {'Recall':>8} "
        f"{'Precision':>10} {'F2':>6} {'ECE':>7}"
    )
    print("  " + "-" * 55)

    for band in bands:
        sub = results[results["age_band_key"] == band]
        metrics = _group_metrics(sub, score_col, confirmed_col)
        group_results[band] = metrics
        print(
            f"  {band:<8} {metrics['n']:>6,} {metrics['auroc']:>7.3f} "
            f"{metrics['recall']:>8.3f} {metrics['precision']:>10.3f} "
            f"{metrics['f2']:>6.3f} {metrics['ece']:>7.4f}"
        )

    recalls = [m["recall"] for m in group_results.values()]
    precisions = [m["precision"] for m in group_results.values()]
    aurocs = [m["auroc"] for m in group_results.values() if not np.isnan(m["auroc"])]

    fairness = {
        "recall_gap": float(max(recalls) - min(recalls)),
        "precision_gap": float(max(precisions) - min(precisions)),
        "auroc_gap": float(max(aurocs) - min(aurocs)) if aurocs else float("nan"),
        "worst_recall_band": bands[recalls.index(min(recalls))],
        "worst_precision_band": bands[precisions.index(min(precisions))],
    }
    print(
        f"\n[evaluate] Fairness gaps:"
        f"\n  Recall gap:    {fairness['recall_gap']:.3f}  "
        f"(worst: {fairness['worst_recall_band']})"
        f"\n  Precision gap: {fairness['precision_gap']:.3f}  "
        f"(worst: {fairness['worst_precision_band']})"
        f"\n  AUROC gap:     {fairness['auroc_gap']:.3f}"
    )

    output = {
        "overall": overall,
        "by_age_band": group_results,
        "fairness_gaps": fairness,
    }

    out_path = model_dir / "stage2_evaluation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[evaluate] Saved -> {out_path}")
    return output


def main() -> None:
    """CLI entry point for Stage 2 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 per age group")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()
    _cfg = load_config()
    if args.mode:
        _cfg.run.mode = args.mode
    evaluate(_cfg)


if __name__ == "__main__":
    main()
