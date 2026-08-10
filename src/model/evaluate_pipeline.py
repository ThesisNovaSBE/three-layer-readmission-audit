"""End-to-end pipeline evaluation on the Stage 1 held-out test set.

Both Stage 1 and Stage 2 models are evaluated on patients that neither model
has ever seen:

- Stage 1 is evaluated on ``artifact["test_idx"]`` — its standard held-out split.
- Stage 2 scores for those same patients come from ``stage2_results.csv``.
  Because Stage 2 was fine-tuned only on Stage 1's *training* partition, any
  patient in Stage 1's test partition is also out-of-sample for Stage 2.

The script reports three evaluation layers:

1. **Stage 1 alone** — AUROC / AUPRC / recall / precision on the test partition.
2. **Stage 2 alone** — AUROC / recall / precision on the subset of test-partition
   patients that were Stage 1-flagged *and* had a discharge note (the Stage 2
   inference population).
3. **Full pipeline** — treating "Stage 2 confirmed" as the final binary prediction
   for all test-partition patients (unconfirmed = negative regardless of Stage 1
   score), reports pipeline-level precision / recall / F1 / F2.

Per-age-band breakdowns are included for all three layers.

Output: ``models/pipeline_evaluation.json`` (printed summary + JSON).

Usage::

    python -m src.model.evaluate_pipeline
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.metrics import auprc as compute_auprc


# ── helpers ───────────────────────────────────────────────────────────────────

def _precision_recall_f(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision, recall, F1, F2 from binary arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2   = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 5), "recall": round(rec, 5),
        "f1": round(f1, 5), "f2": round(f2, 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _band_breakdown(
    y_true: np.ndarray,
    scores: np.ndarray | None,
    pred: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Per-age-band metrics breakdown."""
    bands: dict = {}
    for band in sorted(subgroups["age_band"].dropna().unique(), key=str):
        mask = (subgroups["age_band"] == band).values
        yt = y_true[mask]
        yp = pred[mask]
        entry: dict = {"n": int(mask.sum()), "pos_rate": float(yt.mean())}
        if scores is not None and len(np.unique(yt)) > 1:
            entry["auroc"] = float(roc_auc_score(yt, scores[mask]))
        entry.update(_precision_recall_f(yt, yp))
        bands[str(band)] = entry
    return bands


def _stage1_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    subgroups: pd.DataFrame,
) -> dict:
    """Stage 1 evaluation report dict."""
    pred = (scores >= threshold).astype(int)
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(compute_auprc(y_true, scores)),
        "threshold": float(threshold),
    }
    report.update(_precision_recall_f(y_true, pred))
    report["by_age_band"] = _band_breakdown(y_true, scores, pred, subgroups)
    return report


def _stage2_report(
    y_true: np.ndarray,
    s2_scores: np.ndarray,
    s2_confirmed: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Stage 2 evaluation report dict (only Stage 1-flagged patients with notes)."""
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        report["auroc"] = float(roc_auc_score(y_true, s2_scores))
        report["auprc"] = float(compute_auprc(y_true, s2_scores))
    else:
        report["auroc"] = None
        report["auprc"] = None
    report.update(_precision_recall_f(y_true, s2_confirmed))
    report["by_age_band"] = _band_breakdown(y_true, s2_scores, s2_confirmed, subgroups)
    return report


def _pipeline_report(
    y_true: np.ndarray,
    pipeline_pred: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Pipeline-level evaluation treating Stage 2 confirmed as the final label."""
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "confirmed_rate": float(pipeline_pred.mean()),
    }
    report.update(_precision_recall_f(y_true, pipeline_pred))
    report["by_age_band"] = _band_breakdown(y_true, None, pipeline_pred, subgroups)
    return report


def _load_test_partition(
    artifact: dict, cfg: AppConfig
) -> tuple[object, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load Stage 1 test partition features, labels, subgroups and hadm_ids.

    Returns:
        (x_test, y_test, sub_test, hadm_test)
    """
    matrix = load_feature_matrix(cfg, artifact["mode"])
    features, y, _, subgroups, _ = split_xy(matrix)
    idx = artifact["test_idx"]
    x_test = features.iloc[idx][artifact["feature_cols"]]
    y_test = y[idx]
    sub_test = subgroups.iloc[idx].reset_index(drop=True)
    hadm_test = matrix.iloc[idx]["hadm_id"].values
    return x_test, y_test, sub_test, hadm_test


def _eval_stage2(
    model_dir: object,
    hadm_test: np.ndarray,
    y_test: np.ndarray,
    sub_test: pd.DataFrame,
    s1_scores: np.ndarray,
    s1_threshold: float,
) -> tuple[dict, np.ndarray]:
    """Load Stage 2 results for the test partition and compute evaluation report.

    Args:
        model_dir:     path to models directory.
        hadm_test:     hadm_ids for the Stage 1 test partition.
        y_test:        ground-truth labels aligned to hadm_test.
        sub_test:      subgroup dataframe aligned to hadm_test.
        s1_scores:     Stage 1 probability scores aligned to hadm_test.
        s1_threshold:  Stage 1 classification threshold.

    Returns:
        (s2_report, pipeline_pred) — pipeline_pred is the final binary output.
    """
    s2_path = model_dir / "stage2_results.csv"
    if not s2_path.exists():
        print("[pipeline_eval] stage2_results.csv not found — skipping Stage 2 layer.")
        return {"available": False}, (s1_scores >= s1_threshold).astype(int)

    s2_all = pd.read_csv(s2_path)
    s2_rows = s2_all[s2_all["hadm_id"].isin(set(hadm_test))].set_index("hadm_id")

    s2_scores_arr = np.array([s2_rows["stage2_score"].get(h, float("nan")) for h in hadm_test])
    s2_conf_arr = np.array([s2_rows["stage2_confirmed"].get(h, 0) for h in hadm_test], dtype=int)
    has_s2 = ~np.isnan(s2_scores_arr)

    if has_s2.sum() == 0:
        return {"available": False}, s2_conf_arr

    sub_s2 = sub_test[has_s2].reset_index(drop=True)
    report = _stage2_report(y_test[has_s2], s2_scores_arr[has_s2], s2_conf_arr[has_s2], sub_s2)
    report["available"] = True
    report["n_test_with_s2_scores"] = int(has_s2.sum())
    print(f"[pipeline_eval] Stage 2 (flagged+notes n={has_s2.sum():,}): "
          f"AUROC={report.get('auroc', 'n/a')}  "
          f"recall={report['recall']:.3f}  precision={report['precision']:.3f}")
    return report, s2_conf_arr


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_pipeline(cfg: AppConfig) -> dict:
    """Run the end-to-end pipeline evaluation and save the report.

    Args:
        cfg: validated project config.

    Returns:
        Dict with stage1, stage2, and pipeline sub-reports.
    """
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    x_test, y_test, sub_test, hadm_test = _load_test_partition(artifact, cfg)
    s1_scores = artifact["estimator"].predict_proba(x_test)[:, 1]
    s1_report = _stage1_report(y_test, s1_scores, artifact["threshold"], sub_test)
    print(f"\n[pipeline_eval] Stage 1 (test n={s1_report['n']:,}): "
          f"AUROC={s1_report['auroc']:.4f}  "
          f"recall={s1_report['recall']:.3f}  precision={s1_report['precision']:.3f}")

    s2_report, pipeline_pred = _eval_stage2(
        model_dir, hadm_test, y_test, sub_test, s1_scores, artifact["threshold"]
    )
    pipe_report = _pipeline_report(y_test, pipeline_pred, sub_test)
    print(f"[pipeline_eval] Pipeline end-to-end (n={pipe_report['n']:,}): "
          f"precision={pipe_report['precision']:.3f}  "
          f"recall={pipe_report['recall']:.3f}  "
          f"F1={pipe_report['f1']:.3f}  F2={pipe_report['f2']:.3f}")

    report = {
        "note": (
            "Stage 1 test partition used for all three layers. "
            "Stage 2 scores come from stage2_results.csv — "
            "patients in Stage 1's test partition are out-of-sample for Stage 2."
        ),
        "stage1": s1_report,
        "stage2": s2_report,
        "pipeline": pipe_report,
    }
    _print_band_table(s1_report, s2_report, pipe_report)

    out_path = model_dir / "pipeline_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[pipeline_eval] Saved -> {out_path}")
    return report


def _print_band_table(s1: dict, s2: dict, pipe: dict) -> None:
    """Print a per-age-band comparison table."""
    print("\n  Age band  │ S1 AUROC │ S1 Rec │ S2 AUROC │ S2 Prec │ Pipe F1")
    print("  " + "─" * 64)
    s1_bands = s1.get("by_age_band", {})
    s2_bands = s2.get("by_age_band", {}) if s2.get("available") else {}
    pi_bands = pipe.get("by_age_band", {})
    for band in sorted(s1_bands.keys(), key=str):
        s1b = s1_bands[band]
        s2b = s2_bands.get(band, {})
        pib = pi_bands.get(band, {})
        s1_auroc = f"{s1b.get('auroc', 0):.3f}" if s1b.get("auroc") else "  n/a"
        s2_auroc = f"{s2b.get('auroc', 0):.3f}" if s2b.get("auroc") else "  n/a"
        print(
            f"  {band:<9} │  {s1_auroc}  │  {s1b.get('recall', 0):.3f} "
            f"│   {s2_auroc} │  {s2b.get('precision', 0):.3f}  │  {pib.get('f1', 0):.3f}"
        )


def main() -> None:
    """CLI entry point for end-to-end pipeline evaluation."""
    cfg = load_config()
    evaluate_pipeline(cfg)


if __name__ == "__main__":
    main()
