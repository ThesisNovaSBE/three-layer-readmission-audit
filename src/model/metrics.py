"""Metrics for imbalanced readmission prediction.

Primary: AUPRC (average precision). Secondary: AUROC.
Operating point reporting for the high-recall mandate, plus calibration (Brier)
and per-subgroup AUROC for a fairness check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the area under the precision-recall curve."""
    return float(average_precision_score(y_true, y_score))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the area under the ROC curve."""
    return float(roc_auc_score(y_true, y_score))


def select_threshold_for_recall(
    y_true: np.ndarray, y_score: np.ndarray, target_recall: float
) -> float:
    """Return the highest probability threshold whose recall is still >= target_recall.

    Rationale: lowering the threshold only raises recall and lowers precision, so
    the precision-maximising operating point that still meets the recall floor is
    the *highest* threshold satisfying recall >= target. Stage 2 then recovers
    precision from the flagged set. Falls back to 0.0 (flag everything) if the
    target recall is unreachable.
    """
    _, recall, thresholds = precision_recall_curve(y_true, y_score)
    # recall/precision have length len(thresholds)+1; align with thresholds[:].
    rec = recall[:-1]
    valid = np.where(rec >= target_recall)[0]
    if len(valid) == 0:
        return 0.0
    return float(thresholds[valid[-1]])


def operating_point(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict:
    """Return performance metrics at a fixed decision threshold."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "f2": float(f2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def metrics_at_recall_points(
    y_true: np.ndarray, y_score: np.ndarray, recall_points: list[float]
) -> list[dict]:
    """Return precision/specificity trade-off at several recall targets."""
    out = []
    for r in recall_points:
        thr = select_threshold_for_recall(y_true, y_score, r)
        op = operating_point(y_true, y_score, thr)
        op["recall_target"] = float(r)
        out.append(op)
    return out


def subgroup_auroc(
    y_true: np.ndarray, y_score: np.ndarray, subgroups: pd.DataFrame
) -> dict:
    """Return AUROC within each level of each subgroup column (fairness check)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    result: dict[str, dict] = {}
    for col in subgroups.columns:
        result[col] = {}
        for level, idx in subgroups.groupby(col, observed=True).groups.items():
            mask = subgroups.index.isin(idx)
            yt, ys = y_true[mask], y_score[mask]
            if len(np.unique(yt)) < 2:
                result[col][str(level)] = {
                    "auroc": None,
                    "n": int(mask.sum()),
                    "pos_rate": float(yt.mean()) if len(yt) else None,
                }
            else:
                result[col][str(level)] = {
                    "auroc": auroc(yt, ys),
                    "n": int(mask.sum()),
                    "pos_rate": float(yt.mean()),
                }
    return result


def full_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    recall_points: list[float],
    subgroups: pd.DataFrame | None = None,
) -> dict:
    """Return a complete evaluation report dict."""
    report = {
        "auprc": auprc(y_true, y_score),
        "auroc": auroc(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
        "operating_point": operating_point(y_true, y_score, threshold),
        "recall_tradeoff": metrics_at_recall_points(y_true, y_score, recall_points),
    }
    if subgroups is not None:
        report["subgroup_auroc"] = subgroup_auroc(y_true, y_score, subgroups)
    return report
