"""Metrics for imbalanced readmission prediction.

Primary: AUPRC (average precision). Secondary: AUROC.
Operating point reporting for the capacity-constrained primary policy (with
recall-floor as a secondary comparison table), plus calibration (Brier) and
per-subgroup AUROC for a fairness check.
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


def select_threshold_for_capacity(y_score: np.ndarray, capacity_k: float) -> float:
    """Return the threshold that flags exactly the top ``capacity_k`` fraction.

    Rationale: a recall floor answers "how many alerts until we catch enough
    true positives" — it says nothing about whether a hospital can act on the
    resulting alert volume. A capacity constraint answers a different,
    deployable question directly: "given a transitional-care team that can
    follow up on K% of discharges, who are the highest-risk K%?" This is the
    primary Stage 1 operating-point policy as of session 15 (2026-08-25);
    ``select_threshold_for_recall`` is retained for comparability with prior
    literature, which mostly reports recall-floor operating points.

    Args:
        y_score:    predicted probabilities.
        capacity_k: fraction of the population to flag, in (0, 1].

    Returns:
        The score threshold such that approximately ``capacity_k`` of
        ``y_score`` is >= threshold.
    """
    if not 0.0 < capacity_k <= 1.0:
        raise ValueError(f"capacity_k must be in (0, 1], got {capacity_k}")
    return float(np.quantile(y_score, 1.0 - capacity_k))


def metrics_at_capacity_points(
    y_true: np.ndarray, y_score: np.ndarray, capacity_points: list[float]
) -> list[dict]:
    """Return precision@K / lift@K / recall@K at several capacity fractions K.

    ``lift`` is precision@K divided by the base rate — how many times better
    than randomly screening K% of admissions the model does.
    """
    base_rate = float(np.mean(y_true))
    out = []
    for k in capacity_points:
        thr = select_threshold_for_capacity(y_score, k)
        op = operating_point(y_true, y_score, thr)
        op["capacity_k"] = float(k)
        op["lift"] = float(op["precision"] / base_rate) if base_rate > 0 else 0.0
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
    capacity_points: list[float] | None = None,
) -> dict:
    """Return a complete evaluation report dict.

    ``operating_point`` is evaluated at ``threshold`` — the model's primary,
    already-selected operating point (capacity-constrained as of session 15,
    2026-08-25; see ``select_threshold_for_capacity``). ``recall_tradeoff`` is
    reported for comparability with prior literature. ``capacity_tradeoff`` is
    reported when ``capacity_points`` is given, independent of which policy
    produced ``threshold``.
    """
    report = {
        "auprc": auprc(y_true, y_score),
        "auroc": auroc(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
        "operating_point": operating_point(y_true, y_score, threshold),
        "recall_tradeoff": metrics_at_recall_points(y_true, y_score, recall_points),
    }
    if capacity_points:
        report["capacity_tradeoff"] = metrics_at_capacity_points(
            y_true, y_score, capacity_points
        )
    if subgroups is not None:
        report["subgroup_auroc"] = subgroup_auroc(y_true, y_score, subgroups)
    return report
