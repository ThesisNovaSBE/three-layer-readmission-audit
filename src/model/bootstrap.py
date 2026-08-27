"""Bootstrap confidence intervals for any (y_true, y_score) -> float metric.

Resamples at the patient level (``groups``, typically ``subject_id``) rather
than the admission level when groups are provided — a patient with multiple
admissions should have all of them resampled together, not independently, or
the resample understates variance for patients who appear more than once.
Falls back to admission-level resampling when no groups are given.

No metric currently reported anywhere in this codebase has a confidence
interval attached (remediation review task 0.6, still open as of session 15).
This module exists so that gap can be closed by wrapping existing metric
functions (``auroc``, ``auprc``, ``operating_point``'s components, etc.)
rather than reimplementing resampling logic per metric.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Return a point estimate and percentile bootstrap CI for a metric.

    Args:
        y_true:      binary labels.
        y_score:     scores/predictions passed to ``metric_fn``.
        metric_fn:   callable ``(y_true, y_score) -> float``, e.g.
                     ``src.model.metrics.auroc``.
        groups:      optional cluster labels (e.g. ``subject_id``) for
                     patient-level resampling. If ``None``, each admission is
                     resampled independently.
        n_resamples: number of bootstrap resamples (remediation review: 1000).
        confidence:  confidence level for the interval (default 95%).
        seed:        RNG seed for reproducibility.

    Returns:
        Dict with ``point_estimate`` (metric on the full, unresampled data),
        ``ci_lower``, ``ci_upper``, ``n_resamples_used`` (resamples where
        ``metric_fn`` did not raise — e.g. a resample with only one class
        present is skipped rather than crashing the whole CI), and
        ``n_resamples_requested``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)

    point_estimate = float(metric_fn(y_true, y_score))

    if groups is None:
        unique_groups = np.arange(len(y_true))
        group_of = unique_groups
    else:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_of = groups

    values: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(group_of == g)[0] for g in sampled])
        try:
            values.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            # e.g. only one class present in this resample — skip it rather
            # than fail the whole CI.
            continue

    alpha = (1.0 - confidence) / 2.0
    if values:
        ci_lower = float(np.percentile(values, 100 * alpha))
        ci_upper = float(np.percentile(values, 100 * (1 - alpha)))
    else:
        ci_lower = ci_upper = float("nan")

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_resamples_used": len(values),
        "n_resamples_requested": n_resamples,
    }
