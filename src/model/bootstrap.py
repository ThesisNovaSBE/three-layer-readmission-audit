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
import pandas as pd


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

    # Precompute each group's row positions ONCE via pandas' Cython groupby
    # (O(n log n) total), not per resample. The previous version called
    # np.where(group_of == g) once per SAMPLED group PER resample --
    # O(n_unique_groups * n) work, repeated n_resamples times -- which at
    # tens of thousands of unique patients and n_resamples=1000 made this
    # function impractically slow in practice (confirmed: a real
    # evaluate_pipeline.py run needing 4 bootstrap_ci calls at ~100k-row
    # scale ran 39+ minutes without finishing). A dict lookup replaces the
    # per-resample scan; the sequence of resamples drawn from ``rng`` is
    # unchanged, so results are numerically identical, just much faster.
    if groups is None:
        unique_groups = np.arange(len(y_true))
        group_indices: dict | None = None
    else:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_indices = pd.Series(groups).groupby(groups).indices

    values: list[float] = []
    for _ in range(n_resamples):
        if group_indices is None:
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_indices[g] for g in sampled])
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
