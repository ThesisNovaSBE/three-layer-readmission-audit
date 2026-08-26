"""Stage 1 probability calibration (isotonic regression).

Raw XGBoost probabilities are known to be poorly calibrated (session 15
found Brier=0.2135, worse than a constant base-rate predictor). This module
fits an isotonic calibrator on out-of-fold training predictions — already
computed for threshold selection in ``train.py``, so this adds no new
leakage surface — and stores it in the Stage 1 artifact so every downstream
consumer (evaluate.py, evaluate_pipeline.py, stage2/predict.py, api.py)
scores patients on calibrated probabilities.

Isotonic regression is monotonic, so it does not change AUROC/AUPRC or which
admissions get flagged under a capacity- or recall-based policy (both only
depend on rank order) — it changes what the reported probability *means*,
which matters for Brier score, any threshold reported as a probability, and
the percentile-rank discordance measure Stage 3 relies on (which is already
rank-invariant, but a well-calibrated score is still the more defensible
input to reason over).
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def fit_isotonic_calibrator(
    y_true: np.ndarray, y_score: np.ndarray
) -> IsotonicRegression:
    """Fit an isotonic regression calibrator.

    Args:
        y_true:  binary labels.
        y_score: uncalibrated model scores (e.g. out-of-fold predictions).

    Returns:
        Fitted :class:`~sklearn.isotonic.IsotonicRegression`, clipped to
        [0, 1] out-of-bounds behaviour so it's safe to apply to any future
        score, not just ones inside the training range.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(y_score, y_true)
    return calibrator


def apply_calibration(artifact: dict, scores: np.ndarray) -> np.ndarray:
    """Apply the artifact's stored calibrator to raw scores, if one exists.

    Backward compatible: artifacts trained before calibration was added
    (no ``"calibrator"`` key) pass raw scores through unchanged.

    Args:
        artifact: Stage 1 artifact dict (from ``joblib.load``).
        scores:   raw ``estimator.predict_proba(...)[:, 1]`` scores.

    Returns:
        Calibrated scores, or ``scores`` unchanged if no calibrator is stored.
    """
    calibrator = artifact.get("calibrator")
    if calibrator is None:
        return np.asarray(scores)
    return calibrator.predict(np.asarray(scores))


def calibration_report(
    y_true: np.ndarray, raw_scores: np.ndarray, calibrated_scores: np.ndarray
) -> dict:
    """Return a before/after calibration report.

    Includes the isotonic regression's fitted breakpoints as a simple
    reliability-curve representation (x = raw score, y = calibrated
    probability at each breakpoint) — avoids a plotting dependency while
    still letting anyone reconstruct the calibration curve.

    Args:
        y_true:            binary labels.
        raw_scores:        uncalibrated scores for the same population.
        calibrated_scores: calibrated scores for the same population.

    Returns:
        Dict with ``brier_before``, ``brier_after``, ``n``, ``base_rate``.
    """
    return {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "brier_before": float(brier_score_loss(y_true, raw_scores)),
        "brier_after": float(brier_score_loss(y_true, calibrated_scores)),
    }
