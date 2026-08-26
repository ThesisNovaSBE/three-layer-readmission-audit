"""Tests for Stage 1 isotonic calibration (pure, no data required)."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.calibration import (
    apply_calibration,
    calibration_report,
    fit_isotonic_calibrator,
)


def test_fit_isotonic_calibrator_improves_brier(binary_scores):
    """Calibration must not make Brier score worse on the data it was fit on."""
    y, s = binary_scores
    # Deliberately miscalibrate: push scores toward the extremes.
    miscalibrated = np.clip(s * 2.5 - 0.3, 0.0, 1.0)
    calibrator = fit_isotonic_calibrator(y, miscalibrated)
    calibrated = calibrator.predict(miscalibrated)
    report = calibration_report(y, miscalibrated, calibrated)
    assert report["brier_after"] <= report["brier_before"]


def test_calibrator_output_bounded():
    """Calibrated scores must stay within [0, 1] even for out-of-range inputs."""
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    calibrator = fit_isotonic_calibrator(y, s)
    out = calibrator.predict(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_calibrator_preserves_rank_order():
    """Isotonic calibration must be monotonic — it cannot reorder scores."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.15, 0.3, 0.35, 0.5, 0.6, 0.8, 0.9])
    calibrator = fit_isotonic_calibrator(y, s)
    calibrated = calibrator.predict(s)
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_apply_calibration_passes_through_without_calibrator():
    """Artifacts with no stored calibrator must return raw scores unchanged."""
    scores = np.array([0.1, 0.5, 0.9])
    out = apply_calibration({}, scores)
    assert np.array_equal(out, scores)


def test_apply_calibration_uses_stored_calibrator():
    """apply_calibration must use the artifact's calibrator when present."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.2, 0.3, 0.7, 0.8])
    calibrator = fit_isotonic_calibrator(y, s)
    out = apply_calibration({"calibrator": calibrator}, s)
    assert np.array_equal(out, calibrator.predict(s))


def test_calibration_report_keys(binary_scores):
    """calibration_report must include all expected keys."""
    y, s = binary_scores
    calibrator = fit_isotonic_calibrator(y, s)
    report = calibration_report(y, s, calibrator.predict(s))
    for key in ("n", "base_rate", "brier_before", "brier_after"):
        assert key in report


def test_calibration_report_base_rate(binary_scores):
    """base_rate must equal the actual positive fraction."""
    y, s = binary_scores
    calibrator = fit_isotonic_calibrator(y, s)
    report = calibration_report(y, s, calibrator.predict(s))
    assert report["base_rate"] == pytest.approx(y.mean())
