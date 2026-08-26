"""Tests for patient-level bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.bootstrap import bootstrap_ci
from src.model.metrics import auroc


def test_bootstrap_ci_point_estimate_matches_full_data(binary_scores):
    """point_estimate must equal the metric computed on the unresampled data."""
    y, s = binary_scores
    result = bootstrap_ci(y, s, auroc, n_resamples=50)
    assert result["point_estimate"] == pytest.approx(auroc(y, s))


def test_bootstrap_ci_bounds_bracket_point_estimate(binary_scores):
    """The CI should (usually) contain the point estimate for a stable metric."""
    y, s = binary_scores
    result = bootstrap_ci(y, s, auroc, n_resamples=200, seed=1)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]


def test_bootstrap_ci_uses_requested_resample_count(binary_scores):
    """n_resamples_used must not exceed n_resamples_requested."""
    y, s = binary_scores
    result = bootstrap_ci(y, s, auroc, n_resamples=100)
    assert result["n_resamples_requested"] == 100
    assert result["n_resamples_used"] <= 100
    assert result["n_resamples_used"] > 0


def test_bootstrap_ci_respects_patient_level_groups():
    """Resampling by group must keep each patient's admissions together.

    With one patient contributing most of the positive labels, admission-level
    resampling and patient-level resampling should generally differ in CI
    width — this test just checks patient-level resampling runs correctly and
    produces a valid (non-degenerate) result rather than crashing on
    duplicate-index concatenation.
    """
    rng = np.random.default_rng(0)
    subject_id = np.repeat(np.arange(20), 5)  # 20 patients, 5 admissions each
    y = (rng.uniform(size=100) < 0.3).astype(int)
    s = np.clip(y * 0.5 + rng.uniform(0.0, 0.5, 100), 0.0, 1.0)

    result = bootstrap_ci(y, s, auroc, groups=subject_id, n_resamples=50, seed=2)
    assert result["n_resamples_used"] > 0
    assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0


def test_bootstrap_ci_skips_degenerate_resamples():
    """A metric that raises on single-class resamples must not crash the CI."""
    y = np.array([0, 0, 0, 0, 1])  # heavily imbalanced — some resamples will be all-0
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    result = bootstrap_ci(y, s, auroc, n_resamples=50, seed=3)
    assert result["n_resamples_used"] <= 50
