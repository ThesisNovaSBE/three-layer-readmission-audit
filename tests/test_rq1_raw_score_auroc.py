"""Tests for scripts/rq1_raw_score_auroc.py -- the Platt inversion, on
synthetic data only (no files, no model, no feature matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.rq1_raw_score_auroc import (
    _platt_forward,
    _platt_inverse,
    _recover_raw_stage2_logit,
)


def test_platt_inverse_round_trips_on_synthetic():
    """For p = sigmoid(a*z + b), inverting then re-applying must recover p and z.

    z is kept in a range where p stays inside the inverse's [1e-7, 1-1e-7] clip
    (the realistic regime); for p saturated past the clip only the probability
    round-trip is guaranteed, not exact z recovery."""
    rng = np.random.default_rng(0)
    z = rng.uniform(-4.0, 4.0, size=500)
    for a, b in [(2.0140, -2.1804), (1.6, -2.089), (0.5, 0.3)]:
        p = _platt_forward(z, a, b)
        z_rec = _platt_inverse(p, a, b)
        assert np.allclose(z_rec, z, atol=1e-9)
        assert np.max(np.abs(_platt_forward(z_rec, a, b) - p)) < 1e-6


def test_platt_inverse_rejects_nonpositive_slope():
    """a <= 0 would flip the ranking -- must raise."""
    with pytest.raises(ValueError, match="not > 0"):
        _platt_inverse(np.array([0.3, 0.6]), a=0.0, b=0.1)
    with pytest.raises(ValueError, match="not > 0"):
        _platt_inverse(np.array([0.3, 0.6]), a=-1.5, b=0.1)


def test_recover_raw_stage2_logit_per_band():
    """Per-band inversion recovers each band's raw logit and round-trips < 1e-6."""
    cal = {"18-40": [2.0, -2.18], "70+": [1.6, -2.09]}
    rng = np.random.default_rng(1)
    z_true = rng.uniform(-4.0, 4.0, size=400)
    bands = np.array(["18-40", "70+"] * 200)
    stage2_score = np.empty(400)
    for band in ("18-40", "70+"):
        a, b = cal[band]
        m = bands == band
        stage2_score[m] = _platt_forward(z_true[m], a, b)

    z_rec, max_err = _recover_raw_stage2_logit(stage2_score, bands, cal)
    assert max_err < 1e-6
    assert np.allclose(z_rec, z_true, atol=1e-9)
    # AUROC is monotone-invariant within a band but the recovered logit is the
    # actual z used for the cross-band (raw) ranking.
    assert z_rec.shape == z_true.shape


def test_recover_raw_stage2_logit_missing_band_fails():
    """A band present in the data but absent from the calibrators must abort."""
    cal = {"18-40": [2.0, -2.18]}
    stage2_score = np.array([0.2, 0.5, 0.7])
    bands = np.array(["18-40", "70+", "18-40"])
    with pytest.raises(SystemExit, match="no Platt calibrator"):
        _recover_raw_stage2_logit(stage2_score, bands, cal)
