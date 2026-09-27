"""Tests for quantitative Stage 1-vs-Stage 2 discordance computation.

Moved out of test_stage3_explain.py (2026-09-27) alongside the extraction
of src/stage3/discordance.py from explain.py -- no behavioral change, same
tests, new home matching the new module.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.stage3.discordance import (
    DISCORDANCE_MODES,
    compute_discordance,
    sweep_discordance_thresholds,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_modes_tuple_nonempty():
    """DISCORDANCE_MODES must contain all three expected mode strings."""
    assert len(DISCORDANCE_MODES) == 3
    assert "CONCORDANT" in DISCORDANCE_MODES
    assert "NOTE_MITIGATES" in DISCORDANCE_MODES
    assert "NOTE_AMPLIFIES" in DISCORDANCE_MODES


# ── compute_discordance ─────────────────────────────────────────────────────────

def test_discordance_concordant_when_ranks_match():
    """Equal percentile ranks must be CONCORDANT."""
    cohort = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = compute_discordance(0.3, 0.3, cohort, cohort, displacement_pp=20.0)
    assert result["mode"] == "CONCORDANT"
    assert result["displacement"] == 0.0


def test_discordance_note_mitigates_when_stage2_ranks_lower():
    """Stage 2 ranking the patient much lower than Stage 1 must be NOTE_MITIGATES."""
    cohort_s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # 0.9 -> high rank
    cohort_s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.15])  # 0.15 -> low rank
    result = compute_discordance(0.9, 0.15, cohort_s1, cohort_s2, displacement_pp=20.0)
    assert result["mode"] == "NOTE_MITIGATES"
    assert result["displacement"] < 0


def test_discordance_note_amplifies_when_stage2_ranks_higher():
    """Stage 2 ranking the patient much higher than Stage 1 must be NOTE_AMPLIFIES."""
    cohort_s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.15])  # 0.15 -> low rank
    cohort_s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # 0.9 -> high rank
    result = compute_discordance(0.15, 0.9, cohort_s1, cohort_s2, displacement_pp=20.0)
    assert result["mode"] == "NOTE_AMPLIFIES"
    assert result["displacement"] > 0


def test_discordance_invariant_to_monotonic_rescaling():
    """Rank displacement must be unchanged by a monotonic rescaling of one score.

    This is the property raw-probability subtraction does not have: it makes
    the measure robust to Stage 1 and Stage 2 being unequally calibrated.
    """
    cohort_s1 = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80])
    cohort_s2 = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
    base = compute_discordance(0.50, 0.55, cohort_s1, cohort_s2)

    # Monotonic rescale of stage2's cohort and query value (e.g. a different
    # calibration curve) — ranks, and therefore displacement, must be identical.
    rescaled_cohort_s2 = cohort_s2**2
    rescaled = compute_discordance(0.50, 0.55**2, cohort_s1, rescaled_cohort_s2)
    assert rescaled["displacement"] == base["displacement"]
    assert rescaled["mode"] == base["mode"]


def test_discordance_empty_cohort_defaults_to_50th_percentile():
    """An empty cohort must not raise — falls back to the 50th percentile."""
    result = compute_discordance(0.5, 0.5, np.array([]), np.array([]))
    assert result["r1"] == 50.0
    assert result["r2"] == 50.0


# ── sweep_discordance_thresholds ─────────────────────────────────────────────

def test_sweep_default_thresholds():
    """Default sweep must cover the standard 10/15/20/25/30 pp range."""
    displacements = np.array([-40.0, -10.0, 0.0, 15.0, 45.0])
    sweep = sweep_discordance_thresholds(displacements)
    assert set(sweep.keys()) == {"10.0", "15.0", "20.0", "25.0", "30.0"}


def test_sweep_fractions_sum_to_one():
    """At every threshold, the three mode fractions must sum to 1."""
    rng = np.random.default_rng(0)
    displacements = rng.uniform(-100, 100, 200)
    sweep = sweep_discordance_thresholds(displacements)
    for dist in sweep.values():
        total = dist["NOTE_MITIGATES"] + dist["NOTE_AMPLIFIES"] + dist["CONCORDANT"]
        assert total == pytest.approx(1.0)


def test_sweep_narrower_threshold_flags_more_discordance():
    """A narrower threshold must classify at least as many patients as discordant."""
    displacements = np.array([-25.0, -18.0, -12.0, 5.0, 18.0, 25.0, 0.0])
    sweep = sweep_discordance_thresholds(displacements, thresholds_pp=[10.0, 30.0])
    discordant_10 = sweep["10.0"]["NOTE_MITIGATES"] + sweep["10.0"]["NOTE_AMPLIFIES"]
    discordant_30 = sweep["30.0"]["NOTE_MITIGATES"] + sweep["30.0"]["NOTE_AMPLIFIES"]
    assert discordant_10 >= discordant_30


def test_sweep_matches_compute_discordance_at_same_threshold():
    """The sweep's classification must agree with compute_discordance for a
    single patient at the same threshold — same rule, independently reached."""
    cohort = np.linspace(0.0, 1.0, 101)
    result = compute_discordance(0.20, 0.90, cohort, cohort, displacement_pp=20.0)
    sweep = sweep_discordance_thresholds(np.array([result["displacement"]]), [20.0])
    dist = sweep["20.0"]
    expected_mode = result["mode"]
    assert dist[expected_mode] == pytest.approx(1.0)


def test_sweep_empty_displacements_does_not_crash():
    """An empty input must return zeroed fractions, not raise (division by zero)."""
    sweep = sweep_discordance_thresholds(np.array([]), [20.0])
    assert sweep["20.0"] == {"NOTE_MITIGATES": 0.0, "NOTE_AMPLIFIES": 0.0, "CONCORDANT": 0.0}
