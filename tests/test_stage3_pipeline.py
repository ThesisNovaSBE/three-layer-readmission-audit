"""Tests for src/stage3/pipeline.py helpers that resolve Stage 1 scores for
the discordance ranking -- specifically that r1 ranks on the RAW (tie-free)
Stage 1 score, and that an older results file (no stage1_score_raw column)
fails loudly rather than silently ranking on the tied calibrated score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stage3.discordance import _percentile_rank, compute_discordance
from src.stage3.pipeline import _cohort_scores, _lookup_patient, _prepare_patient


def _results_with_raw() -> pd.DataFrame:
    """Calibrated Stage 1 score is fully tied; raw score is distinct."""
    return pd.DataFrame({
        "hadm_id": [1, 2, 3],
        "subject_id": [1, 2, 3],
        "stage1_score": [0.5, 0.5, 0.5],       # tied after isotonic calibration
        "stage1_score_raw": [0.2, 0.5, 0.9],   # distinct raw XGBoost probability
        "stage1_threshold": [0.3, 0.3, 0.3],
        "stage2_score": [0.3, 0.4, 0.5],
        "stage2_confirmed": [1, 1, 1],
    })


def test_cohort_scores_returns_raw_stage1_not_calibrated():
    """The Stage 1 side of the cohort arrays must be the raw score column."""
    s1_raw, s2 = _cohort_scores(_results_with_raw())
    assert list(s1_raw) == [0.2, 0.5, 0.9]      # raw, not the tied 0.5s
    assert list(s2) == [0.3, 0.4, 0.5]


def test_lookup_patient_exposes_raw_score():
    """_lookup_patient must surface stage1_score_raw for the discordance call."""
    patient = _lookup_patient(3, _results_with_raw())
    assert patient["stage1_score_raw"] == 0.9
    assert patient["stage1_score"] == 0.5       # calibrated still available for the prompt


def test_r1_follows_raw_order_when_calibrated_is_tied():
    """With the calibrated score tied across the cohort, r1 must still
    separate patients by their raw score (the whole point of the change)."""
    df = _results_with_raw()
    s1_raw, s2 = _cohort_scores(df)
    low = _lookup_patient(1, df)    # raw 0.2
    high = _lookup_patient(3, df)   # raw 0.9
    d_low = compute_discordance(low["stage1_score_raw"], low["stage2_score"], s1_raw, s2)
    d_high = compute_discordance(high["stage1_score_raw"], high["stage2_score"], s1_raw, s2)
    assert d_high["r1"] > d_low["r1"]
    # Ranking on the tied calibrated score instead would give both the same r1.
    tied = df["stage1_score"].to_numpy()
    assert _percentile_rank(0.5, tied) == _percentile_rank(0.5, tied)


def _results_without_raw() -> pd.DataFrame:
    df = _results_with_raw()
    return df.drop(columns=["stage1_score_raw"])


def test_lookup_patient_raises_without_raw_column():
    """An older results file with no stage1_score_raw must raise, not fall
    back to the calibrated score."""
    with pytest.raises(KeyError, match="stage1_score_raw"):
        _lookup_patient(1, _results_without_raw())


def test_cohort_scores_raises_without_raw_column():
    """Same guard on the cohort side."""
    with pytest.raises(KeyError, match="stage1_score_raw"):
        _cohort_scores(_results_without_raw())


def test_prepare_patient_raises_on_empty_note():
    """Fix 2: a whitespace-only note (and not the blind-note control) must
    raise before any SHAP/attention work -- cfg is never touched, so pass
    None."""
    with pytest.raises(ValueError, match="no discharge note"):
        _prepare_patient(
            1, None, results_df=_results_with_raw(), artifact={},
            feature_matrix=pd.DataFrame(), note_text="   ",
        )


def test_prepare_patient_empty_note_allowed_under_suppress_note():
    """The blind-note control (suppress_note=True) intentionally passes no
    note, so the empty-note guard must NOT fire -- it should get past the
    note check and fail later (here, on the empty artifact), never on the
    note itself."""
    with pytest.raises(Exception) as exc_info:  # pylint: disable=broad-except
        _prepare_patient(
            1, None, results_df=_results_with_raw(), artifact={"feature_cols": []},
            feature_matrix=pd.DataFrame(), note_text="", suppress_note=True,
        )
    assert "no discharge note" not in str(exc_info.value)
