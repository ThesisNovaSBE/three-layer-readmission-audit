"""Tests for the readmission label computation (all-cause vs. unplanned)."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.features import compute_readmission_label
from src.schemas import TARGET_COL, TARGET_COL_UNPLANNED


@pytest.fixture
def cfg():
    """Minimal config stub with just the fields compute_readmission_label reads."""
    return SimpleNamespace(cohort=SimpleNamespace(readmission_window_days=30))


def _admissions(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal admissions DataFrame for one patient's admission history."""
    defaults = {"subject_id": 1, "hospital_expire_flag": 0, "admission_type": "EW EMER."}
    df = pd.DataFrame([{**defaults, **r} for r in rows])
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    return df


def test_readmission_within_window_flagged(cfg):  # pylint: disable=redefined-outer-name
    """A return within the window, unplanned, must be flagged in both labels."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05"},
        {"hadm_id": 2, "admittime": "2026-01-15", "dischtime": "2026-01-20",
         "admission_type": "EW EMER."},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 1
    assert first[TARGET_COL_UNPLANNED] == 1


def test_readmission_outside_window_not_flagged(cfg):  # pylint: disable=redefined-outer-name
    """A return after the window must not be flagged in either label."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05"},
        {"hadm_id": 2, "admittime": "2026-03-01", "dischtime": "2026-03-05"},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 0
    assert first[TARGET_COL_UNPLANNED] == 0


def test_elective_outcome_counts_all_cause_not_unplanned(cfg):  # pylint: disable=redefined-outer-name
    """P7b: a return within window whose admission_type is ELECTIVE must count
    toward the all-cause label but be excluded from the unplanned label."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05"},
        {"hadm_id": 2, "admittime": "2026-01-15", "dischtime": "2026-01-20",
         "admission_type": "ELECTIVE"},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 1
    assert first[TARGET_COL_UNPLANNED] == 0


def test_same_day_surgical_outcome_excluded_from_unplanned(cfg):  # pylint: disable=redefined-outer-name
    """SURGICAL SAME DAY ADMISSION as the outcome type is also a planned return."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05"},
        {"hadm_id": 2, "admittime": "2026-01-10", "dischtime": "2026-01-11",
         "admission_type": "SURGICAL SAME DAY ADMISSION"},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 1
    assert first[TARGET_COL_UNPLANNED] == 0


def test_death_excludes_both_labels(cfg):  # pylint: disable=redefined-outer-name
    """In-hospital death must exclude both labels, even with a nominal next admission."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05",
         "hospital_expire_flag": 1},
        {"hadm_id": 2, "admittime": "2026-01-15", "dischtime": "2026-01-20"},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 0
    assert first[TARGET_COL_UNPLANNED] == 0


def test_last_admission_no_next_not_flagged(cfg):  # pylint: disable=redefined-outer-name
    """A patient's final admission (no subsequent visit) must not be flagged."""
    adm = _admissions([
        {"hadm_id": 1, "admittime": "2026-01-01", "dischtime": "2026-01-05"},
    ])
    result = compute_readmission_label(adm, cfg)
    first = result[result["hadm_id"] == 1].iloc[0]
    assert first[TARGET_COL] == 0
    assert first[TARGET_COL_UNPLANNED] == 0
