"""Tests for aggregate_discordance with mock data (no disk I/O)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.stage3.categorize import aggregate_discordance
from src.stage3.explain import DISCORDANCE_CATEGORIES, DISCORDANCE_MODES


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(
    modes: list[str],
    cats: list[str],
    failed: list[bool] | None = None,
    confirmed: list[int] | None = None,
    age_bands: list[str] | None = None,
) -> pd.DataFrame:
    n = len(modes)
    data: dict = {
        "hadm_id": list(range(1000, 1000 + n)),
        "discordance_mode": modes,
        "primary_category": cats,
        "annotation_failed": failed if failed is not None else [False] * n,
        "stage2_confirmed": confirmed if confirmed is not None else [1] * n,
    }
    if age_bands is not None:
        data["age_band"] = age_bands
    return pd.DataFrame(data)


# ── Basic structure ───────────────────────────────────────────────────────────

def test_aggregate_keys_present(mock_stage2_results):
    result = aggregate_discordance(mock_stage2_results)
    for key in (
        "n_patients", "n_valid_annotations", "n_annotation_failures",
        "annotation_failure_rate_pct", "mode_distribution",
        "category_distribution", "category_by_discordance_mode",
        "note_mitigates", "note_amplifies",
    ):
        assert key in result, f"missing key: {key}"


def test_empty_df_returns_n_zero():
    result = aggregate_discordance(pd.DataFrame())
    assert result["n_patients"] == 0


def test_n_patients_correct(mock_stage2_results):
    result = aggregate_discordance(mock_stage2_results)
    assert result["n_patients"] == len(mock_stage2_results)


# ── Failure exclusion ─────────────────────────────────────────────────────────

def test_annotation_failures_excluded_from_mode_distribution():
    # 4 patients: 3 valid CONCORDANT, 1 failed (annotation_failed=True)
    df = _make_df(
        modes=["CONCORDANT", "CONCORDANT", "CONCORDANT", "NOTE_AMPLIFIES"],
        cats=["structured_confirmed"] * 3 + ["social_support"],
        failed=[False, False, False, True],
    )
    result = aggregate_discordance(df)
    assert result["n_valid_annotations"] == 3
    assert result["n_annotation_failures"] == 1
    # NOTE_AMPLIFIES must not appear in valid-annotation counts
    assert result["mode_distribution"]["counts"]["NOTE_AMPLIFIES"] == 0
    assert result["mode_distribution"]["counts"]["CONCORDANT"] == 3


def test_all_failures_gives_zero_valid():
    df = _make_df(
        modes=["CONCORDANT"] * 5,
        cats=["social_support"] * 5,
        failed=[True] * 5,
    )
    result = aggregate_discordance(df)
    assert result["n_valid_annotations"] == 0
    assert result["n_annotation_failures"] == 5
    assert result["annotation_failure_rate_pct"] == 100.0


def test_failure_rate_pct_calculation():
    df = _make_df(
        modes=["CONCORDANT"] * 10,
        cats=["social_support"] * 10,
        failed=[True] * 2 + [False] * 8,
    )
    result = aggregate_discordance(df)
    assert result["annotation_failure_rate_pct"] == pytest.approx(20.0)


# ── Mode / category distributions ────────────────────────────────────────────

def test_mode_distribution_all_modes_present():
    result = aggregate_discordance(_make_df(
        modes=["CONCORDANT", "NOTE_MITIGATES", "NOTE_AMPLIFIES"],
        cats=["structured_confirmed", "social_support", "frailty_markers"],
    ))
    counts = result["mode_distribution"]["counts"]
    for mode in DISCORDANCE_MODES:
        assert mode in counts


def test_mode_distribution_percentages_sum_to_100(mock_stage2_results):
    # Use only valid rows
    df = mock_stage2_results[~mock_stage2_results["annotation_failed"]].copy()
    if len(df) == 0:
        pytest.skip("all rows failed annotation in fixture")
    result = aggregate_discordance(df)
    total_pct = sum(result["mode_distribution"]["percentages"].values())
    assert total_pct == pytest.approx(100.0, abs=0.5)


def test_category_distribution_all_categories_present():
    df = _make_df(
        modes=["CONCORDANT"] * 3,
        cats=["social_support", "discharge_planning", "structured_confirmed"],
    )
    result = aggregate_discordance(df)
    cat_counts = result["category_distribution"]["counts"]
    for cat in DISCORDANCE_CATEGORIES:
        assert cat in cat_counts


# ── Confirmed / rejected breakdown ────────────────────────────────────────────

def test_confirmed_rejected_split():
    df = _make_df(
        modes=["CONCORDANT"] * 6,
        cats=["social_support"] * 6,
        confirmed=[1, 1, 1, 1, 0, 0],
    )
    result = aggregate_discordance(df)
    assert result["n_confirmed"] == 4
    assert result["n_rejected"] == 2


# ── Note-level breakdowns ─────────────────────────────────────────────────────

def test_note_mitigates_n_counted_correctly():
    df = _make_df(
        modes=["NOTE_MITIGATES", "NOTE_MITIGATES", "CONCORDANT"],
        cats=["social_support", "discharge_planning", "structured_confirmed"],
    )
    result = aggregate_discordance(df)
    assert result["note_mitigates"]["n"] == 2


def test_note_amplifies_n_counted_correctly():
    df = _make_df(
        modes=["NOTE_AMPLIFIES", "CONCORDANT", "CONCORDANT"],
        cats=["frailty_markers", "social_support", "structured_confirmed"],
    )
    result = aggregate_discordance(df)
    assert result["note_amplifies"]["n"] == 1


# ── Age-group breakdown ───────────────────────────────────────────────────────

def test_age_group_breakdown_present_when_column_exists():
    df = _make_df(
        modes=["CONCORDANT", "NOTE_MITIGATES"],
        cats=["social_support", "discharge_planning"],
        age_bands=["18-40", "70+"],
    )
    result = aggregate_discordance(df)
    assert "category_by_age_group" in result
    assert "18-40" in result["category_by_age_group"]
    assert "70+" in result["category_by_age_group"]


def test_age_group_breakdown_empty_when_column_missing():
    df = _make_df(
        modes=["CONCORDANT"],
        cats=["social_support"],
    )
    result = aggregate_discordance(df)
    assert result["category_by_age_group"] == {}
