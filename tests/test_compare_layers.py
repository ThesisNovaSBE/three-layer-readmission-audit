"""Tests for the RQ1 comparison's join/alignment and paired-diff logic.

Pure logic tests against small synthetic DataFrames — no real model or data
needed. ``compare_layers()`` itself (the full pipeline) is not covered here
since it requires a trained Stage 1 artifact and stage2_results_all.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.compare_layers import _metrics_block, _paired_auroc_diff_ci


@pytest.fixture
def stage1_df():
    """20 test-partition admissions; only 12 have a note (simulated below)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hadm_id": np.arange(1000, 1020),
        "subject_id": np.repeat(np.arange(10), 2),
        "readmission_30d": rng.integers(0, 2, 20),
        "stage1_score": rng.uniform(0.1, 0.9, 20),
    })


@pytest.fixture
def stage2_df(stage1_df):  # pylint: disable=redefined-outer-name
    """Only a subset of hadm_ids have a Stage 2 score — the notes-covered subset."""
    covered = stage1_df["hadm_id"].to_numpy()[:12]
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "hadm_id": covered,
        "stage2_score": rng.uniform(0.1, 0.9, len(covered)),
    })


def test_inner_join_keeps_only_notes_covered_admissions(stage1_df, stage2_df):  # pylint: disable=redefined-outer-name
    """The joined population must be exactly the notes-covered subset."""
    joined = stage1_df.merge(stage2_df, on="hadm_id", how="inner")
    assert len(joined) == 12
    assert set(joined["hadm_id"]) == set(stage2_df["hadm_id"])
    assert set(joined["hadm_id"]).issubset(set(stage1_df["hadm_id"]))


def test_join_preserves_both_scores(stage1_df, stage2_df):  # pylint: disable=redefined-outer-name
    """Every joined row must carry both stage1_score and stage2_score."""
    joined = stage1_df.merge(stage2_df, on="hadm_id", how="inner")
    assert "stage1_score" in joined.columns
    assert "stage2_score" in joined.columns
    assert joined["stage1_score"].notna().all()
    assert joined["stage2_score"].notna().all()


def test_no_note_coverage_gives_empty_join():
    """If Stage 2 covers nothing, the join must be empty, not raise."""
    s1 = pd.DataFrame({"hadm_id": [1, 2, 3], "stage1_score": [0.1, 0.5, 0.9]})
    s2 = pd.DataFrame({"hadm_id": [], "stage2_score": []})
    joined = s1.merge(s2, on="hadm_id", how="inner")
    assert len(joined) == 0


# ── _metrics_block ───────────────────────────────────────────────────────────

def test_metrics_block_keys(binary_scores):
    """_metrics_block must return the expected structure."""
    y, s = binary_scores
    groups = np.arange(len(y))
    block = _metrics_block(y, s, groups, threshold=0.5)
    for key in ("n", "pos_rate", "auroc", "auprc", "operating_point"):
        assert key in block
    assert block["n"] == len(y)


# ── _paired_auroc_diff_ci ─────────────────────────────────────────────────────

def test_paired_diff_zero_for_identical_scores(binary_scores):
    """Comparing a model against itself must give a point estimate of exactly 0."""
    y, s = binary_scores
    groups = np.arange(len(y))
    result = _paired_auroc_diff_ci(y, s, s, groups, n_resamples=20)
    assert result["point_estimate"] == pytest.approx(0.0)


def test_paired_diff_matches_manual_auroc_subtraction(binary_scores):
    """point_estimate must equal auroc(a) - auroc(b) computed independently."""
    from src.model.metrics import auroc as auroc_fn  # pylint: disable=import-outside-toplevel

    y, s = binary_scores
    rng = np.random.default_rng(5)
    s_b = np.clip(s + rng.normal(0, 0.1, len(s)), 0.0, 1.0)
    groups = np.arange(len(y))
    result = _paired_auroc_diff_ci(y, s, s_b, groups, n_resamples=10)
    assert result["point_estimate"] == pytest.approx(auroc_fn(y, s) - auroc_fn(y, s_b))


def test_paired_diff_ci_brackets_point_estimate(binary_scores):
    """The CI should contain the point estimate for a reasonably stable diff."""
    y, s = binary_scores
    rng = np.random.default_rng(7)
    s_b = np.clip(s + rng.normal(0, 0.15, len(s)), 0.0, 1.0)
    groups = np.arange(len(y))
    result = _paired_auroc_diff_ci(y, s, s_b, groups, n_resamples=200, seed=3)
    assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
