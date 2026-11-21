"""Tests for scripts/compare_pipeline_control_paired.py -- the paired
bootstrap of pipeline-minus-control differences. Synthetic data only; the real
reconstruction (reconstruct_policies) needs the cluster data and is exercised
at runtime by the cross-check against pipeline_evaluation.json."""

from __future__ import annotations

import numpy as np

from scripts.compare_pipeline_control_paired import (
    _Policies,
    _paired_diff_ci,
    _top_m_by_raw,
    compute_paired_differences,
)
from src.model.evaluate_pipeline import _precision_metric, _recall_metric


def _clear_gap_arrays():
    """100 one-admission-per-patient cases where arm A is strictly better than
    arm B on both precision and recall."""
    y = np.array([1] * 60 + [0] * 40)
    pred_a = y.copy()  # perfect: precision 1.0, recall 1.0
    pred_b = np.zeros(100, dtype=int)
    pred_b[:30] = 1   # 30 true positives
    pred_b[60:80] = 1  # 20 false positives -> precision 0.6, recall 0.5
    groups = np.arange(100)  # every admission its own patient
    return y, pred_a, pred_b, groups


def test_paired_diff_positive_and_excludes_zero_when_a_better():
    y, pred_a, pred_b, groups = _clear_gap_arrays()
    prec = _paired_diff_ci(y, pred_a, pred_b, groups, _precision_metric)
    rec = _paired_diff_ci(y, pred_a, pred_b, groups, _recall_metric)
    assert prec["point_estimate"] > 0.3 and prec["excludes_zero"] is True
    assert rec["point_estimate"] > 0.3 and rec["excludes_zero"] is True
    assert prec["ci_lower"] > 0.0 and rec["ci_lower"] > 0.0


def test_paired_diff_zero_when_arms_identical():
    y, pred_a, _pred_b, groups = _clear_gap_arrays()
    d = _paired_diff_ci(y, pred_a, pred_a, groups, _precision_metric)
    assert d["point_estimate"] == 0.0
    assert d["ci_lower"] == 0.0 and d["ci_upper"] == 0.0
    assert d["excludes_zero"] is False  # a CI of [0,0] contains 0


def test_paired_diff_is_deterministic_under_seed():
    y, pred_a, pred_b, groups = _clear_gap_arrays()
    a = _paired_diff_ci(y, pred_a, pred_b, groups, _recall_metric, seed=42)
    b = _paired_diff_ci(y, pred_a, pred_b, groups, _recall_metric, seed=42)
    assert a == b
    assert a["n_resamples_used"] == 1000


def test_top_m_by_raw_flags_highest_scores_at_target_count():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    pred = _top_m_by_raw(scores, target_rate=0.4)  # m = round(0.4*5) = 2
    assert pred.sum() == 2
    assert list(pred) == [0, 1, 0, 1, 0]  # the 0.9 and 0.7 admissions


def test_top_m_by_raw_zero_rate_flags_nothing():
    assert _top_m_by_raw(np.array([0.5, 0.6, 0.7]), target_rate=0.0).sum() == 0


def test_compute_paired_differences_structure_and_direction():
    y, pred_a, pred_b, groups = _clear_gap_arrays()
    pol = _Policies(
        y_true=y, groups=groups, s1_scores_raw=np.linspace(0, 1, 100),
        pipeline=pred_a, control=pred_b,
        secondary=pred_a, secondary_control=pred_b,
        conditional=pred_a, conditional_control=pred_b,
    )
    report = compute_paired_differences(pol)
    assert set(report["comparisons"]) == {
        "primary_full_vs_control",
        "secondary_both_override_vs_control",
        "conditional_discordant_only_vs_control",
    }
    assert report["n_resamples"] == 1000 and report["seed"] == 42
    prim = report["comparisons"]["primary_full_vs_control"]
    assert prim["precision_diff"]["point_estimate"] > 0.3
    assert prim["recall_diff"]["excludes_zero"] is True
    assert prim["control_alert_count"] == int(pred_b.sum())
