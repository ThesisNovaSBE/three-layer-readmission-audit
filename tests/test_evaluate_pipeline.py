"""Tests for the Stage 3 decision overlay and control arm in evaluate_pipeline.py.

Pure-logic tests against small synthetic DataFrames / arrays. The rest of
evaluate_pipeline.py needs a trained Stage 1 artifact and real feature
matrix, so isn't covered here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model.evaluate_pipeline import _apply_stage3_decisions, _control_arm_report


def test_no_stage3_file_returns_unchanged_prediction(tmp_path):
    """Without a batch Stage 3 result, the C9-corrected prediction must pass through untouched."""
    hadm_test = np.array([1, 2, 3, 4])
    pipeline_pred_full = np.array([1, 0, 1, 1])
    flagged = np.array([True, False, True, True])

    pred, info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert np.array_equal(pred, pipeline_pred_full)
    assert info["available"] is False


def test_stage3_decision_overrides_stage2_confirmed(tmp_path):
    """Where Stage 3 has a decision, it must win over the prior C9-corrected prediction."""
    (tmp_path / "stage3_batch_results.csv").write_text(
        "hadm_id,decision\n1,override\n2,uphold\n"
    )
    hadm_test = np.array([1, 2, 3])
    # Prior prediction (stage2_confirmed / C9): all positive.
    pipeline_pred_full = np.array([1, 1, 1])
    flagged = np.array([True, True, True])

    pred, info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert pred[0] == 0  # Stage 3 overrode admission 1's alert
    assert pred[1] == 1  # Stage 3 upheld admission 2's alert
    assert pred[2] == 1  # no Stage 3 result for admission 3 -- prior prediction kept
    assert info["available"] is True
    assert info["n_test_with_stage3_decision"] == 2


def test_admissions_without_stage3_result_keep_prior_prediction(tmp_path):
    """Admissions not covered by Stage 3 (no note, or annotation_failed) must be untouched."""
    pd.DataFrame({"hadm_id": [1], "decision": ["override"]}).to_csv(
        tmp_path / "stage3_batch_results.csv", index=False
    )
    hadm_test = np.array([1, 2])
    pipeline_pred_full = np.array([1, 1])  # admission 2's prior C9 fallback
    flagged = np.array([True, True])

    pred, _info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert pred[1] == 1  # unchanged -- Stage 3 never audited this admission


def test_annotation_failed_rows_do_not_override(tmp_path):
    """A row with decision=None (annotation_failed) must not count as a Stage 3 decision."""
    pd.DataFrame({"hadm_id": [1], "decision": [None]}).to_csv(
        tmp_path / "stage3_batch_results.csv", index=False
    )
    hadm_test = np.array([1])
    pipeline_pred_full = np.array([1])
    flagged = np.array([True])

    pred, info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert pred[0] == 1  # unchanged -- no usable Stage 3 decision
    assert info["n_test_with_stage3_decision"] == 0


def test_never_flagged_admissions_stay_negative(tmp_path):
    """Admissions Stage 1 never flagged must be 0 regardless of any Stage 3 row."""
    pd.DataFrame({"hadm_id": [1], "decision": ["uphold"]}).to_csv(
        tmp_path / "stage3_batch_results.csv", index=False
    )
    hadm_test = np.array([1])
    pipeline_pred_full = np.array([1])
    flagged = np.array([False])  # Stage 1 never flagged this admission

    pred, _info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert pred[0] == 0


def test_coverage_fraction_computed_correctly(tmp_path):
    """coverage_of_flagged must be n_with_stage3 / n_flagged."""
    pd.DataFrame({"hadm_id": [1, 2], "decision": ["uphold", "override"]}).to_csv(
        tmp_path / "stage3_batch_results.csv", index=False
    )
    hadm_test = np.array([1, 2, 3, 4])
    pipeline_pred_full = np.array([1, 1, 1, 0])
    flagged = np.array([True, True, True, False])  # 3 flagged, 2 with a Stage 3 decision

    _pred, info = _apply_stage3_decisions(tmp_path, hadm_test, pipeline_pred_full, flagged)

    assert info["coverage_of_flagged"] == 2 / 3


# ── _control_arm_report ───────────────────────────────────────────────────────

@pytest.fixture
def subgroups_20():  # pylint: disable=missing-function-docstring
    return pd.DataFrame({"age_band": ["18-40", "41-55", "56-70", "70+"] * 5})


def test_control_arm_matches_target_alert_rate(subgroups_20):  # pylint: disable=redefined-outer-name
    """The control arm's alert rate must match the target within quantile precision."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 20)
    s1_scores = rng.uniform(0.0, 1.0, 20)
    target_rate = 0.25

    report = _control_arm_report(y, s1_scores, target_rate, subgroups_20)

    assert report["confirmed_rate"] == pytest.approx(target_rate, abs=0.06)
    assert report["target_alert_rate"] == target_rate


def test_control_arm_is_pipeline_report_shaped(subgroups_20):  # pylint: disable=redefined-outer-name
    """Output must have the same shape as _pipeline_report, plus threshold fields."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20)
    s1_scores = rng.uniform(0.0, 1.0, 20)

    report = _control_arm_report(y, s1_scores, 0.5, subgroups_20)

    for key in ("n", "pos_rate", "confirmed_rate", "precision", "recall",
                "f1", "f2", "by_age_band", "threshold", "target_alert_rate"):
        assert key in report


def test_control_arm_full_capacity_flags_everyone(subgroups_20):  # pylint: disable=redefined-outer-name
    """target_alert_rate=1.0 must flag the entire population."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 20)
    s1_scores = rng.uniform(0.0, 1.0, 20)

    report = _control_arm_report(y, s1_scores, 1.0, subgroups_20)

    assert report["confirmed_rate"] == pytest.approx(1.0)
