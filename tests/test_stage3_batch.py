"""Tests for the batch Stage 3 audit runner's control flow.

Mocks explain_patient and the loaders so these test resumability, per-row
error isolation, and CSV output shape — not real model inference.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.stage3.batch import _OUTPUT_FIELDS, run_batch_audit, run_sensitivity_sweep


def _fake_result(hadm_id: int, *, fail: bool = False) -> SimpleNamespace:
    """Build a stand-in for ExplanationResult.model_dump()'s output."""
    return SimpleNamespace(model_dump=lambda: {
        "hadm_id": hadm_id, "stage1_score": 0.6, "stage1_threshold": 0.35,
        "stage2_score": 0.4, "stage2_confirmed": True, "r1": 70.0, "r2": 40.0,
        "displacement": -30.0, "discordance_mode": "NOTE_MITIGATES",
        "decision": None if fail else "override",
        "primary_clinical_domain": None if fail else "social_support",
        "supporting_quote": "" if fail else "Patient has strong family support at home.",
        "quote_verified": None if fail else True,
        "planned_return": None if fail else "no",
        "clinical_justification": "" if fail else "Note documents strong support.",
        "annotation_failed": fail,
    })


@pytest.fixture
def results_df():  # pylint: disable=missing-function-docstring
    return pd.DataFrame({
        "hadm_id": [10, 11, 12, 13],
        "subject_id": [1, 2, 3, 4],
        "readmission_30d": [0, 1, 0, 1],
        "stage1_score": [0.5, 0.6, 0.4, 0.7],
        "stage1_threshold": [0.35] * 4,
        "stage2_score": [0.3, 0.5, 0.2, 0.6],
        "stage2_confirmed": [0, 1, 0, 1],
    })


def _patched(results_df, explain_side_effect):  # pylint: disable=redefined-outer-name
    """Context manager patching batch.py's loaders and explain_patient."""
    return (
        patch("src.stage3.batch._load_artifact", return_value={}),
        patch("src.stage3.batch._load_results", return_value=results_df),
        patch("src.stage3.batch.load_feature_matrix", return_value=pd.DataFrame()),
        patch("src.stage3.batch.explain_patient", side_effect=explain_side_effect),
    )


def test_writes_one_row_per_admission(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A clean run must write exactly one row per target admission."""
    out = tmp_path / "out.csv"

    def side_effect(hadm_id, *_a, **_kw):
        return _fake_result(hadm_id)

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    assert set(written["hadm_id"]) == {10, 11, 12, 13}
    assert list(written.columns) == _OUTPUT_FIELDS


def test_one_failure_does_not_stop_the_batch(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A raised exception on one admission must not prevent the rest from being written."""
    out = tmp_path / "out.csv"

    def side_effect(hadm_id, *_a, **_kw):
        if hadm_id == 11:
            raise RuntimeError("simulated Ollama timeout")
        return _fake_result(hadm_id)

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 3
    assert 11 not in set(written["hadm_id"])


def test_resume_skips_already_written_admissions(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """--resume must not re-call explain_patient for hadm_ids already in the CSV."""
    out = tmp_path / "out.csv"
    calls: list[int] = []

    def side_effect(hadm_id, *_a, **_kw):
        calls.append(hadm_id)
        return _fake_result(hadm_id)

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out)

    assert sorted(calls) == [10, 11, 12, 13]
    calls.clear()

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out, resume=True)

    assert not calls
    written = pd.read_csv(out)
    assert len(written) == 4


def test_limit_restricts_target_count(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """--limit must restrict how many admissions are audited."""
    out = tmp_path / "out.csv"

    def side_effect(hadm_id, *_a, **_kw):
        return _fake_result(hadm_id)

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out, limit=2)

    written = pd.read_csv(out)
    assert len(written) == 2


def test_annotation_failed_rows_still_written(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A parse failure (annotation_failed=True) must still produce a row, not be dropped."""
    out = tmp_path / "out.csv"

    def side_effect(hadm_id, *_a, **_kw):
        return _fake_result(hadm_id, fail=hadm_id == 12)

    p1, p2, p3, p4 = _patched(results_df, side_effect)
    with p1, p2, p3, p4:
        run_batch_audit(cfg=SimpleNamespace(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    row = written[written["hadm_id"] == 12].iloc[0]
    assert bool(row["annotation_failed"]) is True


# ── run_sensitivity_sweep ─────────────────────────────────────────────────────

def test_sensitivity_sweep_raises_without_batch_results(tmp_path):
    """Must fail clearly if run before the batch audit has ever produced output."""
    with pytest.raises(FileNotFoundError):
        run_sensitivity_sweep(model_dir=tmp_path)


def test_sensitivity_sweep_writes_expected_json(tmp_path):
    """Must read displacement from an existing batch CSV and write the sweep JSON."""
    batch_csv = tmp_path / "stage3_batch_results.csv"
    pd.DataFrame({
        "hadm_id": [1, 2, 3, 4],
        "displacement": [-40.0, -5.0, 10.0, 35.0],
    }).to_csv(batch_csv, index=False)

    sweep = run_sensitivity_sweep(model_dir=tmp_path)

    out_path = tmp_path / "discordance_sensitivity.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["n_patients"] == 4
    assert saved["sweep"] == sweep
