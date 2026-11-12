"""Tests for scripts/run_validation_controls.py.

Fix 2: the arm test must treat a non-decision (PREP_FAILED / annotation_failed,
decision_model == NaN) as a MATCH when the reference arm reproduces it, not a
spurious mismatch -- otherwise a prep-failure that both runs reproduce
identically would fail the acceptance gate.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import scripts.run_validation_controls as rvc


def _write_main_run(model_dir) -> pd.DataFrame:
    """A 3-admission main run: one concordant uphold, one PREP_FAILED (no
    decision), one discordant override."""
    real = pd.DataFrame({
        "hadm_id": [1, 2, 3],
        "decision_model": ["uphold", np.nan, "override"],
        "discordance_mode": ["CONCORDANT", "PREP_FAILED", "NOTE_MITIGATES"],
        "annotation_failed": [False, True, False],
    })
    real.to_csv(model_dir / "stage3_batch_results.csv", index=False)
    return real


def _arm_patches(control: pd.DataFrame):
    """Patch everything _run_arm_test needs except the comparison itself."""
    return (
        patch.object(rvc, "_load_results", return_value=pd.DataFrame({"hadm_id": [1, 2, 3]})),
        patch.object(rvc, "_load_artifact", return_value={}),
        patch.object(rvc, "load_feature_matrix", return_value=pd.DataFrame()),
        patch.object(rvc, "_preload_notes", return_value={}),
        patch.object(rvc, "_run_batched_control", return_value=control),
    )


def test_arm_test_passes_when_prep_failed_row_is_reproduced(tmp_path, capsys):
    """A PREP_FAILED row (decision NaN) in the first n passes the arm test when
    the reference arm reproduces the same non-decision (NaN == NaN)."""
    real = _write_main_run(tmp_path)
    control = real[["hadm_id", "decision_model"]].copy()  # reference reproduces all 3, incl NaN

    p1, p2, p3, p4, p5 = _arm_patches(control)
    with p1, p2, p3, p4, p5:
        rvc._run_arm_test(3, cfg=None, model_dir=tmp_path)  # must NOT raise SystemExit

    out = capsys.readouterr().out
    assert "3/3 decisions match" in out
    assert "PASS" in out
    assert "1 are PREP_FAILED" in out


def test_arm_test_still_fails_on_a_real_decision_mismatch(tmp_path):
    """The NaN-as-match rule must not mask a genuine decision mismatch."""
    real = _write_main_run(tmp_path)
    control = real[["hadm_id", "decision_model"]].copy()
    control.loc[control["hadm_id"] == 1, "decision_model"] = "override"  # flip a real decision

    p1, p2, p3, p4, p5 = _arm_patches(control)
    with p1, p2, p3, p4, p5, pytest.raises(SystemExit):
        rvc._run_arm_test(3, cfg=None, model_dir=tmp_path)
