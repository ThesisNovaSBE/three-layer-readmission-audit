"""Tests for Stage 2 checkpoint-resume logic.

Only covers pure path/string/JSON logic (no torch/transformers execution) --
requires the torch/transformers extras to even import src.stage2.train.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.stage2.train import _checkpoint_step, _find_resume_checkpoint

_FP = {"n_train": 100, "target_col": "readmission_30d_unplanned",
       "max_seq_length": 4096, "model_name": "models/clinical_longformer"}


def test_checkpoint_step_parses_the_step_number():
    """_checkpoint_step must extract the integer step from the dir name."""
    assert _checkpoint_step(Path("checkpoint-1500")) == 1500
    assert _checkpoint_step(Path("/some/dir/checkpoint-50")) == 50


def test_find_resume_checkpoint_picks_highest_step_not_lexicographic_max(tmp_path):
    """The true latest checkpoint (by step count) must be picked, not the
    lexicographically-largest directory name.

    Regression test: "checkpoint-500" sorts AFTER "checkpoint-1500" as a
    plain string (since '5' > '1'), which previously caused a real, silent
    bug -- resuming from an older checkpoint than the actual latest one.
    """
    for step in (500, 1000, 1500):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "RUN_FINGERPRINT.json").write_text(json.dumps(_FP))

    resumed = _find_resume_checkpoint(tmp_path, _FP)

    assert resumed is not None
    assert resumed.endswith("checkpoint-1500")


def test_find_resume_checkpoint_returns_none_when_dir_missing(tmp_path):
    """A checkpoint dir that doesn't exist yet must return None, not raise."""
    assert _find_resume_checkpoint(tmp_path / "does_not_exist", _FP) is None


def test_find_resume_checkpoint_returns_none_when_dir_empty(tmp_path):
    """An existing but empty checkpoint dir must return None."""
    empty = tmp_path / "stage2_checkpoints"
    empty.mkdir()
    assert _find_resume_checkpoint(empty, _FP) is None


def test_find_resume_checkpoint_writes_fingerprint_on_fresh_start(tmp_path):
    """Starting fresh (no existing checkpoints) must record the fingerprint
    so a later resume in the SAME run can be validated against it."""
    fresh = tmp_path / "stage2_checkpoints"
    assert _find_resume_checkpoint(fresh, _FP) is None
    assert json.loads((fresh / "RUN_FINGERPRINT.json").read_text()) == _FP


def test_find_resume_checkpoint_rejects_mismatched_fingerprint(tmp_path):
    """Checkpoints left over from a DIFFERENT run (different label column,
    training-sample size, or model) must never be silently resumed from --
    this is the actual mechanism that would have caught a stale rsync'd
    checkpoint directory from an earlier, unrelated run on the cluster.
    """
    stale = tmp_path / "stage2_checkpoints"
    stale.mkdir()
    (stale / "checkpoint-2000").mkdir()
    (stale / "RUN_FINGERPRINT.json").write_text(
        json.dumps({**_FP, "target_col": "readmission_30d"})  # old, all-cause run
    )

    resumed = _find_resume_checkpoint(stale, _FP)

    assert resumed is None
    assert not (stale / "checkpoint-2000").exists()  # not left where it could be resumed from
    moved = list(tmp_path.glob("stage2_checkpoints_stale_*"))
    assert len(moved) == 1
    assert (moved[0] / "checkpoint-2000").exists()  # old checkpoint preserved, not lost


def test_find_resume_checkpoint_treats_missing_fingerprint_as_stale(tmp_path):
    """Checkpoints with no fingerprint file at all (e.g. from before this
    check existed) must not be trusted either."""
    old = tmp_path / "stage2_checkpoints"
    old.mkdir()
    (old / "checkpoint-500").mkdir()

    resumed = _find_resume_checkpoint(old, _FP)

    assert resumed is None
    assert not (old / "checkpoint-500").exists()  # not left where it could be resumed from
    assert list(tmp_path.glob("stage2_checkpoints_stale_*"))  # old content preserved aside
