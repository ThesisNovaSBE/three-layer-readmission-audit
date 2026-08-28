"""Tests for Stage 2 checkpoint-resume logic.

Only covers pure path/string logic (no torch/transformers execution) --
requires the torch/transformers extras to even import src.stage2.train.
"""

from __future__ import annotations

from pathlib import Path

from src.stage2.train import _checkpoint_step, _find_resume_checkpoint


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

    resumed = _find_resume_checkpoint(tmp_path)

    assert resumed is not None
    assert resumed.endswith("checkpoint-1500")


def test_find_resume_checkpoint_returns_none_when_dir_missing(tmp_path):
    """A checkpoint dir that doesn't exist yet must return None, not raise."""
    assert _find_resume_checkpoint(tmp_path / "does_not_exist") is None


def test_find_resume_checkpoint_returns_none_when_dir_empty(tmp_path):
    """An existing but empty checkpoint dir must return None."""
    empty = tmp_path / "stage2_checkpoints"
    empty.mkdir()
    assert _find_resume_checkpoint(empty) is None
