"""Tests for src/stage2/predict.py guardrails.

Fix 3: _score_core must refuse to score Stage 2 when stage2_calibration.json
is missing, rather than silently falling back to a single global threshold
(which would ship miscalibrated probabilities and a wrong per-band operating
point to Stage 3).
"""

from __future__ import annotations

import pytest

import src.stage2.predict as predict_mod
from src.config import load_config


def test_score_core_raises_without_calibration(tmp_path, monkeypatch):
    """A missing stage2_calibration.json must raise FileNotFoundError before
    any scoring, not fall back to the global threshold."""
    monkeypatch.setattr(predict_mod, "get_model_dir", lambda: tmp_path)
    monkeypatch.setattr(predict_mod, "get_stage2_model_path", lambda _md: tmp_path / "model")
    cfg = load_config()
    with pytest.raises(FileNotFoundError, match="calibrate"):
        predict_mod._score_core(
            cfg, flagged_only=True, stage2_threshold=None,
            artifact={"mode": "full"}, limit=None,
        )
