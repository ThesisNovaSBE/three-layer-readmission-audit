"""Tests for configuration loading and Pydantic schema validation."""

from __future__ import annotations

import pytest

from src.config import load_config
from src.config_schema import AppConfig, Stage2Config, Stage3Config


def test_load_config_returns_appconfig():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)


def test_default_stage1_model():
    cfg = load_config()
    assert cfg.stage1.model == "xgboost"


def test_stage2_fields_present():
    cfg = load_config()
    assert cfg.stage2.max_seq_length == 2048
    assert cfg.stage2.focal_loss is True
    assert cfg.stage2.batch_size > 0
    assert 0 < cfg.stage2.val_fraction < 1
    assert 0 < cfg.stage2.calibration_fraction < 1
    assert cfg.stage2.val_fraction + cfg.stage2.calibration_fraction < 1


def test_stage3_new_fields_present():
    cfg = load_config()
    assert hasattr(cfg.stage3, "discordance_analysis")
    assert hasattr(cfg.stage3, "attention_extraction")
    assert hasattr(cfg.stage3, "top_attention_sentences")
    assert hasattr(cfg.stage3, "top_shap_features")
    assert cfg.stage3.top_attention_sentences > 0
    assert cfg.stage3.top_shap_features > 0


def test_pydantic_dot_access():
    """AppConfig fields must be accessible via dot notation, not dict subscript."""
    cfg = load_config()
    # These should not raise
    _ = cfg.stage1.model
    _ = cfg.stage2.model_name
    _ = cfg.stage3.ollama_model
    _ = cfg.run.mode
    _ = cfg.data.mimic_iv_dir


def test_pydantic_rejects_dict_access():
    """AppConfig must NOT be subscriptable (Pydantic v2 behaviour)."""
    cfg = load_config()
    with pytest.raises(TypeError):
        _ = cfg["stage1"]  # type: ignore[index]


def test_stage3_config_defaults():
    s3 = Stage3Config()
    assert s3.enabled is False
    assert s3.discordance_analysis is True
    assert s3.attention_extraction is True


def test_age_group_targets_sum():
    """Total age-group training targets should be around 249k."""
    cfg = load_config()
    total = sum(cfg.stage2.age_group_train_targets.values())
    assert total > 100_000, f"Training targets too low: {total}"


def test_run_mode_switching():
    cfg = load_config()
    cfg.run.mode = "full"
    assert cfg.run.active().n_rows == 0
    cfg.run.mode = "quick"
    assert cfg.run.active().n_rows > 0
