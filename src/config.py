"""Load config.yaml and .env, resolve data paths, detect real vs. synthetic mode."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.config_schema import AppConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate the project configuration.

    Environment variables (from ``.env`` or the shell) take precedence over
    ``config.yaml`` values for MIMIC-IV data paths.

    Args:
        config_path: path to the YAML file. Defaults to
                     ``<project_root>/config.yaml``.

    Returns:
        Validated :class:`~src.config_schema.AppConfig` instance.
    """
    load_dotenv(_PROJECT_ROOT / ".env")

    path = config_path or _PROJECT_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = AppConfig.model_validate(raw)

    if os.getenv("MIMIC_IV_DIR"):
        cfg.data.mimic_iv_dir = os.getenv("MIMIC_IV_DIR", "")
    if os.getenv("MIMIC_IV_NOTE_DIR"):
        cfg.data.mimic_iv_note_dir = os.getenv("MIMIC_IV_NOTE_DIR", "")

    return cfg


def has_real_data(cfg: AppConfig) -> bool:
    """Return ``True`` if the MIMIC-IV directory is configured and exists."""
    return bool(cfg.data.mimic_iv_dir) and Path(cfg.data.mimic_iv_dir).exists()


def get_data_dir() -> Path:
    """Return the absolute path to the project ``data/`` directory."""
    return _PROJECT_ROOT / "data"


def get_model_dir() -> Path:
    """Return the absolute path to the project ``models/`` directory."""
    return _PROJECT_ROOT / "models"
