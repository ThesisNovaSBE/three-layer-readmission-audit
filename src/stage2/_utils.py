"""Shared utilities for Stage 2 modules.

Centralises the age-band key mapping and the Stage 2 model path helper so they
are not duplicated across calibrate.py, evaluate.py, predict.py and train.py.
"""

from __future__ import annotations

from pathlib import Path


# Maps pandas Interval string representations to config key strings.
_BAND_MAP: dict[str, str] = {
    "(17, 40]":  "18-40",
    "(40, 55]":  "41-55",
    "(55, 70]":  "56-70",
    "(70, 120]": "70+",
}


def band_key(age_band: object) -> str:
    """Convert a pandas Interval age_band to its config key string.

    Args:
        age_band: a pandas ``Interval`` or any object whose ``str()``
                  representation matches one of the MIMIC-IV age bins.

    Returns:
        A string key such as ``"18-40"`` or ``"70+"``.
    """
    return _BAND_MAP.get(str(age_band), str(age_band))


def get_stage2_model_path(model_dir: Path) -> Path:
    """Return the path to the fine-tuned Stage 2 model directory.

    Args:
        model_dir: project model directory (from ``get_model_dir()``).

    Returns:
        ``Path`` to the best model checkpoint directory.

    Raises:
        FileNotFoundError: if the directory does not exist.
    """
    path = model_dir / "stage2_longformer_best"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at {path}. "
            "Run `python -m src.stage2.train` first."
        )
    return path
