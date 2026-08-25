"""Shared utilities for Stage 2 modules.

Centralises the age-band key mapping, the Stage 2 model path helper, and the
structured feature column definitions so they are not duplicated across
calibrate.py, evaluate.py, predict.py, train.py, dataset.py, and model.py.
"""

from __future__ import annotations

from pathlib import Path

# Structured feature columns fed to the MLP branch of FusionLongformer.
# stage1_score is deliberately excluded: FusionLongformer must produce an
# estimate independent of Stage 1 so that discordance between the two scores
# reflects the note's marginal contribution, not feature-set overlap.
# has_discharge_note is excluded because it is always 1 for Stage 2 patients
# (selection bias — no note = no Stage 2 inference).
STRUCT_FEATURE_COLS: list[str] = [
    "age",
    "los_days",
    "charlson_index",
    "n_prior_admissions",
    "days_since_last_discharge",
    "is_medicaid_medicare",
    "admission_type_emergency",
    "admission_type_observation",
]

# Subset of STRUCT_FEATURE_COLS that need z-score normalisation.
# Binary features (0/1) are left as-is.
CONTINUOUS_STRUCT_COLS: frozenset[str] = frozenset({
    "age",
    "los_days",
    "charlson_index",
    "n_prior_admissions",
    "days_since_last_discharge",
})


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
