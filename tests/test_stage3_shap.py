"""Tests for SHAP feature extraction (no real XGBoost model required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.stage3.shap_extract import _label, extract_shap_features, extract_shap_for_patient


# ── Label helper ─────────────────────────────────────────────────────────────

def test_label_known_column():
    """Known column names map to human-readable labels."""
    assert _label("creatinine_last") == "creatinine (last)"


def test_label_unknown_column_replaces_underscores():
    """Unknown columns fall back to underscore-replaced name."""
    assert _label("my_custom_col") == "my custom col"


# ── Mock artifact ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "creatinine_last", "age", "los_days", "came_via_ed", "n_prior_admissions"
]


def _make_artifact(importances: list[float] | None = None) -> dict:
    """Build a minimal mock artifact with a sklearn-like model."""
    if importances is None:
        importances = [0.3, 0.25, 0.2, 0.15, 0.1]
    mock_model = MagicMock()
    mock_model.feature_importances_ = np.array(importances)
    return {"estimator": mock_model, "feature_cols": FEATURE_COLS}


def _make_patients(n: int = 3) -> pd.DataFrame:
    """Create a random patient DataFrame with the test feature columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, (n, len(FEATURE_COLS))),
        columns=FEATURE_COLS,
    )


# ── Basic extraction (gain fallback, no shap library) ────────────────────────

def test_returns_list_per_patient():
    """Result must have one entry per patient."""
    artifact = _make_artifact()
    patients = _make_patients(4)
    results = extract_shap_features(artifact, patients, top_k=3)
    assert len(results) == 4


def test_each_patient_has_top_k_features():
    """Each patient entry must have exactly top_k feature strings."""
    artifact = _make_artifact()
    patients = _make_patients(2)
    results = extract_shap_features(artifact, patients, top_k=3)
    for patient_features in results:
        assert len(patient_features) == 3


def test_top_k_clamped_to_available_features():
    """Requesting more features than available must not crash."""
    artifact = _make_artifact()
    patients = _make_patients(1)
    results = extract_shap_features(artifact, patients, top_k=100)
    assert len(results[0]) <= len(FEATURE_COLS)


def test_feature_strings_are_nonempty_strings():
    """All returned features must be non-empty strings."""
    artifact = _make_artifact()
    patients = _make_patients(2)
    results = extract_shap_features(artifact, patients, top_k=2)
    for patient_features in results:
        for s in patient_features:
            assert isinstance(s, str)
            assert len(s) > 0


def test_feature_strings_contain_label():
    """Highest-importance feature must appear first."""
    artifact = _make_artifact(importances=[0.9, 0.02, 0.02, 0.02, 0.02])
    patients = _make_patients(1)
    results = extract_shap_features(artifact, patients, top_k=1)
    assert "creatinine (last)" in results[0][0]


def test_missing_columns_filled_with_zeros():
    """Patients missing some feature columns must not crash."""
    artifact = _make_artifact()
    patients = pd.DataFrame({"creatinine_last": [1.5, 2.0], "age": [65, 72]})
    results = extract_shap_features(artifact, patients, top_k=2)
    assert len(results) == 2


def test_empty_patients_returns_empty_list():
    """Zero-row input must return an empty list."""
    artifact = _make_artifact()
    patients = _make_patients(0)
    results = extract_shap_features(artifact, patients, top_k=3)
    assert not results


# ── Single-patient wrapper ────────────────────────────────────────────────────

def test_extract_shap_for_patient_returns_list_of_strings():
    """Single-patient wrapper must return a list of non-empty strings."""
    artifact = _make_artifact()
    patients = _make_patients(1)
    result = extract_shap_for_patient(artifact, patients.iloc[0], top_k=3)
    assert isinstance(result, list)
    assert len(result) == 3
    for s in result:
        assert isinstance(s, str) and len(s) > 0


def test_extract_shap_for_patient_matches_batch():
    """Single-patient wrapper must return same strings as batch for that row."""
    artifact = _make_artifact()
    patients = _make_patients(3)
    batch = extract_shap_features(artifact, patients, top_k=2)
    for i in range(3):
        single = extract_shap_for_patient(artifact, patients.iloc[i], top_k=2)
        assert single == batch[i]


# ── SHAP path (if shap is installed) ─────────────────────────────────────────

def test_shap_direction_labels_present_when_shap_installed():
    """When shap is available, output must include direction and SHAP value."""
    pytest.importorskip("shap")
    import xgboost as xgb  # pylint: disable=import-outside-toplevel
    from sklearn.datasets import make_classification  # pylint: disable=import-outside-toplevel

    x_arr, y_arr = make_classification(n_samples=60, n_features=5, random_state=0)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(x_arr, y_arr)

    artifact = {"estimator": model, "feature_cols": FEATURE_COLS}
    patients = pd.DataFrame(x_arr[:3], columns=FEATURE_COLS)
    results = extract_shap_features(artifact, patients, top_k=2)

    for patient_features in results:
        for s in patient_features:
            assert ("↑ risk" in s or "↓ risk" in s), f"no direction in: {s}"
            assert "SHAP=" in s
