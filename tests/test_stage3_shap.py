"""Tests for SHAP feature extraction (no real XGBoost model required)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.stage3.shap_extract import _label, extract_shap_features


# ── Label helper ─────────────────────────────────────────────────────────────

def test_label_known_column():
    assert _label("creatinine_last") == "creatinine (last)"


def test_label_unknown_column_replaces_underscores():
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
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, (n, len(FEATURE_COLS))),
        columns=FEATURE_COLS,
    )


# ── Basic extraction (gain fallback, no shap library) ────────────────────────

def test_returns_list_per_patient():
    artifact = _make_artifact()
    patients = _make_patients(4)
    results = extract_shap_features(artifact, patients, top_k=3)
    assert len(results) == 4


def test_each_patient_has_top_k_features():
    artifact = _make_artifact()
    patients = _make_patients(2)
    results = extract_shap_features(artifact, patients, top_k=3)
    for patient_features in results:
        assert len(patient_features) == 3


def test_top_k_clamped_to_available_features():
    artifact = _make_artifact()
    patients = _make_patients(1)
    # Ask for more features than exist
    results = extract_shap_features(artifact, patients, top_k=100)
    assert len(results[0]) <= len(FEATURE_COLS)


def test_feature_strings_are_nonempty_strings():
    artifact = _make_artifact()
    patients = _make_patients(2)
    results = extract_shap_features(artifact, patients, top_k=2)
    for patient_features in results:
        for s in patient_features:
            assert isinstance(s, str)
            assert len(s) > 0


def test_feature_strings_contain_label():
    # creatinine_last has highest importance → should appear first
    artifact = _make_artifact(importances=[0.9, 0.02, 0.02, 0.02, 0.02])
    patients = _make_patients(1)
    results = extract_shap_features(artifact, patients, top_k=1)
    assert "creatinine (last)" in results[0][0]


def test_missing_columns_filled_with_zeros():
    artifact = _make_artifact()
    # Patient DataFrame missing some columns
    patients = pd.DataFrame({"creatinine_last": [1.5, 2.0], "age": [65, 72]})
    results = extract_shap_features(artifact, patients, top_k=2)
    assert len(results) == 2


def test_empty_patients_returns_empty_list():
    artifact = _make_artifact()
    patients = _make_patients(0)
    results = extract_shap_features(artifact, patients, top_k=3)
    assert results == []


# ── SHAP path (if shap is installed) ─────────────────────────────────────────

def test_shap_direction_labels_present_when_shap_installed():
    pytest.importorskip("shap")
    from sklearn.datasets import make_classification
    from sklearn.pipeline import Pipeline
    import xgboost as xgb

    X, y = make_classification(n_samples=60, n_features=5, random_state=0)
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X, y)

    artifact = {"estimator": model, "feature_cols": FEATURE_COLS}
    patients = pd.DataFrame(X[:3], columns=FEATURE_COLS)
    results = extract_shap_features(artifact, patients, top_k=2)

    for patient_features in results:
        for s in patient_features:
            assert ("↑ risk" in s or "↓ risk" in s), f"no direction in: {s}"
            assert "SHAP=" in s
