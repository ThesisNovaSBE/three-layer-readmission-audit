"""Shared pytest fixtures for the readmission prediction test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    """Project config in quick / synthetic mode."""
    c = load_config()
    c.run.mode = "quick"
    return c


@pytest.fixture
def binary_scores():
    """Reproducible y_true / y_score pair for metric tests (100 samples, 20% positive)."""
    rng = np.random.default_rng(42)
    y = np.array([0] * 80 + [1] * 20)
    # Scores are correlated with labels but not perfect
    scores = np.clip(y * 0.6 + rng.uniform(0.0, 0.4, 100), 0.0, 1.0)
    return y, scores


@pytest.fixture
def mock_stage2_results():
    """Minimal stage2_results DataFrame for Stage 3 tests."""
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "hadm_id": range(1000, 1000 + n),
        "subject_id": range(100, 100 + n),
        "readmission_30d": rng.integers(0, 2, n),
        "stage1_score": rng.uniform(0.35, 0.95, n),
        "stage1_threshold": [0.354] * n,
        "stage2_score": rng.uniform(0.1, 0.9, n),
        "stage2_confirmed": rng.integers(0, 2, n),
        "age_band": rng.choice(["18-40", "41-55", "56-70", "70+"], n),
        "discordance_mode": rng.choice(
            ["CONCORDANT", "NOTE_MITIGATES", "NOTE_AMPLIFIES"], n
        ),
        "primary_category": rng.choice(
            ["social_support", "discharge_planning", "structured_confirmed",
             "frailty_markers", "care_complexity"], n
        ),
        "explanation": ["Test explanation."] * n,
        "annotation_failed": [False] * n,
    })
