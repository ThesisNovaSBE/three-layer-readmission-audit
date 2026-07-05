"""Patient-level (grouped) splitting helpers — no patient in two splits.

Group = subject_id everywhere. We use StratifiedGroupKFold so the label rate is
balanced across folds while keeping every patient's admissions together.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_train_test_split(
    y: np.ndarray, groups: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) split by group, stratified on y.

    Approximates ``test_size`` by choosing ``n_splits = round(1/test_size)`` and
    using the first fold as the held-out test set.
    """
    n_splits = max(2, round(1.0 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def make_cv(n_splits: int = 5, seed: int = 42) -> StratifiedGroupKFold:
    """Return a configured :class:`StratifiedGroupKFold` cross-validator."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
