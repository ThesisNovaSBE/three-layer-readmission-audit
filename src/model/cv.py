"""Cross-validation helpers shared by tune.py and train.py.

All splitting is grouped by patient (subject_id). For XGBoost we carve a small
inner grouped validation set out of each fold's training data so early stopping
never peeks at the fold's scoring data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config_schema import AppConfig
from src.model.models import build_estimator, default_params, fit_estimator
from src.model.splits import make_cv


def fit_predict_fold(
    name: str,
    features: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    params: dict,
    cfg: AppConfig,
    seed: int,
) -> np.ndarray:
    """Fit on fold-train, return predicted probabilities for fold-validation.

    Args:
        name:      model identifier string.
        features:  full feature matrix (all folds combined).
        y:         all labels.
        groups:    patient-level group IDs.
        train_idx: row indices for the training portion of this fold.
        val_idx:   row indices for the validation portion of this fold.
        params:    hyperparameter dict.
        cfg:       validated project config.
        seed:      random seed.

    Returns:
        Predicted probabilities for the validation rows (shape ``(len(val_idx),)``).
    """
    x_train, y_train = features.iloc[train_idx], y[train_idx]
    x_val = features.iloc[val_idx]
    estimator = build_estimator(name, params, y_train, cfg, seed)

    if name == "xgboost":
        g_train = groups[train_idx]
        inner_tr, inner_va = next(
            GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            .split(x_train, y_train, g_train)
        )
        fit_estimator(
            estimator, name,
            x_train.iloc[inner_tr], y_train[inner_tr],
            x_val=x_train.iloc[inner_va], y_val=y_train[inner_va],
        )
    else:
        fit_estimator(estimator, name, x_train, y_train)

    return estimator.predict_proba(x_val)[:, 1]  # type: ignore[union-attr]


def oof_predict(
    name: str,
    features: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: AppConfig,
    seed: int,
    *,
    n_splits: int,
    params: dict | None = None,
) -> np.ndarray:
    """Return out-of-fold predicted probabilities over a grouped, stratified CV.

    Args:
        name:     model identifier string.
        features: feature matrix for the training set.
        y:        training labels.
        groups:   patient-level group IDs.
        cfg:      validated project config.
        seed:     random seed.
        n_splits: number of cross-validation folds.
        params:   hyperparameter dict; defaults to ``default_params(name)``.

    Returns:
        Array of out-of-fold predicted probabilities (same length as ``y``).
    """
    params = params or default_params(name)
    features = features.reset_index(drop=True)
    y = np.asarray(y)
    cv = make_cv(n_splits, seed)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(features, y, groups):
        oof[va] = fit_predict_fold(
            name, features, y, groups, tr, va,
            params=params, cfg=cfg, seed=seed,
        )
    return oof
