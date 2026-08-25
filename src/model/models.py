"""Model factory and Optuna search spaces for the three Stage 1 algorithms.

Algorithms (selected via config ``stage1.model``):

- ``logistic_regression``: L2 + class_weight="balanced", interpretable baseline
  (imputation + scaling done inside a Pipeline → per-fold, no leakage)
- ``xgboost``: hist tree method (CPU), eval_metric="aucpr",
  scale_pos_weight = neg/pos, n_estimators via early stopping
- ``histgradientboosting``: fast sklearn GBT, native NaN handling,
  class_weight="balanced"
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as _xgb_err:
    raise ImportError(
        "XGBoost is required for the 'xgboost' model. "
        "Install with: pip install xgboost"
    ) from _xgb_err

from src.config_schema import AppConfig

MODEL_NAMES = ["logistic_regression", "xgboost", "histgradientboosting"]


def scale_pos_weight(y: np.ndarray) -> float:
    """Return the negative-to-positive class ratio for XGBoost weighting."""
    y_arr = np.asarray(y)
    pos = float((y_arr == 1).sum())
    neg = float((y_arr == 0).sum())
    return neg / pos if pos else 1.0


def default_params(name: str) -> dict:
    """Return sensible defaults for the quick baseline (pre-tuning)."""
    if name == "logistic_regression":
        return {"C": 1.0}
    if name == "xgboost":
        return {
            "learning_rate": 0.05, "max_depth": 5, "min_child_weight": 3,
            "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.0,
            "reg_lambda": 1.0, "reg_alpha": 0.0,
        }
    if name == "histgradientboosting":
        return {
            "learning_rate": 0.1, "max_leaf_nodes": 31, "max_depth": None,
            "min_samples_leaf": 50, "l2_regularization": 0.0,
        }
    raise ValueError(f"Unknown model: {name}")


def suggest_params(name: str, trial: object) -> dict:
    """Return Optuna search-space suggestions for the given model.

    Args:
        name:  model identifier string.
        trial: an :class:`optuna.Trial` instance.
    """
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True)}
    if name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            # max_delta_step > 0 makes updates more conservative on imbalanced data
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            # max_bin controls histogram granularity for hist tree method
            "max_bin": trial.suggest_int("max_bin", 128, 512),
        }
    if name == "histgradientboosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
            "max_depth": trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-3, 10.0, log=True
            ),
        }
    raise ValueError(f"Unknown model: {name}")


def build_estimator(
    name: str, params: dict, y_train: np.ndarray, cfg: AppConfig, seed: int
) -> object:
    """Construct an unfitted estimator for the given model name.

    Args:
        name:    model identifier.
        params:  hyperparameter dict (from tuning or defaults).
        y_train: training labels (used to compute scale_pos_weight for XGBoost).
        cfg:     validated project config.
        seed:    random seed for reproducibility.
    """
    if name == "logistic_regression":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                C=params.get("C", 1.0), penalty="l2", class_weight="balanced",
                solver="liblinear", max_iter=2000, random_state=seed)),
        ])

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=cfg.stage1.n_estimators_cap,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=cfg.stage1.early_stopping_rounds,
            scale_pos_weight=scale_pos_weight(y_train),
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if name == "histgradientboosting":
        return HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            random_state=seed, max_iter=1000, **params)

    raise ValueError(f"Unknown model: {name}")


def fit_estimator(
    estimator: object,
    name: str,
    x_train: object,
    y_train: np.ndarray,
    *,
    x_val: object = None,
    y_val: np.ndarray | None = None,
) -> object:
    """Fit the estimator, wiring up an eval_set for XGBoost early stopping.

    Args:
        estimator: unfitted sklearn-compatible estimator.
        name:      model identifier string.
        x_train:   training features.
        y_train:   training labels.
        x_val:     validation features for XGBoost early stopping (keyword-only).
        y_val:     validation labels for XGBoost early stopping (keyword-only).

    Returns:
        The fitted estimator.
    """
    if name == "xgboost" and x_val is not None:
        estimator.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        estimator.fit(x_train, y_train)
    return estimator
