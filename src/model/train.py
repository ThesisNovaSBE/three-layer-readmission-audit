"""Train the Stage 1 model selected in config.yaml and save the artifact.

- Patient-level held-out test split (touched only by evaluate.py).
- Uses tuned hyperparameters from models/<model>_best_params.json if present,
  otherwise sensible defaults.
- Selects the decision threshold for the high-recall mandate using out-of-fold
  predictions on the TRAINING set (no leakage from the test set).
- Saves model + threshold + split + metadata to models/ (gitignored).

Usage::

    python -m src.model.train                 # uses config.yaml run.mode
    python -m src.model.train --mode full
    python -m src.model.train --model logistic_regression
"""

from __future__ import annotations

import argparse
import json

import joblib
from sklearn.model_selection import GroupShuffleSplit

from src.config import load_config, get_model_dir
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.calibration import calibration_report, fit_isotonic_calibrator
from src.model.cv import oof_predict
from src.model.metrics import (
    auprc,
    auroc,
    select_threshold_for_capacity,
    select_threshold_for_recall,
)
from src.model.models import build_estimator, default_params, fit_estimator
from src.model.splits import grouped_train_test_split


def _load_params(name: str, model_dir) -> dict:
    """Load tuned params from disk or fall back to defaults."""
    path = model_dir / f"{name}_best_params.json"
    if path.exists():
        print(f"[train] Using tuned params from {path.name}")
        return json.loads(path.read_text())
    print("[train] No tuned params found — using defaults.")
    return default_params(name)


def _calibrate(y_train, oof) -> dict:
    """Fit isotonic calibration on OOF predictions and report before/after Brier.

    Returns a dict with ``calibrator``, ``oof_calibrated``, and ``report``
    (the ``calibration_report`` dict) so callers only need one local.
    """
    calibrator = fit_isotonic_calibrator(y_train, oof)
    oof_calibrated = calibrator.predict(oof)
    report = calibration_report(y_train, oof, oof_calibrated)
    print(
        f"[train] Calibration (OOF): Brier {report['brier_before']:.4f} -> "
        f"{report['brier_after']:.4f}"
    )
    return {"calibrator": calibrator, "oof_calibrated": oof_calibrated, "report": report}


def _select_thresholds(cfg: AppConfig, y_train, oof) -> dict:
    """Select both operating-point thresholds and log them.

    Returns a dict with ``threshold`` (the one actually used, per
    ``cfg.stage1.threshold_strategy``), ``strategy``, ``recall_floor_threshold``,
    and ``capacity_threshold`` — the latter two always both computed so the
    unused strategy remains available for secondary reporting.
    """
    target_recall = cfg.stage1.target_recall
    recall_floor_threshold = select_threshold_for_recall(y_train, oof, target_recall)
    capacity_threshold = select_threshold_for_capacity(oof, cfg.stage1.capacity_k)
    strategy = cfg.stage1.threshold_strategy
    threshold = capacity_threshold if strategy == "capacity" else recall_floor_threshold
    print(
        f"[train] OOF AUPRC={auprc(y_train, oof):.4f} "
        f"AUROC={auroc(y_train, oof):.4f}\n"
        f"[train] threshold@recall>={target_recall}: {recall_floor_threshold:.4f} (secondary)\n"
        f"[train] threshold@capacity={cfg.stage1.capacity_k:.0%}: "
        f"{capacity_threshold:.4f} (primary if strategy=capacity)\n"
        f"[train] Using strategy='{strategy}' -> threshold={threshold:.4f}"
    )
    return {
        "threshold": threshold, "strategy": strategy,
        "target_recall": target_recall,
        "recall_floor_threshold": recall_floor_threshold,
        "capacity_threshold": capacity_threshold,
    }


def _fit_final_estimator(name, params, x_train, y_train, g_train, cfg, seed):
    """Build and fit the final Stage 1 estimator on the full training split."""
    estimator = build_estimator(name, params, y_train, cfg, seed)
    if name == "xgboost":
        itr, iva = next(
            GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            .split(x_train, y_train, g_train)
        )
        fit_estimator(
            estimator, name, x_train.iloc[itr], y_train[itr],
            x_val=x_train.iloc[iva], y_val=y_train[iva],
        )
    else:
        fit_estimator(estimator, name, x_train, y_train)
    return estimator


def train(cfg: AppConfig) -> None:
    """Train the Stage 1 model and save the artifact.

    Args:
        cfg: validated project config.
    """
    mode = cfg.run.mode
    seed = cfg.run.random_state
    name = cfg.stage1.model
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_feature_matrix(cfg, mode)
    features, y, groups, _subgroups, feat_cols = split_xy(matrix)
    print(f"[train] mode={mode} model={name} rows={len(y):,} "
          f"features={len(feat_cols)} pos_rate={y.mean():.1%}")

    train_idx, test_idx = grouped_train_test_split(
        y, groups, test_size=cfg.stage1.test_size, seed=seed)
    x_train = features.iloc[train_idx].reset_index(drop=True)
    y_train, g_train = y[train_idx], groups[train_idx]

    params = _load_params(name, model_dir)

    print("[train] Computing out-of-fold predictions for threshold selection...")
    oof = oof_predict(
        name, x_train, y_train, g_train, cfg, seed,
        n_splits=cfg.run.active().cv_folds, params=params,
    )

    # Isotonic calibration (remediation review task 0.1, 2026-08-26). Fit on
    # the same OOF predictions used for threshold selection — already
    # held-out relative to the model that produced them, so this adds no new
    # leakage surface. Monotonic, so it does not change which admissions get
    # flagged (see src/model/calibration.py), only what the reported
    # probability means.
    cal_info = _calibrate(y_train, oof)

    thr_info = _select_thresholds(cfg, y_train, cal_info["oof_calibrated"])
    threshold = thr_info["threshold"]

    estimator = _fit_final_estimator(name, params, x_train, y_train, g_train, cfg, seed)

    artifact = {
        "estimator": estimator, "model_name": name, "threshold": float(threshold),
        "feature_cols": feat_cols, "params": params, "seed": seed, "mode": mode,
        "train_idx": train_idx, "test_idx": test_idx,
        "target_recall": thr_info["target_recall"],
        "threshold_strategy": thr_info["strategy"],
        "recall_floor_threshold": float(thr_info["recall_floor_threshold"]),
        "capacity_threshold": float(thr_info["capacity_threshold"]),
        "capacity_k": cfg.stage1.capacity_k,
        "calibrator": cal_info["calibrator"],
    }
    out_path = model_dir / f"stage1_{name}.joblib"
    joblib.dump(artifact, out_path)
    print(f"[train] Saved model -> {out_path}")

    cal_path = model_dir / "stage1_calibration.json"
    cal_path.write_text(json.dumps({
        **cal_info["report"],
        "reliability_curve": {
            "raw_score": cal_info["calibrator"].X_thresholds_.tolist(),
            "calibrated_probability": cal_info["calibrator"].y_thresholds_.tolist(),
        },
    }, indent=2))
    print(f"[train] Saved calibration report -> {cal_path}")
    print("[train] Done. Run `python -m src.model.evaluate` to score the held-out test set.")


def main() -> None:
    """CLI entry point for Stage 1 training."""
    parser = argparse.ArgumentParser(description="Train Stage 1 model")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "histgradientboosting"],
        default=None,
    )
    args = parser.parse_args()

    _cfg = load_config()
    if args.mode:
        _cfg.run.mode = args.mode
    if args.model:
        _cfg.stage1.model = args.model
    train(_cfg)


if __name__ == "__main__":
    main()
