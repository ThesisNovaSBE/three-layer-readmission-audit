"""Optuna hyperparameter search for the Stage 1 model (optimises CV AUPRC).

- Search spaces come from src/model/models.py (ranges per docs/MODELING_PLAN.md).
- Grouped, stratified CV on the TRAINING portion only (test set never touched).
- MedianPruner stops unpromising trials early.
- Study is persisted to models/optuna_stage1_<model>_<target>_journal.log
  (Optuna's JournalFileBackend, not sqlite) -- resubmitting after a
  preemption/crash resumes toward the same trial target instead of
  restarting the whole search. The filename embeds which label
  (MODEL_TARGET_COL in src/schemas.py) it was searched against, so changing
  the target starts a fresh study instead of resuming trials scored against
  a different objective. JournalFileBackend, not sqlite, deliberately: sqlite's
  file-locking is unreliable on the NFS-style shared filesystems typical of
  HPC project directories (KISSKI/Grete), especially with >1 concurrent
  trial (below) -- this is the kind of thing that shows up as intermittent,
  hard-to-reproduce "database is locked" failures on the cluster and not
  locally. JournalFileBackend uses atomic symlink-based locking instead,
  which Optuna recommends specifically for this environment.
- ``stage1.tune_n_jobs`` (default 1) runs that many trials concurrently
  against the same GPU -- a single fit on this dataset is small relative to
  an A100's capacity, so running several at once is what actually uses it,
  rather than one trial saturating a tiny fraction of the GPU at a time.
- Writes best params to models/<model>_best_params.json for train.py to pick up.

Usage::

    python -m src.model.tune                  # uses config.yaml run.mode
    python -m src.model.tune --mode full
    python -m src.model.tune --model histgradientboosting
    python -m src.model.tune --mode full --device cuda --tune-n-jobs 8   # KISSKI/Grete
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

try:
    import optuna
except ImportError as _opt_err:
    raise ImportError(
        "Optuna is required for hyperparameter tuning. "
        "Install with: pip install optuna"
    ) from _opt_err

from src.config import load_config, get_model_dir
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.cv import fit_predict_fold
from src.model.metrics import auprc
from src.model.models import suggest_params
from src.model.splits import grouped_train_test_split, make_cv
from src.schemas import MODEL_TARGET_COL

# Short, filename-safe tag identifying which label a study/journal file
# belongs to -- derived directly from MODEL_TARGET_COL (not a suffix/name
# guess) so a THIRD future label variant can't silently collide with an
# existing tag the way a hardcoded "unplanned"/"allcause" binary would.
_TARGET_TAG = re.sub(r"[^a-z0-9]+", "", MODEL_TARGET_COL.lower())


def _load_or_create_study(model_dir, name: str, seed: int) -> optuna.Study:
    """Create (or resume) the Optuna study for one Stage 1 model.

    Persisted to disk (not in-memory) so a preempted/crashed cluster job can
    resume instead of losing the whole search -- load_if_exists picks up any
    trials already recorded under this study name. JournalFileBackend (not
    sqlite) for NFS-safe locking on shared HPC filesystems -- see module
    docstring.

    The journal filename and study name both embed ``_TARGET_TAG`` (which
    label column ``MODEL_TARGET_COL`` currently points at) so switching the
    model's target creates a fresh study instead of "resuming" trials whose
    recorded objective values were computed against a *different* label --
    confirmed necessary 2026-09-04, when the target was switched from
    all-cause to unplanned readmission mid-project with an existing
    ~290-trial journal already on disk for the old target.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
    journal_path = model_dir / f"optuna_stage1_{name}_{_TARGET_TAG}_journal.log"
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(journal_path))
    )
    return optuna.create_study(
        study_name=f"stage1_{name}_{_TARGET_TAG}",
        storage=storage,
        direction="maximize", sampler=sampler, pruner=pruner,
        load_if_exists=True,
    )


def run_study(cfg: AppConfig) -> dict:
    """Run an Optuna hyperparameter study and save the best parameters.

    Args:
        cfg: validated project config.

    Returns:
        Dict of best hyperparameter values.
    """
    mode = cfg.run.mode
    seed = cfg.run.random_state
    name = cfg.stage1.model
    n_splits = cfg.run.active().cv_folds
    n_trials = cfg.run.active().optuna_trials

    matrix = load_feature_matrix(cfg, mode)
    features, y, groups, _subgroups, _feat_cols = split_xy(matrix)

    train_idx, _test_idx = grouped_train_test_split(
        y, groups, test_size=cfg.stage1.test_size, seed=seed)
    x_train = features.iloc[train_idx].reset_index(drop=True)
    y_train, g_train = y[train_idx], groups[train_idx]

    print(
        f"[tune] mode={mode} model={name} device={cfg.stage1.device} "
        f"tune_n_jobs={cfg.stage1.tune_n_jobs} trials={n_trials} "
        f"cv_folds={n_splits} train_rows={len(y_train):,}"
    )

    def objective(trial: optuna.Trial) -> float:
        """Optimisation target: mean CV AUPRC."""
        params = suggest_params(name, trial)
        cross_val = make_cv(n_splits, seed)
        fold_scores = []
        for fold_i, (tr, va) in enumerate(cross_val.split(x_train, y_train, g_train)):
            proba = fit_predict_fold(
                name, x_train, y_train, g_train, tr, va,
                params=params, cfg=cfg, seed=seed,
            )
            fold_scores.append(auprc(y_train[va], proba))
            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    study = _load_or_create_study(model_dir, name, seed)

    n_done = len(study.trials)
    n_remaining = max(n_trials - n_done, 0)
    if n_done:
        print(f"[tune] Resuming study: {n_done} trial(s) already recorded, "
              f"{n_remaining} remaining toward target={n_trials}.")
    if n_remaining:
        study.optimize(
            objective, n_trials=n_remaining, show_progress_bar=True,
            n_jobs=cfg.stage1.tune_n_jobs,
        )

    print(f"[tune] Best CV AUPRC: {study.best_value:.4f}")
    print(f"[tune] Best params: {study.best_params}")

    out_path = model_dir / f"{name}_best_params.json"
    out_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"[tune] Saved best params -> {out_path}")
    return study.best_params


def main() -> None:
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Optuna search for Stage 1 model")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "histgradientboosting"],
        default=None,
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="XGBoost only; no effect on LR/HGB. cuda requires an NVIDIA GPU "
             "(KISSKI/Grete A100) -- never on Mac.",
    )
    parser.add_argument(
        "--tune-n-jobs", type=int, default=None,
        help="Concurrent Optuna trials against the same GPU/CPU. Keep at 1 "
             "locally; the cluster job passes a higher value explicitly.",
    )
    args = parser.parse_args()

    _cfg = load_config()
    if args.mode:
        _cfg.run.mode = args.mode
    if args.model:
        _cfg.stage1.model = args.model
    if args.device:
        _cfg.stage1.device = args.device
    if args.tune_n_jobs:
        _cfg.stage1.tune_n_jobs = args.tune_n_jobs
    run_study(_cfg)


if __name__ == "__main__":
    main()
