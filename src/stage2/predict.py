"""Stage 2 inference: score admissions with Clinical-Longformer.

Two modes, sharing the same scoring pipeline:

- ``predict_stage2`` — Stage 1-flagged admissions only (``stage1_score >=
  threshold``). This is the cascade population: Layer 3's audit workload.
  Output: ``models/stage2_results.csv``.
- ``predict_stage2_all`` — every test-partition admission with a note,
  independent of Stage 1's flag. This is required for RQ1 (does text carry
  signal at all) — comparing Stage 1 against Stage 2 is only fair if both
  are evaluated on the same population; restricting Stage 2 to Stage 1's
  positives would both bias it toward an already-high-risk subset and make
  the comparison impossible. Output: ``models/stage2_results_all.csv``.
  See docs/ARCHITECTURE.md.

If ``models/stage2_calibration.json`` exists, applies Platt-scaled per-age-group
thresholds. Falls back to the global threshold from ``config.yaml`` if not
found. The calibrators were fit on a slice of Stage 1's *training* patients
(not the flagged subset), so they are valid to apply in both modes.

Usage::

    python -m src.stage2.predict
    python -m src.stage2.predict --all               # population-wide, for RQ1
    python -m src.stage2.predict --threshold 0.4      # override global fallback
    python -m src.stage2.predict --limit 200           # smoke-test on a slice
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

try:
    # Found 2026-08-27, NOT FULLY FIXED: running this module's CLI standalone
    # (`python -m src.stage2.predict`, with or without --all) segfaults
    # (SIGSEGV, exit 139) on macOS ARM. Confirmed pre-existing and
    # reproducible on the pre-session-17 code too -- not introduced by
    # today's refactor. Root cause is the MPS/XGBoost C-extension conflict
    # documented elsewhere in this codebase (api.py, setup_stage2.py):
    # torch's C extensions initialise at import time, and joblib.load()-ing
    # the XGBoost artifact after torch has initialised conflicts on Apple
    # Silicon. The env vars below are the standard mitigation (see
    # setup_stage2.py) but do NOT fully fix it here, because
    # `from src.stage2.dataset import ...` below transitively imports torch
    # too -- by the time _score_core() calls joblib.load(), torch has
    # already initialised regardless of this module's own import order. The
    # real fix (making every torch-touching import lazy, deferred until
    # after an artifact is already loaded) is a bigger change than this
    # session's scope; not attempted here to avoid risking setup_stage2.py's
    # already-working pre-load pattern. Safe ways to run this locally on a
    # Mac today: go through setup_stage2.py (pre-loads the artifact before
    # importing torch), or run on Linux/the cluster, where MPS doesn't exist
    # and this never triggers. See docs/ARCHITECTURE.md.
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer, LongformerForSequenceClassification
except ImportError as _torch_err:
    raise ImportError(
        "Stage 2 requires PyTorch and Transformers. "
        "Install with: pip install torch transformers"
    ) from _torch_err

# DataLoader, tqdm, AutoTokenizer, LongformerForSequenceClassification are
# used by _run_inference() defined later in this module.

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.calibration import apply_calibration
# Aliased: Stage 2 must score/report against the same target Stage 1 uses
# (MODEL_TARGET_COL, unplanned readmission per src/schemas.py).
from src.schemas import MODEL_TARGET_COL as TARGET_COL
from src.stage2._utils import band_key, get_stage2_model_path
from src.stage2.dataset import ClinicalNotesDataset, build_notes_dataframe, load_notes


def _prepare_population(
    artifact: dict, matrix: pd.DataFrame, *, flagged_only: bool
) -> pd.DataFrame:
    """Score Stage 1 on the test partition and attach age_band_key.

    Args:
        artifact:     pre-loaded Stage 1 artifact.
        matrix:       full feature matrix for the mode used during Stage 1
                      training.
        flagged_only: if True, return only admissions with
                      ``stage1_score >= threshold`` (the cascade population).
                      If False, return every test-partition admission (the
                      population-wide, RQ1 comparison mode).

    Returns:
        DataFrame with columns: hadm_id, subject_id, age_band,
        readmission_30d, stage1_score, stage1_threshold, age_band_key.
    """
    features = split_xy(matrix)[0]
    test_idx = artifact["test_idx"]
    x_test = features.iloc[test_idx][artifact["feature_cols"]]
    raw_scores = artifact["estimator"].predict_proba(x_test)[:, 1]
    stage1_scores = apply_calibration(artifact, raw_scores)
    stage1_threshold = artifact["threshold"]

    test_meta = matrix.iloc[test_idx][
        ["hadm_id", "subject_id", "age_band", TARGET_COL]
    ].reset_index(drop=True)
    test_meta["stage1_score"] = stage1_scores
    test_meta["stage1_threshold"] = stage1_threshold
    test_meta["age_band_key"] = test_meta["age_band"].apply(band_key)

    if flagged_only:
        population = test_meta[
            test_meta["stage1_score"] >= stage1_threshold
        ].reset_index(drop=True)
        print(
            f"[stage2/predict] Stage 1 flagged {len(population):,} / "
            f"{len(test_meta):,} test admissions "
            f"(threshold={stage1_threshold:.4f})"
        )
    else:
        population = test_meta
        print(
            f"[stage2/predict] Scoring all {len(population):,} test admissions "
            f"(population-wide, not gated by Stage 1's flag)"
        )
    return population


def _run_inference(
    notes_df: pd.DataFrame,
    stage2_path: object,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Run Longformer inference and return softmax probabilities for class 1.

    Args:
        notes_df:    DataFrame with ``text`` and ``TARGET_COL`` columns.
        stage2_path: path to the fine-tuned model directory.
        batch_size:  inference batch size.
        max_length:  tokenization max length.

    Returns:
        List of float probabilities (one per row in notes_df).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    dataset = ClinicalNotesDataset(
        notes_df["text"].tolist(), notes_df[TARGET_COL].tolist(),
        tokenizer, max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = LongformerForSequenceClassification.from_pretrained(str(stage2_path))
    model.to(device)
    model.eval()

    all_probs: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[stage2] Scoring notes"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
    return all_probs


def _apply_calibration(
    notes_df: pd.DataFrame,
    calibration: dict,
    global_threshold: float,
) -> pd.DataFrame:
    """Apply Platt scaling and per-group thresholds to raw stage2_scores.

    Converts raw softmax probabilities to logits, applies the per-band
    calibration coefficients, converts back to calibrated probabilities,
    then applies the per-band decision threshold.

    Args:
        notes_df:         DataFrame with ``stage2_score`` and ``age_band_key``.
        calibration:      calibration dict loaded from JSON.
        global_threshold: fallback threshold for bands missing from calibration.

    Returns:
        Updated DataFrame with calibrated ``stage2_score`` and
        ``stage2_confirmed`` columns.
    """
    cal_calibrators = calibration["calibrators"]
    cal_thresholds = calibration["thresholds"]

    raw_scores = notes_df["stage2_score"].values
    raw_logits = np.log(
        np.clip(raw_scores, 1e-7, 1 - 1e-7)
        / np.clip(1 - raw_scores, 1e-7, 1 - 1e-7)
    )

    cal_probs = np.empty(len(notes_df))
    for i, (logit, band) in enumerate(zip(raw_logits, notes_df["age_band_key"].values)):
        coef, intercept = cal_calibrators.get(band, [1.0, 0.0])
        cal_logit = coef * logit + intercept
        cal_probs[i] = 1.0 / (1.0 + np.exp(-cal_logit))

    notes_df = notes_df.copy()
    notes_df["stage2_score"] = cal_probs
    band_thr = notes_df["age_band_key"].map(cal_thresholds).fillna(global_threshold)
    notes_df["stage2_confirmed"] = (notes_df["stage2_score"] >= band_thr).astype(int)
    print(f"[stage2/predict] Per-group thresholds: {cal_thresholds}")
    return notes_df


def _score_core(
    cfg: AppConfig,
    *,
    flagged_only: bool,
    stage2_threshold: float | None,
    artifact: dict | None,
    limit: int | None,
) -> pd.DataFrame:
    """Shared scoring pipeline for both ``predict_stage2`` and
    ``predict_stage2_all``: prepare population, run inference, calibrate.

    Args:
        limit: if given, scores only the first ``limit`` admissions of the
               prepared population — for smoke-testing the population-wide
               path locally without a full cluster run.
    """
    model_dir = get_model_dir()
    stage2_path = get_stage2_model_path(model_dir)

    global_threshold = (
        stage2_threshold if stage2_threshold is not None
        else cfg.stage2.threshold
    )
    batch_size = cfg.stage2.batch_size * 2
    max_length = cfg.stage2.max_seq_length

    cal_json_path = model_dir / "stage2_calibration.json"
    calibration: dict | None = None
    if cal_json_path.exists():
        calibration = json.loads(cal_json_path.read_text())
        print(f"[stage2/predict] Loaded calibration from {cal_json_path}")
    else:
        print(
            f"[stage2/predict] No calibration found — "
            f"using global threshold={global_threshold:.3f}"
        )

    if artifact is None:
        artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    matrix = load_feature_matrix(cfg, artifact["mode"])
    population = _prepare_population(artifact, matrix, flagged_only=flagged_only)
    if limit is not None:
        population = population.iloc[:limit].reset_index(drop=True)
        print(f"[stage2/predict] --limit applied: scoring {len(population):,} admissions")

    population_hadm_ids = set(population["hadm_id"].astype(int).tolist())
    notes = load_notes(cfg, hadm_ids=population_hadm_ids)
    notes_df = build_notes_dataframe(
        notes, population[["hadm_id", "subject_id", TARGET_COL]]
    )
    print(f"[stage2/predict] Discharge notes available for {len(notes_df):,} admissions")

    all_probs = _run_inference(notes_df, stage2_path, batch_size, max_length)

    notes_df = notes_df.copy()
    notes_df["stage2_score"] = all_probs
    notes_df = notes_df.merge(
        population[["hadm_id", "stage1_score", "stage1_threshold", "age_band_key"]],
        on="hadm_id", how="left",
    )

    if calibration is not None:
        notes_df = _apply_calibration(notes_df, calibration, global_threshold)
        thr_display = "per-group"
    else:
        notes_df["stage2_confirmed"] = (
            notes_df["stage2_score"] >= global_threshold
        ).astype(int)
        thr_display = f"{global_threshold:.3f}"

    confirmed = int(notes_df["stage2_confirmed"].sum())
    print(
        f"[stage2/predict] Stage 2 confirmed: {confirmed:,} / {len(notes_df):,} "
        f"({confirmed / max(len(notes_df), 1):.1%} | threshold={thr_display})"
    )

    result_cols = [
        "hadm_id", "subject_id", TARGET_COL,
        "stage1_score", "stage1_threshold", "stage2_score", "stage2_confirmed",
    ]
    return notes_df[[c for c in result_cols if c in notes_df.columns]]


def predict_stage2(
    cfg: AppConfig,
    stage2_threshold: float | None = None,
    artifact: dict | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run Stage 2 inference on patients flagged by Stage 1 (the cascade population).

    Args:
        cfg:               validated project config.
        stage2_threshold:  probability cutoff for Stage 2 confirmation.
                           Defaults to ``cfg.stage2.threshold``.
        artifact:          pre-loaded Stage 1 artifact dict. If ``None``,
                           loaded from disk. Pass a pre-loaded artifact when
                           torch is already imported (joblib.load of an XGBoost
                           model crashes on macOS if called after torch).
        limit:             score only the first N admissions (smoke-testing).

    Returns:
        DataFrame with per-admission Stage 1 + Stage 2 scores and confirmation
        flag. Saved to ``models/stage2_results.csv``.
    """
    out = _score_core(
        cfg, flagged_only=True, stage2_threshold=stage2_threshold,
        artifact=artifact, limit=limit,
    )
    out_path = get_model_dir() / "stage2_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[stage2/predict] Saved results -> {out_path}")
    return out


def predict_stage2_all(
    cfg: AppConfig,
    stage2_threshold: float | None = None,
    artifact: dict | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run Stage 2 inference on every test-partition admission with a note.

    Population-wide, independent of Stage 1's flag — required for a fair
    RQ1 comparison against Stage 1 (see module docstring and
    docs/ARCHITECTURE.md). ``stage2_confirmed`` in the output is a diagnostic
    only here (it reuses the cascade threshold) — it is not a deployment
    decision for admissions Stage 1 never flagged.

    Args:
        cfg, stage2_threshold, artifact, limit: as in :func:`predict_stage2`.

    Returns:
        DataFrame as in :func:`predict_stage2`. Saved to
        ``models/stage2_results_all.csv``.
    """
    out = _score_core(
        cfg, flagged_only=False, stage2_threshold=stage2_threshold,
        artifact=artifact, limit=limit,
    )
    out_path = get_model_dir() / "stage2_results_all.csv"
    out.to_csv(out_path, index=False)
    print(f"[stage2/predict] Saved results -> {out_path}")
    return out


def main() -> None:
    """CLI entry point for Stage 2 inference."""
    parser = argparse.ArgumentParser(description="Stage 2 inference")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Stage 2 confirmation probability threshold (default: from config)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Score every test-partition admission with a note, not just "
             "Stage 1-flagged ones (population-wide, for the RQ1 comparison). "
             "Writes models/stage2_results_all.csv instead of stage2_results.csv.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N admissions of the population — for "
             "smoke-testing --all locally without a full cluster run.",
    )
    args = parser.parse_args()
    _cfg = load_config()
    if args.all:
        predict_stage2_all(_cfg, stage2_threshold=args.threshold, limit=args.limit)
    else:
        predict_stage2(_cfg, stage2_threshold=args.threshold, limit=args.limit)


if __name__ == "__main__":
    main()
