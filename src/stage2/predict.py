"""Stage 2 inference: classify Stage 1 flagged patients with FusionLongformer.

Loads the Stage 1 artifact to identify flagged admissions (score >= threshold),
then runs the fine-tuned FusionLongformer on their discharge notes to confirm
or reject each flag.

If ``models/stage2_fusion_config.json`` exists, the inference model is a
:class:`~src.stage2.model.FusionLongformer` (structured feature fusion).
Otherwise the script falls back to a plain ``LongformerForSequenceClassification``
for compatibility with checkpoints trained before the fusion update.

If ``models/stage2_calibration.json`` exists, applies Platt-scaled per-age-group
thresholds. Falls back to the global threshold from ``config.yaml`` if not found.

Output: ``models/stage2_results.csv`` with columns:

    hadm_id, subject_id, readmission_30d, stage1_score, stage2_score,
    stage2_confirmed

Usage::

    python -m src.stage2.predict
    python -m src.stage2.predict --threshold 0.4   # override global fallback
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer, LongformerForSequenceClassification
except ImportError as _torch_err:
    raise ImportError(
        "Stage 2 requires PyTorch and Transformers. "
        "Install with: pip install torch transformers"
    ) from _torch_err

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.schemas import TARGET_COL
from src.stage2._utils import STRUCT_FEATURE_COLS, band_key, get_stage2_model_path
from src.stage2.dataset import (
    ClinicalNotesDataset,
    build_notes_dataframe,
    load_notes,
    normalize_struct_features,
)
from src.stage2.model import FusionLongformer


def _prepare_flagged(
    artifact: dict, matrix: pd.DataFrame
) -> pd.DataFrame:
    """Identify Stage 1 flagged admissions and attach struct features.

    Scores all test-partition admissions with Stage 1 and returns only those
    at or above the threshold.  The structured feature columns required by
    :class:`FusionLongformer` are pulled from the feature matrix and attached
    so they are available for inference.

    Args:
        artifact: pre-loaded Stage 1 artifact.
        matrix:   full feature matrix for the mode used during Stage 1 training.

    Returns:
        DataFrame of flagged admissions with columns:
        hadm_id, subject_id, age_band, readmission_30d,
        stage1_score, stage1_threshold, age_band_key,
        plus all available struct feature columns.
    """
    features = split_xy(matrix)[0]
    test_idx = artifact["test_idx"]
    x_test = features.iloc[test_idx][artifact["feature_cols"]]
    stage1_scores = artifact["estimator"].predict_proba(x_test)[:, 1]
    stage1_threshold = artifact["threshold"]

    struct_cols = [c for c in STRUCT_FEATURE_COLS if c in matrix.columns]
    test_meta = matrix.iloc[test_idx][
        ["hadm_id", "subject_id", "age_band", TARGET_COL] + struct_cols
    ].reset_index(drop=True)
    test_meta["stage1_score"] = stage1_scores
    test_meta["stage1_threshold"] = stage1_threshold
    test_meta["age_band_key"] = test_meta["age_band"].apply(band_key)

    flagged = test_meta[
        test_meta["stage1_score"] >= stage1_threshold
    ].reset_index(drop=True)
    print(
        f"[stage2/predict] Stage 1 flagged {len(flagged):,} / "
        f"{len(test_meta):,} test admissions "
        f"(threshold={stage1_threshold:.4f})"
    )
    return flagged


def _load_model(stage2_path) -> object:
    """Load FusionLongformer if available, else fall back to plain Longformer.

    Fusion config and weights are saved inside ``stage2_path`` by
    :func:`~src.stage2.train._save_results`.

    Args:
        stage2_path: path to the fine-tuned model directory.

    Returns:
        An unfitted model ready for ``.eval()`` and inference.
    """
    fusion_cfg_path = stage2_path / "stage2_fusion_config.json"
    if fusion_cfg_path.exists():
        fusion_cfg = json.loads(fusion_cfg_path.read_text())
        model = FusionLongformer(str(stage2_path), fusion_cfg["n_struct_features"])
        weights_path = stage2_path / "fusion_weights.pt"
        model.load_state_dict(
            torch.load(str(weights_path), map_location="cpu", weights_only=True)
        )
        print(f"[stage2/predict] Loaded FusionLongformer "
              f"(n_struct={fusion_cfg['n_struct_features']}) from {stage2_path}")
        return model
    print(f"[stage2/predict] No fusion config found — loading plain Longformer from {stage2_path}")
    return LongformerForSequenceClassification.from_pretrained(str(stage2_path))


def _run_inference(
    notes_df: pd.DataFrame,
    stage2_path,
    batch_size: int,
    max_length: int,
    scaler_stats: dict | None,
) -> list[float]:
    """Run inference and return softmax probabilities for class 1.

    Supports both :class:`FusionLongformer` (with structured features) and the
    legacy plain ``LongformerForSequenceClassification``.

    Args:
        notes_df:     DataFrame with ``text``, ``TARGET_COL``, and any struct
                      feature columns.
        stage2_path:  path to the fine-tuned model directory.
        model_dir:    project model directory.
        batch_size:   inference batch size.
        max_length:   tokenization max sequence length.
        scaler_stats: normalisation stats from training (``means`` + ``stds``).
                      ``None`` for legacy (non-fusion) models.

    Returns:
        List of float probabilities (one per row in notes_df).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    model = _load_model(stage2_path)
    model.to(device)
    model.eval()

    struct_arr: np.ndarray | None = None
    if scaler_stats is not None:
        struct_arr, _, _ = normalize_struct_features(
            notes_df,
            means=scaler_stats.get("means"),
            stds=scaler_stats.get("stds"),
        )

    dataset = ClinicalNotesDataset(
        notes_df["text"].tolist(), notes_df[TARGET_COL].tolist(),
        tokenizer, max_length, struct_features=struct_arr,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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


def predict_stage2(
    cfg: AppConfig,
    stage2_threshold: float | None = None,
    artifact: dict | None = None,
) -> pd.DataFrame:
    """Run Stage 2 inference on patients flagged by Stage 1.

    Args:
        cfg:               validated project config.
        stage2_threshold:  probability cutoff for Stage 2 confirmation.
                           Defaults to ``cfg.stage2.threshold``.
        artifact:          pre-loaded Stage 1 artifact dict. If ``None``,
                           loaded from disk. Pass a pre-loaded artifact when
                           torch is already imported (joblib.load of an XGBoost
                           model crashes on macOS if called after torch).

    Returns:
        DataFrame with per-admission Stage 1 + Stage 2 scores and confirmation
        flag.
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

    scaler_json_path = model_dir / "stage2_struct_scaler.json"
    scaler_stats: dict | None = None
    if scaler_json_path.exists():
        scaler_stats = json.loads(scaler_json_path.read_text())
        print(f"[stage2/predict] Loaded struct scaler from {scaler_json_path}")

    if artifact is None:
        artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    matrix = load_feature_matrix(cfg, artifact["mode"])
    flagged = _prepare_flagged(artifact, matrix)

    flagged_hadm_ids = set(flagged["hadm_id"].astype(int).tolist())
    notes = load_notes(cfg, hadm_ids=flagged_hadm_ids)
    notes_df = build_notes_dataframe(
        notes, flagged[["hadm_id", "subject_id", TARGET_COL]]
    )

    # Attach struct feature columns from the flagged DataFrame for fusion inference
    struct_cols = [c for c in STRUCT_FEATURE_COLS if c in flagged.columns]
    if struct_cols and scaler_stats is not None:
        notes_df = notes_df.merge(
            flagged[["hadm_id"] + struct_cols], on="hadm_id", how="left"
        )

    print(
        f"[stage2/predict] Discharge notes available for "
        f"{len(notes_df):,} flagged admissions"
    )

    all_probs = _run_inference(
        notes_df, stage2_path, batch_size, max_length, scaler_stats
    )

    notes_df = notes_df.copy()
    notes_df["stage2_score"] = all_probs
    notes_df = notes_df.merge(
        flagged[["hadm_id", "stage1_score", "stage1_threshold", "age_band_key"]],
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
        f"({confirmed / max(len(notes_df), 1):.1%} retained | threshold={thr_display})"
    )

    result_cols = [
        "hadm_id", "subject_id", TARGET_COL,
        "stage1_score", "stage1_threshold", "stage2_score", "stage2_confirmed",
    ]
    out = notes_df[[c for c in result_cols if c in notes_df.columns]]

    out_path = model_dir / "stage2_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[stage2/predict] Saved results -> {out_path}")
    return out


def main() -> None:
    """CLI entry point for Stage 2 inference."""
    parser = argparse.ArgumentParser(description="Stage 2 inference on flagged patients")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Stage 2 confirmation probability threshold (default: from config)",
    )
    args = parser.parse_args()
    _cfg = load_config()
    predict_stage2(_cfg, stage2_threshold=args.threshold)


if __name__ == "__main__":
    main()
