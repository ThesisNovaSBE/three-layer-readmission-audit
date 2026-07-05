"""Stage 2 inference: classify Stage 1 flagged patients using fine-tuned Clinical-Longformer.

Loads the Stage 1 artifact to identify flagged admissions (score >= threshold),
then runs the fine-tuned Clinical-Longformer on their discharge notes to
confirm or reject each flag.

If models/stage2_calibration.json exists, applies Platt-scaled per-age-group
thresholds.  Falls back to the global threshold from config.yaml if not found.

Outputs a CSV to models/stage2_results.csv with columns:
    hadm_id, subject_id, readmission_30d, stage1_score, stage2_score, stage2_confirmed

Usage:
    python -m src.stage2.predict
    python -m src.stage2.predict --threshold 0.4   # override global fallback threshold
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, LongformerForSequenceClassification

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix, split_xy
from src.stage2.dataset import load_notes, build_notes_dataframe, ClinicalNotesDataset
from src.schemas import TARGET_COL


_BAND_MAP = {
    "(17, 40]":  "18-40",
    "(40, 55]":  "41-55",
    "(55, 70]":  "56-70",
    "(70, 120]": "70+",
}


def _band_key(age_band) -> str:
    return _BAND_MAP.get(str(age_band), str(age_band))


def predict_stage2(
    cfg: dict,
    stage2_threshold: float | None = None,
    artifact: dict | None = None,
) -> pd.DataFrame:
    """Run Stage 2 inference on patients flagged by Stage 1.

    Args:
        cfg: loaded config dict.
        stage2_threshold: probability cutoff for Stage 2 confirmation.
                          Defaults to cfg["stage2"]["threshold"].
        artifact: pre-loaded Stage 1 artifact dict. If None, loaded from disk.
                  Pass a pre-loaded artifact when torch is already imported
                  (joblib.load of an XGBoost model crashes on macOS if called
                  after torch is in memory).

    Returns:
        DataFrame with per-admission Stage 1 + Stage 2 scores and confirmation flag.
    """
    model_dir = get_model_dir()
    stage2_path = model_dir / "stage2_longformer_best"
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Stage 2 model not found at {stage2_path}. "
            "Run `python -m src.stage2.train` first."
        )

    global_threshold = stage2_threshold if stage2_threshold is not None \
                       else cfg["stage2"].get("threshold", 0.5)
    batch_size = cfg["stage2"]["batch_size"] * 2

    # Load per-group calibration if available
    cal_json_path = model_dir / "stage2_calibration.json"
    calibration: dict | None = None
    if cal_json_path.exists():
        calibration = json.loads(cal_json_path.read_text())
        print(f"[stage2/predict] Loaded calibration from {cal_json_path}")
    else:
        print(f"[stage2/predict] No calibration found — using global threshold={global_threshold:.3f}")

    # ── Load Stage 1 predictions on the test set ──
    stage1_name = cfg["stage1"]["model"]
    if artifact is None:
        artifact = joblib.load(model_dir / f"stage1_{stage1_name}.joblib")
    # Use the mode the Stage 1 model was trained on to reconstruct the exact
    # same feature matrix (and thus valid train_idx / test_idx positions).
    mode = artifact["mode"]
    max_length = cfg["stage2"]["max_seq_length"]
    matrix = load_feature_matrix(cfg, mode)
    X, y, groups, subgroups, _ = split_xy(matrix)

    test_idx = artifact["test_idx"]
    Xte = X.iloc[test_idx][artifact["feature_cols"]]
    stage1_scores = artifact["estimator"].predict_proba(Xte)[:, 1]
    stage1_threshold = artifact["threshold"]

    test_meta = matrix.iloc[test_idx][["hadm_id", "subject_id", "age_band", TARGET_COL]].reset_index(drop=True)
    test_meta["stage1_score"] = stage1_scores
    test_meta["stage1_threshold"] = stage1_threshold
    test_meta["age_band_key"] = test_meta["age_band"].apply(_band_key)

    flagged = test_meta[test_meta["stage1_score"] >= stage1_threshold].reset_index(drop=True)
    print(f"[stage2/predict] Stage 1 flagged {len(flagged):,} / {len(test_meta):,} test admissions "
          f"(threshold={stage1_threshold:.4f})")

    # ── Load notes for flagged patients only (chunked to avoid memory crash) ──
    flagged_hadm_ids = set(flagged["hadm_id"].astype(int).tolist())
    notes = load_notes(cfg, hadm_ids=flagged_hadm_ids)
    labels_df = flagged[["hadm_id", "subject_id", TARGET_COL]]
    notes_df = build_notes_dataframe(notes, labels_df)
    print(f"[stage2/predict] Discharge notes available for {len(notes_df):,} flagged admissions")

    # ── Tokenise ──
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    dataset = ClinicalNotesDataset(
        notes_df["text"].tolist(),
        notes_df[TARGET_COL].tolist(),
        tokenizer,
        max_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ── Detect device — force CPU on macOS (MPS breaks Longformer attention) ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── Inference ──
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

    notes_df = notes_df.copy()
    notes_df["stage2_score"] = all_probs

    # Merge age_band_key for per-group thresholding
    notes_df = notes_df.merge(
        flagged[["hadm_id", "stage1_score", "stage1_threshold", "age_band_key"]],
        on="hadm_id", how="left",
    )

    if calibration is not None:
        # Apply Platt scaling + per-group threshold
        cal_calibrators = calibration["calibrators"]
        cal_thresholds  = calibration["thresholds"]

        cal_probs = np.empty(len(notes_df))
        for i, (logit, band) in enumerate(zip(
            np.log(np.clip(notes_df["stage2_score"].values, 1e-7, 1 - 1e-7) /
                   np.clip(1 - notes_df["stage2_score"].values, 1e-7, 1 - 1e-7)),
            notes_df["age_band_key"].values,
        )):
            coef, intercept = cal_calibrators.get(band, [1.0, 0.0])
            cal_logit = coef * logit + intercept
            cal_probs[i] = 1.0 / (1.0 + np.exp(-cal_logit))

        notes_df["stage2_score"] = cal_probs
        band_thr = notes_df["age_band_key"].map(cal_thresholds).fillna(global_threshold)
        notes_df["stage2_confirmed"] = (notes_df["stage2_score"] >= band_thr).astype(int)
        print(f"[stage2/predict] Per-group thresholds: {cal_thresholds}")
    else:
        notes_df["stage2_confirmed"] = (notes_df["stage2_score"] >= global_threshold).astype(int)

    out = notes_df

    confirmed = int(out["stage2_confirmed"].sum())
    thr_display = "per-group" if calibration is not None else f"{global_threshold:.3f}"
    print(f"[stage2/predict] Stage 2 confirmed: {confirmed:,} / {len(out):,} "
          f"({confirmed / max(len(out), 1):.1%} retained | threshold={thr_display})")

    result_cols = ["hadm_id", "subject_id", TARGET_COL,
                   "stage1_score", "stage1_threshold", "stage2_score", "stage2_confirmed"]
    out = out[[c for c in result_cols if c in out.columns]]

    out_path = model_dir / "stage2_results.csv"
    out.to_csv(out_path, index=False)
    print(f"[stage2/predict] Saved results -> {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 inference on flagged patients")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Stage 2 confirmation probability threshold (default: 0.5)")
    args = parser.parse_args()
    cfg = load_config()
    predict_stage2(cfg, stage2_threshold=args.threshold)
