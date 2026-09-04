"""Post-training calibration for Stage 2 Clinical-Longformer.

Runs Platt scaling (logistic regression on raw logits) per age group on the
calibration split, then selects per-group decision thresholds that satisfy a
recall floor while maximising F2.

Output: ``models/stage2_calibration.json``::

    {
      "calibrators": {"18-40": [coef, intercept], ...},
      "thresholds":  {"18-40": 0.31, "41-55": 0.28, ...},
      "strategy":    "recall_floor",
      "recall_floor": 0.65,
      "calibration_metrics": {...}
    }

Usage::

    python -m src.stage2.calibrate
    python -m src.stage2.calibrate --force
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
    from transformers import AutoTokenizer, LongformerForSequenceClassification
except ImportError as _torch_err:
    raise ImportError(
        "Stage 2 requires PyTorch and Transformers. "
        "Install with: pip install torch transformers"
    ) from _torch_err

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
# Aliased: Stage 2 must calibrate on the same target Stage 1 uses
# (MODEL_TARGET_COL, unplanned readmission per src/schemas.py).
from src.schemas import MODEL_TARGET_COL as TARGET_COL
from src.stage2._utils import band_key, get_stage2_model_path
from src.stage2.dataset import ClinicalNotesDataset, build_notes_dataframe, load_notes
from src.stage2.splits import build_splits


# ── Inference helpers ─────────────────────────────────────────────────────────

def _get_logits(
    model: object,
    dataset: ClinicalNotesDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return raw class-1 logits for every sample."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits: list[float] = []
    model.eval()  # type: ignore[union-attr]
    with torch.no_grad():
        for batch in tqdm(loader, desc="[calibrate] scoring"):
            inputs = {
                k: v.to(device) for k, v in batch.items()
                if k not in {"labels", "group_weight"}
            }
            logits = model(**inputs).logits  # type: ignore[operator]
            all_logits.extend(logits[:, 1].cpu().numpy().tolist())
    return np.array(all_logits)


# ── Threshold selection ───────────────────────────────────────────────────────

def _recall_floor_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    recall_floor: float,
    n_steps: int = 200,
) -> float:
    """Return the threshold satisfying recall >= recall_floor with max F2.

    If no threshold achieves the recall floor, returns the one with the
    highest recall.

    Args:
        probs:        calibrated probabilities for the group.
        labels:       true binary labels.
        recall_floor: minimum required recall.
        n_steps:      number of threshold candidates to evaluate.

    Returns:
        Selected decision threshold (float in [0.01, 0.99]).
    """
    best_thr = 0.5
    best_f2 = -1.0
    best_recall = -1.0

    for thr in np.linspace(0.01, 0.99, n_steps):
        preds = (probs >= thr).astype(int)
        rec = recall_score(labels, preds, zero_division=0)
        if rec < recall_floor:
            continue
        pre = precision_score(labels, preds, zero_division=0)
        f2 = (5 * pre * rec) / (4 * pre + rec + 1e-9)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)
            best_recall = rec

    if best_f2 < 0:
        for thr in np.linspace(0.01, 0.99, n_steps):
            preds = (probs >= thr).astype(int)
            rec = recall_score(labels, preds, zero_division=0)
            if rec > best_recall:
                best_recall = rec
                best_thr = float(thr)

    return best_thr


# ── Per-band processing ───────────────────────────────────────────────────────

def _process_band(
    raw_logits: np.ndarray,
    labels_arr: np.ndarray,
    mask: np.ndarray,
    strategy: str,
    recall_floor: float,
) -> tuple[list[float], float, dict]:
    """Fit Platt scaling and select threshold for one age group.

    Args:
        raw_logits:   raw class-1 logits for ALL samples.
        labels_arr:   true binary labels for ALL samples.
        mask:         boolean mask selecting this group's rows.
        strategy:     threshold selection strategy (``"recall_floor"``).
        recall_floor: minimum required recall if strategy is ``"recall_floor"``.

    Returns:
        Tuple of ``(calibrator_params, threshold, metrics_dict)``.
    """
    x_band = raw_logits[mask].reshape(-1, 1)
    y_band = labels_arr[mask]

    if y_band.sum() < 5 or (y_band == 0).sum() < 5:
        print(
            f"  [calibrate] too few samples ({mask.sum()}) — "
            "using identity calibration"
        )
        coef_list: list[float] = [1.0, 0.0]
    else:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(x_band, y_band)
        coef_list = [float(lr.coef_[0][0]), float(lr.intercept_[0])]

    coef, intercept = coef_list
    cal_logits = coef * raw_logits[mask] + intercept
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))

    thr = (
        _recall_floor_threshold(cal_probs, y_band, recall_floor)
        if strategy == "recall_floor" else 0.5
    )

    preds = (cal_probs >= thr).astype(int)
    rec = recall_score(y_band, preds, zero_division=0)
    pre = precision_score(y_band, preds, zero_division=0)
    try:
        band_auroc = float(roc_auc_score(y_band, cal_probs))
    except ValueError:
        band_auroc = float("nan")
    f2 = (5 * pre * rec) / (4 * pre + rec + 1e-9)

    metrics: dict = {
        "n": int(mask.sum()),
        "pos_rate": float(y_band.mean()),
        "threshold": thr,
        "recall": float(rec),
        "precision": float(pre),
        "f2": float(f2),
        "auroc": band_auroc,
    }
    return coef_list, thr, metrics


# ── Note loading and scoring helpers ─────────────────────────────────────────

def _load_cal_notes(
    cfg: AppConfig,
    artifact: dict,
    splits: dict,
) -> pd.DataFrame:
    """Load calibration-split discharge notes with labels and age bands.

    Args:
        cfg:      validated project config.
        artifact: pre-loaded Stage 1 artifact (used by :func:`build_splits`
                  if splits are not yet cached).
        splits:   dict with ``"cal"`` key from :func:`build_splits`.

    Returns:
        DataFrame with columns ``hadm_id``, ``subject_id``, ``age_band``,
        ``TARGET_COL``, and ``text``.
    """
    _ = artifact  # kept for API symmetry; splits already computed by caller
    cal_split = splits["cal"]
    cal_hadm_ids = set(cal_split["hadm_id"].astype(int).tolist())
    print(f"[calibrate] Loading notes for {len(cal_hadm_ids):,} calibration admissions ...")

    notes = load_notes(cfg, hadm_ids=cal_hadm_ids)
    notes_df = build_notes_dataframe(
        notes, cal_split[["hadm_id", "subject_id", TARGET_COL]]
    )
    notes_df = notes_df.merge(
        cal_split[["hadm_id", "age_band"]], on="hadm_id", how="left"
    )
    print(f"[calibrate] Notes available for {len(notes_df):,} admissions")
    return notes_df


def _score_cal_notes(
    cal_df: pd.DataFrame,
    stage2_path: object,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Stage 2 inference on the calibration split and return raw logits.

    Args:
        cal_df:      DataFrame with ``text``, ``TARGET_COL``, and ``age_band``
                     columns.
        stage2_path: path to the fine-tuned model directory.
        batch_size:  inference batch size.
        max_length:  tokenization max length.

    Returns:
        Tuple ``(raw_logits, labels_arr, bands)`` — each a 1-D numpy array
        aligned row-by-row with ``cal_df``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_path))
    dataset = ClinicalNotesDataset(
        cal_df["text"].tolist(),
        cal_df[TARGET_COL].tolist(),
        tokenizer,
        max_length,
    )
    model = LongformerForSequenceClassification.from_pretrained(str(stage2_path))
    model.to(device)

    raw_logits = _get_logits(model, dataset, batch_size, device)
    labels_arr = np.array(cal_df[TARGET_COL].tolist())
    bands = np.array(cal_df["age_band"].apply(band_key).tolist())
    return raw_logits, labels_arr, bands


# ── Main calibration routine ──────────────────────────────────────────────────

def calibrate(cfg: AppConfig, artifact: dict | None = None, force: bool = False) -> dict:
    """Run Platt scaling and per-group threshold selection on the calibration split.

    Args:
        cfg:      validated project config.
        artifact: pre-loaded Stage 1 artifact; loaded from disk if ``None``.
        force:    recompute even if calibration JSON already exists.

    Returns:
        Calibration dict (same structure written to JSON).
    """
    model_dir = get_model_dir()
    cal_path = model_dir / "stage2_calibration.json"

    if not force and cal_path.exists():
        print("[calibrate] Loading existing calibration from", cal_path)
        return json.loads(cal_path.read_text())

    stage2_path = get_stage2_model_path(model_dir)

    batch_size = cfg.stage2.batch_size * 2
    max_length = cfg.stage2.max_seq_length
    recall_floor = cfg.stage2.recall_floor
    strategy = cfg.stage2.threshold_strategy

    if artifact is None:
        artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    cal_df = _load_cal_notes(cfg, artifact, splits=build_splits(cfg, artifact=artifact))
    raw_logits, labels_arr, bands = _score_cal_notes(
        cal_df, stage2_path, batch_size, max_length
    )

    calibrators: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    cal_metrics: dict[str, dict] = {}

    print("\n[calibrate] Per-age-group Platt scaling:")
    for band in sorted(set(bands)):
        mask = bands == band
        coef_list, thr, metrics = _process_band(
            raw_logits, labels_arr, mask, strategy, recall_floor
        )
        calibrators[band] = coef_list
        thresholds[band] = thr
        cal_metrics[band] = metrics
        print(
            f"  {band:<8}  n={mask.sum():>5,}  thr={thr:.3f}  "
            f"rec={metrics['recall']:.3f}  pre={metrics['precision']:.3f}  "
            f"F2={metrics['f2']:.3f}  AUROC={metrics['auroc']:.3f}"
        )

    result = {
        "calibrators": calibrators,
        "thresholds": thresholds,
        "strategy": strategy,
        "recall_floor": recall_floor,
        "calibration_metrics": cal_metrics,
    }

    cal_path.write_text(json.dumps(result, indent=2))
    print(f"\n[calibrate] Saved calibration -> {cal_path}")
    return result


def main() -> None:
    """CLI entry point for Stage 2 calibration."""
    parser = argparse.ArgumentParser(description="Calibrate Stage 2 per age group")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if calibration JSON exists")
    args = parser.parse_args()
    _cfg = load_config()
    calibrate(_cfg, force=args.force)


if __name__ == "__main__":
    main()
