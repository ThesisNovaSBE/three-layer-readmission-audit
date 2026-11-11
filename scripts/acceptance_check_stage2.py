"""CPU acceptance check for the corrected Stage 2 calibration.

Run this after the Stage 2 job (slurm_stage2_finish.sh) finishes, before
committing to the six-day Stage 3 rerun. Pure CPU, no GPU, seconds to run.

Reads ``models/stage2_results_all.csv`` -- the population-wide test
admissions with a note (``predict --all``), NOT the Stage 1-flagged file.
This matters: the flagged file is selected by Stage 1's own score, so its
recall/calibration are conditioned on a high-risk subset and are not a fair
check of whether Stage 2 is calibrated on the test distribution. The
population-wide file is.

Per age band, reports:
  1. recall at the band's decision threshold vs. the calibration-set recall
     stored in ``stage2_calibration.json`` -- the headline acceptance check:
     after the fit/apply fix these should be close (the old bug made the
     flagged-group recall ~0.974 while the calibration set saw 0.80-0.89).
  2. mean calibrated probability vs. observed readmission rate
     (calibration-in-the-large) -- a well-calibrated band has these close.
  3. Expected Calibration Error (10 equal-width bins) -- should be well
     below the pre-fix 0.196.

Usage::

    python -m scripts.acceptance_check_stage2
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import get_model_dir, load_config
from src.data.features import load_feature_matrix
from src.schemas import MODEL_TARGET_COL as TARGET_COL
from src.stage2._utils import band_key


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error, 10 equal-width bins -- same definition as
    src/stage2/evaluate.py's helper, duplicated here to keep this a
    standalone diagnostic script."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    if total == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        frac_pos = labels[mask].mean()
        mean_prob = probs[mask].mean()
        ece += mask.sum() / total * abs(mean_prob - frac_pos)
    return float(ece)


def _recall(confirmed: np.ndarray, labels: np.ndarray) -> float:
    """Recall of a binary decision vector against true labels."""
    tp = int(((confirmed == 1) & (labels == 1)).sum())
    fn = int(((confirmed == 0) & (labels == 1)).sum())
    return tp / (tp + fn) if (tp + fn) > 0 else float("nan")


def _load_scored_with_bands(cfg, model_dir) -> tuple[pd.DataFrame, dict]:
    """Load the population-wide results CSV, attach each admission's age band
    from the feature matrix, and return (df, calibration dict)."""
    results_path = model_dir / "stage2_results_all.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run `python -m src.stage2.predict --all` "
            "first (slurm_stage2_finish.sh does this)."
        )
    df = pd.read_csv(results_path)
    print(f"[acceptance] Loaded {len(df):,} population-wide test admissions "
          f"with a note from {results_path.name}")

    calibration = json.loads((model_dir / "stage2_calibration.json").read_text())

    # Take the mode from the Stage 1 artifact, exactly as src/stage2/evaluate.py
    # does, so this loads the same matrix the real run used (cfg.run.mode is
    # the local "quick" default and would load the wrong matrix).
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")
    matrix = load_feature_matrix(cfg, artifact["mode"])
    age_df = matrix[["hadm_id", "age_band"]].copy()
    age_df["age_band_key"] = age_df["age_band"].apply(band_key)
    df = df.merge(age_df[["hadm_id", "age_band_key"]], on="hadm_id", how="left")
    missing = df["age_band_key"].isna().sum()
    if missing:
        print(f"[acceptance] WARNING: {missing} rows missing age_band -- excluded")
        df = df.dropna(subset=["age_band_key"])
    return df, calibration


def _band_metrics(sub: pd.DataFrame, thr: float, recall_cal: float) -> dict:
    """Per-band acceptance metrics: recall at the band threshold vs. the
    calibration-set recall, mean calibrated prob vs. observed rate, and ECE."""
    probs = sub["stage2_score"].to_numpy()
    labels = sub[TARGET_COL].to_numpy()
    recall_test = _recall((probs >= thr).astype(int), labels)
    return {
        "n": int(len(sub)),
        "threshold": float(thr),
        "recall_test_population": recall_test,
        "recall_calibration_set": recall_cal,
        "recall_gap": recall_test - recall_cal,
        "mean_calibrated_prob": float(probs.mean()),
        "observed_readmission_rate": float(labels.mean()),
        "ece": _ece(probs, labels),
    }


def main() -> None:
    """Load population-wide Stage 2 results, attach age bands, and report the
    per-band acceptance metrics against the stored calibration."""
    cfg = load_config()
    model_dir = get_model_dir()
    df, calibration = _load_scored_with_bands(cfg, model_dir)
    cal_thresholds = calibration["thresholds"]
    cal_metrics = calibration["calibration_metrics"]

    report: dict = {"n_total": int(len(df)), "bands": {}}

    print(f"\n{'Band':<8} {'N':>7} {'thr':>6} "
          f"{'recall_test':>12} {'recall_cal':>11} {'Δrecall':>8} "
          f"{'mean_prob':>10} {'obs_rate':>9} {'ECE':>7}")
    print("-" * 92)

    for band in sorted(df["age_band_key"].unique()):
        sub = df[df["age_band_key"] == band]
        thr = cal_thresholds.get(band)
        if thr is None:
            print(f"{band:<8} {len(sub):>7,}  (no threshold in calibration -- skipped)")
            continue
        recall_cal = cal_metrics.get(band, {}).get("recall", float("nan"))
        m = _band_metrics(sub, thr, recall_cal)
        report["bands"][band] = m
        print(f"{band:<8} {m['n']:>7,} {thr:>6.3f} "
              f"{m['recall_test_population']:>12.3f} {recall_cal:>11.3f} {m['recall_gap']:>+8.3f} "
              f"{m['mean_calibrated_prob']:>10.3f} {m['observed_readmission_rate']:>9.3f} "
              f"{m['ece']:>7.4f}")

    overall_probs = df["stage2_score"].to_numpy()
    overall_labels = df[TARGET_COL].to_numpy()
    overall_ece = _ece(overall_probs, overall_labels)
    report["overall"] = {
        "n": int(len(df)),
        "mean_calibrated_prob": float(overall_probs.mean()),
        "observed_readmission_rate": float(overall_labels.mean()),
        "ece": overall_ece,
    }
    print("-" * 92)
    print(f"{'OVERALL':<8} {len(df):>7,} {'':>6} {'':>12} {'':>11} {'':>8} "
          f"{overall_probs.mean():>10.3f} {overall_labels.mean():>9.3f} {overall_ece:>7.4f}")

    out_path = model_dir / "stage2_acceptance_check.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[acceptance] Saved -> {out_path}")
    print("[acceptance] PASS criteria: per-band recall_test close to recall_cal "
          "(not ~0.97 vs 0.80-0.89), mean_prob close to obs_rate, and ECE well "
          "below the pre-fix 0.196. If recall_test is far above recall_cal or "
          "ECE stays high, the calibration is still wrong -- do not start Stage 3.")


if __name__ == "__main__":
    main()
