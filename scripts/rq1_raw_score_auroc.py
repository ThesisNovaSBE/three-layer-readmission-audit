"""RQ1 sensitivity check: raw vs calibrated scores in the average AUROC comparison.

The headline RQ1 comparison (`comparison_rq1.json`) reports AUROC on the
*calibrated* Stage 1 (isotonic) and Stage 2 (per-age-band Platt) scores. Both
calibrations are monotone within a model but not across age bands (Stage 2's
Platt uses a different slope/intercept per band), so calibration can reorder
cross-band pairs and move a rank metric like AUROC. This script checks how much:
it recovers each model's raw score and recomputes the average AUROC comparison
on the notes-covered test population, raw vs calibrated.

The raw Stage 2 score is not stored, so it is recovered by inverting the Platt
mapping ``p = sigmoid(a*z + b)`` per age band (``z = z1 - z0`` is the raw
Longformer logit): ``z = (logit(p) - b) / a``. Stage 1's raw score
(``stage1_score_raw``, the uncalibrated XGBoost probability) is already in the
results file.

Writes ``models/rq1_raw_score_auroc.json`` (aggregates only; no admission
identifiers). CPU only -- pandas + numpy + sklearn's ``roc_auc_score``, no torch.
Runs in a couple of minutes on a login node (the one non-trivial read is the
processed feature matrix, loaded for the ``age_band`` column only).

Usage::

    python -m scripts.rq1_raw_score_auroc
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_data_dir, get_model_dir, load_config
from src.config_schema import AppConfig
from src.model.bootstrap import bootstrap_ci
from src.model.metrics import auroc
from src.schemas import MODEL_TARGET_COL
from src.stage2._utils import band_key

_N_RESAMPLES = 1000
_SEED = 42
_ROUNDTRIP_TOL = 1e-6        # re-applying Platt must reproduce stage2_score to this
_REPRO_TOL = 1e-4            # S2cal-S1cal must reproduce comparison_rq1 to this
_REQUIRED = ("hadm_id", "subject_id", MODEL_TARGET_COL,
             "stage1_score", "stage1_score_raw", "stage2_score")


def _platt_forward(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt mapping ``p = sigmoid(a*z + b)``."""
    return 1.0 / (1.0 + np.exp(-(a * np.asarray(z, dtype=float) + b)))


def _platt_inverse(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Invert the Platt mapping: ``z = (logit(p) - b) / a``. Requires ``a > 0``."""
    if a <= 0:
        raise ValueError(f"Platt slope a={a} is not > 0; inversion would flip the ranking.")
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return (np.log(p / (1 - p)) - b) / a


def _recover_raw_stage2_logit(
    stage2_score: np.ndarray, band_key_arr: np.ndarray, calibrators: dict
) -> tuple[np.ndarray, float]:
    """Per-age-band Platt inversion -> raw Stage 2 logit. Asserts a > 0 per band
    and that re-applying Platt reproduces ``stage2_score`` (max abs err < 1e-6).
    Returns (raw_logit, max_roundtrip_abs_error)."""
    z = np.empty(len(stage2_score), dtype=float)
    recon = np.empty(len(stage2_score), dtype=float)
    for band in np.unique(band_key_arr):
        if band not in calibrators:
            raise SystemExit(
                f"[rq1_raw] age band {band!r} has no Platt calibrator in "
                "stage2_calibration.json; cannot invert. Re-run `src.stage2.calibrate`."
            )
        a, b = calibrators[band]
        mask = band_key_arr == band
        z[mask] = _platt_inverse(stage2_score[mask], a, b)
        recon[mask] = _platt_forward(z[mask], a, b)
    max_err = float(np.max(np.abs(recon - stage2_score)))
    if max_err >= _ROUNDTRIP_TOL:
        raise SystemExit(
            f"[rq1_raw] Platt round-trip error {max_err:.2e} >= {_ROUNDTRIP_TOL:.0e}; "
            "the inversion does not reproduce stage2_score. Aborting."
        )
    return z, max_err


def _paired_auroc_diff_ci(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, groups: np.ndarray,
    n_resamples: int = _N_RESAMPLES, seed: int = _SEED,
) -> dict:
    """Patient-level paired bootstrap CI for ``AUROC(a) - AUROC(b)`` (same
    resampled admissions score both arms each draw). Same scheme as
    ``src.model.bootstrap`` / ``compare_stages._paired_auroc_diff_ci``, but the
    per-group row positions are precomputed ONCE (pandas Cython groupby) instead
    of an ``np.where`` scan per group per draw. A resample with only one class
    present (AUROC undefined) is skipped, matching the existing bootstraps."""
    y_true, score_a, score_b = np.asarray(y_true), np.asarray(score_a), np.asarray(score_b)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    group_indices = pd.Series(groups).groupby(groups).indices  # precomputed once
    point = float(auroc(y_true, score_a) - auroc(y_true, score_b))
    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        try:
            diffs.append(float(auroc(y_true[idx], score_a[idx])
                               - auroc(y_true[idx], score_b[idx])))
        except ValueError:
            continue  # one class present in this resample -- skip, like bootstrap_ci
    lo = float(np.percentile(diffs, 2.5)) if diffs else float("nan")
    hi = float(np.percentile(diffs, 97.5)) if diffs else float("nan")
    return {
        "point_estimate": round(point, 6),
        "ci_lower": round(lo, 6), "ci_upper": round(hi, 6),
        "excludes_zero": bool(diffs) and (lo > 0 or hi < 0),
        "confidence": 0.95,
        "n_resamples_used": len(diffs), "n_resamples_requested": n_resamples,
    }


def _single_ci(y_true: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict:
    """AUROC + patient-level bootstrap CI via the shared bootstrap_ci."""
    return bootstrap_ci(y_true, score, auroc, groups=groups,
                        n_resamples=_N_RESAMPLES, seed=_SEED)


def _load_age_band(cfg: AppConfig, results: pd.DataFrame) -> pd.Series:
    """Recover ``age_band`` per admission. Prefer an ``age_band`` column already
    in the results; otherwise join it (only that column) from the processed
    feature matrix on ``hadm_id`` -- stage2_results_all.csv does not store it."""
    if "age_band" in results.columns:
        return results["age_band"]
    path = get_data_dir().parent / cfg.output.processed_dir / "features.csv"
    if not path.exists():
        raise SystemExit(
            f"[rq1_raw] {path} not found and stage2_results_all.csv has no age_band "
            "column; cannot select per-band Platt calibrators."
        )
    feat = pd.read_csv(path, usecols=["hadm_id", "age_band"]).drop_duplicates("hadm_id")
    merged = results[["hadm_id"]].merge(feat, on="hadm_id", how="left")
    if merged["age_band"].isna().any():
        n_missing = int(merged["age_band"].isna().sum())
        raise SystemExit(
            f"[rq1_raw] {n_missing} admission(s) in stage2_results_all.csv have no "
            "age_band in the feature matrix; cannot invert Platt for them."
        )
    return merged["age_band"]


def compute_report(cfg: AppConfig, model_dir: Path) -> dict:
    """Load inputs, recover raw scores, and build the aggregates-only report."""
    results = pd.read_csv(model_dir / "stage2_results_all.csv")
    missing = [c for c in _REQUIRED if c not in results.columns]
    if missing:
        raise SystemExit(
            f"[rq1_raw] stage2_results_all.csv is missing column(s) {missing}. "
            "Re-run `python -m src.stage2.predict --all` (the current predict.py "
            "writes stage1_score_raw; an older file predates it)."
        )
    band_arr = _load_age_band(cfg, results).apply(band_key).to_numpy()
    calibrators = json.loads((model_dir / "stage2_calibration.json").read_text())["calibrators"]

    y = results[MODEL_TARGET_COL].to_numpy()
    groups = results["subject_id"].to_numpy()
    s1cal = results["stage1_score"].to_numpy(dtype=float)
    s1raw = results["stage1_score_raw"].to_numpy(dtype=float)
    s2cal = results["stage2_score"].to_numpy(dtype=float)
    s2raw, roundtrip_err = _recover_raw_stage2_logit(s2cal, band_arr, calibrators)

    paired = {
        "s2cal_minus_s1cal": _paired_auroc_diff_ci(y, s2cal, s1cal, groups),
        "s2cal_minus_s1raw": _paired_auroc_diff_ci(y, s2cal, s1raw, groups),
        "s2raw_minus_s1raw": _paired_auroc_diff_ci(y, s2raw, s1raw, groups),
    }
    _assert_reproduces_rq1(model_dir, paired["s2cal_minus_s1cal"]["point_estimate"])

    return {
        "note": (
            "RQ1 sensitivity: average AUROC with raw vs calibrated scores on the "
            "notes-covered test population. Stage 2 raw = Platt-inverted logit "
            "(z1-z0); Stage 1 raw = uncalibrated XGBoost probability. Paired "
            "patient-level bootstrap (1000 resamples, seed 42). Aggregates only; "
            "no admission identifiers."
        ),
        "n": int(len(results)),
        "prevalence": round(float(y.mean()), 6),
        "platt_inversion_max_abs_error": roundtrip_err,
        "auroc": {
            "stage1_calibrated": _single_ci(y, s1cal, groups),
            "stage1_raw": _single_ci(y, s1raw, groups),
            "stage2_calibrated": _single_ci(y, s2cal, groups),
            "stage2_raw": _single_ci(y, s2raw, groups),
        },
        "paired_auroc_diff": paired,
    }


def _assert_reproduces_rq1(model_dir: Path, recomputed: float) -> None:
    """Fail loudly if S2cal-S1cal does not reproduce comparison_rq1.json."""
    path = model_dir / "comparison_rq1.json"
    if not path.exists():
        print(f"[rq1_raw] {path} absent -- skipping the reproduction check.")
        return
    target = json.loads(path.read_text())["auroc_diff_stage2_minus_stage1"]["point_estimate"]
    if abs(recomputed - target) > _REPRO_TOL:
        raise SystemExit(
            f"[rq1_raw] S2cal-S1cal point estimate {recomputed:.6f} does not reproduce "
            f"comparison_rq1.json's {target:.6f} (|diff| > {_REPRO_TOL:.0e}). The "
            "population or scores differ from the committed RQ1 result. Aborting."
        )
    print(f"[rq1_raw] reproduces comparison_rq1 S2cal-S1cal: {recomputed:.6f} "
          f"vs {target:.6f} (ok).")


def _print_report(report: dict) -> None:
    """Human-readable summary to stdout."""
    print(f"\n[rq1_raw] n={report['n']:,}  prevalence={report['prevalence']:.4f}  "
          f"(Platt round-trip max err {report['platt_inversion_max_abs_error']:.2e})")
    print("  AUROC [95% CI]:")
    for name, d in report["auroc"].items():
        print(f"    {name:20s} {d['point_estimate']:.4f} "
              f"[{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]  (n_used={d['n_resamples_used']})")
    print("  Paired AUROC difference [95% CI] (A - B):")
    for name, d in report["paired_auroc_diff"].items():
        print(f"    {name:20s} {d['point_estimate']:+.4f} "
              f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]  "
              f"excludes_zero={d['excludes_zero']}")


def main() -> None:
    """CLI entry point."""
    cfg = load_config()
    model_dir = get_model_dir()
    report = compute_report(cfg, model_dir)
    out_path = model_dir / "rq1_raw_score_auroc.json"
    out_path.write_text(json.dumps(report, indent=2))
    _print_report(report)
    print(f"\n[rq1_raw] Saved -> {out_path}")


if __name__ == "__main__":
    main()
