"""Does Stage 3 actually beat Stage 1 alone within the discordant subgroup
specifically -- not just in aggregate across the whole flagged+noted cohort?

The real `evaluate_pipeline.py` RQ2 result showed that swapping the
*concordant* subgroup's treatment from Stage 3's real decision to "Stage 1's
flag stands" recovered precision and recall in aggregate -- which tells us
Stage 3 was net harmful on concordant cases, but does NOT by itself tell us
whether Stage 3 adds value on discordant cases, since Stage 3's treatment of
the discordant group was never varied in that comparison. This script runs
the one additional comparison needed to answer that directly: within EACH
subgroup separately, Stage 3's real decision vs. Stage 1 alone (always
positive, since every admission here is Stage-1-flagged by construction).

Runs entirely locally -- no cluster/GPU needed. `stage2_results.csv` has the
true label (`readmission_30d_unplanned`) and `subject_id` for this cohort;
`stage3_batch_results.csv` already has `discordance_mode` and
`decision_model` as stored columns from the real batch run. A simple
hadm_id join gives everything needed.

Also writes a full per-admission JSON export (merged stage2 + stage3 data,
true label included) for further ad hoc investigation beyond what this
script itself computes.

Usage::

    python scripts/analyze_discordance_subgroups.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.bootstrap import bootstrap_ci

_MODELS_DIR = Path("models")


def _precision_recall_f(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Same shape as evaluate_pipeline.py's helper of the same name --
    duplicated rather than imported since this is a standalone
    investigative script, not part of the core pipeline."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 5), "recall": round(rec, 5),
        "f1": round(f1, 5), "f2": round(f2, 5),
        "n": int(len(y_true)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _with_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict:
    """_precision_recall_f plus patient-clustered bootstrap CIs (1000
    resamples) on precision and recall -- subgroups here (~5K admissions)
    are small enough that this runs in well under a minute, unlike the
    full 104,242-admission cohort in evaluate_pipeline.py (~2h15m there)."""
    report = _precision_recall_f(y_true, y_pred)
    report["precision_ci"] = bootstrap_ci(
        y_true, y_pred, lambda yt, yp: _precision_recall_f(yt, yp)["precision"], groups=groups
    )
    report["recall_ci"] = bootstrap_ci(
        y_true, y_pred, lambda yt, yp: _precision_recall_f(yt, yp)["recall"], groups=groups
    )
    return report


def _subgroup_comparison(df: pd.DataFrame) -> dict:
    """Stage 3's real decision vs. Stage 1 alone (always positive), within
    one discordance subgroup."""
    y_true = df["readmission_30d_unplanned"].to_numpy()
    groups = df["subject_id"].to_numpy()
    stage3_pred = (df["decision_model"] == "uphold").astype(int).to_numpy()
    stage1_pred = np.ones_like(stage3_pred)
    return {
        "n": int(len(df)),
        "stage1_alone": _with_ci(y_true, stage1_pred, groups),
        "stage3_decision": _with_ci(y_true, stage3_pred, groups),
    }


def main() -> None:
    """Load, merge, split by discordance, compare, export."""
    s2 = pd.read_csv(_MODELS_DIR / "stage2_results.csv")
    s3 = pd.read_csv(_MODELS_DIR / "stage3_batch_results.csv")
    merged = s2.merge(s3, on="hadm_id", how="inner", suffixes=("", "_s3"))
    print(f"Merged cohort: {len(merged):,} admissions "
          f"(stage2={len(s2):,}, stage3={len(s3):,})")

    has_decision = merged["decision_model"].isin(["uphold", "override"])
    n_excluded = int((~has_decision).sum())
    scored = merged[has_decision].copy()
    print(f"Excluded (no valid Stage 3 decision -- failed/insufficient_evidence): "
          f"{n_excluded:,}")

    concordant = scored[scored["discordance_mode"] == "CONCORDANT"]
    discordant = scored[scored["discordance_mode"] != "CONCORDANT"]
    print(f"Concordant: {len(concordant):,}  |  Discordant: {len(discordant):,}")

    print("\nComputing concordant subgroup (bootstrap CIs, ~1 min)...")
    concordant_result = _subgroup_comparison(concordant)
    print("Computing discordant subgroup (bootstrap CIs, ~1 min)...")
    discordant_result = _subgroup_comparison(discordant)

    result = {
        "n_total_cohort": int(len(merged)),
        "n_excluded_no_decision": n_excluded,
        "concordant": concordant_result,
        "discordant": discordant_result,
    }
    out_path = _MODELS_DIR / "discordance_subgroup_evaluation.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")

    for label, res in (("CONCORDANT", concordant_result), ("DISCORDANT", discordant_result)):
        s1, s3r = res["stage1_alone"], res["stage3_decision"]
        print(f"\n=== {label} (n={res['n']:,}) ===")
        print(f"  Stage 1 alone : precision={s1['precision']:.3f} "
              f"[{s1['precision_ci']['ci_lower']:.3f}, {s1['precision_ci']['ci_upper']:.3f}]  "
              f"recall={s1['recall']:.3f} "
              f"[{s1['recall_ci']['ci_lower']:.3f}, {s1['recall_ci']['ci_upper']:.3f}]")
        print(f"  Stage 3       : precision={s3r['precision']:.3f} "
              f"[{s3r['precision_ci']['ci_lower']:.3f}, {s3r['precision_ci']['ci_upper']:.3f}]  "
              f"recall={s3r['recall']:.3f} "
              f"[{s3r['recall_ci']['ci_lower']:.3f}, {s3r['recall_ci']['ci_upper']:.3f}]")

    # Full per-admission export for further ad hoc investigation --
    # everything from both source files, plus the true label, in one file.
    grounds_cols = ["mitigating_grounds", "aggravating_grounds"]
    export_df = merged.copy()
    for col in grounds_cols:
        export_df[col] = export_df[col].apply(
            lambda v: json.loads(v) if isinstance(v, str) and v else []
        )
    export_path = _MODELS_DIR / "stage3_full_evaluation_export.json"
    export_df.to_json(export_path, orient="records", indent=2)
    print(f"Saved per-admission export ({len(export_df):,} rows) -> {export_path}")


if __name__ == "__main__":
    main()
