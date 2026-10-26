"""Selection-bias check: does Stage 1's test partition differ systematically
between admissions with a discharge note (the population RQ1/RQ2 are
evaluated on) and those without (~37%, per docs/ARCHITECTURE.md's open
item -- this table did not exist anywhere in the repo before this script)?

Uses ``has_discharge_note`` (src/data/features.py:build_features), which is
computed directly against the real MIMIC-IV-Note file at feature-build time
-- the authoritative note-coverage flag, not something inferred from which
admissions happen to appear in a downstream results CSV.

Usage::

    python -m scripts.analyze_selection_bias
"""

from __future__ import annotations

import json

import joblib
import numpy as np

from src.config import get_model_dir, load_config
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL_UNPLANNED

_COMPARE_COLS = [
    "age", "gender_M", "los_days", "charlson_index", "n_comorbidities",
    "n_prior_admissions", "n_prior_ed_visits", "days_since_last_discharge",
    "admission_type_emergency", "discharged_to_facility", "came_via_ed",
    "is_medicaid_medicare",
]


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (pooled-SD Cohen's d) between two groups.

    The conventional covariate-balance diagnostic in the causal-inference /
    epidemiology literature (Austin, 2009, Comm. Stat. Simul. Comput.):
    |SMD| >= 0.1 flags a meaningfully imbalanced covariate between groups,
    independent of sample size (unlike a p-value, which is near-guaranteed
    to be significant at n in the tens of thousands even for a trivial
    difference).
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def main() -> None:
    """Compare notes-covered vs. notes-excluded admissions on Stage 1's test partition."""
    cfg = load_config()
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    matrix = load_feature_matrix(cfg, artifact["mode"])
    test = matrix.iloc[artifact["test_idx"]].reset_index(drop=True)

    if test["has_discharge_note"].isna().any():
        n_nan = int(test["has_discharge_note"].isna().sum())
        print(f"[selection_bias] WARNING: {n_nan} test rows have unknown "
              "note status (NaN) -- excluded from both groups below.")
    test = test.dropna(subset=["has_discharge_note"])

    has_note = test["has_discharge_note"] == 1
    covered, excluded = test[has_note], test[~has_note]

    print(f"[selection_bias] Stage 1 test partition: n={len(test):,}")
    print(f"  notes-covered:     n={len(covered):,} ({len(covered) / len(test):.1%})")
    print(f"  notes-excluded:    n={len(excluded):,} ({len(excluded) / len(test):.1%})")

    rows = []
    for col in _COMPARE_COLS:
        a = covered[col].to_numpy(dtype=float)
        b = excluded[col].to_numpy(dtype=float)
        rows.append({
            "feature": col,
            "covered_mean": float(np.nanmean(a)),
            "excluded_mean": float(np.nanmean(b)),
            "smd": _smd(a, b),
        })

    outcome_a = covered[TARGET_COL_UNPLANNED].to_numpy(dtype=float)
    outcome_b = excluded[TARGET_COL_UNPLANNED].to_numpy(dtype=float)
    outcome_smd = _smd(outcome_a, outcome_b)

    print(f"\n  {'Feature':<30} {'Covered':>10} {'Excluded':>10} {'SMD':>8}")
    print("  " + "-" * 62)
    for r in rows:
        flag = "  <-- |SMD|>=0.1" if abs(r["smd"]) >= 0.1 else ""
        print(f"  {r['feature']:<30} {r['covered_mean']:>10.3f} "
              f"{r['excluded_mean']:>10.3f} {r['smd']:>8.3f}{flag}")
    print(f"\n  {'readmission_30d_unplanned':<30} {outcome_a.mean():>10.3f} "
          f"{outcome_b.mean():>10.3f} {outcome_smd:>8.3f}"
          f"{'  <-- |SMD|>=0.1' if abs(outcome_smd) >= 0.1 else ''}")

    report = {
        "n_test": int(len(test)),
        "n_notes_covered": int(len(covered)),
        "n_notes_excluded": int(len(excluded)),
        "notes_coverage_rate": float(len(covered) / len(test)),
        "outcome_rate_covered": float(outcome_a.mean()),
        "outcome_rate_excluded": float(outcome_b.mean()),
        "outcome_smd": outcome_smd,
        "baseline_characteristics": rows,
        "note": (
            "SMD (standardized mean difference, pooled SD) >= 0.1 in absolute "
            "value is the conventional threshold for a meaningfully "
            "imbalanced covariate (Austin 2009). Computed on Stage 1's test "
            "partition (n matches evaluate.py/evaluate_pipeline.py exactly), "
            "split by has_discharge_note (src/data/features.py), which is "
            "computed directly against the real MIMIC-IV-Note file at "
            "feature-build time -- not inferred from a downstream results CSV."
        ),
    }
    out_path = model_dir / "selection_bias_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[selection_bias] Saved -> {out_path}")


if __name__ == "__main__":
    main()
