"""RQ1: Structured (Layer 1) vs. notes-only (Layer 2), head-to-head.

Answers "does narrative text carry signal structured data doesn't" by
scoring both models on the *same* population and comparing directly —
neither model gates or feeds the other here (that's the separate cascade
story in ``evaluate_pipeline.py`` / Layer 3).

Stage 2 only ever covers admissions with a discharge note (~63% of Stage 1's
test partition in past runs), so the only population both models can be
fairly compared on is that notes-covered subset. Restricting the comparison
to it, rather than reporting each model on its own natural population, is a
deliberate methodological choice — see docs/ARCHITECTURE.md and the
BLOCKED-ON-DECISION note this module's docstring exists to make concrete:
until that choice is explicitly confirmed, treat ``stage1_notes_cohort`` vs
``stage2_notes_cohort`` as the candidate headline RQ1 comparison and
``stage1_full_population`` as supporting context, not the other way round.

Requires ``models/stage2_results_all.csv`` (population-wide Stage 2 scores —
run ``python -m src.stage2.predict --all`` first; see src/stage2/predict.py).

Usage::

    python -m src.model.compare_layers
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.bootstrap import bootstrap_ci
from src.model.calibration import apply_calibration
from src.model.metrics import (
    auprc,
    auroc,
    operating_point,
    select_threshold_for_capacity,
)


def _load_stage1_test(artifact: dict, cfg: AppConfig) -> pd.DataFrame:
    """Return Stage 1's test-partition scores as a flat, mergeable DataFrame."""
    matrix = load_feature_matrix(cfg, artifact["mode"])
    features, y, groups, _subgroups, _feat = split_xy(matrix)
    idx = artifact["test_idx"]
    x_test = features.iloc[idx][artifact["feature_cols"]]
    raw = artifact["estimator"].predict_proba(x_test)[:, 1]
    stage1_score = apply_calibration(artifact, raw)
    return pd.DataFrame({
        "hadm_id": matrix.iloc[idx]["hadm_id"].to_numpy(),
        "subject_id": groups[idx],
        "readmission_30d_unplanned": y[idx],
        "stage1_score": stage1_score,
    })


def _metrics_block(
    y_true: np.ndarray, y_score: np.ndarray, groups: np.ndarray, threshold: float
) -> dict:
    """AUROC/AUPRC with patient-level bootstrap CIs, plus one operating point."""
    return {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "auroc": bootstrap_ci(y_true, y_score, auroc, groups=groups),
        "auprc": bootstrap_ci(y_true, y_score, auprc, groups=groups),
        "operating_point": operating_point(y_true, y_score, threshold),
    }


def _paired_auroc_diff_ci(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Patient-level paired bootstrap CI for AUROC(a) - AUROC(b).

    Paired (same resampled indices score both models each draw) rather than
    two independent CIs subtracted — a paired CI is narrower and correct
    here because both models are being compared on literally the same
    admissions each resample, not independent samples.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    point = float(auroc(y_true, score_a) - auroc(y_true, score_b))

    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        try:
            diffs.append(float(auroc(y_true[idx], score_a[idx]) - auroc(y_true[idx], score_b[idx])))
        except ValueError:
            continue

    if diffs:
        ci_lower = float(np.percentile(diffs, 2.5))
        ci_upper = float(np.percentile(diffs, 97.5))
    else:
        ci_lower = ci_upper = float("nan")
    return {
        "point_estimate": point, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "confidence": 0.95, "n_resamples_used": len(diffs),
        "n_resamples_requested": n_resamples,
    }


def compare_layers(cfg: AppConfig) -> dict:
    """Compare Stage 1 (structured) against Stage 2 (notes-only) on the same population.

    Loads Stage 1's test-partition scores and Stage 2's population-wide
    scores, inner-joins on ``hadm_id``, and reports both models' metrics on
    the joined (notes-covered) population plus Stage 1's own full-test-set
    numbers as context. Saves ``models/comparison_rq1.json``.
    """
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")
    stage1_df = _load_stage1_test(artifact, cfg)

    stage2_path = model_dir / "stage2_results_all.csv"
    if not stage2_path.exists():
        raise FileNotFoundError(
            f"{stage2_path} not found. Run `python -m src.stage2.predict --all` first."
        )
    stage2_df = pd.read_csv(stage2_path)[["hadm_id", "stage2_score"]]

    joined = stage1_df.merge(stage2_df, on="hadm_id", how="inner")
    coverage = len(joined) / max(len(stage1_df), 1)
    print(f"[compare_layers] Joined (notes-covered) population: {len(joined):,} / "
          f"{len(stage1_df):,} Stage 1 test admissions ({coverage:.1%} note coverage)")

    y = joined["readmission_30d_unplanned"].to_numpy()
    groups = joined["subject_id"].to_numpy()
    s1_scores = joined["stage1_score"].to_numpy()
    s2_scores = joined["stage2_score"].to_numpy()

    # Match Stage 2's operating point to Stage 1's alert volume within this
    # cohort ("at equal alert volume") rather than an arbitrary threshold —
    # Stage 2 has no canonical standalone threshold outside the cascade.
    stage1_flag_rate = float((s1_scores >= artifact["threshold"]).mean())
    stage2_threshold = select_threshold_for_capacity(s2_scores, max(stage1_flag_rate, 1e-6))

    stage1_notes_cohort = _metrics_block(y, s1_scores, groups, artifact["threshold"])
    stage2_notes_cohort = _metrics_block(y, s2_scores, groups, stage2_threshold)

    y_full = stage1_df["readmission_30d_unplanned"].to_numpy()
    stage1_full_population = _metrics_block(
        y_full, stage1_df["stage1_score"].to_numpy(),
        stage1_df["subject_id"].to_numpy(), artifact["threshold"],
    )

    auroc_diff = _paired_auroc_diff_ci(y, s2_scores, s1_scores, groups)

    report = {
        "note": (
            "RQ1: Stage 1 (structured) vs. Stage 2 (notes-only), both scored "
            "on the notes-covered subset of the Stage 1 test partition -- the "
            "only population both models can be fairly compared on. "
            "stage1_full_population is deployment context, not (yet confirmed "
            "as) the headline RQ1 number -- see docs/ARCHITECTURE.md."
        ),
        "note_coverage_of_test_partition": coverage,
        "stage1_notes_cohort": stage1_notes_cohort,
        "stage2_notes_cohort": stage2_notes_cohort,
        "stage1_full_population": stage1_full_population,
        "auroc_diff_stage2_minus_stage1": auroc_diff,
    }

    out_path = model_dir / "comparison_rq1.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[compare_layers] Saved -> {out_path}")
    print(
        f"[compare_layers] Stage 1 AUROC={stage1_notes_cohort['auroc']['point_estimate']:.4f}  "
        f"Stage 2 AUROC={stage2_notes_cohort['auroc']['point_estimate']:.4f}  "
        f"diff={auroc_diff['point_estimate']:+.4f} "
        f"[{auroc_diff['ci_lower']:+.4f}, {auroc_diff['ci_upper']:+.4f}]"
    )
    return report


def main() -> None:
    """CLI entry point for the RQ1 comparison."""
    cfg = load_config()
    compare_layers(cfg)


if __name__ == "__main__":
    main()
