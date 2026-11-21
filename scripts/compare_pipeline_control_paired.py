"""Patient-level PAIRED bootstrap of the pipeline-minus-control difference in
precision and recall (RQ2).

``pipeline_evaluation.json`` reports each policy and its matched Stage-1
control arm with its own (independent) bootstrap CI. Two independent CIs that
overlap do not settle whether the *difference* is real -- the paired bootstrap
does: it resamples the same patients for both arms in every draw and takes the
difference inside the draw, so the shared-patient covariance is kept and the
CI is the correct one for "did the cascade help or hurt at matched volume?".
Same resampling scheme as the existing bootstraps (src/model/bootstrap.py and
src/model/compare_stages.py:_paired_auroc_diff_ci): patient clusters
(``subject_id``), ``np.random.default_rng(42)``, 1,000 resamples.

Three comparisons, each pipeline-arm minus its volume-matched Stage-1 control:

* primary     -- full-cohort pipeline vs its matched control;
* secondary   -- the both-override withdrawal policy vs its matched control;
* conditional -- discordant-only triggering vs a Stage-1 control matched to
                 ITS alert volume (built exactly like the other control arms,
                 top-m by raw Stage-1 score).

Writes ``models/pipeline_paired_differences.json`` (aggregates only: point
differences, CIs, counts -- no admission identifiers, no per-patient data) and
prints a short table.

Reconstructs the per-admission predictions by importing
``src.model.evaluate_pipeline`` unchanged; CPU only (XGBoost ``predict_proba``
plus CSV reads, no torch/LLM). Runs where the data lives (the cluster).

Usage::

    python -m scripts.compare_pipeline_control_paired
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.model.evaluate_pipeline import (
    _apply_stage3_decisions,
    _conditional_predictions,
    _eval_stage2,
    _load_test_partition,
    _precision_metric,
    _recall_metric,
    apply_calibration,
)

_N_RESAMPLES = 1000
_SEED = 42
_CROSSCHECK_TOL = 5e-4  # recomputed point P/R must match pipeline_evaluation.json


def _paired_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = _N_RESAMPLES,
    seed: int = _SEED,
) -> dict:
    """Patient-level paired bootstrap CI for ``metric(a) - metric(b)``.

    The same resampled admissions score both arms in each draw (paired), the
    same scheme as :func:`src.model.compare_stages._paired_auroc_diff_ci`:
    ``default_rng(seed)``, resample unique ``groups`` with replacement, gather
    their row positions, evaluate the difference on that resample.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_indices = pd.Series(groups).groupby(groups).indices  # precomputed, == scheme

    point = float(metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b))
    diffs: list[float] = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        diffs.append(float(metric_fn(y_true[idx], pred_a[idx])
                           - metric_fn(y_true[idx], pred_b[idx])))

    lo = float(np.percentile(diffs, 2.5)) if diffs else float("nan")
    hi = float(np.percentile(diffs, 97.5)) if diffs else float("nan")
    return {
        "point_estimate": round(point, 5),
        "ci_lower": round(lo, 5),
        "ci_upper": round(hi, 5),
        "excludes_zero": bool(diffs) and (lo > 0 or hi < 0),
        "confidence": 0.95,
        "n_resamples_used": len(diffs),
        "n_resamples_requested": n_resamples,
    }


def _top_m_by_raw(s1_scores_raw: np.ndarray, target_rate: float) -> np.ndarray:
    """Stage-1-alone control-arm prediction: flag the top
    ``m = round(target_rate * n)`` admissions by RAW Stage-1 score (stable
    descending argsort). Identical construction to
    :func:`src.model.evaluate_pipeline._control_arm_report`'s primary arm."""
    raw = np.asarray(s1_scores_raw, dtype=float)
    n = len(raw)
    m = round(target_rate * n)
    pred = np.zeros(n, dtype=int)
    if m > 0:
        order = np.argsort(-raw, kind="stable")
        pred[order[:m]] = 1
    return pred


class _Policies(NamedTuple):
    """Aligned per-admission arrays for the test partition: labels, patient
    groups, raw Stage-1 scores, and the six prediction arms the paired
    bootstrap compares (three pipeline policies + their matched controls)."""

    y_true: np.ndarray
    groups: np.ndarray
    s1_scores_raw: np.ndarray
    pipeline: np.ndarray
    control: np.ndarray
    secondary: np.ndarray
    secondary_control: np.ndarray
    conditional: np.ndarray
    conditional_control: np.ndarray


def _read_stage3_decisions(
    model_dir: Path, hadm_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(decision_model, decision_rule, discordance_mode) arrays aligned to
    ``hadm_test`` -- read from stage3_batch_results.csv the same way
    evaluate_pipeline's ``_both_agree_report`` / ``_conditional_triggering_report``
    do (ids used only for the join; never emitted)."""
    s3 = pd.read_csv(model_dir / "stage3_batch_results.csv").set_index("hadm_id")

    def col(name: str) -> pd.Series:
        return s3[name] if name in s3.columns else pd.Series(dtype=object)

    dec, rule, mode = col("decision_model"), col("decision_rule"), col("discordance_mode")
    return (
        np.array([dec.get(h, None) for h in hadm_test], dtype=object),
        np.array([rule.get(h, None) for h in hadm_test], dtype=object),
        np.array([mode.get(h, None) for h in hadm_test], dtype=object),
    )


def reconstruct_policies(cfg: AppConfig) -> _Policies:
    """Rebuild the six prediction arms exactly as ``evaluate_pipeline()`` does,
    importing its helpers without modification."""
    model_dir = get_model_dir()
    import joblib  # noqa: PLC0415  pylint: disable=import-outside-toplevel
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    _x, y_test, _sub, hadm_test, groups = _load_test_partition(artifact, cfg)
    s1_raw = artifact["estimator"].predict_proba(_x)[:, 1]
    s1_cal = apply_calibration(artifact, s1_raw)
    _s2, pipeline_pred_full, _has_s2, flagged = _eval_stage2(
        model_dir, hadm_test, y_test, _sub, s1_cal, artifact["threshold"]
    )
    pipeline_pred, _info = _apply_stage3_decisions(
        model_dir, hadm_test, pipeline_pred_full, flagged
    )

    dec_arr, rule_arr, mode_arr = _read_stage3_decisions(model_dir, hadm_test)
    has_s3 = np.isin(dec_arr, ["uphold", "override"])
    both_override = (dec_arr == "override") & (rule_arr == "override")
    secondary = np.where(flagged, np.where(has_s3, (~both_override).astype(int),
                                           pipeline_pred), 0)
    conditional = _conditional_predictions(mode_arr, pipeline_pred, flagged)

    n = len(y_test)
    return _Policies(
        y_true=np.asarray(y_test), groups=np.asarray(groups), s1_scores_raw=s1_raw,
        pipeline=pipeline_pred,
        control=_top_m_by_raw(s1_raw, pipeline_pred.sum() / n),
        secondary=secondary,
        secondary_control=_top_m_by_raw(s1_raw, secondary.sum() / n),
        conditional=conditional,
        conditional_control=_top_m_by_raw(s1_raw, conditional.sum() / n),
    )


def _crosscheck(pol: _Policies, model_dir: Path) -> None:
    """Fail loudly if a reconstructed arm's point precision/recall disagrees
    with pipeline_evaluation.json -- proves the reconstruction matches the
    committed report before any difference is reported."""
    path = model_dir / "pipeline_evaluation.json"
    if not path.exists():
        print("[paired] pipeline_evaluation.json absent -- skipping cross-check.")
        return
    pipe = json.loads(path.read_text())["pipeline"]
    want = {
        "pipeline": (pol.pipeline, pipe["full_cohort"]),
        "control": (pol.control, pipe["control_arm_stage1_matched"]),
        "secondary": (pol.secondary, pipe["secondary_both_agree"]["report"]),
        "secondary_control": (pol.secondary_control,
                              pipe["secondary_both_agree"]["control_arm_stage1_matched"]),
        "conditional": (pol.conditional, pipe["conditional_triggering"]["report"]),
    }
    for name, (pred, ref) in want.items():
        for metric_fn, key in ((_precision_metric, "precision"), (_recall_metric, "recall")):
            got = metric_fn(pol.y_true, pred)
            if abs(got - ref[key]) > _CROSSCHECK_TOL:
                raise SystemExit(
                    f"[paired] reconstruction mismatch on {name}.{key}: "
                    f"recomputed {got:.5f} vs pipeline_evaluation.json {ref[key]:.5f} "
                    f"(tol {_CROSSCHECK_TOL}). Not reporting a difference built on a "
                    "reconstruction that doesn't match the committed report."
                )
    print("[paired] cross-check OK -- reconstructed arms match pipeline_evaluation.json.")


def compute_paired_differences(pol: _Policies) -> dict:
    """The three paired pipeline-minus-control difference CIs (precision and
    recall each), as an aggregates-only report dict."""
    pairs = {
        "primary_full_vs_control": (pol.pipeline, pol.control),
        "secondary_both_override_vs_control": (pol.secondary, pol.secondary_control),
        "conditional_discordant_only_vs_control": (pol.conditional, pol.conditional_control),
    }
    out: dict = {
        "note": (
            "Paired patient-level bootstrap of (pipeline policy - matched Stage 1 "
            "control) in precision and recall. Positive => the pipeline arm is "
            "higher; 'excludes_zero' true => the 95% CI does not contain 0. "
            "Aggregates only; no admission identifiers."
        ),
        "n_resamples": _N_RESAMPLES, "seed": _SEED, "confidence": 0.95,
        "comparisons": {},
    }
    for name, (arm, control) in pairs.items():
        out["comparisons"][name] = {
            "n_admissions": int(len(pol.y_true)),
            "control_alert_count": int(control.sum()),
            "precision_diff": _paired_diff_ci(pol.y_true, arm, control, pol.groups,
                                              _precision_metric),
            "recall_diff": _paired_diff_ci(pol.y_true, arm, control, pol.groups,
                                           _recall_metric),
        }
    return out


def _print_table(report: dict) -> None:
    """Short console table of the paired differences."""
    print("\n  Comparison (pipeline - control)        metric      diff     95% CI        excl0")
    print("  " + "-" * 78)
    for name, comp in report["comparisons"].items():
        for metric in ("precision_diff", "recall_diff"):
            d = comp[metric]
            print(f"  {name[:36]:36s} {metric[:-5]:9s} "
                  f"{d['point_estimate']:+.4f}  "
                  f"[{d['ci_lower']:+.4f},{d['ci_upper']:+.4f}]  "
                  f"{'yes' if d['excludes_zero'] else 'no'}")


def main() -> None:
    """CLI entry point."""
    cfg = load_config()
    model_dir = get_model_dir()
    pol = reconstruct_policies(cfg)
    _crosscheck(pol, model_dir)
    report = compute_paired_differences(pol)
    out_path = model_dir / "pipeline_paired_differences.json"
    out_path.write_text(json.dumps(report, indent=2))
    _print_table(report)
    print(f"\n[paired] Saved -> {out_path}")


if __name__ == "__main__":
    main()
