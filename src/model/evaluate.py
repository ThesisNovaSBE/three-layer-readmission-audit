"""Evaluate a trained Stage 1 model on the held-out test set.

Reports (saved to models/<model>_metrics.json and printed):

- AUPRC (primary), AUROC (secondary), Brier calibration score
- Operating point at the configured recall target
  (recall / precision / specificity / F2 + confusion matrix)
- Precision/specificity trade-off at recall 0.80 / 0.85 / 0.90
- Per-subgroup AUROC (sex, age band) as a fairness check
- Bootstrap 95% CIs (patient-level resampling) on AUROC, AUPRC, and
  operating-point precision/recall

Usage::

    python -m src.model.evaluate
    python -m src.model.evaluate --model logistic_regression
"""

from __future__ import annotations

import argparse
import json

import joblib

from src.config import load_config, get_model_dir
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.bootstrap import bootstrap_ci
from src.model.calibration import apply_calibration
from src.model.metrics import auprc, auroc, full_report


# Published AUROC benchmarks on MIMIC-IV 30-day readmission for context.
# Sources: Rajkomar et al. (2018) Nature Medicine; Xiao et al. (2018) MIMIC-Extract;
# van Walraven et al. (2010) LACE index; Fraccaro et al. (2016) EHR notes baseline.
_PUBLISHED_AUROC = {
    "LACE index (van Walraven 2010)": 0.694,
    "EHR structured baseline (Xiao 2018)": 0.715,
    "Deep EHR (Rajkomar 2018)": 0.773,
    "Clinical notes baseline (Fraccaro 2016)": 0.684,
}


def _print_report(report: dict, name: str) -> None:
    """Print a human-readable evaluation summary to stdout."""
    print("\n" + "=" * 64)
    print(f"  Stage 1 evaluation — {name}  (held-out test set)")
    print("=" * 64)
    print(f"  AUPRC (primary): {report['auprc']:.4f}")
    print(f"  AUROC          : {report['auroc']:.4f}")
    print(f"  Brier (calib.) : {report['brier']:.4f}")
    print(f"  Base rate      : {report['base_rate']:.1%}")

    if "bootstrap_ci" in report:
        ci = report["bootstrap_ci"]
        print(f"\n  Bootstrap 95% CI (patient-level, n={ci['auroc']['n_resamples_used']} "
              f"resamples):")
        print(f"    AUROC:     {ci['auroc']['point_estimate']:.4f} "
              f"[{ci['auroc']['ci_lower']:.4f}, {ci['auroc']['ci_upper']:.4f}]")
        print(f"    AUPRC:     {ci['auprc']['point_estimate']:.4f} "
              f"[{ci['auprc']['ci_lower']:.4f}, {ci['auprc']['ci_upper']:.4f}]")
        print(f"    Precision: {ci['precision']['point_estimate']:.4f} "
              f"[{ci['precision']['ci_lower']:.4f}, {ci['precision']['ci_upper']:.4f}]")
        print(f"    Recall:    {ci['recall']['point_estimate']:.4f} "
              f"[{ci['recall']['ci_lower']:.4f}, {ci['recall']['ci_upper']:.4f}]")

    op = report["operating_point"]
    print(f"\n  Primary operating point ({report.get('threshold_strategy', 'n/a')}, "
          f"threshold={op['threshold']:.4f}):")
    print(
        f"    recall={op['recall']:.3f}  precision={op['precision']:.3f}  "
        f"specificity={op['specificity']:.3f}  F2={op['f2']:.3f}  "
        f"flagged={(op['tp'] + op['fp']) / max(op['tp'] + op['fp'] + op['tn'] + op['fn'], 1):.1%}"
    )
    print(f"    confusion: TP={op['tp']} FP={op['fp']} TN={op['tn']} FN={op['fn']}")

    if "capacity_tradeoff" in report:
        print("\n  Capacity-constrained trade-off (precision@K / lift@K) — primary:")
        for row in report["capacity_tradeoff"]:
            flagged_pct = (row["tp"] + row["fp"]) / max(
                row["tp"] + row["fp"] + row["tn"] + row["fn"], 1
            )
            print(
                f"    K={row['capacity_k']:.0%} -> flagged={flagged_pct:.1%} "
                f"precision={row['precision']:.3f} recall={row['recall']:.3f} "
                f"lift={row['lift']:.2f}x (thr={row['threshold']:.4f})"
            )

    print("\n  Recall-floor trade-off — secondary, for comparability with prior literature:")
    for row in report["recall_tradeoff"]:
        print(
            f"    recall>={row['recall_target']:.2f} -> recall={row['recall']:.3f} "
            f"precision={row['precision']:.3f} specificity={row['specificity']:.3f} "
            f"(thr={row['threshold']:.4f})"
        )

    print("\n  Published AUROC benchmarks (MIMIC-IV / MIMIC-III, 30-day readmission):")
    our_auroc = report["auroc"]
    for ref, val in _PUBLISHED_AUROC.items():
        delta = our_auroc - val
        sign = "+" if delta >= 0 else ""
        print(f"    {ref:<44} {val:.3f}  (ours {sign}{delta:+.3f})")

    print("\n  Subgroup AUROC (fairness check):")
    for col, levels in report["subgroup_auroc"].items():
        print(f"    [{col}]")
        for level, stats in levels.items():
            auroc_str = "n/a " if stats["auroc"] is None else f"{stats['auroc']:.3f}"
            print(
                f"      {level:<8} AUROC={auroc_str}  "
                f"n={stats['n']}  pos_rate={stats['pos_rate']:.1%}"
            )
    print("=" * 64 + "\n")


def _bootstrap_report(y_test, score, threshold, groups_test) -> dict:
    """Return patient-level bootstrap 95% CIs for AUROC, AUPRC, precision, recall.

    Precision/recall are evaluated at the fixed operating-point ``threshold``
    on each resample — a metric_fn closure over the threshold, matching the
    ``bootstrap_ci`` signature of ``(y_true, y_score) -> float``.
    """
    def _precision(yt, ys):
        pred = ys >= threshold
        tp, fp = int((pred & (yt == 1)).sum()), int((pred & (yt == 0)).sum())
        return tp / (tp + fp) if (tp + fp) else 0.0

    def _recall(yt, ys):
        pred = ys >= threshold
        tp, fn = int((pred & (yt == 1)).sum()), int((~pred & (yt == 1)).sum())
        return tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "auroc": bootstrap_ci(y_test, score, auroc, groups=groups_test),
        "auprc": bootstrap_ci(y_test, score, auprc, groups=groups_test),
        "precision": bootstrap_ci(y_test, score, _precision, groups=groups_test),
        "recall": bootstrap_ci(y_test, score, _recall, groups=groups_test),
    }


def evaluate(cfg: AppConfig) -> dict:
    """Evaluate the Stage 1 model on the held-out test set.

    Args:
        cfg: validated project config.

    Returns:
        Evaluation report dict (also written to ``models/<model>_metrics.json``).
    """
    name = cfg.stage1.model
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{name}.joblib")

    matrix = load_feature_matrix(cfg, artifact["mode"])
    features, y, groups, subgroups, _feat = split_xy(matrix)

    test_idx = artifact["test_idx"]
    x_test = features.iloc[test_idx][artifact["feature_cols"]]
    y_test = y[test_idx]
    sub_test = subgroups.iloc[test_idx].reset_index(drop=True)
    groups_test = groups[test_idx]

    raw_score = artifact["estimator"].predict_proba(x_test)[:, 1]
    score = apply_calibration(artifact, raw_score)
    report = full_report(
        y_test, score, artifact["threshold"],
        cfg.stage1.recall_report_points, subgroups=sub_test,
        capacity_points=cfg.stage1.capacity_report_points,
    )

    report["bootstrap_ci"] = _bootstrap_report(
        y_test, score, artifact["threshold"], groups_test
    )
    report["threshold_strategy"] = artifact.get("threshold_strategy", "recall_floor")
    report["published_auroc_benchmarks"] = _PUBLISHED_AUROC
    _print_report(report, name)

    out_path = model_dir / f"{name}_metrics.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[evaluate] Saved metrics -> {out_path}")
    return report


def main() -> None:
    """CLI entry point for Stage 1 evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 model")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "histgradientboosting"],
        default=None,
    )
    args = parser.parse_args()

    _cfg = load_config()
    if args.model:
        _cfg.stage1.model = args.model
    evaluate(_cfg)


if __name__ == "__main__":
    main()
