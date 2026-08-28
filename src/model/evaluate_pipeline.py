"""End-to-end pipeline evaluation on the Stage 1 held-out test set.

Both Stage 1 and Stage 2 models are evaluated on patients that neither model
has ever seen:

- Stage 1 is evaluated on ``artifact["test_idx"]`` — its standard held-out split.
- Stage 2 scores for those same patients come from ``stage2_results.csv``.
  Because Stage 2 was fine-tuned only on Stage 1's *training* partition, any
  patient in Stage 1's test partition is also out-of-sample for Stage 2.

The script reports four evaluation layers:

1. **Stage 1 alone** — AUROC / AUPRC / recall / precision on the test partition.
2. **Stage 2 alone** — AUROC / recall / precision on the subset of test-partition
   patients that were Stage 1-flagged *and* had a discharge note (the Stage 2
   inference population).
3. **Full pipeline** — the final binary prediction for all test-partition
   patients. Per docs/ARCHITECTURE.md this should be Stage 3's uphold/override
   decision (see ``src/stage3/batch.py``) wherever a batch Stage 3 run covers
   the admission; where it doesn't yet, falls back to "Stage 2 confirmed"
   with the C9 Stage-1 fallback for note-less admissions. Reports
   pipeline-level precision / recall / F1 / F2.
4. **Control arm** — Stage 1 alone, thresholded to match the full pipeline's
   own alert volume (layer 3 above). The load-bearing comparison for RQ2:
   without it there is no answer to whether the cascade + audit caught more
   than simply tightening Stage 1's threshold to the same alert budget would
   have. Always report alongside the full pipeline, never omit.

Per-age-band breakdowns are included for all four layers.

Output: ``models/pipeline_evaluation.json`` (printed summary + JSON).

Usage::

    python -m src.model.evaluate_pipeline
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import get_model_dir, load_config
from src.config_schema import AppConfig
from src.data.features import load_feature_matrix, split_xy
from src.model.calibration import apply_calibration
from src.model.metrics import auprc as compute_auprc
from src.model.metrics import select_threshold_for_capacity


# ── helpers ───────────────────────────────────────────────────────────────────

def _precision_recall_f(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return precision, recall, F1, F2 from binary arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2   = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 5), "recall": round(rec, 5),
        "f1": round(f1, 5), "f2": round(f2, 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _band_breakdown(
    y_true: np.ndarray,
    scores: np.ndarray | None,
    pred: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Per-age-band metrics breakdown."""
    bands: dict = {}
    for band in sorted(subgroups["age_band"].dropna().unique(), key=str):
        mask = (subgroups["age_band"] == band).values
        yt = y_true[mask]
        yp = pred[mask]
        entry: dict = {"n": int(mask.sum()), "pos_rate": float(yt.mean())}
        if scores is not None and len(np.unique(yt)) > 1:
            entry["auroc"] = float(roc_auc_score(yt, scores[mask]))
        entry.update(_precision_recall_f(yt, yp))
        bands[str(band)] = entry
    return bands


def _stage1_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    subgroups: pd.DataFrame,
) -> dict:
    """Stage 1 evaluation report dict."""
    pred = (scores >= threshold).astype(int)
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(compute_auprc(y_true, scores)),
        "threshold": float(threshold),
    }
    report.update(_precision_recall_f(y_true, pred))
    report["by_age_band"] = _band_breakdown(y_true, scores, pred, subgroups)
    return report


def _stage2_report(
    y_true: np.ndarray,
    s2_scores: np.ndarray,
    s2_confirmed: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Stage 2 evaluation report dict (only Stage 1-flagged patients with notes)."""
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        report["auroc"] = float(roc_auc_score(y_true, s2_scores))
        report["auprc"] = float(compute_auprc(y_true, s2_scores))
    else:
        report["auroc"] = None
        report["auprc"] = None
    report.update(_precision_recall_f(y_true, s2_confirmed))
    report["by_age_band"] = _band_breakdown(y_true, s2_scores, s2_confirmed, subgroups)
    return report


def _pipeline_report(
    y_true: np.ndarray,
    pipeline_pred: np.ndarray,
    subgroups: pd.DataFrame,
) -> dict:
    """Pipeline-level evaluation treating Stage 2 confirmed as the final label."""
    report: dict = {
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
        "confirmed_rate": float(pipeline_pred.mean()),
    }
    report.update(_precision_recall_f(y_true, pipeline_pred))
    report["by_age_band"] = _band_breakdown(y_true, None, pipeline_pred, subgroups)
    return report


def _control_arm_report(
    y_true: np.ndarray,
    s1_scores: np.ndarray,
    target_alert_rate: float,
    subgroups: pd.DataFrame,
) -> dict:
    """Stage 1 alone, thresholded to match the full system's alert volume.

    The load-bearing comparison for RQ2 (colleague review, 2026-08-27):
    without this, there is no answer to "did the cascade + audit add
    anything, or would tightening Stage 1's own threshold to the same alert
    budget have caught as much?" Distinct from compare_layers.py's matched-
    capacity operating point, which matches Stage 2 to Stage 1's *pre-audit*
    flag rate for RQ1 -- this matches Stage 1 to the whole system's
    *post-audit* rate (Layer 3's decision, not Layer 2's threshold) for RQ2.

    Args:
        y_true:             ground-truth labels for the test partition.
        s1_scores:           Stage 1 scores for the same population.
        target_alert_rate:  the full system's final alert rate to match
                             (``pipeline.full_cohort.confirmed_rate``).
        subgroups:           age-band subgroup dataframe, aligned to y_true.

    Returns:
        A `_pipeline_report`-shaped dict, plus ``threshold`` and
        ``target_alert_rate``, directly comparable to
        ``pipeline.full_cohort``.
    """
    threshold = select_threshold_for_capacity(s1_scores, max(target_alert_rate, 1e-6))
    pred = (s1_scores >= threshold).astype(int)
    report = _pipeline_report(y_true, pred, subgroups)
    report["threshold"] = float(threshold)
    report["target_alert_rate"] = float(target_alert_rate)
    return report


def _load_test_partition(
    artifact: dict, cfg: AppConfig
) -> tuple[object, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load Stage 1 test partition features, labels, subgroups and hadm_ids.

    Returns:
        (x_test, y_test, sub_test, hadm_test)
    """
    matrix = load_feature_matrix(cfg, artifact["mode"])
    features, y, _, subgroups, _ = split_xy(matrix)
    idx = artifact["test_idx"]
    x_test = features.iloc[idx][artifact["feature_cols"]]
    y_test = y[idx]
    sub_test = subgroups.iloc[idx].reset_index(drop=True)
    hadm_test = matrix.iloc[idx]["hadm_id"].values
    return x_test, y_test, sub_test, hadm_test


def _eval_stage2(
    model_dir: object,
    hadm_test: np.ndarray,
    y_test: np.ndarray,
    sub_test: pd.DataFrame,
    s1_scores: np.ndarray,
    s1_threshold: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Load Stage 2 results for the test partition and compute evaluation report.

    C9 fallback (session 15, 2026-08-25): ~37% of Stage 1-flagged patients have
    no discharge note and therefore no Stage 2 score. The old pipeline_pred
    silently scored these as negative (0), which is wrong — they were never
    seen by Stage 2 at all, so Stage 1's own flag is the only available
    prediction for them. ``pipeline_pred_full`` now uses, per admission:
    not flagged -> 0; flagged + has note -> stage2_confirmed;
    flagged + no note -> 1 (Stage 1 fallback).

    Args:
        model_dir:     path to models directory.
        hadm_test:     hadm_ids for the Stage 1 test partition.
        y_test:        ground-truth labels aligned to hadm_test.
        sub_test:      subgroup dataframe aligned to hadm_test.
        s1_scores:     Stage 1 probability scores aligned to hadm_test.
        s1_threshold:  Stage 1 classification threshold.

    Returns:
        (s2_report, pipeline_pred_full, has_s2, flagged) — ``pipeline_pred_full``
        is the C9-corrected final binary output over the whole test partition;
        ``has_s2`` and ``flagged`` are boolean masks used to build the
        notes-cohort vs. full-cohort split.
    """
    flagged = s1_scores >= s1_threshold
    s2_path = model_dir / "stage2_results.csv"
    if not s2_path.exists():
        print("[pipeline_eval] stage2_results.csv not found — skipping Stage 2 layer.")
        return {"available": False}, flagged.astype(int), np.zeros_like(flagged), flagged

    s2_all = pd.read_csv(s2_path)
    s2_rows = s2_all[s2_all["hadm_id"].isin(set(hadm_test))].set_index("hadm_id")

    s2_scores_arr = np.array([s2_rows["stage2_score"].get(h, float("nan")) for h in hadm_test])
    s2_conf_arr = np.array([s2_rows["stage2_confirmed"].get(h, 0) for h in hadm_test], dtype=int)
    has_s2 = ~np.isnan(s2_scores_arr)

    # C9: flagged-but-no-note patients fall back to Stage 1's own positive flag.
    pipeline_pred_full = np.where(has_s2, s2_conf_arr, flagged.astype(int))
    pipeline_pred_full = np.where(flagged, pipeline_pred_full, 0)

    if has_s2.sum() == 0:
        return {"available": False}, pipeline_pred_full, has_s2, flagged

    sub_s2 = sub_test[has_s2].reset_index(drop=True)
    report = _stage2_report(y_test[has_s2], s2_scores_arr[has_s2], s2_conf_arr[has_s2], sub_s2)
    report["available"] = True
    report["n_test_with_s2_scores"] = int(has_s2.sum())
    report["n_flagged_no_note"] = int((flagged & ~has_s2).sum())
    report["note_coverage_of_flagged"] = (
        float(has_s2.sum() / flagged.sum()) if flagged.sum() else 0.0
    )
    print(f"[pipeline_eval] Stage 2 (flagged+notes n={has_s2.sum():,}, "
          f"note coverage of flagged={report['note_coverage_of_flagged']:.1%}): "
          f"AUROC={report.get('auroc', 'n/a')}  "
          f"recall={report['recall']:.3f}  precision={report['precision']:.3f}")
    return report, pipeline_pred_full, has_s2, flagged


def _apply_stage3_decisions(
    model_dir: object,
    hadm_test: np.ndarray,
    pipeline_pred_full: np.ndarray,
    flagged: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Overlay Stage 3's decision onto the final prediction, where available.

    Per docs/ARCHITECTURE.md, the pipeline's final prediction should be
    Layer 3's uphold/override decision, not Stage 2's confirm/reject
    threshold — Stage 2's score is evidence Layer 3 reasons over, not the
    final word. Requires a batch Stage 3 run (``src/stage3/batch.py``);
    where it doesn't cover an admission (not flagged, no note, or an
    ``annotation_failed`` audit), the existing C9-corrected prediction
    (``stage2_confirmed``, falling back to Stage 1's flag for note-less
    admissions) is left unchanged — this function only ever narrows the gap
    between "what's implemented" and "what the design calls for", never
    silently drops coverage.

    Returns:
        (pipeline_pred_with_stage3, info) — ``info`` reports coverage; if no
        batch Stage 3 result exists yet, ``pipeline_pred_full`` is returned
        unchanged and ``info["available"]`` is ``False``.
    """
    s3_path = model_dir / "stage3_batch_results.csv"
    if not s3_path.exists():
        return pipeline_pred_full, {
            "available": False,
            "note": (
                "stage3_batch_results.csv not found -- final prediction is "
                "stage2_confirmed (+ C9 Stage 1 fallback), not yet Stage 3's "
                "decision. Run `python -m src.stage3.batch` first."
            ),
        }

    s3_all = pd.read_csv(s3_path)
    s3_rows = s3_all.set_index("hadm_id")
    decisions = s3_rows["decision"] if "decision" in s3_rows.columns else pd.Series(dtype=object)

    decision_arr = np.array([decisions.get(h, None) for h in hadm_test], dtype=object)
    has_s3 = np.array([d in ("uphold", "override") for d in decision_arr])

    pred = np.where(has_s3, (decision_arr == "uphold").astype(int), pipeline_pred_full)
    pred = np.where(flagged, pred, 0)  # never-flagged admissions always stay negative

    coverage = float(has_s3.sum() / flagged.sum()) if flagged.sum() else 0.0
    info = {
        "available": True,
        "n_test_with_stage3_decision": int(has_s3.sum()),
        "coverage_of_flagged": coverage,
    }
    print(f"[pipeline_eval] Stage 3 decisions applied to {has_s3.sum():,} / "
          f"{int(flagged.sum()):,} flagged admissions ({coverage:.1%} coverage)")
    return pred, info


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_pipeline(cfg: AppConfig) -> dict:
    """Run the end-to-end pipeline evaluation and save the report.

    Args:
        cfg: validated project config.

    Returns:
        Dict with stage1, stage2, and pipeline sub-reports.
    """
    model_dir = get_model_dir()
    artifact = joblib.load(model_dir / f"stage1_{cfg.stage1.model}.joblib")

    x_test, y_test, sub_test, hadm_test = _load_test_partition(artifact, cfg)
    s1_scores = apply_calibration(artifact, artifact["estimator"].predict_proba(x_test)[:, 1])
    s1_report = _stage1_report(y_test, s1_scores, artifact["threshold"], sub_test)
    print(f"\n[pipeline_eval] Stage 1 (test n={s1_report['n']:,}): "
          f"AUROC={s1_report['auroc']:.4f}  "
          f"recall={s1_report['recall']:.3f}  precision={s1_report['precision']:.3f}")

    s2_report, pipeline_pred_full, has_s2, flagged = _eval_stage2(
        model_dir, hadm_test, y_test, sub_test, s1_scores, artifact["threshold"]
    )
    pipeline_pred_full, s3_info = _apply_stage3_decisions(
        model_dir, hadm_test, pipeline_pred_full, flagged
    )

    # Full cohort (C9, primary): every test admission, note-less flagged
    # patients fall back to Stage 1's own flag rather than being silently
    # scored negative.
    full_report_ = _pipeline_report(y_test, pipeline_pred_full, sub_test)
    print(f"[pipeline_eval] Pipeline, full cohort (n={full_report_['n']:,}, C9 fallback applied): "
          f"precision={full_report_['precision']:.3f}  "
          f"recall={full_report_['recall']:.3f}  "
          f"F1={full_report_['f1']:.3f}  F2={full_report_['f2']:.3f}")

    # Notes cohort (secondary): restricted to admissions Stage 2 actually saw
    # (not flagged, or flagged with a note) — matches how "+21% precision"
    # style claims were computed before the C9 fix. Kept for comparability;
    # never report this number alone (P2/C9 in the 2026-08-19 review).
    notes_mask = (~flagged) | has_s2
    sub_notes = sub_test[notes_mask].reset_index(drop=True)
    notes_report = _pipeline_report(
        y_test[notes_mask], pipeline_pred_full[notes_mask], sub_notes
    )
    print(f"[pipeline_eval] Pipeline, notes cohort only (n={notes_report['n']:,}, "
          f"excludes {int((~notes_mask).sum()):,} flagged-no-note admissions): "
          f"precision={notes_report['precision']:.3f}  recall={notes_report['recall']:.3f}")

    # Control arm (colleague review, 2026-08-27): Stage 1 alone, thresholded
    # to the full system's own alert volume. Load-bearing for RQ2 -- without
    # it there is no answer to "did the cascade + audit add anything, or
    # would tightening Stage 1's threshold alone have caught as much at the
    # same alert budget?"
    control_report = _control_arm_report(
        y_test, s1_scores, full_report_["confirmed_rate"], sub_test
    )
    print(f"[pipeline_eval] Control arm (Stage 1 @ matched alert rate="
          f"{control_report['target_alert_rate']:.1%}, thr={control_report['threshold']:.4f}): "
          f"precision={control_report['precision']:.3f}  "
          f"recall={control_report['recall']:.3f}  "
          f"F1={control_report['f1']:.3f}  F2={control_report['f2']:.3f}")

    report = {
        "note": (
            "Stage 1 test partition used for all three layers. Stage 2 scores "
            "come from stage2_results.csv — patients in Stage 1's test "
            "partition are out-of-sample for Stage 2. 'pipeline.full_cohort' "
            "is the primary, deployable number (100% of test admissions; "
            "flagged-no-note patients fall back to the Stage 1 flag per the "
            "C9 fix, session 15 2026-08-25). 'pipeline.notes_cohort' excludes "
            "flagged-no-note admissions and must always be reported alongside "
            "full_cohort, never alone. The final prediction is Stage 3's "
            "uphold/override decision wherever a batch Stage 3 run covers the "
            "admission (see 'stage3' below for coverage); Stage 2's threshold "
            "is only a fallback where Stage 3 hasn't run yet. "
            "'pipeline.control_arm_stage1_matched' is Stage 1 alone at the "
            "same alert volume as the full system — the single comparison "
            "that determines whether the cascade earns its complexity; "
            "report it alongside full_cohort every time, never omit it."
        ),
        "stage1": s1_report,
        "stage2": s2_report,
        "stage3": s3_info,
        "pipeline": {
            "full_cohort": full_report_,
            "notes_cohort": notes_report,
            "control_arm_stage1_matched": control_report,
        },
    }
    _print_band_table(s1_report, s2_report, full_report_)

    out_path = model_dir / "pipeline_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[pipeline_eval] Saved -> {out_path}")
    return report


def _print_band_table(s1: dict, s2: dict, pipe: dict) -> None:
    """Print a per-age-band comparison table."""
    print("\n  Age band  │ S1 AUROC │ S1 Rec │ S2 AUROC │ S2 Prec │ Pipe F1")
    print("  " + "─" * 64)
    s1_bands = s1.get("by_age_band", {})
    s2_bands = s2.get("by_age_band", {}) if s2.get("available") else {}
    pi_bands = pipe.get("by_age_band", {})
    for band in sorted(s1_bands.keys(), key=str):
        s1b = s1_bands[band]
        s2b = s2_bands.get(band, {})
        pib = pi_bands.get(band, {})
        s1_auroc = f"{s1b.get('auroc', 0):.3f}" if s1b.get("auroc") else "  n/a"
        s2_auroc = f"{s2b.get('auroc', 0):.3f}" if s2b.get("auroc") else "  n/a"
        print(
            f"  {band:<9} │  {s1_auroc}  │  {s1b.get('recall', 0):.3f} "
            f"│   {s2_auroc} │  {s2b.get('precision', 0):.3f}  │  {pib.get('f1', 0):.3f}"
        )


def main() -> None:
    """CLI entry point for end-to-end pipeline evaluation."""
    cfg = load_config()
    evaluate_pipeline(cfg)


if __name__ == "__main__":
    main()
