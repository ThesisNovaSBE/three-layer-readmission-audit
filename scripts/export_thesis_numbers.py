"""Generate the single source of truth for Chapter 4 numbers.

Reads ONLY committed result JSONs (under ``models/``) plus
``results/stage2_training_log_excerpt.txt`` for the Stage 2 training sample, and
writes:

* ``docs/THESIS_NUMBERS.md`` — one section per thesis table/claim, every value
  tagged with its source file and JSON key.
* ``docs/results_numbers.tex`` — one ``\\newcommand`` per value.

Rounding is uniform and deterministic: ``decimal.ROUND_HALF_UP`` on the JSON's
own decimal string (see :func:`_r`). AUROC/AUPRC: 4 dp; precision/recall/F1/F2/
ECE: 3 dp; paired differences: 4 dp; shares in % : 1 dp; counts: thousands
separators.

A few facts are not in any JSON and are passed through verbatim, clearly
labelled: the Stage 1 method facts (from code/artifact, task B5), the feature
count (task B2), and GPU-hours (from sacct). These are defined as module
constants so they live in exactly one place.

Usage::

    python -m scripts.export_thesis_numbers
"""

from __future__ import annotations

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from src.config import get_model_dir

# ── Facts not in any JSON (single source; clearly labelled in the output) ────
STAGE1_METHOD = {
    "cv_folds": 5,
    "optuna_trials": 400,
    "capacity_k": 0.15,
    "threshold": 0.3235,
    "calibration": "isotonic (fitted on out-of-fold predictions)",
    "threshold_selected_on": "top-15% capacity on the calibrated out-of-fold predictions",
    "early_stopping_on": "a single 15% patient-grouped hold-out of the training partition",
}
N_FEATURES = 55  # src/data/features.py: 17 base + 8 labs x 4 aggregates + 6 vitals (B2)
GPU_HOURS = {"total": 335.99, "jobs": 57, "final_stage3": 104.84}

_JSONS = ("pipeline_evaluation", "xgboost_metrics", "stage2_evaluation",
          "stage2_acceptance_check", "stage2_calibration", "comparison_rq1",
          "rq1_raw_score_auroc", "discordance_sensitivity",
          "discordance_subgroup_evaluation", "stage3_run_summary",
          "stage3_validation_controls_summary", "batch_size_comparison",
          "pipeline_paired_differences", "selection_bias_evaluation",
          "planned_return_crosscheck")


def _r(value: float | int, places: int) -> str:
    """ROUND_HALF_UP ``value`` to ``places`` decimals, as a fixed-width string."""
    q = Decimal(1).scaleb(-places) if places else Decimal(1)
    return str(Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP))


def auroc(v: float) -> str:
    """AUROC/AUPRC — 4 decimals."""
    return _r(v, 4)


def metric(v: float) -> str:
    """precision/recall/F1/F2/ECE — 3 decimals."""
    return _r(v, 3)


def diff(v: float) -> str:
    """paired difference — 4 decimals."""
    return _r(v, 4)


def pct(v: float) -> str:
    """share as a percentage — 1 decimal."""
    return _r(Decimal(str(v)) * 100, 1) + "%"


def cnt(v: float) -> str:
    """integer count with a thousands separator."""
    return f"{int(v):,}"


class Out:
    """Accumulates markdown rows and LaTeX \\newcommand lines together."""

    def __init__(self):
        self.md: list[str] = []
        self.tex: list[str] = []

    def h(self, text: str):
        """Start a section with a table header."""
        self.md += [f"\n## {text}\n", "| Value | Number | Source |", "|---|---|---|"]

    def sub(self, text: str):
        """Start a sub-block with a table header."""
        self.md += [f"\n**{text}**\n", "| Value | Number | Source |", "|---|---|---|"]

    def row(self, label: str, number: str, source: str, cmd: str | None = None):
        """Add one table row (and optionally one \\newcommand)."""
        self.md.append(f"| {label} | {number} | `{source}` |")
        if cmd:
            self.tex.append(f"\\newcommand{{\\{cmd}}}{{{number}}}")

    def note(self, text: str):
        """Add a blockquote note."""
        self.md.append(f"\n> {text}\n")


def _sec_populations(out: Out, d: dict) -> None:
    """tab:populations + tab:selection."""
    pe = d["pipeline_evaluation"]
    fc = pe["pipeline"]["full_cohort"]
    s3 = pe["stage3"]
    out.h("tab:populations — populations, readmission rates, Stage 3 outcomes")
    out.row("Test partition n", cnt(fc["n"]),
            "pipeline_evaluation.json pipeline.full_cohort.n",
            "TestN")
    out.row("Full-cohort readmission rate", pct(fc["pos_rate"]),
            "pipeline_evaluation.json pipeline.full_cohort.pos_rate", "FullPosRate")
    out.row("Notes cohort n", cnt(pe["pipeline"]["notes_cohort"]["n"]),
            "pipeline_evaluation.json pipeline.notes_cohort.n", "NotesN")
    out.row("Flagged+noted n (Stage 2/3)", cnt(d["stage2_evaluation"]["overall"]["n"]),
            "stage2_evaluation.json overall.n", "FlaggedN")
    out.row("Decided (uphold/override)", cnt(s3["n_test_with_stage3_decision"]),
            "pipeline_evaluation.json stage3.n_test_with_stage3_decision", "Decided")
    out.row("Insufficient evidence", cnt(s3["n_insufficient_evidence"]),
            "pipeline_evaluation.json stage3.n_insufficient_evidence", "Insufficient")
    out.row("Invalid answers", cnt(s3["n_llm_annotation_failed"]),
            "pipeline_evaluation.json stage3.n_llm_annotation_failed", "Invalid")
    out.row("Preparation failures", cnt(s3["n_prep_failed"]),
            "pipeline_evaluation.json stage3.n_prep_failed", "PrepFailed")
    sel = d["selection_bias_evaluation"]
    out.h("tab:selection — selection bias")
    out.row("Notes coverage of test", pct(sel["notes_coverage_rate"]),
            "selection_bias_evaluation.json notes_coverage_rate", "NotesCoverage")
    out.row("Outcome rate covered / excluded",
            f"{pct(sel['outcome_rate_covered'])} / {pct(sel['outcome_rate_excluded'])}",
            "selection_bias_evaluation.json outcome_rate_*")
    out.row("Outcome SMD", metric(sel["outcome_smd"]),
            "selection_bias_evaluation.json outcome_smd",
            "OutcomeSMD")
    n_imb = sum(1 for c in sel["baseline_characteristics"] if abs(c.get("smd", 0)) >= 0.1)
    out.row("Characteristics with |SMD|>=0.1", f"{n_imb} of {len(sel['baseline_characteristics'])}",
            "selection_bias_evaluation.json baseline_characteristics[*].smd", "NImbalanced")


def _sec_stage1(out: Out, d: dict) -> None:
    """tab:stage1-tradeoff + tab:stage1-age."""
    xgb = d["xgboost_metrics"]
    bc = xgb["bootstrap_ci"]
    ac, pc = bc["auroc"], bc["auprc"]
    op = xgb["operating_point"]
    out.h("tab:stage1-tradeoff — Stage 1 discrimination & operating point")
    out.row("AUROC", f"{auroc(xgb['auroc'])} [{auroc(ac['ci_lower'])}, {auroc(ac['ci_upper'])}]",
            "xgboost_metrics.json auroc / bootstrap_ci.auroc", "StageOneAUROC")
    out.row("AUPRC", f"{auroc(xgb['auprc'])} [{auroc(pc['ci_lower'])}, {auroc(pc['ci_upper'])}]",
            "xgboost_metrics.json auprc / bootstrap_ci.auprc", "StageOneAUPRC")
    out.row("Precision/Recall/F2 @capacity-15%",
            f"{metric(op['precision'])} / {metric(op['recall'])} / {metric(op['f2'])}",
            "xgboost_metrics.json operating_point", "StageOnePrec")
    out.row("Threshold (capacity-15%)", _r(op["threshold"], 4),
            "xgboost_metrics.json operating_point.threshold", "StageOneThreshold")
    out.sub("tab:stage1-age — Stage 1 by age band (pipeline_evaluation.json stage1.by_age_band)")
    for band, m in d["pipeline_evaluation"]["stage1"]["by_age_band"].items():
        out.row(f"{band}: recall / precision / AUROC",
                f"{metric(m['recall'])} / {metric(m['precision'])} / {auroc(m['auroc'])}",
                f"pipeline_evaluation.json stage1.by_age_band.{band}")


def _sec_rq1(out: Out, d: dict) -> None:
    """tab:rq1-average + raw-vs-calibrated sensitivity."""
    cmp = d["comparison_rq1"]
    s1n, s2n = cmp["stage1_notes_cohort"], cmp["stage2_notes_cohort"]
    ad = cmp["auroc_diff_stage2_minus_stage1"]
    out.h("tab:rq1-average — Stage 1 vs Stage 2 on the notes-covered population")
    out.row("Population n", cnt(s1n["n"]), "comparison_rq1.json stage1_notes_cohort.n", "RQOneN")
    out.row("Stage 1 AUROC", f"{auroc(s1n['auroc']['point_estimate'])} "
            f"[{auroc(s1n['auroc']['ci_lower'])}, {auroc(s1n['auroc']['ci_upper'])}]",
            "comparison_rq1.json stage1_notes_cohort.auroc", "RQOneStageOneAUROC")
    out.row("Stage 2 AUROC", f"{auroc(s2n['auroc']['point_estimate'])} "
            f"[{auroc(s2n['auroc']['ci_lower'])}, {auroc(s2n['auroc']['ci_upper'])}]",
            "comparison_rq1.json stage2_notes_cohort.auroc", "RQOneStageTwoAUROC")
    out.row("Paired AUROC diff (S2-S1)", f"{diff(ad['point_estimate'])} "
            f"[{diff(ad['ci_lower'])}, {diff(ad['ci_upper'])}]",
            "comparison_rq1.json auroc_diff_stage2_minus_stage1", "RQOneAUROCDiff")
    out.sub("RQ1 raw-vs-calibrated sensitivity (rq1_raw_score_auroc.json)")
    for k, v in d["rq1_raw_score_auroc"]["auroc"].items():
        out.row(f"AUROC {k}", f"{auroc(v['point_estimate'])} "
                f"[{auroc(v['ci_lower'])}, {auroc(v['ci_upper'])}]",
                        f"rq1_raw_score_auroc.json auroc.{k}")


def _sec_stage2(out: Out, d: dict) -> None:
    """Stage 2 training, flagged metrics, acceptance, calibration."""
    s2, s2acc, s2cal = d["stage2_evaluation"], d["stage2_acceptance_check"], d["stage2_calibration"]
    ov, fg = s2["overall"], s2["fairness_gaps"]
    rg, pg, ag = metric(fg["recall_gap"]), metric(fg["precision_gap"]), metric(fg["auroc_gap"])
    out.h("Stage 2 — training, flagged-group metrics, calibration")
    log = (Path(__file__).resolve().parents[1] / "results"
           / "stage2_training_log_excerpt.txt").read_text()
    tr = re.search(r"Train:\s*([\d,]+)\s*\|\s*Val:\s*([\d,]+)", log)
    out.row("Training sample / validation", f"{tr.group(1)} / {tr.group(2)}" if tr else "pending",
            "stage2_training_log_excerpt.txt", "StageTwoTrainN")
    out.row("Flagged AUROC / AUPRC", f"{auroc(ov['auroc'])} / {auroc(ov['auprc'])}",
            "stage2_evaluation.json overall", "StageTwoAUROC")
    out.row("Flagged recall / precision / F2",
            f"{metric(ov['recall'])} / {metric(ov['precision'])} / {metric(ov['f2'])}",
            "stage2_evaluation.json overall", "StageTwoRecall")
    out.row("Flagged ECE", metric(ov["ece"]), "stage2_evaluation.json overall.ece", "StageTwoECE")
    out.row("Fairness recall/precision/AUROC gap",
            f"{rg} / {pg} / {ag} "
            f"(worst recall {fg['worst_recall_band']})",
            "stage2_evaluation.json fairness_gaps", "StageTwoRecallGap")
    out.sub("Stage 2 per age band (stage2_evaluation.json by_age_band)")
    for band, m in s2["by_age_band"].items():
        out.row(f"{band}: recall / precision / ECE",
                f"{metric(m['recall'])} / {metric(m['precision'])} / {metric(m['ece'])}",
                f"stage2_evaluation.json by_age_band.{band}")
    out.sub("Stage 2 acceptance check on all 62,759 (stage2_acceptance_check.json)")
    out.row("Overall ECE / n", f"{metric(s2acc['overall']['ece'])} / {cnt(s2acc['n_total'])}",
            "stage2_acceptance_check.json overall.ece / n_total", "StageTwoAcceptECE")
    out.sub("Platt parameters & thresholds (stage2_calibration.json)")
    for band in s2cal["calibrators"]:
        a, b = s2cal["calibrators"][band]
        out.row(f"{band}: slope / intercept / threshold",
                f"{_r(a, 4)} / {_r(b, 4)} / {_r(s2cal['thresholds'][band], 4)}",
                f"stage2_calibration.json calibrators/thresholds.{band}")


def _sec_discordance(out: Out, d: dict) -> None:
    """tab:discordance-sweep + RQ1 correct-alert comparison."""
    disc, sub = d["discordance_sensitivity"], d["discordance_subgroup_evaluation"]
    sw = disc["sweep"]["20.0"]
    cvals = [disc["sweep"][t]["CONCORDANT"] for t in disc["sweep"]]
    out.h("tab:discordance-sweep & RQ1 correct-alert comparison")
    out.row("At +/-20pp: CONCORDANT/MITIGATES/AMPLIFIES",
            f"{pct(sw['CONCORDANT'])} / {pct(sw['NOTE_MITIGATES'])} / {pct(sw['NOTE_AMPLIFIES'])}",
            "discordance_sensitivity.json sweep.20.0", "ConcordantShare")
    out.row("Concordant share across sweep (10-30pp)", f"{pct(min(cvals))}-{pct(max(cvals))}",
            "discordance_sensitivity.json sweep.*.CONCORDANT", "ConcordantRange")
    for grp in ("concordant", "discordant"):
        g = sub[grp]
        f2d = g["stage3_decision"]["f2"] - g["stage1_alone"]["f2"]
        out.row(f"{grp} (n={cnt(g['n'])}): correct-alert share S1 / Stage3, F2 change",
                f"{metric(g['stage1_alone']['precision'])} / "
                f"{metric(g['stage3_decision']['precision'])}, {diff(f2d)}",
                f"discordance_subgroup_evaluation.json {grp}", f"{grp.capitalize()}FTwoChange")


def _sec_rq2(out: Out, d: dict) -> None:
    """RQ2 — auditor behaviour & validation controls."""
    run = d["stage3_run_summary"]
    vc = d["stage3_validation_controls_summary"]
    bs = d["batch_size_comparison"]
    dm = run["decision_model_counts"]
    out.h("RQ2 — auditor behaviour & validation controls")
    out.row("uphold / override / insufficient / invalid",
            f"{cnt(dm.get('uphold', 0))} / {cnt(dm.get('override', 0))} / "
            f"{cnt(run['n_insufficient_evidence'])} / {cnt(run['n_invalid_llm_answers'])}",
            "stage3_run_summary.json decision_model_counts", "Uphold")
    out.row("model/rule agreement (decided)", pct(run["model_rule_agreement_among_decided"]),
            "stage3_run_summary.json model_rule_agreement_among_decided", "ModelRuleAgree")
    out.row("all-quotes-verified (decided)", pct(run["all_quotes_verified_share_among_decided"]),
            "stage3_run_summary.json all_quotes_verified_share_among_decided", "QuotesVerified")
    out.row("note_truncated count / share",
            f"{cnt(run['note_truncated_count'])} / {pct(run['note_truncated_share'])}",
            "stage3_run_summary.json note_truncated_*", "TruncatedShare")
    d1 = vc["d1_blind_note"]["vs_reference_arm"]
    d2 = vc["d2_no_stage2"]["vs_reference_arm"]
    out.row("Note-withheld insufficient_evidence",
            f"{cnt(d1['blind_note_distribution'].get('insufficient_evidence', 0))}/{cnt(d1['n'])}",
            "stage3_validation_controls_summary.json d1_blind_note", "BlindNote")
    out.row("Stage-2-withheld agreement vs reference", pct(d2["decision_model_agreement_vs_real"]),
            "stage3_validation_controls_summary.json d2_no_stage2", "NoStageTwoAgree")
    out.row("Repeated-reference agreement",
            pct(vc["d3_self_agreement"]["decision_model_agreement"]),
            "stage3_validation_controls_summary.json d3_self_agreement", "RepeatedAgree")
    out.row("Batch-size check (decision / grounds)",
            f"{pct(bs['decision_model_agreement'])} / {pct(bs['grounds_agreement'])}",
            "batch_size_comparison.json", "BatchDecisionAgree")
    out.note("Reference-arm-vs-main-run agreement (145/150, 96.7%) is NOT in any committed JSON: "
             "it was measured on the cluster (validation job 17462868, "
             "scripts/run_validation_controls.py) and the per-admission control CSVs are "
             "gitignored. Source documented in results/README.md.")


def _sec_rq3(out: Out, d: dict) -> None:
    """tab:rq3-main + tab:rq3-age + paired differences."""
    pe = d["pipeline_evaluation"]
    fc = pe["pipeline"]["full_cohort"]
    ctrl = pe["pipeline"]["control_arm_stage1_matched"]
    out.h("tab:rq3-main — pipeline vs control at matched volume")
    out.row("Full pipeline: precision / recall",
            f"{metric(fc['precision'])} / {metric(fc['recall'])}",
            "pipeline_evaluation.json pipeline.full_cohort", "PipePrec")
    out.row("Control arm: precision / recall",
            f"{metric(ctrl['precision'])} / {metric(ctrl['recall'])}",
            "pipeline_evaluation.json pipeline.control_arm_stage1_matched", "CtrlPrec")
    out.row("Control alert count", cnt(ctrl["realised_alert_count"]),
            "pipeline_evaluation.json pipeline.control_arm_stage1_matched.realised_alert_count",
                    "CtrlAlerts")
    nc = pe["pipeline"]["notes_cohort"]
    out.row("Notes cohort: precision / recall",
            f"{metric(nc['precision'])} / {metric(nc['recall'])}",
            "pipeline_evaluation.json pipeline.notes_cohort")
    cond = pe["pipeline"]["conditional_triggering"]
    out.row("Conditional: precision / recall",
            f"{metric(cond['report']['precision'])} / {metric(cond['report']['recall'])}",
            "pipeline_evaluation.json pipeline.conditional_triggering.report", "CondPrec")
    out.row("Conditional LLM calls (blanket/conditional/saved)",
            f"{cnt(cond['blanket_llm_calls'])} / {cnt(cond['conditional_llm_calls'])} / "
            f"{cnt(cond['llm_calls_saved'])}",
            "pipeline_evaluation.json pipeline.conditional_triggering")
    sec = pe["pipeline"]["secondary_both_agree"]
    out.row("Secondary policy: precision / recall / withdrawn",
            f"{metric(sec['report']['precision'])} / {metric(sec['report']['recall'])} / "
            f"{cnt(sec['n_alerts_withdrawn'])}",
            "pipeline_evaluation.json pipeline.secondary_both_agree", "SecPrec")
    out.sub("Paired pipeline-minus-control differences (pipeline_paired_differences.json)")
    for name, c in d["pipeline_paired_differences"]["comparisons"].items():
        p, r = c["precision_diff"], c["recall_diff"]
        out.row(f"{name}: precision diff / recall diff",
                f"{diff(p['point_estimate'])} [{diff(p['ci_lower'])}, {diff(p['ci_upper'])}] / "
                f"{diff(r['point_estimate'])} [{diff(r['ci_lower'])}, {diff(r['ci_upper'])}]",
                f"pipeline_paired_differences.json comparisons.{name}",
                "PrimaryPrecDiff" if name.startswith("primary") else None)
    out.sub("tab:rq3-age — pipeline full cohort by age band")
    for band, m in fc["by_age_band"].items():
        out.row(f"{band}: precision / recall", f"{metric(m['precision'])} / {metric(m['recall'])}",
                f"pipeline_evaluation.json pipeline.full_cohort.by_age_band.{band}")


def _sec_planned_method(out: Out, d: dict) -> None:
    """tab:planned + Stage 1 method/feature facts + compute."""
    plan = d["planned_return_crosscheck"]
    ct = plan["cross_tab"]
    out.h("tab:planned — planned-return cross-check (n=4,402)")
    out.row("n comparable", cnt(plan["n_comparable"]),
            "planned_return_crosscheck.json n_comparable",
            "PlannedN")
    out.row("Agreement / majority baseline",
            f"{pct(plan['agreement_rate'])} / {pct(plan['majority_baseline_agreement'])}",
            "planned_return_crosscheck.json agreement_rate", "PlannedAgree")
    out.row("2x2 struct-planned (no/yes)",
            f"{cnt(ct['structured=planned']['no'])} / {cnt(ct['structured=planned']['yes'])}",
            "planned_return_crosscheck.json cross_tab.structured=planned")
    out.row("2x2 struct-unplanned (no/yes)",
            f"{cnt(ct['structured=unplanned']['no'])} / {cnt(ct['structured=unplanned']['yes'])}",
            "planned_return_crosscheck.json cross_tab.structured=unplanned")
    out.h("Stage 1 method facts (from code/artifact, not a JSON — tasks B2/B5)")
    out.row("Structured features used", str(N_FEATURES),
            "src/data/features.py build_features (17 base + 8 labs x 4 + 6 vitals)", "NFeatures")
    out.row("Optuna trials / CV folds",
            f"{STAGE1_METHOD['optuna_trials']} / {STAGE1_METHOD['cv_folds']}",
            "config.yaml run.full + src/model/cv.py")
    out.row("Early stopping", STAGE1_METHOD["early_stopping_on"],
            "src/model/train.py _fit_final_estimator")
    out.row("Threshold selection", STAGE1_METHOD["threshold_selected_on"],
            "src/model/train.py _select_thresholds")
    out.row("Calibration", STAGE1_METHOD["calibration"], "src/model/train.py _calibrate")
    out.h("Compute (from sacct, not a JSON)")
    out.row("Total GPU-hours / jobs",
            f"{GPU_HOURS['total']} / {GPU_HOURS['jobs']}",
            "sacct via MODEL_CARD.md",
            "GPUHours")
    out.row("Final Stage 3 run GPU-hours", str(GPU_HOURS["final_stage3"]),
            "sacct via MODEL_CARD.md")


def build(out: Out) -> None:
    """Fill ``out`` with every Chapter-4 number."""
    d = {name: json.loads((get_model_dir() / f"{name}.json").read_text()) for name in _JSONS}
    _sec_populations(out, d)
    _sec_stage1(out, d)
    _sec_rq1(out, d)
    _sec_stage2(out, d)
    _sec_discordance(out, d)
    _sec_rq2(out, d)
    _sec_rq3(out, d)
    _sec_planned_method(out, d)


def main() -> None:
    """Write THESIS_NUMBERS.md and results_numbers.tex."""
    out = Out()
    out.md.append("# Thesis numbers — single source of truth for Chapter 4\n")
    out.md.append("Generated by `scripts/export_thesis_numbers.py` from committed JSONs only. "
                  "Every value is tagged with its source file and key. Do not hand-edit.\n")
    build(out)
    docs = Path(__file__).resolve().parents[1] / "docs"
    (docs / "THESIS_NUMBERS.md").write_text("\n".join(out.md) + "\n")
    (docs / "results_numbers.tex").write_text(
        "\n".join(["% Generated by scripts/export_thesis_numbers.py -- do not hand-edit.",
                   *out.tex]) + "\n")
    print(f"[export_thesis_numbers] wrote docs/THESIS_NUMBERS.md ({len(out.md)} lines) "
          f"and docs/results_numbers.tex ({len(out.tex)} commands)")


if __name__ == "__main__":
    main()
