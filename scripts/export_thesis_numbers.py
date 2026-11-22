"""Generate the single source of truth for Chapter 4 numbers.

Reads ONLY committed result JSONs (under ``models/``) plus
``results/stage2_training_log_excerpt.txt`` for the Stage 2 training sample, and
writes ``docs/THESIS_NUMBERS.md``: one section per thesis table/claim, every
value tagged with its source file and JSON key. It is the only numbers file;
there is deliberately no LaTeX export.

Rounding is uniform and deterministic: ``decimal.ROUND_HALF_UP`` on the JSON's
own decimal string (see :func:`_r`). AUROC/AUPRC/Brier/thresholds: 4 dp;
precision/recall/specificity/F1/F2/ECE/lift: 3 dp; differences: 4 dp with an
explicit sign; shares in %: 1 dp; counts: thousands separators.

Values marked "derived" are computed here from other keys of the same JSONs
(e.g. alert count = tp + fp, rate = count / n); the formula is stated in the
source column.

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

PE = "pipeline_evaluation.json"


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
    """difference — 4 decimals with an explicit sign (+0.0009, -0.0040)."""
    s = _r(v, 4)
    if Decimal(s) == 0:
        return "0.0000"
    return s if s.startswith("-") else "+" + s


def pct(v: float) -> str:
    """share as a percentage — 1 decimal."""
    return _r(Decimal(str(v)) * 100, 1) + "%"


def cnt(v: float) -> str:
    """integer count with a thousands separator."""
    return f"{int(v):,}"


def ci(block: dict, fmt=metric) -> str:
    """``point [lower, upper]`` from a {point_estimate, ci_lower, ci_upper} block."""
    return f"{fmt(block['point_estimate'])} [{fmt(block['ci_lower'])}, {fmt(block['ci_upper'])}]"


def alerts(m: dict) -> int:
    """Alert count of a confusion-matrix block (tp + fp)."""
    return m["tp"] + m["fp"]


class Out:
    """Accumulates markdown rows."""

    def __init__(self):
        self.md: list[str] = []

    def h(self, text: str):
        """Start a section with a table header."""
        self.md += [f"\n## {text}\n", "| Value | Number | Source |", "|---|---|---|"]

    def sub(self, text: str):
        """Start a sub-block with a table header."""
        self.md += [f"\n**{text}**\n", "| Value | Number | Source |", "|---|---|---|"]

    def row(self, label: str, number: str, source: str):
        """Add one table row."""
        self.md.append(f"| {label} | {number} | `{source}` |")

    def note(self, text: str):
        """Add a blockquote note."""
        self.md.append(f"\n> {text}\n")


def _sec_populations(out: Out, d: dict) -> None:
    """tab:populations + tab:selection."""
    pe = d["pipeline_evaluation"]
    fc = pe["pipeline"]["full_cohort"]
    s3 = pe["stage3"]
    sel = d["selection_bias_evaluation"]
    op = d["xgboost_metrics"]["operating_point"]
    s2 = d["stage2_evaluation"]["overall"]
    sub = d["discordance_subgroup_evaluation"]
    rq1op = d["comparison_rq1"]["stage1_notes_cohort"]["operating_point"]

    out.h("tab:populations — populations, readmission rates, Stage 3 outcomes")
    out.row("Test partition n", cnt(fc["n"]), f"{PE} pipeline.full_cohort.n")
    out.row("Test partition readmission rate", pct(fc["pos_rate"]),
            f"{PE} pipeline.full_cohort.pos_rate")
    out.row("With discharge summary: n / readmission rate",
            f"{cnt(sel['n_notes_covered'])} / {pct(sel['outcome_rate_covered'])}",
            "selection_bias_evaluation.json n_notes_covered / outcome_rate_covered")
    out.row("Without discharge summary: n / readmission rate",
            f"{cnt(sel['n_notes_excluded'])} / {pct(sel['outcome_rate_excluded'])}",
            "selection_bias_evaluation.json n_notes_excluded / outcome_rate_excluded")
    flagged = alerts(op)
    out.row("Flagged by Stage 1: n / share of test / readmission rate",
            f"{cnt(flagged)} / {pct(flagged / fc['n'])} / {pct(op['precision'])}",
            "xgboost_metrics.json operating_point (derived: n = tp+fp; share = n / test n; "
            "rate = precision)")
    out.row("Flagged with discharge summary (Stage 2/3 input): n / readmission rate",
            f"{cnt(s2['n'])} / {pct(s2['pos_rate'])}",
            "stage2_evaluation.json overall.n / overall.pos_rate")
    no_note_pos = op["tp"] - rq1op["tp"]
    out.row("Flagged without discharge summary (keep Stage 1 alert): n / readmission rate",
            f"{cnt(pe['stage2']['n_flagged_no_note'])} / "
            f"{pct(no_note_pos / pe['stage2']['n_flagged_no_note'])}",
            f"{PE} stage2.n_flagged_no_note (rate derived: (xgboost_metrics operating_point.tp "
            "- comparison_rq1 stage1_notes_cohort.operating_point.tp) / n)")
    dec_n = sub["concordant"]["n"] + sub["discordant"]["n"]
    dec_pos = sub["concordant"]["stage1_alone"]["tp"] + sub["discordant"]["stage1_alone"]["tp"]
    out.row("Decided by Stage 3 (uphold/override): n / readmission rate",
            f"{cnt(s3['n_test_with_stage3_decision'])} / {pct(dec_pos / dec_n)}",
            f"{PE} stage3.n_test_with_stage3_decision (rate derived: discordance_subgroup_evaluation "
            "concordant+discordant stage1_alone.tp / n)")
    out.row("Insufficient evidence", cnt(s3["n_insufficient_evidence"]),
            f"{PE} stage3.n_insufficient_evidence")
    out.row("Invalid answers", cnt(s3["n_llm_annotation_failed"]),
            f"{PE} stage3.n_llm_annotation_failed")
    out.row("Preparation failures", cnt(s3["n_prep_failed"]), f"{PE} stage3.n_prep_failed")
    out.row("Notes-cohort evaluation population (secondary row of the RQ3 table)",
            cnt(pe["pipeline"]["notes_cohort"]["n"]), f"{PE} pipeline.notes_cohort.n")

    out.h("tab:selection — selection bias")
    out.row("Notes coverage of test", pct(sel["notes_coverage_rate"]),
            "selection_bias_evaluation.json notes_coverage_rate")
    out.row("Outcome rate covered / excluded",
            f"{pct(sel['outcome_rate_covered'])} / {pct(sel['outcome_rate_excluded'])}",
            "selection_bias_evaluation.json outcome_rate_*")
    out.row("Outcome SMD", metric(sel["outcome_smd"]), "selection_bias_evaluation.json outcome_smd")
    n_imb = sum(1 for c in sel["baseline_characteristics"] if abs(c.get("smd", 0)) >= 0.1)
    out.row("Characteristics with abs(SMD) >= 0.1", f"{n_imb} of {len(sel['baseline_characteristics'])}",
            "selection_bias_evaluation.json baseline_characteristics[*].smd")


def _sec_stage1(out: Out, d: dict) -> None:
    """tab:stage1-tradeoff + tab:stage1-age."""
    xgb = d["xgboost_metrics"]
    bc = xgb["bootstrap_ci"]
    op = xgb["operating_point"]
    base = xgb["base_rate"]
    out.h("tab:stage1-tradeoff — Stage 1 discrimination & operating point")
    out.row("AUROC", ci(bc["auroc"], auroc), "xgboost_metrics.json bootstrap_ci.auroc")
    out.row("AUPRC", ci(bc["auprc"], auroc), "xgboost_metrics.json bootstrap_ci.auprc")
    out.row("Brier score", auroc(xgb["brier"]), "xgboost_metrics.json brier")
    out.row("Base rate", pct(base), "xgboost_metrics.json base_rate")
    out.row("Threshold (capacity-15%)", auroc(op["threshold"]),
            "xgboost_metrics.json operating_point.threshold")
    out.row("Precision @capacity-15%", ci(bc["precision"]), "xgboost_metrics.json bootstrap_ci.precision")
    out.row("Recall @capacity-15%", ci(bc["recall"]), "xgboost_metrics.json bootstrap_ci.recall")
    out.row("Specificity / F2 @capacity-15%", f"{metric(op['specificity'])} / {metric(op['f2'])}",
            "xgboost_metrics.json operating_point")
    for g, m in xgb["subgroup_auroc"]["gender"].items():
        out.row(f"Gender {g}: n / readmission rate / AUROC",
                f"{cnt(m['n'])} / {pct(m['pos_rate'])} / {auroc(m['auroc'])}",
                f"xgboost_metrics.json subgroup_auroc.gender.{g}")

    out.sub("Stage 1 operating points — capacity rule (xgboost_metrics.json capacity_tradeoff)")
    for i, m in enumerate(xgb["capacity_tradeoff"]):
        n = m["tp"] + m["fp"] + m["tn"] + m["fn"]
        out.row(f"K = {pct(m['capacity_k'])}: threshold / flagged share / precision / recall / "
                "specificity / F2 / lift",
                f"{auroc(m['threshold'])} / {pct(alerts(m) / n)} / {metric(m['precision'])} / "
                f"{metric(m['recall'])} / {metric(m['specificity'])} / {metric(m['f2'])} / "
                f"{metric(m['lift'])}",
                f"xgboost_metrics.json capacity_tradeoff[{i}] (flagged share derived: (tp+fp)/n)")
    out.sub("Stage 1 operating points — recall rule (xgboost_metrics.json recall_tradeoff)")
    for i, m in enumerate(xgb["recall_tradeoff"]):
        n = m["tp"] + m["fp"] + m["tn"] + m["fn"]
        out.row(f"Recall target {m['recall_target']}: threshold / flagged share / precision / recall / "
                "specificity / F2 / lift",
                f"{auroc(m['threshold'])} / {pct(alerts(m) / n)} / {metric(m['precision'])} / "
                f"{metric(m['recall'])} / {metric(m['specificity'])} / {metric(m['f2'])} / "
                f"{metric(m['precision'] / base)}",
                f"xgboost_metrics.json recall_tradeoff[{i}] (flagged share derived: (tp+fp)/n; "
                "lift derived: precision / base_rate)")

    out.sub("tab:stage1-age — Stage 1 by age band (pipeline_evaluation.json stage1.by_age_band)")
    for band, m in d["pipeline_evaluation"]["stage1"]["by_age_band"].items():
        out.row(f"{band}: n / readmission rate / recall / precision / AUROC",
                f"{cnt(m['n'])} / {pct(m['pos_rate'])} / {metric(m['recall'])} / "
                f"{metric(m['precision'])} / {auroc(m['auroc'])}",
                f"{PE} stage1.by_age_band.{band}")


def _sec_rq1(out: Out, d: dict) -> None:
    """tab:rq1-average + raw-vs-calibrated sensitivity."""
    cmp = d["comparison_rq1"]
    s1n, s2n = cmp["stage1_notes_cohort"], cmp["stage2_notes_cohort"]
    out.h("tab:rq1-average — Stage 1 vs Stage 2 on the notes-covered population")
    out.row("Population n / readmission rate", f"{cnt(s1n['n'])} / {pct(s1n['pos_rate'])}",
            "comparison_rq1.json stage1_notes_cohort.n / pos_rate")
    out.row("Stage 1 AUROC", ci(s1n["auroc"], auroc), "comparison_rq1.json stage1_notes_cohort.auroc")
    out.row("Stage 2 AUROC", ci(s2n["auroc"], auroc), "comparison_rq1.json stage2_notes_cohort.auroc")
    out.row("Paired AUROC diff (S2-S1)", ci(cmp["auroc_diff_stage2_minus_stage1"], diff),
            "comparison_rq1.json auroc_diff_stage2_minus_stage1")
    out.row("Stage 1 AUPRC", ci(s1n["auprc"], auroc), "comparison_rq1.json stage1_notes_cohort.auprc")
    out.row("Stage 2 AUPRC", ci(s2n["auprc"], auroc), "comparison_rq1.json stage2_notes_cohort.auprc")
    for name, blk in (("Stage 1", s1n), ("Stage 2", s2n)):
        m = blk["operating_point"]
        key = "stage1_notes_cohort" if name == "Stage 1" else "stage2_notes_cohort"
        out.row(f"{name} at matched volume: threshold / alerts / precision / recall / "
                "specificity / F2",
                f"{auroc(m['threshold'])} / {cnt(alerts(m))} / {metric(m['precision'])} / "
                f"{metric(m['recall'])} / {metric(m['specificity'])} / {metric(m['f2'])}",
                f"comparison_rq1.json {key}.operating_point (alerts derived: tp+fp)")
    out.sub("RQ1 raw-vs-calibrated sensitivity (rq1_raw_score_auroc.json)")
    for k, v in d["rq1_raw_score_auroc"]["auroc"].items():
        out.row(f"AUROC {k}", ci(v, auroc), f"rq1_raw_score_auroc.json auroc.{k}")


def _sec_stage2(out: Out, d: dict) -> None:
    """Stage 2 training, flagged metrics, acceptance, calibration."""
    s2, s2acc, s2cal = d["stage2_evaluation"], d["stage2_acceptance_check"], d["stage2_calibration"]
    ov, fg = s2["overall"], s2["fairness_gaps"]
    out.h("Stage 2 — training, flagged-group metrics, calibration")
    log = (Path(__file__).resolve().parents[1] / "results"
           / "stage2_training_log_excerpt.txt").read_text()
    tr = re.search(r"Train:\s*([\d,]+)\s*\|\s*Val:\s*([\d,]+)", log)
    out.row("Training sample / validation", f"{tr.group(1)} / {tr.group(2)}" if tr else "pending",
            "stage2_training_log_excerpt.txt")
    out.row("Flagged AUROC / AUPRC", f"{auroc(ov['auroc'])} / {auroc(ov['auprc'])}",
            "stage2_evaluation.json overall")
    out.row("Flagged recall / precision / F2",
            f"{metric(ov['recall'])} / {metric(ov['precision'])} / {metric(ov['f2'])}",
            "stage2_evaluation.json overall")
    out.row("Flagged ECE", metric(ov["ece"]), "stage2_evaluation.json overall.ece")
    out.row("Fairness recall/precision/AUROC gap",
            f"{metric(fg['recall_gap'])} / {metric(fg['precision_gap'])} / {metric(fg['auroc_gap'])} "
            f"(worst recall {fg['worst_recall_band']}, worst precision {fg['worst_precision_band']})",
            "stage2_evaluation.json fairness_gaps")
    out.sub("Stage 2 per age band (stage2_evaluation.json by_age_band)")
    for band, m in s2["by_age_band"].items():
        out.row(f"{band}: n / readmission rate / AUROC / recall / precision / ECE",
                f"{cnt(m['n'])} / {pct(m['pos_rate'])} / {auroc(m['auroc'])} / {metric(m['recall'])} / "
                f"{metric(m['precision'])} / {metric(m['ece'])}",
                f"stage2_evaluation.json by_age_band.{band}")
    out.sub(f"Stage 2 acceptance check on all {cnt(s2acc['n_total'])} (stage2_acceptance_check.json)")
    out.row("Overall ECE / n", f"{metric(s2acc['overall']['ece'])} / {cnt(s2acc['n_total'])}",
            "stage2_acceptance_check.json overall.ece / n_total")
    bands = s2acc["bands"]
    n_pass = sum(abs(b["recall_gap"]) <= 0.05 for b in bands.values()) + sum(
        abs(b["mean_calibrated_prob"] - b["observed_readmission_rate"]) <= 0.03 for b in bands.values())
    out.row("Stage 2 calibration criteria met (per band: abs(recall gap) <= 0.05 and "
            "abs(calibration-in-the-large) <= 0.03; NOT the Stage 3 acceptance arm test)",
            f"{n_pass}/{2 * len(bands)}",
            "stage2_acceptance_check.json bands.* (derived; matches flags.reported)")
    out.sub("Platt parameters & thresholds (stage2_calibration.json)")
    for band in s2cal["calibrators"]:
        a, b = s2cal["calibrators"][band]
        out.row(f"{band}: slope / intercept / threshold",
                f"{_r(a, 4)} / {_r(b, 4)} / {_r(s2cal['thresholds'][band], 4)}",
                f"stage2_calibration.json calibrators/thresholds.{band}")


def _sec_discordance(out: Out, d: dict) -> None:
    """tab:discordance-sweep + RQ1 correct-alert comparison."""
    disc, sub = d["discordance_sensitivity"], d["discordance_subgroup_evaluation"]
    out.h(f"tab:discordance-sweep — discordance categories (n = {cnt(disc['n_patients'])})")
    for t, s in disc["sweep"].items():
        out.row(f"+/-{float(t):g}pp: CONCORDANT / NOTE_MITIGATES / NOTE_AMPLIFIES",
                f"{pct(s['CONCORDANT'])} / {pct(s['NOTE_MITIGATES'])} / {pct(s['NOTE_AMPLIFIES'])}",
                f"discordance_sensitivity.json sweep.{t}")
    out.row("Primary threshold (pp)", f"{disc['current_config_value_pp']:g}",
            "discordance_sensitivity.json current_config_value_pp")

    out.h("RQ1 correct-alert comparison (discordance_subgroup_evaluation.json)")
    out.row("Decided admissions / excluded (no decision)",
            f"{cnt(sub['n_total_cohort'] - sub['n_excluded_no_decision'])} / "
            f"{cnt(sub['n_excluded_no_decision'])}",
            "discordance_subgroup_evaluation.json n_total_cohort - n_excluded_no_decision")
    for grp in ("concordant", "discordant"):
        g = sub[grp]
        s1, s3 = g["stage1_alone"], g["stage3_decision"]
        out.row(f"{grp} n", cnt(g["n"]), f"discordance_subgroup_evaluation.json {grp}.n")
        out.row(f"{grp}: correct-alert share (precision) before / after audit",
                f"{ci(s1['precision_ci'])} / {ci(s3['precision_ci'])}",
                f"discordance_subgroup_evaluation.json {grp}.stage1_alone/stage3_decision.precision_ci")
        out.row(f"{grp}: recall before / after audit",
                f"{ci(s1['recall_ci'])} / {ci(s3['recall_ci'])}",
                f"discordance_subgroup_evaluation.json {grp}.stage1_alone/stage3_decision.recall_ci")
        out.row(f"{grp}: F2 before / after audit / change",
                f"{metric(s1['f2'])} / {metric(s3['f2'])} / {diff(s3['f2'] - s1['f2'])}",
                f"discordance_subgroup_evaluation.json {grp}.*.f2 (change derived)")


def _sec_rq2(out: Out, d: dict) -> None:
    """RQ2 — auditor behaviour & validation controls."""
    run = d["stage3_run_summary"]
    vc = d["stage3_validation_controls_summary"]
    bs = d["batch_size_comparison"]
    dm, dr = run["decision_model_counts"], run["decision_rule_counts"]
    out.h("RQ2 — auditor behaviour & validation controls")
    out.row("Decision model: uphold / override / insufficient / invalid",
            f"{cnt(dm.get('uphold', 0))} / {cnt(dm.get('override', 0))} / "
            f"{cnt(run['n_insufficient_evidence'])} / {cnt(run['n_invalid_llm_answers'])}",
            "stage3_run_summary.json decision_model_counts / n_insufficient_evidence / "
            "n_invalid_llm_answers")
    out.row("Decision rule: uphold / override / missing",
            f"{cnt(dr.get('uphold', 0))} / {cnt(dr.get('override', 0))} / {cnt(dr.get('missing', 0))}",
            "stage3_run_summary.json decision_rule_counts")
    out.row("model/rule agreement (decided)", pct(run["model_rule_agreement_among_decided"]),
            "stage3_run_summary.json model_rule_agreement_among_decided")
    out.row("all-quotes-verified (decided)", pct(run["all_quotes_verified_share_among_decided"]),
            "stage3_run_summary.json all_quotes_verified_share_among_decided")
    out.row("note_truncated count / share",
            f"{cnt(run['note_truncated_count'])} / {pct(run['note_truncated_share'])}",
            "stage3_run_summary.json note_truncated_*")

    out.sub(f"Validation controls (n = {cnt(vc['n_sample'])}, stage3_validation_controls_summary.json)")
    d1 = vc["d1_blind_note"]["vs_reference_arm"]
    d2 = vc["d2_no_stage2"]["vs_reference_arm"]
    ref = d2["real_distribution"]
    arm = d2["no_stage2_distribution"]
    order = ("uphold", "override", "insufficient_evidence")
    out.row("Reference arm: uphold / override / insufficient",
            " / ".join(cnt(ref.get(k, 0)) for k in order),
            "stage3_validation_controls_summary.json d2_no_stage2.vs_reference_arm.real_distribution")
    out.row("Stage-2-withheld arm: uphold / override / insufficient",
            " / ".join(cnt(arm.get(k, 0)) for k in order),
            "stage3_validation_controls_summary.json d2_no_stage2.vs_reference_arm.no_stage2_distribution")
    out.row("Stage-2-withheld agreement vs reference", pct(d2["decision_model_agreement_vs_real"]),
            "stage3_validation_controls_summary.json d2_no_stage2.vs_reference_arm."
            "decision_model_agreement_vs_real")
    out.row("Note-withheld arm: insufficient_evidence",
            f"{cnt(d1['blind_note_distribution'].get('insufficient_evidence', 0))}/{cnt(d1['n'])}",
            "stage3_validation_controls_summary.json d1_blind_note.vs_reference_arm.blind_note_distribution")
    out.row("Note-withheld agreement vs reference", pct(d1["decision_model_agreement_vs_real"]),
            "stage3_validation_controls_summary.json d1_blind_note.vs_reference_arm."
            "decision_model_agreement_vs_real")
    out.row("Repeated-reference agreement (decision model / decision rule)",
            f"{pct(vc['d3_self_agreement']['decision_model_agreement'])} / "
            f"{pct(vc['d3_self_agreement']['decision_rule_agreement'])}",
            "stage3_validation_controls_summary.json d3_self_agreement")
    out.row(f"Batch-size check (n = {bs['n']}): decision / grounds agreement",
            f"{pct(bs['decision_model_agreement'])} / {pct(bs['grounds_agreement'])}",
            "batch_size_comparison.json")
    out.note("Reference-arm-vs-main-run agreement (145/150, 96.7%) is NOT in any committed JSON: "
             "it was measured on the cluster (validation job 17462868, "
             "scripts/run_validation_controls.py) and the per-admission control CSVs are "
             "gitignored. Source documented in results/README.md. The same holds for the "
             "Stage 3 acceptance arm test (first 8 admissions, original batch grouping, passed "
             "8/8; results/README.md, MODEL_CARD.md).")


def _policy_row(out: Out, label: str, m: dict, n: int, src: str) -> None:
    """One RQ3 row: alerts, alert rate, precision [CI], recall [CI], F1, F2."""
    a = alerts(m)
    out.row(f"{label}: alerts / alert rate", f"{cnt(a)} / {pct(a / n)}",
            f"{src} (derived: tp+fp; / n)")
    out.row(f"{label}: precision [95% CI] / recall [95% CI]",
            f"{ci(m['precision_ci'])} / {ci(m['recall_ci'])}", f"{src}.precision_ci / recall_ci")
    out.row(f"{label}: F1 / F2", f"{metric(m['f1'])} / {metric(m['f2'])}", f"{src}.f1 / f2")


def _sec_rq3(out: Out, d: dict) -> None:
    """tab:rq3-main + tab:rq3-age + paired differences."""
    pp = d["pipeline_evaluation"]["pipeline"]
    fc, ctrl = pp["full_cohort"], pp["control_arm_stage1_matched"]
    s1 = ctrl["secondary_threshold_arm_calibrated"]
    cond, sec = pp["conditional_triggering"], pp["secondary_both_agree"]
    pdiff = d["pipeline_paired_differences"]["comparisons"]

    out.h("tab:rq3-main — pipeline vs control at matched volume")
    _policy_row(out, "Stage 1 alone (primary threshold 0.3235)", s1, s1["n"],
                f"{PE} pipeline.control_arm_stage1_matched.secondary_threshold_arm_calibrated")
    _policy_row(out, "Full pipeline", fc, fc["n"], f"{PE} pipeline.full_cohort")
    _policy_row(out, "Control arm (Stage 1 at matched volume)", ctrl, ctrl["n"],
                f"{PE} pipeline.control_arm_stage1_matched")
    out.row("Control arm: realised alert count (JSON key)", cnt(ctrl["realised_alert_count"]),
            f"{PE} pipeline.control_arm_stage1_matched.realised_alert_count")
    _policy_row(out, "Notes-cohort evaluation population (secondary row)", pp["notes_cohort"],
                pp["notes_cohort"]["n"], f"{PE} pipeline.notes_cohort")
    _policy_row(out, "Conditional triggering (discordant only)", cond["report"], cond["report"]["n"],
                f"{PE} pipeline.conditional_triggering.report")
    out.row("Conditional LLM calls (blanket / conditional / saved)",
            f"{cnt(cond['blanket_llm_calls'])} / {cnt(cond['conditional_llm_calls'])} / "
            f"{cnt(cond['llm_calls_saved'])}", f"{PE} pipeline.conditional_triggering")
    cc = pdiff["conditional_discordant_only_vs_control"]
    out.row("Conditional matched control: alerts",
            cnt(cc["control_alert_count"]),
            "pipeline_paired_differences.json comparisons.conditional_discordant_only_vs_control."
            "control_alert_count")
    out.row("Conditional minus its matched control: precision diff / recall diff",
            f"{ci(cc['precision_diff'], diff)} / {ci(cc['recall_diff'], diff)}",
            "pipeline_paired_differences.json comparisons.conditional_discordant_only_vs_control")
    out.note("The conditional policy's matched control is reported by its alert count and the "
             "paired difference only; its absolute precision/recall/F1/F2 are not stored in any "
             "committed JSON (team decision, option A).")
    _policy_row(out, "Secondary policy (withdraw only if both override)", sec["report"],
                sec["report"]["n"], f"{PE} pipeline.secondary_both_agree.report")
    out.row("Secondary policy: alerts withdrawn", cnt(sec["n_alerts_withdrawn"]),
            f"{PE} pipeline.secondary_both_agree.n_alerts_withdrawn")
    _policy_row(out, "Secondary policy matched control", sec["control_arm_stage1_matched"],
                sec["control_arm_stage1_matched"]["n"],
                f"{PE} pipeline.secondary_both_agree.control_arm_stage1_matched")

    out.sub("Paired pipeline-minus-control differences (pipeline_paired_differences.json)")
    for name, c in pdiff.items():
        out.row(f"{name}: precision diff / recall diff",
                f"{ci(c['precision_diff'], diff)} / {ci(c['recall_diff'], diff)}",
                f"pipeline_paired_differences.json comparisons.{name}")

    out.sub("tab:rq3-age — pipeline vs control arm by age band")
    for band, m in fc["by_age_band"].items():
        c = ctrl["by_age_band"][band]
        out.row(f"{band}: pipeline alerts / precision / recall",
                f"{cnt(alerts(m))} / {metric(m['precision'])} / {metric(m['recall'])}",
                f"{PE} pipeline.full_cohort.by_age_band.{band} (alerts derived: tp+fp)")
        out.row(f"{band}: control alerts / precision / recall",
                f"{cnt(alerts(c))} / {metric(c['precision'])} / {metric(c['recall'])}",
                f"{PE} pipeline.control_arm_stage1_matched.by_age_band.{band} (alerts derived: tp+fp)")


def _sec_planned_method(out: Out, d: dict) -> None:
    """tab:planned + Stage 1 method/feature facts + compute."""
    plan = d["planned_return_crosscheck"]
    ct = plan["cross_tab"]
    out.h(f"tab:planned — planned-return cross-check (n={cnt(plan['n_comparable'])})")
    out.row("n comparable", cnt(plan["n_comparable"]), "planned_return_crosscheck.json n_comparable")
    out.row("Agreement / majority baseline",
            f"{pct(plan['agreement_rate'])} / {pct(plan['majority_baseline_agreement'])}",
            "planned_return_crosscheck.json agreement_rate / majority_baseline_agreement")
    out.row("2x2 struct-planned (auditor no/yes)",
            f"{cnt(ct['structured=planned']['no'])} / {cnt(ct['structured=planned']['yes'])}",
            "planned_return_crosscheck.json cross_tab.structured=planned")
    out.row("2x2 struct-unplanned (auditor no/yes)",
            f"{cnt(ct['structured=unplanned']['no'])} / {cnt(ct['structured=unplanned']['yes'])}",
            "planned_return_crosscheck.json cross_tab.structured=unplanned")
    out.h("Stage 1 method facts (from code/artifact, not a JSON — tasks B2/B5)")
    out.row("Structured features used", str(N_FEATURES),
            "src/data/features.py build_features (17 base + 8 labs x 4 + 6 vitals)")
    out.row("Optuna trials / CV folds",
            f"{STAGE1_METHOD['optuna_trials']} / {STAGE1_METHOD['cv_folds']}",
            "config.yaml run.full + src/model/cv.py")
    out.row("Early stopping", STAGE1_METHOD["early_stopping_on"],
            "src/model/train.py _fit_final_estimator")
    out.row("Threshold selection", STAGE1_METHOD["threshold_selected_on"],
            "src/model/train.py _select_thresholds")
    out.row("Calibration", STAGE1_METHOD["calibration"], "src/model/train.py _calibrate")
    out.h("Compute (from sacct, not a JSON)")
    out.row("Total GPU-hours / jobs", f"{GPU_HOURS['total']} / {GPU_HOURS['jobs']}",
            "sacct via MODEL_CARD.md")
    out.row("Final Stage 3 run GPU-hours", str(GPU_HOURS["final_stage3"]), "sacct via MODEL_CARD.md")


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
    """Write THESIS_NUMBERS.md."""
    out = Out()
    out.md.append("# Thesis numbers — single source of truth for Chapter 4\n")
    out.md.append("Generated by `scripts/export_thesis_numbers.py` from committed JSONs only. "
                  "Every value is tagged with its source file and key; values marked "
                  "\"derived\" state their formula. Do not hand-edit.\n")
    build(out)
    docs = Path(__file__).resolve().parents[1] / "docs"
    (docs / "THESIS_NUMBERS.md").write_text("\n".join(out.md) + "\n")
    print(f"[export_thesis_numbers] wrote docs/THESIS_NUMBERS.md ({len(out.md)} lines)")


if __name__ == "__main__":
    main()
