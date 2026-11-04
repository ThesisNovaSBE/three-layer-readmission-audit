# Results

Committed aggregate metrics — **no patient data, no MIMIC identifiers**.

Files placed here are version-controlled snapshots of model performance. They
are safe to commit because they contain only aggregate statistics (AUROC,
recall, counts, percentages), never row-level patient records.

**`xgboost_metrics.json`** (added 2026-09-05) is the real Stage 1 retrain —
400-trial Optuna search, capacity-constrained threshold, isotonic
calibration, targeting unplanned readmission (AUROC 0.7215, AUPRC 0.3965 on
the held-out test set). See `MODEL_CARD.md` for the full breakdown.

**`stage2_evaluation.json`** (added 2026-10-08, real retrain from
2026-09-09/10) — the current Clinical-Longformer, unplanned-label +
corrected age-group oversampling + 4096-token window. Supersedes a
2026-08-01 pre-fairness-rebuild snapshot that was removed as stale
2026-08-28.

**`pipeline_evaluation.json`** (added 2026-10-08, updated 2026-10-26) —
the real RQ1/RQ2 result: Stage 1 alone, Stage 2 alone, full pipeline,
control arm, and the conditional-triggering post-hoc analysis, all from
the completed full Stage 3 batch run (2026-10-06, all 9,899
flagged+noted admissions). The headline: at matched alert volume, Stage
1 alone beats the full cascade. The 2026-10-26 update added bootstrap
confidence intervals to the `notes_cohort` and `conditional_triggering`
arms (previously only `full_cohort` and `control_arm` had them) — fixed
alongside a real O(n²)-ish performance bug in `src/model/bootstrap.py`
that made this computation impractically slow before the fix. See
`MODEL_CARD.md` for the full writeup.

**`selection_bias_evaluation.json`** (added 2026-10-26) — does the
~60% of Stage 1's test partition with a discharge note (the population
RQ1/RQ2 are evaluated on) differ systematically from the ~40% without
one? Yes, demographically: the notes-covered group is older, sicker,
and far more often an emergency admission. No, in outcome rate: the
readmission rate itself is nearly balanced between the two groups
(SMD=0.053, below the conventional 0.1 imbalance threshold). See
`MODEL_CARD.md`'s "Selection bias" section.

**`planned_return_crosscheck.json`** (added 2026-10-26) — does Stage
3's note-derived `planned_return` field agree with the structured,
`admission_type`-based "unplanned" proxy? Not reassuringly: 81.7% raw
agreement is actually *below* the 95.8% a trivial "always predict
unplanned" baseline would score, since Stage 3 calls "planned" 4.5x
more often than the structured data supports. Reported as an open,
unresolved discrepancy, not evidence either signal is simply wrong. See
`MODEL_CARD.md`'s `planned_return` section.

**`stage3_validation_controls_summary.json`** (added 2026-10-30) — the
D1/D2/D3 robustness controls testing whether Stage 3 actually uses its
inputs, long open since session 19 (2026-08-28) and finally executed
this week. Blind-note control: the auditor answered
"insufficient_evidence" on 150/150 admissions with the note withheld,
versus 0/150 in the real run — the strongest available evidence the
auditor is genuinely grounded in the note. No-Stage-2 control: 82.5%
decision agreement. Self-agreement at `temperature=0`: 100% across two
independent runs of every admission. See `MODEL_CARD.md`'s "Is Stage 3
actually using its inputs?" section for the full writeup.

**`discordance_sensitivity.json`** (added 2026-10-08) — the
`±10pp`–`±30pp` sweep of the discordance-mode threshold over the real,
completed Stage 3 run.

**`discordance_subgroup_evaluation.json`** (added 2026-10-08) — the
direct follow-up test the headline RQ2 result alone doesn't answer: Stage
3's real decision vs. "Stage 1's flag stands," computed separately within
the concordant and discordant subgroups. By F2, Stage 3 underperforms
Stage 1 alone in **both** subgroups, more so on discordant cases — see
`MODEL_CARD.md` and `scripts/analyze_discordance_subgroups.py`.

A prior `stage3_discordance_analysis.json` snapshot was removed earlier
(session 16) because its schema predated the current Stage 3 design
(percentile-rank displacement, uphold/override decision — see
`docs/ARCHITECTURE.md` §2) and no longer matched the code that would
produce a new one; no replacement was needed once `pipeline_evaluation.json`
and `discordance_sensitivity.json` above covered the same ground under the
current design.

## How to update

After a training/evaluation run produces a metrics file in `models/` (e.g.
`models/stage2_evaluation.json`, `models/stage1_metrics.json`) that you want
as a permanent, citable snapshot, copy it here and commit it:

```bash
cp models/<file>.json results/<file>.json
```

The `models/` directory version may be overwritten by future runs; the
`results/` copy is the permanent record — so only copy a file here once
you're confident its schema and numbers are current (check
`docs/ARCHITECTURE.md` first).
