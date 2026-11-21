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

**`pipeline_evaluation.json`** (added 2026-10-08, rerun merged
2026-11-21) — the real RQ1 + RQ3 result: Stage 1 alone, Stage 2 alone,
full pipeline, control arm, the conditional-triggering post-hoc
analysis, and the secondary both-override policy, all from the completed
full Stage 3 batch rerun (2026-11-21, all 9,899 flagged+noted
admissions). The headline: at matched alert volume, adding the full
cascade does not improve on Stage 1 alone — in the rerun Stage 1 is
nominally ahead on precision (0.444 vs 0.439) and recall (0.328 vs
0.324) but the 95% CIs overlap on both, so the clear recall gap seen in
the old run is gone. The paired pipeline-minus-control difference test
lives in `pipeline_paired_differences.json`
(`scripts/compare_pipeline_control_paired.py`). See `MODEL_CARD.md` for
the full writeup.

**`selection_bias_evaluation.json`** (added 2026-10-26) — does the
~60% of Stage 1's test partition with a discharge note (the population
RQ1–RQ3 are evaluated on) differ systematically from the ~40% without
one? Yes, demographically: the notes-covered group is older, sicker,
and far more often an emergency admission. No, in outcome rate: the
readmission rate itself is nearly balanced between the two groups
(SMD=0.053, below the conventional 0.1 imbalance threshold). See
`MODEL_CARD.md`'s "Selection bias" section.

**`planned_return_crosscheck.json`** (added 2026-10-26, rerun
2026-11-21) — does Stage 3's note-derived `planned_return` field agree
with the structured, `admission_type`-based "unplanned" proxy? Not
reassuringly: 93.4% raw agreement is still *below* the 95.8% a trivial
"always predict unplanned" baseline would score. In the rerun the
tighter `planned_return` definition brings the LLM's "planned" rate onto
the structured proxy's (4.2% vs 4.2%, down from 19.2% vs 4.2% in the old
run), but the two still disagree substantially on *which* admissions are
planned. Reported as an open, unresolved discrepancy, not evidence
either signal is simply wrong. See `MODEL_CARD.md`'s `planned_return`
section.

**`stage3_validation_controls_summary.json`** (RQ2) — the robustness
controls testing whether Stage 3 actually uses its inputs: four control
arms on 150 admissions (reference, repeated reference, note-withheld,
Stage-2-withheld), an acceptance arm test (first 8) and a batch-size check
(first 20, one at a time). **Results pending (validation job).** Must be
committed **without** its `sample_hadm_ids` list (MIMIC-IV-Note DUA — no
admission identifiers in tracked files); run
`python -m scripts.check_results_no_identifiers` on it before committing.
See `MODEL_CARD.md`'s RQ2 section.

**`discordance_sensitivity.json`** (added 2026-10-08) — the
`±10pp`–`±30pp` sweep of the discordance-mode threshold over the real,
completed Stage 3 run.

**`discordance_subgroup_evaluation.json`** (added 2026-10-08) — the
direct follow-up test the headline RQ3 result alone doesn't answer: Stage
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
