# Results

Committed aggregate metrics — **no patient data, no MIMIC identifiers**.

Files placed here are version-controlled snapshots of model performance. They
are safe to commit because they contain only aggregate statistics (AUROC,
recall, counts, percentages), never row-level patient records.

**`xgboost_metrics.json`** (added 2026-09-05) is the real Stage 1 retrain —
400-trial Optuna search, capacity-constrained threshold, isotonic
calibration, targeting unplanned readmission (AUROC 0.7215, AUPRC 0.3965 on
the held-out test set). See `MODEL_CARD.md` for the full breakdown.

Stage 2/Stage 3 snapshots are still absent. The previous Stage 2 snapshot
(`stage2_evaluation.json`, dated 2026-08-01) was from the pre-fairness-rebuild
model and was removed as stale (2026-08-28); a prior
`stage3_discordance_analysis.json` snapshot was removed earlier (session 16)
because its schema predated the current Stage 3 design (percentile-rank
displacement, uphold/override decision — see `docs/ARCHITECTURE.md` §2) and
no longer matched the code that would produce a new one. Add Stage 2's
snapshot once it's retrained under the current (unplanned-label,
corrected-oversampling) config — not the pre-2026-09-05 v1 checkpoint.

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
