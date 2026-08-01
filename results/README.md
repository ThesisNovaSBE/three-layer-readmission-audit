# Results

Committed aggregate metrics — **no patient data, no MIMIC identifiers**.

These files are version-controlled snapshots of model performance. They are safe
to commit because they contain only aggregate statistics (AUROC, recall, counts,
percentages), never row-level patient records.

| File | Stage | Description |
|------|-------|-------------|
| `stage1_metrics.json` | 1 | XGBoost evaluation on held-out test set |
| `stage2_evaluation.json` | 2 | Clinical-Longformer per-age-group metrics |
| `stage3_discordance_analysis.json` | 3 | Population-level discordance distribution |

## How to update

After each training run, copy the relevant file from `models/` here:

```bash
cp models/stage2_evaluation.json results/stage2_evaluation.json
cp models/stage3_discordance_analysis.json results/stage3_discordance_analysis.json
```

Then commit both. The `models/` directory versions may be overwritten by future
runs; the `results/` copies are the permanent record.
