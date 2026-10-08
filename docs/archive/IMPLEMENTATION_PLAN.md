# Implementation & Training Plan
_Last updated: 2026-08-16 — generated after full methodological evaluation_

> **Superseded 2026-08-25.** The Stage 3 design described here (raw
> `stage1_score - stage2_score` discordance, ε=0.15) was found arithmetically
> fragile and replaced with percentile-rank displacement. See
> `docs/ARCHITECTURE.md` for the current design and what's actually
> implemented in code. Kept here for history.

---

## Research Questions (locked before testing)

| # | Question | Key evidence |
|---|----------|-------------|
| Q1 | Can the cascade improve alert precision while preserving recall? | `evaluate_pipeline.py` + N4 decision curve + bootstrap CIs |
| Q2 | Do clinical notes contain non-redundant readmission signal orthogonal to structured features? | N1 ablation — especially the "L1@stricter-threshold" arm |
| Q3 | What clinical phenomena explain the structured/narrative predictive gap? | N2 disagreement analysis + Stage 3 population-level discordance + κ validation |

---

## What Is Already Implemented

### Stage 1
- [x] `src/data/features.py` — fixed `admission_type_emergency` (maps MIMIC strings "EW EMER.", "DIRECT EMER."), added `admission_type_observation`, `is_medicaid_medicare`, `has_discharge_note`
- [x] `src/model/models.py` — expanded Optuna search space: `colsample_bylevel`, `colsample_bynode`, `max_delta_step`, `max_bin`; wider ranges on depth/regularization; 150→400 trials
- [x] `config.yaml` — `full.optuna_trials: 400`
- [x] `scripts/slurm_stage1_tune.sh` — GWDG Grete script (A100, 16 CPUs, 64GB, 8h); runs features → tune → train → evaluate → evaluate_pipeline
- [x] `src/model/evaluate_pipeline.py` — blind end-to-end evaluation on Stage 1 test partition; reports Stage 1 alone / Stage 2 alone / full pipeline; per-age-band breakdowns; saves `models/pipeline_evaluation.json`

### Stage 2
- [x] `src/stage2/_utils.py` — `STRUCT_FEATURE_COLS` (8 features — stage1_score removed for Stage 2 independence) and `CONTINUOUS_STRUCT_COLS` (5 features)
- [x] `src/stage2/model.py` — `FusionLongformer`: Longformer backbone (768-dim CLS) + structured MLP (→64-dim) → 832-dim → classifier; HuggingFace Trainer-compatible
- [x] `src/stage2/dataset.py` — head+tail truncation in `__getitem__`; `normalize_struct_features()` with z-score for continuous cols, pass-through for binary cols
- [x] `src/stage2/train.py` — updated for FusionLongformer; `_StructData` NamedTuple; `_make_struct_data()`; saves `fusion_weights.pt`, `stage2_fusion_config.json`, `stage2_struct_scaler.json`
- [x] `src/stage2/predict.py` — `_load_model()` checks for `stage2_fusion_config.json`, falls back to plain Longformer for backward compat; attaches struct features for fusion inference

---

## Still To Implement (in priority order)

### Blocks training — do first
- [ ] **Stage 2 SLURM script for KISSKI A100** (`scripts/slurm_stage2_train.sh`)
  - Partition: gpu-a100, 1 GPU, 32GB RAM, ~12h
  - Runs: `python -m src.stage2.train` → `python -m src.stage2.predict` → `python -m src.stage2.calibrate`
  - Email: lennartstenzel@gmail.com

- [ ] **C9 fallback in `evaluate_pipeline.py`** — for the 37% of L1-flagged patients with no discharge note, use L1 output as the final prediction. Report two rows in every pipeline table: "notes cohort" (63%) and "full cohort" (100%). Without this, "+21% precision" is misleading.

- [ ] **Stage 3 discordance fix** — currently phi4-mini classifies the mode; it should only explain it. The mode must be determined quantitatively:
  ```
  NOTE_MITIGATES  if stage1_score - stage2_score > ε  (suggest ε=0.15)
  NOTE_AMPLIFIES  if stage2_score - stage1_score > ε
  CONCORDANT      otherwise
  ```
  Rewrite the Stage 3 prompt: "Given that the structured model scored X and the note model scored Y — a {mode} divergence — explain what in the note drives this, and identify the clinical domain."

### Needed for ablation
- [ ] **N1 ablation runner** — the current plan is missing the most important arm:
  - arm_1: Stage 1 only (existing)
  - arm_2: Notes-only Longformer on all patients (existing plan)
  - arm_3: Full cascade L1→L2 (existing)
  - arm_4: Late score fusion (logistic blend, existing plan)
  - **arm_5: Stage 1 at the threshold that matches cascade precision** ← MISSING, MOST IMPORTANT
  - arm_6: FusionLongformer on all patients (not gated by L1) ← nice to have

- [ ] **`src/stage2/optimize_thresholds.py`** — per-age-band threshold optimization after v2 trains; updates `stage2_calibration.json`

- [ ] **Bootstrap CIs (C4)** — add to all metric reporters; 1000 resamples; report 95% CI alongside every AUROC, precision, recall, F2

### After results
- [ ] **N2 disagreement analysis** — for NOTE_MITIGATES cases: extract note spans that differ from structured risk; categorize by domain (frailty, social support, discharge planning, palliative intent)
- [ ] **N4 decision curve analysis** — net benefit vs. treat-all, treat-none, and L1-only at threshold
- [ ] **OOF Stage 1 scores** — currently L2 trains on in-sample L1 scores (mild leakage). Either implement a 5-fold OOF pass to produce unbiased training-set L1 scores, or document as a limitation and quantify the calibration gap between training-set and test-set L1 score distributions.

---

## Training Sequence (exact order)

```
STEP 0 — PREP (local)
  # Feature matrix must be rebuilt — new features were added
  rm -f data/processed/features.csv
  python -m src.data.features

STEP 1 — STAGE 1 TUNE + TRAIN (GWDG Grete)
  sbatch scripts/slurm_stage1_tune.sh
  # This runs: features → tune (400 trials) → train → evaluate → evaluate_pipeline
  # ~8h wall time on A100
  # Note: evaluate_pipeline will run but Stage 2 results won't exist yet — that's OK

STEP 2 — [WHILE STAGE 1 RUNS] WRITE THESIS
  # Chapters 1, 2, 3 + data section do not depend on results
  # See thesis writing plan below

STEP 3 — STAGE 2 TRAIN (KISSKI)
  # Write slurm_stage2_train.sh first
  sbatch scripts/slurm_stage2_train.sh
  # This runs: python -m src.stage2.train → predict → calibrate
  # ~12h wall time on A100

STEP 4 — FULL PIPELINE EVALUATION (local or KISSKI)
  python -m src.model.evaluate_pipeline
  # Now both Stage 1 and Stage 2 results exist → full blind evaluation

STEP 5 — N1 ABLATION (local or KISSKI)
  # Run all 5 arms including "L1@stricter-threshold"
  # arm_5 can run BEFORE Step 3 — it only needs the Stage 1 artifact

STEP 6 — THRESHOLD OPTIMIZATION
  python -m src.stage2.optimize_thresholds
  # Updates stage2_calibration.json with per-band thresholds
  # Re-run evaluate_pipeline after this

STEP 7 — N2 DISAGREEMENT ANALYSIS
  # Requires Step 4 results + the Stage 2 confirm/reject labels
  # Script to be written

STEP 8 — STAGE 3 (with fixed discordance mode)
  python -m src.stage3.predict   # or however it's invoked
  # Fix discordance mode BEFORE running this — see "Still to implement" above

STEP 9 — METRICS + STATISTICS
  # Bootstrap CIs on all reported metrics
  # Decision curve analysis (N4)
  # κ validation against human annotators for Stage 3 discordance classification
```

---

## Methodological Fixes Required Before Reporting Results

1. **Discordance mode** — fix Stage 3 before any L3 evaluation (see above)
2. **Missing ablation arm** — add L1@stricter-threshold to N1 before calling ablation complete
3. **Denominator integrity (C9)** — every pipeline table must show full-cohort numbers alongside notes-cohort numbers
4. **Bootstrap CIs** — all metrics need CIs before appearing in the thesis results chapter
5. **Calibration gap** — quantify the difference between training-set and test-set L1 score distributions (takes one histogram plot); use to argue the OOF leakage risk is small, or implement OOF fix

---

## Thesis Writing Plan (parallel to training)

### Write NOW — no results needed
- **Chapter 1: Introduction** — gap, motivation, Q1/Q2/Q3 stated as *hypotheses* (not findings)
- **Chapter 2: Literature Review** — readmission prediction models (plateau at 0.70), alert fatigue, multimodal EHR integration, clinical NLP and Longformer, LLMs in clinical AI, fairness in clinical prediction
- **Chapter 3: Methodology** — three-layer architecture, FusionLongformer design and rationale, head+tail truncation rationale (cite discharge note structure literature), data pipeline, patient-level splitting, fairness-weighted training
- **Data section** — MIMIC-IV cohort description, inclusion/exclusion, feature engineering, cohort statistics, note coverage (63%)

### Write after Stage 1 results (Step 1 done)
- Stage 1 results: AUROC, AUPRC, precision/recall at threshold, feature importance, age-band breakdown

### Write after Stage 2 v2 + ablation (Steps 3–6 done)
- Stage 2 results: AUROC on test partition, pipeline precision/recall/F2
- Ablation table (all 5 arms)
- Fairness evaluation: age-band recall before/after fairness retraining
- Pipeline evaluation: full-cohort and notes-cohort rows

### Write after N2 + Stage 3 (Steps 7–8 done)
- Disagreement analysis: NOTE_MITIGATES case characterization
- Stage 3 discordance distribution (population-level finding)
- κ validation result

### Write last
- **Chapter 5: Discussion** — what Q1/Q2/Q3 results mean; limitations (OOF scores, distribution shift, attention faithfulness, 37% note-less); future work
- **Conclusion**

---

## Open Backlog (for reference)

| ID | Item | Blocked on |
|----|------|-----------|
| C1 | Reproducible end-to-end run + results manifest | Stage 2 v2 training |
| C2 | Stage 2 GPU training (KISSKI) | SLURM script (write first) |
| C4 | Bootstrap CIs, DeLong test, calibration curves | Stage 2 v2 results |
| C5 | SHAP interpretability for Stage 1 | Stage 1 retraining |
| C6 | Feature audit (vitals missingness, itemid validation) | Nothing |
| C9 | Cascade policy for note-less patients + full-cohort reporting | evaluate_pipeline.py update |
| N1 | Ablation grid (add L1@stricter-threshold arm) | Stage 2 v2 (partial: arm_5 can run now) |
| N2 | Disagreement analysis | Stage 2 v2 + evaluate_pipeline |
| N4 | Decision-curve / clinical-utility analysis | Stage 2 v2 results |
