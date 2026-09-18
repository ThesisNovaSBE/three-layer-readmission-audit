# Model Card — Readmission Prediction Pipeline

> **See `docs/ARCHITECTURE.md` for the current design.** Stage 1 and Stage 2
> below are both the real 2026-09 retrains — Stage 1 (2026-09-05): 400-trial
> Optuna search, capacity-constrained threshold, isotonic calibration,
> **unplanned** readmission as the target (switched from all-cause the same
> day, see `docs/ARCHITECTURE.md` §6) on the full real MIMIC-IV dataset
> (n=521,191). Stage 2 (2026-09-09/10): retrained under the corrected
> unplanned-label + rebalanced age-group targets + 4096-token window,
> including a calibration-staleness bug found and fixed mid-session (see
> `sessions/2026-09-13_session-23.md`) — the numbers below are from
> *after* that fix, not the initial (miscalibrated) run. Both on GWDG
> KISSKI (A100 80GB). Stage 3 has switched from Ollama/phi4-mini to
> MedGemma-27B-text-it (2026-09-10), served via plain HF transformers +
> lm-format-enforcer (2026-09-15, after vLLM proved structurally
> incompatible with KISSKI's CUDA 12.8 driver ceiling) — matches the
> current code, but has only been smoke-tested at small scale, not run at
> full scale yet.

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features; isotonic-calibrated (since 2026-08-26); capacity-constrained operating point (primary, since 2026-08-25) with recall-floor kept as a secondary comparison table. Two label variants exist in the feature matrix, `readmission_30d` (all-cause) and `readmission_30d_unplanned` (excludes outcome admissions with a planned `admission_type`; added 2026-08-26) — **the model's actual target is `readmission_30d_unplanned`** as of 2026-09-05 (`MODEL_TARGET_COL` in `src/schemas.py`), matching this project's stated scope; every model trained before that date, including the original artifact, silently used all-cause instead
- **Stage 2:** Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`), note-only (no structured features) — 4096-token context (raised from 2048 on 2026-08-25), trained on real MIMIC-IV-Note discharge summaries; produces an independent, note-based risk estimate, not a gate on Stage 1's flag. A jointly-trained structured+note "FusionLongformer" variant was built and dropped on 2026-08-26 without ever completing a training run — see `docs/ARCHITECTURE.md` §2.
- **Stage 3:** Independent LLM audit via HF `transformers.generate()` + `lm-format-enforcer` (`google/medgemma-27b-text-it`, temperature=0, switched from Ollama/phi4-mini 2026-09-10 once the batch run moved to cluster GPU hardware, then from vLLM 2026-09-15 after KISSKI's CUDA 12.8 driver ceiling proved incompatible with vLLM's kernel stack) — reaches its own uphold/override decision rather than explaining a decision Stage 2 already made
- **Developed by:** Nova SBE thesis team (M.Sc. Business Analytics)
- **Model type:** Three-layer LLM-auditing classification pipeline
- **Language:** English (clinical notes)

## Intended Use

- **Primary use:** Predict unplanned 30-day hospital readmissions from MIMIC-IV data
- **Primary users:** Clinical decision support research
- **Out of scope:** Direct clinical deployment without further validation

## Training Data

- MIMIC-IV v3.1 (structured tables) — credentialed access via PhysioNet
- MIMIC-IV-Note (discharge summaries) — credentialed access via PhysioNet
- Population: Adult patients (age >= 18), excluding in-hospital deaths and elective readmissions

## Metrics

### Stage 1 — XGBoost on structured MIMIC-IV 3.1 (full mode, n=521,191, held-out test set, target=unplanned readmission)

Held-out test set: n=104,242, base rate 19.0%. Primary operating point is
capacity-constrained (top 15% by score), not a recall floor — see
`docs/ARCHITECTURE.md` §5 for why. Recall-floor rows kept below as the
secondary, literature-comparable view.

| Metric | Value |
|--------|-------|
| AUROC | 0.7215 [0.7147, 0.7293] (95% CI, patient-level bootstrap) |
| AUPRC | 0.3965 [0.3735, 0.4252] (95% CI) |
| Brier score | 0.1370 |
| Recall @ thr=0.3235 (capacity=15%) | 0.352 [0.3343, 0.3735] |
| Precision @ thr=0.3235 | 0.431 [0.4156, 0.4494] |
| Specificity @ thr=0.3235 | 0.891 |
| F2 @ thr=0.3235 | 0.366 |
| TP / FP / TN / FN | 6,989 / 9,223 / 75,186 / 12,844 |

Capacity trade-off: K=5% → precision=0.566, recall=0.149, lift=2.97x.
K=10% → precision=0.472, recall=0.267, lift=2.48x. K=20% → precision=0.401,
recall=0.423, lift=2.11x.

Recall-floor, secondary (for comparability with prior literature):
recall≥0.80 → precision=0.268 @ thr=0.1359. recall≥0.85 → precision=0.247 @
thr=0.1175. recall≥0.90 → precision=0.233 @ thr=0.1021.

Beats 3 of 4 published AUROC benchmarks cited in `evaluate.py`: LACE index
(0.694, +0.028), Xiao 2018 EHR baseline (0.715, +0.007), Fraccaro 2016 notes
baseline (0.684, +0.038); behind Rajkomar 2018 deep EHR (0.773, −0.051).

Subgroup AUROC (fairness): Female=0.718 (n=54,512, pos_rate=17.7%),
Male=0.724 (n=49,730, pos_rate=20.5%) — near-equal. Age: 18–40=0.750,
41–55=0.737, 56–70=0.728, **70+=0.665** — elderly patients are
meaningfully harder to predict, and their recall at the primary operating
point is only 19.0% vs. 35–45% for the other three bands. This is the
documented "v1 recall gap" that Stage 2's age-group oversampling exists to
address — see `config.yaml`'s `stage2.age_group_train_targets` comment.

### Stage 2 — Clinical-Longformer (real retrain, completed 2026-09-09/10, target=unplanned readmission)

Trained on 141,767 age-stratified notes (real per-band availability came in
lower than the config's targets across the board — every positive-label
cell hit the "use all available" fallback; see `config.yaml`'s
`age_group_train_targets` comment and `sessions/2026-09-13_session-23.md`).
`EarlyStoppingCallback` (patience=3) stopped training at ~1.02 of the
configured 10 epochs once validation AUPRC peaked — Val AUPRC=0.372,
AUROC=0.704. A calibration-staleness bug (reused a pre-existing calibration
file without checking it matched this model) was found and fixed the same
session; numbers below are post-fix.

**Test-set evaluation (notes cohort, n=9,899 Stage 1-flagged-with-note admissions):**

| Metric | Value |
|--------|-------|
| AUROC | 0.622 |
| AUPRC | 0.538 |
| Recall | 0.974 |
| Precision | 0.437 |
| F2 | 0.782 |
| ECE | 0.1957 |

Per-age-group:

| Band | N | AUROC | Recall | Precision | F2 | ECE |
|------|---|-------|--------|-----------|-----|-----|
| 18-40 | 1,570 | 0.626 | 0.956 | 0.507 | 0.812 | 0.2028 |
| 41-55 | 2,836 | 0.617 | 0.975 | 0.437 | 0.782 | 0.2324 |
| 56-70 | 3,514 | 0.632 | 0.983 | 0.432 | 0.783 | 0.1879 |
| 70+   | 1,979 | 0.581 | 0.974 | 0.392 | 0.751 | 0.1543 |

Fairness gaps: recall gap 0.027 (worst: 18-40), precision gap 0.115 (worst:
70+), AUROC gap 0.051. 70+ remains the hardest band for Stage 2 too, though
the recall gap specifically is small — the age-group oversampling fix
appears to have helped recall parity even though precision/AUROC gaps
remain, worth digging into further for the fairness discussion.

**Training config:**

| Parameter | Value |
|-----------|-------|
| Training notes | 141,767 (real availability; see note above) |
| GPU | NVIDIA A100-SXM4-80GB (GWDG KISSKI) |
| Precision | bf16 |
| Batch size | 8 (effective 16 with grad. accum. ×2) |
| Gradient checkpointing | disabled (80 GB VRAM sufficient) |
| Sequence length | 4096 tokens (raised from 2048 on 2026-08-25) |

### RQ1 — Does the note text add signal beyond structured data?

`compare_layers.py` scores Stage 1 and Stage 2 **independently, on the
identical 62,759-admission notes-covered population** (neither model gates
or feeds the other) — the fair, apples-to-apples comparison this project's
own docs call for:

| Model | AUROC |
|-------|-------|
| Stage 1 (structured) | 0.7093 |
| Stage 2 (notes-only) | 0.7101 |
| Difference | +0.0008 [-0.0041, +0.0054] (95% CI) |

**Null result** — the CI straddles zero. This is explicitly anticipated and
reportable per this project's own design docs, not a failure of either
model: on this population, the discharge note alone carries no more (and no
less) predictive signal than the structured record alone.

### Stage 1+2 — Combined pipeline

| Metric | Value |
|--------|-------|
| Stage 1 alone (test n=104,242) | AUROC=0.7215, recall=0.352, precision=0.431 |
| Stage 2 alone (flagged+noted, n=9,899, 61.1% note coverage of flagged) | AUROC=0.622, recall=0.974, precision=0.437 |
| Pipeline, full cohort (n=104,242, C9 no-note fallback applied) | precision=0.437, recall=0.347, F1=0.387, F2=0.362 |
| Pipeline, notes cohort only (n=97,929) | precision=0.437, recall=0.241 |
| Control arm (Stage 1 alone @ matched 15.1% alert rate) | precision=0.431, recall=0.352, F1=0.388, F2=0.366 |

The control arm is nearly identical to the full pipeline here — but unlike
the previous (mismatched-vintage) version of this table, this is now a real
finding, not an artifact: the current cascade (Stage 1 → Stage 2 prune
only) does not yet beat matched-budget Stage 1 alone. This is expected and
incomplete, not a negative result to draw conclusions from yet — Stage 3
(the actual "auditor" layer this pipeline is designed around) hasn't run at
full scale yet. The real RQ2 answer is pending that.

## Stage 3 — Independent LLM Audit (MedGemma-27B via HF transformers + lm-format-enforcer)

Rewritten 2026-08-25, extended 2026-08-28. Switched serving from
Ollama/phi4-mini to `google/medgemma-27b-text-it` on 2026-09-10, once the
batch run moved to cluster GPU hardware, first via vLLM (its
guided/structured decoding preserved the schema-constrained JSON
generation Ollama's `format=` provided), then to plain HF
`transformers.generate()` + `lm-format-enforcer`'s
`prefix_allowed_tokens_fn` on 2026-09-15 — same schema-constrained-JSON
guarantee, different serving mechanism — after KISSKI's CUDA 12.8 driver
ceiling proved structurally incompatible with vLLM's flashinfer/CUTLASS
kernels regardless of vLLM/torch version (see `sessions/` for the full
diagnosis). MedGemma is the only evaluated candidate benchmarked directly
on MIMIC-IV-style reasoning; see `sessions/2026-09-13_session-23.md` and
`config.yaml`'s `stage3.model_name` comment for the full model comparison
and license check. Available both on-demand (one patient per call, via the
API) and in batch (`src/stage3/batch.py`, every Stage 1-flagged,
note-covered admission) — batch has only been smoke-tested at small scale
as of this writing, not run at full scale. For each patient, the model
receives:
- Stage 1's score + top-k SHAP-ranked structured risk factors
- Stage 2's independently-derived, note-based score
- A quantitatively pre-computed discordance mode (never chosen by the LLM)
- The discharge note itself (near-full text, ~20,000-char safety cap — not a
  5-sentence attention summary)

The model reaches its **own** independent decision — it is not asked to
narrate or classify a decision Stage 2 already made.

**Output per patient:**

| Field | Description |
|-------|-------------|
| `mitigating_grounds`, `aggravating_grounds` | Two-sided grounds the model extracted from the note, each with its own verified quote |
| `decision_model` | `uphold` / `override` / `insufficient_evidence` — the model's own judgment |
| `decision_rule` | The same three-way decision, recomputed deterministically in code from the extracted grounds — a consistency check, not a second model opinion |
| `all_quotes_verified` | True only if every extracted ground's quote was found verbatim in the note |
| `planned_return` | Independent yes/no/not_stated field on whether the note documents a scheduled return |
| `clinical_justification` | 2-4 sentence justification citing note content |
| `r1`, `r2`, `displacement`, `discordance_mode` | Quantitative context (percentile ranks + mode), computed before the LLM call |
| `note_truncated`, `model_name` | Logged per row for truncation/scale-comparison analysis |

**Grounds taxonomy** (fixed list; a ground outside it, or with an empty
quote, is a parse failure, not a new category):
- *Mitigating:* `palliative_intent` · `planned_return` ·
  `strong_discharge_support` · `structured_driver_contradicted`
- *Aggravating:* `lives_alone_no_support` · `no_followup_arranged` ·
  `functional_dependence` · `cognitive_impairment` · `nonadherence_risk` ·
  `unstable_at_discharge`

**Discordance mode** is computed from percentile-rank displacement of
stage1_score vs. stage2_score within the flagged+noted cohort — not raw
score subtraction, which was tried and rejected as fragile to unequal
calibration between the two model families (see `docs/ARCHITECTURE.md`).

**Research contribution:** No prior work in the literature review's
49-study systematic search uses an LLM as an independent auditor of another
model's output (as opposed to predictor, feature extractor, or explainer of
its own prediction). `src/stage3/batch.py:run_batch_audit` produces Stage 3
decisions across every Stage 1-flagged, note-covered admission — needed to
evaluate RQ2 (net reclassification vs. structured triage) and characterise
disagreement at scale — but has not yet been *run* at full scale. The
Stage 1/Stage 2 retrain that previously blocked this is complete; the
current blocker is the Stage 3 batch run itself (serving-mechanism
stabilization, see `sessions/`) — see `docs/ARCHITECTURE.md` §4.

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — per-age-band fairness analysis has been run (see Stage 1's subgroup AUROC/recall breakdown above and Stage 2's age-group oversampling); the documented "70+" recall gap is a real, disclosed limitation, not an unstarted analysis
