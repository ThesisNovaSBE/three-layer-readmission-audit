# Model Card — Readmission Prediction Pipeline

> **See `docs/ARCHITECTURE.md` for the current design.** Stage 1 below is the
> real 2026-09-05 retrain (400-trial Optuna search, capacity-constrained
> threshold, isotonic calibration, **unplanned** readmission as the target —
> switched from all-cause the same day, see `docs/ARCHITECTURE.md` §6) on
> the full real MIMIC-IV dataset (n=521,191) on GWDG KISSKI (A100 80GB).
> Stage 2 below is still the v1 checkpoint (pre-dates the unplanned-label
> switch, the corrected age-group oversampling targets, and the 4096-token
> window) — its retrain under the current config has not run yet. The
> Stage 3 section matches the current independent-auditor design already in
> code, but has not been run at full scale yet either.

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features; isotonic-calibrated (since 2026-08-26); capacity-constrained operating point (primary, since 2026-08-25) with recall-floor kept as a secondary comparison table. Two label variants exist in the feature matrix, `readmission_30d` (all-cause) and `readmission_30d_unplanned` (excludes outcome admissions with a planned `admission_type`; added 2026-08-26) — **the model's actual target is `readmission_30d_unplanned`** as of 2026-09-05 (`MODEL_TARGET_COL` in `src/schemas.py`), matching this project's stated scope; every model trained before that date, including the original artifact, silently used all-cause instead
- **Stage 2:** Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`), note-only (no structured features) — 4096-token context (raised from 2048 on 2026-08-25), trained on MIMIC-III; produces an independent, note-based risk estimate, not a gate on Stage 1's flag. A jointly-trained structured+note "FusionLongformer" variant was built and dropped on 2026-08-26 without ever completing a training run — see `docs/ARCHITECTURE.md` §2.
- **Stage 3:** Independent LLM audit via Ollama (`phi4-mini`, temperature=0) — reaches its own uphold/override decision rather than explaining a decision Stage 2 already made
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

### Stage 2 — Clinical-Longformer v1 (fine-tuned on 15k stratified notes, capped eval set of 3k)

> **Note:** metrics below are from the v1 checkpoint (completed 2026-08-01),
> which pre-dates the unplanned-label switch, the corrected age-group
> oversampling targets (2026-09-04 — the v1 run's 70+ oversampling was
> likely also broken the same way, since the underlying cause wasn't
> label-specific), and the current 4096-token window. A v2 retrain under
> the current config has not been run yet — this section will be updated
> once it completes.

| Metric | Value |
|--------|-------|
| AUROC | 0.6404 |
| AUPRC | 0.3411 |
| Best epoch | 2 / 5 (early stopping at epoch 4) |
| Training notes | 15,000 (stratified subsample; 21.1% positive) — v1 only |
| Eval notes (checkpoint selection) | 3,000 (stratified cap) — v1 only |

**Retraining config (v2, not yet run):**

| Parameter | Value |
|-----------|-------|
| Training notes | ~166,787 (recomputed 2026-09-04 against real per-band note availability — see `config.yaml`'s `age_group_train_targets` comment; previous ~249,000 figure was based on unachievable all-cause-era targets) |
| GPU | NVIDIA A100-SXM4-80GB (GWDG KISSKI) |
| Precision | bf16 |
| Batch size | 8 (effective 16 with grad. accum. ×2) |
| Gradient checkpointing | disabled (80 GB VRAM sufficient) |
| Sequence length | 4096 tokens (raised from 2048 on 2026-08-25) |

### Stage 1+2 — Combined pipeline

> **Not a valid long-term reference — transitional only.** The table below
> is `models/pipeline_evaluation.json` as of 2026-09-05: the *new* Stage 1
> (unplanned target, capacity threshold) combined with the *old* v1 Stage 2
> (all-cause-trained, pre-oversampling-fix). Combining a new-target Stage 1
> with an old-target Stage 2 is not a coherent long-term comparison — it's
> included here only because it's what the current committed artifacts
> actually produce, not as a claim about the pipeline's real performance.
> The previous version of this section (thr₁=0.354, thr₂=0.3 sweep) was
> itself from before the capacity-threshold/calibration changes and is
> superseded, not just outdated. **This whole section will be replaced**
> once Stage 2 is retrained under the current config.

| Metric | Value |
|--------|-------|
| Stage 1 alone (test n=104,242) | AUROC=0.7215, recall=0.352, precision=0.431 |
| Stage 2 alone (flagged+noted, n=8,786, 54.2% note coverage of flagged) | AUROC=0.663, recall=0.914, precision=0.274 |
| Pipeline, full cohort (n=104,242, C9 no-note fallback applied) | precision=0.431, recall=0.352, F1=0.388, F2=0.365 |
| Pipeline, notes cohort only (n=89,940) | precision=0.410, recall=0.057 |
| Control arm (Stage 1 alone @ matched 15.5% alert rate) | precision=0.431, recall=0.352, F1=0.388, F2=0.366 |

The control arm being nearly identical to the full pipeline here is an
artifact of the Stage 1/Stage 2 label mismatch above, not a real finding
about whether Stage 2 adds value — do not cite this as an RQ2 result.

## Stage 3 — Independent LLM Audit (phi4-mini)

Rewritten 2026-08-25, extended 2026-08-28. Available both on-demand (one
patient per call, via the API) and in batch (`src/stage3/batch.py`, every
Stage 1-flagged, note-covered admission). For each patient, phi4-mini
receives:
- Stage 1's score + top-k SHAP-ranked structured risk factors
- Stage 2's independently-derived, note-based score
- A quantitatively pre-computed discordance mode (never chosen by the LLM)
- The discharge note itself (near-full text, ~20,000-char safety cap — not a
  5-sentence attention summary)

phi4-mini reaches its **own** independent decision — it is not asked to
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
disagreement at scale — but has not yet been *run* at full scale, pending
the Stage 1/Stage 2 retrain; see `docs/ARCHITECTURE.md` §4.

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — fairness analysis is planned
