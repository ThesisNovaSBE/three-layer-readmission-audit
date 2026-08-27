# Model Card — Readmission Prediction Pipeline

> **See `docs/ARCHITECTURE.md` for the current design.** The Stage 1/Stage 2
> metrics below are from before the 2026-08-25 session (recall-floor
> threshold, 2048-token window, no calibration) and are kept as the last
> known-good reference point — retraining under the new config
> (capacity-constrained threshold, 4096-token window, isotonic calibration)
> is pending. The Stage 3 section has been rewritten to match the new
> independent-auditor design, already in code.

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features; isotonic-calibrated (since 2026-08-26); capacity-constrained operating point (primary, since 2026-08-25) with recall-floor kept as a secondary comparison table. Two label variants available: `readmission_30d` (all-cause, primary/comparability) and `readmission_30d_unplanned` (excludes outcome admissions with a planned `admission_type`; added 2026-08-26, not yet used in any training/evaluation script)
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

### Stage 1 — XGBoost on structured MIMIC-IV 3.1 (full mode, n=521,191, held-out test set)

| Metric | Value |
|--------|-------|
| AUROC | 0.7058 |
| AUPRC | 0.4057 (base rate 20.2%) |
| Brier score | 0.2135 |
| Recall @ thr=0.354 | 0.848 |
| Precision @ thr=0.354 | 0.256 |
| Specificity @ thr=0.354 | 0.376 |
| F2 @ thr=0.354 | 0.580 |
| TP / FP / TN / FN | 17,864 / 51,911 / 31,249 / 3,212 |

Subgroup AUROC (fairness): Female=0.705, Male=0.704 (equal). Age: 18–40=0.723, 41–55=0.726, 56–70=0.706, 70+=0.660 (elderly patients harder to predict — noted as limitation).

### Stage 2 — Clinical-Longformer v1 (fine-tuned on 15k stratified notes, capped eval set of 3k)

> **Note:** A full retrain on 249k notes is running on GWDG KISSKI (A100 80GB, bf16) as of 2026-07-30.
> Metrics below are from the v1 checkpoint. This section will be updated once retraining completes.

| Metric | Value |
|--------|-------|
| AUROC | 0.6404 |
| AUPRC | 0.3411 |
| Best epoch | 2 / 5 (early stopping at epoch 4) |
| Training notes | 15,000 (stratified subsample; 21.1% positive) — v1 only |
| Eval notes (checkpoint selection) | 3,000 (stratified cap) — v1 only |

**Retraining config (v2, in progress):**

| Parameter | Value |
|-----------|-------|
| Training notes | ~249,000 (~99% of available MIMIC-IV-Note training data) |
| GPU | NVIDIA A100-SXM4-80GB (GWDG KISSKI) |
| Precision | bf16 |
| Batch size | 8 (effective 16 with grad. accum. ×2) |
| Gradient checkpointing | disabled (80 GB VRAM sufficient) |
| Sequence length | 2048 tokens |

### Stage 1+2 — Combined pipeline (notes cohort, thr₁=0.354, thr₂=0.3)

**Evaluation population:** Patients flagged by Stage 1 who have a discharge note (43,776 of 69,775 flagged).
In real clinical use every discharged patient has a note; the 37% gap is a MIMIC-IV-Note coverage limitation.
All Stage 1+2 recall figures are relative to the 11,186 positives *within the notes cohort*.

| Metric | Stage 1 on notes cohort | Stage 1+2 (thr₂=0.3) |
|--------|--------------------------|----------------------|
| Precision | 0.256 | 0.309 (+21%) |
| Recall (notes cohort) | 1.000 | 0.709 |
| F2 | 0.632 | 0.563 |
| Confirmed flags | 43,776 | 25,699 (58.7% retained) |

**Threshold sweep (notes cohort, n=43,776, positives=11,186):**

| thr₂ | Confirmed | Precision | Recall | F2 |
|------|-----------|-----------|--------|----|
| Stage 1 baseline | 43,776 | 0.256 | 1.000 | 0.632 |
| 0.2 | 37,078 | 0.275 | 0.912 | 0.623 |
| **0.3** | **25,699** | **0.309** | **0.709** | **0.563** |
| 0.4 | 16,739 | 0.345 | 0.516 | 0.469 |
| 0.5 | 6,300 | 0.410 | 0.231 | 0.253 |

**Operating point:** thr₂=0.3 — +21% precision over Stage 1 alone while retaining 70.9% of true positives in the notes cohort (F2=0.563 vs 0.632 baseline). thr₂=0.2 maximises F2 (0.623) with only a −8.8% recall cost if clinical completeness is the priority.

**Calibration note (v1):** Stage 2 scores cluster near the 0.5 boundary (no confirmed cases above 0.6), indicating underconfidence. This is attributable to training on only 15k notes (6% of available data). The v2 retrain on 249k notes on A100 80GB is expected to resolve this; per-group Platt scaling is applied post-training regardless.

## Stage 3 — Independent LLM Audit (phi4-mini)

Rewritten 2026-08-25. Stage 3 is on-demand (one patient per call, via the
API), not yet run in batch. For each patient, phi4-mini receives:
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
| `decision` | `uphold` / `override` — phi4-mini's own judgment |
| `primary_clinical_domain` | Dominant clinical domain behind the decision |
| `clinical_justification` | 2-4 sentence justification citing note content |
| `r1`, `r2`, `displacement`, `discordance_mode` | Quantitative context (percentile ranks + mode), computed before the LLM call |

**Clinical domain taxonomy:**
`social_support` · `frailty` · `palliative_intent` · `care_coordination` ·
`clinical_trajectory` · `other`

**Discordance mode** is computed from percentile-rank displacement of
stage1_score vs. stage2_score within the flagged+noted cohort — not raw
score subtraction, which was tried and rejected as fragile to unequal
calibration between the two model families (see `docs/ARCHITECTURE.md`).

**Research contribution:** No prior work in the literature review's
49-study systematic search uses an LLM as an independent auditor of another
model's output (as opposed to predictor, feature extractor, or explainer of
its own prediction). A batch runner producing Stage 3 decisions across the
full test partition — needed to evaluate RQ2 (net reclassification vs.
structured triage) and RQ3 (disagreement characterisation) at scale — does
not exist yet; see `docs/ARCHITECTURE.md` §4.

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — fairness analysis is planned
