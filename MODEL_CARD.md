# Model Card — Readmission Prediction Pipeline

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features
- **Stage 2:** Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`) on discharge notes — 4096-token context, trained on MIMIC-III
- **Stage 3 (optional):** Local generative explanation via Ollama (`phi4-mini`)
- **Developed by:** Nova SBE thesis team (M.Sc. Business Analytics)
- **Model type:** Two-stage triage-and-verify classification pipeline
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

## Stage 3 — Cross-Modal Discordance Analysis (phi4-mini)

Stage 3 processes **all** Stage 1 flagged patients (confirmed and rejected by Stage 2).
For each patient, phi4-mini receives:
- Top-k SHAP features from Stage 1 (structured risk drivers)
- Top-n sentences from the discharge note with highest Clinical-Longformer attention weight
- Stage 2 decision and probability

**Output per patient:**

| Field | Description |
|-------|-------------|
| `discordance_mode` | `CONCORDANT` / `NOTE_MITIGATES` / `NOTE_AMPLIFIES` |
| `primary_category` | Dominant clinical domain driving any discordance |
| `explanation` | One-sentence clinician-facing summary |

**Discordance category taxonomy:**
`social_support` · `discharge_planning` · `functional_status` · `frailty_markers` ·
`medication_adherence` · `housing_social_risk` · `care_complexity` · `cognition` ·
`structured_confirmed`

**Research contribution:** The population-level aggregation of discordance modes and categories across all patients produces an empirical finding on which clinical domains in discharge notes explain the predictive gap between structured EHR data and free-text. This is a novel result enabled by the three-stage pipeline architecture.

*Stage 3 results pending Stage 2 retraining on 249k notes (A100 80GB, in progress).*

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — fairness analysis is planned
