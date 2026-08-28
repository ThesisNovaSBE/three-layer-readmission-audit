# Stage 1 Modeling Plan — Structured Readmission Prediction

> Scope: Stage 1 only — predict unplanned 30-day readmission from the **structured** tables of MIMIC-IV. Clinical notes (Stage 2) are out of scope here.
> Target hardware: MacBook Pro 14" (Apple Silicon, M-5). All training runs locally and fast (CPU, multi-core).
> Read `PROJECT_TLDR.md` and the latest `sessions/` entry first.

---

## 1. What the literature does (and what it implies for us)

**Readmission is a rare, imbalanced event.** Reported 30-day all-cause rates land around **2–6% positive** depending on cohort definition; heart-failure / disease-specific cohorts run higher. This dominates every design choice below.

**Performance ceilings are modest.** Classical ML on MIMIC structured data typically reports **AUROC ≈ 0.65–0.72** and **low AUPRC (≈ 0.10–0.20)**. We should not expect AUROC > 0.75 from structured data alone — closing that gap is precisely the motivation for the Stage 2 notes model. Set expectations accordingly in the thesis.

**Gradient-boosted trees are the consensus winner** on tabular EHR. XGBoost / LightGBM / HistGradientBoosting consistently match or beat logistic regression and random forests, with logistic regression retained as the interpretable baseline.

**Standard imbalance handling:**
- XGBoost: `scale_pos_weight = n_negative / n_positive`, and `eval_metric="aucpr"` (AUPRC is the right early-stopping/selection signal under imbalance).
- Tree models: `class_weight="balanced"` (where supported) or sample weights.
- Threshold tuning *after* training is what actually delivers high recall — not resampling alone.

**Feature families that recur:** the **LACE** components (Length of stay, Acuity/admission type, Comorbidities, ED visits), demographics, prior-utilization counts, length of stay, and last/aggregate lab & vital values. Comorbidity indices (Charlson / Elixhauser) are common and high-value.

**Leakage control is non-negotiable:** split by **`subject_id`** (patient), never by admission, because one patient has many admissions. Use **`StratifiedGroupKFold`** (group = `subject_id`, stratify = label).

Sources at the bottom.

---

## 2. Modeling plan

### 2.1 Algorithms to compare (3)

| Model | Role | Why |
|-------|------|-----|
| **Logistic Regression** | Interpretable baseline | Coefficients = transparent risk factors; required reference point. Use L2, `class_weight="balanced"`, scaled features. |
| **XGBoost** | Primary challenger | Strongest on tabular EHR; native imbalance & missing-value handling; `hist` tree method is fast on CPU. |
| **HistGradientBoosting** (sklearn) | Lightweight GBT | No extra C++ deps, very fast on Apple Silicon CPU, native NaN handling, `class_weight="balanced"`. Good robustness check against XGBoost. |

### 2.2 Features to engineer from MIMIC-IV structured tables

From `admissions`, `patients`, `labevents` / `chartevents` (or `omr`), `diagnoses_icd`:

- **Demographics:** age at admission, gender, (optionally race/insurance — flag for fairness analysis, don't necessarily model).
- **Index-admission characteristics:** length of stay (days), admission type (emergency flag), admission location, discharge location, time of admission (day/night, weekend).
- **Prior utilization:** number of prior admissions, number of prior ED visits, days since last discharge.
- **Comorbidity burden:** Charlson and/or Elixhauser index from `diagnoses_icd`; optionally a handful of high-signal condition flags (CHF, COPD, diabetes, renal failure).
- **Labs (last value + optionally min/max/mean during stay):** glucose, creatinine, hemoglobin, WBC, platelets, sodium, potassium, bicarbonate (already in `config.yaml`).
- **Vitals (last value during stay):** heart rate, systolic/diastolic BP, temperature, respiratory rate, SpO2.
- **Counts:** number of distinct diagnoses, number of medications (if `prescriptions` used later).

This matches the schema already drafted in `src/schemas.py` (`FEATURE_MATRIX_COLS`) — comorbidity index and prior-ED counts are the main additions to wire in.

### 2.3 Cohort & label (recap from config)

- Adults (age ≥ 18); exclude in-hospital deaths; exclude elective readmissions from the positive label.
- Label = 1 if the **same patient** has a subsequent qualifying admission within **30 days** of discharge; else 0.
- One row per index admission.

### 2.4 Class imbalance strategy

1. **Cost-weighting first** (cheap, no data distortion): `scale_pos_weight` (XGB) / `class_weight="balanced"` (LR, HGB).
2. **Decision-threshold tuning** — selects the operating point (see 2.6; capacity-constrained is primary, recall-floor a secondary comparison table).
3. **Resampling only if needed** as an ablation: SMOTE / random undersampling **fit inside CV folds on training data only** (never before the split — that leaks). Treat as secondary.

### 2.5 Train / val / test split (no leakage)

- **Group = `subject_id` everywhere.** All of a patient's admissions stay in one split.
- **Hold-out test: 20%** by patient (`StratifiedGroupKFold` or grouped stratified split, fixed `random_state=42`). Touched once, at the end.
- **Tuning on the remaining 80%** via **5-fold `StratifiedGroupKFold`** (stratify on label, group on `subject_id`).
- Fit scalers/imputers **inside** each fold (pipeline), never on the full data.

### 2.6 Metrics (updated 2026-08-28 — see `docs/ARCHITECTURE.md` §2 for the
current, authoritative operating-point policy)

- **Primary:** **AUPRC (average precision)** — the honest headline metric under heavy imbalance.
- **Secondary:** AUROC (comparability with literature).
- **Operating-point report:** at the chosen threshold — **recall (sensitivity), precision, specificity, F2-score** (F2 weights recall over precision), and the confusion matrix.
- **Threshold selection rule (primary, since 2026-08-25):** capacity-constrained
  — flag the top `stage1.capacity_k` fraction of admissions by risk score
  (default 15%), a follow-up capacity a hospital can actually act on. The
  original recall-floor rule (pick the lowest threshold achieving
  **recall ≥ target**, start 0.85, `config.yaml stage1.target_recall`) is
  kept and reported as a **secondary** comparison table for literature
  comparability — it alone flags ~67% of admissions, not a deployable
  triage. Stage 2 is an independent second opinion, not a filter that
  "recovers" Stage 1's precision — see `docs/ARCHITECTURE.md`.
- Report calibration (reliability curve / Brier) as a nice-to-have.

---

## 3. Optuna search space & timing (Apple Silicon)

Run Optuna maximizing **mean CV AUPRC** (5-fold StratifiedGroupKFold). Use XGBoost `hist` tree method (CPU, multi-threaded) — no CUDA on Mac; do **not** chase GPU.

### 3.1 XGBoost search space

| Param | Range | Sampler |
|-------|-------|---------|
| `n_estimators` | use **early stopping** (cap 2000) + `learning_rate` instead of tuning directly | — |
| `learning_rate` | `[1e-3, 0.3]` | log-uniform |
| `max_depth` | `[3, 10]` | int |
| `min_child_weight` | `[1, 10]` | int (or log-uniform `[1e-1, 1e2]`) |
| `subsample` | `[0.6, 1.0]` | uniform |
| `colsample_bytree` | `[0.5, 1.0]` | uniform |
| `gamma` | `[0, 5]` | uniform |
| `reg_lambda` | `[1e-3, 10]` | log-uniform |
| `reg_alpha` | `[1e-3, 10]` | log-uniform |
| `scale_pos_weight` | around `neg/pos` (e.g. `[0.5×, 2×]` that ratio) | log-uniform |

Fixed: `tree_method="hist"`, `eval_metric="aucpr"`, early stopping ~50 rounds.

### 3.2 HistGradientBoosting search space

| Param | Range |
|-------|-------|
| `learning_rate` | `[1e-2, 0.3]` log-uniform |
| `max_iter` | early stopping (cap 1500) |
| `max_leaf_nodes` | `[15, 255]` |
| `max_depth` | `{None, 4, 6, 8, 12}` |
| `min_samples_leaf` | `[10, 200]` log-uniform |
| `l2_regularization` | `[1e-3, 10]` log-uniform |

Fixed: `class_weight="balanced"`, `early_stopping=True`.

### 3.3 Logistic Regression

Small grid is enough: `C ∈ {0.01, 0.1, 1, 10}`, `penalty="l2"`, `class_weight="balanced"`, `solver="liblinear"` or `"lbfgs"`. No Optuna needed.

### 3.4 Expected wall-clock on an M-series 14" (cohort ~40–60k admissions, ~25–40 features)

| Step | Rough time | Notes |
|------|-----------|-------|
| Single LR fit | < 1 s | baseline is instant |
| Single XGBoost fit (early stopping) | ~1–10 s | `hist`, all performance cores |
| Single HGB fit | ~1–5 s | very fast on CPU |
| **Quick baseline** (all 3, default params, 5-fold CV) | **~1–3 min** | sanity check / first numbers |
| **Full Optuna search** (XGB, ~100–200 trials × 5-fold CV) | **~10–30 min** | parallelize with `n_jobs`; use Optuna pruning (`MedianPruner`) to cut bad trials early |
| Optuna for HGB (~100 trials) | **~5–15 min** | |
| **Final model** (refit best params on full train, eval on test) | **< 1 min** | |

Whole Stage 1 modeling cycle is comfortably **under an hour** end-to-end on the laptop. If a search drags, lower trial count or enable aggressive pruning before reaching for any GPU/cloud — none is needed at this scale.

---

## 4. Proposed workflow (no code yet)

1. Build the feature matrix (synthetic first, then real MIMIC) → matches `src/schemas.py`.
2. Grouped stratified split by `subject_id` → train(80%)/test(20%); 5-fold CV inside train.
3. Quick baseline: all 3 models, default params, report AUPRC/AUROC.
4. Optuna search (XGB, HGB) maximizing CV AUPRC with pruning.
5. Select operating threshold for recall ≥ target on validation.
6. Refit best model on full train, evaluate **once** on test; report AUPRC, AUROC, and the recall/precision/F2 operating point + confusion matrix.
7. Persist model + threshold + metrics to `models/` (gitignored); log to `MODEL_CARD.md`.

---

## Sources

- [A Comparative Study of Structured and Narrative EHR Data for 30-Day Readmission Risk Assessment (MDPI Electronics, 2025)](https://www.mdpi.com/2079-9292/14/20/4033)
- [Predicting 30-Day Hospital Readmission for Heart Failure Using EHR Embeddings (JMIR Medical Informatics, 2025)](https://medinform.jmir.org/2025/1/e73020)
- [30-day Hospital Readmission Prediction using MIMIC Data (ResearchGate)](https://www.researchgate.net/publication/349934249_30-day_Hospital_Readmission_Prediction_using_MIMIC_Data)
- [Predicting 30-Day Hospital Readmission in Patients With Diabetes Using ML on EHR (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12085305/)
- [Predictive Modeling of Hospital Readmission: Challenges and Solutions (arXiv 2106.08488)](https://arxiv.org/pdf/2106.08488)
- [In-hospital mortality, readmission, and prolonged LOS prediction (JAMIA Open, 2024)](https://academic.oup.com/jamiaopen/article/7/3/ooae074/7758162)
- [Predicting unplanned ICU readmissions: a multimodality evaluation (Scientific Reports, 2023)](https://www.nature.com/articles/s41598-023-42372-y)
- [XGBoost Hyperparameter Tuning With Optuna (Forecastegy)](https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/)
- [Bayesian Optimization of XGBoost Hyperparameters with Optuna (XGBoosting)](https://xgboosting.com/bayesian-optimization-of-xgboost-hyperparameters-with-optuna/)
- [MIMIC-III ICU Readmission Analysis (GitHub, patient-level splitting reference)](https://github.com/Jeffreylin0925/MIMIC-III_ICU_Readmission_Analysis)
