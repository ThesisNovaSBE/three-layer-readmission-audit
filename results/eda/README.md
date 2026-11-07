# results/eda — Exploratory Data Analysis (MIMIC-IV 3.1 + MIMIC-IV-Note 2.2)

Aggregate-only EDA of the data **exactly as the pipeline uses it**. The
cohort is rebuilt from raw tables with the pipeline's own
`compute_readmission_label` / `build_features` and asserted equal to the
model cohort (521,191 admissions). Reasoning and design decisions are in
`sessions/2026-11-06_session-37.md`.

**Regenerate:** `python -m src.eda.structured` (~1 min once cached) and
`python -m src.eda.notes` (~45 min the first time, then cached). The
row-level caches live in `data/interim/eda_cache/` (gitignored).

**Privacy:** counts below 11 are shown as `"<11"`, and rates with a small
numerator or denominator are `null`. Distributions are reported as p1–p99
only (no min/max), and histograms drop values outside [p1, p99] and zero
out bins under 11. No IDs, dates or note text appear anywhere. This is
enforced by `src/eda/privacy.py` and `tests/test_eda_privacy.py`.

## Files

| File | Content |
|---|---|
| `structured_summary.json` | inventory, cohort flow, outcome/label audit, utilisation, Table 1 (+SMD), subgroup rates, temporal drift, note coverage, missingness, univariate AUROC, redundancy, itemid audit, diagnoses, per-feature distributions |
| `notes_summary.json` | note inventory, coverage, length/tokens/truncation, sections, placeholder-only sections, Stage 3 concept keywords, note timing, de-id density |
| `figures/s01..s13_*.png` | structured-data figures |
| `figures/n01..n06_*.png` | note figures |

## Headline numbers

| | |
|---|---|
| MIMIC-IV admissions → model cohort | 546,028 → **521,191** (216,395 patients) |
| … with a discharge note (Stage 2/3 population) | **313,740 (60.2%)**, 140,340 patients |
| Unplanned 30-day readmission (model target) | **19.0%** (99,149); all-cause 20.2% |
| Planned returns removed by the unplanned label | 6,234 (5.9% of all-cause) |
| Median days to readmission | 8.6; 43.7% within 7 days; 13.3% within 1 day |
| Discharge-note length | median **3,268** Longformer tokens (p25 2,450, p75 4,236) |
| Notes longer than the 4,096-token Stage 2 window | **27.9%** |

## Findings that matter for the thesis

Ordered by how directly they touch a claim, a design decision, or code.

### 1. About 43% of the target is observation stays (`s08`, `outcome`)
Of all-cause 30-day returns, the *next* admission is EU OBSERVATION
(18.0%), OBSERVATION ADMIT (17.4%), DIRECT OBSERVATION (4.0%) or AMBULATORY
OBSERVATION (0.8%). That is 40.2% of all-cause returns and ≈42.7% of the
unplanned target. In US billing, observation is outpatient status, and
CMS HRRP does not count it as a readmission. Our label is therefore
broader than the policy definition the introduction motivates with. This
needs either a limitation sentence or a sensitivity analysis on a label
that excludes observation returns.

### 2. Vital-sign features encode "had an ICU stay" (`s09`, `missingness`, `measurement_quality`)
All six `*_last` vitals come from ICU `chartevents`. They are **100%
missing for non-ICU admissions** and 86% missing overall (ICU share =
14.0%). Because XGBoost routes NaN natively, "vitals present" acts as an
ICU flag. Where present, the values are stale: the last vital is a
median **76 h before discharge** (p95 ≈ 380 h), against about 11 h for
labs. They are not discharge-time vitals. This closes the open "vitals
missingness" feature-audit item in `docs/ARCHITECTURE.md`.

### 3. `temperature_last` mixes °F and °C (`measurement_quality.per_itemid`)
`VITAL_ITEMIDS["temperature"] = [223761, 223762]`. 223761 is °F (median
98.4) and 223762 is °C (median 37.0, about 6.6k admissions). Both feed
one feature. Trees can partly split around this, so the impact is
probably small, but the feature is semantically wrong. The fix is to
convert 223762 to °F (or drop it) in `src/data/features.py`, which would
require a Stage 1 retrain.

### 4. Notes barely cover 2020–2022 (`temporal`, `note_coverage`)
Discharge-note coverage is 68–74% in 2008–2019 but **8.2% in 2020–2022**.
The structured cohort includes about 89k admissions from those years; the
Stage 2/3 population almost entirely excludes them, COVID era included.
Coverage is also low for EU OBSERVATION admissions (26.6%) versus
emergency admissions (81–85%). This adds a temporal dimension to the
existing selection-bias analysis (`selection_bias_evaluation.json`).

### 5. Truncation hits readmitted and older patients more (`n01`, `length`)
Readmitted patients' notes are longer (median 3,582 vs 3,193 tokens), so
**36.0% of readmitted vs 26.0% of non-readmitted notes are truncated**.
Truncation also rises with age: 18.1% (18–40), 25.8% (41–55), 30.4%
(56–70), **32.5% (70+)**. Stage 2 therefore loses the most text exactly
for the positive class and for the age group with the known
discrimination gap. This is a candidate partial explanation for the 70+
gap that hasn't been tested.

### 6. Truncation removes the discharge sections, not the history (`n02`, `sections`)
The note order is fixed: history first, discharge sections last. Among
notes that contain each section, the share starting *after* token 4,096
is about 0% for Chief Complaint through Pertinent Results, 2.8% for
Brief Hospital Course, 13.8% for Discharge Medications, **19.8% for
Discharge Disposition, 22.0% for Discharge Instructions, and 27.8% for
Followup Instructions**. Head-only truncation drops the parts most
relevant to post-discharge risk. Head+tail truncation is a cheap,
well-motivated alternative to mention as future work.

### 7. De-identification removed the social context (`sections`, `deid`)
**Social History is placeholder-only (`___`) in 97.1%** of the notes that
have it, and **Followup Instructions in 100%**. Discharge Instructions
and Disposition are intact (<1% blanked). The phrase "lives alone"
appears in only 0.8% of notes. Consequence for Stage 3: the
`lives_alone_no_support` and `no_followup_arranged` grounds have almost
no textual basis in MIMIC-IV-Note. Their low firing rate is a property of
the dataset, not necessarily of the auditor. Notes carry a median of 53
placeholders each.

### 8. Label edge cases (`outcome`)
- **Competing risk:** 8,304 admissions (2.0% of negatives) died within 30
  days of discharge without readmission and are labelled negative.
- **13.3% of readmissions start within 1 day** of discharge (`s02`). Some
  are likely transfers or administrative re-registrations. 47 index
  admissions overlap with the next admission (negative gap) and are
  labelled negative by the `days_to_next >= 0` rule.
- 226 cohort admissions have `discharge_location = DIED` but
  `hospital_expire_flag = 0` (inconsistent source coding; negligible).

### 9. Readmissions concentrate in frequent users (`s12`, `utilisation`)
77% of cohort admissions belong to patients with 2 or more admissions.
Patients with 11+ admissions supply 18.6% of admissions but **37.5% of
readmissions** (rate 38.4%). Single-admission patients (23% of
admissions) have a 0.6% rate, which is structural: a positive needs a
following admission. This empirically justifies the patient-level split,
and it is why `n_prior_admissions` is the strongest single feature.

### 10. Where the structured signal lives (`s11`, `univariate_auroc`, `redundancy`)
Single-feature AUROC tops out at 0.64: `n_prior_admissions` 0.639,
`n_prior_ed_visits` 0.630, `days_since_last_discharge` 0.625, hemoglobin
(min/mean/last) about 0.60, `charlson_index` 0.580. Age carries almost no
signal on its own (SMD −0.02). 30 feature pairs have |Spearman ρ| ≥ 0.8,
mostly the four aggregates of each lab plus `charlson_index` ↔
`n_comorbidities` (0.94).

### 11. Other things worth knowing
- **37% of curated lab values have no `hadm_id`** (12.6M of 33.6M;
  outpatient labs) and are dropped by the pipeline. 1.0% of the curated
  lab/vital values linked to an admission are charted before `admittime`
  (mostly pre-admission ED draws).
- `discharge_location` is missing for **28.7%** of the cohort (rate 18.9%).
- Highest-risk subgroups: discharged to a psychiatric facility 51.8%,
  against advice 34.0%, primary diagnosis Mental 31.8%. Base rates are
  flat across age bands (17.7–21.1%), so the 70+ gap is not a prevalence
  effect.
- ICD-10 share jumps from 0% to 40% in 2014–2016 (the Oct-2015
  switch); mean Charlson rises from 1.61 to 2.25 over the period (coding
  drift and/or acuity).
- Note timing: median 3.4 h *after* discharge; 26% stored before
  discharge, and about 29% more than 24 h after. A real discharge-time
  deployment would not have the final note for a sizeable share of
  patients.
- Lexical Stage 3 concepts (no negation handling): nonadherence/AMA
  (4.2% of notes; rate 26.1% vs 19.4%), substance use (12.9%; 23.2% vs
  19.1%), palliative/hospice (4.2%; 23.5% vs 19.5%). Cognitive impairment
  and home services show no base-rate difference.
- 41,820 cohort admissions without a discharge note do have radiology
  reports, a possible fallback text source that is unused today.

## Caveats

- The real-year groups are approximate (MIMIC date-shifting is resolved
  only to 3-year anchor windows).
- Keyword concepts measure the *words*, not clinical facts ("denies
  alcohol use" counts as a mention).
- The section detection uses the 15 standard MIMIC-IV discharge
  headers; non-standard notes count as "section absent".
