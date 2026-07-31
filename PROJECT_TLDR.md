This is the single source of truth for project context. LLMs and teammates should read this file first before doing any work. Then read the most recent file in `sessions/` to see current state.

# LLM-Augmented Hospital Readmission Prediction — A Two-Stage "Triage-and-Verify" Approach

**Program:** M.Sc. Business Analytics, Nova SBE
**Advisor:** Prof. Yufei Shen
**Team size:** 4

## Goal

Predict unplanned hospital readmission as accurately as possible using a two-stage pipeline:

1. A classical ML model flags high-risk patients from **structured** MIMIC-IV data (tuned for high recall).
2. A clinical language model reads the **clinical notes** (MIMIC-IV-Note) of flagged patients and prunes false positives.

## Pipeline Stages

### Stage 1 — Classical ML (Structured Data) — IMPLEMENTED

- Models: Logistic Regression, XGBoost, HistGradientBoosting (selectable via `config.yaml`)
- Input: Structured EHR features from MIMIC-IV (demographics, index-admission traits, prior utilisation, Charlson comorbidity index, last + aggregate labs, last vitals)
- Objective: **High recall** — primary metric AUPRC; threshold chosen for recall ≥ target (default 0.85)
- Imbalance: `scale_pos_weight` / `class_weight="balanced"` + threshold tuning
- Split: patient-level (`subject_id`) grouped + stratified; tuning via Optuna (CV AUPRC); see `docs/MODELING_PLAN.md`
- Plan & approach: **`docs/MODELING_PLAN.md`**

### Stage 2 — Clinical Encoder (Notes) — IMPLEMENTED / RETRAINING ON HPC

- Model: Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`, Li et al. 2023)
- Input: Discharge notes (MIMIC-IV-Note) for patients flagged in Stage 1
- Objective: Confirm or reject each flag, reducing false positives
- Training: Focal loss + per-age-group loss weights; 2048-token sequences; patient-level splits (60/20/20 finetune/val/cal)
- Calibration: Platt scaling per age group + recall-floor threshold selection (recall ≥ 0.65, maximise F2)
- Requires real MIMIC-IV-Note; cannot run on synthetic data
- Key modules: `src/stage2/splits.py`, `src/stage2/train.py`, `src/stage2/calibrate.py`, `src/stage2/evaluate.py`, `src/stage2/predict.py`
- **Training scale:** 249k stratified notes (~99% of available MIMIC-IV-Note training data); 70+ group oversampled
- **HPC training:** GWDG KISSKI cluster (A100 80GB, bf16, batch_size=8); job submitted 2026-07-30
- **Job script:** `train_stage2.sh` — Slurm job with pre-flight checks; auto-resumes from checkpoint on resubmission
- **Full setup:** `docs/HPC_DEPLOYMENT.md`

### Stage 3 — Cross-Modal Discordance Analysis + Explanation — REDESIGNED

- Model: `phi4-mini` via Ollama
- Input: **All** Stage 1 flagged patients (both confirmed AND rejected by Stage 2)
- Objective (two levels):
  1. **Per-patient**: clinician-facing explanation synthesising Stage 1 structured signals AND Stage 2 discharge note evidence
  2. **Population-level**: empirical finding — which clinical domains in discharge notes explain the predictive gap between structured EHR data and free text?
- Discordance classification per patient: `CONCORDANT` | `NOTE_MITIGATES` | `NOTE_AMPLIFIES`
- Primary category taxonomy: `social_support`, `discharge_planning`, `functional_status`, `frailty_markers`, `medication_adherence`, `housing_social_risk`, `care_complexity`, `cognition`, `structured_confirmed`
- Optional technical upgrade: extracts Clinical-Longformer attention spans (top-N sentences the model attended to) and feeds them into the phi4-mini prompt — Stage 3 is then mechanistically downstream of Stage 2's internals
- Outputs: `models/stage3_discordance.csv` (per-patient) + `models/stage3_discordance_analysis.json` (population)
- Run: `ollama pull phi4-mini` then `python setup_stage3.py --limit 500`
- Key modules: `src/stage3/attention.py`, `src/stage3/shap_extract.py`, `src/stage3/explain.py`, `src/stage3/categorize.py`, `src/stage3/run.py`

## Data

- **MIMIC-IV** (structured tables) + **MIMIC-IV-Note** (clinical notes)
- Source: PhysioNet — credentialed access required

## Team Roles

- 1 person: literature review
- 2 people: core coding (data pipeline + models)
- 1 person (joining later): evaluation, explanation layer, integration

## Running Stage 1 (works without MIMIC, on synthetic data)

```bash
python setup_demo.py                 # generate synthetic data + build features
python -m src.model.tune             # Optuna search (writes best params)
python -m src.model.train            # train final model + pick threshold
python -m src.model.evaluate         # AUPRC/AUROC + operating point + fairness
```

`config.yaml` controls the model (`stage1.model`), run mode (`run.mode: quick|full`), recall target, and paths. Add `--mode full` / `--model xgboost` to override on the CLI.

## Design Principles

- **Simplicity** — keep the architecture straightforward
- **Small/local models** — no large cloud APIs in the pipeline
- **Build it ourselves** — understand every component
- **Code quality** — Pylint 10.00/10 enforced via `.pylintrc`; Pydantic v2 config validation throughout (`src/config_schema.py`)
