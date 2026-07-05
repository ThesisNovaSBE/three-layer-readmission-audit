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

### Stage 2 — Clinical Encoder (Notes) — IMPLEMENTED

- Model: Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`, Li et al. 2023)
- Input: Discharge notes (MIMIC-IV-Note) for patients flagged in Stage 1
- Objective: Confirm or reject each flag, reducing false positives
- Training: Focal loss + per-age-group loss weights; 2048-token sequences; patient-level splits (60/20/20 finetune/val/cal)
- Calibration: Platt scaling per age group + recall-floor threshold selection (recall ≥ 0.65, maximise F2)
- Requires real MIMIC-IV-Note; cannot run on synthetic data
- Key modules: `src/stage2/splits.py`, `src/stage2/train.py`, `src/stage2/calibrate.py`, `src/stage2/evaluate.py`, `src/stage2/predict.py`

### Stage 3 — Explanation (Optional) — IMPLEMENTED

- Model: `phi4-mini` via Ollama
- Input: Confirmed high-risk cases from Stage 2
- Objective: Plain-language explanation of why a patient is flagged
- Run: `ollama pull phi4-mini` then `python setup_stage3.py`
- See `src/stage3/` for implementation

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
