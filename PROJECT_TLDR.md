This is the single source of truth for project context. LLMs and teammates should read this file first before doing any work. Then read `docs/ARCHITECTURE.md` for the current pipeline design, then the most recent file in `sessions/` to see current state.

# LLM-Augmented Hospital Readmission Prediction — A Two-Stage "Triage-and-Verify" Approach

**Program:** M.Sc. Business Analytics, Nova SBE
**Advisor:** Prof. Yufei Shen
**Team size:** 4

## Goal

Predict 30-day hospital readmission using a three-layer pipeline:

1. A classical ML model flags high-risk patients from **structured** MIMIC-IV data at a deployable, capacity-constrained operating point.
2. A fine-tuned clinical language model reads the **clinical notes** (MIMIC-IV-Note) of flagged patients and produces an independent combined risk estimate.
3. A reasoning LLM (phi4-mini) audits each flagged case — reading both scores and the note — and makes its own independent uphold/override decision with a clinical justification.

See `docs/ARCHITECTURE.md` for the full design and rationale.

## Pipeline Stages

Full design detail (and what's implemented vs. still pending) lives in
**`docs/ARCHITECTURE.md`** — the summary below is intentionally short.

### Stage 1 — XGBoost (structured screen) — IMPLEMENTED

- Models: Logistic Regression, XGBoost, HistGradientBoosting (selectable via `config.yaml`)
- Input: ~40 structured EHR features from MIMIC-IV (demographics, index-admission traits, prior utilisation, Charlson comorbidity index, last + aggregate labs, last vitals)
- **Operating point: capacity-constrained (primary, since 2026-08-25)** — flags the top `capacity_k` (default 15%) of admissions by risk score. Recall-floor (`recall >= 0.85`) kept as a secondary table for literature comparability; it alone flagged 66.94% of admissions, which is not a deployable triage.
- Imbalance: `scale_pos_weight` / `class_weight="balanced"`
- Split: patient-level (`subject_id`) grouped + stratified; tuning via Optuna (CV AUPRC)

### Stage 2 — FusionLongformer / plain Longformer (independent combined estimate) — IMPLEMENTED, RETRAIN PENDING

- Model: Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`, Li et al. 2022), 4096-token window (raised from 2048 on 2026-08-25 — see `docs/ARCHITECTURE.md` §2)
- Input: Discharge notes (MIMIC-IV-Note) + 8 structured features (a different, smaller set than Stage 1's ~40 — deliberate; see ARCHITECTURE.md on why no "identical feature space" claim is made)
- Produces an independent combined risk estimate, not a Stage-1-gating confirm/reject (the `stage2_confirmed` column still exists for the ablation's "cascade" arm but is not the pipeline's final word — Stage 3 is)
- Training: Focal loss + per-age-group loss weights; patient-level splits (60/20/20 finetune/val/cal)
- Calibration: Platt scaling per age group
- Requires real MIMIC-IV-Note; cannot run on synthetic data
- Key modules: `src/stage2/splits.py`, `src/stage2/train.py`, `src/stage2/calibrate.py`, `src/stage2/evaluate.py`, `src/stage2/predict.py`
- **Training scale:** 249k stratified notes (~99% of available MIMIC-IV-Note training data); 70+ group oversampled
- **HPC training:** GWDG KISSKI cluster (A100 80GB, bf16, batch_size=8) — retrain pending at the new 4096-token window
- **Job script:** `train_stage2.sh` — Slurm job with pre-flight checks; auto-resumes from checkpoint on resubmission
- **Full setup:** `docs/HPC_DEPLOYMENT.md`

### Stage 3 — Independent LLM Auditor — REWRITTEN 2026-08-25

- Model: `phi4-mini` via Ollama, `temperature=0` (pinned for reproducibility)
- Input per patient: Stage 1 score + SHAP-ranked reasons, Stage 2 score, near-full discharge note text, and a pre-computed (not LLM-chosen) discordance mode
- **phi4-mini reaches its own independent `uphold`/`override` decision** — it does not narrate or classify a decision Stage 2 already made
- Discordance mode (`CONCORDANT` / `NOTE_MITIGATES` / `NOTE_AMPLIFIES`) is computed quantitatively from percentile-rank displacement of stage1_score vs. stage2_score within the flagged+noted cohort (`src/stage3/explain.py:compute_discordance`) — not raw score subtraction, and never an LLM choice
- Domain taxonomy: `social_support`, `frailty`, `palliative_intent`, `care_coordination`, `clinical_trajectory`, `other`
- On-demand only (per-patient, called from the API); no batch runner yet — see `docs/ARCHITECTURE.md` §4 for what that blocks
- Key modules: `src/stage3/explain.py`, `src/stage3/pipeline.py`, `src/stage3/models.py`, `src/stage3/attention.py` (optional auxiliary hint only), `src/stage3/shap_extract.py`

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
