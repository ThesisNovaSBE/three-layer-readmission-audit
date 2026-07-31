# LLM-Augmented Hospital Readmission Prediction

> **LLMs/agents working in this repo MUST read `PROJECT_TLDR.md` and the latest file in `sessions/` before starting work.**

A three-stage "Triage-and-Verify" pipeline for predicting unplanned 30-day hospital readmissions, built as a Master's thesis at Nova SBE (M.Sc. Business Analytics).

**Stage 1 — Triage (XGBoost):** A high-recall classifier trained on 521,191 MIMIC-IV structured admissions flags at-risk patients (AUROC=0.706, recall=0.848).

**Stage 2 — Verify (Clinical-Longformer):** A fine-tuned `yikuan8/Clinical-Longformer` reads discharge notes of flagged patients and prunes false positives (+21% precision, retains 70.9% of true positives in the notes cohort).

**Stage 3 — Analyse + Explain (phi4-mini):** A local generative model (Ollama) annotates every Stage 1 flagged patient (confirmed and rejected) with a cross-modal discordance label — whether the discharge note agrees with, mitigates, or amplifies the structured EHR risk signal. Population-level aggregation across all patients produces a quantitative finding on what clinical domains in notes change readmission predictions beyond structured data. A clinician-facing explanation is also generated per patient.

**Frontend:** A React + TypeScript + Vite dashboard visualises the pipeline logic and patient-level results — useful for demos and thesis presentations.

---

## Results

### Stage 1 — XGBoost (MIMIC-IV v3.1, n=521,191, held-out test)

| Metric | Value |
|--------|-------|
| AUROC | 0.706 |
| AUPRC | 0.406 |
| Recall @ thr=0.354 | 0.848 |
| Precision @ thr=0.354 | 0.256 |
| F2 | 0.580 |

### Stage 1+2 — Combined (notes cohort, thr₂=0.3)

Evaluated on the 43,776 Stage 1–flagged patients who have discharge notes (the deployable population — in real clinical use every discharged patient has a note).

| Metric | Stage 1 baseline | Stage 1+2 |
|--------|-----------------|-----------|
| Precision | 0.256 | **0.309** (+21%) |
| Recall (notes cohort) | 1.000 | 0.709 |
| F2 | 0.632 | 0.563 |
| Confirmed flags | 43,776 | 25,699 |

---

## Quick Start — Without MIMIC (Synthetic Data)

No MIMIC access? The pipeline runs on synthetic data out of the box.

```bash
git clone <repo-url> && cd 20260401_Thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data + train Stage 1
python setup_demo.py
```

The synthetic generator creates fake patient records that match the MIMIC-IV column schema so all downstream code runs identically.

---

## Full Pipeline — With MIMIC Access

```bash
# 1. Configure data paths
cp .env.example .env   # edit MIMIC_IV_DIR and MIMIC_IV_NOTE_DIR

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Stage 1: train XGBoost
python -m src.model.train

# 4. Stage 2: fine-tune Clinical-Longformer on discharge notes
#    GPU/HPC (recommended): sbatch train_stage2.sh          — see docs/HPC_DEPLOYMENT.md
#    Local GPU:              python setup_stage2.py --mode full
python setup_stage2.py --mode full

# 5. Stage 3: generate explanations for confirmed patients
#    Requires Ollama running + phi4-mini pulled: ollama pull phi4-mini
python setup_stage3.py --limit 50   # --limit for demo; remove for full run
```

---

## GPU / HPC Training (GWDG KISSKI)

Stage 2 training on 249k notes requires a capable GPU. The reference setup uses the GWDG KISSKI cluster (A100 80GB) via `train_stage2.sh`:

```bash
# On the cluster — submit the Slurm job
sbatch train_stage2.sh

# Monitor
squeue -u $USER
tail -f /projects/extern/kisski/kisski-nova-rpcl/dir.project/logs/stage2_<jobid>.log
```

Full setup guide (SSH config, data transfer, conda env, crash recovery): **`docs/HPC_DEPLOYMENT.md`**

---

## Frontend (Demo / Thesis Presentation)

```bash
cd frontend
npm install
npm run dev   # opens at http://localhost:5173
```

The dashboard has two views:
- **Pipeline** — visual Stage 1 → 2 → 3 diagram with real metrics and how-it-works explanation (for professor / committee presentations)
- **Patients** — sortable table of confirmed high-risk patients with Stage 1/2 scores; click any row for the Stage 3 clinical explanation

---

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `setup_demo.py` | Synthetic data demo — no MIMIC needed |
| `setup_stage2.py` | Fine-tune Stage 2 end-to-end |
| `setup_stage3.py --limit N` | Generate Stage 3 explanations (N = sample size) |
| `train_stage2.sh` | Slurm job script for HPC/GPU training (GWDG KISSKI A100) |
| `diagnose_cluster.sh` | Cluster diagnostic job (checks env, GPU, paths) |
| `analyse_age_fairness.py` | Age-group fairness analysis on Stage 1 results |

---

## Data Policy

> **MIMIC-IV data must NEVER be committed to this repository.**
>
> MIMIC-IV is governed by a PhysioNet Data Use Agreement.
> The `data/` directory and all `.csv`/`.joblib`/model weight files are gitignored.

---

## Project Structure

```
├── PROJECT_TLDR.md              # Project context (read first)
├── MODEL_CARD.md                # Full model documentation + metrics
├── config.yaml                  # All configurable parameters
├── .pylintrc                    # Pylint config (enforces 10.00/10)
├── setup_demo.py                # One-command demo (synthetic data)
├── setup_stage2.py              # Stage 2 fine-tuning runner
├── setup_stage3.py              # Stage 3 explanation runner
├── train_stage2.sh              # Slurm job script — GWDG KISSKI A100 80GB
├── analyse_age_fairness.py      # Age-group fairness analysis
├── frontend/                    # React + TS + Vite dashboard
│   └── src/
│       ├── components/          # PipelineDiagram, PatientTable, PatientModal
│       └── data/mockPatients.ts # Synthetic demo patients
├── src/
│   ├── config.py                # Config loader (returns AppConfig)
│   ├── config_schema.py         # Pydantic v2 AppConfig model tree
│   ├── schemas.py               # Column contracts
│   ├── data/
│   │   ├── cohort.py            # MIMIC-IV cohort extraction
│   │   ├── comorbidity.py       # Charlson comorbidity index
│   │   ├── features.py          # Feature engineering
│   │   └── synthetic.py         # Synthetic data generator
│   ├── model/                   # Stage 1: train, tune, evaluate, predict, cv
│   ├── stage2/
│   │   ├── _utils.py            # Shared helpers (band_key, model path)
│   │   ├── dataset.py           # ClinicalNotesDataset + note loading
│   │   ├── splits.py            # Patient-level finetune/val/cal splits
│   │   ├── train.py             # Fine-tune Clinical-Longformer (focal loss)
│   │   ├── calibrate.py         # Platt scaling + per-group threshold selection
│   │   ├── evaluate.py          # Stage 2 evaluation metrics
│   │   └── predict.py           # Stage 2 inference on Stage 1 flags
│   └── stage3/                  # Stage 3: explain, run
├── sessions/                    # Work session logs (read latest for context)
├── tests/                       # Unit & integration tests
├── docs/
│   ├── MODELING_PLAN.md         # Stage 1 modeling strategy
│   └── HPC_DEPLOYMENT.md        # GWDG KISSKI cluster setup guide
├── data/                        # LOCAL ONLY — gitignored
└── models/                      # Trained artifacts — gitignored
```
