# LLM-Augmented Hospital Readmission Prediction

> **LLMs/agents working in this repo MUST read `PROJECT_TLDR.md`, `docs/ARCHITECTURE.md`, and the latest file in `sessions/` before starting work.**

A three-layer LLM-auditing pipeline for predicting 30-day hospital readmissions, built as a Master's thesis at Nova SBE (M.Sc. Business Analytics). Full design rationale: **`docs/ARCHITECTURE.md`**.

**Layer 1 — XGBoost (structured screen):** Trained on 521,191 MIMIC-IV structured admissions, flags the top-K% highest-risk admissions (capacity-constrained operating point, default K=15%; recall-floor kept as a secondary comparison table).

**Layer 2 — Clinical-Longformer, note-only (independent risk estimate):** A fine-tuned `yikuan8/Clinical-Longformer` (4096-token window) reads only the discharge note of flagged patients — no structured features — and produces an independent risk estimate, not a gate on Stage 1's flag. (A jointly-trained structured+note "FusionLongformer" variant was built and dropped on 2026-08-26 without ever completing a training run — see `docs/ARCHITECTURE.md` §2.)

**Layer 3 — phi4-mini (independent auditor):** A local reasoning model (Ollama, temperature=0) reads Stage 1's score + SHAP reasons, Stage 2's score, a quantitatively-computed discordance signal, and the discharge note itself, then reaches its **own** uphold/override judgment with a clinical justification — it does not narrate a decision Stage 2 already made.

**Frontend:** A React + TypeScript + Vite dashboard visualises the pipeline logic and patient-level results — useful for demos and thesis presentations.

---

## Results

> **Note (2026-09-05):** Stage 1 below is the real retrain — 400-trial
> Optuna search, capacity-constrained threshold, isotonic calibration,
> targeting **unplanned** readmission (switched from all-cause the same
> day — every model before this one silently used all-cause despite that
> being the project's stated scope; see `docs/ARCHITECTURE.md` §6). Stage 2
> is still the v1 checkpoint — its retrain under the current config
> (unplanned label, corrected age-group oversampling, 4096 tokens) hasn't
> run yet. See `MODEL_CARD.md` for the full breakdown and `docs/ARCHITECTURE.md`
> §4 for what's still pending.

### Stage 1 — XGBoost (MIMIC-IV v3.1, n=521,191, held-out test n=104,242, target=unplanned)

| Metric | Value |
|--------|-------|
| AUROC | 0.7215 |
| AUPRC | 0.3965 (base rate 19.0%) |
| Recall @ thr=0.3235 (capacity=15%, primary policy) | 0.352 |
| Precision @ thr=0.3235 | 0.431 |
| F2 | 0.366 |

Recall-floor (secondary, for literature comparability): recall≥0.85 →
precision=0.247. Beats 3 of 4 published benchmarks cited in `evaluate.py`
(LACE 0.694, Xiao 2018 0.715, Fraccaro 2016 0.684); behind Rajkomar 2018
(0.773). Weakest subgroup: age 70+ (AUROC 0.665, recall only 19.0% vs.
35–45% for other age bands) — see `MODEL_CARD.md` for the full fairness
breakdown.

### Stage 1+2 — Combined

Not shown here — Stage 1's target just changed and Stage 2 hasn't been
retrained to match yet, so a combined number right now would mix a
new-target Stage 1 with an old-target Stage 2 and wouldn't mean anything
as a real result. See `MODEL_CARD.md`'s Stage 1+2 section for the current
(explicitly transitional) numbers and why they shouldn't be cited as-is.

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
#    GPU/HPC (recommended): sbatch train_stage2.sh          — see below
#    Local GPU:              python setup_stage2.py --mode full
python setup_stage2.py --mode full

# 5. Stage 3: on-demand audit for one patient (no batch runner yet — see
#    docs/ARCHITECTURE.md §4). Requires Ollama running + phi4-mini pulled:
#    ollama pull phi4-mini
python -m src.stage3.pipeline <hadm_id>
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

`train_stage2.sh` includes pre-flight checks (Stage 1 artifact, local model
cache, MIMIC-IV-Note path, CUDA) that fail fast with a clear message before
the expensive part of the job starts — read its header before adapting it
to a different cluster.

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
| `python -m src.stage3.pipeline <hadm_id>` | On-demand Stage 3 audit for one patient (no batch runner yet) |
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
│                                 # (Stage 3 has no setup script — on-demand only, src/stage3/pipeline.py)
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
│   └── stage3/
│       ├── explain.py           # Prompt building, discordance calc, phi4-mini call
│       ├── pipeline.py          # explain_patient() — the on-demand entry point
│       ├── models.py            # ExplanationResult (Pydantic)
│       ├── attention.py         # Optional auxiliary attention-span extraction
│       └── shap_extract.py      # SHAP feature extraction from Stage 1
├── sessions/                    # Work session logs (read latest for context)
├── tests/                       # Unit & integration tests
├── docs/
│   ├── ARCHITECTURE.md          # Current pipeline design — read this first
│   └── MODELING_PLAN.md         # Stage 1 modeling strategy
├── data/                        # LOCAL ONLY — gitignored
└── models/                      # Trained artifacts — gitignored
```
