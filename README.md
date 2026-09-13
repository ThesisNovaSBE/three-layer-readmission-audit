# LLM-Augmented Hospital Readmission Prediction

> **LLMs/agents working in this repo MUST read `PROJECT_TLDR.md`, `docs/ARCHITECTURE.md`, and the latest file in `sessions/` before starting work.**

A three-layer LLM-auditing pipeline for predicting 30-day hospital readmissions, built as a Master's thesis at Nova SBE (M.Sc. Business Analytics). Full design rationale: **`docs/ARCHITECTURE.md`**.

**Layer 1 — XGBoost (structured screen):** Trained on 521,191 MIMIC-IV structured admissions, flags the top-K% highest-risk admissions (capacity-constrained operating point, default K=15%; recall-floor kept as a secondary comparison table).

**Layer 2 — Clinical-Longformer, note-only (independent risk estimate):** A fine-tuned `yikuan8/Clinical-Longformer` (4096-token window) reads only the discharge note of flagged patients — no structured features — and produces an independent risk estimate, not a gate on Stage 1's flag. (A jointly-trained structured+note "FusionLongformer" variant was built and dropped on 2026-08-26 without ever completing a training run — see `docs/ARCHITECTURE.md` §2.)

**Layer 3 — MedGemma-27B (independent auditor):** A local reasoning model (served via vLLM with guided/structured decoding, temperature=0 — switched from Ollama/phi4-mini 2026-09-10 once the batch run moved to cluster GPU hardware) reads Stage 1's score + SHAP reasons, Stage 2's score, a quantitatively-computed discordance signal, and the discharge note itself, then reaches its **own** uphold/override judgment with a clinical justification — it does not narrate a decision Stage 2 already made.

**Frontend:** A React + TypeScript + Vite dashboard visualises the pipeline logic and patient-level results — useful for demos and thesis presentations.

---

## Results

> **Note (2026-09-13):** Stage 1 and Stage 2 below are both real retrains —
> 400-trial Optuna search for Stage 1, corrected age-group oversampling +
> 4096 tokens for Stage 2, both targeting **unplanned** readmission
> (switched from all-cause 2026-09-05 — every model before that silently
> used all-cause despite that being the project's stated scope; see
> `docs/ARCHITECTURE.md` §6). See `MODEL_CARD.md` for the full breakdown,
> including per-age-group fairness metrics and the real RQ1 comparison.
> Stage 3 (MedGemma-27B via vLLM) has code and cluster infrastructure in
> place but has only been smoke-tested at small scale, not run at full
> scale — see `docs/ARCHITECTURE.md` §4 for what's still pending.

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

### Stage 2 — Clinical-Longformer (MIMIC-IV-Note, notes cohort n=9,899, target=unplanned)

| Metric | Value |
|--------|-------|
| AUROC | 0.622 |
| AUPRC | 0.538 |
| Recall | 0.974 |
| Precision | 0.437 |

Weakest subgroup: age 70+ (AUROC 0.581) — same pattern as Stage 1.
Full per-age-group breakdown in `MODEL_CARD.md`.

### RQ1 — Does the note text add signal beyond structured data?

Scored independently on the same 62,759-admission notes-covered
population (neither model gates the other): Stage 1 AUROC 0.7093 vs.
Stage 2 AUROC 0.7101, diff +0.0008 [-0.0041, +0.0054]. **Null result** —
expected and reportable per this project's own design docs, not a failure.

### Stage 1+2 — Combined

The current cascade (Stage 1 → Stage 2 prune only, no Stage 3 yet) is
nearly identical to Stage 1 alone at the same alert budget (control arm) —
expected and incomplete, since Stage 3 (the actual "auditor" layer) hasn't
run at full scale yet. Full numbers and caveats in `MODEL_CARD.md`'s
Stage 1+2 section.

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

# 5. Stage 3: on-demand audit for one patient, or batch (src/stage3/batch.py)
#    for every Stage 1-flagged, note-covered admission. Requires the
#    MedGemma-27B weights pre-downloaded (download_stage3_model.sh) and
#    vLLM installed (cluster/GPU only — see docs/ARCHITECTURE.md §4):
python -m src.stage3.pipeline <hadm_id>
#    python -m src.stage3.batch          # full batch run
#    python -m src.stage3.batch --limit 10   # smoke test
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
│       ├── explain.py           # Prompt building, discordance calc, vLLM/MedGemma call
│       ├── pipeline.py          # explain_patient() — the on-demand entry point
│       ├── batch.py             # Batch audit runner — every flagged, note-covered admission
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
