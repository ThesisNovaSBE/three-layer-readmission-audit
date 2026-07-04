# LLM-Augmented Hospital Readmission Prediction

> **LLMs/agents working in this repo MUST read `PROJECT_TLDR.md` and the latest file in `sessions/` before starting work.**

A two-stage "Triage-and-Verify" pipeline for predicting unplanned hospital readmissions, built as a Master's thesis at Nova SBE.

**Stage 1 — Triage (Classical ML):** A high-recall classifier (Logistic Regression / XGBoost / HistGradientBoosting) flags at-risk patients from structured MIMIC-IV data.

**Stage 2 — Verify (Clinical Encoder):** A fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`) reads the discharge notes of flagged patients and prunes false positives.

**Stage 3 — Explain (Optional):** A local generative model (Ollama) writes plain-language explanations for confirmed cases.

---

## Quick Start — Without MIMIC (Synthetic Data)

No MIMIC access? The pipeline runs on synthetic data out of the box.

```bash
# 1. Clone and set up
git clone <repo-url> && cd 20260401_Thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the demo (generates synthetic data + trains Stage 1)
python setup_demo.py
```

The synthetic generator (`src/data/synthetic.py`) creates fake patient records that match the real MIMIC-IV column schema, so all downstream code works identically.

## Full Pipeline — With MIMIC Access

If you have credentialed access to MIMIC-IV on PhysioNet:

```bash
# 1. Copy and fill in your local data paths
cp .env.example .env
# Edit .env to point MIMIC_IV_DIR and MIMIC_IV_NOTE_DIR to your local extracts

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the pipeline
python -m src.data.cohort       # Extract cohort
python -m src.data.features     # Engineer features
python -m src.model.train       # Train Stage 1 models
```

## Data Policy

> **The real MIMIC-IV data must NEVER be committed to this repository.**
>
> MIMIC-IV is a credentialed dataset governed by a PhysioNet Data Use Agreement.
> It stays only on the local machines of team members who have signed the DUA.
> The `data/` directory is gitignored. Do not override this.

## Project Structure

```
├── PROJECT_TLDR.md          # Project context (read this first)
├── MODEL_CARD.md            # Model documentation
├── config.yaml              # All configurable parameters
├── .env.example             # Template for local data paths
├── requirements.txt         # Python dependencies
├── setup_demo.py            # One-command demo with synthetic data
├── src/
│   ├── config.py            # Config + env loader
│   ├── schemas.py           # Column schemas / contracts
│   ├── data/
│   │   ├── cohort.py        # Cohort extraction from MIMIC-IV
│   │   ├── features.py      # Feature engineering
│   │   └── synthetic.py     # Synthetic data generator
│   └── model/
│       ├── train.py          # Model training
│       ├── evaluate.py       # Evaluation metrics
│       └── predict.py        # Inference
├── sessions/                # Work session logs (read latest for context)
├── notebooks/               # Exploratory analysis
├── tests/                   # Unit & integration tests
├── data/                    # LOCAL ONLY — gitignored
└── models/                  # Trained artifacts — gitignored
```
