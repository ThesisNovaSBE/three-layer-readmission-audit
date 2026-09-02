# Contributing

This is a master's thesis repository. External contributions are not expected,
but the guidelines below ensure reproducibility and code quality for anyone
extending this work.

## Data

MIMIC-IV data must **never** be committed. All data files are gitignored.
Obtain access independently at <https://physionet.org/content/mimiciv/>.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
```

For GPU training on a SLURM cluster, see `train_stage2.sh` for the reference
job script (conda-based; adjust the environment/module lines for your site).

## Code quality

This project targets **Pylint 10.00/10** (enforced by `.pylintrc`).

```bash
pylint src/
pytest                        # unit tests (no data required)
```

Before committing, ensure both pass with no errors. `pytest.ini` declares an
`integration` marker for future slow tests that build synthetic data or train
models; no test currently uses it.

## Style

- Python 3.10+, type-annotated, `from __future__ import annotations`
- Imports: `isort` order (stdlib → third-party → local)
- No inline comments unless the *why* is non-obvious
- Pydantic v2 config: dot-access only (`cfg.stage1.model`), never subscript

## Repo structure

```
src/
  config.py / config_schema.py   # AppConfig (Pydantic v2)
  data/                          # cohort extraction, synthetic data
  model/                         # Stage 1 XGBoost + metrics
  stage2/                        # Clinical-Longformer fine-tuning
  stage3/                        # Independent LLM audit of Stage 1's flags
tests/                           # pytest suite (no MIMIC data)
sessions/                        # per-session development notes
docs/                            # Architecture, methodology, and narrative notes
```
