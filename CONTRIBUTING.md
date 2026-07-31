# Contributing

This is a master's thesis repository. External contributions are not expected,
but the guidelines below ensure reproducibility and code quality for anyone
extending this work.

## Data

MIMIC-IV data must **never** be committed. All data files are gitignored.
Obtain access independently at <https://physionet.org/content/mimiciv/>.

## Environment

```bash
conda env create -f environment.yml
conda activate thesis-env
pip install -e .
```

## Code quality

This project targets **Pylint 10.00/10** (enforced by `.pylintrc`).

```bash
pylint src/
pytest                        # unit tests (no data required)
pytest -m integration         # slow integration tests (~60 s each)
```

Before committing, ensure both pass with no errors.

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
  stage3/                        # Cross-modal discordance analysis
tests/                           # pytest suite (no MIMIC data)
sessions/                        # per-session development notes
docs/                            # HPC deployment, methodology notes
```
