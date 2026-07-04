"""Cohort extraction from real MIMIC-IV data.

Reads raw MIMIC-IV tables, applies inclusion/exclusion criteria, and produces
the cohort with a 30-day readmission label.
"""

from pathlib import Path

import pandas as pd

from src.config import load_config, has_real_data, get_data_dir


def extract_cohort(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()

    if not has_real_data(cfg):
        raise FileNotFoundError(
            "Real MIMIC-IV data not found. Set MIMIC_IV_DIR in .env or config.yaml, "
            "or use src.data.synthetic to generate sample data."
        )

    # TODO: implement real MIMIC-IV cohort extraction
    raise NotImplementedError("Real MIMIC-IV cohort extraction not yet implemented.")


if __name__ == "__main__":
    extract_cohort()
