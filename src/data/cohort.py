"""Cohort extraction from real MIMIC-IV data.

Reads raw MIMIC-IV tables, applies inclusion/exclusion criteria, and produces
the cohort with a 30-day readmission label.
"""

from __future__ import annotations

import pandas as pd

from src.config import has_real_data, load_config
from src.config_schema import AppConfig


def extract_cohort(cfg: AppConfig | None = None) -> pd.DataFrame:
    """Extract the patient cohort from real MIMIC-IV data.

    Args:
        cfg: validated project config. Loaded from disk if ``None``.

    Returns:
        DataFrame of cohort admissions with a ``readmission_30d`` label column.

    Raises:
        FileNotFoundError: if MIMIC-IV data is not configured or not found.
        NotImplementedError: real cohort extraction is not yet implemented.
    """
    cfg = cfg or load_config()

    if not has_real_data(cfg):
        raise FileNotFoundError(
            "Real MIMIC-IV data not found. Set MIMIC_IV_DIR in .env or config.yaml, "
            "or use src.data.synthetic to generate sample data."
        )

    raise NotImplementedError("Real MIMIC-IV cohort extraction not yet implemented.")


if __name__ == "__main__":
    extract_cohort()
