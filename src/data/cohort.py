"""Cohort extraction from real MIMIC-IV data.

Reads raw MIMIC-IV tables, applies inclusion/exclusion criteria from
``config.yaml``, and computes the 30-day readmission label.

The feature engineering pipeline (``src.data.features``) calls the same
helper functions internally, so ``extract_cohort`` is primarily useful as a
standalone step for inspection or documentation purposes.
"""

from __future__ import annotations

import pandas as pd

from src.config import has_real_data, load_config
from src.config_schema import AppConfig
from src.schemas import TARGET_COL


def extract_cohort(cfg: AppConfig | None = None) -> pd.DataFrame:
    """Extract the patient cohort from real MIMIC-IV data.

    Applies the inclusion / exclusion criteria defined in ``config.yaml``
    (``cohort.*``) and computes the 30-day unplanned readmission label.

    Args:
        cfg: validated project config. Loaded from disk if ``None``.

    Returns:
        DataFrame with one row per hospital admission, containing at minimum
        ``subject_id``, ``hadm_id``, ``admittime``, ``dischtime``,
        ``admission_type``, ``discharge_location``, and ``readmission_30d``.

    Raises:
        FileNotFoundError: if MIMIC-IV data paths are not configured.
    """
    # pylint: disable=import-outside-toplevel  # avoids circular import at module level
    from src.data.features import (
        _apply_cohort_filters,
        compute_readmission_label,
        load_raw_tables,
    )

    cfg = cfg or load_config()

    if not has_real_data(cfg):
        raise FileNotFoundError(
            "Real MIMIC-IV data not found. Set MIMIC_IV_DIR in .env or "
            "config.yaml, or use src.data.synthetic for sample data."
        )

    tables = load_raw_tables(cfg)
    admissions = compute_readmission_label(tables["admissions"], cfg)
    cohort = _apply_cohort_filters(admissions, tables["patients"], cfg)

    keep = [
        "subject_id", "hadm_id", "admittime", "dischtime",
        "admission_type", "discharge_location", "hospital_expire_flag",
        TARGET_COL,
    ]
    return cohort[[c for c in keep if c in cohort.columns]].reset_index(drop=True)


if __name__ == "__main__":
    result = extract_cohort()
    print(f"Cohort: {len(result):,} admissions, "
          f"readmission rate: {result[TARGET_COL].mean():.1%}")
