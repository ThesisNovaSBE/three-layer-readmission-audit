"""Feature engineering for Stage 1 — structured readmission prediction.

Single code path for BOTH data sources:

- Real MIMIC-IV  (if ``mimic_iv_dir`` is set and exists) — loaded and
  normalised here.
- Synthetic       (otherwise) — read from ``data/synthetic/`` (generated if
  missing).

Both are normalised to the canonical raw schema in ``src/schemas.py``, then
turned into the Stage 1 feature matrix:

  demographics | index-admission traits | prior utilisation | comorbidity index
  | last + aggregate labs | last vitals | target

Usage::

    python -m src.data.features                # writes data/processed/features.csv
    python -m src.data.features --mode full     # accepted for CLI consistency;
                                                 # always builds from the full
                                                 # available dataset either way
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import get_data_dir, has_real_data, load_config
from src.config_schema import AppConfig
from src.data import synthetic as synth
from src.data.comorbidity import charlson_per_admission
from src.schemas import ID_COLS, TARGET_COL, TARGET_COL_UNPLANNED

# MIMIC-IV v3 uses "EW EMER." / "DIRECT EMER." — not the legacy "EMERGENCY" string.
# The legacy string is retained so synthetic data (which uses "EMERGENCY") still works.
_EMERGENCY_TYPES = frozenset({"EMERGENCY", "EW EMER.", "DIRECT EMER."})
# Admission types on the OUTCOME (next) admission that indicate it was
# scheduled rather than an unplanned return — e.g. a chemotherapy cycle or
# staged surgery re-admitted electively. Structured-data proxy for "planned";
# see readmission_30d_unplanned below and P7b in the 2026-08-19 remediation
# review (the all-cause label counts these as readmissions, which is not
# what "unplanned hospital readmission" in the project goal means).
_PLANNED_READMISSION_TYPES = frozenset({"ELECTIVE", "SURGICAL SAME DAY ADMISSION"})
_OBSERVATION_TYPES = frozenset({
    "EU OBSERVATION", "OBSERVATION ADMIT",
    "DIRECT OBSERVATION", "AMBULATORY OBSERVATION",
})

# ── Curated MIMIC-IV itemid maps (adjust against d_labitems if needed) ──
LAB_ITEMIDS = {
    "glucose": [50931, 50809], "creatinine": [50912], "hemoglobin": [51222, 50811],
    "white_blood_cells": [51301, 51300], "platelets": [51265],
    "sodium": [50983, 50824], "potassium": [50971, 50822], "bicarbonate": [50882],
}
VITAL_ITEMIDS = {
    "heart_rate": [220045], "systolic_bp": [220179, 220050],
    "diastolic_bp": [220180, 220051], "temperature": [223761, 223762],
    "respiratory_rate": [220210], "spo2": [220277],
}


# ──────────────────────────────────────────────────────────────────────────────
# Loading raw tables
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_tables(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    """Load raw data tables from MIMIC-IV or the synthetic generator."""
    if has_real_data(cfg):
        print("[features] Using REAL MIMIC-IV data.")
        return _load_real_mimic(cfg)
    print("[features] Real MIMIC-IV not found — using SYNTHETIC data.")
    return _load_synthetic(cfg)


def _load_synthetic(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    """Load (or generate) the synthetic raw tables."""
    syn_dir = get_data_dir() / "synthetic"
    needed = ["patients", "admissions", "measurements", "diagnoses_icd"]
    if not all((syn_dir / f"{nm}.csv").exists() for nm in needed):
        synth.generate_synthetic_dataset(
            n_patients=cfg.data.synthetic_n_patients,
            seed=cfg.data.synthetic_seed,
            output_dir=syn_dir,
        )
    tables = {nm: pd.read_csv(syn_dir / f"{nm}.csv") for nm in needed}
    for col in ("admittime", "dischtime", "edregtime"):
        tables["admissions"][col] = pd.to_datetime(
            tables["admissions"][col], errors="coerce")
    tables["measurements"]["charttime"] = pd.to_datetime(
        tables["measurements"]["charttime"], errors="coerce")
    return tables


def _load_real_mimic(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    """Load and merge raw MIMIC-IV tables from disk."""
    root = Path(cfg.data.mimic_iv_dir)
    hosp = root / "hosp"
    icu = root / "icu"

    patients = pd.read_csv(
        hosp / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        usecols=[
            "subject_id", "hadm_id", "admittime", "dischtime",
            "admission_type", "admission_location", "discharge_location",
            "insurance", "marital_status", "race", "edregtime",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime", "edregtime"],
    )
    diagnoses = pd.read_csv(
        hosp / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )

    labs = _load_real_measurements(
        hosp / "labevents.csv.gz", LAB_ITEMIDS,
        cols=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"],
    )
    vitals = _load_real_measurements(
        icu / "chartevents.csv.gz", VITAL_ITEMIDS,
        cols=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"],
    )
    measurements = pd.concat([labs, vitals], ignore_index=True)

    return {
        "patients": patients,
        "admissions": admissions,
        "measurements": measurements,
        "diagnoses_icd": diagnoses,
    }


def _load_real_measurements(
    path: Path, itemid_map: dict, cols: list[str]
) -> pd.DataFrame:
    """Read a large MIMIC events file in chunks, keep curated itemids."""
    wanted = {iid: name for name, iids in itemid_map.items() for iid in iids}
    keep = []
    for chunk in pd.read_csv(
        path, usecols=cols, parse_dates=["charttime"], chunksize=2_000_000
    ):
        sub = chunk[chunk["itemid"].isin(wanted)].copy()
        if len(sub):
            sub["item"] = sub["itemid"].map(wanted)
            keep.append(sub[["subject_id", "hadm_id", "item", "valuenum", "charttime"]])
    return (
        pd.concat(keep, ignore_index=True) if keep
        else pd.DataFrame(columns=["subject_id", "hadm_id", "item", "valuenum", "charttime"])
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feature building
# ──────────────────────────────────────────────────────────────────────────────

def compute_readmission_label(admissions: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Compute the 30-day readmission label(s) from admission timing (real data).

    Sets two columns:

    ``TARGET_COL`` (``readmission_30d``, all-cause) — any subsequent
    admission within the window, excluding only in-hospital deaths. This is
    kept as the primary/comparability label (matches most of the readmission
    literature) but is not what "unplanned readmission" means.

    ``TARGET_COL_UNPLANNED`` (``readmission_30d_unplanned``) — the same,
    additionally requiring the OUTCOME admission's own ``admission_type`` not
    indicate a planned return (elective / same-day surgical). A structured-
    data proxy, not a note-derived one: scheduled chemotherapy cycles and
    staged surgical returns are excluded; other forms of planned return not
    reflected in ``admission_type`` are not caught. See P7b in the
    2026-08-19 remediation review and ``docs/ARCHITECTURE.md``.
    """
    window = cfg.cohort.readmission_window_days
    adm = admissions.sort_values(["subject_id", "admittime"]).copy()
    adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)
    adm["next_admission_type"] = adm.groupby("subject_id")["admission_type"].shift(-1)
    days_to_next = (adm["next_admittime"] - adm["dischtime"]).dt.total_seconds() / 86400

    is_readmission = (
        (days_to_next >= 0) & (days_to_next <= window)
        & (adm["hospital_expire_flag"] == 0)
    )
    adm[TARGET_COL] = is_readmission.astype(int)

    next_is_planned = (
        adm["next_admission_type"].str.upper().isin(_PLANNED_READMISSION_TYPES)
    ).fillna(False)
    adm[TARGET_COL_UNPLANNED] = (is_readmission & ~next_is_planned).astype(int)

    return adm.drop(columns=["next_admittime", "next_admission_type"])


def _apply_cohort_filters(
    adm: pd.DataFrame, patients: pd.DataFrame, cfg: AppConfig
) -> pd.DataFrame:
    """Apply inclusion / exclusion criteria from the cohort config."""
    coh = cfg.cohort
    adm = adm.merge(
        patients[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left"
    )
    adm = adm[adm["anchor_age"] >= coh.min_age]
    if coh.exclude_in_hospital_death:
        adm = adm[adm["hospital_expire_flag"] == 0]
    if coh.exclude_elective:
        adm = adm[adm["admission_type"].str.upper() != "ELECTIVE"]
    return adm.reset_index(drop=True)


def _aggregate_measurements(
    meas: pd.DataFrame, items: list[str], aggs: list[str]
) -> pd.DataFrame:
    """Return per-admission aggregates for the requested items.

    ``"last"`` means the value at the latest charttime.
    """
    meas = meas[meas["item"].isin(items)].copy()
    out = None
    if "last" in aggs:
        last = (
            meas.sort_values("charttime")
            .groupby(["hadm_id", "item"])["valuenum"]
            .last()
            .unstack()
        )
        last.columns = [f"{c}_last" for c in last.columns]
        out = last
    other = [a for a in aggs if a != "last"]
    if other:
        agg = meas.groupby(["hadm_id", "item"])["valuenum"].agg(other).unstack()
        agg.columns = [f"{item}_{stat}" for stat, item in agg.columns]
        out = agg if out is None else out.join(agg, how="outer")
    return out.reset_index() if out is not None else pd.DataFrame({"hadm_id": []})


def _load_note_hadm_ids(cfg: AppConfig) -> set | None:
    """Return the set of hadm_ids that have a discharge note, or None if unavailable.

    Reads only the ``hadm_id`` column — no text loaded — so the call is fast
    even against the multi-GB notes file.  Returns None when the notes directory
    is not configured or the file cannot be found, which the caller treats as
    unknown (NaN) rather than absent (0).
    """
    note_dir = cfg.data.mimic_iv_note_dir
    if not note_dir:
        return None
    for candidate in [
        Path(note_dir) / "note" / "discharge.csv.gz",
        Path(note_dir) / "discharge.csv.gz",
    ]:
        if candidate.exists():
            ids = pd.read_csv(candidate, usecols=["hadm_id"])["hadm_id"].dropna()
            return set(ids.astype("int64"))
    return None


def build_features(tables: dict[str, pd.DataFrame], cfg: AppConfig) -> pd.DataFrame:
    """Build the Stage 1 feature matrix from raw data tables.

    Args:
        tables: dict with keys ``patients``, ``admissions``, ``measurements``,
                ``diagnoses_icd`` — each a normalised DataFrame.
        cfg:    validated project config.

    Returns:
        Feature matrix DataFrame including identifiers, features, subgroup
        columns, and the target column.
    """
    patients = tables["patients"]
    admissions = tables["admissions"]
    measurements = tables["measurements"]
    diagnoses = tables["diagnoses_icd"]

    if TARGET_COL not in admissions.columns:
        admissions = compute_readmission_label(admissions, cfg)

    adm = _apply_cohort_filters(admissions, patients, cfg)

    # Index-admission traits
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    adm["admission_type_emergency"] = (
        adm["admission_type"].str.upper().isin(_EMERGENCY_TYPES)
    ).astype(int)
    adm["admission_type_observation"] = (
        adm["admission_type"].str.upper().isin(_OBSERVATION_TYPES)
    ).astype(int)
    adm["is_medicaid_medicare"] = (
        adm["insurance"].str.upper().isin({"MEDICAID", "MEDICARE"})
        if "insurance" in adm.columns
        else 0
    ).astype(int)
    adm["came_via_ed"] = (
        adm["edregtime"].notna()
        | (adm["admission_location"].fillna("").str.upper() == "EMERGENCY ROOM")
    ).astype(int)
    adm["discharged_to_facility"] = adm["discharge_location"].fillna("").str.upper().isin(
        ["SKILLED NURSING FACILITY", "REHAB", "CHRONIC/LONG TERM CARE",
         "CHRONIC/LONG TERM ACUTE CARE"]
    ).astype(int)
    adm["admit_hour"] = adm["admittime"].dt.hour
    adm["admit_is_weekend"] = (adm["admittime"].dt.dayofweek >= 5).astype(int)
    adm["admit_is_night"] = (
        (adm["admit_hour"] < 7) | (adm["admit_hour"] >= 19)
    ).astype(int)

    # Prior utilisation (computed on full timeline, then merged onto cohort)
    full = admissions.sort_values(["subject_id", "admittime"]).copy()
    full["n_prior_admissions"] = full.groupby("subject_id").cumcount()
    full["had_ed"] = (
        full["edregtime"].notna()
        | (full["admission_location"].fillna("").str.upper() == "EMERGENCY ROOM")
    ).astype(int)
    full["n_prior_ed_visits"] = (
        full.groupby("subject_id")["had_ed"].cumsum() - full["had_ed"]
    )
    full["prev_dischtime"] = full.groupby("subject_id")["dischtime"].shift(1)
    full["days_since_last_discharge"] = (
        (full["admittime"] - full["prev_dischtime"]).dt.total_seconds() / 86400
    )
    prior = full[["hadm_id", "n_prior_admissions", "n_prior_ed_visits",
                  "days_since_last_discharge"]]
    adm = adm.merge(prior, on="hadm_id", how="left")

    # Demographics
    adm["age"] = adm["anchor_age"]
    adm["gender_M"] = (adm["gender"].str.upper() == "M").astype(int)

    # Note availability — NaN when notes directory not configured (XGBoost routes
    # NaN to its learned default branch; no imputation needed).
    note_ids = _load_note_hadm_ids(cfg)
    if note_ids is not None:
        adm["has_discharge_note"] = adm["hadm_id"].isin(note_ids).astype(int)
    else:
        adm["has_discharge_note"] = float("nan")

    # Comorbidity index
    como = charlson_per_admission(diagnoses)
    adm = adm.merge(como, on="hadm_id", how="left")
    adm["charlson_index"] = adm["charlson_index"].fillna(0)
    adm["n_comorbidities"] = adm["n_comorbidities"].fillna(0)

    # Labs and vitals
    lab_feats = _aggregate_measurements(
        measurements, cfg.features.lab_items, cfg.features.lab_aggregates)
    vital_feats = _aggregate_measurements(
        measurements, cfg.features.vital_items, ["last"])
    adm = adm.merge(lab_feats, on="hadm_id", how="left")
    adm = adm.merge(vital_feats, on="hadm_id", how="left")

    # Subgroup columns (fairness analysis only)
    adm["gender"] = adm["gender"].str.upper()
    adm["age_band"] = pd.cut(
        adm["age"], bins=[17, 40, 55, 70, 120],
        labels=["18-40", "41-55", "56-70", "70+"],
    )

    # Assemble final matrix
    feature_cols = [
        "age", "gender_M", "los_days",
        "admission_type_emergency", "admission_type_observation",
        "is_medicaid_medicare", "has_discharge_note",
        "came_via_ed", "discharged_to_facility",
        "admit_hour", "admit_is_weekend", "admit_is_night",
        "n_prior_admissions", "n_prior_ed_visits", "days_since_last_discharge",
        "charlson_index", "n_comorbidities",
    ]
    all_items = cfg.features.lab_items + cfg.features.vital_items
    feature_cols += [
        c for c in adm.columns
        if any(c.startswith(f"{it}_") for it in all_items)
    ]

    keep = ID_COLS + feature_cols + ["gender", "age_band", TARGET_COL, TARGET_COL_UNPLANNED]
    return adm[keep].copy()


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Return model-input columns: everything except identifiers, subgroups, target."""
    exclude = set(ID_COLS + ["gender", "age_band", TARGET_COL, TARGET_COL_UNPLANNED])
    return [c for c in matrix.columns if c not in exclude]


def load_feature_matrix(cfg: AppConfig, mode: str | None = None) -> pd.DataFrame:
    """Load the processed feature matrix (building it if absent).

    In quick mode the matrix is deterministically subsampled to
    ``run.quick.n_rows`` for fast iteration. The same ``(cfg, mode, seed)``
    always yields the same matrix, so train / tune / evaluate share an
    identical split.

    Args:
        cfg:  validated project config.
        mode: ``"quick"`` or ``"full"``; defaults to ``cfg.run.mode``.

    Returns:
        Feature matrix DataFrame.
    """
    mode = mode or cfg.run.mode
    out_dir = get_data_dir().parent / cfg.output.processed_dir
    path = out_dir / "features.csv"

    if path.exists():
        matrix = pd.read_csv(path)
    else:
        matrix = build_features(load_raw_tables(cfg), cfg)
        out_dir.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(path, index=False)

    n_rows = cfg.run.active(mode).n_rows
    if n_rows and len(matrix) > n_rows:
        matrix = matrix.sample(
            n=n_rows, random_state=cfg.run.random_state
        ).reset_index(drop=True)
    return matrix


def split_xy(matrix: pd.DataFrame):
    """Return ``(features, y, groups, subgroups, feature_cols)`` for modelling."""
    feat_cols = feature_columns(matrix)
    features = matrix[feat_cols].reset_index(drop=True)
    y = matrix[TARGET_COL].to_numpy()
    groups = matrix["subject_id"].to_numpy()
    subgroups = matrix[["gender", "age_band"]].reset_index(drop=True)
    return features, y, groups, subgroups, feat_cols


def main() -> None:
    """Build and persist the feature matrix.

    Always builds from the full available dataset (real MIMIC-IV if
    configured, else the full synthetic set) regardless of ``--mode`` --
    quick/full only affects row-subsampling at *load* time
    (``load_feature_matrix``), not here. ``--mode`` is still accepted (and
    was already being silently ignored as an unparsed argument before this
    fix, since this script had no argparse at all) purely for CLI
    consistency with train.py/tune.py/evaluate.py.
    """
    parser = argparse.ArgumentParser(description="Build the Stage 1 feature matrix")
    parser.add_argument("--mode", choices=["quick", "full"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg.run.mode = args.mode
    tables = load_raw_tables(cfg)
    matrix = build_features(tables, cfg)

    out_dir = get_data_dir().parent / cfg.output.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.csv"
    matrix.to_csv(out_path, index=False)

    print(f"[features] Wrote {len(matrix):,} rows x {matrix.shape[1]} cols -> {out_path}")
    print(f"[features] Model features: {len(feature_columns(matrix))}")
    print(f"[features] Readmission rate: {matrix[TARGET_COL].mean():.1%}")


if __name__ == "__main__":
    main()
