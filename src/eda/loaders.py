"""Raw-table loaders for the EDA, with an on-disk cache under ``data/interim/``.

The cache lives under ``/data/`` (gitignored, DUA-protected, same as every
other local MIMIC derivative) -- it holds row-level data and must never be
moved into ``results/``.

Why a separate loader from ``src/data/features.py``: the pipeline's loader
maps lab/vital ``itemid`` to a canonical name and drops the ``itemid``,
which is exactly the information needed to audit the itemid map (e.g. the
``temperature`` feature merges a Fahrenheit and a Celsius itemid). This
loader keeps it, then hands the same table to the pipeline's own
``build_features`` so the EDA describes precisely the matrix the models see.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.config import get_data_dir, load_config
from src.config_schema import AppConfig
from src.data.features import LAB_ITEMIDS, VITAL_ITEMIDS, build_features

CACHE_DIR = get_data_dir() / "interim" / "eda_cache"


def _cached(name: str, builder, force: bool = False) -> pd.DataFrame:
    """Return the cached pickle ``name`` or build + cache it."""
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists() and not force:
        return pd.read_pickle(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = builder()
    df.to_pickle(path)
    print(f"[eda/loaders] built {name}: {len(df):,} rows in {time.time() - t0:,.0f}s")
    return df


def mimic_paths(cfg: AppConfig | None = None) -> tuple[Path, Path]:
    """Return (mimic_iv_dir, mimic_iv_note_dir); raises if not configured."""
    cfg = cfg or load_config()
    if not cfg.data.mimic_iv_dir or not Path(cfg.data.mimic_iv_dir).exists():
        raise FileNotFoundError("MIMIC_IV_DIR not configured (see .env.example).")
    return Path(cfg.data.mimic_iv_dir), Path(cfg.data.mimic_iv_note_dir)


def load_patients(cfg: AppConfig | None = None) -> pd.DataFrame:
    """patients incl. ``dod`` (post-discharge death) and ``anchor_year_group``."""
    hosp = mimic_paths(cfg)[0] / "hosp"
    return _cached("patients", lambda: pd.read_csv(
        hosp / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year",
                 "anchor_year_group", "dod"],
        parse_dates=["dod"],
    ))


def load_admissions(cfg: AppConfig | None = None) -> pd.DataFrame:
    """All admissions columns the pipeline and the EDA use."""
    hosp = mimic_paths(cfg)[0] / "hosp"
    return _cached("admissions", lambda: pd.read_csv(
        hosp / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_type", "admission_location", "discharge_location",
                 "insurance", "language", "marital_status", "race", "edregtime",
                 "hospital_expire_flag"],
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime"],
    ))


def load_diagnoses(cfg: AppConfig | None = None) -> pd.DataFrame:
    """diagnoses_icd (all codes, all positions)."""
    hosp = mimic_paths(cfg)[0] / "hosp"
    return _cached("diagnoses_icd", lambda: pd.read_csv(
        hosp / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
        dtype={"icd_code": "string"},
    ))


def load_icustays(cfg: AppConfig | None = None) -> pd.DataFrame:
    """icustays -- explains where the (ICU-only) vital-sign features come from."""
    icu = mimic_paths(cfg)[0] / "icu"
    return _cached("icustays", lambda: pd.read_csv(
        icu / "icustays.csv.gz", usecols=["hadm_id", "stay_id", "los"]))


def _scan_events(path: Path, itemid_map: dict[str, list[int]]) -> pd.DataFrame:
    """Chunked scan of a large events file, keeping curated itemids (+ itemid)."""
    wanted = {iid: name for name, iids in itemid_map.items() for iid in iids}
    keep, n_rows, n_no_hadm = [], 0, 0
    for chunk in pd.read_csv(
        path, usecols=["hadm_id", "itemid", "valuenum", "charttime"],
        parse_dates=["charttime"], chunksize=5_000_000,
        dtype={"itemid": "int32", "valuenum": "float32", "hadm_id": "float64"},
    ):
        n_rows += len(chunk)
        sub = chunk[chunk["itemid"].isin(wanted)]
        n_no_hadm += int(sub["hadm_id"].isna().sum())
        sub = sub.dropna(subset=["hadm_id"])
        if len(sub):
            sub = sub.assign(hadm_id=sub["hadm_id"].astype("int64"),
                             item=sub["itemid"].map(wanted))
            keep.append(sub)
    out = pd.concat(keep, ignore_index=True)
    out["item"] = out["item"].astype("category")
    out.attrs["n_rows_scanned"] = n_rows
    out.attrs["n_curated_without_hadm"] = n_no_hadm
    print(f"[eda/loaders] {path.name}: scanned {n_rows:,} rows, kept {len(out):,} "
          f"curated rows ({n_no_hadm:,} curated rows had no hadm_id)")
    return out


def load_measurements(cfg: AppConfig | None = None) -> pd.DataFrame:
    """Curated labs (labevents) + vitals (chartevents), with ``itemid`` kept.

    The scan of ~160M labevents and ~430M chartevents rows takes roughly
    20-40 min on a laptop; it runs once and is cached. ``source`` tells labs
    from vitals; ``attrs`` carry the total rows scanned for the inventory.
    """
    root = mimic_paths(cfg)[0]

    def _build() -> pd.DataFrame:
        labs = _scan_events(root / "hosp" / "labevents.csv.gz", LAB_ITEMIDS)
        vitals = _scan_events(root / "icu" / "chartevents.csv.gz", VITAL_ITEMIDS)
        scans = pd.DataFrame({
            "table": ["labevents", "chartevents"],
            "n_rows_scanned": [labs.attrs["n_rows_scanned"], vitals.attrs["n_rows_scanned"]],
            "n_curated_without_hadm": [labs.attrs["n_curated_without_hadm"],
                                       vitals.attrs["n_curated_without_hadm"]],
        })
        scans.to_pickle(CACHE_DIR / "event_scan_counts.pkl")
        both = pd.concat([labs.assign(source="lab"), vitals.assign(source="vital")],
                         ignore_index=True)
        both["item"] = both["item"].astype("category")
        both["source"] = both["source"].astype("category")
        return both

    return _cached("measurements", _build)


def load_event_scan_counts() -> pd.DataFrame | None:
    """Rows scanned per events table (written by :func:`load_measurements`)."""
    path = CACHE_DIR / "event_scan_counts.pkl"
    return pd.read_pickle(path) if path.exists() else None


def load_feature_matrix_fresh(cfg: AppConfig | None = None) -> pd.DataFrame:
    """Build the Stage 1 feature matrix from raw data with the pipeline's own code.

    Deliberately does NOT read ``data/processed/features.csv``: that file is
    a local build artifact and can be stale (the copy on this laptop as of
    2026-11-06 predates the unplanned-readmission target). Cached separately
    so it never overwrites the pipeline's file.
    """
    cfg = cfg or load_config()

    def _build() -> pd.DataFrame:
        tables = {
            "patients": load_patients(cfg)[["subject_id", "gender", "anchor_age"]],
            "admissions": load_admissions(cfg).drop(columns=["deathtime", "language"]),
            "measurements": load_measurements(cfg)[
                ["hadm_id", "item", "valuenum", "charttime"]],
            "diagnoses_icd": load_diagnoses(cfg),
        }
        return build_features(tables, cfg)

    return _cached("feature_matrix", _build)
