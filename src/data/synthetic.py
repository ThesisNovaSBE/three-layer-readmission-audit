"""Generate synthetic data matching the MIMIC-IV structured schema.

Produces fake RAW tables that mirror MIMIC-IV column names, so that
`src/data/features.py` runs identically on synthetic and real data:

- patients          (subject_id, gender, anchor_age)
- admissions        (+ admission/discharge location, edregtime, readmission_30d label)
- measurements      (long format: subject_id, hadm_id, item, valuenum, charttime)
- diagnoses_icd     (subject_id, hadm_id, seq_num, icd_code, icd_version)

The readmission label is drawn from a latent risk model so the data is actually
learnable (signal in age, comorbidity, prior utilisation, emergency, length of
stay, discharge destination, and a couple of labs), at a realistic ~10% base rate.

Usage:
    python -m src.data.synthetic              # writes raw tables to data/synthetic/
    python -m src.data.synthetic --n 5000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_data_dir, load_config
from src.config_schema import AppConfig
from src.data.comorbidity import CHARLSON_ICD10, charlson_per_admission


_LAB_DISTRIBUTIONS = {
    "glucose": (120.0, 45.0),
    "creatinine": (1.1, 0.8),
    "hemoglobin": (12.5, 2.0),
    "white_blood_cells": (9.0, 4.0),
    "platelets": (250.0, 80.0),
    "sodium": (139.0, 4.0),
    "potassium": (4.1, 0.5),
    "bicarbonate": (24.0, 4.0),
}

_VITAL_DISTRIBUTIONS = {
    "heart_rate": (82.0, 16.0),
    "systolic_bp": (125.0, 20.0),
    "diastolic_bp": (72.0, 12.0),
    "temperature": (36.8, 0.5),
    "respiratory_rate": (18.0, 4.0),
    "spo2": (96.5, 2.5),
}

_INSURANCE_TYPES = ["Medicare", "Medicaid", "Other"]
_MARITAL_STATUSES = ["MARRIED", "SINGLE", "DIVORCED", "WIDOWED"]
_RACES = ["WHITE", "BLACK", "ASIAN", "HISPANIC", "OTHER"]
_ADMIT_LOCATIONS = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL",
                    "WALK-IN/SELF REFERRAL"]
_DISCHARGE_FACILITY = ["SKILLED NURSING FACILITY", "REHAB", "CHRONIC/LONG TERM CARE"]
_DISCHARGE_LOCATIONS = ["HOME", "HOME HEALTH CARE", *_DISCHARGE_FACILITY]

_MEAS_REPS = 4                     # measurements per item per admission
_MEAS_FRACTIONS = [0.1, 0.4, 0.7, 0.95]
# Non-Charlson "filler" diagnosis codes (ICD-10)
_FILLER_ICD10 = ["Z00", "R10", "R51", "M545", "K219", "F329", "R079"]


def generate_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a patients DataFrame with n synthetic subjects."""
    return pd.DataFrame({
        "subject_id": np.arange(100_000, 100_000 + n),
        "gender": rng.choice(["M", "F"], size=n, p=[0.52, 0.48]),
        "anchor_age": rng.integers(18, 90, size=n),
    })


def generate_admissions(patients: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return an admissions DataFrame with random encounter data per patient."""
    rows = []
    hadm_id = 200_000
    for _, pat in patients.iterrows():
        n_admits = rng.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.15, 0.05])
        base_date = pd.Timestamp("2120-01-01") + pd.Timedelta(days=int(rng.integers(0, 1000)))
        for _ in range(n_admits):
            los_hours = max(12, rng.normal(144, 96))
            admittime = base_date + pd.Timedelta(days=int(rng.integers(0, 60)))
            dischtime = admittime + pd.Timedelta(hours=los_hours)
            is_emergency = rng.random() < 0.65
            died = rng.random() < 0.03

            if is_emergency:
                admit_loc = "EMERGENCY ROOM"
                edregtime = admittime - pd.Timedelta(minutes=int(rng.integers(30, 600)))
            else:
                admit_loc = rng.choice(_ADMIT_LOCATIONS[1:])
                edregtime = pd.NaT

            rows.append({
                "subject_id": pat["subject_id"],
                "hadm_id": hadm_id,
                "admittime": admittime,
                "dischtime": dischtime,
                "admission_type": "EMERGENCY" if is_emergency else rng.choice(
                    ["URGENT", "ELECTIVE", "SURGICAL"]),
                "admission_location": admit_loc,
                "discharge_location": ("HOME" if died else rng.choice(
                    _DISCHARGE_LOCATIONS, p=[0.55, 0.20, 0.12, 0.08, 0.05])),
                "insurance": rng.choice(_INSURANCE_TYPES),
                "marital_status": rng.choice(_MARITAL_STATUSES),
                "race": rng.choice(_RACES),
                "edregtime": edregtime,
                "hospital_expire_flag": int(died),
            })
            hadm_id += 1
            base_date = dischtime + pd.Timedelta(days=int(rng.integers(5, 200)))
    return pd.DataFrame(rows)


def generate_diagnoses(
    admissions: pd.DataFrame, patients: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ICD-10 codes per admission; comorbidity burden rises with age."""
    age_by_subject = dict(zip(patients["subject_id"], patients["anchor_age"]))
    categories = list(CHARLSON_ICD10.keys())
    rows = []
    for _, adm in admissions.iterrows():
        age = age_by_subject[adm["subject_id"]]
        n_comorbid = rng.poisson(0.3 + age / 45.0)
        seq = 1
        for _ in range(n_comorbid):
            cat = rng.choice(categories)
            code = rng.choice(CHARLSON_ICD10[cat])     # prefix used directly as code
            rows.append({"subject_id": adm["subject_id"], "hadm_id": adm["hadm_id"],
                         "seq_num": seq, "icd_code": code, "icd_version": 10})
            seq += 1
        for _ in range(int(rng.integers(1, 4))):       # filler codes
            rows.append({"subject_id": adm["subject_id"], "hadm_id": adm["hadm_id"],
                         "seq_num": seq, "icd_code": rng.choice(_FILLER_ICD10),
                         "icd_version": 10})
            seq += 1
    return pd.DataFrame(rows)


def _draw_baselines(admissions: pd.DataFrame, dists: dict, rng: np.random.Generator) -> dict:
    """One baseline value per admission per item."""
    n = len(admissions)
    return {item: np.clip(rng.normal(mu, sigma, size=n), 0.1, None)
            for item, (mu, sigma) in dists.items()}


def generate_measurements(
    admissions: pd.DataFrame, baselines: dict, all_dists: dict, rng: np.random.Generator
) -> pd.DataFrame:
    """Expand per-admission baselines into long-format repeated measurements."""
    los_seconds = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds().values
    admit_ns = admissions["admittime"].values.astype("datetime64[ns]")
    sid = admissions["subject_id"].values
    hid = admissions["hadm_id"].values

    frames = []
    for item, (_, sigma) in all_dists.items():
        base = baselines[item]
        for frac in _MEAS_FRACTIONS:
            vals = np.clip(base + rng.normal(0, sigma * 0.25, size=len(base)), 0.1, None)
            offsets = (los_seconds * frac * 1e9).astype("timedelta64[ns]")
            frames.append(pd.DataFrame({
                "subject_id": sid,
                "hadm_id": hid,
                "item": item,
                "valuenum": vals,
                "charttime": admit_ns + offsets,
            }))
    return pd.concat(frames, ignore_index=True)


def assign_readmission_label(
    admissions: pd.DataFrame, lab_baselines: dict, rng: np.random.Generator,
    patients: pd.DataFrame, diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """Latent-risk Bernoulli label: learnable signal at a realistic base rate."""
    adm = admissions.sort_values(["subject_id", "admittime"]).copy()
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    adm["n_prior_admissions"] = adm.groupby("subject_id").cumcount()
    adm = adm.merge(patients[["subject_id", "anchor_age"]], on="subject_id", how="left")

    charlson = charlson_per_admission(diagnoses).set_index("hadm_id")["charlson_index"]
    adm["charlson_index"] = adm["hadm_id"].map(charlson).fillna(0).values

    # Align lab baselines (indexed by the original admissions order) onto adm via hadm_id
    base_df = pd.DataFrame({"hadm_id": admissions["hadm_id"].values,
                            "creatinine": lab_baselines["creatinine"],
                            "white_blood_cells": lab_baselines["white_blood_cells"]})
    adm = adm.merge(base_df, on="hadm_id", how="left")

    is_emergency = (adm["admission_type"] == "EMERGENCY").astype(int)
    to_facility = adm["discharge_location"].isin(_DISCHARGE_FACILITY).astype(int)

    logit = (
        -2.9
        + 0.020 * (adm["anchor_age"] - 60)
        + 0.350 * adm["charlson_index"]
        + 0.300 * adm["n_prior_admissions"]
        + 0.450 * is_emergency
        + 0.015 * (adm["los_days"] - 6)
        + 0.600 * to_facility
        + 0.250 * (adm["creatinine"] - 1.1)
        + 0.050 * (adm["white_blood_cells"] - 9.0)
        + rng.normal(0, 0.5, size=len(adm))
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.random(len(adm)) < prob).astype(int)
    # Deaths cannot be readmitted
    label = np.where(adm["hospital_expire_flag"].values == 1, 0, label)

    adm["readmission_30d"] = label
    # Synthetic data has no note-derived or admission-type-based "planned
    # return" signal beyond what's already folded into the logistic risk
    # model above, so the unplanned variant is set equal to the all-cause
    # label here — sufficient for the code-correctness checks synthetic data
    # is used for, not a claim about real-data planned-return rates.
    adm["readmission_30d_unplanned"] = label
    return adm[admissions.columns.tolist() + ["readmission_30d", "readmission_30d_unplanned"]]


def generate_synthetic_dataset(
    n_patients: int = 2000, seed: int = 42, output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate raw synthetic tables. Returns a dict of DataFrames."""
    rng = np.random.default_rng(seed)

    patients = generate_patients(n_patients, rng)
    admissions = generate_admissions(patients, rng)
    diagnoses = generate_diagnoses(admissions, patients, rng)

    all_dists = {**_LAB_DISTRIBUTIONS, **_VITAL_DISTRIBUTIONS}
    baselines = _draw_baselines(admissions, all_dists, rng)
    measurements = generate_measurements(admissions, baselines, all_dists, rng)

    admissions = assign_readmission_label(
        admissions, baselines, rng, patients, diagnoses)

    tables = {
        "patients": patients,
        "admissions": admissions,
        "measurements": measurements,
        "diagnoses_icd": diagnoses,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"Synthetic raw tables written to {output_dir}/")
        print(f"  Patients:     {len(patients):,}")
        print(f"  Admissions:   {len(admissions):,}")
        print(f"  Measurements: {len(measurements):,}")
        print(f"  Diagnoses:    {len(diagnoses):,}")
        print(f"  Readmission rate: {admissions['readmission_30d'].mean():.1%}")

    return tables


def _main(cfg: AppConfig, n_patients: int | None, seed: int | None) -> None:
    """Generate and persist synthetic data tables."""
    n_out = n_patients or cfg.data.synthetic_n_patients
    seed_out = seed or cfg.data.synthetic_seed
    generate_synthetic_dataset(
        n_patients=n_out,
        seed=seed_out,
        output_dir=get_data_dir() / "synthetic",
    )


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV-like data")
    _parser.add_argument("--n", type=int, default=None, help="Number of patients")
    _parser.add_argument("--seed", type=int, default=None, help="Random seed")
    _args = _parser.parse_args()
    _main(load_config(), _args.n, _args.seed)
