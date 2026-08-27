"""Column schemas for the readmission prediction pipeline.

These schemas define the contract between data stages. Both the real MIMIC-IV
loader and the synthetic generator must produce DataFrames matching these schemas.

Raw tables mirror the MIMIC-IV column names so that `src/data/features.py` can run
unchanged on real data and on synthetic data.
"""

PATIENTS_COLS = {
    "subject_id": "int64",
    "gender": "object",
    "anchor_age": "int64",
}

ADMISSIONS_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "admittime": "datetime64[ns]",
    "dischtime": "datetime64[ns]",
    "admission_type": "object",
    "admission_location": "object",
    "discharge_location": "object",
    "insurance": "object",
    "marital_status": "object",
    "race": "object",
    "edregtime": "datetime64[ns]",   # nullable; presence indicates an ED visit
    "hospital_expire_flag": "int64",
    "readmission_30d": "int64",       # label (set by cohort.py / synthetic generator)
}

# Long-format measurements (mirrors MIMIC-IV labevents / chartevents after itemid mapping)
# One row per measurement.
MEASUREMENTS_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "item": "object",        # canonical name, e.g. "glucose", "heart_rate"
    "valuenum": "float64",
    "charttime": "datetime64[ns]",
}

# Diagnoses (mirrors MIMIC-IV diagnoses_icd)
DIAGNOSES_COLS = {
    "subject_id": "int64",
    "hadm_id": "int64",
    "seq_num": "int64",
    "icd_code": "object",
    "icd_version": "int64",   # 9 or 10
}

# Identifier columns carried through the feature matrix but never used as model inputs.
ID_COLS = ["subject_id", "hadm_id"]
TARGET_COL = "readmission_30d"

# All-cause label minus outcome admissions whose own admission_type indicates
# a planned return (elective / same-day surgical). Structured-data proxy for
# "unplanned" — see docs/ARCHITECTURE.md and compute_readmission_label() in
# src/data/features.py. Carried through the feature matrix, never a model
# input, same as TARGET_COL.
TARGET_COL_UNPLANNED = "readmission_30d_unplanned"

# Columns kept only for fairness/subgroup analysis (not model inputs).
SUBGROUP_COLS = ["gender", "age_band"]
