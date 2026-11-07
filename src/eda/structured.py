"""EDA of the structured MIMIC-IV data, as the pipeline uses it.

Every section answers a question the thesis has to be able to defend
(see ``sessions/2026-11-06_session-37.md`` for the reasoning behind each):

==========================  ==============================================
Section                     Thesis question it answers
==========================  ==============================================
inventory                   What data exists, and what do we actually use?
cohort_flow                 How do 546k admissions become the model cohort?
outcome                     How common is the target, and is the label sound
                            (planned returns, same-day transfers, deaths
                            after discharge = competing risk)?
utilisation                 How clustered are admissions in patients (why a
                            patient-level split is mandatory)?
table1                      Who is in the cohort, and how do readmitted
                            admissions differ (SMD)?
subgroups                   Base rates per fairness subgroup (the 70+ gap).
temporal                    Does the base rate / coding drift over years?
note_coverage               Who has a discharge note (Stage 2/3 population)?
missingness                 Which features are mostly missing, and is the
                            missingness itself informative?
univariate_auroc            How much signal does each feature carry alone?
redundancy                  Which features are near-duplicates?
measurement_quality         Is the curated lab/vital itemid map sound (units,
                            post-discharge values, timing of "last" value)?
diagnoses                   ICD-9/ICD-10 mix (affects Charlson), Charlson
                            distribution, primary-diagnosis chapters.
==========================  ==============================================

Outputs (aggregate-only, see ``src/eda/privacy.py``):
``results/eda/structured_summary.json`` and ``results/eda/figures/s*.png``.

Usage::

    python -m src.eda.structured
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.data.features import (
    _EMERGENCY_TYPES,
    _apply_cohort_filters,
    compute_readmission_label,
    feature_columns,
)
from src.eda import loaders as L
from src.eda import plots
from src.eda.privacy import (
    MIN_CELL,
    rate_by_group,
    safe_count,
    safe_histogram,
    safe_quantiles,
    safe_rate,
    safe_value_counts,
    smd,
    write_json,
)
from src.schemas import TARGET_COL, TARGET_COL_UNPLANNED

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "eda"
TARGET = TARGET_COL_UNPLANNED  # the model target (src/schemas.py MODEL_TARGET_COL)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def race_group(race: pd.Series) -> pd.Series:
    """Collapse MIMIC-IV's ~33 race strings into 6 reporting groups."""
    r = race.fillna("UNKNOWN").str.upper()
    out = pd.Series("OTHER", index=race.index)
    out[r.str.startswith("WHITE")] = "WHITE"
    out[r.str.startswith("BLACK")] = "BLACK"
    out[r.str.startswith("HISPANIC") | r.str.contains("SOUTH AMERICAN")] = "HISPANIC/LATINO"
    out[r.str.startswith("ASIAN")] = "ASIAN"
    out[r.isin(["UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"])] = "UNKNOWN"
    return out


def admission_year_group(adm: pd.DataFrame, patients: pd.DataFrame) -> pd.Series:
    """Approximate *real* calendar 3-year group of each admission.

    MIMIC shifts every patient's dates into the future; ``anchor_year_group``
    gives the real 3-year window that ``anchor_year`` falls into. Shifting
    by (admission year - anchor_year) gives an approximate real year for
    each admission, accurate to the 3-year window width.
    """
    p = patients.set_index("subject_id")
    anchor_year = adm["subject_id"].map(p["anchor_year"])
    start = adm["subject_id"].map(p["anchor_year_group"]).str[:4].astype(float)
    approx = start + 1 + (adm["admittime"].dt.year - anchor_year)
    bins = [2007, 2010, 2013, 2016, 2019, 2023]
    labels = ["2008-2010", "2011-2013", "2014-2016", "2017-2019", "2020-2022"]
    return pd.cut(approx, bins=bins, labels=labels).astype(str)


_ICD9_CHAPTERS = [  # (upper bound of numeric range, chapter)
    (139, "Infectious"), (239, "Neoplasms"), (279, "Endocrine/metabolic"),
    (289, "Blood"), (319, "Mental"), (389, "Nervous/sense"), (459, "Circulatory"),
    (519, "Respiratory"), (579, "Digestive"), (629, "Genitourinary"),
    (679, "Pregnancy"), (709, "Skin"), (739, "Musculoskeletal"),
    (759, "Congenital"), (779, "Perinatal"), (799, "Symptoms/ill-defined"),
    (999, "Injury/poisoning"),
]
_ICD10_CHAPTERS = {
    "A": "Infectious", "B": "Infectious", "C": "Neoplasms", "D": "Blood",
    "E": "Endocrine/metabolic", "F": "Mental", "G": "Nervous/sense",
    "H": "Nervous/sense", "I": "Circulatory", "J": "Respiratory",
    "K": "Digestive", "L": "Skin", "M": "Musculoskeletal", "N": "Genitourinary",
    "O": "Pregnancy", "P": "Perinatal", "Q": "Congenital",
    "R": "Symptoms/ill-defined", "S": "Injury/poisoning", "T": "Injury/poisoning",
    "V": "External causes", "W": "External causes", "X": "External causes",
    "Y": "External causes", "Z": "Health-services contact",
}


def icd_chapter(code: pd.Series, version: pd.Series) -> pd.Series:
    """Map ICD-9-CM / ICD-10-CM codes to a harmonised chapter label.

    ICD-10 D50-D89 (blood) vs D00-D49 (neoplasms) is split by number; ICD-9
    V/E codes map to health-services contact / external causes.
    """
    code = code.astype(str).str.upper().str.strip()
    out = pd.Series("Unknown", index=code.index, dtype="object")

    v10 = version == 10
    out[v10] = code[v10].str[0].map(_ICD10_CHAPTERS).fillna("Unknown")
    d_neo = v10 & code.str.startswith("D") & (pd.to_numeric(code.str[1:3], errors="coerce") < 50)
    out[d_neo] = "Neoplasms"

    v9 = version == 9
    out[v9 & code.str.startswith("V")] = "Health-services contact"
    out[v9 & code.str.startswith("E")] = "External causes"
    num = pd.to_numeric(code.str[:3], errors="coerce")
    v9n = v9 & num.notna()
    out[v9n] = pd.cut(num[v9n], bins=[-1] + [b for b, _ in _ICD9_CHAPTERS],
                      labels=[c for _, c in _ICD9_CHAPTERS], ordered=False).astype(str)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────────

def inventory(cfg) -> dict:
    """List every MIMIC-IV / -Note file, its size, and whether the pipeline uses it."""
    root, note_root = L.mimic_paths(cfg)
    used = {"patients", "admissions", "diagnoses_icd", "labevents", "chartevents",
            "discharge", "icustays"}
    rows = []
    for base, module in [(root / "hosp", "hosp"), (root / "icu", "icu"),
                         (note_root / "note", "note")]:
        for f in sorted(base.glob("*.csv.gz")):
            name = f.name.replace(".csv.gz", "")
            rows.append({"module": module, "table": name,
                         "size_mb": round(f.stat().st_size / 1e6, 1),
                         "used_by_pipeline": name in used and name != "icustays",
                         "used_by_eda": name in used})
    scans = L.load_event_scan_counts()
    if scans is not None:
        for _, r in scans.iterrows():
            for row in rows:
                if row["table"] == r["table"]:
                    row["n_rows"] = int(r["n_rows_scanned"])
    return {"tables": rows,
            "mimic_iv_version": root.name, "mimic_iv_note_version": note_root.name}


def cohort_flow(adm: pd.DataFrame, patients: pd.DataFrame, cfg, matrix) -> dict:
    """CONSORT-style counts from all admissions to the model cohort."""
    a = adm.merge(patients[["subject_id", "anchor_age"]], on="subject_id", how="left")
    steps = [("All MIMIC-IV admissions", a)]
    a = a[a["anchor_age"] >= cfg.cohort.min_age]
    steps.append((f"Age >= {cfg.cohort.min_age}", a))
    if cfg.cohort.exclude_in_hospital_death:
        a = a[a["hospital_expire_flag"] == 0]
        steps.append(("Exclude in-hospital death", a))
    if cfg.cohort.exclude_elective:
        a = a[a["admission_type"].str.upper() != "ELECTIVE"]
        steps.append(("Exclude elective index admission", a))
    noted = matrix[matrix["has_discharge_note"] == 1]
    out = [{"step": s, "n_admissions": safe_count(len(d)),
            "n_patients": safe_count(d["subject_id"].nunique())} for s, d in steps]
    out.append({"step": "Model cohort (feature matrix)", "n_admissions": safe_count(len(matrix)),
                "n_patients": safe_count(matrix["subject_id"].nunique())})
    out.append({"step": "... with a discharge note (Stage 2/3 population)",
                "n_admissions": safe_count(len(noted)),
                "n_patients": safe_count(noted["subject_id"].nunique())})
    return {"steps": out,
            "excluded_elective_n": safe_count(
                ((adm["admission_type"].str.upper() == "ELECTIVE")).sum()),
            "matches_pipeline": int(len(steps[-1][1])) == len(matrix)}


def outcome(adm: pd.DataFrame, patients: pd.DataFrame, cfg, matrix) -> dict:
    """Target prevalence and label-construction checks."""
    window = cfg.cohort.readmission_window_days
    full = adm.sort_values(["subject_id", "admittime"]).copy()
    full["next_admittime"] = full.groupby("subject_id")["admittime"].shift(-1)
    full["next_type"] = full.groupby("subject_id")["admission_type"].shift(-1)
    full["gap_days"] = (full["next_admittime"] - full["dischtime"]).dt.total_seconds() / 86400
    coh = full[full["hadm_id"].isin(set(matrix["hadm_id"]))].copy()
    n = len(coh)

    pos_all = coh[(coh["gap_days"] >= 0) & (coh["gap_days"] <= window)]
    planned = pos_all["next_type"].str.upper().isin({"ELECTIVE", "SURGICAL SAME DAY ADMISSION"})

    # Competing risk: died after discharge within the window, never readmitted
    # -> labelled negative. patients.dod covers out-of-hospital death (state
    # death records, up to 1 year after last discharge).
    dod = coh["subject_id"].map(patients.set_index("subject_id")["dod"])
    days_to_death = (dod - coh["dischtime"]).dt.total_seconds() / 86400
    neg = matrix.set_index("hadm_id").loc[coh["hadm_id"], TARGET].to_numpy() == 0
    died_window = (days_to_death >= 0) & (days_to_death <= window)
    died_neg = died_window.to_numpy() & neg

    return {
        "n_cohort": safe_count(n),
        "all_cause_30d": {"n": safe_count(matrix[TARGET_COL].sum()),
                          "rate": safe_rate(matrix[TARGET_COL].sum(), n)},
        "unplanned_30d (model target)": {"n": safe_count(matrix[TARGET].sum()),
                                         "rate": safe_rate(matrix[TARGET].sum(), n)},
        "planned_returns_removed": {"n": safe_count(planned.sum()),
                                    "share_of_all_cause": safe_rate(planned.sum(), len(pos_all))},
        "next_admission_type_of_readmissions": safe_value_counts(pos_all["next_type"]),
        "days_to_readmission": {
            "quantiles": safe_quantiles(pos_all["gap_days"]),
            "histogram_daily": safe_histogram(pos_all["gap_days"], bins=window,
                                              clip=(0.0, 1.0)),
            "within_1_day": {"n": safe_count((pos_all["gap_days"] < 1).sum()),
                             "share": safe_rate((pos_all["gap_days"] < 1).sum(), len(pos_all))},
            "within_7_days_share": safe_rate((pos_all["gap_days"] <= 7).sum(), len(pos_all)),
        },
        "overlapping_next_admission (gap < 0)": safe_count((coh["gap_days"] < 0).sum()),
        "competing_risk_death": {
            "died_within_window_total": safe_count(died_window.sum()),
            "died_within_window_and_labelled_negative": safe_count(died_neg.sum()),
            "share_of_negatives": safe_rate(died_neg.sum(), neg.sum()),
        },
    }


def utilisation(matrix: pd.DataFrame) -> dict:
    """Admissions per patient -- the reason splits must be patient-level."""
    per_pt = matrix.groupby("subject_id").agg(n=("hadm_id", "size"), pos=(TARGET, "sum"))
    bucket = pd.cut(per_pt["n"], bins=[0, 1, 2, 3, 5, 10, np.inf],
                    labels=["1", "2", "3", "4-5", "6-10", "11+"])
    rows = []
    for b, sub in per_pt.groupby(bucket, observed=False):
        rows.append({"admissions_per_patient": str(b), "n_patients": safe_count(len(sub)),
                     "n_admissions": safe_count(sub["n"].sum()),
                     "share_of_admissions": safe_rate(sub["n"].sum(), len(matrix)),
                     "share_of_positives": safe_rate(sub["pos"].sum(), matrix[TARGET].sum()),
                     "readmission_rate": safe_rate(sub["pos"].sum(), sub["n"].sum())})
    return {"n_patients": safe_count(len(per_pt)),
            "admissions_per_patient": safe_quantiles(per_pt["n"]),
            "by_bucket": rows,
            "share_admissions_from_repeat_patients": safe_rate(
                per_pt.loc[per_pt["n"] > 1, "n"].sum(), len(matrix))}


def enrich(matrix: pd.DataFrame, adm: pd.DataFrame, patients: pd.DataFrame,
           icu: pd.DataFrame) -> pd.DataFrame:
    """Attach EDA-only descriptive columns (not model features) to the matrix."""
    a = adm.set_index("hadm_id")
    m = matrix.copy()
    for col in ["admission_type", "admission_location", "discharge_location",
                "insurance", "language", "marital_status"]:
        m[col] = m["hadm_id"].map(a[col]).fillna("MISSING")
    m["race_group"] = race_group(m["hadm_id"].map(a["race"]))
    sub = adm[adm["hadm_id"].isin(set(m["hadm_id"]))]
    m["year_group"] = m["hadm_id"].map(
        pd.Series(admission_year_group(sub, patients).to_numpy(), index=sub["hadm_id"]))
    m["had_icu_stay"] = m["hadm_id"].isin(set(icu["hadm_id"])).astype(int)
    m["age_band"] = m["age_band"].astype(str)
    return m


def table1(m: pd.DataFrame) -> dict:
    """Cohort characteristics overall and by outcome, with SMD."""
    pos, neg = m[m[TARGET] == 1], m[m[TARGET] == 0]
    numeric = ["age", "los_days", "charlson_index", "n_comorbidities",
               "n_prior_admissions", "n_prior_ed_visits", "days_since_last_discharge"]
    binary = ["gender_M", "admission_type_emergency", "admission_type_observation",
              "came_via_ed", "discharged_to_facility", "is_medicaid_medicare",
              "had_icu_stay", "has_discharge_note", "admit_is_weekend", "admit_is_night"]
    out = {"n": {"overall": safe_count(len(m)), "readmitted": safe_count(len(pos)),
                 "not_readmitted": safe_count(len(neg))},
           "numeric": {}, "binary": {}}
    for c in numeric:
        out["numeric"][c] = {"overall": safe_quantiles(m[c]),
                             "readmitted": safe_quantiles(pos[c]),
                             "not_readmitted": safe_quantiles(neg[c]),
                             "smd": smd(pos[c], neg[c])}
    for c in binary:
        out["binary"][c] = {"overall": safe_rate(m[c].sum(), len(m)),
                            "readmitted": safe_rate(pos[c].sum(), len(pos)),
                            "not_readmitted": safe_rate(neg[c].sum(), len(neg)),
                            "smd": smd(pos[c], neg[c])}
    return out


SUBGROUP_COLS = ["age_band", "gender", "race_group", "insurance", "marital_status",
                 "language", "admission_type", "admission_location",
                 "discharge_location", "year_group", "had_icu_stay"]


def subgroups(m: pd.DataFrame) -> dict:
    """Readmission base rate per descriptive subgroup."""
    return {c: rate_by_group(m, c, TARGET, top=15) for c in SUBGROUP_COLS}


def temporal(m: pd.DataFrame, dx: pd.DataFrame) -> dict:
    """Base-rate and coding drift across (approximate real) year groups."""
    coh_dx = dx[dx["hadm_id"].isin(set(m["hadm_id"]))]
    ver = coh_dx.groupby("hadm_id")["icd_version"].max()
    m = m.assign(icd10=m["hadm_id"].map(ver).eq(10).astype(int))
    rows = []
    for yg, sub in m.groupby("year_group"):
        rows.append({"year_group": yg, "n": safe_count(len(sub)),
                     "readmission_rate": safe_rate(sub[TARGET].sum(), len(sub)),
                     "share_icd10": safe_rate(sub["icd10"].sum(), len(sub)),
                     "share_with_note": safe_rate(sub["has_discharge_note"].sum(), len(sub)),
                     "mean_charlson": round(float(sub["charlson_index"].mean()), 3)
                     if len(sub) >= MIN_CELL else None})
    return {"by_year_group": rows}


def note_coverage(m: pd.DataFrame) -> dict:
    """Discharge-note availability overall, by subgroup and by outcome."""
    has = m["has_discharge_note"] == 1
    out = {"share_with_note": safe_rate(has.sum(), len(m)),
           "readmission_rate_with_note": safe_rate(m.loc[has, TARGET].sum(), has.sum()),
           "readmission_rate_without_note": safe_rate(m.loc[~has, TARGET].sum(), (~has).sum()),
           "by_subgroup": {}}
    for c in ["age_band", "race_group", "admission_type", "year_group", "discharge_location"]:
        out["by_subgroup"][c] = rate_by_group(m, c, "has_discharge_note", top=12)
    return out


def missingness(m: pd.DataFrame, feats: list[str]) -> dict:
    """Missing rate per model feature, overall and by outcome; ICU vs non-ICU."""
    pos = m[TARGET] == 1
    icu = m["had_icu_stay"] == 1
    rows = []
    for c in feats:
        miss = m[c].isna()
        if miss.sum() == 0:
            continue
        rows.append({"feature": c,
                     "missing_rate": safe_rate(miss.sum(), len(m)),
                     "missing_rate_readmitted": safe_rate(miss[pos].sum(), pos.sum()),
                     "missing_rate_not_readmitted": safe_rate(miss[~pos].sum(), (~pos).sum()),
                     "missing_rate_icu": safe_rate(miss[icu].sum(), icu.sum()),
                     "missing_rate_non_icu": safe_rate(miss[~icu].sum(), (~icu).sum()),
                     "readmission_rate_if_missing": safe_rate(
                         m.loc[miss, TARGET].sum(), miss.sum()),
                     "readmission_rate_if_present": safe_rate(
                         m.loc[~miss, TARGET].sum(), (~miss).sum())})
    rows.sort(key=lambda r: -(r["missing_rate"] or 0))
    complete = m[feats].notna().all(axis=1)
    return {"features_with_missing": rows,
            "complete_case_share": safe_rate(complete.sum(), len(m)),
            "icu_share": safe_rate(icu.sum(), len(m))}


def univariate_auroc(m: pd.DataFrame, feats: list[str]) -> list[dict]:
    """AUROC of each feature alone (non-missing rows), direction-agnostic.

    ``auroc_oriented`` = max(AUC, 1-AUC); ``direction`` says whether higher
    values mean higher risk. A cheap ranking of where the signal lives.
    """
    rows = []
    y = m[TARGET]
    for c in feats:
        ok = m[c].notna()
        if ok.sum() < 1000 or y[ok].nunique() < 2:
            continue
        auc = roc_auc_score(y[ok], m.loc[ok, c])
        rows.append({"feature": c, "auroc_oriented": round(max(auc, 1 - auc), 4),
                     "direction": "higher=riskier" if auc >= 0.5 else "lower=riskier",
                     "n_used": safe_count(ok.sum())})
    return sorted(rows, key=lambda r: -r["auroc_oriented"])


def redundancy(m: pd.DataFrame, feats: list[str], threshold: float = 0.8) -> list[dict]:
    """Feature pairs with |Spearman rho| >= threshold (on a 100k sample)."""
    sample = m[feats].sample(n=min(100_000, len(m)), random_state=42)
    corr = sample.corr(method="spearman")
    rows = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": round(float(r), 3)})
    return sorted(rows, key=lambda r: -abs(r["spearman"]))


def measurement_quality(meas: pd.DataFrame, adm: pd.DataFrame, cohort_ids: set) -> dict:
    """Audit the curated itemid map: per-itemid value ranges, post-discharge values."""
    meas = meas[meas["hadm_id"].isin(cohort_ids)]
    per_item = []
    for (item, iid), sub in meas.groupby(["item", "itemid"], observed=True):
        per_item.append({"item": str(item), "itemid": int(iid),
                         "source": str(sub["source"].iloc[0]),
                         "n_values": safe_count(len(sub)),
                         "n_admissions": safe_count(sub["hadm_id"].nunique()),
                         "valuenum": safe_quantiles(sub["valuenum"], digits=2)})
    disch = meas["hadm_id"].map(adm.set_index("hadm_id")["dischtime"])
    admit = meas["hadm_id"].map(adm.set_index("hadm_id")["admittime"])
    after = meas["charttime"] > disch
    before = meas["charttime"] < admit
    last = meas.assign(dt=(disch - meas["charttime"]).dt.total_seconds() / 3600) \
        .sort_values("charttime").groupby(["hadm_id", "item"], observed=True).tail(1)
    hours_before = {str(it): safe_quantiles(sub["dt"], digits=1)
                    for it, sub in last.groupby("item", observed=True)}
    return {"per_itemid": per_item,
            "values_charted_after_discharge": {"n": safe_count(after.sum()),
                                               "share": safe_rate(after.sum(), len(meas))},
            "values_charted_before_admission": {"n": safe_count(before.sum()),
                                                "share": safe_rate(before.sum(), len(meas))},
            "hours_between_last_value_and_discharge": hours_before}


def diagnoses(m: pd.DataFrame, dx: pd.DataFrame) -> dict:
    """ICD version mix, codes per admission, primary-diagnosis chapters."""
    coh = dx[dx["hadm_id"].isin(set(m["hadm_id"]))]
    prim = coh[coh["seq_num"] == 1].copy()
    prim["chapter"] = icd_chapter(prim["icd_code"], prim["icd_version"])
    m2 = m[["hadm_id", TARGET]].merge(prim[["hadm_id", "chapter"]], on="hadm_id", how="left")
    m2["chapter"] = m2["chapter"].fillna("No diagnosis coded")
    return {"icd_version_share_of_codes": safe_value_counts(coh["icd_version"]),
            "codes_per_admission": safe_quantiles(coh.groupby("hadm_id").size()),
            "admissions_without_any_code": safe_count(
                (~m["hadm_id"].isin(set(coh["hadm_id"]))).sum()),
            "charlson_index_distribution": safe_value_counts(m["charlson_index"].clip(upper=10)
                                                             .astype(int).astype(str)),
            "primary_diagnosis_chapter": rate_by_group(m2, "chapter", TARGET)}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run() -> dict:
    """Compute every section, write JSON + figures, return the summary dict."""
    cfg = load_config()
    patients, adm = L.load_patients(cfg), L.load_admissions(cfg)
    dx, icu = L.load_diagnoses(cfg), L.load_icustays(cfg)
    matrix = L.load_feature_matrix_fresh(cfg)
    feats = feature_columns(matrix)
    m = enrich(matrix, adm, patients, icu)

    # Sanity: the EDA's own label must equal the pipeline's.
    relabel = _apply_cohort_filters(compute_readmission_label(
        adm.drop(columns=["deathtime", "language"]), cfg), patients, cfg)
    assert len(relabel) == len(matrix), "EDA cohort != pipeline cohort"

    print("[eda/structured] computing sections ...")
    summary = {
        "target": TARGET,
        "privacy": {"min_cell": MIN_CELL, "note": "counts <11 suppressed; aggregates only"},
        "inventory": inventory(cfg),
        "cohort_flow": cohort_flow(adm, patients, cfg, matrix),
        "outcome": outcome(adm, patients, cfg, matrix),
        "utilisation": utilisation(matrix),
        "table1": table1(m),
        "subgroups": subgroups(m),
        "temporal": temporal(m, dx),
        "note_coverage": note_coverage(m),
        "missingness": missingness(m, feats),
        "univariate_auroc": univariate_auroc(m, feats),
        "redundancy": redundancy(m, feats),
        "measurement_quality": measurement_quality(L.load_measurements(cfg), adm,
                                                   set(matrix["hadm_id"])),
        "diagnoses": diagnoses(m, dx),
        "emergency_type_strings": sorted(_EMERGENCY_TYPES),
        "feature_distributions": {
            c: {"quantiles": safe_quantiles(m[c]),
                "hist_readmitted": safe_histogram(m.loc[m[TARGET] == 1, c]),
                "hist_not_readmitted": safe_histogram(m.loc[m[TARGET] == 0, c])}
            for c in feats},
    }
    write_json(summary, RESULTS_DIR / "structured_summary.json")
    plots.structured_figures(summary, RESULTS_DIR / "figures")
    print(f"[eda/structured] wrote {RESULTS_DIR / 'structured_summary.json'}")
    return summary


if __name__ == "__main__":
    run()
