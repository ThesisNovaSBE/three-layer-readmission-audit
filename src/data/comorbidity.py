"""Charlson Comorbidity Index from ICD-9 / ICD-10 diagnosis codes.

Implements the Quan et al. (2005) "enhanced" ICD coding of the Charlson
comorbidities, with the original Charlson (1987) weights. Code matching is done
by prefix: a diagnosis code matches a category if it starts with any of the
listed prefixes (dots removed, upper-cased).

This is deliberately self-contained ("build it ourselves") and works the same on
real MIMIC-IV `diagnoses_icd` (columns: icd_code, icd_version) and on the
synthetic generator's diagnoses table.

References:
- Quan H et al. Coding algorithms for defining comorbidities in ICD-9-CM and
  ICD-10 administrative data. Med Care. 2005.
- Charlson ME et al. A new method of classifying prognostic comorbidity. 1987.
"""

from __future__ import annotations

import pandas as pd

# Original Charlson weights per category
CHARLSON_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "chronic_pulmonary_disease": 1,
    "rheumatic_disease": 1,
    "peptic_ulcer_disease": 1,
    "mild_liver_disease": 1,
    "diabetes_no_complication": 1,
    "diabetes_with_complication": 2,
    "hemiplegia_paraplegia": 2,
    "renal_disease": 2,
    "malignancy": 2,
    "moderate_severe_liver_disease": 3,
    "metastatic_solid_tumor": 6,
    "aids_hiv": 6,
}

# ICD-10 (Quan 2005) code prefixes per Charlson category
CHARLSON_ICD10 = {
    "myocardial_infarction": ["I21", "I22", "I252"],
    "congestive_heart_failure": [
        "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
        "I427", "I428", "I429", "I43", "I50", "P290",
    ],
    "peripheral_vascular_disease": [
        "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
        "K551", "K558", "K559", "Z958", "Z959",
    ],
    "cerebrovascular_disease": ["G45", "G46", "H340", "I60", "I61", "I62",
                                 "I63", "I64", "I65", "I66", "I67", "I68", "I69"],
    "dementia": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "chronic_pulmonary_disease": [
        "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46",
        "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
        "J684", "J701", "J703",
    ],
    "rheumatic_disease": ["M05", "M06", "M315", "M32", "M33", "M34", "M351",
                           "M353", "M360"],
    "peptic_ulcer_disease": ["K25", "K26", "K27", "K28"],
    "mild_liver_disease": [
        "B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714",
        "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764",
        "K768", "K769", "Z944",
    ],
    "diabetes_no_complication": ["E100", "E101", "E106", "E108", "E109",
                                  "E110", "E111", "E116", "E118", "E119",
                                  "E120", "E121", "E126", "E128", "E129",
                                  "E130", "E131", "E136", "E138", "E139",
                                  "E140", "E141", "E146", "E148", "E149"],
    "diabetes_with_complication": ["E102", "E103", "E104", "E105", "E107",
                                    "E112", "E113", "E114", "E115", "E117",
                                    "E122", "E123", "E124", "E125", "E127",
                                    "E132", "E133", "E134", "E135", "E137",
                                    "E142", "E143", "E144", "E145", "E147"],
    "hemiplegia_paraplegia": ["G041", "G114", "G801", "G802", "G81", "G82",
                               "G830", "G831", "G832", "G833", "G834", "G839"],
    "renal_disease": ["I120", "I131", "N032", "N033", "N034", "N035", "N036",
                       "N037", "N052", "N053", "N054", "N055", "N056", "N057",
                       "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940",
                       "Z992"],
    "malignancy": ["C0", "C1", "C2", "C3", "C40", "C41", "C43", "C45", "C46",
                    "C47", "C48", "C49", "C5", "C6", "C70", "C71", "C72",
                    "C73", "C74", "C75", "C76", "C81", "C82", "C83", "C84",
                    "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95",
                    "C96", "C97"],
    "moderate_severe_liver_disease": ["I850", "I859", "I864", "I982", "K704",
                                       "K711", "K721", "K729", "K765", "K766",
                                       "K767"],
    "metastatic_solid_tumor": ["C77", "C78", "C79", "C80"],
    "aids_hiv": ["B20", "B21", "B22", "B24"],
}

# ICD-9-CM (Quan 2005) code prefixes per Charlson category
CHARLSON_ICD9 = {
    "myocardial_infarction": ["410", "412"],
    "congestive_heart_failure": ["39891", "40201", "40211", "40291", "40401",
                                  "40403", "40411", "40413", "40491", "40493",
                                  "4254", "4255", "4257", "4258", "4259", "428"],
    "peripheral_vascular_disease": ["0930", "4373", "440", "441", "4431",
                                     "4432", "4438", "4439", "4471", "5571",
                                     "5579", "V434"],
    "cerebrovascular_disease": ["36234", "430", "431", "432", "433", "434",
                                 "435", "436", "437", "438"],
    "dementia": ["290", "2941", "3312"],
    "chronic_pulmonary_disease": ["4168", "4169", "490", "491", "492", "493",
                                   "494", "495", "496", "500", "501", "502",
                                   "503", "504", "505", "5064", "5081", "5088"],
    "rheumatic_disease": ["4465", "7100", "7101", "7102", "7103", "7104",
                           "7140", "7141", "7142", "7148", "725"],
    "peptic_ulcer_disease": ["531", "532", "533", "534"],
    "mild_liver_disease": ["07022", "07023", "07032", "07033", "07044",
                            "07054", "0706", "0709", "570", "571", "5733",
                            "5734", "5738", "5739", "V427"],
    "diabetes_no_complication": ["2500", "2501", "2502", "2503", "2508",
                                  "2509"],
    "diabetes_with_complication": ["2504", "2505", "2506", "2507"],
    "hemiplegia_paraplegia": ["3341", "342", "343", "3440", "3441", "3442",
                               "3443", "3444", "3445", "3446", "3449"],
    "renal_disease": ["40301", "40311", "40391", "40402", "40403", "40412",
                       "40413", "40492", "40493", "582", "5830", "5831",
                       "5832", "5834", "5836", "5837", "585", "586", "5880",
                       "V420", "V451", "V56"],
    "malignancy": ["14", "15", "16", "170", "171", "172", "174", "175", "176",
                    "179", "18", "190", "191", "192", "193", "194", "1950",
                    "1951", "1952", "1953", "1954", "1955", "1956", "1957",
                    "1958", "200", "201", "202", "203", "204", "205", "206",
                    "207", "208", "2386"],
    "moderate_severe_liver_disease": ["4560", "4561", "4562", "5722", "5723",
                                       "5724", "5728"],
    "metastatic_solid_tumor": ["196", "197", "198", "199"],
    "aids_hiv": ["042", "043", "044"],
}


def _normalize(code: str) -> str:
    return str(code).replace(".", "").replace(" ", "").upper()


def _matches(code: str, prefixes: list[str]) -> bool:
    return any(code.startswith(p.upper()) for p in prefixes)


def charlson_categories(codes_with_versions: list[tuple[str, int]]) -> dict[str, int]:
    """Return a 0/1 flag per Charlson category for one admission's diagnosis codes.

    codes_with_versions: list of (icd_code, icd_version) where icd_version is 9 or 10.
    """
    flags = {cat: 0 for cat in CHARLSON_WEIGHTS}
    for raw_code, version in codes_with_versions:
        code = _normalize(raw_code)
        table = CHARLSON_ICD10 if int(version) == 10 else CHARLSON_ICD9
        for cat, prefixes in table.items():
            if flags[cat] == 0 and _matches(code, prefixes):
                flags[cat] = 1

    # Hierarchy rules: the more severe variant cancels the milder one.
    if flags["diabetes_with_complication"]:
        flags["diabetes_no_complication"] = 0
    if flags["moderate_severe_liver_disease"]:
        flags["mild_liver_disease"] = 0
    if flags["metastatic_solid_tumor"]:
        flags["malignancy"] = 0
    return flags


def charlson_index(flags: dict[str, int]) -> int:
    """Return the total Charlson score by summing weighted comorbidity flags."""
    return sum(CHARLSON_WEIGHTS[cat] * present for cat, present in flags.items())


def charlson_per_admission(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Compute the Charlson index per admission from a diagnoses table.

    Expects columns: hadm_id, icd_code, icd_version.
    Returns a DataFrame with columns: hadm_id, charlson_index, n_comorbidities.
    """
    rows = []
    for hadm_id, grp in diagnoses.groupby("hadm_id"):
        pairs = list(zip(grp["icd_code"], grp["icd_version"]))
        flags = charlson_categories(pairs)
        rows.append({
            "hadm_id": hadm_id,
            "charlson_index": charlson_index(flags),
            "n_comorbidities": sum(flags.values()),
        })
    return pd.DataFrame(rows)
