"""Extract top-k risk-driving features from the Stage 1 XGBoost model.

Uses SHAP TreeExplainer when the ``shap`` library is installed, falling back
to gain-based feature importances if it is not.  Either way the output is a
list of human-readable strings suitable for inclusion in Stage 3 prompts.

Usage::

    from src.stage3.shap_extract import extract_shap_features
    feature_strings = extract_shap_features(artifact, patient_rows, top_k=5)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_FEATURE_LABELS: dict[str, str] = {
    "age": "age (years)",
    "los_days": "length of stay (days)",
    "came_via_ed": "admitted via ED",
    "n_prior_admissions": "prior admissions",
    "days_since_last_discharge": "days since last discharge",
    "charlson_index": "Charlson comorbidity score",
    "n_comorbidities": "number of comorbidities",
    "creatinine_last": "creatinine (last)",
    "glucose_last": "glucose (last)",
    "hemoglobin_last": "hemoglobin (last)",
    "white_blood_cells_last": "WBC (last)",
    "platelets_last": "platelets (last)",
    "sodium_last": "sodium (last)",
    "potassium_last": "potassium (last)",
    "bicarbonate_last": "bicarbonate (last)",
    "heart_rate_last": "heart rate (last)",
    "systolic_bp_last": "systolic BP (last)",
    "diastolic_bp_last": "diastolic BP (last)",
    "temperature_last": "temperature (last)",
    "respiratory_rate_last": "respiratory rate (last)",
    "spo2_last": "SpO2 (last)",
    "discharged_to_facility": "discharged to SNF / facility",
    "admission_type_emergency": "emergency admission",
}


def _label(col: str) -> str:
    return _FEATURE_LABELS.get(col, col.replace("_", " "))


def extract_shap_features(
    artifact: dict,
    patient_rows: pd.DataFrame,
    top_k: int = 5,
) -> list[list[str]]:
    """Compute per-patient top-k risk-driving features as readable strings.

    Args:
        artifact:     Stage 1 artifact dict (keys: estimator, feature_cols).
        patient_rows: DataFrame of patients aligned to the training schema.
        top_k:        number of top features per patient.

    Returns:
        List of lists — one inner list per patient, each containing up to
        top_k strings such as
        ``"creatinine (last): 3.2 (↑ risk, SHAP=+0.18)"``.
    """
    model = artifact["estimator"]
    feature_cols: list[str] = artifact["feature_cols"]

    available = [c for c in feature_cols if c in patient_rows.columns]
    x = patient_rows[available].fillna(0).values

    use_shap = False
    shap_vals: np.ndarray | None = None
    gains: np.ndarray | None = None

    try:
        import shap  # optional dependency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(x)
        use_shap = True
    except ImportError:
        raw_gains = model.feature_importances_
        gains = np.array([
            raw_gains[feature_cols.index(c)] if c in feature_cols else 0.0
            for c in available
        ])

    results: list[list[str]] = []
    for i in range(len(patient_rows)):
        if use_shap and shap_vals is not None:
            contribs = shap_vals[i]
            order = np.argsort(np.abs(contribs))[::-1][:top_k]
            features = []
            for j in order:
                col = available[j]
                val = x[i, j]
                contrib = float(contribs[j])
                direction = "↑ risk" if contrib > 0 else "↓ risk"
                features.append(
                    f"{_label(col)}: {val:.3g} ({direction}, SHAP={contrib:+.3f})"
                )
        else:
            order = np.argsort(gains)[::-1][:top_k]  # type: ignore[arg-type]
            features = []
            for j in order:
                col = available[j]
                val = x[i, j]
                features.append(f"{_label(col)}: {val:.3g}")
        results.append(features)

    return results
