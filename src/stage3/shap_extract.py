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

_EXPLAINER_CACHE: dict[int, object] = {}

# requirements.txt pins only shap>=0.44 -- the cluster env may resolve a
# different version than whatever was used to verify the NaN-input fix
# locally. If TreeExplainer ever fails for any reason (version mismatch
# or otherwise), every patient silently loses real SHAP attribution in
# favor of gain-based importances (no direction, no SHAP= value) with no
# visible signal in a 9,899-patient run unless this is counted and
# printed loudly as it happens.
_FALLBACK_COUNT = 0

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


def _format_value(val: float) -> str:
    """Render a feature value for the prompt, labeling true missingness
    instead of printing a bare 'nan'."""
    return "not recorded" if np.isnan(val) else f"{val:.3g}"


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
    # Keep missing values as NaN, not 0 -- XGBoost was trained and scores
    # with missing values left as missing (routed to each split's learned
    # default branch; see src/data/features.py). Filling with 0 here used
    # to make SHAP explain a different, fabricated record than the one the
    # model actually scored (e.g. spo2_last=0 for a patient who was never
    # on a monitor, not spo2_last=missing) -- confirmed to flip the top-5
    # driver set for 68% of the 9,899-admission full run.
    x = patient_rows[available].values

    use_shap = False
    shap_vals: np.ndarray | None = None
    gains: np.ndarray | None = None

    try:
        import shap  # optional dependency  # pylint: disable=import-outside-toplevel
        cache_key = id(model)
        if cache_key not in _EXPLAINER_CACHE:
            # Constructing a TreeExplainer is not free -- confirmed 391ms/call
            # against the real Stage 1 model (2026-09-13). This function is
            # called once per patient in a batch run (~9,800 times), so an
            # uncached explainer would waste ~64 minutes of pure
            # reconstruction across a full run for an object that is
            # identical every time the same model is passed in.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(model)
        explainer = _EXPLAINER_CACHE[cache_key]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_vals = explainer.shap_values(x)
        use_shap = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Deliberately broad, not just ImportError: verified 2026-09-13 that
        # shap.TreeExplainer raises its own InvalidModelError (not
        # ImportError) for any model type it doesn't support -- narrower
        # handling here would crash every patient in a batch run instead of
        # falling back to gain importances as this function's docstring
        # promises, if the installed shap version ever can't handle the
        # real XGBoost artifact for any reason.
        global _FALLBACK_COUNT  # pylint: disable=global-statement
        _FALLBACK_COUNT += len(patient_rows)
        print(
            f"[shap_extract] WARNING: TreeExplainer failed ({exc!r}) -- "
            f"falling back to gain-based importances for {len(patient_rows)} "
            f"patient(s). Fallback total so far: {_FALLBACK_COUNT}. These "
            "statements will have no SHAP= value or direction.",
            flush=True,
        )
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
                val_str = _format_value(x[i, j])
                contrib = float(contribs[j])
                direction = "↑ risk" if contrib > 0 else "↓ risk"
                features.append(
                    f"{_label(col)}: {val_str} ({direction}, SHAP={contrib:+.3f})"
                )
        else:
            order = np.argsort(gains)[::-1][:top_k]  # type: ignore[arg-type]
            features = []
            for j in order:
                col = available[j]
                val_str = _format_value(x[i, j])
                features.append(f"{_label(col)}: {val_str}")
        results.append(features)

    return results


def extract_shap_for_patient(
    artifact: dict,
    patient_row: pd.Series,
    top_k: int = 5,
) -> list[str]:
    """Single-patient wrapper around :func:`extract_shap_features`.

    Args:
        artifact:    Stage 1 artifact dict (keys: estimator, feature_cols).
        patient_row: One patient's feature values as a pandas Series.
        top_k:       Number of top features to return.

    Returns:
        List of up to top_k human-readable feature strings.
    """
    row_df = pd.DataFrame([patient_row])
    result = extract_shap_features(artifact, row_df, top_k=top_k)
    return result[0] if result else []
