"""Extract top-k risk-driving features from the Stage 1 model.

For the XGBoost Stage 1 model this uses XGBoost's own ``pred_contribs``
(exact Tree SHAP, computed by the booster itself -- no ``shap`` library,
no TreeExplainer), with missing values kept as NaN so the attribution
describes the record the model actually scored. Any integrity failure on
the XGBoost path RAISES (it becomes a PREP_FAILED row upstream) rather than
silently returning a wrong attribution. Gain-based importances remain ONLY
as the path for a non-XGBoost estimator. Either way the output is a list of
human-readable strings for the Stage 3 prompt.

Usage::

    from src.stage3.shap_extract import extract_shap_features
    feature_strings = extract_shap_features(artifact, patient_rows, top_k=5)
"""

from __future__ import annotations

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


def _format_value(val: float) -> str:
    """Render a feature value for the prompt, labeling true missingness
    instead of printing a bare 'nan'."""
    return "not recorded" if np.isnan(val) else f"{val:.3g}"


def _is_xgboost(model: object) -> bool:
    """True for an XGBoost sklearn estimator (XGBClassifier/XGBRegressor).

    Used instead of a bare ``hasattr(model, "get_booster")`` so a stand-in
    mock in tests isn't misdetected as XGBoost; behaviourally equivalent for
    the real Stage 1 artifact, which is an XGBClassifier."""
    try:
        import xgboost as xgb  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return isinstance(model, xgb.XGBModel)


def _xgb_shap_values(
    model: object, feature_frame: pd.DataFrame, feature_cols: list[str], available: list[str]
) -> np.ndarray:
    """Per-sample SHAP values from XGBoost's native ``pred_contribs`` (exact
    Tree SHAP, no ``shap`` library).

    RAISES (never falls back to gain) on any integrity failure, because for
    the real XGBoost Stage 1 model a bad attribution must become a
    PREP_FAILED row, not a silently-wrong prompt:
      - the feature columns must be the full training schema, in order;
      - the per-sample contributions must reconstruct the model's raw margin
        (sum over features + bias == predict(output_margin=True)).
    """
    import xgboost as xgb  # pylint: disable=import-outside-toplevel
    if available != feature_cols:
        raise ValueError(
            "SHAP needs the full Stage 1 feature schema in training order: got "
            f"{len(available)} columns, expected {len(feature_cols)}."
        )
    frame = feature_frame.astype(float)
    # Match the trees used by the sklearn wrapper's predict(): the Stage 1
    # model was trained with early stopping (best_iteration << n_estimators),
    # and predict() uses (0, best_iteration+1) by default, whereas a bare
    # booster pred_contribs would use EVERY boosted round. Without aligning
    # the iteration_range the contributions reconstruct a different margin
    # and the integrity check below fails for every admission.
    best_it = getattr(model, "best_iteration", None)
    iteration_range = (0, best_it + 1) if best_it is not None else (0, 0)
    contrib = model.get_booster().predict(  # type: ignore[attr-defined]
        xgb.DMatrix(frame, missing=np.nan),
        pred_contribs=True, iteration_range=iteration_range,
    )
    shap_vals = contrib[:, :-1]  # final column is the bias/base term
    margins = model.predict(frame, output_margin=True)  # type: ignore[call-arg]
    if not np.allclose(contrib.sum(axis=1), margins, atol=1e-4):
        raise ValueError(
            "XGBoost pred_contribs do not reconstruct the model margin "
            "(sum of contributions != predict(output_margin=True)); SHAP "
            "attribution is unreliable."
        )
    return shap_vals


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
        ``"creatinine (last): 3.2 (↑ risk, SHAP=+0.18 log-odds)"``.
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

    shap_vals: np.ndarray | None = None
    gains: np.ndarray | None = None
    if _is_xgboost(model):
        shap_vals = _xgb_shap_values(model, patient_rows[available], feature_cols, available)
    else:
        # Gain-based importances: only for a non-XGBoost estimator, which has
        # no exact per-sample attribution. No fallback from the XGBoost path.
        raw_gains = model.feature_importances_
        gains = np.array([
            raw_gains[feature_cols.index(c)] if c in feature_cols else 0.0
            for c in available
        ])

    results: list[list[str]] = []
    for i in range(len(patient_rows)):
        if shap_vals is not None:
            contribs = shap_vals[i]
            order = np.argsort(np.abs(contribs))[::-1][:top_k]
            features = []
            for j in order:
                col = available[j]
                val_str = _format_value(x[i, j])
                contrib = float(contribs[j])
                direction = "↑ risk" if contrib > 0 else "↓ risk"
                features.append(
                    f"{_label(col)}: {val_str} ({direction}, SHAP={contrib:+.3f} log-odds)"
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
