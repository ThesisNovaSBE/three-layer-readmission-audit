"""Stage 3: generate plain-language readmission risk explanations via Ollama.

Uses a local generative model (default: phi4-mini) to write 3-5 sentence
plain-language summaries of why a patient was confirmed high-risk.

Prerequisites:

- Ollama installed and running: https://ollama.com
- Model pulled: ``ollama pull phi4-mini``
- ``stage3.enabled: true`` in ``config.yaml``

Usage::

    python -m src.stage3.run
    python -m src.stage3.run --input models/stage2_results.csv
"""

from __future__ import annotations

import math
import textwrap
from typing import Any

try:
    import ollama
except ImportError as _ollama_err:
    raise ImportError(
        "Ollama is required for Stage 3. "
        "Install with: pip install ollama  and  ollama pull phi4-mini"
    ) from _ollama_err

from src.config_schema import AppConfig


_SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical decision support assistant helping hospital staff understand
    readmission risk predictions made by an ML model.

    Your task is to translate model outputs and patient risk factors into a clear,
    concise explanation for clinical review.

    Rules:
    - Write exactly 3 to 5 sentences.
    - Use plain language; avoid unnecessary jargon.
    - Only refer to factors explicitly present in the patient profile below.
    - Do not diagnose, prescribe, or make clinical recommendations.
    - Do not invent or infer information not stated in the profile.
""").strip()


_USER_TEMPLATE = textwrap.dedent("""
    A patient has been flagged as high risk for unplanned 30-day hospital readmission
    by a two-stage ML pipeline (Stage 1: structured EHR data; Stage 2: clinical notes).

    Patient risk profile:
    - Age: {age} years
    - Gender: {gender}
    - Length of stay: {los_days} days
    - Admission type: {admission_type}
    - Came via emergency department: {came_via_ed}
    - Discharged to: {discharge_location}
    - Prior admissions (this hospitalisation): {n_prior_admissions}
    - Days since last discharge: {days_since_last_discharge}
    - Charlson Comorbidity Index: {charlson_index} (number of conditions: {n_comorbidities})
    - Last creatinine: {creatinine}
    - Last WBC: {wbc}
    - Stage 1 risk score: {stage1_score} (flag threshold: {stage1_threshold})
    - Stage 2 confirmation score: {stage2_score}

    Write a brief, plain-language explanation of why this patient is at elevated risk
    of readmission within 30 days.
""").strip()


def _fmt(val: Any, decimals: int = 1) -> str:
    """Format a value for inclusion in the prompt, handling NaN/None cleanly."""
    if val is None:
        return "not available"
    try:
        if math.isnan(float(val)):
            return "not available"
    except (TypeError, ValueError):
        pass
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def build_prompt(patient: dict) -> str:
    """Construct the user prompt from a patient feature dict.

    Args:
        patient: dict containing patient features and stage scores.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    return _USER_TEMPLATE.format(
        age=_fmt(patient.get("age"), 0),
        gender=patient.get("gender", "unknown"),
        los_days=_fmt(patient.get("los_days")),
        admission_type=(
            "emergency" if patient.get("admission_type_emergency") else "non-emergency"
        ),
        came_via_ed="yes" if patient.get("came_via_ed") else "no",
        discharge_location=(
            "facility (SNF/rehab)" if patient.get("discharged_to_facility") else "home"
        ),
        n_prior_admissions=int(patient.get("n_prior_admissions", 0)),
        days_since_last_discharge=_fmt(patient.get("days_since_last_discharge"), 0),
        charlson_index=_fmt(patient.get("charlson_index"), 0),
        n_comorbidities=int(patient.get("n_comorbidities", 0)),
        creatinine=_fmt(patient.get("creatinine_last")),
        wbc=_fmt(patient.get("white_blood_cells_last")),
        stage1_score=_fmt(patient.get("stage1_score"), 3),
        stage1_threshold=_fmt(patient.get("stage1_threshold"), 3),
        stage2_score=_fmt(patient.get("stage2_score"), 3),
    )


def generate_explanation(patient: dict, cfg: AppConfig) -> str:
    """Call Ollama to generate a plain-language readmission risk explanation.

    Args:
        patient: dict containing patient features and stage scores.
        cfg:     validated project config.

    Returns:
        Explanation string. Empty string if ``stage3.enabled`` is ``False``.

    Raises:
        ollama.ResponseError: if Ollama is not running or model not pulled.
    """
    if not cfg.stage3.enabled:
        return ""

    model = cfg.stage3.ollama_model
    temperature = cfg.stage3.temperature

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(patient)},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def generate_explanations_batch(
    patients: list[dict],
    cfg: AppConfig,
    verbose: bool = True,
) -> list[dict]:
    """Generate explanations for a list of confirmed high-risk patients.

    Returns the input list with an added ``'explanation'`` key per patient
    dict. Failures per patient are caught individually so one bad call does
    not abort the batch.

    Args:
        patients: list of patient feature dicts.
        cfg:      validated project config.
        verbose:  if ``True``, print a preview of each explanation.

    Returns:
        List of dicts with an added ``'explanation'`` key.
    """
    if not cfg.stage3.enabled:
        print("[stage3] Disabled in config (stage3.enabled: false). Skipping.")
        return [{**p, "explanation": ""} for p in patients]

    model = cfg.stage3.ollama_model
    print(f"[stage3] Generating explanations with '{model}' for {len(patients):,} patients...")

    results = []
    for i, patient in enumerate(patients):
        try:
            explanation = generate_explanation(patient, cfg)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Catch all exceptions so a single failed API call does not abort
            # the entire batch; the error is surfaced in the explanation field.
            explanation = f"[generation failed: {exc}]"

        results.append({**patient, "explanation": explanation})

        if verbose:
            hadm = patient.get("hadm_id", i)
            preview = explanation[:90].replace("\n", " ")
            print(f"  [{i + 1}/{len(patients)}] hadm_id={hadm}: {preview}...")

    return results
