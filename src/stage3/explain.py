"""Stage 3: cross-modal discordance annotation and clinician explanation.

Each confirmed or rejected Stage 1 flag is annotated with:

1. ``discordance_mode`` — whether the discharge note and structured EHR data
   tell the same story (CONCORDANT), the note is reassuring despite bad numbers
   (NOTE_MITIGATES), or the note reveals additional risk not in the structured
   data (NOTE_AMPLIFIES).

2. ``primary_category`` — the dominant clinical domain driving any discordance.

3. ``explanation`` — a single sentence for the clinical record that synthesises
   both Stage 1 structured signals and Stage 2 discharge note evidence.

phi4-mini is called once per patient and asked to return a JSON object.
``format="json"`` is passed to Ollama to enforce structured output.

Constants
---------
DISCORDANCE_MODES      : valid values for the discordance_mode field.
DISCORDANCE_CATEGORIES : valid values for the primary_category field.
"""

from __future__ import annotations

import json
import math
import textwrap
from typing import Any

try:
    import ollama
except ImportError as _err:
    raise ImportError(
        "Ollama is required for Stage 3.  "
        "Install with: pip install ollama  then  ollama pull phi4-mini"
    ) from _err

from src.config_schema import AppConfig


# ── Taxonomy constants ─────────────────────────────────────────────────────────

DISCORDANCE_MODES: tuple[str, ...] = (
    "CONCORDANT",      # note and structured data tell the same story
    "NOTE_MITIGATES",  # note has reassuring factors despite bad structured numbers
    "NOTE_AMPLIFIES",  # note reveals additional risk not captured in structured data
)

DISCORDANCE_CATEGORIES: tuple[str, ...] = (
    "social_support",       # family support, home care, social isolation
    "discharge_planning",   # follow-up scheduled, care coordination quality
    "functional_status",    # mobility, ADL capacity, independence level
    "frailty_markers",      # explicit frailty language, debility, falls history
    "medication_adherence", # patient understanding, pill management, polypharmacy
    "housing_social_risk",  # lives alone, housing instability, homelessness
    "care_complexity",      # multiple specialists, complex discharge regimen
    "cognition",            # delirium, dementia, confusion documented in note
    "structured_confirmed", # note mirrors / reinforces the lab or vital signals
)


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical NLP system.  Your task is to classify the relationship
    between structured EHR risk signals (Stage 1) and discharge note evidence
    (Stage 2) for hospital readmission prediction.

    Return ONLY a JSON object — no text outside the JSON.
    Temperature is low; be consistent and precise.
""").strip()

_USER_TEMPLATE = textwrap.dedent("""
    A patient was processed by a two-stage readmission prediction pipeline:
      Stage 1 (XGBoost on structured EHR data) flagged this patient as HIGH RISK.
      Stage 2 (Clinical-Longformer on discharge note) {stage2_verdict}.

    STAGE 1 — top risk drivers from structured data:
    {shap_features}

    STAGE 2 — key sentences from discharge note (highest model attention):
    {note_sentences}

    TASK: Classify the relationship between the structured risk signal and the
    discharge note evidence.

    Return a JSON object with exactly these three fields:

    "discordance_mode": one of {modes}
      CONCORDANT     — note and structured data agree on risk level
      NOTE_MITIGATES — note contains reassuring content despite bad structured numbers
      NOTE_AMPLIFIES — note reveals additional risk not captured in structured data

    "primary_category": the single most important clinical domain. Choose one of:
      {categories}

    "explanation": one sentence for the clinical record explaining, in plain
      language, why Stage 2 {confirmed_or_rejected} the Stage 1 flag and what
      information in the discharge note drove that decision.

    Return ONLY valid JSON.  No commentary outside the JSON object.
""").strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(val: Any, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    try:
        if math.isnan(float(val)):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


_PARSE_FAILURE: dict = {
    "discordance_mode": None,
    "primary_category": None,
    "explanation": "",
    "annotation_failed": True,
}


def _parse_response(raw: str) -> dict:
    """Parse phi4-mini JSON response.

    Returns a dict with ``annotation_failed=True`` (and ``None`` mode/category)
    on any parse error so failures are excluded from population aggregation
    rather than silently counted as CONCORDANT.
    """
    data: dict | None = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                pass

    if data is None:
        return {**_PARSE_FAILURE, "explanation": raw.strip()[:300]}

    mode = data.get("discordance_mode", "")
    cat = data.get("primary_category", "")
    valid_mode = mode if mode in DISCORDANCE_MODES else None
    valid_cat = cat if cat in DISCORDANCE_CATEGORIES else None

    if valid_mode is None or valid_cat is None:
        return {**_PARSE_FAILURE, "explanation": data.get("explanation", raw.strip()[:300])}

    return {
        "discordance_mode": valid_mode,
        "primary_category": valid_cat,
        "explanation": data.get("explanation", ""),
        "annotation_failed": False,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

_NOTE_TRUNCATE_CHARS = 1_500  # chars of raw note to pass when attention is absent


def build_prompt(
    stage2_confirmed: bool,
    stage2_score: float,
    shap_feature_strings: list[str],
    attention_sentences: list[str],
    note_text: str = "",
) -> str:
    """Build the Stage 3 user prompt for one patient.

    Args:
        stage2_confirmed:     True if Stage 2 confirmed the Stage 1 flag.
        stage2_score:         Stage 2 probability after calibration.
        shap_feature_strings: top-k feature strings from extract_shap_features().
        attention_sentences:  top-n sentences from extract_attention_spans().
        note_text:            raw discharge note text; used as fallback when
                              attention_sentences is empty (e.g. --no-attention).

    Returns:
        Formatted prompt string.
    """
    verdict = (
        f"CONFIRMED the flag (score={stage2_score:.3f})"
        if stage2_confirmed
        else f"REJECTED the flag (score={stage2_score:.3f})"
    )
    confirmed_or_rejected = "confirmed" if stage2_confirmed else "rejected"

    if shap_feature_strings:
        shap_block = "\n".join(f"  - {f}" for f in shap_feature_strings)
    else:
        shap_block = "  (not available)"

    if attention_sentences:
        note_block = "\n".join(
            f"  [{j + 1}] {s}" for j, s in enumerate(attention_sentences)
        )
    elif note_text.strip():
        truncated = note_text.strip()[:_NOTE_TRUNCATE_CHARS]
        note_block = f"  [full note, truncated to {_NOTE_TRUNCATE_CHARS} chars]\n  {truncated}"
    else:
        note_block = "  (discharge note not available)"

    return _USER_TEMPLATE.format(
        stage2_verdict=verdict,
        shap_features=shap_block,
        note_sentences=note_block,
        modes=str(DISCORDANCE_MODES),
        categories=str(DISCORDANCE_CATEGORIES),
        confirmed_or_rejected=confirmed_or_rejected,
    )


def annotate_patient(
    patient: dict,
    shap_feature_strings: list[str],
    attention_sentences: list[str],
    cfg: AppConfig,
) -> dict:
    """Generate discordance annotation + clinician explanation for one patient.

    Args:
        patient:              dict containing at least ``stage2_confirmed`` and
                              ``stage2_score`` keys.
        shap_feature_strings: from extract_shap_features(), may be empty.
        attention_sentences:  from extract_attention_spans(), may be empty.
        cfg:                  validated project config.

    Returns:
        Dict with keys ``discordance_mode``, ``primary_category``,
        ``explanation``.  On Ollama error, safe defaults are returned.
    """
    confirmed = bool(patient.get("stage2_confirmed", 0))
    score = float(patient.get("stage2_score", 0.0))
    note_text = str(patient.get("note_text", ""))
    prompt = build_prompt(confirmed, score, shap_feature_strings, attention_sentences, note_text)

    try:
        response = ollama.chat(
            model=cfg.stage3.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},  # low temp for consistent classification
            format="json",
        )
        raw = response.message.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {**_PARSE_FAILURE, "explanation": f"[ollama error: {exc}]"}

    return _parse_response(raw)


def annotate_batch(
    patients: list[dict],
    shap_features_list: list[list[str]],
    attention_spans: dict[int, list[str]],
    cfg: AppConfig,
    verbose: bool = True,
) -> list[dict]:
    """Annotate a batch of patients with discordance labels and explanations.

    Args:
        patients:           list of patient dicts (must have ``hadm_id``,
                            ``stage2_confirmed``, ``stage2_score``).
        shap_features_list: one list of strings per patient (same order).
        attention_spans:    dict mapping hadm_id -> list of top sentences.
        cfg:                validated project config.
        verbose:            print progress lines.

    Returns:
        Input patients list with added keys: ``discordance_mode``,
        ``primary_category``, ``explanation``.
    """
    model = cfg.stage3.ollama_model
    n = len(patients)
    print(f"[stage3/explain] Annotating {n:,} patients with '{model}' ...")

    results = []
    for i, (patient, shap_strs) in enumerate(zip(patients, shap_features_list)):
        hadm_id = int(patient.get("hadm_id", 0))
        attn = attention_spans.get(hadm_id, [])
        annotation = annotate_patient(patient, shap_strs, attn, cfg)
        results.append({**patient, **annotation})

        if verbose and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")

    return results
