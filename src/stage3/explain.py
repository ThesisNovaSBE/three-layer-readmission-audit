"""Stage 3 prompt construction and phi4-mini response parsing.

For a single patient, builds a prompt that reasons about the *tension* between
what Stage 1 (structured EHR data) and Stage 2 (discharge note) said, then
calls phi4-mini to produce a clinical narrative explaining that tension.

The key input the prompt uses that the old single-sentence approach lacked is
the **score delta**: Stage 2 score minus Stage 1 score.  A large negative delta
means the note strongly reassured; near zero means the modalities agreed; a
large positive delta means the note revealed additional risk.  phi4-mini is
asked to explain *why* that delta exists, citing specific note content.

Constants
---------
DISCORDANCE_MODES      : valid values for the ``discordance_mode`` field.
DISCORDANCE_CATEGORIES : valid values for the ``primary_category`` field.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

try:
    import ollama
except ImportError as _err:
    raise ImportError(
        "Ollama is required for Stage 3. "
        "Install with: pip install ollama  then  ollama pull phi4-mini"
    ) from _err

from src.config_schema import AppConfig


# ── Taxonomy ───────────────────────────────────────────────────────────────────

DISCORDANCE_MODES: tuple[str, ...] = (
    "CONCORDANT",      # note and structured data agree on risk level
    "NOTE_MITIGATES",  # note contains reassuring factors despite bad numbers
    "NOTE_AMPLIFIES",  # note reveals additional risk not captured in structured data
)

DISCORDANCE_CATEGORIES: tuple[str, ...] = (
    "social_support",       # family support, home care, social isolation
    "discharge_planning",   # follow-up arranged, care coordination quality
    "functional_status",    # mobility, ADL capacity, independence level
    "frailty_markers",      # explicit frailty language, debility, falls history
    "medication_adherence", # patient understanding, polypharmacy management
    "housing_social_risk",  # lives alone, housing instability
    "care_complexity",      # multiple specialists, complex discharge regimen
    "cognition",            # delirium, dementia, confusion documented in note
    "structured_confirmed", # note directly mirrors the lab or vital signals
)

_NOTE_TRUNCATE_CHARS = 1_500


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a clinical decision support system explaining a two-stage hospital
    readmission prediction model to clinicians.

    Your job: explain in plain clinical language why the discharge note agreed
    or disagreed with the structured EHR risk signal for one specific patient.
    Be precise — cite actual clinical content from the note.
    Do not invent information not present in the inputs.
    Return ONLY valid JSON. No text outside the JSON object.
""").strip()

_USER_TEMPLATE = textwrap.dedent("""
    A patient was assessed by a two-stage 30-day readmission prediction system.

    ── STAGE 1 (XGBoost · structured EHR) ──────────────────────────────
    Score: {stage1_score:.3f}   Flagged at threshold ≥ {stage1_threshold:.3f}
    Top structured risk factors (SHAP-ranked):
    {shap_block}

    ── STAGE 2 (Clinical-Longformer · discharge note) ───────────────────
    Score: {stage2_score:.3f}   Threshold: {stage2_threshold:.3f}
    Decision: {verdict}
    Score delta (Stage 2 − Stage 1): {delta:+.3f}  [{delta_label}]

    Key discharge note content:
    {note_block}

    ── HOW TO READ THE DELTA ────────────────────────────────────────────
      Strongly negative → the note contained reassuring content that
                          outweighs what the structured numbers implied.
      Near zero         → both modalities agree on the risk level.
      Strongly positive → the note revealed additional risk not present
                          in the structured data.

    ── YOUR TASK ────────────────────────────────────────────────────────
    Explain why Stage 2 {confirmed_or_rejected} the Stage 1 flag.
    Be specific: what clinical content in the note explains the delta?
    How does the note relate to the top structured risk factors above?

    Return ONLY a JSON object with exactly these three fields:

    "discordance_mode": one of {modes}
      CONCORDANT     — note and structured data agree on risk
      NOTE_MITIGATES — note contains reassuring content despite bad numbers
      NOTE_AMPLIFIES — note reveals additional risk not in structured data

    "primary_category": the single most relevant clinical domain, one of:
      {categories}

    "narrative": 2-3 sentences in plain clinical language. Cite specific
      content from the note. Explain the connection between the structured
      risk factors and what the note says.
""").strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _delta_label(delta: float) -> str:
    """Return a short human-readable interpretation of the score delta."""
    if delta < -0.30:
        return "note strongly mitigates structured risk"
    if delta < -0.10:
        return "note somewhat mitigates structured risk"
    if delta <= 0.10:
        return "modalities broadly agree"
    if delta <= 0.30:
        return "note somewhat amplifies risk"
    return "note strongly amplifies risk"


def _note_block(attention_sentences: list[str], note_text: str) -> str:
    """Return the best available note representation for the prompt."""
    if attention_sentences:
        lines = "\n".join(
            f"  [{i + 1}] {s}" for i, s in enumerate(attention_sentences)
        )
        return f"(top-attended sentences from Clinical-Longformer)\n{lines}"
    if note_text.strip():
        truncated = note_text.strip()[:_NOTE_TRUNCATE_CHARS]
        return f"(discharge note, truncated to {_NOTE_TRUNCATE_CHARS} chars)\n  {truncated}"
    return "  (discharge note not available)"


_PARSE_FAILURE: dict[str, Any] = {
    "discordance_mode": None,
    "primary_category": None,
    "narrative": "",
    "annotation_failed": True,
}


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse phi4-mini JSON response into a flat annotation dict.

    Returns ``annotation_failed=True`` (with ``None`` mode/category) on any
    parse error so callers can distinguish a real CONCORDANT from a silent
    failure.
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
        return {**_PARSE_FAILURE, "narrative": raw.strip()[:300]}

    mode = data.get("discordance_mode", "")
    cat = data.get("primary_category", "")
    valid_mode = mode if mode in DISCORDANCE_MODES else None
    valid_cat = cat if cat in DISCORDANCE_CATEGORIES else None

    if valid_mode is None or valid_cat is None:
        return {**_PARSE_FAILURE, "narrative": data.get("narrative", raw.strip()[:300])}

    return {
        "discordance_mode": valid_mode,
        "primary_category": valid_cat,
        "narrative": data.get("narrative", ""),
        "annotation_failed": False,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(
    stage1_score: float,
    stage1_threshold: float,
    stage2_score: float,
    stage2_threshold: float,
    stage2_confirmed: bool,
    shap_feature_strings: list[str],
    attention_sentences: list[str],
    note_text: str = "",
) -> str:
    """Build the Stage 3 user prompt for one patient.

    Args:
        stage1_score:         XGBoost probability for this patient.
        stage1_threshold:     Stage 1 flag threshold.
        stage2_score:         Calibrated Longformer probability.
        stage2_threshold:     Stage 2 decision threshold for this age group.
        stage2_confirmed:     True if Stage 2 confirmed the Stage 1 flag.
        shap_feature_strings: Top-k SHAP strings from extract_shap_for_patient().
        attention_sentences:  Top-n attended sentences (empty if not extracted).
        note_text:            Raw discharge note; used when attention is absent.

    Returns:
        Formatted prompt string ready for phi4-mini.
    """
    delta = stage2_score - stage1_score
    verdict = (
        f"CONFIRMED (score {stage2_score:.3f} ≥ threshold {stage2_threshold:.3f})"
        if stage2_confirmed
        else f"REJECTED (score {stage2_score:.3f} < threshold {stage2_threshold:.3f})"
    )
    confirmed_or_rejected = "confirmed" if stage2_confirmed else "rejected"

    shap_block = (
        "\n".join(f"  - {f}" for f in shap_feature_strings)
        if shap_feature_strings
        else "  (not available)"
    )

    return _USER_TEMPLATE.format(
        stage1_score=stage1_score,
        stage1_threshold=stage1_threshold,
        stage2_score=stage2_score,
        stage2_threshold=stage2_threshold,
        verdict=verdict,
        delta=delta,
        delta_label=_delta_label(delta),
        shap_block=shap_block,
        note_block=_note_block(attention_sentences, note_text),
        confirmed_or_rejected=confirmed_or_rejected,
        modes=str(DISCORDANCE_MODES),
        categories=str(DISCORDANCE_CATEGORIES),
    )


def call_phi4mini(prompt: str, cfg: AppConfig) -> dict[str, Any]:
    """Call phi4-mini via Ollama and return the parsed annotation dict.

    Args:
        prompt: built by :func:`build_prompt`.
        cfg:    validated project config (reads ``stage3.ollama_model`` and
                ``stage3.temperature``).

    Returns:
        Dict with keys ``discordance_mode``, ``primary_category``,
        ``narrative``, ``annotation_failed``.
    """
    try:
        response = ollama.chat(
            model=cfg.stage3.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": cfg.stage3.temperature},
            format="json",
        )
        raw = response.message.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {**_PARSE_FAILURE, "narrative": f"[ollama error: {exc}]"}

    return _parse_response(raw)
