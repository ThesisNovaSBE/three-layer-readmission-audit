"""Stage 3: independent LLM auditing of Stage 1-flagged admissions.

Rewritten 2026-08-25 (session 15). Prior versions asked phi4-mini either to
narrate Stage 1's own SHAP values (session 11's "old Stage 3" — rejected as a
well-trodden explainer pattern with no novel finding) or to freely classify a
9-category discordance taxonomy while also acting as an explainer of Stage 2's
decision. Neither implements what the literature review's own gap analysis
calls for: an LLM that independently audits *another model's* output.

The design here — confirmed across three planning artifacts (the 2026-08-16
methodological evaluation, the 2026-08-17 Project Brief, and the Working
Paper) — has phi4-mini act as a deliberating auditor. It receives Stage 1's
score and SHAP attributions, Stage 2's independently-derived note-based
score, and the discharge note itself, and returns its own uphold/override
judgment — not a narration of a decision already made by Stage 2.

Two things are deliberately NOT delegated to the LLM:

1. **Discordance mode.** Computed quantitatively from percentile-rank
   displacement of stage1_score vs. stage2_score within the flagged+noted
   cohort (see :func:`compute_discordance`), not asked of the model. Percentile
   rank is used instead of a raw-probability difference (the original design)
   because Stage 1 and Stage 2 are different model families and are not
   guaranteed to be equally well-calibrated even after isotonic calibration —
   rank displacement is invariant to that risk. Stage 1 uses ~40 structured
   features; Stage 2 (2026-08-26: reverted to a plain, note-only
   Clinical-Longformer — the multimodal "FusionLongformer" variant was
   explored and dropped as unnecessary complexity for a model that was never
   actually trained) uses none. The two models are informationally
   independent by construction, and no claim stronger than "they use
   different information, and when they disagree, that is worth
   investigating" is made.
2. **Whether the auditor's own decision is reproducible.** Ollama temperature
   is pinned at 0 (``cfg.stage3.temperature``) for every evaluation run.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import numpy as np

try:
    import ollama
except ImportError as _err:
    raise ImportError(
        "Ollama is required for Stage 3. "
        "Install with: pip install ollama  then  ollama pull phi4-mini"
    ) from _err

from src.config_schema import AppConfig


# ── Taxonomy ───────────────────────────────────────────────────────────────────

DECISIONS: tuple[str, ...] = ("uphold", "override")

DISCORDANCE_MODES: tuple[str, ...] = (
    "CONCORDANT",      # Stage 1 and Stage 2 rank this patient similarly
    "NOTE_MITIGATES",  # Stage 2 ranks the patient markedly lower risk than Stage 1
    "NOTE_AMPLIFIES",  # Stage 2 ranks the patient markedly higher risk than Stage 1
)

CLINICAL_DOMAINS: tuple[str, ...] = (
    "social_support",     # family support, home care, social isolation, living alone
    "frailty",             # explicit frailty language, debility, falls history
    "palliative_intent",   # hospice, comfort-focused care, goals-of-care discussion
    "care_coordination",   # follow-up arranged, discharge planning quality, polypharmacy
    "clinical_trajectory", # trend in labs/vitals/course not captured by static features
    "other",
)

# Colleague review, 2026-08-27, item 3: does the note itself mention a
# planned return (chemo cycle, staged surgery, scheduled dialysis)? This is
# independent of, not a replacement for, the structured admission_type-based
# readmission_30d_unplanned proxy in src/data/features.py -- the two use
# different evidence and are reported separately; their agreement rate is
# itself a characterisation finding (how much the structured proxy
# undercounts planned returns), not something to silently merge.
PLANNED_RETURN_ANSWERS: tuple[str, ...] = ("yes", "no", "not_stated")

# Generous safety cap, not a routine truncation. Session 14 measured median
# MIMIC discharge notes at ~2,649 tokens; at Stage 2's 4096-token window
# (~4-5 chars/token in clinical text) that is comfortably under 20,000 chars.
# The old cap (1,500 chars, ~250 words) truncated to ~6% of what Stage 2 sees
# and was flagged in the Working Paper as an unresolved confound for
# discordance interpretation — this raises it so both stages read materially
# the same document for all but pathologically long notes.
_NOTE_MAX_CHARS = 20_000


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an independent clinical auditor reviewing a 30-day hospital
    readmission risk alert raised by a structured EHR model.

    You will be given: the structured model's risk score and its stated
    reasons (SHAP attributions), a second, independently-derived risk estimate
    from a model that reads the discharge note, and the discharge note itself.

    Your job is to reach your OWN judgment about whether this alert should be
    upheld or overridden — you are not explaining or narrating a decision
    someone else already made. Read the note as a clinician would. Be
    specific and cite actual clinical content. Do not invent information not
    present in the inputs.

    Return ONLY valid JSON. No text outside the JSON object.
""").strip()

_USER_TEMPLATE = textwrap.dedent("""
    A structured model flagged this patient as high-risk for 30-day
    readmission.

    ── STRUCTURED RISK ESTIMATE (XGBoost) ────────────────────────────────
    Score: {stage1_score:.3f}   Flagged at threshold >= {stage1_threshold:.3f}
    Top structured risk factors (SHAP-ranked):
    {shap_block}

    ── NOTE-BASED RISK ESTIMATE (Clinical-Longformer, discharge note only) ──
    Score: {stage2_score:.3f}
    This model reads only the discharge note — no structured features. It
    was trained independently and does not know the score above.

    ── QUANTITATIVE DISCORDANCE ────────────────────────────────────────────
    Within the cohort of flagged, note-covered patients, this patient ranks
    at the {r1:.0f}th percentile on the structured estimate and the
    {r2:.0f}th percentile on the note-based estimate (displacement:
    {displacement:+.0f} percentile points -> {mode}).
      NOTE_MITIGATES -> the note-based estimate ranks this patient markedly
                         lower risk than the structured estimate alone.
      NOTE_AMPLIFIES -> the note-based estimate ranks this patient markedly
                         higher risk than the structured estimate alone.
      CONCORDANT     -> the two estimates rank this patient similarly.
    This is context, not a conclusion — reach your own judgment from the
    evidence below.

    ── DISCHARGE NOTE ───────────────────────────────────────────────────────
    {note_block}
    {attention_block}

    ── YOUR TASK ────────────────────────────────────────────────────────────
    Decide, independently, whether this readmission-risk alert should be
    upheld or overridden. Base your decision on the discharge note and both
    risk estimates together — not on the discordance label alone.

    Return ONLY a JSON object with exactly these five fields:

    "decision": one of {decisions}
      uphold   — the alert should stand; the note does not provide grounds
                 to suppress it
      override — the note provides specific, documented grounds to suppress
                 this alert (e.g. a planned return, a robust support system,
                 or explicitly reassuring clinical trajectory)

    "primary_clinical_domain": the single most relevant domain, one of:
      {domains}

    "supporting_quote": the exact sentence from the discharge note above that
      most directly supports your decision. Copy it verbatim — do not
      paraphrase or summarise. This is checked automatically against the
      note text, so an invented or altered quote will be caught.

    "planned_return": does the note mention a planned return — a scheduled
      chemotherapy cycle, a staged surgery, scheduled dialysis, or similar —
      regardless of whether it affected your decision? One of
      {planned_return_options}.

    "clinical_justification": 2-4 sentences. Cite specific content from the
      note. State what drove your decision.
""").strip()


# ── Discordance (quantitative, not LLM-determined) ──────────────────────────────

def _percentile_rank(value: float, population: np.ndarray) -> float:
    """Return the percentile rank of ``value`` within ``population`` (0-100)."""
    population = np.asarray(population, dtype=float)
    if population.size == 0:
        return 50.0
    return float((population <= value).mean() * 100.0)


def compute_discordance(
    stage1_score: float,
    stage2_score: float,
    cohort_stage1_scores: np.ndarray,
    cohort_stage2_scores: np.ndarray,
    displacement_pp: float = 20.0,
) -> dict[str, float | str]:
    """Return percentile-rank displacement and discordance mode for one patient.

    Displacement is invariant to any residual, unequal miscalibration between
    Stage 1 (XGBoost) and Stage 2 (Clinical-Longformer) — two different model
    families are not guaranteed to share the same calibration error even after
    isotonic calibration, so a raw ``stage2_score - stage1_score`` difference
    is not a reliable measure of disagreement. Rank displacement only requires
    that each score is a meaningful risk ordering within its own cohort.

    Args:
        stage1_score:          this patient's Stage 1 probability.
        stage2_score:          this patient's Stage 2 probability.
        cohort_stage1_scores:  Stage 1 scores for the flagged+noted cohort.
        cohort_stage2_scores:  Stage 2 scores for the same cohort.
        displacement_pp:       |displacement| >= this (percentile points)
                                is classified as discordant.

    Returns:
        Dict with ``r1``, ``r2``, ``displacement``, ``mode``.
    """
    r1 = _percentile_rank(stage1_score, cohort_stage1_scores)
    r2 = _percentile_rank(stage2_score, cohort_stage2_scores)
    displacement = r2 - r1
    if displacement <= -displacement_pp:
        mode = "NOTE_MITIGATES"
    elif displacement >= displacement_pp:
        mode = "NOTE_AMPLIFIES"
    else:
        mode = "CONCORDANT"
    return {"r1": r1, "r2": r2, "displacement": displacement, "mode": mode}


def sweep_discordance_thresholds(
    displacements: np.ndarray, thresholds_pp: list[float] | None = None
) -> dict[str, dict[str, float]]:
    """Report the discordance mode distribution across a range of thresholds.

    ``stage3.discordance_displacement_pp`` (20, provisional) has never been
    validated empirically — this answers how sensitive the reported mode
    distribution is to that choice, per docs/ARCHITECTURE.md. Only the mode
    classification depends on the threshold; ``displacement`` values
    (already computed per-patient by :func:`compute_discordance`) don't need
    recomputing — pass the ``displacement`` column of a batch audit result.

    Args:
        displacements: array of ``r2 - r1`` values, one per audited patient.
        thresholds_pp: displacement-point thresholds to sweep. Defaults to
                       ``[10, 15, 20, 25, 30]``.

    Returns:
        Dict keyed by threshold (as a string) to a dict of
        mode -> fraction of patients classified into that mode.
    """
    thresholds_pp = thresholds_pp or [10.0, 15.0, 20.0, 25.0, 30.0]
    displacements = np.asarray(displacements, dtype=float)
    n = len(displacements)
    out: dict[str, dict[str, float]] = {}
    for thr in thresholds_pp:
        mitigates = int((displacements <= -thr).sum())
        amplifies = int((displacements >= thr).sum())
        concordant = n - mitigates - amplifies
        out[str(thr)] = {
            "NOTE_MITIGATES": mitigates / n if n else 0.0,
            "NOTE_AMPLIFIES": amplifies / n if n else 0.0,
            "CONCORDANT": concordant / n if n else 0.0,
        }
    return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _note_block(note_text: str) -> str:
    """Return the discharge note, capped at a generous safety limit."""
    text = note_text.strip()
    if not text:
        return "  (discharge note not available)"
    if len(text) > _NOTE_MAX_CHARS:
        text = text[:_NOTE_MAX_CHARS]
        return f"(discharge note, capped at {_NOTE_MAX_CHARS:,} chars)\n  {text}"
    return f"(full discharge note, {len(text):,} chars)\n  {text}"


def _attention_block(attention_sentences: list[str]) -> str:
    """Return an optional auxiliary hint of what Stage 2 attended to most.

    Informational only — per Jain & Wallace (2019), attention weights are not
    a faithful explanation of Stage 2's decision. Never presented as "the
    reason", only as "regions of the note Stage 2's attention concentrated on".
    """
    if not attention_sentences:
        return ""
    lines = "\n".join(f"  [{i + 1}] {s}" for i, s in enumerate(attention_sentences))
    return (
        "\n(For reference only — regions of the note the note-based model's "
        "attention concentrated on, not a faithful explanation of its score:)\n"
        f"{lines}"
    )


_PARSE_FAILURE: dict[str, Any] = {
    "decision": None,
    "primary_clinical_domain": None,
    "supporting_quote": "",
    "quote_verified": None,
    "planned_return": None,
    "clinical_justification": "",
    "annotation_failed": True,
}


def verify_quote(supporting_quote: str, note_text: str) -> bool:
    """Return whether ``supporting_quote`` appears verbatim in ``note_text``.

    Computed in code, not asked of the LLM — this is what turns "the model
    says it quoted the note" into something automatically checkable, and is
    the mechanism that makes human spot-checking (colleague review,
    2026-08-27, item 2) tractable instead of impossible: a reviewer only
    needs to check the *decisions* on the (hopefully small) subset where
    ``quote_verified`` is False, not re-read every note from scratch.
    """
    if not supporting_quote.strip():
        return False
    return supporting_quote.strip() in note_text


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse phi4-mini JSON response into a flat annotation dict.

    Returns ``annotation_failed=True`` (with ``None`` decision/domain) on any
    parse error so callers can distinguish a real "uphold" from a silent
    failure. ``quote_verified`` is NOT set here — it needs ``note_text``,
    which this function doesn't have; see :func:`call_llm`, which calls
    :func:`verify_quote` after parsing.
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
        return {**_PARSE_FAILURE, "clinical_justification": raw.strip()[:300]}

    decision = data.get("decision", "")
    domain = data.get("primary_clinical_domain", "")
    quote = data.get("supporting_quote", "")
    planned_return = data.get("planned_return", "")

    valid_decision = decision if decision in DECISIONS else None
    valid_domain = domain if domain in CLINICAL_DOMAINS else None
    valid_quote = quote if isinstance(quote, str) and quote.strip() else None
    valid_planned_return = (
        planned_return if planned_return in PLANNED_RETURN_ANSWERS else None
    )

    if (
        valid_decision is None or valid_domain is None
        or valid_quote is None or valid_planned_return is None
    ):
        return {
            **_PARSE_FAILURE,
            "clinical_justification": data.get(
                "clinical_justification", raw.strip()[:300]
            ),
        }

    return {
        "decision": valid_decision,
        "primary_clinical_domain": valid_domain,
        "supporting_quote": valid_quote,
        "quote_verified": None,  # filled in by call_llm, which has note_text
        "planned_return": valid_planned_return,
        "clinical_justification": data.get("clinical_justification", ""),
        "annotation_failed": False,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def build_prompt(
    stage1_score: float,
    stage1_threshold: float,
    stage2_score: float,
    discordance: dict[str, float | str],
    shap_feature_strings: list[str],
    note_text: str,
    attention_sentences: list[str] | None = None,
) -> str:
    """Build the Stage 3 user prompt for one patient.

    Args:
        stage1_score:          XGBoost probability for this patient.
        stage1_threshold:      Stage 1 flag threshold.
        stage2_score:          Calibrated Clinical-Longformer probability.
        discordance:           output of :func:`compute_discordance`.
        shap_feature_strings:  top-k SHAP strings from extract_shap_for_patient().
        note_text:             raw discharge note text.
        attention_sentences:   optional Stage 2 attention spans (auxiliary only).

    Returns:
        Formatted prompt string ready for phi4-mini.
    """
    shap_block = (
        "\n".join(f"  - {f}" for f in shap_feature_strings)
        if shap_feature_strings
        else "  (not available)"
    )

    return _USER_TEMPLATE.format(
        stage1_score=stage1_score,
        stage1_threshold=stage1_threshold,
        stage2_score=stage2_score,
        r1=discordance["r1"],
        r2=discordance["r2"],
        displacement=discordance["displacement"],
        mode=discordance["mode"],
        shap_block=shap_block,
        note_block=_note_block(note_text),
        attention_block=_attention_block(attention_sentences or []),
        decisions=str(DECISIONS),
        domains=str(CLINICAL_DOMAINS),
        planned_return_options=str(PLANNED_RETURN_ANSWERS),
    )


def call_llm(
    prompt: str,
    cfg: AppConfig,
    model_name: str | None = None,
    note_text: str = "",
) -> dict[str, Any]:
    """Call an Ollama-hosted model and return the parsed annotation dict.

    Generalised from the original ``call_phi4mini`` so the same prompt can be
    run through a different model — e.g. ``cfg.stage3.robustness_model`` — as
    a robustness check on whether the auditor's value depends on model
    scale, without duplicating the prompt/parsing logic. All models here are
    assumed Ollama-hosted (local); routing to a cloud API is a separate,
    currently unmade decision — see docs/ARCHITECTURE.md (PhysioNet's data
    use terms need checking before patient text is sent to any third party).

    Args:
        prompt:     built by :func:`build_prompt`.
        cfg:        validated project config (reads ``stage3.temperature``
                    — pinned at 0 for reproducibility).
        model_name: Ollama model tag to use. Defaults to
                    ``cfg.stage3.ollama_model`` (the primary auditor model).
        note_text:  the same raw note text passed to :func:`build_prompt` —
                    used only to verify ``supporting_quote`` against it
                    (see :func:`verify_quote`), not re-sent to the model.

    Returns:
        Dict with keys ``decision``, ``primary_clinical_domain``,
        ``supporting_quote``, ``quote_verified``, ``planned_return``,
        ``clinical_justification``, ``annotation_failed``.
    """
    model_name = model_name or cfg.stage3.ollama_model
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": cfg.stage3.temperature},
            format="json",
        )
        raw = response.message.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {**_PARSE_FAILURE, "clinical_justification": f"[ollama error: {exc}]"}

    annotation = _parse_response(raw)
    if not annotation["annotation_failed"]:
        annotation["quote_verified"] = verify_quote(annotation["supporting_quote"], note_text)
    return annotation
