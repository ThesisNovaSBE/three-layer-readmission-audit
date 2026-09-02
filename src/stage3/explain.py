"""Stage 3: independent LLM auditing of Stage 1-flagged admissions.

Rewritten 2026-08-25 (session 15), extended 2026-08-28 (session 19). Prior
versions asked phi4-mini either to narrate Stage 1's own SHAP values
(session 11's "old Stage 3" — rejected as a well-trodden explainer pattern
with no novel finding) or to freely classify a 9-category discordance
taxonomy while also acting as an explainer of Stage 2's decision. Neither
implements what the literature review's own gap analysis calls for: an LLM
that independently audits *another model's* output.

The design here has phi4-mini act as a deliberating auditor. It receives
Stage 1's score and SHAP attributions, Stage 2's independently-derived
note-based score, and the discharge note itself, and returns its own
uphold/override/insufficient_evidence judgment — not a narration of a
decision already made by Stage 2.

Session 19 replaced the single free-text ``primary_clinical_domain`` with a
fixed, two-sided grounds taxonomy (mitigating vs. aggravating), each ground
requiring its own verbatim quote, and added a second, code-computed
``decision_rule`` alongside the model's own ``decision_model`` — not a
replacement for the model's free judgment (see docs/ARCHITECTURE.md §5 item
3a: small additions to the schema, not a switch to pure extraction), but a
deterministic, human-checkable cross-check computed from the same
extraction. Their agreement rate is a reportable consistency metric; their
disagreement is itself a finding about small local models as judges.

Three things are deliberately NOT delegated to the LLM:

1. **Discordance mode.** Computed quantitatively from percentile-rank
   displacement of stage1_score vs. stage2_score within the flagged+noted
   cohort (see :func:`compute_discordance`), not asked of the model. Percentile
   rank is used instead of a raw-probability difference because Stage 1 and
   Stage 2 are different model families and are not guaranteed to be equally
   well-calibrated even after isotonic calibration — rank displacement is
   invariant to that risk. Stage 1 uses ~40 structured features; Stage 2 (a
   plain, note-only Clinical-Longformer) uses none. The two models are
   informationally independent by construction.
2. **``decision_rule``.** Deterministically recomputed in code from the
   grounds the model itself extracted (see :func:`compute_decision_rule`) —
   not a second opinion asked of the model, a check on whether the model's
   own stated decision actually follows its own stated rubric.
3. **Whether the auditor's own decision is reproducible.** Ollama temperature
   is pinned at 0 (``cfg.stage3.temperature``) for every evaluation run.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidationError

try:
    import ollama
except ImportError as _err:
    raise ImportError(
        "Ollama is required for Stage 3. "
        "Install with: pip install ollama  then  ollama pull phi4-mini"
    ) from _err

from src.config_schema import AppConfig


# ── Taxonomy ───────────────────────────────────────────────────────────────────

DECISIONS: tuple[str, ...] = ("uphold", "override", "insufficient_evidence")

DISCORDANCE_MODES: tuple[str, ...] = (
    "CONCORDANT",      # Stage 1 and Stage 2 rank this patient similarly
    "NOTE_MITIGATES",  # Stage 2 ranks the patient markedly lower risk than Stage 1
    "NOTE_AMPLIFIES",  # Stage 2 ranks the patient markedly higher risk than Stage 1
)

# Two-sided grounds taxonomy (session 19), replacing the single free-choice
# primary_clinical_domain. Each ground the model cites must carry its own
# verbatim quote (validated in _parse_response, verified against the note in
# call_llm) -- fixed list only; anything outside it is a parse failure, not a
# new category (don't let the model invent grounds).
# See _MITIGATING_DESCRIPTIONS / _AGGRAVATING_DESCRIPTIONS below for what each means.
MITIGATING_GROUNDS: tuple[str, ...] = (
    "palliative_intent",
    "planned_return",
    "strong_discharge_support",
    "structured_driver_contradicted",
)

AGGRAVATING_GROUNDS: tuple[str, ...] = (
    "lives_alone_no_support",
    "no_followup_arranged",
    "functional_dependence",
    "cognitive_impairment",
    "nonadherence_risk",
    "unstable_at_discharge",
)

_MITIGATING_DESCRIPTIONS: dict[str, str] = {
    "palliative_intent": "hospice, palliative, or comfort-focused care documented — "
                          "readmission isn't the relevant outcome",
    "planned_return": "a scheduled return is documented (chemo cycle, staged "
                       "procedure, planned dialysis admission)",
    "strong_discharge_support": "follow-up appointment arranged and "
                                 "caregiver/support present and clinically "
                                 "stable at discharge",
    "structured_driver_contradicted": "the note explicitly contradicts what "
                                       "drove the structured model's alert",
}

_AGGRAVATING_DESCRIPTIONS: dict[str, str] = {
    "lives_alone_no_support": "lives alone or no caregiver documented",
    "no_followup_arranged": "no follow-up appointment documented",
    "functional_dependence": "needs help with daily activities, or a new mobility aid",
    "cognitive_impairment": "delirium, dementia, or confusion documented",
    "nonadherence_risk": "active substance use, missed appointments, or "
                          "medication-management concerns",
    "unstable_at_discharge": "unresolved clinical issues at the point of discharge",
}

# Colleague review, 2026-08-27, item 3: does the note itself mention a
# planned return (chemo cycle, staged surgery, scheduled dialysis)? This is
# independent of, not a replacement for, the structured admission_type-based
# readmission_30d_unplanned proxy in src/data/features.py -- the two use
# different evidence and are reported separately. It is ALSO independent of
# "planned_return" appearing in MITIGATING_GROUNDS above: the standalone
# field always gets an answer (for the label-audit comparison); the ground
# is cited only when it actually drove the decision. Do not collapse these
# two into one field.
PLANNED_RETURN_ANSWERS: tuple[str, ...] = ("yes", "no", "not_stated")

# Generous safety cap, not a routine truncation. Session 14 measured median
# MIMIC discharge notes at ~2,649 tokens; at Stage 2's 4096-token window
# (~4-5 chars/token in clinical text) that is comfortably under 20,000 chars.
_NOTE_MAX_CHARS = 20_000

# Below this length a note cannot ground either a mitigating or an
# aggravating finding, regardless of what the model claims to have
# extracted -- decision_rule reports insufficient_evidence rather than
# trusting an extraction from an uninformative note. A code-side judgment
# call, not tuned to any labelled outcome: chosen well below the shortest
# real discharge-summary body seen in exploratory review (session 14).
_MIN_INFORMATIVE_NOTE_CHARS = 200


# ── LLM output shape (also the schema-constrained generation target) ───────────

class _GroundHit(BaseModel):
    """One extracted ground with its supporting quote, as returned by the LLM."""

    ground: str
    quote: str


class _LLMOutput(BaseModel):
    """Raw shape of the LLM's JSON response, before taxonomy/decision validation.

    Passed to Ollama as a JSON schema (``format=``) for schema-constrained
    generation, and used to parse the response -- this is what nearly
    eliminates malformed-JSON parse failures. Membership in the fixed
    grounds/decision/planned_return taxonomies is still validated separately
    in :func:`_parse_response`, since a JSON schema can constrain shape but
    this project keeps enum validation explicit and testable in Python.
    """

    mitigating_grounds: list[_GroundHit] = []
    aggravating_grounds: list[_GroundHit] = []
    planned_return: str
    clinical_justification: str
    decision: str


# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are an independent clinical auditor reviewing a 30-day hospital
    readmission risk alert raised by a structured EHR model.

    You will be given: the structured model's risk score and its stated
    reasons (SHAP attributions), a second, independently-derived risk estimate
    from a model that reads the discharge note, and the discharge note itself.

    Your job is to reach your OWN judgment about whether this alert should be
    upheld or overridden — you are not explaining or narrating a decision
    someone else already made. Extract evidence from the note FIRST, then
    decide — do not decide first and invent supporting evidence afterward.
    Read the note as a clinician would. Be specific and cite actual clinical
    content. Do not invent information not present in the inputs, and do not
    cite a ground that is not actually documented in the note.

    Return ONLY valid JSON. No text outside the JSON object.
""").strip()

_USER_TEMPLATE = textwrap.dedent("""
    A structured model flagged this patient as high-risk for 30-day
    readmission.

    ── STRUCTURED RISK ESTIMATE (XGBoost) ────────────────────────────────
    Score: {stage1_score:.3f}   Flagged at threshold >= {stage1_threshold:.3f}
    Top structured risk factors (SHAP-ranked):
    {shap_block}

    {stage2_evidence_block}

    ── DISCHARGE NOTE ───────────────────────────────────────────────────────
    {note_block}
    {attention_block}

    ── YOUR TASK ────────────────────────────────────────────────────────────
    Read the discharge note and both risk estimates. Identify which of the
    following grounds, if any, are documented in the note. Only cite a
    ground if the note actually documents it — do not invent one to justify
    a decision you have already reached. Extract first, decide after.

    Mitigating grounds (support overriding/cancelling the alert):
    {mitigating_block}

    Aggravating grounds (support upholding the alert):
    {aggravating_block}

    Return ONLY a JSON object with exactly these five fields:

    "mitigating_grounds": list of objects {{"ground": <one of the mitigating
      grounds above>, "quote": <exact verbatim sentence from the note>}}.
      Empty list if none apply.

    "aggravating_grounds": same shape, drawn from the aggravating grounds
      above. Empty list if none apply.

    "planned_return": does the note mention a planned return — a scheduled
      chemotherapy cycle, a staged surgery, scheduled dialysis, or similar —
      regardless of whether it affected your decision? One of
      {planned_return_options}.

    "clinical_justification": 2-4 sentences, written AFTER you have
      identified the grounds above. Synthesise what you found — do not
      introduce claims not reflected in the grounds you extracted.

    "decision": one of {decisions}
      uphold                 — the alert should stand; the grounds you found
                                (if any) do not justify suppressing it
      override                — the mitigating grounds you found justify
                                suppressing this alert, and no aggravating
                                ground contradicts them
      insufficient_evidence  — the note is too short or uninformative to
                                assess either way
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

def is_note_truncated(note_text: str) -> bool:
    """Whether ``note_text`` exceeds the safety cap applied in the prompt.

    Exposed separately from :func:`_note_block` so callers (pipeline.py,
    batch.py) can log truncation as structured data, not just prose inside
    the prompt string.
    """
    return len(note_text.strip()) > _NOTE_MAX_CHARS


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


def _grounds_block(grounds: tuple[str, ...], descriptions: dict[str, str]) -> str:
    """Render a grounds taxonomy as a labelled list for the prompt."""
    return "\n".join(f"  - {g}: {descriptions[g]}" for g in grounds)


def _stage2_evidence_block(
    stage2_score: float, discordance: dict[str, float | str], hide: bool
) -> str:
    """Render Stage 2's score + the quantitative discordance section.

    ``hide=True`` (the no-Stage-2 validation control, session 19 Phase D2)
    withholds this evidence entirely rather than passing a zeroed or
    placeholder score — the point of the control is to test whether Stage 2
    earns its place in the prompt, not to feed the model a misleading value.
    """
    if hide:
        return (
            "── NOTE-BASED RISK ESTIMATE ──────────────────────────────────────────\n"
            "    (withheld for this run — reason from the structured estimate and\n"
            "    the discharge note alone.)"
        )
    return textwrap.dedent(f"""\
        ── NOTE-BASED RISK ESTIMATE (Clinical-Longformer, discharge note only) ──
        Score: {stage2_score:.3f}
        This model reads only the discharge note — no structured features. It
        was trained independently and does not know the score above.

        ── QUANTITATIVE DISCORDANCE ────────────────────────────────────────────
        Within the cohort of flagged, note-covered patients, this patient ranks
        at the {discordance["r1"]:.0f}th percentile on the structured estimate and the
        {discordance["r2"]:.0f}th percentile on the note-based estimate (displacement:
        {discordance["displacement"]:+.0f} percentile points -> {discordance["mode"]}).
          NOTE_MITIGATES -> the note-based estimate ranks this patient markedly
                             lower risk than the structured estimate alone.
          NOTE_AMPLIFIES -> the note-based estimate ranks this patient markedly
                             higher risk than the structured estimate alone.
          CONCORDANT     -> the two estimates rank this patient similarly.
        This is context, not a conclusion — reach your own judgment from the
        evidence below.""")


def verify_quote(quote: str, note_text: str) -> bool:
    """Return whether ``quote`` appears verbatim in ``note_text``.

    Computed in code, not asked of the LLM — this is what turns "the model
    says it quoted the note" into something automatically checkable, and is
    the mechanism that makes human spot-checking tractable instead of
    impossible: a reviewer only needs to check the (hopefully small) subset
    where a quote is unverified, not re-read every note from scratch.
    """
    if not quote.strip():
        return False
    return quote.strip() in note_text


def compute_decision_rule(
    mitigating_grounds: list[dict[str, str]],
    aggravating_grounds: list[dict[str, str]],
    note_text: str,
) -> str:
    """Deterministically recompute the decision from extracted grounds.

    Independent of ``decision_model`` (the LLM's own stated decision) —
    reported alongside it as a consistency metric and a fully transparent
    fallback (docs/ARCHITECTURE.md §2, §5 item 3a: a small addition to the
    schema, not a replacement for the model's free judgment). Not asked of
    the model.

    ``insufficient_evidence`` here is a code-side judgment about the note's
    length, not a claim the LLM makes about itself — a note this short
    cannot ground either a mitigating or an aggravating finding regardless
    of what was extracted from it.
    """
    if len(note_text.strip()) < _MIN_INFORMATIVE_NOTE_CHARS:
        return "insufficient_evidence"
    if mitigating_grounds and not aggravating_grounds:
        return "override"
    return "uphold"


_PARSE_FAILURE: dict[str, Any] = {
    "mitigating_grounds": [],
    "aggravating_grounds": [],
    "planned_return": None,
    "clinical_justification": "",
    "decision_model": None,
    "decision_rule": None,
    "all_quotes_verified": None,
    "annotation_failed": True,
}


def _validate_grounds(
    raw_grounds: list[_GroundHit], allowed: tuple[str, ...]
) -> list[dict[str, str]] | None:
    """Return validated {ground, quote} dicts, or None if any entry is invalid.

    Fixed list only — a ground outside ``allowed``, or one with an empty
    quote, fails the whole response (don't let the model invent categories
    or cite a ground without evidence).
    """
    out: list[dict[str, str]] = []
    for hit in raw_grounds:
        if hit.ground not in allowed or not hit.quote.strip():
            return None
        out.append({"ground": hit.ground, "quote": hit.quote})
    return out


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse the LLM's JSON response into a flat annotation dict.

    Returns ``annotation_failed=True`` (with ``None``/empty fields) on any
    parse or validation error so callers can distinguish a real decision
    from a silent failure. ``quote_verified`` per ground and
    ``decision_rule`` are NOT set here — both need ``note_text``, which this
    function doesn't have; see :func:`call_llm`.
    """
    parsed: _LLMOutput | None = None
    try:
        parsed = _LLMOutput.model_validate_json(raw)
    except ValidationError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = _LLMOutput.model_validate_json(raw[start:end])
            except ValidationError:
                pass

    if parsed is None:
        return {**_PARSE_FAILURE, "clinical_justification": raw.strip()[:300]}

    mitigating = _validate_grounds(parsed.mitigating_grounds, MITIGATING_GROUNDS)
    aggravating = _validate_grounds(parsed.aggravating_grounds, AGGRAVATING_GROUNDS)
    valid_decision = parsed.decision if parsed.decision in DECISIONS else None
    valid_planned_return = (
        parsed.planned_return if parsed.planned_return in PLANNED_RETURN_ANSWERS else None
    )

    if (
        mitigating is None or aggravating is None
        or valid_decision is None or valid_planned_return is None
    ):
        return {**_PARSE_FAILURE, "clinical_justification": parsed.clinical_justification}

    return {
        "mitigating_grounds": mitigating,
        "aggravating_grounds": aggravating,
        "planned_return": valid_planned_return,
        "clinical_justification": parsed.clinical_justification,
        "decision_model": valid_decision,
        "decision_rule": None,       # filled in by call_llm, which has note_text
        "all_quotes_verified": None,  # filled in by call_llm, which has note_text
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
    *,
    hide_stage2: bool = False,
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
        hide_stage2:           if True, withhold Stage 2's score and the
                                discordance section entirely (the no-Stage-2
                                validation control, session 19 Phase D2) —
                                tests whether Stage 2 earns its place in the
                                prompt, not fed a misleading placeholder.

    Returns:
        Formatted prompt string ready for the LLM.
    """
    shap_block = (
        "\n".join(f"  - {f}" for f in shap_feature_strings)
        if shap_feature_strings
        else "  (not available)"
    )

    return _USER_TEMPLATE.format(
        stage1_score=stage1_score,
        stage1_threshold=stage1_threshold,
        shap_block=shap_block,
        stage2_evidence_block=_stage2_evidence_block(stage2_score, discordance, hide_stage2),
        note_block=_note_block(note_text),
        attention_block=_attention_block(attention_sentences or []),
        mitigating_block=_grounds_block(MITIGATING_GROUNDS, _MITIGATING_DESCRIPTIONS),
        aggravating_block=_grounds_block(AGGRAVATING_GROUNDS, _AGGRAVATING_DESCRIPTIONS),
        decisions=str(DECISIONS),
        planned_return_options=str(PLANNED_RETURN_ANSWERS),
    )


def call_llm(
    prompt: str,
    cfg: AppConfig,
    model_name: str | None = None,
    note_text: str = "",
) -> dict[str, Any]:
    """Call an Ollama-hosted model and return the parsed annotation dict.

    Uses schema-constrained generation (``format=<JSON schema>``, not the
    generic ``format="json"``) — this is what nearly eliminates malformed-
    JSON parse failures, per the colleague review that motivated this design.

    Generalised so the same prompt can be run through a different model —
    e.g. ``cfg.stage3.robustness_model`` — as a robustness check on whether
    the auditor's value depends on model scale, without duplicating the
    prompt/parsing logic. All models here are assumed Ollama-hosted (local);
    routing to a cloud API is a separate, currently unmade decision — see
    docs/ARCHITECTURE.md.

    Args:
        prompt:     built by :func:`build_prompt`.
        cfg:        validated project config (reads ``stage3.temperature``
                    — pinned at 0 for reproducibility).
        model_name: Ollama model tag to use. Defaults to
                    ``cfg.stage3.ollama_model`` (the primary auditor model).
        note_text:  the same raw note text passed to :func:`build_prompt` —
                    used to verify each ground's quote against it and to
                    compute ``decision_rule``, not re-sent to the model.

    Returns:
        Dict with keys ``mitigating_grounds``, ``aggravating_grounds``,
        ``planned_return``, ``clinical_justification``, ``decision_model``,
        ``decision_rule``, ``all_quotes_verified``, ``annotation_failed``.
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
            format=_LLMOutput.model_json_schema(),
        )
        raw = response.message.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {**_PARSE_FAILURE, "clinical_justification": f"[ollama error: {exc}]"}

    annotation = _parse_response(raw)
    if annotation["annotation_failed"]:
        return annotation

    mitigating = [
        {**g, "quote_verified": verify_quote(g["quote"], note_text)}
        for g in annotation["mitigating_grounds"]
    ]
    aggravating = [
        {**g, "quote_verified": verify_quote(g["quote"], note_text)}
        for g in annotation["aggravating_grounds"]
    ]
    annotation["mitigating_grounds"] = mitigating
    annotation["aggravating_grounds"] = aggravating
    annotation["all_quotes_verified"] = all(
        g["quote_verified"] for g in mitigating + aggravating
    )
    annotation["decision_rule"] = compute_decision_rule(
        annotation["mitigating_grounds"], annotation["aggravating_grounds"], note_text
    )
    return annotation
