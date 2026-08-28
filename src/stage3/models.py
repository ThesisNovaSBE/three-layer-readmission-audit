"""Pydantic output model for Stage 3 on-demand patient explanations."""

from __future__ import annotations

from pydantic import BaseModel


class GroundHit(BaseModel):
    """One extracted ground, its verbatim quote, and whether that quote
    verified against the source note (computed in code, not asked of the
    model — see :func:`src.stage3.explain.verify_quote`)."""

    ground: str
    quote: str
    quote_verified: bool


class ExplanationResult(BaseModel):
    """Structured output from a single-patient Stage 3 audit.

    Serialises cleanly to JSON for the API response and frontend consumption.
    ``annotation_failed=True`` means the LLM returned unparseable or
    taxonomy-violating output; the other fields still contain the raw inputs
    so the frontend can display the structured data even without a decision.

    ``mitigating_grounds`` / ``aggravating_grounds`` (session 19): the two-
    sided grounds taxonomy the model extracts from the note, each with its
    own verified quote — replaces the single free-choice
    ``primary_clinical_domain`` from the 2026-08-25 design. ``planned_return``
    is a separate, always-answered field (not one of the grounds, even
    though "planned_return" also exists as a mitigating ground the model may
    cite when it drives the decision) — it is the model's own note-derived
    answer to whether a scheduled return is documented, independent of the
    structured admission_type-based proxy in ``src/data/features.py``;
    reported separately, not merged.

    Two decisions, not one:

    - ``decision_model`` is the LLM's own independent uphold / override /
      insufficient_evidence judgment (not a narration of Stage 2's
      confirm/reject output — Stage 2's score is one piece of evidence the
      auditor reasons over, not a decision it explains). This is the value
      that drives the final pipeline prediction.
    - ``decision_rule`` is the same three-way decision recomputed
      deterministically in code from the grounds the model extracted (see
      :func:`src.stage3.explain.compute_decision_rule`) — not asked of the
      model. Reported alongside ``decision_model``, never in place of it;
      their agreement rate is a reportable consistency metric for small
      local models as judges, and ``decision_rule`` is a fully transparent
      fallback if ``decision_model`` proves unreliable.

    ``all_quotes_verified`` is True only if every extracted ground's quote
    was found verbatim in the note (vacuously True if no grounds were
    extracted) — the single number to check before trusting a row's grounds
    at all. ``note_truncated`` and ``model_name`` are logged per row so a
    scale-robustness comparison (mixing model sizes) and truncation-asymmetry
    analysis can be reconstructed from the output alone.
    """

    hadm_id: int
    stage1_score: float
    stage1_threshold: float
    stage2_score: float
    stage2_confirmed: bool
    r1: float
    r2: float
    displacement: float
    discordance_mode: str
    top_shap_features: list[str]
    attention_sentences: list[str]
    mitigating_grounds: list[GroundHit]
    aggravating_grounds: list[GroundHit]
    all_quotes_verified: bool | None
    planned_return: str | None
    clinical_justification: str
    decision_model: str | None
    decision_rule: str | None
    note_truncated: bool
    model_name: str
    annotation_failed: bool
