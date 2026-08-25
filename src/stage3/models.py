"""Pydantic output model for Stage 3 on-demand patient explanations."""

from __future__ import annotations

from pydantic import BaseModel


class ExplanationResult(BaseModel):
    """Structured output from a single-patient Stage 3 audit.

    Serialises cleanly to JSON for the API response and frontend consumption.
    ``annotation_failed=True`` means phi4-mini returned unparseable output;
    the other fields still contain the raw inputs so the frontend can display
    the structured data even without a decision.

    ``decision`` is phi4-mini's own independent uphold/override judgment (not
    a narration of Stage 2's confirm/reject output — Stage 2's score is one
    piece of evidence the auditor reasons over, not a decision it explains).
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
    decision: str | None
    primary_clinical_domain: str | None
    clinical_justification: str
    annotation_failed: bool
