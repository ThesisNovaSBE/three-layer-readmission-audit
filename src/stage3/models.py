"""Pydantic output model for Stage 3 on-demand patient explanations."""

from __future__ import annotations

from pydantic import BaseModel


class ExplanationResult(BaseModel):
    """Structured output from a single-patient Stage 3 explanation.

    Serialises cleanly to JSON for the API response and frontend consumption.
    ``annotation_failed=True`` means phi4-mini returned unparseable output;
    the other fields still contain the raw inputs so the frontend can display
    the structured data even without a narrative.
    """

    hadm_id: int
    stage1_score: float
    stage1_threshold: float
    stage2_score: float
    stage2_confirmed: bool
    score_delta: float
    top_shap_features: list[str]
    attention_sentences: list[str]
    discordance_mode: str | None
    primary_category: str | None
    narrative: str
    annotation_failed: bool
