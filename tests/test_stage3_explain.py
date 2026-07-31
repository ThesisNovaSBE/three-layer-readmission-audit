"""Tests for Stage 3 prompt building and JSON parsing (no Ollama required)."""

from __future__ import annotations

import json

import pytest

from src.stage3.explain import (
    DISCORDANCE_CATEGORIES,
    DISCORDANCE_MODES,
    _parse_response,
    build_prompt,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_modes_tuple_nonempty():
    assert len(DISCORDANCE_MODES) == 3
    assert "CONCORDANT" in DISCORDANCE_MODES
    assert "NOTE_MITIGATES" in DISCORDANCE_MODES
    assert "NOTE_AMPLIFIES" in DISCORDANCE_MODES


def test_categories_tuple_nonempty():
    assert len(DISCORDANCE_CATEGORIES) >= 8
    assert "structured_confirmed" in DISCORDANCE_CATEGORIES
    assert "social_support" in DISCORDANCE_CATEGORIES


# ── build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_contains_stage2_verdict_confirmed():
    prompt = build_prompt(
        stage2_confirmed=True,
        stage2_score=0.72,
        shap_feature_strings=["creatinine (last): 3.2 (↑ risk, SHAP=+0.18)"],
        attention_sentences=["Patient was discharged home."],
    )
    assert "CONFIRMED" in prompt
    assert "0.720" in prompt


def test_build_prompt_contains_stage2_verdict_rejected():
    prompt = build_prompt(
        stage2_confirmed=False,
        stage2_score=0.21,
        shap_feature_strings=[],
        attention_sentences=[],
    )
    assert "REJECTED" in prompt
    assert "0.210" in prompt


def test_build_prompt_includes_shap_features():
    features = ["creatinine (last): 3.2 (↑ risk)", "age (years): 78"]
    prompt = build_prompt(True, 0.8, features, [])
    for f in features:
        assert f in prompt


def test_build_prompt_includes_attention_sentences():
    sentences = ["Good social support noted.", "Adequate follow-up arranged."]
    prompt = build_prompt(False, 0.3, [], sentences)
    for s in sentences:
        assert s in prompt


def test_build_prompt_fallback_when_no_shap():
    prompt = build_prompt(True, 0.9, [], [])
    assert "not available" in prompt


def test_build_prompt_fallback_when_no_attention():
    prompt = build_prompt(True, 0.9, [], [])
    assert "not available" in prompt


def test_build_prompt_modes_in_prompt():
    prompt = build_prompt(True, 0.5, [], [])
    for mode in DISCORDANCE_MODES:
        assert mode in prompt


# ── _parse_response ───────────────────────────────────────────────────────────

def _good_json(
    mode: str = "CONCORDANT",
    cat: str = "structured_confirmed",
    explanation: str = "Note confirms structured data.",
) -> str:
    return json.dumps({
        "discordance_mode": mode,
        "primary_category": cat,
        "explanation": explanation,
    })


def test_parse_valid_concordant():
    result = _parse_response(_good_json("CONCORDANT", "structured_confirmed"))
    assert result["discordance_mode"] == "CONCORDANT"
    assert result["primary_category"] == "structured_confirmed"
    assert result["annotation_failed"] is False


def test_parse_valid_note_mitigates():
    result = _parse_response(_good_json("NOTE_MITIGATES", "social_support"))
    assert result["discordance_mode"] == "NOTE_MITIGATES"
    assert result["annotation_failed"] is False


def test_parse_valid_note_amplifies():
    result = _parse_response(_good_json("NOTE_AMPLIFIES", "frailty_markers"))
    assert result["discordance_mode"] == "NOTE_AMPLIFIES"
    assert result["annotation_failed"] is False


def test_parse_bad_json_sets_annotation_failed():
    result = _parse_response("not json at all")
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None
    assert result["primary_category"] is None


def test_parse_unknown_mode_sets_annotation_failed():
    bad = json.dumps({
        "discordance_mode": "DISAGREEMENT",  # not a valid mode
        "primary_category": "social_support",
        "explanation": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None


def test_parse_unknown_category_sets_annotation_failed():
    bad = json.dumps({
        "discordance_mode": "CONCORDANT",
        "primary_category": "unknown_category",
        "explanation": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["primary_category"] is None


def test_parse_json_embedded_in_text():
    """phi4-mini sometimes wraps the JSON in markdown or prose."""
    raw = 'Here is the analysis: {"discordance_mode": "NOTE_AMPLIFIES", "primary_category": "cognition", "explanation": "Delirium noted."}'
    result = _parse_response(raw)
    assert result["annotation_failed"] is False
    assert result["discordance_mode"] == "NOTE_AMPLIFIES"
    assert result["primary_category"] == "cognition"


def test_parse_explanation_preserved():
    explanation = "Patient has robust social support reducing readmission risk."
    result = _parse_response(_good_json("NOTE_MITIGATES", "social_support", explanation))
    assert result["explanation"] == explanation


def test_parse_failure_does_not_default_to_concordant():
    """Failures must not silently inflate CONCORDANT counts."""
    result = _parse_response("{}")
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None
