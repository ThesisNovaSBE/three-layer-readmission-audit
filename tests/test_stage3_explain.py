"""Tests for Stage 3 prompt construction and response parsing (no Ollama needed)."""

from __future__ import annotations

import json

from src.stage3.explain import (
    DISCORDANCE_CATEGORIES,
    DISCORDANCE_MODES,
    _delta_label,
    _parse_response,
    build_prompt,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_modes_tuple_nonempty():
    """DISCORDANCE_MODES must contain all three expected mode strings."""
    assert len(DISCORDANCE_MODES) == 3
    assert "CONCORDANT" in DISCORDANCE_MODES
    assert "NOTE_MITIGATES" in DISCORDANCE_MODES
    assert "NOTE_AMPLIFIES" in DISCORDANCE_MODES


def test_categories_tuple_nonempty():
    """DISCORDANCE_CATEGORIES must contain at least eight clinical domains."""
    assert len(DISCORDANCE_CATEGORIES) >= 8
    assert "structured_confirmed" in DISCORDANCE_CATEGORIES
    assert "social_support" in DISCORDANCE_CATEGORIES


# ── _delta_label ──────────────────────────────────────────────────────────────

def test_delta_label_strongly_negative():
    """Large negative delta must indicate strong mitigation."""
    assert "strongly mitigates" in _delta_label(-0.5)


def test_delta_label_somewhat_negative():
    """Moderate negative delta must indicate partial mitigation."""
    assert "somewhat mitigates" in _delta_label(-0.2)


def test_delta_label_near_zero():
    """Delta near zero must indicate agreement between modalities."""
    assert "agree" in _delta_label(0.0)
    assert "agree" in _delta_label(0.05)
    assert "agree" in _delta_label(-0.05)


def test_delta_label_somewhat_positive():
    """Moderate positive delta must indicate partial amplification."""
    assert "somewhat amplifies" in _delta_label(0.2)


def test_delta_label_strongly_positive():
    """Large positive delta must indicate strong amplification."""
    assert "strongly amplifies" in _delta_label(0.6)


# ── build_prompt ──────────────────────────────────────────────────────────────

def _make_prompt(confirmed: bool = True, **kwargs) -> str:
    """Helper: build a test prompt with sensible defaults, overridable via kwargs."""
    defaults = {
        "stage1_score": 0.70,
        "stage1_threshold": 0.35,
        "stage2_score": 0.80 if confirmed else 0.10,
        "stage2_threshold": 0.30,
        "stage2_confirmed": confirmed,
        "shap_feature_strings": ["creatinine (last): 3.2 (↑ risk, SHAP=+0.18)"],
        "attention_sentences": [],
        "note_text": "",
    }
    defaults.update(kwargs)
    return build_prompt(**defaults)


def test_build_prompt_contains_stage1_score():
    """Stage 1 score must appear in the prompt."""
    assert "0.700" in _make_prompt()


def test_build_prompt_contains_stage1_threshold():
    """Stage 1 threshold must appear in the prompt."""
    assert "0.350" in _make_prompt()


def test_build_prompt_contains_stage2_score():
    """Stage 2 score must appear in the prompt."""
    assert "0.800" in _make_prompt()


def test_build_prompt_confirmed_verdict():
    """Confirmed decision must produce CONFIRMED in the prompt."""
    assert "CONFIRMED" in _make_prompt(confirmed=True)


def test_build_prompt_rejected_verdict():
    """Rejected decision must produce REJECTED in the prompt."""
    assert "REJECTED" in _make_prompt(confirmed=False)


def test_build_prompt_delta_present():
    """Score delta must appear in the prompt."""
    prompt = _make_prompt(
        stage1_score=0.70, stage2_score=0.10, stage2_confirmed=False
    )
    assert "-0.600" in prompt


def test_build_prompt_shap_features_included():
    """All SHAP feature strings must appear in the prompt."""
    prompt = _make_prompt(shap_feature_strings=["creatinine: 3.2", "age: 78"])
    assert "creatinine: 3.2" in prompt
    assert "age: 78" in prompt


def test_build_prompt_attention_sentences_used_over_note_text():
    """Attention sentences must take priority over raw note_text."""
    prompt = _make_prompt(
        attention_sentences=["Patient discharged home with good support."],
        note_text="Should not appear.",
    )
    assert "Patient discharged home with good support." in prompt
    assert "Should not appear." not in prompt


def test_build_prompt_note_text_fallback_when_no_attention():
    """Raw note text must appear in the prompt when attention spans are absent."""
    prompt = _make_prompt(
        attention_sentences=[],
        note_text="Strong family support documented. Follow-up arranged.",
    )
    assert "Strong family support documented" in prompt
    assert "(discharge note not available)" not in prompt


def test_build_prompt_no_note_fallback_message():
    """When both attention and note_text are absent the prompt must say so."""
    prompt = _make_prompt(attention_sentences=[], note_text="")
    assert "not available" in prompt


def test_build_prompt_modes_listed():
    """All three discordance modes must be listed in the prompt."""
    prompt = _make_prompt()
    for mode in DISCORDANCE_MODES:
        assert mode in prompt


def test_build_prompt_categories_listed():
    """All clinical categories must be listed in the prompt."""
    prompt = _make_prompt()
    for cat in DISCORDANCE_CATEGORIES:
        assert cat in prompt


# ── _parse_response ───────────────────────────────────────────────────────────

def _good_json(
    mode: str = "CONCORDANT",
    cat: str = "structured_confirmed",
    narrative: str = "Note confirms the structured risk.",
) -> str:
    """Return a well-formed phi4-mini JSON response string."""
    return json.dumps({
        "discordance_mode": mode,
        "primary_category": cat,
        "narrative": narrative,
    })


def test_parse_valid_concordant():
    """Valid CONCORDANT response must parse without failures."""
    result = _parse_response(_good_json("CONCORDANT", "structured_confirmed"))
    assert result["discordance_mode"] == "CONCORDANT"
    assert result["primary_category"] == "structured_confirmed"
    assert result["annotation_failed"] is False


def test_parse_valid_note_mitigates():
    """Valid NOTE_MITIGATES response must parse without failures."""
    result = _parse_response(_good_json("NOTE_MITIGATES", "social_support"))
    assert result["discordance_mode"] == "NOTE_MITIGATES"
    assert result["annotation_failed"] is False


def test_parse_valid_note_amplifies():
    """Valid NOTE_AMPLIFIES response must parse without failures."""
    result = _parse_response(_good_json("NOTE_AMPLIFIES", "frailty_markers"))
    assert result["discordance_mode"] == "NOTE_AMPLIFIES"
    assert result["annotation_failed"] is False


def test_parse_bad_json_sets_annotation_failed():
    """Unparseable output must set annotation_failed=True with None fields."""
    result = _parse_response("not json at all")
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None
    assert result["primary_category"] is None


def test_parse_unknown_mode_sets_annotation_failed():
    """Unrecognised discordance_mode must set annotation_failed=True."""
    bad = json.dumps({
        "discordance_mode": "DISAGREEMENT",
        "primary_category": "social_support",
        "narrative": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None


def test_parse_unknown_category_sets_annotation_failed():
    """Unrecognised primary_category must set annotation_failed=True."""
    bad = json.dumps({
        "discordance_mode": "CONCORDANT",
        "primary_category": "unknown_category",
        "narrative": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["primary_category"] is None


def test_parse_json_embedded_in_text():
    """JSON embedded in prose (as phi4-mini sometimes produces) must parse correctly."""
    raw = (
        'Here is the analysis: {"discordance_mode": "NOTE_AMPLIFIES", '
        '"primary_category": "cognition", "narrative": "Delirium noted."}'
    )
    result = _parse_response(raw)
    assert result["annotation_failed"] is False
    assert result["discordance_mode"] == "NOTE_AMPLIFIES"
    assert result["primary_category"] == "cognition"


def test_parse_narrative_preserved():
    """The narrative string from phi4-mini must be preserved verbatim."""
    narrative = "Patient has robust social support reducing readmission risk."
    result = _parse_response(_good_json("NOTE_MITIGATES", "social_support", narrative))
    assert result["narrative"] == narrative


def test_parse_failure_does_not_default_to_concordant():
    """Empty JSON must not silently inflate CONCORDANT counts."""
    result = _parse_response("{}")
    assert result["annotation_failed"] is True
    assert result["discordance_mode"] is None
