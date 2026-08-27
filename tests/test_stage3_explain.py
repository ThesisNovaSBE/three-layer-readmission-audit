"""Tests for Stage 3 prompt construction, discordance, and response parsing.

No Ollama needed — these test pure prompt-building and parsing logic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.stage3.explain import (
    CLINICAL_DOMAINS,
    DECISIONS,
    DISCORDANCE_MODES,
    _parse_response,
    build_prompt,
    compute_discordance,
    sweep_discordance_thresholds,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_decisions_tuple():
    """DECISIONS must contain exactly uphold/override."""
    assert set(DECISIONS) == {"uphold", "override"}


def test_modes_tuple_nonempty():
    """DISCORDANCE_MODES must contain all three expected mode strings."""
    assert len(DISCORDANCE_MODES) == 3
    assert "CONCORDANT" in DISCORDANCE_MODES
    assert "NOTE_MITIGATES" in DISCORDANCE_MODES
    assert "NOTE_AMPLIFIES" in DISCORDANCE_MODES


def test_domains_tuple_nonempty():
    """CLINICAL_DOMAINS must contain at least six clinical domains."""
    assert len(CLINICAL_DOMAINS) >= 6
    assert "social_support" in CLINICAL_DOMAINS
    assert "other" in CLINICAL_DOMAINS


# ── compute_discordance ─────────────────────────────────────────────────────────

def test_discordance_concordant_when_ranks_match():
    """Equal percentile ranks must be CONCORDANT."""
    cohort = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = compute_discordance(0.3, 0.3, cohort, cohort, displacement_pp=20.0)
    assert result["mode"] == "CONCORDANT"
    assert result["displacement"] == 0.0


def test_discordance_note_mitigates_when_stage2_ranks_lower():
    """Stage 2 ranking the patient much lower than Stage 1 must be NOTE_MITIGATES."""
    cohort_s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # 0.9 -> high rank
    cohort_s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.15])  # 0.15 -> low rank
    result = compute_discordance(0.9, 0.15, cohort_s1, cohort_s2, displacement_pp=20.0)
    assert result["mode"] == "NOTE_MITIGATES"
    assert result["displacement"] < 0


def test_discordance_note_amplifies_when_stage2_ranks_higher():
    """Stage 2 ranking the patient much higher than Stage 1 must be NOTE_AMPLIFIES."""
    cohort_s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.15])  # 0.15 -> low rank
    cohort_s2 = np.array([0.1, 0.2, 0.3, 0.4, 0.9])  # 0.9 -> high rank
    result = compute_discordance(0.15, 0.9, cohort_s1, cohort_s2, displacement_pp=20.0)
    assert result["mode"] == "NOTE_AMPLIFIES"
    assert result["displacement"] > 0


def test_discordance_invariant_to_monotonic_rescaling():
    """Rank displacement must be unchanged by a monotonic rescaling of one score.

    This is the property raw-probability subtraction does not have: it makes
    the measure robust to Stage 1 and Stage 2 being unequally calibrated.
    """
    cohort_s1 = np.array([0.05, 0.20, 0.35, 0.50, 0.65, 0.80])
    cohort_s2 = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
    base = compute_discordance(0.50, 0.55, cohort_s1, cohort_s2)

    # Monotonic rescale of stage2's cohort and query value (e.g. a different
    # calibration curve) — ranks, and therefore displacement, must be identical.
    rescaled_cohort_s2 = cohort_s2**2
    rescaled = compute_discordance(0.50, 0.55**2, cohort_s1, rescaled_cohort_s2)
    assert rescaled["displacement"] == base["displacement"]
    assert rescaled["mode"] == base["mode"]


def test_discordance_empty_cohort_defaults_to_50th_percentile():
    """An empty cohort must not raise — falls back to the 50th percentile."""
    result = compute_discordance(0.5, 0.5, np.array([]), np.array([]))
    assert result["r1"] == 50.0
    assert result["r2"] == 50.0


# ── sweep_discordance_thresholds ─────────────────────────────────────────────

def test_sweep_default_thresholds():
    """Default sweep must cover the standard 10/15/20/25/30 pp range."""
    displacements = np.array([-40.0, -10.0, 0.0, 15.0, 45.0])
    sweep = sweep_discordance_thresholds(displacements)
    assert set(sweep.keys()) == {"10.0", "15.0", "20.0", "25.0", "30.0"}


def test_sweep_fractions_sum_to_one():
    """At every threshold, the three mode fractions must sum to 1."""
    rng = np.random.default_rng(0)
    displacements = rng.uniform(-100, 100, 200)
    sweep = sweep_discordance_thresholds(displacements)
    for dist in sweep.values():
        total = dist["NOTE_MITIGATES"] + dist["NOTE_AMPLIFIES"] + dist["CONCORDANT"]
        assert total == pytest.approx(1.0)


def test_sweep_narrower_threshold_flags_more_discordance():
    """A narrower threshold must classify at least as many patients as discordant."""
    displacements = np.array([-25.0, -18.0, -12.0, 5.0, 18.0, 25.0, 0.0])
    sweep = sweep_discordance_thresholds(displacements, thresholds_pp=[10.0, 30.0])
    discordant_10 = sweep["10.0"]["NOTE_MITIGATES"] + sweep["10.0"]["NOTE_AMPLIFIES"]
    discordant_30 = sweep["30.0"]["NOTE_MITIGATES"] + sweep["30.0"]["NOTE_AMPLIFIES"]
    assert discordant_10 >= discordant_30


def test_sweep_matches_compute_discordance_at_same_threshold():
    """The sweep's classification must agree with compute_discordance for a
    single patient at the same threshold — same rule, independently reached."""
    cohort = np.linspace(0.0, 1.0, 101)
    result = compute_discordance(0.20, 0.90, cohort, cohort, displacement_pp=20.0)
    sweep = sweep_discordance_thresholds(np.array([result["displacement"]]), [20.0])
    dist = sweep["20.0"]
    expected_mode = result["mode"]
    assert dist[expected_mode] == pytest.approx(1.0)


def test_sweep_empty_displacements_does_not_crash():
    """An empty input must return zeroed fractions, not raise (division by zero)."""
    sweep = sweep_discordance_thresholds(np.array([]), [20.0])
    assert sweep["20.0"] == {"NOTE_MITIGATES": 0.0, "NOTE_AMPLIFIES": 0.0, "CONCORDANT": 0.0}


# ── build_prompt ──────────────────────────────────────────────────────────────

def _discordance(mode: str = "CONCORDANT", r1: float = 50.0, r2: float = 50.0) -> dict:
    return {"r1": r1, "r2": r2, "displacement": r2 - r1, "mode": mode}


def _make_prompt(**kwargs) -> str:
    """Helper: build a test prompt with sensible defaults, overridable via kwargs."""
    defaults = {
        "stage1_score": 0.70,
        "stage1_threshold": 0.35,
        "stage2_score": 0.80,
        "discordance": _discordance(),
        "shap_feature_strings": ["creatinine (last): 3.2 (↑ risk, SHAP=+0.18)"],
        "note_text": "",
        "attention_sentences": [],
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


def test_build_prompt_discordance_mode_present():
    """The pre-computed discordance mode must appear in the prompt as context."""
    prompt = _make_prompt(discordance=_discordance("NOTE_AMPLIFIES", r1=20.0, r2=85.0))
    assert "NOTE_AMPLIFIES" in prompt
    assert "20" in prompt
    assert "85" in prompt


def test_build_prompt_shap_features_included():
    """All SHAP feature strings must appear in the prompt."""
    prompt = _make_prompt(shap_feature_strings=["creatinine: 3.2", "age: 78"])
    assert "creatinine: 3.2" in prompt
    assert "age: 78" in prompt


def test_build_prompt_note_text_included():
    """Raw note text must appear in the prompt — it is the auditor's primary evidence."""
    prompt = _make_prompt(note_text="Strong family support documented. Follow-up arranged.")
    assert "Strong family support documented" in prompt
    assert "(discharge note not available)" not in prompt


def test_build_prompt_no_note_fallback_message():
    """When note_text is empty the prompt must say so."""
    prompt = _make_prompt(note_text="")
    assert "not available" in prompt


def test_build_prompt_attention_is_auxiliary_not_primary():
    """Attention sentences must appear only as a labelled auxiliary hint,

    never replacing the raw note text (unlike the pre-2026-08-25 design).
    """
    prompt = _make_prompt(
        note_text="Full note body here.",
        attention_sentences=["A high-attention sentence."],
    )
    assert "Full note body here." in prompt
    assert "A high-attention sentence." in prompt
    assert "not a faithful explanation" in prompt


def test_build_prompt_decisions_listed():
    """Both decision options must be listed in the prompt."""
    prompt = _make_prompt()
    for decision in DECISIONS:
        assert decision in prompt


def test_build_prompt_domains_listed():
    """All clinical domains must be listed in the prompt."""
    prompt = _make_prompt()
    for domain in CLINICAL_DOMAINS:
        assert domain in prompt


def test_build_prompt_no_stage2_confirmed_narration():
    """The prompt must not ask the model to narrate a Stage 2 confirm/reject verdict."""
    prompt = _make_prompt()
    assert "CONFIRMED" not in prompt
    assert "REJECTED" not in prompt


# ── _parse_response ───────────────────────────────────────────────────────────

def _good_json(
    decision: str = "uphold",
    domain: str = "care_coordination",
    justification: str = "Note confirms the structured risk.",
) -> str:
    """Return a well-formed phi4-mini JSON response string."""
    return json.dumps({
        "decision": decision,
        "primary_clinical_domain": domain,
        "clinical_justification": justification,
    })


def test_parse_valid_uphold():
    """Valid uphold response must parse without failures."""
    result = _parse_response(_good_json("uphold", "care_coordination"))
    assert result["decision"] == "uphold"
    assert result["primary_clinical_domain"] == "care_coordination"
    assert result["annotation_failed"] is False


def test_parse_valid_override():
    """Valid override response must parse without failures."""
    result = _parse_response(_good_json("override", "social_support"))
    assert result["decision"] == "override"
    assert result["annotation_failed"] is False


def test_parse_bad_json_sets_annotation_failed():
    """Unparseable output must set annotation_failed=True with None fields."""
    result = _parse_response("not json at all")
    assert result["annotation_failed"] is True
    assert result["decision"] is None
    assert result["primary_clinical_domain"] is None


def test_parse_unknown_decision_sets_annotation_failed():
    """Unrecognised decision must set annotation_failed=True."""
    bad = json.dumps({
        "decision": "maybe",
        "primary_clinical_domain": "social_support",
        "clinical_justification": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["decision"] is None


def test_parse_unknown_domain_sets_annotation_failed():
    """Unrecognised primary_clinical_domain must set annotation_failed=True."""
    bad = json.dumps({
        "decision": "uphold",
        "primary_clinical_domain": "unknown_domain",
        "clinical_justification": "whatever",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["primary_clinical_domain"] is None


def test_parse_json_embedded_in_text():
    """JSON embedded in prose (as phi4-mini sometimes produces) must parse correctly."""
    raw = (
        'Here is the analysis: {"decision": "override", '
        '"primary_clinical_domain": "frailty", '
        '"clinical_justification": "Falls history noted."}'
    )
    result = _parse_response(raw)
    assert result["annotation_failed"] is False
    assert result["decision"] == "override"
    assert result["primary_clinical_domain"] == "frailty"


def test_parse_justification_preserved():
    """The justification string from phi4-mini must be preserved verbatim."""
    justification = "Patient has robust social support reducing readmission risk."
    result = _parse_response(_good_json("override", "social_support", justification))
    assert result["clinical_justification"] == justification


def test_parse_failure_does_not_default_to_uphold():
    """Empty JSON must not silently default to a decision."""
    result = _parse_response("{}")
    assert result["annotation_failed"] is True
    assert result["decision"] is None
