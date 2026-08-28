"""Tests for Stage 3 prompt construction, discordance, and response parsing.

No Ollama needed — these test pure prompt-building and parsing logic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.stage3.explain import (
    AGGRAVATING_GROUNDS,
    DECISIONS,
    DISCORDANCE_MODES,
    MITIGATING_GROUNDS,
    PLANNED_RETURN_ANSWERS,
    _parse_response,
    build_prompt,
    compute_decision_rule,
    compute_discordance,
    is_note_truncated,
    sweep_discordance_thresholds,
    verify_quote,
)


# ── Constants ─────────────────────────────────────────────────────────────────

def test_decisions_tuple():
    """DECISIONS must contain exactly uphold/override/insufficient_evidence."""
    assert set(DECISIONS) == {"uphold", "override", "insufficient_evidence"}


def test_modes_tuple_nonempty():
    """DISCORDANCE_MODES must contain all three expected mode strings."""
    assert len(DISCORDANCE_MODES) == 3
    assert "CONCORDANT" in DISCORDANCE_MODES
    assert "NOTE_MITIGATES" in DISCORDANCE_MODES
    assert "NOTE_AMPLIFIES" in DISCORDANCE_MODES


def test_mitigating_grounds_nonempty():
    """MITIGATING_GROUNDS must contain the four documented grounds."""
    assert len(MITIGATING_GROUNDS) == 4
    assert "palliative_intent" in MITIGATING_GROUNDS
    assert "planned_return" in MITIGATING_GROUNDS


def test_aggravating_grounds_nonempty():
    """AGGRAVATING_GROUNDS must contain the six documented grounds."""
    assert len(AGGRAVATING_GROUNDS) == 6
    assert "lives_alone_no_support" in AGGRAVATING_GROUNDS
    assert "unstable_at_discharge" in AGGRAVATING_GROUNDS


def test_grounds_taxonomies_disjoint():
    """No ground name may appear in both the mitigating and aggravating lists."""
    assert set(MITIGATING_GROUNDS).isdisjoint(set(AGGRAVATING_GROUNDS))


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


# ── is_note_truncated ─────────────────────────────────────────────────────────

def test_is_note_truncated_false_for_short_note():
    """A short note must not be reported as truncated."""
    assert is_note_truncated("Short discharge note.") is False


def test_is_note_truncated_true_for_long_note():
    """A note exceeding the safety cap must be reported as truncated."""
    assert is_note_truncated("x" * 25_000) is True


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
    """Stage 2 score must appear in the prompt by default (not hidden)."""
    assert "0.800" in _make_prompt()


def test_build_prompt_hide_stage2_withholds_score_and_discordance():
    """hide_stage2=True must withhold Stage 2's score and the discordance section."""
    prompt = _make_prompt(hide_stage2=True)
    assert "0.800" not in prompt
    assert "withheld for this run" in prompt
    assert "QUANTITATIVE DISCORDANCE" not in prompt


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
    """All three decision options must be listed in the prompt."""
    prompt = _make_prompt()
    for decision in DECISIONS:
        assert decision in prompt


def test_build_prompt_mitigating_grounds_listed():
    """All mitigating grounds must be listed in the prompt."""
    prompt = _make_prompt()
    for ground in MITIGATING_GROUNDS:
        assert ground in prompt


def test_build_prompt_aggravating_grounds_listed():
    """All aggravating grounds must be listed in the prompt."""
    prompt = _make_prompt()
    for ground in AGGRAVATING_GROUNDS:
        assert ground in prompt


def test_build_prompt_planned_return_options_listed():
    """All planned_return answer options must be listed in the prompt."""
    prompt = _make_prompt()
    for option in PLANNED_RETURN_ANSWERS:
        assert option in prompt


def test_build_prompt_requests_grounds_with_quotes():
    """The prompt must explicitly request verbatim quotes for each ground."""
    prompt = _make_prompt()
    assert "mitigating_grounds" in prompt
    assert "aggravating_grounds" in prompt
    assert "verbatim" in prompt


def test_build_prompt_evidence_requested_before_decision():
    """Grounds/evidence must be requested before decision in the JSON field
    list — an autoregressive model conditions on what it has already
    written, so asking for decision first invites post-hoc rationalisation.
    """
    prompt = _make_prompt()
    idx_grounds = prompt.index('"mitigating_grounds"')
    idx_justification = prompt.index('"clinical_justification"')
    idx_decision = prompt.rindex('"decision"')
    assert idx_grounds < idx_decision
    assert idx_justification < idx_decision


def test_build_prompt_no_stage2_confirmed_narration():
    """The prompt must not ask the model to narrate a Stage 2 confirm/reject verdict."""
    prompt = _make_prompt()
    assert "CONFIRMED" not in prompt
    assert "REJECTED" not in prompt


# ── _parse_response ───────────────────────────────────────────────────────────

def _good_json(
    decision: str = "uphold",
    justification: str = "Note confirms the structured risk.",
    mitigating: list | None = None,
    aggravating: list | None = None,
    planned_return: str = "no",
) -> str:
    """Return a well-formed LLM JSON response string."""
    return json.dumps({
        "mitigating_grounds": mitigating if mitigating is not None else [],
        "aggravating_grounds": aggravating if aggravating is not None else [
            {"ground": "lives_alone_no_support", "quote": "Patient lives alone."}
        ],
        "planned_return": planned_return,
        "clinical_justification": justification,
        "decision": decision,
    })


def test_parse_valid_uphold():
    """Valid uphold response must parse without failures."""
    result = _parse_response(_good_json("uphold"))
    assert result["decision_model"] == "uphold"
    assert result["annotation_failed"] is False


def test_parse_valid_override():
    """Valid override response must parse without failures."""
    result = _parse_response(_good_json(
        "override",
        mitigating=[{"ground": "palliative_intent", "quote": "Comfort-focused care planned."}],
        aggravating=[],
    ))
    assert result["decision_model"] == "override"
    assert result["annotation_failed"] is False


def test_parse_bad_json_sets_annotation_failed():
    """Unparseable output must set annotation_failed=True with empty/None fields."""
    result = _parse_response("not json at all")
    assert result["annotation_failed"] is True
    assert result["decision_model"] is None
    assert result["mitigating_grounds"] == []


def test_parse_unknown_decision_sets_annotation_failed():
    """Unrecognised decision must set annotation_failed=True."""
    bad = json.dumps({
        "mitigating_grounds": [],
        "aggravating_grounds": [],
        "planned_return": "no",
        "clinical_justification": "whatever",
        "decision": "maybe",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["decision_model"] is None


def test_parse_unknown_ground_sets_annotation_failed():
    """A ground outside the fixed taxonomy must fail the whole response
    (don't let the model invent categories)."""
    bad = json.dumps({
        "mitigating_grounds": [{"ground": "made_up_ground", "quote": "some quote"}],
        "aggravating_grounds": [],
        "planned_return": "no",
        "clinical_justification": "whatever",
        "decision": "override",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True


def test_parse_ground_from_wrong_side_sets_annotation_failed():
    """A mitigating ground listed under aggravating_grounds (or vice versa)
    is not in that list's allowed taxonomy and must fail."""
    bad = json.dumps({
        "mitigating_grounds": [],
        "aggravating_grounds": [{"ground": "palliative_intent", "quote": "Hospice care planned."}],
        "planned_return": "no",
        "clinical_justification": "whatever",
        "decision": "uphold",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True


def test_parse_missing_quote_sets_annotation_failed():
    """A ground with a missing or empty quote must set annotation_failed=True.

    The quote is required, at the same enforcement level as decision, so it
    can't be silently omitted.
    """
    bad = json.dumps({
        "mitigating_grounds": [{"ground": "palliative_intent", "quote": ""}],
        "aggravating_grounds": [],
        "planned_return": "no",
        "clinical_justification": "whatever",
        "decision": "override",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True


def test_parse_unknown_planned_return_sets_annotation_failed():
    """planned_return outside the fixed taxonomy must set annotation_failed=True."""
    bad = json.dumps({
        "mitigating_grounds": [],
        "aggravating_grounds": [],
        "planned_return": "maybe",
        "clinical_justification": "whatever",
        "decision": "uphold",
    })
    result = _parse_response(bad)
    assert result["annotation_failed"] is True
    assert result["planned_return"] is None


def test_parse_valid_planned_return_yes():
    """planned_return='yes' must parse through cleanly."""
    result = _parse_response(_good_json(planned_return="yes"))
    assert result["planned_return"] == "yes"
    assert result["annotation_failed"] is False


def test_parse_empty_grounds_lists_are_valid():
    """A response with no grounds on either side must still parse cleanly."""
    result = _parse_response(_good_json(mitigating=[], aggravating=[]))
    assert result["annotation_failed"] is False
    assert result["mitigating_grounds"] == []
    assert result["aggravating_grounds"] == []


def test_parse_decision_rule_and_quote_verified_not_set_by_parse_response():
    """_parse_response alone must not set decision_rule/all_quotes_verified --
    it has no note_text. call_llm fills these in after parsing."""
    result = _parse_response(_good_json())
    assert result["decision_rule"] is None
    assert result["all_quotes_verified"] is None


def test_parse_json_embedded_in_text():
    """JSON embedded in prose (as some models sometimes produce) must parse correctly."""
    raw = (
        'Here is the analysis: {"mitigating_grounds": [], '
        '"aggravating_grounds": [{"ground": "cognitive_impairment", '
        '"quote": "Patient has documented delirium."}], '
        '"planned_return": "no", '
        '"clinical_justification": "Delirium noted.", '
        '"decision": "uphold"}'
    )
    result = _parse_response(raw)
    assert result["annotation_failed"] is False
    assert result["decision_model"] == "uphold"
    assert result["aggravating_grounds"][0]["ground"] == "cognitive_impairment"


def test_parse_justification_preserved():
    """The justification string from the LLM must be preserved verbatim."""
    justification = "Patient has robust social support reducing readmission risk."
    result = _parse_response(_good_json("override", justification, mitigating=[
        {"ground": "strong_discharge_support", "quote": "Family support confirmed at discharge."}
    ], aggravating=[]))
    assert result["clinical_justification"] == justification


def test_parse_failure_does_not_default_to_uphold():
    """Empty JSON must not silently default to a decision."""
    result = _parse_response("{}")
    assert result["annotation_failed"] is True
    assert result["decision_model"] is None


# ── compute_decision_rule ────────────────────────────────────────────────────

_LONG_NOTE = "Patient discharged home in stable condition. " * 10


def test_decision_rule_override_when_only_mitigating():
    """Mitigating grounds with no aggravating grounds must yield override."""
    mitigating = [{"ground": "palliative_intent", "quote": "Comfort care planned."}]
    assert compute_decision_rule(mitigating, [], _LONG_NOTE) == "override"


def test_decision_rule_uphold_when_any_aggravating_present():
    """Any aggravating ground must force uphold, even alongside mitigating grounds."""
    mitigating = [{"ground": "palliative_intent", "quote": "Comfort care planned."}]
    aggravating = [{"ground": "unstable_at_discharge", "quote": "Vitals still abnormal."}]
    assert compute_decision_rule(mitigating, aggravating, _LONG_NOTE) == "uphold"


def test_decision_rule_uphold_when_no_grounds():
    """No grounds on either side must default to uphold, given an informative note."""
    assert compute_decision_rule([], [], _LONG_NOTE) == "uphold"


def test_decision_rule_insufficient_evidence_for_short_note():
    """A note below the informativeness threshold must yield insufficient_evidence
    regardless of what grounds were (implausibly) extracted from it."""
    mitigating = [{"ground": "palliative_intent", "quote": "Comfort care."}]
    assert compute_decision_rule(mitigating, [], "Too short.") == "insufficient_evidence"
    assert compute_decision_rule([], [], "") == "insufficient_evidence"


# ── verify_quote ──────────────────────────────────────────────────────────────

def test_verify_quote_true_for_exact_substring():
    """An exact verbatim quote must verify as True."""
    note = "Patient discharged home. Strong family support documented. Follow-up arranged."
    assert verify_quote("Strong family support documented.", note) is True


def test_verify_quote_false_for_paraphrase():
    """A paraphrased (non-verbatim) quote must verify as False -- catches hallucination."""
    note = "Patient discharged home. Strong family support documented."
    assert verify_quote("The patient has good social support", note) is False


def test_verify_quote_false_for_empty_quote():
    """An empty quote must never verify as True."""
    note = "Some discharge note text."
    assert verify_quote("", note) is False
    assert verify_quote("   ", note) is False


def test_verify_quote_strips_whitespace():
    """Leading/trailing whitespace on the quote must not cause a false negative."""
    note = "Patient discharged home. Strong family support documented."
    assert verify_quote("  Strong family support documented.  ", note) is True
