"""Tests for scripts/check_results_no_identifiers.py."""

from __future__ import annotations

from scripts.check_results_no_identifiers import find_identifier_leaks


def test_clean_aggregates_pass():
    clean = {
        "n": 104242, "tp": 6989, "precision": 0.4311,
        "by_age_band": {"70+": {"n": 27350, "recall": 0.19}},
        "note": "Methodology note with no identifiers and no de-id markers.",
    }
    assert find_identifier_leaks(clean) == []


def test_hadm_subject_keys_flagged():
    leaks = find_identifier_leaks({"sample_hadm_ids": [], "subject_id": 1})
    assert any("identifier-like key" in leak for leak in leaks)
    assert len(leaks) >= 2


def test_eight_digit_id_value_flagged():
    # 99999999 is an obviously-synthetic value that still lands in the
    # [10_000_000, 99_999_999] range the guard treats as an identifier.
    leaks = find_identifier_leaks({"ids": [99999999]})
    assert any("MIMIC id range" in leak for leak in leaks)


def test_six_digit_counts_not_flagged():
    # cohort n (~104k) and tp/fp counts are <= 6 digits -- never confused for ids
    assert find_identifier_leaks({"n": 104242, "fp": 95447}) == []


def test_mimic_deid_marker_string_flagged():
    assert find_identifier_leaks({"x": "Patient seen on ___ by Dr. ___"})
    assert find_identifier_leaks({"x": "History: [**2110-1-1**] admission"})


def test_free_text_key_with_string_flagged():
    assert find_identifier_leaks({"clinical_justification": "comfort care documented"})


def test_bool_not_treated_as_identifier():
    assert find_identifier_leaks({"available": True, "flagged": False}) == []


def test_without_hadm_count_keys_not_flagged():
    # "*_without_hadm*" are aggregate count keys, not identifiers
    assert find_identifier_leaks({"discharge_notes_without_hadm": 123,
                                  "radiology_without_hadm_share": 0.4}) == []


def test_tokenizer_files_skipped(tmp_path):
    import json as _json  # noqa: PLC0415
    from scripts.check_results_no_identifiers import check_file
    # a vocab.json with BPE tokens that LOOK like id keys / de-id markers
    p = tmp_path / "vocab.json"
    p.write_text(_json.dumps({"Ġsubject": 1, "___": 2}))
    assert check_file(p) == []
    # but a non-tokenizer file with the same content is still flagged
    q = tmp_path / "results.json"
    q.write_text(_json.dumps({"Ġsubject": 1}))
    assert check_file(q)
