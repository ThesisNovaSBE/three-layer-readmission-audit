"""Tests for scripts/summarize_stage3_run.py -- synthetic data only."""

from __future__ import annotations

import json

import pandas as pd

from scripts.summarize_stage3_run import (
    count_oom_retries,
    schema_field_order,
    summarize_csv,
    summarize_raw,
)


def _synthetic_df() -> pd.DataFrame:
    """Five rows: 2 decided (one agreeing, one not), 1 insufficient, 1 prep
    failure, 1 invalid LLM answer."""
    return pd.DataFrame({
        "decision_model": ["uphold", "override", "insufficient_evidence", None, None],
        "decision_rule": ["uphold", "uphold", None, None, None],
        "all_quotes_verified": [True, False, None, None, None],
        "note_truncated": [False, True, False, False, False],
        "planned_return": ["no", "yes", "not_stated", None, None],
        "discordance_mode": ["CONCORDANT", "NOTE_MITIGATES", "CONCORDANT",
                             "PREP_FAILED", "NOTE_AMPLIFIES"],
        "annotation_failed": [False, False, False, True, True],
    })


def test_summarize_csv_decisions_and_agreement():
    s = summarize_csv(_synthetic_df())
    assert s["n_rows"] == 5
    assert s["n_decided"] == 2
    assert s["model_rule_agreement_among_decided"] == 0.5     # uphold==uphold, override!=uphold
    assert s["all_quotes_verified_share_among_decided"] == 0.5
    assert s["note_truncated_count"] == 1
    assert s["note_truncated_share"] == 0.2
    assert s["n_insufficient_evidence"] == 1
    assert s["n_prep_failed"] == 1
    assert s["n_invalid_llm_answers"] == 1                    # row 4: failed, not prep-failed
    assert s["decision_model_counts"]["uphold"] == 1
    assert s["planned_return_distribution"] == {
        "no": 1, "yes": 1, "not_stated": 1, "missing": 2
    }


def _raw_line(hadm, answer_keys_values):
    """A raw.jsonl line: hadm_id + raw_response (a JSON string with the given
    ordered keys)."""
    raw = json.dumps(dict(answer_keys_values))  # preserves order
    return json.dumps({"hadm_id": hadm, "raw_response": raw})


def test_summarize_raw_key_order_and_unexpected_keys(tmp_path):
    order = schema_field_order()
    assert order[0] == "mitigating_grounds" and order[-1] == "decision"
    raw_path = tmp_path / "out.raw.jsonl"
    raw_path.write_text("\n".join([
        _raw_line(1, [("mitigating_grounds", []), ("decision", "uphold")]),      # in order
        _raw_line(2, [("aggravating_grounds", []), ("mitigating_grounds", [])]),  # swapped
        json.dumps({"hadm_id": 3, "raw_response": "not valid json"}),            # unparseable
        _raw_line(4, [("mitigating_grounds", []), ("surprise", 1), ("decision", "x")]),  # extra key
    ]) + "\n")

    r = summarize_raw(raw_path, order)
    assert r["raw_sidecar_present"] is True
    assert r["n_raw_lines"] == 4
    assert r["n_raw_unparseable"] == 1
    assert r["n_keys_out_of_schema_order"] == 1
    assert r["out_of_place_field_counts"] == {"mitigating_grounds": 1}
    assert r["n_answers_with_unexpected_keys"] == 1


def test_summarize_raw_absent_sidecar(tmp_path):
    assert summarize_raw(tmp_path / "nope.raw.jsonl", schema_field_order()) == {
        "raw_sidecar_present": False
    }


def test_count_oom_retries_counts_only_oom_retried_batches(tmp_path):
    log = tmp_path / "stage3_batch_1.log"
    log.write_text(
        "[stage3/batch] [generate] chunk hadm_ids=[1, 2, 3] FAILED: CUDA out of memory. "
        "Tried to allocate 2.00 GiB -- retrying 3 patient(s) individually\n"
        "[stage3/batch] [generate] chunk hadm_ids=[4, 5] FAILED: some transient error "
        "-- retrying 2 patient(s) individually\n"
        "[stage3/batch] [1200/9899] ok=1200 failed=0 annotation_failed=0\n"
    )
    out = count_oom_retries(str(tmp_path / "stage3_batch_*.log"))
    assert out["log_files_scanned"] == 1
    assert out["oom_retried_batches"] == 1  # only the CUDA-OOM retry line, not the transient one
