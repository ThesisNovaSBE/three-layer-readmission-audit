"""Tests for the batch Stage 3 audit runner's control flow.

Mocks explain_patient and the loaders so these test resumability, per-row
error isolation, and CSV output shape — not real model inference.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from src.stage3.batch import (
    _OUTPUT_FIELDS,
    _assert_output_matches_current_inputs,
    check_self_agreement,
    run_batch_audit,
    run_blind_note_control,
    run_no_stage2_control,
    run_sensitivity_sweep,
)
from src.stage3._annotation import BatchGenerationError

# hadm_id -> (stage1_score, stage1_score_raw, stage2_score), mirroring the
# results_df fixture below.
_FIXTURE_SCORES = {
    10: (0.5, 0.45, 0.3), 11: (0.6, 0.55, 0.5),
    12: (0.4, 0.35, 0.2), 13: (0.7, 0.65, 0.6),
}


def _fake_result(hadm_id: int, *, fail: bool = False, decision: str = "override") -> SimpleNamespace:
    """Build a stand-in for ExplanationResult -- exposes both attribute access
    (as check_self_agreement uses) and .model_dump() (as run_batch_audit uses)."""
    # Match the results_df fixture per hadm_id so the stale-output guard (which
    # compares written stage2_score/stage1_score_raw against stage2_results.csv)
    # passes on a legitimate --resume. _FIXTURE_SCORES mirrors the fixture below.
    s1, s1_raw, s2 = _FIXTURE_SCORES.get(hadm_id, (0.6, 0.55, 0.4))
    data = {
        "hadm_id": hadm_id, "stage1_score": s1, "stage1_score_raw": s1_raw,
        "stage1_threshold": 0.35,
        "stage2_score": s2, "stage2_confirmed": True, "r1": 70.0, "r2": 40.0,
        "displacement": -30.0, "discordance_mode": "NOTE_MITIGATES",
        "top_shap_features": [] if fail else ["creatinine (last): 3.2 (up risk, SHAP=+0.18)"],
        "attention_sentences": [] if fail else ["Comfort care planned."],
        "mitigating_grounds": [] if fail else [
            {"ground": "palliative_intent", "quote": "Comfort care planned.",
             "quote_verified": True}
        ],
        "aggravating_grounds": [],
        "all_quotes_verified": None if fail else True,
        "planned_return": None if fail else "no",
        "clinical_justification": "" if fail else "Note documents comfort-focused care.",
        "decision_model": None if fail else decision,
        "decision_rule": None if fail else decision,
        "note_truncated": False,
        "model_name": "phi4-mini",
        "annotation_failed": fail,
    }
    result = SimpleNamespace(**data)
    result.model_dump = lambda: dict(data)
    return result


@pytest.fixture
def results_df():  # pylint: disable=missing-function-docstring
    return pd.DataFrame({
        "hadm_id": [10, 11, 12, 13],
        "subject_id": [1, 2, 3, 4],
        "readmission_30d": [0, 1, 0, 1],
        "stage1_score": [0.5, 0.6, 0.4, 0.7],
        "stage1_score_raw": [0.45, 0.55, 0.35, 0.65],
        "stage1_threshold": [0.35] * 4,
        "stage2_score": [0.3, 0.5, 0.2, 0.6],
        "stage2_confirmed": [0, 1, 0, 1],
    })


def _patched(results_df, explain_side_effect):  # pylint: disable=redefined-outer-name
    """Context manager patching batch.py's loaders and explain_patient --
    for the validation-control / self-agreement tests below, which still
    call explain_patient() directly (unbatched, small samples only)."""
    return (
        patch("src.stage3.batch._load_artifact", return_value={}),
        patch("src.stage3.batch._load_results", return_value=results_df),
        patch("src.stage3.batch.load_feature_matrix", return_value=pd.DataFrame()),
        patch("src.stage3.batch.explain_patient", side_effect=explain_side_effect),
        patch("src.stage3.batch._preload_notes", return_value={}),
    )


def _cfg(batch_size: int = 10) -> SimpleNamespace:
    """cfg stand-in exposing what run_batch_audit reads directly
    (generation_batch_size, model_name for prep-failure rows) --
    everything else flows through mocked _prepare_patient/call_llm_batch/
    _assemble_result instead of real config."""
    return SimpleNamespace(
        stage3=SimpleNamespace(generation_batch_size=batch_size, model_name="fake-model")
    )


def _fake_prepared(hadm_id: int) -> SimpleNamespace:
    """Minimal stand-in for pipeline._PreparedPatient -- only exposes what
    run_batch_audit reads before calling call_llm_batch (.prompt,
    .prompt_note_text), plus .hadm_id as a test-only convenience so the
    mocked _assemble_result can build the right _fake_result without
    parsing it back out of the prompt string."""
    return SimpleNamespace(
        prompt=f"prompt-{hadm_id}", prompt_note_text=f"note-{hadm_id}", hadm_id=hadm_id,
    )


def _patched_batch(  # pylint: disable=redefined-outer-name
    results_df, prepare_side_effect, call_llm_batch_side_effect, assemble_side_effect,
):
    """Context manager patching batch.py's loaders and the
    prepare -> call_llm_batch -> assemble pipeline run_batch_audit uses
    (2026-09-15, batched-generation rewrite)."""
    return (
        patch("src.stage3.batch._load_artifact", return_value={}),
        patch("src.stage3.batch._load_results", return_value=results_df),
        patch("src.stage3.batch.load_feature_matrix", return_value=pd.DataFrame()),
        patch("src.stage3.batch._preload_notes", return_value={}),
        patch("src.stage3.batch._prepare_patient", side_effect=prepare_side_effect),
        patch("src.stage3.batch.call_llm_batch", side_effect=call_llm_batch_side_effect),
        patch("src.stage3.batch._assemble_result", side_effect=assemble_side_effect),
    )


def _ok_call_llm_batch(prompts, *_a, **_kw):
    """Default call_llm_batch stand-in: every prompt succeeds."""
    return [{"annotation_failed": False} for _ in prompts]


def _ok_assemble(prepared, annotation):
    """Default _assemble_result stand-in: build a _fake_result keyed off
    the prepared stand-in's hadm_id."""
    return _fake_result(prepared.hadm_id, fail=annotation["annotation_failed"])


def test_writes_one_row_per_admission(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A clean run must write exactly one row per target admission."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    assert set(written["hadm_id"]) == {10, 11, 12, 13}
    assert list(written.columns) == _OUTPUT_FIELDS


def test_grounds_columns_are_json_serialised(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """mitigating_grounds/aggravating_grounds must round-trip through the CSV as JSON."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out, limit=1)

    written = pd.read_csv(out)
    grounds = json.loads(written.iloc[0]["mitigating_grounds"])
    assert grounds[0]["ground"] == "palliative_intent"
    shap_features = json.loads(written.iloc[0]["top_shap_features"])
    assert shap_features == ["creatinine (last): 3.2 (up risk, SHAP=+0.18)"]
    attention = json.loads(written.iloc[0]["attention_sentences"])
    assert attention == ["Comfort care planned."]


def test_prep_failure_does_not_stop_the_batch_and_still_writes_a_row(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """A raised exception preparing one admission must not prevent the rest
    from being written, AND the failed admission itself must still get
    exactly one row (annotation_failed=True) -- not be silently dropped.
    Before this fix, a prep failure was only printed to the job log and
    never written to the CSV, which meant it would be retried on every
    single --resume forever across a multi-day run's chained job
    resubmissions, with no single place to reconcile the final row count
    against the target admission count."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        if hadm_id == 11:
            raise RuntimeError("simulated note-loading failure")
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    assert set(written["hadm_id"]) == {10, 11, 12, 13}
    row = written[written["hadm_id"] == 11].iloc[0]
    assert bool(row["annotation_failed"]) is True
    assert "PREP FAILED" in row["clinical_justification"]
    # basic Stage 1/2 fields must still come from results_df, not be blanked
    assert row["stage1_score"] == pytest.approx(0.6)


def test_missing_note_becomes_prep_failed_row(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """Fix 2: an admission with no note (absent from notes_lookup) must be
    recorded by the REAL _prepare_patient as a PREP_FAILED row, never audited.
    Only the loaders are mocked; _prepare_patient and _prep_failure_row run
    for real, so the empty-note guard actually fires."""
    out = tmp_path / "out.csv"
    with patch("src.stage3.batch._load_artifact",
               return_value={"estimator": None, "feature_cols": []}), \
         patch("src.stage3.batch._load_results", return_value=results_df), \
         patch("src.stage3.batch.load_feature_matrix", return_value=pd.DataFrame()), \
         patch("src.stage3.batch._preload_notes", return_value={}):  # no notes for anyone
        run_batch_audit(cfg=_cfg(), out_path=out, limit=1)

    written = pd.read_csv(out)
    assert len(written) == 1
    row = written.iloc[0]
    assert row["discordance_mode"] == "PREP_FAILED"
    assert bool(row["annotation_failed"]) is True
    assert "no discharge note" in row["clinical_justification"]


# ── Fix 4: stale-output guard ─────────────────────────────────────────────────

def _full_output_row(hadm_id, stage2_score, stage1_score_raw):
    """A complete _OUTPUT_FIELDS row with the two guard-checked columns set."""
    row = {c: 0 for c in _OUTPUT_FIELDS}
    row.update({"hadm_id": hadm_id, "stage2_score": stage2_score,
                "stage1_score_raw": stage1_score_raw})
    return row


def test_stale_output_guard_noop_when_absent(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """No existing file -> the guard does nothing (fresh run / arm test)."""
    _assert_output_matches_current_inputs(tmp_path / "missing.csv", results_df)


def test_stale_output_guard_passes_when_matching(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """An existing file whose rows match the current results must pass."""
    out = tmp_path / "out.csv"
    pd.DataFrame([_full_output_row(10, 0.3, 0.45)])[_OUTPUT_FIELDS].to_csv(out, index=False)
    _assert_output_matches_current_inputs(out, results_df)  # hadm 10 fixture: s2=0.3, raw=0.45


def test_stale_output_guard_rejects_header_mismatch(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A file missing a current column (e.g. stage1_score_raw) must SystemExit."""
    out = tmp_path / "out.csv"
    cols = [c for c in _OUTPUT_FIELDS if c != "stage1_score_raw"]
    pd.DataFrame(columns=cols).to_csv(out, index=False)
    with pytest.raises(SystemExit, match="header"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_rejects_value_drift(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """Correct header but a stage2_score that drifted from the current
    stage2_results.csv (regenerated since) must SystemExit."""
    out = tmp_path / "out.csv"
    # hadm 10's current stage2_score is 0.3; write 0.999 to simulate a stale run.
    pd.DataFrame([_full_output_row(10, 0.999, 0.45)])[_OUTPUT_FIELDS].to_csv(out, index=False)
    with pytest.raises(SystemExit, match="stage2_score"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_rejects_unknown_hadm_id(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """An existing row whose hadm_id isn't in the current stage2_results.csv
    must stop with the clear move-aside message, not a bare KeyError."""
    out = tmp_path / "out.csv"
    pd.DataFrame([_full_output_row(9999, 0.3, 0.45)])[_OUTPUT_FIELDS].to_csv(out, index=False)
    with pytest.raises(SystemExit, match="absent from the current"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_allows_header_only(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A header-only file (job died during model load) must pass -- resuming
    into it is correct, not a crash."""
    out = tmp_path / "out.csv"
    pd.DataFrame(columns=_OUTPUT_FIELDS).to_csv(out, index=False)
    _assert_output_matches_current_inputs(out, results_df)  # no raise


def test_header_only_file_resumes(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """run_batch_audit(resume=True) into a header-only file must proceed and
    write all rows, not crash on the empty body."""
    out = tmp_path / "out.csv"
    pd.DataFrame(columns=_OUTPUT_FIELDS).to_csv(out, index=False)

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out, resume=True)

    assert set(pd.read_csv(out)["hadm_id"]) == {10, 11, 12, 13}


def test_stale_output_guard_rejects_nan_stage2_score(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A stored NaN stage2_score against a real current value must SystemExit
    (allclose with equal_nan=True still rejects NaN-vs-number)."""
    out = tmp_path / "out.csv"
    pd.DataFrame([_full_output_row(10, float("nan"), 0.45)])[_OUTPUT_FIELDS].to_csv(out, index=False)
    with pytest.raises(SystemExit, match="stage2_score"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_rejects_cutoff_last_row(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A row with no annotation_failed value (last line cut off by a kill)
    must SystemExit telling the user to delete that line."""
    out = tmp_path / "out.csv"
    row = _full_output_row(10, 0.3, 0.45)
    row["annotation_failed"] = float("nan")
    pd.DataFrame([row])[_OUTPUT_FIELDS].to_csv(out, index=False)
    with pytest.raises(SystemExit, match="cut off"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_rejects_unparseable_last_line(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A kill inside a quoted JSON field leaves an unterminated record that
    pandas can't parse -- caught and turned into the same cut-off message."""
    out = tmp_path / "out.csv"
    header = ",".join(_OUTPUT_FIELDS)
    # Second field opens a quote that never closes before EOF.
    out.write_text(f"{header}\n10,\"[{{\"ground\":\n")
    with pytest.raises(SystemExit, match="cut off"):
        _assert_output_matches_current_inputs(out, results_df)


def test_stale_output_guard_rejects_duplicate_results(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """Duplicate hadm_ids in the current stage2_results.csv must SystemExit
    (set_index would otherwise misalign the comparison)."""
    out = tmp_path / "out.csv"
    pd.DataFrame([_full_output_row(10, 0.3, 0.45)])[_OUTPUT_FIELDS].to_csv(out, index=False)
    dup = pd.concat([results_df, results_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate hadm_ids"):
        _assert_output_matches_current_inputs(out, dup)


def test_resume_does_not_retry_a_prep_failure(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """Once an admission has a prep-failure row, --resume must not attempt
    to prepare it again -- the entire point of writing the row in the first
    place. A deterministic failure (missing note, malformed feature row)
    would otherwise fail identically on every future resubmission forever."""
    out = tmp_path / "out.csv"
    calls: list[int] = []

    def prepare_se(hadm_id, *_a, **_kw):
        calls.append(hadm_id)
        if hadm_id == 11:
            raise RuntimeError("permanent, deterministic failure")
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    calls.clear()
    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out, resume=True)

    assert 11 not in calls


def test_chunk_generation_failure_does_not_stop_the_batch(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A whole-chunk call_llm_batch failure must not prevent later chunks
    from being processed -- batch_size=2 over 4 patients makes two chunks;
    only the first (hadm_ids 10, 11) should be lost."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    def call_llm_batch_se(prompts, *_a, **_kw):
        if "prompt-10" in prompts:
            raise RuntimeError("simulated generation OOM")
        return [{"annotation_failed": False} for _ in prompts]

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, call_llm_batch_se, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(batch_size=2), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 2
    assert set(written["hadm_id"]) == {12, 13}


def test_batch_generation_error_retries_individually_and_recovers(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """A BatchGenerationError on the whole chunk must trigger a per-patient
    retry; if the batch-of-one succeeds, no patient is lost."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    def call_llm_batch_se(prompts, *_a, **_kw):
        if len(prompts) > 1:  # the real chunk -- simulate a batch-level OOM
            raise BatchGenerationError("simulated chunk OOM")
        return [{"annotation_failed": False}]  # batch-of-one retry fits

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, call_llm_batch_se, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(batch_size=2), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    assert set(written["hadm_id"]) == {10, 11, 12, 13}
    assert not written["annotation_failed"].any()


def test_batch_generation_error_persistent_writes_failed_row_not_lost(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """If the individual retry also raises, that patient must still get a
    written annotation_failed row (not silently dropped), while the others
    in the chunk recover."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    def call_llm_batch_se(prompts, *_a, **_kw):
        # hadm_id 10 fails at both batch and individual-retry level; all
        # others succeed (batch of one, or a chunk without prompt-10).
        if "prompt-10" in prompts:
            raise BatchGenerationError("persistent OOM for 10")
        return [{"annotation_failed": False} for _ in prompts]

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, call_llm_batch_se, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(batch_size=2), out_path=out)

    written = pd.read_csv(out).set_index("hadm_id")
    assert set(written.index) == {10, 11, 12, 13}  # none lost
    assert bool(written.loc[10, "annotation_failed"]) is True
    assert not written.loc[[11, 12, 13], "annotation_failed"].any()


# ── Per-patient alignment through the real _process_chunk ─────────────────────

_ALIGN_SCORE = {10: 0.11, 11: 0.22, 12: 0.33, 13: 0.44}


def _align_prepared(hadm_id: int):
    """Prepared stand-in whose prompt AND note each embed this patient's own
    hadm_id, and whose patient dict carries a distinct stage2_score -- so a
    prompt/note desync or a wrong-order zip in _process_chunk becomes
    detectable in the written row."""
    return SimpleNamespace(
        prompt=f"AUDIT hadm_id={hadm_id}",
        prompt_note_text=f"note-of-{hadm_id}",
        patient={"hadm_id": hadm_id, "stage2_score": _ALIGN_SCORE[hadm_id]},
        hadm_id=hadm_id,
    )


def _align_assemble(prepared, annotation):
    """Build a full output row combining the SCORE side (hadm_id + stage2_score
    from the prepared patient) with the PROMPT/NOTE side (the ids the stub LLM
    echoed back from what it was actually handed). If _process_chunk paired the
    wrong annotation with this prepared patient, these disagree."""
    data = {k: "" for k in _OUTPUT_FIELDS}
    data.update({
        "hadm_id": prepared.patient["hadm_id"],
        "stage2_score": prepared.patient["stage2_score"],
        "mitigating_grounds": [], "aggravating_grounds": [],
        "top_shap_features": [], "attention_sentences": [],
        "clinical_justification":
            f"prompt={annotation['prompt_hadm_id']};note={annotation['note_hadm_id']}",
        "annotation_failed": annotation["annotation_failed"],
    })
    return SimpleNamespace(model_dump=lambda d=data: dict(d))


def _echo_call_llm_batch(prompts, *_a, note_texts=None, **_kw):
    """Stub LLM: echo, per item, the hadm_id embedded in the PROMPT and the
    hadm_id embedded in the NOTE this call was handed -- never from list
    position -- so downstream can prove prompt, note and output stay paired."""
    out = []
    for prompt, note in zip(prompts, note_texts):
        out.append({
            "annotation_failed": False,
            "prompt_hadm_id": int(prompt.split("hadm_id=")[1]),
            "note_hadm_id": int(note.split("note-of-")[1]),
        })
    return out


def _assert_rows_self_consistent(out_path):
    """Every row's hadm_id, stage2_score and echoed prompt/note ids must all
    point at the same patient."""
    written = pd.read_csv(out_path)
    assert set(written["hadm_id"]) == {10, 11, 12, 13}
    for _, row in written.iterrows():
        hadm_id = int(row["hadm_id"])
        prompt_part, note_part = row["clinical_justification"].split(";")
        assert prompt_part == f"prompt={hadm_id}", f"prompt not aligned for {hadm_id}: {row}"
        assert note_part == f"note={hadm_id}", f"note not aligned for {hadm_id}: {row}"
        assert row["stage2_score"] == _ALIGN_SCORE[hadm_id]


def test_process_chunk_keeps_every_row_aligned_to_its_patient(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """Through the real _process_chunk (chunk of 4), each written row's
    hadm_id, stage2_score and quote-verification note must belong to the
    same patient -- a batched-generation misalignment would break this."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _align_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _echo_call_llm_batch, _align_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(batch_size=4), out_path=out)  # one chunk of 4

    _assert_rows_self_consistent(out)


def test_individual_retry_preserves_per_patient_alignment(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """The individual-retry path must preserve the same per-patient pairing:
    force a whole-chunk BatchGenerationError so every patient is retried in a
    batch of one, then assert each row is still self-consistent."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _align_prepared(hadm_id)

    def flaky_echo(prompts, *_a, note_texts=None, **_kw):
        if len(prompts) > 1:  # the whole chunk -- force the retry path
            raise BatchGenerationError("simulated chunk OOM")
        return _echo_call_llm_batch(prompts, note_texts=note_texts)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, flaky_echo, _align_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(batch_size=4), out_path=out)

    _assert_rows_self_consistent(out)


def test_resume_skips_already_written_admissions(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """--resume must not re-call _prepare_patient for hadm_ids already in the CSV."""
    out = tmp_path / "out.csv"
    calls: list[int] = []

    def prepare_se(hadm_id, *_a, **_kw):
        calls.append(hadm_id)
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    assert sorted(calls) == [10, 11, 12, 13]
    calls.clear()

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out, resume=True)

    assert not calls
    written = pd.read_csv(out)
    assert len(written) == 4


def test_setup_batch_skips_preload_when_nothing_pending(
    tmp_path, results_df  # pylint: disable=redefined-outer-name
):
    """When --resume finds everything already done, _preload_notes must
    never be called at all -- confirmed real 2026-09-27: the underlying
    load_notes() raises ValueError ("No discharge notes found after
    filtering") on an empty hadm_ids set instead of returning nothing, so
    a --resume run landing on an already-fully-done batch would otherwise
    crash instead of exiting cleanly. A plausible way for one of the full
    run's several job segments to end."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    def preload_notes_se(*_a, **_kw):
        raise AssertionError("_preload_notes must not be called when pending is empty")

    with patch("src.stage3.batch._load_artifact", return_value={}), \
            patch("src.stage3.batch._load_results", return_value=results_df), \
            patch("src.stage3.batch.load_feature_matrix", return_value=pd.DataFrame()), \
            patch("src.stage3.batch._preload_notes", side_effect=preload_notes_se), \
            patch("src.stage3.batch._prepare_patient", side_effect=prepare_se), \
            patch("src.stage3.batch.call_llm_batch", side_effect=_ok_call_llm_batch), \
            patch("src.stage3.batch._assemble_result", side_effect=_ok_assemble):
        run_batch_audit(cfg=_cfg(), out_path=out, resume=True)

    written = pd.read_csv(out)
    assert len(written) == 4


def test_limit_restricts_target_count(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """--limit must restrict how many admissions are audited."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, _ok_call_llm_batch, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out, limit=2)

    written = pd.read_csv(out)
    assert len(written) == 2


def test_annotation_failed_rows_still_written(tmp_path, results_df):  # pylint: disable=redefined-outer-name
    """A parse failure (annotation_failed=True) must still produce a row, not be dropped."""
    out = tmp_path / "out.csv"

    def prepare_se(hadm_id, *_a, **_kw):
        return _fake_prepared(hadm_id)

    def call_llm_batch_se(prompts, *_a, **_kw):
        return [{"annotation_failed": p == "prompt-12"} for p in prompts]

    p1, p2, p3, p4, p5, p6, p7 = _patched_batch(
        results_df, prepare_se, call_llm_batch_se, _ok_assemble
    )
    with p1, p2, p3, p4, p5, p6, p7:
        run_batch_audit(cfg=_cfg(), out_path=out)

    written = pd.read_csv(out)
    assert len(written) == 4
    row = written[written["hadm_id"] == 12].iloc[0]
    assert bool(row["annotation_failed"]) is True


# ── run_sensitivity_sweep ─────────────────────────────────────────────────────

def test_sensitivity_sweep_raises_without_batch_results(tmp_path):
    """Must fail clearly if run before the batch audit has ever produced output."""
    with pytest.raises(FileNotFoundError):
        run_sensitivity_sweep(model_dir=tmp_path)


def test_sensitivity_sweep_writes_expected_json(tmp_path):
    """Must read displacement from an existing batch CSV, EXCLUDE prep-failure
    rows (PREP_FAILED / NaN displacement), and write the sweep JSON."""
    batch_csv = tmp_path / "stage3_batch_results.csv"
    pd.DataFrame({
        "hadm_id": [1, 2, 3, 4, 5],
        "displacement": [-40.0, -5.0, 10.0, 35.0, float("nan")],
        "discordance_mode": ["NOTE_MITIGATES", "CONCORDANT", "CONCORDANT",
                             "NOTE_AMPLIFIES", "PREP_FAILED"],
    }).to_csv(batch_csv, index=False)

    sweep = run_sensitivity_sweep(model_dir=tmp_path)

    out_path = tmp_path / "discordance_sensitivity.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["n_patients"] == 4  # the PREP_FAILED row is excluded
    assert saved["n_excluded_prep_failed"] == 1
    assert saved["sweep"] == sweep


# ── run_blind_note_control / run_no_stage2_control ──────────────────────────

def test_blind_note_control_calls_explain_patient_with_suppress_note(results_df):  # pylint: disable=redefined-outer-name
    """run_blind_note_control must call explain_patient with suppress_note=True."""
    calls = []

    def side_effect(hadm_id, _cfg, **kwargs):
        calls.append(kwargs)
        return _fake_result(hadm_id)

    p1, p2, p3, p4, p5 = _patched(results_df, side_effect)
    with p1, p2, p3, p4, p5:
        out = run_blind_note_control(SimpleNamespace(), [10, 11])

    assert len(out) == 2
    assert all(kw.get("suppress_note") is True for kw in calls)
    assert all(kw.get("suppress_stage2") is False for kw in calls)


def test_no_stage2_control_calls_explain_patient_with_suppress_stage2(results_df):  # pylint: disable=redefined-outer-name
    """run_no_stage2_control must call explain_patient with suppress_stage2=True."""
    calls = []

    def side_effect(hadm_id, _cfg, **kwargs):
        calls.append(kwargs)
        return _fake_result(hadm_id)

    p1, p2, p3, p4, p5 = _patched(results_df, side_effect)
    with p1, p2, p3, p4, p5:
        out = run_no_stage2_control(SimpleNamespace(), [10])

    assert len(out) == 1
    assert calls[0]["suppress_stage2"] is True
    assert calls[0]["suppress_note"] is False


# ── check_self_agreement ──────────────────────────────────────────────────────

def test_self_agreement_reports_full_agreement_for_identical_output(results_df):  # pylint: disable=redefined-outer-name
    """Two identical runs (deterministic mock) must report 100% agreement."""
    def side_effect(hadm_id, *_a, **_kw):
        return _fake_result(hadm_id, decision="uphold")

    p1, p2, p3, p4, p5 = _patched(results_df, side_effect)
    with p1, p2, p3, p4, p5:
        report = check_self_agreement(SimpleNamespace(), [10, 11])

    assert report["n"] == 2
    assert report["decision_model_agreement"] == 1.0
    assert report["decision_rule_agreement"] == 1.0
    assert report["grounds_agreement"] == 1.0
    assert report["disagreements"] == []


def test_self_agreement_detects_a_mismatch(results_df):  # pylint: disable=redefined-outer-name
    """A patient whose two runs disagree must be reported, not averaged away."""
    call_count = {"n": 0}

    def side_effect(hadm_id, *_a, **_kw):
        call_count["n"] += 1
        # Every other call for hadm_id 11 returns a different decision.
        decision = "uphold" if call_count["n"] % 2 == 1 else "override"
        return _fake_result(hadm_id, decision=decision if hadm_id == 11 else "uphold")

    p1, p2, p3, p4, p5 = _patched(results_df, side_effect)
    with p1, p2, p3, p4, p5:
        report = check_self_agreement(SimpleNamespace(), [10, 11])

    assert 11 in report["disagreements"]
    assert 10 not in report["disagreements"]
    assert report["decision_model_agreement"] == 0.5
