"""Tests for scripts/merge_stage3_parts.py: target order, duplicate handling
(main-file precedence), completeness (missing admissions), and header checks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.merge_stage3_parts import merge_stage3_parts

_HEADER = ["hadm_id", "decision_model", "decision_rule"]


def _write_results(path, hadm_ids, tag, header=None):
    """Write a minimal Stage-3-shaped result CSV; decision_model carries the
    file tag so the kept row's origin is checkable."""
    header = header or _HEADER
    rows = {"hadm_id": hadm_ids}
    for col in header:
        if col == "hadm_id":
            continue
        rows[col] = [f"{tag}-{h}" for h in hadm_ids]
    pd.DataFrame(rows)[header].to_csv(path, index=False)


def _write_stage2(path, order):
    """Write a stage2_results.csv whose hadm_id column defines the target order."""
    pd.DataFrame({"hadm_id": order, "subject_id": list(range(len(order)))}).to_csv(
        path, index=False
    )


def test_merge_orders_by_target_order_and_backs_up_main(tmp_path):
    """Merged rows follow stage2_results.csv order (not numeric sort), and the
    unmerged main file is preserved as <stem>.premerge.csv."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 12], "main")
    _write_results(part, [11, 13], "part")
    order = [13, 10, 12, 11]  # deliberately unsorted
    _write_stage2(tmp_path / "stage2_results.csv", order)

    report = merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")

    merged = pd.read_csv(main)
    assert list(merged["hadm_id"]) == order
    assert report["n_dropped_duplicates"] == 0
    backup = pd.read_csv(tmp_path / "main.premerge.csv")
    assert list(backup["hadm_id"]) == [10, 12]  # original main content preserved


def test_merge_keeps_main_row_on_duplicate_and_counts_drops(tmp_path):
    """A hadm_id in both files keeps the MAIN file's row; drops are counted."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 11], "main")
    _write_results(part, [11, 12, 13], "part")  # 11 duplicates main
    _write_stage2(tmp_path / "stage2_results.csv", [10, 11, 12, 13])

    report = merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")

    merged = pd.read_csv(main).set_index("hadm_id")
    assert merged.loc[11, "decision_model"] == "main-11"  # main won
    assert merged.loc[12, "decision_model"] == "part-12"
    assert report["n_dropped_duplicates"] == 1


def test_merge_exits_on_missing_admission(tmp_path):
    """A target admission absent from every input file must fail the merge."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 11], "main")
    _write_results(part, [12], "part")  # 13 is missing entirely
    _write_stage2(tmp_path / "stage2_results.csv", [10, 11, 12, 13])

    with pytest.raises(SystemExit):
        merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")
    # main file left untouched (no partial write)
    assert list(pd.read_csv(main)["hadm_id"]) == [10, 11]


def test_merge_exits_on_extra_admission(tmp_path):
    """A hadm_id not in the target set (e.g. a wrong-run part) must fail."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 11], "main")
    _write_results(part, [12, 99], "part")  # 99 is not a target
    _write_stage2(tmp_path / "stage2_results.csv", [10, 11, 12])

    with pytest.raises(SystemExit):
        merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")


def test_merge_exits_on_header_mismatch(tmp_path):
    """A part whose header differs from the main file's must fail the merge."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 11], "main")
    _write_results(part, [12, 13], "part", header=["hadm_id", "decision_model"])  # missing col
    _write_stage2(tmp_path / "stage2_results.csv", [10, 11, 12, 13])

    with pytest.raises(SystemExit):
        merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")


def test_merge_raw_sidecars_dedup_and_order(tmp_path):
    """The .raw.jsonl sidecars merge the same way: main precedence on a
    duplicate hadm_id, emitted in target order, with a premerge backup."""
    main, part = tmp_path / "main.csv", tmp_path / "part.csv"
    _write_results(main, [10, 11], "main")
    _write_results(part, [11, 12, 13], "part")
    _write_stage2(tmp_path / "stage2_results.csv", [10, 11, 12, 13])
    (tmp_path / "main.raw.jsonl").write_text(
        '{"hadm_id": 10, "raw_response": "main-10"}\n'
        '{"hadm_id": 11, "raw_response": "main-11"}\n'
    )
    (tmp_path / "part.raw.jsonl").write_text(
        '{"hadm_id": 11, "raw_response": "part-11"}\n'
        '{"hadm_id": 12, "raw_response": "part-12"}\n'
        '{"hadm_id": 13, "raw_response": "part-13"}\n'
    )

    merge_stage3_parts(main, [part], tmp_path / "stage2_results.csv")

    import json  # noqa: PLC0415  pylint: disable=import-outside-toplevel
    lines = [json.loads(ln) for ln in (tmp_path / "main.raw.jsonl").read_text().splitlines()]
    assert [r["hadm_id"] for r in lines] == [10, 11, 12, 13]   # target order
    kept_11 = next(r for r in lines if r["hadm_id"] == 11)
    assert kept_11["raw_response"] == "main-11"                 # main won
    assert (tmp_path / "main.premerge.raw.jsonl").exists()
