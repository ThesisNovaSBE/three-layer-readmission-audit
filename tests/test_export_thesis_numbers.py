"""Tests for scripts/export_thesis_numbers.py: the generator runs, rounds with
ROUND_HALF_UP, signs differences, keeps the markdown tables intact, and its key
values match the source JSONs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_thesis_numbers import Out, _r, auroc, build, diff, metric
from src.config import get_model_dir

_ROOT = Path(__file__).resolve().parents[1]


def _md() -> str:
    out = Out()
    build(out)
    return "\n".join(out.md)


def test_round_half_up_rule():
    """ROUND_HALF_UP: a trailing 5 rounds away from zero (the B16 rule)."""
    assert _r(-0.00545, 4) == "-0.0055"      # not -0.0054
    assert _r(0.00545, 4) == "0.0055"
    assert _r(0.43927, 3) == "0.439"
    assert _r(0.7215074, 4) == "0.7215"


def test_differences_carry_a_sign():
    """Differences are written +0.0009 / -0.0040, never unsigned."""
    assert diff(0.0009316) == "+0.0009"
    assert diff(-0.0040202) == "-0.0040"
    assert diff(0.0) == "0.0000"


def test_every_table_row_has_three_cells():
    """No stray pipe (e.g. |SMD|) may break a markdown table row."""
    for line in _md().splitlines():
        if line.startswith("| ") and not line.startswith("|---"):
            assert line.count(" | ") == 2, line


def test_key_values_match_source_jsons():
    """Pipeline/control precision-recall in the output equal the JSON (rounded)."""
    pe = json.loads((get_model_dir() / "pipeline_evaluation.json").read_text())
    fc = pe["pipeline"]["full_cohort"]
    ctrl = pe["pipeline"]["control_arm_stage1_matched"]
    md = _md()
    assert metric(fc["precision"]) in md and metric(fc["recall"]) in md
    assert metric(ctrl["precision"]) in md and metric(ctrl["recall"]) in md
    xgb = json.loads((get_model_dir() / "xgboost_metrics.json").read_text())
    assert auroc(xgb["auroc"]) == "0.7215"


def test_population_counts_add_up():
    """Covered + excluded = test n; flagged with + without summary = flagged."""
    m = get_model_dir()
    sel = json.loads((m / "selection_bias_evaluation.json").read_text())
    pe = json.loads((m / "pipeline_evaluation.json").read_text())
    s2 = json.loads((m / "stage2_evaluation.json").read_text())
    op = json.loads((m / "xgboost_metrics.json").read_text())["operating_point"]
    assert sel["n_notes_covered"] + sel["n_notes_excluded"] == sel["n_test"]
    assert s2["overall"]["n"] + pe["stage2"]["n_flagged_no_note"] == op["tp"] + op["fp"]


def test_main_writes_only_the_markdown_file():
    """Running the module writes THESIS_NUMBERS.md; the LaTeX export is gone."""
    from scripts.export_thesis_numbers import main
    main()
    assert (_ROOT / "docs" / "THESIS_NUMBERS.md").exists()
    assert not (_ROOT / "docs" / "results_numbers.tex").exists()
