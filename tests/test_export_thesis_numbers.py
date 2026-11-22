"""Tests for scripts/export_thesis_numbers.py: the generator runs, rounds with
ROUND_HALF_UP, and its key values match the source JSONs."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_thesis_numbers import Out, _r, auroc, build, metric
from src.config import get_model_dir

_ROOT = Path(__file__).resolve().parents[1]


def test_round_half_up_rule():
    """ROUND_HALF_UP: a trailing 5 rounds away from zero (the B16 rule)."""
    assert _r(-0.00545, 4) == "-0.0055"      # not -0.0054
    assert _r(0.00545, 4) == "0.0055"
    assert _r(0.43927, 3) == "0.439"
    assert _r(0.7215074, 4) == "0.7215"


def test_generator_runs_and_emits_rows_and_commands():
    """build() populates both the markdown rows and the LaTeX commands."""
    out = Out()
    build(out)
    assert len(out.md) > 50
    assert any("\\newcommand" in line for line in out.tex)
    assert any("THESIS_NUMBERS" not in line for line in out.md)  # trivially true; rows exist


def test_key_values_match_source_jsons():
    """Pipeline/control precision-recall in the output equal the JSON (rounded)."""
    pe = json.loads((get_model_dir() / "pipeline_evaluation.json").read_text())
    fc = pe["pipeline"]["full_cohort"]
    ctrl = pe["pipeline"]["control_arm_stage1_matched"]
    out = Out()
    build(out)
    md = "\n".join(out.md)
    # the generator writes "precision / recall" for the full pipeline and control
    assert f"{metric(fc['precision'])} / {metric(fc['recall'])}" in md
    assert f"{metric(ctrl['precision'])} / {metric(ctrl['recall'])}" in md
    xgb = json.loads((get_model_dir() / "xgboost_metrics.json").read_text())
    assert auroc(xgb["auroc"]) == "0.7215"


def test_generated_files_present_after_main():
    """Running the module writes both deliverables."""
    from scripts.export_thesis_numbers import main
    main()
    assert (_ROOT / "docs" / "THESIS_NUMBERS.md").exists()
    assert (_ROOT / "docs" / "results_numbers.tex").exists()
