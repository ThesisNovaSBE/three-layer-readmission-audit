"""Tests for the EDA's disclosure controls and helper mappings.

These guard the one property that matters most for a MIMIC EDA: nothing
row-level or small-cell can reach a committed file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import privacy as P
from src.eda.structured import icd_chapter, race_group

ROOT = Path(__file__).resolve().parents[1]


def test_safe_count_suppresses_small_cells_but_keeps_zero():
    """Safe count suppresses small cells but keeps zero."""
    assert P.safe_count(0) == 0
    assert P.safe_count(1) == P.SUPPRESSED
    assert P.safe_count(P.MIN_CELL - 1) == P.SUPPRESSED
    assert P.safe_count(P.MIN_CELL) == P.MIN_CELL


def test_safe_rate_suppresses_small_numerator_or_denominator():
    """Safe rate suppresses small numerator or denominator."""
    assert P.safe_rate(3, 5000) is None           # small numerator
    assert P.safe_rate(2, 5) is None              # small denominator
    assert P.safe_rate(0, 5000) == 0.0            # structural zero is fine
    assert P.safe_rate(50, 200) == 0.25


def test_safe_quantiles_never_reports_min_or_max():
    """Safe quantiles never reports min or max."""
    out = P.safe_quantiles(np.arange(1000))
    assert "min" not in out and "max" not in out
    assert out["p1"] > 0 and out["p99"] < 999


def test_safe_quantiles_small_sample_returns_counts_only():
    """Safe quantiles small sample returns counts only."""
    out = P.safe_quantiles([1, 2, 3])
    assert out["n"] == P.SUPPRESSED and "mean" not in out


def test_histogram_has_no_small_bins():
    """Histogram has no small bins."""
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.normal(0, 1, 5000), [50.0, 60.0]])  # two outliers
    h = P.safe_histogram(vals, bins=30)
    assert all(c == 0 or c >= P.MIN_CELL for c in h["counts"])
    assert h["edges"][-1] < 50                      # outliers clipped away


def test_value_counts_folds_small_categories_into_other():
    """Value counts folds small categories into other."""
    s = pd.Series(["A"] * 50 + ["B"] * 30 + ["rare1"] * 3 + ["rare2"] * 2)
    rows = P.safe_value_counts(s)
    cats = [r["category"] for r in rows]
    assert "rare1" not in cats and "Other" in cats
    other = next(r for r in rows if r["category"] == "Other")
    assert other["n"] == P.SUPPRESSED               # 5 < MIN_CELL


@pytest.mark.parametrize("bad", [
    {"subject_id": 1},
    {"rows": [{"hadm_id": 2}]},
    {"x": {"text": "note"}},
])
def test_identifier_guard_rejects_row_level_keys(bad):
    """Identifier guard rejects row level keys."""
    with pytest.raises(ValueError):
        P.assert_no_identifiers(bad)


def test_identifier_guard_rejects_raw_frames():
    """Identifier guard rejects raw frames."""
    with pytest.raises(TypeError):
        P.assert_no_identifiers({"df": pd.DataFrame({"a": [1]})})


def test_write_json_refuses_identifiers(tmp_path):
    """Write json refuses identifiers."""
    with pytest.raises(ValueError):
        P.write_json({"hadm_id": 1}, tmp_path / "x.json")
    assert not (tmp_path / "x.json").exists()


def test_committed_eda_outputs_contain_no_identifier_keys():
    """Committed eda outputs contain no identifier keys."""
    for path in (ROOT / "results" / "eda").glob("*.json"):
        P.assert_no_identifiers(json.loads(path.read_text()))


def test_notebooks_are_committed_without_outputs():
    """Notebook outputs could carry row-level MIMIC data (df.head(), note text)."""
    for nb_path in (ROOT / "notebooks").glob("*.ipynb"):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                assert not cell.get("outputs"), f"{nb_path.name} has stored outputs"
                assert cell.get("execution_count") is None, f"{nb_path.name} not cleared"


def test_icd_chapter_mapping():
    """Icd chapter mapping."""
    codes = pd.Series(["I214", "D649", "C349", "4280", "V5811", "E8889", "Z5111"])
    vers = pd.Series([10, 10, 10, 9, 9, 9, 10])
    assert icd_chapter(codes, vers).tolist() == [
        "Circulatory", "Blood", "Neoplasms", "Circulatory",
        "Health-services contact", "External causes", "Health-services contact"]


def test_race_group_collapses_variants():
    """Race group collapses variants."""
    s = pd.Series(["WHITE - RUSSIAN", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - CUBAN",
                   "ASIAN - CHINESE", "UNABLE TO OBTAIN", "AMERICAN INDIAN/ALASKA NATIVE", None])
    assert race_group(s).tolist() == ["WHITE", "BLACK", "HISPANIC/LATINO", "ASIAN",
                                      "UNKNOWN", "OTHER", "UNKNOWN"]
