"""Tests for scripts/compare_batch_size.py (N11)."""

from __future__ import annotations

import json

import pandas as pd

import scripts.compare_batch_size as cbs


def _write(tmp_path, name, df):
    df.to_csv(tmp_path / name, index=False)


def test_compare_reports_decision_rule_and_grounds_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(cbs, "get_model_dir", lambda: tmp_path)
    main_df = pd.DataFrame({
        "hadm_id": [1, 2],
        "decision_model": ["uphold", "override"],
        "decision_rule": ["uphold", "override"],
        "mitigating_grounds": ['[{"ground": "palliative_intent", "quote": "q"}]', "[]"],
        "aggravating_grounds": ["[]", "[]"],
    })
    bs1 = main_df.copy()
    bs1.loc[bs1["hadm_id"] == 2, "decision_model"] = "uphold"  # one decision flips
    _write(tmp_path, "stage3_batch_results.csv", main_df)
    _write(tmp_path, "stage3_bs1_check.csv", bs1)

    cbs.main()

    report = json.loads((tmp_path / "batch_size_comparison.json").read_text())
    assert report["n"] == 2
    assert report["decision_model_agreement"] == 0.5   # hadm 2 differs
    assert report["decision_rule_agreement"] == 1.0    # rules unchanged
    assert report["grounds_agreement"] == 1.0          # grounds identical


def test_compare_skips_when_input_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cbs, "get_model_dir", lambda: tmp_path)
    cbs.main()  # neither CSV exists -- must not raise
    assert "skipping" in capsys.readouterr().out
    assert not (tmp_path / "batch_size_comparison.json").exists()
