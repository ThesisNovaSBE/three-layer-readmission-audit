"""Tests for scripts/acceptance_check_stage2.py gate logic (N9).

BLOCKING: any band |recall gap| > 0.10 or overall ECE > 0.10 -> exit 1.
REPORTED-only (never blocking): band |recall gap| <= 0.05, per-band
|mean prob - observed rate| <= 0.03, overall ECE <= 0.03.
"""

from __future__ import annotations

from scripts.acceptance_check_stage2 import _gate_flags


def test_gate_flags_blocks_on_large_recall_gap():
    report = {
        "bands": {
            "18-40": {"recall_gap": 0.02, "mean_calibrated_prob": 0.20,
                      "observed_readmission_rate": 0.19},
            "70+":   {"recall_gap": 0.15, "mean_calibrated_prob": 0.25,
                      "observed_readmission_rate": 0.20},
        },
        "overall": {"ece": 0.04},
    }
    flags = _gate_flags(report)
    assert flags["blocking"]["failed"] is True
    assert flags["blocking"]["bands_recall_gap_over_0.10"] == ["70+"]
    assert flags["blocking"]["overall_ece_over_0.10"] is False
    # reported-only flags reflect the tighter thresholds
    assert flags["reported"]["all_bands_recall_gap_within_0.05"] is False     # 70+ gap 0.15
    assert flags["reported"]["all_bands_calibration_in_large_within_0.03"] is False  # 70+ 0.05
    assert flags["reported"]["overall_ece_within_0.03"] is False              # 0.04 > 0.03


def test_gate_flags_blocks_on_high_overall_ece():
    report = {
        "bands": {"18-40": {"recall_gap": 0.01, "mean_calibrated_prob": 0.20,
                            "observed_readmission_rate": 0.20}},
        "overall": {"ece": 0.12},
    }
    flags = _gate_flags(report)
    assert flags["blocking"]["failed"] is True
    assert flags["blocking"]["overall_ece_over_0.10"] is True
    assert flags["blocking"]["bands_recall_gap_over_0.10"] == []


def test_gate_flags_passes_clean_report():
    report = {
        "bands": {"18-40": {"recall_gap": 0.02, "mean_calibrated_prob": 0.20,
                            "observed_readmission_rate": 0.19}},
        "overall": {"ece": 0.02},
    }
    flags = _gate_flags(report)
    assert flags["blocking"]["failed"] is False
    assert flags["reported"]["all_bands_recall_gap_within_0.05"] is True
    assert flags["reported"]["all_bands_calibration_in_large_within_0.03"] is True
    assert flags["reported"]["overall_ece_within_0.03"] is True
