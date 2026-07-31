"""Tests for Stage 1 metric functions (pure, no data required)."""

from __future__ import annotations

import numpy as np
import pytest

from src.model.metrics import (
    auprc,
    auroc,
    full_report,
    metrics_at_recall_points,
    operating_point,
    select_threshold_for_recall,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    return y, s


@pytest.fixture
def random_scores(binary_scores):
    return binary_scores


# ── AUROC / AUPRC ─────────────────────────────────────────────────────────────

def test_auroc_perfect(perfect_scores):
    y, s = perfect_scores
    assert auroc(y, s) == pytest.approx(1.0)


def test_auroc_above_chance(random_scores):
    y, s = random_scores
    assert auroc(y, s) > 0.5


def test_auprc_bounded(random_scores):
    y, s = random_scores
    val = auprc(y, s)
    assert val > 0.0
    assert val <= 1.0 + 1e-9


def test_auprc_perfect(perfect_scores):
    y, s = perfect_scores
    assert auprc(y, s) == pytest.approx(1.0)


# ── Threshold selection ────────────────────────────────────────────────────────

def test_threshold_achieves_recall(random_scores):
    y, s = random_scores
    target = 0.85
    thr = select_threshold_for_recall(y, s, target)
    y_pred = (s >= thr).astype(int)
    tp = ((y == 1) & (y_pred == 1)).sum()
    fn = ((y == 1) & (y_pred == 0)).sum()
    actual_recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert actual_recall >= target or thr == 0.0, (
        f"recall {actual_recall:.3f} < target {target} with threshold {thr}"
    )


def test_threshold_falls_back_when_impossible(perfect_scores):
    y, s = perfect_scores
    # Target recall of 1.0 is achievable but very demanding
    thr = select_threshold_for_recall(y, s, 1.0)
    assert 0.0 <= thr <= 1.0


# ── Operating point ───────────────────────────────────────────────────────────

def test_operating_point_structure(random_scores):
    y, s = random_scores
    op = operating_point(y, s, threshold=0.5)
    for key in ("threshold", "recall", "precision", "specificity", "f2",
                "tp", "fp", "tn", "fn"):
        assert key in op, f"missing key: {key}"


def test_operating_point_confusion_matrix_sums(random_scores):
    y, s = random_scores
    op = operating_point(y, s, threshold=0.5)
    assert op["tp"] + op["fp"] + op["tn"] + op["fn"] == len(y)


def test_operating_point_all_flagged(random_scores):
    y, s = random_scores
    op = operating_point(y, s, threshold=0.0)
    assert op["fn"] == 0  # threshold=0 flags everything → no false negatives
    assert op["recall"] == pytest.approx(1.0)


def test_operating_point_none_flagged(random_scores):
    y, s = random_scores
    op = operating_point(y, s, threshold=1.1)
    assert op["tp"] == 0
    assert op["fp"] == 0


# ── Recall trade-off ──────────────────────────────────────────────────────────

def test_recall_tradeoff_sorted(random_scores):
    y, s = random_scores
    points = metrics_at_recall_points(y, s, [0.80, 0.85, 0.90])
    thresholds = [p["threshold"] for p in points]
    # Higher recall target → lower threshold (flag more aggressively)
    assert thresholds[0] >= thresholds[1] >= thresholds[2] or thresholds == [0.0, 0.0, 0.0]


# ── Full report ───────────────────────────────────────────────────────────────

def test_full_report_keys(random_scores):
    y, s = random_scores
    report = full_report(y, s, threshold=0.5, recall_points=[0.85])
    for key in ("auprc", "auroc", "brier", "base_rate", "operating_point", "recall_tradeoff"):
        assert key in report


def test_full_report_base_rate(random_scores):
    y, s = random_scores
    report = full_report(y, s, threshold=0.5, recall_points=[])
    assert report["base_rate"] == pytest.approx(y.mean())
