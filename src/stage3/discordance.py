"""Quantitative Stage 1-vs-Stage 2 discordance -- computed in code, never
asked of the LLM (see src/stage3/explain.py's module docstring, item 1).

Extracted out of explain.py (2026-09-27) purely to keep that module under
pylint's line-count limit after the strong_discharge_support taxonomy
split added ~90 lines -- no behavioral change, same functions, same
callers (src/stage3/pipeline.py:compute_discordance,
src/stage3/batch.py:sweep_discordance_thresholds).
"""

from __future__ import annotations

import numpy as np

DISCORDANCE_MODES: tuple[str, ...] = (
    "CONCORDANT",      # Stage 1 and Stage 2 rank this patient similarly
    "NOTE_MITIGATES",  # Stage 2 ranks the patient markedly lower risk than Stage 1
    "NOTE_AMPLIFIES",  # Stage 2 ranks the patient markedly higher risk than Stage 1
)


def _percentile_rank(value: float, population: np.ndarray) -> float:
    """Return the percentile rank of ``value`` within ``population`` (0-100)."""
    population = np.asarray(population, dtype=float)
    if population.size == 0:
        return 50.0
    return float((population <= value).mean() * 100.0)


def compute_discordance(
    stage1_score: float,
    stage2_score: float,
    cohort_stage1_scores: np.ndarray,
    cohort_stage2_scores: np.ndarray,
    displacement_pp: float = 20.0,
) -> dict[str, float | str]:
    """Return percentile-rank displacement and discordance mode for one patient.

    Displacement is invariant to any residual, unequal miscalibration between
    Stage 1 (XGBoost) and Stage 2 (Clinical-Longformer) — two different model
    families are not guaranteed to share the same calibration error even after
    isotonic calibration, so a raw ``stage2_score - stage1_score`` difference
    is not a reliable measure of disagreement. Rank displacement only requires
    that each score is a meaningful risk ordering within its own cohort.

    Args:
        stage1_score:          this patient's Stage 1 probability.
        stage2_score:          this patient's Stage 2 probability.
        cohort_stage1_scores:  Stage 1 scores for the flagged+noted cohort.
        cohort_stage2_scores:  Stage 2 scores for the same cohort.
        displacement_pp:       |displacement| >= this (percentile points)
                                is classified as discordant.

    Returns:
        Dict with ``r1``, ``r2``, ``displacement``, ``mode``.
    """
    r1 = _percentile_rank(stage1_score, cohort_stage1_scores)
    r2 = _percentile_rank(stage2_score, cohort_stage2_scores)
    displacement = r2 - r1
    if displacement <= -displacement_pp:
        mode = "NOTE_MITIGATES"
    elif displacement >= displacement_pp:
        mode = "NOTE_AMPLIFIES"
    else:
        mode = "CONCORDANT"
    return {"r1": r1, "r2": r2, "displacement": displacement, "mode": mode}


def sweep_discordance_thresholds(
    displacements: np.ndarray, thresholds_pp: list[float] | None = None
) -> dict[str, dict[str, float]]:
    """Report the discordance mode distribution across a range of thresholds.

    ``stage3.discordance_displacement_pp`` (20, provisional) has never been
    validated empirically — this answers how sensitive the reported mode
    distribution is to that choice, per docs/ARCHITECTURE.md. Only the mode
    classification depends on the threshold; ``displacement`` values
    (already computed per-patient by :func:`compute_discordance`) don't need
    recomputing — pass the ``displacement`` column of a batch audit result.

    Args:
        displacements: array of ``r2 - r1`` values, one per audited patient.
        thresholds_pp: displacement-point thresholds to sweep. Defaults to
                       ``[10, 15, 20, 25, 30]``.

    Returns:
        Dict keyed by threshold (as a string) to a dict of
        mode -> fraction of patients classified into that mode.
    """
    thresholds_pp = thresholds_pp or [10.0, 15.0, 20.0, 25.0, 30.0]
    displacements = np.asarray(displacements, dtype=float)
    n = len(displacements)
    out: dict[str, dict[str, float]] = {}
    for thr in thresholds_pp:
        mitigates = int((displacements <= -thr).sum())
        amplifies = int((displacements >= thr).sum())
        concordant = n - mitigates - amplifies
        out[str(thr)] = {
            "NOTE_MITIGATES": mitigates / n if n else 0.0,
            "NOTE_AMPLIFIES": amplifies / n if n else 0.0,
            "CONCORDANT": concordant / n if n else 0.0,
        }
    return out
