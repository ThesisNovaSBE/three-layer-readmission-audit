"""Population-level discordance analysis: aggregate Stage 3 annotations.

Reads the per-patient Stage 3 discordance CSV and produces a structured
summary of:

- Distribution of discordance modes (CONCORDANT / NOTE_MITIGATES /
  NOTE_AMPLIFIES) across all Stage 1 flagged patients.
- Distribution of primary categories broken down by discordance mode and
  age group — this is the core empirical finding for the thesis.
- Counts and percentages for confirmed vs. rejected Stage 1 flags.

Output: ``models/stage3_discordance_analysis.json``

Usage::

    python -m src.stage3.categorize
    from src.stage3.categorize import aggregate_discordance
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import get_model_dir
from src.stage3.explain import DISCORDANCE_CATEGORIES, DISCORDANCE_MODES


# ── Private helpers ────────────────────────────────────────────────────────────

def _mode_counts(valid_df: pd.DataFrame) -> dict[str, int]:
    """Return discordance mode counts initialised to zero for all valid modes."""
    counts: dict[str, int] = {m: 0 for m in DISCORDANCE_MODES}
    if "discordance_mode" not in valid_df.columns:
        return counts
    vc = valid_df["discordance_mode"].value_counts()
    for mode in DISCORDANCE_MODES:
        counts[mode] = int(vc.get(mode, 0))
    return counts


def _cat_counts(valid_df: pd.DataFrame) -> dict[str, int]:
    """Return primary_category counts initialised to zero for all valid categories."""
    counts: dict[str, int] = {c: 0 for c in DISCORDANCE_CATEGORIES}
    if "primary_category" not in valid_df.columns:
        return counts
    vc = valid_df["primary_category"].value_counts()
    for cat in DISCORDANCE_CATEGORIES:
        counts[cat] = int(vc.get(cat, 0))
    return counts


def _cat_by_mode(valid_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return category value counts for each discordance mode."""
    if "discordance_mode" not in valid_df.columns:
        return {m: {} for m in DISCORDANCE_MODES}
    result: dict[str, dict[str, int]] = {}
    for mode in DISCORDANCE_MODES:
        sub = valid_df[valid_df["discordance_mode"] == mode]
        result[mode] = (
            {k: int(v) for k, v in sub["primary_category"].value_counts().items()}
            if len(sub) > 0 else {}
        )
    return result


def _cat_by_age(valid_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Return category value counts for each age band."""
    if "age_band" not in valid_df.columns or "primary_category" not in valid_df.columns:
        return {}
    result: dict[str, dict[str, int]] = {}
    for band, grp in valid_df.groupby("age_band"):
        result[str(band)] = {
            k: int(v) for k, v in grp["primary_category"].value_counts().items()
        }
    return result


def _mode_subset(valid_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return rows for a single discordance mode (empty DataFrame if column missing)."""
    if "discordance_mode" not in valid_df.columns:
        return pd.DataFrame()
    return valid_df[valid_df["discordance_mode"] == mode]


# ── Public API ─────────────────────────────────────────────────────────────────

def aggregate_discordance(df: pd.DataFrame) -> dict:
    """Compute population-level discordance statistics from annotated results.

    Args:
        df: DataFrame produced by run_stage3; must contain columns
            ``discordance_mode``, ``primary_category``, ``stage2_confirmed``,
            and optionally ``age_band``.

    Returns:
        Nested dict suitable for JSON serialisation.
    """
    n = len(df)
    if n == 0:
        return {"n_patients": 0}

    confirmed = int(df["stage2_confirmed"].sum()) if "stage2_confirmed" in df.columns else n
    rejected = n - confirmed

    # Exclude annotation failures from distribution analysis
    fail_col = "annotation_failed"
    if fail_col in df.columns:
        fail_mask = df[fail_col].fillna(True)
        n_failed = int(fail_mask.sum())
        valid_df = df[~fail_mask]
    else:
        n_failed = 0
        valid_df = df

    n_valid = len(valid_df)
    denom = max(n_valid, 1)
    failure_rate = round(n_failed / max(n, 1) * 100, 1)

    mode_cts = _mode_counts(valid_df)
    mode_pct = {k: round(v / denom * 100, 1) for k, v in mode_cts.items()}

    cat_cts = _cat_counts(valid_df)
    cat_pct = {k: round(v / denom * 100, 1) for k, v in cat_cts.items()}

    mitigated = _mode_subset(valid_df, "NOTE_MITIGATES")
    amplified = _mode_subset(valid_df, "NOTE_AMPLIFIES")
    mit_cats = mitigated["primary_category"].value_counts().to_dict() if len(mitigated) > 0 else {}
    amp_cats = amplified["primary_category"].value_counts().to_dict() if len(amplified) > 0 else {}

    return {
        "n_patients": n,
        "n_valid_annotations": n_valid,
        "n_annotation_failures": n_failed,
        "annotation_failure_rate_pct": failure_rate,
        "n_confirmed": confirmed,
        "n_rejected": rejected,
        "pct_confirmed": round(confirmed / n * 100, 1),
        "mode_distribution": {"counts": mode_cts, "percentages": mode_pct},
        "category_distribution": {"counts": cat_cts, "percentages": cat_pct},
        "category_by_discordance_mode": _cat_by_mode(valid_df),
        "category_by_age_group": _cat_by_age(valid_df),
        "note_mitigates": {
            "n": len(mitigated),
            "pct_of_rejected": round(len(mitigated) / max(rejected, 1) * 100, 1),
            "top_categories": {k: int(v) for k, v in list(mit_cats.items())[:5]},
        },
        "note_amplifies": {
            "n": len(amplified),
            "pct_of_confirmed": round(len(amplified) / max(confirmed, 1) * 100, 1),
            "top_categories": {k: int(v) for k, v in list(amp_cats.items())[:5]},
        },
    }


def run_categorize(df: pd.DataFrame | None = None) -> dict:
    """Load Stage 3 results from disk (or use provided df) and aggregate.

    Args:
        df: optional pre-loaded DataFrame; loaded from disk if None.

    Returns:
        Discordance analysis dict; also saved to
        ``models/stage3_discordance_analysis.json``.
    """
    model_dir = get_model_dir()
    in_path = model_dir / "stage3_discordance.csv"

    if df is None:
        if not in_path.exists():
            raise FileNotFoundError(
                f"Stage 3 results not found at {in_path}.  "
                "Run setup_stage3.py first."
            )
        df = pd.read_csv(in_path)

    analysis = aggregate_discordance(df)

    out_path = model_dir / "stage3_discordance_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"[stage3/categorize] Saved analysis -> {out_path}")
    return analysis


if __name__ == "__main__":
    run_categorize()
