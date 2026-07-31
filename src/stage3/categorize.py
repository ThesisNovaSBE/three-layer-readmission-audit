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
from pathlib import Path

import pandas as pd

from src.config import get_model_dir
from src.stage3.explain import DISCORDANCE_CATEGORIES, DISCORDANCE_MODES


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
        n_failed = int(df[fail_col].fillna(True).sum())
        valid_df = df[~df[fail_col].fillna(True)]
    else:
        n_failed = 0
        valid_df = df

    n_valid = len(valid_df)
    failure_rate = round(n_failed / max(n, 1) * 100, 1)

    # ── Mode distribution (valid annotations only) ────────────────────────────
    mode_counts: dict[str, int] = {m: 0 for m in DISCORDANCE_MODES}
    if "discordance_mode" in valid_df.columns:
        for mode in valid_df["discordance_mode"].dropna():
            if mode in mode_counts:
                mode_counts[mode] += 1

    denom = max(n_valid, 1)
    mode_pct = {k: round(v / denom * 100, 1) for k, v in mode_counts.items()}

    # ── Category distribution (overall, valid only) ───────────────────────────
    cat_counts: dict[str, int] = {c: 0 for c in DISCORDANCE_CATEGORIES}
    if "primary_category" in valid_df.columns:
        for cat in valid_df["primary_category"].dropna():
            if cat in cat_counts:
                cat_counts[cat] += 1

    cat_pct = {k: round(v / denom * 100, 1) for k, v in cat_counts.items()}

    # ── Category by discordance mode (valid only) ─────────────────────────────
    cat_by_mode: dict[str, dict[str, int]] = {}
    for mode in DISCORDANCE_MODES:
        sub = valid_df[valid_df["discordance_mode"] == mode] if "discordance_mode" in valid_df.columns else pd.DataFrame()
        if len(sub) == 0:
            cat_by_mode[mode] = {}
            continue
        counts = sub["primary_category"].value_counts().to_dict()
        cat_by_mode[mode] = {k: int(v) for k, v in counts.items()}

    # ── Category by age group (valid only) ────────────────────────────────────
    cat_by_age: dict[str, dict[str, int]] = {}
    if "age_band" in valid_df.columns and "primary_category" in valid_df.columns:
        for band, grp in valid_df.groupby("age_band"):
            counts = grp["primary_category"].value_counts().to_dict()
            cat_by_age[str(band)] = {k: int(v) for k, v in counts.items()}

    # ── Note_mitigates / amplifies breakdown ──────────────────────────────────
    mitigated = valid_df[valid_df.get("discordance_mode", pd.Series(dtype=str)) == "NOTE_MITIGATES"] if "discordance_mode" in valid_df.columns else pd.DataFrame()
    mitigated_cats = mitigated["primary_category"].value_counts().to_dict() if len(mitigated) > 0 else {}

    amplified = valid_df[valid_df["discordance_mode"] == "NOTE_AMPLIFIES"] if "discordance_mode" in valid_df.columns else pd.DataFrame()
    amplified_cats = amplified["primary_category"].value_counts().to_dict() if len(amplified) > 0 else {}

    return {
        "n_patients": n,
        "n_valid_annotations": n_valid,
        "n_annotation_failures": n_failed,
        "annotation_failure_rate_pct": failure_rate,
        "n_confirmed": confirmed,
        "n_rejected": rejected,
        "pct_confirmed": round(confirmed / n * 100, 1),
        "mode_distribution": {
            "counts": mode_counts,
            "percentages": mode_pct,
        },
        "category_distribution": {
            "counts": cat_counts,
            "percentages": cat_pct,
        },
        "category_by_discordance_mode": cat_by_mode,
        "category_by_age_group": cat_by_age,
        "note_mitigates": {
            "n": len(mitigated),
            "pct_of_rejected": round(len(mitigated) / max(rejected, 1) * 100, 1),
            "top_categories": {k: int(v) for k, v in list(mitigated_cats.items())[:5]},
        },
        "note_amplifies": {
            "n": len(amplified),
            "pct_of_confirmed": round(len(amplified) / max(confirmed, 1) * 100, 1),
            "top_categories": {k: int(v) for k, v in list(amplified_cats.items())[:5]},
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
