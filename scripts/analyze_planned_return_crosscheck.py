"""Cross-check: does Stage 3's note-derived ``planned_return`` field agree
with the structured, admission_type-based "unplanned" proxy
(``readmission_30d_unplanned`` in src/data/features.py)?

The two ask related but distinct questions, at different times:

- The structured proxy is a RETROSPECTIVE fact about the readmission that
  actually happened: among patients who WERE readmitted within 30 days
  (``readmission_30d == 1``), was that outcome admission's own
  ``admission_type`` a scheduled one (elective / same-day surgical)? Only
  defined for patients who were actually readmitted.
- Stage 3's ``planned_return`` is a PROSPECTIVE read of the index
  admission's discharge note: does the note itself mention a scheduled
  future return (chemo cycle, staged surgery, scheduled dialysis)? Defined
  for every Stage-3-audited patient, whether or not they are later
  readmitted.

Restricting to the subset that WAS actually readmitted is what makes the
two comparable: for those patients, "the readmission that happened was
planned" (structured) and "the note said a planned return was coming"
(Stage 3) are now two independent answers to close to the same question.

Never examined anywhere in the repo before this script -- see
docs/ARCHITECTURE.md §6 ("the two are not yet cross-checked against each
other; whether they agree ... is unexamined").

Usage::

    python -m scripts.analyze_planned_return_crosscheck
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import get_model_dir
from src.schemas import TARGET_COL, TARGET_COL_UNPLANNED


def main() -> None:
    """Cross-tabulate Stage 3's planned_return against the structured proxy."""
    model_dir = get_model_dir()
    s3 = pd.read_csv(model_dir / "stage3_batch_results.csv")
    feat = pd.read_csv(
        "data/processed/features.csv",
        usecols=["hadm_id", TARGET_COL, TARGET_COL_UNPLANNED],
    )

    merged = s3.merge(feat, on="hadm_id", how="left")
    n_total = len(merged)

    valid = merged[~merged["annotation_failed"]].copy()
    n_valid = len(valid)

    readmitted = valid[valid[TARGET_COL] == 1].copy()
    n_readmitted = len(readmitted)

    readmitted["structured_planned"] = readmitted[TARGET_COL_UNPLANNED] == 0
    readmitted["llm_planned"] = readmitted["planned_return"] == "yes"
    readmitted["llm_not_stated"] = readmitted["planned_return"] == "not_stated"

    comparable = readmitted[~readmitted["llm_not_stated"]].copy()
    n_comparable = len(comparable)
    agree = (comparable["structured_planned"] == comparable["llm_planned"]).mean()
    # Sanity benchmark: structured_planned is a rare class (~4%), so a
    # trivial "always predict unplanned" baseline already scores high on
    # raw agreement -- reporting agreement alone, without this, would read
    # as more reassuring than it is if Stage 3's real agreement rate turns
    # out to be BELOW what doing nothing achieves.
    majority_baseline = 1.0 - comparable["structured_planned"].mean()

    structured_label = readmitted["structured_planned"].map(
        {True: "structured=planned", False: "structured=unplanned"}
    )
    cross = pd.crosstab(readmitted["planned_return"].fillna("annotation_failed"), structured_label)

    print(f"[planned_return_crosscheck] Stage 3 batch: n={n_total:,}")
    print(f"  valid annotation (not annotation_failed): n={n_valid:,}")
    print(f"  actually readmitted within 30d (readmission_30d==1): n={n_readmitted:,}")
    print(f"  comparable (excludes LLM 'not_stated'): n={n_comparable:,}")
    print()
    print("  Cross-tab (rows = Stage 3's planned_return, cols = structured proxy):")
    print(cross.to_string())
    print()
    print(f"  Agreement rate: {agree:.1%}")
    print(
        f"  Majority-baseline agreement (always predict 'unplanned'): "
        f"{majority_baseline:.1%}  <-- Stage 3's real agreement is "
        f"{'BELOW' if agree < majority_baseline else 'above'} this trivial baseline"
    )
    print(
        f"  Structured says planned: {int(comparable['structured_planned'].sum()):,} "
        f"({comparable['structured_planned'].mean():.1%})"
    )
    print(
        f"  Stage 3 says planned:    {int(comparable['llm_planned'].sum()):,} "
        f"({comparable['llm_planned'].mean():.1%})"
    )

    report = {
        "n_stage3_batch": n_total,
        "n_valid_annotation": n_valid,
        "n_actually_readmitted": n_readmitted,
        "n_comparable": n_comparable,
        "agreement_rate": float(agree),
        "majority_baseline_agreement": float(majority_baseline),
        "structured_planned_rate": float(comparable["structured_planned"].mean()),
        "llm_planned_rate": float(comparable["llm_planned"].mean()),
        "cross_tab": cross.to_dict(),
        "note": (
            "Restricted to patients who were actually readmitted within 30 "
            "days (readmission_30d==1) -- only for them does 'was the "
            "readmission planned' have a structured answer at all. Excludes "
            "annotation_failed rows (no valid planned_return) and the single "
            "LLM 'not_stated' row (neither agreement nor disagreement)."
        ),
    }
    out_path = model_dir / "planned_return_crosscheck.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[planned_return_crosscheck] Saved -> {out_path}")


if __name__ == "__main__":
    main()
