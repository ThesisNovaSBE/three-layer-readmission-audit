"""Compare a batch-size-1 Stage 3 re-audit against the main (batch-size-4)
run on the same admissions (N11).

bf16 batched generation is sensitive to which admissions share a batch, so a
decision can differ between batch sizes even at temperature 0. This quantifies
that: decision_model / decision_rule / grounds agreement between
``models/stage3_bs1_check.csv`` (produced by ``src.stage3.batch --limit 20
--batch-size 1 --out ...``) and ``models/stage3_batch_results.csv``, joined on
hadm_id. ``--limit 20`` audits the first 20 admissions in stage2_results.csv
order, which are exactly the first 20 of the main run, so the two cover the
same patients.

Diagnostic only: this never fails the job. Missing inputs or no overlap just
print a notice and return.

Usage::

    python -m scripts.compare_batch_size
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import get_model_dir


def _grounds_set(cell: object) -> frozenset:
    """A (ground, quote) set for one cell, so grounds compare order-independently."""
    try:
        items = json.loads(cell) if isinstance(cell, str) and cell else []
    except (ValueError, TypeError):
        return frozenset()
    return frozenset((g.get("ground"), g.get("quote")) for g in items)


def _same_grounds(row: pd.Series) -> bool:
    """Both mitigating and aggravating grounds match between the two runs."""
    for kind in ("mitigating", "aggravating"):
        if _grounds_set(row[f"{kind}_grounds_main"]) != _grounds_set(row[f"{kind}_grounds_bs1"]):
            return False
    return True


def main() -> None:
    """Join the two runs on hadm_id and report decision/rule/grounds agreement."""
    model_dir = get_model_dir()
    main_path = model_dir / "stage3_batch_results.csv"
    bs1_path = model_dir / "stage3_bs1_check.csv"
    if not main_path.exists() or not bs1_path.exists():
        print(f"[compare_batch_size] missing input ({main_path} or {bs1_path}) -- skipping.")
        return

    merged = pd.read_csv(main_path).merge(
        pd.read_csv(bs1_path), on="hadm_id", suffixes=("_main", "_bs1")
    )
    n = len(merged)
    if n == 0:
        print("[compare_batch_size] no overlapping hadm_ids -- skipping.")
        return

    dec_agree = float((merged["decision_model_main"].fillna("<none>")
                       == merged["decision_model_bs1"].fillna("<none>")).mean())
    rule_agree = float((merged["decision_rule_main"].fillna("<none>")
                        == merged["decision_rule_bs1"].fillna("<none>")).mean())
    grounds_agree = float(merged.apply(_same_grounds, axis=1).mean())

    report = {
        "n": int(n),
        "decision_model_agreement": dec_agree,
        "decision_rule_agreement": rule_agree,
        "grounds_agreement": grounds_agree,
    }
    out_path = model_dir / "batch_size_comparison.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[compare_batch_size] n={n}: decision_model {dec_agree:.1%}, "
          f"decision_rule {rule_agree:.1%}, grounds {grounds_agree:.1%} -> {out_path}")


if __name__ == "__main__":
    main()
