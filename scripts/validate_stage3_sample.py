"""Aggregate stats + per-patient dump for a Stage 3 batch validation run.

Run on the cluster against a fresh stage3_batch_results.csv to sanity-check
generation quality before committing to the full run. Not part of the
pipeline itself.
"""

import argparse
import json
from collections import Counter

import pandas as pd


def _parse_list_field(value):
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/mnt/vast-kisski/home/l.yorkstenzel/u29346/thesis/models/stage3_batch_results.csv",
    )
    parser.add_argument("--dump-limit", type=int, default=50)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    n = len(df)
    print(f"=== Aggregate stats (n={n}) ===")

    if "all_quotes_verified" in df.columns:
        verified = df["all_quotes_verified"].astype(bool)
        print(f"all_quotes_verified: {verified.sum()}/{n} ({100 * verified.mean():.1f}%)")

    if "decision_model" in df.columns:
        print("decision_model distribution:", dict(Counter(df["decision_model"])))
    if "decision_rule" in df.columns:
        print("decision_rule distribution:", dict(Counter(df["decision_rule"])))
    if "decision_model" in df.columns and "decision_rule" in df.columns:
        agree = (df["decision_model"] == df["decision_rule"]).mean()
        print(f"decision_model vs decision_rule agreement: {100 * agree:.1f}%")

    ground_counter = Counter()
    justification_lengths = []
    mid_word_cutoffs = 0

    for _, row in df.iterrows():
        for col in ("mitigating_grounds", "aggravating_grounds"):
            if col in df.columns:
                for g in _parse_list_field(row[col]):
                    ground_counter[g["ground"]] += 1
        if "clinical_justification" in df.columns:
            just = str(row["clinical_justification"])
            justification_lengths.append(len(just))
            if just and not just.rstrip().endswith((".", "!", "?")):
                mid_word_cutoffs += 1

    print("\nGround frequency across all patients:")
    for ground, count in ground_counter.most_common():
        print(f"  {ground}: {count}")

    if justification_lengths:
        avg_len = sum(justification_lengths) / len(justification_lengths)
        print(f"\nclinical_justification avg length: {avg_len:.0f} chars")
        print(f"clinical_justification not ending on sentence boundary: {mid_word_cutoffs}/{n}")

    print(f"\n=== Per-patient dump (first {args.dump_limit}) ===")
    for i, row in df.head(args.dump_limit).iterrows():
        print(f"\n--- [{i}] hadm_id={row.get('hadm_id')} ---")
        print(f"decision_model={row.get('decision_model')} decision_rule={row.get('decision_rule')}")
        print(f"all_quotes_verified={row.get('all_quotes_verified')}")
        for label, col in (("mitigating", "mitigating_grounds"), ("aggravating", "aggravating_grounds")):
            hits = _parse_list_field(row.get(col))
            print(f"{label}_grounds:")
            for g in hits:
                print(f"    [{g['ground']}] verified={g.get('quote_verified')} quote={g['quote']!r}")
            if not hits:
                print("    (none)")
        just = str(row.get("clinical_justification", ""))
        print(f"clinical_justification: {just}")


if __name__ == "__main__":
    main()
