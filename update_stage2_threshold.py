"""Update stage2_confirmed column in stage2_results.csv to a new threshold.

Usage:
    python update_stage2_threshold.py --threshold 0.3
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, required=True)
args = parser.parse_args()

df = pd.read_csv("models/stage2_results.csv")
df["stage2_confirmed"] = (df["stage2_score"] >= args.threshold).astype(int)
df.to_csv("models/stage2_results.csv", index=False)

confirmed = int(df["stage2_confirmed"].sum())
tp = int(((df["stage2_confirmed"] == 1) & (df["readmission_30d"] == 1)).sum())
print(f"Updated stage2_results.csv: thr={args.threshold}")
print(f"  Confirmed: {confirmed:,}  |  Precision: {tp/max(confirmed,1):.3f}")
