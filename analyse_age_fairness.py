"""Measure Stage 1 and Stage 2 precision/FP rate by age group.

Determines whether the age-group bias in Stage 1 persists through Stage 2,
and whether it is large enough to warrant age-stratified threshold calibration.

Decision criteria (printed at the end):
  Gap < 3 pp  → Stage 2 already corrects for age bias. No stratification needed.
  Gap 3–8 pp  → Moderate gap. Equal-PPV stratification is worthwhile.
  Gap > 8 pp  → Large gap. Strong motivation for full meta-calibration model.

Usage:
    python analyse_age_fairness.py
"""

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, get_model_dir
from src.data.features import load_feature_matrix
from src.schemas import TARGET_COL

# ── Load config + Stage 1 artifact (no torch imported — safe on macOS) ──
cfg = load_config()
model_dir = get_model_dir()
stage1_name = cfg["stage1"]["model"]
artifact = joblib.load(model_dir / f"stage1_{stage1_name}.joblib")
mode = artifact["mode"]
print(f"Loaded Stage 1 artifact  ({mode} mode, threshold={artifact['threshold']:.4f})\n")

# ── Reconstruct Stage 1 test set with age bands ──
matrix = load_feature_matrix(cfg, mode)
test_idx = artifact["test_idx"]
test_df = matrix.iloc[test_idx][
    ["hadm_id", "subject_id", TARGET_COL, "age", "age_band"]
].reset_index(drop=True)

feat_cols = artifact["feature_cols"]
Xte = matrix.iloc[test_idx][feat_cols]
test_df["stage1_score"] = artifact["estimator"].predict_proba(Xte)[:, 1]
test_df["stage1_flagged"] = (test_df["stage1_score"] >= artifact["threshold"]).astype(int)

# ── Stage 1 metrics by age band ──
print("=" * 70)
print("STAGE 1 — metrics by age band (full test set, n={:,})".format(len(test_df)))
print("=" * 70)
print(f"{'Age band':>12} {'N':>7} {'Base rate':>10} {'Flagged':>8} "
      f"{'Precision':>10} {'Recall':>8} {'FPR':>8}")
print("-" * 70)

s1_rows = []
for band, grp in test_df.groupby("age_band", observed=True):
    n = len(grp)
    pos = int(grp[TARGET_COL].sum())
    flagged = grp[grp["stage1_flagged"] == 1]
    tp = int((flagged[TARGET_COL] == 1).sum())
    fp = len(flagged) - tp
    fn = pos - tp
    tn = n - len(flagged) - fn
    prec = tp / max(len(flagged), 1)
    rec  = tp / max(pos, 1)
    fpr  = fp / max(fp + tn, 1)
    print(f"{str(band):>12} {n:>7,} {pos/n:>10.1%} {len(flagged):>8,} "
          f"{prec:>10.3f} {rec:>8.3f} {fpr:>8.3f}")
    s1_rows.append({"band": str(band), "n": n, "precision": prec, "recall": rec, "fpr": fpr})

s1_df = pd.DataFrame(s1_rows)
s1_prec_range = s1_df["precision"].max() - s1_df["precision"].min()
print(f"\n  Stage 1 precision range across age bands: {s1_prec_range:.3f} "
      f"({s1_prec_range*100:.1f} pp)")

# ── Load Stage 2 results and merge age ──
results_path = model_dir / "stage2_results.csv"
if not results_path.exists():
    print("\nERROR: models/stage2_results.csv not found. Run setup_stage2.py first.")
    raise SystemExit(1)

s2_df = pd.read_csv(results_path)
s2_df = s2_df.merge(
    test_df[["hadm_id", "age", "age_band"]],
    on="hadm_id", how="left"
)
missing_age = s2_df["age_band"].isna().sum()
if missing_age:
    print(f"\nWARNING: {missing_age:,} Stage 2 patients could not be matched to age band.")

s2_df = s2_df.dropna(subset=["age_band"])

# ── Stage 2 metrics by age band (notes cohort, current threshold 0.3) ──
print("\n" + "=" * 70)
print(f"STAGE 2 — metrics by age band (notes cohort, n={len(s2_df):,}, thr=0.3)")
print("=" * 70)
print(f"{'Age band':>12} {'N notes':>8} {'Confirmed':>10} {'TP':>6} "
      f"{'Precision':>10} {'Recall':>8} {'vs S1 prec':>11}")
print("-" * 70)

s2_rows = []
for band, grp in s2_df.groupby("age_band", observed=True):
    n = len(grp)
    pos_in_cohort = int(grp[TARGET_COL].sum())
    confirmed = grp[grp["stage2_confirmed"] == 1]
    tp = int((confirmed[TARGET_COL] == 1).sum())
    prec = tp / max(len(confirmed), 1)
    rec  = tp / max(pos_in_cohort, 1)
    s1_prec = s1_df.loc[s1_df["band"] == str(band), "precision"].values
    s1_p = s1_prec[0] if len(s1_prec) else float("nan")
    delta = prec - s1_p
    print(f"{str(band):>12} {n:>8,} {len(confirmed):>10,} {tp:>6,} "
          f"{prec:>10.3f} {rec:>8.3f} {delta:>+10.3f}")
    s2_rows.append({"band": str(band), "n_notes": n, "precision": prec,
                    "recall": rec, "s1_precision": s1_p})

s2_df_agg = pd.DataFrame(s2_rows)
s2_prec_range = s2_df_agg["precision"].max() - s2_df_agg["precision"].min()

print(f"\n  Stage 2 precision range across age bands: {s2_prec_range:.3f} "
      f"({s2_prec_range*100:.1f} pp)")

# ── Threshold sweep per age band (equal-PPV candidates) ──
print("\n" + "=" * 70)
print("EQUAL-PPV THRESHOLD SWEEP — what threshold achieves global precision per band?")
print("=" * 70)
global_prec = s2_df_agg["precision"].mean()
print(f"  Global average Stage 2 precision: {global_prec:.3f}\n")
print(f"  {'Age band':>12} {'Current prec':>13} {'Gap to global':>14} "
      f"{'Equal-PPV thr':>14} {'Recall at thr':>13}")
print("-" * 70)

for _, row in s2_df_agg.iterrows():
    band_data = s2_df[s2_df["age_band"].astype(str) == row["band"]]
    pos_in_cohort = int((band_data[TARGET_COL] == 1).sum())
    gap = row["precision"] - global_prec
    # Find threshold that achieves global_prec for this band
    best_thr = None
    best_rec = None
    for thr in np.arange(0.05, 0.96, 0.01):
        conf = band_data[band_data["stage2_score"] >= thr]
        if len(conf) == 0:
            break
        tp = int((conf[TARGET_COL] == 1).sum())
        p = tp / len(conf)
        if p >= global_prec:
            best_thr = thr
            best_rec = tp / max(pos_in_cohort, 1)
            break
    thr_str = f"{best_thr:.2f}" if best_thr else "n/a (impossible)"
    rec_str = f"{best_rec:.3f}" if best_rec else "—"
    print(f"  {row['band']:>12} {row['precision']:>13.3f} {gap:>+13.3f} pp  "
          f"{thr_str:>14} {rec_str:>13}")

# ── Verdict ──
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if s2_prec_range < 0.03:
    verdict = (
        "SMALL GAP (<3 pp). Stage 2 already largely corrects for age bias via\n"
        "  discharge note context. Age-stratified calibration is NOT justified —\n"
        "  the notes are doing their job. Report this as a positive finding."
    )
elif s2_prec_range < 0.08:
    verdict = (
        "MODERATE GAP (3–8 pp). Age bias partially persists through Stage 2.\n"
        "  Equal-PPV stratification (Operation 2) is worthwhile and defensible.\n"
        "  The full meta-calibration model is optional — probably future work."
    )
else:
    verdict = (
        "LARGE GAP (>8 pp). Age bias persists strongly through Stage 2.\n"
        "  Strong motivation for the full meta-calibration approach:\n"
        "  retrain pipeline with calibration split + FP-surface GAM."
    )
print(f"  Stage 2 precision range: {s2_prec_range*100:.1f} pp\n")
print(f"  {verdict}")
print("=" * 70)
