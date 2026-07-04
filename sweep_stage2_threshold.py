"""Sweep Stage 2 thresholds using already-computed scores in stage2_results.csv.

No model inference needed — just re-applies thresholds to saved scores.

Metrics are reported relative to the NOTES COHORT (patients with discharge notes)
which is the correct evaluation population: in real clinical use every discharged
patient has a discharge note, so the 25,999 Stage 1 flags without notes in MIMIC-IV
are a dataset coverage gap, not a clinical reality.

Usage:
    python sweep_stage2_threshold.py
"""

import pandas as pd

df = pd.read_csv("models/stage2_results.csv")

# Notes cohort: all Stage 1-flagged patients who have a discharge note.
# This is the deployable population — Stage 2 can only act on patients with notes.
notes_positives = int((df["readmission_30d"] == 1).sum())  # TP within notes cohort
notes_total = len(df)

print(f"\nNotes cohort: {notes_total:,} flagged patients with discharge notes")
print(f"Positives in notes cohort: {notes_positives:,} ({notes_positives/notes_total:.1%})")
print(f"\nRecall is relative to notes cohort positives ({notes_positives:,}),")
print(f"not the full test set — patients without notes are out of scope for this pipeline.\n")

print(f"{'Threshold':>10} {'Confirmed':>10} {'Retained':>10} {'Precision':>10} {'Recall':>10} {'F2':>8}")
print("-" * 62)
# Stage 1 baseline within notes cohort (thr=0.1 = keep everyone)
prec0 = notes_positives / notes_total
print(f"{'Stage1':>10} {notes_total:>10,} {'100.0%':>10} {prec0:>10.3f} {'1.000':>10} {5*prec0*1/(4*prec0+1):>8.3f}  (all flagged with notes)")
print("-" * 62)
for thr in [0.2, 0.3, 0.4, 0.5]:
    confirmed = df[df["stage2_score"] >= thr]
    n = len(confirmed)
    tp = int((confirmed["readmission_30d"] == 1).sum())
    prec = tp / max(n, 1)
    rec = tp / notes_positives
    f2 = 5 * prec * rec / (4 * prec + rec) if (prec + rec) > 0 else 0
    print(f"{thr:>10.1f} {n:>10,} {n/notes_total:>9.1%} {prec:>10.3f} {rec:>10.3f} {f2:>8.3f}")
