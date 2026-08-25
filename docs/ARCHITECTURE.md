# Architecture — Current State

**Last updated:** 2026-08-25 (session 15). This is the single current source of
truth for the pipeline design. It supersedes `docs/IMPLEMENTATION_PLAN.md`,
`docs/THESIS_NARRATIVE.md`, and `docs/SANITY_CHECK_2026-07-06.md` — those are
kept for history (each now has a banner pointing here) but describe designs
this project has since moved past. If you are a new session picking up this
project, read this file, then the latest `sessions/` entry.

---

## 1. How we got here (context for why this doc exists)

Between 2026-07-31 and 2026-08-19 the Stage 3 design went through three
undocumented-in-one-place iterations: (1) the original build had phi4-mini
narrate Stage 1's own SHAP values — rejected in session 11 as "essentially
SHAP explanation + LLM, a well-trodden pattern with no novel finding"; (2)
the "Cross-Modal Discordance Analysis" replacement (session 11) had phi4-mini
freely classify a 9-category discordance taxonomy from a raw
`stage2_score - stage1_score` difference; (3) three separate planning
artifacts (an evaluation critique, a Project Brief, and a Working Paper — all
2026-08-16/17) independently converged on fixing this by making phi4-mini an
*independent* auditor with a *quantitatively computed* discordance measure.
A colleague's remediation review (2026-08-19) confirmed the raw-subtraction
approach in (3) was still arithmetically fragile (P1) even with shared
calibration, because two different model families are not guaranteed to have
equal residual miscalibration.

Session 15 (2026-08-25) implemented the resolution: percentile-rank
displacement instead of raw subtraction, and phi4-mini returns its own
uphold/override decision rather than narrating or classifying anyone else's.
See that session log for the full reasoning trail.

---

## 2. The three layers, as implemented now

### Layer 1 — XGBoost (structured screen)

- Input: ~40 structured MIMIC-IV features (demographics, admission traits,
  prior utilisation, Charlson comorbidity index, lab/vital aggregates).
- Output: calibrated risk score.
- **Operating point policy (changed 2026-08-25): capacity-constrained,
  primary.** Flags the top `stage1.capacity_k` fraction of admissions
  (default 15%) rather than the lowest threshold meeting a recall floor. The
  old recall-floor policy (`recall >= 0.85`) flagged 66.94% of all
  admissions — not a deployable triage. Recall-floor is retained and reported
  as a secondary table for comparability with prior literature
  (`src/model/metrics.py:select_threshold_for_recall`,
  `select_threshold_for_capacity`).
- Config: `stage1.threshold_strategy`, `stage1.capacity_k`,
  `stage1.capacity_report_points` (`config.yaml`).

### Layer 2 — FusionLongformer / plain Longformer (independent combined estimate)

- Clinical-Longformer backbone, 4096-token window (changed from 2048 —
  session 14 measured 74% of discharge notes exceeding 2048 tokens, median
  ~2,649; the 2048 setting was based on an earlier, since-corrected estimate).
- `STRUCT_FEATURE_COLS` (8 features) is a *different, smaller* feature set
  than Stage 1's ~40. **We do not claim these are "the same structured
  features" or that any Stage1/Stage2 score gap is cleanly attributable to
  the note alone** — that claim doesn't hold given the feature-set mismatch,
  and no code change was made to force it to hold (cutting Stage 1 to 8
  features was considered and rejected — unjustified retraining cost for no
  clear benefit). The weaker, defensible claim: Stage 1 and Stage 2 are two
  independently-trained models using different information; when they
  disagree, that's worth investigating.
- Produces `stage2_score`. `stage2_confirmed` (a per-age-band threshold
  decision) is still computed and still useful for the N1 ablation's
  "cascade" arm, but is **not** the final word on a patient — see Layer 3.

### Layer 3 — phi4-mini (independent auditor)

Rewritten 2026-08-25 (`src/stage3/explain.py`, `src/stage3/pipeline.py`,
`src/stage3/models.py`). Inputs: Stage 1's score + SHAP-ranked reasons,
Stage 2's score, the discharge note itself (near-full text, not a 5-sentence
attention summary), and a pre-computed discordance mode. Output: `decision`
(`uphold` / `override`) — phi4-mini's **own** judgment, not a narration of
Stage 2's — plus `primary_clinical_domain` and `clinical_justification`.

**Discordance mode is computed quantitatively, never by the LLM**
(`compute_discordance` in `src/stage3/explain.py`):

```
r1 = percentile rank of stage1_score within the flagged+noted cohort
r2 = percentile rank of stage2_score within the same cohort
displacement = r2 - r1                          # range [-100, +100]
NOTE_MITIGATES  if displacement <= -20
NOTE_AMPLIFIES  if displacement >= +20
CONCORDANT      otherwise
```

Percentile rank (not raw `stage2_score - stage1_score`) because Stage 1 and
Stage 2 are different model families and are not guaranteed to share
equivalent calibration error even after isotonic calibration — rank
displacement is invariant to that risk by construction. The `20`
percentile-point threshold (`stage3.discordance_displacement_pp`) is a
provisional value; validate empirically (percentile sweep of the observed
displacement distribution) before reporting it as a result — nobody has done
this yet.

`stage3.temperature` is pinned at `0.0`. Uphold/override is the primary
outcome variable for RQ2; a stochastic auditor makes any reported effect a
sample from a distribution rather than a fixed, checkable quantity.

---

## 3. What changed in code on 2026-08-25 (session 15)

All of the below required no model retraining and are already implemented:

- `config.yaml` / `src/config_schema.py`: `stage2.max_seq_length` 2048→4096;
  `stage3.temperature` 0.3→0.0; new `stage1.threshold_strategy`,
  `stage1.capacity_k`, `stage1.capacity_report_points`; new
  `stage3.discordance_displacement_pp`.
- `src/model/metrics.py`: added `select_threshold_for_capacity`,
  `metrics_at_capacity_points` (precision@K, lift@K).
- `src/model/train.py`, `src/model/evaluate.py`: Stage 1 now selects and
  reports both operating points; capacity-constrained is primary.
- `src/model/evaluate_pipeline.py`: added the C9 fallback — flagged-but-
  no-note patients now fall back to Stage 1's own flag instead of being
  silently scored negative. Reports `pipeline.full_cohort` (primary, 100% of
  test admissions) and `pipeline.notes_cohort` (secondary, matches how
  "+21% precision" was computed before this fix) separately; never report
  `notes_cohort` alone.
- `src/stage3/explain.py`: full rewrite (see §2 above).
- `src/stage3/models.py`, `src/stage3/pipeline.py`: updated to the new
  `ExplanationResult` schema (`decision`, `r1`, `r2`, `displacement`,
  `primary_clinical_domain`, `clinical_justification` replace the old
  `discordance_mode`-as-LLM-opinion / `primary_category` / `narrative`).
- `data/processed/features.csv` rebuilt with the session-14 fixes (fixed
  `admission_type_emergency` mapping, 3 new features, 521,191 rows).
- `tests/`: `test_stage3_explain.py` rewritten for the new API (incl. a test
  that percentile-rank displacement is invariant to monotonic rescaling —
  the property raw subtraction lacks); `test_metrics.py` and
  `test_config.py` extended/updated. 65 passed, 1 skipped; pylint 10.00/10.

---

## 4. What is explicitly deferred (not done in this session)

These require either GPU retraining or a further design decision and were
intentionally left for the next round of "exact stage behaviour" discussion:

- **Retrain Stage 1** with the rebuilt feature matrix (400-trial Optuna,
  session-14 feature fixes) — `scripts/slurm_stage1_tune.sh` is ready.
- **Retrain Stage 2** at `max_seq_length=4096` — no script changes needed,
  just a fresh KISSKI/Grete job.
- **`api.py` / `src/stage2/predict.py` still treat `stage2_confirmed` as a
  gate** in places (e.g. `list_patients(confirmed_only=True)`). Per §2,
  Stage 2 no longer gates conceptually — Stage 3 is meant to see every
  flagged+noted patient regardless of Stage 2's confirm/reject call. Whether
  and how to change the API/frontend contract (which the current frontend
  depends on) is exactly the "exact stage behaviour" discussion still to
  have — not changed this session to avoid breaking the running demo without
  a coordinated frontend update.
- **No batch Stage 3 runner yet.** `explain_patient` is on-demand,
  single-patient only. `evaluate_pipeline.py`'s pipeline metric still uses
  `stage2_confirmed` as the final pipeline prediction (with the C9 fix on
  top) rather than Stage 3's decision, because there's no batch path to
  produce Stage 3 decisions for a whole test partition yet. Needed before
  RQ2/RQ3 can be evaluated at scale.
- **N1 ablation runner** (5+ arms incl. the critical L1-at-matched-capacity
  arm) — not yet written.
- **ε / displacement threshold validation** — `discordance_displacement_pp`
  is a provisional 20; needs the percentile-sweep sensitivity check before
  being reported as a methodological choice rather than a guess.
- **Bootstrap CIs** — not yet added to any reported metric.

---

## 5. Open methodological notes carried forward

- Stage 1's decision threshold is still selected on the same OOF/validation
  data used for early-stopping — a mild contamination risk flagged once in
  an earlier review and never revisited. Still open.
- The label counts *all* 30-day returns, not just unplanned ones (elective
  index admissions are excluded, elective *outcome* admissions are not) —
  `readmission_30d_unplanned` (label variant excluding documented planned
  returns) is still unbuilt.
- Truncation asymmetry between Stage 2 (4096 tokens) and Stage 3 (now a
  20,000-character safety cap, effectively near-full note) is much smaller
  than before this session but not eliminated for pathologically long notes.
