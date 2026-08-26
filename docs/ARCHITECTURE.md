# Architecture — Current State

**Last updated:** 2026-08-26 (session 16). This is the single current source of
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

### Layer 2 — Clinical-Longformer, note-only (independent note-based estimate)

- Plain `LongformerForSequenceClassification` fine-tuned on discharge notes
  only — **no structured features**. Clinical-Longformer backbone, 4096-token
  window (changed from 2048 — session 14 measured 74% of discharge notes
  exceeding 2048 tokens, median ~2,649; the 2048 setting was based on an
  earlier, since-corrected estimate).
- **FusionLongformer (structured features + note, jointly trained) was
  designed, built, and reverted on 2026-08-26 without ever completing a
  training run.** `models/` never contained fusion weights — every evaluated
  Stage 2 number that exists (`stage2_evaluation.json`, dated 2026-08-01) is
  from the plain note-only model. Decision: drop fusion as a modeling target.
  Reasons: (1) note-only is the cleaner independence story for Layer 3's
  discordance measure — Stage 1 (~40 structured features) and Stage 2 (note
  text, zero structured features) are unambiguously informationally
  independent, no caveating required; (2) it was never trained, so dropping
  it costs nothing already built, while training it for the first time would
  cost a full KISSKI/Grete GPU run for a design whose only role was Table 2
  Row 3 of the comparison study (a secondary RQ1 sub-question, not the
  thesis's actual contribution); (3) `src/stage2/train.py` had been fully
  rewritten around fusion with no way left to reproduce the working,
  already-evaluated plain model — reverting `train.py`/`dataset.py`/
  `predict.py`/`_utils.py` to their pre-fusion versions (git commit
  `8a4af36`) fixes that as a side effect. `src/stage2/model.py`
  (`FusionLongformer` class) was deleted; `STRUCT_FEATURE_COLS` no longer
  exists. If joint fusion is wanted later, treat it as new work, not a
  resumption — the code for it is gone.
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

## 3. What changed in code

All of the below required no model retraining and are already implemented.

**Session 15 (2026-08-25):**

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
  `test_config.py` extended/updated.

**Session 16 (2026-08-26):**

- **FusionLongformer dropped.** `src/stage2/{train,dataset,predict,_utils}.py`
  reverted to their pre-fusion versions (git commit `8a4af36`); `model.py`
  (the `FusionLongformer` class) deleted. Stage 2 is a plain, note-only
  Clinical-Longformer — see §2. `setup_stage2.py` also had unrelated
  pre-existing pylint issues (dead imports, missing docstrings) fixed while
  touching this code.
- **Isotonic calibration added for Stage 1** (`src/model/calibration.py`).
  Fit on OOF predictions (the same ones used for threshold selection — no
  new leakage surface), stored in the artifact, applied everywhere a
  Stage 1 score is produced (`evaluate.py`, `evaluate_pipeline.py`,
  `stage2/predict.py`, `api.py`). Saves `models/stage1_calibration.json`
  (Brier before/after + reliability-curve breakpoints). Monotonic, so it
  does not change which admissions get flagged, only what the reported
  probability means.
- **Bootstrap CIs added** (`src/model/bootstrap.py`): generic
  `bootstrap_ci(y_true, y_score, metric_fn, groups=...)` with patient-level
  (`subject_id`) cluster resampling. Wired into `evaluate.py` for
  AUROC/AUPRC/precision/recall at the primary operating point. **Not yet
  wired into `evaluate_pipeline.py`** — the full/notes-cohort pipeline
  numbers still lack CIs.
- **`readmission_30d_unplanned` label variant added** (P7b). New column in
  the feature matrix: the all-cause label minus outcome admissions whose own
  `admission_type` indicates a planned return (`ELECTIVE`,
  `SURGICAL SAME DAY ADMISSION`) — a structured-data proxy, not a
  note-derived one (`src/data/features.py:compute_readmission_label`). Not
  yet wired into any training/evaluation script — `split_xy` still returns
  the all-cause label by default; using the unplanned variant is a future
  analysis, not automatic.
- **Deleted `models/stage3_discordance_analysis.json`** (git-tracked,
  produced by the pre-session-15 Stage 3 design; schema no longer matches
  the current code, and it demonstrated the exact P2 bug — `NOTE_AMPLIFIES`
  count exceeding `n_confirmed`). Recoverable from git history if needed.

---

## 4. What is explicitly deferred (not done yet)

These require either GPU retraining or a further design decision and were
intentionally left for the next round of "exact stage behaviour" discussion:

- **Retrain Stage 1** with the rebuilt feature matrix (400-trial Optuna,
  session-14 feature fixes, isotonic calibration) —
  `scripts/slurm_stage1_tune.sh` is ready.
- **Retrain Stage 2** at `max_seq_length=4096` — no script changes needed,
  just a fresh KISSKI/Grete job (plain Longformer only, per §2).
- **`api.py` / `src/stage2/predict.py` still treat `stage2_confirmed` as a
  gate** in places (e.g. `list_patients(confirmed_only=True)`). Per §2,
  Stage 2 no longer gates conceptually — Stage 3 is meant to see every
  flagged+noted patient regardless of Stage 2's confirm/reject call. Whether
  and how to change the API/frontend contract (which the current frontend
  depends on) is exactly the "exact stage behaviour" discussion still to
  have — not changed yet to avoid breaking the running demo without a
  coordinated frontend update.
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
- **Bootstrap CIs not yet in `evaluate_pipeline.py`** — only Stage 1's
  standalone `evaluate.py` has them so far (see §3).
- **`readmission_30d_unplanned` not yet used anywhere** — the column exists
  in the feature matrix but no script trains or evaluates against it yet.
- Feature audit (vitals missingness, lab itemid validation against
  `d_labitems`) — not touched.

---

## 5. Open methodological notes carried forward

- Stage 1's decision threshold is still selected on the same OOF/validation
  data used for early-stopping — a mild contamination risk flagged once in
  an earlier review and never revisited. Still open.
- `readmission_30d_unplanned` only catches planned returns visible in
  `admission_type` (elective / same-day surgical) — a structured-data proxy.
  Planned returns not reflected there (e.g. informally scheduled follow-up
  admissions) are not caught; the remediation review's original proposal
  (an LLM-extracted `planned_return` flag from the note) would catch more
  but depends on evidence-extraction infrastructure that doesn't exist.
- Truncation asymmetry between Stage 2 (4096 tokens) and Stage 3 (now a
  20,000-character safety cap, effectively near-full note) is much smaller
  than before session 15 but not eliminated for pathologically long notes.
- `training-changes` is still unmerged into `main` (44+ commits ahead as of
  session 15). Nothing in this doc addresses that — it's a repo-hygiene
  decision (merge vs. formally adopt as the working branch), not a code fix.
