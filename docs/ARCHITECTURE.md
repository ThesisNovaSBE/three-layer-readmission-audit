# Architecture — Current State

**Last updated:** 2026-08-28 (session 18). This is the single current source of
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
Stage 2's — plus `primary_clinical_domain`, `clinical_justification`,
`supporting_quote`, `quote_verified`, and `planned_return` (added
2026-08-28, colleague review item 2/3 — see session 18):

- `supporting_quote`: the exact note passage the LLM says its judgment rests
  on. `verify_quote()` checks it appears verbatim in the note text
  (`quote_verified: bool`) — a cheap, automatic hallucination check that
  makes human review of an override tractable (spot-check the quote, not the
  whole note) rather than a leap of faith. All 5 fields (decision, domain,
  quote non-empty, `quote_verified`, `planned_return`) are enforced at the
  same level in `_parse_response` — a missing/invalid quote fails parsing
  exactly like a missing decision does, not silently ignored.
- `planned_return`: requires the LLM to state, as its own structured field
  (`PLANNED_RETURN_ANSWERS = ("yes", "no", "not_stated")`), whether the note
  documents this admission as a planned return — directly answers the
  colleague's review question about whether the auditor distinguishes
  planned from unplanned returns rather than conflating them. (Also see
  `TARGET_COL_UNPLANNED` in `src/schemas.py`/`src/data/features.py`, which
  computes the same distinction at the label level, independent of Stage 3.)

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

### RQ1 and RQ2 — which module answers what

- **RQ1 — does text carry signal at all?** Layer 1 vs. Layer 2, scored
  independently on the *same* population (not gated by Layer 1's flag) and
  compared head-to-head. `src/stage2/predict.py:predict_stage2_all` produces
  the population-wide Layer 2 scores this needs;
  `src/model/compare_layers.py` does the comparison. Neither model feeds the
  other here — this is not the cascade.
- **RQ2 — does the auditor add value over Layer 1 alone?** Layer 3 audits
  Layer 1's positive predictions using Layer 2's score as evidence.
  `src/stage3/batch.py:run_batch_audit` runs this at scale;
  `src/model/evaluate_pipeline.py` reports the resulting pipeline metrics
  with Layer 3's decision (not Layer 2's threshold) as the final prediction
  wherever a batch audit covers the admission. The load-bearing comparison
  (colleague review item 1, 2026-08-28) is against a **control arm**:
  Layer 1 alone, its threshold raised to match the full system's alert
  volume exactly (`evaluate_pipeline.py:_control_arm_report`, using
  `select_threshold_for_capacity`) — not an unconstrained Layer 1. Without
  this, an apparent precision gain from the cascade could just be Layer 1
  flagging fewer admissions, which raising its own threshold would do for
  free. Reported as `report["pipeline"]["control_arm_stage1_matched"]`.

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

**Session 17 (2026-08-27):** built everything from the session-16 gap audit
that needed no GPU and no open decision.

- **Population-wide Layer 2 scoring** — `src/stage2/predict.py`. New
  `predict_stage2_all` (shares a refactored `_score_core`/`_prepare_population`
  pipeline with the existing `predict_stage2`), scores every test-partition
  admission with a note, not just Layer 1's positives. Output
  `models/stage2_results_all.csv`. `--all` / `--limit` CLI flags added.
  Required for RQ1 — see §2.5.
  **Bug found while smoke-testing, not fully fixed:** running this module's
  CLI standalone (`python -m src.stage2.predict`, with or without `--all`)
  segfaults (SIGSEGV) on macOS ARM — confirmed pre-existing on the
  pre-session-17 code too, not introduced today. Root cause: the same
  MPS/XGBoost C-extension conflict documented elsewhere in this codebase
  (`api.py`, `setup_stage2.py`) — `joblib.load()`-ing the XGBoost artifact
  after torch has initialised crashes on Apple Silicon, and
  `src.stage2.dataset` transitively imports torch, so this module's own
  import order can't prevent it alone. Added the standard `PYTORCH_ENABLE_MPS_FALLBACK`
  mitigation (doesn't fully fix it) and documented the real cause inline.
  **Verified working**: calling `predict_stage2_all(cfg, artifact=<preloaded>)`
  with the artifact loaded before torch imports (`setup_stage2.py`'s own
  pattern) succeeds — confirmed by an ad hoc script, 15-admission smoke test.
  A real fix (lazy-importing everything torch-touching) is a bigger change
  than this session's scope; not attempted to avoid risking
  `setup_stage2.py`'s already-working path. Until fixed: never invoke
  `python -m src.stage2.predict` directly on macOS; drive it the way
  `setup_stage2.py` does, or run on Linux/the cluster.
- **`api.py` stopped gating on Layer 2's threshold.** `list_patients`'s
  `confirmed_only` default flipped `True`→`False`; docstrings updated. Layer
  2's score is evidence for Layer 3, not a display gate, per §2.
  **Follow-up found, not fixed:** the frontend (`frontend/src/api.ts`) has
  its own independent `confirmedOnly = true` default and UI copy built
  around "confirmed" patients (`PatientTable.tsx`, `PipelineDiagram.tsx`) —
  changing the backend default didn't change frontend behaviour. Needs an
  actual frontend change plus a browser check, not a blind edit; not done
  this session.
- **N1 comparison runner (RQ1)** — new `src/model/compare_layers.py`.
  Inner-joins Stage 1's test-partition scores with
  `stage2_results_all.csv` on `hadm_id`, reports both models' AUROC/AUPRC
  (bootstrap 95% CI, patient-level) and an operating point matched to Layer
  1's alert volume, plus a paired bootstrap CI on the AUROC difference.
  Output `models/comparison_rq1.json`.
- **Batch Stage 3 runner (RQ2)** — new `src/stage3/batch.py`. Runs
  `explain_patient` over every admission in `stage2_results.csv`
  (preloading artifact/results/feature matrix once), writes incrementally
  and crash-safe to `models/stage3_batch_results.csv`, supports `--limit`
  and `--resume`. One bad admission no longer kills the run.
- **Discordance-threshold sensitivity sweep** — `sweep_discordance_thresholds`
  in `src/stage3/explain.py` (pure function, reclassifies existing
  `displacement` values at multiple pp thresholds — no rank recomputation
  needed) plus `run_sensitivity_sweep` / `--sweep` in `src/stage3/batch.py`,
  consuming `stage3_batch_results.csv`. Output
  `models/discordance_sensitivity.json`.
- **Layer 3's model call generalised** — `call_phi4mini` → `call_llm(prompt,
  cfg, model_name=None)` in `src/stage3/explain.py`; `explain_patient` gained
  a matching `model_name` passthrough. New `stage3.robustness_model` config
  field (default `null` — a real open decision, not a default to assume; see
  §5). Makes the scale-robustness arm wireable without further code changes
  once a model is chosen.
- **`evaluate_pipeline.py`'s final prediction now prefers Layer 3's
  decision.** New `_apply_stage3_decisions`: wherever
  `stage3_batch_results.csv` has a decision for an admission, it replaces
  `stage2_confirmed` (+ C9 fallback) as the final prediction; admissions
  Layer 3 hasn't covered keep the prior C9-corrected value unchanged — this
  only ever narrows the gap to the design, never drops coverage. Report now
  includes a `stage3` block reporting coverage.
- Tests: `test_compare_layers.py`, `test_stage3_batch.py`,
  `test_evaluate_pipeline.py` (new); `test_stage3_explain.py` extended for
  the sweep. 108 passed, 1 skipped; pylint 10.00/10.

**Session 18 (2026-08-28):** built the BUILD-NOW items from a colleague
review (control arm, evidence quoting, planned-return question) plus the
narrative-doc rewrite; the colleague's other two items (retrain Stage 1,
per-group context) and the user's own four open architecture questions were
deliberately left as open decisions, not implemented — see §5.

- **Control arm** — `evaluate_pipeline.py:_control_arm_report` (§2.5 above).
  Stage 1 alone, threshold raised to match the full system's realized alert
  rate, evaluated with the same `_pipeline_report` shape as the cascade.
  This is now the load-bearing comparison for "does the cascade+audit add
  value" — see §2.5.
- **Evidence quoting** — `src/stage3/explain.py`/`models.py`/`pipeline.py`.
  New `supporting_quote` + `quote_verified` fields (§2 above), enforced at
  the same level as `decision` in `_parse_response`. `verify_quote()` is a
  plain substring check (`supporting_quote.strip() in note_text`) — cheap by
  design, not a semantic entailment check; it catches fabricated quotes, not
  misleading-but-verbatim ones.
- **Planned-return question** — new `planned_return` field on the same
  three files, taxonomy `PLANNED_RETURN_ANSWERS = ("yes", "no",
  "not_stated")` (§2, §6 above).
- **Narrative docs rewritten** — `docs/THESIS_NARRATIVE.md`'s one-sentence
  summary, abstract, storyline options, and "if the professor asks" section,
  plus the paste-ready session-start prompt and N5 literature-positioning
  prompt in `docs/SANITY_CHECK_2026-07-06.md`, no longer describe a
  "high-recall screen" that a Stage 2 "second reader" "vetoes" — they now
  describe the capacity-constrained screen / independent second opinion /
  independent auditor design this file has described since session 15.
  Abstract numbers replaced with a retrain-pending placeholder rather than
  the stale recall-floor figures (AUROC 0.706, recall 0.85) they previously
  carried.
- Tests: `test_evaluate_pipeline.py`, `test_stage3_explain.py`,
  `test_stage3_batch.py` extended for the new fields. 121 passed, 1 skipped;
  pylint 10.00/10.

---

## 4. What is explicitly deferred (not done yet)

Everything in this section needs either GPU/compute time or an explicit
decision — the code to run once either is resolved already exists as of
session 17.

- **Retrain Stage 1** with the rebuilt feature matrix (400-trial Optuna,
  session-14 feature fixes, isotonic calibration) —
  `scripts/slurm_stage1_tune.sh` is ready.
- **Retrain Stage 2** at `max_seq_length=4096` — no script changes needed,
  just a fresh KISSKI/Grete job (plain Longformer only, per §2).
- **Run `predict_stage2_all` at full scale** once Stage 2 is retrained — the
  code exists (session 17) but has only been smoke-tested on a small
  `--limit` slice against the stale, pre-retrain model.
- **Run `compare_layers.py`** once both retrains + the full-scale
  `stage2_results_all.csv` exist, to produce the first real RQ1 numbers.
  Blocked on the headline-denominator decision below before the result is
  reported as "the" RQ1 answer, not on writing more code.
- **Run `batch.py` at full scale** once Stage 1 + Stage 2 are retrained, to
  produce real RQ2 numbers and let `evaluate_pipeline.py`'s `stage3` coverage
  block become non-empty.
- **Run `batch.py --sweep`** once the above exists, to actually validate
  `discordance_displacement_pp` rather than leave it at the provisional 20.
- **Wire and run the Layer 3 robustness arm** — blocked on the model-choice
  decision below, not on code.
- **N1 ablation runner** (5+ arms) beyond the two-arm RQ1 comparison
  `compare_layers.py` already covers — not yet written. The single most
  load-bearing arm, L1-at-matched-capacity, no longer needs a separate
  runner: `evaluate_pipeline.py:_control_arm_report` (session 18,
  2026-08-28) computes it inline as part of every pipeline evaluation.
- **Bootstrap CIs not yet in `evaluate_pipeline.py`** — `evaluate.py` and
  `compare_layers.py` have them; the cascade's full/notes-cohort pipeline
  numbers still don't.
- **`readmission_30d_unplanned` not yet used anywhere** — the column exists
  in the feature matrix but no script trains or evaluates against it yet.
- Feature audit (vitals missingness, lab itemid validation against
  `d_labitems`) — not touched.

---

## 5. Decisions still needed (blocking, not just deferred)

- **Which model runs Layer 3's robustness arm, and is a cloud API even
  permitted on this data?** MIMIC-IV/MIMIC-IV-Note are governed by a
  PhysioNet Data Use Agreement; sending credentialed data to a third-party
  cloud API is very likely restricted without a specific agreement.
  Recommendation: default to a larger *local* Ollama model, not a cloud API,
  unless the DUA is explicitly checked and permits it. `stage3.robustness_model`
  is deliberately left `null` pending this — do not set it without checking.
- **RQ1's headline comparison denominator.** Layer 2 only ever covers
  admissions with a note (~63% in past runs); Layer 1 covers 100%.
  Recommendation (per the remediation review's denominator-integrity point):
  the headline RQ1 number is both models restricted to the notes-covered
  subset (`stage1_notes_cohort` vs. `stage2_notes_cohort` in
  `comparison_rq1.json`) — the only population both models can be fairly
  compared on — with `stage1_full_population` reported as separate
  deployment context, not blended into the headline claim. Needs explicit
  confirmation before any number from `compare_layers.py` goes in the thesis
  text as "the" RQ1 result.
- **Conditional Stage 3 triggering (raised 2026-08-28, not implemented).**
  Currently `batch.py` audits every Stage 1-flagged, note-covered admission.
  An open question (not yet decided by the user, explicitly deferred pending
  their own evaluation) is whether Stage 3 should instead run only on
  discordant cases (`discordance_mode != CONCORDANT`) — cheaper, and arguably
  a cleaner story ("the auditor is called in when the two screens disagree"),
  at the cost of never getting an independent audit opinion on concordant
  flags to check whether the auditor would have agreed anyway. Do not
  implement without the user's explicit go-ahead.
- **Three more open architecture questions raised 2026-08-28, all
  explicitly left for the user to evaluate, not implemented:** whether
  Stage 1 and Stage 2 should be forced onto identical data splits for
  cleaner comparability (currently: same test partition, but Stage 2 is
  further restricted to the notes-covered subset within it); what exactly
  Stage 3 should be understood to reason "over" beyond the current
  score+SHAP+note+discordance-mode bundle (e.g. should it also see Stage 1's
  raw feature values, not just SHAP strings); and whether per-subgroup
  (e.g. age-band) context should be surfaced to Stage 3 explicitly rather
  than left implicit in the note text and scores.

---

## 6. Open methodological notes carried forward

- Stage 1's decision threshold is still selected on the same OOF/validation
  data used for early-stopping — a mild contamination risk flagged once in
  an earlier review and never revisited. Still open.
- `readmission_30d_unplanned` (label-level, `src/data/features.py`) only
  catches planned returns visible in `admission_type` (elective / same-day
  surgical) — a structured-data proxy. Planned returns not reflected there
  (e.g. informally scheduled follow-up admissions) are not caught there.
  Stage 3's `planned_return` field (added 2026-08-28, §2 above) now gives an
  LLM-extracted, note-based answer to the same question at audit time — but
  the two are not yet cross-checked against each other; whether they agree,
  and what to do when they don't, is unexamined.
- Truncation asymmetry between Stage 2 (4096 tokens) and Stage 3 (now a
  20,000-character safety cap, effectively near-full note) is much smaller
  than before session 15 but not eliminated for pathologically long notes.
- `training-changes` is still unmerged into `main` (44+ commits ahead as of
  session 15). Nothing in this doc addresses that — it's a repo-hygiene
  decision (merge vs. formally adopt as the working branch), not a code fix.
- Frontend (`frontend/src/`) still assumes the old "Stage 2 gates the
  patient list" model independently of the backend (see session 17 note in
  §3) — needs its own pass with an actual browser check, not covered here.
