# Architecture — Current State

**Last updated:** 2026-10-30 (session 34; §3 carries the most recent
content). This is the single current source of
truth for the pipeline design. It supersedes `docs/archive/IMPLEMENTATION_PLAN.md`,
`docs/THESIS_NARRATIVE.md`, and `docs/archive/SANITY_CHECK_2026-07-06.md` — the
archived two are kept for history (each has a banner pointing here) but
describe designs this project has since moved past; `THESIS_NARRATIVE.md`
stays in `docs/` (not archived) since its storytelling framing is still
needed for writing the thesis itself. If you are a new session picking up
this project, read this file, then the latest `sessions/` entry.

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

## 2. The three stages, as implemented now

### Stage 1 — XGBoost (structured screen)

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

### Stage 2 — Clinical-Longformer, note-only (independent note-based estimate)

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
  Reasons: (1) note-only is the cleaner independence story for Stage 3's
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
  "cascade" arm, but is **not** the final word on a patient — see Stage 3.

### Stage 3 — MedGemma-27B via HF transformers + lm-format-enforcer (independent auditor)

Rewritten 2026-08-25, extended 2026-08-28 (session 18) and 2026-08-28
(session 19) (`src/stage3/explain.py`, `src/stage3/pipeline.py`,
`src/stage3/models.py`). Serving switched from Ollama/phi4-mini to
`google/medgemma-27b-text-it` 2026-09-10 (session 23), first via vLLM, then
to plain HF `transformers.generate()` + `lm-format-enforcer` 2026-09-15
once KISSKI's CUDA 12.8 driver ceiling proved structurally incompatible
with vLLM's flashinfer/CUTLASS kernels regardless of vLLM/torch version —
see `config.yaml`'s `stage3.model_name` comment for the model/license
comparison and `sessions/` for the full serving-mechanism diagnosis.
Inputs: Stage 1's score + SHAP-ranked reasons,
Stage 2's score, the discharge note itself (near-full text, not a 5-sentence
attention summary), and a pre-computed discordance mode.

**Grounds taxonomy (session 19), replacing the single free-choice
`primary_clinical_domain`.** The LLM extracts a list of `mitigating_grounds`
and `aggravating_grounds`, each with its own verbatim quote, from two fixed
lists (`MITIGATING_GROUNDS`, `AGGRAVATING_GROUNDS` in `explain.py`) — fixed
list only; a ground outside the taxonomy, or one with an empty quote, fails
the whole response (don't let the model invent categories). Each quote is
independently checked against the note text (`verify_quote`, per ground);
`all_quotes_verified` is True only if every extracted quote verified.
`planned_return` remains a separate, always-answered field (yes/no/not_stated
per `PLANNED_RETURN_ANSWERS`) — independent of `"planned_return"` also
existing as a mitigating ground the model may cite when it drives the
decision; the two are not merged (also see `TARGET_COL_UNPLANNED` in
`src/schemas.py`/`src/data/features.py`, which computes the same distinction
at the label level, independently of Stage 3).

**Two decisions, not one (session 19).** `decision_model` is the LLM's own
independent uphold / override / insufficient_evidence judgment (not a
narration of Stage 2's — Stage 2's score is evidence the auditor reasons
over, not a decision it explains); this is what drives the final pipeline
prediction (`evaluate_pipeline.py:_apply_stage3_decisions`). `decision_rule`
is the same three-way decision recomputed **deterministically in code**
(`compute_decision_rule`) from the grounds the model itself extracted — not
asked of the model, not a replacement for its free judgment (§5 item 3a:
a small addition to the schema, not a switch to pure extraction), reported
alongside `decision_model` as a consistency metric and a fully transparent
fallback. `insufficient_evidence` (new third value) is a code-side judgment
about the note's length — a note below `_MIN_INFORMATIVE_NOTE_CHARS`
cannot ground either finding regardless of what was extracted from it.
**Critically, `decision_rule` is computed from quote-verified grounds
only** (`_finalize_annotation` filters on `quote_verified` before calling
`compute_decision_rule`) — a ground with a fabricated, non-verbatim quote
cannot drive it. Without this filter the "fully transparent, can't-be-
fooled fallback" claim would be false: a hallucinated ground could still
flip the deterministic decision, which is precisely the failure mode this
whole mechanism exists to rule out.

**Schema-constrained generation (session 19; serving mechanism changed
session 23, changed again 2026-09-15, same guarantee preserved each time).**
`call_llm` passes a JSON schema (`_LLMOutput.model_json_schema()`) as a
guided-decoding constraint — originally to Ollama's `format=` parameter,
then to vLLM's `GuidedDecodingParams`, now to `lm-format-enforcer`'s
`prefix_allowed_tokens_fn` hook into plain HF `transformers.generate()`
(the vLLM→lm-format-enforcer switch was forced by KISSKI's CUDA 12.8
driver ceiling being structurally incompatible with vLLM's
flashinfer/CUTLASS kernels — see `sessions/` for the diagnosis; not the
generic `format="json"`) — this is what nearly eliminates malformed-JSON
parse failures. The prompt
was also reordered so the model emits
grounds/evidence *before* `decision` (previously `decision` was asked first)
— an autoregressive model conditions on what it has already written, so
asking for decision first invited the model to invent justification to fit
a decision already reached.

`note_truncated` and `model_name` (session 19) are logged on every row —
required groundwork for the (still-blocked, see §5) model-scale robustness
arm, and for characterising the truncation-asymmetry question in §6.

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

- **RQ1 — does text carry signal at all?** Stage 1 vs. Stage 2, scored
  independently on the *same* population (not gated by Stage 1's flag) and
  compared head-to-head. `src/stage2/predict.py:predict_stage2_all` produces
  the population-wide Stage 2 scores this needs;
  `src/model/compare_stages.py` does the comparison. Neither model feeds the
  other here — this is not the cascade.
- **RQ2 — does the auditor add value over Stage 1 alone?** Stage 3 audits
  Stage 1's positive predictions using Stage 2's score as evidence.
  `src/stage3/batch.py:run_batch_audit` runs this at scale;
  `src/model/evaluate_pipeline.py` reports the resulting pipeline metrics
  with Stage 3's decision (not Stage 2's threshold) as the final prediction
  wherever a batch audit covers the admission. The load-bearing comparison
  (colleague review item 1, 2026-08-28) is against a **control arm**:
  Stage 1 alone, its threshold raised to match the full system's alert
  volume exactly (`evaluate_pipeline.py:_control_arm_report`, using
  `select_threshold_for_capacity`) — not an unconstrained Stage 1. Without
  this, an apparent precision gain from the cascade could just be Stage 1
  flagging fewer admissions, which raising its own threshold would do for
  free. Reported as `report["pipeline"]["control_arm_stage1_matched"]`.
  `run_batch_audit` always audits **every** Stage 1-flagged, note-covered
  admission (blanket, unconditional) — this was raised as an open question
  (should Stage 3 only run on discordant cases?) and resolved 2026-08-28
  (session 19, see §5) as: keep the blanket run, and answer the question as
  a **post-hoc analysis** of a completed run instead
  (`evaluate_pipeline.py:_conditional_triggering_report`,
  `report["pipeline"]["conditional_triggering"]`) — this reports the
  LLM-call count and metric delta conditional (discordant-only) triggering
  would have cost/saved, without ever forfeiting the ability to check
  whether the auditor would have agreed on concordant cases anyway.

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

- **Population-wide Stage 2 scoring** — `src/stage2/predict.py`. New
  `predict_stage2_all` (shares a refactored `_score_core`/`_prepare_population`
  pipeline with the existing `predict_stage2`), scores every test-partition
  admission with a note, not just Stage 1's positives. Output
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
- **`api.py` stopped gating on Stage 2's threshold.** `list_patients`'s
  `confirmed_only` default flipped `True`→`False`; docstrings updated. Stage
  2's score is evidence for Stage 3, not a display gate, per §2.
  **Follow-up found, not fixed:** the frontend (`frontend/src/api.ts`) has
  its own independent `confirmedOnly = true` default and UI copy built
  around "confirmed" patients (`PatientTable.tsx`, `PipelineDiagram.tsx`) —
  changing the backend default didn't change frontend behaviour. Needs an
  actual frontend change plus a browser check, not a blind edit; not done
  this session.
- **N1 comparison runner (RQ1)** — new `src/model/compare_stages.py`.
  Inner-joins Stage 1's test-partition scores with
  `stage2_results_all.csv` on `hadm_id`, reports both models' AUROC/AUPRC
  (bootstrap 95% CI, patient-level) and an operating point matched to Stage
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
- **Stage 3's model call generalised** — `call_phi4mini` → `call_llm(prompt,
  cfg, model_name=None)` in `src/stage3/explain.py`; `explain_patient` gained
  a matching `model_name` passthrough. New `stage3.robustness_model` config
  field (default `null` — a real open decision, not a default to assume; see
  §5). Makes the scale-robustness arm wireable without further code changes
  once a model is chosen.
- **`evaluate_pipeline.py`'s final prediction now prefers Stage 3's
  decision.** New `_apply_stage3_decisions`: wherever
  `stage3_batch_results.csv` has a decision for an admission, it replaces
  `stage2_confirmed` (+ C9 fallback) as the final prediction; admissions
  Stage 3 hasn't covered keep the prior C9-corrected value unchanged — this
  only ever narrows the gap to the design, never drops coverage. Report now
  includes a `stage3` block reporting coverage.
- Tests: `test_compare_stages.py`, `test_stage3_batch.py`,
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
  prompt in `docs/archive/SANITY_CHECK_2026-07-06.md`, no longer describe a
  "high-recall screen" that a Stage 2 "second reader" "vetoes" — they now
  describe the capacity-constrained screen / independent second opinion /
  independent auditor design this file has described since session 15.
  Abstract numbers replaced with a retrain-pending placeholder rather than
  the stale recall-floor figures (AUROC 0.706, recall 0.85) they previously
  carried.
- Tests: `test_evaluate_pipeline.py`, `test_stage3_explain.py`,
  `test_stage3_batch.py` extended for the new fields. 121 passed, 1 skipped;
  pylint 10.00/10.

**Session 19 (2026-08-28):** evaluated a colleague's independent pipeline
restructuring proposal against the current implementation (most of it
matched what session 18 had already built, or restated §5's already-decided
items); built the BUILD-NOW items from that evaluation, plus two decisions
made this session (keep blanket Stage 3 triggering + analyse conditional
triggering post-hoc; keep K=0.15 primary) — see §5.

- **Grounds taxonomy replacing `primary_clinical_domain`** —
  `src/stage3/explain.py`: `MITIGATING_GROUNDS` (4) / `AGGRAVATING_GROUNDS`
  (6), each extracted ground with its own quote, independently verified.
  Fixed list only — an invalid ground or empty quote fails the whole
  response, same enforcement level as before.
- **`decision_model` + `decision_rule`** — the LLM's own decision, reported
  alongside a deterministic recomputation from the extracted grounds
  (`compute_decision_rule`). Adds `insufficient_evidence` as a third
  decision value (code-side, based on note length). `evaluate_pipeline.py:
  _apply_stage3_decisions` updated to read `decision_model` and treat
  `insufficient_evidence` as no-coverage, not a silent uphold.
- **Schema-constrained Ollama generation** — `call_llm` now passes a JSON
  schema (`format=_LLMOutput.model_json_schema()`), not `format="json"`.
- **Prompt reordered** — grounds/evidence requested before `decision`,
  fixing a real autoregressive-conditioning issue in the prior prompt
  (decision was asked first, evidence after).
- **`note_truncated` / `model_name` logged per row** — groundwork for the
  still-blocked model-scale robustness arm.
- **Conditional-triggering post-hoc analysis** —
  `evaluate_pipeline.py:_conditional_triggering_report` (§2, §5 above).
  Does not change `run_batch_audit`'s targeting.
- **Two validation controls (Phase D1/D2)** — `src/stage3/batch.py:
  run_blind_note_control`, `run_no_stage2_control`; `explain_patient` gained
  `suppress_note`/`suppress_stage2` params, `build_prompt` gained
  `hide_stage2`. Diagnostic functions, not wired into the default batch path.
- **Self-agreement check (Phase D3)** — `src/stage3/batch.py:
  check_self_agreement`, calls `explain_patient` twice per admission and
  reports exact-match agreement on `decision_model`/`decision_rule`/grounds.
- **Selection-bias table gap identified, not yet built** — no
  included-vs-excluded (notes-covered vs. not) baseline-characteristics
  table exists anywhere in the repo; flagged as a genuine gap during the
  colleague-proposal evaluation, not yet implemented — add to §4.
- `frontend/src/types.ts`, `PatientModal.tsx` updated to match the new
  schema (grounds lists, two decisions, `all_quotes_verified`).
- Tests: `test_stage3_explain.py`, `test_stage3_batch.py`,
  `test_evaluate_pipeline.py` extended/rewritten for the new schema and new
  functions. 143 passed, 1 skipped; pylint 10.00/10; frontend `tsc --noEmit`
  clean.

**Session 31 (2026-10-26):** closed out every open §4 item that needed no
GPU/cluster time — pure local analysis on the already-committed Stage
1/2/3 result files, after a freshness audit confirmed those files are
internally consistent and match every number already committed in
`MODEL_CARD.md`.

- **Selection-bias table built** (`scripts/analyze_selection_bias.py`,
  new) — see §4, now resolved.
- **`planned_return` cross-check built** (`scripts/
  analyze_planned_return_crosscheck.py`, new) — see §6, now resolved.
  **Not the same thing as the human-annotator validation plan in
  `docs/STAGE3_VALIDATION.md`** — that's a separate, not-yet-executed
  Cohen's-kappa protocol against an independent clinical annotator; this
  is a code-only cross-check against the structured label.
- **Real performance bug found and fixed in `src/model/bootstrap.py`,
  not something gone looking for — it blocked the next item.**
  `bootstrap_ci`'s resampling loop called `np.where(group_of == g)` once
  per *sampled patient*, per resample — O(n_unique_patients × n_rows)
  work repeated `n_resamples` times. At this project's real scale (tens
  of thousands of unique patients, 1000 resamples), re-running
  `evaluate_pipeline.py` to add the two missing CIs below ran 39+ minutes
  without finishing. Fixed by precomputing each patient's row positions
  once via `pandas`' Cython groupby before the resample loop, then doing
  O(1) dict lookups inside it — **verified bit-for-bit identical output**
  against the old algorithm on a controlled case, then benchmarked at
  realistic scale: 24 seconds instead of 39+ unfinished minutes. Speeds
  up every bootstrap CI in the codebase, not just the two below.
- **Bootstrap CIs for `notes_cohort` and `conditional_triggering`** — the
  infra already existed and was already wired into `full_cohort`/
  `control_arm_stage1_matched` (two `groups=` pass-throughs were
  missing). Reran the full evaluation post-fix and confirmed every
  already-reported number (`full_cohort`, `control_arm`) is
  byte-identical to the pre-fix run — pure speed + coverage addition, no
  change to any previously-reported result.
- **D1/D2/D3 validation-control CLI entry point built**
  (`scripts/run_validation_controls.py` + `scripts/
  slurm_stage3_validation.sh`, new). Session 19's `run_blind_note_control`
  / `run_no_stage2_control` / `check_self_agreement` (`src/stage3/
  batch.py`) existed as callable functions but were never wired to an
  actual entry point or run against the real model — this is that entry
  point, on a sample (default n=150), writing incrementally and
  resumable like the main batch run. **Submitted to KISSKI
  (job 16873264) but the D1 blind-note control ran ~4 hours on a single
  patient without completing — 607.5 core-hours spent, zero patients
  written** (cancelled, not a crash: `nvidia-smi` showed 100% GPU
  utilization throughout, so it was genuinely computing, not hung/idle —
  model loading itself took only 95 seconds per the job's own log).
  ~~Leading hypothesis, not yet confirmed: the schema-constrained
  decoder may behave pathologically when asked to extract quotes from a
  note that says "not available."~~ **Wrong — see session 34. The job
  was never stuck; it completed successfully. A logging bug made it
  look stuck.**

**Session 34 (2026-10-30):** resolved the D1/D2/D3 validation controls
end to end, after two more rounds of cluster troubleshooting.

- **The "4-hour stall" from session 31 was never a stall.** An isolated,
  single-patient timing test (job 16947546) ran all three conditions
  cleanly: NORMAL=212.1s (includes a one-time ~87s model load),
  D2 no-Stage-2=109.4s, D1 blind-note=69.5s — blind-note was the
  *fastest* of the three and correctly returned `insufficient_evidence`.
  Scaled the real run down to `--n 30` out of caution
  (`scripts/slurm_stage3_validation.sh`) and resubmitted (job 16948923).
  That job's log showed `D1 blind-note: 30 target, 150 already done, 0
  pending` — the original, supposedly-stuck job 16873264 had in fact
  completed all 150 D1 patients and 40 D2 patients before being
  cancelled. **Root cause: `scripts/run_validation_controls.py`'s
  progress `print()` calls had no explicit `flush`.** Redirected to a
  SLURM log file (not a terminal), Python block-buffers stdout; the
  per-patient `[i/n]` checkpoints were sitting in that buffer, never
  written to disk, when `scancel` killed the process without a clean
  exit. The CSV data itself was never at risk (the CSV writer was
  flushed explicitly after every row) — only the human-readable log was
  invisible. Fixed: `flush=True` added to every `print()` call in that
  script. 607.5 core-hours were not wasted; they produced 150 real D1
  rows and 40 real D2 rows that job 16948923 then reused instead of
  recomputing.
- **Real D1/D2/D3 results** (`models/stage3_validation_controls_summary.json`):
  - **D1 blind-note (n=150): 0% decision agreement with the real run,
    and that is the best possible outcome, not a red flag.** Every
    single one of the 150 blind-note responses was
    `insufficient_evidence`; the real run (same patients, note present)
    never used `insufficient_evidence` at all (104 uphold, 46 override).
    The auditor does not guess uphold/override when it has nothing to
    read — it reliably recognizes that and says so. This is the
    strongest available evidence that Stage 3's real decisions are
    actually grounded in the note, not just the two numeric scores.
  - **D2 no-Stage-2 (n=40): 82.5% agreement.** Removing Stage 2's score
    and the discordance signal flips the decision on 7 of 40 patients,
    shifting toward more uphold calls when Stage 2 is hidden (27→30
    uphold, 13→10 override). Stage 2's evidence earns its place in the
    prompt to a real, moderate degree without dominating it.
  - **D3 self-agreement (n=30): 100% agreement** on both
    `decision_model` and `decision_rule` across two independent runs of
    every patient. `temperature=0` delivers exactly what it was pinned
    for — a fixed, reproducible decision, not a sample from a
    distribution. Verifies, rather than just assumes, the premise the
    entire RQ2 design rests on.
  - **Taken together: the net-harmful RQ2 finding is not an artifact of
    a broken or inattentive auditor.** Stage 3 demonstrably reads the
    note, demonstrably uses Stage 2's evidence, and is demonstrably
    reproducible — and still underperforms Stage 1 alone. That is a
    materially stronger, more defensible claim than an unvalidated
    negative result would have been.

---

## 4. What is explicitly deferred (not done yet)

Everything in this section needs either GPU/compute time or an explicit
decision — the code to run once either is resolved already exists as of
session 17.

- ~~Retrain Stage 1~~ **Done (2026-09-05, session 22/23)** — 400-trial
  Optuna search, unplanned-readmission target, AUROC 0.7215. See
  `MODEL_CARD.md`.
- ~~Retrain Stage 2~~ **Done (2026-09-09/10, session 23)** — 4096 tokens,
  corrected age-group oversampling, unplanned target. See `MODEL_CARD.md`.
- ~~Run `predict_stage2_all` at full scale~~ **Done (session 23)** — real
  population-wide scores exist in `models/stage2_results_all.csv`.
- ~~Run `compare_stages.py`~~ **Done (session 23)** — first real RQ1
  numbers: Stage 1 AUROC 0.7093 vs. Stage 2 AUROC 0.7101, a null result
  (see `MODEL_CARD.md`). The headline-denominator decision below (notes-
  covered subset as headline) still needs explicit confirmation before this
  is written up as "the" RQ1 answer in the thesis text, but the numbers
  themselves are real and current.
- ~~Run `batch.py` at full scale~~ **Done (2026-10-06)** — all 9,899
  flagged+noted admissions, 0.94% annotation failure rate (93 patients,
  token-budget truncation, fall back to Stage 2's threshold). Real RQ2
  result: at matched alert volume, Stage 1 alone (control arm) beats the
  full cascade on both precision (0.450 vs 0.435) and recall (0.316 vs
  0.295, 95% CIs effectively non-overlapping on recall) — blanket Stage 3
  auditing is net harmful to screening performance, not neutral. A
  post-hoc conditional-triggering re-analysis (Stage 3 only on discordant
  cases) recovers both metrics versus the blanket run at 48% fewer LLM
  calls, which initially read as "Stage 3 adds value on discordant cases" —
  **but that comparison never varied Stage 3's treatment of the discordant
  group, so it couldn't have tested that.** A direct follow-up
  (`scripts/analyze_discordance_subgroups.py`, 2026-10-07): Stage 3's real
  decision vs. "Stage 1's flag stands," within each subgroup separately,
  with CIs. By F2, Stage 3 underperforms Stage 1 alone in BOTH subgroups,
  and the drop is larger on discordant cases (−0.168) than concordant
  (−0.130) — see `MODEL_CARD.md` for the full table. The correct
  statement: Stage 3 trades recall for little-to-no precision gain almost
  everywhere it's applied; removing it from concordant cases is a clean
  win, but it is not adding value on discordant cases either.
- ~~Run `batch.py --sweep`~~ **Done (2026-10-06)** —
  `models/discordance_sensitivity.json`: at the provisional ±20pp,
  CONCORDANT=47.6%, NOTE_MITIGATES=29.8%, NOTE_AMPLIFIES=22.7%. Sweep
  across ±10-30pp shows the mode split is meaningfully sensitive to the
  threshold choice (CONCORDANT ranges 27.0%-65.2%), worth discussing as a
  robustness caveat rather than treating ±20pp as uniquely correct.
- ~~Wire and run the Stage 3 robustness arm~~ **Decided against, 2026-10-30
  — see §5.** Not a deferred item anymore; closed.
- **N1 ablation runner** (5+ arms) beyond the two-arm RQ1 comparison
  `compare_stages.py` already covers — not yet written. The single most
  load-bearing arm, L1-at-matched-capacity, no longer needs a separate
  runner: `evaluate_pipeline.py:_control_arm_report` (session 18,
  2026-08-28) computes it inline as part of every pipeline evaluation.
- ~~Bootstrap CIs not yet in `evaluate_pipeline.py`~~ **Done (2026-10-26,
  session 31)** — `_add_precision_recall_ci` (patient-clustered, 1000
  resamples) now covers all four pipeline arms, including `notes_cohort`
  (precision=0.433 [0.416, 0.450], recall=0.180 [0.168, 0.193]) and
  `conditional_triggering` (precision=0.437 [0.420, 0.457],
  recall=0.319 [0.301, 0.339]) — fixing a real resampling-performance bug
  in `bootstrap.py` along the way (see §3, session 31).
- Feature audit (vitals missingness, lab itemid validation against
  `d_labitems`) — not touched.
- ~~Included-vs-excluded (notes-covered vs. not) selection-bias table~~
  **Done (2026-10-26, session 31)** — `scripts/analyze_selection_bias.py`.
  Real result: the notes-covered subset of Stage 1's test partition
  (60.2%, n=62,759) is clinically distinct from the excluded 39.8%
  (n=41,483) — older (59.7 vs. 52.1), sicker (Charlson 2.24 vs. 1.43),
  far more often an emergency admission (SMD=0.73, the largest gap by a
  wide margin), more often discharged to a facility — but has a nearly
  balanced outcome rate (19.8% vs. 17.8%, SMD=0.053, below the
  conventional 0.1 imbalance flag). RQ1/RQ2 are evaluated on a population
  skewed toward older, sicker, emergency-admitted patients — a real scope
  caveat for the thesis text — but that skew is not an outcome-rate
  confound.

---

## 5. Decisions still needed (blocking, not just deferred)

**Resolved 2026-08-28 (session 19):**

- **Conditional Stage 3 triggering.** Kept blanket (audit every flagged,
  note-covered admission) as the execution mode — `run_batch_audit`'s
  targeting logic is unchanged. Discordant-only triggering is answered
  instead as a post-hoc analysis of a completed blanket run
  (`_conditional_triggering_report`, §2 above): committing to conditional
  triggering as the design would mean never learning what auditing
  concordant cases found, and no full batch run has happened yet even
  once, so there was no cost being saved by skipping it now. The
  cost/benefit trade-off is reported as a measured result, not assumed.
- **Operating point K.** Stays at `capacity_k = 0.15` (primary); 20%/10%/5%
  remain secondary `capacity_report_points`, unchanged. A colleague's
  proposal to make K=20% primary had no comparable justification to the
  clinical-capacity argument behind the existing 0.15 choice (session 15) —
  not changed without new operational evidence.

- ~~Which model runs Stage 3's robustness arm, and is a cloud API even
  permitted on this data?~~ **Resolved 2026-10-30 (user decision): not
  pursued.** A robustness arm means a full second Stage 3 batch run
  against a different (larger) model — not a small follow-up analysis.
  The original full run across all 9,899 flagged-and-noted admissions
  took 6 days of wall-clock cluster time. A second run at that cost is
  not affordable within the thesis timeline, and nothing about the
  already-validated D1/D2/D3 results (session 34) suggests the primary
  model is behaving unreliably in a way that would make a second model
  the more urgent use of remaining cluster time. The cloud-API question
  (MIMIC-IV/MIMIC-IV-Note are governed by a PhysioNet Data Use
  Agreement; sending credentialed data to a third-party cloud API is
  very likely restricted without a specific agreement) is moot now that
  the arm itself is not being run. `stage3.robustness_model` stays
  `null`; this is a closed decision, not a deferred one.
**Resolved 2026-10-30 (user decision, after reviewing session 31's new
findings):**

- **RQ1's headline comparison denominator.** Confirmed: the headline RQ1
  number is both models restricted to the notes-covered subset
  (`stage1_notes_cohort` vs. `stage2_notes_cohort` in `comparison_rq1.json`
  — AUROC 0.7093 vs. 0.7101) — the only population both models can be
  fairly compared on, since comparing each model on a different
  population isn't a comparison at all. `stage1_full_population` (AUROC
  0.7215) is reported as separate deployment context, never blended into
  the headline claim. This matches how `README.md`/`MODEL_CARD.md` already
  present the numbers — this confirms that presentation as the deliberate,
  decided choice, not a provisional default awaiting sign-off.
- **Should Stage 1 and Stage 2 be forced onto identical data splits?**
  Confirmed: no — the current setup (same held-out test partition; Stage
  2's training data further restricted to the notes-covered subset within
  Stage 1's *training* patients) is the correct design, not an open
  question. Restricting Stage 1's own training data to notes-covered
  admissions only would have thrown away a large fraction of real training
  data for no benefit — Stage 1 doesn't need a note to train or predict.
  The actual fairness requirement is that both models are *evaluated* on
  the exact same test admissions, which they already are; each model
  trains on the most data it can legitimately use, while the comparison
  itself stays apples-to-apples.
- **What should Stage 3 reason over, beyond the current
  score+SHAP+note+discordance-mode bundle?** Confirmed: leave as-is — do
  not add Stage 1's raw feature values. Two reasons: (1) giving the LLM
  all ~40 raw structured features, instead of the top-5 SHAP-ranked,
  direction-and-magnitude-labeled strings already provided, would dilute
  the prompt with mostly-irrelevant structured noise on top of near-full
  note text already competing for the model's attention under a hard
  token budget — more raw input is not obviously more signal; (2) more
  importantly, this risks drifting back toward the exact design this
  project already rejected in session 11 (§1 above): an LLM narrating a
  structured model's own SHAP values, "a well-trodden pattern with no
  novel finding." SHAP-ranked strings keep Stage 3 doing what it's
  actually for — an independent audit grounded in the clinical note — not
  a second explainer of Stage 1's decision.
- **Should per-subgroup (e.g. age-band) context be surfaced to Stage 3
  explicitly?** Confirmed: no — not implemented, and deliberately not
  pursued. Stage 1 (isotonic calibration, fairness subgroup reporting) and
  Stage 2 (age-group oversampling, per-age-group loss weights, per-band
  Platt calibration and thresholds) already carry explicit, targeted
  bias-mitigation machinery. Layering a systematic age-band context onto
  Stage 3 as well risks compounding those corrections in an uncontrolled
  way, or introducing a new vector for inconsistent treatment (the auditor
  reasoning differently about identical clinical content depending on a
  stated age band), rather than reducing bias further. One honest caveat:
  `age` is one of Stage 1's ~40 features, so it can appear *incidentally*
  in a given patient's top-5 SHAP block if age happens to be a top risk
  driver for that specific admission (`src/stage3/shap_extract.py`'s
  `_FEATURE_LABELS`) — a side effect of SHAP ranking, not the deliberate,
  systematic age-band framing this question is about.

---

## 6. Open methodological notes carried forward

- Stage 1's decision threshold is still selected on the same OOF/validation
  data used for early-stopping — a mild contamination risk flagged once in
  an earlier review and never revisited. Still open.
- `readmission_30d_unplanned` (label-level, `src/data/features.py`, now the
  model's actual target via `MODEL_TARGET_COL` — see below) only catches
  planned returns visible in `admission_type` (elective / same-day
  surgical) — a structured-data proxy, not CMS's full ICD-procedure/
  diagnosis-code-based Planned Readmission Algorithm. Planned returns not
  reflected in `admission_type` (e.g. informally scheduled follow-up
  admissions) are not caught. This is not an ad hoc simplification, though
  — it matches the operationalization used by at least one peer-reviewed
  MIMIC-IV benchmark pipeline (Extensive Data Processing Pipeline for
  MIMIC-IV, arXiv:2204.13841) and informal MIMIC community convention
  (MIT-LCP/mimic-code discussion #1215), and Rajkomar et al. 2018 (the most
  rigorous "unplanned" precedent in this project's own citation list, see
  MODELING_PLAN.md) explicitly states there is no single standard
  definition in the field. Separately, and NOT addressed by this proxy
  either way: this project's cohort does not exclude OBSERVATION-type
  stays from the readmission count at all, whereas CMS's measure excludes
  them from the denominator entirely (they aren't billed inpatient
  admissions) — a scope question distinct from the planned/unplanned
  distinction, disclosed here rather than fixed, since resolving it would
  mean re-deriving the cohort definition, not just the label (confirmed
  2026-09-04, alongside the switch of the model's actual target from
  all-cause to this column). Stage 3's `planned_return` field (added
  2026-08-28, §2 above) now gives an LLM-extracted, note-based answer to
  the same question at audit time. ~~The two are not yet cross-checked
  against each other~~ **Done (2026-10-26, session 31,
  `scripts/analyze_planned_return_crosscheck.py`) — and the finding is
  not reassuring.** Restricted to Stage-3-audited patients who were
  actually readmitted within 30 days (n=4,375, the only population where
  "was the readmission planned" has a structured answer at all): raw
  agreement is 81.7%, but a trivial "always predict unplanned" baseline
  already scores 95.8% (structured-planned rate is only 4.2%) — **Stage
  3's actual agreement is below the trivial baseline.** Stage 3 calls
  "planned" 4.5× more often than the structured proxy does (19.2% vs.
  4.2%); of Stage 3's 842 "yes" calls, only 112 (13.3%) match the
  structured proxy. Reads as the LLM being substantially more liberal
  about reading a note as describing a planned return than the
  structured data would license — worth stating explicitly as an open
  discrepancy between the two signals in the thesis text, not quietly
  citing the 81.7% number alone.
- `MODEL_TARGET_COL` (`src/schemas.py`) is the single switch controlling
  which label column (`TARGET_COL`/all-cause vs `TARGET_COL_UNPLANNED`)
  every training/eval script actually uses — set to unplanned as of
  2026-09-04. Before that, every Stage 1 model this project ever produced
  (including the original artifact and several retraining attempts)
  silently trained on all-cause despite every scope-level doc
  (`MODELING_PLAN.md`, `THESIS_NARRATIVE.md`, `MODEL_CARD.md`) stating the
  study targets unplanned readmission — caught during a pre-flight audit,
  not before. If retraining ever needs to compare against the all-cause
  label again, change only this one constant; every consumer (Stage 1's
  `split_xy()`, every Stage 2 module) follows it automatically.
- Truncation asymmetry between Stage 2 (4096 tokens) and Stage 3 (now a
  20,000-character safety cap, effectively near-full note) is much smaller
  than before session 15 but not eliminated for pathologically long notes.
- `training-changes` is still unmerged into `main` (44+ commits ahead as of
  session 15). Nothing in this doc addresses that — it's a repo-hygiene
  decision (merge vs. formally adopt as the working branch), not a code fix.
- Frontend (`frontend/src/`) still assumes the old "Stage 2 gates the
  patient list" model independently of the backend (see session 17 note in
  §3) — needs its own pass with an actual browser check, not covered here.
