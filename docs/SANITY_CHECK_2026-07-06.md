> **Superseded 2026-08-25.** Most C1-C12/N1-N8 items here predate the Stage 2
> fairness rebuild, the Stage 3 rewrite, and the capacity-constrained
> operating point. See `docs/ARCHITECTURE.md` for current state and
> `sessions/2026-08-25_session-15.md` for what changed. Kept for history —
> useful background on why decisions were made, not a current task list.

# Thesis Sanity Check & Working Backlog — Revision 2 (2026-07-06)

> **Purpose.** A single honest review of (1) where the code actually stands, (2) whether the
> research narrative holds up, and (3) a concrete, prioritised backlog. Every problem has a
> ready-to-paste **AI prompt** in the Prompt Library so any teammate or agent can pick it up
> with full context.
>
> **Revision 2 changes.** Aligned every task and prompt with the narrative strategy in
> `docs/THESIS_NARRATIVE.md` (alert-fatigue lead · second-reader framing · disagreement
> analysis as star exhibit · alerts-per-true-catch metric · strict denominator honesty).
> Updated status: Stage 1 LR/HGB comparison is **in progress** (Martin); Stage 2 rebuild
> training **failed on a teammate's laptop for lack of compute** — a GPU plan with a
> documented fallback is now part of C2. New tasks added: **M1** (alert-burden metrics),
> **W1–W4** (writing/story infrastructure), **N8** (temporal validation), **F1** (frontend
> sync). Prompts now instruct agents to read `THESIS_NARRATIVE.md` and log sessions.
>
> **How to use.** Work top-to-bottom by tier. Each task has an ID (`C…` code, `N…` narrative
> /science, `M…` metrics, `W…` writing, `F…` frontend). One prompt per ID at the bottom.

---

## Part A — Status quo (what is real vs. what is claimed)

### What exists in code (mature)
- **Stage 1** (`src/model/`): LR / XGBoost / HistGradientBoosting, Optuna tuning on CV AUPRC,
  patient-level grouped splits, threshold-for-recall, fairness subgroup AUROC. Pydantic v2
  config, pylint 10/10.
- **Stage 2** (`src/stage2/`): *rebuilt* Clinical-Longformer pipeline — 2048 tokens, focal
  loss, age×label stratified sampling, per-age-group loss weights, Platt calibration,
  fairness eval.
- **Stage 3** (`src/stage3/`): Ollama / phi4-mini explanation generator with a guarded prompt.
- **Frontend** (`frontend/`): React dashboard (pipeline explainer + patient table, mock data).
- **Narrative strategy**: `docs/THESIS_NARRATIVE.md` — framing options, chapter red thread,
  abstract for Prof. Shen.

### What has actually been *run*
| Component | Claimed in MODEL_CARD | Reality |
|---|---|---|
| Stage 1 XGBoost (real, 521k) | AUROC 0.706, recall 0.848 | Ran on Lennart's machine; **not reproducible locally** (local artifact is the June-24 synthetic quick-mode model; `data/processed/` empty). Manifest missing. |
| Stage 1 LR / HGB comparison | — | **In progress (Martin, real data, full mode).** |
| Stage 2 rebuilt (45k notes, 2048 tok) | — | **Coded, training ATTEMPTED and ABORTED** — a teammate's laptop lacked the compute (CPU-infeasible at 2048 tokens). Needs a GPU plan or a documented reduced-scale fallback (see C2). |
| Stage 2 v1 numbers | AUROC 0.640, thr₂=0.3, +21% precision | Real but produced by the **old v1** model (15k notes, 512 tok) that current code no longer implements. |
| Stage 3 | pipeline "ready" | **Never run**; frontend uses mock patients. |

**Bottom line:** unchanged from Rev 1 — the code is ahead of the evidence. The two active
work fronts (Stage 1 comparison; Stage 2 GPU training) are exactly the right ones. Everything
else below turns those numbers into a defensible thesis that tells the story we chose.

---

## Part B — Code issues

### Strengths (keep)
Patient-level grouped splitting everywhere · no-leakage discipline (per-fold preprocessing,
OOF thresholding, test-once) · validated config schema · synthetic fallback · fairness-aware
Stage 2 rebuild · disciplined session logs.

### Issues

| ID | Severity | Status | Issue |
|----|----------|--------|-------|
| **C1** | High | open | No reproducible end-to-end run; no results manifest tying data version → git commit → numbers. Narrative claims (abstract!) must be traceable. |
| **C2** | High | **blocked on GPU** | Rebuilt Stage 2 untrained; training aborted on a laptop. Need: (a) cloud-GPU execution plan, (b) a *documented reduced-scale fallback* that fits available hardware, (c) MODEL_CARD refresh with v1 clearly deprecated. |
| **C3** | High | **in progress** | Stage 1 LR/HGB runs under way. Still needed: the comparison table + CIs + identical-split verification, written up for §3.3. |
| **C4** | High | open | No bootstrap CIs, no DeLong test, no Stage 1 calibration curve. The abstract's "+21% precision" needs an uncertainty statement. |
| **C5** | Medium | open | No interpretability output (SHAP/coefficients). Feeds the "second reader you can question" story and §3.3. |
| **C6** | High | open | Vitals from `icu/chartevents` are ICU-only → unquantified missingness for non-ICU admissions; lab itemids unvalidated against `d_labitems`. |
| **C7** | Medium | open | Feature cache is mode-blind (already bit us once: synthetic served to a real run). |
| **C8** | Medium | open | No tests (leakage assertion, feature build, metrics, Charlson). |
| **C9** | High | open | No cascade policy for flagged patients **without notes** (~37%); full-cohort metrics unreported. Core to denominator honesty (N3). |
| **C10** | Medium | open | No EDA artifacts; blocks §3.2 and the funnel figure (W3). |
| **C11** | Low | open | `cohort.py` stub vs. label logic in `features.py`. |
| **C12** | Low | open | Fragile macOS joblib/torch ordering workaround. |

---

## Part C — Narrative & science gaps

The chosen storyline (see `THESIS_NARRATIVE.md`): **alert fatigue** is the problem; the LLM
is a **second reader** that vetoes false alarms before they reach a human; the science is in
**what the notes know that the labs don't**; supporting themes: all-local deployment and the
self-critical **fairness rebuild**. Every gap below is an ingredient of that story.

| ID | Status | Gap |
|----|--------|-----|
| **N1** | open | Cascade vs. joint/fusion model unjustified → the **ablation grid** is "the one table that settles it" (Results centrepiece). |
| **N2** | open | **Disagreement analysis** — the star exhibit. Where and why does the note overturn the numbers? |
| **N3** | open | Denominator integrity: every metric must state full-cohort AND notes-cohort, with coverage %. The abstract's honesty depends on it. |
| **N4** | open | Clinical utility unquantified → decision-curve / net-benefit + **alerts-per-true-catch** (see M1) express "so what" in clinician units. |
| **N5** | open | Novelty positioning (§2.4): verification layer vs. structured-only / notes-only / fusion literature. Now anchored by the narrative doc. |
| **N6** | open | Stage 3 unevaluated (faithfulness). Feeds the "make every alert explain itself" close + the expert interviews. |
| **N7** | open | Prediction time-point statement + automated leakage audit (at-discharge framing). |
| **N8** | new | **Temporal validation** — train on earlier `anchor_year_group`s, test on later. Supports the deployability claim; cheap robustness win. |

### New infrastructure tasks (narrative & delivery)

| ID | What |
|----|------|
| **M1** | Implement **alert-burden metrics** as first-class outputs: alerts-per-true-catch (1/precision), total alert volume, % alerts removed by Stage 2, TPs retained — computed in evaluate scripts, wired into MODEL_CARD/README/frontend. The story's recurring number. |
| **W1** | Thesis **chapter skeleton** in `nova_sbe_thesis.tex` mapped to the red thread (chapter-opening questions, section stubs, figure placeholders), respecting team section ownership. |
| **W2** | The **composite patient vignette** ("Mrs. M.") — fictional, clinically plausible, consistent with our feature set; versions for intro/methods/results/discussion. |
| **W3** | The **reusable funnel diagram** (schematic → annotated → populated) as a publication-quality figure, consistent with the frontend's visual language. |
| **W4** | **Claims traceability audit**: every number in the abstract, README, MODEL_CARD, frontend, and thesis text traces to the results manifest (C1). One table: claim → source file → run ID. |
| **F1** | **Frontend sync**: replace mock metrics/patients with real pipeline outputs (after C2/N6), add alert-burden view (M1), align copy with second-reader framing. |

---

## Part D — Prioritised backlog

**Tier 1 — make the evidence real (blocks thesis claims)**
1. `C3` finish Stage 1 comparison (in progress) → comparison table + writeup
2. `C2` Stage 2 GPU plan + train (with documented fallback if GPU unavailable)
3. `C1` reproducible run + results manifest
4. `C9` + `N3` cascade fallback policy + dual-denominator reporting
5. `M1` alert-burden metrics (cheap; makes every result narrative-ready)

**Tier 2 — make it defensible (turns numbers into science)**
6. `C4` CIs + DeLong + calibration curve
7. `N1` ablation grid (cascade vs. structured-only vs. notes-only vs. fusion)
8. `N2` disagreement analysis (star exhibit)
9. `C6` feature audit (itemids + vitals missingness)
10. `N4` decision-curve / net-benefit
11. `C5` interpretability (SHAP + LR coefficients)
12. `N8` temporal validation

**Tier 3 — make it complete (writing, robustness, delivery)**
13. `W1` chapter skeleton → then `N5` novelty section, `N7` time-point statement, `W2` vignette
14. `C10` EDA notebook + `W3` funnel figure
15. `N6` Stage 3 run + faithfulness rubric (feeds interviews)
16. `C8` tests · `C7` mode-aware cache · `W4` claims audit · `F1` frontend sync · `C11`/`C12` cleanups

---

## Prompt Library

> **Include in EVERY session** (paste alongside the task prompt):
> *Repo `20260401_Thesis`. Before working, read `PROJECT_TLDR.md`, `docs/THESIS_NARRATIVE.md`,
> and the latest file in `sessions/`. Constraints: patient-level (`subject_id`) grouping and
> zero data leakage; fixed seeds; validated Pydantic `AppConfig`; pylint 10/10; design
> principles = simplicity, small/local models, build-it-ourselves. Real MIMIC-IV data and
> note text must NEVER be committed (data/, models/ are gitignored). Framing: the pipeline is
> a high-recall screen + an LLM "second reader" that prunes false alarms; report metrics for
> BOTH the full flagged cohort and the notes cohort; include alerts-per-true-catch where
> precision appears. At the end, write a NEW session log `sessions/YYYY-MM-DD_session-NN.md`
> (never edit older ones).*

### C1 — Reproducible end-to-end run + results manifest
```
Context: Headline Stage 1 metrics (AUROC 0.706, recall 0.848, n=521,191) were produced on a
teammate's machine; local artifacts are stale; the thesis abstract in docs/THESIS_NARRATIVE.md
cites these numbers with no traceability.
Task: (1) Add run_all.py (or Makefile) that executes feature build -> Stage 1 train -> evaluate
in --mode full from a clean checkout, reading MIMIC paths from .env. (2) Create
RESULTS_MANIFEST.json + a human-readable RESULTS_MANIFEST.md recording, per reported metric:
MIMIC-IV version, cohort row counts, git commit, seed, config hash, timestamp, output file.
(3) Every future evaluate run appends/updates its manifest entry automatically. (4) Document
VS Code click-to-run steps (the team does not use the terminal).
Acceptance: a teammate with MIMIC access reproduces Stage 1 numbers from a clean checkout and
the manifest entry matches; the W4 claims audit can cite manifest run IDs.
```

### C2 — Stage 2: GPU training plan + documented fallback + MODEL_CARD refresh
```
Context: The rebuilt Stage 2 (src/stage2/: 2048 tokens, focal loss, age×label stratified 45k
notes, Platt calibration; see sessions/2026-07-05_session-09.md) has NEVER completed training.
A training attempt on a teammate's laptop was aborted — CPU/consumer hardware cannot handle
2048-token Longformer fine-tuning. MODEL_CARD.md still shows old v1 numbers (15k notes, 512
tokens, AUROC 0.640) that the current code no longer produces. The thesis narrative
(THESIS_NARRATIVE.md) needs refreshed Stage 2 numbers, and its fairness storyline (Option E)
depends on the rebuilt model's per-age-group results.
Task: (1) Write docs/stage2_training_plan.md with two executable paths:
  PATH A (preferred): cloud GPU — Colab Pro+ A100 (~4h, bf16) or equivalent; exact steps to
  mount the repo + MIMIC-IV-Note securely (data must never persist in the cloud after the run;
  document deletion), config flags per session-09's GPU table, and `python setup_stage2.py
  --mode full` (splits are CPU-fast and can be prebuilt locally).
  PATH B (fallback if no GPU access): a reduced-scale config that a MacBook/Colab-free tier can
  finish: max_seq_length 1024, ~20k stratified notes, fp32, gradient checkpointing; estimate
  wall-clock; document the trade-off honestly (what the thesis must then say about sequence
  truncation and sample size as limitations).
(2) Execute whichever path is available. (3) Run calibrate -> predict -> evaluate; save
stage2_calibration.json, stage2_results.csv, stage2_evaluation.json. (4) Update MODEL_CARD.md
and README.md: new results labelled v2, old numbers moved to a clearly-marked "v1 (deprecated,
512-token model)" appendix section. (5) Update the manifest (C1) and flag the abstract numbers
in THESIS_NARRATIVE.md for refresh (W4).
Acceptance: a completed Stage 2 training with per-age-group metrics + ECE; MODEL_CARD shows
v2; the fairness targets from session-09 (recall gap < 6pp, precision gap < 3pp, ECE < 0.05)
are evaluated and reported whether met or not.
```

### C3 — Stage 1 three-model comparison: finish + write up
```
Context: LR and HistGradientBoosting full-mode runs on real MIMIC-IV are in progress (XGBoost
done: AUROC 0.706 / AUPRC 0.406 / recall 0.848 @ thr=0.354). The thesis §3.3 needs a
comparison that justifies XGBoost as the Stage 1 screener within the second-reader narrative.
Task: (1) Verify all three models used the IDENTICAL patient-level split (same seed/config;
assert identical test_idx across artifacts). (2) Build docs/stage1_model_comparison.md with one
table: AUPRC, AUROC, Brier, and at the recall>=0.85 operating point: recall, precision,
specificity, F2, ALERTS-PER-TRUE-CATCH (1/precision), total alert volume. (3) Add bootstrap
95% CIs (subject-level, cf. C4) at least for AUPRC/AUROC. (4) Short writeup: why the winner
wins, why LR stays in the thesis (interpretability anchor), 3-5 sentences each.
Acceptance: one reproducible table + writeup ready to paste into §3.3, with CIs and identical-
split verification stated.
```

### C4 — Statistical rigour (CIs + DeLong + calibration)
```
Context: All reported metrics are point estimates. The abstract claims "+21% precision" and
"71% of true positives retained" with no uncertainty. Committee will ask.
Task: Add src/model/stats.py: (1) subject-level (clustered) bootstrap 95% CIs for
AUROC/AUPRC/precision/recall at the operating point — used by BOTH Stage 1 evaluate and
src/stage2/evaluate.py; (2) DeLong test for AUROC differences between the three Stage 1 models
on the shared test set; (3) reliability curve + ECE for Stage 1 (Stage 2 already gets ECE).
Wire into evaluate outputs + manifest. Matplotlib only.
Acceptance: every headline number in MODEL_CARD carries a CI; model comparison has p-values;
calibration figure saved to docs/figures/.
```

### C5 — Interpretability (SHAP + LR coefficients)
```
Context: The narrative sells a "second reader you can question": Stage 1's decisions should be
inspectable. LR is in the lineup precisely for interpretability, but nothing is produced.
Task: Add src/model/interpret.py: (1) standardized LR coefficients + 95% CIs, sorted, with
plain-language feature labels ("days since last discharge", not days_since_last_discharge);
(2) SHAP for the XGBoost artifact on a seeded test sample: beeswarm + top-20 bar + CSV of
mean|SHAP|; (3) docs/stage1_interpretability.md: which drivers are ACTIONABLE at discharge
(follow-up scheduling, discharge destination) vs not (age) — this distinction feeds the
Discussion chapter. No retraining; read saved artifacts.
Acceptance: figures + table usable directly in §3.3/§5; actionable-vs-fixed framing written.
```

### C6 — Feature audit: itemids + vitals missingness
```
Context: features.py maps vitals from icu/chartevents — ICU-only, so every non-ICU admission
has all-NaN vitals; unquantified. Lab itemid lists are unvalidated against d_labitems. The
narrative claims an honest, deployable pipeline; silent missingness undermines it.
Task: (1) Validate every itemid in LAB_ITEMIDS/VITAL_ITEMIDS against hosp/d_labitems.csv.gz
and icu/d_items.csv.gz (print official label + row coverage). (2) Quantify per-feature
missingness on the real cohort, split by ICU vs non-ICU admission. (3) Decide + implement:
recommended default = keep vitals with an explicit has_icu_stay indicator feature and document
that vitals are ICU-conditional; alternatively drop vitals — justify whichever with numbers.
(4) Write docs/feature_audit.md; re-run Stage 1 if the feature set changes and update manifest.
Acceptance: missingness table by ICU status; implemented, justified decision; itemid map
verified against official dictionaries.
```

### C7 — Mode-aware feature cache
```
Context: load_feature_matrix() caches data/processed/features.csv and reuses it regardless of
data source (synthetic vs real), cohort config, or MIMIC version. This silently served
synthetic data to a real-data session once already.
Task: Key the cache on {source, MIMIC version, cohort-config hash}: write a sidecar
features.meta.json and rebuild on mismatch; add --rebuild flag. Print an UNMISSABLE banner at
load stating source + row count + positive rate (e.g. "REAL MIMIC-IV 3.1 | 521,191 rows |
20.2% positive").
Acceptance: switching source/config never serves a stale matrix; the banner makes the active
dataset obvious in every run log.
```

### C8 — Test suite
```
Context: tests/ is empty. The highest-value tests guard the claims the thesis stakes its
credibility on: no leakage, correct label, correct Charlson.
Task: pytest suite (synthetic data only, fast, deterministic): (1) zero subject_id overlap
between train/test from grouped_train_test_split (leakage assertion); (2) build_features
returns expected columns/dtypes/positive-rate range; (3) readmission label on a handcrafted
mini-timeline (readmit at 29d=1, 31d=0, death=0, elective handling); (4)
select_threshold_for_recall + operating_point on tiny arrays; (5) charlson_per_admission on
known ICD-9 and ICD-10 codes incl. hierarchy rules (metastatic cancels malignancy); (6) the
N7 time-point audit test once written. Document one-command local run (VS Code play-button
compatible script is fine).
Acceptance: pytest green; breaking the grouping or the label definition fails a test.
```

### C9 — Cascade policy for note-less patients + full-cohort reporting
```
Context: ~37% of Stage 1-flagged admissions have no discharge note in MIMIC-IV-Note. Current
evaluation silently drops them, so "Stage 1+2" metrics describe only the notes cohort. The
narrative (THESIS_NARRATIVE.md) commits to denominator honesty (N3), and deployment needs a
defined behaviour.
Task: (1) Implement the fallback policy in src/stage2/predict.py: a flagged admission with no
note KEEPS its Stage 1 flag (stage2_confirmed=1, stage2_score=NaN, fallback=true) — the second
reader can only veto what it can read; make the policy a config switch (keep|drop) defaulting
to keep. (2) Update src/stage2/evaluate.py to report BOTH populations side by side: full
flagged cohort (with fallback) and notes cohort, each with precision, recall, F2,
alerts-per-true-catch, and the coverage % stated. (3) One paragraph of rationale for the
thesis (§3.5) including the "in real deployment every discharged patient has a note" argument
and its limits.
Acceptance: no metric appears without its denominator; deployment-realistic (full-cohort)
numbers exist and are the headline; MODEL_CARD/README updated.
```

### C10 — EDA notebook for §3.2
```
Context: notebooks/ is empty; §3.2.2 (Martin's section) needs cohort characterisation; the
funnel figure (W3) needs the underlying counts.
Task: notebooks/01_eda.ipynb (seeded, re-runnable, outputs stripped of patient-level rows):
cohort attrition flow (raw admissions -> adult -> non-elective -> survived -> final, exact
counts per step), readmission base rate overall + by age band/sex/insurance/admission type,
LOS + Charlson distributions, per-feature missingness heatmap (tie to C6), correlation
overview. Save publication-quality figures to docs/figures/ (dataviz-consistent styling).
Write 10 bullet takeaways at the end for direct use in §3.2.2.
Acceptance: figures + attrition counts + takeaways; nothing patient-level committed.
```

### C11 — cohort.py separation (cleanup)
```
Context: src/data/cohort.py is a stub; cohort filters + label live inside features.py. §3.2.1
("Cohort Definition and Label Engineering") would benefit from code that mirrors the thesis
structure.
Task: Move cohort selection + label computation into cohort.py with a clean interface
(build_cohort(tables, cfg) -> cohort DataFrame consumed by build_features); keep behaviour
byte-identical (prove with a C8 regression test on synthetic data); update docs/module
docstrings so code layout matches thesis §3.2/§3.3 structure.
Acceptance: no dead stub; features.py only does features; regression test passes.
```

### C12 — Harden macOS torch/joblib workaround
```
Context: Loading an XGBoost joblib after importing torch segfaults on macOS; current code
passes pre-loaded artifacts around to control import order — fragile and undocumented for new
contributors.
Task: Split the pipeline stages into separate OS processes: Stage 1 scoring writes
stage1_scores.csv; Stage 2 (torch) reads the CSV and never imports joblib/xgboost in-process
(and vice versa). setup_stage2.py orchestrates via subprocess. Add a loud comment + README
note about the constraint.
Acceptance: no code path loads an xgboost joblib after torch import in the same process; a new
contributor cannot trip the segfault by accident.
```

### N1 — Ablation grid: cascade vs. alternatives ("the one table")
```
Context: The committee's first question: why a cascade instead of one multimodal model or
notes-only? THESIS_NARRATIVE.md makes this ablation the Results centrepiece ("the one table
that settles it").
Task: On the SAME patient-level test split (Stage 1 artifact's test_idx), evaluate four arms:
(a) structured-only = Stage 1; (b) notes-only = the fine-tuned Longformer scoring ALL test
admissions with notes; (c) cascade = Stage 1 -> Stage 2 with the C9 fallback; (d) late fusion
= logistic blend of stage1_score + stage2_score fit on the validation split. Report per arm:
AUPRC, AUROC, precision + alerts-per-true-catch at recall>=0.85 (where achievable), total
alert volume, and COMPUTE COST (notes scored per 1,000 admissions — the cascade's efficiency
argument). Note that arms (b),(c),(d) exist only on note-covered admissions; report (a) on
both denominators. Write docs/design_justification.md: conceptual rationale (efficiency,
robustness to missing notes, modular interpretability, workflow mirror) + the empirical table.
Acceptance: one table + argument that the cascade earns its place — or an honest finding that
fusion wins, which the Discussion must then address.
```

### N2 — Disagreement analysis (the star exhibit)
```
Context: The thesis's most original contribution (THESIS_NARRATIVE.md Option C): where and why
does the note overturn the structured prediction? This directly answers "what do the notes
know that the labs don't?" and fills Discussion §5.1.
Task: Build src/analysis/disagreement.py: (1) on the test set, cross-tabulate Stage 1 flag ×
Stage 2 verdict × true label (vetoed-correctly, vetoed-wrongly, confirmed-correctly,
confirmed-wrongly); (2) compare structured-feature distributions between vetoed-correctly and
confirmed-correctly groups (standardized mean differences) — what does Stage 1 over-weight
that the note corrects?; (3) qualitative pass on a seeded sample of ~30 vetoed-correctly and
~15 vetoed-wrongly notes: tag recurring themes (recovered frailty, strong home support,
planned/elective return, palliative intent, resolved acute cause) — PARAPHRASE themes only,
never commit note text; (4) quantify the cost of wrong vetoes: how many true readmissions did
Stage 2 remove, and what did those cases look like? Write docs/disagreement_analysis.md with
the quantitative contrasts + theme table + 3 paraphrased mini-vignettes (clearly marked as
paraphrased composites).
Acceptance: a written analysis answering "what do notes add"; feeds §5.1, the vignette (W2),
and the interview discussion guide.
```

### N3 — Denominator integrity (reporting standard)
```
Context: Rev-1 issue, now a narrative commitment: THESIS_NARRATIVE.md promises full-cohort AND
notes-cohort reporting everywhere. Depends on C9's fallback policy.
Task: After C9 lands, sweep ALL surfaces — MODEL_CARD.md, README.md, docs/*.md, frontend
metric cards, and the abstract in THESIS_NARRATIVE.md — so that every Stage 1+2 metric states
its denominator explicitly and the note-coverage % appears wherever the notes cohort is used.
Add a "Reporting standard" box to MODEL_CARD.md so future numbers follow the rule.
Acceptance: grep-level check passes — no precision/recall claim without a stated denominator;
abstract updated or explicitly flagged pending C2 numbers.
```

### N4 — Decision-curve / clinical-utility analysis
```
Context: "Precision 0.31" doesn't answer "so what". The narrative's alert-fatigue framing
needs net benefit in clinician units. Complements M1 (which implements the raw alert-burden
metrics).
Task: Add src/model/decision_curve.py: (1) decision-curve analysis (net benefit vs threshold
probability) for Stage 1 alone, cascade (with C9 fallback), and treat-all/treat-none
references, on the full test cohort; (2) a simple cost scenario table: assume a
transitional-care intervention cost and a prevented-readmission saving from literature (cite
2-3 sources; both configurable in config.yaml), compute expected net saving per 1,000
discharges for Stage 1 vs cascade at the operating point; (3) figures to docs/figures/,
writeup to docs/clinical_utility.md, explicitly linking to the alerts-per-true-catch numbers
(M1). All assumptions explicit and configurable.
Acceptance: a net-benefit figure + cost table that turns the precision gain into a clinical/
economic argument for §5.2.
```

### N5 — Novelty positioning (§2.4 draft)
```
Context: THESIS_NARRATIVE.md fixed the framing: the contribution is the VERIFICATION layer
(LLM as second reader on a high-recall screen) + disagreement analysis + fairness-audited
local deployment — not raw AUROC. §2.4 ("Research Gap and Contribution", Thomas's chapter, but
draft it for him).
Task: Draft 500-700 words for §2.4 positioning against three literature families: (a)
structured-only readmission models (plateau ~0.70 AUROC, alert-fatigue problem documented);
(b) notes-only clinical LMs (Huang et al. 2019 ClinicalBERT readmission — replaces rather than
complements); (c) multimodal fusion (opaque, compute-heavy, all-or-nothing on missing
modalities). State the gap: no prior work uses the LLM as a selective false-positive verifier
in a high-recall cascade with fairness calibration. Cite li2023comparative for the Longformer
choice. BibTeX entries for any new citations into references.bib.
Acceptance: a paste-ready §2.4 draft consistent with the ablation (N1) and abstract.
```

### N6 — Stage 3 run + faithfulness evaluation
```
Context: Stage 3 (phi4-mini via Ollama) has never produced a real explanation. The narrative
closes on "make every alert explain itself" — currently unbacked. Also feeds the expert-
interview validation (clinician raters).
Task: (1) Ollama + phi4-mini setup documented; run `python setup_stage3.py --limit 50` on
confirmed test-set patients. (2) Add an automated faithfulness check in src/stage3/verify.py:
every number and clinical fact in the output must appear in the input profile (regex/number
matching + a term whitelist); flag violations. (3) Manually rate all 50 on a rubric: grounded
(no invented facts) / complete (uses the salient risk factors) / appropriately hedged (no
diagnosis or treatment advice) / readable. (4) Produce an interview-ready rating form (10
examples, Likert scales) for clinician validation. (5) docs/stage3_evaluation.md with results
+ 5 anonymised example outputs (profiles are synthetic-composite, never raw MIMIC rows).
Acceptance: 50 rated real explanations; hallucination-flagger in CI-able form; rating form
ready for the interviews.
```

### N7 — Prediction time-point statement + leakage audit
```
Context: The pipeline predicts AT DISCHARGE (Stage 1: data available by discharge; Stage 2:
the discharge summary). This must be stated crisply in §3.1 and PROVEN — a leakage accusation
is the fastest way to lose a committee.
Task: (1) Write the §3.1 "prediction setup" paragraph (when the prediction fires, what
information is available, what the label window is). (2) Add an automated audit (test in
tests/, cf. C8): assert every measurement charttime <= dischtime of its admission; assert the
label derives only from the NEXT admission's admittime; assert discharge notes are charted at/
before discharge or explicitly document MIMIC-IV-Note's charttime semantics if not. (3) Run
the audit on the real cohort once and record the result in the manifest.
Acceptance: paragraph written; audit test green on synthetic AND real data.
```

### N8 — Temporal validation (new)
```
Context: The narrative claims deployability; a random split can overstate it. MIMIC-IV
provides anchor_year_group per patient — a coarse but usable time axis.
Task: (1) Check feasibility: distribution of anchor_year_group in the cohort (patients per
group). (2) If viable, add a temporal split mode: train on earlier anchor_year_groups, test on
the latest group (patient-level grouping preserved automatically since anchor_year_group is
per-patient). (3) Re-evaluate the tuned Stage 1 XGBoost under the temporal split; compare
AUROC/AUPRC/operating point vs the random split; report drift. (4) Short writeup in
docs/temporal_validation.md — if performance holds, it strengthens §5.2; if it degrades,
that's an honest limitation for §5.3.
Acceptance: a temporal-vs-random comparison table + interpretation, or a documented finding
that anchor_year_group granularity makes this infeasible.
```

### M1 — Alert-burden metrics (new; cheap, high leverage)
```
Context: THESIS_NARRATIVE.md makes "alerts per true catch" (1/precision) the recurring
clinician-facing number (3.9 -> 3.2 with Stage 2 on v1 numbers). It must be computed, not
hand-derived, and appear everywhere precision does.
Task: (1) Add to src/model/metrics.py: alerts_per_true_catch = 1/precision, plus for cascade
comparisons: total alerts, % alerts removed vs Stage 1, % true positives retained; include in
operating_point() output and both evaluate scripts. (2) Update MODEL_CARD.md/README tables to
carry the new row. (3) Provide a tiny helper that renders the "funnel numbers" (admissions ->
flagged -> confirmed, with TP counts) as a JSON the frontend (F1) and the W3 figure can both
consume.
Acceptance: alert-burden numbers are computed outputs present in metrics JSONs, MODEL_CARD,
and a funnel.json artifact.
```

### W1 — Thesis chapter skeleton (LaTeX)
```
Context: nova_sbe_thesis.tex has the section structure; THESIS_NARRATIVE.md defines the red
thread (chapter-opening questions, funnel-first Results, vignette). Writing has not started.
Task: Flesh out the LaTeX skeleton WITHOUT writing full prose: per section (respecting
ownership: Benjamin 1/4/5/6, Thomas 2, Martin 3.1-3.3, Lennart 3.4-3.6): the chapter-opening
question as an epigraph/comment, 3-6 bullet content stubs per subsection reflecting the
narrative doc, figure/table placeholders with captions already written (funnel, ablation
table, comparison table, disagreement themes, SHAP, decision curve), and \gls{} abbreviations
registered for recurring terms. Compile must stay green (pdflatex + biber per README).
Acceptance: the .tex compiles; every section has stubs + captioned placeholders; teammates can
write into a scaffold that already tells the story.
```

### W2 — Composite patient vignette ("Mrs. M.")
```
Context: The narrative opens and closes the thesis with a recurring composite patient. She
must be FICTIONAL (never a real MIMIC record) yet clinically plausible and consistent with our
feature set and the disagreement-analysis themes (N2).
Task: Write 4 short versions of the vignette: (a) Introduction hook (~120 words, human,
concrete); (b) Methods walk-through (her data as it appears to Stage 1: the actual feature
values a "Mrs. M." would have — age 78, CHF, Charlson ~5, 2 prior admissions, discharged home
alone); (c) Results version (her journey through the funnel with realistic scores); (d)
Discussion resolution (what the note said that the numbers couldn't — align with N2 themes).
Include a footnote template stating she is a fictional composite. Deliver as
docs/vignette.md + LaTeX-ready snippets for W1's placeholders.
Acceptance: four consistent vignette versions, feature-accurate, clearly marked fictional.
```

### W3 — The reusable funnel figure
```
Context: One diagram reused three times (intro schematic -> methods annotated -> results
populated) is the thesis's visual anchor; the frontend has a similar pipeline view. Consumes
funnel.json from M1.
Task: Produce a publication-quality figure (matplotlib or TikZ — pick one, justify) in three
variants sharing identical layout/colour language: (1) schematic (no numbers), (2) annotated
(stage names, model names, threshold logic), (3) populated (real counts + alert-burden numbers
from funnel.json). Colour-blind-safe palette, consistent with frontend styling. Export PDF
(LaTeX) + SVG (frontend/slides) to docs/figures/.
Acceptance: three variants, one visual language, real numbers auto-fed from funnel.json (no
hand-typed counts).
```

### W4 — Claims traceability audit
```
Context: Numbers now live in 6 places: abstract (THESIS_NARRATIVE.md), MODEL_CARD.md,
README.md, frontend mockPatients.ts, docs/*.md, and soon the thesis .tex. They WILL drift
(the abstract already carries v1 Stage 2 numbers flagged for refresh).
Task: (1) Build docs/CLAIMS_AUDIT.md: one row per quantitative claim -> current value ->
source surface(s) -> manifest run ID (C1) -> status (current / stale / pending-C2). (2) Add a
small script (scripts/check_claims.py) that greps the surfaces for the tracked numbers and
reports mismatches against the manifest — run before any submission/presentation. (3) Fix all
stale numbers found, or mark them pending with a visible TODO.
Acceptance: a single table showing every claim is traceable; the checker runs clean or lists
known pendings.
```

### F1 — Frontend sync with real results + narrative framing
```
Context: The React dashboard (frontend/) uses mock patients and v1 metrics; its copy predates
the second-reader framing. It is the professor-demo surface, so it must match the thesis
story exactly. Depends on: C2 (new Stage 2 numbers), M1 (funnel.json), N6 (real Stage 3
explanations).
Task: (1) Replace hardcoded metrics with values loaded from funnel.json / metrics JSONs
(build-time import is fine; no backend). (2) Reframe copy: "second reader", alerts-per-true-
catch, both denominators shown (N3). (3) Replace mock patients with synthetic-composite
patients whose Stage 3 explanations come from the real N6 run (paraphrased/anonymised — no
MIMIC rows). (4) Add a small alert-burden panel: alerts before/after Stage 2, TPs retained.
Keep the two-tab structure (Pipeline for the committee, Patients for the demo).
Acceptance: no hand-typed metric in the frontend; demo tells the same story as the thesis;
`npm run dev` works from a clean install.
```
