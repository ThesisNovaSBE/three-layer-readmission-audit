# Model Card — Readmission Prediction Pipeline

> **See `docs/ARCHITECTURE.md` for the current design.** Stage 1 and Stage 2
> below are both the real 2026-09 retrains — Stage 1 (2026-09-05): 400-trial
> Optuna search, capacity-constrained threshold, isotonic calibration,
> **unplanned** readmission as the target (switched from all-cause the same
> day, see `docs/ARCHITECTURE.md` §6) on the full real MIMIC-IV dataset
> (n=521,191). Stage 2 (2026-09-09/10): retrained under the corrected
> unplanned-label + rebalanced age-group targets + 4096-token window,
> including a calibration-staleness bug found and fixed mid-session (see
> `sessions/2026-09-13_session-23.md`) — the numbers below are from
> *after* that fix, not the initial (miscalibrated) run. Both on GWDG
> KISSKI (A100 80GB). Stage 3 has switched from Ollama/phi4-mini to
> MedGemma-27B-text-it (2026-09-10), served via plain HF transformers +
> lm-format-enforcer (2026-09-15, after vLLM proved structurally
> incompatible with KISSKI's CUDA 12.8 driver ceiling) — matches the
> current code, and has now run at full scale (2026-10-06, all 9,899
> flagged+noted admissions, see §"Stage 1+2+3 — Combined pipeline" below
> for the real RQ2 result).

## Model Details

- **Stage 1:** Classical ML classifiers (Logistic Regression, XGBoost, HistGradientBoosting) on structured EHR features; isotonic-calibrated (since 2026-08-26); capacity-constrained operating point (primary, since 2026-08-25) with recall-floor kept as a secondary comparison table. Two label variants exist in the feature matrix, `readmission_30d` (all-cause) and `readmission_30d_unplanned` (excludes outcome admissions with a planned `admission_type`; added 2026-08-26) — **the model's actual target is `readmission_30d_unplanned`** as of 2026-09-05 (`MODEL_TARGET_COL` in `src/schemas.py`), matching this project's stated scope; every model trained before that date, including the original artifact, silently used all-cause instead
- **Stage 2:** Fine-tuned Clinical-Longformer (`yikuan8/Clinical-Longformer`), note-only (no structured features) — 4096-token context (raised from 2048 on 2026-08-25), trained on real MIMIC-IV-Note discharge summaries; produces an independent, note-based risk estimate, not a gate on Stage 1's flag. A jointly-trained structured+note "FusionLongformer" variant was built and dropped on 2026-08-26 without ever completing a training run — see `docs/ARCHITECTURE.md` §2.
- **Stage 3:** Independent LLM audit via HF `transformers.generate()` + `lm-format-enforcer` (`google/medgemma-27b-text-it`, temperature=0, switched from Ollama/phi4-mini 2026-09-10 once the batch run moved to cluster GPU hardware, then from vLLM 2026-09-15 after KISSKI's CUDA 12.8 driver ceiling proved incompatible with vLLM's kernel stack) — reaches its own uphold/override decision rather than explaining a decision Stage 2 already made
- **Developed by:** Nova SBE thesis team (M.Sc. Business Analytics)
- **Model type:** Three-stage LLM-auditing classification pipeline
- **Language:** English (clinical notes)

## Intended Use

- **Primary use:** Predict unplanned 30-day hospital readmissions from MIMIC-IV data
- **Primary users:** Clinical decision support research
- **Out of scope:** Direct clinical deployment without further validation

## Training Data

- MIMIC-IV v3.1 (structured tables) — credentialed access via PhysioNet
- MIMIC-IV-Note (discharge summaries) — credentialed access via PhysioNet
- Population: Adult patients (age >= 18), excluding in-hospital deaths and elective readmissions

## Metrics

### Stage 1 — XGBoost on structured MIMIC-IV 3.1 (full mode, n=521,191, held-out test set, target=unplanned readmission)

Held-out test set: n=104,242, base rate 19.0%. Primary operating point is
capacity-constrained (top 15% by score), not a recall floor — see
`docs/ARCHITECTURE.md` §5 for why. Recall-floor rows kept below as the
secondary, literature-comparable view.

| Metric | Value |
|--------|-------|
| AUROC | 0.7215 [0.7147, 0.7293] (95% CI, patient-level bootstrap) |
| AUPRC | 0.3965 [0.3735, 0.4252] (95% CI) |
| Brier score | 0.1370 |
| Recall @ thr=0.3235 (capacity=15%) | 0.352 [0.3343, 0.3735] |
| Precision @ thr=0.3235 | 0.431 [0.4156, 0.4494] |
| Specificity @ thr=0.3235 | 0.891 |
| F2 @ thr=0.3235 | 0.366 |
| TP / FP / TN / FN | 6,989 / 9,223 / 75,186 / 12,844 |

Capacity trade-off: K=5% → precision=0.566, recall=0.149, lift=2.97x.
K=10% → precision=0.472, recall=0.267, lift=2.48x. K=20% → precision=0.401,
recall=0.423, lift=2.11x.

Recall-floor, secondary (for comparability with prior literature):
recall≥0.80 → precision=0.268 @ thr=0.1359. recall≥0.85 → precision=0.247 @
thr=0.1175. recall≥0.90 → precision=0.233 @ thr=0.1021.

Beats 3 of 4 published AUROC benchmarks cited in `evaluate.py`: LACE index
(0.694, +0.028), Xiao 2018 EHR baseline (0.715, +0.007), Fraccaro 2016 notes
baseline (0.684, +0.038); behind Rajkomar 2018 deep EHR (0.773, −0.051).

Subgroup AUROC (fairness): Female=0.718 (n=54,512, pos_rate=17.7%),
Male=0.724 (n=49,730, pos_rate=20.5%) — near-equal. Age: 18–40=0.750,
41–55=0.737, 56–70=0.728, **70+=0.665** — elderly patients are
meaningfully harder to predict, and their recall at the primary operating
point is only 19.0% vs. 35–45% for the other three bands. This is the
documented "v1 recall gap" that Stage 2's age-group oversampling exists to
address — see `config.yaml`'s `stage2.age_group_train_targets` comment.

### Stage 2 — Clinical-Longformer (real retrain, completed 2026-09-09/10, target=unplanned readmission)

Trained on 141,767 age-stratified notes (real per-band availability came in
lower than the config's targets across the board — every positive-label
cell hit the "use all available" fallback; see `config.yaml`'s
`age_group_train_targets` comment and `sessions/2026-09-13_session-23.md`).
`EarlyStoppingCallback` (patience=3) stopped training at ~1.02 of the
configured 10 epochs once validation AUPRC peaked — Val AUPRC=0.372,
AUROC=0.704. A calibration-staleness bug (reused a pre-existing calibration
file without checking it matched this model) was found and fixed the same
session; numbers below are post-fix.

**Test-set evaluation (notes cohort, n=9,899 Stage 1-flagged-with-note admissions):**

| Metric | Value |
|--------|-------|
| AUROC | 0.622 |
| AUPRC | 0.538 |
| Recall | 0.974 |
| Precision | 0.437 |
| F2 | 0.782 |
| ECE | 0.1957 |

Per-age-group:

| Band | N | AUROC | Recall | Precision | F2 | ECE |
|------|---|-------|--------|-----------|-----|-----|
| 18-40 | 1,570 | 0.626 | 0.956 | 0.507 | 0.812 | 0.2028 |
| 41-55 | 2,836 | 0.617 | 0.975 | 0.437 | 0.782 | 0.2324 |
| 56-70 | 3,514 | 0.632 | 0.983 | 0.432 | 0.783 | 0.1879 |
| 70+   | 1,979 | 0.581 | 0.974 | 0.392 | 0.751 | 0.1543 |

Fairness gaps: recall gap 0.027 (worst: 18-40), precision gap 0.115 (worst:
70+), AUROC gap 0.051. 70+ remains the hardest band for Stage 2 too, though
the recall gap specifically is small — the age-group oversampling fix
appears to have helped recall parity even though precision/AUROC gaps
remain, worth digging into further for the fairness discussion.

**Training config:**

| Parameter | Value |
|-----------|-------|
| Training notes | 141,767 (real availability; see note above) |
| GPU | NVIDIA A100-SXM4-80GB (GWDG KISSKI) |
| Precision | bf16 |
| Batch size | 8 (effective 16 with grad. accum. ×2) |
| Gradient checkpointing | disabled (80 GB VRAM sufficient) |
| Sequence length | 4096 tokens (raised from 2048 on 2026-08-25) |

### RQ1 — Does the note text add signal beyond structured data?

`compare_stages.py` scores Stage 1 and Stage 2 **independently, on the
identical 62,759-admission notes-covered population** (neither model gates
or feeds the other) — the fair, apples-to-apples comparison this project's
own docs call for:

| Model | AUROC |
|-------|-------|
| Stage 1 (structured) | 0.7093 |
| Stage 2 (notes-only) | 0.7101 |
| Difference | +0.0008 [-0.0041, +0.0054] (95% CI) |

**Null result** — the CI straddles zero. This is explicitly anticipated and
reportable per this project's own design docs, not a failure of either
model: on this population, the discharge note alone carries no more (and no
less) predictive signal than the structured record alone.

### Selection bias: does the notes-covered population look like the excluded one?

`scripts/analyze_selection_bias.py` (2026-10-26) compares baseline
characteristics between Stage 1's test-partition admissions that have a
discharge note (the population RQ1/RQ2 are evaluated on) and those that
don't, using `has_discharge_note` (computed directly against the real
MIMIC-IV-Note file, not inferred). **They are clinically distinct
populations.** Of 104,242 test admissions, 60.2% (n=62,759) have a note,
39.8% (n=41,483) don't. The notes-covered group is older (59.7 vs. 52.1),
sicker (Charlson comorbidity index 2.24 vs. 1.43), far more often an
emergency admission (SMD=0.73 — the largest gap of any feature checked by
a wide margin), and more often discharged to a facility. Standardized
mean differences (Austin 2009; |SMD| ≥ 0.1 flags a meaningfully
imbalanced covariate, independent of sample size) exceed that threshold
on 8 of 12 baseline characteristics checked.

**The readmission outcome rate itself, however, is nearly balanced**
(19.8% covered vs. 17.8% excluded, SMD=0.053 — below the imbalance flag).
So the skew is real and worth disclosing as scope — RQ1/RQ2 generalize to
an older, sicker, more acutely-admitted population, not to all
admissions equally — but it is not an outcome-rate confound driving the
RQ1/RQ2 results themselves.

### Stage 1+2+3 — Combined pipeline (real RQ2 result, full run 2026-10-06)

All 9,899 flagged+noted admissions audited by Stage 3
(`google/medgemma-27b-text-it`, temperature=0); 93 annotation failures
(0.94%, token-budget truncations) fall back to Stage 2's threshold per the
documented fallback chain. 95% CIs are patient-clustered bootstrap (1000
resamples).

| Metric | Value |
|--------|-------|
| Stage 1 alone (test n=104,242) | AUROC=0.7215, recall=0.352, precision=0.431 |
| Stage 2 alone (flagged+noted, n=9,899, 61.1% note coverage of flagged) | AUROC=0.622, recall=0.974, precision=0.437 |
| Pipeline, full cohort (n=104,242, blanket Stage 3, C9 fallback applied) | precision=0.435 [0.417, 0.457], recall=0.295 [0.276, 0.315], F1=0.351, F2=0.315 |
| Pipeline, notes cohort only (n=97,929) | precision=0.433, recall=0.180 |
| Control arm (Stage 1 alone @ matched 12.9% alert rate) | precision=0.450 [0.433, 0.469], recall=0.316 [0.298, 0.338], F1=0.372, F2=0.336 |
| Conditional triggering (discordant-only, post-hoc; 5,131/9,793 LLM calls, 4,662 saved) | precision=0.437, recall=0.319 (no CI computed yet) |

**Headline RQ2 result: at matched alert volume, Stage 1 alone (the control
arm) beats the full three-stage cascade on both precision and recall.**
The precision CIs overlap substantially, but recall looks like a real gap
— the control arm's lower bound (0.298) sits above the full pipeline's own
point estimate (0.295). Applying Stage 3 to every flagged admission is net
*harmful* to screening performance: the auditor's overrides cancel some
alerts that turn out to be correct, and that cost outweighs what it
correctly catches.

The **conditional-triggering** analysis (`scripts/analyze_discordance_subgroups.py`)
re-scores the same completed run under a different rule: a CONCORDANT
admission (Stage 1 and Stage 2 agree) is treated as if Stage 3 had never
been called — Stage 1's flag simply stands — while a DISCORDANT admission
keeps its real Stage 3 decision. This is not a different execution mode
(the real batch run always audits everyone; this is a post-hoc re-filter
of its output), run this way deliberately so Stage 3's actual decisions on
concordant cases were available to check, not just assumed useless. Under
this rule, both precision *and* recall improve versus the blanket pipeline
(0.437/0.319 vs. 0.435/0.295) using 4,662 fewer LLM calls (48% reduction).

**This full-cohort comparison only varies treatment on the concordant
group — it does not, by itself, show Stage 3 adds value on discordant
cases**, since the discordant group's treatment (Stage 3's real decision)
is identical in both the blanket and conditional numbers above. A direct
follow-up test — Stage 3's real decision vs. "Stage 1's flag stands,"
computed *within each subgroup separately* — gives the fuller, more honest
picture (95% CIs, patient-clustered bootstrap):

| Subgroup | Policy | Precision | Recall | F2 |
|---|---|---|---|---|
| Concordant (n=4,662) | Stage 1 alone | 0.447 [0.427, 0.467] | 1.000 | 0.801 |
| | Stage 3's decision | 0.443 [0.421, 0.466] | 0.771 [0.741, 0.796] | **0.671** |
| Discordant (n=5,131) | Stage 1 alone | 0.409 [0.392, 0.426] | 1.000 | 0.776 |
| | Stage 3's decision | 0.423 [0.402, 0.444] | 0.682 [0.656, 0.708] | **0.608** |

By F2 (weights recall 4x precision — appropriate here since a missed
readmission is costlier than a false alarm), **Stage 3 makes things worse
in both subgroups, and the drop is larger on discordant cases (−0.168)
than concordant (−0.130)** — recall falls further there (32 points vs. 23
points) and F2 punishes that hard. Stage 3 does buy a small, real
precision gain on discordant cases (+1.4pp) that it doesn't buy on
concordant cases (flat/slightly negative) — but that gain doesn't come
close to compensating for the recall it gives up, in either subgroup.

**The correct, complete statement**: Stage 3's overrides trade recall for
little-to-no precision gain almost everywhere they're applied. That trade
is unambiguously bad on concordant cases (zero redeeming precision value)
and still a bad trade — just a smaller, partially-compensated one — on
discordant cases. The earlier framing ("auditing concordant cases is where
the harm concentrates") is not wrong about concordant cases being the
clean win from removing Stage 3, but it should not be read as "Stage 3
helps on discordant cases" — the subgroup analysis above directly tests
that claim and finds it doesn't hold on F2. See `models/
discordance_subgroup_evaluation.json` for the full numbers.

## Stage 3 — Independent LLM Audit (MedGemma-27B via HF transformers + lm-format-enforcer)

Rewritten 2026-08-25, extended 2026-08-28. Switched serving from
Ollama/phi4-mini to `google/medgemma-27b-text-it` on 2026-09-10, once the
batch run moved to cluster GPU hardware, first via vLLM (its
guided/structured decoding preserved the schema-constrained JSON
generation Ollama's `format=` provided), then to plain HF
`transformers.generate()` + `lm-format-enforcer`'s
`prefix_allowed_tokens_fn` on 2026-09-15 — same schema-constrained-JSON
guarantee, different serving mechanism — after KISSKI's CUDA 12.8 driver
ceiling proved structurally incompatible with vLLM's flashinfer/CUTLASS
kernels regardless of vLLM/torch version (see `sessions/` for the full
diagnosis). MedGemma is the only evaluated candidate benchmarked directly
on MIMIC-IV-style reasoning; see `sessions/2026-09-13_session-23.md` and
`config.yaml`'s `stage3.model_name` comment for the full model comparison
and license check. Available both on-demand (one patient per call, via the
API) and in batch (`src/stage3/batch.py`, every Stage 1-flagged,
note-covered admission) — the full batch run completed 2026-10-06 across
all 9,899 flagged+noted admissions. For each patient, the model receives:
- Stage 1's score + top-k SHAP-ranked structured risk factors
- Stage 2's independently-derived, note-based score
- A quantitatively pre-computed discordance mode (never chosen by the LLM)
- The discharge note itself (near-full text, ~20,000-char safety cap — not a
  5-sentence attention summary)

The model reaches its **own** independent decision — it is not asked to
narrate or classify a decision Stage 2 already made.

**Output per patient:**

| Field | Description |
|-------|-------------|
| `mitigating_grounds`, `aggravating_grounds` | Two-sided grounds the model extracted from the note, each with its own verified quote |
| `decision_model` | `uphold` / `override` / `insufficient_evidence` — the model's own judgment |
| `decision_rule` | The same three-way decision, recomputed deterministically in code from the extracted grounds *restricted to quote-verified entries only* — a consistency check, not a second model opinion, and not driven by evidence that didn't check out |
| `all_quotes_verified` | True only if every extracted ground's quote was found verbatim in the note |
| `planned_return` | Independent yes/no/not_stated field on whether the note documents a scheduled return |
| `clinical_justification` | 2-4 sentence justification citing note content |
| `r1`, `r2`, `displacement`, `discordance_mode` | Quantitative context (percentile ranks + mode), computed before the LLM call |
| `note_truncated`, `model_name` | Logged per row for truncation/scale-comparison analysis |

**Grounds taxonomy** (fixed list; a ground outside it, or with an empty
quote, is a parse failure, not a new category):
- *Mitigating:* `palliative_intent` · `planned_return` ·
  `structured_driver_contradicted`
- *Aggravating:* `lives_alone_no_support` · `no_followup_arranged` ·
  `functional_dependence` · `cognitive_impairment` · `nonadherence_risk` ·
  `unstable_at_discharge`

**`strong_discharge_support`** (2026-09-27) is a fourth mitigating ground,
but is NOT directly citable by the model like the others above — a
50-patient validation run found the model citing it on 47/50 patients
(94%) on weak single-criterion evidence (including once on the bare
`Discharge Disposition: Extended Care` field, a pattern an explicit
negative example in the prompt already excluded), i.e. a prose "ALL THREE,
not just one" instruction didn't reliably stop the model treating one weak
signal as sufficient. Instead the model extracts three atomic quotes
independently (`followup_plan_quote`, `named_caregiver_quote`,
`clinically_stable_quote`), and code (`_extract_discharge_support_ground`)
credits `strong_discharge_support` only when all three are present and
individually verified against the note — the same fix pattern as
`decision_rule`: never trust the model with compound AND logic it can get
wrong.

**Discordance mode** is computed from percentile-rank displacement of
stage1_score vs. stage2_score within the flagged+noted cohort — not raw
score subtraction, which was tried and rejected as fragile to unequal
calibration between the two model families (see `docs/ARCHITECTURE.md`).

**Research contribution:** No prior work in the literature review's
49-study systematic search uses an LLM as an independent auditor of another
model's output (as opposed to predictor, feature extractor, or explainer of
its own prediction). `src/stage3/batch.py:run_batch_audit` produced Stage 3
decisions across every Stage 1-flagged, note-covered admission (completed
2026-10-06, 9,899/9,899, 0.94% annotation failure rate) — see "Stage
1+2+3 — Combined pipeline" above for the real RQ2 result this made
possible.

### Does Stage 3's `planned_return` agree with the structured-data proxy?

`scripts/analyze_planned_return_crosscheck.py` (2026-10-26) compares
Stage 3's note-derived `planned_return` field against the structured,
`admission_type`-based "unplanned" proxy (`readmission_30d_unplanned`),
restricted to Stage-3-audited patients who were actually readmitted
within 30 days (n=4,375 — the only population where "was the readmission
planned" has a structured answer at all; this question was raised but
never examined as of the 2026-09-13 architecture pass).

**The finding is not reassuring.** Raw agreement is 81.7% — which reads
as fine in isolation, but a trivial "always predict unplanned" baseline
already scores 95.8% (the structured-planned rate is only 4.2%), so
**Stage 3's actual agreement is below what doing nothing would achieve.**
Stage 3 calls "planned" 4.5× more often than the structured proxy
supports (19.2% vs. 4.2%); of Stage 3's 842 "yes" calls, only 112 (13.3%)
match the structured proxy. This reads as the LLM being substantially
more liberal about reading a note as describing a planned return than
the structured data would license — both signals have known, disclosed
limitations (the structured proxy only catches planned returns visible
in `admission_type`; Stage 3's answer could in principle be catching
real scheduled returns the structured proxy misses), so this is reported
as an open, unresolved discrepancy between the two signals, not evidence
that either one is simply wrong.

### Is Stage 3 actually using its inputs? Three validation controls (2026-10-30)

Session 19 built three diagnostic controls to test whether the auditor
genuinely relies on the evidence it is given, rather than ignoring part
of it. They had never been run against the real model until
`scripts/run_validation_controls.py` executed all three on KISSKI
(`models/stage3_validation_controls_summary.json`).

**Blind-note control (n=150):** withholds the discharge note entirely.
The auditor returned `insufficient_evidence` on **150 of 150**
admissions. On the same 150 admissions with the note present, the real
run never used `insufficient_evidence` once (104 uphold, 46 override).
Raw agreement with the real decision is 0%, and that is the intended
result, not a failure: the auditor does not guess uphold or override
when it has nothing to read. It consistently recognizes the absence of
evidence and says so. This is the strongest available evidence that
Stage 3's real decisions are grounded in the note itself.

**No-Stage-2 control (n=40):** withholds Stage 2's score and the
discordance signal, note still present. Agreement with the real
decision is 82.5%; removing Stage 2's evidence flips the decision on 7
of 40 admissions, shifting toward more upholds (27→30 uphold, 13→10
override). Stage 2's score measurably influences the audit without
dominating it.

**Self-agreement control (n=30):** runs the identical patient through
Stage 3 twice. Agreement is 100% on both `decision_model` and
`decision_rule` across all 30 admissions. `temperature=0` delivers a
fixed, reproducible decision in practice, not just in principle, which
the full RQ2 result (a single run across 9,899 admissions) depends on.

**Taken together**, the net-harmful RQ2 finding is not an artifact of a
broken or inattentive auditor. Stage 3 demonstrably reads the note,
demonstrably uses Stage 2's evidence, and is demonstrably reproducible,
and it still underperforms Stage 1 alone at matched alert volume. That
is a materially stronger claim than an unvalidated negative result would
have been.

## Limitations

- Trained and evaluated on MIMIC-IV only (single US academic medical center)
- Temporal and demographic generalization not validated
- Not intended for real-time clinical use

## Ethical Considerations

- MIMIC-IV data is de-identified but originates from real patient encounters
- All data handling follows the PhysioNet Data Use Agreement
- Readmission prediction models may encode demographic biases present in the training data — per-age-band fairness analysis has been run (see Stage 1's subgroup AUROC/recall breakdown above and Stage 2's age-group oversampling); the documented "70+" recall gap is a real, disclosed limitation, not an unstarted analysis
