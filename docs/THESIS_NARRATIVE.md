> **Superseded 2026-08-25** for architecture/Stage 3 mechanics — see
> `docs/ARCHITECTURE.md` for the current pipeline design. The storytelling
> framing was updated 2026-08-28 to match: Stage 1 is capacity-constrained
> (top-K% by risk), Stage 2 is an independent note-based second opinion (not
> a gatekeeper that "vetoes"), and Stage 3 is an independent auditor of
> Stage 1's flags — using Stage 2's score as evidence, quoting the note,
> reaching its own uphold/override decision — rather than a narrator of
> Stage 2's decision.

# Thesis Narrative & Storytelling Guide

> **Purpose.** How we tell the story of this thesis — the framing options, the recommended
> red thread through every chapter, concrete storytelling devices, and a ~400-word abstract
> to give Prof. Shen as a preview.
>
> Companion docs: `PROJECT_TLDR.md` (what we built) · `docs/SANITY_CHECK_2026-07-06.md`
> (what still needs fixing before these claims are fully backed).

---

## 1. The thesis in one sentence

> **A classical ML model flags the highest-risk share of admissions it has budget to act on
> from structured EHR data; a clinical language model independently reads the discharge
> notes of the same population and forms its own opinion; and a small local LLM audits every
> flagged case against both scores and the note itself, upholding or overriding the alert
> and explaining, with a quoted passage, why.**

**Sanity check verdict:** the idea is sound and clinically resonant — *if* we frame it as an
**audit layer** (LLM as independent auditor, not as narrator of a decision already made by
Stage 2) rather than "LLM predicts readmission too." The known argumentation gaps (why a
cascade beats a matched-budget structured-only baseline, notes-cohort denominator, clinical
utility quantification) are catalogued in the sanity-check doc; the storyline below is
designed so that answering those gaps *is* the story.

---

## 2. Storyline options

### Option A — "The boy who cried wolf" (alert fatigue) ⭐ recommended lead
- **Hook:** Risk models already exist — and clinicians ignore them, because a model flags far
  more admissions than any ward can act on, so most alerts turn out to be noise. The
  bottleneck isn't prediction, it's *trust*.
- **Story:** We don't build a better wolf-detector; we add an independent auditor who checks
  each alarm against the patient's chart — and its own reading of the notes — before it
  reaches a human, and stands behind every override with a quoted reason.
- **Strengths:** True to our numbers (this is exactly what the auditor is for vs. a
  matched-budget structured-only baseline), clinically urgent, differentiates from "yet
  another readmission model."
- **Risk:** Needs the control-arm comparison (structured-only at the same alert budget) to
  fully land (`_control_arm_report`, done; large-scale run pending Stage 1/2 retrain).

### Option B — "The independent auditor" (workflow mirror)
- **Hook:** Hospitals already work this way: a broad screen, an independent second opinion,
  then a documented decision on whether to act. Our pipeline is that workflow, automated —
  screen (XGBoost), independent read (Clinical-Longformer), audit (local LLM).
- **Strengths:** Instantly intuitive to clinicians and committee; makes the three-stage
  architecture feel *inevitable* rather than arbitrary; matches the literature-review gap
  directly (LLM-as-auditor, not LLM-as-predictor/explainer).
- **Risk:** A metaphor, not a result — must be backed by the control-arm comparison showing
  the audited pipeline beats the structured screen alone at the same alert budget
  (backlog N1).

### Option C — "Two kinds of evidence" (epistemic story)
- **Hook:** Structured data knows *what* happened (labs, codes, LOS); notes know *what it
  means* (frailty, lives alone, palliative intent, planned return). Neither alone suffices.
- **Story:** The scientific payload is the **disagreement analysis** — the cases where the
  note overturns the numbers, and *why*.
- **Strengths:** The most genuinely academic framing; produces the thesis's star exhibit.
- **Risk:** Depends on an analysis not yet run (backlog N2).

### Option D — "AI that fits in the hospital basement" (pragmatics)
- **Hook:** No cloud APIs, no patient data leaving the building, models that run on a
  laptop/single GPU. Deployable under GDPR/HIPAA constraints by design.
- **Strengths:** Real differentiator vs. GPT-4-based literature; our advisor-endorsed design
  principles (simple, small, local) become a *contribution*, not a limitation.
- **Risk:** Supporting theme, not a thesis by itself.

### Option E — "Fair triage" (equity story)
- **Hook:** Our own audit showed the pipeline underserved patients 70+ (13 pp recall gap
  introduced by Stage 2 v1) — so we rebuilt Stage 2 with age-stratified training, focal loss,
  and per-group calibration.
- **Strengths:** Honest, self-critical science; fairness is a first-class thesis theme.
- **Risk:** Rebuilt Stage 2 not yet trained (backlog C2) — currently a promise, not a result.

### Recommended blend
**Lead with A (alert fatigue), structure with B (independent auditor), deliver scientific
depth with C (disagreement analysis), and let D + E run as supporting themes.** One sentence:
*"Clinicians drown in false alarms; we add an independent auditor that checks the chart and
its own reading of the notes before the alarm fires, show precisely what the notes
contribute, and do it with small, local, fairness-audited models."*

---

## 3. The red thread, chapter by chapter

| Chapter | Narrative move |
|---|---|
| **1. Introduction** | Open with the composite patient vignette (below) + the alert-fatigue statistic. Pose the question: *can we make risk alerts trustworthy enough to act on?* |
| **2. Literature review** | Build to the gap in three steps: structured models plateau (AUROC ≈ 0.7)… LLMs are used as predictors, feature-extractors, or explainers of a model's own output… but never to audit another model's decision. **Gap: nobody uses the LLM as an *independent auditor*.** |
| **3. Data & Methodology** | Frame each stage as a member of the clinical team: the screener working within a fixed follow-up capacity (Stage 1, top-K% by risk), the specialist reading the note cold, forming an independent opinion (Stage 2, Clinical-Longformer), the auditor who reviews the case file — both scores, the SHAP reasons, the note itself — and reaches their own verdict (Stage 3, local + guarded prompt). State the prediction time-point (at discharge) explicitly. |
| **4. Results** | Tell it as the patient's journey through the funnel: X admissions → Y flagged (top-K%) → Z upheld / W overridden by the auditor. Report the control-arm comparison (structured-only at the same alert budget) alongside precision/recall, full-cohort AND notes-cohort. The RQ1/RQ2 comparison table (does text add signal / does the auditor add value over the matched-budget baseline) is "the one table that settles it." |
| **5. Discussion** | Return to the vignette: what did the note say that the numbers couldn't? (disagreement analysis, quote-verified). Then honesty: coverage gap, calibration, single-center. Fairness rebuild as self-critical science. |
| **6. Conclusion** | The workflow argument: this isn't a model, it's a *division of labour* between a fixed-capacity screen, an independent read, and an accountable auditor — a pattern that generalises beyond readmission. |

### Storytelling devices
1. **A recurring composite patient** ("Mrs. M., 78, CHF, lives alone, discharged Tuesday —
   readmitted Friday"). Open the thesis with her; revisit her at each stage; resolve in the
   Discussion. (Composite/fictional — never a real MIMIC record.)
2. **Alert-budget-matched comparison** as the recurring clinical metric: does the audited
   pipeline outperform the structured screen alone at the *same* alert volume
   (`_control_arm_report`)? Every results table repeats this comparison. [Numbers pending
   Stage 1/2 retrain — see `docs/ARCHITECTURE.md` §4.]
3. **Name the pipeline.** Prefer "screen, independent read, audit" over the retired
   "triage-and-verify" framing — use it consistently as a
   brand (figure titles, table captions), so the committee remembers *the pattern*, not just
   the numbers.
4. **One diagram, reused.** The Stage 1→2→3 funnel with live numbers appears in the intro
   (schematic), methods (annotated), and results (populated). Same visual language as the
   frontend demo.
5. **Chapter-opening questions.** Each chapter opens with the single question it answers
   ("Why don't clinicians trust risk scores?" → "What do the notes know that the labs
   don't?").

---

## 4. Abstract (preview for Prof. Shen, ~400 words)

> **REWRITTEN 2026-08-28 to match the current architecture** — the previous version
> described a high-recall XGBoost screen (recall ≥ 0.85, flagging 67% of admissions) and a
> Stage 2 that "vetoes" alarms as a gatekeeper. Neither is current: Stage 1 is now
> capacity-constrained (flags the top ~15% by risk), and Stage 2 is an independent second
> opinion Stage 3 reasons over, not a gate. Numbers below are placeholders pending
> retraining (`docs/ARCHITECTURE.md` §4) — replace before sending to Prof. Shen.

> **An LLM Auditor for Hospital Readmission Alerts: Independent Verification Beyond
> Prediction**
>
> Unplanned 30-day readmissions are costly, penalised, and partly preventable — yet the
> prediction models meant to flag them are routinely ignored. Trained on structured
> electronic health records, such models plateau near AUROC 0.70, and at any clinically
> useful sensitivity they generate more false alarms than a hospital can act on — alert
> fatigue wins. The information that separates a genuinely fragile patient from a
> statistically similar one — frailty, social support, discharge planning, palliative
> intent — is rarely in the structured record at all. It is written in the clinical notes.
>
> This thesis proposes and evaluates a three-layer pipeline built around a role no prior
> study in a systematic 49-study literature review has used a language model for: auditing
> another model's output. A structured XGBoost model flags the highest-risk [K]% of
> admissions using MIMIC-IV data (521,191 admissions) — a capacity a hospital can actually
> follow up on, not a recall floor that flags most of it. A fine-tuned clinical language
> model (Clinical-Longformer) reads the same population's discharge notes independently,
> producing its own risk estimate — evidence for what follows, not a verdict. A locally
> deployed reasoning model (phi4-mini, via Ollama) then audits every flagged case: given the
> structured score and its stated reasons, the independent note-based score, and the note
> itself, it reaches its own uphold-or-override judgment, quotes the note passage it relied
> on, and states its reasoning in plain language. No patient data leaves the hospital
> environment at any stage: every model is small, open, and locally deployable by design.
>
> Beyond aggregate metrics, the thesis asks two separable questions: does narrative text add
> predictive signal structured data alone doesn't (tested directly, on the same population,
> independent of the cascade — a null result here is expected and reportable), and does an
> independently-informed auditor improve on the structured screen alone, including against
> the simplest alternative of just raising its threshold to the same alert budget. A
> disagreement analysis examines *where* and *why* the auditor overrides a structured alert;
> a fairness audit addressed a recall disparity affecting patients over 70 through
> age-stratified training and per-group calibration; and every override is quote-verified
> against the source note, making human review of the audit's reasoning tractable rather
> than a leap of faith.
>
> The result is intended less as another readmission model than as a transferable pattern
> for clinical AI: a cheap, broad screen, an independent second reading, and an accountable
> auditor over both — not a black box that predicts, but a system whose disagreements are
> visible and whose overrides explain themselves.

*(Numbers to fill in once Stage 1/Stage 2 are retrained under the current config —
`docs/ARCHITECTURE.md` §4. Word count target unchanged at ~400.)*

---

## 5. If the professor asks…

- **"Why not one multimodal model?"** → Efficiency (the note-reading and audit stages run
  only on the flagged capacity, not the full population), robustness to missing notes,
  modular interpretability, and the workflow analogy — plus the control-arm comparison (N1)
  to show it empirically against a matched-budget structured-only baseline.
- **"Why not just raise Stage 1's threshold to the alert budget you want and skip the LLMs?"**
  → That's exactly the control arm (`_control_arm_report`): Stage 1 alone, same alert volume,
  same evaluation. The cascade has to beat *that*, not an unconstrained Stage 1 — numbers
  pending retrain.
- **"What's novel?"** → The *auditor* framing (LLM independently reviews another model's
  flagged output, on stated evidence, with a quoted and verified justification for every
  override) — a role absent from a systematic 49-study literature review — plus the
  disagreement analysis and a fairness-audited, fully local implementation, not raw AUROC.
