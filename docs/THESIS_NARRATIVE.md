# Thesis Narrative & Storytelling Guide

> **Purpose.** How we tell the story of this thesis — the framing options, the recommended
> red thread through every chapter, concrete storytelling devices, and a ~400-word abstract
> to give Prof. Shen as a preview.
>
> Companion docs: `PROJECT_TLDR.md` (what we built) · `docs/SANITY_CHECK_2026-07-06.md`
> (what still needs fixing before these claims are fully backed).

---

## 1. The thesis in one sentence

> **A classical ML model casts a wide net for 30-day readmission risk from structured EHR
> data; a clinical language model then *reads the discharge notes* of flagged patients and
> vetoes the false alarms — cutting alert burden while keeping recall high — and a small
> local LLM explains each confirmed flag in plain language.**

**Sanity check verdict:** the idea is sound and clinically resonant — *if* we frame it as a
**verification layer** (LLM as second reader) rather than "LLM predicts readmission too."
The known argumentation gaps (cascade-vs-fusion justification, notes-cohort denominator,
clinical utility quantification) are catalogued in the sanity-check doc; the storyline below
is designed so that answering those gaps *is* the story.

---

## 2. Storyline options

### Option A — "The boy who cried wolf" (alert fatigue) ⭐ recommended lead
- **Hook:** Risk models already exist — and clinicians ignore them, because at usable recall
  they cry wolf 3 times out of 4. The bottleneck isn't prediction, it's *trust*.
- **Story:** We don't build a better wolf-detector; we add a second reader who checks each
  alarm against the patient's chart before it reaches a human.
- **Strengths:** True to our numbers (precision is exactly what Stage 2 improves), clinically
  urgent, differentiates from "yet another readmission model."
- **Risk:** Needs the decision-curve / alerts-per-catch analysis to fully land (backlog N4).

### Option B — "The second reader" (workflow mirror)
- **Hook:** Hospitals already work this way: a broad screen, then a specialist review, then an
  explanation to the care team. Our pipeline is that workflow, automated — triage (XGBoost),
  verify (Clinical-Longformer), explain (local LLM).
- **Strengths:** Instantly intuitive to clinicians and committee; makes the three-stage
  architecture feel *inevitable* rather than arbitrary.
- **Risk:** A metaphor, not a result — must be backed by the ablation showing the cascade
  beats one-shot alternatives (backlog N1).

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
**Lead with A (alert fatigue), structure with B (second reader), deliver scientific depth
with C (disagreement analysis), and let D + E run as supporting themes.** One sentence:
*"Clinicians drown in false alarms; we add an automated second reader that checks the chart
before the alarm fires, show precisely what the notes contribute, and do it with small,
local, fairness-audited models."*

---

## 3. The red thread, chapter by chapter

| Chapter | Narrative move |
|---|---|
| **1. Introduction** | Open with the composite patient vignette (below) + the alert-fatigue statistic. Pose the question: *can we make risk alerts trustworthy enough to act on?* |
| **2. Literature review** | Build to the gap in three steps: structured models plateau (AUROC ≈ 0.7)… notes-only LLMs exist but replace rather than complement… fusion models are opaque and heavy. **Gap: nobody uses the LLM as a *verifier*.** |
| **3. Data & Methodology** | Frame each stage as a member of the clinical team: the tireless screener (Stage 1, tuned for recall), the careful specialist (Stage 2, reads the whole note — hence Longformer), the communicator (Stage 3, local + guarded prompt). State the prediction time-point (at discharge) explicitly. |
| **4. Results** | Tell it as the patient's journey through the funnel: X admissions → Y flagged → Z confirmed. Report *alerts per true catch* alongside precision/recall, full-cohort AND notes-cohort. The ablation table (structured-only / notes-only / cascade / fusion) is "the one table that settles it." |
| **5. Discussion** | Return to the vignette: what did the note say that the numbers couldn't? (disagreement analysis). Then honesty: coverage gap, calibration, single-center. Fairness rebuild as self-critical science. |
| **6. Conclusion** | The workflow argument: this isn't a model, it's a *division of labour* between cheap-broad and expensive-careful — a pattern that generalises beyond readmission. |

### Storytelling devices
1. **A recurring composite patient** ("Mrs. M., 78, CHF, lives alone, discharged Tuesday —
   readmitted Friday"). Open the thesis with her; revisit her at each stage; resolve in the
   Discussion. (Composite/fictional — never a real MIMIC record.)
2. **Alerts-per-true-catch (1/precision)** as the recurring clinical metric: ~3.9 → ~3.2
   with Stage 2, alongside the absolute alert-volume reduction (43,776 → 25,699 on the notes
   cohort). Every results table repeats this row.
3. **Name the pipeline.** "Triage-and-Verify" is already ours — use it consistently as a
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

> **Triage and Verify: Reducing False Alarms in Hospital Readmission Prediction with a
> Locally Deployed Clinical Language Model**
>
> Unplanned 30-day readmissions are costly, penalised, and partly preventable — yet the
> prediction models meant to flag them are routinely ignored. Trained on structured
> electronic health records, such models plateau near AUROC 0.70; tuned for the high recall
> that patient safety demands, they bury clinicians in false alarms, and alert fatigue wins.
> The information that separates a genuinely fragile patient from a statistically similar
> one — frailty, social support, discharge planning, palliative intent — is rarely in the
> structured record at all. It is written in the clinical notes.
>
> This thesis proposes and evaluates a *triage-and-verify* pipeline that mirrors how
> hospitals already reason. A deliberately high-recall XGBoost model screens all admissions
> using structured MIMIC-IV data (521,191 admissions; AUROC 0.706; recall 0.85). A
> fine-tuned clinical language model (Clinical-Longformer) then acts as an automated second
> reader: it reads the discharge summaries of flagged patients only and vetoes alarms the
> narrative does not support, raising precision by 21% while retaining 71% of true positives
> among note-covered admissions — fewer, more credible alerts per true catch. Finally, a
> small instruction-tuned generative model, running entirely locally via Ollama, converts
> each confirmed flag into a brief plain-language rationale grounded exclusively in the
> patient's recorded risk factors. No patient data leaves the hospital environment at any
> stage: every model is small, open, and locally deployable by design.
>
> Beyond aggregate metrics, the thesis contributes a disagreement analysis examining *where*
> and *why* narrative evidence overturns structured predictions — making explicit what notes
> add over labs and codes — and a fairness audit that exposed, and then addressed through
> age-stratified training and per-group calibration, a recall disparity affecting patients
> over 70. An ablation against structured-only, notes-only, and score-fusion alternatives
> tests whether the cascade design earns its place; planned expert interviews with clinicians
> and MIMIC-affiliated researchers assess the clinical plausibility of the pipeline and the
> faithfulness of its generated explanations.
>
> The result is intended less as another readmission model than as a transferable pattern
> for clinical AI: pair a cheap, sensitive screen with a careful, expensive reader — and
> make every alert explain itself.

*(~370 words. Numbers reflect current results; Stage 2 figures are from the v1 model and
will be refreshed after the fairness-aware rebuild is trained — see backlog C2.)*

---

## 5. If the professor asks…

- **"Why not one multimodal model?"** → Efficiency (the expensive reader runs only on ~13%
  of admissions), robustness to missing notes, modular interpretability, and the workflow
  analogy — plus the planned ablation (N1) to show it empirically.
- **"Isn't 0.31 precision still low?"** → That's 3.2 alerts per caught readmission vs. 3.9
  without verification, at 41% fewer total alerts; decision-curve analysis (N4) will express
  this as net clinical benefit.
- **"What's novel?"** → The *verification* framing (LLM as second reader on a high-recall
  screen), the disagreement analysis, and a fairness-audited, fully local implementation —
  not raw AUROC.
