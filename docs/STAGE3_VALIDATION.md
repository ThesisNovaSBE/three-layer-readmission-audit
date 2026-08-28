# Stage 3 Manual Validation Plan

> **Partly superseded 2026-08-25, re-targeted 2026-08-28 (session 19).** As
> of the current `src/stage3/explain.py`, the LLM no longer chooses the
> discordance mode — that is computed quantitatively (percentile-rank
> displacement; see `docs/ARCHITECTURE.md`), so there is nothing there to
> validate against human judgment. The LLM now outputs `decision_model`
> (uphold / override / insufficient_evidence) plus `mitigating_grounds` /
> `aggravating_grounds` — a fixed two-sided taxonomy with a verbatim quote
> per ground, replacing the single free-choice `primary_clinical_domain`
> from the 2026-08-25 design. A second, code-computed `decision_rule` is
> also reported (not LLM-generated — nothing to validate here either, but
> its *agreement rate with* `decision_model` is itself a useful thing to
> discuss alongside the human-agreement numbers below). The validation
> *protocol* below (sampling design, kappa methodology) is still sound —
> re-targeted at agreement on `decision_model` and the extracted grounds.

## Motivation

Stage 3 uses phi4-mini to independently decide, per patient, whether to
uphold or override the structured risk alert, and to classify the primary
clinical domain behind that decision.  Because this is LLM-generated output,
the thesis must report how well it agrees with human clinical judgment —
otherwise the population-level finding has no validity anchor.

## Target Sample

| Group | N | Rationale |
|---|---|---|
| CONCORDANT | 15 | Baseline — easiest case, model should agree |
| NOTE_MITIGATES | 20 | Core finding: note contradicts structured risk |
| NOTE_AMPLIFIES | 15 | Secondary finding: note adds risk |
| **Total** | **50** | Sufficient for Cohen's kappa with 95% CI |

Cases are sampled uniformly at random from each mode after Stage 3 completes.

## Annotation Protocol

Each case is presented to one annotator (the thesis author) and one independent
annotator (a clinical collaborator or fellow student with clinical training):

```
Patient ID:        [hadm_id — de-identified]
Stage 1 top features:
  [SHAP output]

Top attention sentences from discharge note:
  [extracted spans]

Model annotation:
  decision_model:      [uphold / override / insufficient_evidence]
  decision_rule:        [uphold / override / insufficient_evidence]
  mitigating_grounds:  [ground: quote, ...]
  aggravating_grounds: [ground: quote, ...]
  clinical_justification: [...]

Your annotation:
  decision:            _______________
  mitigating_grounds:  _______________
  aggravating_grounds: _______________
  Do you agree with the justification? [YES / PARTIAL / NO]
  Notes: _______________
```

Annotators are blinded to each other's labels until both are complete.

## Metrics

- **Cohen's kappa (κ)** on `decision_model` (primary) — the three-way
  uphold/override/insufficient_evidence agreement between the annotator and
  the LLM.
- **Set-overlap agreement (Jaccard index)** on the extracted grounds
  (secondary) — grounds are multi-label (a case can have zero, one, or
  several on each side), not a single categorical choice, so simple percent
  agreement doesn't apply cleanly; report the overlap between the
  annotator's and the model's mitigating-grounds sets and aggravating-
  grounds sets separately.
- **`decision_model` vs. `decision_rule` agreement** (descriptive, not a
  human-agreement metric) — report alongside the above as context on how
  often the model's free judgment follows its own extracted grounds.
- **Annotation failure rate** cross-check: compare manual failure judgements
  to `annotation_failed=True` flag in output.

### Interpretation thresholds (Landis & Koch 1977)

| κ | Strength |
|---|---|
| < 0.40 | Poor — model not reliable enough for thesis claim |
| 0.41–0.60 | Moderate — use with caveats |
| 0.61–0.80 | Substantial — acceptable for thesis |
| > 0.80 | Almost perfect — strong result |

## What to Do With the Results

**If κ > 0.61:** Report the kappa value prominently in the thesis. Use it to
justify the population-level discordance findings as a validated research result.

**If κ = 0.41–0.60:** Report with explicit caveats. Note that phi4-mini's
classifications are indicative rather than ground-truth. Shift the thesis claim
from "we found X% NOTE_MITIGATES" to "our LLM classifier, with moderate
agreement to human judgment, suggests X%."

**If κ < 0.40:** The Stage 3 finding is not publishable as a research result.
Options: (a) improve the prompt and re-annotate, (b) frame Stage 3 as a
proof-of-concept / future work section only, (c) add a second LLM judge (GPT-4o
or Claude) and report LLM-LLM agreement alongside human-LLM agreement.

## Jain & Wallace Attention Caveat

When reporting attention-span extraction in the thesis, include the following
methodological note:

> "Attention weights are not guaranteed to constitute faithful explanations of
> model decisions (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).  The spans
> extracted here are used to ground the phi4-mini prompt with contextually
> relevant note content, not as standalone proof of which sentences caused the
> Longformer's prediction.  A gradient-based attribution method (e.g. Integrated
> Gradients) would be required for causally faithful explanations."

## Timeline

| Step | Est. effort |
|---|---|
| Export 50 annotation cases as CSV | 1 h |
| Self-annotation | 3 h |
| Independent annotator completes labels | 1 week |
| Compute kappa, write results section | 1 h |

## Output Files

- `docs/validation_cases.csv` — 50 sampled cases (no raw note text — MIMIC DUA)
- `docs/validation_labels.csv` — annotator labels + kappa computation
- `docs/validation_report.md` — summary to paste into thesis methodology section
