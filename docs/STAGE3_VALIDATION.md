# Stage 3 Manual Validation Plan

## Motivation

Stage 3 uses phi4-mini to classify each patient into one of three discordance
modes (CONCORDANT / NOTE_MITIGATES / NOTE_AMPLIFIES) and one of nine clinical
categories.  Because this is LLM-generated output, the thesis must report how
well it agrees with human clinical judgment — otherwise the population-level
finding has no validity anchor.

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

phi4-mini annotation:
  discordance_mode:   [CONCORDANT / NOTE_MITIGATES / NOTE_AMPLIFIES]
  primary_category:   [...]
  explanation:        [...]

Your annotation:
  discordance_mode:   _______________
  primary_category:   _______________
  Do you agree with the explanation? [YES / PARTIAL / NO]
  Notes: _______________
```

Annotators are blinded to each other's labels until both are complete.

## Metrics

- **Cohen's kappa (κ)** on `discordance_mode` (primary)
- **Percent agreement** on `primary_category` (secondary — 9 classes, too many for reliable kappa on 50 cases)
- **Annotation failure rate** cross-check: compare manual failure judgements to `annotation_failed=True` flag in output

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
