# Phase-6 single-seed improvement screening

## Objective

Phase 6 searches for a meaningful improvement over the parsimonious M0 model without spending repeated-seed or multi-fold compute on unpromising ideas. Model selection remains restricted to `meta_validation`; the locked meta-test is not accessed.

## Execution rule

- Development seed: `20260813` only.
- Training has no epoch-count cap and stops after eight consecutive epochs without improvement.
- Interventions are tested separately before combinations are considered.
- A candidate is promoted only if it improves multiple calibration budgets, improves or preserves SBP error, and does not worsen the participant-level error tail materially.
- Multi-seed and multi-fold confirmation are deferred until a candidate passes this screening gate.

## Current candidates

| Candidate | Single changed factor | Purpose | Slurm job |
|---|---|---|---:|
| M0 reference | none | Existing rolling-support comparison result | 783 |
| Fixed-first protocol ablation | support policy | Match meta-training support to the frozen evaluation support | 818 |
| Robust loss | Huber loss, delta 0.5 standardized units | Reduce sensitivity to large residuals | 819 |
| BP-change sampling | meta-train episode sampler | Emphasize calibration-drift episodes without validation leakage | 822 |
| Robust anchor | coordinate-wise median support residual | Reduce sensitivity to an atypical cuff calibration event | 823 |
| PPG quality gate | PPG-only personalization gate | Attenuate unreliable personalization without query BP/error | 824 |
| Demographic conditioning | cleaned age/sex vector | Test whether subject context adds information beyond PPG and cuff anchors | 825 |

No repeated-seed or multi-fold job is part of this submission. Every candidate
changes only one factor relative to the original M0 configuration; combinations
will be considered only after isolated screening.

## Residual-tail audit

The development-only K=5 audit contains 697 participants and 103,564 common queries. Participant mean-MAE quantiles are:

| Quantile | Mean MAE (mmHg) |
|---:|---:|
| 50% | 7.722 |
| 90% | 14.665 |
| 95% | 16.919 |
| 99% | 22.492 |

The prespecified worst-20% participant analysis is reported in
[PHASE6_HIGH_ERROR_ANALYSIS.md](PHASE6_HIGH_ERROR_ANALYSIS.md). The tail has
larger within-participant BP variability, larger support-to-query BP changes,
more targets outside the support BP range, and a later average event horizon.
MIMIC also has a higher tail proportion than VitalDB. PPG amplitude and mean
age are nearly unchanged. Coverage-error calculations based on observed query
error remain oracle-only and cannot be used as deployable quality screening.

## Demographic audit decision

Age and sex were audited and converted to a participant-level development
table. Age uses the participant median of finite values from 18 to 100 years;
25 participants without a valid adult age are retained with `age_valid=0`, and
76 participants with multiple valid age records are summarized by the median.
Age normalization is fitted on meta-train participants only. Sex uses the
participant mode with an explicit unknown channel. This cleaned representation
is evaluated as one isolated candidate rather than being added to every model.

## Leakage controls

- Participant split remains subject-disjoint.
- Validation support remains the first K events and queries remain event 6 onward.
- Meta-validation targets are used only for evaluation and early stopping.
- No query BP, query residual, or true error may be used by a deployable quality gate.
- The locked meta-test remains untouched.
