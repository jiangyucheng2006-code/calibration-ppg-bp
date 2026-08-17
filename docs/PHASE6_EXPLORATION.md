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

| Candidate | Meta-training support | Loss | Purpose |
|---|---|---|---|
| M0 reference | latest K prior events | MSE | Existing comparison result |
| Fixed-first M0 | events 1 to K | MSE | Match meta-training support to the frozen evaluation support |
| Robust-loss M0 | latest K prior events | Huber, delta 0.5 standardized units | Reduce sensitivity to large residuals |

Slurm jobs 818 and 819 run the fixed-first and robust-loss candidates, respectively. No repeated-seed or multi-fold job is part of this submission.

## Residual-tail audit

The development-only K=5 audit contains 697 participants and 103,564 common queries. Participant mean-MAE quantiles are:

| Quantile | Mean MAE (mmHg) |
|---:|---:|
| 50% | 7.722 |
| 90% | 14.665 |
| 95% | 16.919 |
| 99% | 22.492 |

The top 10% error group has larger within-participant BP variability and a later average event horizon. MIMIC also has a higher high-error proportion than VitalDB. Coverage-error calculations based on observed query error are labelled oracle-only; they are not deployable quality screening and must not be presented as headline model accuracy.

## Demographic audit decision

Age and sex were audited on the development manifest before any conditioning model was submitted. Sex is complete and internally consistent. Age is not ready for model input: 76 participants have more than one age value within the same participant, 25 participants are younger than 18, and the minimum recorded value is zero. Demographic conditioning is therefore paused until a prespecified aggregation and validity rule is implemented using development data only.

## Leakage controls

- Participant split remains subject-disjoint.
- Validation support remains the first K events and queries remain event 6 onward.
- Meta-validation targets are used only for evaluation and early stopping.
- No query BP, query residual, or true error may be used by a deployable quality gate.
- The locked meta-test remains untouched.
