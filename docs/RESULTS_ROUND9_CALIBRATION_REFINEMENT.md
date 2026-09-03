# Round-9 calibration-refinement results

## Conclusion

Round 9 did **not** identify a candidate that passes the prespecified internal
promotion gate. The numerical winner, R9-1 adaptive population/personal
fusion, improves Overall participant-macro mean MAE by only `0.0363 mmHg`
relative to the architecture-matched internal reference and is `0.0029 mmHg`
worse in VitalDB. It therefore is not promoted as the new base model.

R9-7 temporal-delta consistency is the only candidate that improves both
internal PulseDB source strata, but its Overall gain is only `0.0097 mmHg`, far
below the required `0.15 mmHg`. The result supports moving to a substantively
different representation-learning experiment rather than combining additional
small correction heads.

## Experimental boundary

- Calibration regime: fixed-first `K=5` pseudo-cuff/reference-BP events.
- Query regime: all eligible event-6-onward queries.
- Selection split: participant-disjoint meta-train internal folds.
- Fit: folds 0--2; patience-8 early stopping: fold 3; candidate ranking: fold 4.
- Fold-4 evaluation: 628 participants and 96,332 queries.
- MIMIC: 285 participants and 74,373 queries.
- VitalDB: 343 participants and 21,959 queries.
- Seed: `20260823`.
- Meta-validation was not used for training, early stopping, prediction,
  scoring, or candidate ranking in this screen.
- The locked meta-test was not accessed.

These are exploratory internal-fold results. They are not directly comparable
to the Round-8 meta-validation values because the evaluated participants are
different. MIMIC and VitalDB are internal PulseDB source strata, not independent
external validation datasets.

## Execution and integrity

Slurm jobs 975--983 and deterministic report job 984 all completed with exit
code `0:0`; every stderr file is empty. Extended report job 985 also completed
with exit code `0:0` and empty stderr. The work and NAS report directories are
byte-identical. All nine settings use the same fold-4 query keys and targets.

The full server suite passed 97 tests. The report code additionally verifies
the internal split, seed, meta-validation-use flags, locked-test flag, query-key
identity, target identity, finite predictions, and Overall/MIMIC/VitalDB
coverage.

## Primary participant-macro result

Negative values in the final column indicate improvement over R9-0.

| Setting | SBP MAE | DBP MAE | Mean MAE | Mean-MAE change |
|---|---:|---:|---:|---:|
| R9-0 R8 architecture reference | 10.5314 | 5.9448 | 8.2381 | 0.0000 |
| R9-1 Adaptive base-personal fusion | 10.5091 | 5.8946 | **8.2019** | **-0.0363** |
| R9-2 Soft BP-range experts | 10.5434 | 5.9453 | 8.2444 | +0.0063 |
| R9-3 DBP-specific physiology | 10.5384 | 5.9847 | 8.2616 | +0.0235 |
| R9-4 Bias regularization | 10.5680 | 6.0020 | 8.2850 | +0.0469 |
| R9-5 Short causal attention | 10.6413 | 6.0257 | 8.3335 | +0.0954 |
| R9-6 Personal BP direction | 10.4964 | 5.9719 | 8.2341 | -0.0040 |
| R9-7 Temporal delta consistency | 10.5109 | 5.9460 | 8.2284 | -0.0097 |
| R9-8 Support dropout consistency | 10.6053 | 6.0173 | 8.3113 | +0.0732 |

MAE values are participant-macro and expressed in mmHg.

## Internal source-stratified result

| Setting | MIMIC mean MAE | Change | VitalDB mean MAE | Change |
|---|---:|---:|---:|---:|
| R9-0 R8 architecture reference | 8.9065 | 0.0000 | 7.6827 | 0.0000 |
| R9-1 Adaptive base-personal fusion | **8.8231** | **-0.0834** | 7.6856 | +0.0029 |
| R9-2 Soft BP-range experts | 8.9004 | -0.0061 | 7.6993 | +0.0166 |
| R9-3 DBP-specific physiology | 8.9231 | +0.0165 | 7.7119 | +0.0292 |
| R9-4 Bias regularization | 8.9526 | +0.0461 | 7.7303 | +0.0476 |
| R9-5 Short causal attention | 9.0393 | +0.1327 | 7.7470 | +0.0643 |
| R9-6 Personal BP direction | 8.9111 | +0.0045 | 7.6717 | -0.0111 |
| R9-7 Temporal delta consistency | 8.9003 | -0.0063 | **7.6702** | **-0.0125** |
| R9-8 Support dropout consistency | 9.0129 | +0.1064 | 7.7283 | +0.0455 |

Overall was recomputed from all participants and events; it is not the average
of the two source-specific values.

## Requested diagnostic view for the numerical winner

The event-pooled diagnostic table below is secondary to participant-macro MAE.
ME is prediction minus reference BP; STD is the sample standard deviation of
that signed error.

| Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Overall | SBP | 12.0494 | 0.4739 | -0.8432 | 16.1666 | 29.79% | 53.75% | 70.32% | FAIL* | FAIL (D)* |
| Overall | DBP | 6.4960 | 0.5057 | -0.9509 | 9.1677 | 51.54% | 79.39% | 91.93% | FAIL* | PASS (B)* |
| MIMIC | SBP | 12.7236 | 0.4472 | -0.9906 | 16.9726 | 28.55% | 51.43% | 67.72% | FAIL* | FAIL (D)* |
| MIMIC | DBP | 6.7139 | 0.4779 | -1.0530 | 9.4655 | 50.63% | 78.10% | 91.07% | FAIL* | PASS (B)* |
| VitalDB | SBP | 9.7657 | 0.5342 | -0.3439 | 13.0604 | 33.99% | 61.61% | 79.12% | FAIL* | FAIL (D)* |
| VitalDB | DBP | 5.7581 | 0.5889 | -0.6050 | 8.0688 | 54.60% | 83.77% | 94.83% | FAIL* | PASS (B)* |

`AAMI` and `BHS` are retrospective numerical screens only. PulseDB, invasive
ABP-derived labels, the internal participant split, and repeated events do not
satisfy a formal device-validation protocol; no standards-compliance claim is
made.

## Interpretation by intervention

- **Adaptive fusion:** produces the largest Overall improvement, mainly from
  MIMIC, but does not transfer to VitalDB and misses the effect-size gate.
- **Soft BP-range experts:** is essentially neutral Overall and slightly worse
  in VitalDB. A smooth predefined range split does not solve extrapolation.
- **DBP-specific physiology:** worsens both source strata. The current frozen
  physiology features do not add useful DBP information beyond the reference.
- **Bias regularization:** worsens MAE and does not reduce the error STD enough
  to change the numerical-screen conclusion.
- **Short causal attention:** is the worst candidate. Replacing the recurrent
  causal state with bounded attention loses useful temporal information under
  this implementation.
- **Personal BP direction:** slightly improves VitalDB but slightly worsens
  MIMIC, producing an effectively neutral Overall result.
- **Temporal-delta consistency:** is directionally consistent across both
  sources, but its effect is too small to justify promotion.
- **Support dropout consistency:** worsens both sources; randomly removing one
  of only five support events reduces useful calibration information more than
  it improves robustness.

## Decision and next gate

No Round-9 candidate is promoted, and no meta-validation evaluation is run for
these candidates. R8-4 remains the current single-seed meta-validation
candidate, but it is not yet a confirmed final model.

The next justified experiment is a partial end-to-end calibration-aware PPG
encoder: keep early waveform layers frozen, fine-tune the final encoder block
using query-to-support PPG pairs, and add an auxiliary BP-change objective. This
tests whether the current ceiling comes from frozen population embeddings. It
must use the same participant-disjoint internal selection protocol before any
further meta-validation evaluation.

## Public aggregate artifacts

- [Participant-macro table](../results/round9/participant_macro_internal.csv)
- [Overall/MIMIC/VitalDB pooled diagnostic table](../results/round9/pooled_diagnostics_internal.csv)
- [Changes versus the internal reference](../results/round9/comparison_vs_reference_internal.csv)
- [Machine-readable selection decision](../results/round9/selection.json)

Event-level predictions, participant identifiers, checkpoints, logs, and
personal server paths are intentionally not published.
