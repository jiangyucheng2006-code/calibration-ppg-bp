# Phase-6B tail-aware factorial screening

Last updated: 2026-08-19.

## Conclusion

The fourth-round development screen is complete. Five new jobs tested the
remaining cells of the prespecified `PPG quality gate x Huber loss x
participant-CVaR` factorial. Together with the previously completed M0,
Huber-only, and quality-gate-only runs, eight settings were compared on the
same seed, participants, and future-query events.

`Quality gate + Huber` is the best full-coverage setting. Across
`K = 1, 2, 3, 5`, its Overall participant-macro mean MAE is 8.888 mmHg,
compared with 9.137 mmHg for fixed-first M0 and 8.962 mmHg for the quality
gate alone. The paired participant bootstrap difference versus M0 is
-0.249 mmHg (exploratory 95% interval -0.353 to -0.150). Adding Huber to the
quality gate contributes a further -0.074 mmHg (interval -0.124 to -0.026).

Participant-CVaR does not improve full-cohort performance. It is therefore
not promoted into the provisional base model. It slightly improves the
observed-error tail in an oracle analysis, but that gain cannot justify a
deployable method because true query error is unavailable at inference.

The quality-gate-plus-Huber gain is source-asymmetric: the four-K improvement
versus M0 is 0.472 mmHg in MIMIC but only 0.065 mmHg in VitalDB. This remains a
single-seed development result and does not yet establish a final model.

## Integrity gates

- Jobs 840--844 completed with exit code `0:0`; all stderr files are empty.
- All eight settings use seed `20260813`, the fixed-first protocol, and the
  identical event-6-onward query set.
- Each setting contains 697 meta-validation participants and 103,564 query
  events at every K.
- Query keys and targets match exactly across settings.
- Work and NAS result artifacts are byte-identical.
- Every run records `locked_test_accessed=false`; the quarantined meta-test was
  not accessed.
- Participant-macro MAE is primary. Event-pooled metrics are secondary.
- MIMIC and VitalDB are internal PulseDB source strata, not external
  validation datasets.

## Full factorial: participant-macro results

Values are the average of participant-macro SBP and DBP MAE, in mmHg. The
four-K mean averages the prespecified `K = 1, 2, 3, 5` results.

| Setting | K=1 | K=2 | K=3 | K=5 | Four-K mean | Change vs M0 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-first M0 | 9.559 | 9.189 | 9.018 | 8.782 | 9.137 | -- |
| Huber | 9.440 | 9.137 | 8.981 | 8.756 | 9.078 | -0.059 |
| PPG quality gate | 9.386 | 9.035 | 8.840 | 8.587 | 8.962 | -0.175 |
| Participant CVaR | 9.617 | 9.216 | 9.017 | 8.763 | 9.153 | +0.016 |
| **Quality gate + Huber** | **9.350** | **8.959** | **8.756** | **8.485** | **8.888** | **-0.249** |
| Quality gate + CVaR | 9.386 | 9.057 | 8.871 | 8.630 | 8.986 | -0.151 |
| Huber + CVaR | 9.563 | 9.174 | 8.970 | 8.709 | 9.104 | -0.033 |
| Quality gate + Huber + CVaR | 9.337 | 8.991 | 8.790 | 8.522 | 8.910 | -0.227 |

CVaR-only is worse than M0 on the four-K average. Adding CVaR to the quality
gate, Huber, or their combination also worsens the corresponding full-cohort
average. The smallest setting with the best full-cohort result is therefore
`Quality gate + Huber`.

## Overall and source-stratified four-K results

Each cell lists `SBP MAE / DBP MAE / mean MAE`, all participant-macro and in
mmHg. Overall is recomputed over all participants rather than formed by
averaging the two source rows.

| Setting | Overall | MIMIC | VitalDB |
|---|---:|---:|---:|
| Fixed-first M0 | 11.627 / 6.648 / 9.137 | 12.682 / 7.099 / 9.891 | 10.751 / 6.273 / 8.512 |
| Huber | 11.652 / 6.504 / 9.078 | 12.688 / 6.879 / 9.783 | 10.794 / 6.194 / 8.494 |
| PPG quality gate | 11.377 / 6.547 / 8.962 | 12.162 / 6.929 / 9.546 | 10.725 / 6.231 / 8.478 |
| Participant CVaR | 11.704 / 6.602 / 9.153 | 12.705 / 6.942 / 9.824 | 10.874 / 6.320 / 8.597 |
| **Quality gate + Huber** | **11.300 / 6.475 / 8.888** | **12.027 / 6.811 / 9.419** | **10.697 / 6.197 / 8.447** |
| Quality gate + CVaR | 11.329 / 6.643 / 8.986 | 12.071 / 7.087 / 9.579 | 10.714 / 6.275 / 8.494 |
| Huber + CVaR | 11.657 / 6.551 / 9.104 | 12.636 / 6.841 / 9.739 | 10.846 / 6.309 / 8.578 |
| Quality gate + Huber + CVaR | 11.310 / 6.510 / 8.910 | 12.033 / 6.823 / 9.428 | 10.710 / 6.250 / 8.480 |

The winner is directionally better than M0 in both sources, but almost all of
the magnitude comes from MIMIC. For `Quality gate + Huber` versus M0, the
four-K paired participant-bootstrap differences are:

| Scope | Mean difference | Exploratory 95% interval |
|---|---:|---:|
| Overall | -0.249 | -0.353 to -0.150 |
| MIMIC | -0.472 | -0.667 to -0.288 |
| VitalDB | -0.065 | -0.161 to +0.030 |

Intervals use 20,000 participant-cluster bootstrap repetitions and preserve
the participant as the resampling unit. They are exploratory and unadjusted:
all candidates were screened on the same meta-validation set, so these are not
confirmatory hypothesis tests.

## Winning setting: requested K=5 diagnostic table

`ME = prediction - reference`. Percentages are event-pooled secondary
diagnostics; they do not replace participant-macro MAE for model selection.

| Scope | BP | MAE | R2 | ME | STD | <=5 mmHg | <=10 mmHg | <=15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Overall | SBP | 11.640 | 0.515 | -1.253 | 15.283 | 29.62% | 53.75% | 71.21% | FAIL* | Grade D, FAIL* |
| Overall | DBP | 6.816 | 0.485 | -0.634 | 9.425 | 48.25% | 78.12% | 90.97% | FAIL* | Grade C, FAIL* |
| MIMIC | SBP | 12.043 | 0.510 | -1.548 | 15.754 | 28.70% | 52.32% | 69.57% | FAIL* | Grade D, FAIL* |
| MIMIC | DBP | 7.080 | 0.469 | -0.777 | 9.805 | 46.87% | 76.83% | 89.94% | FAIL* | Grade C, FAIL* |
| VitalDB | SBP | 10.200 | 0.508 | -0.200 | 13.419 | 32.91% | 58.83% | 77.06% | FAIL* | Grade D, FAIL* |
| VitalDB | DBP | 5.873 | 0.555 | -0.125 | 7.904 | 53.20% | 82.75% | 94.65% | PASS* | Grade B, PASS* |

The asterisks denote retrospective numerical screens only. PulseDB is not a
prospective validation of a finished device, so formal AAMI/ISO/IEEE or BHS
device compliance is not established.

## Fixed-reference worst-30% analysis

The reference tail is defined once from fixed-first M0 at K=5 and reused for
every candidate. The Overall tail contains exactly
`ceil(0.30 x 697) = 210` participants. This definition uses observed query
error, so it is an oracle diagnostic and is not available to a deployed
model.

| Setting on the fixed M0 tail | Tail mean MAE |
|---|---:|
| Fixed-first M0 | 13.714 |
| Huber | 13.524 |
| PPG quality gate | 12.921 |
| Participant CVaR | 13.444 |
| Quality gate + Huber | 12.812 |
| Quality gate + CVaR | 12.816 |
| Huber + CVaR | 13.423 |
| **Quality gate + Huber + CVaR** | **12.765** |

The three-factor setting is 0.047 mmHg better than quality-gate-plus-Huber on
this oracle tail, but it is 0.022 mmHg worse on the full four-K cohort. This is
not sufficient to promote CVaR. It instead supports the separate, already
specified Phase-6C question: train a cross-fitted input-only risk model to
recognise similar difficult cases and evaluate a specialist or fusion model
without access to true query error.

## Development decision

1. Promote `Quality gate + Huber` from a screening candidate to the
   **provisional full-coverage candidate**.
2. Do not promote participant-CVaR into the base model; retain its tail result
   as an ablation and oracle diagnostic.
3. Do not access the locked meta-test yet.
4. Evaluate the separate Phase-6C deployable hard-case classifier and
   specialist/fusion models before deciding whether a two-stage system is
   justified.
5. If the Phase-6C router fails, retain the simpler quality-gate-plus-Huber
   full-coverage model rather than claiming that the oracle worst 30% can be
   identified.
6. Before final model selection, repeat only the surviving candidate(s) across
   multiple seeds and require Overall, MIMIC, and VitalDB reporting.

## Reproducibility artifacts

The machine-readable tables are stored under
`results/phase6b_factorial/`:

- Overall, MIMIC, and VitalDB requested diagnostic tables;
- participant-macro results by scope and K;
- fixed-reference oracle-tail comparisons;
- paired participant-cluster bootstrap comparisons.

The private tail-membership file containing participant identifiers is
intentionally not published. The locked meta-test remains quarantined.
