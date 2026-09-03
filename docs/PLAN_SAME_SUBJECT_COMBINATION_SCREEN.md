# Same-subject module-combination screen

## Purpose

This terminal development screen asks whether modules that helped individually
can improve the selected participant-indexed rank-4 LoRA model when combined.
It is a **seen-participant** experiment under
`development-calbased-analogue-v1`, not an unseen-participant K-shot result and
not an official PulseDB CalBased reproduction.

The held-out role is not readable by training, checkpoint selection, either
split report, or the final cross-split report. The participant-disjoint
`event120-v1` locked meta-test is unchanged.

## Evidence used to choose modules

The preceding one-factor screen ranked these six modules highest by Overall
participant-macro SBP/DBP mean MAE:

| Rank | Single-component setting | Mean MAE (mmHg) |
|---:|---|---:|
| 1 | participant-indexed rank-4 LoRA | 3.0853 |
| 2 | FiLM | 4.7803 |
| 3 | support attention | 5.2440 |
| 4 | adaptive multiple-event weighting | 5.2921 |
| 5 | support reliability weighting | 5.3316 |
| 6 | calibration-relative correction | 5.3379 |

LoRA is therefore fixed as the core in every candidate. Modules that worsened
or barely changed the matched reference are not added merely to enlarge the
network.

## Frozen candidate matrix

The matrix is bounded to 15 candidates rather than exhaustively searching all
`2^6` subsets:

1. `lora`
2. `lora_film`
3. `lora_attention`
4. `lora_multi_event`
5. `lora_reliability`
6. `lora_calibration_relative`
7. `lora_film_attention`
8. `lora_film_multi_event`
9. `lora_film_reliability`
10. `lora_film_calibration_relative`
11. `lora_attention_calibration_relative`
12. `lora_multi_event_reliability`
13. `lora_film_attention_calibration_relative`
14. `lora_film_multi_event_reliability`
15. `lora_all_six`

This covers every LoRA-plus-one effect, selected mechanistically related
three-module combinations, two four-module combinations, and the full six-
module model. All candidates use the compact ResNet encoder, the train-role
participant BP-mean anchor, rank-4 subject-indexed LoRA, Huber loss, seed
`20260902`, no epoch cap, and patience 8.

## Split modes and evaluation

Every candidate is trained independently under both:

- `random_disjoint`;
- `chronological_blocked`.

Each split report uses the identical full internal-validation cohort within
that mode and emits Overall, MIMIC, and VitalDB participant-macro results from
the same saved predictions. Event-pooled AAMI/BHS columns are secondary
numerical screens only and do not establish device or clinical compliance.

## Prespecified decision rule

For a combination to replace `lora`, it must:

1. improve Overall participant-macro mean MAE by at least 0.15 mmHg in both
   split modes;
2. improve MIMIC and VitalDB mean MAE in both split modes;
3. use full internal-validation coverage and access no held-out targets.

Among candidates passing all conditions, select the lowest average Overall
mean MAE across the two split modes. If none passes, retain `lora`.

This screen selects a development candidate only. Before a one-time held-out
evaluation, complete the identity-shuffle, shared-adapter, anchor-removal, and
rank/parameter controls and freeze the final implementation.

## Claim boundary

Every candidate uses a participant-indexed adapter jointly learned from 320
labelled train-role windows for each already-seen participant. Results cannot be
described as calibration-free performance, unseen-user K=1/2/3/5 calibration,
or external validation.

## Completed decision

All 30 candidate runs and the three reporting jobs completed successfully.
No non-reference combination met the prespecified 0.15-mmHg gain in both
split modes. The final selector therefore retained plain LoRA according to the
fallback rule. See
[RESULTS_SAME_SUBJECT_COMBINATIONS.md](RESULTS_SAME_SUBJECT_COMBINATIONS.md)
and the path-free aggregate files under
`../results/same_subject_combinations/` for the completed evidence.
