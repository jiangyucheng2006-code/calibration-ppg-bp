# Same-subject terminal module-combination screen: result

## Conclusion

All 30 training jobs, both split-specific reports, and the final cross-split
selector completed successfully with exit code `0:0`. No non-reference
combination passed the prespecified promotion rule. The development selector
therefore **retains participant-indexed rank-4 LoRA** as the base method.

The lowest random-disjoint result is `lora_film_reliability`, with Overall
participant-macro SBP/DBP/mean MAE of **3.8668/2.1544/3.0106 mmHg**. The
lowest chronological-blocked result is `lora_film_attention`, with
**4.8925/2.6737/3.7831 mmHg**. These are different models. Their Overall mean
MAE gains over the paired LoRA reference are only 0.0347 and 0.0896 mmHg,
respectively, and neither reaches the frozen 0.15-mmHg margin in both split
modes.

This is a development-only, seen-participant result based on 320 labelled
training windows per participant. It is not unseen-participant K=1/2/3/5
calibration, not an exact reproduction of the official PulseDB CalBased
benchmark, and not external validation. The same-subject held-out role and the
participant-disjoint locked meta-test were not accessed.

## Frozen comparison

- Protocol: `development-calbased-analogue-v1`.
- Screen: `same-subject-combination-v1`.
- Seed: `20260902`.
- Candidates: 15 in each of `random_disjoint` and
  `chronological_blocked`.
- Cohort: 2,051 participants: 1,011 MIMIC and 1,040 VitalDB.
- Internal validation: 82,040 windows: 40,440 MIMIC and 41,600 VitalDB.
- Per participant: 320 labelled train-role windows and 40 internal-validation
  windows; the 40-window held-out role remained sealed.
- Shared core: compact ResNet, participant train-role BP-mean anchor,
  participant-indexed rank-4 LoRA, Huber loss, no epoch cap, and patience 8.
- Selection metric: Overall participant-macro mean of SBP and DBP MAE.
- Promotion rule: at least 0.15-mmHg Overall mean-MAE gain over LoRA in both
  split modes, plus positive MIMIC and VitalDB gains in both modes. If no
  combination passes, retain LoRA.

Overall, MIMIC, and VitalDB were recomputed from the same saved predictions in
each run. MIMIC and VitalDB are internal PulseDB source strata, not independent
external datasets.

## Best numerical combinations across both split modes

The cross-split mean below is the average of the two Overall participant-macro
mean MAEs. It is a ranking aid, not the promotion criterion.

| Rank | Candidate | Random mean MAE | Chronological mean MAE | Cross-split mean | Random gain vs LoRA | Chronological gain vs LoRA | Promotion |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `lora_film_reliability` | 3.0106 | 3.7889 | **3.3997** | 0.0347 | 0.0839 | Fail |
| 2 | `lora_film` | 3.0258 | 3.7907 | 3.4082 | 0.0195 | 0.0821 | Fail |
| 3 | `lora_film_multi_event` | 3.0267 | 3.7901 | 3.4084 | 0.0186 | 0.0827 | Fail |
| 4 | `lora_film_calibration_relative` | 3.0315 | 3.7919 | 3.4117 | 0.0138 | 0.0808 | Fail |
| 5 | `lora_film_attention_calibration_relative` | 3.0390 | 3.7938 | 3.4164 | 0.0063 | 0.0789 | Fail |
| Reference | `lora` | 3.0453 | 3.8727 | 3.4590 | 0.0000 | 0.0000 | **Retained by fallback rule** |

`lora_film_attention` is the chronological winner at 3.7831 mmHg, but its
random-disjoint mean MAE is 3.0955 mmHg, 0.0503 mmHg worse than LoRA. The
all-six model is also not superior: it records 3.0632 mmHg under random
disjoint and 3.7978 mmHg under chronological blocked. More modules therefore
do not produce a reliable monotonic gain.

## Split-specific numerical winners

Participant-macro MAE is the primary result.

| Split | Scope | Participants | Events | Candidate | SBP MAE | DBP MAE | Mean MAE |
|---|---|---:|---:|---|---:|---:|---:|
| Random disjoint | Overall | 2,051 | 82,040 | `lora_film_reliability` | 3.8668 | 2.1544 | **3.0106** |
| Random disjoint | MIMIC | 1,011 | 40,440 | `lora_film_reliability` | 4.2284 | 2.3353 | **3.2818** |
| Random disjoint | VitalDB | 1,040 | 41,600 | `lora_film_reliability` | 3.5153 | 1.9785 | **2.7469** |
| Chronological blocked | Overall | 2,051 | 82,040 | `lora_film_attention` | 4.8925 | 2.6737 | **3.7831** |
| Chronological blocked | MIMIC | 1,011 | 40,440 | `lora_film_attention` | 4.6546 | 2.4743 | **3.5645** |
| Chronological blocked | VitalDB | 1,040 | 41,600 | `lora_film_attention` | 5.1237 | 2.8676 | **3.9956** |

## Retained LoRA reference by split and source

This is the model selected by the prespecified fallback rule, not the lowest
raw score within either individual split.

| Split | Scope | Participants | Events | SBP MAE | DBP MAE | Mean MAE |
|---|---|---:|---:|---:|---:|---:|
| Random disjoint | Overall | 2,051 | 82,040 | 3.9062 | 2.1843 | **3.0453** |
| Random disjoint | MIMIC | 1,011 | 40,440 | 4.2713 | 2.3683 | **3.3198** |
| Random disjoint | VitalDB | 1,040 | 41,600 | 3.5514 | 2.0055 | **2.7784** |
| Chronological blocked | Overall | 2,051 | 82,040 | 4.9664 | 2.7791 | **3.8727** |
| Chronological blocked | MIMIC | 1,011 | 40,440 | 4.7811 | 2.5859 | **3.6835** |
| Chronological blocked | VitalDB | 1,040 | 41,600 | 5.1465 | 2.9669 | **4.0567** |

Chronological blocking increases the LoRA Overall mean MAE by 0.8275 mmHg
relative to random disjoint windows. This supports a substantial temporal-
shift/interpolation gap even when every evaluated participant is already
known to the model.

## Requested event-pooled diagnostic table for split winners

Participant-macro MAE above is primary. The following R-squared, signed error,
error standard deviation, threshold percentages, and standards-style fields
are secondary event-pooled diagnostics reconstructed from the same saved
predictions.

| Split | Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Random | Overall | SBP | 3.8668 | 0.9263 | 0.1379 | 5.7452 | 74.57% | 93.05% | 97.58% | PASS* | Grade A* |
| Random | Overall | DBP | 2.1544 | 0.9232 | -0.0071 | 3.5548 | 91.36% | 98.29% | 99.38% | PASS* | Grade A* |
| Random | MIMIC | SBP | 4.2284 | 0.9226 | 0.2607 | 6.3282 | 71.68% | 91.31% | 96.69% | PASS* | Grade A* |
| Random | MIMIC | DBP | 2.3353 | 0.9074 | -0.0606 | 4.0462 | 89.92% | 97.57% | 98.99% | PASS* | Grade A* |
| Random | VitalDB | SBP | 3.5153 | 0.9272 | 0.0185 | 5.1124 | 77.38% | 94.74% | 98.44% | PASS* | Grade A* |
| Random | VitalDB | DBP | 1.9785 | 0.9400 | 0.0449 | 3.0001 | 92.76% | 98.99% | 99.76% | PASS* | Grade A* |
| Chronological | Overall | SBP | 4.8925 | 0.8759 | -0.2195 | 7.2376 | 65.91% | 87.71% | 95.00% | PASS* | Grade B* |
| Chronological | Overall | DBP | 2.6737 | 0.8787 | -0.0688 | 4.3322 | 85.83% | 96.61% | 98.90% | PASS* | Grade A* |
| Chronological | MIMIC | SBP | 4.6546 | 0.9022 | -0.2691 | 7.0803 | 68.95% | 88.68% | 95.30% | PASS* | Grade A* |
| Chronological | MIMIC | DBP | 2.4743 | 0.8973 | -0.1062 | 4.1412 | 87.83% | 96.71% | 98.73% | PASS* | Grade A* |
| Chronological | VitalDB | SBP | 5.1237 | 0.8272 | -0.1714 | 7.3870 | 62.96% | 86.76% | 94.71% | PASS* | Grade B* |
| Chronological | VitalDB | DBP | 2.8676 | 0.8528 | -0.0325 | 4.5099 | 83.88% | 96.51% | 99.06% | PASS* | Grade A* |

The starred entries are retrospective numerical screens only. They do not
establish formal device, clinical, AAMI/ISO/IEEE, or BHS compliance because
the dataset, reference procedure, and validation protocol do not constitute a
formal device-validation study.

## Interpretation

1. **The LoRA effect is much larger than the add-on effects.** The previous
   one-factor screen reduced Overall mean MAE by 2.8771 mmHg when moving from
   the paired residual reference to participant-indexed LoRA. In this terminal
   screen, the best additional reduction is only 0.0347 mmHg under random
   disjoint windows and 0.0896 mmHg under chronological blocking.
2. **The best add-on is not stable across split modes.** Reliability weighting
   leads under random disjoint windows, while attention leads under
   chronological blocking. This weakens the case for promoting either as a
   general improvement.
3. **Complexity does not add monotonically.** Several combinations worsen the
   random-disjoint reference, and the full six-module model is not the best in
   either split. Further unconstrained module stacking is not justified.
4. **The claim is still persistent known-user personalization.** Every
   participant has a jointly learned personal adapter trained from 320 labelled
   windows. The result does not show that a new participant can be calibrated
   from one to five independent cuff events.

## Decision and next gate

Do not select `lora_film_reliability` merely because it has the lowest average
score. The improvement is below the frozen margin, and the random/chronological
leaders disagree. Retain plain participant-indexed rank-4 LoRA as the
development base.

Before any one-time held-out evaluation, run the minimum mechanism controls:

1. shuffled participant index;
2. shared adapter instead of participant-indexed adapters;
3. removal of the participant train-role BP-mean anchor;
4. LoRA rank and parameter-matched controls.

These controls determine whether the large LoRA gain reflects a meaningful
person-specific PPG mapping or mainly identity memorization, anchor use, and
parameter-table capacity. Only after that mechanism is frozen should one model
be refitted and evaluated once on the sealed same-subject held-out role. The
participant-disjoint event-level K=1/2/3/5 track remains a separate primary
research question.

## Public result files

- [`participant_macro.csv`](../results/same_subject_combinations/participant_macro.csv):
  all 15 candidates, both split modes, and all three participant-macro scopes.
- [`event_pooled_diagnostics.csv`](../results/same_subject_combinations/event_pooled_diagnostics.csv):
  requested diagnostic fields for every candidate, scope, split, and BP
  endpoint.
- [`cross_split_comparison.csv`](../results/same_subject_combinations/cross_split_comparison.csv):
  paired random/chronological gains and final gate fields.
- [`selected_lora_views.csv`](../results/same_subject_combinations/selected_lora_views.csv):
  the retained LoRA reference in Overall, MIMIC, and VitalDB for both splits.
- [`split_winner_views.csv`](../results/same_subject_combinations/split_winner_views.csv):
  the two split-specific numerical winners.
- [`selection.json`](../results/same_subject_combinations/selection.json):
  path-free final decision and held-out-access declaration.
- [`manifest.json`](../results/same_subject_combinations/manifest.json):
  execution, report hashes, integrity checks, and claim boundary.
