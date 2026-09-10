# Personal enrollment on the original participant partition

Protocol: `legacy-split-enrollment-v1`. Authorized 10 September 2026.
Status: GPU smoke and real preparation passed; fresh shared fitting submitted
and running. See STATUS.md for job IDs, audited counts and snapshot hashes.
No prior checkpoint is eligible for initialization.

## Question and the two candidates

Can a previously unseen person establish an accurate personal predictor using
a substantial labelled history, after the population model is trained on the
original, disjoint training participants? Compare only:

1. **New-person LoRA**: a new rank-4 personal feature adapter and registration
   BP anchor, with the shared network frozen.
2. **New-person LoRA + reference memory**: exactly the same fitted adapter,
   plus the current fixed personal reference-memory mechanism.

The user explicitly requested this focused pair. Additional architectures,
ablation controls, seeds and K=1/2/3/5 sweeps are not included. Earlier memory-only
ablations motivate the comparison but cannot substitute for matched controls
if a subsequent publication claims that every component is necessary.

## Data contract

Reuse the existing audited official 2,506-person, 400-window/person **input
pool**. Apply the ORIGINAL source-stratified participant assignments, with
`subject_splits.csv` SHA-256
`8705f7cd75d92201bd203c00fb4d8ad8c738c02d7c4b56e5747210ffba504cd7`.
Do not randomly sample another 100/200 people or change any saved parent role.

Before content exclusions, the pool intersects the old assignments as follows:

| Original outer role | MIMIC | VitalDB | Total |
|---|---:|---:|---:|
| meta_train | 858 | 881 | 1,739 |
| meta_validation | 188 | 196 | 384 |
| meta_test | 167 | 216 | 383 |

This is not the entire original 5,361-person sampling frame or its event120
eligible cohort. It preserves their outer assignments **within the current
window pool**. The original 120-second event sampling is not reinstated.

Population fitting uses all retained source-pool windows of meta_train people.
For validation and test people, allocate approximately 90% registration and
10% query, outcome-blind with fixed seed 20260910. Registration has a nested
8/9 fit and 1/9 validation split (normally 320/40/40 per 400 windows). Exact
waveform duplicates and positive same-record overlaps are indivisible groups,
so budgets may differ slightly from nominal. Minimum query and nested
validation budgets are five windows each; at least five inner-fit windows.

Quarantine all identities in a connected exact-content/positive-interval
component that crosses outer roles. This does not infer that two IDs identify
the same person. Preserve original files and assignments and report every
quarantine and input-only insufficient-history exclusion. A remaining
cross-role content/interval collision is a hard failure, not a waived check.
Within-role verified source duplicates remain explicitly disclosed.

This is a separately named custom protocol, NOT exact official CalBased or
CalFree. Current original official TRAIN/TEST window assignments do not govern
the new within-person roles. They are not overwritten. Random windows do not
establish that historical data predict later dates.

## Shared training and validation

- Same existing `resnet_small` encoder, 256-D features, residual BP head and
  rank-4 feature LoRA; no new architecture or module combination.
- Fresh model, no prior weights, adapters or memories. Only meta_train people
  fit shared weights, population LoRA, anchors, normalization and BP scaler.
- All retained meta_train windows are permitted fitting data. There is no
  final refit incorporating validation or test participants.
- Epoch selection uses participant-macro MAE on **meta_validation** queries
  with their own registration-only BP anchors and zero personal adapters.
  This is a shared-feature transfer criterion, not an additional reported
  candidate, and does not optimize post-adaptation performance directly.
- AdamW, learning rate 3e-4, weight decay 1e-4, normalized Huber delta 0.5,
  batch size 64, gradient clip 5, eight non-improving epochs, no epoch cap.
  Fixed base seed 20260910; the selection helper initialization uses seed +1
  and training stochastic operations use seed +2, both recorded in code.
- Validate the two full personal methods on validation people after selecting
  the base. Freeze both methods regardless of their ordering before final-test
  personal enrollment. No recursive optimization from test feedback.

## New-person training and persistent history

Each new person starts with a zero-correction adapter, not another person's
profile. As in the current pipeline, personal seed base is 20260909 plus a
stable identity-derived offset; A is small random and B is zero. Exactly
2,048 personal parameters train; encoder, residual head, scaler and BN/dropout
behavior remain fixed. Own nested registration validation selects epochs,
including epoch zero, with patience eight. Initialize a fresh adapter and
refit all registration records for the selected epoch count.

Reference memory is post-adapter 256-D cosine top-five retrieval, temperature
0.1, with the unchanged registration-only leave-40-block q95 distance fusion.
Fallback to LoRA if no legal donor set is available; do not discard a query.
Record actual reference counts and fallback coverage.

Save/reload the adapter, anchor, shared-checkpoint hash, registration IDs,
feature vectors, permitted reference BP and gate state. Check shared-state
immutability and prediction equivalence. Both candidates use the **same fitted
LoRA**: the paired contrast changes only memory at prediction time.

## Label separation and validation evidence

Preparation first creates and audits input-only assignments. It reads only
permitted training/registration labels and validation targets. Final test
query BP stays in the original label store, absent from all training inputs.
The personal processes do not load that label store. The scorer verifies both
complete prediction sets and profile hashes, freezes their files, and only
then reads the exact final query labels. Tests deliberately inject labels,
change identities, duplicate windows, corrupt the plan and reuse a forbidden
checkpoint. The old 30/200-person entry points retain their original guards.

## Reporting and limitations

Emit participant-macro MAE and the requested diagnostic table independently
for Overall, MIMIC and VitalDB: Setting, BP, MAE, R2, ME, STD, cumulative
percentages within 5/10/15 mmHg, numerical AAMI/BHS screens. These screens are
not clinical/device certification. Recompute Overall from combined predictions.
Use paired participant bootstrap, 2,000 source-stratified resamples, seed
20260909, pointwise 95% intervals conditional on the fitted model. The primary
contrast is LoRA minus LoRA+memory Overall participant-macro mean SBP/DBP MAE.

Prior experiments on this dataset have informed the method. Fresh weights
establish per-run participant exclusion, not a pristine confirmatory holdout.
Registration windows have ABP-derived labels, not independent cuff events.
MIMIC and VitalDB are internal PulseDB strata, not external datasets. No
clinical, cross-day, motion, wrist-device or contact-pressure claim is tested.

## Execution

One immutable snapshot and one fresh shared fit serve the two candidates.
GPU smoke precedes formal submission. CPU preparation precedes the shared fit;
two balanced personal shards run validation, followed by its frozen-method
receipt; two test shards then run and the final CPU scorer joins them. This
creates multiple scheduler stages, not multiple added model candidates.
Use hpc-2's RTX 5080/5070 Ti for personal shards. CPU stages request no GPU.
Train/write in `work`, archive completed stages and logs to `nas`.
