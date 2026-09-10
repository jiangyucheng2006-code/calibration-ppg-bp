# Full-cohort subject-disjoint personal enrollment

Protocol: `full-cohort-enrollment-v1`. Authorized 10 September 2026.
This corrects the unintended official-pool restriction in the preceding batch.
The old subset jobs1745-1751 were cancelled with authorization; their artifacts
and original data remain intact. This document is a prospective plan, not results.

## Cohort and unchanged outer assignments

Scan all5,361 original participants and all5,245,454 indexed windows. Do not
intersect with official CalBased membership, impose400windows/person, reuse
event120eligibility, sample another200people, or limit the number of subjects.

| Original role | MIMIC | VitalDB | Total |
|---|---:|---:|---:|
| Population training | 1,696 | 2,056 | 3,752 |
| New-person validation | 364 | 441 | 805 |
| New-person final evaluation | 363 | 441 | 804 |
| Total | 2,423 | 2,938 | 5,361 |

These are assignment counts BEFORE necessary validity/provenance/eligibility
checks. Preserve subject_splits.csv and its SHA-256
`8705f7cd75d92201bd203c00fb4d8ad8c738c02d7c4b56e5747210ffba504cd7`.
No subject is reassigned to make a desired score or cohort size.

## Objective eligibility and complete accounting

1. Verify each work-area MAT file against the original index hash. Read only
   PPG_F and its object references, not ABP or query BP values. Reuse the prior
   schema-validity flag, which includes basic finite/reference validity, not an
   error or BP-range threshold. Require finite nonconstant1250-sample PPG,
   consistent125Hz/10-second timing, and complete source/record identities.
2. Quarantine complete identity components whose exact float32 PPG content or
   positive recorded-span overlaps link different outer roles. This does not
   establish that two IDs are the same real person. Original files are unchanged.
3. Within an outer role, retain one deterministic canonical row per identical
   waveform hash, ordered by source-qualified identity and segment ID. Record
   all collapsed duplicate rows. This keeps the strict memory audit and avoids
   counting copied waveforms as additional observations.
4. All valid population-train windows remain usable, even when a person has
   only one. Validation/test people need at least three separable groups for
   inner fitting, inner selection and final query. Use a minimum of one window
   in each, not the old five-window minimum. Short-history estimates will be
   noisy; report actual budgets and participant-macro uncertainty. Never remove
   a person because memory has too few donors: it falls back to LoRA.
5. Save full_cohort_accounting.csv covering every original person,
   window_exclusions.parquet covering every unused row, and personal_budgets.parquet.
   Included plus excluded rows must equal the entire original index count.

Metadata inspection found292of5,361 source files have fewer than three windows;
707have fewer than15. These are full-source counts, not final target exclusions.
They explain why it is impossible to promise that every target person can
simultaneously supply three nonempty subsets. No fixed400-window requirement
or prediction-error-based rejection is used.

## Exactly two paired candidates

- New-person LoRA: shared resnet_small,256-dimensional features, rank4 feature
  LoRA with2,048 new parameters/person, plus a registration-only personal BP anchor.
- New-person LoRA plus reference memory: the same fitted adapter, with the same
  cosine top5, temperature0.1, registration-only leave40-block q95 distance fusion.

Personal artifacts retain subject/account ID, base-checkpoint hash, anchor,
adapter, reference features/BP, registration IDs and distance-gate state.
Reloaded predictions and frozen shared parameters must match.

No additional backbone, baseline, seed sweep or K=1/2/3/5 experiment is added.
Memory similarities are computed in128-query blocks to bound memory; all donors
are still considered and weighting is unchanged. Test blocked versus unblocked
retrieval parity. Low-history q95/donor failure uses unchanged LoRA, not exclusion.

## Fit, select, enroll, score

1. Fit a fresh population model only on original meta_train identities. No old
   all-person or cancelled subset checkpoint is an eligible initialization.
2. For each validation/test person, assign approximately90% registration and10%
   query with deterministic seed20260910. Keep positive-overlap groups together.
   Inside registration, approximately8/9 is inner fit and1/9 inner selection;
   reserve at least one group per role. These are window fractions, not cuff counts.
3. Population epoch selection uses only meta_validation queries and their
   registration anchor with zero personal adapters. It does not borrow a fitted
   training person's adapter. This is a selection criterion, not a third candidate.
4. Each new person's LoRA selects epochs inside that person's registration,
   then initializes afresh and refits all permitted registration data for the
   selected count. Shared encoder/head/scaler/BN/dropout stay frozen.
5. Finish validation for both candidates, retain both as prespecified, and save
   a frozen-method receipt. Only then fit final-cohort personal states. Freeze
   both complete prediction sets before final-query targets enter the scorer.

Population/personal patience8, no epoch cap; seed and previous learning settings
are retained. Scheduler wall-time limits are operational limits, not permission
to report an unfinished run as completed. Any failed prerequisite cancels its
dependent jobs; do not bypass leakage checks or silently substitute subset data.

## Evaluation and claims

Report participant-macro SBP/DBP/mean MAE and paired participant bootstrap
intervals. Recompute Overall/MIMIC/VitalDB separately from the same prediction
sets. Include Setting,BP,MAE,R2,ME,STD,within5/10/15percentages,AAMI and BHS
retrospective numerical screens. The latter are not clinical certification.

This is an exploratory custom full-cohort, person-disjoint enrollment study,
not exact official CalBased/CalFree. Public-data histories informed earlier
method choices, so this is not a pristine independent confirmatory dataset.
Random within-person windows do not establish earlier-history/later-date
prediction, long-term stability, real wrist transfer, or few-cuff calibration.
PPG labels remain ABP-derived; no human experiment is performed here.

## Execution artifacts

Full data materialization and audit are CPU stages in ~/work. Population fitting
uses hpc-2 RTX5080; two independent personal shards use5080and5070Ti after it.
Formal submission follows successful synthetic end-to-end and CUDA smoke tests.
Code snapshot, source/split hashes, plans, logs, progress and successful outputs
are retained. Original NAS masters and previous experiments remain unchanged.
No raw waveforms, personal records, or model tensors are published to GitHub.
