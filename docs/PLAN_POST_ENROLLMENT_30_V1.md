# Post-training enrollment of 30 entirely excluded people

## Material Passport

- Origin: project experiment planning and implementation, 2026-09-09.
- Version: post-enrollment-30-v1.
- Verification status: implementation in progress; no new accuracy result yet.

## Question and deployment direction

The user can supply paired personal cuff blood pressure and PPG recordings.
The intended system is a shared population model, a newly fitted personal LoRA,
and a persistent personal reference-memory bank. A new account must be able to
create its own profile without selecting another person's adapter or retraining
the population network. This public-data experiment tests that onboarding step.
It does not collect human data or authorize clinical use.

## Fixed split and label budgets

- Source: the verified official CalBased 2,506-person store, 1,213 MIMIC and
  1,293 VitalDB people. Preserve its exact within-person 360 TRAIN / 40 TEST rows.
- Using sorted subject identities, NumPy default_rng seed 20260909, sample 15
  people from each source, without reading BP, errors, age or signal quality.
- Remove these 30 people's complete membership from population fitting and
  population validation/test. This is a logical split, not raw-file deletion.
- Remaining 2,476 people: 792,320 inner-fitting windows; 99,040 inner-validation
  windows. Select epochs with patience 8, then start fresh and refit 891,360
  labelled windows for the selected number of epochs. Subsequently score the
  remaining 99,040 original test windows without feeding the result into training.
- New 30: 10,800 registration windows, 1,200 test windows. For each person,
  320 registration windows fit an adapter; 40 registration windows select its
  epoch (patience 8; epoch 0 is eligible). Initialize that adapter afresh and
  refit all 360 registration windows for the selected number of epochs.
- The 30 new subjects are not used for population model selection, target
  normalization, BP anchors, batch-normalization statistics, encoder training,
  memory fitting, or pretraining. No old all-subject checkpoint may initialize it.

This is a new subject-excluded protocol derived from official membership, **not
an exact official benchmark reproduction**. Personal identities intentionally
overlap personal registration and personal test only after population fitting.
These are ABP-labelled 10-second windows, **not 360 independent cuff events**.
MIMIC/VitalDB are internal PulseDB source strata, not independent external tests.

## Model, controls and order

1. Train the paired architecture from scratch on the remaining people:
   ResNet-small, 256 features, shared regression head, per-person rank-4 LoRA,
   train-only BP anchor, standardized-target Huber loss. Use the existing base
   learning rate 3e-4, AdamW weight decay 1e-4, Huber delta .5, batch 64,
   gradient clip 5, seed 20260909 and patience 8; no fixed epoch-count cap.
2. Validate internally, refit the remaining cohort, and freeze a benchmark
   prediction file before joining only the corresponding reference targets.
3. For each new person, copy shared encoder/head only. Freeze all shared
   parameters and batch-normalization state. Initialize new A ~ N(0,.01), B=0;
   fit only the 2,048 personal LoRA parameters, using the same optimizer/loss.
   Individual epoch selection uses their allowed registration validation only.
4. Compare on the exact same 40 per-person test windows:
   personal registration BP mean; shared model with that BP anchor and zero
   adapter; new personal LoRA; new personal LoRA plus fixed personal memory.
5. Fixed memory preserves cosine top-5 retrieval, temperature .1 and the
   distance-based fusion formula from the successful fixed-memory method.
   The q95 distance scale is now fitted independently within each new person's
   registration bank using leave-40-window-block retrieval, not pooled across
   future users. All donor BP belongs to registration; query BP is unavailable.
   This per-person gate fitting is an explicit onboarding implementation change.
6. Persist the shared model version/hash, personal adapter, BP anchor,
   normalization parameters, registration IDs, memory vectors/reference BP,
   gate scale and complete profile hashes. Verify saved/reloaded predictions.
7. Only after all 30 profiles and their four-method predictions are frozen does
   a separate scoring stage read their exact test targets. Never use the
   remaining cohort benchmark or new-person test to revise this finite batch.

## Required checks and reporting

- Assert whole-subject, waveform-content and physiological-interval separation
  between population and enrollment cohorts, and exact role membership.
- Preserve the already authorized, source-verified official within-person
  duplicate-content exceptions; count/disclose their effect in each new cohort.
  Reject newly introduced copies or cross-cohort content/interval overlap.
- No test-target file is opened by a trainer or personal-memory constructor.
  No old person's adapter is copied into the new account. Freeze shared state
  during personal training and test serialization equivalence.
- Provide Overall, MIMIC, VitalDB participant-macro MAE and paired comparisons.
  Separately provide Setting, BP, MAE, R², ME, STD, ≤5/10/15 mmHg percentages,
  qualified AAMI/BHS numerical diagnostics from identical predictions.
- Report all 30, including people whose error remains high or whose selected
  adapter is epoch 0; no replacement of randomly sampled difficult participants.
- Record code/data hashes, seed, counts, job IDs, timings, checkpoints and logs.
  Raw signals, individual IDs/profiles and predictions stay private on server.

## Claim limits and next gate

This dataset has already informed earlier model development. Fresh subject
exclusion prevents direct fitting exposure in this run but does not turn it into
an untouched confirmatory cohort. Thirty people (15 per source) are an
exploratory test with limited precision, not a clinical-validation sample.
Randomized historical windows are not chronological future predictions and
cannot establish long-term stability, wrist/cuff transfer or clinical AAMI/BHS
certification. The next evidence, if onboarding works, is a separately frozen
earlier-registration/later-session evaluation, then ethically approved paired
real-device data. Do not automatically start either extension from this batch.
