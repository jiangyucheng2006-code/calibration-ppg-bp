# Personal memory v2: targeted experiments

## Material Passport

- Mode: implementation and bounded server submission, authorized 2026-09-07.
- Question: can train-only personal measurement memory improve an existing
  registered-user LoRA model for reasons beyond nearby-in-time interpolation?
- Source: [v1 results](RESULTS_PERSONAL_MEMORY_V1.md) and
  [publication-oriented plan](PLAN_PERSONAL_MEMORY_PAPER.md).
- Status: implementation; submission and completion require separate receipts.
- No held-out evaluation, new participant recruitment, or clinical claim.

## Fixed contract

The outer protocol remains `development-calbased-analogue-v1`, parent
`meta_train` only. There are 2,051 registered people (1,011 MIMIC and 1,040
VitalDB), each with 320 labelled train windows and 40 internal-validation
windows. Each window is 10 seconds, not an independent cuff measurement.
Persistent personal LoRA is retained. Random-disjoint and chronological-blocked
are separate experiments; the former is not a prospective prediction claim.
All 82,040 internal-validation queries per mode must remain in every main table.
Insufficient legal references cause an unchanged-LoRA fallback, not exclusion.

Frozen source features/checkpoints and train-only BP scalers are reused.
Every loaded cache file must match its manifest hash. Canonical subject,
window, waveform-content, recording, clock and physical-interval checks remain
active. Query BP can select development checkpoints and score predictions,
but cannot select references, update a personal state or enter prediction.
The source encoder was trained on the train labels: none of these features
or train residuals is called out-of-fold (OOF).

## Finite experiment matrix

| Route | Actual intervention | What remains fixed |
|---|---|---|
| E1: matched relation training | Replace the train-query 40-window block exclusion with an explicitly named physical-gap-v2 strategy: exclude self and all overlapping intervals, then select legal top-five references. Chronological mode remains past-only on a comparable clock. | Original relation architecture, zero initialization, optimizer, loss, label budget, validation references, and original train/validation fusion alpha. |
| E2: BP-state retrieval | Train a bias-free 256-to-32 projection from same-person train BP differences. Retrieve top five using the new space. | Original LoRA/features, legal donor rules including the original train block, bank budget, zero relation correction, and each query's original 256D fusion alpha. |
| E3: fixed reliability refinement | Reduce the old alpha using the effective number of contributing references and their standardized prediction dispersion. Evaluate on both zero-relation and frozen trained-relation branches. | Reference selection, LoRA, relation weights, evaluation queries, and all BP-label access. No gate is trained. |

E1 and E2 are two independent trainable candidates, not a combined model.
Across two modes this is **four new model fits**. E3 adds four frozen evaluation
settings across the modes, not four trained networks. A learned E3 error gate
is explicitly deferred until encoder-level cross-fitting can be implemented
and justified. It must not be represented as already submitted.

### E1 details

The original block excluded 40 time-sorted train windows but did not impose
the same exclusion around validation queries. This may change the reference
distance distribution; it is a hypothesis, not an established failure cause.
The first intervention changes only training donor selection. It retains the
old alpha for both train and validation, so it is not simultaneously a new gate.
Old/new train and validation distance/coverage diagnostics are saved.
This v2 strategy is documented rather than silently modifying the v1 cache
or disabling outer split checks. Near/mid/far resampling is not added in the
same fit; a later experiment would be separately specified if necessary.

The shared relation remains 256D pair features, 64 hidden units, antisymmetric
two-output correction and zero final layer. Use normalized absolute Huber
loss (delta 0.5) plus 0.25 pair-delta Huber loss. AdamW learning rate 0.0003,
weight decay 0.0001, gradient clipping 5, batch 256, and 200,000 sampled
train-query exposures per epoch. Exposures are not unique additional windows.

### E2 details

Projection outputs are unit-normalized; cosine distance is trained against
a bounded target derived from same-person BP differences standardized using
the existing train-only SBP/DBP standard deviations. Sampling combines
original-waveform-near pairs with temporally stratified legal train pairs;
BP-dissimilar near pairs receive explicit extra training weight. Pair BP
supervision is restricted to train, while validation retrieval sees no query BP.

Report the original fixed blend, initial projection, best projection, and
memory-only predictions. A 32D projection does not by itself imply reduced
total storage: original 256D features and the source model are still retained.
This is a testable application of supervised metric learning, not a claim
that metric learning or low-dimensional projections are new inventions.

### E3 details

For normalized reference weights `w`, effective count is
`ESS = 1 / sum(w**2)`. For five references, `ESS/5` lies in [0.2, 1].
Let `scatter` be their weighted root-mean-square dispersion around the
weighted predicted BP, standardized by the two train BP standard deviations.
Use the fixed rule `alpha_new = alpha_old * (ESS/5) / (1 + scatter)`.
Fallback remains alpha=0. This formula is frozen before viewing v2 outcomes.

Dispersion may reflect true BP change rather than measurement noise. Therefore
the rule is a hypothesis, not an assumed quality label; score its behavior in
large-BP-change strata and retain failures. No branch-error learner is fitted
to optimistic, in-sample residuals.

## Frozen evidence controls

First reproduce the original full v1 blended predictions on identical keys.
Only after that technical check, evaluate:

- extra temporal exclusions of 0, 60 and 300 seconds;
- random-mode supplementary past-only retrieval, separately labelled;
- original LoRA, feature kNN, epoch-zero fixed blend and trained v1 blend;
- nearest legal recording-time train BP;
- a random-reference comparator matched to the selected feature neighbors'
  train-fitted temporal-distance strata, with common fallback coverage.

Time comparison requires comparable recording clocks. Different recordings
must not be assigned an invented elapsed time. The 300-second and past-only
analyses are new exploratory conditions, not retroactive v1 preregistration.
Matching broad time strata does not establish exact time-distance matching;
report residual distance differences and avoid claiming causal identification.

All outcomes include Overall/MIMIC/VitalDB participant-macro MAE and the
requested pooled diagnostic fields: Setting, BP, MAE, R2, ME, STD,
<=5/10/15 mmHg percentages and qualified AAMI/BHS numerical screens.
Stratify score-only absolute deviation from personal train median BP into
0–10, 10–20 and >20 mmHg; never use these query-target strata for prediction.

## Execution and decision policy

The authorized first batch is finite: two hardware smoke checks, two frozen
diagnostic/E3 jobs, four E1/E2 fits, two mode reports and one joint report.
Only hpc-2 RTX 5080/RTX 5070 Ti are used, subject to live Slurm verification.
No downloading or retraining of the original population backbone is required.
Training/data I/O is in the hot work area, with durable NAS archives.

Technical failure blocks dependent work. The new time diagnostics are
informative, not a retrospectively tuned launch threshold: the four finite
E1/E2 fits are authorized even if the old memory's gap performance is poor,
because they test whether different training/retrieval fixes that limitation.
This execution refinement supersedes the earlier proposal to condition every
neural fit on a favorable old-model gap result. Negative diagnostics still
restrict scientific claims and prevent automatic further architecture sweeps.

Use seed 20260907, patience eight and no epoch cap, with a 72-hour Slurm wall
limit per training job. Save epoch-zero outputs, best epoch, all epoch metrics,
optimizer-step count, runtime, hashes and private participant predictions.
Do not restart a failed scientific run without recording its cause.

Compare each candidate against the same stronger frozen/continued LoRA chosen
by Overall MAE per mode, as well as fixed blend and trained v1 blend. The
historical numerical-upgrade rule stays >=0.15 mmHg Overall mean-MAE gain in
both modes and positive gains in both sources. Beating LoRA alone does not
establish a new module's contribution if the fixed blend is equally good.
No candidate promotion, multi-seed confirmation, unseen-user claim or final
test access is automatic. Those decisions follow this finite batch's report.

## What is deliberately not added now

No new backbone sweep, stacking E1+E2+E3, learned OOF gate, 70/30 exclusion,
new-user calibration, or pressure/motion/device experiment is included.
The highest-priority addition is the temporal/control evidence above, rather
than a fourth architectural direction. Publication suitability depends on
the evidence obtained and cannot be promised before the experiments.
