# Personal memory: candidate retention and bounded confirmation plan

## Plan status

- Date: 2026-09-08.
- Purpose: retain a useful registered-user method, identify the source of its
  benefit, and prepare a defensible benchmark evaluation.
- Status: proposed plan; no training, held-out evaluation or GitHub publication
  is authorized or performed by this document.
- Evidence: completed v2 aggregate reports at repository commit
  `8c8b610be90ffa674aea4e5366b041860f1d8030`; official PulseDB paper and repository
  checked on this date. Proposed analyses have not been executed.
- Existing execution code and environment remain unchanged. Exact commands,
  resource bounds and an immutable execution manifest must precede submission.

## Clarification after the follow-up discussion (2026-09-08)

Fixed memory and E1 have already been evaluated in both modes; E1 was actually
trained. Stage 1 below means reuse these completed comparisons and add only
missing analyses, not repeat the same fits as a new discovery experiment.
Fixed memory has no additional trainable relation/gate; its encoder/LoRA was
already trained. E1 chronological improves fixed memory from 3.7967 to 3.6334
(0.1633 mmHg), while its increment over the trained v1 blend is only 0.0237.

Confirmation does not forbid targeted method development. One optional first
candidate is a learned trust fusion between the current registered-user LoRA
prediction and explicit retrieved-memory prediction. Unlike the present fixed
distance rule or fixed E3 ESS/scatter formula, it would learn from PPG/history
and branch disagreement when memory is useful. Its training examples must use
encoder-level cross-fitted predictions entirely inside permitted training
roles; validation/test errors cannot train the gate. Preserve all queries and
the LoRA fallback. Compare against the existing fixed rule on identical inputs.
This is proposed, not implemented or shown to improve results.

Gating, OOF residual learning and joint fine-tuning already appear in earlier
project phases. Their generic ideas are not new. The narrow untested question
is their benefit for this specific explicit-history/LoRA two-branch setup.
Literature novelty remains unverified. Jointly updating personal adaptation and
the retrieval relation is only a later alternative: earlier LoRA+PRS joint
continuation did not establish a benefit, and the current memory pipeline uses
frozen encoder caches. Do not launch both as an unrestricted sweep.

Official membership preparation/audit may proceed independently of this method
decision once execution is authorized; there is no scientific need to finish
all optional upgrades before preparing the benchmark. Full official training
must still follow the historical-exposure and protected-cohort checks in Stage 3.

## Decision: keep the method without rewriting the historical gate

Keep personal memory as the main development direction, with E1 as the leading
learned candidate. Keep participant LoRA as the paired accuracy reference.
Retaining a candidate is not the same as formally replacing that reference.
The completed v2 screen remains `promotion=false` under its predeclared
requirement of at least 0.15 mmHg improvement in both split modes. That threshold
is an internal screening decision, not a PulseDB requirement or a clinical
minimum important difference. This plan does not lower it retrospectively.

Primary numbers below are participant-macro mean MAE, averaging SBP and DBP,
in mmHg. Modes are separate outcomes and must not be averaged together.

| Method | Random disjoint | Chronological blocked | Retention decision |
|---|---:|---:|---|
| Stronger paired LoRA | 2.8764 | 3.7124 | Required reference |
| Fixed memory blend | 2.6475 | 3.7967 | Required simple control |
| E1 matched relation | 2.6475 | 3.6334 | Leading learned candidate |
| E3 reliability on selected v1 relation | 2.6381 | 3.6448 | Optional refinement |

E1 improves the paired LoRA by 0.2289 / 0.0790 mmHg (approximately 8.0% / 2.1%).
MIMIC and VitalDB each improve in both modes. These are genuine numerical
observations, not yet statistically confirmed or clinically meaningful effects.

E1 random selects epoch zero: its result equals the fixed blend, so the random
benefit cannot be attributed to new E1 relation learning. Its chronological
improvement over the previously selected v1 blend is only 0.0237 mmHg. E3 adds
only 0.0094 / 0.0124 over v1 and worsens the exploratory large personal-DBP-
deviation group. Stop expanding E2. Do not mix the best method from each split
into an unregistered composite result.

## Official PulseDB split: verified definition

The original paper, Section 2.6 and Table 4, specifies random within-participant
sampling for CalBased, not an earlier-training/later-testing requirement.
After removing the separate AAMI cohort, 400 segments are randomly selected per
eligible person. Ten percent of people form CalFree; among the other people,
10% of their segments form CalBased testing and the rest form training.
Thus official Group A contains 2,506 shared participants, with 360 training and
40 CalBased test segments per person. These are 10-second segments, not cuffs.

Official identities and segment membership come from `Train_Info.mat` and
`CalBased_Test_Info.mat`. `Generate_Subsets.m` loads those assignments; arbitrary
random re-splitting is not exact official reproduction. CalFree uses disjoint
people; AAMI calibration/testing are separate subsets, not alternate names for
the ordinary CalBased split. The official source does not require this project's
0.15 mmHg two-mode improvement gate.

- [Original paper, Section 2.6 and Table 4](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full)
- [Official Info-file guide](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/Info_Files/File_Preparation_Guide.md)
- [Official subset generation](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/Generate_Subsets.m)

Provenance: Europe PMC `TITLE:PulseDB`, resultType=core, returned one result,
PMCID PMC9944565 / DOI 10.3389/fdgth.2022.1090854. Section 2.6 and Table 4 were
also read through its full-text XML API. GitHub main resolved to the pinned
commit above. No official patient-level Info file was downloaded in this task.

Our 2,051-person, 320/40/40 development protocol is a custom analogue, not the
official 2,506-person 360/40 benchmark. Chronology remains an informative
additional control. Random CalBased evaluation can support registered-user
new-window interpolation; it does not establish future-only BP tracking.

## Stage 1: complete mechanism analysis without new backbone fitting

Use the existing permitted training caches and full internal-validation cohort.
Do not open either the 40-window sealed role or the locked unseen-user cohort.

1. Compare the paired LoRA, fixed blend and E1 on identical query keys. Retain
   both frozen and continued LoRA controls and disclose the predeclared stronger
   paired-reference rule. Never choose the reference using held-out results.
2. Report participant-paired MAE differences, improvement fractions and
   participant-bootstrap intervals, separately for Overall/MIMIC/VitalDB and
   each BP/mode. Current validation has been searched repeatedly; its intervals
   describe cohort variability, not independent confirmation.
3. Evaluate E1 itself at reference exclusion gaps 0/60/300 s. Retain every query
   with the frozen LoRA fallback; report fallback rates and actual reference-time
   distances. Existing v1 gap findings do not substitute for this E1 analysis.
4. Compare against personal training-BP mean, waveform-selected memory without
   LoRA, and nearest legal recording-time BP where comparable clocks exist.
   The last is a timestamp-assisted diagnostic, not a PPG-only deployment model.
   Apply identical permitted-reference rules and report clock availability.
5. Quantify residual time-distance mismatch in feature-versus-random controls.
   A broad time bin is not exact temporal matching. Do not infer a causal
   waveform effect from unmatched distances or use query BP to choose references.
6. Retain the BP-deviation subgroup as score-only analysis; do not remove difficult
   people/windows from main metrics or treat target-defined groups as deployable
   quality labels. A past-only retrieval constraint on a random-trained encoder
   must not be described as prospective validation.

Important existing warning: nearest legal-time BP reaches 2.2851 in the random
v1 diagnostic, versus 2.6475 for the v1 memory blend. This motivates the simple
control; it neither proves PPG useless nor proves the official split invalid.

## Stage 2: a small paired confirmation matrix

Core arms: LoRA; LoRA plus fixed memory; LoRA plus E1. E3 is an optional frozen
secondary analysis, not another unrestricted module sweep. Use the same three
arms in both modes; train and select only within the allowed development roles.

Target three independently seeded base-LoRA pipelines per mode, with shared
source checkpoints across arms within each seed. Reuse an existing source seed
only after its complete training lineage and matching controls are verified.
This means at most six base fits and six E1 fits across both modes, fewer when
qualified existing fits can be reused. Fixed blending needs inference, not a
new backbone. Changing only E1 seeds on one cache tests head stability, not
whole-method stability. Save all seeds and failed runs, not only the best seed.

Report seed-specific and averaged paired differences, source-specific effects,
training/inference cost and per-user memory size. Quantify participant and seed
uncertainty separately; repeated seed-window rows are not independent people.
Use fixed early stopping and inner-validation selection rules. Do not tune
memory size, time exclusions or alpha against sealed outcomes.

## Stage 3: freeze one method and choose a final evaluation route

Before any new final evaluation, freeze preprocessing, full label budget,
reference selection, checkpoint rule, model variants and reporting code.
Proposed future primary comparison: the frozen memory method versus paired
LoRA on random new-window prediction. Chronology remains a named, fully reported
secondary robustness endpoint, not a hidden condition removed after failure.
On untouched evaluation, report paired effect and its 95% interval; an interval
entirely on the improvement side supports numerical superiority in that setting,
not clinical significance. Do not infer chronological non-inferiority from a
non-significant result. A non-inferiority margin requires a separate justification.
This proposed confirmation scope does not change the completed v2 gate.

Two possible routes require an explicit evaluation decision before execution:

- Current analogue: after method selection, refit on the permitted 320+40
  development windows and evaluate the reserved 40 once, including a matched
  LoRA comparator. Still label this a custom benchmark, not official CalBased.
- Official CalBased: first audit exact Info-file membership and hashes against
  every historical training/validation/locked role. The published 360/40 split
  is not automatically untouched in our project. If historical training used
  official test windows, reusing those checkpoints is invalid; fresh fitting is
  necessary. If model development has used those outcomes, fresh fitting alone
  cannot erase that development exposure. Disclose benchmark reuse and retain
  a separate untouched confirmation. Any conflict with protected unseen users
  requires a separate protocol decision; never silently unlock or relabel them.

Official training must include an inner-validation plan before the official
test is opened. Reproduce exact membership and use only allowed PPG and declared
personal history; no query ABP or query BP may enter model inputs or retrieval.
An audited restricted subset must be named as such, not the full official set.

## Deliverables and stopping decisions

Each stage must produce a dated execution receipt, data/code hashes and tables
for both modes, each separately showing Overall, MIMIC and VitalDB. Required
columns remain Setting, BP, MAE, R2, ME, STD, <=5/10/15 mmHg, and qualified
AAMI/BHS numerical screens. These are not clinical certification. Keep private
waveforms, participant tables and checkpoints off the public repository.

Retain the simpler fixed-memory variant if E1 adds no repeatable value under the
chosen scope. Retain E1 if its learned contribution is supported, especially in
chronological prediction. Do not require every secondary subgroup point estimate
to improve, but disclose harms, uncertainty and any trade-offs. If only nearby
random-window gains survive, keep that limited result and frame the contribution
as history-assisted interpolation rather than continuous physiological tracking.

The next useful action is Stage 1, not another large architecture search. New
training and final evaluation wait for execution authorization and their frozen
manifests. Runtime estimates require a resource check; none are promised here.
