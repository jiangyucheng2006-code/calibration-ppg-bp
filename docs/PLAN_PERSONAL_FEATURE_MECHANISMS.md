# Personal feature adaptation: mechanism and improvement study

Date: 2026-09-06. Active scope: registered-user personalization with the same
participants in training and development validation, and disjoint windows.
The previous compact-profile screen failed to beat paired LoRA. This study
tests waveform alignment and personal transformation before advancing a new
method. It does not resume the separate unseen-user K-shot experiment.

## Frozen checkpoint diagnostic

Use the completed LoRA (1399/1408) and no-support 32D profile (1401/1410)
under both random_disjoint and chronological_blocked. Keep the same 82,040
internal-validation targets per mode and all 2,051 people. Read only train
and internal_validation roles; leave held-out sealed.

For each frozen checkpoint:

1. Natural PPG must reproduce saved predictions: maximum absolute prediction
   difference at most 0.05 mmHg and average difference at most 0.005 mmHg.
   These numerical tolerances accommodate device arithmetic, not model error.
2. Two seeded, fixed-point-free PPG permutations within each participant break
   window-level correspondence while retaining participant morphology. Their
   maps depend only on IDs and random numbers; donor PPG can be from any
   validation window, so this is a counterfactual diagnostic, not deployment.
3. Exchange personal parameters between participants of the same source while
   keeping each recipient's BP anchor and PPG. This probes personal mapping.
4. Set LoRA's B matrix, or the profile's personal code/bias, to zero. This is
   removal from a frozen trained network, not the result of training without
   that component. In particular co-adaptation may make zeroing destructive.
5. Compare train-role personal BP mean and median without query PPG.

Encode each natural window once on GPU, then reuse its features and waveform
descriptor for head-only perturbation evaluations. Parameters are restored
after each perturbation; original checkpoints are immutable. Save all
condition predictions privately and Overall/MIMIC/VitalDB aggregate tables.

Interpret jointly: PPG permutation degradation supports reliance on current
window correspondence; personal swapping degradation supports sensitivity to
the correct person's state. Neither proves causal physiology, and their
effects are not additive percentages of model benefit. Follow with trained
shared-adapter/capacity controls to distinguish co-adaptation and capacity.

## Prospective improvement contract

Keep the existing cohort, both split modes, training budget, Huber loss,
patience eight without an epoch cap, complete validation coverage and personal
state artifacts. Use a paired rank-4 LoRA reference and isolate feature
adaptation form from personal parameter count. The finite candidate matrix
and equations will be frozen before new model submission, after the diagnostic.

An accuracy candidate merits confirmation only with at least 0.15 mmHg
Overall participant-macro mean-MAE gain over paired LoRA in both modes and
positive gains in MIMIC and VitalDB in both modes. Multiple candidates are
exploratory; multiple seeds are a later confirmation step. Do not interpret
fewer personal parameters as proven non-inferiority or faster inference.

## Precedents and novelty boundary

The architectural literature lookup is targeted, not exhaustive. LoRA and
FiLM are established controls; shared adaptation directions with a compact
personal coefficient vector are related to vector-based adaptation. Their
use does not itself justify claiming a new mathematical primitive. The
study tests whether personal waveform-feature transformation and its
parameter sharing explain the observed PPG-BP advantage.
