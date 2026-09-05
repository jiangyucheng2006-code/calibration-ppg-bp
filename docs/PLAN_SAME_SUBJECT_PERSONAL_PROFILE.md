# Compact personal-profile residual experiment

## Status and scope

Completed; see the [full result](RESULTS_SAME_SUBJECT_PERSONAL_PROFILES.md).
All 16 runs succeeded, but the prespecified primary did not outperform paired
LoRA. The original hypothesis and decision rule below are retained unchanged.

This is a prespecified development experiment for persistent, seen-participant
personalization. It is **not** an unseen-participant few-shot claim and it is
**not** an official PulseDB CalBased reproduction.

- Source parent split: `meta_train` only.
- Participants: the same retained participants occur in train and internal
  validation.
- Samples: exact 10-second windows, intervals, and waveform content remain
  disjoint between roles.
- Development split modes: `random_disjoint` and `chronological_blocked`, always
  reported separately.
- Read roles: train and internal validation only.
- Held-out role: sealed for this screen.
- Main metric: participant-macro MAE, with Overall, MIMIC, and VitalDB reported
  separately.

## Hypothesis

Rank-4 participant LoRA stores 2,048 trainable values per person at the
256-dimensional feature layer. The proposed model tests whether persistent
personalization can instead be expressed more compactly and interpretably as:

1. a train-role participant BP mean anchor;
2. a shared PPG-to-relative-BP component;
3. a stable participant bias;
4. a query-dependent change relative to that participant's train-role PPG/BP
   support profile; and
5. a reliability gate that shrinks the dynamic correction when the current PPG
   is unlike the personal support profile.

The primary model stores a 32-dimensional participant state and a two-value
participant bias: 34 trainable values per participant, versus 2,048 for the
rank-4 LoRA reference. Five deterministic non-self train-role supports provide
the historical PPG/BP context for every query.

## Frozen candidate matrix

| Candidate | Purpose |
|---|---|
| `residual_reference` | Shared compact ResNet plus participant BP mean |
| `subject_lora_rank4` | Paired LoRA numerical reference |
| `personal_profile_support_only` | Remove free participant parameters |
| `personal_profile_code32_no_support` | Remove historical PPG/BP supports |
| `personal_profile_code32_no_reliability` | Remove reliability shrinkage |
| `personal_profile_code32_reliability` | Prespecified primary model |
| `personal_profile_code64_reliability` | Personal-state capacity control |
| `personal_profile_code32_stable_only` | Remove dynamic personal correction |

All candidates use the same preprocessing, waveform encoder family, seed,
optimizer, early-stopping rule, windows, and reporting code. The LoRA and
ordinary residual references are retrained in the same screen rather than
copied from an older run.

## Personal-state artifacts

Every run saves:

- `subject_index.json`, the exact participant-to-row mapping;
- `participant_profile_index.parquet`, the train-role BP anchor and exact
  support-event identifiers for each participant;
- the same mapping inside `best.pt`;
- the participant-indexed state and bias tables inside the checkpoint for
  candidates that use them;
- the train-only BP anchor and support-selection contract in `run.json`; and
- internal-validation predictions and Overall/MIMIC/VitalDB metrics.

This prevents a participant-indexed tensor from becoming detached from the
identity mapping needed to use it.

## Prespecified decision rule

The primary model is promoted only if it improves participant-macro mean MAE
over the paired LoRA reference by at least 0.15 mmHg Overall in both split
modes, while also improving both MIMIC and VitalDB in both modes. Other profile
variants are mechanism or capacity controls; a lower number from an ablation is
reported as a numerical observation rather than silently replacing the primary
hypothesis.

No held-out target may be opened because of a development-screen result.
