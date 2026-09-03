# Same-subject single-component screen

## Purpose

This development-only screen starts from the current best formulation under
the seen-participant PulseDB analogue:

```text
train-role participant BP mean + compact PPG ResNet residual, Huber loss
```

The discovery split is `random_disjoint`, because it is the current best
same-subject split result. Every candidate uses the same 2,051 participants,
the same 320 train-role windows per participant before an explicitly stated
training-only filter, and all 40 internal-validation windows per participant.
The 40-window held-out role remains sealed.

This is not the participant-disjoint K=1/2/3/5 primary experiment and not an
official PulseDB CalBased reproduction. It is a seen-participant/new-window
development benchmark using many labeled train windows per participant.

## Execution status

Completed on 2026-09-01. Input-only similarity preparation job 1268, all 19
paired training jobs 1269--1287, and common report job 1288 completed with exit
code `0:0`. The selected candidate is `residual_subject_lora_rank4`, with
Overall participant-macro SBP/DBP/mean MAE
`3.9663/2.2043/3.0853` mmHg. See the
[formal result report](RESULTS_SAME_SUBJECT_SINGLE_COMPONENT.md) and public
aggregate files under `results/same_subject_single_component/`.

The source remains the immutable snapshot with archive SHA-256
`85f7c4b9adb5ecd969cd69c6b2c0fc7f7edc2108cd7bbf3de40a0bc49ce6da5b`.
No held-out target was accessed.

## Frozen comparison rule

- Seed: `20260831`.
- Candidate selection: Overall participant-macro mean MAE.
- Early stopping: no epoch cap; stop after eight non-improving epochs.
- Loss: Huber for every candidate; only the explicitly named weighted-loss
  candidate changes per-window loss weights.
- Query coverage: all 82,040 internal-validation windows for every candidate.
- Source reporting: Overall, PulseDB MIMIC, and PulseDB VitalDB reconstructed
  from the same saved predictions.
- Promotion: at least 0.15 mmHg Overall participant-macro mean-MAE improvement
  over the paired reference and positive improvement in both source strata.
- A passing candidate is confirmed on `chronological_blocked` before any
  component combination is considered.

## Candidate matrix

| Candidate | Only changed factor |
|---|---|
| `residual_reference` | Paired rerun of the current compact-ResNet residual reference |
| `residual_quality_gate` | PPG-only soft shrinkage of the residual |
| `residual_quality_weighted_loss` | Train-fitted input-only quality weights in Huber loss |
| `residual_ppg_quality_filter` | Remove only the lowest train-role quality decile |
| `residual_calibration_relative` | Train-support morphology/BP-relative correction |
| `residual_support_attention` | Query-conditioned attention over five train-role supports |
| `residual_support_reliability` | Query-independent support reliability weights |
| `residual_film` | Support-conditioned FiLM modulation |
| `residual_multi_event_weighting` | Adaptive weighting of five train-role BP events |
| `residual_subject_lora_rank4` | Seen-subject rank-4 feature adapter |
| `residual_inception_time_wide` | Replace only the compact ResNet encoder |
| `residual_patch_transformer` | Replace only the encoder with a Patch Transformer |
| `residual_conformer` | Replace only the encoder with a Conformer |
| `residual_cnn_bilstm` | Replace only the encoder with a CNN-BiLSTM |
| `residual_cnn_gru` | Replace only the encoder with a compact CNN-GRU |
| `residual_soft_moe` | Four learned residual experts with a PPG soft gate |
| `residual_prototype_moe` | Four waveform-feature prototypes route residual experts |
| `residual_demographics_direct` | Direct cleaned age/sex concatenation |
| `residual_beat_similarity_filter` | Train on similarity >=0.90, retain every validation query |

The support candidates select six train-role windows at evenly spaced frozen
selection ranks and expose five non-self supports to each training query. No
support is selected from BP magnitude, validation error, or a future role.
Validation queries use five train-role supports only.

The quality proxy is fitted separately within the MIMIC and VitalDB train
roles from raw filtered-PPG amplitude dispersion. It is an input-only proxy,
not a validated universal signal-quality index. The exact beat-similarity
candidate uses the previously documented within-window algorithm and a 0.90
threshold only on training windows; it cannot improve its score by discarding
hard validation queries.

The demographic candidate uses the existing development-only cleaning rule:
adult-age median, explicit age-validity flag, train-fitted age normalization,
and female/male/unknown channels. It cannot be generalized to datasets lacking
equivalent fields without a separate missing-data policy.

## Deliberately deferred combinations and systems

`Quality Gate + Huber + calibration-relative` is not in this matrix because
it changes more than one component. It is eligible only if the isolated
quality-gate and calibration-relative candidates each provide evidence of
benefit.

The previous difficult-30% classifier/specialist and offline waveform-cluster
router are multi-stage systems rather than single modules. Their earlier
participant-disjoint versions had limited discrimination or unstable groups.
This screen includes soft and prototype MoE candidates as isolated routing
mechanisms, but it does not relabel validation participants from their true
errors or claim that the worst 30% can be identified. A new two-stage tail
system would require cross-fitted train-role errors and a separate prospective
protocol.

## Leakage gates

Before accepting the report, require:

1. only the `train` and `internal_validation` roles were readable;
2. train-role BP alone formed participant means and support BP inputs;
3. no candidate altered the internal-validation query keys or targets;
4. the held-out role, participant-disjoint meta-validation, and locked
   meta-test were not accessed;
5. Overall/MIMIC/VitalDB metrics came from the same saved predictions;
6. work and NAS run artifacts were archived and byte-compared.
