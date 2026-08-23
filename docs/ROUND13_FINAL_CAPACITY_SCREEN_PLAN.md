# Round 13: final architecture and capacity screen

## Decision question

This is a final, development-only screen of whether the current calibrated
PPG-BP model is limited by the population encoder architecture or its capacity.
It is not a new calibration protocol. Every candidate retains the same K=5
fixed-first QGH calibration head, loss, optimizer settings, target scaling,
query set, and leakage-safe fold boundary.

## What has already been tested

Round 11 already showed that a larger or attention-based network is not
automatically better. The primary values below are participant-macro mean MAE
on meta-train fold 4; lower is better.

| Backbone | Population parameters | Overall mean MAE (mmHg) | Change vs small ResNet |
|---|---:|---:|---:|
| `resnet_small` | 665,490 | 8.6453 | 0.0000 |
| `inception_time` | 512,162 | 8.7227 | +0.0774 |
| `resnet_deep` | 3,827,002 | 8.9036 | +0.2583 |
| `patch_transformer` | 710,530 | 8.9132 | +0.2679 |
| `conformer` | 1,587,330 | 8.9789 | +0.3336 |

Round 12 additionally found that a 16.13-million-parameter residual-attention
encoder did not improve the result. These observations motivate a controlled
scale curve rather than assuming that more parameters must help.

## Candidate matrix

The repeated base models make all within-family comparisons use the same seed
and current code. Parameter counts are measured by the CUDA smoke job before
formal submission.

| Candidate | Planned population parameters | Controlled question |
|---|---:|---|
| `resnet_small` | 665,490 | Exact within-round reference |
| `resnet_depth2` | 1,333,010 | Same channels/resolution, two residual blocks per stage |
| `resnet_wide1p5` | 1,456,282 | Same four blocks/resolution, approximately 1.5-times wider channels |
| `inception_time` | 512,162 | Base multiscale convolution control |
| `inception_time_wide` | 1,123,954 | Whether more multiscale convolution channels help |
| `patch_transformer` | 710,530 | Base Transformer repeat |
| `patch_transformer_deep` | 1,372,034 | Same tokenization and width, 4 to 8 layers |
| `patch_transformer_wide` | 2,730,498 | Same tokenization and 4 layers, 128 to 256 model width |
| `patch_transformer_highres` | 715,522 | Same base capacity, more short overlapping PPG tokens |
| `patch_transformer_longpatch` | 716,034 | Same base capacity, near-beat-length patches |
| `conformer` | 1,587,330 | Base local-convolution plus attention repeat |
| `conformer_large` | 5,230,018 | Larger Conformer control |
| `convnext_1d` | 5,477,826 | Modern scalable convolutional alternative without attention tokens |

The ResNet depth and width candidates keep the original downsampling schedule.
The Transformer depth, width, high-resolution, and long-patch candidates change
one main factor relative to the repeated base Transformer. This makes the
result more interpretable than a single arbitrary large network.
Every encoder still outputs 256 features, so the QGH calibration component
adds the same 461,828 trainable parameters to every population model. The CUDA
smoke test must reproduce the planned counts exactly before formal submission.

## Fixed experimental boundary

- fit: meta-train folds 0, 1, and 2;
- early stopping: meta-train fold 3, patience 8, no epoch cap;
- internal ranking: meta-train fold 4;
- K=5 chronological fixed-first support events;
- Huber loss and the established quality gate;
- feature output dimension fixed at 256 for every encoder;
- a common physical microbatch of 32 for population training and 16 for QGH,
  with four-step gradient accumulation to preserve effective batches of 128
  and 64 for every candidate;
- 99,968 sampled examples per epoch, which is exactly divisible by both
  effective batch sizes and avoids a candidate-independent short final step;
- one seed (`20260827`) for screening;
- meta-validation and the locked meta-test are not accessed;
- Overall, PulseDB MIMIC, and PulseDB VitalDB are computed from the same saved
  fold-4 predictions, with participant-macro MAE as the primary metric.

## Execution and safety gates

1. Run the complete unit-test suite on the deployed source.
2. Run one GPU forward/backward smoke test for all 13 candidates and record
   exact parameter counts and peak allocated memory.
3. Submit population training, then dependent QGH training, then dependent
   fold-4 evaluation for every candidate.
4. Generate one report only after every evaluation succeeds.
5. Use 12 GiB host memory per GPU job so the RTX 5080 and RTX 5070 Ti can run
   concurrently. If the common 32/16 CUDA microbatches do not fit a candidate,
   stop and revise the common setting rather than silently confounding one
   model with a different physical batch.

The internal promotion gate remains an Overall participant-macro mean-MAE
improvement of at least 0.15 mmHg over the same-round `resnet_small`, with
improvement in both MIMIC and VitalDB. If no candidate passes, architecture
scaling is closed after this round; no multi-seed confirmation and no locked
test access are justified.
