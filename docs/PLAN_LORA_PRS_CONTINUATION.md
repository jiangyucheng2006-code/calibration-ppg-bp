# LoRA with persistent personal residual correction

Frozen plan: 2026-09-06. Screen: `lora-prs-continuation-v1`.
This is a finite, exploratory continuation study, not an already successful
upgrade, an official PulseDB benchmark, or an unseen-user calibration study.

## Why this experiment

The completed personal-profile study did not outperform paired participant
LoRA. Frozen diagnostics showed that LoRA relies on both the correct personal
state and the current PPG window. We therefore retain the fitted LoRA instead
of asking a new compact profile to replace its learned transformation.

The hypothesis is that a compact, persistent output correction can capture
systematic errors left by LoRA, without relearning the whole predictor.
This uses familiar residual/adaptation concepts; combining them alone does
not establish methodological novelty or guarantee better accuracy.

Related combinations were already tested: LoRA + FiLM, attention, support
reliability, calibration-relative correction, and multi-event weighting,
including larger combinations. See
[historical combination matrix](PLAN_SAME_SUBJECT_COMBINATION_SCREEN.md).
Standalone PRS/personal-profile models were also tested. The new comparison
is specifically a checkpoint-initialized LoRA with persistent personal bias
and a compact waveform-dependent residual branch. It does not repeat the
old support-relative branch or copy the unsuccessful PRS reliability gate.
The separately queued [feature mechanism study](PLAN_PERSONAL_FEATURE_MECHANISMS.md)
continues unchanged.

## Five settings, each in two split modes

| Setting | Change relative to continued LoRA | Added personal values | Stored personal values | Personally trainable values |
|---|---|---:|---:|---:|
| `lora_continue` | Same fitted LoRA, continued at a small learning rate | 0 | 2,048 | 2,048 |
| `lora_prs_bias` | Two persistent personal SBP/DBP offsets | 2 | 2,050 | 2,050 |
| `lora_prs_dynamic` | Personal offsets + 32D code + waveform-dependent shared correction head | 34 | 2,082 | 2,082 |
| `lora_prs_dynamic_shrink` | Same dynamic correction, with a penalty discouraging unnecessarily large changes | 34 | 2,082 | 2,082 |
| `lora_prs_dynamic_frozen` | Same dynamic correction, but all original LoRA weights and running statistics frozen | 34 | 2,082 | 34 |

These counts exclude the two non-learned train-role BP anchors per person.
The primary proposed upgrade is `lora_prs_dynamic`. The bias-only candidate
checks whether a simple offset suffices; shrinkage tests over-correction;
the frozen-base candidate checks whether joint modification is necessary.
All five retain the person's original LoRA state; PRS is not substituted for it.

With PPG feature `z`, subject index `s`, and fixed train-BP anchor `a_s`:

```text
z_s = z + B_s A_s z / 4
base prediction = a_s + h(z_s)
new prediction  = base prediction + b_s + g(z_s, c_s, d(PPG))
```

`b_s` is a two-value personal offset; `c_s` is a 32D personal code;
`d(PPG)` is the existing ten-value PPG-only descriptor. The shared head has
input dimension 298, a 128-unit SiLU hidden layer, dropout 0.2 and two outputs.
Its final layer and the personal offset start at zero, preserving the original
LoRA prediction exactly. There is one BP anchor, not two. Prediction/offsets
inside the network are in train-standardized BP units and are converted to
mmHg by the existing fixed scaler.

## Immutable initialization and training contract

- Random source: completed paired LoRA job1399, seed20260904; SHA-256
  `04c965346c8721a0bae8e74c8ec84f670dad8cdd145694dbff32e5a186fa92f2`.
- Chronological source: completed paired LoRA job1408, seed20260904; SHA-256
  `225760ceea3c88ae486b6d5d147978e9c8a16781a9c481eccd9cae831c108f81`.
- Verify source protocol, completed status, checkpoint hash, data-manifest
  hash, exact participant-to-row mapping and train-only target scaler before
  fitting. Load every original LoRA parameter strictly, including personal state.
- New screening seed20260907; fresh optimizer. Existing base learning rate
  `1e-5`; new correction learning rate `3e-4`; AdamW weight decay `1e-4`.
  Huber delta0.5 in standardized units; global gradient clipping5.
- Shrinkage is `0.01 * mean(correction**2)` in standardized units. Other
  candidates have no extra correction penalty.
- Batch64, four data workers, 200,000 replacement-sampled train examples per
  epoch, common seeded sampler. Patience8, no epoch-count cap. Each Slurm
  allocation has a finite72-hour resource limit; this is not an epoch limit.
- Epoch0 is evaluated and saved before training. It remains eligible for
  internal-validation checkpoint selection. If all continuation epochs are
  worse, retain epoch0 rather than overwriting the good starting model.
  This protects the selected development score, not an unknown test score.
- The frozen-base setting freezes BatchNorm statistics and dropout state as
  well as gradients. Original source files are never overwritten.
- Initial predictions are recomputed on the assigned GPU. Previous strict
  reproduction checks detected cross-GPU arithmetic differences; we do not
  silently compare a new GPU's predictions with a cached source result as
  though they were bit-identical. The five candidates must share identical
  starting predictions on that split's assigned GPU.

## Data and leakage boundary

Unchanged `development-calbased-analogue-v1`: 2,051 people (1,011 MIMIC,
1,040 VitalDB), 320 labelled train windows and40 internal-validation windows
per person. The40 held-out windows remain sealed. All people derive from the
original meta-train parent. Each window is10s; this is not a claim of320
independent cuff measurements or five-shot new-user calibration.

Both `random_disjoint` and `chronological_blocked` are run independently.
Participant overlap is intentional; role overlap in windows, physiological
intervals and exact waveform content remains prohibited. Random splitting
is known-user interpolation, not guaranteed future prediction.

Only train labels fit population weights, personal adapters, codes, offsets,
BP anchors and scalers. Internal-validation labels only select checkpoints
and exploratory candidates; they never enter inference inputs or adaptation.
No tail removal, extra support labels, or validation-informed personal fitting
is introduced. Training the correction on the LoRA training subset is ordinary
continuation, not an independent out-of-fold residual diagnostic.

## Reporting and decision

Emit full-coverage Overall/MIMIC/VitalDB participant-macro SBP/DBP/mean MAE
and the requested event-pooled tables: Setting, BP, MAE, R², ME, STD,
≤5/10/15mmHg percentages, qualified AAMI/BHS numerical screens. The latter
do not establish clinical/device standards compliance.

Compare every candidate with **both** the recomputed frozen starting model
and `lora_continue`. A potential upgrade requires at least0.15mmHg Overall
mean-MAE gain in both split modes, plus positive MIMIC and VitalDB gains in
both modes, against both comparators. A numerical leader failing this gate
is reported but not automatically promoted. Multiple seeds and final held-out
confirmation remain later steps; repeated development screening can overfit
the internal-validation set.

## Execution

Tests cover nontrivial zero-correction equivalence, forward/backward and
checkpoint state, parameter accounting, personal-row isolation, frozen
parameters/statistics, source provenance/scaler/mapping rejection, epoch0
fallback, and the paired promotion gate.

Submit two GPU smoke jobs, ten training jobs and three dependent reports.
Each training job starts only after its GPU smoke succeeds; each split report
waits for all five training jobs; final comparison waits for both split reports.
Short smoke checks use real train-role PPG and verified original checkpoints.
Previous queued/running jobs are not canceled or changed. New jobs use hpc-2:
RTX5080 for random, RTX5070Ti for chronological. All active I/O stays in work;
completed outputs/checkpoints and submission manifests are archived to NAS.

Entry points: `lora_prs_train`, `lora_prs_smoke`, `lora_prs_report` and
`scripts/submit_lora_prs.sh`. The exact snapshot and submitted IDs belong in
[STATUS](STATUS.md). Public artifacts contain code, plans and aggregate results,
not raw signals, personal identifiers, personal weights or prediction tables.
