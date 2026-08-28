# PulseDB same-subject development analogue

## Purpose

This is a separate, subject-dependent benchmark for testing how much of the
reported PulseDB calibration-based accuracy comes from seeing labelled windows
from the same participant during model fitting.

It does not replace the primary `event120-v1` experiment. The primary study
continues to ask whether a model can adapt to an unseen participant from only
`K = 1, 2, 3, or 5` temporally separated reference-BP events. The new track
asks a different and easier question: how accurately can a model predict new
10-second windows for participants who already contributed many labelled
10-second windows to training?

## Relation to the official PulseDB protocol

The official PulseDB calibration-based benchmark assigns the same 2,506
participants to training and calibration-based testing. For each participant,
360 selected 10-second windows are used for training and 40 different windows
are used for testing. The official windows are disjoint, but the split is not
participant-disjoint and does not require session or temporal-block separation.

The official roles are encoded in `Train_Info.mat` and
`CalBased_Test_Info.mat`. Those files are not currently present on the server,
and exact use of the full official Group A could expose participants reserved
by this project as a locked unseen-participant test. Therefore the first
experiment is deliberately named a **development analogue**, not an official
PulseDB CalBased reproduction.

Primary sources:

- [PulseDB dataset paper](https://doi.org/10.3389/fdgth.2022.1090854)
- [PulseDB official repository](https://github.com/pulselabteam/PulseDB)
- [Official Info-file preparation guide](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/Info_Files/File_Preparation_Guide.md)

Architecture-family sources motivating the PPG-only adaptations:

- Jamil et al., [Self-Attention ResUNet on PulseDB](https://doi.org/10.1109/JSEN.2024.3468316);
- Chen et al., [rU-Net, multi-scale fusion, and transfer learning](https://doi.org/10.1109/JBHI.2024.3483301);
- Shaikh and Forouzanfar, [Dual-Stream CNN-LSTM on PulseDB](https://doi.org/10.1109/JSEN.2024.3512197);
- Tang et al., [Iterative demographic attentional fusion CNN-Transformer](https://doi.org/10.1109/APSIPAASC63619.2025.10849156).

These publications use multimodal inputs and/or demographic variables in their
reported methods. This track borrows only broad encoder ideas and therefore
labels every corresponding implementation as a PPG-only adaptation.

## Frozen development boundary

Protocol identifier: `development-calbased-analogue-v1`.

- Eligible participants come only from the existing frozen `meta_train`
  partition.
- `meta_validation` and the locked `meta_test` are rejected by the builder.
- A participant must have at least 400 schema-valid, finite, nonconstant,
  1,250-sample `PPG_F` windows with finite SBP and DBP.
- Exactly 400 windows per eligible participant are assigned to roles:
  - 320 `train` windows;
  - 40 `internal_validation` windows;
  - 40 `heldout_test` windows.
- The same participant set appears in all three roles.
- A `segment_uid` can appear in only one role.
- The initial model screen may load only `train` and
  `internal_validation`. It must not read held-out targets.
- After one candidate and its training rule are selected, the candidate may be
  refitted on `train + internal_validation` (360 windows per participant) and
  evaluated once on `heldout_test`.
- The first screen is single-seed and development-only. Repeated-seed work is
  justified only after a useful validation gain.

The expected eligible cohort from the current full index is 2,058 participants
(1,018 MIMIC and 1,040 VitalDB). These counts remain prospective until the
versioned builder and its audit finish successfully on the server.

## Two split modes

### Random disjoint windows

`random_disjoint` is the literature-aligned main analysis. A deterministic
participant-specific hash selects and assigns 400 windows without replacement.
It mimics the key same-subject/random-window property of the official
calibration-based benchmark while preserving an independent internal
validation role.

This mode may place adjacent physiological states in different roles. The
audit must therefore report record overlap, adjacent-window cross-role pairs,
minimum train-to-validation/test time separation, and exact duplicate keys.

### Chronological blocked control

`chronological_blocked` is a stricter control. Training windows precede
validation and held-out windows according to record order and within-record
time. It tests whether a gain survives when later windows, rather than randomly
interleaved windows, are predicted for the same participant.

The random and chronological results must never be averaged. Their difference
is itself a protocol-sensitivity result.

## PPG-only candidate screen

All neural candidates receive only PPG at query time. Participant identity and
statistics derived from that participant's labelled training windows are
allowed only for explicitly named seen-subject methods.

The first matrix contains:

1. `subject_train_mean`: predict every validation window from that
   participant's training-label mean;
2. direct compact-ResNet PPG regression;
3. direct PPG regression with the existing InceptionTime-wide and Patch
   Transformer encoders;
4. direct PPG-only adaptations of literature architecture classes;
5. `subject_mean_residual_ppg`: participant training-label mean plus a PPG
   network that predicts within-participant SBP/DBP change;
6. deferred adaptations of the current QGH and calibration-relative methods, reported
   explicitly as seen-subject plus support variants rather than ordinary
   K-shot results.

The Self-Attention ResUNet, residual U-Net, CNN-BiLSTM/CRNN, and
CNN-Transformer families in this matrix are **PPG-only adaptations**. They are
not exact reproductions of papers that used ECG, demographic variables, or
different validation procedures.

## Evaluation and claim boundary

For every accepted prediction table, report:

- Overall, PulseDB MIMIC, and PulseDB VitalDB from the same predictions;
- participant-macro SBP, DBP, and mean MAE as primary;
- window-pooled MAE, R2, signed ME, error STD, and within 5/10/15 mmHg as
  secondary diagnostics;
- retrospective AAMI-style and historical BHS numerical screens with no
  device-validation claim;
- participant count, window count, and split/audit hashes.

A low error on this track supports only seen-participant, new-window
prediction. It does not demonstrate unseen-participant generalization, true
few-shot cuff calibration, robustness to pressure or motion, cross-device
transfer, or clinical validation.

The scientifically useful comparison is the protocol gap:

```text
same participant + hundreds of labelled windows
versus
unseen participant + K=1/2/3/5 independent reference events
```

If the former is much more accurate, the difference must be attributed first
to task information and split design, not automatically to a superior neural
architecture.
