# Round-12 literature-derived backbone screen

## Objective

Test whether neural-network families reported in calibrated or personalized
PPG blood-pressure studies improve the current compact ResNet when evaluated
under this project's stricter, leakage-safe K=5 protocol.

## Fixed factors

- input: PPG only;
- support: each participant's first five eligible labeled events;
- query: all later eligible events from the same participant;
- development boundary: `meta_train` only;
- fitting participants: internal folds 0--2;
- early stopping: fold 3, patience eight, no epoch-count cap;
- candidate ranking: fold 4;
- one seed (`20260826`) for screening;
- population model followed by frozen-population QGH calibration;
- Huber loss and PPG-only quality gate for QGH;
- identical 256-dimensional encoder output and identical prediction head;
- no meta-validation and no locked meta-test access.

## Candidates

| Setting | Literature rationale | What is actually changed |
|---|---|---|
| `resnet_small` | Current reference | nothing |
| `tcn_bp` | subject-disjoint PulseDB calibration paper | PPG encoder only |
| `fewshot_resnet_attention` | PulseDB five-shot paper | PPG encoder only; no SSL yet |
| `bp_crnn` | personalized transfer/AAMI-BHS literature | PPG encoder only |
| `resunet_encoder` | PulseDB residual U-Net/AAMI-BHS paper | PPG encoder only; no ECG/demographics/ABP loss |

## Advancement rule

Primary ranking uses participant-macro mean MAE on fold 4. A candidate advances
only if it improves Overall by at least 0.15 mmHg and improves both MIMIC and
VitalDB internal source strata. A single-seed winner is not a final result: it
must later pass independent-seed confirmation before any locked evaluation.

## Interpretation guardrail

This is an architecture-family comparison, not a reproduction of published
numbers. Published papers use different signals, support budgets, cohort
definitions, train/test overlap and target definitions. A failed candidate
means the family did not help under this fixed protocol; it does not falsify
the paper. A successful candidate justifies a later method-level replication
such as self-supervised pretraining or supervised morphology auxiliaries.
