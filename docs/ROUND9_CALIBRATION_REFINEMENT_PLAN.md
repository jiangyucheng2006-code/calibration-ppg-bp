# Round 9: calibration-relative refinement

## Objective

Round 9 asks a narrower question than earlier tail-routing experiments: can the
model learn the *change from each participant's observed calibration events to
later PPG events* more accurately than the Round-8 architecture?

This is an exploratory K=5, single-seed screen. It does not establish final
generalization, external validity, AAMI compliance, or BHS compliance.

## Selection boundary

- Participants remain disjoint across all frozen project splits.
- The fixed-first five calibration events precede every event-6-onward query.
- Meta-train folds 0--2 fit each candidate.
- Meta-train fold 3 controls patience-8 early stopping with no epoch-count cap.
- Meta-train fold 4 ranks the candidates.
- Meta-validation is not used for training, early stopping, prediction,
  scoring, or candidate ranking in this screen.
- The locked meta-test is not accessed.
- The internal winner must improve mean participant-macro MAE by at least
  0.15 mmHg and improve both MIMIC and VitalDB internal source strata before
  one frozen meta-validation evaluation is considered.

This internal split is deliberately stricter than repeatedly choosing models
on the existing meta-validation results after many exploratory rounds.

## Candidate matrix

All candidates use the same query set, target scaling fitted on folds 0--2,
seed, optimization policy, Huber loss, pairwise calibration-delta auxiliary
loss, support-range auxiliary loss, and causal event order. Only the named
intervention changes.

| ID | Candidate | Plain-language purpose |
|---|---|---|
| R9-0 | R8 architecture reference | Refit the Round-8 architecture under the new internal selection protocol. |
| R9-1 | Adaptive base-personal fusion | Learn when to trust the population estimate and when to trust participant calibration. |
| R9-2 | Soft BP-range experts | Smoothly combine below-range, within-range, and above-range corrections instead of forcing one hard case. |
| R9-3 | DBP-specific physiology | Add morphology-derived correction only to DBP so an ineffective SBP auxiliary path cannot dilute the useful signal. |
| R9-4 | Bias regularization | Penalize systematic over- or under-estimation in each training batch. |
| R9-5 | Short causal attention | Let each query focus on the most relevant recent past events instead of compressing all history through one recurrent state. |
| R9-6 | Personal BP direction | Infer each participant's PPG-to-BP direction from all pairwise differences among the five calibration events. |
| R9-7 | Temporal delta consistency | Ask consecutive predicted BP changes to follow the direction and magnitude of observed training changes. |
| R9-8 | Support dropout consistency | Randomly hide one calibration event during training and require stable predictions, reducing dependence on one atypical cuff event. |

## Why these candidates

Earlier experiments showed that causal time context and explicit
query-to-calibration differences were more useful than demographics,
similarity filtering, hard 70/30 routing, error-tail specialists, or
unsupervised waveform clusters. Round 9 therefore concentrates on the direct
calibration-to-query relation and robustness of that relation.

The following routes are not repeated in this screen:

- error-ranked exclusion, because it is oracle-only at inference time;
- the previous hard/soft difficult-participant router, because its gain was
  negligible and source-asymmetric;
- the previous waveform phenotype experts, because their clusters were
  unstable and every expert worsened its own assigned group;
- scalar beat-similarity filtering, because similarity below 0.90 did not
  predict larger BP error;
- demographic conditioning and 128-D compression, because neither improved
  the Round-8 full-coverage result.

## High-upside second stage

The R9-0--R9-8 screen operates on frozen population PPG embeddings. This makes
the comparison fast and controlled, but also places a ceiling on improvement:
the encoder was not originally optimized to represent *within-participant BP
change*.

If no candidate clears the internal gate, the next scientifically distinct
experiment should not be another small correction head. It should partially
fine-tune the last PPG-encoder block using query-to-support waveform pairs and
an auxiliary BP-change objective, while keeping participant-disjoint internal
selection and the same fixed-first K=5 protocol. That end-to-end route is more
computationally expensive and carries greater overfitting risk, but it is the
most plausible route to a larger improvement because it changes what the PPG
representation itself learns.

If one lightweight candidate clears the gate, it can also be combined once
with this partial end-to-end encoder adaptation. The combination must be
treated as a new candidate and cannot inherit the earlier candidate's evidence.

## Required outputs

The internal screen stores one checkpoint, learning history, and fold-4
prediction table per candidate. The deterministic report recomputes
participant-macro SBP, DBP, and mean MAE for Overall, MIMIC, and VitalDB from
the same prediction keys. No candidate is promoted from a pooled Overall value
alone.
