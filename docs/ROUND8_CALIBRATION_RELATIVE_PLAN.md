# Round-8 calibration-relative development screen

## Purpose

Round 8 tests whether the model can use the *relationship* between each of the
five calibration events and a later PPG query more effectively than the
current Quality Gate + Huber reference. It also implements three collaborator
requests as isolated development experiments: a beat-similarity threshold,
direct demographic concatenation, and a 128-dimensional PPG representation.

This is K=5, development-only, single-seed screening. The locked meta-test is
not accessed. A candidate is promoted only if it improves Overall
participant-macro mean MAE by at least 0.15 mmHg relative to job 841 and also
improves both internal PulseDB source strata, MIMIC and VitalDB.

## Candidate matrix

| ID | Candidate | What changes | Primary comparison |
|---|---|---|---|
| Reference | Quality Gate + Huber, 256-D | Existing full-coverage model (job 841) | Frozen reference |
| R8-1 | Pairwise delta | Compare the query separately with every calibration PPG/BP pair, then combine the five estimates | Reference |
| R8-2 | Pairwise delta + causal time | Add a recurrent state that can use only the current and earlier query events | R8-1 |
| R8-3 | Pairwise delta + BP-range task | Add an auxiliary task predicting whether query BP is below, within, or above the five support BP values | R8-1 |
| R8-4 | Pairwise delta + causal time + BP-range task | Combine the two independently screened additions | R8-2 and R8-3 |
| R8-5 | R8-4 + generic PPG-change features | Add target-free waveform shape/change summaries for support-to-query comparison | R8-4 |
| R8-6 | Similarity >=0.90 sensitivity | Apply the existing reference only where finite within-window beat similarity is at least 0.90 | Coverage analysis only |
| R8-7 | Direct demographics + 256-D | Concatenate five cleaned demographic values directly to the 256-D PPG representation | Reference |
| R8-8 | Quality Gate + Huber, 128-D | Retrain the population encoder and personalised model with a 128-D PPG representation | Reference |

R8-6 is deliberately not ranked as a full-coverage model. Removing difficult
or unmeasurable windows changes the population being scored. Its report must
therefore show retained-query and retained-participant coverage for Overall,
MIMIC, and VitalDB alongside the conditional error.

## Calibration-relative model

For a query representation `q` and each of the five support representations
`s_i`, the pair module receives only inference-visible information:

- query and support PPG representations;
- signed and absolute representation differences;
- support cuff BP;
- difference between the frozen population prediction and each support event;
- the query-to-support event gap;
- for R8-5 only, generic target-free waveform-shape differences.

The model estimates five support-relative BP changes and learns attention
weights to combine them. The output is initialized as the existing Quality
Gate + Huber prediction, so a new correction starts from zero rather than from
an arbitrary BP estimate.

The causal-time variants process queries in temporal order. At event `t`, the
recurrent state can use events up to `t`; it cannot see later PPG events or any
query BP label. Query BP is used only as a meta-train loss target or a
meta-validation score.

## Training and leakage controls

- Fixed first five calibration events; event 6 and later are queries.
- Participant-disjoint meta-train cross-fitting outputs are the Round-8
  training inputs; folds 0--3 fit the model and fold 4 controls early stopping.
- No epoch-count cap; stop after eight consecutive non-improving epochs.
- The meta-validation query set is used once after model selection.
- Source, participant identity, true query error, future events, and query BP
  are excluded from model inputs.
- R8-7 uses the audited five-value representation: standardized adult age,
  age-valid flag, and female/male/unknown indicators. Invalid age is encoded
  with an explicit missingness flag; the age scaler is fitted on meta-train.
- R8-8 changes only encoder width. It does not add demographics or another
  model component.

## Required report

Every full-coverage candidate must contain exactly the same 697 participants
and 103,564 K=5 queries as the reference. The report is generated from saved
predictions and separately recomputes:

1. Overall;
2. PulseDB MIMIC internal source stratum;
3. PulseDB VitalDB internal source stratum.

Participant-macro SBP, DBP, and mean MAE are primary. Event-pooled MAE, R2,
ME, STD, proportions within 5/10/15 mmHg, and retrospective AAMI/BHS numerical
screens are secondary. The numerical screens do not establish standards
compliance or clinical validity.
