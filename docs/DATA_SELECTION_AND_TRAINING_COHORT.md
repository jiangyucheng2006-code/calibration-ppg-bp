# PulseDB data selection and first-run training cohort

Last updated: 2026-08-15.

## Short answer

The first training run did not use all 5.25 million PulseDB segments as
independent samples. All 5,361 participant files and all 5,245,454 indexed
segments passed the structural audit, but the experiment then:

1. quarantined 804 participants as a locked test set;
2. collapsed adjacent 10-second segments into one deterministic representative
   event per record-anchored 120-second block;
3. retained only participants with at least five calibration-candidate events
   and at least five later common-query events; and
4. used participant-balanced sampling during neural-network training.

The large reduction is therefore mainly a deliberate independence, temporal,
calibration-feasibility, and leakage-control decision. It is not evidence that
most original waveforms were corrupt.

## Selection funnel

| Stage | Rule | Participants | Rows | First-run use |
|---|---|---:|---:|---|
| Extracted PulseDB v2 | Exact official source counts: 2,423 MIMIC + 2,938 VitalDB files | 5,361 | -- | Audited |
| Full schema/segment audit | Required structure, identity, time, PPG, BP, duplicate, and overlap gates | 5,361 | 5,245,454 valid segments | Candidate cohort |
| Frozen subject split | Source-stratified 70/15/15 split, seed `20260809` | 3,752 train / 805 validation / 804 locked test | 4,463,294 development / 782,160 locked segments | Locked test excluded from development |
| Development quality gate | `segment_schema_valid` and `IncludeFlag == 1` | 4,557 development | 4,463,294 | No additional rows were lost; every development row had `IncludeFlag == 1` |
| 120-second eventization | One segment closest to each record-anchored bin centre | 4,557 development | 608,582 events | Event candidates |
| Eligibility gate | At least 10 events: first five support candidates plus at least five later queries | 3,840 development | 606,010 events | Included in the materialized development store |
| Eligible meta-train | Frozen split intersected with eligibility | 3,143 | 498,961 events | Model fitting only |
| Eligible meta-validation | Frozen split intersected with eligibility | 697 | 107,049 events | Checkpoint/model selection only |
| Common validation query | Event 6 onward, identical for K=1/2/3/5 | 697 | 103,564 queries | Every first-run reported setting |

The 3,840 eligible development participants comprise:

- meta-train MIMIC: 1,427 participants, 384,740 events;
- meta-train VitalDB: 1,716 participants, 114,221 events;
- meta-validation MIMIC: 316 participants, 82,454 events; and
- meta-validation VitalDB: 381 participants, 24,595 events.

The 606,010 selected event rows represent 4,456,193 original schema-valid
10-second segments inside their temporal bins. The other segments are not
separate training observations because only one deterministic representative
is retained per 120-second event.

After the rule was frozen, it was applied unchanged to the locked test. This
produced 681 eligible locked participants, 104,874 model-input event rows, and
101,469 evaluator-only query targets. None of these locked targets has been used
for the first-run results or the ongoing repeated-seed development suite.

## Raw-file acceptance conditions

A participant file is accepted only when all of the following hold:

- it appears in one of the two expected source directories and follows the
  `pNNNNNN.mat` filename convention;
- it has the MATLAB 7.3 header and HDF5 signature;
- it contains a `Subj_Wins` group and all required fields: `PPG_Raw`, `PPG_F`,
  `ABP_Raw`, `ABP_F`, `SegSBP`, `SegDBP`, `T`, `SubjectID`, `CaseID`,
  `SegmentID`, `WinID`, `WinSeqID`, `IncludeFlag`, `PPG_ABP_Corr`, and
  `ABP_Lag`;
- required fields use one consistent storage mode and contain aligned window
  counts with no null references;
- the decoded participant identity is non-empty, matches the filename, and is
  unique after qualifying it by source;
- the file contains at least one segment that passes the segment-level gates;
  and
- the same source-qualified participant does not occur in more than one raw
  file.

The actual full-data audit found 5,361 of 5,361 files schema-valid, zero invalid
files, and zero duplicate source-qualified participant IDs.

## Segment acceptance and exclusion conditions

A segment is eligible for the indexed cohort only when:

- `T` has at least two samples, all values are finite, and time is strictly
  increasing;
- `PPG_Raw` and `PPG_F` lengths match `T`;
- filtered PPG (`PPG_F`) is finite, non-empty, and non-constant;
- required scalar metadata are finite;
- SBP is greater than DBP;
- subject and record identifiers are non-empty;
- its interval is not duplicated within the record; and
- it does not have positive-duration overlap with the previously accepted
  chronological interval in that record.

For event construction, a segment must additionally have `IncludeFlag == 1`.
In the full development cohort this flag removed no rows. The audit recorded a
broad diagnostic BP-range flag (`SBP < 50`, `SBP > 260`, `DBP < 20`, or
`DBP > 180` mmHg), but this was deliberately **not** used as an exclusion
threshold. Changing a BP-range or signal-quality threshold after looking at
model errors would create selection bias and is not allowed for the locked
test.

## Event and calibration conditions

Within each participant-record pair, bins are anchored at the first available
segment and have a fixed width of 120 seconds. The segment whose start time is
closest to the bin centre represents the event; ties are resolved
deterministically by start time and segment ID. BP is copied only from that
representative segment and is not averaged across the bin.

A participant is included only if at least 10 such events exist. Chronological
events 1--5 form a fixed support-candidate pool. For K=1/2/3/5, the first K
events are calibration support, unused candidates stay unused, and every K is
evaluated on the identical event-6-onward query set. Thus increasing K does not
silently produce an easier or different test set.

PulseDB reference BP is derived from invasive ABP. These events are therefore
pseudo-cuff events for algorithm development, not literal independent cuff
measurements.

## What enters the neural network

- Signal input: only the selected `PPG_F` waveform, 1,250 samples per event
  (approximately 10 seconds at 125 Hz).
- Waveform normalization: per-window z-score in the loader. A waveform with a
  nonfinite or near-zero (`<= 1e-8`) standard deviation is rejected.
- Targets: SBP and DBP, standardized using means and population standard
  deviations fitted on `meta_train` only.
- Population model: trained with participant-balanced sampling over eligible
  meta-train events; the first run requested 200,000 sampled examples per
  epoch.
- M0/M1/M2: trained with participant-balanced variable-K episodes; the first
  run requested 100,000 sampled episodes per epoch. Training uses only
  chronologically prior support; validation always uses the frozen first-K
  support pool.
- Validation: all 697 participants and all 103,564 common query events are
  scored for each applicable K. Participant-macro MAE is the primary model-
  selection metric.

Sampling with replacement means that not every eligible meta-train event is
necessarily visited in every epoch. The full eligible store remains available,
but the sampler prevents long-record participants from dominating only because
they contribute more events.

## Information explicitly not used as routine query input

The model does not receive reference ABP waveforms, `PPG_ABP_Corr`, `ABP_Lag`,
query SBP/DBP, future support events, ECG, demographics, source-specific BP
distributions, or locked-test outcomes. ABP-derived and quality fields may be
retained for offline audit or later prespecified subgroup analysis, but using
them as ordinary PPG-only inference features would violate the intended
deployment claim.

## Reproducibility and leakage evidence

- frozen split file SHA-256:
  `8705f7cd75d92201bd203c00fb4d8ad8c738c02d7c4b56e5747210ffba504cd7`;
- full segment index SHA-256:
  `4bd0d281b1fb3b0f715b23e41315405c6b13ad696452469f57c2c771d08e569d`;
- development episode SHA-256:
  `0a176f1392ed03845fbaa3aca9fd2c0fc1f2d35ec9aca0f00f0a1f08b2c7e2bf`;
- automated subject-disjointness, temporal-order, common-query, and locked-
  label-isolation audit: passed with zero failures.

These rules were fixed before the first model-error inspection. Later error-
based subgroup analysis may identify difficult cases, but any new exclusion or
reweighting rule must be learned on development participants only and then
applied unchanged to the locked test or an external dataset.
