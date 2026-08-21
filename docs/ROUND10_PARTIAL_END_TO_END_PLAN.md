# Round-10 leakage-safe partial end-to-end plan

Date frozen: 2026-08-21. Development seed: `20260824`.

## Decision and motivation

Round 9 did not produce a promotable candidate. Its best Overall improvement
was 0.036 mmHg participant-macro mean MAE and was not consistent across the
two internal PulseDB source strata. All Round-9 candidates operated on a
frozen 256-dimensional PPG representation, so the next scientifically distinct
question is whether calibration-aware gradients must update the PPG encoder
itself.

Round 10 is a development-only screen of partial end-to-end adaptation. It is
not a final result and it does not access the locked meta-test. It also repairs
an internal-screening limitation: the older frozen representation had been
trained before the later fold-4 candidate comparison was defined. Round 10
therefore rebuilds both base networks without using fold 4 for fitting or early
stopping.

## Data roles and leakage boundary

The five participant-disjoint folds inside `meta_train` have exactly one role
each:

| Role | Participant folds | Allowed use |
|---|---|---|
| Model fitting | 0, 1, 2 | Population model, QGH base, calibration head, selected encoder suffix |
| Early stopping | 3 | Patience-8 checkpoint selection only |
| Candidate ranking | 4 | One comparison after each candidate is frozen |
| Meta-validation | none | Not read, predicted, scored, or ranked in Round 10 |
| Locked meta-test | none | Quarantined |

Target normalization is fitted on folds 0--2 only. Participant identities are
disjoint across fitting, early-stopping, and ranking roles. For every
participant, events 1--5 provide K=5 calibration PPG and BP, and the identical
common-query set beginning at event 6 is predicted. Query BP is a loss target
only in the fitting role and is never a model input.

## Prerequisite models

1. Rebuild a 256-dimensional multiscale 1-D ResNet population model on folds
   0--2, with fold 3 controlling patience-8 early stopping.
2. Rebuild the current Quality Gate + Huber (QGH) M0 base under the same split,
   fixed-first K=5 support, and frozen population mapping.
3. Materialize only meta-train waveform pointers, five calibration events,
   QGH base predictions, and audited fold roles. No meta-validation or test
   artifact is included.

These prerequisites make the fold-4 comparison independent of supervised
representation fitting and checkpoint selection.

## Candidate matrix

Each candidate uses the same newly built population encoder, QGH base
prediction, K=5 support events, calibration-relative pair head, and causal GRU.
Only the listed factor changes.

| ID | Method | Trainable PPG representation | Additional objective or output rule |
|---|---|---|---|
| T10-0 | Frozen reference | None | Architecture-matched calibration head |
| T10-1 | Projection adaptation | Final projection only | None |
| T10-2 | Last-block adaptation | Last residual block + projection | None |
| T10-3 | Last-two-block adaptation | Last two residual blocks + projection | None |
| T10-4 | Full-encoder control | Entire encoder | Low learning-rate overfitting control |
| T10-5 | Direction-aware | Last block + projection | Predict sign of query-to-support SBP/DBP change |
| T10-6 | Temporal-consistency | Last block + projection | Penalize implausible mismatch in consecutive BP change |
| T10-7 | Adaptive fusion | Last block + projection | Learn when to trust QGH base or personalized estimate |
| T10-8 | Joint interaction | Last block + projection | Direction and temporal-consistency objectives together |

The main candidate is T10-2. T10-1 and T10-3 identify how much representation
adaptation is needed; T10-4 is a deliberately higher-variance control. T10-5
and T10-6 test the two signals that were most defensible after Rounds 7--9:
calibration-relative BP change and causal within-participant history. T10-7
tests cautious fusion rather than unconditional correction. T10-8 tests their
interaction only after the isolated components are retained in the same
screen.

Negative or weak prior factors are not repeated: oracle worst-tail routing,
unsupervised waveform clusters, demographic conditioning with unresolved age
anomalies, 128-dimensional compression, and rejection of windows using the
beat-similarity threshold.

## Optimisation

- Primary loss: standardized SBP/DBP Huber-style loss inherited from the
  calibration-relative head, with its range auxiliary term.
- Calibration-head learning rate: `5e-4`.
- Encoder learning rates: projection `1e-4`, last block `5e-5`, last two blocks
  `3e-5`, and full encoder `1e-5`.
- Optimizer: AdamW, weight decay `1e-4`, gradient norm clipped at 5.
- Epoch cap: none. Stop after eight consecutive fold-3 epochs without
  improvement in participant-macro mean MAE.
- Long records: training samples one chronological segment of at most 256
  future events per participant per epoch. Evaluation processes the complete
  chronological record in 256-event chunks while carrying the causal GRU
  state, so no query is removed and chunking does not reset temporal history.
- BatchNorm running statistics remain frozen during encoder adaptation because
  participant-sequence batches are small.

## Outcomes and promotion rule

The primary endpoint is K=5 participant-macro mean MAE on fold 4. SBP and DBP
participant-macro MAE are also reported. `Overall`, `PulseDB MIMIC`, and
`PulseDB VitalDB` are recomputed from the same saved predictions. Event-pooled
MAE, R-squared, mean error, error SD, percentages within 5/10/15 mmHg, and
retrospective AAMI/BHS numerical screens are secondary diagnostics.

A candidate advances only if all conditions hold:

1. Overall participant-macro mean MAE improves by at least 0.15 mmHg relative
   to T10-0;
2. MIMIC participant-macro mean MAE improves;
3. VitalDB participant-macro mean MAE improves.

If no candidate passes, no Round-10 model is evaluated on meta-validation. If
one passes, the experiment stops for review before any meta-validation or
multi-seed confirmation is submitted. The locked meta-test remains untouched
until the complete final method and seed policy are frozen.

## Interpretation limits

Round 10 is a single-seed internal screen and cannot establish a final method.
Its fold-4 comparison is participant-disjoint, but repeated rounds still create
researcher degrees of freedom; the fixed 0.15-mmHg and two-source gate limits
promotion of negligible or source-specific changes. PulseDB MIMIC and VitalDB
are internal source strata, not independent external validation datasets.
