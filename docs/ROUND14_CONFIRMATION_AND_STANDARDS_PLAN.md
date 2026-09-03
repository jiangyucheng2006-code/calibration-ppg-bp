# Round 14: paired confirmation and standards-oriented development plan

## Decision questions

Round 14 has two separate development lines.

1. **Line A asks whether the Round-13 architecture result is reproducible.**
   It performs a paired five-seed comparison of the compact ResNet and the
   wider InceptionTime encoder under the unchanged K=5 Quality Gate + Huber
   (QGH) calibration protocol.
2. **Line B asks whether complete calibration-relative methods can improve
   both the primary participant-level error and the numerical error
   distribution.** It tests two prespecified full wider-InceptionTime causal
   methods: a calibration-relative predictor and a standards-oriented version
   with physical-mmHg losses. A third complete candidate adds a source-by-
   relative-range smooth worst-group SBP penalty only if its preflight group-
   count gate is satisfied.

Line A is confirmatory with respect to the internal Round-13 discovery. Line B
remains a prospective method-development screen. A result from Line B cannot
be counted as confirmation of Line A because its head and objective differ.
Neither line accesses meta-validation or the locked meta-test.

## Frozen data and evaluation boundary

The following boundary applies to every trained setting unless a section below
states a stricter rule:

- input signal: one 10-second PPG window per event;
- support: the first five eligible chronological pseudo-cuff/reference-BP
  events for each participant (`K=5`, fixed-first);
- query: event 6 onward, with the same eligible query keys for every paired
  comparison;
- fitting: participant-disjoint `meta_train` internal folds 0, 1, and 2;
- early stopping: internal fold 3, patience eight, with no epoch-count cap;
- internal evaluation and candidate ranking: fold 4;
- calibration protocol: support PPG and support BP may be used; query BP,
  later reference BP, and future query information may not be model inputs;
- source identity may be used only for aggregate reporting and, in the
   optional group-robust candidate, for fit-fold loss grouping; it is never a model
  input or an inference-time router;
- meta-validation: not used for fitting, early stopping, hyperparameter
  choice, model selection, prediction, or reporting;
- locked meta-test: not accessed;
- primary metric: participant-macro SBP MAE, DBP MAE, and their mean;
- required views: Overall, PulseDB MIMIC, and PulseDB VitalDB, recomputed from
  the same saved predictions;
- MIMIC and VitalDB are internal PulseDB source strata, not independent
  external-validation datasets.

The data-store manifest, participant-fold file, query-key set, target values,
preprocessing configuration, effective batch sizes, feature dimension, and
training audit fields must be hash-locked before submission. A run is invalid
if any safety flag indicates meta-validation access, locked-test access, query
BP input, future-query input, or source input to the predictor.

## Line A: paired architecture confirmation

### Compared settings

Every seed trains and evaluates both settings on identical data and queries:

- reference: `resnet_small + QGH`;
- candidate: `inception_time_wide + QGH`.

The population encoder is the only intended architectural difference. Both
encoders output 256 features and use the same QGH calibration head, optimizer,
Huber loss, sampled examples per epoch, effective batches, early stopping, and
fold boundary.

### Prespecified seeds

- Round-13 discovery seed: `20260827`;
- new confirmation seeds: `20260828`, `20260829`, `20260830`, and `20260831`.

The accepted, hash-verified Round-13 predictions are reused for the discovery
seed. Both backbones are retrained from scratch for each new seed. Seed numbers
may not be replaced after observing convergence or fold-4 results. A technical
failure is rerun with the same seed and frozen configuration; it is not
substituted with a more favorable seed.

### Paired reporting

For each seed and scope, define the paired gain as

```text
gain = mean participant-macro MAE(resnet_small + QGH)
       - mean participant-macro MAE(inception_time_wide + QGH)
```

A positive gain favors the wider InceptionTime encoder. The report must contain:

- all five per-seed SBP, DBP, and mean participant-macro MAEs;
- all five paired Overall, MIMIC, and VitalDB gains;
- across-seed mean and standard deviation for each setting;
- mean and standard deviation of the paired gain;
- the number of seeds with a positive Overall gain;
- a separate four-new-seed-only sensitivity table so the discovery seed is not
  hidden inside the combined summary;
- complete event-pooled diagnostics for transparency, with participant-macro
  results remaining primary.

Because there are only five seeds and one internal participant split, exact
paired differences and their consistency are more informative than a
standalone significance-test label. No claim of population-level replication
is made from a seed-only experiment.

### Confirmation gate

The four genuinely new seeds define the primary Line-A confirmation. The
Round-13 discovery seed remains in the descriptive five-seed tables but cannot
rescue a failed new-seed result. The wider InceptionTime architecture passes
Line A only if all four conditions hold over seeds `20260828--20260831`:

1. mean Overall participant-macro mean-MAE gain is at least 0.15 mmHg;
2. mean MIMIC gain is greater than 0 mmHg;
3. mean VitalDB gain is greater than 0 mmHg; and
4. at least three of four seeds have a positive Overall gain.

The five-seed summary remains mandatory context and is reported as combined
seed-stability evidence, not independent replication. The VitalDB SBP/DBP
asymmetry observed in Round 13 must also be reported explicitly; averaging the
two endpoints may not conceal a consistently worse SBP result.

If the gate passes, `inception_time_wide + QGH` becomes the development
architecture reference for future work. If it fails, the compact ResNet remains
the reference and the Round-13 gain is treated as seed-sensitive. Either result
is reported without accessing meta-validation or the locked meta-test.

## Prespecified five-seed ensemble diagnostic

After all five paired seeds pass the query-identity audit, a CPU-only diagnostic
averages predictions across seeds with fixed equal weights. For endpoint
`b` in `{SBP, DBP}`:

```text
prediction_ensemble_b = (1 / 5) * sum(prediction_b_seed_s for s in five seeds)
```

Separate five-seed ensembles are formed for `resnet_small + QGH` and
`inception_time_wide + QGH` so their comparison remains paired. No seed is
weighted according to fold-4 performance, and no participant-specific or
source-specific ensemble weights are fitted.

This is a preregistered variance-reduction diagnostic, not a third candidate
in the Line-A confirmation gate. It is computed only from already saved
predictions, cannot rescue a failed per-seed confirmation, and cannot be used
to tune any training setting. Overall, MIMIC, VitalDB, participant-macro, and
pooled numerical diagnostics are reported using the same common query set.

## Line B: complete standards-oriented candidate

### Rationale

Round 13 indicates that multiscale convolution width can help, while repeated
increases in generic network size do not. Earlier development also indicates
that calibration-relative and causal information can be useful. Line B
therefore tests one coherent prediction method rather than another large
backbone sweep or a combinatorial module ablation.

### Architecture-matched reference

The Line-B reference is the `inception_time_wide + QGH` run for prespecified
seed `20260828`. It is trained in Line A and reused only as the exactly matched
Line-B reference: the candidates use the same seed, population and QGH
checkpoints, participant folds, and fold-4 query keys. Reusing this artifact
avoids an unnecessary duplicate training run; it is not an unmatched or
post-hoc reference.

### Full candidate C1: calibration-relative causal prediction

Candidate C1 contains all of the following components from the start:

1. **Wide InceptionTime PPG encoder.** The 10-second query and support PPG
   windows share the same population encoder.
2. **Calibration-relative prediction.** The head predicts BP change relative
   to a support-derived personal anchor rather than predicting absolute BP
   without context. Only the first five support BP labels define the anchor.
3. **Causal query history.** A causal recurrent state may use current and
   earlier query PPG embeddings, support information, and its own latent state.
   It may not use query BP labels, later query embeddings, or future reference
   measurements.
4. **Relative-to-support-range auxiliary task.** During folds 0--2 fitting,
   the model predicts whether the query BP target is below, within, or above
   the range of the first five support BP labels. Query BP is used only as the
   auxiliary supervision label on fitting folds. The label is never a model
   input, is unavailable at inference, and is not used to route fold-3 or
   fold-4 predictions.

C1 is evaluated as one complete alternative to QGH. Round 14 does not launch
a large `A`, `A+B`, `A+B+C` factorial ablation grid. Candidate-level
comparisons cannot be interpreted as isolated component effects. If a
candidate fails, its fold-4 result is not used to retune individual
coefficients and rerun the same screen.

### Full candidate C2: standards-oriented calibration-relative prediction

C2 contains the complete C1 method and adds two fixed physical-mmHg training
terms. This is a separate complete candidate designed to move the entire error
distribution, not an after-the-fact C1 ablation.

### Physical-mmHg threshold-aware objective

Let `e_b = prediction_b - target_b` in mmHg for endpoint `b`. In addition to
the established C1 objective, C2 adds a physical-unit Huber term and a bounded
differentiable threshold term at the historical 5, 10, and 15 mmHg reporting
thresholds:

```text
p_tau(e_b) = sigmoid((abs(e_b) - tau) / 1 mmHg)
L_threshold = 0.50*p_5 + 0.30*p_10 + 0.20*p_15
L_total = L_C1 + 0.10*L_Huber_mmHg + 0.25*L_threshold
```

The physical Huber term uses a fixed 5-mmHg transition and is divided by its
squared transition value to remain numerically comparable with C1. The
1-mmHg threshold temperature and all weights above are frozen before training.
The bounded threshold term encourages errors to remain below the reported
physical thresholds without allowing a few extreme samples to dominate. The
model is still selected by participant-macro development performance, not by
repeatedly changing loss weights to optimize the fold-4 BHS table.

The term is called *standards-oriented* only because it uses the same numerical
error thresholds as the retrospective reporting table. It does not turn the
training run into a standards-compliance study.

### Optional full candidate C3: source-by-relative-range group robustness

C3 contains the entire C2 method and adds a fixed GroupDRO-inspired smooth
worst-group SBP penalty during fit-fold training. It is a separate full
robustness candidate, not an inference-time router and not a post-result repair
of individual participants.

The implemented penalty is GroupDRO-inspired but is not canonical GroupDRO:
it computes a smooth excess over the groups present in each minibatch and does
not maintain persistent group weights across minibatches. It is therefore
reported as a *smooth worst-group SBP penalty*, not as evidence that a
particular canonical GroupDRO algorithm was evaluated.

Training events are assigned to one of six prespecified groups:

```text
source in {MIMIC, VitalDB}
SBP target relative to the first-five support SBP range in {
  below_support,
  within_support,
  above_support
}
```

Group assignment may use query SBP targets only on folds 0--2 for loss
weighting. The group label is not provided to the predictor and is unavailable
at inference. Fold-3 and fold-4 target values are used only after prediction
for early-stopping scoring or aggregate evaluation; they never select a route.

C3 is run only if a preflight audit on folds 0--2 finds at least 50 participants
and 1,000 eligible training events in every group. If the gate fails, C3 is
recorded as not feasible and groups are not merged after inspecting fold-4
performance. The smooth-worst temperature and penalty weight must be written
to the frozen machine-readable configuration before submission. They are
fixed here at `0.25` standardized-loss units and `0.25`, respectively, and may
not be changed after fold-4 results are observed.

The initial Round-14 submission therefore queues C1 and C2 only. C3 is held
out of that dependency chain until the new-seed cache has produced the frozen
six-group preflight audit. If eligible, it is submitted as a separate
prespecified follow-up and cannot cancel the C1/C2 report; if ineligible, only
the `not feasible` audit outcome is recorded.

### Line-B screening seed and advancement rules

Line B uses prespecified development seed `20260828` for the initial full
candidate screen. C1, C2, and, if eligible, C3 are paired with the same-seed
architecture-matched reference on identical fold-4 queries.

Each Line-B candidate is checked independently against the following gates;
the report does not first choose an Overall winner and then test only that
winner. A candidate qualifies as a **primary-model candidate** only if it:

1. improves Overall participant-macro mean MAE by at least 0.15 mmHg;
2. improves participant-macro mean MAE in both MIMIC and VitalDB;
3. has full query coverage identical to the reference; and
4. does not worsen either SBP or DBP participant-macro MAE by more than 0.05
   mmHg Overall.

A candidate may instead qualify as a **tail-focused candidate** only if all of
the following hold:

1. Overall participant-macro mean MAE is non-inferior within 0.05 mmHg;
2. neither source-stratum mean MAE worsens by more than 0.10 mmHg;
3. event-pooled error standard deviation decreases for both SBP and DBP;
4. for each endpoint, the average of the within-5/10/15-mmHg percentages
   improves by at least 1.0 percentage point; and
5. no individual within-threshold percentage decreases by more than 0.5
   percentage point.

Passing either development gate advances the complete candidate to a later
paired multi-seed confirmation. It does not authorize meta-validation or
locked-test access. A result that meets neither gate is recorded as negative;
no coefficient, group definition, or routing rule is changed after viewing
fold 4.

The numerically lowest Overall result is reported descriptively, but it does
not suppress another prespecified candidate that independently passes a gate.
Because fold 4 has been reused across prior development rounds, all Line-B
results remain exploratory and may motivate a frozen later confirmation; they
are not independent-subject replication or final evidence.

## Prespecified exclusions

Round 14 does not repeat or recombine the following previously unproductive or
out-of-scope routes:

- another broad Transformer, Conformer, ConvNeXt, residual-attention, or generic
  parameter-scale sweep;
- hard 70/30 routing, hard waveform clusters, mixture-of-experts routing, or a
  classifier that sends participants to separate predictors;
- CVaR, hardest-tail oversampling, or unconstrained high-error reweighting;
- query filtering by beat similarity, because it changes coverage rather than
  improving full-coverage prediction;
- demographic conditioning while audited age anomalies remain unresolved;
- source identity as a predictor input;
- a large factorial addition/removal grid assembled after inspecting fold-4
  results.

This exclusion is deliberate: Round 14 is designed to distinguish a
reproducible architecture gain from one complete physiology- and
calibration-motivated method, rather than increasing complexity without a
matching performance gain.

## Required reports and integrity checks

Every formal report must include:

- Overall, MIMIC, and VitalDB participant-macro SBP, DBP, and mean MAE;
- per-seed and across-seed Line-A results, plus exact paired gains;
- discovery-seed and four-new-seed-only summaries;
- event-pooled MAE, R-squared, mean error, error standard deviation, and
  within-5/10/15-mmHg percentages for SBP and DBP;
- retrospective AAMI-style numerical screen and historical BHS grade, clearly
  marked as diagnostic only;
- participant and query counts for every scope;
- identical-query and identical-target hashes across paired settings;
- source-code, data-manifest, fold, configuration, and prediction-artifact
  hashes;
- all safety flags, including explicit confirmation that meta-validation and
  the locked meta-test were not accessed.

Overall is recomputed from all eligible participants and events. It is not an
average of the MIMIC and VitalDB rows. Source-stratified R-squared and error
standard deviations are never averaged to construct Overall. Participant-level
or event-level predictions, personal identifiers, raw signals, checkpoints,
logs, and private infrastructure details are not published.

## Why PulseDB numerical screening cannot prove standards compliance

Even if a model numerically satisfies the commonly quoted mean-error,
error-standard-deviation, or 5/10/15-mmHg thresholds on this internal fold, the
correct statement is only that it **passes a retrospective AAMI-style or
historical BHS numerical screen on PulseDB**. It is not valid to claim that the
model or a device is AAMI-, ISO-, IEEE-, or BHS-compliant.

Formal compliance or device validation requires a prospective, device-specific
protocol under the relevant standard, including the prescribed reference
measurement procedure, participant and BP distributions, measurement pairing,
repeat-measurement handling, calibration independence, operating conditions,
and analysis population. PulseDB is a retrospective waveform resource; its
MIMIC and VitalDB subsets are internal sources, and its ABP-derived labels and
correlated windows are not a substitute for a standards-governed cuff/device
validation study. Model-development reuse of internal fold 4 is another reason
that it cannot serve as a final compliance cohort.

Accordingly, Round 14 can identify a better development candidate and quantify
whether its error distribution moves toward desirable numerical thresholds.
It cannot establish clinical safety, external generalization, or formal
standards compliance. Those claims require later independent external and
prospective validation after the model and calibration protocol are frozen.
