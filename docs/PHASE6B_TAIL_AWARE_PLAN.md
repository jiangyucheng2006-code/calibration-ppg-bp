# Phase-6B tail-aware personalization plan

## Status and scope

This document defines a development-only plan for improving the high-error tail
of the calibrated PPG--BP model. It does not authorize locked meta-test access,
change the frozen `event120-v1` cohort, or replace the fixed-first primary
protocol. The selected Phase-6 single-factor model remains the reference until
one tail-aware intervention passes the development gate below.

**Result update, 2026-08-19:** the full factorial is complete. Quality gate
plus Huber is the best full-coverage setting (four-K Overall participant-macro
mean MAE 8.888 mmHg versus 9.137 for fixed-first M0). Participant-CVaR does not
improve the full cohort and is not promoted. Full results, source-separated
tables, paired participant bootstrap intervals, and the fixed-reference oracle
tail diagnostic are reported in
[RESULTS_PHASE6B_FACTORIAL.md](RESULTS_PHASE6B_FACTORIAL.md).

The immediate scientific problem is not simply to lower a pooled mean. The
existing development analysis indicates that high-error participants have
larger within-participant BP variability, larger support-to-query BP change,
more targets outside the support BP range, and later query horizons. High BP is
over-represented but participant mean BP has only a weak association with
error. Consequently, the tail plan must distinguish calibration drift and
input uncertainty from BP-range imbalance.

## Decision summary

1. Removing the observed worst 30% is an **oracle diagnostic only**. It may
   quantify an upper bound for a perfect risk detector, but it is not a
   deployable exclusion rule.
2. A deployable risk model must use only information available before or at the
   current PPG query: support PPG/BP, current and previous PPG, elapsed time,
   source/device metadata if genuinely available, demographics, and model
   uncertainty. It may not use query BP, query residual, future events, or
   locked-test labels.
3. The first tail-training candidate should be participant-level Huber-CVaR,
   because it directly optimizes the worst 30% participant-loss tail without
   adding an inference-time input.
4. Input-only selective prediction is evaluated as a risk--coverage and
   recalibration policy, not as a way to hide difficult cases. Flagged events
   must have an explicit action, such as requesting a new cuff measurement.
5. A two-expert model and adaptive recalibration are promoted only if an
   input-only risk score can identify high-error cases on held-out development
   participants.
6. All result packages must report the prespecified aggregate result and the
   two PulseDB sources separately: overall, MIMIC, and VitalDB.

## Implemented Phase-6C two-stage experiment

The first deployable two-stage experiment is now specified as follows. It is a
development-only `K=5` prototype because the observed worst-30% analysis and
the cross-fitted target are defined from the five-cuff M0 setting. It does not
change the project's primary `K=1/2/3/5` result matrix.

1. Divide only `meta_train` participants into five source-stratified,
   participant-disjoint folds.
2. For every fold, retrain the population model and M0 without that fold and
   predict the held-out fold at `K=5`. The concatenated predictions are
   out-of-fold (OOF); no participant is labelled by a model trained on that
   participant.
3. Within MIMIC and VitalDB separately, label exactly the highest-error
   `ceil(0.30*N)` OOF participants as difficult. Source is used to prevent one
   database from monopolising the label, but source is not an input to the
   risk classifier.
4. Train a small risk MLP from current M0 predictions, first-five support BP
   dispersion, raw filtered-PPG summary statistics and query-to-support PPG
   distances, plus event horizon. Query BP, query error, target-range flags,
   and participant identity are prohibited inputs. One cross-fitting fold is
   reserved for risk-model early stopping; its threshold is fixed from the
   risk-training folds rather than fitted to `meta_validation`.
5. Train three specialist candidates from the same OOF labels: all
   `meta_train` participants with the difficult group sampled four times as
   often; difficult participants only; and difficult participants only with
   the PPG quality gate. These runs use no meta-validation tail labels.
6. Compare the ordinary M0, each specialist alone, hard routing, and soft
   risk-weighted fusion on `meta_validation`. Hard routing uses the frozen
   event-risk threshold; soft fusion uses the risk probability as the expert
   mixing weight.

The specialists are not accepted merely because they improve the oracle tail.
Promotion requires an input-visible risk classifier with useful held-out
participant AUPRC, a gain in the predicted-high-risk group, no material loss in
full-coverage performance, and consistent Overall/MIMIC/VitalDB reporting.
The current classifier uses only information available at the current query,
so its decision is an event-level risk decision rather than an immediate
post-calibration participant diagnosis. Extending this prototype to K=1/2/3
requires separate features that use no more than the corresponding K cuff
measurements.

## 1. Three different tail definitions

### 1.1 Oracle participant tail

Rank participants by participant-macro mean MAE and define the highest-error
30% as the oracle tail. Recalculate metrics for the remaining 70% and for the
removed 30% separately.

This analysis answers: *How much error is concentrated in a minority of
participants?* It does not answer: *Can those participants be identified from
PPG before their reference BP is known?*

Every oracle-retained result must be labelled, for example,
`oracle retained-70% diagnostic`. It must not be used for a headline accuracy,
standards claim, training threshold, or locked-test exclusion.

### 1.2 Deployable participant risk

A participant-risk model predicts whether a participant is likely to remain
difficult using only the first K support events and information genuinely
available at calibration time. It cannot use aggregates over future query PPG
if the intended decision is made immediately after initial calibration.

### 1.3 Deployable event risk

An event-risk model predicts whether the current query is unreliable using the
current PPG and history up to that event. This is the appropriate unit for
abstention, a tail expert, or an adaptive cuff request. Participant-tail and
event-tail results must be reported separately.

## 2. Cross-fitted construction of a deployable risk model

Tail membership is supervised by model error, so it must be created without
in-sample optimism:

1. Split `meta_train` participants into participant-disjoint cross-fitting
   folds.
2. Train the selected base model on all but one fold and generate out-of-fold
   predictions for the held-out participants.
3. Define participant- or event-tail targets from these out-of-fold errors
   only.
4. Train the risk model on deployment-visible features.
5. Fix one risk threshold or target coverage using cross-fitted meta-train
   risk-training predictions; use `meta_validation` to compare the frozen
   policy, not to fit its threshold.
6. Freeze the base model, feature computation, risk model, and threshold before
   any locked-test evaluation.

Using the base model's error on the same participants used to fit that model is
not an acceptable substitute for cross-fitting. Using meta-validation or
locked-test tail membership as a risk-model feature or training target is also
prohibited.

### 2.1 Input-only candidate features

| Feature family | Examples | Deployment-visible? |
|---|---|---:|
| Raw PPG integrity | clipping, flat-line fraction, missingness, spikes, raw dynamic range | Yes |
| PPG signal quality | beat-template correlation, peak-interval stability, physiologically plausible heart rate, spectral entropy, in-band energy | Yes |
| Morphology shift | distance between current-query and support PPG embeddings; change from recent PPG | Yes |
| Model uncertainty | disagreement or predictive SD across independently trained models; heteroscedastic output scale | Yes |
| Support reliability | support BP range/MAD; population-model support-residual range/MAD; disagreement between support anchors | Yes |
| Calibration horizon | time or event count since the last cuff event | Yes |
| Known context | MIMIC/VitalDB or device domain, age, sex, and explicit missingness indicators | Yes, if available in the intended setting |
| Current model state | current predicted BP and its distance from the support BP range | Yes |
| Query BP/error | reference BP, residual, absolute error, or whether true BP is outside support range | **No** |

Raw amplitude and clipping features must be computed before per-window z-score
normalization, because normalization can remove acquisition-quality
information. A PPG signal-quality index validates signal usability for its
specified task; it does not by itself validate BP accuracy. Orphanidou et al.
developed PPG/ECG signal-quality indices for reliable heart-rate recovery, not
a BP-accuracy certificate ([DOI 10.1109/JBHI.2014.2338351](https://doi.org/10.1109/JBHI.2014.2338351)).

## 3. Tail-aware training candidates

### 3.1 Participant-level Huber-CVaR -- Priority 1

Let the loss for participant `s` be the mean of query-level SBP and DBP Huber
losses, with every participant receiving equal weight regardless of query
count:

```text
L_s = mean_q 0.5 * (Huber(SBP_error_q) + Huber(DBP_error_q))
```

For a 70th-percentile threshold, CVaR targets the highest-loss 30%:

```text
J_CVaR = eta + 1 / (1 - 0.70) * mean_s relu(L_s - eta)
```

`eta` is optimized jointly or estimated by a sufficiently large
participant-level batch. The loss must be aggregated by participant before
CVaR; event-level top-30% loss would allow participants with many queries to
dominate the objective.

Huber is used inside CVaR to prevent a very small number of corrupt labels or
waveforms from controlling the gradient. Huber alone down-weights very large
errors and may therefore reduce attention to genuine systematic failures;
Huber-CVaR separates these two roles. Huber's robust-estimation basis is given
by [Huber, 1964, DOI 10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732).
Tail/subpopulation risk optimization is supported by distributionally robust
learning work including [Duchi, Hashimoto, and Namkoong, DOI
10.1287/opre.2022.2363](https://doi.org/10.1287/opre.2022.2363), while average
top-k loss has also been studied for regression
([Fan et al., NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6c524f9d5d7027454a783c841250ba71-Abstract.html)).

Required ablations are ordinary mean loss, Huber-only, participant-CVaR with
the ordinary base loss, and Huber-CVaR. The 30% tail fraction must be fixed
before locked-test access.

### 3.2 Source-balanced loss and group DRO -- Priority 2

If the independent MIMIC and VitalDB reports show a meaningful source gap,
first test the smallest source intervention:

```text
J_source_equal = 0.5 * L_MIMIC + 0.5 * L_VitalDB
```

Group DRO may subsequently update source weights toward the currently
worst-performing source. This uses source labels only during training; a
single shared predictor does not require source at inference. Naive group DRO
can overfit small groups, so early stopping and stronger regularization are
required ([Sagawa et al., ICLR 2020](https://openreview.net/forum?id=ryxGuJrFvS)).

Source balancing cannot replace latent-tail optimization: it addresses the two
known database domains, not difficult subgroups within each domain.

### 3.3 BP-range imbalance -- Priority 2, conditional

High- and low-BP events can be under-represented in continuous-label training.
Two relevant generic methods are label/feature distribution smoothing
([Yang et al., ICML 2021](https://proceedings.mlr.press/v139/yang21m.html)) and
Balanced MSE
([Ren et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html)).

The minimum experiment is a meta-train-only smoothed SBP/DBP density weight,
bounded to prevent rare labels from producing unbounded gradients and
normalized to mean one. It changes training only and requires no query BP at
inference.

This is conditional because the current analysis shows that BP variability
and support-to-query drift are more strongly associated with error than
participant mean BP. A range-balanced result must report low, middle, and high
BP intervals and must not hide a material loss in the common central range.

### 3.4 Input-only uncertainty/quality gate -- Priority 1 diagnostic and policy

The gate estimates risk from model disagreement, query-to-support embedding
distance, PPG quality, support-residual dispersion, and calibration horizon.
An ensemble provides a strong general uncertainty baseline and can express
higher uncertainty under distribution shift
([Lakshminarayanan et al., NeurIPS 2017](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)).
SelectiveNet provides a regression-capable learned reject-option precedent
([Geifman and El-Yaniv, ICML 2019](https://proceedings.mlr.press/v97/geifman19a)).

The gate has two valid uses:

- route the event to a tail expert; or
- request a new cuff measurement and resume prediction afterward.

It is not valid to report only the retained 70% without an action and without
coverage. Selective regression can improve aggregate retained risk while
worsening particular subgroup outcomes, so coverage must be reported by
source and participant subgroup
([Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html)).

If conformal intervals are added, calibration must respect participant
clustering. Naive event-wise conformal calibration does not justify coverage
under many correlated queries from the same participant. Conformalized
quantile regression is a relevant interval method, subject to its exchangeability
conditions
([Romano, Patterson, and Candes, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)).

### 3.5 Input-gated stable/tail mixture of experts -- Priority 3

Use two experts rather than a broad architecture sweep:

```text
y_hat = (1 - w) * y_hat_stable + w * y_hat_tail
w = sigmoid(g(input_only_risk_features))
```

- `stable` is the selected base model;
- `tail` is trained with cross-fitted high-risk meta-train participants or
  events up-weighted;
- the gate sees only deployment-visible features from Section 2.1.

Mixture-of-experts models use an input-dependent gating network to allocate
examples to local experts
([Jacobs et al., 1991, DOI 10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79)).
This candidate is promoted only if the preceding risk detector has useful
held-out AUPRC and calibration. Otherwise the expert router is likely to learn
source prevalence or demographic shortcuts rather than the actual error tail.

## 4. Second-cuff and multi-anchor calibration

A participant with distinct early and late BP regimes may benefit from a later
calibration anchor, but the second anchor cannot be chosen retrospectively by
searching for the largest true BP difference.

### 4.1 Valid schedules

1. **Fixed schedule:** request a second cuff after a prespecified elapsed time
   or event count.
2. **Input-only trigger:** request a cuff when a frozen uncertainty,
   query-to-support distance, quality, or horizon threshold is crossed.

At trigger event `t`, the reference BP becomes available only because a cuff is
requested. Event `t` is then a support/calibration event and must not also be
counted as a query prediction. The new BP may influence predictions only from
event `t+1` onward.

Selecting the second event after examining its BP, query error, or the whole
future sequence is an oracle analysis and is not a valid calibration policy.

### 4.2 Separate protocol branch

Maintenance calibration changes the number and timing of cuff events and is
therefore reported separately from fixed-first K=1/2/3/5. Strategies must be
compared at equal cuff burden and on an aligned post-trigger query set. Required
outcomes are:

- participant-macro MAE before and after recalibration;
- cuffs requested per participant;
- time or events to recalibration;
- proportion never requesting a second cuff;
- overall, MIMIC, and VitalDB results;
- error versus cuff-burden and coverage curves.

A small longitudinal study found that calibration stability and recalibration
needs can be model-specific; its toe-PAT finding in ten participants does not
establish a schedule for PPG-only M0
([Yavarimanesh et al., IEEE TBME 2022, DOI
10.1109/TBME.2021.3136492](https://doi.org/10.1109/TBME.2021.3136492)).

## 5. Fourth-round experiment priority

All broad Phase-6B comparisons are exploratory single-seed screens. A
candidate is repeated across seeds only after it shows a meaningful
development gain. Each job changes one main factor relative to the selected
base model.

| Priority | Candidate | Single primary change | Gate to continue |
|---:|---|---|---|
| 0 | Existing-prediction tail report | No retraining; overall/source reports and oracle 70/30 diagnostic | Quantify tail concentration and source gap |
| 1 | Participant-CVaR | CVaR at the fixed 70th percentile | Lower worst-30% participant MAE without material overall loss |
| 1 | Huber-CVaR | Huber event loss inside participant-CVaR | Improve tail over CVaR or improve stability without losing its gain |
| 1 | Input-only risk gate | Add frozen PPG/support/uncertainty risk score | Useful held-out AUPRC/calibration and favorable risk--coverage curve |
| 2 | Source-equal or source group DRO | Equalize or robustly weight MIMIC/VitalDB | Improve worst source without material other-source loss |
| 2 | BP distribution smoothing/balanced loss | Meta-train-only continuous-label weighting | Improve both BP extremes without material central-range loss |
| 2 | Fixed second-cuff schedule | One prespecified maintenance cuff | Benefit after the second cuff at an explicit cuff burden |
| 3 | Stable/tail two-expert model | Cross-fitted tail expert plus input-only gate | Improve flagged tail and full-coverage aggregate |
| 3 | Risk-triggered second cuff | Input-only cuff trigger | Better error--cuff-burden trade-off than fixed schedule |

Combinations are considered only after isolated effects are known. In
particular, do not start with CVaR + group DRO + demographic conditioning +
quality gate in one run, because the source of any gain would be unidentifiable.

## 6. Evaluation and reporting matrix

Participant-macro metrics remain primary. Event-pooled metrics are secondary
diagnostics. Every accepted result package must contain three cohort views:

1. overall PulseDB development result;
2. MIMIC-only result;
3. VitalDB-only result.

Within each view, report:

- all-participant/all-query result;
- oracle worst-30% and oracle retained-70% diagnostics, explicitly labelled;
- deployable predicted-high-risk and predicted-low-risk results;
- K=1/2/3/5 on the identical event-6-onward common query definition unless a
  maintenance-calibration branch explicitly changes query eligibility;
- SBP, DBP, and their prespecified mean;
- participant-macro MAE with participant-cluster bootstrap confidence
  intervals;
- pooled MAE, R-squared, prediction-minus-reference ME, error SD, and
  cumulative percentages within 5/10/15 mmHg as secondary diagnostics;
- AAMI-style and historical BHS numerical screens with the non-compliance
  disclaimer below;
- for gates: coverage at 100/90/80/70%, risk--coverage AUC, rejected-tail MAE,
  and coverage by source, sex, age, and BP range;
- for recalibration: cuff count and error--cuff-burden curves.

The primary promotion decision uses all-participant performance and the
worst-30% participant loss. A lower retained-70% error alone is insufficient.

## 7. Leakage, exclusion, and fairness controls

- Query BP and query error are evaluation-only outside meta-train supervised
  loss construction.
- No true-error tail membership is available to a deployment gate.
- No filter, threshold, loss weight, BP bin, or recalibration rule is selected
  on locked-test results.
- An event used as a new cuff anchor cannot also be counted as a query.
- A participant with many windows must not receive more weight in the
  participant-CVaR objective merely because of query count.
- Signal-quality exclusions must use thresholds fitted on development data and
  be reported as coverage, not silently removed.
- Exclusion or abstention rates must be audited by source, age, sex, and BP
  range. Difficult physiological ranges must not be removed merely to improve
  a numerical standards screen.
- If a risk score uses a future PPG sequence summary, it is retrospective only;
  a real-time gate may use only current and past PPG.

## 8. AAMI, BHS, ISO, and IEEE claim boundary

The result-table screens are not device validation:

- the AAMI-style column checks only retrospective numerical limits such as
  bias and error SD;
- BHS A/B/C/D uses a historical 5/10/15-mmHg grading protocol
  ([O'Brien et al., 1993, DOI
  10.1097/00004872-199306000-00013](https://doi.org/10.1097/00004872-199306000-00013));
- the AAMI/ESH/ISO collaboration statement defines a complete device
  validation procedure, not a two-number criterion
  ([DOI 10.1097/HJH.0000000000001634](https://doi.org/10.1097/HJH.0000000000001634));
- [ISO 81060-3:2022](https://committee.iso.org/standard/71161.html?browse=tc)
  addresses clinical investigation of continuous automated non-invasive
  sphygmomanometers and is currently marked by ISO as due for revision;
- [IEEE 1708-2025](https://standards.ieee.org/ieee/1708/7031/) is the current
  active IEEE standard for wearable cuffless BP devices as of this plan.

PulseDB uses retrospective ABP-derived references and 120-second pseudo-cuff
events; it is not a prospective investigation of a finished wearable device.
Deleting the observed worst 30%, filtering BP extremes, or reporting retained
coverage cannot establish AAMI, BHS, ISO, or IEEE compliance. Public wording
must remain `AAMI-style numerical screen`, `historical BHS numerical grade`,
and `formal device compliance not established`.

Some personalized PPG papers report numbers that their authors describe as
meeting AAMI/BHS thresholds. For example, Leitner et al. use population
pretraining followed by participant-specific partial-layer transfer learning
and report results with 50 personal five-second samples
([DOI 10.1109/JBHI.2021.3085526](https://doi.org/10.1109/JBHI.2021.3085526)).
Li et al. use LoRA personalization, a pulse-pressure penalty, and a
sampling-rate-robust extension, again with substantially more personal
waveform samples than K=1--5 independent events
([DOI 10.1109/JBHI.2026.3665810](https://doi.org/10.1109/JBHI.2026.3665810)).
These are method precedents and author-reported numerical claims, not evidence
that this PulseDB experiment satisfies a clinical device-validation standard.
Schlesinger et al.'s single-anchor Siamese paper states that its result *almost*
complies with an AAMI recommendation rather than claiming a pass
([DOI 10.1109/ICASSP40776.2020.9053446](https://doi.org/10.1109/ICASSP40776.2020.9053446)).

## 9. Promotion and stop rules

A tail-aware candidate may enter repeated-seed confirmation only if it:

1. improves the prespecified worst-30% participant-macro loss on development;
2. does not materially degrade overall participant-macro MAE;
3. does not obtain its gain solely from one source while substantially harming
   the other;
4. preserves the fixed-first leakage rules or is clearly designated as a
   separate maintenance-calibration protocol;
5. has saved participant/event predictions from which overall, MIMIC, and
   VitalDB results can be rebuilt; and
6. requires no query BP or true error at inference.

If no input-only risk model can predict the tail better than a simple baseline,
do not proceed to a complex mixture-of-experts router. In that case, retain the
best full-coverage base model, report the irreducible development tail, and
prioritize genuinely informative future data: controlled pressure, motion,
device, and repeated-cuff measurements.
