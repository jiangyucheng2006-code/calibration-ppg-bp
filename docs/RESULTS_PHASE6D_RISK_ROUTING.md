# Phase-6D risk identification and specialist routing results

## Scope and integrity

Phase-6D is a single-seed, development-only K=5 experiment. The first five
chronological pseudo-cuff/reference-BP events are support, and event 6 onward
is the common query set. All results use 697 meta-validation participants and
103,564 queries: 316 participants/80,874 queries from MIMIC and 381/22,690
from VitalDB. MIMIC and VitalDB are internal PulseDB source strata, not
independent external validation datasets.

Slurm jobs 880--890 all completed with exit code `0:0`; every stderr file is
empty. The final prediction artifact contains 18 settings x 103,564 queries,
has no duplicate setting/query keys or non-finite predictions, and matches the
same targets. Work and NAS report directories are byte-identical. Every run
records `locked_test_accessed=false`.

## Leakage-safe difficult-label construction

Five participant-disjoint, source-stratified folds inside meta-train produced
Quality Gate + Huber out-of-fold predictions for all 3,143 eligible training
participants. Exactly the worst 30% within each source were labelled difficult:
429/1,427 MIMIC and 515/1,716 VitalDB participants. The risk classifier was
then fitted from 22 deployment-visible PPG, support-BP, model-prediction, and
event-horizon features. Query BP, query error, source, participant identity,
future PPG, and locked-test information were excluded.

## Can the difficult 30% be identified?

The frozen meta-train participant threshold predicted 217/697 (31.1%)
meta-validation participants as high risk. It correctly identified 102/210
true difficult participants but also selected 115 false positives.

| Scope | AUPRC | AUROC | Precision | Recall | Specificity | F1 | Balanced accuracy | Predicted high risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 0.457 | 0.659 | 0.470 | 0.486 | 0.764 | 0.478 | 0.625 | 31.1% |
| MIMIC | 0.479 | 0.699 | 0.451 | 0.537 | 0.719 | 0.490 | 0.628 | 35.8% |
| VitalDB | 0.459 | 0.619 | 0.490 | 0.443 | 0.801 | 0.466 | 0.622 | 27.3% |

The Overall AUPRC is above the 30.1% difficult-class prevalence, and predicted
high-risk participants have mean MAE 10.140 versus 7.737 mmHg in the predicted
low-risk group. Thus the classifier contains real ranking information, but it
does not precisely recover the difficult tail: 22.5% of the predicted-low
group still belongs to the true worst 30%.

## Do difficult-case specialists help?

The fixed evaluation-only true tail contains 210 participants. This tail is
used only after prediction to diagnose expert value and never to route a
deployable prediction.

| Specialist | Overall tail delta | MIMIC tail delta | VitalDB tail delta | Interpretation |
|---|---:|---:|---:|---|
| Quality Gate + Huber, difficult weight 2x | -0.170 | -0.263 | -0.093 | Improves both sources |
| Quality Gate + Huber, difficult weight 4x | -0.121 | -0.279 | +0.010 | Small VitalDB loss |
| Quality Gate + Huber, difficult-only | +0.141 | +0.004 | +0.254 | Worse; over-specialization |

Values are expert minus general-model participant-macro mean MAE in mmHg;
negative values favour the expert. Moderate 2x weighting is the only expert
that improves the fixed difficult tail in both sources. Fourfold weighting is
too aggressive, and difficult-only training loses useful information from the
full training population.

## Complete-pipeline result

Participant-macro MAE is the primary outcome.

| Setting | Scope | SBP MAE | DBP MAE | Mean MAE | Delta vs general |
|---|---|---:|---:|---:|---:|
| General Quality Gate + Huber | Overall | 10.803 | 6.168 | 8.485 | 0.000 |
| 2x specialist standalone | Overall | 10.776 | 6.156 | 8.466 | -0.019 |
| 2x specialist, event hard route | Overall | 10.779 | 6.184 | 8.481 | -0.004 |
| **2x specialist, event soft fusion** | **Overall** | **10.717** | **6.131** | **8.424** | **-0.061** |
| General Quality Gate + Huber | MIMIC | 11.576 | 6.573 | 9.075 | 0.000 |
| 2x specialist, event hard route | MIMIC | 11.509 | 6.592 | 9.050 | -0.025 |
| **2x specialist, event soft fusion** | **MIMIC** | **11.457** | **6.515** | **8.986** | **-0.089** |
| General Quality Gate + Huber | VitalDB | 10.161 | 5.831 | 7.996 | 0.000 |
| 2x specialist, event hard route | VitalDB | 10.173 | 5.846 | 8.010 | +0.014 |
| **2x specialist, event soft fusion** | **VitalDB** | **10.104** | **5.812** | **7.958** | **-0.038** |

The binary hard route does not pass the promotion gate: its Overall change is
negligible and VitalDB worsens. The continuous risk-weighted soft fusion is the
best Phase-6D setting and improves Overall, MIMIC, and VitalDB, but the absolute
gain is small (0.061 mmHg, approximately 0.72%).

A prespecified-style exploratory participant-cluster paired bootstrap with
20,000 repetitions gives candidate-minus-general differences:

| Scope | Mean delta | Exploratory 95% interval |
|---|---:|---:|
| Overall | -0.061 | [-0.097, -0.026] |
| MIMIC | -0.089 | [-0.156, -0.027] |
| VitalDB | -0.038 | [-0.075, -0.002] |

These are unadjusted development-set intervals after candidate screening, not
confirmatory evidence. Independent training seeds are required.

## Requested pooled diagnostic table for the winning route

| Scope | BP | MAE | R² | ME | STD | <=5 | <=10 | <=15 | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Overall | SBP | 11.597 | 0.518 | -1.214 | 15.241 | 29.7% | 54.0% | 71.5% | FAIL* | Grade D* |
| Overall | DBP | 6.808 | 0.487 | -0.621 | 9.410 | 48.5% | 78.1% | 91.0% | FAIL* | Grade C* |
| MIMIC | SBP | 11.995 | 0.513 | -1.473 | 15.708 | 28.7% | 52.5% | 69.9% | FAIL* | Grade D* |
| MIMIC | DBP | 7.074 | 0.471 | -0.755 | 9.788 | 47.1% | 76.8% | 89.9% | FAIL* | Grade C* |
| VitalDB | SBP | 10.178 | 0.509 | -0.293 | 13.405 | 33.1% | 59.1% | 77.1% | FAIL* | Grade D* |
| VitalDB | DBP | 5.859 | 0.555 | -0.141 | 7.897 | 53.5% | 82.7% | 94.6% | PASS* | Grade B* |

The asterisks denote retrospective numerical screens only. PulseDB does not
meet the complete population, reference, measurement, and protocol conditions
needed to claim formal AAMI/BHS or device-validation compliance.

## Decision

Phase-6D partially succeeds. The model can rank difficult cases better than
chance, and moderate difficult-case weighting provides a useful expert.
However, the requested binary 70%/30% hard-routing system is not accurate or
beneficial enough to promote. The continuous risk-weighted soft fusion is a
provisional candidate for independent-seed confirmation. It must not replace
the Quality Gate + Huber reference or trigger locked-test access until the
  small gain reproduces across seeds and both source strata.
