# Round 14: paired confirmation and complete-method results

## Result in one paragraph

Round 14 produced two different conclusions. First, the wider InceptionTime
backbone did **not** confirm the Round-13 architecture gain: across the four
prespecified new seeds its mean improvement over the compact ResNet was only
0.0559 mmHg, below the frozen 0.15-mmHg confirmation threshold, even though
all four Overall gains were positive. Second, the complete
`calibration_relative` method passed its single-seed internal primary gate:
its Overall participant-macro mean MAE was 8.3520 mmHg, compared with
8.5107 mmHg for the matched wider-InceptionTime QGH anchor. The corresponding
gains were 0.1586 mmHg Overall, 0.1253 mmHg in PulseDB MIMIC, and 0.1863 mmHg
in PulseDB VitalDB. This makes `calibration_relative` a candidate for
independent-seed confirmation, not a final model. Meta-validation and the
locked meta-test were not accessed.

## Evaluation boundary

- Calibration budget: `K = 5`, with the first five eligible events used as
  calibration support and every eligible event from event 6 onward used as a
  future query.
- Fit folds: participant-disjoint internal meta-train folds 0--2.
- Early stopping: internal fold 3, with patience 8 and no epoch cap.
- Candidate selection: internal fold 4 only.
- Common evaluation population per run: 628 participants and 96,332 queries.
- Internal PulseDB source strata: 285 MIMIC participants with 74,373 queries
  and 343 VitalDB participants with 21,959 queries.
- Primary endpoint: participant-macro SBP/DBP MAE and their arithmetic mean.
- Overall, MIMIC, and VitalDB rows were recomputed from the same saved
  predictions. MIMIC and VitalDB are internal PulseDB source strata, not
  external validation datasets.

## Line A: paired backbone confirmation

The confirmation decision used only seeds `20260828`--`20260831`. The
Round-13 discovery seed `20260827` appears only in the descriptive five-seed
summary and could not influence the gate.

### Four-new-seed gain over compact ResNet QGH

Positive values favour `inception_time_wide + QGH`.

| Scope | SBP gain, mean ± SD | DBP gain, mean ± SD | Mean-MAE gain, mean ± SD | Positive seeds |
|---|---:|---:|---:|---:|
| Overall | 0.0492 ± 0.0731 | 0.0627 ± 0.0644 | 0.0559 ± 0.0623 | 4/4 |
| MIMIC | 0.0391 ± 0.0857 | 0.0463 ± 0.0802 | 0.0427 ± 0.0577 | 3/4 |
| VitalDB | 0.0575 ± 0.1102 | 0.0763 ± 0.0656 | 0.0669 ± 0.0708 | 4/4 |

The candidate passed the two-source direction check and the 3-of-4 direction
check, but failed the prespecified requirement that the four-new-seed Overall
mean gain be at least 0.15 mmHg. Therefore:

**`inception_time_wide + QGH` did not pass paired confirmation.**

### Five-seed descriptive summary

These values include the discovery seed and are descriptive only.

| Setting | Scope | SBP participant-macro MAE | DBP participant-macro MAE | Mean participant-macro MAE |
|---|---|---:|---:|---:|
| `resnet_small + QGH` | Overall | 11.0290 ± 0.0880 | 6.3364 ± 0.0360 | 8.6827 ± 0.0529 |
| `resnet_small + QGH` | MIMIC | 11.9761 ± 0.1719 | 6.6881 ± 0.0741 | 9.3321 ± 0.1209 |
| `resnet_small + QGH` | VitalDB | 10.2421 ± 0.0980 | 6.0442 ± 0.0411 | 8.1431 ± 0.0463 |
| `inception_time_wide + QGH` | Overall | 10.9598 ± 0.1383 | 6.2406 ± 0.0794 | 8.6002 ± 0.1060 |
| `inception_time_wide + QGH` | MIMIC | 11.8670 ± 0.1989 | 6.5890 ± 0.1353 | 9.2280 ± 0.1642 |
| `inception_time_wide + QGH` | VitalDB | 10.2059 ± 0.0947 | 5.9510 ± 0.0485 | 8.0785 ± 0.0595 |

The separate equal-weight five-seed ensemble was a CPU-only variance
diagnostic. Its Overall SBP/DBP/mean participant-macro MAE was
10.6808/6.1485/8.4146 mmHg for compact ResNet and
10.6035/6.0641/8.3338 mmHg for wider InceptionTime. Because this ensemble was
excluded from the prospective gate, it cannot rescue the failed confirmation.

## Line B: complete calibration-method screen

Line B used seed `20260828` and compared each complete method with the
same-seed `inception_time_wide + QGH` anchor on the identical query set.
`calibration_relative` predicts BP change relative to the calibration state
and uses only causal, input-visible query history. The
`calibration_relative_standards` route adds fixed physical-mmHg robust and
5/10/15-mmHg threshold-oriented losses.

### Participant-macro primary results

| Setting | Scope | N participants | N queries | SBP MAE | DBP MAE | Mean MAE |
|---|---|---:|---:|---:|---:|---:|
| `inception_time_wide_qgh` | Overall | 628 | 96,332 | 10.8106 | 6.2107 | 8.5107 |
| `inception_time_wide_qgh` | MIMIC | 285 | 74,373 | 11.7006 | 6.4938 | 9.0972 |
| `inception_time_wide_qgh` | VitalDB | 343 | 21,959 | 10.0711 | 5.9754 | 8.0233 |
| `calibration_relative` | Overall | 628 | 96,332 | **10.6536** | **6.0505** | **8.3520** |
| `calibration_relative` | MIMIC | 285 | 74,373 | **11.6071** | **6.3366** | **8.9719** |
| `calibration_relative` | VitalDB | 343 | 21,959 | **9.8613** | **5.8127** | **7.8370** |
| `calibration_relative_standards` | Overall | 628 | 96,332 | 10.6580 | 6.1057 | 8.3819 |
| `calibration_relative_standards` | MIMIC | 285 | 74,373 | 11.6038 | 6.3958 | 8.9998 |
| `calibration_relative_standards` | VitalDB | 343 | 21,959 | 9.8721 | 5.8646 | 7.8684 |

### Prespecified gate outcomes

| Candidate | Overall gain | MIMIC gain | VitalDB gain | Primary gate | Tail gate | Decision |
|---|---:|---:|---:|---|---|---|
| `calibration_relative` | 0.1586 | 0.1253 | 0.1863 | PASS | FAIL | Advance to independent-seed confirmation |
| `calibration_relative_standards` | 0.1288 | 0.0974 | 0.1549 | FAIL | FAIL | Do not advance |

The standards-oriented route reduced pooled error standard deviation for both
endpoints, but it missed both the primary gain threshold and the complete
tail-focused threshold-percentage gate. The optional six-group C3 route was
not evaluated in this completed result, and no C3 result is claimed.

## Pooled diagnostic results

The following event-pooled results are secondary diagnostics; participant-
macro MAE above remains the primary endpoint.

| Setting | Scope | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `inception_time_wide_qgh` | Overall | SBP | 12.2843 | 0.4643 | -1.1150 | 16.2979 | 28.72% | 52.22% | 69.11% | FAIL* | Grade D* |
| `inception_time_wide_qgh` | Overall | DBP | 6.7608 | 0.4825 | -1.0502 | 9.3715 | 48.42% | 78.31% | 91.43% | FAIL* | Grade C* |
| `calibration_relative` | Overall | SBP | 12.1891 | 0.4749 | -1.1825 | 16.1300 | 28.86% | 52.42% | 69.44% | FAIL* | Grade D* |
| `calibration_relative` | Overall | DBP | 6.6635 | 0.4929 | -0.8681 | 9.2949 | 49.36% | 78.58% | 91.76% | FAIL* | Grade C* |
| `calibration_relative_standards` | Overall | SBP | 12.3223 | 0.4630 | -2.2905 | 16.1943 | 28.63% | 52.22% | 69.16% | FAIL* | Grade D* |
| `calibration_relative_standards` | Overall | DBP | 6.7363 | 0.4800 | -1.6459 | 9.3090 | 48.94% | 78.57% | 91.52% | FAIL* | Grade C* |
| `inception_time_wide_qgh` | MIMIC | SBP | 12.8910 | 0.4453 | -1.4806 | 16.9653 | 27.33% | 50.21% | 66.75% | FAIL* | Grade D* |
| `inception_time_wide_qgh` | MIMIC | DBP | 6.9643 | 0.4596 | -1.2203 | 9.6120 | 47.30% | 77.17% | 90.74% | FAIL* | Grade C* |
| `calibration_relative` | MIMIC | SBP | 12.8286 | 0.4538 | -1.3796 | 16.8426 | 27.40% | 50.21% | 66.91% | FAIL* | Grade D* |
| `calibration_relative` | MIMIC | DBP | 6.8815 | 0.4686 | -0.8957 | 9.5664 | 48.14% | 77.39% | 91.04% | FAIL* | Grade C* |
| `calibration_relative_standards` | MIMIC | SBP | 13.0071 | 0.4394 | -2.4842 | 16.9398 | 27.02% | 49.79% | 66.48% | FAIL* | Grade D* |
| `calibration_relative_standards` | MIMIC | DBP | 6.9596 | 0.4544 | -1.7267 | 9.5817 | 47.66% | 77.40% | 90.81% | FAIL* | Grade C* |
| `inception_time_wide_qgh` | VitalDB | SBP | 10.2294 | 0.4857 | 0.1233 | 13.7282 | 33.41% | 59.02% | 77.10% | FAIL* | Grade D* |
| `inception_time_wide_qgh` | VitalDB | DBP | 6.0715 | 0.5469 | -0.4740 | 8.4812 | 52.23% | 82.16% | 93.76% | FAIL* | Grade B* |
| `calibration_relative` | VitalDB | SBP | 10.0233 | 0.5080 | -0.5149 | 13.4174 | 33.83% | 59.93% | 78.02% | FAIL* | Grade D* |
| `calibration_relative` | VitalDB | DBP | 5.9253 | 0.5627 | -0.7745 | 8.3092 | 53.52% | 82.59% | 94.22% | FAIL* | Grade B* |
| `calibration_relative_standards` | VitalDB | SBP | 10.0029 | 0.5069 | -1.6346 | 13.3434 | 34.07% | 60.44% | 78.25% | FAIL* | Grade D* |
| `calibration_relative_standards` | VitalDB | DBP | 5.9803 | 0.5542 | -1.3721 | 8.3136 | 53.30% | 82.51% | 93.93% | FAIL* | Grade B* |

\* AAMI and BHS entries are retrospective numerical screens on PulseDB.
They are not formal device-validation or standards-compliance claims. In
particular, no complete setting passes the Overall AAMI-style numerical
screen, and the paired Overall historical BHS grades remain D for SBP and C
for DBP.

## Reproducibility and leakage audit

- All 32 submitted jobs completed with exit code 0, and every stderr file was
  empty.
- All 33 Round-14 result directories were byte-identical between the active
  work area and the durable NAS archive.
- The four exploratory-job stdout/stderr pairs were additionally archived
  before this public report was prepared.
- The source snapshot was immutable and its work/NAS archives had matching
  hashes.
- All available run safety flags were false for locked-test access,
  meta-validation fitting/early stopping/ranking/prediction, query-BP model
  input, future-query input, and source-label input.
- The Line-A confirmation JSON explicitly records
  `meta_validation_accessed=false` and `locked_test_accessed=false`. The
  Line-B selection JSON records that meta-validation was not used for
  candidate ranking and that the locked test was not accessed; the separate
  audit of all available Line-B run metadata found no meta-validation access
  in fitting, early stopping, ranking, or prediction.

Only aggregate tables and sanitized provenance hashes are published. Raw
signals, participant identifiers, event-level predictions, checkpoints,
private tail membership, server paths, and execution logs are excluded.

## Decision and next step

1. Retain the compact ResNet as the architecture reference for QGH-only
   comparisons because the wider InceptionTime gain did not pass paired
   confirmation.
2. Advance the **complete** `calibration_relative` method to a new paired
   independent-seed confirmation. This advances the method as a whole; it does
   not retroactively confirm the wider InceptionTime backbone in isolation.
3. Do not advance `calibration_relative_standards` and do not tune its loss
   weights after observing this fold.
4. Continue to quarantine meta-validation and the locked meta-test until the
   complete-method confirmation policy is frozen and passed.
5. Do not claim AAMI, BHS, ISO, IEEE, clinical, or device compliance from these
   retrospective PulseDB diagnostics.

## Machine-readable aggregates

- [Line-A confirmation tables](../results/round14/confirmation/)
- [Line-B complete-method tables](../results/round14/complete_methods/)
- [Prospective Round-14 plan](ROUND14_CONFIRMATION_AND_STANDARDS_PLAN.md)
