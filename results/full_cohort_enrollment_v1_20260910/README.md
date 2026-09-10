# Full-cohort subject-disjoint enrollment: completed results

Reviewed on 11 September 2026. Protocol: `full-cohort-enrollment-v1`.
Run: `full-cohort-enrollment-v1_20260910-150200`.

## Main finding

Both prespecified methods completed on all 762 eligible final-evaluation
participants and the same 78,237 query windows. Personal LoRA plus reference
memory reduced participant-macro SBP/DBP MAE from **3.8556/2.1661** to
**3.2081/1.8336 mmHg**. Mean MAE fell from **3.0108 to 2.5209 mmHg**,
a paired reduction of **0.4900 mmHg (16.27%)**. Both PulseDB source strata
improved; the result is not restricted to the validation cohort.

The shared population model did not train on any of these 762 people.
Each new person subsequently provided approximately 90% of their eligible
windows for a new personal adapter and reference bank. The remaining windows
were scored only after both prediction sets were frozen. This is substantial
labelled enrollment, not calibration-free prediction or 1–5 cuff calibration.

## Execution and cohort

Jobs 1764–1772 all finished `COMPLETED`, exit `0:0`. The final scorer finished
at 2026-09-10 16:49:02 UTC / 11 September 00:49:02 China time. The user queue
was empty at the live inspection at 17:10:49 UTC / 01:10:49 China time.

| Stage | Duration |
|---|---:|
| Full-source materialization | 47 min 27 s |
| Preparation and cohort audit | 5 min 13 s |
| Shared population fit | 6 h 5 min 16 s |
| Personal validation, two parallel shards | 1 h 25 min 23 s / 1 h 25 min 14 s |
| Validation scoring | 11 s |
| Personal final evaluation, two parallel shards | 1 h 23 min 22 s / 1 h 27 min 35 s |
| Final scoring | 22 s |

Population fitting selected epoch 5 and stopped after epoch 13 with patience 8.
This was one population fit followed by many individual profile fits, not
multiple population backbones. There was no new fit or resubmission during
this review.

| Role | MIMIC people | VitalDB people | Total people | Labelled fit/registration windows | Query windows |
|---|---:|---:|---:|---:|---:|
| Population training | 1,694 | 2,056 | 3,750 | 3,672,008 | — |
| New-person validation | 344 | 419 | 763 | 691,929 | 76,909 |
| New-person final evaluation | 343 | 419 | 762 | 703,869 | 78,237 |

All 5,361 original identities and 5,245,454 indexed windows were considered.
The analytic cohort contains 5,275 people. Eighty-three people lacked enough
separable history for the personal roles; three were quarantined for
cross-outer content/interval provenance. A total of 22,502 rows were excluded
or collapsed under the documented audit. No high-error person was removed.
No official-membership intersection, 400-window cap or subject cap was used.

## Primary final results: equal weight per participant

Each person's window MAE is calculated first; participants then have equal
weight. Mean MAE is the mean of SBP and DBP participant-macro MAE.

| Scope | Participants | Query windows | Method | SBP MAE | DBP MAE | Mean MAE |
|---|---:|---:|---|---:|---:|---:|
| Overall | 762 | 78,237 | Personal LoRA | 3.8556 | 2.1661 | 3.0108 |
| Overall | 762 | 78,237 | Personal LoRA + memory | **3.2081** | **1.8336** | **2.5209** |
| MIMIC | 343 | 54,664 | Personal LoRA | 3.7949 | 2.0948 | 2.9448 |
| MIMIC | 343 | 54,664 | Personal LoRA + memory | **3.2322** | **1.7896** | **2.5109** |
| VitalDB | 419 | 23,573 | Personal LoRA | 3.9053 | 2.2244 | 3.0649 |
| VitalDB | 419 | 23,573 | Personal LoRA + memory | **3.1884** | **1.8695** | **2.5290** |

[Exact primary values](test/participant_macro.csv).

| Scope | Mean-MAE reduction | Relative reduction | Saved 95% paired bootstrap interval |
|---|---:|---:|---:|
| Overall, primary contrast | 0.4900 | 16.27% | [0.4513, 0.5320] |
| MIMIC, exploratory | 0.4339 | 14.73% | [0.3764, 0.5022] |
| VitalDB, exploratory | 0.5359 | 17.49% | [0.4860, 0.5903] |

By participant mean MAE, **626/762 improve (82.15%), 97 are unchanged, and
39 worsen**. Positive average benefit is not a guarantee for every wearer.
No cause is inferred for the unchanged profiles from this count alone.

Intervals resample paired participants within each source, 2,000 replicates,
seed 20260909. They are conditional on this fitted model, not training-seed
uncertainty; secondary intervals are pointwise exploratory summaries, not
multiple-comparison-corrected findings. See the [paired intervals](test/paired_participant_intervals.csv)
and [uncertainty contract](test/uncertainty_contract.json).

## Requested diagnostic tables: pooled windows, not participant-macro

The following tables weight each query window equally. Their MAE therefore
differs from the primary table above because personal query counts vary.
Do not combine a participant-macro MAE with pooled R²/ME/STD in one unlabeled
row. Error is prediction minus reference; STD is sample SD of these signed
errors, not the SD across training seeds or across participant MAEs.
All error units are mmHg; threshold columns are cumulative percentages.

### Overall

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| LoRA | SBP | 3.4251 | 0.9409 | 0.0245 | 5.2910 | 79.28% | 94.67% | 98.13% | PASS* | A* |
| LoRA | DBP | 1.8949 | 0.9170 | -0.1212 | 3.6509 | 93.62% | 98.57% | 99.37% | PASS* | A* |
| LoRA + memory | SBP | 2.7235 | 0.9570 | -0.0559 | 4.5117 | 85.61% | 96.23% | 98.58% | PASS* | A* |
| LoRA + memory | DBP | 1.5091 | 0.9377 | -0.0970 | 3.1626 | 95.24% | 98.83% | 99.50% | PASS* | A* |

### MIMIC

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| LoRA | SBP | 3.4744 | 0.9410 | 0.0763 | 5.3904 | 79.01% | 94.36% | 97.93% | PASS* | A* |
| LoRA | DBP | 1.9225 | 0.9154 | -0.1430 | 3.7870 | 93.49% | 98.30% | 99.18% | PASS* | A* |
| LoRA + memory | SBP | 2.7968 | 0.9568 | -0.0314 | 4.6120 | 85.13% | 95.91% | 98.41% | PASS* | A* |
| LoRA + memory | DBP | 1.5365 | 0.9386 | -0.0967 | 3.2254 | 94.92% | 98.62% | 99.36% | PASS* | A* |

### VitalDB

| Setting | BP | MAE | R² | ME | STD | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg | AAMI | BHS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| LoRA | SBP | 3.3107 | 0.9300 | -0.0955 | 5.0510 | 79.90% | 95.39% | 98.59% | PASS* | A* |
| LoRA | DBP | 1.8309 | 0.9215 | -0.0706 | 3.3134 | 93.91% | 99.19% | 99.82% | PASS* | A* |
| LoRA + memory | SBP | 2.5535 | 0.9500 | -0.1126 | 4.2694 | 86.72% | 96.98% | 98.97% | PASS* | A* |
| LoRA + memory | DBP | 1.4455 | 0.9351 | -0.0977 | 3.0120 | 95.99% | 99.32% | 99.82% | PASS* | A* |

\* PASS/A denotes the prespecified retrospective numerical screen only.
The code checks absolute ME ≤5 and error STD ≤8 for the AAMI-style column,
and the historical BHS cumulative percentages for the grade. These numbers
do not establish compliance with a complete clinical device-validation
protocol, including its reference, sampling and repeated-measure requirements.
See the [AAMI/ESH/ISO collaboration statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC5796427/)
and [original BHS protocol](https://bihs.org.uk/wp-content/uploads/2017/08/BHS_Protocol_Revision_J_Hypertens_1993_df.pdf).
The exact unrounded [diagnostic CSV](test/diagnostic_tables.csv) and
[original server-formatted report](test/RESULT_TABLES.md) are preserved.

## Validation versus final evaluation

| Role | Participants | LoRA SBP / DBP MAE | LoRA + memory SBP / DBP MAE | Mean-MAE reduction |
|---|---:|---:|---:|---:|
| Validation | 763 | 4.0121 / 2.2543 | 3.3595 / 1.8714 | 0.5177 |
| Final evaluation | 762 | 3.8556 / 2.1661 | 3.2081 / 1.8336 | 0.4900 |

Both are participant-macro values on different people. They show consistent
direction under this protocol; a lower final point estimate does not mean
another training improvement occurred between the two cohorts. Both candidates
were retained before final scoring, regardless of validation ordering.

## Interpretation and next step

The evidence supports retaining personal LoRA plus reference memory as the
candidate for further testing in the substantial-history, new-user setting.
It does not establish few-label usability, continual improvement, cross-day
stability, wrist-device transfer, independent external validity or clinical
certification. The split is custom, not unchanged official CalBased/CalFree.
Prior project experiments on PulseDB informed method choices, so these are
exploratory results rather than a pristine independent confirmation.

The current full-cohort batch contains exactly two methods. The richer
200-person ablation results are separate and must not be presented as
full-cohort ablations. This review did not train a new baseline or module.

The user requests seeing these results before further training. Keep the
existing 90% reference unchanged. The already discussed lower-budget,
common-query comparison and later chronological archive-updating study remain
next-step proposals; do not submit them automatically from this review or
choose their settings from final-test errors.

## Why retain the subject-disjoint population split

For the intended scenario of a new wearer joining after population training,
this is the more directly relevant evaluation framework. The distinction is
the scope of learned information, not simply whether some personal fitting
occurs at all:

| Evaluation framework | Target person contributes to population fitting | Permitted personal information before scoring | Supported question |
|---|---|---|---|
| Historical seen-user studies | Yes | Declared training-role labels and persistent personal state | Other-window accuracy for already registered training participants |
| Current new-user enrollment | No | Separate labelled registration subset; fresh personal adapter and reference bank | Whether an excluded person can be enrolled after the shared model is frozen |
| Earlier K-shot study | No | Only the specified 1/2/3/5 prior reference events | Adaptation from a small number of prior reference events |

Retain the current outer subject-disjoint framework as the preferred basis for
further new-user enrollment research, while preserving seen-user studies as
separate benchmark evidence. This is a recommendation about the research
question, not a retrospective change to any split or a new claim of few-shot
performance. The present benefit over paired LoRA is observed in both source
strata, but its magnitude cannot be directly ranked against earlier studies
with different people, label budgets and queries.

The personal registration subset is deliberately used for fitting; the final
query subset is not. Report these as separate roles instead of saying that
the complete target-user cohort is never used for any training. For stronger
practical claims, establish lower-budget performance and earlier-history/
later-query prediction under separately predefined protocols. Current scores
do not prove either property, and the extensive prior use of PulseDB limits
independent confirmatory interpretation.

## Verification and storage

Independent arithmetic on the saved scored predictions reproduced primary
MAEs and pooled diagnostic columns to a maximum absolute difference of
1.42e-14. The two methods have identical query keys and reference targets,
finite predictions and zero duplicate query keys. Source counts sum to
Overall. The scorer receipt states predictions were frozen before target
access and that scoring provides no training feedback. Both personal shards
completed 381/381 profiles with saved/reloaded prediction parity and frozen
shared-state checks.

Seven small final report/receipt files were copied locally; their hashes match
both work and NAS. Raw waveforms, checkpoints and personal records were not
downloaded. See [verification receipt](verification_receipt.json),
[review limits](REVIEW_NOTES.md), and [final evaluation receipt](test/evaluation_receipt.json).

Server result directory:
`/home/jiangyu.cheng/work/ppg_bp/outputs/full-cohort-enrollment-v1_20260910-150200/test_score`.
The same relative output directory is archived under
`/home/jiangyu.cheng/nas/ppg_bp/outputs/`.
The original 01:10 China-time review updated local reports only; its saved
receipt therefore records no public push. A later publication update on
11 September adds these aggregate artifacts to the existing research branch.
Neither operation changes frozen protocols, submits new jobs, or writes
global memory. The original verification receipt remains unchanged.
