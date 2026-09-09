# Verified project status

Last updated: 2026-09-09.

## Results and collaborator report prepared for publication

Both completed result batches now have dedicated reports and three-source
tables: [official CalBased](../results/official_calbased_v1_20260909/README.md)
and [30-person post-training enrollment](../results/post_enrollment_30_v1_20260909/README.md).
[The Chinese method report](PERSONAL_LORA_MEMORY_METHOD_ZH.md) and its
[editable Word version](个人LoRA与参考记忆血压估计研究说明.docx) explain the actual
ResNet/feature-LoRA implementation, fixed reference memory, enrollment and the
proposed cuff-paired wrist study. The root README points to these current results.

This is a publication/documentation update: no new fit, participant filtering,
prediction changes or human-data collection. Longitudinal history updates and
additional matched controls remain proposed, not implemented or verified here.

## Thirty-person post-training enrollment — completed, verified 15:09 CST

Jobs 1719–1726 all COMPLETED 0:0; final evaluation ended at 14:45:01 CST.
The queue is empty. Population inner training ran 40 epochs, selected epoch 32,
then a fresh final model refit all permitted population rows for 32 epochs.
No selected enrollment subject participated in either population fit.

| Method | Overall SBP / DBP MAE | MIMIC SBP / DBP MAE | VitalDB SBP / DBP MAE |
|---|---:|---:|---:|
| Personal BP mean | 11.5854 / 6.0402 | 10.5278 / 5.2594 | 12.6429 / 6.8209 |
| Shared + own BP anchor | 10.8464 / 5.7255 | 10.7799 / 5.5361 | 10.9128 / 5.9150 |
| New-person LoRA | 4.0749 / 2.2775 | 3.8486 / 2.2589 | 4.3011 / 2.2961 |
| New-person LoRA + memory | 3.1515 / 1.7466 | 3.0666 / 1.7583 | 3.2364 / 1.7349 |

All 30 people/1,200 queries retained, 15 people per source, 360 labelled
registration windows/person. Compared with new-person LoRA, memory improves
Overall mean MAE by 0.7271 mmHg (22.89%); MIMIC/VitalDB by 21.00%/24.65%.
It improves mean MAE in 29/30 people (one worsens by 0.0847); SBP improves in
28/30, DBP in 30/30. No hypothesis tests/CIs or independent re-training here.
LoRA and LoRA+memory both meet the saved numerical AAMI/BHS-A screens in all
scopes; these are not clinical-device validation or certification.

All profiles actually trained: selected personal epochs 7–726, median257;
2048 parameters/person. All30 shared-state-freeze and save/reload checks pass.
Recomputed all primary MAEs from saved scored predictions without touching
source BP labels. Twenty-two aggregate/receipt files match work/NAS/local hashes;
three inspected source modules match the actual server runtime hashes.

[Full report, requested diagnostic tables and verification](../results/post_enrollment_30_v1_20260909/README.md).
The remaining2476-person LoRA benchmark is 3.7752/2.0810 Overall; not a paired
comparator to the new30. This supports accumulated-data post-training enrollment,
not few-cuff, chronological/long-term stability or external clinical validity.
No new training, protocol edits or GitHub push in this status-check turn.
The earlier running snapshot below is retained as execution history.

## Thirty-person post-training enrollment — partition verified; population fit running

The user authorized a new whole-subject-excluded onboarding experiment on
2026-09-09: train a fresh population model without 30 randomly sampled people,
then create their personal LoRA and reference memory from their own registration
records. [Frozen design and label budgets](PLAN_POST_ENROLLMENT_30_V1.md).
This retains the earlier official benchmark unchanged and is separately named
`post-enrollment-30-v1`, not an exact-official reproduction or a few-cuff study.

The user confirms they can supply paired cuff BP and PPG for future real-world
personal archives. The present work uses public data only, with no recruitment
or clinical-readiness claim. Four matched controls, shared-state freezing and
serialized-profile replay are required before interpreting new-user accuracy.
Implementation is complete and the finite batch is submitted. Runtime code
revision: `5ee29e7`, based on `8711792`. The immutable server snapshot is
`/home/jiangyu.cheng/work/ppg_bp/code/post_enrollment_5ee29e7`.

- Smoke 1719: COMPLETED, exit 0:0, 31 seconds. All 39 server tests passed,
  including end-to-end synthetic enrollment/scoring and the unchanged official
  protocol tests. Real RTX 5080 personal forward/backward computation passed.
- First smoke 1718 failed before formal fitting: a partial-state-load check
  incorrectly assumed BatchNorm missing-key behavior, and a synthetic gradient
  test used an untrained zero-output head. Both were corrected and re-tested.
  The failed log is retained; no formal fit used that version.
- Prepare 1720 COMPLETED 0:0 in 36 seconds (CPU only). Its materialized split
  has 2,476 population people (1,198 MIMIC / 1,278 VitalDB), 891,360 labelled
  registration/training rows and 99,040 test inputs; the new cohort has exactly
  30 people (15/15), 10,800 registration rows and 1,200 test inputs.
- Cross-cohort subject/content overlap is zero. The 30-person registration/test
  content overlap is also zero. The remaining population retains the 35
  already disclosed source-native TRAIN/TEST duplicate-content pairs.
- Population inner job 1721 is RUNNING on hpc-2/RTX 5080. Its live receipt
  confirms 2,476 fitted people, 792,320 inner-fit rows, no enrollment-label
  access and no old checkpoint. Subsequent jobs are PENDING (Dependency).
- 1721 population inner fit → 1722 fresh selected-epoch full refit →
  1723 frozen remaining-population benchmark → 1724/1725 personal profiles on
  RTX 5080/RTX 5070 Ti respectively → 1726 final 30-person scoring.
- Work batch: `/home/jiangyu.cheng/work/ppg_bp/outputs/post-enrollment-30-v1_20260909-102000`.
  Submission record and per-stage outputs/logs are archived under the matching
  `/home/jiangyu.cheng/nas/ppg_bp/outputs/` batch and `nas/ppg_bp/logs/`.
- No accuracy result is available yet. No result-driven candidate changes,
  raw-data deletion, human-data collection, or GitHub push occurred this turn.

## Official CalBased — all six candidates scored, verified 08:39 CST

The recovered batch completed normally. Slurm jobs 1700–1717 all report
`COMPLETED`, exit `0:0`; the account queue is empty. Final evaluation 1717
finished at **2026-09-09 06:42:21 CST**. This supersedes the active-fitting
snapshot below, without changing the frozen protocol.

- Inner LoRA trained 54 epochs, selected epoch 46, then stopped after eight
  non-improving epochs. Final LoRA was freshly fitted for 46 epochs on all
  902,160 official training windows. The three OOF fits completed 25 epochs each.
- All six candidates were frozen before the evaluator joined official test
  targets. The receipt reports 2,506 people / 100,240 test windows, no removed
  test rows, no test-based model selection, and no training feedback generated.
- Participant-macro SBP/DBP MAE: LoRA **3.8207/2.0994** Overall,
  **4.1920/2.2802** MIMIC, **3.4724/1.9298** VitalDB. Fixed personal memory
  gives **3.4210/1.8858**, **3.8073/2.0699**, **3.0587/1.7132**, respectively.
- v1, E1, scalar trust and BP-specific trust each genuinely trained eight
  internal-selection epochs (6,256 optimizer steps), but none beat its epoch-0
  fixed-memory initialization. All selected epoch 0. Their final reported
  scores reproduce the fixed-memory result to four decimals; this is not
  evidence that four learned additions improved it. v1/E1 final refits correctly
  have zero optimizer steps; trust heads remain frozen at the selected state.
- All reported pooled SBP/DBP AAMI numerical screens are PASS and BHS screens
  Grade A in all three scopes. These are retrospective numerical diagnostics,
  **not clinical certification**. This single-seed result is not a significance
  or unseen-user generalization claim.
- The authorized exact-official source duplicates remain disclosed, including
  35 test windows with matching training PPG content. Internal/OOF splits are
  index-disjoint, not strictly content-independent.
- The ten aggregate/receipt files in `evaluate/` were verified byte-identical
  between work and NAS. [Full result tables](../results/official_calbased_v1_20260909/RESULT_TABLES.md).

This inspection did not change code, data, configuration, checkpoints or the
server queue. No new jobs or GitHub push were performed. Further method changes
must not use this already-opened official test as an untouched selection set.


## Official CalBased — exact membership retained; formal fitting active, 23:41 CST

The user explicitly chose to **retain all official rows, including source-native
duplicates**, without a custom deduplicated split or duplicate-excluded score.
The six methods, 360/40 official budgets and existing inner assignments remain
unchanged. This supersedes the earlier pending-decision status below.

- Runtime `8592a1bf68be`: all 317 source/test/script/plan Git blobs verified on
  the immutable server snapshot. The exception is pinned to the independently
  source-verified duplicate audit, not a blanket bypass. New copies, repeated
  row IDs, invalid official membership and target-access violations still fail.
- Smoke 1700 COMPLETED 0:0: 111 tests, no skips, real RTX5080 forward/backward.
  The original shared memory-preparation suite also passed all 30 tests.
- CPU recovery 1701 COMPLETED 0:0 in 3:21. All 64 PPG shards have been copied,
  file- and window-rehashed; exact official membership verification passed.
  Inner, all three OOF and final full-cohort runtime/memory audits passed.
  The old failed store and all raw masters are unchanged. No full raw recopy.
- New store: `~/work/ppg_bp/data/processed/pulsedb-official-calbased-v1_20260908-234500`.
  The suffix is a run identifier, not a claim about the exact submission time.
  No new official model scores yet. The finite chain 1702–1717 was submitted
  after both preparation and smoke succeeded. 1702 is actively fitting inner
  LoRA on RTX5080 (801,920 fit windows); 1703 fits OOF0 on RTX5070Ti (534,480
  fit windows). Both cover 2,506 people and record test-target access as false.
  1704–1717 are PENDING Dependency. [Stage receipt](OFFICIAL_CALBASED_EXECUTION_20260908.md).
- Reports will explicitly disclose 35 cross-official duplicate test windows
  and the unchanged internal/OOF source copies. They are index-disjoint, not
  strictly content-independent. All test targets remain sealed until scoring.

Batch: `~/work/ppg_bp/outputs/official-calbased-v1_20260908-234500`.
Its `submission.tsv` is byte-identical to the NAS archive. Work-area logs are
`~/work/ppg_bp/logs/official_calbased_<jobid>.log`. No new methods, repeated
seed screens or test-feedback optimization were added. This server-side finite
chain does not require the desktop to remain connected.

## Official CalBased — source duplicates confirmed, 23:18 CST

**The batch failed during preparation; none of the six formal models trained.**
This supersedes the 15:24 submission snapshot below. Own queue is empty.

- Smoke 1673 and metadata preflight 1675 completed. Hot staging 1674 also
  completed: all 2,506 required MAT files / 461.55 GiB were copied and verified.
  No repeat download or full raw copy is needed.
- Materialization 1676 failed with exit 1:0 after 17:36 at 17:10:18 CST.
  Jobs 1677–1692 were automatically cancelled before starting. All 64 signal
  shards exist, but the processed store is correctly marked `failed`, not ready.
- Full-cohort float32 content checks found 35 train–test duplicate groups,
  affecting 35 / 100,240 test windows (0.0349%), all MIMIC and four test people.
  Across both roles there are 154 duplicate pairs: 117 within TRAIN, two
  within TEST and 35 cross-role. All 308 implicated original Info name/index
  entries were independently rechecked directly against the original MAT Info.
- Raw-source verification confirms **all 154 pairs have exactly equal original
  float64 PPG_F, PPG_Raw and T arrays**, despite different record IDs. Five pairs
  also cross subject IDs; one of those is cross-role. Prepared arrays match
  source arrays. This is not a float32 rounding or shard-row indexing error.
- Of the within-TRAIN duplicate groups, 23 cross internal train/validation and
  82 cross encoder-fit folds. Simply bypassing the first checker would leave
  additional affected consumers and biased internal selection.
- No test BP was accessed. No model scores, resumed fitting, data deletion,
  protocol change or GitHub publication occurred in this inspection.

The [execution receipt](OFFICIAL_CALBASED_EXECUTION_20260908.md) records the
diagnostic command and evidence. Before resubmission, an explicit protocol
decision is needed: preserve exact official membership with a disclosed
duplicate audit and duplicate-excluded sensitivity report, or adopt a named
deduplicated derivative. Internal split/crossfit duplicates also need grouped
handling. Neither alternative has been authorized or implemented by this check.

## Official CalBased — recovery submitted, 15:24 CST

The same six methods have been resubmitted after an authorized preparation
repair. **Data staging is running; formal GPU fitting has not started.**
This supersedes, but does not erase, the failed-batch inspection below.

- Runtime revision: `d20264a14ee5`; immutable server code snapshot
  `~/work/ppg_bp/code/official_calbased_d20264a`. All 313 tracked runtime,
  test, script and plan files matched the local Git object hashes.
- Smoke1673 COMPLETED0:0: **103 tests, zero skips**, plus actual RTX5080
  forward/backward. Metadata preflight1675 COMPLETED0:0 in82seconds.
- Full official, inner and OOF timestamp checks passed: zero positive
  recorded-sample-span overlaps; respectively4,7,1 touching pairs reported.
  Official membership and internal split seed are unchanged. This is not a
  blanket waiver of overlap or content-duplicate checks.
- Hot staging1674 RUNNING. At15:24CST,102/2506 files,23.69GiB of461.55GiB
  verified. Required NAS files all exist; sufficient hot storage is available.
  Originals remain read-only. Scheduler allocation is2CPU/8GB, **no GPU**.
- Materialization1676 waits for1674 and1675, then runs full content checks and
  the actual training/memory metadata-consumer audit before releasing GPU fits.
- Inner/OOF stages1677–1685 and final candidates1686–1691 are PENDINGDependency.
  Frozen six-method evaluation1692 follows; only that stage reads official
  test BP. Overall/MIMIC/VitalDB tables remain prespecified.
- Batch: `~/work/ppg_bp/outputs/official-calbased-v1_20260908-152000`.
  Its `submission.tsv` is byte-identical to the NAS archive. The detailed
  [recovery receipt](OFFICIAL_CALBASED_EXECUTION_20260908.md) lists every stage.
  No new scores, model improvement or completed training are claimed.

The finite chain runs server-side after the desktop disconnects. It does not
launch new experiments from test feedback. No GitHub push was performed during
this recovery; this page and the local Skill continuity record were updated.

## Official CalBased — superseding status at 14:55 CST: preparation failed

- Live scheduler inspection: own queue empty. Preparation1648 FAILEDexit1:0
  after27minutes, at14:13:59CST. Smoke1651 succeeded, but1652–1667 were all
  automaticallyCANCELLED with zero elapsed time and no start timestamp.
  **No formal model in this official batch began fitting; there are no scores.**
- The failure is the new physical-interval check, not participant overlap.
  Metadata-only reproduction finds exactly4 cross-role pairs (1MIMIC/3VitalDB),
  all0.008seconds under the new end-plus-sampling-interval definition. Their
  recorded last/first timestamps touch, with no positive recorded sample-span
  overlap. Old audits treated this case as touching rather than positive overlap.
- Raw T/PPG_Raw checks of all four pairs confirm1250samples, equal boundary
  timestamps, unequal boundaryPPG values and unequal complete arrays. This is
  evidence for a timestamp-boundary contract mismatch, not proof of duplicate
  ten-second data or proof of unrestricted physical independence.
- Another preflight gap: the chosen work raw directories contain only5MIMIC
  and5Vital files. Required checked files remain in the NAS raw master but were
  not staged to those work paths. Both boundary semantics and work staging need
  addressing before resubmission; turning off the check alone is insufficient.
- This inspection changed no training code, official membership or server jobs,
  and read no official test BP. Old prepared manifest still says preparing;
  scheduler FAILED and traceback are authoritative. Historical submission status
  below is retained as a dated observation, not the current state.

## Official CalBased — data preparation and verified implementation

- The user authorized a separate exact-official benchmark, not another custom
  same-subject random split. Both official Info files are downloaded and their
  public SHA-1 and NAS/work SHA-256 verified. Identity-only audits confirm
  2,506 people, 902,160 train and 100,240 test windows, exactly360/40 perperson.
- Six methods and their fitting/evaluation boundaries are predeclared in
  [the executable plan](PLAN_OFFICIAL_CALBASED_20260908.md). Four retained
  methods plus two prospective learned-trust variants; no result is presumed.
- Job1647 failed before reading data because a minimal code package omitted
  package imports. Corrected job1648 is running on hpc-2 with2CPU/16GB and no
  GPU. Its six data-contract tests passed. The failure and corrected snapshot
  are retained rather than hidden.
- Server smoke1649 completed all65 collected tests with no skips; real RTX5080
  forward/backward passed. Five additional rawMAT-to-store integration tests
  also passed on the server. The final source adds eight freezer contract tests,
  passed locally; the submitted snapshot will re-run the complete suite.
- Verified submission at approximately14:05CST: immutable code0872d0f,
  batchofficial-calbased-v1_20260908-140300, jobs1651–1667 submitted. Final
  snapshot smoke1651 COMPLETEDexit0:0; all78tests passed,0skips,8.72seconds,
  plus RTX5080forward/backward. Data preparation1648 remainsRUNNING with
  officialtrain800k/902160 identities decoded at the last inspected log.
  Training1652–1666 and evaluator1667 arePENDING(Dependency), not yet fitted.
  [Full stage receipt](OFFICIAL_CALBASED_EXECUTION_20260908.md).
- All new weights start fresh; previous protected-cohort overlap is documented,
  not portrayed as new unseen-user evidence. Official test BP does not enter
  fitting or personal memory. Six frozen predictions will be scored once by a
  separate evaluator, producing Overall/MIMIC/VitalDB tables.

## Personal-memory v2 — all jobs complete; no eligible upgrade

- Live inspection at12:35CST: own queue empty. Jobs1636–1646 all COMPLETED
  with exit0:0, including both full GPU smokes, two frozen diagnostic jobs,
  four fits, two per-mode reports and final selection. No recovery is needed.
- E1 meanMAE random/chronological2.6475227/3.6334262; E2 2.7125509/3.7881403;
  E3 on selectedv1 relation2.6381435/3.6447506. E1random and E2random both
  select epoch0. Fourfits stopped normally after8 non-improving epochs.
- E1chronological adds0.0237124 over v1 but only0.0789577 over pairedLoRA.
  E3 adds0.0093792/0.0123879 over v1. All four candidates fail the historical
  both-mode>=0.15 gate; pairedLoRA remains the reference. No automatic promotion.
- v1 randomblend worsens2.6475→2.9664/3.3023 with60/300s reference exclusions;
  chrono3.6571→3.6756/3.7850. NewE1/E2 and gap-dependentE3 were not evaluated
  in these controls. Broad time-stratum matching is not exact time matching.
- E3 worsens DBP MAE in >20mmHg absolute train-median-deviation subgroups by
  0.2221random/0.1116chronological versus v1. This is not an adjacent-time-change
  measurement, and the target-defined subgroup is scoring-only.
- All33 aggregate inputs checked byte-identical between serverwork/NAS;
  local transferred archiveSHA3bda883d4c305e3ab76ec8cf1f1ce5c9ad843ec780588c5d57961308ec5fb0d6.
  Public deliverable is aggregate-only; private predictions/checkpoints stay server-side.
- Publisher20tests and existingcomparison8tests pass locally with no skips;
  repeated publication is byte-identical. Generated48primary/96pooledrows,
  plus training/time/subgroup diagnostics. Metadata warning about unused
  synthetic CLI split-mode defaults is disclosed; real mode comes from cache.
- [Formal results and conclusion](RESULTS_PERSONAL_MEMORY_V2.md),
  [full requested tables](../results/personal_memory_v2/RESULT_TABLES.md), and
  [execution receipt](PERSONAL_MEMORY_V2_EXECUTION.md) supersede pending snapshots
  below. No new training or held-out evaluation was performed in this task.

## Historical v2 submission — implementation and technical validation

- User authorized the three targeted routes; the finite matrix is in
  [PLAN_PERSONAL_MEMORY_V2.md](PLAN_PERSONAL_MEMORY_V2.md). E1/E2 each have one
  independent fit per mode. E3 is a fixed reliability diagnostic, not a trained
  error gate. Encoder-level OOF gate learning remains deferred.
- Original v1 caches, source LoRA, validation cohort and held-out sealing are
  unchanged. E1 changes only train donors/weights with explicit physical-gap-v2
  policy; old validation references and both-role alpha remain identical.
- E2 learns an 8,192-parameter retrieval projection, retains the original
  40-window train block and query alpha. Both neural candidates retain epoch0
  and complete validation predictions; synthetic/real smoke is labelled as such.
- Frozen diagnostics reproduce v1 full predictions first, then compare gaps,
  past-only references and time-stratum-matched reference selection. Four fixed
  E3 evaluations across both modes do not count as new neural fits.
- Local technical checks after recovery: 145 personal-memory unit tests discovered;
  111 pass and 34 explicitly skip because local PyTorch is absent. These skips
  are not CUDA verification. Full server tests and actual GPU smoke are required
  before dependent work can proceed. GitHub's previous result commit CI passed.
- Initial snapshot 8ff89d0 was submitted as jobs1625–1635. Both smoke jobs
  (1625/1627) failed before model fitting: one empty-tail diagnostic could not
  be serialized to strict JSON, and two CPU tests exposed shared NumPy/Torch
  storage. Fix df2fba9 preserves the scientific settings and adds regressions.
  All nine dependent jobs were verified unstarted before cancellation. Failed
  logs and the original immutable snapshot are preserved; no result is lost.
- Recovery snapshot df2fba9 is hash-verified and submitted as jobs1636–1646.
  At21:18CST, both smoke jobs1636/1638 are RUNNING and have passed all145
  server unit tests; synthetic/real-data GPU checks are still in progress.
  Remaining jobs wait on dependencies. See the
  [dated execution receipt](PERSONAL_MEMORY_V2_EXECUTION.md) for exact jobs,
  immutable code, failure recovery and output paths. No v2 result or promotion
  is currently claimed.

## Latest completion — personal-memory report repaired and formal results

- User authorized report recovery, formal result publication and a feasible
  next-paper plan. No new model training or held-out evaluation was requested
  or performed in this task.
- Report repair commit20a381462312a48d3138b2ab5f835d2f6493eea2 changes explicit
  scope-column access, not predictions, metrics or promotion rules. Eight
  regressions passed locally (pandas3.0.1) and on the server (pandas2.3.3).
- Final aggregate recovery completed at12:26UTC (20:26CST). The original
  failed job1624 and all training outputs remain preserved. New final outputs
  and the immutable report-code archive match NAS copies byte for byte.
  No GPU allocation or model rerun was needed for this small aggregate task.
- The final selector is now complete, with eligible_candidates=[]; random
  uses continuedLoRA, chronological uses frozenLoRA (only floating precision
  separates it from continuedLoRA). Main-reference status is unchanged.
- [Complete formal results](RESULTS_PERSONAL_MEMORY_V1.md) include36 macro
  and72 pooled diagnostic rows. Offline publisher adds11 passing regressions,
  same-cohort/standard-label/paired-reference checks and input hashes.
  [Public aggregates](../results/personal_memory_v1/) exclude patient data.
  All11 original aggregate input files are byte-identical across local,
  server work and NAS copies. The report and next plan are linked from README
  on the existing method/personal-feature-mechanisms branch.
- [Next-paper plan](PLAN_PERSONAL_MEMORY_PAPER.md) records observed effects,
  nearest-time competing explanation, finite diagnostic/E1–E3 proposals,
  provenance-safe calibration/gating, confirmatory evidence and literature
  boundaries. It does not change the frozen protocol or submit experiments.
- The sections below are historical snapshots, not current pending-job claims.

## Latest live inspection — personal-memory results, 20:06 CST

This section supersedes the running/pending snapshots below. Status-only
inspection: no new training, code repair or GitHub push was performed.

- All ten model fits completed successfully; no scientific skips. The queue
  is empty. Two per-mode reports completed. The final combined report failed
  on a pandas column/attribute collision (`row.view`); predictions and mode
  reports are intact. Repairing the report does not require retraining.
- Overall participant-macro mean MAE (mmHg):

| Candidate | Random-disjoint | Chronological-blocked |
|---|---:|---:|
| Continued LoRA control | 2.8764 | 3.7124 |
| Single reference relation | 3.0375 | 3.9663 |
| Uniform reference relation | 3.1895 | 3.7679 |
| Retrieved reference relation | 2.6973 | 3.6965 |
| Distance-weighted memory + LoRA | **2.6475** | **3.6571** |

- The numerical leader's source-stratified participant-macro results:

| Mode | Source | SBP MAE | DBP MAE | Mean MAE | Mean improvement versus continued control |
|---|---|---:|---:|---:|---:|
| Random | MIMIC | 3.7702 | 2.0847 | 2.9275 | 0.2132 |
| Random | VitalDB | 3.0243 | 1.7265 | 2.3754 | 0.2441 |
| Chronological | MIMIC | 4.4284 | 2.4207 | 3.4245 | 0.0528 |
| Chronological | VitalDB | 4.9641 | 2.8024 | 3.8832 | 0.0577 |

- Overall gains are 0.2289 and 0.0552 mmHg. Manual comparison fails the
  frozen requirement of >=0.15 in both modes. The combined selector itself
  did not finish. This is a numerical lead, not a promoted main model.
- Genuine fits: LoRA controls took57m34s/30m57s; small relation fits took
  38s–2m12s. They reuse frozen256D features and optimize65,730 parameters
  with batch256. Random retrieved/blend trained8epochs but selected epoch0:
  their reported improvement is train-reference BP interpolation/fixed fusion,
  not a learned relation-network benefit. Chronological blend selected epoch1
  after9epochs; mean MAE fell from initial3.796670 to3.657139.
- Simple random memory retrieval deteriorated from2.6973 to3.1590 when
  references within60s were excluded; the frozen LoRA baseline is2.8801.
  This flags temporal proximity as an important explanation, not automatic
  prohibited leakage. This diagnostic did not test the full blended model.
- Both mode reports include Overall/MIMIC/VitalDB macro summaries and the
  requested pooled MAE/R2/ME/STD/threshold/AAMI/BHS columns. Best blend meets
  the retrospective AAMI-style numerical screen throughout; BHS is A except
  chronological VitalDB SBP (B). These are not clinical validation claims.
- All2051 participants/82,040 validation windows remain per mode. Personal
  training budget320 labels; registered-user internal development only.
  Held-out roles remain sealed. No independent-seed uncertainty yet.
- Six selected aggregate report/selection files match their NAS copies byte
  for byte. Local complete per-mode aggregates are saved under the ignored
  `local_archive/personal_memory_status_20260907/` directory. Public results
  have not been updated by this inspection.

## Latest verified completion and new authorized direction

- Live Slurm check: prior queue empty. Feature screen has all16 successful
  results (failed1486 replaced by1565); reports1497/1566/1567 complete0:0.
  PRS screen has all10 results and reports1505/1512/1513 complete0:0.
- [Feature final results](RESULTS_PERSONAL_FEATURE_MECHANISMS.md) and
  [PRS final results](RESULTS_LORA_PRS_CONTINUATION.md) include all requested
  Overall/MIMIC/VitalDB tables under both modes. Fourteen aggregate source
  files were checked against byte-identical NAS copies. No patient rows are published.
- Continued LoRA is strongest within the two latest screens: meanMAE
  2.880098 random /3.712384 chronological. PRS has no positive structural
  gain over matched continued LoRA; both old promotion selectors remain empty.
- User authorized the Personal Memory–Augmented Blood Pressure Estimation
  in Registered Users direction. Implementation/submission is in progress;
  source checkpoints are the completed continued-LoRA jobs1500/1507.
  No new result is claimed. The dated historical snapshots below are not
  current running-job claims.

## Personal-memory screen submitted — 2026-09-07

**Current replacement chain:** code9e7ff58, immutable
`personal_memory_9e7ff58`; manifest`personal_memory_20260907-084550.tsv`.
Smoke1607/1609 complete0:0 with87 passing tests each and real GPU optimization.
Cache1608/1610 running; gate1611; training1612–1616 and1618–1622;
reports1617/1623/final1624. The first export encountered only a documented
float64-vs-source-float32 label representation mismatch (2 entries/mode).
All source predictions replayed exactly. The check now uses the source loader's
float32 representation with the original2e-5 bound; labels/protocol unchanged.
Oldfailed1590/1592 artifacts retained, only14 superseded pending dependents
1593–1606 canceled. Details and both immutable hashes are in the receipt.
Replacement full exports now pass:738,360 raw-window hashes per mode checked,
all82,040 source predictions reproduced exactly, maximum and mean difference0.
Neighbour preparation/diagnostic scoring is the current phase.

The following bullets record the first submitted chain, now superseded:

- [Frozen executable plan](PLAN_PERSONAL_MEMORY_V1.md) and
  [submission/verification receipt](PERSONAL_MEMORY_EXECUTION_20260907.md).
- Code4788e06 is public on `method/personal-feature-mechanisms`; immutable
  source/archive hashes match local/server/NAS. Prior result commit37748e7
  is public on the same branch; no extra branch or unfinished draft was created.
- GPU smoke1589/1591 completed0:0: all85 focused tests pass on each device,
  followed by four finite training steps on real train PPG. Original personal
  source parameters are not updated by smoke. Both hpc-2 cards are used.
- Cache/diagnostic1590/1592 are running. Global gate1593, training1594–1598
  and1600–1604, reports1599/1605/final1606 wait on afterok dependencies.
  Total18 jobs include10 conditional model fits. If no measured complementarity,
  the neural jobs record a scientific skip; no silent new architecture expansion.
- Personal memory uses lawful train-only references, actual source post-LoRA
  features, global lineage audits, blocked train retrieval and causal same-clock
  chronological retrieval. Fixed distance fusion uses no query BP or learned
  validation-error gate. The encoder is supervised in-sample, not OOF.
- Source budget remains320 labels/person; retrievedm5 is not K-shot. All40
  validation windows remain, with base fallback where history is insufficient.
  No held-out role was opened. New performance results remain pending.

## Publication-oriented research blueprint and memory probe

- [Research question, candidate and paper outline](PUBLICATION_RESEARCH_BLUEPRINT_20260907.md).
  This is a proposed research route, not a validated model or submission-ready paper.
  Current strong LoRA remains the comparator; PRS continuation gains were not
  structural gains over matched continued LoRA. Do not rename precedents as inventions.
- Proposed finite first stage: train-only full personal-memory kNN BP/residual
  diagnostics before a reference-conditioned relation network and support-distance
  fusion. Previous fixed five-support attention is not full320 feature retrieval.
  Literature contains related paired calibration and retrieval; no first/SOTA claim.
- Added pure-NumPy `personal_memory_probe.py` and 36 passing synthetic tests.
  Guard coverage: train-only memory, dev-only queries, canonical identities,
  global exact ID/hash/interval conflicts, comparable causal timestamps, no query
  labels, immutable arrays, explicit residual prediction provenance. These checks
  do not certify upstream encoder/split lineage or real-data performance.
- No real-memory inference, new neural training, server changes, test unsealing
  or GitHub push in this planning turn. Prior job status below is a dated snapshot,
  not a fresh live queue check. Both split modes and existing promotion rules remain.

## Historical snapshot — feature-screen serial-loader recovery

- User authorized repairing only the failed shared_bilinear64 random run and
  its blocked reports. [Recovery record](FEATURE_RECOVERY_20260907.md).
- Recovery source `a8cdba40bb0806accbe4fe2d49c27d04bce0ae69`; operational snapshot
  `feature_recovery_a8cdba4`, archive SHA256
  `bd64fb88064abaed2cc1dc7fd447ca9111f581affac8c1f937d216f4e23327f1`.
  Actual model/trainer imports still use the original immutable
  `personal_feature_v1`, source-tree hash
  `daa12286d09a90c7cd529e8991b0f6c7b3f8be4d672ea45ed7e0543940b49965`.
- GPU smoke1564 COMPLETED0:0 on RTX5080: real train samples match serial
  collation, four finite training steps, exact checkpoint reload. Smoke
  weights are not used for the retry; held-out roles remain sealed.
- Retry1565 RUNNING at10:01CST on hpc-2. Same model/seed20260906/batch64/data/
  optimizer/early stopping, restarted from epoch zero; only workers4→0.
  No bitwise replay claim. Original1486 checkpoint/logs are retained.
- New random report1566 waits afterok1565 and uses seven completed original
  runs plus retry1565. New final1567 waits afterok1566 and reuses completed
  chronological report1497. Obsolete pending reports1488/1498 were canceled
  only after replacement dependencies were verified. No other jobs canceled.
- Manifest `feature_recovery_20260907-020009.tsv` archived identically to NAS.
  No new architecture candidates submitted: one smoke, one retry, two reports.

## Completed LoRA + PRS continuation and interpretation

- PRS jobs1499–1513 all COMPLETED0:0. Overall meanMAE random/chronological:
  continuedLoRA2.880098/3.712384; bias2.885003/3.713496;
  dynamic2.893681/3.712990; shrink2.891248/3.712833;
  frozen2.908674/3.739217. Final eligible_candidates=[]; noPRS upgrade selected.
- ContinuedLoRA gains0.159347/0.082083 versus the starting model. An old-start
  comparison alone must not attribute this improvement to the PRS module.
- The separate mechanism study remains the main feature-personalization
  question; PRS was an auxiliary output-correction test. Correct PPG and
  personal state are supported by diagnostics, not a unique physiological
  explanation. Shared/personal capacity is a confound to quantify; nonlinear
  rank4 did not outperform linear rank4. No automatic additional model sweep.
- Full internal-validation source views remain Overall/MIMIC/VitalDB. All
  statements are single-seed development findings; no held-out access.

## LoRA + PRS continuation: submitted, awaiting GPU checks

- [Frozen plan](PLAN_LORA_PRS_CONTINUATION.md): five settings under two modes,
  ten training jobs. This retains the original fitted LoRA and tests compact
  personal correction; it is not a rerun of the historical support-relative
  combinations and is not yet evidence of an improvement.
- Source commit `d3515b342dc567a776764920bbb6e421fbeea519`; immutable snapshot
  `code_snapshots/lora_prs_d3515b3` on the server. Archive SHA-256
  `573153452c7dceee633d5a1c66f75a45caaf8276d1744481b02559c1a52d3472`.
  Local/server hashes match; NAS archive is byte-identical.
- Nine new focused CPU checks and the full repository CPU suite passed.
  Bash syntax checks passed. Both original checkpoint hashes verified.
  GPU checks are **queued, not yet passed**:1499(random, RTX5080),
  1506(chronological, RTX5070Ti). They include unit tests and real train-role
  checkpoint/PPG forward-backward checks. Training uses afterok dependencies.

| Setting | Random job | Chronological job |
|---|---:|---:|
| LoRA continuation control | 1500 | 1507 |
| LoRA + personal BP offsets | 1501 | 1508 |
| LoRA + dynamic PRS | 1502 | 1509 |
| LoRA + dynamic PRS with shrinkage | 1503 | 1510 |
| Frozen LoRA + dynamic PRS | 1504 | 1511 |

- Split reports1505/1512 wait for all five corresponding jobs; final report1513
  waits for both reports. Each split emits Overall/MIMIC/VitalDB macro and
  complete requested diagnostic tables. Final comparison checks gains against
  both the initial model and the continued LoRA control.
- Last live check at approximately2026-09-06 04:10CST: new smoke jobs pending
  GPU availability, training/report jobs pending their success dependencies.
  Earlier feature-study jobs1481/1490 were running on hpc-2; their remaining
  jobs and source snapshot were not modified or canceled.
- Submission manifest `lora_prs_20260905-200916.tsv` is in the work and NAS
  `outputs/submission_manifests` directories, verified identical. Total new
  scheduler entries15: two smoke + ten training + three reporting jobs.
- No new-result or standards claim is available yet. Held-out data remains
  sealed; no new subject exclusions, extra labels, or calibration budget changes.

## Personal feature mechanism study: diagnostics complete, new training submitted

- Frozen diagnostic jobs1474–1477 completed0:0. All four reproduce original
  and cached-feature predictions exactly. [Formal result](RESULTS_PERSONAL_MECHANISMS.md)
  includes Overall/MIMIC/VitalDB plus all requested diagnostic columns.
- LoRA mean MAE natural / within-person PPG permutation / personal-state swap:
  random3.0394 /10.6230,10.5665 /11.0013; chronological3.7945 /5.4063,5.3846 /10.2862.
  Frozen sensitivity supports matched waveform and personal state dependence;
  it is not a causal attribution or a retrained ablation.
- Full repository CPU test suite passed; 12 new focused checks passed.
  GPU smoke1478/1479 passed all nine checks on RTX5080/RTX5070Ti.
- [Frozen candidate matrix](PLAN_PERSONAL_FEATURE_MECHANISMS.md): eight models
  with unchanged data/label access, a paired rank4 LoRA and primary nonlinear
  rank4 personal response, plus sharing/capacity/feature-form controls.
- Submitted training: random1480–1487, chronological1489–1496. Split reports
  1488/1497 depend on successful completion of all their training runs; final
  cross-split gate1498 depends on both reports. No new result is claimed yet.
- At the updated post-submission check1480 and1489 are running, one per GPU.
  1480 has completed full epochs and all82,040 validation windows with finite
  metrics and empty stderr. The other14 training jobs are pending resources/reservations.
  All3 reports show Dependency, as intended. This snapshot is not a live dashboard.
- Random candidates request hpc-2 RTX5080; chronological candidates request
  hpc-2 RTX5070Ti. Matching hardware within each split avoids a candidate/GPU
  confound. Both have passed smoke tests and now have one allocated training job.
- Immutable training source commit `ef51b6d`, snapshot
  `/home/jiangyu.cheng/work/ppg_bp/code_snapshots/personal_feature_v1`.
  NAS source archive `personal_feature_ef51b6d.tar.gz`.
  Private work/NAS manifest `personal_feature_20260905-165514.tsv` records all19 jobs.
  The date is UTC; submission was on2026-09-06 local time.
- Local skill now prioritizes persistent seen-user personalization and records
  this workflow. Unseen K-shot remains separate; held-out access is still prohibited.

## Compact personal-profile screen completed

All 16 training jobs (1398–1405, 1407–1414), split reports 1406/1415 and
final report 1416 completed with exit code `0:0`. The final report was written
on 2026-09-05 at 14:53 UTC+8. At the 2026-09-06 check the user queue is empty
and all 19 corresponding stderr files are empty.

Paired LoRA is best in both modes and all three source scopes. Its Overall
SBP/DBP/mean MAE is `3.9018/2.1771/3.0394` mmHg for random disjoint windows
and `4.8789/2.7101/3.7945` for chronological blocking. The prespecified
32D reliability-gated profile gives `5.1591/2.8768/4.0179` and
`5.5740/3.0540/4.3140`, respectively. No new profile passes the prospective
upgrade gate; LoRA remains the accuracy reference.

Within the new profiles, a dynamic personal correction helps, adding the
current reliability gate slightly worsens both modes, and expanding the
personal code to 64D helps but does not close the LoRA gap. These are
single-seed observations, not repeat-seed confirmation or proof of mechanism.
The next recommended work is to test PPG versus identity contributions and
separate personal capacity from the location of personalization in the
network. No new training was submitted during this publication task.

The audit reread all 16 prediction tables, verified identical development
keys/targets within each mode and recomputed all three scopes' MAE, R², ME,
STD and error percentages. Metadata, profile mappings, predictions and
diagnostics match their NAS copies; all three report directories match too.
Checkpoint binary hashes were not rechecked during this publication audit.
The held-out role remains sealed. The experiment uses 320 labelled training
windows per known participant; it is not a new-user K-shot or exact official
CalBased result.

[Full report and next recommendations](RESULTS_SAME_SUBJECT_PERSONAL_PROFILES.md)
and [aggregate evidence](../results/same_subject_personal_profiles/) are the
authoritative record for this screen.

## Same-subject terminal combination screen completed

A prespecified 15-candidate screen kept participant-indexed rank-4 LoRA as the
common core and combined it with FiLM, support attention, adaptive multiple-
event weighting, support reliability weighting, and calibration-relative
correction. All 30 training jobs, both split reports, and the final selector
completed with exit code `0:0`; all three report directories are byte-
identical between the active work area and NAS archive.

`lora_film_reliability` is numerically lowest under random disjoint windows at
Overall participant-macro SBP/DBP/mean MAE
`3.8668/2.1544/3.0106` mmHg. `lora_film_attention` is lowest under
chronological blocking at `4.8925/2.6737/3.7831` mmHg. Their Overall mean-MAE
gains over the paired LoRA reference are only 0.0347 and 0.0896 mmHg,
respectively. No non-reference combination reaches the frozen 0.15-mmHg gain
in both modes, so the prespecified fallback rule retains plain LoRA.

The full Overall/MIMIC/VitalDB result, requested event-pooled diagnostic table,
cross-split ranking, integrity evidence, and next mechanism-control gate are in
[RESULTS_SAME_SUBJECT_COMBINATIONS.md](RESULTS_SAME_SUBJECT_COMBINATIONS.md),
with machine-readable path-free files under `results/same_subject_combinations/`.
The same-subject held-out role and the participant-disjoint locked meta-test
remain sealed. The result is persistent seen-participant personalization based
on 320 labelled training windows per participant, not unseen-user K=1/2/3/5
calibration or an official PulseDB CalBased reproduction.

## Same-subject single-component screen completed

All 19 paired training settings and common report job 1288 completed with exit
code `0:0`. `residual_subject_lora_rank4` is the internal-validation winner at
Overall participant-macro SBP/DBP/mean MAE
`3.9663/2.2043/3.0853` mmHg. The paired `residual_reference` records
`7.7155/4.2092/5.9624` mmHg, so the winner improves mean MAE by 2.8771 mmHg
(48.25%). It also improves MIMIC from 5.9817 to 3.3604 mmHg and VitalDB from
5.9436 to 2.8180 mmHg, passing the prospective discovery gate in all three
views.

The winning model uses a compact ResNet, the participant's train-role BP mean,
and a rank-4 feature adapter indexed by the already-seen participant. It uses
no five-event support set at inference (`support_count=0`); its personal
adapter and shared network were jointly learned from 320 labelled training
windows per participant. Its strong result therefore supports persistent
known-user personalization, not unseen-user K=1/2/3/5 calibration.

Every candidate contains the same 2,051 participants and 82,040 internal-
validation windows. Work and NAS copies of all candidate run metadata,
predictions, and diagnostics are byte-identical, as are all nine common report
files. The only nonempty stderr is the known Patch Transformer nested-tensor
performance warning. The same-subject held-out role and participant-disjoint
locked meta-test remain sealed.

The full result, interpretation, standards qualification, and next gate are in
[RESULTS_SAME_SUBJECT_SINGLE_COMPONENT.md](RESULTS_SAME_SUBJECT_SINGLE_COMPONENT.md),
with public aggregate files under `results/same_subject_single_component/`.
The candidate passes random-disjoint discovery but still requires a paired
chronological-blocked confirmation and mechanism ablations before any
winner-only held-out release.

## Same-subject PulseDB dual-split screen completed

All 18 model and baseline jobs completed successfully for the separate
`development-calbased-analogue-v1` track. The accepted cohort contains 2,051
participants: 1,011 PulseDB MIMIC and 1,040 PulseDB VitalDB. Each split uses
320 labelled training windows and 40 internal-validation windows per
participant; the 40-window held-out role remains sealed.

`subject_mean_residual_ppg` is the validation winner under both independently
reported split modes. It combines the participant's training-label mean with a
compact-ResNet PPG residual. Overall participant-macro SBP/DBP/mean MAE is
7.6485/4.1693/5.9089 mmHg under `random_disjoint` and
8.0361/4.3716/6.2039 mmHg under `chronological_blocked`. The chronological
mean MAE is 0.2949 mmHg worse. Chronological VitalDB is the weakest source view
at 8.6213/4.8730/6.7472 mmHg, while chronological MIMIC records
7.4341/3.8558/5.6449 mmHg.

The original report jobs stopped after training because a target-consistency
assertion required bit-exact equality across float32 and float64
serializations. The scientific outputs were intact: composite event keys were
identical, and the maximum serialization-only target difference was about
2.24e-5 mmHg. The repaired reporter sorts by participant/source/event keys,
accepts at most 1e-4 mmHg absolute serialization noise, and continues to reject
a 0.01-mmHg mismatch. The full server test suite passed, both reports were
rebuilt from unchanged saved predictions, and work/NAS report directories are
byte-identical.

No new model was trained for the repair, no new experiment was submitted, and
the held-out role was not accessed. See
[RESULTS_SAME_SUBJECT_DUAL_SPLIT.md](RESULTS_SAME_SUBJECT_DUAL_SPLIT.md) and
the public tables under `results/same_subject_dual_split/`. These results are a
development-only same-subject analogue, not unseen-participant performance,
not K=1/2/3/5 calibration, and not an exact official PulseDB CalBased
reproduction.

## Historical same-subject submission record (superseded)

A new, strictly separate `development-calbased-analogue-v1` track has been
implemented and queued to measure the protocol gap between the primary
unseen-participant few-shot task and a seen-participant/new-window task. It is
not an official PulseDB CalBased reproduction and it does not replace
`event120-v1`.

Only the frozen `meta_train` parent split is eligible. The pre-audit cohort has
2,058 participants. Each retained participant contributes 320 training, 40
internal-validation, and 40 sealed held-out 10-second windows. The two
independently reported split modes are `random_disjoint` and the stricter
`chronological_blocked` control.

The first full materialization failed closed after finding exact PPG content
under different storage locators. No model started and no accuracy result was
produced. The repair does not disable that gate: an input-only pre-audit now
hashes selected `PPG_F` samples without loading held-out BP targets. Every
participant represented in an exact-content duplicate group is excluded from
both split modes, after which all manifests and stores are rebuilt. Accepted
stores require zero exact-content duplicates both across and within roles.

The queued first screen contains one train-label-mean baseline and eight
PPG-only neural candidates: subject-mean residual PPG, compact ResNet,
InceptionTime-wide, Patch Transformer, Self-Attention ResUNet adaptation,
rU-Net/ResUNet adaptation, CNN-BiLSTM adaptation, and CNN-Transformer/AFF
adaptation. Original-paper multimodal or demographic inputs are not used.

The repaired immutable training snapshot is commit `addd3b1`, archive SHA-256
`a57d8ab452cd76bea10f021a82f5c619faaee298a9ee17ddbc3169e5f75b9dec`.
It passed all 224 server tests. The preceding model implementation also passed
CUDA forward/backward finite-gradient smoke checks on both the RTX 5080 and
RTX 5070 Ti. The first full-data attempt identified only seven raw-time
overlaps among 823,200 random-mode windows. A deterministic role-swap repair
removed all seven while preserving the 2,058-person cohort and exact
320/40/40 counts; the repeated strict full-data audit passed with zero
cross-role interval and locator overlap.

Failed preparation job 1179 produced no model result. Its partial outputs were
moved intact to recoverable quarantine, and its never-started jobs 1180--1199
were cancelled. Repaired data-preparation job 1202 is running on `hpc-2`;
random-mode jobs 1203--1211 with report 1212 and chronological-mode jobs
1213--1221 with report 1222 are dependency-queued. No GPU model can start
unless job 1202 completes every audit successfully. The retained cohort size
will be reported from that completed audit rather than assumed in advance.

Model selection reads only train and internal-validation data. Held-out BP
targets remain physically separated and have not been accessed. After this
screen, exactly one selected candidate may be refitted on 360 labelled windows
per participant and evaluated once on the held-out 40-window role.

See [PULSEDB_SAME_SUBJECT_ANALOGUE_PLAN.md](PULSEDB_SAME_SUBJECT_ANALOGUE_PLAN.md)
for the protocol and claim boundary.

## Round-14 paired confirmation and complete-method screen completed

All 32 Round-14 jobs completed successfully with empty stderr. All 33 output
directories are byte-identical between the active work area and durable NAS
archive, and the four exploratory-job log pairs were archived before public
reporting. The immutable source snapshot and its archived copy have matching
hashes.

Line A does not confirm the wider InceptionTime architecture. Across the four
prespecified new seeds, `inception_time_wide + QGH` improves mean participant-
macro MAE over `resnet_small + QGH` by 0.0559 mmHg Overall, 0.0427 in MIMIC,
and 0.0669 in VitalDB. All four Overall directions are positive, but the
Overall mean gain is below the frozen 0.15-mmHg confirmation threshold. The
five-seed ensemble remains a descriptive CPU diagnostic and cannot rescue the
failed gate.

Line B identifies one exploratory complete-method candidate. On seed
`20260828`, `calibration_relative` records Overall participant-macro
SBP/DBP/mean MAE of 10.6536/6.0505/8.3520 mmHg, compared with
10.8106/6.2107/8.5107 for the matched wider-InceptionTime QGH anchor. Mean-MAE
gains are 0.1586 mmHg Overall, 0.1253 in MIMIC, and 0.1863 in VitalDB. It
passes the primary internal gate but fails the tail gate, so it advances only
to independent-seed confirmation. The standards-oriented variant improves
Overall mean MAE by 0.1288 mmHg and passes neither gate. The optional
six-group C3 route was not evaluated in this completed result.

The `calibration_relative` event-pooled Overall SBP/DBP MAE is
12.1891/6.6635 mmHg. Both endpoints fail the retrospective AAMI-style
numerical screen, with historical BHS Grade D for SBP and Grade C for DBP.
These diagnostics do not establish formal device or standards compliance.
Meta-validation and the locked meta-test remain inaccessible. See
[RESULTS_ROUND14_CONFIRMATION_AND_COMPLETE_METHODS.md](RESULTS_ROUND14_CONFIRMATION_AND_COMPLETE_METHODS.md)
and the aggregate files under `results/round14/`.

## Round-13 final architecture/capacity screen completed

The wider InceptionTime QGH setting is the Round-13 numerical winner. Overall
participant-macro SBP/DBP/mean MAE is 10.8759/6.1636/8.5197 mmHg, compared
with 11.0255/6.3922/8.7089 for the same-round compact ResNet QGH reference.
The mean-MAE gain is 0.1891 mmHg Overall, 0.3497 in PulseDB MIMIC, and 0.0557
in PulseDB VitalDB. It passes the prespecified single-seed internal gate and
therefore advances to seed-stability confirmation; it is not yet the final
model.

Event-pooled Overall SBP/DBP MAE is 12.3256/6.8079 mmHg. Both endpoints fail
the retrospective AAMI-style numerical screen, with historical BHS Grade D
for SBP and Grade C for DBP. These are internal numerical diagnostics only.
See [RESULTS_ROUND13_FINAL_CAPACITY_SCREEN.md](RESULTS_ROUND13_FINAL_CAPACITY_SCREEN.md)
and the aggregate tables under `results/round13/`.

## Round-10 partial end-to-end screen completed

Round 10 is complete. Recovery jobs 1017--1025 and deterministic report job
1026 all completed with exit code `0:0`, every stderr file is empty, and every
work output matches its NAS archive. The full server suite passes 119 tests.
All nine candidates use the identical K=5 fold-4 comparison of 628 participants
and 96,332 queries after folds 0--2 fitting and fold-3 patience-8 early
stopping. Meta-validation and the locked meta-test were not accessed.

T10-8, last-block adaptation with pair-direction and temporal-consistency
objectives, is the numerical winner. Its Overall participant-macro SBP/DBP/
mean MAE is 10.7820/6.1093/8.4457 mmHg, versus 10.8946/6.1697/8.5322 for the
frozen-encoder reference. Mean MAE improves by 0.0865 mmHg Overall, 0.0725 in
MIMIC, and 0.0981 in VitalDB.

Although both internal source strata improve, the Overall gain is below the
prespecified 0.15-mmHg promotion threshold. No Round-10 candidate is promoted
or evaluated on meta-validation. See the complete
[Round-10 result](RESULTS_ROUND10_PARTIAL_END_TO_END.md), the prospective
[plan](ROUND10_PARTIAL_END_TO_END_PLAN.md), and aggregate tables under
`results/round10/`.

The earlier failed chains produced no scientific result. Jobs 991--1003
exposed an argument-validation defect, while jobs 1007--1016 exposed a
mixed-precision loss-boundary defect. Both were corrected and regression-
tested before the accepted recovery run. The exact accepted snapshot is
`event120-v1_round10_dtype_fix_20260821-160940.tar.gz`, SHA-256
`2c239ba85501baca2a561b0c3076778c6661d225e975ccc120178f4df30b6e7d`.

## Round-9 calibration refinement completed

Round 9 is complete. Jobs 975--985 all completed with exit code `0:0`, empty
stderr, and byte-identical work/NAS reports. The screen compares one
architecture-matched R8 reference with eight isolated changes at K=5.

To reduce repeated model-selection pressure on the previously viewed
meta-validation set, folds 0--2 of participant-disjoint meta-train fit each
candidate, fold 3 controls patience-8 early stopping, and fold 4 ranks the
candidates. Meta-validation is not used for training, early stopping,
prediction, scoring, or candidate ranking in this screen. The locked meta-test
is not accessed. The server suite passes 97 tests. The fold-4 comparison covers
628 participants and 96,332 common queries.

No candidate passes the internal promotion gate. R9-1 adaptive fusion is the
numerical winner but improves Overall participant-macro mean MAE by only 0.036
mmHg and worsens VitalDB by 0.003 mmHg. R9-7 is the only method to improve both
source strata, but its Overall gain is only 0.010 mmHg. No Round-9 candidate is
promoted or evaluated on meta-validation. See the complete
[Round-9 result](RESULTS_ROUND9_CALIBRATION_REFINEMENT.md) and aggregate files
under `results/round9/`. The next justified route is partial end-to-end
PPG-encoder adaptation to query-to-calibration BP change.

## Ten-second PPG beat-to-beat similarity audit

The within-window morphology audit is complete for all 103,564 K=5
meta-validation query PPG windows from 697 participants; the locked meta-test
was not accessed. The median within-window pairwise beat correlation is 0.9923
Overall, 0.9932 for PulseDB MIMIC, and 0.9783 for PulseDB VitalDB. However, the
window-level 10th percentiles are 0.5792, 0.7741, and 0.3088, respectively.
Thus typical morphology is highly repeatable, but there is a marked low-
similarity tail, especially in VitalDB. Full aggregate results and method
limitations are documented in
[PPG_BEAT_TO_BEAT_SIMILARITY.md](PPG_BEAT_TO_BEAT_SIMILARITY.md); private
participant/event rows are not published.

The follow-up similarity--error analysis is also complete. In both the Quality
Gate + Huber reference and the R7-5 causal GRU, event-level correlations are
near zero and within-participant comparisons do not show higher error below
0.90 similarity. Participant-level correlations are weakly positive rather
than negative. Therefore the current normalized morphology-similarity score is
not promoted as a quality gate, rejection rule, or specialist-routing feature.
See [PPG_BEAT_SIMILARITY_ERROR_RELATION.md](PPG_BEAT_SIMILARITY_ERROR_RELATION.md).

## Round-7 new training submitted

Round 7 supersedes the earlier interpretation that the completed Phase-6E
cluster head fully represented the requested waveform-category system. Nine
new candidates are now submitted under seed `20260821`: six newly trained OOF
residual methods, a separately trained waveform-phenotype router with hard and
soft independent category experts, and a deeper waveform-embedding causal GRU.

The full server suite passed 72 tests before submission. Slurm jobs 932--942
form the new chain; no previous candidate job is substituted into its final
report. The exact submitted-code snapshot is
`event120-v1_round7_nine_routes_20260819-1437.tar.gz`, SHA-256
`56888eb537b573d5d4295f53ee0ae99967a00c23f7f5e5732ad3bf2b3baa9a91`.
The design and leakage boundary are documented in
[ROUND7_NINE_ROUTE_PLAN.md](ROUND7_NINE_ROUTE_PLAN.md).

## Phase-6E result

The seven-route K=5 development screen is complete. All jobs 915--924 and the
cluster-audit correction jobs 925--926 completed with exit code `0:0`, empty
stderr, and byte-identical work/NAS report artifacts. The locked meta-test was
not accessed.

The causal GRU residual corrector is the numerical winner: Overall
participant-macro SBP/DBP/mean MAE is 10.674/6.141/8.408 mmHg versus
10.803/6.168/8.485 for Quality Gate + Huber. MIMIC mean MAE improves from
9.075 to 8.969 and VitalDB from 7.996 to 7.942. The Overall gain is only
0.078 mmHg, below the frozen 0.15-mmHg promotion threshold, so no route is
promoted and Quality Gate + Huber remains the development base.

The morphology-cluster MoE used an explicitly exploratory K=8 fallback because
none of K=8/16/32 reached the 0.75 meta-train stability gate. Its Overall gain
was only 0.005 mmHg and MIMIC worsened. It does not establish stable learned
waveform phenotypes. Full aggregate results are in
[RESULTS_PHASE6E_SEVEN_ROUTES.md](RESULTS_PHASE6E_SEVEN_ROUTES.md) and
`results/phase6e/`.

## Phase-6E seven-route screen submitted

Phase-6E now tests seven K=5 continuous-error corrections against the frozen
Quality Gate + Huber development comparator: ridge residual correction,
residual MLP, confidence-gated residual MLP, difficult-participant 2x weighted
MLP, causal GRU residual correction, supervised mixture-of-experts, and an
unsupervised morphology-cluster mixture-of-experts. The last route learns
waveform phenotypes from frozen PPG embeddings rather than imposing a fixed
70%/30% partition; a new query is softly assigned to cluster-specific residual
experts and every query retains a prediction.

The server regression suite passed 70 tests and all seven routes passed a
real-data end-to-end preflight. The corrected submission chain is jobs
915--924: preparation 915, routes 916--921, waveform embedding 922,
morphology-cluster route 923, and unified report 924. Preparation job 915
completed successfully with empty stderr. The exact submitted-code archive is
`event120-v1_phase6e_seven_routes_postfix_20260819-1351.tar.gz` with SHA-256
`acf357bfc2085656e8e078775f1d5d1fed935940a359b87a917a0fb30864e1d8`.
An earlier dependency chain 900--907 was superseded after job 900 exposed an
NAS permission-preservation incompatibility; jobs 901--907 were cancelled
without running, and the corrected chain also reduces the cluster job's memory
request to the verified node limit.

All candidate targets are derived from participant-disjoint Quality Gate +
Huber out-of-fold meta-train predictions. Folds 0--3 fit candidate parameters,
fold 4 performs internal selection, and meta-validation is evaluated only
after freezing. The locked meta-test remains inaccessible. A candidate advances
only if Overall participant-macro mean MAE improves by at least 0.15 mmHg and
both MIMIC and VitalDB improve; otherwise Phase-6E remains a negative screen.

## Completed gates

- PulseDB v2 controlled extraction and integrity validation completed.
- All 5,361 participant files passed the project schema gate.
- The normalized segment index contains 5,245,454 valid segments.
- Frozen source-stratified participant split:
  - 3,752 meta-train participants;
  - 805 meta-validation participants;
  - 804 quarantined meta-test participants.
- `event120-v1` was selected using development participants only.
- The 120-second protocol yields 3,840 eligible development participants and
  681 eligible locked-test participants under the fixed eligibility rule.
- Development event table: 606,010 representative event rows.
- Locked input table: 104,874 event rows; common-query targets are held in a
  separate evaluator-only artifact.
- Leakage audit passed for subject disjointness, temporal order, common-query
  equality across K, and locked-query label isolation.
- Development and locked-input PPG waveforms were materialized to the hot work
  area without query BP in the model-input store.
- GPU smoke testing passed on an NVIDIA RTX 5080.
- The server regression suite contained 25 passing tests at the training-queue
  freeze.

## Development-only training

The five prespecified seed pipelines (`20260813`--`20260817`) completed for the
population model, M0, M1, M2, and the shared calibration controls. Scheduler
jobs 782--807 all exited `0:0`; the final aggregation and independently rebuilt
report were archived identically to work and NAS. The locked meta-test was not
accessed.

M0 has the lowest five-seed mean participant-macro MAE at every calibration
budget. Values below are mean ± sample SD across training seeds:

| K | SBP MAE | DBP MAE | Mean MAE |
|---:|---:|---:|---:|
| 1 | 12.920 ± 0.111 | 7.242 ± 0.096 | 10.081 ± 0.096 |
| 2 | 12.240 ± 0.136 | 6.812 ± 0.095 | 9.526 ± 0.112 |
| 3 | 11.894 ± 0.144 | 6.576 ± 0.095 | 9.235 ± 0.116 |
| 5 | 11.318 ± 0.110 | 6.253 ± 0.086 | 8.785 ± 0.094 |

All rows contain the same 697 meta-validation participants and 103,564 future
query events per K. Across K, M0 has mean MAE 9.407 ± 0.104 mmHg, followed by
M1 at 9.453 ± 0.143 and M2 at 9.510 ± 0.102. The M0--M1 difference is small
relative to seed variability; M0 is therefore the parsimonious provisional
finalist, not a conclusively superior model.

The unlimited-epoch runs all stopped by patience-8 early stopping. M0 selected
best epochs 10--33 across seeds, confirming that the original fixed 25-epoch
cap was not adequate for every initialization while showing that the new runs
did converge under the prespecified stopping rule.

The five-seed [extended report](RESULTS_PHASE5_REPEAT5.md) contains the required
Setting/BP/MAE/R²/ME/STD/5--10--15-mmHg/AAMI/BHS columns. All 88 method/K/BP
summary rows fail the AAMI numerical screen in every seed, and no row obtains
a BHS Grade A or B in any seed. These are retrospective numerical screens, not
formal device-validation determinations.

The complete first-run data-selection funnel and acceptance/exclusion rules are
documented in
[DATA_SELECTION_AND_TRAINING_COHORT.md](DATA_SELECTION_AND_TRAINING_COHORT.md).

## Current gate

The convergence and repeat-seed gates are complete. Before a one-time locked-
test evaluation, use development data only to complete the planned residual-
tail analysis, decide whether it motivates one prespecified training change,
and then freeze M0 (or a justified alternative), the seed/checkpoint policy,
statistics, exclusions, and reporting script.

The fixed-first single-factor Phase-6 screen is complete and documented in
[RESULTS_PHASE6_SCREENING.md](RESULTS_PHASE6_SCREENING.md). Jobs 826--830 all
completed successfully on the same seed and query set. The PPG-only quality
gate improved the four-K Overall participant-macro mean MAE from 9.137 to
8.962 mmHg and improved all four K values, but most of the gain occurred in
MIMIC; it is a provisional component, not yet a confirmed replacement for M0.

The exact observed-error worst 30% contains 210 of 697 participants at K=5.
Their M0 mean MAE is 13.714 mmHg versus 6.656 mmHg for the oracle retained 70%.
They show more support-to-query BP change, greater within-participant BP
variability, and later query horizons. These are oracle associations and cannot
be used as a deployable filter.

Round 4 is complete and documented in
[RESULTS_PHASE6B_FACTORIAL.md](RESULTS_PHASE6B_FACTORIAL.md). All five new jobs
completed successfully; together with the three existing factorial cells they
provide the full quality-gate x Huber x participant-tail-CVaR comparison.
Quality gate plus Huber has the lowest four-K Overall participant-macro mean
MAE: 8.888 mmHg versus 9.137 for fixed-first M0 and 8.962 for quality gate
alone. The participant-cluster bootstrap difference versus M0 is -0.249 mmHg
(exploratory 95% interval -0.353 to -0.150). The gain is larger in MIMIC
(-0.472) than VitalDB (-0.065; interval includes zero), so this is a
provisional single-seed candidate rather than a frozen final model.

Participant-CVaR does not improve the full cohort, either alone or when added
to the quality gate and/or Huber. The three-factor setting is slightly best on
the fixed observed-error tail at K=5, but worse than quality gate plus Huber on
the full four-K cohort. Because true tail membership uses query error, that
result remains an oracle diagnostic and does not justify a deployment-time
router. The locked meta-test remains untouched.

An additional Phase-6C two-stage experiment has been implemented and submitted
to answer the separate deployment question: can difficult cases be recognised
without seeing their reference BP, and can a dedicated model improve them?
Five source-stratified participant folds inside `meta_train` create K=5
out-of-fold M0 errors and exact within-source worst-30% labels. A 22-feature
input-visible risk MLP is then trained without query BP, true error, source, or
participant identity. Three specialist variants test 4x difficult-group
sampling, difficult-group-only training, and difficult-group-only training
with the PPG quality gate. Each will be evaluated alone, through a frozen hard
router, and through soft risk-weighted fusion with M0.

The new implementation passed 61 server regression tests and a read-only real-
store preflight covering all 3,143 meta-train participants and all 103,564 K=5
meta-validation query rows. The dependent jobs are queued; no Phase-6C result
is reported yet. This prototype is deliberately K=5. It may be extended to
K=1/2/3 only with support-budget-specific features that do not use unavailable
cuff measurements. Promotion requires useful held-out participant AUPRC and a
full-coverage gain in Overall, MIMIC, and VitalDB; a single-seed gain would
still require confirmation.

Phase-6C subsequently completed without scheduler or artifact-integrity
failures. Its input-only risk classifier showed moderate, not yet decisive,
held-out meta-train discrimination (Overall participant AUPRC 0.469;
precision 0.444 and recall 0.442 at the frozen threshold). The best original
specialist route improved K=5 Overall participant-macro mean MAE only slightly,
from 8.783 to 8.736 mmHg. This supports testing the two-stage idea further but
does not yet establish reliable automatic identification or specialist value.

Phase-6D is complete and reported in
[RESULTS_PHASE6D_RISK_ROUTING.md](RESULTS_PHASE6D_RISK_ROUTING.md). Jobs
880--890 all completed with exit code `0:0` and empty stderr. The corrected
Quality Gate + Huber risk classifier achieves Overall participant AUPRC 0.457,
AUROC 0.659, precision 0.470, and recall 0.486. Predicted-high-risk
participants have mean MAE 10.140 versus 7.737 mmHg in the predicted-low group,
showing useful but incomplete separation.

Moderate 2x difficult-participant weighting is the only specialist that
improves the fixed evaluation-only difficult tail in both sources. Fourfold
weighting is too aggressive, and difficult-only training worsens the tail. The
binary event hard route changes Overall mean MAE only from 8.485 to 8.481 mmHg
and worsens VitalDB, so the requested 70%/30% hard-routing system is not
promoted. Continuous event-risk soft fusion is the Phase-6D winner at 8.424
mmHg Overall, 8.986 MIMIC, and 7.958 VitalDB, improving the general model by
0.061/0.089/0.038 mmHg. Exploratory 20,000-repetition paired participant
bootstrap intervals exclude zero in all three scopes, but the gain is small
and comes from the same single-seed development screen. Soft fusion advances
only to independent-seed confirmation; locked meta-test remains untouched.

## Round-11A backbone screen completed

All formal jobs 1033--1048 completed with exit code `0:0`, and all 16 work/NAS
result pairs are byte-identical. The patch-Transformer stderr contains only a
PyTorch nested-tensor optimization warning; the other formal stderr files are
empty. Every setting uses the same 628 fold-4 participants and 96,332 K=5
queries after folds 0--2 fitting and fold-3 patience-8 early stopping.

The compact ResNet remains the numerical winner. QGH participant-macro
SBP/DBP/mean MAE is 11.0136/6.2771/8.6453 mmHg Overall, with mean MAE 9.3631
in MIMIC and 8.0489 in VitalDB. InceptionTime is the closest alternative but
worsens the three means by 0.0774/0.0741/0.0801 mmHg. Deeper ResNet, patch
Transformer and Conformer worsen Overall by 0.2583, 0.2679 and 0.3336 mmHg,
respectively. No alternative improves either internal source stratum.

`winner_backbone=resnet_small` and `passes_internal_gate=false`; no new
backbone is promoted. The result supports retaining the smaller encoder and
rejects complexity-for-complexity's-sake under the current protocol. It does
not establish a universal negative conclusion about all attention models.
Meta-validation and the locked meta-test were not accessed. The accepted
report is [RESULTS_ROUND11_BACKBONE_SCREEN.md](RESULTS_ROUND11_BACKBONE_SCREEN.md),
with public aggregate files under `results/round11a/`.

Stage 11B may now use the retained compact ResNet to perform a structural subtraction
ablation of the personal correction MLP, PPG-only quality gate and
query-conditioned support attention, followed by a paired MSE-versus-Huber
loss comparison on the selected minimal structure.
Stage 11C will then compare universal-only, universal + stable personal bias,
universal + dynamic change and the full three-part decomposition. These later
stages are intentionally not submitted before the upstream winner is known.
The complete prospective design is in
[ROUND11_SYSTEMATIC_MODEL_REVISION_PLAN.md](ROUND11_SYSTEMATIC_MODEL_REVISION_PLAN.md).

## Not yet established

- no locked-test result or frozen final configuration;
- no real pressure-, motion-, or device-shift robustness result;
- no independent external validation;
- no clinical validation or standards claim.

## Round-12 literature-derived backbone screen completed

A broad calibrated/personalized PPG-BP literature audit was completed before
new jobs were defined. The central finding is that no audited paper currently
demonstrates PulseDB + PPG-only + participant-disjoint development + strictly
chronological first K<=5 labeled events + no later label access + SBP/DBP
AAMI/BHS acceptance at the same time. Several very low PulseDB errors instead
use same-participant high-volume window splits, multimodal inputs, or ongoing
reference-BP updates. The evidence table and source links are preserved in
[LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md](LITERATURE_AUDIT_CALIBRATED_PPG_BP_20260822.md).

Round 12 therefore performs a controlled architecture-family screen rather
than copying incomparable headline scores. It compares the unchanged compact
ResNet with a causal TCN, a five-shot residual-attention encoder, a compact
CNN-GRU and a 1D residual U-Net. All five use the same PPG-only QGH calibration
head, fixed-first K=5 support, folds 0--2/3/4, Huber loss, quality gate and
single seed. Meta-validation and the locked meta-test remain quarantined. The
prospective specification is in
[ROUND12_LITERATURE_BACKBONE_PLAN.md](ROUND12_LITERATURE_BACKBONE_PLAN.md).

Server verification passed all 131 regression tests. CUDA smoke job 1049
completed with exit code `0:0` after forward propagation, backpropagation and
an optimizer update for all existing and new backbones. Formal jobs 1050--1065
then all completed with exit code `0:0`, all stderr files are empty, and all 16
work/NAS output pairs are byte-identical.

The compact ResNet remains the numerical winner at Overall participant-macro
SBP/DBP/mean MAE 11.0294/6.2592/8.6443 mmHg. The residual U-Net, TCN,
five-shot residual-attention encoder and CNN-GRU worsen Overall mean MAE by
0.1120, 0.1778, 0.4141 and 0.4245 mmHg, respectively, and every candidate also
worsens mean MAE in both internal PulseDB source strata.
`passes_internal_gate=false`; no new architecture advances to meta-validation,
multi-seed confirmation, or the locked meta-test. The accepted aggregate report is
[RESULTS_ROUND12_LITERATURE_BACKBONES.md](RESULTS_ROUND12_LITERATURE_BACKBONES.md),
with machine-readable tables under `results/round12/`.

## Round-8 calibration-relative screen completed

All Round-8 jobs completed successfully with empty stderr, and the locked
meta-test was not accessed. This was a K=5, single-seed development screen
built around the Quality Gate + Huber job-841 reference. Every full-coverage
candidate used the same 697 participants and 103,564 query events.

R8-4, which combines pairwise calibration-relative prediction, causal query
history, and a support-BP-range auxiliary task, is the numerical winner. Its
Overall participant-macro SBP/DBP/mean MAE is
10.487/6.021/8.254 mmHg, compared with 10.803/6.168/8.485 mmHg for the
reference. Mean MAE also improves in both internal PulseDB source strata:
9.075 to 8.852 mmHg in MIMIC and 7.996 to 7.758 mmHg in VitalDB.

The `similarity >=0.90` sensitivity retains only 79.81% of Overall queries and
has strongly source-dependent coverage, so it is not a full-coverage model and
is not promoted. Direct demographic concatenation is nearly neutral, while a
128-dimensional PPG representation is slightly worse than the 256-dimensional
reference. R8-4 advances as a candidate base for a further development round;
multi-seed confirmation is deferred until that method round is complete.

The full result, scope limitations, candidate interpretation, and links to all
public Overall/MIMIC/VitalDB tables are in
[RESULTS_ROUND8_CALIBRATION_RELATIVE.md](RESULTS_ROUND8_CALIBRATION_RELATIVE.md).
The prespecified design remains available in
[ROUND8_CALIBRATION_RELATIVE_PLAN.md](ROUND8_CALIBRATION_RELATIVE_PLAN.md).

## Round-13 final architecture/capacity screen completed

Round 13 completed the controlled single-seed comparison of 13 population PPG
encoders. Every candidate retains the same K=5 fixed-first Quality Gate + Huber
calibration model, folds 0--2/3/4, query set, effective batches, sampled
examples per epoch, seed, and 256-dimensional encoder output. Meta-validation
and the locked meta-test remain quarantined.

The wider InceptionTime encoder is the numerical winner. Its Overall
participant-macro SBP/DBP/mean MAE is 10.8759/6.1636/8.5197 mmHg, compared with
11.0255/6.3922/8.7089 mmHg for the same-round compact ResNet. Mean MAE improves
by 0.1891 mmHg Overall, 0.3497 mmHg in the internal MIMIC stratum, and 0.0557
mmHg in the internal VitalDB stratum. This satisfies the prespecified internal
gate, so `winner_backbone=inception_time_wide` and
`passes_internal_gate=true`.

The result does not support network scaling in general. A wider InceptionTime
helps, but deeper/wider ResNets, deeper/wider or differently tokenized Patch
Transformers, a larger Conformer, and ConvNeXt-1D do not improve the reference.
The VitalDB mean gain is also small and is driven by DBP while SBP is slightly
worse, so source- and endpoint-level confirmation remains necessary.

All 173 pre-submission regression tests and both formal-batch GPU smoke tests
passed. All 13 population, QGH, and fold-4 evaluation chains completed. A
post-run SHA-256 inventory found identical relative file sets, sizes, and
contents for all 41 work/archive output pairs. One final housekeeping step
returned a nonzero status because the archive target does not support
preserving POSIX permission metadata; this occurred after the scientific
outputs completed and did not produce a content mismatch.

The accepted result and full Overall/MIMIC/VitalDB tables are in
[RESULTS_ROUND13_FINAL_CAPACITY_SCREEN.md](RESULTS_ROUND13_FINAL_CAPACITY_SCREEN.md),
with machine-readable aggregate files under `results/round13/`. The prospective
specification remains in
[ROUND13_FINAL_CAPACITY_SCREEN_PLAN.md](ROUND13_FINAL_CAPACITY_SCREEN_PLAN.md).
The wider InceptionTime candidate advances only to independent-seed
confirmation and does not yet replace the compact ResNet reference. Neither
meta-validation nor the locked meta-test has been accessed.
