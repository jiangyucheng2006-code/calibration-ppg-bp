# Exact official CalBased: personal LoRA and explicit history

Status: implementation and preflight, 2026-09-08. This is a finite six-candidate
experiment, not a new result or a claim of clinical validation.

## Research question and authorization

Can explicit personal history improve a persistent subject-specific LoRA model
when both use the exact official PulseDB CalBased training and test assignments?
The user authorized this new protocol and its full official cohort on 2026-09-08.
The old development-analogue experiments, parent subject partitions, checkpoints,
and historical promotion decisions remain unchanged.

Protocol identifier: `pulsedb-official-calbased-v1`.

### Authorized source-duplicate policy, 2026-09-08

The user explicitly requests preserving the official assignments including
their source-native duplicate signals. No row is dropped, reassigned or replaced;
all 902,160 TRAIN and 100,240 TEST rows, exact 360/40 budgets, six methods and
the existing seed/inner assignments remain unchanged. A deduplicated test subset
is not substituted or added to this run.

Policy ID: `official-source-duplicates-retained-20260908`. The private source
audit is pinned by SHA-256
`ed3ce6fbc707ce4c4c635c9cf32a0dd25abb144ed4d54892947f6242376193ac`.
Only its 154 verified duplicate pairs are exempted from exact-content rejection:
35 cross official TRAIN/TEST, 117 within TRAIN and two within TEST, all MIMIC.
There are 35 affected TEST rows. The original float64 PPG_F, PPG_Raw and T arrays
and original Info identities/indices were verified before this decision, without
test BP. New duplicates, repeated row IDs, invalid official membership, changed
waveforms, positive same-record interval overlap and target-access violations
are not permitted by this exception. Old experiment defaults remain strict.

The unchanged inner assignments have 23 duplicate groups across fitting and
validation, and 82 across encoder-fit folds. Consequently, internal validation
and OOF are **index-disjoint but not strictly content-disjoint**: exclusion of
an exact row does not exclude its source copy. State this limitation for trust
training and checkpoint selection rather than claiming fully independent OOF.
The final full official report must include this disclosure. It is an exact
membership benchmark, not proof of leakage-free waveform generalization or
clinical validity. No official TEST row labels enter fitting or adaptation.

The official cohort overlaps participants previously protected or used in other
project experiments. New models therefore cannot be presented as independently
validated unseen-user models. Fresh initialization prevents old checkpoint-label
reuse; it does not erase historical researcher exposure to this dataset. This is
an explicitly declared official benchmark replication, not a new external study.

## Exact source and membership

Membership comes from `Train_Info.mat` and `CalBased_Test_Info.mat`, linked in
the [official preparation guide](https://github.com/pulselabteam/PulseDB/blob/db0824f18d9a462458e46fe94c31283a93a5c0d5/Info_Files/File_Preparation_Guide.md).
The original split is described in [the PulseDB paper, Section 2.6](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full).

| Source | Participants | Official TRAIN | Official CalBased TEST |
|---|---:|---:|---:|
| MIMIC | 1,213 | 436,680 | 48,520 |
| VitalDB | 1,293 | 465,480 | 51,720 |
| Overall | 2,506 | 902,160 | 100,240 |

Each person has 360 training and 40 test windows. These are 10-s PPG windows
with ABP-derived BP labels, not 360 independent cuff measurements. Same-subject
overlap is intentional. The official `Subj_Name` and one-based `Subj_SegIDX`
are mapped to their original subject MAT file and segment index; the project
does not substitute its previous 2,051-person 320/40/40 split.

Files verified against the public Box SHA-1, and copied NAS-to-work with SHA-256:

| File | SHA-256 |
|---|---|
| Train_Info.mat | `9d6be19cb4d0b1f8b2db9c5beae66bc6a5ae89eaf8caeaa1d829a715e3e74088` |
| CalBased_Test_Info.mat | `35914d57e38dfa27c4ff36d751365a7ace1ab772b4dc813404c9f4936d54c698` |

Only identity/index fields are decoded from Info files during preparation.
Training targets are loaded using the exact official TRAIN window predicate.
Test inputs are stored without BP. Exact IDs, physiological intervals, PPG
content hashes, source counts and historical parent overlap are audited. A
membership discrepancy is not silently repaired by dropping or moving windows.

## Six prespecified candidates

| ID | Candidate | Change from its paired reference |
|---|---|---|
| L0 | Persistent LoRA | Fresh common backbone and saved personal adapters; paired baseline |
| M0 | Fixed personal-memory blend | Frozen PPG retrieval and fixed distance-dependent blend |
| M1 | Learned memory relation, v1 | Learn reference-to-query BP differences; exclude the query's 40-window donor block during relation training |
| M2 | Retrieval-matched relation, E1 | Match relation-training retrieval more closely to inference; exclude self/physical overlap instead of an entire donor block |
| T1 | Cross-fitted scalar trust | Learn one mixing weight for the LoRA and fixed-memory branches |
| T2 | Cross-fitted BP-specific trust | Learn separate SBP and DBP mixing weights using the same inputs and training data as T1 |

T1 and T2 are hypotheses, not demonstrated innovations. Gating, retrieval and
low-rank adaptation have precedents. Neither candidate adds an unacknowledged
E1 branch. Trust inputs describe PPG feature distances, legal reference count,
reference BP dispersion, and disagreement between two predictions. They do not
contain the query BP, prediction error, ABP, ECG, future test BP, or a label-defined
"difficult" group. All test queries are retained; invalid retrieval falls back
to LoRA, not selective deletion.

## Development, refit and test access

1. Within each official TRAIN set, deterministic seed `20260908` reserves 40
   windows for internal validation, leaving 320 for fitting. No official TEST
   window participates in this step.
2. A fresh LoRA is fitted on all 320 windows/person per epoch. Participant-macro
   mean(SBP MAE, DBP MAE) selects the checkpoint; patience is eight epochs with
   no arbitrary epoch cap. All four existing candidate components use the same
   selected encoder and same queries.
3. Three scratch LoRA fits exclude complementary blocks of the internal 320
   windows. Their budget is **25 epochs fixed before training**, not selected
   using excluded-fold outcomes. Encoder, personal parameters, target scalers,
   anchors and donor banks all exclude that fold. Each trust-training query is
   predicted once by the model that did not fit its label. This is internal
   cross-fitting for gate training, not three repeated benchmark seeds.
4. Relation and trust heads use patience eight on the internal 40 windows.
   Relation fits retain the previous zero-initialized/fixed-blend comparison;
   an epoch-zero selection will be reported as no learned-relation improvement.
   Heads use 200,000 sampled supervised examples per epoch from their complete
   eligible training pools, with batch size 256; this is an optimizer budget,
   not a claim that only those labels were available.
5. Freeze all candidate settings and the inner-selected epoch counts. Refit
   LoRA from scratch using all official 360 windows/person for its selected
   number of epochs. Refit M1/M2 on this new feature space for their frozen
   selected epoch counts. M0 remains fixed.
6. T1/T2 retain their gate and input-scaler weights learned from inner-320 OOF
   data; only the LoRA and reference bank switch to full-360. Thus the gate's
   supervised budget is **320/person**, while the complete registered-user
   system has **360/person**. This transfer is deliberate and may underperform;
   it is not concealed as a full-360 OOF gate refit.
7. Each frozen method emits predictions on all exact official test inputs, with
   no target columns. After all six prediction files and checkpoint hashes are
   saved, a separate evaluator binds them to the frozen plan and reads only the
   matching official TEST labels to produce the prespecified tables. There is
   no automated return path from test errors to training or hyperparameters.

The five retrieved reference windows are selected from the available training
bank. They are not the complete calibration budget. This study does not test
unseen-user K=1/2/3/5, prospective drift, or pressure/motion robustness.

## Reporting and interpretation

Recompute Overall, MIMIC and VitalDB separately on identical query IDs.
Participant-macro SBP, DBP and mean MAE are primary. Include the requested
window-pooled diagnostic columns: Setting, BP, MAE, R², ME, STD, ≤5, ≤10, ≤15,
AAMI and BHS. Numerical AAMI-style/BHS-grade checks are not clinical device
certification. No cherry-picked best-per-source combination is permitted.

The old 0.15-mmHg dual-split promotion gate remains an old experiment rule;
it is not an official PulseDB requirement and is not rewritten by this batch.
Prior E1/v1 gains justify retaining candidates, not presuming official gains.
Report uncertainty and all candidates, including negative results. Multiple
whole-pipeline seeds and temporal/external confirmation remain later work.

## Execution and resource limits

### Recovery amendment, 2026-09-08

The six methods, optimizer budgets, official membership and internal split seed
remain unchanged. The initial preparation failed before any formal GPU fit.
Its logs and immutable code snapshot are retained, not overwritten.

The recovered implementation uses `recorded-sample-span-v1`: the right endpoint
is the recorded last sample `T[-1]` (`end_time_s`), not `T[-1] + dt`. Four original
cross-role pairs have identical end/start timestamps and unequal PPG boundary
values and whole raw arrays. These are recorded as touching boundaries, not
claimed to prove physiological independence. Positive recorded-span overlap
above `1e-7` s still fails, including an actual one-sample overlap. Preparation,
inner/OOF training and memory-cache/retrieval metadata share this contract.
Historical protocol code and historical results are not reinterpreted.

The original Info file hashes are checked on every preparation. Independently
decoded identity-only membership caches may be reused only with hard-pinned
SHA-256 identities, the official name/index conversion, redundant identity
agreement and the exact 2506-person 360/40 count checks. No test BP is read by
the metadata preflight, copying step or PPG materializer.

Only the required 2506 raw MAT files are copied from the NAS master to work:
495,585,295,746 bytes in total (approximately 461.55 GiB). The original ten hot
pilot files are checked and reused where applicable. Copying is single-stream,
checks raw-file hashes against the full-cohort audit, read-back verifies the
new file and installs it without overwriting an existing target. NAS originals
remain unchanged. The official processed signal arrays contain only the
selected 1,002,400 windows (approximately 5.01 GB of float32 PPG).

CPU-only hot staging and metadata preflight precede full materialization. A
post-materialization audit exercises all inner/OOF/final metadata consumers
before GPU fitting. Exact content duplicates within or across official roles
remain explicit failures requiring inspection, never silent exclusions.

Use only hpc-2's RTX 5080 and RTX 5070 Ti, at most two allocated GPUs at once.
Data preparation is a two-CPU Slurm job with no GPU. Hot training inputs,
environment, logs and outputs remain under `~/work/ppg_bp`; durable artifacts
are copied to `~/nas/ppg_bp`. No local raw-data download is required.

Each batch uses an immutable code snapshot, explicit dependencies and unique
output paths. Smoke tests precede real fitting. Failed stages retain their logs
and stop dependent jobs; no existing user job is cancelled or replaced.
