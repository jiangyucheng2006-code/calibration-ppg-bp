# Official CalBased execution receipt — 2026-09-08

## Superseding inspection, 14:55 CST

Preparation1648 failed at14:13:59CST (exit1:0,27minutes). Its invalid-dependency
cascade automatically cancelled1652–1667 before any fit started. Smoke1651
remainsCOMPLETED. There are **zero newly trained official models or results**.

The exact checker reproduces4 timestamp-boundary conflicts, each0.008seconds,
affecting1MIMIC and3VitalDB participants. The new right edge is last sample time
plusdt, whereas the old audit separately counted touching last/first timestamps.
Raw T and PPG_Raw confirm equal boundary timestamps but unequal boundary values
and unequal whole arrays. No test BP was read during this diagnosis.

The specified work raw paths also contain only the original10pilotMAT files;
checked full-cohort files remain on NAS. A repair must handle both the boundary
contract consistently and stage required data to the hot workspace. No code or
partition changes, recovery submission or publication occurred in this status
inspection. The table below preserves the earlier14:05 submission snapshot.

## Original submission snapshot

Status inspected approximately14:05CST: submitted, awaiting completion of data
preparation. No new official benchmark scores are available or implied.

Code commit: `0872d0f` on `method/personal-feature-mechanisms`.
Archive SHA-256: `61218e7814d7e447d6444af3b5876d6588f16ca1253a10b4c5b4f3112f3a0e3e`.
Immutable server snapshot: `~/work/ppg_bp/code/official_calbased_0872d0f`.
Batch: `~/work/ppg_bp/outputs/official-calbased-v1_20260908-140300`.
Exact scheduler receipt: `submission.tsv` in that directory and its NAS mirror.

The [prespecified plan](PLAN_OFFICIAL_CALBASED_20260908.md) defines the six
methods and all access/budget boundaries. Seventeen pipeline stages plus the
data-preparation stage do not mean seventeen different candidate methods.

| Stage | Job | Resource | Verified status |
|---|---:|---|---|
| Official data preparation | 1648 | hpc-2, 2CPU, 16GB, noGPU | RUNNING |
| Full snapshot smoke | 1651 | RTX5080, 8GB hostRAM | COMPLETED, exit0:0 |
| Inner LoRA, 320/40 inside official TRAIN | 1652 | RTX5080 | PENDING dependency |
| Crossfit LoRA fold0 | 1653 | RTX5070Ti | PENDING dependency |
| Crossfit LoRA fold1 | 1654 | RTX5070Ti | PENDING dependency |
| Crossfit LoRA fold2 | 1655 | RTX5070Ti | PENDING dependency |
| Inner fixed memory | 1656 | RTX5080 | PENDING dependency |
| Inner v1 relation | 1657 | RTX5080 | PENDING dependency |
| Inner E1 relation | 1658 | RTX5080 | PENDING dependency |
| Inner scalar trust | 1659 | RTX5080 | PENDING dependency |
| Inner BP-specific trust | 1660 | RTX5070Ti | PENDING dependency |
| Final full360 LoRA | 1661 | RTX5080 | PENDING dependency |
| Final fixed memory | 1662 | RTX5080 | PENDING dependency |
| Final v1 relation | 1663 | RTX5080 | PENDING dependency |
| Final E1 relation | 1664 | RTX5070Ti | PENDING dependency |
| Final scalar trust | 1665 | RTX5080 | PENDING dependency |
| Final BP-specific trust | 1666 | RTX5070Ti | PENDING dependency |
| Frozen official evaluation, three source scopes | 1667 | 2CPU, noGPU | PENDING dependency |

All fits are restricted to hpc-2; no hpc-1 resources are requested. A pending
dependency allocates no GPU. Training starts only after its prerequisites
succeed. A failed prerequisite does not silently trigger relaxed data checks.

## Verification and retained failures

- Public official Info SHA-1 and local/remote SHA-256 identities verified.
  Identity-only counts exactly match2506×360/40; original private identity tables
  and raw files are not included in Git.
- Initial minimal preparation package omitted `events` and `splits` imported by
  the package initializer. Job1647 failed in2seconds before real data reading.
  Corrected immutable preparation snapshot/job1648 retained this failure.
- Smoke1649 tested the first complete implementation:65tests passed with no
  skips, and actual RTX5080 forward/backward passed.
- Final snapshot smoke1651: **78tests passed, zero skips**, unittest runtime
  8.720seconds; total scheduler elapsed17seconds. This includes all five memory
  methods' inner/final synthetic fits, scratchLoRA inner/OOF/final, rawMAT array
  ordering/hash checks, test-label traps, frozen prediction checks and scoring
  integration. GPU forward/backward passed separately in the same job.
- Source-record identity audit found6906 `(source,record_id)` groups and zero
  groups shared acrossdifferent subjects. No actual collision was repaired.
- Ordinary upstream software syntax and scientific correctness are distinct:
  these checks support implementation integrity, not a guarantee of model gains
  or of all real official windows passing the remaining full-cohort audits.
- Local Skill continuity and protocol amendment were updated. The Skill format
  checker could not execute locally because PyYAML is absent; unchanged YAML
  frontmatter and narrowly edited protocol text were inspected directly.

## Outputs when complete

Intermediate selection uses only the internal official-TRAIN validation role.
The final scorer writes `evaluate/RESULT_TABLES.md`, metrics and scoring receipts
after all six no-target prediction files are frozen. Overall, MIMIC and VitalDB
are recomputed separately. Test scores do not generate automatic new training.
Full checkpoints, personal state, source IDs and raw-level predictions stay on
the server/NAS; only appropriate aggregate reports should be published later.
