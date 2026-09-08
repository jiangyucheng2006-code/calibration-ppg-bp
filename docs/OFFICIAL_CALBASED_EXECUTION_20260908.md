# Official CalBased execution receipt — 2026-09-08

## Recovery submission, verified at 15:24 CST

User authorized repairing preparation and resubmitting the **same six methods**.
No official membership, internal seed, model candidate or optimizer budget was
changed. The old failed batch remains intact.

Runtime revision: `d20264a14ee5`, immutable snapshot
`~/work/ppg_bp/code/official_calbased_d20264a`. Intermittent SSH resets prevented
the initial complete archive transfer. Recovery copied the existing immutable
0872d0f runtime into a new directory and applied a checksum-verified delta;
**all313 tracked runtime/source/test/script/plan files then matched the local
Git blob hashes**. The failed partial archive was never extracted or executed.
Verified delta SHA-256:
`591a64a2c7e7d3f9be5c1630b619a5e2ce0f50026fd837f3ec4c19931f3ec46c`.

Batch: `~/work/ppg_bp/outputs/official-calbased-v1_20260908-152000`.
Processed store: `~/work/ppg_bp/data/processed/pulsedb-official-calbased-v1_20260908-152000`.
Staging receipt: `~/work/ppg_bp/data/manifests/official_calbased_stage_20260908-152000.json`.
The batch's `submission.tsv` is byte-identical to its NAS mirror.

| Stage | Job | Status at inspection |
|---|---:|---|
| Complete snapshot smoke + real RTX5080 check | 1673 | COMPLETED0:0,103tests,zero skips |
| Exact2506-person raw-file hot staging | 1674 | RUNNING;102files,23.69GiB verified |
| Full identity/time metadata preflight | 1675 | COMPLETED0:0,82seconds |
| Materialization + post-prepare runtime audit | 1676 | PENDING after1674/1675 |
| Inner LoRA,320/40 | 1677 | PENDING dependency |
| Encoder crossfit fold0 | 1678 | PENDING dependency |
| Encoder crossfit fold1 | 1679 | PENDING dependency |
| Encoder crossfit fold2 | 1680 | PENDING dependency |
| Inner fixed memory | 1681 | PENDING dependency |
| Inner v1 relation | 1682 | PENDING dependency |
| Inner E1 relation | 1683 | PENDING dependency |
| Inner scalar trust | 1684 | PENDING dependency |
| Inner BP-specific trust | 1685 | PENDING dependency |
| Final full360 LoRA | 1686 | PENDING dependency |
| Final fixed memory | 1687 | PENDING dependency |
| Final v1 relation | 1688 | PENDING dependency |
| Final E1 relation | 1689 | PENDING dependency |
| Final scalar trust | 1690 | PENDING dependency |
| Final BP-specific trust | 1691 | PENDING dependency |
| One-way frozen official evaluation | 1692 | PENDING dependency |

There are six scientific candidates, not twenty independent model proposals.
Preparation, internal selection, three encoder crossfit fits and final scoring
are required supporting stages. Only hpc-2 is used; at most one RTX5080 and one
RTX5070Ti job may run concurrently. Hot staging requested one CPU; the scheduler
allocated two logical CPUs/8GB and noGPU. Pending jobs allocate noGPU.

### Evidence for this repair

- All103 server tests passed in9.038seconds, including synthetic rawMAT/shard,
  inner/OOF/final neural fits, trust heads, frozen scoring and target-access
  traps. Actual RTX5080 forward/backward also passed. These do not constitute
  completed real-cohort training.
- Official Info originals and pinned identity-only caches verified; exact
  counts remain2506 people,902160train/100240test,360/40 perperson.
- The full metadata preflight reports official cross-role overlap0/touch4,
  inner train/validation overlap0/touch7, OOF overlap0/touch1. All consumers
  now use actual last-sample endpoints under`recorded-sample-span-v1` with
  tolerance1e-7s. Positive overlaps and exact signal duplicates still fail.
- Required2506 NAS MAT files exist and total495585295746bytes. The new stager
  copies/checksums only that whitelist; pre-existing hot files are verified,
  never silently replaced. No local raw-data download was made.
- Training targets are accessed only for exact official TRAIN IDs. The
  metadata preflight and raw-copy utility do not parse BP, and prepared test
  inputs remain target-free. Full materialization/content checks and the
  postprepare audit were still pending at this inspection.
- Interrupted submission attempts were checked for absence of the new batch
  directory and queued fits before retry. No duplicate model chain was sent.
  Task-scoped SSH options`IPQoS=none` and`KexAlgorithms=curve25519-sha256`
  succeeded on the final submission; this is an observed workaround, not a
  proven network root cause. No global SSH/security configuration was changed.

Local execution/status documents and Skill progress were updated. The runtime
commit and this receipt have not been pushed to GitHub in this recovery turn.
No new official model results are available yet.

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
