# Personal memory v2: execution record

Latest verification: 2026-09-08 12:35 CST. All11 jobs completed with exit0:0;
the user's queue is empty. The four fits completed normally by early stopping.
No new jobs were submitted during this result inspection. See
[the complete result report](RESULTS_PERSONAL_MEMORY_V2.md).

| Fit | Job | Selected / completed epochs | Elapsed |
|---|---:|---:|---|
| E1 random | 1640 | 0 / 8 | 00:04:25 |
| E2 random | 1641 | 0 / 8 | 00:03:56 |
| E1 chronological | 1643 | 8 / 16 | 00:03:45 |
| E2 chronological | 1644 | 3 / 11 | 00:03:02 |

The sections below retain the original submission snapshot verified at
2026-09-07 21:18 CST; statements that smoke was running are historical.

## Scientific scope

See [the fixed experiment plan](PLAN_PERSONAL_MEMORY_V2.md). E1 and E2 each
have one fit in each split mode: four new fits total. E3 is a frozen reliability
rule evaluated with and without the existing relation correction; it is not a
trained error gate. All outputs remain development-only and full-coverage.
No held-out evaluation, automatic model promotion or extra architecture sweep.

## Code and recovery

- Initial source commit: `8ff89d077eaed95c6a4062fb44476293f4cb82c7`.
- Initial smoke jobs 1625 and 1627 failed before fitting. Empty retained-tail
  diagnostics could not be serialized to strict JSON, and CPU tests exposed
  shared NumPy/Torch array storage. The fixes do not change the data contract,
  model settings or outcome thresholds.
- The nine dependent jobs 1626, 1628–1635 were verified PENDING and cancelled.
  No running model fit was cancelled. Initial logs and snapshots are retained.
- Recovery source commit: `df2fba9d1f46836baedeb7352177318c49f82356`.
- Recovery code archive SHA-256:
  `8bfbd36e120496de009293d00396747cd149209500efc853a3da2e9bdb59534e`.
  The upload was hash-verified before extraction. The snapshot is read-only;
  work and NAS archive copies were verified byte-identical.
- Local checks: 111 passed, 34 explicitly skipped (local PyTorch unavailable).
  Both recovery server smoke jobs subsequently passed all 145 unit tests.
  Real-data/GPU smoke remains in progress at this receipt's timestamp.

## Recovery job receipt

Batch: `personal-memory-v2_20260907-131749` (UTC timestamp).

| Stage | Random-disjoint | Chronological-blocked | Dependency |
|---|---:|---:|---|
| Unit, synthetic and real-data smoke | 1636 | 1638 | None |
| Frozen time controls and E3 | 1637 | 1639 | Respective smoke succeeds |
| E1 matched relation training | 1640 | 1643 | Both diagnostic jobs succeed |
| E2 BP-state retrieval training | 1641 | 1644 | Both diagnostic jobs succeed |
| Per-mode report | 1642 | 1645 | Respective E1 and E2 succeed |
| Combined report | 1646 | 1646 | Both per-mode reports succeed |

The combined report is one job, not two. Total: 11 jobs. GPU stages request
only hpc-2: RTX 5080 for random-disjoint, RTX 5070 Ti for chronological-blocked.
At this inspection, smoke jobs 1636/1638 are RUNNING; all remaining jobs are
PENDING on dependencies. A failed prerequisite intentionally stops the chain.

Training uses seed 20260907, eight non-improving epochs for early stopping,
no epoch cap, and a 72-hour Slurm wall limit per fit. The scheduler wall limit
is not a promise of convergence or automatic restart. E3 has no fitting loop.

## Artifact locations

- Immutable code:
  `/home/jiangyu.cheng/work/ppg_bp/code_snapshots/personal_memory_v2_df2fba9`.
- Work outputs:
  `/home/jiangyu.cheng/work/ppg_bp/outputs/personal-memory-v2_20260907-131749_<mode>_<stage>`.
- Logs:
  `/home/jiangyu.cheng/work/ppg_bp/logs/personal_memory_v2_<job_id>.log`.
- Submission manifest:
  `/home/jiangyu.cheng/work/ppg_bp/outputs/submission_manifests/personal_memory_v2_20260907-131749.tsv`.
- NAS archive: corresponding paths beneath `/home/jiangyu.cheng/nas/ppg_bp`;
  output/log archiving runs on job exit. Current running output is on work.

Future reports must distinguish epoch-zero fixed blending, genuinely learned
improvements and temporal-neighbor effects, with Overall/MIMIC/VitalDB views.
Completion and scientific effectiveness are not established by submission or
by passing unit tests. This document is a dated snapshot, not a live monitor.
