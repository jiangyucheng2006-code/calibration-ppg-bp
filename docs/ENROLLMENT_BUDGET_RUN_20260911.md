# Enrollment budget execution record

Protocol: `enrollment-budget-v1`. Current status: **all stages completed**.
The [final report](../results/enrollment_budget_v1_20260911/README.md) contains
all eight budgets, both methods and the Overall/MIMIC/VitalDB result tables.

## Verified completion: 11 September 2026

Jobs 1783–1818 all completed with exit 0:0 on hpc-2. Final scorer 1818 ended
at 09:07:47 UTC / 17:07:47 China time. This is 32 completed personal-fitting
shards, preparation, two scorers and the successful smoke check. The current
user queue was empty at the result inspection; no stopped or unfinished arm
was inferred merely from an empty GPU display.

The scoring receipt retains exactly 762 final people and 78,237 query windows
in each of sixteen prediction sets. All predictions were frozen before target
access. A separate read-only audit at 09:56 UTC recomputed 48 primary summary
rows, 96 diagnostic rows and seven planned primary interval comparisons;
maximum numeric discrepancy was 1.42e-14. Final aggregate copies match work,
NAS and the local transferred files. The 90% predictions are byte-identical
to the previous parent 90% result, not an additional independent replication.

This publication adds reports, aggregate tables and verification receipts.
No new job, model change, data exclusion or protocol amendment was made.
The earlier status snapshots below are preserved as execution history.

## Verified live state: 11 September 2026, 15:33 China time

All jobs 1783–1813 completed with exit 0:0. All eight validation conditions
were scored together on the same 763 people / 76,909 queries; the receipt
confirms all sixteen prediction sets were frozen first and no score feedback
was passed to fitting. No validation metrics were analyzed in this status check.

Final personal runs for 20%–70% are complete, but the all-budget final scorer
has not run. At 15:33:15, 80% job 1814 had completed 356/381 profiles and job
1815 had completed 347/381. Both were RUNNING on hpc-2, with completed profile
reload/shared-state checks passing and no reported errors. The final 90%
jobs 1816/1817 and scorer 1818 are waiting on normal dependencies.

This is 28 completed, two running and two pending personal-fitting shards.
The approximate remaining duration is 1.5–2 hours based on current progress
and observed 90% validation runtimes, assuming no interruption. The final
scoring receipt is still absent; do not present completed predictions as final
MAE/AAMI/BHS results. No retraining, resubmission or protocol change was made.

## Verified live state: 11 September 2026, 02:15 China time

Preparation job 1784 completed successfully in 82 seconds. Jobs 1785 and 1786
started at 02:10:51 on hpc-2, with the requested RTX 5080 and RTX 5070 Ti
respectively. At 02:15:12, each had completed and reload-verified 53 personal
profiles (53/382 and 53/381); both jobs were RUNNING, with no restart or reported
error. These are the 20% validation arms, not final-test accuracy results.

The prepared plan SHA256 is
`49e023ba7a323baf7fd72cebed4bfb46e17ddd7c1e7b56f6f04fbc20366fb665`.
Its bytes match locally, in work and in NAS. The copied plan confirms all eight
budgets, exactly 763 validation and 762 final people, the original parent plan
and shared checkpoint, and unchanged final query identities. Source-level
query counts remain 54,664 MIMIC and 23,573 VitalDB at every final budget.

| Nominal budget | MIMIC median enrollment windows | VitalDB median enrollment windows | Final people using the one-group zero-adapter fallback |
| --- | --- | --- | --- |
| 20% | 207 | 100 | 50 |
| 30% | 311 | 150 | 37 |
| 40% | 415 | 201 | 23 |
| 50% | 519 | 251 | 14 |
| 60% | 622 | 301 | 14 |
| 70% | 726 | 352 | 0 |
| 80% | 830 | 402 | 0 |
| 90% | 934 | 453 | 0 |

The fallback counts are prespecified treatment of short histories, not removed
people; all their queries remain in evaluation. These substantial median counts
also show why even 20% must not be called a few independent cuff measurements.
Across the final cohort, the 20% bank contains 156,127 labeled windows and the
90% bank contains 703,869. Every arm predicts the same 78,237 query windows.

## Verified implementation and checks

- Runtime commit: `2b5dd222b7fb469a9f02853b2ab5edf4fc11de64`.
- Immutable snapshot: `code/enrollment-budget-2b5dd22` under the server project.
- Source-tree SHA256:
  `34c5f5b013f8964c1ee96b35f6ff088ee7aa3b789230b712a4d5174bd0c26fce`.
- Compressed source archive SHA256:
  `3850f50b424d234d9702d72c1795a106ec823dbb17b009f9d3e852ad350518ab`.
- Smoke job 1783: COMPLETED, exit 0:0, 1 min 39 s, RTX 5080 on hpc-2.
- 77 passing tests: 5 new budget, 7 full-cohort, 11 legacy-enrollment,
  16 post-enrollment, 8 ablation, and 30 memory-preparation tests.
- GPU forward/backward and frozen shared-encoder gradient checks passed.
- Tests exercise all eight budgets and both cohorts end to end, fresh personal
  states, exact fixed-query coverage, one-group zero-step fallback, saved/reloaded
  equivalence, label-free allocation, rejection of predictions/errors, parent
  contract checks, scoring freezes and matched participant intervals.
- Local syntax, shell syntax, diff checks and updated Skill validation passed.

The first smoke attempt, job 1782 on snapshot `a455ba5`, failed an assertion
that predicted-BP metadata must be rejected explicitly. Such fields were not
used for selection; the missing defensive rejection was corrected before any
formal budget job was submitted. Redundant inner-role metadata is also updated
for each new budget. The original failed smoke log/snapshot is retained. This
was a synthetic test failure, not a failed clinical/data result.

## Finite queue

Run directory:
`outputs/enrollment-budget-v1_20260910-180706` under the server project.

| Budget | Validation, 5080 | Validation, 5070 Ti | Final, 5080 | Final, 5070 Ti |
| --- | --- | --- | --- | --- |
| 20% | 1785 | 1786 | 1802 | 1803 |
| 30% | 1787 | 1788 | 1804 | 1805 |
| 40% | 1789 | 1790 | 1806 | 1807 |
| 50% | 1791 | 1792 | 1808 | 1809 |
| 60% | 1793 | 1794 | 1810 | 1811 |
| 70% | 1795 | 1796 | 1812 | 1813 |
| 80% | 1797 | 1798 | 1814 | 1815 |
| 90% | 1799 | 1800 | 1816 | 1817 |

Preparation: 1784, CPU-only on hpc-2. Validation freeze/scoring: 1801.
Final all-budget scoring: 1818. These three stages request no GPU. Each personal
job requests one named GPU, two CPUs and 12 GB RAM. The two serial GPU lanes
avoid oversubscribing either card. No job requests hpc-1.

There are **eight enrollment conditions, sixteen prediction settings, and
32 personal-fitting shards**, plus three preparation/scoring jobs and the
successful smoke check. A LoRA and a LoRA+memory result share the same trained
personal adapter. The different counts do not imply extra model variants.

Jobs 1784–1818 were accepted by Slurm. All later fitting/scoring stages depend
on successful prerequisites. Failure of a prerequisite does not permit a partial
all-arm comparison or automated new variants. No further manual submission is
needed for the prespecified successful path. No epoch cap or seed sweep was added.

## Outputs and provenance

The private `submission.tsv` and smoke receipt were copied locally under
`local_archive/enrollment_budget_v1_20260911/`. The runtime source archive,
submission table, completed stage outputs and logs are retained in the NAS
archive. Active waveform reads and profile writes remain in the work area.
The existing full-cohort store and shared checkpoint are reused without changes.

Completed final outputs under `test_score/`:

- `all_budgets_participant_macro.csv` and separate Overall/MIMIC/VitalDB views;
- `all_budgets_diagnostics.csv` and the three requested source diagnostic views;
- `p20/` through `p90/`, each with the full metric table and paired memory gains;
- `budget_paired_intervals.csv`, with same-person differences versus 90%;
- frozen-prediction and final-evaluation receipts.

Private `prepare/personal_budgets.parquet` records actual counts, unused rows,
rounding and fallback per person. `prepare/budget_summary.csv` contains the
corresponding source-level summaries. Query windows never become enrollment
rows just because a nominal budget is small.

At submission and during the earlier operational checks, only local project
documentation and the operating Skill were updated; no GitHub push was made
in those steps. The completed result-publication package is now linked above.
The prior full-cohort result is preserved unchanged. No raw-data transfer or
global-memory update is part of this result-publication task.
