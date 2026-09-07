# Personal-memory execution receipt

Date: 2026-09-07. Screen: `personal-memory-v1`.

The completed prior results were published at commit `37748e7`.
The new finite protocol/code was frozen at `4788e06eff472f0e7f8b8f01fda1ed3e07c96e54`.
See [the executable plan](PLAN_PERSONAL_MEMORY_V1.md) for inference inputs, label
budget, safeguards, diagnostic go/no-go rule and unchanged promotion threshold.

## Current chain after reference-precision correction

The first export jobs1590/1592 stopped at a scoring-target representation check,
before any neural experiment ran. In each mode, only2 of164,080 SBP/DBP values
differed from raw MATLAB float64 by slightly more than2e-5mmHg after the old
loader's float32 standardization/de-standardization. Observed maxima were
0.0000217026(random)/0.0000224086(chronological). In the documented source
float32 representation, maximum difference is0.0000152588mmHg in both modes.
All82,040 source-model predictions in each mode replayed **exactly** (max/mean0).

Correction commit `9e7ff583884d299b52d3c6a9619db5665ffaa347` compares labels in
the original loader's float32 representation while retaining the2e-5 tolerance.
It does not change any label, window key, waveform, model or leakage rule.
Regression tests cover this rounding case and reject actual target changes.
Failed exports/logs and the first snapshot are retained. Only14 still-pending
superseded jobs1593–1606 were canceled after checking their ownership/state.

- Active immutable snapshot: `personal_memory_9e7ff58`.
- Archive SHA256: `9e1869cfb978cc3fa9b33d437bb8b527544324d4df8be6e2c67d92125a522cea`.
- Source-tree SHA256: `665158cd45c5d4f8fb0e2764b4acef62cdc81bb18d1380f256bc6b3a5b418f68`.
- Local/uploaded/NAS archives match. Replacement manifest:
  `personal_memory_20260907-084550.tsv`.
- New smoke1607/1609 completed0:0, all87 focused tests pass on both GPUs,
  followed by the real-data four-step check. Cache1608/1610 are running.
- Full replacement exports passed in117.7/116.6s:738,360 raw-window hashes
  verified per mode and all82,040 reference predictions replayed exactly,
  max/mean difference0. Both jobs have moved on to audited neighbour preparation
  and diagnostic scoring; the completed export is not yet a neural training result.

| Active stage/candidate | Random | Chronological |
|---|---:|---:|
| GPU smoke |1607 completed|1609 completed|
| Export + retrieval diagnostics |1608 running|1610 running|
| Continued LoRA |1612 queued|1618 queued|
| Single reference |1613 queued|1619 queued|
| Uniform five references |1614 queued|1620 queued|
| Feature-retrieved references |1615 queued|1621 queued|
| Distance blend |1616 queued|1622 queued|
| Mode result report |1617 queued|1623 queued|

Joint gate1611 precedes those10 training jobs; final report1624 follows both
mode reports. This replaces, rather than adds another architecture batch to,
the original18-job chain. Final full-data retrieval diagnostics remain pending
at this snapshot; no new performance claim is made.

## Original source and verification (superseded operational snapshot)

- Immutable server snapshot: `personal_memory_4788e06`.
- Source archive SHA256: `2e83cdb2674eb56bed3ddfdff63d3776c62aa1a62190d423f542fbda66230b47`.
- Extracted source-tree SHA256: `acf9430f67b8d699457e22538a123c560bc2ee52b1ca30919c1b8fdfbf5e254c`.
- Local/uploaded archive hashes match; work/NAS archives are byte-identical.
  Snapshot is read-only. Shell syntax checks pass.
- Local check:85 collected tests,71 passed and14 neural tests explicitly skipped
  because local PyTorch is absent. Server smoke then ran all85 successfully on
  each GPU; none of those server tests was reported as a skip.
- Real-data GPU checks1589/1591 both completed0:0 in17s:64 real train PPG windows,
  source checkpoint/personal state, raw waveform hashes, record-clock mapping,
  four finite optimizer steps, exact antisymmetry. Smoke weights are discarded.
- Source models remain continued-LoRA1500/1507. Full82,040-query reference replay
  is additionally required within each cache job; smoke is not a substitute.

## Original submission history (superseded)

Submission manifest: `personal_memory_20260907-083934.tsv`, archived to NAS.
All GPU assignments are restricted to hpc-2. Random usesRTX5080;
chronological usesRTX5070Ti. No hpc-1 GPU is requested.

| Stage/candidate | Random | Chronological |
|---|---:|---:|
| Unit + real GPU smoke |1589 completed|1591 completed|
| Frozen export + lawful retrieval diagnostics |1590 running|1592 running|
| Continued LoRA control |1594 queued|1600 queued|
| Single-reference relation |1595 queued|1601 queued|
| Uniform five-reference relation |1596 queued|1602 queued|
| Feature-retrieved relation |1597 queued|1603 queued|
| Distance-constrained blend |1598 queued|1604 queued|
| Full three-scope report |1599 queued|1605 queued|

Joint diagnostic go/no-go1593 waits for both caches. Final joint report1606
waits for both mode reports. **18 scheduler jobs include10 conditional training
jobs**, not18 architectures. Successful diagnostic stops produce explicit
`skipped_no_complementarity` records, not empty or fabricated results.

Status above is a checked submission snapshot, not a claim that all neural
training is already active or completed. No new neural-model MAE is available yet.
The scheduler continues independently of the local computer. Any new failure must
be diagnosed from its exact log; do not bypass data lineage checks or overwrite
the immutable snapshot to make dependent jobs start.

## Reporting boundary

Full participant coverage is retained; failures of memory eligibility revert
to the frozen LoRA prediction. Registered users still supply320 labelled training
windows; m=5 retrieval is not five-shot new-user calibration. Internal validation
is for development selection; held-out targets remain sealed. The neural branch
uses ordinary supervised frozen features, not OOF residual learning. No claim of
novelty, improvement, clinical certification or publication readiness is made.
