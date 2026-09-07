# Personal feature screen: scoped recovery and mechanism interpretation

Date: 2026-09-07. This is an operational retry, not a new architecture search.

## Failed run and recovery boundary

Job1486 (`shared_bilinear64`, random_disjoint) failed after61 completed epochs
in DataLoader worker1 with `could not unlink the shared memory file ... No such
file or directory`. Its best checkpoint is epoch58 but has no optimizer/RNG
state, so a claim of exact interrupted resumption would be incorrect. The
underlying system trigger for the missing shared-memory file remains unknown.

Restart only this candidate from epoch zero, using original immutable source
`personal_feature_v1`, original seed20260906, model, training windows, batch64,
200000 sampled examples/epoch, optimizer, patience8/noepochcap and RTX5080.
Change only DataLoader workers4 to0 to avoid worker IPC/shared-memory collation.
CPU allocation remains4 and memory12G; do not claim bitwise identity with an
uninterrupted four-worker run. Input tensors/order for deterministic loading
are checked, and the retry's run.json records workers0 explicitly.

A separate15-minute GPU smoke checks real train-role sample collation,
four forward/backward steps, finite gradients, and exact checkpoint reload.
Smoke parameters are not reused for the formal retry. No held-out role opens.

Submit a replacement random report using the seven completed original runs
and the retry. Its afterok dependency waits for the retry. Submit a final
comparison depending on the new random report and reusing completed chrono
report1497. Cancel only obsolete pending reports1488/1498 after all replacement
jobs and dependencies are verified. Preserve failed logs/checkpoint and all25
completed training outputs. Active I/O stays in work, durable copies in NAS.

## Mechanism study versus output correction

The mechanism study and the later LoRA+PRS continuation are separate tests.
The primary scientific question remains what persistent person-specific
transformation of current PPG features contributes. PRS output correction was
an additional compensation hypothesis, not a substitute for that question.

Evidence currently available:

- Frozen same-person waveform permutations and personal-state exchanges
  degrade predictions. This supports dependence on matched waveform and
  personal state, but is not a causal decomposition or a trained ablation.
- Trained shared-LoRA versus personal-LoRA controls support the value of
  personal transformation in this registered-user setting. The capacity of
  personal storage also changes, so this is not a parameter-matched proof.
- Rank1 is worse than rank4 in the current single-seed screen. This supports
  testing sufficient personal adaptation capacity, not unlimited rank growth.
- Nonlinear rank4 did not outperform linear rank4; adding nonlinearity is
  not automatically useful. Compact shared-direction alternatives mostly
  trade accuracy for reduced per-person state, not an established upgrade.
- New PRS continuation improves relative to its old starting checkpoint but
  does not beat the paired continued-LoRA control. Additional optimization,
  rather than the new correction branch, explains the observed incremental
  comparison. This is still exploratory, not a definitive causal statement.

Consequently, keep identity-linked feature adaptation and current PPG input.
Do not describe adding output residuals as a demonstrated mechanism upgrade.
Further targeted choices (not submitted here) should isolate personal capacity,
adaptation location, or selective optimization of existing LoRA parameters,
with matched continued-LoRA controls and both split modes. Check previous
rank/location experiments before launching anything; no automatic larger sweep.
