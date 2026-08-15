# Leakage-safe repeat-seed development plan

Last updated: 2026-08-14.

## Scientific boundary

- Split used for training and selection: `meta_train` and `meta_validation`
  only.
- Locked meta-test access: prohibited.
- Calibration protocol: frozen `event120-v1`, with K = 1, 2, 3, and 5 and the
  common future query set beginning at event 6.
- Target scaler: fitted from `meta_train` rows only; a regression test verifies
  that extreme `meta_validation` values cannot change it.
- Seeds are prespecified as `20260813`, `20260814`, `20260815`, `20260816`, and
  `20260817` before the repeat results are observed.

## Training rule

The repeat suite passes `--epochs 0 --patience 8`. Zero is defined by the
training CLI as no epoch-count cap. Training stops after eight consecutive
meta-validation epochs without a strictly lower participant-macro mean MAE.
The Slurm wall-time remains a safety and resource-allocation boundary; a job
that reaches it is a failed/incomplete run and cannot enter the aggregate.

## Three personalized training tasks

For each seed, a new population backbone is trained first. Its checkpoint then
feeds three independent personalized runs:

1. M0: variable-K unweighted residual anchor;
2. M1: M0 plus FiLM support modulation;
3. M2: M1 plus query-conditioned reliability weighting.

The same population checkpoint also feeds the calibration-control evaluator,
which recomputes population mean, last cuff, support mean, residual offset,
head-only, full fine-tuning, and LoRA for that seed. This prevents neural
backbone seed variation from being hidden behind one fixed population model.

## Job and reporting gates

- A short GPU smoke job must complete before population training starts.
- Every downstream job uses `afterok` on its matching population job.
- The CPU aggregation job starts only if all five-seed trained models and
  calibration controls complete successfully.
- Aggregation requires each artifact to state `split=meta_validation` and
  `locked_test_accessed=false`.
- Per-seed metrics and mean/SD summaries are archived to NAS; event-level
  predictions and checkpoints remain private and are not committed to Git.
