# Phase-6D: risk identification, difficult-case specialists, and complete routing

Status: completed. See [the verified result report](RESULTS_PHASE6D_RISK_ROUTING.md).

## Objective

Phase-6D tests whether the difficult participant tail can be recognized from
information available at inference time, whether a specialist improves those
cases, and whether a complete automatic routing pipeline improves full-cohort
performance. This is a single-seed development screen at `K=5`; it is not a
locked-test or external-validation result.

## Leakage-safe construction

1. Split the 3,143 eligible meta-train participants into the existing five
   participant-disjoint, source-stratified folds.
2. For each held-out fold, train/evaluate an M0 model with the current
   Quality Gate + Huber configuration without training on that fold.
3. Within each source, label exactly the worst 30% of participants from these
   out-of-fold K=5 errors. Meta-validation and locked meta-test outcomes are
   not used to create training labels.
4. Train the risk classifier from deployment-visible calibration and PPG
   features only. Query BP, query error, source, participant identity, and
   locked-test information are excluded.
5. Freeze risk thresholds on meta-train cross-fitting outputs before scoring
   meta-validation.

## Candidate specialists

- `QGHuber-weight2`: train on all meta-train participants and give cross-fitted
  difficult participants twofold sampling weight.
- `QGHuber-weight4`: same, with fourfold sampling weight.
- `QGHuber-hard-only`: train only on cross-fitted difficult participants.

All three retain fixed-first support, K=5, Quality Gate, Huber loss, the same
population checkpoint, early stopping with patience 8, and no epoch cap.

## Evaluation layers

The report separates four questions that must not be conflated:

1. **Identification:** AUPRC, AUROC, precision, recall, specificity, F1, and
   balanced accuracy for detecting the evaluation-only meta-validation worst
   30%, with Overall, MIMIC, and VitalDB views.
2. **Specialist value:** specialist versus general-model error on the fixed
   true difficult tail. This is an oracle diagnostic, not a deployable route.
3. **Deployable complete pipeline:** the current-event input-only risk score
   selects the general Quality Gate + Huber model or a specialist before the
   query BP label is read.
4. **Upper-bound diagnostics:** participant-aggregated, fixed-top-30%, and
   true-error oracle routes are clearly labelled retrospective or oracle-only.

Participant-macro SBP/DBP/mean MAE is primary. Pooled MAE, R2, ME, STD,
within-5/10/15-mmHg percentages, and AAMI/BHS numerical screens are secondary.
Every result is recomputed separately for Overall, MIMIC, and VitalDB.

## Promotion gate

A route is only a provisional winner if it improves full-cohort
participant-macro MAE over Quality Gate + Huber, improves or at least does not
materially worsen the predicted-high-risk group, and avoids a material loss in
either source. A positive result must then be confirmed with independent
training seeds before any architecture freeze. The locked meta-test remains
quarantined throughout Phase-6D.
