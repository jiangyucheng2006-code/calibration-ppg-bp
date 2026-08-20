# Round-8 public result tables

These tables are reconstructed from the saved predictions of the completed
Round-8 K=5, single-seed `meta_validation` screen. They contain aggregate
results only; participant-level predictions and model checkpoints are not
published.

## Files

- `participant_macro_full_coverage.csv`: primary Overall, MIMIC, and VitalDB
  participant-macro SBP/DBP/mean MAE for all eight full-coverage settings.
- `diagnostic_full_coverage.csv`: secondary event-pooled MAE, R-squared, mean
  error, error standard deviation, cumulative absolute-error percentages, and
  retrospective AAMI/BHS-style numerical screens.
- `similarity_filter_coverage.csv`: retained query and participant coverage for
  the prespecified beat-similarity threshold.
- `similarity_filter_participant_macro.csv` and
  `similarity_filter_diagnostic.csv`: partial-coverage sensitivity results,
  reported separately from the full-coverage comparison.

MIMIC and VitalDB are internal PulseDB source strata. They are not independent
external validation datasets. The AAMI/BHS-style fields do not establish
formal standards compliance.

See
[`docs/RESULTS_ROUND8_CALIBRATION_RELATIVE.md`](../../docs/RESULTS_ROUND8_CALIBRATION_RELATIVE.md)
for the result interpretation, K coverage, and claim limits.
