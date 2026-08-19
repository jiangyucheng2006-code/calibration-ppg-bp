# Phase-6D aggregate results

This directory contains public aggregate tables for the K=5 development-only
risk-identification and difficult-case-specialist experiment. It intentionally
excludes participant identifiers, participant membership, event-level
predictions, checkpoints, raw data, and infrastructure paths.

- `risk_identification_metrics.csv`: frozen-threshold and fixed-coverage
  identification metrics for Overall, MIMIC, and VitalDB.
- `predicted_risk_group_errors.csv`: participant-macro error separation between
  predicted-low- and predicted-high-risk groups.
- `participant_macro.csv`: primary SBP/DBP/mean MAE for every specialist and
  routing diagnostic.
- `pooled_metrics.csv`: requested event-pooled MAE/R2/ME/STD/5-10-15/AAMI/BHS
  diagnostic table.
- `oracle_tail_comparison.csv`: evaluation-only fixed-tail specialist analysis.
- `paired_bootstrap.csv`: 20,000-repetition exploratory participant-cluster
  paired comparisons against the general Quality Gate + Huber model.

MIMIC and VitalDB are internal PulseDB source strata, not independent external
validation datasets. AAMI/BHS fields are numerical screens rather than formal
standards-compliance claims.
