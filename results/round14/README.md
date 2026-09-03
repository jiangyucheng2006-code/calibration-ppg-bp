# Round 14 aggregate results

This directory contains the aggregate, development-only tables supporting
[the Round-14 result report](../../docs/RESULTS_ROUND14_CONFIRMATION_AND_COMPLETE_METHODS.md).

- `confirmation/` contains the paired compact-ResNet versus wider-
  InceptionTime QGH seed-confirmation summaries.
- `complete_methods/` contains the same-seed complete-method comparison,
  promotion gates, and retrospective pooled diagnostics.

Overall, MIMIC, and VitalDB rows come from the same saved predictions. MIMIC
and VitalDB are internal PulseDB source strata, not external validation
datasets. Participant-macro MAE is the primary endpoint; event-pooled AAMI-
style and historical BHS fields are retrospective numerical screens only.

No raw PPG, participant identifiers, event-level predictions, checkpoints,
private tail-membership records, server paths, or execution logs are included.
Meta-validation and the locked meta-test were not accessed.
