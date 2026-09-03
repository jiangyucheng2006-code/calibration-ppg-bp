# Round-10 aggregate results

These files are public aggregate outputs from the K=5, single-seed Round-10
internal fold-4 screen. They contain no raw PPG waveforms, participant
identifiers, event-level predictions, checkpoints, private server paths, or
locked-test results.

- `participant_macro_internal.csv`: primary SBP, DBP, and mean participant-
  macro MAE for every setting in Overall, MIMIC, and VitalDB.
- `comparison_vs_reference_internal.csv`: change in mean participant-macro MAE
  relative to T10-0; negative candidate-minus-reference values are better.
- `pooled_diagnostics_internal.csv`: secondary event-pooled MAE, R², ME, STD,
  percentages within 5/10/15 mmHg, and qualified AAMI/BHS numerical screens.
- `selection.json`: prespecified internal winner and promotion-gate decision.

MIMIC and VitalDB are PulseDB source strata, not independent external
validation datasets. Overall values are recomputed from all eligible rows and
are not averages of source-specific metrics.

See the [full result report](../../docs/RESULTS_ROUND10_PARTIAL_END_TO_END.md)
and the [prospective plan](../../docs/ROUND10_PARTIAL_END_TO_END_PLAN.md).
