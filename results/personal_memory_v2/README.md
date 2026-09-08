# Personal memory v2: public aggregates

Start with the [conclusion and interpretation](../../docs/RESULTS_PERSONAL_MEMORY_V2.md)
or the [complete requested result tables](RESULT_TABLES.md).

| File | Contents |
|---|---|
| participant_macro_summary.csv | 48 primary rows: 8 methods × 2 splits × 3 sources |
| event_pooled_diagnostics_all_scopes.csv | 96 secondary SBP/DBP rows, including R², ME, STD, thresholds and qualified AAMI/BHS labels |
| comparison_vs_reference.csv | Paired LoRA, fixed-blend and previous-v1 comparisons |
| promotion_gate.csv | All four new candidates fail the fixed joint upgrade rule |
| training_summary.csv | Epoch-zero, selected and best-trained metrics; four fits, three source views each |
| diagnostic_participant_macro.csv / diagnostic_pooled.csv | Frozen v1 temporal/reference-selection controls and fixed E3 evaluations |
| subgroup_diagnostics.csv | Scoring-only BP-deviation groups, not deployable selection or adjacent-time change |
| temporal_diagnostics.csv | Available reference-gap and common-fallback summaries; not exact temporal matching |
| verification.json | Input/output hashes, recorded data contracts, validation scope and metadata caveats |

Raw waveforms, participant identifiers, prediction-level tables and checkpoints
are deliberately absent. MIMIC/VitalDB are internal PulseDB source strata.
All primary queries are retained; there is no worst-30% exclusion in the main
tables. AAMI/BHS PASS labels are retrospective numerical screens, not device
certification or clinical validation.

The tables can be regenerated with NumPy and pandas using
`scripts/publish_personal_memory_v2_results.py --input-root <aggregate-directory> --output-dir <new-output-directory>`.
The publisher validates the 33 completed server aggregate artifacts and their
recorded hashes. It does not reload private predictions or redo training.
The original aggregates must be obtained through the authorized project data
workflow; they are not a hidden dependency on a public patient-data download.

Numeric leader and model promotion are different: E1 is best in chronological
development, E3 is best in random development, but neither is promoted.
See the report for epoch-zero selections, temporal dependence and negative
large-deviation subgroup findings.
