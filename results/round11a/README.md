# Round-11A public aggregate results

These files are exact copies of the accepted job-1048 aggregate report
artifacts. They contain no participant identifiers, event-level predictions,
waveforms, checkpoints, server logs, or private paths.

| File | SHA-256 |
| --- | --- |
| `participant_macro_internal.csv` | `766ea83e29837fafe3d843ad67f78b0539e89f5c08a4fa18eac2f1dcd40230a7` |
| `pooled_diagnostics_internal.csv` | `f12b6f94c3d2cbbf32eaf7ce8b9204177eac3b549cccff2d855de6f68fc39cd8` |
| `comparison_vs_reference_internal.csv` | `9066ccb132bee904a3887058881073fca31c950b80a842908c23582cfcaa5fcd` |
| `model_complexity.csv` | `2f09b51511c1b56b42d3ccf12b97b5d59ede3ca0d6847af50ea021f83533b616` |
| `selection.json` | `d18385a12d15ae7d5746f71005edc3588a268fbaa7898a0d1d6bec7560933304` |

The comparison is K=5 on meta-train internal fold 4 after folds 0--2 fitting
and fold-3 early stopping. PulseDB MIMIC and VitalDB are internal source
strata. Meta-validation and the locked meta-test were not accessed.
