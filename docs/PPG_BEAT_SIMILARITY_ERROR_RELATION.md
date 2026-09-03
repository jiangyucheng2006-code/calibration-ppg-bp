# Beat-to-beat similarity and BP prediction error

## Question and decision

Does lower within-window beat-to-beat PPG morphology similarity identify larger
BP prediction error in the current calibrated models?

**Result: no.** The relationship is weak, non-monotonic, and often in the
opposite direction to the proposed quality interpretation. Beat similarity
should not currently be used to reject windows, route cases, or weight the BP
loss.

## Analysis boundary

- Development split: K=5 `meta_validation` only.
- Locked meta-test accessed: no.
- Identical queries: 103,564 windows from 697 participants.
- Models: the current Quality Gate + Huber reference and the Round-7 numerical
  winner, causal feature GRU.
- Primary analysis: participant median beat similarity versus participant mean
  BP error, Spearman correlation with a 5,000-repetition participant bootstrap.
- Secondary analysis: within the same participant, mean error in windows below
  versus at least 0.90 similarity. The 0.90 threshold is descriptive and is not
  a validated or deployable cutoff.
- MIMIC and VitalDB are internal PulseDB source strata, not external datasets.

## Participant-level association

Negative correlation would support the hypothesis that higher similarity
accompanies lower error. The observed correlations are instead small and
positive.

| Model | Scope | Participants | Spearman rho | 95% bootstrap CI |
|---|---|---:|---:|---:|
| Quality Gate + Huber | Overall | 697 | +0.1270 | [+0.0561, +0.1991] |
| Quality Gate + Huber | MIMIC | 316 | +0.0492 | [-0.0634, +0.1555] |
| Quality Gate + Huber | VitalDB | 381 | +0.1523 | [+0.0498, +0.2471] |
| R7-5 causal GRU | Overall | 697 | +0.1187 | [+0.0464, +0.1905] |
| R7-5 causal GRU | MIMIC | 316 | +0.0652 | [-0.0438, +0.1699] |
| R7-5 causal GRU | VitalDB | 381 | +0.1345 | [+0.0381, +0.2312] |

At the individual-window level, descriptive correlations between similarity and
mean absolute BP error are only +0.0288 Overall, +0.0054 in MIMIC, and +0.0516
in VitalDB for the reference model. The R7-5 values are similarly near zero.

## Within-participant comparison

This comparison controls stable differences between participants by including
only participants who have both lower- and higher-similarity windows.

| Model | Scope | Paired participants | MAE below 0.90 | MAE at least 0.90 | Low minus high | 95% bootstrap CI |
|---|---|---:|---:|---:|---:|---:|
| Quality Gate + Huber | Overall | 578 | 8.4251 | 8.5848 | -0.1597 | [-0.4745, +0.1571] |
| Quality Gate + Huber | MIMIC | 240 | 9.1857 | 9.1910 | -0.0053 | [-0.4875, +0.5044] |
| Quality Gate + Huber | VitalDB | 338 | 7.8850 | 8.1544 | -0.2694 | [-0.6705, +0.1601] |
| R7-5 causal GRU | Overall | 578 | 8.3347 | 8.4780 | -0.1434 | [-0.4583, +0.1745] |
| R7-5 causal GRU | MIMIC | 240 | 9.0234 | 9.0671 | -0.0437 | [-0.5306, +0.4531] |
| R7-5 causal GRU | VitalDB | 338 | 7.8457 | 8.0598 | -0.2141 | [-0.6061, +0.1910] |

The point estimates do not show larger error in low-similarity windows, and all
intervals include zero. Error also does not change monotonically across the
five prespecified descriptive similarity bins.

## Interpretation

This metric measures repetition of time- and amplitude-normalized beat shape.
It deliberately removes absolute amplitude and duration, and a repetitive
artefact can itself be highly similar. BP error may instead be driven more by
support-to-query BP change, event horizon, participant physiology, and domain
shift. A small positive participant-level association must not be interpreted
causally; it may reflect these confounders.

The result changes the immediate model recommendation. Do not add this single
similarity score to the Quality Gate, do not reject windows below 0.90, and do
not create a low-similarity specialist. Beat-level modeling remains a possible
representation experiment only if it includes richer information such as
amplitude, duration, derivatives, beat-interval variability, clipping, and
calibration-to-query morphology change, with each addition tested separately.
The causal temporal route remains the only repeatedly favorable direction, but
its gain is still below the promotion threshold.

## Reproducibility

- Slurm job 954 completed on `hpc-2` with exit code `0:0` and empty stderr.
- Work and NAS outputs are byte-identical.
- The project regression suite passed 76 tests before execution; the targeted
  post-fix suite passed 3 tests.
- Private event/participant joins remain on the server. Only aggregate results
  are public.

