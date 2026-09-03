# Round 11: systematic model revision

## Decision

Round 11 replaces unconstrained module stacking with three ordered questions:

1. Does a stronger PPG backbone improve the unchanged calibrated baseline?
2. Which personalization components are independently useful, and which should be removed?
3. Does an explicit population + stable personal bias + dynamic change decomposition improve tracking?

The stages are sequential. A later stage is launched only after the preceding stage has produced a valid internal result. This prevents architecture, calibration and loss changes from being confounded in one comparison.

## Frozen protocol

- Dataset: frozen `event120-v1` PulseDB development cohort.
- Calibration: the first five independent events of each participant.
- Query: event 6 and all later eligible events.
- Fit/early-stop/rank: meta-train folds 0--2 / fold 3 / fold 4.
- Development screen: one fixed seed; patience 8; no epoch cap.
- Primary metric: participant-macro SBP, DBP and mean MAE.
- Reporting: Overall, PulseDB MIMIC and PulseDB VitalDB from the same predictions.
- The meta-validation and locked meta-test sets remain inaccessible during screening.

## Round 11A: backbone screen

All candidates use a 256-dimensional PPG representation and the same Quality Gate + Huber personalized head. Only the PPG encoder changes.

| Candidate | Question answered | Main temporal bias |
|---|---|---|
| `resnet_small` | Exact architecture reference | compact multiscale residual CNN |
| `resnet_deep` | Is the old network merely under-capacity? | deeper/wider residual CNN |
| `inception_time` | Are explicit long and short convolutional scales useful? | parallel multiscale convolutions |
| `patch_transformer` | Does global interaction between PPG patches help? | patch tokens + self-attention |
| `conformer` | Is combined local pulse morphology and global context useful? | convolution + self-attention |

The screen also records parameter counts. A candidate advances only if its Overall mean participant-macro MAE improves by at least 0.15 mmHg and neither internal source stratum deteriorates. Complexity is not itself evidence of improvement.

## Round 11B: subtractive ablation

This stage uses only the winning backbone from 11A. Its first part is a
`2 x 2 x 2` structural factorial under the currently preferred Huber loss:

- learned personal correction MLP: off versus on;
- PPG-only quality gate: off versus on;
- query-conditioned support attention: off versus on.

The structural winner is then compared with MSE versus Huber as one paired
loss ablation. This hierarchical design tests attention explicitly without
turning every loss/module combination into an unconstrained search. The
minimum necessary model is selected using fold 4. An added component is
retained only when it improves the primary mean MAE and does not create a
clinically important source-specific regression. Other historical modules are
not automatically restored; prior weak candidates remain excluded unless
tested again as a single controlled factor.

## Round 11C: three-part BP decomposition

For query event `q` and support set `S`, the proposed prediction is

`BP_hat(q) = U(q) + B(S) + D(q, S, H_<q)`.

- `U(q)`: universal PPG-to-BP relation learned across participants;
- `B(S)`: participant-stable offset estimated only from the calibration events;
- `D(q, S, H_<q)`: change relative to calibration, using current/past PPG information but never the query BP or future events.

The controlled comparison is `U`, `U+B`, `U+D`, and `U+B+D`. `B` is tested first as an analytical residual mean/median and then, only if justified, as a learned shrinkage estimator. `D` predicts delta BP relative to the support anchor and receives an auxiliary direction/range loss only in a separate ablation.

Besides the primary MAE, the report separates:

- between-participant bias: participant mean signed error;
- within-participant tracking: MAE after participant-wise centering;
- performance versus support-to-query BP change and time distance.

This distinction tests the senior observation directly: whether errors arise mainly from a stable cross-person offset or from failure to track large within-person change.

## Evidence informing the design

- InceptionTime introduced a strong multiscale convolutional baseline for time-series classification: <https://arxiv.org/abs/1909.04939>.
- PatchTST motivates patch tokens for retaining local semantics while reducing attention cost: <https://openreview.net/forum?id=Jbdc0vTOcol>.
- ModernTCN shows that modernized pure convolution remains competitive for long time series, so Transformer is not assumed to win automatically: <https://openreview.net/forum?id=vpJMJerXHU>.
- A PPG-BP benchmarking study reported strong ResNet/Inception-family performance and explicitly compared calibrated protocols; its absolute results are not directly transferable to this split: <https://arxiv.org/abs/2502.19167>.
- A recent PPG BP architecture combined Conformer and Swin Transformer, supporting evaluation of local-plus-global modeling, but its data and evaluation protocol differ from this project: <https://pubmed.ncbi.nlm.nih.gov/41359694/>.
- Evaluation recommendations emphasize explicit calibration and leakage-safe subject/session reporting: <https://www.nature.com/articles/s43856-024-00555-2>.

These papers justify candidates, not expected results. The project-specific winner must be determined by the frozen comparison above.
