# Round-7 nine-route development screen

## Correction of scope

Round 7 is a new K=5 development experiment, not a relabeling of the completed
Phase-6E outputs. Six residual candidates are trained again under seed
`20260821`, and three deeper candidates implement a new waveform-phenotype and
temporal-learning system. The locked meta-test is not accessed.

| Route | New training |
|---|---|
| R7-1 | Linear OOF residual ridge |
| R7-2 | Residual MLP |
| R7-3 | Confidence-gated residual shrinkage |
| R7-4 | Difficult-participant 2x residual MLP |
| R7-5 | Causal feature-sequence GRU |
| R7-6 | Supervised residual mixture-of-experts |
| R7-7 | Two-stage waveform phenotype classifier plus hard-routed independent experts |
| R7-8 | The same independent phenotype experts with uncertainty-aware soft routing |
| R7-9 | Causal GRU over waveform embeddings plus calibration-visible features |

## Two-stage phenotype system

Stage 1 trains a nonlinear autoencoder on frozen PPG waveform embeddings using
meta-train folds 0--3 and fold 4 for early stopping. Its latent representation
is clustered into eight waveform phenotypes using meta-train only. A separate
router is then trained to recognize those pseudo-labels from waveform latent
features. Source, participant identity, query BP, true error, and future events
are not router inputs.

Stage 2 trains eight separate residual prediction networks. Each expert sees
only the meta-train events routed to its own phenotype and has its own feature
scaling, checkpoint, and fold-4 early stopping. R7-7 selects one expert; R7-8
combines expert predictions using router probabilities. New meta-validation
events are categorized by PPG before prediction. This differs from the earlier
joint Cluster-MoE because the router and each category expert are independently
trained artifacts.

R7-9 uses the newly learned waveform latent sequence directly in a causal GRU,
alongside calibration-visible features. It tests whether the weak temporal gain
seen previously becomes larger when the network receives waveform morphology
rather than only summary statistics.

## Leakage and promotion gate

- Reference: K=5 fixed-first Quality Gate + Huber, job 841.
- Residual labels: participant-disjoint Quality Gate + Huber meta-train OOF
  predictions only.
- Candidate fitting: OOF folds 0--3; internal early stopping: OOF fold 4.
- Evaluation: meta-validation once after freezing.
- Locked meta-test: inaccessible.
- Promotion: Overall participant-macro mean MAE must improve by at least
  0.15 mmHg and both MIMIC and VitalDB must improve.

## Submitted jobs

The submitted chain is jobs 932--942. Jobs 932--937 are the six new residual
trainings; job 938 trains the phenotype representation/router; jobs 939/940
train hard/soft independent phenotype specialists; job 941 trains the deeper
waveform-embedding causal GRU; job 942 generates the unified
Overall/MIMIC/VitalDB report. At the first verified snapshot, job 932 had
completed, job 933 was running with empty stderr, and the remaining jobs had
the intended resource or dependency state.
