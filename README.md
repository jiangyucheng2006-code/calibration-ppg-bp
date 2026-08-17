# Calibration PPG BP

Research code for leakage-safe, event-level, few-shot personalization of
photoplethysmography (PPG) blood-pressure models.

The primary experiment asks whether a PPG-aware personalized neural model can
outperform strong, simple calibration controls when each new participant
provides only `K = 1, 2, 3, or 5` independent reference-BP events.

> This repository contains research code, tests, configurations, and Slurm
> scripts. It does not contain PulseDB waveforms, participant data, trained
> checkpoints, or clinical software. The project is not a validated medical
> device and makes no clinical-use claim.

## Study design

- Dataset role: PulseDB v2 for population/meta-training, meta-validation, and a
  quarantined internal meta-test.
- Calibration unit: one temporally separated, ABP-labelled representative event
  (a PulseDB pseudo-cuff event), never an adjacent waveform window.
- Primary calibration budgets: `K = 1, 2, 3, 5`.
- Frozen event protocol: `event120-v1`, using 120-second temporal blocks.
- Fair K comparison: events 1--5 form the support-candidate pool; every K uses
  the identical query set beginning at event 6.
- Split policy: participants are disjoint across meta-train, meta-validation,
  and locked meta-test.
- Primary reporting unit: participant-macro error. Pooled event-level error is
  secondary.

```text
events:  1  2  3  4  5 | 6  7  8 ...
K=1:     S  U  U  U  U | Q  Q  Q ...
K=2:     S  S  U  U  U | Q  Q  Q ...
K=3:     S  S  S  U  U | Q  Q  Q ...
K=5:     S  S  S  S  S | Q  Q  Q ...
```

`S` is a calibration support event, `U` is unused for that K, and `Q` is a
future query. Query BP is evaluator-only and cannot be used for adaptation,
preprocessing selection, early stopping, or model selection.

## Model and baseline matrix

The development comparison contains:

- population BP mean, last-cuff persistence, and support-BP mean;
- a calibration-free multiscale 1-D ResNet population model;
- population residual-offset correction;
- single-anchor Siamese delta prediction;
- head-only, full-network, and LoRA personal fine-tuning;
- `M0`: unweighted variable-K residual anchoring;
- `M1`: M0 plus support-conditioned FiLM modulation;
- `M2`: M1 plus query-conditioned support reliability weighting.

The main research hypothesis concerns the complete event-level protocol and
the M0--M2 progression, not the novelty of FiLM, LoRA, residual correction, or
set pooling in isolation. Robustness to contact pressure, motion, and device or
acquisition shift is a later phase and begins only after the base personalized
model passes the calibration gate.

See [METHODS.md](docs/METHODS.md) for equations and allowed inputs,
[PROTOCOL.md](docs/PROTOCOL.md) for leakage controls, and
[STATUS.md](docs/STATUS.md) for the current verified development state. The
current Phase-5 development comparison is the [five-seed report](docs/RESULTS_PHASE5_REPEAT5.md),
with a [participant-macro table](results/phase5_repeat5_participant_macro.csv),
an [88-row extended summary](results/phase5_repeat5_extended_summary.csv), and
the [440 seed-specific diagnostic rows](results/phase5_repeat5_extended_metrics_by_seed.csv)
needed to reconstruct the summary. The original [single-seed snapshot](docs/RESULTS_PHASE5.md)
is retained as a historical development record. The complete [data-selection
and training-cohort report](docs/DATA_SELECTION_AND_TRAINING_COHORT.md)
documents why the experiment uses a leakage-safe subset of the original
segment rows.

## Repository layout

```text
config/                     immutable public acquisition/configuration metadata
docs/                       study protocol, model definitions, and status
results/                    public rounded development-result tables
scripts/                    data-audit, materialization, Slurm, and training tools
src/pulsedb_fewshot/        Python package
tests/                      synthetic and contract-level regression tests
requirements-train.txt      pinned PyTorch training requirement
pyproject.toml              package and test configuration
```

## Installation and tests

The data-contract and audit modules require Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

GPU training additionally requires the PyTorch build specified in
`requirements-train.txt`. The included Slurm scripts reflect the current lab
cluster and should be reviewed before use on another system.

## Data

PulseDB data are intentionally excluded. Obtain the dataset from the
[official PulseDB repository](https://github.com/pulselabteam/PulseDB) and
follow its source-specific terms. Keep raw data and high-frequency processed
arrays outside Git.

The server-side project convention is:

```text
~/work/ppg_bp/   active code, environment, processed data, logs, and checkpoints
~/nas/ppg_bp/    raw-data master and durable archives
```

Training reads from `~/work`; it does not train directly from NAS.

## Reproducibility boundary

Development hyperparameters and checkpoints are selected only on
meta-validation. Locked meta-test scoring must not be submitted until the final
configuration, comparator, seed policy, and evaluation script are frozen.
Saved predictions, run configuration, source snapshot hash, data-manifest hash,
checkpoint hash, environment, and participant-macro metrics are required for an
accepted experiment.

## Current maturity

The complete PulseDB audit, participant split, `event120-v1` construction,
label-isolated query artifacts, waveform materialization, GPU smoke test, and
five-seed Phase-5 development comparison are complete. M0 has the lowest mean
participant-macro MAE at every K. The repeated runs used no epoch cap and all
stopped through patience-8 early stopping, resolving the original convergence
concern. M0 and M1 remain close, so M0 is the parsimonious provisional finalist
rather than a conclusively superior model. No locked-test, robustness,
external-validation, or final-paper result is reported at this stage.

The current development-only error-tail diagnosis and isolated Phase-6
candidate plan are documented in
[docs/PHASE6_HIGH_ERROR_ANALYSIS.md](docs/PHASE6_HIGH_ERROR_ANALYSIS.md) and
[docs/PHASE6_EXPLORATION.md](docs/PHASE6_EXPLORATION.md).

## License and citation

No open-source license has yet been granted. The public repository is intended
for transparent research review; contact the repository owner before reusing or
redistributing the code. Dataset licenses remain separate and are not changed by
this repository.
