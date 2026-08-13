# Models and baselines

## Notation

For query PPG `x_q` and support set
`S = {(x_i, y_i)} for i=1..K`:

- `E(x)` is the shared PPG encoder representation;
- `f_pop(x)` is the frozen population SBP/DBP prediction;
- `y_i` is the support reference `[SBP, DBP]`;
- `r_i = y_i - f_pop(x_i)` is the support residual;
- all BP equations are implemented in the population model's standardized
  target space and converted back to mmHg for reporting.

## Required baseline ladder

| ID | Method | Calibration information | Purpose |
|---|---|---|---|
| B0 | Population BP mean | none | Non-signal floor |
| B1 | Last-cuff persistence | most recent support BP | Strong trivial temporal anchor |
| B2 | Support-BP mean | K support BP pairs | Simple multi-event anchor |
| B3 | Population multiscale 1-D ResNet | query PPG only | Calibration-free neural reference |
| B4 | Residual-offset correction | support PPG/BP plus frozen B3 | Strong simple PPG-aware calibration |
| B5 | Single-anchor Siamese delta | first support PPG/BP plus query PPG | Literature-aligned K=1 delta comparator |
| B6 | Head-only fine-tuning | K support PPG/BP pairs | Architecture-matched transfer baseline |
| B7 | Full fine-tuning | K support PPG/BP pairs | Small-K overfitting control |
| B8 | LoRA fine-tuning | K support PPG/BP pairs | Parameter-efficient adaptation comparator |

B0 and B3 are calibration-independent. B1, B2, B4, and B6--B8 are evaluated
separately for `K=1,2,3,5`. B5 is currently the direct K=1 first-anchor
comparator; a prespecified multi-anchor average can be reported as a secondary
extension.

## M0: unweighted variable-K residual anchor

For each support event:

```text
r_i = y_i - f_pop(x_i)
b   = mean_i(r_i)
```

Support tokens contain the support representation, support BP, and support
residual. A permutation-invariant mean produces `c(S)`. The query prediction is:

```text
y_hat_q = f_pop(x_q) + b + g(E(x_q), c(S))
```

The learned correction head `g` is zero-initialized so training begins at the
residual-offset baseline rather than an arbitrary personalized predictor.

## M1: support-conditioned FiLM

M1 adds one-pass, parameter-efficient feature modulation:

```text
gamma, beta = h(c(S))
z_tilde_q   = (1 + gamma) * E(x_q) + beta
```

The remaining correction is predicted from the modulated query representation
and support context. No per-user gradient optimization is required at routine
inference.

## M2: query-conditioned support reliability

M2 replaces equal support weighting with masked query-conditioned weights:

```text
alpha_i = softmax_i a(E(x_q), E(x_i), r_i)
b_q     = sum_i alpha_i r_i
c_q     = sum_i alpha_i token(E(x_i), y_i, r_i)
```

The weight calculation may use query PPG and support PPG/BP but cannot use query
BP, query error, future events, ABP-derived query features, or test-time quality
thresholds fitted on query labels.

## Main ablations

Use the same backbone, target scaling, split, event protocol, early-stopping
rule, and participant-macro evaluator wherever applicable:

1. query PPG only;
2. support BP only;
3. residual offset without a learned correction;
4. absolute regression versus anchor/delta prediction;
5. mean support pooling versus query-conditioned weighting;
6. without versus with FiLM;
7. fixed-K versus variable-K training;
8. head-only versus full versus LoRA adaptation;
9. PPG-only versus exploratory PPG+VPG input;
10. base versus robustness-enhanced training, only after the base-model gate.

## Robustness extension

The later M3 phase keeps the selected base architecture fixed and changes the
training procedure through condition-balanced episodic sampling and one
prespecified consistency objective. Clean error, stressed error, degradation,
and worst-group performance must all be reported. Auxiliary condition labels
used during training do not automatically imply that an auxiliary sensor is
available at inference.
