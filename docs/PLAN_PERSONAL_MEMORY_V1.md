# Personal memory: finite registered-user experiment

Date: 2026-09-07. This is an executable research plan, not a result or a novelty claim.
The publication question is **Personal Memory–Augmented Blood Pressure Estimation
from PPG in Registered Users**. Related work and the paper outline are in the
[research blueprint](PUBLICATION_RESEARCH_BLUEPRINT_20260907.md).

## Question and fixed reference

Can an explicit bank of one person's lawful training PPG/BP measurements add
useful local information to that person's already strong parametric LoRA model?
Use completed continued-LoRA checkpoints1500(random)/1507(chronological), not
the weaker old initialization or a newly retrained-from-scratch comparator.
Keep the source model, every personal rank4 A/B, train BP anchor and identity mapping.

Both `development-calbased-analogue-v1` modes retain2,051 participants:
1,011 MIMIC and1,040 VitalDB. Each has320 labelled training windows and40 internal
validation windows. Final held-out windows/targets remain sealed. A10s window
is not an independent cuff measurement. The retrieval count **m=5 is not K-shot**;
the personal training label budget remains320. This is not official CalBased reproduction.

## Stage1 — does memory provide complementary information?

Export actual post-LoRA256D features using a hook on the existing prediction head.
Verify every allowed raw PPG content hash and reproduce all saved82,040 validation
predictions: maximum absolute difference≤0.05 and mean difference≤0.005mmHg.
Use source GPU types, unchanged64 batch inference, and serial loading.

| Setting | Prediction | What it checks |
|---|---|---|
| D0 | Frozen continued LoRA | Strong starting reference |
| D1 | Same-person feature-top5 BP, cosine-softmax T=0.1 | Explicit measurement-memory information |
| D2 | D0 plus top5 training prediction residuals | Simple residual correction; explicitly in-sample, not OOF |
| D3 | Nearest-time legal training BP | Proximity/interpolation explanation |

Train query retrieval excludes its complete40-window time-sorted block within
its recording, not just itself. Before subject filtering, audit global role IDs,
PPG hashes and physiological interval overlap. Chronological donors must be earlier
on the same audited recording clock; no cross-record pseudo-time is invented.
Insufficient5-reference histories fall back to D0 and remain in every primary metric.

Freeze a60s additional exclusion-band sensitivity analysis. Report both sources,
PPG feature-distance quartiles (boundaries fitted on train only), and BP deviation
from personal train median:0–10,10–20,>20mmHg. Query BP defines these **scoring-only**
subgroups; it never selects a reference or routes a prediction.

The exploratory go/no-go rule is fixed before looking at these new results:
continue to neural experiments if D1 or D2 yields either an Overall mean gain≥0.02
in either mode, or gain≥0.15 in a predeclared source×BP-deviation×SBP/DBP cell with
at least100 people and500 queries. Otherwise downstream jobs record
`skipped_no_complementarity`. This deliberately permissive diagnostic gate is
**not a promotion criterion or statistical confirmation**; many development cells
are examined. No parameter is fitted from these validation subgroup errors.

## Stage2 — bounded neural comparisons

Five settings under two modes: at most10 model-training jobs, not a width/depth sweep.

| Candidate | Single main change |
|---|---|
| `lora_continued_control` | Further train the same strong source without memory |
| `pair_single` | One feature-selected personal reference + shared pairwise relation network |
| `pair_uniform` | Five deterministic, time-spread legal references + uniform averaging |
| `pair_retrieved` | Five feature-nearest legal references + fixed cosine-softmax weighting |
| `pair_distance_blend` | Same retrieved relation path, blended with D0 using a fixed PPG-distance rule |

Each pair network uses `[q,r,q-r,q*r]` →64 GELU→2 and evaluates both orders with
half their difference. This makes its predicted BP difference antisymmetric;
it is not a claim of physiological causality or a new algebraic principle.
The last layer is zero initialized. Target differences are normalized using the
unchanged source train BP standard deviation. Loss: normalized absolute Huber
plus0.25 times per-reference BP-difference Huber; delta0.5. All pair variants use
the same objective. Network output is converted to mmHg before evaluation.

### Explicit implementation refinement of the blueprint

This first screen uses **ordinary supervised relation learning on frozen features**,
not OOF errors. The encoder has already seen all train-role labels, so excluding
a retrieval block does not make those features or train residuals out-of-fold.
It is valid ordinary training but must not be described as unseen-error meta-learning.

There is **no learned error-predicting gate** in this batch. The distance fusion is
`clip(1 - nearest_cosine_distance / train_distance_q95, 0, 1)`, fixed from PPG-only
training distances. This avoids fitting a gate on in-sample errors or validation
labels. A future learned error/confidence gate would require the blueprint's
fully nested/encoder-cross-fitted lineage; this implementation does not supply it.
Distance is a representation-space proxy, not a guarantee of physiological coverage.

Pair networks train only shared relation weights; LoRA and the memory bank stay
frozen. Train rows without legal memory have no effective memory gradient.
Every validation query retains a prediction, including the fallback cases.
Personal memory adds approximately320KiB of256D float32 features/person plus
labels/metadata, not less storage than8KiB/person LoRA.

## Optimization, execution and selection

- Seed20260907 only; patience8 without an epoch cap; scheduler walltime72h remains
  an operational safety limit. No multiseed/fold expansion in this screening batch.
- Pair models: batch256, AdamW3e-4, weight decay1e-4, gradient clip5,
  200,000 sampled train queries/epoch. Full internal validation every epoch.
- Continued LoRA: unchanged source architecture, batch64, AdamW1e-5,
  weight decay1e-4, Huber0.5, same200,000 sampled queries/epoch, patience8.
- These are matched starting/data conditions, **not equal compute**: pair models
  reuse frozen cached features while LoRA processes raw waveforms; optimizer step
  counts and stopping epochs differ. Report both runtime and query exposures.
- GPU compute only on hpc-2: random usesRTX5080, chronological usesRTX5070Ti.
  Unit/real-data GPU smoke precedes export; both diagnostics precede neural training.
  Each output and immutable code bundle is archived to NAS; no raw/participant
  predictions or checkpoints are sent to public GitHub.
- Compare neural candidates with the stronger continued-LoRA reference, keeping
  all participants. Original promotion rule: Overall mean MAE improves≥0.15mmHg
  under **both** modes and each of the four source/mode comparisons is positive.
- Publish Overall/MIMIC/VitalDB independently, using the same predictions:
  Setting,BP,MAE,R²,ME,STD,≤5/10/15mmHg percentages,AAMI*,BHS*. Numeric screens
  are not clinical device certification. Final tests are not opened by this workflow.

Completion is not guaranteed improvement. If memory does not help, retain LoRA
and report the result rather than renaming an unsuccessful combination.
