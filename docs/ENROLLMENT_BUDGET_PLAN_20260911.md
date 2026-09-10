# Personal enrollment budget study

Date: 2026-09-11. Protocol: `enrollment-budget-v1`.

Material: original executable research plan, based on the completed full-cohort
enrollment experiment and the user's eight-budget request. Mode: implementation
and bounded server execution. Status at specification: prospective, before any
new budget results. This is exploratory work, not preregistered confirmation.

## Question and hypotheses

How much paired personal PPG/BP history is needed when an already trained shared
network is used to enroll a previously excluded person?

Two possible findings are specified before results: diminishing returns, or
continued error reduction with additional labeled history. Neither is promised.
Performance may also be nonmonotonic or worsen for some people. A plateau would
support a smaller practical enrollment budget only after examining the error
difference and its uncertainty. An increasing-data curve is not evidence that
elapsed use time, unlabelled PPG, or every individual update improves reliability.

## Fixed population and queries

Reuse the frozen shared network from `full-cohort-enrollment-v1_20260910-150200`.
The parent considered all 5,361 original people and retained 5,275 eligible
people: 3,750 population-training, 763 development-validation and 762 final-test
participants. The parent exclusions and exact assignments remain unchanged.
The previous result is retained separately, not overwritten.

The shared network was fitted without validation/test participants in its
gradient updates; validation did select its epoch. Test-person parameters start
fresh. This is subject-disjoint population fitting followed by supervised
personal enrollment, not calibration-free inference.

Reuse exactly 76,909 validation queries and 78,237 final queries, with the same
people, waveform identities and reference targets at every budget. The final
query-set SHA256 is
`fdec318023a1ff0cf2616c51e889eaea5d3f083cf58591d5097be3dcb5f7144a`.
The parent plan SHA256 is
`593bef54dfa758746c8e8645ac988881210150f29fcec2c6ff8a8b18ad54516a`.
The population checkpoint SHA256 is
`5340902ef5bf81c1f8a704380d3376d9b7e1b1df3d509175b0b89c7d494397ed`.

No new shared-backbone fitting or waveform download is required. Reusing this
verified subject-disjoint population model is intentional; reusing its trained
people's personal adapters, or a target person's 90% adapter at a lower budget,
is forbidden. The shared checkpoint and scaler are fixed across all arms.

## Eight nested budgets and two paired outputs

| Nominal enrollment | Fixed queries | Unused remainder, approximately |
| --- | --- | --- |
| 20% | Same parent 10% | 70% |
| 30% | Same parent 10% | 60% |
| 40% | Same parent 10% | 50% |
| 50% | Same parent 10% | 40% |
| 60% | Same parent 10% | 30% |
| 70% | Same parent 10% | 20% |
| 80% | Same parent 10% | 10% |
| 90% | Same parent 10% | 0% |

Percentages refer to all eligible windows of that participant, not a percentage
of the 90% registration pool. Whole content/positive-overlap groups remain
indivisible. A deterministic seeded group order, independent of BP and prediction
errors, creates nested subsets. The requested window count is the floor of the
nominal fraction times eligible windows, with minimum one and capped by the
parent registration count. Complete-group inclusion may overshoot. The 90% arm
uses the exact parent's registration rows, order and inner roles.

Each participant's requested/actual count, actual fraction, fixed query count,
unused count and any fallback are recorded. Therefore a percentage is not an
equal label budget across people, and 20% must not be described as a few cuff
measurements. These are 10-s ABP-labelled public-data windows, not independent
cuff measurements.

Each arm produces both `new_person_lora` and `new_person_lora_memory`, using the
same fitted personal adapter. This gives eight enrollment conditions and sixteen
prediction settings, not sixteen shared-backbone trainings.

## What is fitted at each budget

1. Copy only the shared network parameters; initialize that person's rank-4,
   2,048-parameter adapter with the same person-specific seed across budgets.
2. Use only this budget's registered labels. Within it, keep content/interval
   groups separate between personal epoch-selection fit and validation roles.
3. Select epochs with the existing patience-eight procedure, then discard the
   selection adapter and refit a fresh adapter on the whole allowed budget for
   that fixed number of epochs. No hard epoch cap or new seed sweep is added.
4. Recompute the person's BP anchor, reference memory and distance threshold
   from this budget only. Keep the existing encoder, rank, optimizer, five-donor
   retrieval, temperature and memory fallback rules unchanged.
5. Process query PPG only after personal fitting. Neither query BP nor unused
   registration labels enter the adapter, anchor, selection, memory or gate.

If a low budget contains only one indivisible group, there is no independent
inner validation set. Keep that person, use the permitted personal BP anchor,
and leave the adapter at its zero correction (zero fitted epochs). Memory uses
its existing insufficient-donor/threshold fallback. Do not borrow labels from
larger budgets or queries. Report this coverage explicitly. Extremely short
histories can consequently have identical realized counts at several nominal
fractions; they do not provide evidence for a precise continuous dose response.

## Evaluation and decision rules

All sixteen validation predictions are frozen before scoring, and all eight
budgets/two methods are retained regardless of their ordering. Only then can
the corresponding final-cohort jobs start. All sixteen final predictions must
be frozen before the final scorer opens reference BP. The queue cannot feed
these scores back into training or submit new variants.

Primary metrics are participant-macro SBP, DBP and mean MAE. Provide Overall,
MIMIC and VitalDB views from the same predictions. Also provide the requested
diagnostic columns: Setting, BP, MAE, R², ME, STD, ≤5 mmHg, ≤10 mmHg, ≤15 mmHg,
AAMI and BHS. Numerical screens do not constitute clinical device certification.
MIMIC and VitalDB are internal PulseDB source strata, not external validation.

Compare lower fractions with 90% on exactly paired people and query windows.
Report source-stratified participant bootstrap intervals (2,000 draws, seed
20260911). The primary family is seven Overall mean-MAE differences versus 90%
for LoRA+memory; supply a joint 95% maximum-centered-deviation bootstrap band.
Other intervals are pointwise exploratory summaries, not a multiplicity-adjusted
search for a winner. Uncertainty is conditional on this shared fit and one
nested ordering, not across training seeds.

No clinical noninferiority margin is invented. A nonsignificant difference does
not establish equivalence; do not choose a publication claim solely from the
best final-test point estimate. A smaller candidate budget can be proposed for
subsequent confirmation using a prospectively acceptable error tradeoff.

## Execution and limits

Use an immutable code snapshot. First run synthetic end-to-end and regression
tests plus a GPU forward/backward check. Then prepare the budget manifests.
Two hpc-2 GPU lanes process the two person shards for each budget; 32 personal
jobs cover eight budgets, two cohorts and two shards. Three non-GPU jobs prepare
and score, plus the initial smoke job. Dependencies prevent scoring incomplete
arms and prevent the final cohort preceding the validation freeze.

All active data, profiles and logs remain in the project's hot work area;
completed stage outputs are copied to the NAS archive. This is one finite batch,
not an autonomous retry/optimization loop. Public reports contain aggregates,
not personal profiles, waveforms or reference-label tables.

The earlier K=1/2/3/5 few-event route is discontinued as an active project goal
at the user's request. Preserve its historical code and results, labelled as
archived. The active direction is accumulated labelled personal history with
LoRA plus reference memory. A chronological incremental-update study and real
cuff/PPG measurements remain distinct future validation tasks, not claims of
this random-history budget experiment.
