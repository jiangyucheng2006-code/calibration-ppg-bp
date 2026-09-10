# Enrollment budget and nearest-prior-work review

Date: 11 September 2026. Status: source verification and recommendations only.
No new training, final-test scoring, protocol replacement or public push.

## Conclusions

- The current approximately 90% personal registration fraction is a project
  choice, not a requirement of LoRA or of every PulseDB experiment. The official
  CalBased benchmark has 360 training and 40 testing windows per person; the
  current full-cohort subject-disjoint enrollment study is a custom protocol.
- Comparing 50%, 60%, 70% and 90% is reasonable. Reduced labeling burden is
  certain by construction; improved accuracy is a hypothesis, not an expected
  numerical result. Keep the current 90% batch unchanged as a reference.
- No fully identical implementation was identified in this bounded search.
  This is NOT proof of worldwide novelty. Direct prior work exists for PPG-BP
  LoRA, similar-sample-assisted personalization, reference-conditioned BP
  estimation, and general neural neighbor retrieval. Two particularly relevant
  papers could only be checked at metadata/abstract level.
- Continue evaluating the combination, but do not claim a first use of LoRA,
  personal calibration, historical reference signals, or nearest-neighbor
  regression. Publication depends on a demonstrated contribution, not merely
  an unmatched acronym or an exact architecture not found in a search.

## Verified local contract

Sources: [full-cohort plan](PLAN_FULL_COHORT_ENROLLMENT_V1.md),
`src/pulsedb_fewshot/legacy_enrollment_protocol.py`,
`src/pulsedb_fewshot/full_enrollment_data.py`, and
`src/pulsedb_fewshot/post_enrollment_personal.py`.

The original outer subject roles remain population training, new-person
validation and new-person final evaluation. The 90% fraction concerns each
target person's permitted registration windows, not a 90% reuse of final-query
targets for population training. Approximately 10% forms that person's query
role, with overlap groups kept together. Within registration, approximately
8/9 supports inner fitting and 1/9 selects personal epochs; a fresh adapter is
then refitted on all permitted registration windows. Inner-selection labels
are part of the registration budget.

The shared compact ResNet, 256-dimensional representation, rank-4 personal
adapter (2,048 trainable parameters), and personal BP anchor form the LoRA
path. The reference path uses post-adapter features, same-person registration
BP, cosine top-five retrieval, temperature 0.1, and registration-only
leave-40-block distance calibration for fusion. Five retrieved neighbors do
not mean five calibration labels. The bank can contain all registration
records. This is not a newly established general memory architecture.

The official 360/40 split is documented in Table 4 of the
[PulseDB dataset article](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full).
Changing the number of personal training rows creates a budget variant, not an
exact full-budget official reproduction. The current all-source cohort was
already explicitly designated custom, not exact official CalBased/CalFree.

## Recommended budget comparison, not an executed plan

Use one fixed population checkpoint and the same target people. Freeze a common
query set before inspecting its outcomes. Within the remaining registration
pool, choose nested subsets without using query BP or previous query errors.
For illustration, if a person has 1,000 eligible windows and 100 fixed queries:

| Fraction of all eligible personal windows | Registration | Fixed queries | Unused in that condition |
|---|---:|---:|---:|
| 50% | 500 | 100 | 400 |
| 60% | 600 | 100 | 300 |
| 70% | 700 | 100 | 200 |
| 90% | 900 | 100 | 0 |

These are illustrative counts, not actual fixed-size participant budgets.
Fractions refer to all eligible personal windows, not a fraction of an already
90% registration pool. Prespecify rounding and short-history behavior; do not
remove difficult people after seeing errors. Each condition must recompute
its adapter, anchor, memory bank and gate using only that condition's labels.
Unused labels cannot inform epoch selection or any other personal state.

The primary comparison would pair LoRA with LoRA plus memory at each fraction:
four budgets and two outputs, with one fitted adapter shared by the paired
outputs at each budget. This is not eight population retrainings. Retain the
previously proposed shared+anchor and no-LoRA-feature memory controls for the
separate module-ablation question.

First make choices on development participants, then freeze any later
evaluation. Do not choose the best budget by repeatedly inspecting final-test
errors. Report participant-macro Overall/MIMIC/VitalDB results, paired
uncertainty and actual personal label-count distributions. A formal claim of
non-inferiority requires a prospective margin; similar point estimates alone
do not establish equivalence.

A separate 50/50 or 70/30 whole-split experiment is also possible, but it changes
query composition and is not a clean estimate of the label-budget effect.
Random-window enrollment does not establish prediction on later days. A
chronological study should use earlier registration sessions and fixed later
queries, with recording/session groups and temporal separation declared.

PulseDB provides ABP-derived window labels. Neither 90% nor 50% can be converted
directly into a number of independent cuff measurements. Later deployment
experiments need explicit paired cuff occasions and dates, not percentages
alone. Keeping 50% of a long recording may still require many labels.

## Follow-up: fixed-query comparison versus cumulative personal updates

Discussion update, 11 September 2026. These are recommendations, not a frozen
protocol or authorization to execute. The user wants lower initial personal
label requirements and an archive that grows whenever a new paired cuff
measurement becomes available. Do not silently interpret routine model-only
PPG predictions as new reference BP measurements.

### A. Compare personal label budgets on identical queries

The primary budget comparison should keep the same people and exact query IDs,
not merely the same number of queries. For an illustrative 500-window person
with 100 reserved queries, 200/250/300/350/400 enrollment windows correspond to
40/50/60/70/80% of all eligible personal windows. The remaining
200/150/100/50/0 windows are unused for that condition. This is not data loss:
their labels are deliberately withheld to make the budget intervention real.

Testing a 200-window profile on all remaining 300 windows is also a valid
separate evaluation if roles are genuinely disjoint. However, comparing that
score directly with 400-enrollment/100-query performance changes both the
budget and the query distribution. Report a common-query primary result and
an all-remainder secondary result, labelled separately.

It is impossible to enroll on 450 of 500 unique windows and simultaneously
reserve 100 disjoint queries. To include a 90% arm, use a common 50-window
query set for every arm, or keep the 90% run as a separate historical reference.
This numerical example does not replace the current run's approximately
90%/10% assignment or prescribe a fixed per-person window count.

For a pure data-volume comparison, use nested enrollment subsets within one
common permitted history horizon, selected without future BP/errors. Match or
explicitly analyze the recency of calibration: earliest-prefix growth changes
both data quantity and the distance to testing. All personal inner selection,
anchors, LoRA, memory and distance calibration must obey the assigned budget.

### B. Evaluate actual accumulating-history use in a separate time-ordered protocol

The current `post_enrollment_personal.run_profiles` performs one registration
fit/refit, creates a fixed reference bank, and then predicts the fixed query
role. Saving a profile is implemented; automatic sequential incorporation of
new cuff measurements is not established by that implementation or its scores.

Proposed sequence: fit an initial profile from earlier paired records, predict
the next observation with the available profile, freeze that prediction, and
only then reveal a new reference BP when the prespecified measurement schedule
makes it available. After alignment/quality checks, add the paired record to
that person's history and update only their personal state for future use.
PPG-only observations are not supervised updates. Delayed reference availability
must be respected. Every later prediction uses only earlier released labels.

This is a new predict-before-update evaluation, not permission to feed the
existing static held-out targets into the current batch. A past streamed
observation may become a later training reference only under the new frozen
label-release rule; it is never rescored after learning its target. The
evaluator can retain labels of non-calibration observations without exposing
them to the model. Calibrations should occur at realistic sparse occasions,
not automatically at every ABP-labelled 10-second window.

Keep the population encoder/head/scaler frozen. A minimal staged implementation
would append reliable memory immediately and update personal LoRA periodically
using new and replayed earlier reference records. Since the existing bank uses
post-adapter features, any adapter change requires refreshing the bank in the
same feature space and recalculating permitted anchor/gate state as prescribed.
Version and atomically save the adapter, anchor, bank and gate together.
Never mix old adapted bank vectors with new-adapter query vectors.

Candidate comparisons on identical chronological queries and cuff schedules:
initial profile frozen; memory-only updating; LoRA-only updating; combined
updates. Separate anchor updating as a specified matched policy so it does not
confound the component comparison. Include last-reference persistence and a
simple offset-update control to check whether a fresh BP label alone explains
the benefit. Freeze update steps, sampling, quality and acceptance rules on
development participants, not final-stream errors. A periodic all-available-
history refit can be an implementation reference, not proof of efficient
incremental learning.

Report participant-macro Overall/MIMIC/VitalDB paired errors, cumulative
reference-event counts, calibration-to-query intervals, coverage, update cost,
and the proportion of people made worse. More records may improve coverage,
but noisy references, redundant states, changing physiology and forgetting
can prevent monotonic improvement. Test the gain against a frozen profile on
the same time blocks, not by assuming later blocks have equal difficulty.

The [River progressive-evaluation documentation](https://riverml.xyz/latest/api/evaluate/progressive-val-score/)
describes prediction before delayed target release and learning. It supports
the evaluation ordering, not medical validity or novelty of this application.
The [official PulseDB v2 documentation](https://github.com/pulselabteam/PulseDB)
defines `T` relative to each extracted recording clip. Verify recording/session
ordering before chronological replay; do not globally sort `T` across clips
as if it were an absolute calendar clock. The public-data replay can test
record-time accumulation, not replace multi-day wrist-PPG/cuff validation.

Suggested next decision: retain the present high-history reference; use a
fixed-query lower-budget curve to identify plausible starting budgets, then
test chronological accumulation. No server check, submission, final scoring,
model change, public push or global-memory update was performed here.

## Nearest verified prior work

### Li et al., IEEE JBHI, 2026

[Few-Shot Personalized Blood Pressure Estimation From Photoplethysmography and Physiological Priors via Low-Rank Adaptation](https://pubmed.ncbi.nlm.nih.gov/41701592/).
DOI: 10.1109/JBHI.2026.3665810; PMID: 41701592. The current PubMed record lists
30(9), 7913-7925; earlier indexing recorded online publication in February.

Verified author abstract: Transformer-based population pretraining followed by
personal fine-tuning, physiological pulse-pressure constraints, and sampling-
rate-robust LoRA. It lists UCI, Queensland and CAS-BP, with 50 personal samples
per fine-tuned participant. This directly precedes PPG-BP LoRA personalization.
The abstract does not describe our same-person query-time reference bank, but
that omission cannot establish its absence from the full article. IEEE DOI
resolved to document 11397819; readable full methods were unavailable here.
Its reported MAE is not a matched PulseDB comparison with our protocol.

### Song et al., CWSN 2024 proceedings, published 2025

[SimilarBP: Leveraging Similar Samples for Few-Shot PPG-Based Blood Pressure Measurement](https://link.springer.com/chapter/10.1007/978-981-96-2186-6_22).
DOI: 10.1007/978-981-96-2186-6_22. Yixuan Song, Dong Zhao, Qi Wang and Zhou Fang.

Verified publisher abstract: pretrain a user-independent model, then fine-tune
with personal samples and similar samples from the larger dataset. It uses
individualized contrastive learning for selection and an ABP label-correction
algorithm. The stated use of similar samples in fine-tuning differs from our
implemented same-person inference-time lookup. However, subscription full
text was unavailable; exact architectural overlap and the named public dataset
remain unresolved. Treat this as a close precedent requiring full-text review,
not evidence that our broad similar-reference idea is unoccupied.

### Schlesinger et al., Critical Care Explorations, 2020

[Estimation and Tracking of Blood Pressure Using Routinely Acquired Photoplethysmographic Signals and Deep Neural Networks](https://pubmed.ncbi.nlm.nih.gov/32426737/).
DOI: 10.1097/CCE.0000000000000095; PMCID: PMC7188414.

Retrieved full-text methods describe a subject-disjoint MIMIC-II study and a
Siamese CNN that compares the current PPG spectrogram with a first personal
PPG/BP reference to estimate BP change. It does not retrain a personal network
for each new patient. This is direct precedent for reference-conditioned
prediction and differs from our adapter plus multi-reference neighbor fusion.

### Mekonnen et al., Scientific Reports, 2024

[Accurate and 30-plus days reliable cuffless blood pressure measurements with 9-minutes personal photoplethysmograph data and mixed deduction learning](https://www.nature.com/articles/s41598-024-75583-y).
DOI: 10.1038/s41598-024-75583-y; PMCID: PMC11467377.

Retrieved full-text methods describe paired preceding/current PPG inputs,
personal reference BP, and CNN mixed deduction learning. Nine personal
measurement rounds are used with a larger participant pool; the evaluation
has 15 target people and 88 testing samples. This is a relevant precedent for
accumulated personal references and later-date testing, not the same LoRA and
cosine-neighbor implementation. Its clinical/standards claims were not adopted
or independently validated in this review. The publisher flags an author
correction, DOI 10.1038/s41598-024-79326-x; that correction's contents were not
accessible in the attempted publisher fetch.

### General method precedent: TabR

[TabR: Tabular Deep Learning Meets Nearest Neighbors in 2023](https://arxiv.org/abs/2307.14338)
describes neural prediction using retrieved training features and labels.
The author preprint and HTML were accessible. This is tabular ML, not a direct
PPG study, but it prevents treating neural neighbor-label retrieval itself as
a newly invented mechanism.

## Search provenance and limits

Targeted source/novelty check on 11 September 2026, not a systematic review.
Used web discovery with publisher/author records, PubMed, Europe PMC and
Crossref. No paid access, credential use or access-control bypass. Search
snippets and the project's own public repository were not evidence of novelty.

Europe PMC endpoint: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`.
Queries were URL-encoded, `format=json`, `resultType=core`; result envelopes
were checked for errors and parsed query strings. Retrieved only first pages:

| Query | Index hits | Records inspected |
|---|---:|---:|
| `(photoplethysmography OR PPG) AND "blood pressure" AND ("low-rank" OR LoRA)` | 80 | 20 |
| `(photoplethysmography OR PPG) AND "blood pressure" AND ("memory bank" OR "reference memory" OR "retrieval augmented" OR "nearest neighbor")` | 313 | 20 |
| `"SimilarBP"` | 2 | 2 (both irrelevant) |
| `TITLE:"blood pressure" AND (ABSTRACT:LoRA OR ABSTRACT:"low rank adaptation")` | 1 | 1 |
| `TITLE:"blood pressure" AND (ABSTRACT:"memory bank" OR ABSTRACT:"reference memory" OR ABSTRACT:retrieval OR ABSTRACT:"similar samples")` | 26 | 25 |
| `TITLE:"blood pressure" AND (ABSTRACT:personalized OR ABSTRACT:personalization OR ABSTRACT:calibration) AND (ABSTRACT:"nearest neighbor" OR ABSTRACT:"nearest neighbour" OR ABSTRACT:"reference samples")` | 1 | 1 |

Europe PMC full-text endpoint `/{PMCID}/fullTextXML` supplied PMC7188414 and
PMC11467377; the presence of an article body was checked before reading
method-related passages. A failed PMC browser challenge was not treated as an
unavailable article when the open-access XML endpoint was available.

Crossref endpoint: `https://api.crossref.org/works`, `query.bibliographic`,
`rows=10`. Queries were `photoplethysmography personalized blood pressure
retrieval memory`, `photoplethysmography blood pressure low rank adaptation`,
and `SimilarBP Leveraging Similar Samples`. These broad ranked queries had
2,348,106, 3,401,948 and 338,086 index matches respectively; only the first ten
of each were screened. Those counts are not counts of relevant BP papers.
They resolved the Li and SimilarBP DOIs but do not support exhaustive absence.

Web terms also included PPG/BP with personalized LoRA, SRR-LoRA, memory bank,
reference bank, nearest neighbor, retrieval-augmented, similar samples and
Siamese calibration, plus exact-title/author searches. Many memory/LoRa hits
concerned LSTM, cognition, radio communication or clinical text retrieval and
were irrelevant. Non-indexed work, inaccessible full texts, different
terminology and unpublished research remain coverage gaps.

## Publication-oriented interpretation

A testable contribution is whether an adapted personal representation and
explicit same-person references provide complementary value for post-training
new-user enrollment at a declared label budget. It still requires matched
LoRA-only/reference-only controls, lower-budget evaluation, BP-change tracking
and later-session evidence before stronger practical claims. This review does
not assert acceptance, a world-best result, unique naming or a clinical device.

If an equivalent method is later found, new-method novelty must be narrowed or
withdrawn. A transparently positioned replication, independent validation or
meaningful extension can still be research; an unchanged method with only a new
name cannot be presented as an original invention.
