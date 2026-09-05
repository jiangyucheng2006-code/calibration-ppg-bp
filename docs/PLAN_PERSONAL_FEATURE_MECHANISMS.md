# Personal feature adaptation: mechanism and improvement study

Date: 2026-09-06. Active scope: registered-user personalization with the same
participants in training and development validation, and disjoint windows.
The previous compact-profile screen failed to beat paired LoRA. This study
tests waveform alignment and personal transformation before advancing a new
method. It does not resume the separate unseen-user K-shot experiment.

## Frozen checkpoint diagnostic

Use the completed LoRA (1399/1408) and no-support 32D profile (1401/1410)
under both random_disjoint and chronological_blocked. Keep the same 82,040
internal-validation targets per mode and all 2,051 people. Read only train
and internal_validation roles; leave held-out sealed.

For each frozen checkpoint:

1. Natural PPG must reproduce saved predictions: maximum absolute prediction
   difference at most 0.05 mmHg and average difference at most 0.005 mmHg.
   These numerical tolerances accommodate device arithmetic, not model error.
2. Two seeded, fixed-point-free PPG permutations within each participant break
   window-level correspondence while retaining participant morphology. Their
   maps depend only on IDs and random numbers; donor PPG can be from any
   validation window, so this is a counterfactual diagnostic, not deployment.
3. Exchange personal parameters between participants of the same source while
   keeping each recipient's BP anchor and PPG. This probes personal mapping.
4. Set LoRA's B matrix, or the profile's personal code/bias, to zero. This is
   removal from a frozen trained network, not the result of training without
   that component. In particular co-adaptation may make zeroing destructive.
5. Compare train-role personal BP mean and median without query PPG.

Encode each natural window once on GPU, then reuse its features and waveform
descriptor for head-only perturbation evaluations. Parameters are restored
after each perturbation; original checkpoints are immutable. Save all
condition predictions privately and Overall/MIMIC/VitalDB aggregate tables.

Interpret jointly: PPG permutation degradation supports reliance on current
window correspondence; personal swapping degradation supports sensitivity to
the correct person's state. Neither proves causal physiology, and their
effects are not additive percentages of model benefit. Follow with trained
shared-adapter/capacity controls to distinguish co-adaptation and capacity.

## Prospective improvement contract

Keep the existing cohort, both split modes, training budget, Huber loss,
patience eight without an epoch cap, complete validation coverage and personal
state artifacts. Use a paired rank-4 LoRA reference and isolate feature
adaptation form from personal parameter count. The finite candidate matrix
and equations below are frozen before new model submission, after the diagnostic.

### Eight candidates, each under both split modes

| Candidate | Change tested | Learned values per person |
|---|---|---:|
| subject_lora_rank4 | Unchanged paired accuracy reference | 2,048 |
| shared_lora_rank4 | One shared adapter instead of one per person | 0 |
| subject_lora_rank1 | Smaller personal linear adapter | 512 |
| output_profile32 | Previous no-support output-level profile | 34 |
| feature_affine32 | Personal 32D code controls feature scales/shifts | 34 |
| shared_bilinear32 | Shared feature directions, personal 32D coefficients | 34 |
| shared_bilinear64 | Same bilinear rule, larger personal code | 66 |
| subject_nonlinear_rank4 | Nonlinear personal response with unchanged rank/parameter count | 2,048 |

These counts exclude the two train-derived BP anchor values per person and
the shared network. Total learned parameters are saved separately in run.json.
The prespecified primary accuracy hypothesis is subject_nonlinear_rank4:
changing the shape of the personal response, not adding personal capacity.
The other candidates are mechanism/capacity comparisons and exploratory
parameter-sharing alternatives. They are not all novel algorithms.

Let z be the 256D PPG feature, a_s the standardized train-role personal BP
mean, h a shared two-output head, and s a registered participant index.

- Reference: y = a_s + h(z + B_s A_s z / r), r=4.
- Primary: y = a_s + h(z + B_s [2 SiLU(A_s z)] / r).
  The only architectural change is replacing the linear low-dimensional
  personal response by a smooth nonlinear response. 2 SiLU has unit slope
  at zero. This remains a personal bottleneck adapter with LoRA ancestry;
  it is not presented as a novel mathematical activation or decomposition.
- Affine code: y = a_s + b_s + h((1+gamma(c_s))*z + beta(c_s)).
  A shared bias-free linear decoder maps c_s into 256 scales and 256 shifts.
- Shared bilinear code: y = a_s + b_s + h(z + B[c_s*(Az)]).
  A and B are learned jointly across all registered training people;
  c_s and b_s are saved per person. Unlike VeRA's frozen random matrices,
  these feature directions are trained. The family is related, not unrelated
  original mathematics. No nearest-person lookup, cluster assignment, or
  new-user routing is involved.

All models train end-to-end from scratch with the existing train data. No
validation gradient, personal-state update or target-derived input is allowed.
Every participant retains the same 320 labelled train windows and all 40
internal-validation windows. No worst-tail removal or extra support sampling.
Seed 20260906; AdamW lr=3e-4, weight_decay=1e-4, standardized Huber beta=0.5;
batch64, 200,000 sampled examples per epoch, patience8, no epoch cap.
As before, the finite 72-hour Slurm walltime is a resource limit, not early
stopping; interrupted jobs must not be called completed runs.
Within each split mode all candidates use the same GPU model: random on
RTX5080, chronological on RTX5070Ti, hpc-2 only. This avoids candidate/GPU
confounding after the diagnostic detected mixed-precision device differences.
Sixteen training jobs feed two afterok split reports and one cross-split gate.

An accuracy candidate merits confirmation only with at least 0.15 mmHg
Overall participant-macro mean-MAE gain over paired LoRA in both modes and
positive gains in MIMIC and VitalDB in both modes. Multiple candidates are
exploratory; multiple seeds are a later confirmation step. Do not interpret
fewer personal parameters as proven non-inferiority or faster inference.

## Precedents and novelty boundary

The architectural literature lookup is targeted, not exhaustive. LoRA and
FiLM are established controls; shared adaptation directions with a compact
personal coefficient vector are related to vector-based adaptation. Their
use does not itself justify claiming a new mathematical primitive. The
study tests whether personal waveform-feature transformation and its
parameter sharing explain the observed PPG-BP advantage.

Primary sources checked on 2026-09-06:

- [Hu et al., LoRA](https://arxiv.org/abs/2106.09685): original method freezes
  pretrained weights; our existing PPG implementation instead learns the
  shared encoder and subject adapters jointly, so it is not an exact reproduction.
- [Perez et al., FiLM](https://arxiv.org/abs/1709.07871): established
  feature-wise affine conditioning, used here as a comparison.
- [Kopiczko et al., VeRA](https://arxiv.org/abs/2310.11454): shared adaptation
  matrices with compact learned vectors; a relevant prior-art boundary for
  the proposed learned shared-direction personal coefficients.
- [Houlsby et al., parameter-efficient adapters](https://arxiv.org/abs/1902.00751):
  an established adapter-module family relevant to personal bottleneck networks.

Retrieval provenance: targeted architecture lookup, not exhaustive retrieval;
arXiv abstract/metadata pages accessed on 2026-09-06. The arXiv API query
`id_list=2106.09685,1709.07871,2310.11454&max_results=3` returned3/3 records
through the documented Atom parser. Houlsby was checked via its official arXiv
page. This is not a full-text systematic novelty assessment of PPG-BP papers.

The targeted source check is not an exhaustive PPG-BP novelty review. An
accuracy improvement, if observed, would motivate a broader novelty search
and confirmatory evaluation before a paper claim.
