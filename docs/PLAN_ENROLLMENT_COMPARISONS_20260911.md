# New-user enrollment: matched comparisons and ablations

## Material Passport

- Date: 2026-09-11.
- Mode: experiment planning; conditional next steps, not a submitted batch.
- Origin: the user's request to finish the current subject-disjoint enrollment
  study before comparing published methods and removing personal modules.
- Verification status: existing protocol, personal fitting code and saved
  validation receipt inspected; no new fit or final-test evaluation performed.
- Version: planning note v1. No frozen protocol or historical result is changed.

## Research question

After a fixed population model has been trained without the target people,
does personal LoRA plus reference memory improve prediction of those people's
held-out windows after a declared amount of labeled personal enrollment?

This is supervised new-user personalization, not calibration-free estimation.
Target-person enrollment data are explicitly allowed personal training data;
they are not part of the final scored queries. Population fitting, personal
fitting, model selection and query scoring must retain separate access rules.

## Order of work

1. Complete and audit the existing full-cohort batch before adding jobs.
   Read the frozen paired LoRA and LoRA-plus-memory results, cohort accounting,
   registration budgets, complete query coverage and prediction receipts.
2. Specify a finite, literature-grounded comparison matrix on development data.
   Do not choose a paper's main split simply because its final-test score is
   lower. Match the intended deployment question and declare all explored modes.
3. Run matched personal-stage ablations and published-method comparisons only
   after their configurations and evaluation rules are fixed.
4. Confirm useful effects with repeated training seeds and participant-level
   uncertainty, then add earlier-registration/later-session evidence where the
   data actually support that distinction.

This note does not start a monitor, schedule a future run or authorize automatic
test-feedback optimization. No new entry command or job IDs exist for it.

## Published-method comparisons

Select a small set of reproducible PPG-based methods after checking primary
papers and available implementations. Maintain two separate questions:

- Backbone comparison: use the same personal adaptation and labeled budget on
  each eligible backbone, clearly labeled as our controlled adaptation.
- Complete-method comparison: retain the published personal adaptation logic,
  with documented changes needed for the common PPG-only enrollment task.

Match population identities, registration IDs, scored query IDs, signal input,
label budget, preprocessing access, selection data and reasonable tuning
resources. A calibration-free network is an informative secondary control,
not the only competitor for a method with extensive labeled personal history.
Retain plain LoRA, personal BP mean and residual-offset controls. Do not compare
our internal table directly with a paper's differently split headline result.
PPG-only adaptations of multimodal papers must not be called exact reproductions.

## Minimum personal-stage ablation matrix

All four rows use the same frozen population model, personal BP anchor,
eligible people, registration data and scored queries.

| Setting | Personal LoRA | Reference memory | Question |
|---|---|---|---|
| Shared model plus personal anchor | No | No | What does the shared predictor and personal BP level explain? |
| LoRA | Yes | No | What does fitting personal feature parameters add? |
| Shared model plus memory | No | Yes | Can unadapted shared features and personal records explain the gain? |
| LoRA plus memory | Yes | Yes | Does memory improve the paired personal LoRA? |

Construct each variant correctly. In the no-LoRA memory row, rebuild reference
and query features without personal LoRA; memory computed from LoRA-adapted
features is not a LoRA-removal ablation. Fit required personal parameters under
the same selection rules rather than calling an arbitrary test-time switch-off
a retrained ablation. These controls isolate the personal stage conditional on
a common fitted population model, not the entire population-training pipeline.

If justified, follow with equal versus similarity-weighted donors and fixed
versus distance-based fusion. Earlier 200-person controls provide reusable
implementations, not replacement results for the current full-cohort protocol.

## Leakage and claim gates

- Exclude final-evaluation people from all population fitting and learned
  population preprocessing; never reuse an exposed all-person checkpoint.
- Assign registration and query roles before personal fitting. Never place
  query BP in anchors, adapters, reference banks, gates or learned transforms.
- Choose personal epochs only inside registration; global design choices use
  development participants, not final-query outcomes.
- Audit identity, recording interval and signal-content overlap, not only row
  numbers. Retain all eligible scored queries; error-based removal is not a
  deployment selection method.
- Freeze complete predictions before scoring. Existing validation receipts
  document this access order, but do not prove that the public dataset was
  untouched throughout the project's historical model selection.
- The current random within-person enrollment protocol uses approximately 90%
  labeled registration windows. It does not demonstrate few-cuff calibration,
  prospective long-term prediction or an exact official PulseDB benchmark.
- The public dataset has informed many earlier design decisions. Preserve that
  exploratory status. Repeated test-driven changes or retrospective selection
  of the best split require a new independent confirmation source or an honest
  exploratory report, not relabeling the old data as pristine.

## Reporting and later confirmation

Primary: participant-macro SBP/DBP/mean MAE and paired participant uncertainty.
Report Overall, MIMIC and VitalDB separately. Retain the requested diagnostic
columns: Setting, BP, MAE, R2, ME, STD, within 5/10/15 mmHg, and explicitly
qualified AAMI/BHS numerical screens. The latter are not clinical certification.

Report counts and label budgets, excluded rows/people, failures, model sizes,
personal fitting time and memory storage. Do not treat windows as independent
subjects. No new numerical promotion threshold is declared by this note.

For a longitudinal wearable claim, later use genuinely earlier enrollment and
later-session queries, or an independently collected cuff-PPG study with the
appropriate prior ethics approval. Dataset windows are ABP-derived references,
not independent real cuff measurements.

## Targeted literature verification

This was a bounded protocol-precedent lookup on 2026-09-11, not an exhaustive
review and not approval of either paper's entire methodology or clinical claims.

- Kim YC and Baek HJ (2026), *Evaluating cross-dataset transfer learning for
  photoplethysmography-based blood pressure estimation*, Scientific Reports,
  [DOI:10.1038/s41598-026-43409-8](https://doi.org/10.1038/s41598-026-43409-8).
  Primary full-text methods describe source-patient pretraining, target-patient
  fine-tuning, and separate personal training, validation and test samples.
  This supports personalization as an established evaluation setting, not
  equivalence between its budgets/data and ours.
- Fritzsche C and Senner V (2026), *The role of dataset integrity, calibration
  and signal quality in neural network-based blood pressure estimation*,
  Scientific Reports,
  [DOI:10.1038/s41598-026-66882-7](https://doi.org/10.1038/s41598-026-66882-7).
  The retrieved primary methods section separates initial calibration examples
  from training, validation and test examples under subject-wise splitting.
  It includes multimodal/scalar inputs and is not a ready-made PPG-only match.

Provenance: publisher/PMC web discovery; Europe PMC REST search with the exact
first title (pageSize=5, resultType=core, hitCount=1), returning MED/41906013,
PMC13039200 and its DOI. The `/PMC13039200/fullTextXML` body was verified and
its transfer-learning and data-partition paragraphs inspected. Some publisher
and PMC browser requests failed; the full-text API supplied the first paper.
