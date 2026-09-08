# Personal memory v2: completed results and interpretation

Verified on 2026-09-08. All 11 recovery jobs completed successfully, including
four new fits, two frozen diagnostic jobs and the final joint report. No new
training, final-test evaluation or model promotion was performed during this
result-publication task.

## Conclusion

**No v2 candidate qualifies to replace the paired LoRA reference.** E1 provides
a small additional chronological-mode improvement; E3 provides a smaller,
consistent numerical refinement to the existing memory blend. E2 does not
outperform the existing blend. These are single-seed development observations,
not statistically confirmed improvements.

The most important mechanism result is that the **v1 memory advantage depends
on temporal proximity**. Removing nearby references reverses its advantage in
random-disjoint evaluation. This is not evidence that PPG is useless, and it
is not a direct temporal-robustness test of the new E1/E2 models.

- [Complete metric tables: both modes, Overall/MIMIC/VitalDB](../results/personal_memory_v2/RESULT_TABLES.md)
- [Validated aggregate files and provenance](../results/personal_memory_v2/)
- [Frozen experiment plan](PLAN_PERSONAL_MEMORY_V2.md)
- [Execution and recovery record](PERSONAL_MEMORY_V2_EXECUTION.md)
- [Previous v1 results](RESULTS_PERSONAL_MEMORY_V1.md)

## Protocol and interpretation boundaries

The protocol remains `development-calbased-analogue-v1`, using only the
`meta_train` parent: 2,051 registered people, comprising 1,011 MIMIC and
1,040 VitalDB participants. Each contributes 320 labelled training windows
and 40 internal-validation windows. All 82,040 validation windows per mode
are retained, with 40,440 MIMIC and 41,600 VitalDB windows. Each window is
10 seconds; these are not independent cuff measurements or a five-label
new-user experiment. Retrieval uses five of the available training references.

Subjects intentionally overlap between train and internal validation; exact
windows, overlapping physiological intervals and duplicate content do not.
Validation BP selects checkpoints and scores development predictions, but
does not select references, enter predictor inputs or update personal state.
Held-out targets remain sealed. This is not an official PulseDB CalBased
reproduction, external validation or clinical device validation.

Random-disjoint permits training references later than the query and asks
about within-user interpolation. Chronological-blocked uses past references
on comparable recording clocks. MIMIC/VitalDB are internal source strata,
not two independent external datasets. Primary MAE is participant-macro.

## Overall result

Mean MAE is the mean of SBP and DBP participant-macro MAE, in mmHg; lower is
better. Each comparator is evaluated on the same complete query cohort.

| Method | Random-disjoint | Chronological-blocked | Interpretation |
|---|---:|---:|---|
| Paired stronger LoRA | 2.8764 | 3.7124 | Continued LoRA for random; frozen LoRA for chronological |
| Fixed memory + LoRA, no learned relation | 2.6475 | 3.7967 | Essential simple control |
| Previous v1 selected memory blend | 2.6475 | 3.6571 | Random selects epoch 0; chronological uses trained relation |
| E1: matched relation training | 2.6475 | **3.6334** | No learned random-mode gain; small chronological improvement |
| E2: BP-state retrieval metric | 2.7126 | 3.7881 | Inferior to the existing blend in both modes |
| E3: reliability on zero relation | **2.6381** | 3.7555 | Better than fixed blend, but worse than chronological LoRA |
| E3: reliability on selected v1 relation | **2.6381** | 3.6448 | Small refinement, not a newly trained gate |

The improvement over paired LoRA is 0.2289 / 0.0790 mmHg for E1 and
0.2383 / 0.0676 mmHg for E3 with the selected v1 relation. Neither reaches
the predeclared 0.15 mmHg requirement in **both** modes. The final report
therefore records an empty eligible-candidate list. Choosing the best method
separately in each mode below is descriptive; it does not define one frozen
model or authorize post-hoc promotion.

## Source-stratified results

These are separate full-cohort source views, not results after excluding a
difficult 30%. Positive gain means lower mean MAE than the paired LoRA.

| Mode | Method | Source | SBP MAE | DBP MAE | Mean MAE | Gain vs LoRA |
|---|---|---|---:|---:|---:|---:|
| Random | E1 | Overall | 3.3920 | 1.9031 | 2.6475 | 0.2289 |
| Random | E1 | MIMIC | 3.7702 | 2.0847 | 2.9275 | 0.2132 |
| Random | E1 | VitalDB | 3.0243 | 1.7265 | 2.3754 | 0.2441 |
| Chronological | E1 | Overall | 4.6734 | 2.5934 | 3.6334 | 0.0790 |
| Chronological | E1 | MIMIC | 4.3993 | 2.3999 | 3.3996 | 0.0777 |
| Chronological | E1 | VitalDB | 4.9399 | 2.7816 | 3.8607 | 0.0802 |
| Random | E3, selected v1 relation | Overall | 3.3844 | 1.8919 | 2.6381 | 0.2383 |
| Random | E3, selected v1 relation | MIMIC | 3.7608 | 2.0720 | 2.9164 | 0.2243 |
| Random | E3, selected v1 relation | VitalDB | 3.0184 | 1.7168 | 2.3676 | 0.2519 |
| Chronological | E3, selected v1 relation | Overall | 4.6874 | 2.6021 | 3.6448 | 0.0676 |
| Chronological | E3, selected v1 relation | MIMIC | 4.4121 | 2.4020 | 3.4071 | 0.0702 |
| Chronological | E3, selected v1 relation | VitalDB | 4.9550 | 2.7966 | 3.8758 | 0.0651 |

## What the new training actually contributed

All fits used seed 20260907 and stopped after eight non-improving epochs.
The backbone and personal LoRA were frozen; E1 trains 65,730 relation
parameters and E2 trains an 8,192-parameter projection. This is why each fit
finished in minutes rather than repeating full raw-PPG backbone training.

| Fit | Completed epochs | Selected epoch | Optimizer steps | Slurm elapsed |
|---|---:|---:|---:|---|
| E1 random | 8 | **0** | 6,256 | 4 min 25 s |
| E1 chronological | 16 | 8 | 12,512 | 3 min 45 s |
| E2 random | 8 | **0** | 6,256 | 3 min 56 s |
| E2 chronological | 11 | 3 | 8,602 | 3 min 02 s |

For E1 random, the selected result is the unchanged initial fixed blend.
Its best actually trained epoch reaches 2.6761 mmHg, worse than initial
2.6475. It would be incorrect to attribute the selected result to E1 learning.
For E1 chronological, training improves 3.7967 to 3.6334; however, v1 had
already achieved 3.6571, so the additional benefit over v1 is only **0.0237**.

E2 random selects the initial random projection, not a learned BP-state
metric. Its best actual trained epoch, derived from saved history, is 2.7426.
E2 chronological improves its initial projection from 3.8274 to 3.7881,
but remains worse than both LoRA and the selected v1 memory blend.

E3 reuses frozen v1 predictions and applies a fixed confidence formula;
it has no newly learned gate. Its additional mean-MAE improvement over
selected v1 is **0.0094 random / 0.0124 chronological**. Both sources improve
numerically, but this small single-seed effect is not a demonstrated material
or statistically significant gain.

## Temporal controls: the central limitation

The following controls apply to the **frozen v1 memory blend**, not to new E1,
E2, or E3 under altered time gaps. All queries remain, with original-LoRA
fallback where legal references are insufficient.

| Extra reference exclusion | Random mean MAE | Chronological mean MAE |
|---|---:|---:|
| No extra gap | 2.6475 | 3.6571 |
| Exclude references within 60 s | 2.9664 | 3.6756 |
| Exclude references within 300 s | 3.3023 | 3.7850 |
| Frozen LoRA, unaffected by retrieval exclusion | 2.8801 | 3.7124 |

In random mode, the v1 advantage reverses after the 60-second exclusion.
The nearest legal recording-time BP alone achieves 2.2851, better than
v1's 2.6475. Past-only retrieval raises v1 error to 2.8483; this is only
a retrieval restriction because the source model was still fitted under
the random split, so it is **not prospective model validation**.

There is also evidence that waveform selection contains useful information:
feature-selected uniform references outperform random references drawn from
the same broad temporal-distance strata (random: 2.7610 vs 5.7214;
chronological: 3.9671 vs 5.8644). However, these are coarse strata, not exact
time matching. Residual feature-versus-random time-distance differences are
not fully summarized in this run. This control cannot establish causal
independence from time proximity or justify a claim that PPG is unnecessary.

## Large-deviation subgroup: a negative result for E3

The fixed reliability refinement does not improve every group. Among windows
whose query target (reference-measurement) DBP differs from the person's training-median DBP by more
than 20 mmHg:

| Mode | Participants / windows in group | V1 blend DBP MAE | E3 DBP MAE | E3 worsening |
|---|---:|---:|---:|---:|
| Random | 721 / 1,873 | 7.2810 | 7.5031 | 0.2221 |
| Chronological | 186 / 1,796 | 12.7908 | 12.9024 | 0.1116 |

This is an absolute deviation from personal historical median, **not** an
adjacent-time BP change or a measured rapid-change tracking error. Groups
use reference BP only for post-prediction scoring, never for routing. The
finding suggests that lowering memory reliance can improve the average while
sacrificing some unusual BP states; dispersion cannot simply be equated with
noise. These exploratory subgroup comparisons have no uncertainty analysis.

## Requested agreement tables and standards labels

The linked full tables contain `Setting`, `BP`, `MAE`, `R²`, `ME`, `STD`,
`≤5 mmHg`, `≤10 mmHg`, `≤15 mmHg`, `AAMI`, and `BHS`, separately for both
modes and all three scopes. ME is prediction minus reference; STD is the
sample standard deviation of signed window errors, not the SD of subject
MAEs or repeated-seed uncertainty. Pooled R²/STD/threshold percentages are
secondary diagnostics. MAE agrees with participant-macro MAE here because
each participant contributes the same 40 queries.

`PASS*` means only the project's historical numerical screen: absolute ME
at most 5 mmHg and error STD at most 8 mmHg. The BHS column reports the
historical cumulative-error grade, with A/B shown as numerical PASS. Neither
label establishes compliance with a complete clinical validation protocol.
The selected E1/E3 Overall SBP and DBP rows satisfy these numerical screens
and have BHS Grade A; the paired LoRA Overall rows already do so as well.
Thus passing these columns is not evidence that the new module improves LoRA.

The original [BHS protocol](https://bihs.org.uk/wp-content/uploads/2017/08/BHS_Protocol_Revision_J_Hypertens_1993_df.pdf)
includes more than a table of error proportions. The
[ISO 81060-2 scope](https://www.iso.org/standard/73339.html) concerns clinical
investigation of intermittent cuff-based devices, and the
[AHA cuffless-device statement](https://professional.heart.org/en/science-news/cuffless-devices-for-the-measurement-of-blood-pressure)
emphasizes the need for appropriate validation beyond calibration-adjacent
controlled measurements. Sources checked on 2026-09-08.

## Decision and bounded next step

1. Retain paired LoRA as the main reference; do not promote any v2 candidate.
2. Retain E1 chronological as the stronger learned candidate and E3 as a
   low-complexity refinement for future confirmation, not as proven upgrades.
3. Do not expand E2 or stack modules on the basis of these results.
4. If further work is authorized, prioritize candidate-specific temporal-gap
   evaluation and quantify matching/fallback behavior before more architecture
   searches. A useful paper claim must separate recent-history interpolation
   from prediction under meaningful within-person BP change.

No experiment is submitted by this publication. No claim of novelty, SOTA,
clinical validity, external generalization or publication acceptance follows
from this batch. Multiple-seed/participant-level uncertainty and an untouched
final evaluation remain necessary before confirmatory claims.

## Verification record

- Batch: `personal-memory-v2_20260907-131749`; jobs 1636–1646 all completed
  with exit code 0:0. Both GPU smoke checks passed before dependent work.
- Training snapshot commit: `df2fba9d1f46836baedeb7352177318c49f82356`.
- Source archive SHA-256:
  `8bfbd36e120496de009293d00396747cd149209500efc853a3da2e9bdb59534e`.
- All 33 aggregate input files were verified byte-identical between server
  work and NAS. Transfer archive SHA-256:
  `3bda883d4c305e3ab76ec8cf1f1ce5c9ad843ec780588c5d57961308ec5fb0d6`.
- Local verification: 20 publisher tests and eight existing v2 comparison
  tests pass. Re-running the publisher reproduces all generated files byte
  for byte: 48 primary rows, 96 requested pooled rows, and separate training,
  temporal and subgroup diagnostics. There were no skipped tests in these
  28 checks, including integration against the completed aggregate archive.
- Server mode reports validate prediction keys, source counts, cache/model
  identity and recompute metrics from saved predictions. The local publisher
  validates those aggregate artifacts and their hashes; it does not reload
  private participant predictions. Public outputs contain aggregates only.
- Metadata caveat: the two chronological training records retain the default
  `arguments.split_mode=random_disjoint`. Code inspection confirms that this
  CLI argument only creates synthetic fixtures; real fits derive their mode
  from the hash-verified cache manifest. Top-level run mode, source cache and
  temporal legality use chronological mode. The publisher records this stale
  unused argument as a warning rather than rewriting the original run records.
- Fit seed is 20260907; the separate temporal-matching diagnostic uses its
  fixed seed 20260908. Neither represents a multi-seed training confirmation.
- The earlier failed smoke batch remains documented rather than erased.
