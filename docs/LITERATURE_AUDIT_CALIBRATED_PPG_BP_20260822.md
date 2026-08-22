# Calibrated PPG blood-pressure literature audit (2026-08-22)

## Scope and reproducible search

The search targeted personalized, calibrated, transfer-learning, meta-learning,
few-shot, online-calibration, and test-time-calibration PPG blood-pressure
estimation, with special attention to PulseDB and numerical AAMI/BHS claims.
The search covered Europe PMC/PubMed, Crossref-linked publisher pages, PMC,
arXiv, PMLR, IEEE/author postprints, journal pages, public code repositories,
reference chaining, and general web discovery. Four Europe PMC queries returned
3,799, 11, 23, and 280 records respectively; the first 100 records were screened
for the two broad queries and all records for the two PulseDB-specific queries.

This is a broad, reproducible search, not a claim that every page on the internet
was indexed or accessible. Search terms and decisions are preserved here so the
screen can be updated later.

## Non-negotiable interpretation rules

- A paper is not counted as AAMI compliant merely because its MAE is below
  5 mmHg. The numerical screen requires signed error mean and signed error SD;
  formal device compliance additionally requires the prescribed measurement
  protocol and cohort.
- BHS grading requires the percentages of absolute errors within 5, 10, and
  15 mmHg. A low MAE alone is not a BHS grade.
- PulseDB's published `calibration-based` split places many windows from the
  same participants in training and testing. It is not equivalent to this
  project's first K=1/2/3/5 labeled cuff events followed by later-event testing.
- Results from PPG+ECG, demographics, invasive ABP statistics, random
  window-level splitting, or ongoing labeled updates are not presented as
  comparable PPG-only fixed-K results.
- AAMI/BHS fields in this project remain retrospective numerical screens, not
  claims of clinical device certification.

## Evidence matrix

| Study | Data and calibration | Network | Reported standard result | Protocol audit and use here |
|---|---|---|---|---|
| Wang et al., *PulseDB* (2023), DOI 10.3389/fdgth.2022.1090854 | PulseDB; official calibration-based, calibration-free and AAMI cohort files | 1D ResNet-18 example using ECG+PPG | Dataset supplies AAMI-oriented cohorts; it does not prove a model passes | Authoritative dataset definition. The calibration-based split is same-participant overlap, not K-shot calibration. |
| Fritzsche & Senner, *Scientific Reports* (2026), DOI 10.1038/s41598-026-66882-7 | PulseDB MIMIC/VitalDB; subject-disjoint; first 3/6/9 reference measurements | MLP, residual CNN and causal TCN; PPG/VPG/APG plus scalar features | DBP can satisfy the numerical AAMI screen; SBP does not, even with nine references | Strongest integrity match. The raw-waveform TCN branch is selected for an isolated backbone screen. |
| Tang et al., *Physiological Measurement* (2026), DOI 10.1088/1361-6579/ae52a1 | PulseDB; five-shot personalization | self-supervised masked autoencoder followed by residual-attention or Transformer encoder | Accessible report gives MAE but does not establish AAMI/BHS signed-error criteria | Closest K=5 paper. Public code is marked as being reorganized and contains split/path ambiguities, so only its residual-attention backbone is screened now. |
| Chen et al., *IEEE JBHI* (2024), DOI 10.1109/JBHI.2024.3483301 | PulseDB VitalDB; calibration-based training uses about 360 windows per participant; transfer setting uses up to 100 target windows | 1D/2D residual U-Net, CBAM, BiGRU and multimodal fusion | Reports AAMI numerical acceptance and BHS grade A | Not few-shot and not PPG-only. A reduced 1D residual U-Net family is screened without copying its multimodal inputs. |
| Kasbekar et al., *Bioengineering* (2025), DOI 10.3390/bioengineering12050493 | PulseDB VitalDB/AAMI cohort; prior invasive ABP mean/SD are reused when PPG mean changes | small residual net on Catch22, morphology, ECG-derived and demographic features | Personalized result reports SBP -1.31±7.91 and DBP 0.10±4.59 mmHg; BHS A | Meets numerical screens but uses ongoing invasive-label state and extra inputs. Evidence that the BP anchor dominates; not a deployable backbone candidate. |
| Tang et al., APSIPA (2025), DOI 10.1109/APSIPAASC63619.2025.10849156 | Official PulseDB training and calibration-based test subsets | CNN, Transformer and iterative attentional fusion with PPG+ECG+demographics | Reports AAMI and BHS A | Same-participant high-volume calibration and multimodal inputs; not a fair K-shot comparison. |
| Shaikh & Forouzanfar, *IEEE Sensors Journal* (2025), DOI 10.1109/JSEN.2024.3512197 | PulseDB; 3,027 participants; five-fold evaluation | separate ECG/PPG CNN streams followed by multilayer BiLSTM | Reports AAMI and BHS A | Accessible sources do not resolve whether folds are participant-disjoint; multimodal. It motivates the PPG-only compact CRNN family, not its reported score. |
| Leitner et al., *IEEE JBHI* (2022), DOI 10.1109/JBHI.2021.3085526 | MIMIC-III; subject-specific transfer with at least 50 samples | convolutional-recurrent network with personalized layers | Reports AAMI/BHS-compatible performance | Relevant personalized architecture but 50 samples is ten times K=5 and windows may be strongly correlated. Compact CRNN is screened under our stricter protocol. |
| Jo et al., CHIL/PMLR (2025) | PulseDB; unlabeled and intermittently labeled test-time buffers | ResNet-ViT hybrid with self-supervised and supervised heads | Improves MAE but does not establish AAMI/BHS | Ongoing labels are not fixed first-K calibration. Test-time updating is deferred to a separate method experiment. |
| Shen et al., DMT arXiv:2606.11125 (2026 preprint) | Official PulseDB calibration-based/calfree/AAMI subsets | demographic FiLM, morphology auxiliary loss and six-layer Transformer | Calibration-based subset meets numerical thresholds; calibration-free and AAMI subsets fail badly | Strong result depends on same-participant overlap and demographics. Demographics remain paused because audited ages are anomalous. |
| Moulaeifard et al. (2025), DOI 10.1088/3049-477X/ae01a8 | PulseDB benchmark with in-distribution and participant-disjoint views | XResNet1d101 and other baselines | Calibration-free errors remain much larger than many overlapping-window reports | Important negative control: strict generalization is substantially harder. |
| Li et al., *IEEE JBHI* (2026), DOI 10.1109/JBHI.2026.3665810 | UCI/Queensland/CAS-BP; 50 target samples | Transformer, physiological loss and SRR-LoRA | Reports AAMI/BHS A | Not PulseDB and not K≤5. LoRA is a later personalization-method candidate, not evidence for immediate backbone replacement. |

## Main finding

No audited study was found that simultaneously demonstrates all of the
following: PulseDB, PPG-only input, participant-disjoint development, strictly
chronological first K≤5 labeled calibration events, no later label access, and
both SBP/DBP AAMI or BHS acceptance. The low-error PulseDB papers generally
relax at least one of these conditions. Therefore their published MAE cannot be
used as the expected value for this project.

This gap is scientifically useful. It means the project's strict setting is
not a routine replication of the PulseDB calibration-based benchmark.

## Architecture shortlist for Round 12

All candidates keep the same QGH few-shot calibration head, fixed-first K=5
support policy, Huber loss, folds, query set and early-stopping policy. Only the
PPG encoder changes.

1. `tcn_bp`: causal TCN waveform branch from the subject-disjoint PulseDB
   calibration study. This has the strongest protocol evidence.
2. `fewshot_resnet_attention`: group-normalized residual blocks plus learned
   attention pooling from the direct PulseDB five-shot study. SSL pretraining
   is deliberately deferred so architecture and training method are not mixed.
3. `bp_crnn`: three compact temporal convolutions plus a GRU, derived from
   personalized transfer studies that report AAMI/BHS numerical results.
4. `resunet_encoder`: 1D residual U-Net family reported to reach AAMI/BHS on
   PulseDB, stripped of ECG, demographics and ABP-waveform supervision.
5. `resnet_small`: unchanged internal reference.

The shortlist does not assert exact reproduction of the source papers. Inputs,
calibration heads, loss functions and split regimes differ by design; this is
a controlled architecture-family screen under one common protocol.

## Deferred method experiments

- self-supervised masked-waveform pretraining followed by the winning encoder;
- per-participant LoRA or last-layer adaptation with a K-aware regularizer;
- universal relationship + stable personal offset + dynamic change
  decomposition;
- label-efficient online/test-time calibration with a strictly causal buffer;
- morphology auxiliary learning, only with labels computed from PPG itself;
- demographic conditioning only after age/sex integrity problems are resolved.

## Primary sources

- PulseDB: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9944565/>
- Dataset integrity/calibration study: <https://www.nature.com/articles/s41598-026-66882-7>
- Five-shot PulseDB study: <https://pubmed.ncbi.nlm.nih.gov/41839155/>
- Five-shot public code: <https://github.com/sustech-tlw/Few-Shot-Personalized-Blood-Pressure-Estimation>
- Residual U-Net study: <https://pubmed.ncbi.nlm.nih.gov/39423074/>
- Catch22 personalization: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12109000/>
- Iterative demographic fusion postprint: <https://pure.coventry.ac.uk/ws/portalfiles/portal/105005287/Zheng2025AAM.pdf>
- Dual-stream CNN-LSTM record: <https://pure.etsmtl.ca/en/publications/dual-stream-cnn-lstm-architecture-for-cuffless-blood-pressure-est/>
- Personalized BP-CRNN: <https://pubmed.ncbi.nlm.nih.gov/34077378/>
- Test-time calibration: <https://proceedings.mlr.press/v287/jo25a.html>
- DMT preprint: <https://arxiv.org/abs/2606.11125>
- Generalizable PulseDB benchmark: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12435175/>
- Few-shot LoRA study: <https://pubmed.ncbi.nlm.nih.gov/41701592/>
