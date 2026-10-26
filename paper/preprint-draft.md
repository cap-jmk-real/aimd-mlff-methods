# DRAFT PREPRINT — NO RESULTS YET

> **Status: work-in-progress draft (26 Oct 2026).** This document is a structured outline of a planned methods paper. It contains **no results**; all hypotheses below are untested, and no numbers in it are measured outcomes (thresholds are pre-registered acceptance criteria). Do not cite as findings.

**Working title:** mond: packing arbitrary mixtures, active-learning training data and stable MD with machine-learned force fields — groundwork for simulating electrochemical reactions

**Author:** Julian M. Kleber (sole author)

**License:** The text of this draft is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code in this repository is covered by the repository [LICENSE](../LICENSE).

## Abstract (draft)
We plan to study three ingredients needed for simulating electrochemical reactions with machine-learned force fields (MLFFs) at ab initio molecular dynamics (AIMD) accuracy: automatic packing of arbitrary liquid electrolyte mixtures, data-efficient generation of DFT training labels via uncertainty-driven active learning, and stable, AIMD-faithful MLFF molecular dynamics. Results will be added once the experiments in [`experiments/PLAN.md`](../experiments/PLAN.md) have been run.


## 0. Research questions and hypotheses
- **RQ1 (packing):** Can mond build physically valid boxes of arbitrary liquid mixtures incl. salts automatically?
  - **H1:** mond boxes reach equilibrated density within 2 % of reference (exp./long classical MD) with zero overlaps below 0.8×vdW contact, at time-to-pack ≤ packmol, for S1–S3. Refuted if density error > 2 % or overlaps remain after densification. Metric: density, min-distance histogram, wall time (E5).
- **RQ2 (accuracy):** Which MLFF reaches DFT accuracy on held-out mixture frames at a fixed label budget?
  - **H2a:** Fine-tuned foundation models (MACE-OFF / MACE-MP-0) beat from-scratch MACE/NequIP at equal DFT budget (≤3k labels): lower force RMSE on held-out frames, difference CI excluding 0 over ≥3 seeds. Refuted if CI includes 0 or scratch wins. (E1, E6)
  - **H2b:** Conservative E→F models beat the direct-force baseline (tutorial MPNN) on force MAE and energy MAE. (E1, E6)
- **RQ3 (stability):** Do these MLFFs give stable, AIMD-faithful MD?
  - **H3a:** Best model runs ≥1 ns NVT on ~1k-atom boxes with no failure (bond break, T > 2×T0, or |ΔE| blow-up) in ≥8/9 runs (3 seeds × 3 inits). Metric: time-to-failure.
  - **H3b:** NVE energy drift < 1 meV/atom/ps; conservative models drift less than direct-force (refuted otherwise).
  - **H3c:** RDF integrated absolute error vs AIMD < 0.1 and self-diffusion within 30 % of AIMD; Li⁺ coordination number within ±0.5. (E2)
- **RQ4 (data efficiency):** Does uncertainty-driven active learning cut DFT calls?
  - **H4:** QBC or single-model uncertainty AL reaches target force RMSE and target stable time with ≥2× fewer DFT calls than random/uncorrelated-MD-snapshot selection (area under learning curve, ≥3 seeds). Refuted if ratio < 2 or CI overlaps 1. (E3)
- **RQ5 (transfer):** Do models trained on some compositions transfer to unseen ones?
  - **H5:** On held-out compositions (S2 0.5/2 M, unseen EC:EMC ratio, DMC), zero-shot force RMSE ≤ 1.5× in-distribution, and ≤50 new labels recover in-distribution accuracy and H3a stability. (E4)

## 1. Introduction (RQ1–RQ5)
Motivation: simulating electrochemical reactions (electrolyte decomposition, SEI, electrode/electrolyte interfaces) at DFT accuracy needs (i) correct boxes, (ii) cheap data, (iii) stable long MD. Contributions C1 packing (RQ1), C2 AL with DFT-call accounting (RQ4), C3 stability-validated MLFF MD (RQ2, RQ3, RQ5).
## 2. Related work (all RQs)
Equivariant MLIPs (NequIP, Allegro, MACE, MACE-OFF, SO3krates); foundation models (MACE-MP-0, UMA, OMol25, CHGNet...); stability (Fu et al.); AL (FLARE, DP-GEN, Smith 2018, Zaverkin 2024); electrolyte MLFF (Magdău 2023); packing (packmol, gmx insert-molecules). Refs only from review-1.md verified list.
## 3. Methods
3.1 mond packing (RQ1) · 3.2 reference DFT + xTB pre-sampling, unit-correct Verlet/Langevin, PR #15 (RQ2–4) · 3.3 models and conservative training, losses, splits by composition and time block (RQ2) · 3.4 AL loop and DFT-call accounting (RQ4) · 3.5 stability protocol (RQ3) · 3.6 statistics: ≥3 seeds, bootstrap CIs.
## 4. Results
4.1 Packing, E5 (H1) · 4.2 Accuracy, E1 (H2a, H2b) · 4.3 Stability, E2 (H3a–c) · 4.4 AL efficiency, E3 (H4) · 4.5 Transfer, E4 (H5) · 4.6 Ablations, E6 (H2a, H2b, H3b).
## 5. Discussion and outlook to electrochemistry
Which hypotheses held; failure modes; next: reactive sampling (nanoreactor, Wang 2014), reactive AL (Vandermause 2022, DP-GEN), charge-aware models (SpookyNet), constant-potential MLFF (Wang 2025). No reactive results claimed.
## 6. Limitations (all) — cluster vs periodic DFT, small cells, DFT level, ionic radii.
## 7. Code and data availability.


## Results
_Not yet available. Section 4 lists planned analyses only._
