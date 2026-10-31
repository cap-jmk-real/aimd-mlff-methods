# Handoff: AIMD MLFF Methods Paper (`aimd-mlff-methods`)

**Working Title:** *Methods for electrochemical reaction sampling with machine-learned force fields at AIMD-level accuracy*  
(Alternative / full: *mond: packing arbitrary mixtures, active-learning training data and stable MD with machine-learned force fields — groundwork for simulating electrochemical reactions*)  
**Authors:** Julian M. Kleber (`0000-0001-5518-0932`)  
**Target Venue:** *Journal of Chemical Information and Modeling (JCIM)* / *Digital Discovery* / *ChemRxiv*  
**Status:** **Scaffold & Research Design Complete (Ready for S0 Calibration on Engine)**.

---

## 1. Executive Summary & Objective

This project establishes the computational groundwork for simulating electrochemical interfaces and multi-component electrolyte mixtures using machine-learned force fields (MLFFs) matching ab initio molecular dynamics (AIMD) accuracy.

### Core Scientific Questions & Hypotheses
* **RQ1 (Packing - H1):** Can automated Monte Carlo packing (`mond`) build physically valid boxes of arbitrary liquid mixtures and salts with $<2\%$ density error vs. experiment?
* **RQ2 (Accuracy - H2):** Do fine-tuned equivariant foundation models (MACE-OFF / MACE-MP-0) outperform from-scratch models and non-conservative MPNNs at equal DFT budget ($\le 3\text{k}$ labels)?
* **RQ3 (Stability - H3):** Do conservative equivariant MLFFs achieve $\ge 1\text{ ns}$ stable NVT dynamics without unphysical bond breaking or temperature divergence, and NVE drift $<1\text{ meV/atom/ps}$?
* **RQ4 (Data Efficiency - H4):** Does query-by-committee (QBC) uncertainty active learning reduce required DFT label calls by $\ge 2\times$ over random sampling?
* **RQ5 (Composition Transfer - H5):** Can models trained on specific electrolyte ratios (e.g., EC:EMC) transfer zero-shot or with $\le 50$ adaptation labels to unseen compositions (DMC)?

---

## 2. Directory Layout & File Map

* **Paper Files:**
  * [`paper/main.tex`](paper/main.tex): LaTeX methods manuscript.
  * [`paper/OUTLINE.md`](paper/OUTLINE.md): Detailed IMRaD outline mapped to hypotheses H1–H5.
  * [`paper/inventory.md`](paper/inventory.md): Forensic audit of prior 2025 implementations (`mond`, `mlff-tutorial`, `openmm-mixtures`), detailing the $103.64\times$ Verlet acceleration bug and its resolution.
  * [`paper/references.bib`](paper/references.bib): Verified open-access bibliography.
* **Experiment Runbook & Engine Protocol:**
  * [`experiments/PLAN.md`](experiments/PLAN.md): Complete execution guide for stages S0–S6 on `engine` GPU/CPU.
* **Code Packages:**
  * `src/aimd_mlff_methods/data/splits.py`: Grouped trajectory-block and composition splits (zero frame-level leakage).
  * `src/aimd_mlff_methods/validation/metrics.py`: Accurate force/energy MAE/RMSE with bootstrap CIs and stability rules.
  * `src/aimd_mlff_methods/integrator.py`: Unit-correct velocity Verlet and NVE energy conservation checkers.

---

## 3. Remote Engine Integration (`engine` / `192.168.10.29`)

* **Remote Path:** `/home/s4il0r/aimd-mlff-methods`
* **Compute Resources:** 1× RTX 3060 12GB, 32 CPU threads, 61 GB RAM.
* **Execution Flow:**
  1. **Parallel Setup:** Run DFT labelling across the 32 CPU threads while dedicating the GPU to MACE fine-tuning / MLFF MD.
  2. **Backup Strategy:** Append-only restic backups to Hetzner bucket tagged `aimd_mlff`.

---

## 4. Immediate Step-by-Step Instructions (Incoming Agent Runbook)

### Step 1: Stage 0 (S0) Environment & Timing Calibration on `engine` (COMPLETE)
Environment active at `/home/s4il0r/micromamba/envs/aimd` with PyTorch 2.5.1+cu121, MACE-torch 0.3.16, NequIP 0.6.2, PySCF 2.14.0, tblite 0.7.0, and `gpu4pyscf-cuda12x`.

Benchmark results logged in [`experiments/s0_timing_report.json`](experiments/s0_timing_report.json):
* **tblite (GFN2-xTB):** **11.5 ms** per 25-atom cluster (candidate presampling is ultra-fast).
* **PySCF CPU (32 threads, $\omega$B97M-V/def2-SVP):** **254.1 s (4.2 min)** for 25-atom $\text{Li}^+(\text{MeCN})_4$.
* **PySCF CPU (32 threads, $\omega$B97M-V/def2-TZVP):** **355.6 s (5.9 min)** for 25-atom $\text{Li}^+(\text{MeCN})_4$ (SCF: 4.75 min, nuclear gradients: 1.17 min).
* **Validation:** Energy difference between CPU and GPU is $<2 \times 10^{-12}\text{ Hartree}$ (machine precision match).

### Step 2: Integrator Smoke Test (COMPLETE)
Velocity Verlet and NVE energy conservation test suite:
```bash
pytest tests/ -v  # 13/13 passed in 0.55s
```

### Step 3: Stage 1 (S1) Methanol Sanity & Stage 2 (S2) `mond` Packing Benchmark
Proceed with S1 mini-benchmark and S2 packing density error evaluation (<2% error vs. experiment across S1, S2, S3).

### Step 4: Fill In Results & Compile Paper
Populate experimental tables in [`paper/main.tex`](paper/main.tex) as each stage (S1–S6) completes and compile via:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
```
