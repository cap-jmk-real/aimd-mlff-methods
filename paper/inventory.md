# AIMD/MLFF inventory (26 Oct 2026, 14:35 CET)
Sources searched: mond (main, 3 commits, last ec25bac 2025-06-19), mlff-tutorial (main f9af113 2025-06-19 + 2 cursor/* branches from today), openmm-mixtures (2 commits, last 4e0dd72 2025-06-07), aimd-mlff-methods (scaffold 1c03dcc, today), engine home (find, depth 6), restic bucket (102 snapshots).
Result: **all of Julian's prior AIMD/MLFF work is in these 3 repos (June 2025).** Engine has no psi4/ase/xtb/mace/methanol traces except my own paper_src copies; the bucket holds only nca_research tags (hermes_raw, paper_src, denoiseopt), no AIMD data. Only branches: main (+ today's cursor PR branches). No old notebooks, model checkpoints or training logs outside the repos. Times below are commit times converted to CEST.

## Concepts / experiments
| # | Concept | Evidence | Systems | Method / model | Data | Units | Training | Validation | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| X1 | Mixture packing in OpenMM (predecessor) | openmm-mixtures fbffa6d "added error producing sim", 2025-06-07 13:51 | small-molecule mixtures | OpenMM + GAFF/AM1-BCC charges | – | nm/ps | – | – | **Failed**: charge step (antechamber) errors; abandoned |
| X2 | MC packing of arbitrary mixtures incl. salts (mond) | mond/src/mond/packing/pack.py, a38628b 2025-06-19 13:54; tutorial 01 (ecb1c33 16:48) | water/ethanol 50:50, 10 Å box, ≤500 molecules; salts planned | random pose + vdW overlap check, iterative add + OpenMM LJPME densify (ff14SB/TIP3P, 0.4 nm cutoff, 5000×2 fs) | – | Å packing / nm MD | – | visual only | **Works** (outputs in notebook); not benchmarked vs packmol (his own todo: "packmol could have done the job?") |
| X3 | Own AIMD loop (Psi4) | mond/simulation/aimd.py; tutorial 02 (d3d3d68 2025-06-19 17:11) | single gas-phase methanol, 6 atoms | velocity Verlet, PBE/def2-SVP DF, NVE, no thermostat, dt 0.05 fs, 10,000 steps | ~10k npz frames committed (methanol_aimd_data_prerun) | E,F eV & eV/Å (ASE-converted, correct); v in Å/fs; **a = F/m with m in amu (missing ×1/103.64)** | – | none (no energy-conservation plot) | **Invalid dynamics**: accelerations ~103.6× too large (PR #15 diagnosis). Frames remain valid QM labels for their geometries |
| X4 | Direct-force MPNN | tutorial 03 (f9af113 2025-06-19 19:10) | methanol only | PyG ForceMPNN: Linear embedding of Z as a float, fully connected graph, edge features = raw (dx,dy,dz,r), one message-passing layer, hidden 256, outputs F directly; no energy head | the ~10k frames of X3 (one trajectory, highly correlated) | eV/Å, float64 | MSE on forces only, Adam 1e-4, batch 64, 5000 epochs, CPU, random rotation augmentation; no normalization, no cutoff | **none**: no split, no MAE/RMSE, only training-loss curve and force-histogram comparison on the training data | Unknown accuracy; "might not converge" in todo.md |
| X5 | MLFF MD | tutorial 04 | methanol | same Verlet (same unit bug) driven by X4 | – | as X3 | – | **no outputs** | Never shown to run stably. Inferred extra bug: forces in step are computed from the `data_point` built *before* the position update, so they lag one step (data_point rebuilt only after the step) |
| X6 | Planned, not done (mond/todo.md) | todo.md | gold sample, LiClO4/ACN, electrolytes, nanoreactor | NHC thermostat (NotImplementedError), QE periodic DFT (NotImplementedError), ORCA parallel (fails), xTB docking stub, COM pulling force, SOAP for small→large cells | – | – | – | – | Stopped 2025-06-19; periodic DFT ~6 days on his PC |

## Why the force fields were bad — diagnosis
**Data**
- D1 Units (certain): Verlet uses F[eV/Å]/m[amu] with v in Å/fs → a ~103.6× too large. All trajectories (incl. the pre-run) are unphysical; sampled geometries are a fast, hot, distorted walk, not a 298 K ensemble. Likely why dt had to be 0.05 fs (inferred).
- D2 Coverage (certain): one molecule, gas phase, one trajectory, no intermolecular contacts, no ions, no liquid. Nothing transfers to mixtures.
- D3 Quantity / correlation (certain): ~10k consecutive frames at 0.05 fs (nominal 0.5 ps; effectively a few hundred highly correlated configurations). Effective sample size is small despite the count (inferred).
- D4 Level: PBE/def2-SVP without dispersion is poor for liquids/H-bonding (well known; not tested here).
- D5 PBC wrap with non-periodic Psi4 (code path off in tutorial 2, but wrong if enabled).
**Architecture**
- A1 Direct-force output, no energy → not conservative; energy drift in NVE expected (Fu et al. arXiv:2210.07237 show low force error ≠ stable MD).
- A2 Not invariant/equivariant: raw Cartesian edge vectors, symmetry only by augmentation; atomic number as a continuous scalar (Z=1 vs 6 vs 8 treated as magnitudes).
- A3 Fully connected graph, no cutoff, one layer: does not scale to boxes and has weak many-body expressiveness.
**Training**
- T1 Force-only MSE, no energy term, no per-element shift/scale or force normalization; float64 CPU; fixed LR, no scheduler/early stopping; no seeds.
**Validation**
- V1 No train/val/test split (not even by time block), no MAE/RMSE.
- V2 No MD stability test, no energy drift, no RDF/diffusion vs AIMD; tutorial 4 has no output.
- V3 Inferred integration bug in tutorial 4 (stale forces, X5).
Bottom line: the concept chain (pack → AIMD → learn → MD) is sound; failure came from invalid reference dynamics + a non-conservative, non-equivariant toy model + zero validation. Nothing indicates the idea itself fails.

## Engine status (checked 14:30 CET)
RTX 3060 12 GB, ~0.5 GB used / ~33 % util by an nca_venv job (pid 1594941); 32 CPU threads, 61 GB RAM, 223 GB free disk. nca_venv: Python 3.13, torch 2.6.0+cu124. pip-installable: mace-torch 0.3.16, nequip 0.19.1, ase 3.29, tblite 0.7 (GFN2-xTB), fairchem-core 2.24 (UMA; weights gated, license check needed). **Not pip-installable: psi4, xtb-python.** /usr/bin/orca exists but is probably the GNOME screen reader (check timed out; unverified). pyscf / gpu4pyscf availability: check interrupted, unverified.
