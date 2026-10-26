# Plan: better MLFF approach + experiment plan on engine (26 Oct 2026, 14:40 CET)
Nothing launched, nothing pushed. Hypotheses/RQs: see OUTLINE.md §0. Citations only from the verified list in review-1.md.

## 1. Recommended approach
1. **Fix the reference first**: PR #15 (unit-correct Verlet, PBC only for periodic). Add Langevin/NVT via ASE instead of the own integrator. Regenerate all trajectories; keep old frames only as extra single-point labels.
2. **Primary model: MACE** (Batatia 2022, 10.52202/068431-0830), energy-conserving (F = −∇E), equivariant, small (2 layers, 64–128 channels, L=1, r_cut 5 Å) → fits 12 GB for ≤ ~200-atom training cells and ~1k-atom MD.
3. **Foundation starts, zero-shot and fine-tuned**: MACE-OFF (Kovács 2025, 10.1021/jacs.4c07099; organic molecules/liquids, ωB97M-D3(BJ)/def2-TZVPPD level — so labelling at the same level makes fine-tuning consistent) and MACE-MP-0 (arXiv:2401.00096; covers Li/P/F/Cl). UMA (arXiv:2506.23971) zero-shot only after license check (fairchem-core installs; weights gated). Note: MACE-OFF license is academic/non-commercial (to verify for Julian's use).
4. **Second architecture: NequIP** (10.1038/s41467-022-29939-5) from scratch; Allegro (10.1038/s41467-023-36329-y) only if NequIP MD is too slow. Baseline: Julian's direct-force MPNN (fixed graph), to show the conservative/equivariant gain (H2b).
5. **Loss**: weighted energy + force (MACE default schedule: force-heavy, then energy weight raised in SWA phase), per-element E0 from isolated atoms, data in eV/Å float32 train / float64 MD check.
6. **Active learning**: xTB (tblite GFN2) MD on mond boxes for cheap, broad pre-sampling (never ground truth) → pick candidates by committee disagreement (4 MACE models, different seeds; QBC as in DP-GEN, 10.1016/j.cpc.2020.107206, and Smith 2018, 10.1063/1.5023802) or single-model uncertainty (Zaverkin 2024, 10.1038/s41524-024-01254-1) → DFT label clusters cut around the uncertain atom (≤ ~60–100 atoms, CPU) → retrain/fine-tune → MLFF MD at increasing T; failures feed back. Every DFT call logged (atoms, wall time, level) for H4.
7. **Validation**: splits by composition and time block, ≥3 seeds, force/energy MAE+RMSE, plus stability protocol per Fu et al. (arXiv:2210.07237).

## 2. Engine constraints and DFT choice
RTX 3060 12 GB (currently shared with a small nca job), 32 threads, 61 GB RAM, 223 GB free. No GPU rental. psi4 is not pip-installable in nca_venv → options to time in stage 0: (a) psi4 via a user-space micromamba env (no root), (b) PySCF (+ gpu4pyscf on the 3060; availability unverified), (c) real ORCA if licensed (the /usr/bin/orca present is probably the screen reader). Pick the fastest that gives ωB97M-D3(BJ) (or ωB97X-D)/def2-TZVP(D); def2-SVP allowed only for the AL inner loop if relabelled.

## 3. Stages (all resumable: one job = one directory with status.json, checkpoint every N steps/epochs, idempotent restart; data → restic tag aimd_mlff, append-only, encrypted, Hetzner bucket)
| Stage | What | Compute (estimate, calibrated in S0) | ETA (sequential, from start) |
|---|---|---|---|
| S0 pilot (1–2 days) | install mace-torch/nequip/ase/tblite in a new venv `aimd_venv`; time DFT single points (methanol cluster 30/60/100 atoms, LiPF6 in EC cluster) for options a–c; time MACE training (1k frames, 1 epoch) and MLFF MD ns/day on 1k atoms; xTB MD speed; zero-shot MACE-OFF/MP-0 on 50 labelled frames | ~1 CPU-day, <2 GPU-h | day 2 |
| S1 methanol sanity (re-do of his tutorial correctly) | corrected AIMD (short, 3 inits) + 500–1000 labels; MACE scratch/fine-tune vs MPNN; NVE drift, stability | 1–3 CPU-days, ~6 GPU-h | week 1 |
| S2 packing benchmark E5 | mond vs packmol on S1–S3 boxes, OpenMM density | CPU hours | week 1 (parallel) |
| S3 AL on LiClO4/ACN and EC/EMC/LiPF6 (E3) | 4 strategies × 3 seeds, budget ≤3k labels/system, shared label pool to avoid double DFT | DFT bottleneck: ~1–2 CPU-weeks/system (if ~5–15 min/label) | weeks 2–5 |
| S4 accuracy + ablations (E1, E6) | 5 model variants × 3 seeds × 3 systems | ~1–2 GPU-weeks (queued with nca jobs) | weeks 4–6 |
| S5 stability (E2) | 3 seeds × 3 inits × best models, 1 ns NVT + NVE drift, RDF/D/CN vs short AIMD (small cell, a few ps; most expensive single item) | GPU days + AIMD CPU ~1 week | weeks 5–7 |
| S6 transfer (E4) | held-out compositions, zero-shot + ≤50 labels | days | week 7–8 |
Total ~6–8 weeks of engine time, if DFT ≤ ~15 min/label and ≤3k labels/system. S0 decides; if DFT is slower, shrink clusters or drop S3 system count to 2.

## 4. Cloud-agent PR specs (aimd-mlff-methods)
- PR-A `labelling/`: ASE-calculator wrapper for the chosen DFT backend (psi4 | pyscf | orca), cluster carving around a central atom with H-capping-free solvent shells, DFT-call ledger (CSV/SQLite: id, n_atoms, level, wall time, converged), restartable job queue; unit tests incl. eV/Å units and a 2-atom analytic check.
- PR-B `sampling/`: mond → ASE Atoms bridge; xTB (tblite) and MLFF MD drivers (Langevin NVT, VelocityVerlet NVE) with checkpoint/restart; unit test that NVE drift with a harmonic/LJ calculator is ~0 (guards against the 103.6× bug).
- PR-C `al/`: committee and single-model uncertainty selectors, AL loop orchestrator with budgets and seeds, resume from ledger.
- PR-D `bench/`: stability metrics (time-to-failure rules, NVE drift, RDF error, MSD diffusion, Li⁺ CN), accuracy metrics, bootstrap CIs, plots.
- PR-E `experiments/`: configs S0–S6, `run_stage.sh` with status files, restic backup hook (append-only tag aimd_mlff).
- PR-F mond: fix in upstream `cap-jmk-real/mond` instead of the vendored copy in PR #15; NHC removal or ASE thermostat.
Runs with existing pip packages (no new code): MACE training/fine-tuning (`mace_run_train`), NequIP training, MACE-OFF/MP-0 zero-shot, tblite xTB MD, OpenMM densification, packmol comparison.

## 5. Open questions for Julian
1. DFT engine: OK to install psi4 via user-space micromamba on engine (no root)? Or prefer PySCF? Is there an ORCA license?
2. Reference level: ωB97M-D3(BJ)/def2-TZVPPD (matches MACE-OFF) vs cheaper; acceptable per-label cost?
3. Cluster (non-periodic) labels OK for liquids, or must the paper have periodic DFT (CP2K, much costlier)?
4. Licenses: MACE-OFF is non-commercial (to verify); UMA gated — is the paper/academic use enough?
5. Systems: confirm S1 methanol, S2 LiClO4/ACN, S3 EC/EMC/LiPF6 (+ DMC held out). Any system from his old experiments I missed (the gold sample)?
6. Were there other model attempts not in git (local notebooks, another machine)? Repos, engine and bucket show only the PyG MPNN.
7. GPU sharing with NCA/DenoiseOpt queue: priority for AIMD?
8. Merge PR #15 (and move the mond fix upstream) before S0?
