# Experiments

**Execution plan:** see [`PLAN.md`](PLAN.md) for hypothesis-level specs (H1–H5), compute budgets, parallelism, and decisions.

Place reproducible study definitions here (configs, slurm/bash drivers, analysis notebooks).

Suggested naming:

- `mixtures/` — arbitrary composition sweeps (pairs with openmm-mixtures).
- `stability/` — long MLFF-MD runs and drift metrics.
- `data-gen/` — AIMD / DFT reference trajectories fed into mond + mlff-tutorial pipelines.

Each subdirectory should include a short README with inputs, expected artifacts under `outputs/` (gitignored), and links to the corresponding `paper/` sections.
