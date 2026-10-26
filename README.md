# aimd-mlff-methods

Private research repository for a **methods paper** on simulating electrochemical reactions with learned force fields at **ab initio molecular dynamics (AIMD)**-level accuracy.

## Scope

This project develops and documents groundwork for:

- **Arbitrary electrolyte mixtures** — composition-dependent training data and MD protocols that extend beyond single-solvent tutorials.
- **Stability of MLFF-driven MD** — long-horizon checks, energy drift, force/consistency diagnostics, and failure modes at AIMD-like timesteps and temperatures.
- **Training-data generation** — reproducible pipelines from reference QM (DFT / AIMD) to MLFF datasets, with clear provenance for electrochemical environments.

The manuscript lives under [`paper/`](paper/). Executable studies and configs live under [`experiments/`](experiments/). Thin orchestration code lives under [`src/`](src/) and **depends on** the tutorial codebase rather than vendoring it.

## Related repositories

| Repository | Role |
|------------|------|
| [cap-jmk-real/mlff-tutorial](https://github.com/cap-jmk-real/mlff-tutorial) | Public tutorials: MLFF construction, OpenMM integration, and baseline workflows this research builds on. |
| [cap-jmk-real/mond](https://github.com/cap-jmk-real/mond) | Dataset preparation and featurization for atomistic ML. |
| [cap-jmk-real/openmm-mixtures](https://github.com/cap-jmk-real/openmm-mixtures) | Mixture-aware OpenMM setups aligned with complex electrolyte simulations. |

## Development

Install the tutorial tree as the primary dependency (editable install recommended):

```bash
git clone https://github.com/cap-jmk-real/mlff-tutorial.git
git clone git@github.com:cap-jmk-real/aimd-mlff-methods.git
cd aimd-mlff-methods
pip install -e ".[dev]"
pip install -e ../mlff-tutorial   # path install; see pyproject.toml for pinned remote URL
```

Do **not** copy tutorial sources into this repo; import and extend them from `mlff-tutorial` (see `src/aimd_mlff_methods/README.md`).

## Layout

```
paper/          LaTeX manuscript (methods paper)
experiments/    Study configs, notebooks, and result stubs
src/            Small package wiring experiments to mlff-tutorial
```

## License

Research code and writing in this repository are private unless otherwise noted. See `LICENSE` if present.

## Author

Julian M. Kleber
