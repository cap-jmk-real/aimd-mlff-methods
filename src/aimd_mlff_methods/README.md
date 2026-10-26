# `aimd_mlff_methods` package

This package holds **thin glue** for the methods paper: experiment drivers, config loaders, and references to tutorial modules.

## Dependency policy

- **Use** [mlff-tutorial](https://github.com/cap-jmk-real/mlff-tutorial) as the canonical implementation for MLFF training, OpenMM hooks, and notebook-derived utilities.
- **Do not** duplicate or vendor tutorial code here. Add a path or VCS dependency instead:

  ```bash
  pip install -e /path/to/mlff-tutorial
  ```

- Prefer importing public tutorial modules once they are exposed as a package; until then, document explicit `PYTHONPATH` or submodule pins in `experiments/*/README.md`.

## Related libraries

- [mond](https://github.com/cap-jmk-real/mond) for dataset featurization.
- [openmm-mixtures](https://github.com/cap-jmk-real/openmm-mixtures) for mixture systems in OpenMM.
