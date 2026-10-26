"""ASE-compatible unit constants for MD with eV, Å, fs, and amu."""

# 1 amu expressed as mass in eV·fs²/Å² (matches mond PR #15 / mlff-tutorial fix).
AMU_TO_EV_FS2_PER_A2 = 103.642691909


def masses_amu_to_ev_fs2_per_a2(masses_amu):
    """Convert atomic masses from amu to eV·fs²/Å² for ``a = F / m`` with F in eV/Å."""
    import numpy as np

    return np.asarray(masses_amu, dtype=float) * AMU_TO_EV_FS2_PER_A2
