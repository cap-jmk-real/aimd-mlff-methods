import numpy as np
import pytest

from aimd_mlff_methods.validation.metrics import (
    aggregate_over_seeds,
    compare_radial_distribution,
    energy_force_errors,
    nve_energy_drift_mev_per_atom_ps,
    radial_distribution_function,
    time_to_failure,
)


def test_energy_force_errors_zero_on_match():
    e = np.array([1.0, 2.0, 3.0])
    f = np.random.randn(4, 3)
    e_m, f_m = energy_force_errors(e, e, f, f, bootstrap_seed=0, n_bootstrap=200)
    assert e_m.mae == 0.0 and f_m.mae == 0.0


def test_nve_drift_linear():
    times = np.array([0.0, 1000.0, 2000.0])  # fs
    # 1 meV/atom/ps → 0.001 eV/atom/ps total slope for 2 atoms → 0.002 eV/ps
    energies = 0.002 * (times * 1e-3)
    drift = nve_energy_drift_mev_per_atom_ps(energies, times, n_atoms=2)
    assert drift == pytest.approx(1.0, rel=0.05)


def test_time_to_failure_temperature():
    times = np.array([0.0, 10.0, 20.0])
    temps = np.array([300.0, 400.0, 700.0])
    t_fail, reason = time_to_failure(times, temperatures_K=temps, t0_K=300.0)
    assert t_fail == 20.0 and reason == "temperature"


def test_rdf_compare_identical():
    pos = np.random.randn(20, 3) * 2
    r, g = radial_distribution_function(pos, box_length=20.0, r_max=8.0, n_bins=40)
    err = compare_radial_distribution(r, g, r, g)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_aggregate_requires_three_seeds():
    with pytest.raises(ValueError):
        aggregate_over_seeds([1.0, 2.0], min_seeds=3)
    mean, std = aggregate_over_seeds([1.0, 2.0, 3.0, 4.0], min_seeds=3)
    assert mean == pytest.approx(2.5)
    assert std > 0
