"""NVE tests for velocity Verlet mass units (103.642691909 eV·fs²/Å² per amu)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.units import kB

from aimd_mlff_methods.integrator.verlet import (
    kinetic_energy_ev,
    run_nve_total_energy,
    velocity_verlet_step,
)
from aimd_mlff_methods.units import AMU_TO_EV_FS2_PER_A2
from aimd_mlff_methods.validation.metrics import nve_energy_drift_mev_per_atom_ps


class HarmonicOscillator:
    """Single particle in 1D harmonic potential; energy in eV, positions in Å."""

    def __init__(self, k_ev_per_a2: float, mass_amu: float = 1.0):
        self.k = k_ev_per_a2
        self.mass_amu = mass_amu

    def energy_forces(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        x = positions[0, 0]
        e = 0.5 * self.k * x**2
        f = np.array([[-self.k * x, 0.0, 0.0]])
        return float(e), f


def test_constant_force_displacement_matches_analytic():
    """One step: Δx = 0.5 a dt² with a = 1 Å/fs² when F = AMU_TO eV/Å on 1 amu."""
    dt = 0.1
    masses = np.array([1.0])
    positions = np.zeros((1, 3))
    velocities = np.zeros((1, 3))
    forces = np.array([[AMU_TO_EV_FS2_PER_A2, 0.0, 0.0]])

    def ef(pos):
        return 0.0, forces

    state = velocity_verlet_step(
        positions, velocities, forces, masses, dt, ef, periodic_boundary=False
    )
    expected_dx = 0.5 * dt**2
    assert state.positions[0, 0] == pytest.approx(expected_dx, rel=1e-12)


def test_harmonic_oscillator_nve_low_drift():
    ho = HarmonicOscillator(k_ev_per_a2=1.0)
    masses = np.array([ho.mass_amu])
    pos = np.array([[0.5, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0]])
    dt = 0.01
    n_steps = 5000
    e_tot, times = run_nve_total_energy(pos, vel, masses, dt, n_steps, ho.energy_forces)
    drift = nve_energy_drift_mev_per_atom_ps(e_tot, times, 1)
    assert abs(drift) < 0.05


def test_wrong_mass_scale_inflates_acceleration():
    """Historical bug: ``a = F/m_amu`` instead of ``F/m_ev_fs2`` → ~103.6× larger displacement."""
    dt = 0.1
    masses = np.array([1.0])
    positions = np.zeros((1, 3))
    velocities = np.zeros((1, 3))
    forces = np.array([[AMU_TO_EV_FS2_PER_A2, 0.0, 0.0]])

    def ef(pos):
        return 0.0, forces

    ok = velocity_verlet_step(
        positions, velocities, forces, masses, dt, ef, mass_scale=1.0
    )
    bug = velocity_verlet_step(
        positions,
        velocities,
        forces,
        masses,
        dt,
        ef,
        mass_scale=1.0 / AMU_TO_EV_FS2_PER_A2,
    )
    ratio = bug.positions[0, 0] / ok.positions[0, 0]
    assert ratio == pytest.approx(AMU_TO_EV_FS2_PER_A2, rel=1e-6)


def test_water_emt_nve_energy_reasonable():
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]])
    atoms.calc = EMT()
    masses = atoms.get_masses()
    pos = atoms.get_positions()
    rng = np.random.default_rng(0)
    vel = rng.normal(size=pos.shape)
    # Remove COM drift; scale to ~300 K
    vel -= vel.mean(axis=0)
    t_target = 300.0
    e_kin = kinetic_energy_ev(vel, masses)
    scale = np.sqrt(1.5 * len(atoms) * kB * t_target / e_kin)
    vel *= scale

    def ef(p):
        atoms.set_positions(p)
        return float(atoms.get_potential_energy()), atoms.get_forces()

    dt = 0.25
    n_steps = 400
    e_tot, times = run_nve_total_energy(pos, vel, masses, dt, n_steps, ef)
    drift = nve_energy_drift_mev_per_atom_ps(e_tot, times, len(atoms))
    assert abs(drift) < 10.0
