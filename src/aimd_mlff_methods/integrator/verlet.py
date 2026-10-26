"""Velocity Verlet integrator in ASE-compatible units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from aimd_mlff_methods.units import masses_amu_to_ev_fs2_per_a2

EnergyForcesFn = Callable[[np.ndarray], Tuple[float, np.ndarray]]


@dataclass
class VelocityVerletState:
    """Snapshot after one velocity Verlet step."""

    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    potential_energy: float


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses_amu: np.ndarray,
    dt_fs: float,
    energy_forces_fn: EnergyForcesFn,
    *,
    periodic_boundary: bool = False,
    box_lengths: Optional[np.ndarray] = None,
    mass_scale: float = 1.0,
) -> VelocityVerletState:
    """
    One velocity Verlet step.

    Units
    -----
    positions : Å
    velocities : Å/fs
    forces : eV/Å
    masses_amu : amu (per atom)
    dt_fs : fs
    potential_energy return value : eV

    ``mass_scale`` is applied to the amu→eV·fs²/Å² conversion (1.0 = correct ASE mass).
    Values other than 1.0 are for regression tests of the historical unit bug.
    """
    masses_ev_fs2 = masses_amu_to_ev_fs2_per_a2(masses_amu) * mass_scale
    acc = forces / masses_ev_fs2[:, np.newaxis]

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    positions = positions + velocities * dt_fs + 0.5 * acc * dt_fs**2

    if periodic_boundary:
        if box_lengths is None:
            raise ValueError("box_lengths required when periodic_boundary is True")
        box_lengths = np.asarray(box_lengths, dtype=float)
        positions = positions % box_lengths

    potential_energy, new_forces = energy_forces_fn(positions)
    new_acc = new_forces / masses_ev_fs2[:, np.newaxis]
    velocities = velocities + 0.5 * (acc + new_acc) * dt_fs

    return VelocityVerletState(
        positions=positions,
        velocities=velocities,
        forces=new_forces,
        potential_energy=float(potential_energy),
    )


def run_nve(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_amu: np.ndarray,
    dt_fs: float,
    n_steps: int,
    energy_forces_fn: EnergyForcesFn,
    *,
    periodic_boundary: bool = False,
    box_lengths: Optional[np.ndarray] = None,
    mass_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run NVE dynamics; return ``(potential_energies, times_fs)`` of length ``n_steps + 1``.
    Initial forces and energy come from ``energy_forces_fn(positions)``.
    """
    e_pot, forces = energy_forces_fn(positions)
    energies = [float(e_pot)]
    times = [0.0]
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    f = np.asarray(forces, dtype=float)

    for _ in range(n_steps):
        state = velocity_verlet_step(
            pos,
            vel,
            f,
            masses_amu,
            dt_fs,
            energy_forces_fn,
            periodic_boundary=periodic_boundary,
            box_lengths=box_lengths,
            mass_scale=mass_scale,
        )
        pos, vel, f = state.positions, state.velocities, state.forces
        energies.append(state.potential_energy)
        times.append(times[-1] + dt_fs)

    return np.asarray(energies, dtype=float), np.asarray(times, dtype=float)


def kinetic_energy_ev(
    velocities: np.ndarray,
    masses_amu: np.ndarray,
    *,
    mass_scale: float = 1.0,
) -> float:
    """Kinetic energy in eV from velocities (Å/fs) and masses (amu)."""
    masses_ev_fs2 = masses_amu_to_ev_fs2_per_a2(masses_amu) * mass_scale
    return float(0.5 * np.sum(masses_ev_fs2 * np.sum(velocities**2, axis=1)))


def run_nve_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses_amu: np.ndarray,
    dt_fs: float,
    n_steps: int,
    energy_forces_fn: EnergyForcesFn,
    *,
    periodic_boundary: bool = False,
    box_lengths: Optional[np.ndarray] = None,
    mass_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Like ``run_nve`` but records total energy (pot + kin) each step."""
    e_pot, forces = energy_forces_fn(positions)
    e_kin = kinetic_energy_ev(velocities, masses_amu, mass_scale=mass_scale)
    totals = [float(e_pot) + e_kin]
    times = [0.0]
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    f = np.asarray(forces, dtype=float)

    for _ in range(n_steps):
        state = velocity_verlet_step(
            pos,
            vel,
            f,
            masses_amu,
            dt_fs,
            energy_forces_fn,
            periodic_boundary=periodic_boundary,
            box_lengths=box_lengths,
            mass_scale=mass_scale,
        )
        pos, vel, f = state.positions, state.velocities, state.forces
        e_kin = kinetic_energy_ev(vel, masses_amu, mass_scale=mass_scale)
        totals.append(state.potential_energy + e_kin)
        times.append(times[-1] + dt_fs)

    return np.asarray(totals, dtype=float), np.asarray(times, dtype=float)
