"""Accuracy and stability metrics with bootstrap uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

FailureReason = Literal["bond_break", "temperature", "energy_blowup", "none"]


@dataclass
class ErrorMetrics:
    mae: float
    rmse: float
    mae_ci: Tuple[float, float]
    rmse_ci: Tuple[float, float]


def _mae(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - ref)))


def _rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``statistic(values)``."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        stats.append(statistic(sample))
    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return (low, high)


def energy_force_errors(
    pred_energy: np.ndarray,
    ref_energy: np.ndarray,
    pred_forces: np.ndarray,
    ref_forces: np.ndarray,
    *,
    bootstrap_seed: int = 0,
    n_bootstrap: int = 2000,
) -> Tuple[ErrorMetrics, ErrorMetrics]:
    """Per-frame energy errors and per-component force errors with bootstrap CIs."""
    pred_energy = np.asarray(pred_energy, dtype=float)
    ref_energy = np.asarray(ref_energy, dtype=float)
    pred_forces = np.asarray(pred_forces, dtype=float)
    ref_forces = np.asarray(ref_forces, dtype=float)

    e_abs = np.abs(pred_energy - ref_energy)
    f_abs = np.abs(pred_forces - ref_forces).ravel()

    e_mae = float(np.mean(e_abs))
    e_rmse = float(np.sqrt(np.mean((pred_energy - ref_energy) ** 2)))
    f_mae = float(np.mean(f_abs))
    f_rmse = float(np.sqrt(np.mean((pred_forces - ref_forces) ** 2)))

    e_metrics = ErrorMetrics(
        mae=e_mae,
        rmse=e_rmse,
        mae_ci=bootstrap_ci(e_abs, np.mean, seed=bootstrap_seed, n_bootstrap=n_bootstrap),
        rmse_ci=bootstrap_ci(
            (pred_energy - ref_energy) ** 2,
            lambda x: float(np.sqrt(np.mean(x))),
            seed=bootstrap_seed + 1,
            n_bootstrap=n_bootstrap,
        ),
    )
    f_metrics = ErrorMetrics(
        mae=f_mae,
        rmse=f_rmse,
        mae_ci=bootstrap_ci(f_abs, np.mean, seed=bootstrap_seed + 2, n_bootstrap=n_bootstrap),
        rmse_ci=bootstrap_ci(
            (pred_forces - ref_forces).ravel() ** 2,
            lambda x: float(np.sqrt(np.mean(x))),
            seed=bootstrap_seed + 3,
            n_bootstrap=n_bootstrap,
        ),
    )
    return e_metrics, f_metrics


def nve_energy_drift_mev_per_atom_ps(
    total_energies_ev: np.ndarray,
    times_fs: np.ndarray,
    n_atoms: int,
    *,
    kinetic_energies_ev: Optional[np.ndarray] = None,
) -> float:
    """
    Linear drift of total energy (eV) in meV/atom/ps.

    If ``kinetic_energies_ev`` is provided, uses E_total = E_pot + E_kin.
    """
    times_fs = np.asarray(times_fs, dtype=float)
    e_pot = np.asarray(total_energies_ev, dtype=float)
    if kinetic_energies_ev is not None:
        e_tot = e_pot + np.asarray(kinetic_energies_ev, dtype=float)
    else:
        e_tot = e_pot
    if len(times_fs) < 2:
        return float("nan")
    t_ps = times_fs * 1e-3
    slope_ev_per_ps = float(np.polyfit(t_ps, e_tot, 1)[0])
    return slope_ev_per_ps * 1000.0 / n_atoms


def time_to_failure(
    times_fs: np.ndarray,
    *,
    temperatures_K: Optional[np.ndarray] = None,
    t0_K: float = 300.0,
    max_temperature_factor: float = 2.0,
    total_energies_ev: Optional[np.ndarray] = None,
    energy_blowup_factor: float = 10.0,
    min_bond_lengths: Optional[np.ndarray] = None,
    min_allowed_bond_length: float = 0.5,
) -> Tuple[float, FailureReason]:
    """
    First failure time (fs) or ``inf`` if none. Checks T > max_factor×T0, |E| blow-up, bond lengths.
    """
    times_fs = np.asarray(times_fs, dtype=float)
    if len(times_fs) == 0:
        return float("inf"), "none"

    e0 = None
    if total_energies_ev is not None:
        e = np.asarray(total_energies_ev, dtype=float)
        e0 = abs(e[0]) + 1e-12

    for i, t in enumerate(times_fs):
        if temperatures_K is not None and temperatures_K[i] > max_temperature_factor * t0_K:
            return float(t), "temperature"
        if total_energies_ev is not None and abs(e[i]) > energy_blowup_factor * e0:
            return float(t), "energy_blowup"
        if min_bond_lengths is not None and min_bond_lengths[i] < min_allowed_bond_length:
            return float(t), "bond_break"
    return float("inf"), "none"


def radial_distribution_function(
    positions: np.ndarray,
    box_length: float,
    r_max: float,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple RDF g(r) for N particles in a cubic box (minimum-image, uniform density normalization)."""
    pos = np.asarray(positions, dtype=float)
    n_atoms = pos.shape[0]
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins, dtype=float)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dr = pos[j] - pos[i]
            dr = dr - box_length * np.round(dr / box_length)
            r = float(np.linalg.norm(dr))
            if r < r_max:
                bin_idx = int(r / r_max * n_bins)
                if bin_idx < n_bins:
                    hist[bin_idx] += 2.0
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    vol_shell = 4.0 * np.pi * r_centers**2 * dr
    rho = n_atoms / box_length**3
    norm = n_atoms * rho * vol_shell
    norm = np.where(norm > 0, norm, 1.0)
    g_r = hist / norm
    return r_centers, g_r


def compare_radial_distribution(
    r_ref: np.ndarray,
    g_ref: np.ndarray,
    r_pred: np.ndarray,
    g_pred: np.ndarray,
) -> float:
    """Integrated absolute difference ∫|g_pred - g_ref| dr (trapezoid on common grid)."""
    r_common = np.linspace(max(r_ref.min(), r_pred.min()), min(r_ref.max(), r_pred.max()), 200)
    g_ref_i = np.interp(r_common, r_ref, g_ref)
    g_pred_i = np.interp(r_common, r_pred, g_pred)
    return float(np.trapezoid(np.abs(g_pred_i - g_ref_i), r_common))


def diffusion_coefficient_from_msd(
    times_fs: np.ndarray,
    msd_angstrom2: np.ndarray,
    dimension: int = 3,
) -> float:
    """D in Å²/fs from MSD slope (Einstein relation)."""
    t = np.asarray(times_fs, dtype=float) * 1e-3  # ps
    msd = np.asarray(msd_angstrom2, dtype=float)
    if len(t) < 2:
        return float("nan")
    slope = float(np.polyfit(t, msd, 1)[0])
    return slope / (2.0 * dimension)


def compare_diffusion_coefficients(d_ref: float, d_pred: float) -> float:
    """Relative error |d_pred - d_ref| / |d_ref| (nan if ref is 0)."""
    if abs(d_ref) < 1e-15:
        return float("nan")
    return float(abs(d_pred - d_ref) / abs(d_ref))


def aggregate_over_seeds(
    values_per_seed: Sequence[float],
    *,
    min_seeds: int = 3,
) -> Tuple[float, float]:
    """Sample mean and sample std; raises if fewer than ``min_seeds`` values."""
    arr = np.asarray(values_per_seed, dtype=float)
    if len(arr) < min_seeds:
        raise ValueError(f"need at least {min_seeds} seeds, got {len(arr)}")
    return float(np.mean(arr)), float(np.std(arr, ddof=1))
