"""
Stage 0 (S0) Environment & Timing Calibration Benchmark.

Benchmarks single-point SCF energy and analytic nuclear gradients for:
1. GFN2-xTB (via tblite)
2. PySCF CPU (multithreaded)
3. PySCF GPU (via gpu4pyscf if available)

Across:
- S1 Sanity: Methanol (6 atoms)
- S2 Electrolyte cluster: Li+(ACN)4 (25 atoms)
- S3 Electrolyte cluster: Li+(EC)4 (41 atoms)

Outputs timing results to experiments/s0_timing_report.json.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    system_name: str
    n_atoms: int
    engine: str
    method: str
    basis: Optional[str]
    threads: int
    scf_wall_s: float
    grad_wall_s: float
    total_wall_s: float
    energy: float
    grad_norm: float
    converged: bool
    notes: str = ""


def build_methanol() -> tuple[list[str], np.ndarray, int]:
    """Methanol molecule: 6 atoms, neutral."""
    symbols = ["C", "O", "H", "H", "H", "H"]
    coords = np.array([
        [-0.0474,  0.6654,  0.0000],
        [-0.0474, -0.7554,  0.0000],
        [ 0.9856,  1.0264,  0.0000],
        [-0.5484,  1.0374,  0.8910],
        [-0.5484,  1.0374, -0.8910],
        [ 0.8496, -1.0794,  0.0000],
    ])
    return symbols, coords, 0


def build_li_acn4() -> tuple[list[str], np.ndarray, int]:
    """Li+(MeCN)4 tetrahedral solvated cluster: 25 atoms, +1 charge."""
    # Central Li at origin, 4 ACN ligands along tetrahedral vectors
    symbols = ["Li"]
    coords = [[0.0, 0.0, 0.0]]

    # Normalized tetrahedral directions
    directions = np.array([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ]) / np.sqrt(3.0)

    # ACN internal geometry: N-(1.16)-C-(1.46)-C(methyl)
    # Li-N distance ~2.04 A
    r_li_n = 2.04
    r_n_c = 1.16
    r_c_c = 1.46
    r_c_h = 1.09

    for d in directions:
        pos_n = d * r_li_n
        pos_c1 = d * (r_li_n + r_n_c)
        pos_c2 = d * (r_li_n + r_n_c + r_c_c)

        symbols.extend(["N", "C", "C"])
        coords.extend([pos_n.tolist(), pos_c1.tolist(), pos_c2.tolist()])

        # Construct 3 methyl hydrogens staggered around d
        # Find orthogonal vectors
        if abs(d[0]) < 0.8:
            v_perp = np.cross(d, [1.0, 0.0, 0.0])
        else:
            v_perp = np.cross(d, [0.0, 1.0, 0.0])
        v_perp = v_perp / np.linalg.norm(v_perp)
        v_perp2 = np.cross(d, v_perp)

        cos_tetra = np.cos(np.radians(109.5 - 90.0))  # tilt
        sin_tetra = np.sin(np.radians(109.5 - 90.0))

        for angle_deg in [0.0, 120.0, 240.0]:
            rad = np.radians(angle_deg)
            radial = np.cos(rad) * v_perp + np.sin(rad) * v_perp2
            h_dir = d * sin_tetra + radial * cos_tetra
            pos_h = pos_c2 + h_dir * r_c_h
            symbols.append("H")
            coords.append(pos_h.tolist())

    return symbols, np.array(coords), 1


def build_pyscf_mol(symbols: list[str], coords: np.ndarray, charge: int, basis: str):
    from pyscf import gto
    atom_list = []
    for s, c in zip(symbols, coords):
        atom_list.append(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    atom_str = "; ".join(atom_list)
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, verbose=0)
    return mol


def benchmark_xtb(system_name: str, symbols: list[str], coords: np.ndarray, charge: int) -> BenchmarkResult:
    """Benchmark GFN2-xTB using tblite."""
    import ase
    from tblite.ase import TBLite

    atoms = ase.Atoms(symbols=symbols, positions=coords)
    atoms.calc = TBLite(method="GFN2-xTB", charge=charge)

    t0 = time.time()
    energy = atoms.get_potential_energy()  # in eV
    t1 = time.time()

    forces = atoms.get_forces()  # in eV / A
    t2 = time.time()

    scf_wall = t1 - t0
    grad_wall = t2 - t1
    grad_norm = float(np.linalg.norm(forces))

    return BenchmarkResult(
        system_name=system_name,
        n_atoms=len(symbols),
        engine="tblite",
        method="GFN2-xTB",
        basis=None,
        threads=int(os.environ.get("OMP_NUM_THREADS", "32")),
        scf_wall_s=scf_wall,
        grad_wall_s=grad_wall,
        total_wall_s=scf_wall + grad_wall,
        energy=float(energy),
        grad_norm=grad_norm,
        converged=True,
        notes="Fast presampling baseline",
    )


def benchmark_pyscf(
    system_name: str,
    symbols: list[str],
    coords: np.ndarray,
    charge: int,
    xc: str = "wb97m-v",
    basis: str = "def2-svp",
    use_gpu: bool = False,
) -> BenchmarkResult:
    """Benchmark PySCF SCF + nuclear gradient."""
    from pyscf import dft

    mol = build_pyscf_mol(symbols, coords, charge=charge, basis=basis)
    engine_name = "pyscf-gpu" if use_gpu else "pyscf-cpu"

    mf = dft.RKS(mol, xc=xc)
    if use_gpu:
        try:
            mf = mf.to_gpu()
        except Exception as e:
            return BenchmarkResult(
                system_name=system_name,
                n_atoms=len(symbols),
                engine=engine_name,
                method=xc,
                basis=basis,
                threads=1,
                scf_wall_s=0.0,
                grad_wall_s=0.0,
                total_wall_s=0.0,
                energy=0.0,
                grad_norm=0.0,
                converged=False,
                notes=f"to_gpu failed: {e}",
            )

    t0 = time.time()
    energy = mf.kernel()
    t1 = time.time()

    t_grad0 = time.time()
    g = mf.nuc_grad_method().kernel()
    t_grad1 = time.time()

    scf_wall = t1 - t0
    grad_wall = t_grad1 - t_grad0
    grad_norm = float(np.linalg.norm(g))

    return BenchmarkResult(
        system_name=system_name,
        n_atoms=len(symbols),
        engine=engine_name,
        method=xc,
        basis=basis,
        threads=int(os.environ.get("OMP_NUM_THREADS", "32")),
        scf_wall_s=scf_wall,
        grad_wall_s=grad_wall,
        total_wall_s=scf_wall + grad_wall,
        energy=float(energy),
        grad_norm=grad_norm,
        converged=bool(mf.converged),
        notes=f"Basis nbas={mol.nbas}, nao={mol.nao}",
    )


def run_all_benchmarks(output_path: str = "experiments/s0_timing_report.json"):
    systems = [
        ("S1-Methanol", build_methanol()),
        ("S2-Li(ACN)4", build_li_acn4()),
    ]

    has_gpu = False
    try:
        import gpu4pyscf  # noqa: F401
        has_gpu = True
    except ImportError:
        pass

    results: list[BenchmarkResult] = []

    print("=" * 70)
    print("STAGE 0 (S0) TIMING BENCHMARK EXECUTION")
    print(f"Platform: {platform.node()} ({platform.processor()})")
    print(f"Python: {platform.python_version()}")
    print(f"PySCF GPU available: {has_gpu}")
    print("=" * 70)

    for sys_name, (symbols, coords, charge) in systems:
        print(f"\n--- System: {sys_name} ({len(symbols)} atoms, charge={charge}) ---")

        # 1. GFN2-xTB
        print("  Running tblite GFN2-xTB...")
        try:
            res_xtb = benchmark_xtb(sys_name, symbols, coords, charge)
            results.append(res_xtb)
            print(f"    [xTB] Total: {res_xtb.total_wall_s:.3f}s (SCF: {res_xtb.scf_wall_s:.3f}s, Grad: {res_xtb.grad_wall_s:.3f}s)")
        except Exception as e:
            print(f"    [xTB] FAILED: {e}")

        # 2. PySCF CPU def2-SVP
        print("  Running PySCF CPU (wb97m-v / def2-svp)...")
        try:
            res_cpu_svp = benchmark_pyscf(sys_name, symbols, coords, charge, xc="wb97m-v", basis="def2-svp", use_gpu=False)
            results.append(res_cpu_svp)
            print(f"    [CPU SVP] Total: {res_cpu_svp.total_wall_s:.3f}s (SCF: {res_cpu_svp.scf_wall_s:.3f}s, Grad: {res_cpu_svp.grad_wall_s:.3f}s) | Conv={res_cpu_svp.converged}")
        except Exception as e:
            print(f"    [CPU SVP] FAILED: {e}")

        # 3. PySCF GPU def2-SVP (if available)
        if has_gpu:
            print("  Running PySCF GPU (wb97m-v / def2-svp)...")
            try:
                res_gpu_svp = benchmark_pyscf(sys_name, symbols, coords, charge, xc="wb97m-v", basis="def2-svp", use_gpu=True)
                results.append(res_gpu_svp)
                print(f"    [GPU SVP] Total: {res_gpu_svp.total_wall_s:.3f}s (SCF: {res_gpu_svp.scf_wall_s:.3f}s, Grad: {res_gpu_svp.grad_wall_s:.3f}s) | Conv={res_gpu_svp.converged}")
            except Exception as e:
                print(f"    [GPU SVP] FAILED: {e}")

        # 4. PySCF CPU def2-TZVP
        if len(symbols) <= 25:
            print("  Running PySCF CPU (wb97m-v / def2-tzvp)...")
            try:
                res_cpu_tzvp = benchmark_pyscf(sys_name, symbols, coords, charge, xc="wb97m-v", basis="def2-tzvp", use_gpu=False)
                results.append(res_cpu_tzvp)
                print(f"    [CPU TZVP] Total: {res_cpu_tzvp.total_wall_s:.3f}s (SCF: {res_cpu_tzvp.scf_wall_s:.3f}s, Grad: {res_cpu_tzvp.grad_wall_s:.3f}s) | Conv={res_cpu_tzvp.converged}")
            except Exception as e:
                print(f"    [CPU TZVP] FAILED: {e}")

        if has_gpu and len(symbols) <= 25:
            print("  Running PySCF GPU (wb97m-v / def2-tzvp)...")
            try:
                res_gpu_tzvp = benchmark_pyscf(sys_name, symbols, coords, charge, xc="wb97m-v", basis="def2-tzvp", use_gpu=True)
                results.append(res_gpu_tzvp)
                print(f"    [GPU TZVP] Total: {res_gpu_tzvp.total_wall_s:.3f}s (SCF: {res_gpu_tzvp.scf_wall_s:.3f}s, Grad: {res_gpu_tzvp.grad_wall_s:.3f}s) | Conv={res_gpu_tzvp.converged}")
            except Exception as e:
                print(f"    [GPU TZVP] FAILED: {e}")

    # Write report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "32")),
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved timing calibration report to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "experiments/s0_timing_report.json"
    run_all_benchmarks(out_file)
