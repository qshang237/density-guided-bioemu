"""Synthetic two-state, two-domain protein and map for testing the density potential.

The 'experimental' map is deliberately NOT made with the scorer's own forward model
(avoids the inverse crime): it adds pseudo side-chain blobs, uses a different
Gaussian width, and adds noise.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from density_potential import DensityMap, simulate_density


def helix(n: int, start: np.ndarray, direction: int) -> np.ndarray:
    """Ideal alpha-helix Calpha trace along z (radius 2.3 A, rise 1.5 A, 100 deg/res)."""
    k = np.arange(n)
    ang = np.deg2rad(100.0) * k
    z = direction * 1.5 * k
    return start + np.stack([2.3 * np.cos(ang), 2.3 * np.sin(ang), z], -1)


def domain(center: np.ndarray, n_per_helix: int = 20) -> np.ndarray:
    """Four-helix bundle on a 10 A square, alternating direction."""
    length = 1.5 * (n_per_helix - 1)
    out = []
    for i, (dx, dy) in enumerate([(-5, -5), (5, -5), (5, 5), (-5, 5)]):
        d = 1 if i % 2 == 0 else -1
        z0 = -length / 2 if d == 1 else length / 2
        out.append(helix(n_per_helix, center + np.array([dx, dy, z0]), d))
    return np.concatenate(out)


def rotation(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def two_state_protein(hinge_deg: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (state_A, state_B, scaffold_indices). Domain 2 swings about a hinge in state B."""
    d1 = domain(np.array([0.0, 0.0, 0.0]))
    d2 = domain(np.array([22.0, 0.0, 0.0]))
    a = np.concatenate([d1, d2])
    pivot = np.array([11.0, 0.0, -15.0])
    r = rotation(np.array([0.0, 1.0, 0.0]), hinge_deg)
    d2b = (d2 - pivot) @ r.T + pivot
    b = np.concatenate([d1, d2b])
    scaffold = np.arange(len(d1))
    return a.astype(np.float32), b.astype(np.float32), scaffold


def hinge_state(state_a: np.ndarray, n_scaffold: int, deg: float) -> np.ndarray:
    """Intermediate conformation: domain 2 rotated by `deg` from state A."""
    pivot = np.array([11.0, 0.0, -15.0])
    r = rotation(np.array([0.0, 1.0, 0.0]), deg)
    out = state_a.copy()
    out[n_scaffold:] = (state_a[n_scaffold:] - pivot) @ r.T + pivot
    return out.astype(np.float32)


def experimental_map(
    ca: np.ndarray,
    resolution: float = 4.5,
    voxel: float = 1.5,
    noise: float = 0.3,
    seed: int = 0,
) -> DensityMap:
    """Map from Calpha plus pseudo side-chain blobs, broader blur, and Gaussian noise."""
    rng = np.random.default_rng(seed)
    side = ca + rng.normal(size=ca.shape) * 1.0
    side += 2.4 * (side - ca) / (np.linalg.norm(side - ca, axis=1, keepdims=True) + 1e-6)
    atoms = np.concatenate([ca, side]).astype(np.float32)
    w = np.concatenate([np.ones(len(ca)), 0.8 * np.ones(len(ca))])

    lo = atoms.min(0) - 12.0
    hi = atoms.max(0) + 12.0
    n = np.ceil((hi - lo) / voxel).astype(int)
    origin = torch.tensor(lo, dtype=torch.float32)
    vs = torch.tensor([voxel] * 3, dtype=torch.float32)
    axes = [origin[d] + vs[d] * torch.arange(n[d], dtype=torch.float32) for d in range(3)]
    sigma = 0.27 * resolution  # deliberately different from the scorer's 0.225
    rho = simulate_density(torch.tensor(atoms), axes, sigma, torch.tensor(w, dtype=torch.float32))[0]
    rho = rho / rho.max()
    rho = rho + noise * float(rho.std()) * torch.randn(rho.shape, generator=torch.Generator().manual_seed(seed))
    return DensityMap(data=rho, origin=origin, voxel_size=vs)
