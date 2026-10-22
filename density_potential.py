"""Differentiable cryo-EM density potential for guiding protein ensemble generators.

Milestone 1 scope: Calpha Gaussian forward model, masked real-space correlation,
an optional rigid pose (oracle or fixed), and an adapter that matches the
BioEmu steering potential interface. No map-only registration yet.

Conventions
-----------
* All internal coordinates are in Angstrom. BioEmu passes nanometres; the
  adapter converts.
* Map arrays are stored as (nx, ny, nz), indexed [i, j, k] along x, y, z.
  The Cartesian position of voxel (i, j, k) is origin + voxel_size * (i, j, k).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np
import torch

# ChimeraX molmap default: sigma = resolution / (pi * sqrt(2)) ~= 0.225 * resolution
MOLMAP_SIGMA_FACTOR = 1.0 / (math.pi * math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Map container and I/O
# ---------------------------------------------------------------------------


@dataclass
class DensityMap:
    data: torch.Tensor  # (nx, ny, nz) float
    origin: torch.Tensor  # (3,) Angstrom, position of voxel (0, 0, 0)
    voxel_size: torch.Tensor  # (3,) Angstrom
    mask: Optional[torch.Tensor] = None  # (nx, ny, nz) bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    def axes(self) -> list[torch.Tensor]:
        """1D voxel-centre coordinates along x, y, z (Angstrom)."""
        return [
            self.origin[d] + self.voxel_size[d] * torch.arange(n, dtype=self.data.dtype, device=self.data.device)
            for d, n in enumerate(self.shape)
        ]

    def to(self, device=None, dtype=None) -> "DensityMap":
        dtype = dtype or self.data.dtype
        return DensityMap(
            data=self.data.to(device=device, dtype=dtype),
            origin=self.origin.to(device=device, dtype=dtype),
            voxel_size=self.voxel_size.to(device=device, dtype=dtype),
            mask=None if self.mask is None else self.mask.to(device=device),
        )

    def value_at(self, coords: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of the map at (..., 3) Angstrom coordinates."""
        idx = (coords - self.origin) / self.voxel_size  # fractional voxel index
        size = torch.tensor(self.shape, dtype=coords.dtype, device=coords.device)
        norm = 2.0 * idx / (size - 1) - 1.0  # [-1, 1]
        # grid_sample expects (N, C, D, H, W) with grid (..., [x->W, y->H, z->D])
        vol = self.data.permute(2, 1, 0)[None, None]  # (1,1,nz,ny,nx)
        flat = norm.reshape(1, -1, 1, 1, 3)
        out = torch.nn.functional.grid_sample(vol, flat, mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.reshape(coords.shape[:-1])


def load_mrc(path: str, dtype=torch.float32) -> DensityMap:
    """Read an MRC/CCP4 map into x, y, z order with Cartesian origin in Angstrom.

    Handles MAPC/MAPR/MAPS axis permutation. Origin follows the ChimeraX rule:
    use the ORIGIN header field when it is non-zero, otherwise NSTART * voxel size.
    Always confirm with check 'native model sits in density' before trusting it.
    """
    import mrcfile

    with mrcfile.open(path, permissive=True) as m:
        arr = np.asarray(m.data, dtype=np.float32)
        h = m.header
        crs_to_xyz = (int(h.mapc), int(h.mapr), int(h.maps))  # axis (1=x,2=y,3=z) for col,row,sec
        nstart_crs = np.array([h.nxstart, h.nystart, h.nzstart], dtype=float)
        cell = np.array([h.cella.x, h.cella.y, h.cella.z], dtype=float)
        mxyz = np.array([h.mx, h.my, h.mz], dtype=float)
        origin_hdr = np.array([h.origin.x, h.origin.y, h.origin.z], dtype=float)

    voxel = cell / np.where(mxyz > 0, mxyz, 1)
    # numpy array dims are (sec, row, col); find which array dim carries x, y, z
    dim_of_axis = {crs_to_xyz[2]: 0, crs_to_xyz[1]: 1, crs_to_xyz[0]: 2}
    arr = np.transpose(arr, (dim_of_axis[1], dim_of_axis[2], dim_of_axis[3]))
    nstart_xyz = np.zeros(3)
    for crs_idx, axis in enumerate(crs_to_xyz):
        nstart_xyz[axis - 1] = nstart_crs[crs_idx]
    origin = origin_hdr if np.any(origin_hdr != 0) else nstart_xyz * voxel

    return DensityMap(
        data=torch.tensor(np.ascontiguousarray(arr), dtype=dtype),
        origin=torch.tensor(origin, dtype=dtype),
        voxel_size=torch.tensor(voxel, dtype=dtype),
    )


def save_mrc(dmap: DensityMap, path: str) -> None:
    """Write a DensityMap (standard axis order, origin in header)."""
    import mrcfile

    arr = dmap.data.detach().cpu().numpy().astype(np.float32).transpose(2, 1, 0)  # -> (z, y, x)
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.ascontiguousarray(arr))
        vs = dmap.voxel_size.cpu().numpy()
        m.voxel_size = tuple(float(v) for v in vs)
        o = dmap.origin.cpu().numpy()
        m.header.origin.x, m.header.origin.y, m.header.origin.z = (float(v) for v in o)


def read_ca_pdb(path: str, chain: Optional[str] = None) -> tuple[np.ndarray, list[int]]:
    """Calpha coordinates (Angstrom) and residue numbers from a PDB file (first model)."""
    xyz, resnums = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                if chain is not None and line[21] != chain:
                    continue
                if line[16] not in (" ", "A"):  # keep first altloc only
                    continue
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                resnums.append(int(line[22:26]))
    return np.asarray(xyz, dtype=np.float32), resnums


# ---------------------------------------------------------------------------
# Preprocessing (offline, once per map)
# ---------------------------------------------------------------------------


def preprocess_map(
    dmap: DensityMap,
    threshold: Optional[float] = None,
    dilate_A: float = 3.0,
    pad_A: float = 6.0,
    bin_factor: int = 1,
    region_mask: Optional[np.ndarray] = None,
) -> DensityMap:
    """Build a fixed, model-independent mask; crop; optionally bin; normalise in mask.

    threshold: contour level (use the EMDB recommended level when available).
        If None, mean + 3 * std of the whole box is used as a rough default.
    region_mask: optional boolean array (same grid) restricting to one protomer,
        e.g. from map segmentation. Applied before thresholding.
    bin_factor: integer average pooling to coarsen the grid (speed).
    """
    from scipy import ndimage

    data = dmap.data.detach().cpu().numpy().astype(np.float32)
    vs = dmap.voxel_size.cpu().numpy().astype(float)
    origin = dmap.origin.cpu().numpy().astype(float)

    if threshold is None:
        threshold = float(data.mean() + 3.0 * data.std())
    core = data > threshold
    if region_mask is not None:
        core &= region_mask.astype(bool)
    if not core.any():
        raise ValueError("Mask is empty: threshold too high or wrong region mask.")

    r = np.maximum(np.round(dilate_A / vs).astype(int), 0)
    if r.max() > 0:
        zz, yy, xx = np.ogrid[-r[0] : r[0] + 1, -r[1] : r[1] + 1, -r[2] : r[2] + 1]
        ball = (zz / max(r[0], 1)) ** 2 + (yy / max(r[1], 1)) ** 2 + (xx / max(r[2], 1)) ** 2 <= 1.0
        mask = ndimage.binary_dilation(core, structure=ball)
    else:
        mask = core

    # crop to mask bounding box plus padding
    pad = np.ceil(pad_A / vs).astype(int)
    lo = np.maximum(np.argwhere(mask).min(0) - pad, 0)
    hi = np.minimum(np.argwhere(mask).max(0) + pad + 1, data.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    data, mask = data[sl], mask[sl]
    origin = origin + lo * vs

    if bin_factor > 1:
        f = bin_factor
        n = (np.array(data.shape) // f) * f
        data = data[: n[0], : n[1], : n[2]].reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).mean((1, 3, 5))
        mask = mask[: n[0], : n[1], : n[2]].reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).any((1, 3, 5))
        origin = origin + (f - 1) / 2.0 * vs
        vs = vs * f

    vals = data[mask]
    data = (data - vals.mean()) / (vals.std() + 1e-8)

    dt = dmap.data.dtype
    return DensityMap(
        data=torch.tensor(data, dtype=dt),
        origin=torch.tensor(origin, dtype=dt),
        voxel_size=torch.tensor(vs, dtype=dt),
        mask=torch.tensor(mask),
    )


# ---------------------------------------------------------------------------
# Forward model and score
# ---------------------------------------------------------------------------


def simulate_density(
    coords: torch.Tensor,
    axes: Sequence[torch.Tensor],
    sigma: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of isotropic Gaussians on a grid, computed separably.

    coords: (B, N, 3) Angstrom. Returns (B, nx, ny, nz).
    Exact (no cutoff) and fully differentiable. Cost O(B * N * nx * ny * nz).
    """
    if coords.dim() == 2:
        coords = coords[None]
    g = [torch.exp(-((axes[d][None, None, :] - coords[..., d : d + 1]) ** 2) / (2.0 * sigma**2)) for d in range(3)]
    gx = g[0] if weights is None else g[0] * weights[..., None]
    # Contract in two steps so no (B, N, nx, ny, nz) tensor is ever formed:
    # (B, N, nx*ny) from the x and y factors, then a batched matmul over atoms with z.
    b, n = gx.shape[:2]
    nx, ny, nz = gx.shape[-1], g[1].shape[-1], g[2].shape[-1]
    gxy = (gx[..., :, None] * g[1][..., None, :]).reshape(b, n, nx * ny)
    return torch.bmm(gxy.transpose(1, 2), g[2]).reshape(b, nx, ny, nz)


def masked_cc(sim: torch.Tensor, exp: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation inside a fixed mask. sim (B, ...), exp and mask (...)."""
    s = sim[:, mask]
    e = exp[mask]
    s = s - s.mean(dim=-1, keepdim=True)
    e = e - e.mean()
    return (s * e).sum(-1) / (s.norm(dim=-1) * e.norm() + eps)


class DensityScorer:
    """Calpha Gaussian density vs experimental map, scored by masked CC (higher is better)."""

    def __init__(self, dmap: DensityMap, resolution: float, sigma: Optional[float] = None):
        if dmap.mask is None:
            raise ValueError("Run preprocess_map first so the map has a fixed mask.")
        self.map = dmap
        self.resolution = resolution
        self.sigma = sigma if sigma is not None else MOLMAP_SIGMA_FACTOR * resolution
        self._axes = dmap.axes()

    def __call__(self, coords_A: torch.Tensor) -> torch.Tensor:
        sim = simulate_density(coords_A, self._axes, self.sigma)
        return masked_cc(sim, self.map.data, self.map.mask)

    def to(self, device) -> "DensityScorer":
        self.map = self.map.to(device)
        self._axes = self.map.axes()
        return self


# ---------------------------------------------------------------------------
# Rigid pose helpers (oracle pose for Pilot A)
# ---------------------------------------------------------------------------


def kabsch(mobile: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation R and translation t minimising |mobile @ R.T + t - target|. mobile (B, N, 3), target (N, 3)."""
    mc = mobile.mean(-2, keepdim=True)
    tc = target.mean(-2, keepdim=True)
    h = (mobile - mc).transpose(-1, -2) @ (target - tc)  # (B, 3, 3)
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.linalg.det(vt.transpose(-1, -2) @ u.transpose(-1, -2)))
    diag = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    r = vt.transpose(-1, -2) @ diag @ u.transpose(-1, -2)
    t = tc - mc @ r.transpose(-1, -2)
    return r, t


def oracle_pose_fn(reference_A: np.ndarray, indices: Optional[Sequence[int]] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Pose from a reference model superposed on a residue subset, with stop-gradient on the pose.

    Use a subset that does NOT change between states (e.g. the GltPh scaffold or
    trimerisation domain) so the pose carries no information about which state
    is correct. This is the Pilot A engineering condition, not the final protocol.
    """
    ref = torch.as_tensor(reference_A, dtype=torch.float32)
    idx = None if indices is None else torch.as_tensor(list(indices), dtype=torch.long)

    def apply(coords: torch.Tensor) -> torch.Tensor:
        sub = coords if idx is None else coords[:, idx]
        target = ref if idx is None else ref[idx]
        with torch.no_grad():
            r, t = kabsch(sub.detach(), target.to(coords))
        return coords @ r.transpose(-1, -2) + t

    return apply


# ---------------------------------------------------------------------------
# BioEmu adapter
# ---------------------------------------------------------------------------


class CryoEMDensityPotential:
    """Matches bioemu.steering potential interface: __call__(ca_pos_nm, *, t, sequence) -> energy (B,).

    energy = -weight * CC. BioEmu uses reward = -energy, so higher CC means higher reward.
    `weight` is the guidance strength (inverse temperature); calibrate it with ESS.
    """

    def __init__(
        self,
        scorer: DensityScorer,
        weight: float = 1.0,
        pose_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        guidance_steering: bool = True,
    ):
        self.scorer = scorer
        self.weight = weight
        self.pose_fn = pose_fn
        self.guidance_steering = guidance_steering

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        coords = ca_pos_nm * 10.0  # nm -> Angstrom
        if self.pose_fn is not None:
            coords = self.pose_fn(coords)
        return -self.weight * self.scorer(coords)


# ---------------------------------------------------------------------------
# Small utilities used by the checks
# ---------------------------------------------------------------------------


def rmsd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Plain RMSD without superposition (structures assumed in the same frame)."""
    return ((a - b) ** 2).sum(-1).mean(-1).sqrt()


def refine_coordinates(
    scorer: DensityScorer,
    start_A: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
    bond_weight: float = 0.0,
    reference_A: Optional[torch.Tensor] = None,
) -> dict:
    """Gradient ascent on CC from a starting structure. Returns score/RMSD traces."""
    x = start_A.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    trace = {"cc": [], "rmsd": [], "grad_norm": []}
    for _ in range(steps):
        opt.zero_grad()
        cc = scorer(x[None])[0]
        loss = -cc
        if bond_weight > 0:
            bonds = (x[1:] - x[:-1]).norm(dim=-1)
            loss = loss + bond_weight * ((bonds - 3.8) ** 2).mean()
        loss.backward()
        trace["grad_norm"].append(float(x.grad.norm()))
        opt.step()
        trace["cc"].append(float(cc.detach()))
        if reference_A is not None:
            trace["rmsd"].append(float(rmsd(x.detach(), reference_A)))
    trace["final"] = x.detach()
    return trace
