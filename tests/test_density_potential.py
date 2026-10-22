"""Milestone 1 checks on synthetic data. Run: pytest -q tests/"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from density_potential import (  # noqa: E402
    CryoEMDensityPotential,
    DensityScorer,
    load_mrc,
    oracle_pose_fn,
    preprocess_map,
    refine_coordinates,
    rmsd,
    save_mrc,
)
from synthetic import experimental_map, hinge_state, rotation, two_state_protein  # noqa: E402

RES = 4.5


@pytest.fixture(scope="module")
def system():
    torch.manual_seed(0)
    a, b, scaffold = two_state_protein(hinge_deg=30.0)
    raw = experimental_map(a, resolution=RES)
    dmap = preprocess_map(raw, threshold=0.2 * float(raw.data.max()), dilate_A=3.0)
    scorer = DensityScorer(dmap, resolution=RES)
    return dict(a=torch.tensor(a), b=torch.tensor(b), scaffold=scaffold, scorer=scorer, raw=raw)


def test_mrc_roundtrip_keeps_frame(system, tmp_path):
    p = tmp_path / "m.mrc"
    save_mrc(system["raw"], str(p))
    back = load_mrc(str(p))
    assert torch.allclose(back.origin, system["raw"].origin, atol=1e-3)
    assert torch.allclose(back.data, system["raw"].data, atol=1e-5)
    # native model must sit in high density after reload (catches axis-order bugs)
    dens = back.value_at(system["a"]).mean()
    assert dens > back.data.mean() + 2 * back.data.std()


def test_1_discrimination(system):
    s, a, b = system["scorer"], system["a"], system["b"]
    near = hinge_state(a.numpy(), len(system["scaffold"]), 8.0)
    ccs = s(torch.stack([a, torch.tensor(near), b]))
    assert ccs[0] > ccs[1] > ccs[2], ccs


def test_2_smooth_decrease_with_noise(system):
    s, a = system["scorer"], system["a"]
    g = torch.Generator().manual_seed(1)
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    means = []
    for lv in levels:
        batch = a[None] + lv * torch.randn((8, *a.shape), generator=g)
        means.append(float(s(batch).mean()))
    assert all(x > y for x, y in zip(means, means[1:])), means


def test_3_gradients_finite(system):
    pot = CryoEMDensityPotential(system["scorer"], weight=10.0)
    x = torch.stack([system["a"], system["b"]]) / 10.0  # nm, like BioEmu
    x = (x + 0.05 * torch.randn_like(x)).requires_grad_(True)
    e = pot(x)
    assert e.shape == (2,)
    e.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() < 1e4


def _hinge_torch(a, n_scaffold, deg):
    """Differentiable version of synthetic.hinge_state (rotation about y through the pivot)."""
    pivot = torch.tensor([11.0, 0.0, -15.0])
    t = torch.deg2rad(deg)
    c, s_ = torch.cos(t), torch.sin(t)
    zero, one = torch.zeros(()), torch.ones(())
    r = torch.stack([torch.stack([c, zero, s_]), torch.stack([zero, one, zero]), torch.stack([-s_, zero, c])])
    moved = (a[n_scaffold:] - pivot) @ r.T + pivot
    return torch.cat([a[:n_scaffold], moved])


def test_4a_gradient_on_collective_coordinate_recovers_native(system):
    """The density gradient, projected on the real conformational degree of freedom, points home."""
    s, a, n = system["scorer"], system["a"], len(system["scaffold"])
    theta = torch.tensor(10.0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.5)
    for _ in range(100):
        opt.zero_grad()
        (-s(_hinge_torch(a, n, theta)[None])[0]).backward()
        opt.step()
    assert abs(float(theta)) < 2.0, float(theta)


def test_4b_free_cartesian_ascent_overfits(system):
    """Documented failure mode: unrestrained per-atom ascent raises CC without approaching native.

    In BioEmu the score network plays the role of the missing restraint; this test pins the
    behaviour so that any change to the reward that alters it is noticed.
    """
    s, a = system["scorer"], system["a"]
    start = torch.tensor(hinge_state(a.numpy(), len(system["scaffold"]), 10.0))
    tr = refine_coordinates(s, start, steps=150, lr=0.05, reference_A=a)
    assert tr["cc"][-1] - tr["cc"][0] > 0.2
    assert tr["rmsd"][-1] > min(tr["rmsd"]) - 1e-6 and min(tr["rmsd"]) > 0.9 * tr["rmsd"][0]


def test_5_capture_radius_grows_with_blur(system):
    """Matched blur gives a plateau far from the target; broad blur gives a monotone funnel."""
    a, n, dmap = system["a"], len(system["scaffold"]), system["scorer"].map
    path = torch.stack([torch.tensor(hinge_state(a.numpy(), n, d)) for d in (15, 20, 25, 30, 40)])
    sharp = DensityScorer(dmap, resolution=RES)(path)
    broad = DensityScorer(dmap, resolution=RES, sigma=0.225 * 14.0)(path)
    sharp_drop = float(sharp[0] - sharp[3])
    broad_drop = float(broad[0] - broad[3])
    assert broad_drop > 3 * sharp_drop, (sharp.tolist(), broad.tolist())
    assert all(x > y for x, y in zip(broad.tolist(), broad.tolist()[1:]))


def test_oracle_pose_uses_only_scaffold(system):
    """Pose from scaffold superposition restores the frame without seeing domain 2."""
    s, b = system["scorer"], system["b"]
    r = torch.tensor(rotation(np.array([1.0, 2.0, 3.0]), 70.0), dtype=torch.float32)
    moved = b @ r.T + torch.tensor([30.0, -12.0, 5.0])
    pose = oracle_pose_fn(system["a"].numpy(), system["scaffold"])
    restored = pose(moved[None])[0]
    assert rmsd(restored, b) < 1e-3
    assert torch.isclose(s(restored[None]), s(b[None]), atol=1e-4)
