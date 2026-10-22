"""BioEmu steering potential backed by a cryo-EM density target.

Plugs into bioemu.steering (dpm_smc and dpm_fkc) through Hydra:

    fk_potentials:
      - _target_: density_bioemu.BioEmuDensityPotential
        target_path: targets/GltPh_IFS_6x12.pt
        weight: 30.0
        t_on: 0.5

Interface expected by BioEmu: __call__(ca_pos_nm, *, t, sequence) -> energy of shape (batch,).
BioEmu uses reward = -energy, so energy = -w(t) * CC.

Time schedule: early in reverse diffusion the x0 estimate is a blob and its CC is noise.
Guidance is switched on smoothly, w(t) = weight * sigmoid((t_on - t) / ramp). The ramp is
differentiable in t, so FKC's d(reward)/dt term in the importance weights stays correct.

Pose (Pilot A): each x0 is Kabsch-superposed onto the target's deposited model using only
the state-invariant core, with stop-gradient on the rotation and translation.
"""

from __future__ import annotations

import torch

from density_potential import DensityMap, DensityScorer, kabsch


class BioEmuDensityPotential:
    def __init__(
        self,
        target_path: str,
        weight: float = 30.0,
        t_on: float = 0.5,
        ramp: float = 0.05,
        blur_res: float | None = None,
        guidance_steering: bool = True,
        **_,
    ):
        tgt = torch.load(target_path, map_location="cpu", weights_only=False)
        self.label = tgt["label"]
        self.sequence = tgt["sequence"]
        dmap = DensityMap(tgt["map_data"], tgt["map_origin"], tgt["map_voxel"], tgt["map_mask"])
        sigma = None if blur_res is None else 0.225 * blur_res
        self.scorer = DensityScorer(dmap, resolution=tgt["resolution"], sigma=sigma)
        self.core = tgt["core_idx"]
        self.ref_core = tgt["reference_ca_A"][self.core]
        self.weight, self.t_on, self.ramp = float(weight), float(t_on), float(ramp)
        self.guidance_steering = guidance_steering
        self._device = torch.device("cpu")

    def _to(self, device):
        if device != self._device:
            self.scorer.to(device)
            self.core = self.core.to(device)
            self.ref_core = self.ref_core.to(device)
            self._device = device

    def schedule(self, t) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32, device=self._device)
        return self.weight * torch.sigmoid((self.t_on - t) / self.ramp)

    def cc(self, ca_pos_nm: torch.Tensor) -> torch.Tensor:
        """Masked CC of each structure in the batch, after the oracle core pose."""
        self._to(ca_pos_nm.device)
        x = ca_pos_nm.float() * 10.0  # nm -> A
        if self.sequence is not None and x.shape[1] != len(self.sequence):
            raise ValueError(f"expected {len(self.sequence)} residues, got {x.shape[1]}")
        with torch.no_grad():
            r, tr = kabsch(x[:, self.core].detach(), self.ref_core)
        x = x @ r.transpose(-1, -2) + tr
        return self.scorer(x)

    def __call__(self, ca_pos_nm: torch.Tensor, *, t=None, sequence=None) -> torch.Tensor:
        self._to(ca_pos_nm.device)
        w = self.weight if t is None else self.schedule(t)
        return -w * self.cc(ca_pos_nm)
