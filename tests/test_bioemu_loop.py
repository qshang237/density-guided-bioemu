"""End-to-end check of the density potential inside BioEmu's real SMC and FKC loops.

The score network and the colabfold embedding model are replaced by stand-ins (as in
BioEmu's own tests), so this checks plumbing, shapes, devices, gradients and resampling,
not sample quality. Slow on CPU (~1-2 min).
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TARGET = Path(__file__).resolve().parents[1] / "targets" / "GltPh_IFS_6x12.pt"
pytest.importorskip("bioemu")
if not TARGET.exists():
    pytest.skip("run prepare_targets.py first", allow_module_level=True)

from bioemu.sample import generate_batch  # noqa: E402
from bioemu.shortcuts import CosineVPSDE, DiGSO3SDE  # noqa: E402

from prepare_targets import GLTPH_SEQ  # noqa: E402
from run_condition import CONDITIONS, build_config  # noqa: E402

L = len(GLTPH_SEQ)


def score_stand_in(batch, t):
    # differentiable in pos so the FKC gradient path through x0 is exercised
    return {"pos": -0.5 * batch["pos"], "node_orientations": torch.zeros(batch["node_orientations"].shape[0], 3)}


def run(cond, batch_size=10):
    cfg = build_config(cond, str(TARGET), 30.0, 0.5, None, 5, 12)
    denoiser = hydra.utils.instantiate(cfg)
    sdes = {"node_orientations": DiGSO3SDE(num_sigma=10, num_omega=10, l_max=10), "pos": CosineVPSDE()}
    feats = {"single": np.random.randn(L, 384).astype(np.float32), "pair": np.random.randn(L, L, 128).astype(np.float32)}
    with (
        patch("bioemu.colabfold_inline.model_runner._run_model", return_value=feats),
        patch("bioemu.colabfold_inline.model_runner._load_model_and_params", return_value=(MagicMock(), {})),
        patch("bioemu.colabfold_inline.model_runner.download_alphafold_params"),
        patch("bioemu.get_embeds._get_a3m_string", return_value=f">q\n{GLTPH_SEQ}\n"),
    ):
        return generate_batch(score_model=score_stand_in, sequence=GLTPH_SEQ, sdes=sdes, batch_size=batch_size,
                              seed=0, denoiser=denoiser, cache_embeds_dir=None)


@pytest.mark.parametrize("cond", CONDITIONS)
def test_condition_runs(cond):
    t0 = time.time()
    out = run(cond)
    assert out["pos"].shape == (10, L, 3)
    assert torch.isfinite(out["pos"]).all()
    print(cond, f"{time.time() - t0:.1f}s")
