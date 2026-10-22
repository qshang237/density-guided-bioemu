"""Quick sanity check of raw BioEmu output (.npz batches), before PDB/XTC conversion.

python check_npz.py runs/diag_*
"""
import glob
import sys

import numpy as np
import torch

from density_bioemu import BioEmuDensityPotential

pot = BioEmuDensityPotential("targets/GltPh_IFS_6x12.pt")
for d in sys.argv[1:]:
    files = sorted(glob.glob(f"{d}/batch_*.npz"))
    if not files:
        print(f"{d}: no batches yet")
        continue
    p = np.concatenate([np.load(f)["pos"] for f in files])  # nm
    bonds = np.linalg.norm(np.diff(p, axis=1), axis=-1) * 10
    ok = np.isfinite(p).all()
    cc = pot.cc(torch.tensor(p)).tolist() if ok else []
    print(f"{d}: n={len(p)} NaN={int(np.isnan(p).sum())} max|x|={np.nanmax(np.abs(p)):.1f} nm "
          f"CA-CA median={np.nanmedian(bonds):.2f} A max={np.nanmax(bonds):.1f} A "
          f"CC(IFS) max={max(cc) if cc else float('nan'):.3f} median={np.median(cc) if cc else float('nan'):.3f}")
