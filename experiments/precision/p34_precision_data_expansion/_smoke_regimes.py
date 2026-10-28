"""Quick smoke: known-good vs diverse regimes."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p34_precision_data_expansion.regimes import RegimeSpec  # noqa: E402
from experiments.precision.p34_precision_data_expansion.run_p34 import (  # noqa: E402
    _init_worker,
    discover_bases,
    process_candidate,
)


def main() -> None:
    rng = np.random.default_rng(34)
    bases = discover_bases(rng)
    print("bases", len(bases), bases[0]["physical_root"], flush=True)
    _init_worker(34)
    specs = [
        RegimeSpec("DZ_ONLY_dx0_dy0_dz12_th0", "DZ_ONLY", 0, 0, 12, bucket="productive"),
        RegimeSpec("COMBINED_dx4_dy0_dz12_th0", "COMBINED", 4, 0, 12, bucket="productive"),
        RegimeSpec("DZ_ONLY_dx0_dy0_dz-12_th0", "DZ_ONLY", 0, 0, -12, bucket="underrepresented"),
        RegimeSpec("DX_ONLY_dx12_dy0_dz0_th0", "DX_ONLY", 12, 0, 0, bucket="underrepresented"),
        RegimeSpec("COMBINED_dx8_dy0_dz8_th0", "COMBINED", 8, 0, 8, bucket="underrepresented"),
    ]
    # Try first 5 donors × productive specs
    for bi, base in enumerate(bases[:5]):
        for spec in specs[:2]:
            t = {
                "tag": f"smoke_{base['physical_root']}_{spec.regime_id}",
                "lineage_id": f"P34SMOKE__{base['physical_root']}__{spec.regime_id}",
                "physical_root": base["physical_root"],
                "donor_id": base["sample_id"],
                "donor_path": base["path"],
                "regime_id": spec.regime_id,
                "family": spec.family,
                "bucket": spec.bucket,
                "target_dx_mm": spec.target_dx_mm,
                "target_dy_mm": spec.target_dy_mm,
                "target_dz_mm": spec.target_dz_mm,
                "dtheta_deg": spec.dtheta_deg,
                "v_lin_hand_mms": [0.0, 0.0, 0.0],
                "w_hand_dps": [0.0, 0.0, 0.0],
                "mass_scale": 1.0,
                "friction_scale": 1.0,
                "seen_fps": [],
                "worker_id": 0,
                "worker_seed": 34,
                "force_oracle_on_p0": False,
            }
            t0 = time.time()
            r = process_candidate(t)
            print(
                base["physical_root"],
                spec.regime_id,
                "class",
                r.get("class"),
                "reject",
                r.get("reject"),
                "n",
                r.get("n_actions"),
                "wall",
                round(time.time() - t0, 1),
                "imm",
                r.get("imm_SUCCESS_NEW"),
                "prec",
                r.get("prec_SUCCESS_NEW"),
                "status",
                r.get("oracle_status"),
                flush=True,
            )
            if r.get("class") == "P1":
                return
    # also try diverse on first donor
    base = bases[0]
    for spec in specs[2:]:
        t = {
            "tag": f"smoke_{base['physical_root']}_{spec.regime_id}",
            "lineage_id": f"P34SMOKE__{base['physical_root']}__{spec.regime_id}",
            "physical_root": base["physical_root"],
            "donor_id": base["sample_id"],
            "donor_path": base["path"],
            "regime_id": spec.regime_id,
            "family": spec.family,
            "bucket": spec.bucket,
            "target_dx_mm": spec.target_dx_mm,
            "target_dy_mm": spec.target_dy_mm,
            "target_dz_mm": spec.target_dz_mm,
            "dtheta_deg": spec.dtheta_deg,
            "v_lin_hand_mms": [0.0, 0.0, 0.0],
            "w_hand_dps": [0.0, 0.0, 0.0],
            "mass_scale": 1.0,
            "friction_scale": 1.0,
            "seen_fps": [],
            "worker_id": 0,
            "worker_seed": 34,
            "force_oracle_on_p0": False,
        }
        t0 = time.time()
        r = process_candidate(t)
        print(
            "diverse",
            spec.regime_id,
            "class",
            r.get("class"),
            "reject",
            r.get("reject"),
            "wall",
            round(time.time() - t0, 1),
            flush=True,
        )


if __name__ == "__main__":
    main()
