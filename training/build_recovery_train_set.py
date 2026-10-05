"""Build a physical offset-IC training set for recovery4d SAC.

Legacy D_enter collection is not used. Held-out eval npz files are not read.
Default N=20 is a smoke set; do not treat it as the full training buffer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.recovery_env import save_buffer  # noqa: E402
from training.offset_construct import make_one_training_ic  # noqa: E402

EVAL_OFFSET = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMPACT = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"
OUT_SMOKE = ROOT / "results" / "buffers" / "recovery_train_smoke.npz"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(snaps: list[dict]) -> dict:
    ex = np.array([float(s["e_x"]) for s in snaps], float)
    z = np.array([float(s["object_z"]) for s in snaps], float)
    return {
        "n": len(snaps),
        "e_x_mm_min": float(ex.min() * 1e3),
        "e_x_mm_max": float(ex.max() * 1e3),
        "e_x_mm_mean": float(ex.mean() * 1e3),
        "n_pos": int(np.sum(ex > 0)),
        "n_neg": int(np.sum(ex < 0)),
        "object_z_min": float(z.min()),
        "object_z_max": float(z.max()),
        "v_cmd_max": float(max(np.max(np.abs(s["v_cmd"])) for s in snaps)),
        "source": snaps[0].get("source"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=str(OUT_SMOKE))
    args = p.parse_args()
    h_off = _file_sha(EVAL_OFFSET)
    h_imp = _file_sha(EVAL_IMPACT)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    rng = np.random.default_rng(args.seed)
    snaps = []
    for i in range(args.n):
        snap = make_one_training_ic(cfg, rng)
        snaps.append(snap)
        print(f"  {i:02d}  e_x={1e3*snap['e_x']:+.2f} mm  z={snap['object_z']:.3f}")
    out = Path(args.out)
    save_buffer(out, snaps)
    summary = audit(snaps)
    summary["eval_offset_sha256"] = h_off
    summary["eval_impact_sha256"] = h_imp
    summary["out"] = str(out)
    summary["used_eval_npz"] = False
    print(json.dumps(summary, indent=2))
    h_off2 = _file_sha(EVAL_OFFSET)
    h_imp2 = _file_sha(EVAL_IMPACT)
    if h_off != h_off2 or h_imp != h_imp2:
        raise RuntimeError("held-out eval set files changed")
    print("TRAIN_SET_SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
