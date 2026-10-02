"""Held-out ZERO vs frozen RULE replay. Evaluation-only; no SAC training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from training.replay_core import (  # noqa: E402
    KP,
    load_impact_ic,
    load_offset_ic,
    make_impact_sim,
    make_offset_sim,
    obs_from_sim,
    params_from_frozen,
    restore_replay,
    rollout_from_replay,
    verify_frozen_hashes,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offset-idx", type=int, default=0)
    p.add_argument("--impact-v", type=float, default=8.5)
    p.add_argument("--which", choices=("offset", "impact", "both"), default="both")
    args = p.parse_args()
    hashes = verify_frozen_hashes()
    print("frozen", hashes)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    assert abs(gains["kp_pos"] - KP) < 1e-9
    params, _ = params_from_frozen()

    if args.which in ("offset", "both"):
        sim = make_offset_sim(cfg)
        ic = load_offset_ic(args.offset_idx)
        restore_replay(sim, ic)
        o = obs_from_sim(sim)
        print("offset IC e_x_mm", 1e3 * o["e_x"], "n", o["nL"], o["nR"])
        _, z = rollout_from_replay(sim, ic, gains, params, True, 2.0)
        _, r = rollout_from_replay(sim, ic, gains, params, False, 4.0)
        print("OFFSET ZERO keep", z["grasp_retention"], "rec", z["recovery_success"], "ef", round(z["e_final_mm"], 2))
        print("OFFSET RULE keep", r["grasp_retention"], "rec", r["recovery_success"], "path", r["path"], "ef", round(r["e_final_mm"], 2), r["modes"])

    if args.which in ("impact", "both"):
        sim = make_impact_sim(cfg)
        ic = load_impact_ic(args.impact_v)
        restore_replay(sim, ic)
        o = obs_from_sim(sim)
        print("impact IC v", args.impact_v, "e_x_mm", 1e3 * o["e_x"], "n", o["nL"], o["nR"])
        _, z = rollout_from_replay(sim, ic, gains, params, True, 2.0)
        _, r = rollout_from_replay(sim, ic, gains, params, False, 4.0)
        print("IMPACT ZERO keep", z["grasp_retention"], "rec", z["recovery_success"], "ef", round(z["e_final_mm"], 2))
        print("IMPACT RULE keep", r["grasp_retention"], "rec", r["recovery_success"], "path", r["path"], "ef", round(r["e_final_mm"], 2), r["modes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
