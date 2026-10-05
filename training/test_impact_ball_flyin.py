"""Regression: impact ball flies through mj_step; no park until after last contact."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from training.impact_ball import (
    ImpactBallDriver,
    ball_hits_cylinder,
    ball_pose_twist,
    fly_until_parked,
    run_to_airborne_hold,
)
from training.replay_core import ball_is_parked
from training.visualize_impact_recovery import make_live_impact_sim


def test_flyin_no_qpos_overwrite_before_contact():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_live_impact_sim(cfg)
    gains = gains_from_cfg(cfg)
    run_to_airborne_hold(sim, gains)
    drv = ImpactBallDriver(sim, v_impact=9.0, d_launch=0.30, legacy=False)
    real_apply = __import__("training.impact_ball", fromlist=["apply_ball_free_state"]).apply_ball_free_state
    writes = {"n": 0}

    def wrapped(sim_, pos, vel):
        writes["n"] += 1
        real_apply(sim_, pos, vel)

    import training.impact_ball as ib

    ib.apply_ball_free_state = wrapped
    drv.launch()
    n_launch = writes["n"]
    assert n_launch == 1
    q = int(sim.ball_qadr)
    prev = np.array(sim.data.qpos[q : q + 3], float).copy()
    moved = False
    parked_early = False
    from training.impact_visualization_utils import tick_vw_park
    from training.replay_core import G_HOLD

    while drv.state == "FLYING":
        tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=False)
        now = np.array(sim.data.qpos[q : q + 3], float)
        if float(np.linalg.norm(now - prev)) > 1e-8:
            moved = True
        prev = now.copy()
        drv.note_after_step()
        if ball_is_parked(sim) and drv.state != "PARKED_AFTER_IMPACT":
            parked_early = True
            break
        if float(sim.data.time) - drv.t_launch > 1.0:
            break
    ib.apply_ball_free_state = real_apply
    assert moved, "ball qpos must change via mj_step during FLYING"
    assert writes["n"] == 1, f"extra apply_ball_free_state during FLYING: {writes['n']}"
    assert not parked_early
    assert drv.t_first_contact is not None, "must contact the cylinder"
    assert drv.t_last_contact is not None
    fly_until_parked(sim, drv, gains)
    assert drv.state == "PARKED_AFTER_IMPACT"
    assert ball_is_parked(sim)
    assert drv.t_park is not None
    assert drv.t_park + 1e-12 >= drv.t_last_contact
    # recovery-style ticks keep it parked
    tick_vw_park(sim, np.zeros(3), np.zeros(3), G_HOLD, gains, park=True)
    assert ball_is_parked(sim)


def test_legacy_is_near_surface():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_live_impact_sim(cfg)
    gains = gains_from_cfg(cfg)
    run_to_airborne_hold(sim, gains)
    drv = ImpactBallDriver(sim, v_impact=9.0, d_launch=0.30, legacy=True)
    meta = drv.launch()
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    dist = float(np.linalg.norm(meta["p_launch"] - po))
    assert dist < 0.12, dist


def main() -> int:
    test_legacy_is_near_surface()
    test_flyin_no_qpos_overwrite_before_contact()
    print("test_impact_ball_flyin: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
