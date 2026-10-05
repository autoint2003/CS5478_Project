"""Ball exists from t=0: guided mocap weld, then free flight, no near-impact teleport."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.impact_demo_core import run_episode
import training.impact_ball as ib
import training.impact_demo_core as core


def test_ball_from_t0_no_late_teleport():
    writes = {"n": 0, "t": []}
    real = core.apply_ball_free_state

    def wrapped(sim, pos, vel):
        writes["n"] += 1
        writes["t"].append(float(sim.data.time))
        real(sim, pos, vel)

    core.apply_ball_free_state = wrapped
    ib.apply_ball_free_state = wrapped
    try:
        r = run_episode("zero", screenshots=False, seed=0)
    finally:
        core.apply_ball_free_state = real
        ib.apply_ball_free_state = real
    assert writes["n"] == 1, f"expected one t=0 write, got {writes}"
    assert writes["t"][0] == 0.0 or writes["t"][0] < 1e-9
    assert r["ball_moved_via_mj_step"]
    assert r["t_first_contact"] is not None
    assert r["t_release"] is not None
    assert r["t_release"] < r["t_first_contact"]
    zs = [row["pos"][2] for row in r["speed_log"]]
    assert max(zs) < 2.5, f"ball left robot-scale scene z={max(zs)}"
    ys = [row["pos"][1] for row in r["speed_log"] if row["t"] <= r["t_release"]]
    assert ys[-1] > ys[0] + 0.15, "guided approach must move toward +Y"
    rel = r["release"]
    assert rel is not None
    assert float(rel["dv"]) < 0.35, f"velocity jump at release dv={rel['dv']}"
    assert float(rel["dp"]) < 0.03, f"position jump at release dp={rel['dp']}"
    hx, hy, hz = abs(r["v_hit_hx"]), abs(r["v_hit_hy"]), abs(r["v_hit_hz"])
    assert hx > hy and hx > hz, f"impact not lateral in hand frame {r['v_hit_hand']}"
    samples = r["samples"]
    p0 = np.array(samples["0"]["pos"], float)
    p50 = np.array(samples["50"]["pos"], float)
    assert float(np.linalg.norm(p50 - p0)) > 0.05


if __name__ == "__main__":
    test_ball_from_t0_no_late_teleport()
    print("PASS test_ball_from_t0_no_late_teleport")
