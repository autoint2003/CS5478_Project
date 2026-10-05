"""Playback scheduling: physics invariance headless vs dummy viewer; guide at dt."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.impact_demo_core import RENDER_HZ, run_episode


class DummyViewer:
    def __init__(self):
        self.sync_t: list[float] = []
        self.user_scn = None
        self.opt = None
        self.n_overlay = 0
        self.cam = SimpleNamespace(
            lookat=np.zeros(3),
            distance=1.0,
            azimuth=0.0,
            elevation=0.0,
            type=0,
            fixedcamid=-1,
        )

    def sync(self):
        self.sync_t.append(time.perf_counter())

    def add_overlay(self, *args, **kwargs):
        self.n_overlay += 1

    def is_running(self):
        return True


def _phys(r: dict) -> dict:
    post = r.get("post_impact") or {}
    return {
        "t_first_contact": r["t_first_contact"],
        "t_release": r["t_release"],
        "pre_vel": np.round(np.asarray(r["pre_contact_vel_world"], float), 8).tolist(),
        "v_hit_hand": np.round(np.asarray(r["v_hit_hand"], float), 8).tolist(),
        "post_e_x": None if not post else round(float(post["e_x_GT"]), 8),
        "post_obj_z": None if not post else round(float(post["obj_z"]), 8),
        "success_GT": r["final"]["success_GT"],
        "fail_kind_GT": r["final"]["fail_kind_GT"],
        "nL": r["final"]["nL"],
        "nR": r["final"]["nR"],
    }


def test_headless_matches_dummy_viewer_zero():
    h = run_episode("zero", viewer=None, interactive=False, screenshots=False, seed=0)
    v = DummyViewer()
    vis = run_episode("zero", viewer=v, interactive=False, screenshots=False, seed=0)
    assert _phys(h) == _phys(vis), (_phys(h), _phys(vis))
    pb = vis["playback"]
    assert pb["physics_hz"] == 500.0
    assert pb["guide_updated_every_physics_step"] is True
    motion = pb["guided_ball_motion"]
    assert motion["n"] > 1000
    assert motion["dt_median"] == 0.002 or abs(motion["dt_median"] - 0.002) < 1e-9
    # Not a 50 Hz piecewise guide: every-10th Δp is not an order of magnitude larger.
    ratio = motion["lag10_vs_other_ratio"]
    assert ratio is not None and ratio < 3.0, motion
    sync = pb["viewer_sync"]
    assert sync["n_sync"] < 0.7 * pb["loop_cost_mean"]["n_physics"]
    mean = sync["mean_s"]
    assert mean is not None and 0.012 < mean < 0.022, sync
    assert sync["fps"] > 45.0
    assert vis["final"]["success_GT"] is False
    assert vis.get("t_continue") is not None
    assert vis["t_eval_end"] is not None
    assert vis["t_eval_end"] <= vis["t_continue"] + 1e-6
    assert vis["t_viewer_end"] > vis["t_continue"]
    cl = vis["continue_lift"]
    assert cl["used_rule_resume"] is False
    assert abs(float(cl["lift_dz"]) - 0.18) < 1e-9
    return h, vis


def test_headless_matches_dummy_viewer_heuristic():
    h = run_episode("heuristic", viewer=None, interactive=False, screenshots=False, seed=0)
    vis = run_episode(
        "heuristic", viewer=DummyViewer(), interactive=False, screenshots=False, seed=0
    )
    assert _phys(h) == _phys(vis), (_phys(h), _phys(vis))
    assert vis["final"]["success_GT"] is True
    assert h["final"]["success_GT"] is True
    assert vis.get("t_continue") is not None
    assert vis["t_continue"] >= vis["t_eval_end"] - 1e-6
    return h, vis


if __name__ == "__main__":
    hz, vz = test_headless_matches_dummy_viewer_zero()
    print("ZERO match", _phys(hz))
    print("ZERO playback", vz["playback"]["viewer_sync"])
    print("ZERO ball", vz["playback"]["guided_ball_motion"])
    print("ZERO cost", vz["playback"]["loop_cost_mean"])
    hh, vh = test_headless_matches_dummy_viewer_heuristic()
    print("HEURISTIC match", _phys(hh))
    print("PASS test_impact_demo_playback")
