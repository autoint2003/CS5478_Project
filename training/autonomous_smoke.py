"""Smoke test of the frozen online path. Does not train and does not score 24 cases.

Runs one impact success, one sliding success, and one undisturbed grasp.
Wrist and open-finger rows in the pre-training baseline are this same impact
episode: those benchmark labels share the zm6 ball.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.geometry_switch import (
    ENTRY_PERSISTENCE_S,
    ENTRY_PERSISTENCE_STEPS,
    ENTRY_POSITION_THRESHOLD_M,
    EXIT_ORIENTATION_THRESHOLD_RAD,
    EXIT_POSITION_THRESHOLD_M,
    EXIT_RELATIVE_SPEED_THRESHOLD_MPS,
    EXIT_STABILITY_STEPS,
    EXIT_STABILITY_WINDOW_S,
)
from envs.airborne_obs import OBS_DIM_AIR
from training.autonomous_episode import make_training_env, run_ball_episode, run_undisturbed
from training.recovery_runtime import T_TASK


def _forbid_meter(sim):
    def _blocked(*_a, **_k):
        raise RuntimeError("DeteriorationMeter.update_mode is not the recovery trigger")

    sim.meter.update_mode = _blocked


def main() -> int:
    if ENTRY_PERSISTENCE_STEPS != 3 or EXIT_STABILITY_STEPS != 5:
        raise RuntimeError("persistence sample counts drifted")
    if abs(ENTRY_POSITION_THRESHOLD_M - 0.006) > 1e-12:
        raise RuntimeError("entry threshold")
    if abs(EXIT_POSITION_THRESHOLD_M - 0.0147) > 1e-12:
        raise RuntimeError("exit position")
    if abs(EXIT_ORIENTATION_THRESHOLD_RAD - 0.35) > 1e-12:
        raise RuntimeError("exit orientation")
    if abs(EXIT_RELATIVE_SPEED_THRESHOLD_MPS - 0.08) > 1e-12:
        raise RuntimeError("exit speed")
    if abs(ENTRY_PERSISTENCE_S - 0.040) > 1e-12 or abs(EXIT_STABILITY_WINDOW_S - 0.100) > 1e-12:
        raise RuntimeError("windows")

    env = make_training_env()
    obs = env.observation()
    if obs.shape != (OBS_DIM_AIR,) or OBS_DIM_AIR != 39:
        raise RuntimeError(f"obs {obs.shape}")
    weight = env.saved["model"]["in_proj.weight"]
    if tuple(weight.shape) != (64, 46):
        raise RuntimeError(f"checkpoint input {tuple(weight.shape)}")
    _forbid_meter(env.sim)
    print("ENV", "obs", int(obs.shape[0]), "in_proj", tuple(weight.shape), flush=True)

    impact = run_ball_episode(env, 6.5, -0.006, 0.0, None)
    print("IMPACT", impact, flush=True)
    _forbid_meter(env.sim)
    slide = run_ball_episode(env, 6.5, -0.006, 0.0, (0.0, 0.001, 0.0))
    print("SLIDE", slide, flush=True)
    _forbid_meter(env.sim)
    healthy = run_undisturbed(env)
    print("HEALTHY", healthy, flush=True)

    def require_retained(name, row):
        if not row["entered"] or row["entry_reason"] != "position":
            raise RuntimeError(f"{name} entry {row['entry_reason']}")
        if row["qpos_reset"] or row["clock_reset"]:
            raise RuntimeError(f"{name} state reset")
        if row["exit_kind"] != "geometry":
            raise RuntimeError(f"{name} exit {row['exit_kind']}")
        if row["t12"] != "RETAINED_TO_12":
            raise RuntimeError(f"{name} t12 {row['t12']}")
        if row["t_end"] < T_TASK - 1e-6:
            raise RuntimeError(f"{name} clock {row['t_end']}")
        if abs(row["first_step_dt"] - 0.02) > 1e-6:
            raise RuntimeError(f"{name} first step {row['first_step_dt']}")
        if not (row["t_entry"] < row["t_exit"] < row["t_end"]):
            raise RuntimeError(f"{name} times")
        # The collision sample is 8 ms after contact. Entry must be later.
        if row["t_impact"] is None or row["t_entry"] < row["t_impact"] + 0.040:
            raise RuntimeError(f"{name} entry too close to contact")

    require_retained("impact_zm6", impact)
    require_retained("slide_dy1", slide)
    if healthy["entered"] or not healthy["t12_reached"]:
        raise RuntimeError(f"healthy false entry {healthy}")
    if healthy["max_pos"] >= ENTRY_POSITION_THRESHOLD_M:
        raise RuntimeError(f"healthy pose {healthy['max_pos']}")
    print("SMOKE PASS", flush=True)
    print(
        "WRIST recatch_open20 share the impact_zm6 episode "
        f"entry {impact['t_entry']:.3f} exit {impact['t_exit']:.3f} t12 {impact['t12']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
