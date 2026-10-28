"""Probe why evaluate_handoff reports DROP while paired nominal holds."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import _bilateral, _dropped, _monitor
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p40_c_handoff_investigation.paths import M111_META
from experiments.precision.p40_c_handoff_investigation.run_p40 import (
    load_c_jobs,
    replay_to_landmarks,
    snap_geo,
)
from models.m111_checkpoint import load_m111_book
from training.recovery_execution import ESCAPE_XY, _escaped


def main():
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = load_c_jobs()
    job = next(j for j in jobs if "gbi_002_2e202e" in j["traj_id"])
    land = replay_to_landmarks(env, job, book, rho)
    cap = land["captures"]["r2"]
    ref = job["ep"]["ref"]
    print("r2_step", land["r2_step"], "ESCAPE_XY", ESCAPE_XY)

    restore_capture(env, cap)
    g = snap_geo(env.sim)
    mon = _monitor(env.sim)
    ph = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    po = np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
    print(
        "PRE",
        "pad_bil",
        g["bilateral"],
        "mon_bil",
        int(_bilateral(mon)),
        "esc",
        int(_escaped(env.sim)),
        "xy",
        float(np.linalg.norm((po - ph)[:2])),
        "z",
        float(po[2]),
        "drop",
        int(_dropped(mon, env.sim)),
        "obj_z_geo",
        g["obj_z"],
    )

    restore_capture(env, cap)
    apply_handoff(env.sim, ref)
    for i in range(8):
        nominal_step(env.sim)
        mon = _monitor(env.sim)
        ph = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
        gg = snap_geo(env.sim)
        print(
            f"nom{i}",
            "mon_bil",
            int(_bilateral(mon)),
            "pad",
            gg["bilateral"],
            "esc",
            int(_escaped(env.sim)),
            "drop",
            int(_dropped(mon, env.sim)),
            "xy",
            float(np.linalg.norm((po - ph)[:2])),
            "z",
            float(po[2]),
            "lift_pdes_z",
            float(env.sim.fsm.p_des[2]),
        )

    restore_capture(env, cap)
    h8 = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    print("eval8", h8["reason"], "NEW", h8["SUCCESS_NEW"], "s0", (h8.get("samples") or [None])[0])

    restore_capture(env, cap)
    h0 = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    print("eval0", h0["reason"], "NEW", h0["SUCCESS_NEW"], "s0", (h0.get("samples") or [None])[0])

    # Compare heights R2 vs terminal
    for tag in ("r2", "mid", "terminal"):
        restore_capture(env, land["captures"][tag])
        g = snap_geo(env.sim)
        mon = _monitor(env.sim)
        print(
            tag,
            "obj_z",
            g["obj_z"],
            "p_des_z",
            g["p_des"][2],
            "z_tgt",
            float(__import__("envs.ballistic_impact", fromlist=["z_tgt_of"]).z_tgt_of(env.sim)),
            "esc",
            int(_escaped(env.sim)),
            "drop",
            int(_dropped(mon, env.sim)),
            "e_pos",
            g["e_pos"],
        )


if __name__ == "__main__":
    main()
