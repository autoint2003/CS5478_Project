"""Probe Controller O horizontal sign and Controller P first-contact geometry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.observations.airborne_obs import SCALE_PREL, SCALE_VREL, relative_state
from envs.simulation.capture import take_capture, restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_pose_feedback import (
    controller_rollout,
    load_root,
    obs_correct_action,
)
from scripts.run_lateral_offset_falling_recatch import (
    finger_frame,
    hand_pos,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch


def geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return n or f"geom_{gid}"


def body_name(model, bid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return n or f"body_{bid}"


def list_contacts(sim):
    obj = int(sim.ids.object_body)
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        other_g = g2 if b1 == obj else g1
        other_b = b2 if b1 == obj else b1
        rows.append({
            "geom": geom_name(sim.model, other_g),
            "geom_id": other_g,
            "body": body_name(sim.model, other_b),
            "body_id": other_b,
            "pos": np.asarray(c.pos, float).tolist(),
            "dist": float(c.dist),
        })
    return rows


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    # ---- O sign audit at IC ----
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    obs = np.asarray(env.observation(), np.float32)
    rel = relative_state(env.sim.model, env.sim.data, env.sim.ids)
    p_rel = obs[27:30] * SCALE_PREL
    print("=== O sign audit at IC ===")
    print("p_rel_obs", p_rel)
    print("p_rel_gt ", rel["p_rel_h"])
    print("P_TGT    ", P_TGT)
    print("err_old  ", P_TGT - p_rel, " -> would command opposite of object")
    print("err_chase", p_rel, " -> chase toward object in hand frame")
    gains = {"kp": 1.2, "kz": 0.5, "kw": 0.5, "vz_bias": 0.25, "wy": 0.0}
    a_old = obs_correct_action(obs, "chase", gains, [])
    print("O chase action (old)", a_old)

    # matched +/- horizontal offsets via IC free-fall zero action a few steps won't flip sign;
    # use privileged relative: command for +p_rel_x should be +a[0] for chase
    for sign in (+1.0, -1.0):
        fake = obs.copy()
        fake[27] = sign * 0.05 / SCALE_PREL  # p_rel_hx
        fake[28] = 0.0
        a = obs_correct_action(fake, "chase", gains, [])
        print(f" fake p_rel_x={sign*0.05:+.3f} -> a_xy=({a[0]:+.3f},{a[1]:+.3f})  "
              f"expected_chase_a0={sign:+.1f}*positive")

    # ---- Action response: unit axes ----
    print("\n=== Action response (5 steps) ===")
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    cap0 = take_capture(env)
    Rh0 = np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    for axis, a7 in [
        ("+vx", [0.5, 0, 0, 0, 0, 0, -1]),
        ("-vx", [-0.5, 0, 0, 0, 0, 0, -1]),
        ("+vy", [0, 0.5, 0, 0, 0, 0, -1]),
        ("+vz", [0, 0, 0.5, 0, 0, 0, -1]),
        ("-vz", [0, 0, -0.5, 0, 0, 0, -1]),
    ]:
        restore_capture(env, cap0)
        clear_probe_latch(env.sim)
        p0 = hand_pos(env.sim).copy()
        for _ in range(5):
            step_legal(env.sim, np.asarray(a7, np.float32))
        dp_w = hand_pos(env.sim) - p0
        dp_h = Rh0.T @ dp_w
        print(f" {axis}: dp_hand={dp_h.round(4)} dp_world={dp_w.round(4)}")

    # ---- P rollout first contact ----
    print("\n=== Controller P first contact ===")
    gains_p = {
        "kp": 1.0, "kz": 0.5, "kw": 0.5, "lift": -0.5, "vz_bias": 0.2,
        "wx_approach": 0.0, "wy_approach": 0.14, "wz_approach": 0.20, "kw_app": 0.4,
    }
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first = None
    for i in range(80):
        ff = finger_frame(env.sim)
        po, mid = obj_pos(env.sim), ff["mid"]
        err = ff["Rh"].T @ (po - mid)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(1.5 * err[0] / 0.02, -1, 1))
        a[1] = float(np.clip(1.5 * err[1] / 0.02, -1, 1))
        a[2] = float(np.clip(0.85 + 0.5 * err[2] / 0.05, 0, 1))
        a[6] = -1.0
        vo = np.asarray(env.sim.data.qvel[env.sim.model.jnt_dofadr[env.sim.ids.object_jnt]:
                                          env.sim.model.jnt_dofadr[env.sim.ids.object_jnt] + 3], float)
        cons_before = list_contacts(env.sim)
        step_legal(env.sim, a)
        cons = list_contacts(env.sim)
        if cons and first is None:
            first = {"step": i, "contacts": cons, "err_h": err.tolist(),
                     "po": po.tolist(), "mid": mid.tolist(), "ph": hand_pos(env.sim).tolist(),
                     "a": a.tolist(), "vo": vo.tolist(),
                     "ff_approach": ff["u_approach"].tolist()}
            print("FIRST CONTACT", json.dumps(first, indent=2))
            break
    if first is None:
        print("no contact in 80 steps")

    # ---- O rollout: does horizontal error grow? ----
    print("\n=== Controller O horizontal error trace ===")
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    hist = []
    for i in range(15):
        obs = np.asarray(env.observation(), np.float32)
        hist.append(obs.copy())
        p = obs[27:30] * SCALE_PREL
        a = obs_correct_action(obs, "chase", gains, hist)
        a[3:6] = 0  # isolate translation
        print(f" i={i} p_rel_xy=({p[0]:+.4f},{p[1]:+.4f}) a_xy=({a[0]:+.3f},{a[1]:+.3f}) a2={a[2]:+.3f}")
        step_legal(env.sim, a)


if __name__ == "__main__":
    main()
