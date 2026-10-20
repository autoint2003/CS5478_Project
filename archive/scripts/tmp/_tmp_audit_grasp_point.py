"""Matched audit: perp_off01_r15 SUCCESS_NEW expert vs corr_P axial contacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import cylinder_axis_world, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import eval_full_sn
from scripts.run_lateral_offset_falling_recatch import (
    finger_frame,
    hand_pos,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch
from envs.outcomes.horizon_outcome import NOMINAL_S

CYL_HALF = 0.03  # m
SRC_CA = ROOT / "results/diagnostics/freefall_recatch_collision_aware_feedback/trajectories"


def load_acts(path):
    z = np.load(path)
    return [np.asarray(a, np.float32) for a in z["actions"]], int(z["cut"]) if "cut" in z.files else None


def contacts_axial(sim):
    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    po = obj_pos(sim)
    u = cylinder_axis_world(sim)
    rows = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        side = "L" if other == left else ("R" if other == right else ("hand" if other == int(sim.ids.hand_body) else "other"))
        pos = np.asarray(c.pos, float)
        s = float(np.dot(pos - po, u))  # axial from COM, ±half
        d_end = CYL_HALF - abs(s)
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            pass
        rows.append({
            "side": side,
            "s_axial": s,
            "d_end": d_end,
            "near_tail": abs(s) > 0.018,
            "near_center": abs(s) < 0.012,
            "pos": pos.tolist(),
            "fn": float(force[0]),
        })
    return rows, u, po


def aperture(sim):
    # finger joint opening approx
    idx = sim.ids.finger_jnt
    if hasattr(idx, "__len__"):
        idx = int(idx[0])
    else:
        idx = int(idx)
    return float(sim.data.qpos[idx])


def run_trace(env, ic, acts, label):
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    rows = []
    events = {}
    bil_since = None
    for i, a in enumerate(acts):
        a = np.asarray(a, np.float32)
        step_legal(env.sim, a)
        cs = contact_summary(env.sim)
        cons, u, po = contacts_axial(env.sim)
        ff = finger_frame(env.sim)
        p_rel = p_com_hand(env.sim)
        pose = pose_now(env.sim)
        ax_L = [c["s_axial"] for c in cons if c["side"] == "L"]
        ax_R = [c["s_axial"] for c in cons if c["side"] == "R"]
        mean_ax = float(np.mean([c["s_axial"] for c in cons if c["side"] in ("L", "R")])) if any(c["side"] in ("L", "R") for c in cons) else None
        row = {
            "label": label,
            "step": i,
            "a_grip": float(a[6]),
            "a_xyz": a[:3].tolist(),
            "aper": aperture(env.sim),
            "bil": int(cs["bilateral"]),
            "nL": int(cs["nL"]),
            "nR": int(cs["nR"]),
            "p_rel": p_rel.tolist(),
            "e_theta": pose["e_theta_deg"],
            "e_pos_mm": pose["e_pos_mm"],
            "mean_s_axial": mean_ax,
            "max_abs_s": float(max((abs(c["s_axial"]) for c in cons if c["side"] in ("L", "R")), default=0)),
            "n_near_tail": sum(1 for c in cons if c["side"] in ("L", "R") and c["near_tail"]),
            "n_near_center": sum(1 for c in cons if c["side"] in ("L", "R") and c["near_center"]),
            "sides": sorted({c["side"] for c in cons}),
            "obj_z": float(po[2]),
            "mid_err": (ff["Rh"].T @ (po - ff["mid"])).tolist(),
        }
        rows.append(row)
        if cs["nL"] > 0 and "first_L" not in events:
            events["first_L"] = dict(row)
        if cs["nR"] > 0 and "first_R" not in events:
            events["first_R"] = dict(row)
        if cs["bilateral"] and "first_bil" not in events:
            events["first_bil"] = dict(row)
            bil_since = i
        if float(a[6]) > 0.5 and "close_cmd" not in events:
            events["close_cmd"] = dict(row)
        if float(a[2]) > 0.5 and "descend_cmd" not in events:
            events["descend_cmd"] = dict(row)
        if latched(env.sim):
            events["latch"] = dict(row)
            break
    out = eval_full_sn(env, ic, acts, max(1, (bil_since or len(acts) // 3)), nominal_s=float(NOMINAL_S))
    return rows, events, out


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, exp_acts, cut = load_root("perp_off01_r15")
    # Prefer SUCCESS_NEW if available
    sn = ROOT / "results/expert_generation/horizontal_offset_freefall_recatch/trajectories/perp_off01_r15__baseline.npz"
    # baseline is what load_root may return; check for SN
    p100 = ROOT / "results/expert_generation/horizontal_offset_freefall_recatch/trajectories/perp_off01_r100__SUCCESS_NEW.npz"
    corr_p = SRC_CA / "perp_off01_r15__corr_P.npz"

    print("expert n", len(exp_acts), "cut", cut)
    print("P_TGT", P_TGT, "U_TGT", U_TGT)

    rows_e, ev_e, out_e = run_trace(env, ic, exp_acts, "expert")
    print("EXPERT SN", out_e["SUCCESS_NEW"], "CH", out_e["CATCH_HANDOFF_SUCCESS"],
          "th", out_e.get("SN_mean_theta"), "pos", out_e.get("SN_mean_pos"))
    print(" events", {k: (v["step"], v.get("mean_s_axial"), v.get("n_near_tail"), v.get("p_rel")) for k, v in ev_e.items()})

    acts_p, _ = load_acts(corr_p)
    rows_p, ev_p, out_p = run_trace(env, ic, acts_p, "corr_P")
    print("CORR_P SN", out_p["SUCCESS_NEW"], "CH", out_p["CATCH_HANDOFF_SUCCESS"],
          "th", out_p.get("SN_mean_theta"), "pos", out_p.get("SN_mean_pos"))
    print(" events", {k: (v["step"], v.get("mean_s_axial"), v.get("n_near_tail"), v.get("p_rel")) for k, v in ev_p.items()})

    # bil window comparison
    print("\n=== around bil ===")
    for label, rows, ev in (("expert", rows_e, ev_e), ("corr_P", rows_p, ev_p)):
        fb = ev.get("first_bil", {}).get("step")
        if fb is None:
            print(label, "no bil")
            continue
        print(label, "bil@", fb)
        for r in rows[max(0, fb - 2): fb + 6]:
            print(
                f"  {r['step']} grip={r['a_grip']:+.1f} a2={r['a_xyz'][2]:+.2f} bil={r['bil']} "
                f"s_ax={r['mean_s_axial']} tail={r['n_near_tail']} ctr={r['n_near_center']} "
                f"p_rel={np.round(r['p_rel'],3)} mid_err={np.round(r['mid_err'],3)} "
                f"th={r['e_theta']:.1f} aper={r['aper']:.4f}"
            )


if __name__ == "__main__":
    main()
