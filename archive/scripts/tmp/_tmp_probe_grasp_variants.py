"""Probe grasp-point / closure / wrist overlays on corrected_P_action."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    corrected_P_action,
    wrist_from_rot,
)
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    axial_summary,
    contacts_axial,
)
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def rot_rel_now(sim):
    u = np.asarray(cylinder_axis_hand(sim), float)
    ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    return np.cross(u, ut)


def run(env, ic, name, **kw):
    g = dict(BASE_GAINS)
    g.update(kw.get("gains", {}))
    close_xy = float(kw.get("close_xy", g.get("xy_soft_capture", 0.042)))
    theta_max = float(kw.get("theta_max", 180.0))
    gate_close = bool(kw.get("gate_close", False))
    vel_match = bool(kw.get("vel_match", False))
    wrist_boost = float(kw.get("wrist_boost", 1.0))
    wait_orient = bool(kw.get("wait_orient", False))
    axial_z = bool(kw.get("axial_z", False))

    if gate_close:
        g["xy_soft_capture"] = close_xy
        g["preclose"] = -1.0

    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    acts = []
    first_bil = house = None
    ax_bil = None
    for i in range(40):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=bil,
        )
        th = float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"])
        err_xy = float(meta.get("err_xy_now", 99.0))
        rr = rot_rel_now(env.sim)

        if axial_z and meta.get("phase") in ("align_xy", "scoop", "soft_capture"):
            # bias vertical chase toward P_TGT depth (hand-frame z of COM)
            pz = float(p_com_hand(env.sim)[2])
            a[2] = float(np.clip(a[2] + 0.5 * (P_TGT[2] - pz) / 0.03, -1, 1))

        if wrist_boost > 1.0 and not contact_summary(env.sim)["bilateral"]:
            g2 = dict(g)
            g2["kw_app"] = float(g.get("kw_app", 0.3)) * wrist_boost
            a[3:6] = wrist_from_rot(rr, g2)

        if wait_orient and th > theta_max and err_xy > 0.015:
            # hold height, keep aligning, do not close
            a[2] = float(g.get("retract", -0.12))
            a[6] = -1.0
            a[3:6] = np.clip(1.4 * wrist_from_rot(rr, {**g, "kw_app": 0.55}), -1, 1)

        if gate_close and meta.get("phase") in ("soft_capture", "scoop", "align_xy"):
            if not contact_summary(env.sim)["bilateral"] and bil == 0:
                allow = err_xy <= close_xy and th <= theta_max
                if not allow:
                    a[6] = -1.0
                    if meta.get("descend"):
                        a[2] = float(g.get("soft_descend", 0.35))
                else:
                    a[6] = 1.0
                    a[0] = float(np.clip(0.3 * a[0], -0.4, 0.4))
                    a[1] = float(np.clip(0.3 * a[1], -0.4, 0.4))
                    a[2] = float(g.get("soft_descend", 0.35))

        if vel_match and meta.get("descend") and err_xy < 0.05 and bil == 0:
            # mild descend matching (object falling → increase hand down)
            a[2] = float(np.clip(max(float(a[2]), 0.45), 0.2, 1.0))

        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        cs = contact_summary(env.sim)
        cons, _, _ = contacts_axial(env.sim)
        ax = axial_summary(cons)
        if any(c["side"] == "hand" for c in cons) and house is None:
            house = i
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                ax_bil = ax
            bil += 1
        elif first_bil is not None:
            bil += 1
        if latched(env.sim):
            break
        if first_bil is not None and i >= first_bil + 6:
            break

    SN = CH = lift = 0
    thf = posf = None
    full = list(acts)
    if first_bil is not None:
        pref = acts[: first_bil + 1]
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        for aa in pref:
            step_legal(env.sim, aa)
        soft = []
        for _ in range(4):
            if not contact_summary(env.sim)["bilateral"] or latched(env.sim):
                break
            aa = np.zeros(7, np.float32)
            aa[6] = 1.0
            aa[2] = 0.2
            step_legal(env.sim, aa)
            soft.append(aa.copy())
        pref2 = pref + soft
        p = {
            "n_lift": 36, "lift": -0.95, "n_return": 14, "n_seat": 28, "n_fine": 0,
            "wy": 0.25, "n_wy": 12, "xy_gain": 0.55, "axis_align": True,
            "use_fine_down": False, "return_vz": -0.18, "seat_vz": 0.08,
        }
        stab = build_stabilization(env, ic, pref2, p)
        if stab is None:
            stab = build_stabilization(env, ic, pref, p)
            if stab is not None:
                full = pref + stab
                lift = 1
        else:
            full = pref2 + stab
            lift = 1
        out = eval_full_sn(env, ic, full, max(1, first_bil + 1), nominal_s=SEARCH_NOMINAL_S)
        SN = int(out["SUCCESS_NEW"])
        CH = int(out["CATCH_HANDOFF_SUCCESS"])
        thf = out.get("SN_mean_theta")
        posf = out.get("SN_mean_pos")

    s = None if ax_bil is None else round(float(ax_bil["mean_s_axial"]), 4)
    print(
        f"{name}: bil@{first_bil} s={s} tail={None if not ax_bil else ax_bil['n_near_tail']} "
        f"ctr={None if not ax_bil else ax_bil['n_near_center']} house={house} lift={lift} "
        f"CH={CH} SN={SN} th={None if thf is None else round(thf, 1)} "
        f"pos={None if posf is None else round(posf, 1)}",
        flush=True,
    )


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    run(env, ic, "A_base")
    run(env, ic, "B_axial_z", axial_z=True)
    run(env, ic, "C_late018", gate_close=True, close_xy=0.018, theta_max=180)
    run(env, ic, "C_late022", gate_close=True, close_xy=0.022, theta_max=180)
    run(env, ic, "C_late025", gate_close=True, close_xy=0.025, theta_max=180)
    run(env, ic, "D_vel", vel_match=True)
    run(env, ic, "E_late_axial", gate_close=True, close_xy=0.020, axial_z=True)
    run(env, ic, "F_w2_late", gate_close=True, close_xy=0.020, wrist_boost=2.0)
    run(env, ic, "F_w3_wait", gate_close=True, close_xy=0.022, theta_max=50,
        wrist_boost=2.5, wait_orient=True)
    run(env, ic, "F_full", gate_close=True, close_xy=0.018, theta_max=55,
        wrist_boost=2.0, axial_z=True, vel_match=True, wait_orient=True)
    # induce early scoop / brush like expert
    run(env, ic, "brush_early_scoop",
        gains={"n_align_retract": 2, "xy_early_scoop": 0.055, "xy_soft_capture": 0.030},
        gate_close=True, close_xy=0.016, theta_max=40, wrist_boost=1.5)
    run(env, ic, "late_and_lift_hold",
        gains={"n_align_retract": 5, "xy_early_scoop": 0.040, "soft_descend": 0.25},
        gate_close=True, close_xy=0.016, theta_max=180)


if __name__ == "__main__":
    main()
