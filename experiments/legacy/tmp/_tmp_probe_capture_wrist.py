"""Capture-time wrist on the stable soft=0.022 catch to break ~18° lock."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    corrected_P_action,
    hand_vel,
    jaw_target_err_privileged,
    obj_vel,
    wrist_from_rot,
)
from scripts.run_freefall_recatch_grasp_point_alignment import BASE_GAINS
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import clear_probe_latch


def rot_rel(sim):
    u = np.asarray(cylinder_axis_hand(sim), float)
    ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
    u = u / (np.linalg.norm(u) + 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    return np.cross(u, ut)


def run(env, ic, name, kw_cap=0.0, n_cap_wrist=8, wy_bias=0.0, preclose_wrist_th=40.0, pre_kw=0.0):
    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=0.022, preclose=-1.0,
            n_bil_seat=8, lift=-0.92,
        )
    )
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    acts = []
    first_bil = None
    for i in range(70):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=bil,
        )
        th = float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"])
        rr = rot_rel(env.sim)
        if meta.get("phase") in ("scoop", "soft_capture") and first_bil is None:
            info = jaw_target_err_privileged(env.sim, False, 0.0)
            v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
            a[2] = float(np.clip(float(a[2]) + 0.8 * float(v_rel[2]), 0.2, 1.0))
        if meta.get("phase") == "soft_capture" and first_bil is None:
            a[0] = float(np.clip(0.5 * a[0], -0.25, 0.25))
            a[1] = float(np.clip(0.5 * a[1], -0.25, 0.25))
            # pre-close wrist once tumbling has reduced theta
            if th < preclose_wrist_th and pre_kw > 0:
                a[3:6] = wrist_from_rot(rr, {"kw_app": pre_kw, "wx0": 0, "wy0": wy_bias, "wz0": 0})

        if first_bil is not None:
            age = i - first_bil
            a[6] = 1.0
            if age < 2:
                a[2] = 0.30
            else:
                a[2] = -0.92
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.65 * err[0] / 0.03, -0.8, 0.8))
                a[1] = float(np.clip(0.65 * err[1] / 0.03, -0.8, 0.8))
            if age < n_cap_wrist:
                w = wrist_from_rot(rr, {"kw_app": kw_cap, "wx0": 0, "wy0": wy_bias, "wz0": 0.05})
                a[3:6] = w

        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                print(name, "bil@", i, "th", round(th, 1), flush=True)
            bil += 1
        elif first_bil is not None:
            bil += 1
        if first_bil is not None and (i - first_bil) in (0, 2, 5, 10, 20, 35):
            po = pose_now(env.sim)
            print(
                f"  +{i-first_bil} th={po['e_theta_deg']:.1f} pos={po['e_pos_mm']:.1f} "
                f"bil={po['bilateral']} latch={latched(env.sim)}",
                flush=True,
            )
        if latched(env.sim):
            break
        if first_bil is not None and i >= first_bil + 45:
            break

    if first_bil is None:
        print(name, "NO BIL", flush=True)
        return
    out4 = eval_full_sn(env, ic, acts, max(1, first_bil + 1), nominal_s=SEARCH_NOMINAL_S)
    out12 = eval_full_sn(env, ic, acts, max(1, first_bil + 1), nominal_s=float(NOMINAL_S))
    print(
        name, "SN4", out4["SUCCESS_NEW"], "CH", out4["CATCH_HANDOFF_SUCCESS"],
        "th", out4.get("SN_mean_theta"), "pos", out4.get("SN_mean_pos"),
        "fail", out4.get("fail_class"), "SN12", out12["SUCCESS_NEW"],
        flush=True,
    )


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    run(env, ic, "base", kw_cap=0.0)
    for kw in (0.35, 0.55, 0.75, 1.0):
        run(env, ic, f"cap_kw{kw}", kw_cap=kw, n_cap_wrist=12, wy_bias=0.0)
    for kw in (0.55, 0.75):
        run(env, ic, f"cap_kw{kw}_wy", kw_cap=kw, n_cap_wrist=16, wy_bias=0.20)
    run(env, ic, "pre+cap", kw_cap=0.7, n_cap_wrist=14, pre_kw=0.4, preclose_wrist_th=45)
    run(env, ic, "pre_only", kw_cap=0.0, pre_kw=0.5, preclose_wrist_th=50)
    run(env, ic, "pre_strong", kw_cap=0.8, n_cap_wrist=20, pre_kw=0.6, preclose_wrist_th=55, wy_bias=0.15)


if __name__ == "__main__":
    main()
