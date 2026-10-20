"""Deep-dive vm2.5 good-theta catch hold + stab."""
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
from evaluation.grasp_configuration import cylinder_axis_hand, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    corrected_P_action,
    hand_vel,
    jaw_target_err_privileged,
    obj_vel,
)
from scripts.run_freefall_recatch_grasp_point_alignment import BASE_GAINS
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=0.020, preclose=-1.0,
            n_bil_seat=12, lift=-0.7,
        )
    )
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    acts = []
    first_bil = None
    vm_gain = 2.5
    for i in range(80):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=bil,
        )
        info = jaw_target_err_privileged(env.sim, False, 0.0)
        v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
        if first_bil is None:
            if meta.get("phase") in ("scoop", "soft_capture"):
                a[2] = float(np.clip(float(a[2]) + vm_gain * float(v_rel[2]), 0.2, 1.0))
            if meta.get("phase") == "soft_capture":
                a[0] = float(np.clip(0.45 * a[0], -0.22, 0.22))
                a[1] = float(np.clip(0.45 * a[1], -0.22, 0.22))
        else:
            age = i - first_bil
            a[6] = 1.0
            if age < 5:
                a[2] = float(np.clip(0.25 + 0.4 * float(v_rel[2]), 0.1, 0.45))
                a[0] = float(np.clip(0.15 * a[0], -0.15, 0.15))
                a[1] = float(np.clip(0.15 * a[1], -0.15, 0.15))
            else:
                a[2] = -0.75 if age < 25 else -0.5
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.7 * err[0] / 0.03 + 0.4, -1, 1))
                a[1] = float(np.clip(0.7 * err[1] / 0.03, -0.8, 0.8))
                u = np.asarray(cylinder_axis_hand(env.sim), float)
                ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
                u = u / (np.linalg.norm(u) + 1e-12)
                if float(np.dot(u, ut)) < 0:
                    u = -u
                a[3:6] = np.clip(-0.3 * np.cross(u, ut), -0.45, 0.45)
        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
            bil += 1
        elif first_bil is not None:
            bil += 1
        if first_bil is not None:
            age = i - first_bil
            if age <= 12 or age % 10 == 0:
                po = pose_now(env.sim)
                print(
                    f"+{age} bil={po['bilateral']} th={po['e_theta_deg']:.1f} "
                    f"pos={po['e_pos_mm']:.1f} latch={latched(env.sim)}",
                    flush=True,
                )
            if age >= 50 or latched(env.sim):
                break
        if latched(env.sim) and first_bil is None:
            break

    print("first_bil", first_bil, "n", len(acts), flush=True)
    out4 = eval_full_sn(env, ic, acts, max(1, (first_bil or 1) + 1), nominal_s=SEARCH_NOMINAL_S)
    keys = (
        "SUCCESS_NEW", "CATCH_HANDOFF_SUCCESS", "SN_mean_theta", "SN_mean_pos",
        "fail_class", "handoff_reason", "held_s",
    )
    print("out4", {k: out4.get(k) for k in keys}, flush=True)

    if first_bil is None:
        return
    for cut in (first_bil + 3, first_bil + 5, first_bil + 8, first_bil + 12):
        if cut > len(acts):
            continue
        pref = acts[:cut]
        stab = build_stabilization(
            env, ic, pref,
            {
                "n_lift": 50, "lift": -0.85, "n_return": 22, "n_seat": 40, "n_fine": 10,
                "wy": 0.25, "n_wy": 14, "xy_gain": 0.55, "axis_align": True,
                "use_fine_down": True, "return_vz": -0.12, "seat_vz": 0.03,
            },
        )
        print("cut", cut, "stab", None if stab is None else len(stab), flush=True)
        if stab is not None:
            full = pref + stab
            o = eval_full_sn(env, ic, full, max(1, first_bil + 1), nominal_s=float(NOMINAL_S))
            print(
                " SN12", o["SUCCESS_NEW"], "CH", o["CATCH_HANDOFF_SUCCESS"],
                "th", o.get("SN_mean_theta"), "pos", o.get("SN_mean_pos"),
                "fail", o.get("fail_class"),
                flush=True,
            )


if __name__ == "__main__":
    main()
