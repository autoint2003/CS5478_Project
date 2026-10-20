"""Close only when theta is low enough, while velocity-matching the fall."""
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
from evaluation.grasp_configuration import measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    corrected_P_action,
    hand_vel,
    jaw_target_err_privileged,
    obj_vel,
)
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    aperture,
    axial_summary,
    contacts_axial,
)
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def run(env, ic, name, theta_close=15.0, xy_close=0.025, soft_cap=0.030):
    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=soft_cap,  # enter soft phase earlier for vel match
            preclose=-1.0,
            n_bil_seat=6, lift=-0.90,
        )
    )
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    acts = []
    first_bil = None
    ax_bil = None
    for i in range(60):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=bil,
        )
        th = float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"])
        err_xy = float(meta.get("err_xy_now", 99))
        info = jaw_target_err_privileged(env.sim, False, 0.0)
        v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))

        if first_bil is None and not contact_summary(env.sim)["bilateral"]:
            # always vel-match when near
            if err_xy < 0.05:
                a[2] = float(np.clip(0.55 + 0.9 * float(v_rel[2]), 0.25, 1.0))
            allow = err_xy <= xy_close and th <= theta_close
            if allow:
                a[6] = 1.0
                a[0] = float(np.clip(0.4 * a[0], -0.25, 0.25))
                a[1] = float(np.clip(0.4 * a[1], -0.25, 0.25))
                a[2] = float(np.clip(0.35 + 0.5 * float(v_rel[2]), 0.2, 0.7))
            else:
                a[6] = -1.0
                # keep tracking fall
                if err_xy < 0.04:
                    a[2] = float(np.clip(0.7 + 0.8 * float(v_rel[2]), 0.3, 1.0))
        else:
            # after bil: lift like shared but ensure grip
            a[6] = 1.0
            age = i - (first_bil or i)
            if age < 2:
                a[2] = float(np.clip(0.30 + 0.4 * float(v_rel[2]), 0.15, 0.5))
            else:
                a[2] = -0.92
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.7 * err[0] / 0.03, -0.8, 0.8))
                a[1] = float(np.clip(0.7 * err[1] / 0.03, -0.8, 0.8))

        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                ax_bil = axial_summary(contacts_axial(env.sim)[0])
                ax_bil["theta"] = th
                ax_bil["vrelz"] = float(v_rel[2])
                print(
                    f"{name} bil@{i} th={th:.1f} xy={err_xy:.3f} vrelz={v_rel[2]:.4f} "
                    f"s={ax_bil['mean_s_axial']:.4f} ctr={ax_bil['n_near_center']} "
                    f"aper={aperture(env.sim):.4f}",
                    flush=True,
                )
            bil += 1
        elif first_bil is not None:
            bil += 1
        if latched(env.sim):
            print(f"{name} LATCH@{i} after_bil={first_bil}", flush=True)
            break
        if first_bil is not None and i >= first_bil + 40:
            break

    if first_bil is None:
        print(name, "NO BIL", flush=True)
        return

    # closed-loop eval
    out4 = eval_full_sn(env, ic, acts, max(1, first_bil + 1), nominal_s=SEARCH_NOMINAL_S)
    out12 = eval_full_sn(env, ic, acts, max(1, first_bil + 1), nominal_s=float(NOMINAL_S))
    print(
        name, "CL SN4", out4["SUCCESS_NEW"], "CH", out4["CATCH_HANDOFF_SUCCESS"],
        "th", out4.get("SN_mean_theta"), "pos", out4.get("SN_mean_pos"),
        "fail", out4.get("fail_class"), "SN12", out12["SUCCESS_NEW"],
        flush=True,
    )

    # stab from bil+2
    pref = acts[: first_bil + 3]
    stab = build_stabilization(
        env, ic, pref,
        {
            "n_lift": 48, "lift": -0.95, "n_return": 20, "n_seat": 36, "n_fine": 8,
            "wy": 0.28, "n_wy": 16, "xy_gain": 0.55, "axis_align": True,
            "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.04,
        },
    )
    if stab is not None:
        full = pref + stab
        out12b = eval_full_sn(env, ic, full, max(1, first_bil + 1), nominal_s=float(NOMINAL_S))
        out4b = eval_full_sn(env, ic, full, max(1, first_bil + 1), nominal_s=SEARCH_NOMINAL_S)
        print(
            name, "STAB SN4", out4b["SUCCESS_NEW"], "CH", out4b["CATCH_HANDOFF_SUCCESS"],
            "th", out4b.get("SN_mean_theta"), "pos", out4b.get("SN_mean_pos"),
            "SN12", out12b["SUCCESS_NEW"],
            flush=True,
        )
    else:
        print(name, "STAB failed", flush=True)


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    for th in (25, 20, 15, 12, 10, 8):
        run(env, ic, f"th{th}", theta_close=th, xy_close=0.024)
    run(env, ic, "th15_xy020", theta_close=15, xy_close=0.020)
    run(env, ic, "th12_xy022", theta_close=12, xy_close=0.022)
    run(env, ic, "th18_xy025", theta_close=18, xy_close=0.025)


if __name__ == "__main__":
    main()
