"""Stronger velocity matching before soft=0.020 close for stable good-theta grasp."""
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
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import cylinder_axis_hand, measure as gmeasure, p_com_hand
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
from scripts.run_lateral_recatch_handoff import clear_probe_latch


def run(env, ic, name, soft=0.020, vm_gain=0.8, vm_bias=0.0, n_extra_soft=0):
    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=soft, preclose=-1.0,
            n_bil_seat=10, lift=-0.85,
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
        info = jaw_target_err_privileged(env.sim, False, 0.0)
        v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
        th = float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"])

        if first_bil is None:
            if meta.get("phase") in ("scoop", "soft_capture"):
                a[2] = float(np.clip(float(a[2]) + vm_gain * float(v_rel[2]) + vm_bias, 0.2, 1.0))
            if meta.get("phase") == "soft_capture":
                a[0] = float(np.clip(0.45 * a[0], -0.22, 0.22))
                a[1] = float(np.clip(0.45 * a[1], -0.22, 0.22))
        else:
            age = i - first_bil
            a[6] = 1.0
            if age < 2 + n_extra_soft:
                # stay soft, keep matching
                a[2] = float(np.clip(0.35 + 0.5 * float(v_rel[2]), 0.15, 0.55))
                a[0] = float(np.clip(0.2 * a[0], -0.2, 0.2))
                a[1] = float(np.clip(0.2 * a[1], -0.2, 0.2))
            else:
                a[2] = -0.90
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.75 * err[0] / 0.03 + 0.35, -1, 1))  # bias +x like expert
                a[1] = float(np.clip(0.75 * err[1] / 0.03, -0.8, 0.8))
                # light wrist
                u = np.asarray(cylinder_axis_hand(env.sim), float)
                ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
                u = u / (np.linalg.norm(u) + 1e-12)
                if float(np.dot(u, ut)) < 0:
                    u = -u
                a[3:6] = np.clip(-0.35 * np.cross(u, ut), -0.5, 0.5)

        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        cs = contact_summary(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                ax = axial_summary(contacts_axial(env.sim)[0])
                print(
                    f"{name} bil@{i} th={th:.1f} vrelz={v_rel[2]:.4f} "
                    f"s={ax['mean_s_axial']:.4f} aper={aperture(env.sim):.4f}",
                    flush=True,
                )
            bil += 1
        elif first_bil is not None:
            bil += 1
        if first_bil is not None and (i - first_bil) in (0, 1, 2, 5, 15, 30):
            po = pose_now(env.sim)
            print(
                f"  +{i-first_bil} bil={po['bilateral']} th={po['e_theta_deg']:.1f} "
                f"pos={po['e_pos_mm']:.1f} latch={latched(env.sim)}",
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
        "th", None if out4.get("SN_mean_theta") is None else round(out4["SN_mean_theta"], 2),
        "pos", None if out4.get("SN_mean_pos") is None else round(out4["SN_mean_pos"], 2),
        "fail", out4.get("fail_class"), "SN12", out12["SUCCESS_NEW"],
        flush=True,
    )


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    for vg in (0.8, 1.2, 1.6, 2.0, 2.5):
        run(env, ic, f"vm{vg}", soft=0.020, vm_gain=vg)
    for bias in (0.1, 0.2, 0.3):
        run(env, ic, f"vm1.2_b{bias}", soft=0.020, vm_gain=1.2, vm_bias=bias)
    run(env, ic, "vm1.6_soft3", soft=0.020, vm_gain=1.6, n_extra_soft=3)
    run(env, ic, "vm2_s019", soft=0.019, vm_gain=2.0)
    run(env, ic, "vm2_s021", soft=0.021, vm_gain=2.0)
    # also re-check 0.022 with stronger lift xy bias
    run(env, ic, "vm0.8_s022", soft=0.022, vm_gain=0.8)


if __name__ == "__main__":
    main()
