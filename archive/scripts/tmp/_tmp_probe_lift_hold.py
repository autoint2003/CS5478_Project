"""Hold good-orientation catch (soft<=0.021) through expert-like lift."""
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
)
from scripts.run_freefall_recatch_grasp_point_alignment import BASE_GAINS, aperture
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def catch_to_bil(env, ic, soft=0.020):
    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=soft, preclose=-1.0,
        )
    )
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = 0
    acts = []
    for i in range(40):
        a, meta = corrected_P_action(env.sim, g, True, align_steps=align, bil_hold_steps=0)
        if meta.get("phase") in ("scoop", "soft_capture"):
            info = jaw_target_err_privileged(env.sim, False, 0.0)
            v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
            a[2] = float(np.clip(float(a[2]) + 0.8 * float(v_rel[2]), 0.2, 1.0))
        if meta.get("phase") == "soft_capture":
            a[0] = float(np.clip(0.5 * a[0], -0.25, 0.25))
            a[1] = float(np.clip(0.5 * a[1], -0.25, 0.25))
        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        acts.append(np.clip(a, -1, 1).astype(np.float32).copy())
        if contact_summary(env.sim)["bilateral"]:
            return acts, i, pose_now(env.sim)
        if latched(env.sim):
            break
    return acts, None, None


def lift_variants(env, ic):
    acts0, fb, po = catch_to_bil(env, ic, 0.020)
    print("catch", fb, po)
    if fb is None:
        return

    variants = []
    # 1) expert-like: one soft then strong +x lift
    variants.append(("exp_like", dict(n_soft=1, soft_z=0.35, lift=-0.85, a0=0.70, a1_mode="hold", n_lift=45)))
    variants.append(("exp_a0_0.5", dict(n_soft=1, soft_z=0.35, lift=-0.85, a0=0.50, a1_mode="hold", n_lift=45)))
    variants.append(("exp_a0_1", dict(n_soft=1, soft_z=0.35, lift=-0.90, a0=1.0, a1_mode="ptgt", n_lift=45)))
    variants.append(("no_soft", dict(n_soft=0, soft_z=0.35, lift=-0.85, a0=0.70, a1_mode="ptgt", n_lift=45)))
    variants.append(("soft2", dict(n_soft=2, soft_z=0.25, lift=-0.85, a0=0.70, a1_mode="ptgt", n_lift=45)))
    variants.append(("slow_lift", dict(n_soft=1, soft_z=0.2, lift=-0.55, a0=0.6, a1_mode="ptgt", n_lift=60)))
    variants.append(("squeeze", dict(n_soft=3, soft_z=0.1, lift=-0.85, a0=0.5, a1_mode="ptgt", n_lift=45, squeeze=True)))

    for name, cfg in variants:
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        full = []
        for a in acts0:
            step_legal(env.sim, a)
            full.append(a.copy())
        lost = False
        for s in range(int(cfg["n_soft"])):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            a[2] = float(cfg["soft_z"])
            if cfg.get("squeeze"):
                a[2] = 0.05
            step_legal(env.sim, a)
            full.append(a.copy())
            if not contact_summary(env.sim)["bilateral"] or latched(env.sim):
                print(name, "lost soft", s, pose_now(env.sim))
                lost = True
                break
        if lost:
            continue
        for k in range(int(cfg["n_lift"])):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            a[2] = float(cfg["lift"])
            a[0] = float(cfg["a0"])
            if cfg["a1_mode"] == "ptgt":
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.8 * err[0] / 0.03, -1, 1))
                a[1] = float(np.clip(0.8 * err[1] / 0.03, -1, 1))
                # blend with bias
                a[0] = float(np.clip(0.5 * a[0] + 0.5 * cfg["a0"], -1, 1))
            elif cfg["a1_mode"] == "hold":
                a[1] = 0.0
            # light wrist toward U_TGT
            u = np.asarray(cylinder_axis_hand(env.sim), float)
            ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
            u = u / (np.linalg.norm(u) + 1e-12)
            if float(np.dot(u, ut)) < 0:
                u = -u
            rot = np.cross(u, ut)
            a[3:6] = np.clip(-0.25 * rot, -0.4, 0.4)
            step_legal(env.sim, a)
            full.append(a.copy())
            if k < 4 or k % 15 == 0:
                po = pose_now(env.sim)
                print(
                    name, "lift", k,
                    f"bil={po['bilateral']} th={po['e_theta_deg']:.1f} pos={po['e_pos_mm']:.1f} "
                    f"aper={aperture(env.sim):.4f} latch={latched(env.sim)}",
                    flush=True,
                )
            if latched(env.sim):
                lost = True
                break
        if lost:
            continue
        out4 = eval_full_sn(env, ic, full, max(1, fb + 1), nominal_s=SEARCH_NOMINAL_S)
        out12 = eval_full_sn(env, ic, full, max(1, fb + 1), nominal_s=float(NOMINAL_S))
        print(
            name, "SN4", out4["SUCCESS_NEW"], "CH", out4["CATCH_HANDOFF_SUCCESS"],
            "th", out4.get("SN_mean_theta"), "pos", out4.get("SN_mean_pos"),
            "fail", out4.get("fail_class"), "SN12", out12["SUCCESS_NEW"],
            flush=True,
        )
        # stab alternative
        pref = full[: fb + 1 + int(cfg["n_soft"]) + 5]
        stab = build_stabilization(
            env, ic, pref,
            {
                "n_lift": 48, "lift": -0.95, "n_return": 20, "n_seat": 36, "n_fine": 8,
                "wy": 0.25, "n_wy": 14, "xy_gain": 0.55, "axis_align": True,
                "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.04,
            },
        )
        if stab is not None:
            full2 = pref + stab
            out12b = eval_full_sn(env, ic, full2, max(1, fb + 1), nominal_s=float(NOMINAL_S))
            print(name, "STAB_SN12", out12b["SUCCESS_NEW"], out12b.get("SN_mean_theta"), out12b.get("SN_mean_pos"), flush=True)


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    lift_variants(env, ic)


if __name__ == "__main__":
    main()
