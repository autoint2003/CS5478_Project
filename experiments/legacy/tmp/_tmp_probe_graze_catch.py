"""Probe controlled early graze → tumble → close for perp freefall."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import cylinder_axis_hand, measure as gmeasure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import corrected_P_action
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    axial_summary,
    contacts_axial,
)
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def theta(sim):
    return float(gmeasure(sim, U_TGT, P_TGT)["e_theta_deg"])


def obj_spin(sim):
    bid = int(sim.ids.object_body)
    return float(np.linalg.norm(sim.data.cvel[bid][:3]))


def run_graze(env, ic, name, graze_steps=2, wait_theta=35.0, close_xy=0.020,
              chase_retract=-0.05, scoop_after_spin=True):
    """Phase machine:
    1) chase XY with mild height (possible light contact)
    2) after any contact or graze_steps of near approach, retract and wait for theta drop
    3) when theta < wait_theta and xy ok, soft close like expert
    """
    g = dict(BASE_GAINS)
    g["preclose"] = -1.0
    g["xy_soft_capture"] = close_xy
    g["n_align_retract"] = 8
    g["xy_early_scoop"] = 0.06  # stay retracting longer... we'll override

    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    acts = []
    phase = "chase"
    first_bil = house = None
    ax_bil = None
    grazed = False
    spin_seen = False

    for i in range(50):
        a, meta = corrected_P_action(env.sim, g, True, align_steps=align, bil_hold_steps=bil)
        th = theta(env.sim)
        err_xy = float(meta.get("err_xy_now", 99))
        spin = obj_spin(env.sim)
        cs0 = contact_summary(env.sim)

        if phase == "chase":
            # strong XY chase, mild down to invite light brush (expert-like)
            a[2] = float(chase_retract) if i < graze_steps else 0.35
            a[6] = -1.0
            if cs0["nL"] or cs0["nR"] or any(
                c["side"] == "hand" for c in contacts_axial(env.sim)[0]
            ):
                grazed = True
                phase = "wait_spin"
            elif i >= graze_steps + 3 and err_xy < 0.04:
                # force a brief scoop to graze
                a[2] = 0.7
            if spin > 5.0:
                spin_seen = True
                phase = "wait_spin"

        elif phase == "wait_spin":
            a[6] = -1.0
            a[2] = float(g.get("retract", -0.2))
            # keep chasing XY gently
            a[0] = float(np.clip(0.5 * a[0], -0.6, 0.6))
            a[1] = float(np.clip(0.5 * a[1], -0.6, 0.6))
            if spin > 3.0:
                spin_seen = True
            if th <= wait_theta and err_xy < 0.045:
                phase = "close"
            elif i > 25:
                phase = "close"  # timeout: attempt anyway

        elif phase == "close":
            if err_xy > close_xy and th > wait_theta + 15:
                a[6] = -1.0
                a[2] = 0.35 if scoop_after_spin else float(g.get("retract", -0.1))
            else:
                a[6] = 1.0
                a[2] = float(g.get("soft_descend", 0.35))
                a[0] = float(np.clip(0.3 * a[0], -0.4, 0.4))
                a[1] = float(np.clip(0.3 * a[1], -0.4, 0.4))

        if meta.get("phase") == "align_xy":
            align += 1
        # after bil, let corrected_P seat/lift run: re-call without override
        if bil > 0 or cs0["bilateral"]:
            a, meta = corrected_P_action(
                env.sim, g, True, align_steps=align, bil_hold_steps=bil,
            )

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
                ax_bil = dict(ax)
                ax_bil["theta"] = theta(env.sim)
            bil += 1
        elif first_bil is not None:
            bil += 1
        if latched(env.sim):
            break
        if first_bil is not None and i >= first_bil + 8:
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
            aa[2] = 0.15
            step_legal(env.sim, aa)
            soft.append(aa.copy())
        pref2 = pref + soft
        p = {
            "n_lift": 40, "lift": -0.95, "n_return": 16, "n_seat": 32, "n_fine": 0,
            "wy": 0.30, "n_wy": 16, "xy_gain": 0.55, "axis_align": True,
            "use_fine_down": False, "return_vz": -0.18, "seat_vz": 0.05,
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

    print(
        f"{name}: bil@{first_bil} th_bil={None if not ax_bil else round(ax_bil.get('theta', -1), 1)} "
        f"s={None if not ax_bil else round(ax_bil['mean_s_axial'], 4)} "
        f"ctr={None if not ax_bil else ax_bil['n_near_center']} "
        f"house={house} graze={grazed} spin={spin_seen} lift={lift} CH={CH} SN={SN} "
        f"th={None if thf is None else round(thf, 1)} pos={None if posf is None else round(posf, 1)}",
        flush=True,
    )
    return SN, full, first_bil


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")

    for gs in (1, 2, 3, 4):
        run_graze(env, ic, f"graze{gs}", graze_steps=gs, wait_theta=35)
    for wt in (50, 40, 30, 25, 20):
        run_graze(env, ic, f"wait{wt}", graze_steps=2, wait_theta=wt)
    run_graze(env, ic, "graze2_xy016", graze_steps=2, wait_theta=30, close_xy=0.016)
    run_graze(env, ic, "graze3_ret0", graze_steps=3, wait_theta=30, chase_retract=0.0)
    run_graze(env, ic, "graze2_down", graze_steps=2, wait_theta=30, chase_retract=0.25)


if __name__ == "__main__":
    main()
