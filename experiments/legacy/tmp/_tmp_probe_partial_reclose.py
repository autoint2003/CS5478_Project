"""Allow brief partial-open after first bil so cylinder can finish tumbling."""
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
from scripts.run_freefall_recatch_grasp_point_alignment import BASE_GAINS
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now
from scripts.run_horizontal_offset_freefall_recatch import SEARCH_NOMINAL_S, eval_full_sn
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import clear_probe_latch


def run(env, ic, name, soft=0.022, open_steps=2, open_cmd=-0.2, vm=0.8):
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
    align = bil = 0
    acts = []
    first_bil = None
    for i in range(70):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=max(bil, 0),
        )
        info = jaw_target_err_privileged(env.sim, False, 0.0)
        v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
        th = float(gmeasure(env.sim, U_TGT, P_TGT)["e_theta_deg"])

        if first_bil is None:
            if meta.get("phase") in ("scoop", "soft_capture"):
                a[2] = float(np.clip(float(a[2]) + vm * float(v_rel[2]), 0.2, 1.0))
            if meta.get("phase") == "soft_capture":
                a[0] = float(np.clip(0.5 * a[0], -0.25, 0.25))
                a[1] = float(np.clip(0.5 * a[1], -0.25, 0.25))
        else:
            age = i - first_bil
            if age < open_steps and th > 12:
                a[6] = float(open_cmd)  # brief partial open
                a[2] = float(np.clip(0.35 + 0.5 * float(v_rel[2]), 0.15, 0.6))
            elif age < open_steps + 3:
                a[6] = 1.0
                a[2] = 0.30
            else:
                a[6] = 1.0
                a[2] = -0.92
                p = p_com_hand(env.sim)
                err = np.asarray(P_TGT) - p
                a[0] = float(np.clip(0.7 * err[0] / 0.03, -0.8, 0.8))
                a[1] = float(np.clip(0.7 * err[1] / 0.03, -0.8, 0.8))
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
                print(name, "bil@", i, "th", round(th, 1), flush=True)
            bil += 1
        elif first_bil is not None:
            bil += 1  # sticky for controller; actual may be lost
        if first_bil is not None and (i - first_bil) in (0, 1, 2, 3, 5, 10, 20, 35):
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
        "th", out4.get("SN_mean_theta"), "pos", out4.get("SN_mean_pos"),
        "fail", out4.get("fail_class"), "SN12", out12["SUCCESS_NEW"],
        flush=True,
    )


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    for os_, cmd in [(1, 0.0), (2, -0.2), (2, -0.5), (3, -0.3), (2, 0.3)]:
        run(env, ic, f"open{os_}_c{cmd}", open_steps=os_, open_cmd=cmd)
    # also try soft 0.020 with partial then reclose after more tumble — unlikely
    run(env, ic, "s020_open2", soft=0.020, open_steps=2, open_cmd=-0.3, vm=2.0)


if __name__ == "__main__":
    main()
