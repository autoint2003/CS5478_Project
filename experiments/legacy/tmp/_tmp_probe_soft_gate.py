"""Compare post-bil stability across soft_capture gates."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.success_criteria import contact_summary
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
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import clear_probe_latch


def catch(env, ic, soft, vel_match=True, soft_xy=True, n_after=12):
    g = dict(BASE_GAINS)
    g.update(
        dict(
            kw_app=0.0, wx0=0.0, wy0=0.0, wz0=0.0,
            n_align_retract=6, xy_early_scoop=0.035,
            xy_soft_capture=soft, preclose=-1.0,
            n_bil_seat=8, lift=-0.85,
        )
    )
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil = 0
    first_bil = None
    for i in range(50):
        a, meta = corrected_P_action(
            env.sim, g, True, align_steps=align, bil_hold_steps=bil,
        )
        if vel_match and meta.get("phase") in ("scoop", "soft_capture") and first_bil is None:
            info = jaw_target_err_privileged(env.sim, False, 0.0)
            v_rel = info["Rh"].T @ (obj_vel(env.sim) - hand_vel(env.sim))
            a[2] = float(np.clip(float(a[2]) + 0.8 * float(v_rel[2]), 0.2, 1.0))
        if soft_xy and meta.get("phase") == "soft_capture" and first_bil is None:
            a[0] = float(np.clip(0.5 * a[0], -0.25, 0.25))
            a[1] = float(np.clip(0.5 * a[1], -0.25, 0.25))
        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, np.clip(a, -1, 1).astype(np.float32))
        cs = contact_summary(env.sim)
        po = pose_now(env.sim)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
                vo, vh = obj_vel(env.sim), hand_vel(env.sim)
                ax = axial_summary(contacts_axial(env.sim)[0])
                print(
                    f"=== soft={soft} bil@{i} th={po['e_theta_deg']:.1f} "
                    f"pos={po['e_pos_mm']:.1f} aper={aperture(env.sim):.4f} "
                    f"vrelz={(vo - vh)[2]:.4f} s={ax['mean_s_axial']:.4f} "
                    f"ctr={ax['n_near_center']}",
                    flush=True,
                )
            bil += 1
        elif first_bil is not None:
            bil += 1
        if first_bil is not None:
            age = i - first_bil
            print(
                f"  +{age} bil={int(cs['bilateral'])} th={po['e_theta_deg']:.1f} "
                f"pos={po['e_pos_mm']:.1f} aper={aperture(env.sim):.4f} "
                f"phase={meta.get('phase')} latch={latched(env.sim)}",
                flush=True,
            )
            if age >= n_after or latched(env.sim):
                break
        if latched(env.sim) and first_bil is None:
            print(f"=== soft={soft} LATCH before bil", flush=True)
            break
    return first_bil


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    for s in (0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.025):
        catch(env, ic, s)


if __name__ == "__main__":
    main()
