import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    corrected_O_action,
    corrected_P_action,
    list_obj_contacts,
    rollout_controller,
)
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_lateral_offset_falling_recatch import (
    finger_frame,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch

gains = {
    "kp_chase": 1.7,
    "retract": -0.20,
    "descend": 1.0,
    "kw_app": 0.3,
    "wx0": 0,
    "wy0": 0.12,
    "wz0": 0.18,
    "t_lead_max": 0.1,
    "t_lead_bias": 0.05,
    "v_xy_nom": 0.65,
    "preclose": 1.0,
    "ez_fall_through": 0.03,
    "n_align_retract": 4,
    "xy_early_scoop": 0.048,
    "overchase": 0.25,
    "lift": -0.85,
    "kp_hold": 1.0,
    "n_bil_seat": 3,
    "seat_vz": -0.3,
    "kp": 1.2,
    "kz": 0.5,
    "vz_bias": 0.25,
}


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    for kind in ("corr_P", "corr_O", "orig_P", "orig_O"):
        acts, summary, _, out = rollout_controller(env, ic, kind, gains, log=False)
        print(
            kind,
            "bil", summary["first_bil_step"],
            "jaw", summary["jaw_entry"],
            "house", summary["housing_collision"],
            "house_g", summary["first_housing_geoms"],
            "SN", summary["SUCCESS_NEW"],
            "CH", summary["CATCH_HANDOFF_SUCCESS"],
            "th", summary["SN_mean_theta"],
            "pos", summary["SN_mean_pos"],
        )


if __name__ == "__main__":
    main()
