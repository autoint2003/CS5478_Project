import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.success_criteria import contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_collision_aware_feedback import (
    JAW_XY_ENTER,
    corrected_P_action,
    list_obj_contacts,
)
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_lateral_offset_falling_recatch import (
    contact_sites,
    finger_frame,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch
from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal


def run(gains, label):
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, _, _ = load_root("perp_off01_r15")
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    print("===", label, "===")
    for i in range(40):
        a, meta = corrected_P_action(env.sim, gains, use_predict=True)
        step_legal(env.sim, a)
        ff = finger_frame(env.sim)
        err = ff["Rh"].T @ (obj_pos(env.sim) - ff["mid"])
        cs = contact_summary(env.sim)
        bodies = sorted({c["body"] for c in list_obj_contacts(env.sim)})
        sites = [s["site"] for s in contact_sites(env.sim)]
        if i < 20 or bodies or cs["bilateral"]:
            print(
                f" {i} err_xy={np.linalg.norm(err[:2]):.3f} ez={err[2]:+.3f} "
                f"a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[6]:+.1f}) "
                f"desc={meta.get('descend')} house={meta.get('housing_risk')} "
                f"bil={cs['bilateral']} bodies={bodies} sites={sites[:4]}"
            )
        if latched(env.sim) or cs["bilateral"]:
            print(" STOP", "latch" if latched(env.sim) else "bil", "at", i)
            break


def main():
    base = {
        "kp_chase": 1.8,
        "retract": -0.35,
        "hover": -0.05,
        "descend": 1.0,
        "kw_app": 0.25,
        "wx0": 0.0,
        "wy0": 0.1,
        "wz0": 0.15,
        "t_lead_max": 0.10,
        "t_lead_bias": 0.03,
        "v_xy_nom": 0.7,
        "preclose": 0.0,
        "lift": -0.55,
        "kp_hold": 1.0,
    }
    run(base, "base")
    # tighter enter + stronger retract then scoop like expert
    g2 = dict(base)
    g2.update({"retract": -0.4, "hover": -0.15, "descend": 1.0, "kp_chase": 2.0,
               "t_lead_bias": 0.06, "preclose": -1.0})
    # monkeypatch enter thresholds via gains? they're module globals — probe local override
    import scripts.run_freefall_recatch_collision_aware_feedback as m
    m.JAW_XY_ENTER = 0.035
    m.JAW_XY_SOFT = 0.055
    run(g2, "wider_enter_strong_retract")
    m.JAW_XY_ENTER = 0.022
    m.JAW_XY_SOFT = 0.04
    g3 = dict(base)
    g3.update({"retract": -0.2, "hover": 0.2, "descend": 1.0, "kp_chase": 1.6})
    run(g3, "earlier_descend")


if __name__ == "__main__":
    main()
