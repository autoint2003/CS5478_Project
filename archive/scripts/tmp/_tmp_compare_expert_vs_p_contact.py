import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.full_objective import contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_pose_feedback import load_root
from scripts.run_lateral_offset_falling_recatch import (
    contact_sites,
    finger_frame,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch
from envs.step_legal import step_legal


def contact_detail(sim):
    obj = int(sim.ids.object_body)
    rows = []
    for j in range(int(sim.data.ncon)):
        c = sim.data.contact[j]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        og = g2 if b1 == obj else g1
        ob = b2 if b1 == obj else b1
        bn = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, ob)
        rows.append((og, bn, float(c.dist)))
    return rows


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, acts, _ = load_root("perp_off01_r15")

    print("EXPERT")
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    for i, a in enumerate(acts[:25]):
        step_legal(env.sim, np.asarray(a, np.float32))
        cd = contact_detail(env.sim)
        cs = contact_summary(env.sim)
        sites = contact_sites(env.sim)
        bodies = sorted(set(b for _, b, _ in cd))
        print(
            f" {i} a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[6]:+.1f}) "
            f"bil={cs['bilateral']} bodies={bodies} "
            f"sites={[s['site'] for s in sites]}"
        )

    print("P COM chase")
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    for i in range(25):
        ff = finger_frame(env.sim)
        po = obj_pos(env.sim)
        mid = ff["mid"]
        err = ff["Rh"].T @ (po - mid)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(1.5 * err[0] / 0.02, -1, 1))
        a[1] = float(np.clip(1.5 * err[1] / 0.02, -1, 1))
        a[2] = float(np.clip(0.85 + 0.5 * err[2] / 0.05, 0, 1))
        a[6] = -1.0
        step_legal(env.sim, a)
        cd = contact_detail(env.sim)
        cs = contact_summary(env.sim)
        sites = contact_sites(env.sim)
        bodies = sorted(set(b for _, b, _ in cd))
        print(
            f" {i} err={np.round(err, 3)} a=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) "
            f"bil={cs['bilateral']} bodies={bodies} "
            f"sites={[s['site'] for s in sites]} nL={cs['nL']} nR={cs['nR']}"
        )


if __name__ == "__main__":
    main()
