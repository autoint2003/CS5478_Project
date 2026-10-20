import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, mujoco, torch
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from training.recovery_episode import make_training_env
from evaluation.success_criteria import contact_summary
from scripts.expand_k49_large_chase_dynamic import hand_pos, obj_pos

device = torch.device("cpu")
model, book = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)
env.begin_nominal()
sim = env.sim
sim.data.qpos[sim.ids.finger_jnt] = 0.04
mujoco.mj_forward(sim.model, sim.data)
pl = np.array(sim.data.xpos[sim.ids.left_body])
pr = np.array(sim.data.xpos[sim.ids.right_body])
mid = 0.5 * (pl + pr)
print("L", pl, "R", pr, "mid", mid, "gap", np.linalg.norm(pl - pr))
print("hand", hand_pos(sim))
q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
s = np.sqrt(0.5)
tg = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
for dz in [0.05, 0.08, 0.12, 0.18, 0.25]:
    sim.data.qpos[q : q + 3] = [mid[0], mid[1], mid[2] + dz]
    sim.data.qpos[q + 3 : q + 7] = [s, s, 0, 0]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    cs = contact_summary(sim)
    ntab = 0
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        bodies = (int(sim.model.geom_bodyid[int(c.geom1)]), int(sim.model.geom_bodyid[int(c.geom2)]))
        if sim.ids.object_body not in bodies:
            continue
        og = int(c.geom2) if bodies[0] == sim.ids.object_body else int(c.geom1)
        if og == tg:
            ntab += 1
    print(f"dz={dz:.2f} z={obj_pos(sim)[2]:.3f} nL={cs['nL']} nR={cs['nR']} ntab={ntab}")

sim.data.qpos[q : q + 3] = [mid[0], mid[1], mid[2] + 0.12]
sim.data.qpos[q + 3 : q + 7] = [s, s, 0, 0]
sim.data.qvel[:] = 0
mujoco.mj_forward(sim.model, sim.data)
t0 = float(sim.data.time)
for i in range(300):
    mujoco.mj_step(sim.model, sim.data)
    if float(obj_pos(sim)[2]) < 0.42:
        print("hit low z at dt", float(sim.data.time) - t0, "z", obj_pos(sim)[2], "steps", i)
        break
else:
    print("no low z", obj_pos(sim)[2])
