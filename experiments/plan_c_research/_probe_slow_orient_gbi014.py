import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import data_generation.oracles.plan_c.dynamic_catch as dc
import data_generation.oracles.plan_c.interception as er

sys.modules["data_generation.strategies.ground_dynamic_catch"] = dc
sys.modules["data_generation.strategies.ground_early_recovery"] = er

from controllers.recovery.action_map import TAU_SECURE
from data_generation.oracles.plan_b.release_recatch import _action_from
from data_generation.oracles.plan_c.future_trajectory import object_axis, pad_bilateral, sample_kin
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import restore_ballistic

pack = pickle.loads(
    (ROOT / "results/diagnostics/ground_plan_c_final_success/raw/gbi_014__elevated_snap.pkl").read_bytes()
)
sim, _ = make_ground_sim()
restore_ballistic(sim, pack["elev_snap"])
restore_drop(sim, None)
arm(sim)
print("start", measure(sim, U_TGT, P_TGT)["e_theta_deg"], pad_bilateral(sim)["bilateral"])
for i in range(200):
    Rh = sample_kin(sim)["R_hand"]
    u = object_axis(sim)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    uh = Rh.T @ u
    uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
    w = np.clip(np.cross(uh, np.asarray(U_TGT, float)) * 0.15, -0.2, 0.2)
    a = _action_from(sim, np.zeros(3), w, TAU_SECURE)
    a[6] = 1.0
    step_legal(sim, a)
    if i % 25 == 0:
        m = measure(sim, U_TGT, P_TGT)
        pb = pad_bilateral(sim)
        print(i, round(m["e_theta_deg"], 2), round(m["e_pos_mm"], 1), int(pb["bilateral"]), int(latched(sim)))
    if latched(sim) or not pad_bilateral(sim)["bilateral"]:
        print("lost at", i)
        break
else:
    m = measure(sim, U_TGT, P_TGT)
    print("end", m["e_theta_deg"], m["e_pos_mm"], pad_bilateral(sim)["bilateral"])
