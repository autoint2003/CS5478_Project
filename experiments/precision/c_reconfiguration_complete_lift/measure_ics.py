#!/usr/bin/env python
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from data_generation.oracles.precision.residuals import measure_errors
from models.transformer_policy import DT_POLICY
import numpy as np

EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
OUT = ROOT / "results/diagnostics/c_reconfiguration_complete_lift"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, _ = make_ground_sim()

    class E:
        pass

    env = E()
    env.sim = sim
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    rows = []
    for hid in [
        "hard_910263",
        "hard_910277",
        "hard_910132",
        "hard_910024",
        "hard_910084",
        "hard_910305",
        "hard_910026",
    ]:
        blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
        restore_capture(env, blob["capture"])
        e = measure_errors(env.sim)
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        zh = float(env.sim.data.xpos[env.sim.ids.hand_body][2])
        gz = getattr(env.sim.fsm, "grasp_z", None)
        rows.append(
            {
                "hard_id": hid,
                "dx_mm": float(e["dx_mm"]),
                "dz_mm": float(e["dz_mm"]),
                "dtheta_deg": float(e["dtheta_deg"]),
                "e_pos_mm": float(e["e_pos_mm"]),
                "obj_z": z,
                "hand_z": zh,
                "grasp_z": float(gz) if gz is not None else None,
                "bilateral": bool(e.get("bilateral")),
                "fn_tot": float(e.get("fn_tot") or 0),
                "alpha_up_deg": float(e.get("alpha_up_deg") or 0),
                "abs_dx": abs(float(e["dx_mm"])),
                "abs_dz": abs(float(e["dz_mm"])),
                "both_large": abs(float(e["dx_mm"])) > 6 and abs(float(e["dz_mm"])) > 6,
            }
        )
        print(rows[-1])
    (OUT / "IC_MEASUREMENTS.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
