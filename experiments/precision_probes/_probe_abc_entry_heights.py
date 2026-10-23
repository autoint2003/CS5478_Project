#!/usr/bin/env python
"""Compare Stage-2 IC world-frame heights for Plans A/B/C."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.state_machine import measure_errors
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY


def main():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    def heights(cap):
        restore_capture(env, cap)
        err = measure_errors(env.sim)
        return {
            "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
            "hand_z": float(env.sim.data.xpos[env.sim.ids.hand_body][2]),
            "e_pos": float(err["e_pos_mm"]),
            "dtheta": float(err["dtheta_deg"]),
            "bil": bool(err["bilateral"]),
            "fnL": float(err["fn_L"]),
            "fnR": float(err["fn_R"]),
            "signed": [round(float(x), 2) for x in err["e_signed_mm"]],
        }

    samples = []
    plans = [
        (ROOT / "results/data_generation/stage2_abc_mining/v1", "PILOT_A__*.json", "A", 5),
        (ROOT / "results/data_generation/stage2_abc_mining/v2", "B__*.json", "B", 8),
        (ROOT / "results/data_generation/stage2_abc_mining/v2", "C__*.json", "C", 10),
    ]
    for base, globp, plan, lim in plans:
        n = 0
        for p in sorted((base / "outcomes").glob(globp)):
            o = json.loads(p.read_text(encoding="utf-8"))
            cat = o.get("sft_category")
            if plan in ("A", "B") and cat not in ("NONTRIVIAL_PRECISION", "DIRECT_HANDOFF"):
                continue
            ic = base / "ics" / f"{o['sample_id']}.pkl"
            if not ic.exists():
                continue
            pack = pickle.loads(ic.read_bytes())
            h = heights(pack["capture"])
            samples.append((plan, o["sample_id"], cat, h))
            n += 1
            if n >= lim:
                break

    for plan in "ABC":
        hs = [s for s in samples if s[0] == plan]
        print("===", plan, "n", len(hs))
        for s in hs:
            h = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s[3].items()}
            print(s[1][:48], s[2], h)
        if hs:
            oz = [s[3]["obj_z"] for s in hs]
            hz = [s[3]["hand_z"] for s in hs]
            print(" obj_z", round(min(oz), 3), "-", round(max(oz), 3), "mean", round(sum(oz) / len(oz), 3))
            print(" hand_z", round(min(hz), 3), "-", round(max(hz), 3), "mean", round(sum(hz) / len(hz), 3))


if __name__ == "__main__":
    main()
