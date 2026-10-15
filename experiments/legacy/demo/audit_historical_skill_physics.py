"""Open-loop replay of the three historical schedules on a fresh zm6 impact.

This is the demonstration generator, not the balanced Transformer. It does not
load a checkpoint and does not write the archived evaluation report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import grip_tau, v1_normalized_to_v3
from envs.physical_recovery import physical_pack
from envs.ballistic_impact import assert_noslip
from training.natural_loss_recatch_transformer_v3 import step_v3
from experiments.legacy.phase_decoupled_recovery_pilot import world
from experiments.legacy.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import T_TASK, continue_long, physical_of, restore_dyn, run_impact
from training.replay_core import park_impact_ball
from training.full_3d_airborne_recatch_dataset import TRACK

OUT = ROOT / "results" / "diagnostics" / "raw" / "historical_unified_reconciliation" / "schedule_physics.json"


def hand_angles(sim):
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    up = np.array([0.0, 0.0, 1.0])
    return {
        "hand_x_to_up_deg": round(float(np.degrees(np.arccos(np.clip(rh[:, 0] @ up, -1, 1)))), 2),
        "hand_z_to_up_deg": round(float(np.degrees(np.arccos(np.clip(rh[:, 2] @ up, -1, 1)))), 2),
    }


def snap_row(sim, act):
    o = physical_pack(sim)
    row = {
        "t": round(float(sim.data.time), 3),
        "act": [round(float(x), 4) for x in act],
        "tau": round(float(o["tau"]), 3),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "rh_mm": [round(1e3 * float(x), 3) for x in o["rh"]],
        "g_h": [round(float(x), 3) for x in o["g_h"]],
        "z": round(float(o["obj_z"]), 4),
        "w_rel": round(float(o["w_rel"]), 3),
        "aperture_mm": round(1e3 * float(o["aperture"]), 2),
    }
    row.update(hand_angles(sim))
    return row


def replay(sim, snap, nominal, pads, mu, name, actions):
    info = restore_dyn(sim, snap, nominal, pads, mu, None, None)
    park_impact_ball(sim)
    if not info["pose_ok"]:
        return {"name": name, "restore": False}
    rows = []
    uni = both = none = 0
    for raw in actions:
        act = v1_normalized_to_v3(np.asarray(raw, np.float32))
        rows.append(snap_row(sim, act))
        nL, nR = rows[-1]["nL"], rows[-1]["nR"]
        if nL > 0 and nR > 0:
            both += 1
        elif nL > 0 or nR > 0:
            uni += 1
        else:
            none += 1
        step_v3(sim, act)
        if float(physical_pack(sim)["obj_z"]) < 0.30:
            break
    end_cmd = snap_row(sim, np.zeros(7))
    rec = continue_long(sim, TRACK, T_TASK, name)
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    rh0 = np.array(rows[0]["rh_mm"])
    if name == "gravity_inward" and len(rows) > 43:
        slip = [round(float(rows[43]["rh_mm"][i] - rows[19]["rh_mm"][i]), 3) for i in range(3)]
    else:
        slip = None
    return {
        "name": name,
        "n_cmd": len(actions),
        "logged": len(rows),
        "steps_bilateral": both,
        "steps_unilateral": uni,
        "steps_no_contact": none,
        "tau_open_nm": round(float(grip_tau(-1)), 3),
        "tau_secure_nm": round(float(grip_tau(1)), 3),
        "entry": rows[0],
        "mid": rows[len(rows) // 2],
        "before_continue": end_cmd,
        "unload_drh_mm": slip,
        "t12": "RETAINED_TO_12" if y == 1 else kind,
        "end_n": [int(rec["end"]["nL"]), int(rec["end"]["nR"])],
        "end_z": round(float(rec["end"]["obj_z"]), 4),
        "samples": [rows[i] for i in (0, 11, 19, 20, 35, 43, min(51, len(rows) - 1)) if i < len(rows)],
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sim, gains, nominal, pads, mu, pads_l, pads_r = world()
    assert_noslip(sim)
    print("IMPACT", flush=True)
    impact = run_impact(sim, gains, 6.5, -0.006, 0.0, pads_l, pads_r, nominal, None, None)
    snap = impact["snap"]
    sched = schedules()
    out = []
    for name in ("wrist_reseat", "gravity_inward", "gravity_outward", "recatch_open20"):
        print("REPLAY", name, flush=True)
        row = replay(sim, snap, nominal, pads, mu, name, sched[name])
        out.append(row)
        print(name, row["t12"], "uni", row["steps_unilateral"], "none", row["steps_no_contact"], "slip", row["unload_drh_mm"], flush=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
