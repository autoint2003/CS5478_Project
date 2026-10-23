"""Nominal continuation scored from the current MuJoCo state.

Moved out of the Plan C expert-search script so active oracles do not import
historical entrypoints. The scoring rules are unchanged.
"""

from __future__ import annotations

import numpy as np

from data_generation.oracles.plan_c.future_trajectory import hold_action, pad_bilateral, sample_kin
from envs.outcomes.ground_drop import latched
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import (
    FINAL_WINDOW_S,
    NOMINAL_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)


def evaluate_handoff_from_state(sim, ref, *, stabilize_n: int = 0) -> dict:
    for _ in range(stabilize_n):
        step_legal(sim, hold_action(1.0))
        if latched(sim) or not pad_bilateral(sim)["bilateral"]:
            break
    m0 = measure(sim, U_TGT, P_TGT)
    pb0 = pad_bilateral(sim)
    # Trace around handoff
    trace = []
    apply_handoff(sim, ref)
    t_h = float(sim.data.time)
    reason = None
    samples = []
    next_sample = 0.0
    lost = 0
    lift_done = False
    from envs.ballistic_impact import z_tgt_of
    from envs.outcomes.horizon_outcome import (
        LOST_STEPS,
        _bilateral,
        _dropped,
        _escaped,
        _monitor,
    )

    # First few steps detailed
    for i in range(25):
        if reason is not None:
            break
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        k = sample_kin(sim)
        m = measure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        pb = pad_bilateral(sim)
        row = {
            "t_rel": t_rel,
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "bilateral": int(pb["bilateral"]),
            "fnL": pb["fnL"],
            "fnR": pb["fnR"],
            "obj_z": m["obj_z"],
            "v_hand": float(np.linalg.norm(k["v_hand"])),
            "v_rel": float(np.linalg.norm(k["v_obj"] - k["v_hand"])),
            "p_des_z": float(sim.fsm.p_des[2]),
            "v_cmd_norm": float(np.linalg.norm(sim.fsm.v_cmd)),
        }
        if i < 15 or (not pb["bilateral"]):
            trace.append(row)
        if t_rel + 1e-12 >= next_sample:
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += 0.02
        mon = _monitor(sim)
        if pb["bilateral"] and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            break
        # Pass sim so GROUND_SCENE uses geometric ground drop, not TABLE_DROP=0.44.
        if _dropped(mon, sim) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not pb["bilateral"] else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break

    # Continue remaining nominal if still ok
    while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(sim)
        t_rel = float(sim.data.time) - t_h
        if t_rel + 1e-12 >= next_sample:
            m = measure(sim, U_TGT, P_TGT)
            cs = contact_summary(sim)
            samples.append({
                "t_rel": t_rel,
                "e_theta_deg": m["e_theta_deg"],
                "e_pos_mm": m["e_pos_mm"],
                "bilateral": cs["bilateral"],
                "obj_z": m["obj_z"],
            })
            next_sample += 0.02
        mon = _monitor(sim)
        pb = pad_bilateral(sim)
        if pb["bilateral"] and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            break
        if _dropped(mon, sim) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not pb["bilateral"] else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        mon = _monitor(sim)
        if not _bilateral(mon) or _dropped(mon, sim) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    win = window_stats(samples, float(NOMINAL_S) - FINAL_WINDOW_S)
    success_new = bool(reason == "SUCCESS_12S" and win.get("working_SUCCESS_NEW_hold"))
    return {
        "pre_theta": m0["e_theta_deg"],
        "pre_pos_mm": m0["e_pos_mm"],
        "pre_bilateral": int(pb0["bilateral"]),
        "pre_in_band": int(
            m0["e_theta_deg"] <= WORKING_THETA_DEG and m0["e_pos_mm"] <= WORKING_POS_MM and pb0["bilateral"]
        ),
        "reason": reason,
        "SUCCESS_NEW": success_new,
        "success_12s": reason == "SUCCESS_12S",
        "window": win,
        "drop_t_rel": next((t["t_rel"] for t in trace if not t["bilateral"]), None),
        "trace_head": trace[:12],
    }
