"""Confirm the legal hold, then test the unchanged K48 map and nominal handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.horizon_outcome import NOMINAL_S, nominal_verify
from training.legal_recatch_speed_match import run_speed
from training.legal_recatch_transfer import OUT, WIN, _enable_supports, _prepare, _save_rows
from training.privileged_ball_impact_search import representability, step_legal
from training.recatch_braking_optimization import run
from training.recatch_interface_comparison import _render
from training.replay_core import TABLE_TOP
from training.table_drop import latched, restore_state
from training.unified_k47 import load_codebook

SPEC = {"dive": 1.2, "dive_s": 0.04, "lead": 0.0, "tol": 0.12, "settle": 0.04, "accel": 3.0}
REAL_STEP = mujoco.mj_step


def _install_counter(sim):
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obj = int(sim.ids.object_geom)
    counts = {"table": 0, "floor": 0}

    def wrapped(model, data):
        REAL_STEP(model, data)
        for index in range(int(data.ncon)):
            geoms = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
            if obj not in geoms:
                continue
            if table in geoms:
                counts["table"] += 1
            if floor in geoms:
                counts["floor"] += 1

    mujoco.mj_step = wrapped
    return counts


def _story(rows):
    none = [(r["nL"] == 0 and r["nR"] == 0) for r in rows]
    both = [(r["nL"] > 0 and r["nR"] > 0) for r in rows]
    off = next((i for i, flag in enumerate(none) if flag), None)
    back = None
    if off is not None:
        back = next((i for i in range(off, len(rows)) if both[i]), None)
    loss = None
    if back is not None:
        loss = next((i for i in range(back, len(rows)) if none[i]), None)
    i_min = int(np.argmin([r["bottom"] for r in rows]))
    stable = [r for r in rows if r["phase"] == "stable"]
    tau = np.stack([r["tau_j"] for r in rows])
    return {
        "n": len(rows),
        "t_end": round(float(rows[-1]["t"]), 3),
        "off_t": None if off is None else round(float(rows[off]["t"]), 4),
        "recontact_t": None if back is None else round(float(rows[back]["t"]), 4),
        "free_s": None if off is None or back is None else round(float(rows[back]["t"] - rows[off]["t"]), 4),
        "re_vh": None if back is None else [round(float(v), 3) for v in rows[back]["vh"]],
        "re_vo": None if back is None else [round(float(v), 3) for v in rows[back]["vo"]],
        "re_bottom": None if back is None else round(float(rows[back]["bottom"]), 4),
        "later_loss_t": None if loss is None else round(float(rows[loss]["t"]), 4),
        "min_bottom_m": round(float(rows[i_min]["bottom"]), 4),
        "min_xy": [round(float(v), 4) for v in rows[i_min]["xy"]],
        "min_over_table": bool(0.38 <= rows[i_min]["xy"][0] <= 0.62 and -0.12 <= rows[i_min]["xy"][1] <= 0.12),
        "end_zo": round(float(rows[-1]["zo"]), 4),
        "end_xy": [round(float(v), 4) for v in rows[-1]["xy"]],
        "end_voz": round(float(rows[-1]["vo"][2]), 3),
        "end_vhz": round(float(rows[-1]["vh"][2]), 3),
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "stable_n": len(stable),
        "stable_both": None if not stable else round(float(np.mean([r["nL"] > 0 and r["nR"] > 0 for r in stable])), 3),
        "max_gap_mm": round(float(max(r["gap"] for r in rows)) * 1e3, 1),
        "max_p_err_mm": round(float(max(r["p_err"] for r in rows)) * 1e3, 1),
        "max_finger_slip": round(float(np.max(np.abs([r["vrel_h"][2] for r in rows]))), 3),
        "peak_tau": round(float(np.max(np.abs(tau))), 2),
        "phases": sorted({r["phase"] for r in rows}),
    }


def _replay_actions(sim, snap, nominal, pads, pad_mu, actions):
    from training.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = 0.002
    restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    t0 = float(sim.data.time)
    rows = []
    from training.legal_recatch_transfer import _sample

    for action in actions:
        step_legal(sim, action)
        rows.append(_sample(sim, t0, "k48", np.zeros(3), 0.0))
        if latched(sim) or rows[-1]["bottom"] < TABLE_TOP - 0.02:
            break
    return rows


def _public_hold(hold, sim):
    end = hold.get("end") or {}
    return {
        "reason": hold.get("reason"),
        "success": bool(hold.get("success")),
        "handoff_t": None if hold.get("t_handoff") is None else round(float(hold["t_handoff"]), 4),
        "nominal_elapsed": None if hold.get("nominal_elapsed") is None else round(float(hold["nominal_elapsed"]), 3),
        "lift_completed": hold.get("lift_completed"),
        "table_drop": bool(latched(sim)),
        "end_z": end.get("z"),
        "end_nL": end.get("nL"),
        "end_nR": end.get("nR"),
        "p_des": None if hold.get("handoff_record") is None else hold["handoff_record"].get("p_des"),
        "v_cmd_norm": None if hold.get("handoff_record") is None else hold["handoff_record"].get("v_cmd_norm"),
        "r_des_to_saved_rad": None if hold.get("handoff_record") is None else hold["handoff_record"].get("r_des_to_saved_rad"),
        "grip_ctrl": None if hold.get("handoff_record") is None else hold["handoff_record"].get("grip_ctrl"),
    }


def main():
    sim, snap, nominal, pads, pad_mu = _prepare()
    supports = _enable_supports(sim)
    limits = np.array(sim.model.actuator_ctrlrange[:8], float)
    counts = _install_counter(sim)
    result = run_speed(sim, snap, nominal, pads, pad_mu, SPEC)
    mujoco.mj_step = REAL_STEP
    story = _story(result["rows"])
    story["table_contacts"] = int(counts["table"])
    story["floor_contacts"] = int(counts["floor"])
    story["table_drop"] = bool(latched(sim))
    story["stage"] = result["stage"]
    story["ball_hits"] = int(result["ball_hits"])
    actions = np.stack(result["actions"]).astype(np.float32)
    np.savez_compressed(OUT / "traces" / "legal_A_actions.npz", actions=actions)
    _save_rows("legal_A_best", result["rows"])
    print("LEGAL", story, flush=True)

    book = load_codebook()
    packed = representability(actions, book)
    decoded = np.asarray(packed["quantized"], np.float32)
    err = np.max(np.abs(decoded - actions), axis=1)
    k48 = {
        "n_actions": int(len(actions)),
        "max_linf_to_codeword": packed["max_linf"],
        "fraction_within_0.25": packed["fraction_within_0.25"],
        "representable": bool(packed["max_linf"] is not None and packed["max_linf"] <= 0.25 + 1e-8),
        "max_abs_quant_error": round(float(np.max(err)), 4),
        "median_abs_quant_error": round(float(np.median(err)), 4),
    }
    counts = _install_counter(sim)
    k_rows = _replay_actions(sim, snap, nominal, pads, pad_mu, decoded)
    mujoco.mj_step = REAL_STEP
    k_story = _story(k_rows)
    k_story["table_contacts"] = int(counts["table"])
    k_story["floor_contacts"] = int(counts["floor"])
    k_story["table_drop"] = bool(latched(sim))
    held = bool(
        k_story["stable_n"] == 0
        and k_story["later_loss_t"] is None
        and k_story["recontact_t"] is not None
        and k_story["min_bottom_m"] >= TABLE_TOP
        and not k_story["table_drop"]
        and k_story["end_n"][0] > 0
        and k_story["end_n"][1] > 0
        and k_story["end_voz"] > -0.05
        and k_story["t_end"] >= 1.2
    )
    # The replay has no phase labels, so the hold is the raw ending above the table.
    k48["rollout"] = k_story
    k48["airborne_hold_kept"] = held
    _save_rows("k48_replay", k_rows)
    print("K48", k48["representable"], k48["max_linf_to_codeword"], k_story, flush=True)

    handoff = None
    if held and k48["representable"]:
        from training.multi_disturbance_airborne_feasibility import restore

        from training.nominal_pose_audit import hand_R
        from controllers.geometry_switch import hand_pose

        restore(sim, snap, nominal, pads, pad_mu)
        pose, relative = hand_pose(sim)
        ref = {
            "R_hand": hand_R(sim),
            "r_des": np.array(sim.fsm.r_des, float).copy(),
            "p_nom": np.array(pose, float).copy(),
            "R_rel": np.array(relative, float).copy(),
        }
        # Return to the end of the K48 tape, which is the current sim only if we did not restore.
        # Re-run the tape, then hand off without restoring again.
        _install_counter(sim)
        _replay_actions(sim, snap, nominal, pads, pad_mu, decoded)
        mujoco.mj_step = REAL_STEP

        class Env:
            pass

        env = Env()
        env.sim = sim
        import training.horizon_outcome as horizon

        saved_handoff = {}
        real_handoff = horizon.apply_handoff

        def _remember(sim_now, ref_now):
            record = real_handoff(sim_now, ref_now)
            saved_handoff.update(record)
            return record

        horizon.apply_handoff = _remember
        try:
            raw = nominal_verify(env, ref, seconds=NOMINAL_S)
            horizon.apply_handoff = real_handoff
            raw["handoff_record"] = raw.get("handoff") if isinstance(raw.get("handoff"), dict) else None
            # nominal_verify does not return the apply_handoff dict. Re-read the fields it stored
            # only if present. The function's local handoff is not in the return. Record fsm now.
            handoff = {
                "reason": raw.get("reason"),
                "success": bool(raw.get("success")),
                "nominal_elapsed": None if raw.get("nominal_elapsed") is None else round(float(raw["nominal_elapsed"]), 3),
                "lift_completed": raw.get("lift_completed"),
                "table_drop": bool(latched(sim)),
                "end_z": None if raw.get("end") is None else raw["end"].get("z"),
                "end_nL": None if raw.get("end") is None else raw["end"].get("nL"),
                "end_nR": None if raw.get("end") is None else raw["end"].get("nR"),
                "handoff": {key: saved_handoff.get(key) for key in ("t", "p_des", "v_cmd_norm", "r_des_to_saved_rad", "r_des_to_hand_rad", "grip_ctrl", "physics_edited")},
            }
        except Exception as exc:
            import traceback

            horizon.apply_handoff = real_handoff
            handoff = {"error": str(exc), "traceback": traceback.format_exc(), "success": False}
        print("HANDOFF", handoff, flush=True)
    else:
        print("HANDOFF skipped", flush=True)

    videos = {}
    if "--no-video" in sys.argv:
        payload = {
            "supports": supports,
            "actuator_ctrlrange": limits.tolist(),
            "legal_A": story,
            "k48": k48,
            "nominal_12s": handoff,
            "videos": videos,
        }
        (OUT / "k48_handoff.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print("HANDOFF_ONLY", handoff, flush=True)
        return
    for interface in ("C", "B", "A"):
        replay = run(sim, snap, nominal, pads, pad_mu, WIN, interface=interface, drop_table=False, horizon=2.2)
        dest = OUT / "videos" / f"table_on_{interface}.mp4"
        videos[interface] = str(_render(sim, replay, dest, f"table-on {interface}"))
        print("VIDEO", interface, replay["labels"], videos[interface], flush=True)

    payload = {
        "supports": supports,
        "actuator_ctrlrange": limits.tolist(),
        "legal_A": story,
        "k48": k48,
        "nominal_12s": handoff,
        "videos": videos,
    }
    (OUT / "k48_handoff.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("WROTE", OUT / "k48_handoff.json", flush=True)


if __name__ == "__main__":
    main()
