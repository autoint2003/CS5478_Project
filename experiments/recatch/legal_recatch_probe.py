"""Contact counts and a closed-loop brake from the shared contact state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import AIRBORNE_V3, normalize_action
from envs.deterioration import body_twist
from data_generation.strategies.capture_geometry import capture_geometry
from experiments.recatch.legal_recatch_transfer import (
    OUT,
    WIN,
    _bottom,
    _brake_c,
    _chase_until_contact,
    _enable_supports,
    _pinch_target,
    _prepare,
    _sample,
)
from experiments.ball_impact.privileged_ball_impact_search import step_legal
from experiments.recatch.recatch_braking_optimization import run
from training.replay_core import TABLE_TOP
from envs.table_drop import latched

TRACE = OUT / "traces" / "A_enter_a3.0_m0.0_p0.0.npz"


def _tally(sim):
    import mujoco

    if getattr(sim, "_count_support", None) is None:
        return
    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    obj = int(sim.ids.object_geom)
    for index in range(int(sim.data.ncon)):
        geoms = {int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)}
        if obj not in geoms:
            continue
        if table in geoms:
            sim._count_support["table"] += 1
        if floor in geoms:
            sim._count_support["floor"] += 1


def _release(rows):
    off = None
    back = None
    for i, row in enumerate(rows):
        none = row["nL"] == 0 and row["nR"] == 0
        both = row["nL"] > 0 and row["nR"] > 0
        if none and off is None:
            off = i
        if off is not None and both and back is None:
            back = i
            break
    if off is None:
        return {"released": False}
    t0 = float(rows[off]["t"])
    t1 = float(rows[back]["t"]) if back is not None else float(rows[-1]["t"])
    return {
        "released": True,
        "off_t": round(t0, 4),
        "recontact_t": None if back is None else round(t1, 4),
        "free_s": round(t1 - t0, 4),
        "re_bottom": None if back is None else round(float(rows[back]["bottom"]), 4),
        "re_voz": None if back is None else round(float(rows[back]["vo"][2]), 3),
        "re_vhz": None if back is None else round(float(rows[back]["vh"][2]), 3),
        "re_xy": None if back is None else [round(float(v), 4) for v in rows[back]["xy"]],
    }


def _xy_at_min(rows):
    i = int(np.argmin([r["bottom"] for r in rows]))
    return {
        "t": round(float(rows[i]["t"]), 4),
        "bottom": round(float(rows[i]["bottom"]), 4),
        "xy": [round(float(v), 4) for v in rows[i]["xy"]],
        "over_table": bool(0.38 <= rows[i]["xy"][0] <= 0.62 and -0.12 <= rows[i]["xy"][1] <= 0.12),
    }


def _limits(rows):
    tau = np.stack([r["tau_j"] for r in rows])
    return {
        "peak_abs_tau": round(float(np.max(np.abs(tau))), 2),
        "sat_frac_over_80": round(float(np.mean(np.max(np.abs(tau), axis=1) >= 80.0)), 3),
    }


def _closed_loop(sim, state, rh, interface, horizon=1.6):
    from experiments.recatch.legal_recatch_transfer import _restore_live
    from experiments.recatch.recatch_braking_optimization import _step
    from experiments.recatch.recatch_interface_comparison import _mapped_velocity

    _restore_live(sim, state)
    sim._count_support = {"table": 0, "floor": 0}
    t0 = float(sim.data.time)
    v_cmd_z = float(min(body_twist(sim.model, sim.data, sim.ids.hand_body)[0][2], 0.0))
    p_state = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    rows = []
    phase = "brake"
    zo_brake = float(sim.data.xpos[sim.ids.object_body][2])
    for _ in range(int(horizon / 0.02)):
        geo = capture_geometry(sim)
        vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        if phase == "brake":
            v_cmd_z = min(0.0, v_cmd_z + 6.0 * 0.02)
            if both and v_cmd_z >= -0.02 and float(vo[2]) > -0.15:
                phase = "lift"
        elif phase == "lift":
            v_cmd_z = min(0.20, v_cmd_z + 6.0 * 0.02)
            if both and float(vo[2]) > -0.02 and float(sim.data.xpos[sim.ids.object_body][2]) >= zo_brake + 0.03:
                phase = "stable"
        if phase == "stable":
            v_cmd_z = 0.0
        v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
        if interface == "A":
            hand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            action = normalize_action(hand.T @ v_star, np.zeros(3), -18.0, AIRBORNE_V3)
            step_legal(sim, action)
            _tally(sim)
        else:
            _action, v_apply, tau_apply = _mapped_velocity(sim, v_star, -18.0)
            p_state = p_state + v_apply * 0.02
            v_star = v_apply
            for _sub in range(10):
                _step(sim, p_state, v_apply, tau_apply)
        rows.append(_sample(sim, t0, phase, v_star, -18.0))
        if rows[-1]["bottom"] < TABLE_TOP - 0.02 and not both:
            break
        if phase == "stable" and sum(r["phase"] == "stable" for r in rows) >= 50:
            break
    return {
        "interface": interface,
        "n": len(rows),
        "phase": rows[-1]["phase"],
        "min_bottom_m": round(float(min(r["bottom"] for r in rows)), 4),
        "end_zo_m": round(float(rows[-1]["zo"]), 4),
        "end_voz": round(float(rows[-1]["vo"][2]), 3),
        "end_vhz": round(float(rows[-1]["vh"][2]), 3),
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "table_contacts": int(sim._count_support["table"]),
        "floor_contacts": int(sim._count_support["floor"]),
        "table_drop": bool(latched(sim)),
        "min_xy": _xy_at_min(rows),
        "peak_tau": round(float(np.max(np.abs(np.stack([r["tau_j"] for r in rows])))), 2),
    }


def _trace_story():
    data = np.load(TRACE)
    t = data["t"]
    nL, nR = data["nL"], data["nR"]
    both = (nL > 0) & (nR > 0)
    none = (nL == 0) & (nR == 0)
    i_re = int(np.argmax(both)) if np.any(both) else -1
    # first both after a none
    i_re = -1
    seen = False
    for i in range(len(t)):
        if none[i]:
            seen = True
        elif seen and both[i]:
            i_re = i
            break
    i_loss = -1
    if i_re >= 0:
        for i in range(i_re, len(t)):
            if none[i]:
                i_loss = i
                break
    def pack(i):
        if i < 0:
            return None
        return {
            "t": round(float(t[i]), 4),
            "bottom": round(float(data["bottom"][i]), 4),
            "zo": round(float(data["zo"][i]), 4),
            "xy": [round(float(v), 4) for v in data["xy"][i]],
            "vh": [round(float(v), 3) for v in data["vh"][i]],
            "vo": [round(float(v), 3) for v in data["vo"][i]],
            "vrel_h": [round(float(v), 3) for v in data["vrel_h"][i]],
            "fn": [round(float(v), 2) for v in data["fn"][i]],
            "gap_mm": round(float(data["gap"][i]) * 1e3, 1),
            "p_err_mm": round(float(data["p_err"][i]) * 1e3, 1),
            "v_des_z": round(float(data["v_des_z"][i]), 3),
            "n": [int(nL[i]), int(nR[i])],
        }
    return {
        "recontact": pack(i_re),
        "contact_loss": pack(i_loss),
        "end": pack(len(t) - 1),
        "max_gap_mm": round(float(np.max(data["gap"])) * 1e3, 1),
        "max_p_err_mm": round(float(np.max(data["p_err"])) * 1e3, 1),
    }


def main():
    sim, snap, nominal, pads, pad_mu = _prepare()
    supports = _enable_supports(sim)
    audits = []
    for interface in ("C", "B", "A"):
        sim._count_support = {"table": 0, "floor": 0}
        result = run(sim, snap, nominal, pads, pad_mu, WIN, interface=interface, drop_table=False, horizon=2.4)
        audits.append({
            "interface": interface,
            "labels": result["labels"],
            "min_bottom_m": result["min_bottom_m"],
            "table_drop": bool(latched(sim)),
            "table_contacts": int(sim._count_support["table"]),
            "floor_contacts": int(sim._count_support["floor"]),
            "release": _release(result["rows"]),
            "at_min": _xy_at_min(result["rows"]),
            "limits": _limits(result["rows"]),
            "end_xy": [round(float(v), 4) for v in result["rows"][-1]["xy"]],
        })
        print("AUDIT", interface, audits[-1]["table_contacts"], audits[-1]["at_min"], audits[-1]["release"], flush=True)
    state, rh, contact_t = _chase_until_contact(sim, snap, nominal, pads, pad_mu)
    closed = [_closed_loop(sim, state, rh, name) for name in ("A", "B")]
    # C closed loop at 2 ms, same law, for the same horizon, using the existing tape helper extended.
    from envs.table_drop import restore_state

    restore_state(sim, None)
    rows_c, _tape = _brake_c(sim, state, rh, horizon=1.2)
    story = _trace_story()
    payload = {
        "supports": supports,
        "audit": audits,
        "contact_t": contact_t,
        "closed_loop_from_C_contact": closed,
        "C_open_loop_1p2s_min_bottom": round(float(min(r["bottom"] for r in rows_c)), 4),
        "legal_A_trace": story,
    }
    path = OUT / "probe.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
