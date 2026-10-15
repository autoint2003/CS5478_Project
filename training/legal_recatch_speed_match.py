"""Legal step_legal search that matches downward speed before it closes and brakes."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.recovery_actions import AIRBORNE_V3, normalize_action
from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from training.capture_geometry import capture_geometry
from training.legal_recatch_transfer import (
    OUT,
    _ball_touch,
    _enable_supports,
    _prepare,
    _sample,
    _save_rows,
)
from training.privileged_ball_impact_search import step_legal
from training.recatch_braking_optimization import BASE_SPEC, _advance, _law
from training.recatch_interface_comparison import _capture_window, _render
from training.replay_core import TABLE_TOP
from training.table_drop import latched, restore_state

CSV_PATH = OUT / "SEARCH_RESULTS.csv"


def run_speed(sim, snap, nominal, pads, pad_mu, spec, horizon=2.8):
    from training.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = 0.002
    restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(rel0["p_rel_h"], float)
    phase = "chase"
    free_s = both_s = 0.0
    armed = False
    v_cmd_z = None
    settle_until = None
    zo_brake = None
    brake_t = None
    rows = []
    actions = []
    ball_hits = 0
    matched = False
    for _ in range(int(horizon / 0.02)):
        geo = capture_geometry(sim)
        phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, 0.02)
        if phase in ("close", "hold"):
            phase = "chase"
        now = float(sim.data.time) - t0
        vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        _p, v_law, _e = _law(
            sim, rh, BASE_SPEC["aim_down"], BASE_SPEC["k"], BASE_SPEC["v_clip"], False, BASE_SPEC["corr_clip"],
        )
        if v_cmd_z is None:
            if now < float(spec["dive_s"]):
                vz = -float(spec["dive"])
            else:
                vz = min(float(v_law[2]), float(vo[2]) - float(spec["lead"]))
            vz = float(np.clip(vz, -3.8, 1.0))
            v_star = np.array([float(v_law[0]), float(v_law[1]), vz], float)
            gap = float(geo["inner_gap_m"])
            if gap < 0.040:
                tau = 2.0
            elif gap > 0.060:
                tau = -4.0
            else:
                tau = 0.0
            ready = _capture_window(geo, free_s, armed) and float(vh[2]) <= float(vo[2]) + float(spec["tol"])
            if ready:
                matched = True
                phase = "settle"
                settle_until = now + float(spec["settle"])
                v_cmd_z = float(min(vo[2], 0.0))
                zo_brake = float(vo[2])
                tau = -18.0
                v_star[2] = v_cmd_z
        elif phase == "settle":
            v_cmd_z = float(min(vo[2], 0.0))
            v_star = np.array([float(vo[0]), float(vo[1]), v_cmd_z], float)
            tau = -18.0
            if now >= settle_until and both:
                phase = "brake"
                brake_t = now
                zo_brake = float(sim.data.xpos[sim.ids.object_body][2])
                v_cmd_z = float(min(vh[2], vo[2], 0.0))
        elif phase == "brake":
            if both and float(vh[2]) > float(vo[2]) + 0.08:
                v_cmd_z = float(min(vo[2], 0.0))
            else:
                v_cmd_z = min(0.0, float(v_cmd_z) + float(spec["accel"]) * 0.02)
            if both and v_cmd_z >= -0.02 and float(vo[2]) > -0.15:
                phase = "lift"
            v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
            tau = -18.0
        elif phase == "lift":
            v_cmd_z = min(0.20, float(v_cmd_z) + float(spec["accel"]) * 0.02)
            if both and float(vo[2]) > -0.02 and float(sim.data.xpos[sim.ids.object_body][2]) >= float(zo_brake) + 0.03:
                phase = "stable"
            v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
            tau = -18.0
        else:
            v_cmd_z = 0.0
            v_star = np.array([float(vo[0]), float(vo[1]), 0.0], float)
            tau = -18.0
        hand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        action = normalize_action(hand.T @ v_star, np.zeros(3), float(tau), AIRBORNE_V3)
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
        ball_hits += _ball_touch(sim)
        rows.append(_sample(sim, t0, phase, v_star, tau))
        if latched(sim) or (rows[-1]["bottom"] < TABLE_TOP and phase in ("brake", "lift", "stable", "settle")):
            break
        if phase == "stable" and sum(r["phase"] == "stable" for r in rows) >= 50:
            break
        if rows[-1]["bottom"] < 0.42 and phase == "chase":
            break
    from training.legal_recatch_transfer import _label_legal

    result = _label_legal(rows, actions, ball_hits, brake_t, bool(latched(sim)))
    if result["stage"] in ("NO_RECONTACT", "NO_RELEASE") and not matched:
        result["stage"] = "CANNOT_MATCH_SPEED"
    result["matched"] = matched
    return result


def _grid():
    specs = []
    for dive in (1.2, 2.0, 3.0):
        for dive_s in (0.04, 0.08):
            for lead in (0.0, 0.4):
                for accel in (3.0, 6.0):
                    specs.append({
                        "dive": dive,
                        "dive_s": dive_s,
                        "lead": lead,
                        "tol": 0.12,
                        "settle": 0.04,
                        "accel": accel,
                        "trigger": "speed_match",
                        "match_s": 0.04,
                        "pre_accel": dive,
                    })
    return specs


def _append(rows):
    fields = ["name", "stage", "trigger", "accel", "match_s", "pre_accel", "recontact_t", "brake_t", "min_bottom_m", "end_zo_m", "end_voz", "end_vhz", "end_nL", "end_nR", "ball_hits"]
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _pub(result, name, spec):
    return {
        "name": name,
        "stage": result["stage"],
        "trigger": spec["trigger"],
        "accel": spec["accel"],
        "match_s": spec["settle"],
        "pre_accel": spec["dive"],
        "recontact_t": result["recontact_t"],
        "brake_t": result["brake_t"],
        "min_bottom_m": result["min_bottom_m"],
        "end_zo_m": result["end_zo_m"],
        "end_voz": result["end_voz"],
        "end_vhz": result["end_vhz"],
        "end_nL": result["end_n"][0],
        "end_nR": result["end_n"][1],
        "ball_hits": result["ball_hits"],
        "labels": result["labels"],
        "matched": result["matched"],
    }


def main():
    sim, snap, nominal, pads, pad_mu = _prepare()
    _enable_supports(sim)
    found = []
    best = None
    for spec in _grid():
        result = run_speed(sim, snap, nominal, pads, pad_mu, spec)
        name = f"Amatch_d{spec['dive']}_s{spec['dive_s']}_l{spec['lead']}_a{spec['accel']}"
        pub = _pub(result, name, spec)
        found.append(pub)
        print("M", name, pub["stage"], "bottom", pub["min_bottom_m"], "re", pub["recontact_t"], "matched", pub["matched"], flush=True)
        rank = (bool(result["stable"]), pub["min_bottom_m"])
        if best is None or rank > best[0]:
            best = (rank, name, result, spec)
        if result["stable"] and result["ball_hits"] == 0 and not latched(sim):
            break
    _append(found)
    _name, result, spec = best[1], best[2], best[3]
    trace = _save_rows(best[1], result["rows"])
    video = None
    if result["rows"] and "qpos" in result["rows"][0]:
        dest = OUT / "videos" / f"{best[1]}.mp4"
        video = _render(sim, result, dest, best[1])
    payload = {"best": {**_pub(result, best[1], spec), "trace": trace, "video": None if video is None else str(video)}, "n": len(found)}
    (OUT / "speed_match.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("BEST", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
