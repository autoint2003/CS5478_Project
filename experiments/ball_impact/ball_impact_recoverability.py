"""Find one real ball-impact whose nominal hold fails and legal recovery succeeds.

Does not train and does not change step_legal, K48, TABLE_DROP, or nominal control.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_generation.strategies.recatch_oracle import _nominal_fail, _action_from, _setup

from controllers.recovery_actions import AIRBORNE_V3, normalize_action
from envs.airborne_obs import relative_state, rotvec
from envs.deterioration import body_twist
from data_generation.ball_impact_dataset import aim_ball, cylinder_contacts
from data_generation.strategies.capture_geometry import capture_geometry
from envs.horizon_outcome import NOMINAL_S, nominal_verify
from training.impact_ball import apply_ball_free_state
from experiments.recatch.legal_recatch_transfer import _bottom, _enable_supports, _prepare, _sample
from training.nominal_pose_audit import hand_R
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from experiments.recatch.recovery_ic_expansion import _flight_step, _ref_now
from training.recovery_runtime import make_snap, restore_ballistic
from envs.recovery_success import nominal_step
from training.replay_core import TABLE_TOP, park_impact_ball
from envs.table_drop import latched, restore_state
from training.unified_k47 import load_codebook

OUT = ROOT / "results" / "diagnostics" / "ball_impact_recoverability"
OLD = ROOT / "results" / "diagnostics" / "recovery_ic_expansion" / "IC_MANIFEST.csv"


def _cell_from_row(row):
    return {
        "azimuth_deg": float(row["azimuth_deg"]),
        "speed_mps": float(row["speed_mps"]),
        "z_off_m": float(row["z_off_m"] or 0.0),
        "phi_deg": float(row["phi_deg"] or 0.0),
        "delay_s": 0.0,
        "mass_kg": None,
        "friction": None,
        "id": row["id"],
        "split": row["split"],
        "family": row["family"],
    }


def _unsolved_cells():
    rows = []
    for line in OLD.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 5 or parts[4] != "UNSOLVED":
            continue
        header = OLD.read_text(encoding="utf-8").splitlines()[0].split(",")
        row = dict(zip(header, parts))
        rows.append(_cell_from_row(row))
    return rows


def _fly(sim, base, cell, age_s=None):
    """One real flight. Returns the post-impact snapshot and a short trace."""
    restore_ballistic(sim, base)
    restore_state(sim, None)
    sim.model.opt.timestep = 0.002
    sim.set_guide_weld(False)
    launch = aim_ball(sim, cell["azimuth_deg"], cell["speed_mps"], cell["z_off_m"], cell["phi_deg"])
    before = _pose(sim)
    apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
    if cylinder_contacts(sim)[0]:
        return None, "overlapped"
    contact_t = None
    t0 = float(sim.data.time)
    peak = 0.0
    trace = []
    lost = False
    for _ in range(int(0.55 / 0.002)):
        _flight_step(sim)
        now = float(sim.data.time)
        hits, _other = cylinder_contacts(sim)
        if hits and contact_t is None:
            contact_t = now
        if hits:
            peak = max(peak, max(float(item["fn"]) for item in hits))
        geo = capture_geometry(sim)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        none = int(geo["nL"]) == 0 and int(geo["nR"]) == 0
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            return None, "fell_before_ic"
        if contact_t is None:
            if now > t0 + 0.40:
                break
            continue
        if len(trace) < 8 or none or not both:
            trace.append(_pose(sim))
        if age_s is not None:
            if now >= contact_t + float(age_s):
                break
            continue
        if none and now >= contact_t + 0.012 and _bottom(sim) > 0.50:
            lost = True
            break
        if (not both) and now >= contact_t + 0.030 and _bottom(sim) > 0.50:
            break
        if now >= contact_t + 0.10:
            break
    if contact_t is None:
        return None, "no_contact"
    after = _pose(sim)
    park_impact_ball(sim)
    snap = make_snap(sim)
    return snap, {
        "before": before,
        "after": after,
        "contact_t": round(float(contact_t - t0), 4),
        "peak_n": round(float(peak), 2),
        "lost": bool(lost),
        "launch_v": [round(float(x), 3) for x in launch["v0"]],
    }


def _pose(sim):
    rel = relative_state(sim.model, sim.data, sim.ids)
    geo = capture_geometry(sim)
    _v, w = body_twist(sim.model, sim.data, sim.ids.object_body)
    return {
        "p": [round(float(x), 4) for x in rel["p_obj_w"]],
        "v": [round(float(x), 3) for x in rel["v_obj_w"]],
        "w": [round(float(x), 2) for x in w],
        "p_rel": [round(float(x), 4) for x in rel["p_rel_h"]],
        "v_rel": [round(float(x), 3) for x in rel["v_rel_h"]],
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "bottom": round(_bottom(sim), 4),
        "clearance": round(_bottom(sim) - TABLE_TOP, 4),
    }


def _class_of(pose):
    touch = pose["nL"] + pose["nR"] > 0
    both = pose["nL"] > 0 and pose["nR"] > 0
    v = np.array(pose["v_rel"], float)
    axial = abs(float(v[0]))
    opening = abs(float(v[1]))
    finger = abs(float(v[2]))
    spin = float(np.linalg.norm(pose["w"]))
    if pose["clearance"] < 0.08 and not touch:
        return "E_falling_toward_table"
    if spin > 15.0 and not both:
        return "D_rotation_loss"
    if finger > max(axial, opening) + 0.05 and finger > 0.15:
        return "C_along_finger"
    if not touch and max(axial, opening) > 0.15:
        return "B_left_the_jaw"
    if touch and max(axial, opening) >= finger:
        return "A_lateral_slip_with_contact"
    if not touch:
        return "B_left_the_jaw"
    return "A_lateral_slip_with_contact"




def _wrist(sim, r_saved, gain):
    if gain <= 0.0:
        return np.zeros(3)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    err = rotvec(rd.T @ hand_R(sim))
    # Bring the orientation command onto the hand first, so the arm stops twisting the cylinder.
    return np.clip(float(gain) * err, -4.0, 4.0)


def _oracle(sim, target_rel, r_saved, spec, horizon=2.2):
    t0 = float(sim.data.time)
    rows = []
    actions = []
    quiet = 0
    saw_off = False
    saw_re = False
    mode = "chase"
    for _ in range(int(horizon / 0.02)):
        rel = relative_state(sim.model, sim.data, sim.ids)
        geo = capture_geometry(sim)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        none = int(geo["nL"]) == 0 and int(geo["nR"]) == 0
        if none:
            saw_off = True
        elif saw_off and both:
            saw_re = True
        vo = np.array(rel["v_obj_w"], float)
        vh = np.array(rel["v_hand_w"], float)
        err = np.array(rel["p_rel_h"], float) - np.asarray(target_rel, float)
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        touch = int(geo["nL"]) + int(geo["nR"]) > 0
        err_xy = err.copy()
        err_xy[2] = 0.0
        if touch:
            mode = "hold"
            xy = vo[:2].copy()
            speed = float(np.linalg.norm(xy))
            if speed < 1e-6:
                v_xy = np.zeros(2)
            else:
                v_xy = xy / speed * max(0.0, speed - float(spec["brake"]) * 0.02)
            v_corr = rh @ np.array([
                float(np.clip(spec["k"] * err_xy[0], -0.40, 0.40)),
                float(np.clip(spec["k"] * err_xy[1], -0.40, 0.40)),
                0.0,
            ])
            v_w = np.array([v_xy[0] + v_corr[0], v_xy[1] + v_corr[1], 0.08], float)
            tau = -18.0
            w = _wrist(sim, r_saved, spec["align"])
        else:
            mode = "chase"
            v_corr = rh @ np.array([
                float(np.clip(spec["k"] * err_xy[0], -0.80, 0.80)),
                float(np.clip(spec["k"] * err_xy[1], -0.80, 0.80)),
                0.0,
            ])
            v_w = np.array([vo[0] + v_corr[0], vo[1] + v_corr[1], 0.0], float)
            if float(rel["p_obj_w"][2]) > float(rel["p_hand_w"][2]) + 0.01 and _bottom(sim) > 0.55:
                v_w[2] = float(np.clip(vo[2], -0.20, 0.20))
            gap = float(geo["inner_gap_m"])
            tau = 2.0 if gap < 0.045 else (-18.0 if geo["between"] else 0.0)
            w = np.zeros(3)
        v_w = np.clip(v_w, [-1.0, -1.0, -0.20], [1.0, 1.0, 1.0])
        if touch:
            v_w[2] = max(float(v_w[2]), 0.0)
        centered = float(np.linalg.norm(err_xy)) < 0.012 and float(np.linalg.norm(vo)) < float(spec["v_stop"])
        centered = centered and float(np.linalg.norm(vh)) < float(spec["v_stop"]) and both
        quiet = quiet + 1 if centered and _bottom(sim) >= TABLE_TOP + 0.04 else 0
        if quiet >= int(0.30 / 0.02):
            rows.append(_sample(sim, t0, "handoff", v_w, tau))
            break
        action = _action_from(sim, v_w, w, tau)
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
        rows.append(_sample(sim, t0, mode, v_w, tau))
        if latched(sim) or (_bottom(sim) < TABLE_TOP):
            break
    return {
        "actions": actions,
        "rows": rows,
        "phase": rows[-1]["phase"] if rows else "empty",
        "saw_off": saw_off,
        "saw_re": saw_re,
        "table": bool(latched(sim) or _bottom(sim) < TABLE_TOP),
        "end_v": float(np.linalg.norm(rows[-1]["vo"])) if rows else None,
        "end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])] if rows else [0, 0],
    }




def _replay(sim, snap, actions):
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    t0 = float(sim.data.time)
    rows = []
    for action in actions:
        step_legal(sim, action)
        rows.append(_sample(sim, t0, "replay", np.zeros(3), 0.0))
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            break
    return rows


def _finish(sim, ref):
    class Env:
        pass

    env = Env()
    env.sim = sim
    import envs.horizon_outcome as horizon

    saved = {}
    real = horizon.apply_handoff

    def _remember(sim_now, ref_now):
        record = real(sim_now, ref_now)
        saved.update(record)
        return record

    horizon.apply_handoff = _remember
    try:
        out = nominal_verify(env, ref, seconds=NOMINAL_S)
    finally:
        horizon.apply_handoff = real
    return out, saved


def _variants():
    rows = []
    for k in (4.0, 8.0):
        for brake in (1.5, 3.5):
            for align in (0.0, 2.0):
                rows.append({"k": k, "brake": brake, "align": align, "v_stop": 0.08})
    return rows


def _boundary_cells():
    rows = []
    for azimuth in (10.0, 20.0, 30.0, 40.0):
        for speed in (8.0, 10.0, 12.0):
            for z_off in (0.0, 0.015):
                rows.append({
                    "azimuth_deg": azimuth,
                    "speed_mps": speed,
                    "z_off_m": z_off,
                    "phi_deg": 0.0,
                    "delay_s": 0.0,
                    "mass_kg": None,
                    "friction": None,
                    "id": f"b_az{azimuth:.0f}_v{speed:.0f}_z{z_off:.3f}",
                    "split": "boundary",
                    "family": f"az{azimuth:.0f}",
                })
    return rows


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _try_recover(sim, snap, ref, target, book, spec):
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    rolled = _oracle(sim, target, ref["r_des"], spec)
    if rolled["phase"] != "handoff" or rolled["table"] or not rolled["actions"]:
        return {"ok": False, "stage": "table" if rolled["table"] else rolled["phase"], "rolled": rolled}
    actions = np.stack(rolled["actions"])
    replay = _replay(sim, snap, actions)
    if not replay or latched(sim) or replay[-1]["nL"] == 0 or replay[-1]["nR"] == 0:
        return {"ok": False, "stage": "open_loop", "rolled": rolled}
    packed = representability(actions, book)
    if packed["max_linf"] is None or packed["max_linf"] > 0.25 + 1e-8:
        return {"ok": False, "stage": "k48_range", "rolled": rolled, "linf": packed["max_linf"]}
    tokens = np.asarray(packed["tokens"], np.int64)
    residual = actions - book[tokens]
    decoded = np.asarray(packed["quantized"], np.float32)
    krows = _replay(sim, snap, decoded)
    if not krows or latched(sim) or krows[-1]["nL"] == 0 or krows[-1]["nR"] == 0:
        return {"ok": False, "stage": "k48_replay", "rolled": rolled}
    out, handoff = _finish(sim, ref)
    if not out.get("success") or handoff.get("physics_edited"):
        return {"ok": False, "stage": str(out.get("reason")), "rolled": rolled, "end_v": rolled["end_v"]}
    return {
        "ok": True,
        "stage": "SUCCESS_12S",
        "rolled": rolled,
        "actions": actions,
        "tokens": tokens,
        "residual": residual,
        "decoded": decoded,
        "linf": packed["max_linf"],
        "out": out,
        "handoff": handoff,
        "kind": "airborne_recatch" if rolled["saw_off"] and rolled["saw_re"] else "contact_preserving",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    sim, seated, _nominal, _pads, _pad_mu = _prepare()
    _enable_supports(sim)
    from experiments.recatch.multi_disturbance_airborne_feasibility import restore

    restore(sim, seated, _nominal, _pads, _pad_mu)
    sim.park_ball()
    sim.set_guide_weld(True)
    restore_state(sim, None)
    ref = _ref_now(sim)
    target = relative_state(sim.model, sim.data, sim.ids)["p_rel_h"].copy()
    base = make_snap(sim)
    book = load_codebook()

    analyses = []
    for cell in _unsolved_cells():
        snap, info = _fly(sim, base, cell)
        if snap is None:
            analyses.append({"id": cell["id"], "mechanism": "INVALID", "detail": info})
            print("OLD", cell["id"], info, flush=True)
            continue
        fail = _nominal_fail(sim, snap, ref)
        pose = info["after"]
        mech = _class_of(pose)
        analyses.append({
            "id": cell["id"],
            "azimuth_deg": cell["azimuth_deg"],
            "speed_mps": cell["speed_mps"],
            "z_off_m": cell["z_off_m"],
            "phi_deg": cell["phi_deg"],
            "contact_t": info["contact_t"],
            "peak_n": info["peak_n"],
            "launch_v": info["launch_v"],
            "before_v": info["before"]["v"],
            "after_p": pose["p"],
            "after_v": pose["v"],
            "after_w": pose["w"],
            "p_rel": pose["p_rel"],
            "v_rel": pose["v_rel"],
            "nL": pose["nL"],
            "nR": pose["nR"],
            "clearance_m": pose["clearance"],
            "fully_lost": bool(info["lost"] or (pose["nL"] == 0 and pose["nR"] == 0)),
            "mechanism": mech,
            "nominal_fail_t": fail["t"],
            "nominal_why": fail["why"],
            "oracle_was": "vertical_chase_then_" + cell["id"],
        })
        print("OLD", cell["id"], mech, "clear", pose["clearance"], "n", pose["nL"], pose["nR"], "fail", fail["why"], fail["t"], flush=True)
    (OUT / "unsolved_analysis.json").write_text(json.dumps(analyses, indent=2), encoding="utf-8")

    boundary = []
    promising = []
    for cell in _boundary_cells():
        snap, info = _fly(sim, base, cell)
        if snap is None:
            boundary.append({"id": cell["id"], "nominal": "INVALID", "why": info, "azimuth_deg": cell["azimuth_deg"], "speed_mps": cell["speed_mps"], "z_off_m": cell["z_off_m"]})
            print("BND", cell["id"], info, flush=True)
            continue
        fail = _nominal_fail(sim, snap, ref)
        pose = info["after"]
        row = {
            "id": cell["id"],
            "azimuth_deg": cell["azimuth_deg"],
            "speed_mps": cell["speed_mps"],
            "z_off_m": cell["z_off_m"],
            "phi_deg": 0.0,
            "nominal": "SUCCESS" if fail["ok"] else "FAIL",
            "why": fail["why"],
            "fail_t": fail["t"],
            "mechanism": _class_of(pose),
            "nL": pose["nL"],
            "nR": pose["nR"],
            "clearance_m": pose["clearance"],
            "p_rel_x": pose["p_rel"][0],
            "v_rel_x": pose["v_rel"][0],
            "v_rel_y": pose["v_rel"][1],
            "v_rel_z": pose["v_rel"][2],
            "spin": round(float(np.linalg.norm(pose["w"])), 2),
        }
        boundary.append(row)
        print("BND", cell["id"], row["nominal"], row["mechanism"], "n", pose["nL"], pose["nR"], "clear", pose["clearance"], flush=True)
        if not fail["ok"] and pose["clearance"] > 0.10 and (pose["nL"] + pose["nR"]) > 0:
            score = pose["clearance"] + 0.02 * (pose["nL"] + pose["nR"]) - abs(pose["p_rel"][0])
            promising.append((score, cell, snap, pose, info))
    _csv(OUT / "NOMINAL_FAILURE_BOUNDARY.csv", boundary, [
        "id", "azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "nominal", "why", "fail_t",
        "mechanism", "nL", "nR", "clearance_m", "p_rel_x", "v_rel_x", "v_rel_y", "v_rel_z", "spin",
    ])
    promising.sort(key=lambda item: -item[0])
    chosen = promising[:4]
    if not chosen:
        # Fall back to the mildest old unsolved that still has contact and clearance.
        for cell in _unsolved_cells():
            if cell["speed_mps"] > 12.0 or cell["phi_deg"] != 0.0:
                continue
            snap, info = _fly(sim, base, cell)
            if snap is None:
                continue
            pose = info["after"]
            if pose["clearance"] > 0.10 and pose["nL"] + pose["nR"] > 0:
                chosen.append((0.0, cell, snap, pose, info))
            if len(chosen) >= 3:
                break
    searches = []
    success = None
    for score, cell, snap, pose, info in chosen:
        print("ORACLE", cell["id"], "clear", pose["clearance"], "n", pose["nL"], pose["nR"], flush=True)
        for spec in _variants():
            result = _try_recover(sim, snap, ref, target, book, spec)
            searches.append({
                "id": cell["id"],
                "k": spec["k"],
                "brake": spec["brake"],
                "align": spec["align"],
                "stage": result["stage"],
                "end_v": None if result.get("rolled") is None else result["rolled"].get("end_v"),
                "end_n": None if result.get("rolled") is None else result["rolled"].get("end_n"),
            })
            print(" ", spec["k"], spec["brake"], spec["align"], result["stage"], flush=True)
            if result["ok"]:
                success = (cell, snap, info, pose, result)
                break
        if success is not None:
            break
    _csv(OUT / "ORACLE_SEARCH_RESULTS.csv", searches, ["id", "k", "brake", "align", "stage", "end_v", "end_n"])
    payload = {"success": None}
    if success is not None:
        cell, snap, info, pose, result = success
        folder = OUT / "success"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / "actions.npz",
            actions=result["actions"],
            tokens=result["tokens"],
            residual=result["residual"].astype(np.float32),
            decoded=result["decoded"],
        )
        # Second independent replay.
        again = _replay(sim, snap, result["decoded"])
        out2, hand2 = _finish(sim, ref)
        from experiments.recatch.recatch_interface_comparison import _render

        _render(sim, {"rows": result["rolled"]["rows"]}, OUT / "videos" / "success.mp4", cell["id"])
        payload["success"] = {
            "id": cell["id"],
            "cell": {key: cell[key] for key in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg")},
            "kind": result["kind"],
            "k48_linf": result["linf"],
            "token_only_error": float(np.max(np.abs(book[result["tokens"]] - result["actions"]))),
            "nominal": result["out"].get("reason"),
            "replay2": out2.get("reason"),
            "handoff_p_des": result["handoff"].get("p_des"),
            "handoff_physics_edited": result["handoff"].get("physics_edited"),
            "end_z": None if result["out"].get("end") is None else result["out"]["end"].get("z"),
            "replay2_end_z": None if out2.get("end") is None else out2["end"].get("z"),
            "clearance_at_ic": pose["clearance"],
            "contacts_at_ic": [pose["nL"], pose["nR"]],
        }
        print("SUCCESS", payload["success"], flush=True)
    (OUT / "summary.json").write_text(json.dumps({"analysis_n": len(analyses), "boundary_n": len(boundary), "searches": len(searches), "success": payload["success"]}, indent=2, default=str), encoding="utf-8")
    print("DONE", payload["success"] is not None, flush=True)


WIN = {"k": 8.0, "brake": 3.5, "align": 0.0, "v_stop": 0.08}


def _cell(azimuth, speed, z_off, phi=0.0):
    return {
        "azimuth_deg": float(azimuth),
        "speed_mps": float(speed),
        "z_off_m": float(z_off),
        "phi_deg": float(phi),
        "delay_s": 0.0,
        "mass_kg": None,
        "friction": None,
        "id": f"p_az{azimuth:.0f}_v{speed:.0f}_z{z_off:.3f}_p{phi:.0f}",
        "split": "train" if abs(float(azimuth) - 40.0) < 1e-6 else "valid",
        "family": f"az{float(azimuth):.0f}",
    }




def verify():
    """Replay the discovered impact twice and probe the nearby parameter region."""
    import pickle

    from experiments.recatch.recovery_ic_expansion import _table_touch
    from experiments.recatch.recatch_interface_comparison import _render

    sim, base, ref, target, book = _setup()
    cell = _cell(40.0, 8.0, 0.0)
    cell["id"] = "b_az40_v8_z0.000"
    cell["split"] = "train"
    snap, info = _fly(sim, base, cell)
    fail = _nominal_fail(sim, snap, ref)
    result = _try_recover(sim, snap, ref, target, book, WIN)
    if not result["ok"]:
        print("VERIFY_LOST", result["stage"], flush=True)
        return
    table_hits = 0
    min_bottom = 1.0
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    t0 = float(sim.data.time)
    rows = []
    for action in result["decoded"]:
        step_legal(sim, action)
        table_hits += int(_table_touch(sim))
        min_bottom = min(min_bottom, _bottom(sim))
        rows.append(_sample(sim, t0, "replay", np.zeros(3), -18.0))
    reasons = []
    for _ in range(2):
        restore_ballistic(sim, snap)
        restore_state(sim, None)
        for action in result["decoded"]:
            step_legal(sim, action)
        out, handoff = _finish(sim, ref)
        reasons.append({
            "reason": out.get("reason"),
            "end_z": None if out.get("end") is None else out["end"].get("z"),
            "nL": None if out.get("end") is None else out["end"].get("nL"),
            "nR": None if out.get("end") is None else out["end"].get("nR"),
            "physics_edited": handoff.get("physics_edited"),
            "elapsed": out.get("nominal_elapsed"),
        })
        print("REPLAY", reasons[-1], flush=True)
    folder = OUT / "success"
    folder.mkdir(exist_ok=True)
    with (folder / "snapshot.pkl").open("wb") as handle:
        pickle.dump(snap, handle)
    np.savez_compressed(
        folder / "trace.npz",
        t=np.array([row["t"] for row in rows]),
        bottom=np.array([row["bottom"] for row in rows]),
        zo=np.array([row["zo"] for row in rows]),
        xy=np.array([row["xy"] for row in rows]),
        nL=np.array([row["nL"] for row in rows]),
        nR=np.array([row["nR"] for row in rows]),
        vo=np.array([row["vo"] for row in rows]),
        qpos=np.array([row["qpos"] for row in rows]),
    )
    _render(sim, {"rows": rows}, OUT / "videos" / "success_replay.mp4", "az40_v8")
    record = {
        "nominal_reason": fail["why"],
        "nominal_elapsed": fail["t"],
        "peak_normal_n": info["peak_n"],
        "launch_v": info["launch_v"],
        "contact_t": info["contact_t"],
        "table_geom_contacts_during_replay": int(table_hits),
        "table_drop": bool(latched(sim)),
        "min_bottom_m": round(float(min_bottom), 4),
        "kind": result["kind"],
        "replays": reasons,
        "k48_linf": result["linf"],
    }
    probes = []
    for azimuth, speed, z_off, phi in (
        (35.0, 8.0, 0.0, 0.0),
        (45.0, 8.0, 0.0, 0.0),
        (40.0, 7.0, 0.0, 0.0),
        (40.0, 9.0, 0.0, 0.0),
        (40.0, 8.0, 0.005, 0.0),
        (40.0, 8.0, 0.010, 0.0),
        (40.0, 10.0, 0.0, 0.0),
        (140.0, 8.0, 0.0, 0.0),
    ):
        probe = _cell(azimuth, speed, z_off, phi)
        probe["split"] = "train" if abs(azimuth - 40.0) < 1e-6 else "valid"
        psnap, pinfo = _fly(sim, base, probe)
        if psnap is None:
            probes.append({"id": probe["id"], "split": probe["split"], "stage": pinfo})
            print("PERT", probe["id"], pinfo, flush=True)
            continue
        pfail = _nominal_fail(sim, psnap, ref)
        if pfail["ok"]:
            probes.append({"id": probe["id"], "split": probe["split"], "stage": "TRIVIAL", "nominal": pfail["why"]})
            print("PERT", probe["id"], "TRIVIAL", flush=True)
            continue
        found = _try_recover(sim, psnap, ref, target, book, WIN)
        probes.append({
            "id": probe["id"],
            "split": probe["split"],
            "family": probe["family"],
            "azimuth_deg": azimuth,
            "speed_mps": speed,
            "z_off_m": z_off,
            "nominal": pfail["why"],
            "stage": found["stage"],
            "kind": found.get("kind"),
            "peak_n": pinfo["peak_n"],
            "nL": pinfo["after"]["nL"],
            "nR": pinfo["after"]["nR"],
        })
        print("PERT", probe["id"], found["stage"], flush=True)
        if not found["ok"] and probe["id"].startswith("p_az40_v10"):
            _render(sim, {"rows": found["rolled"]["rows"]}, OUT / "videos" / "fail_az40_v10.mp4", "az40_v10")
    _csv(OUT / "PERTURBATION.csv", probes, ["id", "split", "family", "azimuth_deg", "speed_mps", "z_off_m", "nominal", "stage", "kind", "peak_n", "nL", "nR"])
    record["perturbations"] = probes
    (OUT / "success" / "verification.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("VERIFY_DONE", record["nominal_reason"], record["replays"], "table", table_hits, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify()
    else:
        main()
