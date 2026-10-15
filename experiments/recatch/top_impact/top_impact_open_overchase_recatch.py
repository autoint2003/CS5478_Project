"""Open, over-chase, close and brake after a real top impact.

Does not change step_legal, nominal control, TABLE_DROP, or the K48 codebook.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_generation.strategies.recatch_oracle import _contacts, _pose, _aim, _impact, _finish

from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE
from envs.observations.airborne_obs import relative_state
from envs.simulation.grasp_quality import body_twist
from data_generation.ball_impact_dataset import cylinder_contacts
from experiments.ball_impact.ball_impact_recoverability import _action_from, _nominal_fail, _setup
from data_generation.strategies.capture_geometry import PINCH_HAND, capture_geometry, choose_grip_tau
from envs.outcomes.horizon_outcome import NOMINAL_S, nominal_verify
from envs.simulation.impact_ball import apply_ball_free_state
from experiments.recatch.legal.legal_recatch_transfer import _bottom
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from experiments.recatch.top_impact.recovery_ic_expansion import _flight_step
from training.recovery_execution import make_snap, restore_ballistic
from envs.outcomes.recovery_success import nominal_step
from envs.simulation.scene_state import TABLE_TOP, park_impact_ball
from envs.outcomes.table_drop import latched, restore_state
from training.k47_codebook import load_codebook

OUT = ROOT / "results" / "diagnostics" / "top_impact_open_overchase_recatch"












def _events(rows, gap0):
    def first(pred):
        for row in rows:
            if pred(row):
                return round(float(row["t"]), 3)
        return None

    opened = first(lambda row: row["gap"] > gap0 + 0.003)
    both_off = first(lambda row: row["nL"] == 0 and row["nR"] == 0)
    re_both = None
    seen_off = False
    for row in rows:
        if row["nL"] == 0 and row["nR"] == 0:
            seen_off = True
        elif seen_off and row["nL"] > 0 and row["nR"] > 0:
            re_both = round(float(row["t"]), 3)
            break
    return {
        "t_actual_open": opened,
        "t_both_off": both_off,
        "t_bilateral_recontact": re_both,
        "t_favorable": first(lambda row: row.get("favorable")),
        "t_brake": first(lambda row: row.get("mode") == "brake"),
        "t_vz0": first(lambda row: row.get("mode") in ("brake", "lift") and row["v_obj"][2] > -0.05),
        "t_lift": first(lambda row: row.get("mode") == "lift"),
        "gap_max_mm": round(1000.0 * max(row["gap"] for row in rows), 2),
        "min_bottom": round(min(row["bottom"] for row in rows), 4),
        "min_fz_normal": round(min(row["fz_n"] for row in rows), 3),
    }


def _oracle(sim, snap, spec):
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    t0 = float(sim.data.time)
    gap0 = _pose(sim)["gap"]
    mode = "open"
    both_for = 0
    quiet = 0
    rows = []
    actions = []
    for _ in range(int(1.6 / 0.02)):
        pose = _pose(sim)
        pose["t"] = float(sim.data.time) - t0
        geo = capture_geometry(sim)
        rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        # Pinch below the cylinder center, with the center still inside the open jaw.
        pinch_below = float(pose["pinch"][2]) <= float(pose["p_obj"][2]) - float(spec["lead"])
        centered = abs(float(pose["p_rel"][0])) < 0.025 and abs(float(pose["p_rel"][1])) < 0.02
        favorable = bool(pinch_below and centered and pose["gap"] >= 0.042 and pose["bottom"] > 0.50)
        pose["favorable"] = favorable
        pose["mode"] = mode
        both = pose["nL"] > 0 and pose["nR"] > 0
        if mode in ("open", "chase"):
            tau = TAU_OPEN if pose["gap"] < float(spec["gap"]) else 0.0
            v_center = rh @ np.array([
                float(np.clip(3.0 * pose["p_rel"][0], -0.2, 0.2)),
                float(np.clip(3.0 * pose["p_rel"][1], -0.2, 0.2)),
                0.0,
            ])
            v_w = np.array([v_center[0], v_center[1], -float(spec["down"])])
            mode = "chase" if pose["gap"] >= float(spec["gap"]) else "open"
            if favorable:
                mode = "close"
            elif pose["t"] > 0.28 or pose["bottom"] < 0.48:
                rows.append(pose)
                break
        elif mode == "close":
            tau = TAU_SECURE if both else -8.0
            # Stay slightly below the cylinder while the fingers travel inward.
            v_w = pose["v_obj"].copy()
            v_w[0] = float(np.clip(v_w[0], -0.3, 0.3))
            v_w[1] = float(np.clip(v_w[1], -0.3, 0.3))
            v_w[2] = float(min(pose["v_obj"][2] - 0.30, -0.20))
            both_for = both_for + 1 if both else 0
            if both_for >= 2:
                mode = "brake"
            elif pose["bottom"] < 0.43:
                rows.append(pose)
                break
        elif mode == "brake":
            tau = TAU_SECURE
            speed = float(pose["v_obj"][2])
            vz = 0.0 if speed >= -0.02 else min(0.0, speed + float(spec["brake"]) * 0.02)
            v_w = np.array([
                float(np.clip(pose["v_obj"][0], -0.2, 0.2)),
                float(np.clip(pose["v_obj"][1], -0.2, 0.2)),
                vz,
            ])
            if both and pose["v_obj"][2] > -0.05 and pose["v_hand"][2] > -0.08:
                mode = "lift"
            if not both:
                mode = "close"
        else:
            tau = TAU_SECURE
            v_w = np.array([0.0, 0.0, 0.08])
            quiet = quiet + 1 if both and abs(float(pose["v_obj"][2])) < 0.08 and pose["bottom"] > 0.52 else 0
            if quiet >= 8:
                pose["mode"] = "handoff"
                rows.append(pose)
                break
        v_w = np.clip(v_w, [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
        action = _action_from(sim, v_w, np.zeros(3), tau)
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
        pose = _pose(sim)
        pose["t"] = float(sim.data.time) - t0
        pose["mode"] = mode
        pose["favorable"] = favorable
        rows.append(pose)
        if latched(sim) or pose["bottom"] < TABLE_TOP:
            break
    end_mode = rows[-1]["mode"] if rows else "empty"
    return {
        "actions": actions,
        "rows": rows,
        "reached_handoff": end_mode == "handoff",
        "events": _events(rows, gap0) if rows else {},
        "table": bool(latched(sim) or (rows and rows[-1]["bottom"] < TABLE_TOP)),
    }


def _cells():
    rows = []
    for speed in (6.0, 8.0, 10.0):
        for pitch in (10.0, 25.0):
            for axial in (0.0, 0.022, 0.028):
                for ecc in (0.0, 0.010):
                    rows.append({
                        "speed": speed, "pitch": pitch, "axial": axial, "ecc": ecc,
                        "id": f"s{speed:.1f}_p{pitch:.0f}_a{axial:+.3f}_e{ecc:.3f}",
                    })
    return rows


def _specs():
    return [
        {"gap": 0.044, "lead": 0.006, "down": 2.2, "brake": 6.0},
        {"gap": 0.046, "lead": 0.010, "down": 2.2, "brake": 8.0},
        {"gap": 0.044, "lead": 0.008, "down": 2.6, "brake": 6.0},
        {"gap": 0.042, "lead": 0.004, "down": 2.0, "brake": 5.0},
    ]


def _audit(sim):
    pose = _pose(sim)
    forces = pose["contacts"]
    axis = pose["axis"]
    return {
        "axis_world": [round(float(x), 4) for x in axis],
        "axis_horizontal_deg": round(float(pose["tilt_deg"]), 2),
        "mass_kg": 0.20,
        "radius_m": 0.018,
        "half_length_m": 0.03,
        "ball_radius_m": 0.015,
        "ball_mass_kg": 0.05,
        "gap_mm": round(pose["gap"] * 1000.0, 2),
        "finger_qpos_m": [round(v, 4) for v in pose["q"]],
        "p_obj": [round(float(x), 4) for x in pose["p_obj"]],
        "p_hand": [round(float(x), 4) for x in pose["p_hand"]],
        "p_rel_hand": [round(float(x), 4) for x in pose["p_rel"]],
        "pinch": [round(float(x), 4) for x in pose["pinch"]],
        "nL": pose["nL"],
        "nR": pose["nR"],
        "fz_normal": round(pose["fz_n"], 3),
        "fz_friction": round(pose["fz_f"], 3),
        "contacts": [
            {
                "side": item["side"],
                "pos": [round(float(x), 4) for x in item["pos"]],
                "offset_from_center": [round(float(x), 4) for x in item["offset"]],
                "fn": round(item["fn"], 2),
                "ft": round(item["ft"], 2),
                "fz_normal": round(item["fz_normal"], 3),
                "fz_friction": round(item["fz_friction"], 3),
                "normal": [round(float(x), 3) for x in item["normal_on_cylinder"]],
            }
            for item in forces
        ],
    }


def _public_pose(pose):
    return {
        "p_rel_z": round(float(pose["p_rel"][2]), 4),
        "vz": round(float(pose["v_obj"][2]), 3),
        "tilt": pose["tilt_deg"],
        "nL": pose["nL"],
        "nR": pose["nR"],
        "gap_mm": round(pose["gap"] * 1000.0, 2),
        "bottom": round(pose["bottom"], 4),
        "fz_n": round(pose["fz_n"], 3),
        "fz_f": round(pose["fz_f"], 3),
    }


def _video(rows, dest, title):
    import mujoco
    from experiments.legacy.demo.demonstrations.demo_ballistic_recovery_comparison import caption
    from experiments.legacy.demo.demonstrations.demo_grav_reposition_comparison import write_frames_mp4
    from envs.ballistic_impact import make_sim
    from visualization.impact_view import mjv_camera_from_preset

    sim, _cfg = make_sim()
    look = rows[len(rows) // 3]["p_obj"]
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.55, "azimuth": 130.0, "elevation": -15.0})
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames = []
    next_t = 0.0
    t0 = rows[0]["t"]
    for row in rows:
        if row["t"] - t0 + 1e-9 < next_t:
            continue
        sim.data.qpos[:] = row["qpos"]
        sim.data.qvel[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
        renderer.update_scene(sim.data, cam)
        text = f"{title} t={row['t'] - t0:.2f} n={row['nL']}/{row['nR']} z={row['p_obj'][2]:.3f}"
        frames.append(caption(renderer.render(), text))
        next_t += 1.0 / 15.0
    renderer.close()
    write_frames_mp4(frames, dest)


def _save_trace(path, rows):
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([row["t"] for row in rows]),
        p_obj=np.array([row["p_obj"] for row in rows]),
        p_rel=np.array([row["p_rel"] for row in rows]),
        v_obj=np.array([row["v_obj"] for row in rows]),
        nL=np.array([row["nL"] for row in rows]),
        nR=np.array([row["nR"] for row in rows]),
        gap=np.array([row["gap"] for row in rows]),
        bottom=np.array([row["bottom"] for row in rows]),
        fz_n=np.array([row["fz_n"] for row in rows]),
        fz_f=np.array([row["fz_f"] for row in rows]),
        qpos=np.array([row["qpos"] for row in rows]),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUCCESS_TRACES").mkdir(exist_ok=True)
    (OUT / "FAILURE_TRACES").mkdir(exist_ok=True)
    (OUT / "VIDEOS").mkdir(exist_ok=True)
    sim, base, ref, _target, _book = _setup()
    book = load_codebook()
    audit = _audit(sim)
    # Two seconds of the already-lifted nominal hold, without a ball.
    t0 = float(sim.data.time)
    held = True
    while float(sim.data.time) < t0 + 2.0:
        nominal_step(sim)
        geo = capture_geometry(sim)
        if int(geo["nL"]) == 0 or int(geo["nR"]) == 0 or _bottom(sim) < 0.50:
            held = False
            break
    audit["nominal_hold_2s"] = held
    (OUT / "grasp_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("AUDIT axis", audit["axis_world"], "n", audit["nL"], audit["nR"], "fzN", audit["fz_normal"], "fzF", audit["fz_friction"], "hold", held, flush=True)
    for item in audit["contacts"][:6]:
        print(" ", item["side"], "off", item["offset_from_center"], "fn", item["fn"], "fzN", item["fz_normal"], "fzF", item["fz_friction"], flush=True)

    manifest = []
    promising = []
    for cell in _cells():
        info, err = _impact(sim, base, cell)
        if err:
            manifest.append({"id": cell["id"], "status": err})
            print("MISS", cell["id"], err, flush=True)
            continue
        pose = info["pose"]
        row = {
            "id": cell["id"],
            "speed": cell["speed"],
            "pitch": cell["pitch"],
            "axial": cell["axial"],
            "ecc": cell["ecc"],
            "peak_n": info["peak_n"],
            "impulse": info["impulse"],
            "v_down": info["v_down"],
            "contact_s": info["contact_s"],
            "worst_fz_normal": info["worst_fz_normal"],
            "vz": round(float(pose["v_obj"][2]), 3),
            "prel_z": round(float(pose["p_rel"][2]), 4),
            "d_prel_z": round(float(pose["p_rel"][2] - info["before"]["p_rel"][2]), 4),
            "tilt": round(float(pose["tilt_deg"]), 2),
            "nL": pose["nL"],
            "nR": pose["nR"],
            "bottom": round(pose["bottom"], 4),
            "horizontal": info["horizontal"],
            "progressive": info["progressive"],
            "nominal": "",
        }
        manifest.append(row)
        print("HIT", cell["id"], "F", info["peak_n"], "down", info["v_down"], "dz", row["d_prel_z"], "fz", info["worst_fz_normal"], "vz", row["vz"], "prog", info["progressive"], flush=True)
        serious = abs(cell["axial"]) < 1e-6 or pose["nL"] == 0 or pose["nR"] == 0 or row["vz"] < -0.3
        if info["progressive"] and serious:
            promising.append((info, cell, row))
        elif info["progressive"]:
            row["nominal"] = "skipped_small_slip"
    promising = promising[:8]
    search_rows = []
    timelines = []
    success = None
    for info, cell, row in promising:
        nominal = _nominal_fail(sim, info["snap"], ref)
        row["nominal"] = nominal["why"]
        print("BASE", cell["id"], nominal["why"], nominal["t"], flush=True)
        if nominal["ok"]:
            row["status"] = "TRIVIAL"
            continue
        row["status"] = "NOMINAL_FAIL"
        _save_trace(OUT / "FAILURE_TRACES" / f"{cell['id']}_impact.npz", info["rows"])
        for spec in _specs():
            rolled = _oracle(sim, info["snap"], spec)
            stage = "table" if rolled["table"] else ("handoff" if rolled["reached_handoff"] else rolled["rows"][-1]["mode"] if rolled["rows"] else "empty")
            record = {
                "id": cell["id"],
                "gap": spec["gap"],
                "lead": spec["lead"],
                "down": spec["down"],
                "brake": spec["brake"],
                "stage": stage,
                **rolled["events"],
            }
            if not rolled["reached_handoff"] or rolled["table"] or not rolled["actions"]:
                search_rows.append(record)
                print(" ", spec["lead"], spec["down"], stage, rolled["events"].get("t_actual_open"), rolled["events"].get("t_bilateral_recontact"), flush=True)
                continue
            actions = np.stack(rolled["actions"])
            packed = representability(actions, book)
            record["k48_linf"] = None if packed["max_linf"] is None else round(float(packed["max_linf"]), 4)
            if packed["max_linf"] is None or packed["max_linf"] > 0.25 + 1e-8:
                record["stage"] = "k48_range"
                search_rows.append(record)
                continue
            restore_ballistic(sim, info["snap"])
            restore_state(sim, None)
            for action in packed["quantized"]:
                step_legal(sim, action)
            out, extra = _finish(sim, ref, extra=5.0)
            record["formal"] = out.get("reason")
            record["extension"] = extra
            record["stage"] = out.get("reason")
            search_rows.append(record)
            timelines.append({"id": cell["id"], "spec": spec, **rolled["events"], "formal": out.get("reason"), "extension": extra})
            print(" ", spec, out.get("reason"), extra, flush=True)
            if out.get("success") and extra == "HELD_EXTENSION":
                success = (cell, info, spec, actions, packed, out, extra, rolled)
                break
        if success is not None:
            break
    with (OUT / "IC_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "speed", "pitch", "axial", "ecc", "peak_n", "impulse", "v_down", "contact_s", "worst_fz_normal", "vz", "prel_z", "d_prel_z", "tilt", "nL", "nR", "bottom", "horizontal", "progressive", "nominal", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    with (OUT / "FOUR_PHASE_SEARCH.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "gap", "lead", "down", "brake", "stage", "formal", "extension", "k48_linf", "t_actual_open", "t_both_off", "t_favorable", "t_bilateral_recontact", "t_brake", "t_vz0", "t_lift", "gap_max_mm", "min_bottom", "min_fz_normal"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(search_rows)
    with (OUT / "EVENT_TIMELINES.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "formal", "extension", "t_actual_open", "t_both_off", "t_favorable", "t_bilateral_recontact", "t_brake", "t_vz0", "t_lift"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timelines)
    if success is not None:
        cell, info, spec, actions, packed, out, extra, rolled = success
        folder = OUT / "SUCCESS_TRACES"
        np.savez_compressed(folder / "actions.npz", actions=actions, tokens=packed["tokens"], residual=(actions - book[packed["tokens"]]).astype(np.float32), decoded=np.asarray(packed["quantized"], np.float32))
        with (folder / "snapshot.pkl").open("wb") as handle:
            pickle.dump(info["snap"], handle)
        _save_trace(folder / "recovery.npz", rolled["rows"])
        _video(info["rows"] + rolled["rows"], OUT / "VIDEOS" / "success.mp4", cell["id"])
        (folder / "result.json").write_text(json.dumps({"id": cell["id"], "spec": spec, "formal": out.get("reason"), "extension": extra, "k48": record_linf(packed)}, indent=2), encoding="utf-8")
    elif promising:
        info, cell, _row = promising[0]
        _video(info["rows"], OUT / "VIDEOS" / "impact_nominal.mp4", cell["id"])
    (OUT / "summary.json").write_text(json.dumps({"n_manifest": len(manifest), "n_progressive": len(promising), "success": success is not None}, indent=2), encoding="utf-8")
    print("DONE", success is not None, "progressive", len(promising), flush=True)


def record_linf(packed):
    return None if packed["max_linf"] is None else round(float(packed["max_linf"]), 4)


if __name__ == "__main__":
    main()
