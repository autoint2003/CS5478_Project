"""Top-down ball impacts that tilt a held cylinder, then a legal recatch search.

Does not train and does not change step_legal, nominal control, or SUCCESS_12S.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from training.ball_impact_dataset import cylinder_contacts
from training.ball_impact_recoverability import _nominal_fail, _setup
from training.capture_geometry import capture_geometry
from training.impact_ball import apply_ball_free_state
from training.legal_recatch_transfer import _bottom
from training.recovery_ic_expansion import _flight_step
from training.recovery_runtime import make_snap, restore_ballistic
from training.replay_core import TABLE_TOP, park_impact_ball
from training.table_drop import latched, restore_state

OUT = ROOT / "results" / "diagnostics" / "top_impact_progressive_recatch"


def _aim_top(sim, speed, pitch_deg, axial, lateral, approach):
    """Ball arrives from above. Cylinder qpos is not written."""
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    opening = np.cross(up, axis)
    opening = opening / max(float(np.linalg.norm(opening)), 1e-9)
    radius = float(sim.model.geom_size[int(sim.ids.object_geom)][0])
    ball_r = float(sim.model.geom_size[int(sim.ball_geom)][0])
    p_contact = po + axis * float(axial) + opening * float(lateral) + up * radius
    side = axis if approach == "axial" else opening
    pitch = np.radians(float(pitch_deg))
    incoming = -up * np.cos(pitch) + side * np.sin(pitch)
    incoming = incoming / max(float(np.linalg.norm(incoming)), 1e-9)
    v_imp = incoming * float(speed)
    outward = -incoming
    p_ball = p_contact + outward * (ball_r + 0.001)
    flight = 0.12
    g = np.array(sim.model.opt.gravity, float)
    v0 = v_imp - g * flight
    p_launch = p_ball - v0 * flight - 0.5 * g * flight * flight
    return {"p_launch": p_launch, "v0": v0, "v_imp": v_imp, "p_contact": p_contact}


def _tilt(sim):
    axis = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)[:, 2]
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    horizontal = float(np.linalg.norm(axis[:2]))
    return float(np.degrees(np.arctan2(abs(axis[2]), max(horizontal, 1e-9))))


def _probe(sim, base, cell):
    restore_ballistic(sim, base)
    restore_state(sim, None)
    sim.model.opt.timestep = 0.002
    sim.set_guide_weld(False)
    launch = _aim_top(sim, cell["speed"], cell["pitch"], cell["axial"], cell["lateral"], cell["approach"])
    before = _snap_pose(sim)
    apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
    if cylinder_contacts(sim)[0]:
        return None, "overlapped"
    t0 = float(sim.data.time)
    contact_t = None
    peak = 0.0
    hit_hand = False
    samples = []
    for _ in range(int(0.55 / 0.002)):
        _flight_step(sim)
        now = float(sim.data.time)
        hits, other = cylinder_contacts(sim)
        if hits and contact_t is None:
            contact_t = now
        if hits:
            peak = max(peak, max(float(item["fn"]) for item in hits))
        if other:
            hit_hand = True
        if contact_t is None:
            if now > t0 + 0.35:
                break
            continue
        age = now - contact_t
        if age >= cell.get("sample_at", 0.28) or _bottom(sim) < TABLE_TOP + 0.02 or latched(sim):
            samples.append(_snap_pose(sim))
            break
        if len(samples) == 0 or age >= 0.02 * (len(samples) + 1):
            samples.append(_snap_pose(sim))
    if contact_t is None:
        return None, "no_cylinder_contact"
    pose = samples[-1] if samples else _snap_pose(sim)
    early = samples[0] if samples else pose
    horizontal = float(np.linalg.norm(pose["v"][:2]))
    progressive = bool(
        peak >= 4.0
        and horizontal < 0.45
        and pose["clearance"] > 0.08
        and pose["dist"] < 0.05
        and (pose["tilt"] - before["tilt"] > 8.0 or (early["both"] and not pose["both"]) or pose["slip"] > 0.05)
    )
    return {
        "before": before,
        "early": early,
        "pose": pose,
        "contact_s": round(float(contact_t - t0), 3),
        "peak_n": round(float(peak), 2),
        "v_down": round(float(-launch["v_imp"][2]), 3),
        "hit_hand": bool(hit_hand),
        "progressive": progressive,
        "horizontal": round(horizontal, 3),
        "tilt_gain": round(float(pose["tilt"] - before["tilt"]), 2),
    }, None


def _snap_pose(sim):
    rel = relative_state(sim.model, sim.data, sim.ids)
    geo = capture_geometry(sim)
    _v, w = body_twist(sim.model, sim.data, sim.ids.object_body)
    both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
    return {
        "p": [round(float(x), 4) for x in rel["p_obj_w"]],
        "v": [round(float(x), 3) for x in rel["v_obj_w"]],
        "w": round(float(np.linalg.norm(w)), 2),
        "p_rel": [round(float(x), 4) for x in rel["p_rel_h"]],
        "v_rel": [round(float(x), 3) for x in rel["v_rel_h"]],
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "both": bool(both),
        "tilt": round(_tilt(sim), 2),
        "bottom": round(_bottom(sim), 4),
        "clearance": round(_bottom(sim) - TABLE_TOP, 4),
        "dist": round(float(np.linalg.norm(rel["p_rel_h"])), 4),
        "gap": round(float(geo["inner_gap_m"]) * 1000.0, 2),
        "slip": round(float(np.linalg.norm(rel["v_rel_h"][:2])), 3),
    }


def _cells():
    rows = []
    for speed in (2.0, 3.5, 5.0):
        for pitch in (25.0, 50.0):
            for axial in (-0.024, -0.014, 0.014, 0.024):
                for lateral in (0.0, 0.008):
                    for approach in ("axial", "opening"):
                        rows.append({
                            "speed": speed,
                            "pitch": pitch,
                            "axial": axial,
                            "lateral": lateral,
                            "approach": approach,
                            "id": f"s{speed:.1f}_p{pitch:.0f}_a{axial:+.3f}_l{lateral:.3f}_{approach}",
                        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, base, ref, _target, _book = _setup()
    found = []
    for cell in _cells():
        info, err = _probe(sim, base, cell)
        if err:
            print("MISS", cell["id"], err, flush=True)
            continue
        pose = info["pose"]
        print(
            "HIT", cell["id"], "F", info["peak_n"], "down", info["v_down"],
            "tilt", info["tilt_gain"], "n", pose["nL"], pose["nR"],
            "slip", pose["slip"], "vh", info["horizontal"], "prog", info["progressive"],
            flush=True,
        )
        if info["progressive"]:
            found.append({"cell": cell, "info": info})
    (OUT / "survey.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    print("PROGRESSIVE", len(found), flush=True)


if __name__ == "__main__":
    main()
