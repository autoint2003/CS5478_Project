"""Discover natural post-grasp deterioration under mass/friction uncertainty.

Centered grasp only. No apply_rel_pose, no RULE, no SAC, no impact.

    python training/demo_dynamic_deterioration.py --verify-friction
    python training/demo_dynamic_deterioration.py --friction-regression
    python training/demo_dynamic_deterioration.py --sweep
    python training/demo_dynamic_deterioration.py --mass 0.25 --mu 0.40
    python training/demo_dynamic_deterioration.py --case m0p25_mu0p40 --playback-speed 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import finger_pad_geom_ids, set_finger_object_sliding_mu
from envs.grasp_sim import GraspSim
from envs.observable_reward import read_tactile
from envs.physical_recovery import TABLE_DROP, Z_AIR, physical_pack
from training.impact_demo_core import RENDER_HZ, RealtimePacer, _tilt_deg
from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

LOG_DIR = ROOT / "results" / "logs" / "natural_post_grasp_deterioration_friction_fixed"
FIG_DIR = LOG_DIR / "figures"
MU_MATCH_TOL = 0.05

# Documented train range: mass [0.05, 0.25], friction [0.40, 1.20]
# Nominal yaml: mass 0.08, friction 1.0. Held-out: mass [0.16, 0.22], mu [0.45, 0.70].
MASSES = (0.08, 0.16, 0.25)
MUS = (1.0, 0.70, 0.40)

CAPTURE_HOLD = 0.20  # discovery criterion, not a new task definition
UNI_HOLD = 0.15
BOTH_HOLD = 0.10
SLIP_E = 0.002
SLIP_TILT = 8.0
SLIP_V = 0.03
CLEAR_OFF = 0.010
LOG_DT = 0.0095


def case_id(mass: float, mu: float) -> str:
    return f"m{mass:.2f}_mu{mu:.2f}".replace(".", "p")


def parse_case(s: str) -> tuple[float, float]:
    # m0p25_mu0p40
    parts = str(s).split("_")
    if len(parts) != 2 or not parts[0].startswith("m") or not parts[1].startswith("mu"):
        raise ValueError(f"bad --case {s!r}; expected m0p25_mu0p40")
    mass = float(parts[0][1:].replace("p", "."))
    mu = float(parts[1][2:].replace("p", "."))
    return mass, mu


def _geom_name(model, gid: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"g{gid}"


def live_finger_object_contacts(sim: GraspSim) -> list[dict]:
    model, data, ids = sim.model, sim.data, sim.ids
    obj_g = int(ids.object_geom)
    rows = []
    wr = np.zeros(6)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        bodies = {int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])}
        if obj_g not in (g1, g2):
            continue
        if ids.left_body not in bodies and ids.right_body not in bodies:
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        side = "L" if ids.left_body in bodies else "R"
        rows.append(
            {
                "side": side,
                "geom1": _geom_name(model, g1),
                "geom2": _geom_name(model, g2),
                "geom1_id": g1,
                "geom2_id": g2,
                "geom1_friction": np.array(model.geom_friction[g1], float).tolist(),
                "geom2_friction": np.array(model.geom_friction[g2], float).tolist(),
                "geom1_friction0": float(model.geom_friction[g1][0]),
                "geom2_friction0": float(model.geom_friction[g2][0]),
                "contact_friction": np.array(c.friction, float).tolist(),
                "contact_friction0": float(c.friction[0]),
                "Fn": float(abs(wr[0])),
                "Ft": float(np.hypot(wr[1], wr[2])),
            }
        )
    return rows


def finger_tangent_totals(sim: GraspSim) -> tuple[float, float, float, float]:
    """Sum |Fn|, |Ft| on left/right finger–object contacts."""
    fnL = ftL = fnR = ftR = 0.0
    for r in live_finger_object_contacts(sim):
        if r["side"] == "L":
            fnL += r["Fn"]
            ftL += r["Ft"]
        else:
            fnR += r["Fn"]
            ftR += r["Ft"]
    return fnL, ftL, fnR, ftR


def apply_pair_mu(sim: GraspSim, mu: float) -> dict:
    return set_finger_object_sliding_mu(sim.model, sim.ids, mu, sim.data)


def friction_mass_audit(sim: GraspSim, requested_m: float, requested_mu: float) -> dict:
    model, data, ids = sim.model, sim.data, sim.ids
    obj_g = int(ids.object_geom)
    obj_b = int(ids.object_body)
    left_g, right_g = finger_pad_geom_ids(model, ids)
    pair = live_finger_object_contacts(sim)
    mus = [p["contact_friction0"] for p in pair]
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    table_mu0 = float(model.geom_friction[table_id][0]) if table_id >= 0 else None
    return {
        "requested_mass": float(requested_m),
        "requested_mu": float(requested_mu),
        "body_mass_object": float(model.body_mass[obj_b]),
        "body_subtreemass_object": float(model.body_subtreemass[obj_b]),
        "mj_setConst_called": True,
        "object_geom_friction": np.array(model.geom_friction[obj_g], float).tolist(),
        "left_finger_geom_friction": [np.array(model.geom_friction[g], float).tolist() for g in left_g],
        "right_finger_geom_friction": [np.array(model.geom_friction[g], float).tolist() for g in right_g],
        "table_geom_friction0_unchanged": table_mu0,
        "live_contacts": pair[:16],
        "effective_mu_from_contacts": float(np.median(mus)) if mus else None,
        "note": (
            "YAML mu is realized as sliding friction[0] on object + left/right "
            "finger geoms. Torsional/rolling components are not written."
        ),
    }


def summarize_lr_contacts(contacts: list[dict], requested: float) -> dict:
    out = {"requested_mu": requested, "left": [], "right": []}
    for c in contacts:
        rec = {
            "geom1": c["geom1"],
            "geom2": c["geom2"],
            "object_or_finger_mu": {
                "geom1_friction0": c["geom1_friction0"],
                "geom2_friction0": c["geom2_friction0"],
            },
            "contact.friction": c["contact_friction"],
            "contact.friction[0]": c["contact_friction0"],
        }
        out["left" if c["side"] == "L" else "right"].append(rec)
    def med(side):
        xs = [x["contact.friction[0]"] for x in out[side]]
        return float(np.median(xs)) if xs else None
    out["left_median_contact_mu"] = med("left")
    out["right_median_contact_mu"] = med("right")
    return out


def _row(sim, o, tac, phase: str) -> dict:
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rh = np.asarray(o["rh"], float)
    fnL, ftL, fnR, ftR = finger_tangent_totals(sim)
    mu_c = None
    cons = live_finger_object_contacts(sim)
    if cons:
        mu_c = float(np.median([c["contact_friction0"] for c in cons]))
    util = None
    if mu_c and (fnL + fnR) > 1e-6:
        util = float((ftL + ftR) / (mu_c * (fnL + fnR)))
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "e_x": float(rh[0]),
        "e_y": float(rh[1]),
        "e_z": float(rh[2]),
        "v_rel_hand": np.asarray(o["v_rel_h"], float).tolist(),
        "omega_rel_hand": np.asarray(o["w_rel_h"], float).tolist(),
        "v_rel": float(o["v_rel"]),
        "w_rel": float(o["w_rel"]),
        "tilt_deg": _tilt_deg(sim),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(o["Fn_L"]),
        "Fn_R": float(o["Fn_R"]),
        "Ft_L": ftL,
        "Ft_R": ftR,
        "friction_utilization": util,
        "measured_contact_mu": mu_c,
        "aperture": float(o["aperture"]),
        "tau": float(o["tau"]),
        "hand": ph.tolist(),
        "object": po.tolist(),
        "obj_z": float(po[2]),
        "hand_z": float(ph[2]),
        "clear": float(o["clear"]),
        "u_L": float(tac.get("u_L", float("nan"))),
        "u_R": float(tac.get("u_R", float("nan"))),
        "v_L": float(tac.get("v_L", float("nan"))),
        "v_R": float(tac.get("v_R", float("nan"))),
        "cop_valid": bool(tac.get("estimate_valid", False)),
    }


def _overlay(viewer, mass, mu_eff, phase, t, e_x, e_y) -> None:
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "m", f"{mass:.2f} kg")
        if mu_eff is not None:
            viewer.add_overlay(pos, "mu_eff", f"{mu_eff:.3f}")
        viewer.add_overlay(pos, "PHASE", str(phase).upper())
        viewer.add_overlay(pos, "t", f"{t:.3f} s")
        if e_x is not None:
            viewer.add_overlay(pos, "e_x", f"{1e3 * float(e_x):+.1f} mm")
        if e_y is not None:
            viewer.add_overlay(pos, "e_y", f"{1e3 * float(e_y):+.1f} mm")
    except Exception:
        return


def _cam(sim) -> dict:
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    return {
        "lookat": po + np.array([0.0, 0.0, 0.02]),
        "distance": 0.72,
        "azimuth": 125.0,
        "elevation": -22.0,
    }


def _off_table(o: dict) -> bool:
    return float(o["clear"]) >= CLEAR_OFF and float(o["obj_z"]) > (0.40 + 0.03 + 0.004)


def _bilateral(o: dict) -> bool:
    return int(o["nL"]) > 0 and int(o["nR"]) > 0


def classify_and_events(log: list[dict], fsm_success: bool) -> dict:
    t_close = next((r["t"] for r in log if r["phase"] == "close"), None)
    t_lift = next((r["t"] for r in log if r["phase"] == "lift"), None)
    t_bil = next((r["t"] for r in log if r["nL"] > 0 and r["nR"] > 0), None)

    capture_acc = 0.0
    t_air_bil_start = None
    t_capture_ok = None
    t_first_air = None
    prev_t = None
    for r in log:
        if prev_t is None:
            prev_t = r["t"]
            continue
        dt = max(0.0, r["t"] - prev_t)
        prev_t = r["t"]
        air = _off_table(r) and r["phase"] in ("lift", "hold")
        if air and t_first_air is None:
            t_first_air = r["t"]
        ok = air and r["nL"] > 0 and r["nR"] > 0
        if ok:
            if t_air_bil_start is None:
                t_air_bil_start = r["t"]
            capture_acc += dt
            if t_capture_ok is None and capture_acc >= CAPTURE_HOLD:
                t_capture_ok = r["t"]
        else:
            if t_capture_ok is None:
                capture_acc = 0.0
                t_air_bil_start = None

    airborne_bil_dur = 0.0
    prev_t = None
    for r in log:
        if prev_t is None:
            prev_t = r["t"]
            continue
        dt = max(0.0, r["t"] - prev_t)
        prev_t = r["t"]
        if r["phase"] in ("lift", "hold") and _off_table(r) and r["nL"] > 0 and r["nR"] > 0:
            airborne_bil_dur += dt

    cap_row = None
    if t_capture_ok is not None:
        cap_row = next((r for r in log if r["t"] >= t_capture_ok - 1e-9), log[0])

    def after_cap(r):
        return t_capture_ok is not None and r["t"] >= t_capture_ok - 1e-9

    t_slip = None
    t_deg = None
    t_uni = None
    t_both = None
    t_drop = None
    uni_acc = both_acc = 0.0
    prev_t = None
    for r in log:
        if prev_t is None:
            prev_t = r["t"]
            continue
        dt = max(0.0, r["t"] - prev_t)
        prev_t = r["t"]
        if not after_cap(r):
            continue
        if t_slip is None and cap_row is not None:
            de = abs(r["e_x"] - cap_row["e_x"]) > SLIP_E or abs(r["e_y"] - cap_row["e_y"]) > SLIP_E
            dtilt = (r["tilt_deg"] - cap_row["tilt_deg"]) > SLIP_TILT
            dv = r["v_rel"] > SLIP_V
            if de or dtilt or dv:
                t_slip = r["t"]
        fns = r["Fn_L"] + r["Fn_R"]
        imb = abs(r["Fn_L"] - r["Fn_R"]) / max(fns, 1e-6)
        if t_deg is None and (imb > 0.55 or (r["nL"] == 0) ^ (r["nR"] == 0)):
            t_deg = r["t"]
        if (r["nL"] == 0) ^ (r["nR"] == 0):
            uni_acc += dt
            if t_uni is None:
                t_uni = r["t"]
        else:
            uni_acc = 0.0
        if r["nL"] == 0 and r["nR"] == 0:
            both_acc += dt
            if t_both is None:
                t_both = r["t"]
        else:
            both_acc = 0.0
        if t_drop is None and (
            r["obj_z"] < TABLE_DROP or r["clear"] < -0.005
        ):
            # only after having been off-table
            if t_first_air is not None and r["t"] > t_first_air + 0.05:
                t_drop = r["t"]

    t_fail = None
    fail_kind = None
    if t_capture_ok is not None:
        # recompute sustained
        uni_acc = both_acc = 0.0
        prev_t = None
        t_uni_sus = t_both_sus = None
        for r in log:
            if prev_t is None:
                prev_t = r["t"]
                continue
            dt = max(0.0, r["t"] - prev_t)
            prev_t = r["t"]
            if not after_cap(r):
                continue
            if (r["nL"] == 0) ^ (r["nR"] == 0):
                uni_acc += dt
                if uni_acc >= UNI_HOLD and t_uni_sus is None:
                    t_uni_sus = r["t"]
            else:
                uni_acc = 0.0
            if r["nL"] == 0 and r["nR"] == 0:
                both_acc += dt
                if both_acc >= BOTH_HOLD and t_both_sus is None:
                    t_both_sus = r["t"]
            else:
                both_acc = 0.0
        cands = []
        if t_uni_sus is not None:
            cands.append(("sustained_unilateral", t_uni_sus))
        if t_both_sus is not None:
            cands.append(("both_contacts_lost", t_both_sus))
        if t_drop is not None:
            cands.append(("drop", t_drop))
        if cands:
            fail_kind, t_fail = min(cands, key=lambda x: x[1])

    max_ex = max_ey = max_tilt = 0.0
    after_rows = [r for r in log if after_cap(r)] if t_capture_ok else []
    src = after_rows or log
    if src:
        max_ex = max(abs(r["e_x"]) for r in src)
        max_ey = max(abs(r["e_y"]) for r in src)
        max_tilt = max(r["tilt_deg"] for r in src)

    grew = False
    if cap_row is not None and after_rows:
        grew = (
            max(abs(r["e_x"] - cap_row["e_x"]) for r in after_rows) > SLIP_E
            or max(abs(r["e_y"] - cap_row["e_y"]) for r in after_rows) > SLIP_E
            or max(r["tilt_deg"] for r in after_rows) - cap_row["tilt_deg"] > SLIP_TILT
            or max(r["v_rel"] for r in after_rows) > SLIP_V
        )

    if t_capture_ok is None:
        cls = "FAILED_CAPTURE"
    elif t_fail is not None:
        cls = "POST_GRASP_FAILURE"
    elif fsm_success and grew:
        cls = "DETERIORATES_BUT_SURVIVES"
    elif fsm_success:
        cls = "STABLE_SUCCESS"
    elif grew:
        cls = "DETERIORATES_BUT_SURVIVES"
    else:
        cls = "FAILED_CAPTURE" if t_first_air is None else "DETERIORATES_BUT_SURVIVES"

    first_axis = None
    if t_slip is not None and cap_row is not None:
        r = next((x for x in log if abs(x["t"] - t_slip) < 1e-4 or x["t"] >= t_slip), None)
        if r is not None:
            scores = {
                "e_x": abs(r["e_x"] - cap_row["e_x"]),
                "e_y": abs(r["e_y"] - cap_row["e_y"]),
                "e_z": abs(r["e_z"] - cap_row["e_z"]),
                "tilt_deg": max(0.0, (r["tilt_deg"] - cap_row["tilt_deg"]) / 180.0),
                "v_rel": r["v_rel"],
            }
            first_axis = max(scores, key=scores.get)

    return {
        "t_close_start": t_close,
        "t_first_bilateral": t_bil,
        "t_lift_start": t_lift,
        "t_first_airborne": t_first_air,
        "t_capture_ok": t_capture_ok,
        "t_first_meaningful_relative_slip": t_slip,
        "t_first_contact_degradation": t_deg,
        "t_first_unilateral": t_uni,
        "t_both_contacts_lost": t_both,
        "t_drop": t_drop,
        "t_failure": t_fail,
        "fail_kind_physical": fail_kind,
        "capture_successful": t_capture_ok is not None,
        "airborne_bilateral_duration": airborne_bil_dur,
        "max_abs_e_x": max_ex,
        "max_abs_e_y": max_ey,
        "max_tilt_deg": max_tilt,
        "unilateral": t_uni is not None,
        "both_lost": t_both is not None,
        "drop": t_drop is not None,
        "classification": cls,
        "first_growing_component": first_axis,
        "capture_hold_s": CAPTURE_HOLD,
        "capture_hold_is_discovery_criterion": True,
    }


def precursor_windows(log: list[dict], events: dict) -> dict:
    t_cap = events.get("t_capture_ok")
    t_fail = events.get("t_failure")
    if t_cap is None or t_fail is None:
        return {}
    cap = next((r for r in log if r["t"] >= t_cap - 1e-9), None)
    if cap is None:
        return {}
    t_cop = t_fn = t_ap = t_rel = None
    for r in log:
        if r["t"] < t_cap or r["t"] > t_fail:
            continue
        if t_rel is None and (
            abs(r["e_x"] - cap["e_x"]) > SLIP_E
            or abs(r["e_y"] - cap["e_y"]) > SLIP_E
            or r["v_rel"] > SLIP_V
        ):
            t_rel = r["t"]
        fns = r["Fn_L"] + r["Fn_R"]
        if t_fn is None and abs(r["Fn_L"] - r["Fn_R"]) / max(fns, 1e-6) > 0.45:
            t_fn = r["t"]
        if t_ap is None and abs(r["aperture"] - cap["aperture"]) > 0.002:
            t_ap = r["t"]
        if t_cop is None and cap.get("cop_valid") and r.get("cop_valid"):
            du = abs(r["u_L"] - cap["u_L"]) if np.isfinite(r["u_L"]) and np.isfinite(cap["u_L"]) else 0.0
            dv = abs(r["u_R"] - cap["u_R"]) if np.isfinite(r["u_R"]) and np.isfinite(cap["u_R"]) else 0.0
            if max(du, dv) > 0.002:
                t_cop = r["t"]
    out = {}
    for name, t0 in (
        ("tactile_CoP_motion", t_cop),
        ("Fn_imbalance", t_fn),
        ("aperture_change", t_ap),
        ("GT_relative_motion", t_rel),
    ):
        if t0 is None:
            out[name] = {"t_precursor": None, "window_s": None}
        else:
            out[name] = {"t_precursor": t0, "window_s": float(t_fail - t0)}
    return out


def event_snapshots(log: list[dict], events: dict) -> dict:
    keys = (
        "t_first_bilateral",
        "t_lift_start",
        "t_first_airborne",
        "t_first_meaningful_relative_slip",
        "t_first_unilateral",
        "t_both_contacts_lost",
        "t_drop",
        "t_capture_ok",
        "t_failure",
    )
    out = {}
    for k in keys:
        tv = events.get(k)
        if tv is None:
            out[k] = None
            continue
        r = next((x for x in log if x["t"] >= float(tv) - 1e-9), None)
        if r is None:
            out[k] = None
            continue
        out[k] = {
            "t": r["t"],
            "phase": r["phase"],
            "e_hand": [r["e_x"], r["e_y"], r["e_z"]],
            "v_rel_hand": r["v_rel_hand"],
            "omega_rel_hand": r["omega_rel_hand"],
            "Fn_L": r["Fn_L"],
            "Fn_R": r["Fn_R"],
            "Ft_L": r.get("Ft_L"),
            "Ft_R": r.get("Ft_R"),
            "tilt": r["tilt_deg"],
            "nL": r["nL"],
            "nR": r["nR"],
            "obj_z": r["obj_z"],
        }
    return out


def run_episode(
    mass: float,
    mu: float,
    *,
    interactive: bool = False,
    playback: float = 0.5,
    keep_log: bool = True,
) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = GraspSim(cfg)
    sim.reset(mass, mu, np.zeros(3))
    apply_pair_mu(sim, mu)
    dt = float(sim.model.opt.timestep)
    log = []
    last_log = -1e9
    n_max = int(round(12.0 / dt))
    audit_early = friction_mass_audit(sim, mass, mu)
    viewer = None
    viewer_cm = None
    pacer = None
    if interactive:
        import mujoco.viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = viewer_cm.__enter__()
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        pacer = RealtimePacer(playback, RENDER_HZ, dt)
        pacer.start(0.0)
        pacer.note_sync(0.0)
        viewer.sync()

    t_drop_clock = None
    t_success_clock = None
    mu_samples = []
    last_mu_meas = None
    for _ in range(n_max):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        o = physical_pack(sim)
        _, tac = read_tactile(sim.model, sim.data, sim.ids)
        t = float(sim.data.time)
        phase = str(sim.fsm.phase)
        if keep_log and (t - last_log >= LOG_DT or not log):
            row = _row(sim, o, tac, phase)
            log.append(row)
            last_log = t
            if row.get("measured_contact_mu") is not None:
                mu_samples.append(float(row["measured_contact_mu"]))
                last_mu_meas = float(row["measured_contact_mu"])
        if t_drop_clock is None and phase == "lift" and sim.dropped():
            t_drop_clock = t
        if t_success_clock is None and sim.fsm.success:
            t_success_clock = t
        if viewer is not None and pacer is not None and pacer.should_render(t):
            _overlay(
                viewer,
                mass,
                last_mu_meas if last_mu_meas is not None else mu,
                phase,
                t,
                o["e_x"],
                float(o["rh"][1]),
            )
            viewer.sync()
            pacer.note_sync(t)
            pacer.wait_if_ahead(t)
            if hasattr(viewer, "is_running") and not viewer.is_running():
                break
        if t_drop_clock is not None and t >= t_drop_clock + 1.5:
            break
        if t_success_clock is not None and t >= t_success_clock + 0.50:
            break
        if t >= 12.0:
            break

    if keep_log and log and log[-1]["t"] < float(sim.data.time) - 0.002:
        o = physical_pack(sim)
        _, tac = read_tactile(sim.model, sim.data, sim.ids)
        log.append(_row(sim, o, tac, str(sim.fsm.phase)))

    audit_late = friction_mass_audit(sim, mass, mu)
    events = classify_and_events(log, bool(sim.fsm.success))
    # merge live drop from GraspSim if classification missed table-stay
    if t_drop_clock is not None and events["t_drop"] is None and events["capture_successful"]:
        events["t_drop"] = t_drop_clock
        if events["t_failure"] is None or t_drop_clock < events["t_failure"]:
            events["t_failure"] = t_drop_clock
            events["fail_kind_physical"] = "drop"
            events["classification"] = "POST_GRASP_FAILURE"
            events["drop"] = True
    windows = precursor_windows(log, events)
    meas = float(np.median(mu_samples)) if mu_samples else None
    invalid = meas is None or abs(meas - float(mu)) > MU_MATCH_TOL
    snaps = event_snapshots(log, events)
    oF = physical_pack(sim)
    out = {
        "ok": True,
        "case_id": case_id(mass, mu),
        "mass": mass,
        "requested_mu": mu,
        "measured_contact_mu": meas,
        "mu_match_ok": (not invalid) and meas is not None,
        "invalid_mu_mismatch": bool(invalid),
        "effective_mu": meas,
        "pair_sliding_applied": True,
        "friction_mass_audit": {"at_reset": audit_early, "at_end": audit_late},
        "fsm_success": bool(sim.fsm.success),
        "final_e_x": float(log[-1]["e_x"]) if log else float(oF["e_x"]),
        "final_e_y": float(log[-1]["e_y"]) if log else float(np.asarray(oF["rh"], float)[1]),
        "final_obj_z": float(log[-1]["obj_z"]) if log else float(oF["obj_z"]),
        "final_nL": int(log[-1]["nL"]) if log else int(oF["nL"]),
        "final_nR": int(log[-1]["nR"]) if log else int(oF["nR"]),
        "used_apply_rel_pose": False,
        "used_grasp_offset": [0.0, 0.0, 0.0],
        "used_RULE": False,
        "used_SAC": False,
        **events,
        "event_snapshots": snaps,
        "precursor_windows": windows,
        "log": log if keep_log else [],
    }
    if viewer_cm is not None:
        viewer_cm.__exit__(None, None, None)
    return out


def plot_case(r: dict, path: Path) -> None:
    log = r.get("log") or []
    if not log:
        return
    t = np.array([x["t"] for x in log])
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    ax = axes.ravel()
    ax[0].plot(t, 1e3 * np.array([x["e_x"] for x in log]))
    ax[0].set_ylabel("e_x mm")
    ax[1].plot(t, 1e3 * np.array([x["e_y"] for x in log]))
    ax[1].set_ylabel("e_y mm")
    ax[2].plot(t, [x["tilt_deg"] for x in log])
    ax[2].set_ylabel("tilt deg")
    ax[3].plot(t, [x["v_rel"] for x in log])
    ax[3].set_ylabel("|v_rel|")
    ax[4].plot(t, [x["w_rel"] for x in log])
    ax[4].set_ylabel("|omega_rel|")
    ax[5].plot(t, [x["Fn_L"] for x in log], label="Fn_L")
    ax[5].plot(t, [x["Fn_R"] for x in log], label="Fn_R")
    ax[5].legend()
    ax[5].set_ylabel("Fn")
    ax[6].plot(t, [x["nL"] for x in log], label="nL")
    ax[6].plot(t, [x["nR"] for x in log], label="nR")
    ax[6].legend()
    ax[6].set_ylabel("contacts")
    ax[7].plot(t, [x["obj_z"] for x in log], label="obj_z")
    ax[7].plot(t, [x["clear"] for x in log], label="clear")
    ax[7].legend()
    marks = [
        ("t_first_airborne", "air"),
        ("t_capture_ok", "cap"),
        ("t_first_meaningful_relative_slip", "slip"),
        ("t_first_unilateral", "uni"),
        ("t_both_contacts_lost", "both"),
        ("t_drop", "drop"),
        ("t_failure", "fail"),
    ]
    for a in ax:
        for k, lab in marks:
            tv = r.get(k)
            if tv is not None:
                a.axvline(float(tv), ls="--", lw=0.8, alpha=0.7)
        a.grid(True, alpha=0.3)
    fig.suptitle(f"{r['case_id']}  {r['classification']}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def slim(r: dict) -> dict:
    skip = {"log", "friction_mass_audit"}
    d = {k: v for k, v in r.items() if k not in skip}
    ar = r.get("friction_mass_audit", {}).get("at_reset", {})
    ae = r.get("friction_mass_audit", {}).get("at_end", {})
    d["friction_mass_audit"] = {
        "at_reset": {
            k: ar[k]
            for k in (
                "requested_mass",
                "requested_mu",
                "body_mass_object",
                "body_subtreemass_object",
                "object_geom_friction",
                "mj_setConst_called",
                "table_geom_friction0_unchanged",
                "effective_mu_from_contacts",
            )
            if k in ar
        },
        "at_end_effective_mu": ae.get("effective_mu_from_contacts"),
        "at_end_contact_samples": (ae.get("live_contacts") or [])[:4],
    }
    return d


def run_sweep() -> list[dict]:
    rows = []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for m in MASSES:
        for mu in MUS:
            print(f"running {case_id(m, mu)} ...", flush=True)
            r = run_episode(m, mu, interactive=False, keep_log=True)
            rows.append(r)
            cid = r["case_id"]
            (LOG_DIR / f"{cid}.json").write_text(
                json.dumps({"summary": slim(r), "log": r["log"]}, indent=2, default=str),
                encoding="utf-8",
            )
            if r["classification"] in ("POST_GRASP_FAILURE", "DETERIORATES_BUT_SURVIVES"):
                plot_case(r, FIG_DIR / f"{cid}.png")
    diffs = trajectory_diffs_fixed_mass(rows)
    table = [slim(r) for r in rows]
    (LOG_DIR / "sweep.json").write_text(
        json.dumps({"table": table, "fixed_mass_trajectory_diffs": diffs}, indent=2, default=str),
        encoding="utf-8",
    )
    if not any(r["classification"] == "POST_GRASP_FAILURE" for r in rows):
        for r in rows:
            if r["classification"] == "STABLE_SUCCESS":
                plot_case(r, FIG_DIR / f"{r['case_id']}_stable.png")
                break
    return rows


def trajectory_diffs_fixed_mass(rows: list[dict]) -> dict:
    """Compare mu=1.0 vs 0.7 vs 0.4 at each mass using overlapping log times."""
    out = {}
    for m in MASSES:
        group = {float(r["requested_mu"]): r for r in rows if abs(r["mass"] - m) < 1e-9}
        if set(group) != {1.0, 0.7, 0.4}:
            continue
        def series(mu, key):
            log = group[mu]["log"]
            return np.array([float(x[key]) if x.get(key) is not None else np.nan for x in log])

        rec = {}
        for a, b in ((1.0, 0.7), (1.0, 0.4), (0.7, 0.4)):
            n = min(len(group[a]["log"]), len(group[b]["log"]))
            rec[f"{a}_vs_{b}"] = {
                "max_abs_d_obj_z": float(np.nanmax(np.abs(series(a, "obj_z")[:n] - series(b, "obj_z")[:n]))),
                "max_abs_d_e_x": float(np.nanmax(np.abs(series(a, "e_x")[:n] - series(b, "e_x")[:n]))),
                "max_abs_d_tilt": float(np.nanmax(np.abs(series(a, "tilt_deg")[:n] - series(b, "tilt_deg")[:n]))),
                "max_abs_d_v_rel": float(np.nanmax(np.abs(series(a, "v_rel")[:n] - series(b, "v_rel")[:n]))),
                "max_abs_d_Ft": float(
                    np.nanmax(
                        np.abs(
                            (series(a, "Ft_L") + series(a, "Ft_R"))[:n]
                            - (series(b, "Ft_L") + series(b, "Ft_R"))[:n]
                        )
                    )
                ),
                "class_a": group[a]["classification"],
                "class_b": group[b]["classification"],
            }
        out[f"m={m}"] = rec
    return out


def verify_live_pair_friction() -> dict:
    """Establish bilateral contact and print live contact.friction[0] vs requested mu."""
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    report = {"mujoco_version": getattr(mujoco, "__version__", "unknown"), "rows": [], "pass": True}
    for mu in MUS:
        sim = GraspSim(cfg)
        sim.reset(0.08, mu, np.zeros(3))
        pair_apply = apply_pair_mu(sim, mu)
        got = None
        summary = None
        n_max = int(round(8.0 / float(sim.model.opt.timestep)))
        for _ in range(n_max):
            sim.physics_step(None, in_recovery=False)
            cons = live_finger_object_contacts(sim)
            sides = {c["side"] for c in cons}
            if "L" in sides and "R" in sides:
                summary = summarize_lr_contacts(cons, mu)
                got = True
                break
        mass_ok = abs(float(sim.model.body_mass[sim.ids.object_body]) - 0.08) < 1e-9
        lmu = summary["left_median_contact_mu"] if summary else None
        rmu = summary["right_median_contact_mu"] if summary else None
        ok = (
            got
            and lmu is not None
            and rmu is not None
            and abs(lmu - mu) <= MU_MATCH_TOL
            and abs(rmu - mu) <= MU_MATCH_TOL
        )
        if not ok:
            report["pass"] = False
        report["rows"].append(
            {
                "requested_mu": mu,
                "pair_apply": {
                    "n_geoms": pair_apply["n_geoms"],
                    "object_geom": pair_apply["object_geom"],
                    "n_left": len(pair_apply["left_geoms"]),
                    "n_right": len(pair_apply["right_geoms"]),
                },
                "body_mass": float(sim.model.body_mass[sim.ids.object_body]),
                "subtree_mass": float(sim.model.body_subtreemass[sim.ids.object_body]),
                "mass_ok": mass_ok,
                "got_bilateral": bool(got),
                "summary": summary,
                "match_ok": bool(ok),
                "object_friction_vec": np.array(sim.model.geom_friction[sim.ids.object_geom], float).tolist(),
            }
        )
    return report


def friction_regression_test() -> dict:
    """Same nominal FSM, three mus. Look for physical difference without extra forces."""
    stats = []
    for mu in MUS:
        r = run_episode(0.08, mu, interactive=False, keep_log=True)
        lift = [x for x in r["log"] if x["phase"] in ("close", "lift")]
        ft = np.array([x.get("Ft_L", 0.0) + x.get("Ft_R", 0.0) for x in lift], float)
        vr = np.array([x["v_rel"] for x in lift], float)
        ex = np.array([x["e_x"] for x in lift], float)
        util = np.array(
            [x["friction_utilization"] if x.get("friction_utilization") is not None else np.nan for x in lift],
            float,
        )
        stats.append(
            {
                "mu": mu,
                "measured_contact_mu": r.get("measured_contact_mu"),
                "classification": r["classification"],
                "mean_Ft": float(np.mean(ft)) if ft.size else None,
                "max_Ft": float(np.max(ft)) if ft.size else None,
                "max_v_rel": float(np.max(vr)) if vr.size else None,
                "max_abs_e_x": float(np.max(np.abs(ex))) if ex.size else None,
                "mean_util": float(np.nanmean(util)) if util.size else None,
                "n": len(lift),
            }
        )
    diffs = {}
    if len(stats) == 3:
        diffs["d_mean_Ft_1_vs_0p4"] = abs((stats[0]["mean_Ft"] or 0) - (stats[2]["mean_Ft"] or 0))
        diffs["d_max_v_rel_1_vs_0p4"] = abs((stats[0]["max_v_rel"] or 0) - (stats[2]["max_v_rel"] or 0))
        diffs["d_max_e_x_1_vs_0p4"] = abs((stats[0]["max_abs_e_x"] or 0) - (stats[2]["max_abs_e_x"] or 0))
        diffs["d_mean_util_1_vs_0p4"] = abs((stats[0]["mean_util"] or 0) - (stats[2]["mean_util"] or np.nan))
        # Distinguish if any physically relevant channel moves
        diffs["distinguished"] = bool(
            diffs["d_mean_Ft_1_vs_0p4"] > 0.05
            or diffs["d_max_v_rel_1_vs_0p4"] > 5e-4
            or diffs["d_max_e_x_1_vs_0p4"] > 5e-5
            or (np.isfinite(diffs["d_mean_util_1_vs_0p4"]) and diffs["d_mean_util_1_vs_0p4"] > 0.02)
        )
    return {"stats": stats, "diffs": diffs}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Natural post-grasp deterioration discovery")
    p.add_argument("--verify-friction", action="store_true")
    p.add_argument("--friction-regression", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--case", type=str, default=None)
    p.add_argument("--mass", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--playback-speed", type=float, default=0.5)
    p.add_argument("--playback", type=float, default=None, help="alias of --playback-speed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    playback = float(args.playback if args.playback is not None else args.playback_speed)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify_friction or args.sweep:
        ver = verify_live_pair_friction()
        (LOG_DIR / "friction_verify.json").write_text(json.dumps(ver, indent=2, default=str), encoding="utf-8")
        print(json.dumps(ver, indent=2, default=str))
        if not ver["pass"]:
            print("FRICTION VERIFY FAILED: live pair mu does not match requested. STOP. No sweep.")
            return 2
        print("FRICTION VERIFY PASS")
        if args.verify_friction and not args.sweep:
            return 0
    if args.friction_regression:
        reg = friction_regression_test()
        (LOG_DIR / "friction_regression.json").write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")
        print(json.dumps(reg, indent=2, default=str))
        if not args.sweep:
            return 0
    if args.sweep:
        rows = run_sweep()
        print(json.dumps([slim(r) for r in rows], indent=2, default=str))
        print("INVALID mu mismatches:", sum(bool(r.get("invalid_mu_mismatch")) for r in rows))
        print("POST_GRASP_FAILURE count:", sum(r["classification"] == "POST_GRASP_FAILURE" for r in rows))
        print("wrote", LOG_DIR)
        return 0
    mass = args.mass
    mu = args.mu
    if args.case:
        mass, mu = parse_case(args.case)
    if mass is None or mu is None:
        print("provide --verify-friction, --friction-regression, --sweep, or --mass/--mu (or --case)")
        return 2
    r = run_episode(float(mass), float(mu), interactive=not args.headless, playback=playback, keep_log=True)
    cid = r["case_id"]
    (LOG_DIR / f"{cid}_view.json").write_text(
        json.dumps({"summary": slim(r), "log": r["log"]}, indent=2, default=str),
        encoding="utf-8",
    )
    if r["classification"] in ("POST_GRASP_FAILURE", "DETERIORATES_BUT_SURVIVES"):
        plot_case(r, FIG_DIR / f"{cid}.png")
    print(json.dumps(slim(r), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
