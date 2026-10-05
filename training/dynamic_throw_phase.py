"""Offset-assisted throw with OPEN-phase control. Imported by demo_dynamic_recatch."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import physical_pack
from training.demo_airborne_recovery import (
    apply_masks,
    copy_masks,
    disable_object_table_only,
    prepare_high_parent,
)
from training.demo_airborne_recapture import make_parent_sim
from training.demo_ballistic_impact import make_sim as make_ballistic_sim
from training.demo_ballistic_recovery import restore_ballistic
from training.demo_dynamic_recatch import (
    BOTH_OFF_MIN,
    EARLY_DOC_RHX_MM,
    EARLY_PKL,
    HOLD_S,
    MATCH_DEG,
    OMEGA,
    OPEN_PHASES,
    OUT,
    RAW,
    ROOT,
    TAU_OPEN,
    TAU_SEC,
    V_UP_MIN,
    dump,
    measure,
    restore,
    save_npz,
    slim_row,
    step,
)
from training.demo_teleport_recovery_state import signed_rot_about_y_deg
from training.replay_core import freeze


def nearest_angle_row(log, deg: float):
    if not log:
        return None
    return min(log, key=lambda r: abs(float(r["ang_deg"]) - float(deg)))


def load_early_snap() -> dict:
    if not EARLY_PKL.is_file():
        raise FileNotFoundError(f"missing CENTER-6 EARLY snapshot {EARLY_PKL}")
    import pickle

    with EARLY_PKL.open("rb") as f:
        return pickle.load(f)


def audit_early_state(sim) -> dict:
    o = physical_pack(sim)
    m = measure(sim, tau=TAU_SEC, label="EARLY")
    return {
        "path": str(EARLY_PKL),
        "t": float(sim.data.time),
        "rh": np.asarray(o["rh"], float).tolist(),
        "rh_mm": (1e3 * np.asarray(o["rh"], float)).tolist(),
        "R_rel": np.asarray(o["R_rel"], float).tolist(),
        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "aperture": float(m["aperture"]),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "phase": str(sim.fsm.phase),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "obj_z": float(o["obj_z"]),
        "doc_rhx_mm": EARLY_DOC_RHX_MM,
        "rhx_mm_match_doc": abs(1e3 * float(o["rh"][0]) - EARLY_DOC_RHX_MM) < 0.15,
    }


def restore_offset(sim, snap) -> None:
    restore_ballistic(sim, snap)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)
    disable_object_table_only(sim)


def constrained_swing(sim, gains, restore_fn, ctl=None, max_deg=125.0):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    log = [measure(sim, omega_y=OMEGA, tau=TAU_SEC, label="SWING", Rh0=Rh0)]
    ev = {"ROTATE_START": float(sim.data.time)}
    timeout = float(sim.data.time) + np.deg2rad(max_deg) / OMEGA + 1.2
    best = None
    while float(sim.data.time) < timeout:
        m, stop = step(sim, gains, OMEGA, 0.0, 0.0, TAU_SEC, log, Rh0, "SWING", ev, ctl)
        if stop:
            break
        if m["nL"] > 0 and m["nR"] > 0:
            if best is None or float(m["v_obj"][2]) > float(best["v_obj"][2]):
                best = slim_row(m)
        if abs(float(m["ang_deg"])) >= max_deg - 0.8:
            break
        if m["nL"] == 0 and m["nR"] == 0:
            ev["BOTH_OFF_DURING_SECURE_SWING"] = m["t"]
            break
    marks = {}
    for d in MATCH_DEG:
        r = nearest_angle_row(log, d)
        marks[str(int(d))] = None if r is None else slim_row(r)
    return {"log": log, "marks": marks, "best_supported_vz": best, "events": ev}


def ballistic_after_release(sim, gains, log, Rh0, ev, ctl=None, dur=0.80):
    t0 = float(sim.data.time)
    z0 = float(sim.data.xpos[sim.ids.object_body][2])
    z_max = z0
    t_apex = None
    sep_max = 0.0
    m = log[-1]
    while float(sim.data.time) < t0 + dur:
        m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, Rh0, "BALLISTIC", ev, ctl)
        if stop:
            break
        z = float(m["obj_z"])
        vz = float(m["v_obj"][2])
        sep_max = max(sep_max, float(np.linalg.norm(np.array(m["p_obj"]) - np.array(m["p_hand"]))))
        if z >= z_max:
            z_max = z
            t_apex = m["t"]
        if vz <= 0.0 and "BALLISTIC_APEX" not in ev and t_apex is not None:
            ev["BALLISTIC_APEX"] = t_apex
        if (m["nL"] > 0 or m["nR"] > 0) and "RECONTACT_DURING_BALLISTIC" not in ev:
            ev["RECONTACT_DURING_BALLISTIC"] = m["t"]
    if "BALLISTIC_APEX" not in ev and t_apex is not None:
        ev["BALLISTIC_APEX"] = t_apex
    rh = np.asarray(m["rh"], float)
    return {
        "z0": z0,
        "z_max": z_max,
        "up_disp": float(z_max - z0),
        "t_apex": ev.get("BALLISTIC_APEX"),
        "dt_apex": None if ev.get("BALLISTIC_APEX") is None else float(ev["BALLISTIC_APEX"] - t0),
        "sep_max": sep_max,
        "end_rh": rh.tolist(),
        "end_rhx_mm": 1e3 * float(rh[0]),
        "end_rhz": float(rh[2]),
        "corridor_end": bool(abs(rh[0]) < 0.020 and abs(rh[2] - 0.099) < 0.05),
        "end": slim_row(m),
    }


def recatch_fixed_orientation(sim, gains, log, Rh0, ev, ctl=None):
    ev["RECATCH_PREP"] = float(sim.data.time)
    ev["CLOSE_START"] = float(sim.data.time)
    t0 = float(sim.data.time)
    captured = False
    m = log[-1]
    while float(sim.data.time) < t0 + 0.45:
        m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_SEC, log, Rh0, "CLOSE", ev, ctl)
        if stop:
            break
        if (m["nL"] > 0 or m["nR"] > 0) and "FIRST_RECONTACT" not in ev:
            ev["FIRST_RECONTACT"] = m["t"]
        if m["nL"] > 0 and m["nR"] > 0:
            if "FIRST_BILATERAL" not in ev:
                ev["FIRST_BILATERAL"] = m["t"]
            elif m["t"] - ev["FIRST_BILATERAL"] >= 0.050:
                ev["MOTION_ARREST"] = m["t"]
                captured = True
                break
    if not captured:
        return False, m
    ev["HOLD_START"] = float(sim.data.time)
    lost = 0.0
    dt = float(sim.model.opt.timestep)
    t_hold = float(sim.data.time) + HOLD_S
    while float(sim.data.time) < t_hold:
        m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_SEC, log, Rh0, "HOLD", ev, ctl)
        if stop:
            return False, m
        if m["nL"] == 0 or m["nR"] == 0:
            lost += dt
            if lost >= 0.08:
                ev["HOLD_LOST"] = m["t"]
                return False, m
        else:
            lost = 0.0
    return True, m


def phase_open_trial(sim, gains, restore_fn, theta_open_cmd: float, name: str, ctl=None, recatch=False):
    restore_fn()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    m0 = measure(sim, omega_y=OMEGA, tau=TAU_SEC, label="PARENT", Rh0=Rh0)
    log = [m0]
    ev = {"ROTATE_START": float(sim.data.time), "theta_open_cmd": float(theta_open_cmd)}
    opened = False
    last_side = m0.get("contact_topology")
    dt = float(sim.model.opt.timestep)
    both = 0.0
    timeout = float(sim.data.time) + 1.60
    m = m0
    while float(sim.data.time) < timeout:
        ang = signed_rot_about_y_deg(
            Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        )
        if (not opened) and abs(ang) + 0.4 >= float(theta_open_cmd):
            opened = True
            ev["OPEN_COMMAND"] = float(sim.data.time)
            ev["theta_open_cmd_actual"] = float(ang)
        tau = TAU_OPEN if opened else TAU_SEC
        m, stop = step(
            sim, gains, OMEGA, 0.0, 0.0, tau, log, Rh0, "OPEN" if opened else "ROTATE", ev, ctl
        )
        if stop:
            break
        topo = m.get("contact_topology")
        if topo in ("L", "R"):
            last_side = topo
            if opened and "FIRST_SINGLE_CONTACT" not in ev:
                ev["FIRST_SINGLE_CONTACT"] = m["t"]
                ev["theta_first_single_contact"] = float(m["ang_deg"])
                ev["first_single_side"] = topo
        if m["nL"] == 0 and m["nR"] == 0:
            if "FIRST_BOTH_OFF" not in ev:
                ev["FIRST_BOTH_OFF"] = m["t"]
                ev["theta_first_both_off"] = float(m["ang_deg"])
                ev["both_off_state"] = slim_row(m)
                ev["last_contact_identity"] = last_side
            both += dt
            if both >= BOTH_OFF_MIN:
                break
        else:
            both = 0.0
    ball = None
    captured = False
    if "FIRST_BOTH_OFF" in ev:
        if recatch:
            t_wait = float(sim.data.time) + 0.040
            while float(sim.data.time) < t_wait:
                m, stop = step(sim, gains, 0.0, 0.0, 0.0, TAU_OPEN, log, Rh0, "BALLISTIC", ev, ctl)
                if stop:
                    break
                if float(m["v_obj"][2]) <= 0.0 and "BALLISTIC_APEX" not in ev:
                    ev["BALLISTIC_APEX"] = m["t"]
                    break
            captured, m = recatch_fixed_orientation(sim, gains, log, Rh0, ev, ctl=ctl)
        else:
            ball = ballistic_after_release(sim, gains, log, Rh0, ev, ctl=ctl)
    lat = None
    if "OPEN_COMMAND" in ev and "FIRST_BOTH_OFF" in ev:
        lat = float(ev["FIRST_BOTH_OFF"] - ev["OPEN_COMMAND"])
    boths = ev.get("both_off_state")
    vz = None if boths is None else float(boths["v_obj"][2])
    return {
        "name": name,
        "theta_open_cmd": float(theta_open_cmd),
        "theta_open_cmd_actual": ev.get("theta_open_cmd_actual"),
        "theta_first_single_contact": ev.get("theta_first_single_contact"),
        "theta_first_both_off": ev.get("theta_first_both_off"),
        "open_latency_s": lat,
        "initial_rh": m0["rh"],
        "both_off": boths,
        "v_obj_z_both_off": vz,
        "upward": bool(vz is not None and vz > V_UP_MIN),
        "last_contact_identity": ev.get("last_contact_identity"),
        "ballistic": ball,
        "captured": captured,
        "ok_hold": bool(
            captured
            and ev.get("HOLD_START")
            and "HOLD_LOST" not in ev
            and "FLOOR" not in ev
        ),
        "events": {k: v for k, v in ev.items() if k != "both_off_state"},
        "log": log,
        "end": slim_row(log[-1]) if log else None,
    }


def slim_trial(tr: dict) -> dict:
    return {k: v for k, v in tr.items() if k != "log"}


def append_offset_section(early, swings, trials, recatch_row, verdict: str) -> None:
    p = OUT / "DYNAMIC_THROW_RECAPTURE.md"
    prev = p.read_text(encoding="utf-8") if p.is_file() else ""
    if "OFFSET-ASSISTED RELEASE-PHASE AUDIT" in prev:
        prev = prev.split("# OFFSET-ASSISTED RELEASE-PHASE AUDIT")[0].rstrip()
    a = []
    A = a.append
    A("")
    A("---")
    A("")
    A("# OFFSET-ASSISTED RELEASE-PHASE AUDIT")
    A("")
    A("Wrist keeps rotating during OPEN. Physical release = FIRST_BOTH_OFF, not OPEN_COMMAND. Same `omega_y=+3` and legal open for centered vs offset. No recatch tuning.")
    A("")
    A("## 1. Exact initial offset-state provenance")
    A("")
    A(f"File: `{early['path']}` (same pickle as the validated large-angle gravity-recovery EARLY state).")
    A(
        f"t={early['t']:.3f} s, r_h mm={np.round(early['rh_mm'], 2).tolist()}, "
        f"documented r_h.x={early['doc_rhx_mm']} mm, match={early['rhx_mm_match_doc']}."
    )
    A(f"nL/nR={early['nL']}/{early['nR']}, aperture={early['aperture']:.4f}, phase={early['phase']}.")
    A(f"v_rel_h={early['v_rel_h']}, w_rel_h={early['w_rel_h']}.")
    A(f"p_des={early['p_des']}, v_cmd={early['v_cmd']}, ctrl7={early['ctrl7']}.")
    A("No qpos/qvel edits. Real CENTER-6 r_h.x sign only.")
    A("")
    A("## 2. Centered vs offset constrained swing (tau=-18, no open)")
    A("")
    A("| deg | c v_z | o v_z | c rhx mm | o rhx mm | c lever | o lever | c rho | o rho | c n | o n |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    cm, om = swings["centered"]["marks"], swings["offset"]["marks"]
    for d in MATCH_DEG:
        k = str(int(d))
        c, o = cm.get(k), om.get(k)

        def cell(r, fn):
            return None if r is None else fn(r)

        A(
            f"| {int(d)} | {cell(c, lambda r: r['v_obj'][2])} | {cell(o, lambda r: r['v_obj'][2])} | "
            f"{cell(c, lambda r: 1e3*r['rh'][0])} | {cell(o, lambda r: 1e3*r['rh'][0])} | "
            f"{cell(c, lambda r: r.get('lever_arm'))} | {cell(o, lambda r: r.get('lever_arm'))} | "
            f"{cell(c, lambda r: r.get('rho_max'))} | {cell(o, lambda r: r.get('rho_max'))} | "
            f"{cell(c, lambda r: str(r['nL'])+'/'+str(r['nR']))} | {cell(o, lambda r: str(r['nL'])+'/'+str(r['nR']))} |"
        )
    A("")
    bc = swings["centered"].get("best_supported_vz") or {}
    bo = swings["offset"].get("best_supported_vz") or {}
    A(
        f"Peak bilateral v_obj,z: centered ang={bc.get('ang_deg')} vz={None if not bc else bc['v_obj'][2]}; "
        f"offset ang={bo.get('ang_deg')} vz={None if not bo else bo['v_obj'][2]}."
    )
    A("")
    A("## 3. OPEN_COMMAND -> FIRST_BOTH_OFF latency")
    A("")
    A("| case | theta_cmd | theta_cmd_act | theta_single | theta_BOTH_OFF | latency s |")
    A("|---|---|---|---|---|---|")
    for tr in trials:
        A(
            f"| {tr['name']} | {tr['theta_open_cmd']} | {tr.get('theta_open_cmd_actual')} | "
            f"{tr.get('theta_first_single_contact')} | {tr.get('theta_first_both_off')} | {tr.get('open_latency_s')} |"
        )
    A("")
    A("## 4-6. FIRST_BOTH_OFF velocity/pose and ballistic")
    A("")
    A("| case | init rhx mm | theta_off | v_obj,z | v_obj | |w_obj| | |w_hand| | rh_off mm | last | dt_apex | rise m | sep_max | corridor |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tr in trials:
        b = tr.get("both_off") or {}
        ball = tr.get("ballistic") or {}
        wo = np.asarray(b.get("w_obj") or [0, 0, 0], float)
        wh = np.asarray(b.get("w_hand") or [0, 0, 0], float)
        rh = b.get("rh")
        A(
            f"| {tr['name']} | {1e3*tr['initial_rh'][0]:.2f} | {tr.get('theta_first_both_off')} | "
            f"{tr.get('v_obj_z_both_off')} | {b.get('v_obj')} | {float(np.linalg.norm(wo)):.3f} | "
            f"{float(np.linalg.norm(wh)):.3f} | {None if rh is None else [round(1e3*x,2) for x in rh]} | "
            f"{tr.get('last_contact_identity')} | {ball.get('dt_apex')} | {ball.get('up_disp')} | "
            f"{ball.get('sep_max')} | {ball.get('corridor_end')} |"
        )
    A("")
    A("## 7. Late-open vs early-open (offset)")
    A("")
    early_t = next((t for t in trials if t["name"] == "offset_early"), None)
    late_t = next((t for t in trials if t["name"] == "offset_late"), None)
    if early_t and late_t:
        A(
            f"EARLY: cmd {early_t['theta_open_cmd']} deg -> BOTH_OFF {early_t.get('theta_first_both_off')} deg, "
            f"v_z={early_t.get('v_obj_z_both_off')}, latency={early_t.get('open_latency_s')} s."
        )
        A(
            f"LATE: cmd {late_t['theta_open_cmd']} deg -> BOTH_OFF {late_t.get('theta_first_both_off')} deg, "
            f"v_z={late_t.get('v_obj_z_both_off')}, latency={late_t.get('open_latency_s')} s."
        )
        ez, lz = early_t.get("v_obj_z_both_off"), late_t.get("v_obj_z_both_off")
        if ez is not None and lz is not None:
            A(f"Phase effect Delta v_z (early-late) = {ez-lz:+.4f} m/s.")
    A("")
    A("## 8. Privileged recatch")
    A("")
    if recatch_row is None:
        A("Not run.")
    else:
        A("ONE fixed-orientation hold attempt (no return-to-nominal).")
        A(f"ok_hold={recatch_row.get('ok_hold')} captured={recatch_row.get('captured')}")
        evs = recatch_row.get("events") or {}
        A(
            "Events: "
            + ", ".join(
                f"{k}={evs.get(k)}"
                for k in (
                    "OPEN_COMMAND",
                    "FIRST_BOTH_OFF",
                    "theta_first_both_off",
                    "CLOSE_START",
                    "FIRST_RECONTACT",
                    "FIRST_BILATERAL",
                    "MOTION_ARREST",
                    "HOLD_START",
                    "HOLD_LOST",
                )
                if k in evs
            )
        )
    A("")
    A("## 9. Limitations")
    A("")
    A(verdict)
    A("")
    A("Coarse early/medium/late only. Bounds unchanged. No synthetic offset. No MP4.")
    A("")
    A("```text")
    A("python training/demo_dynamic_recatch.py --mode centered_throw")
    A("python training/demo_dynamic_recatch.py --mode offset_late_open")
    A("python training/demo_dynamic_recatch.py --mode offset_early_open")
    A("python training/demo_dynamic_recatch.py --mode offset_throw_recatch")
    A("```")
    p.write_text(prev + "\n" + "\n".join(a) + "\n", encoding="utf-8")
    print("updated", p, flush=True)


def make_restorers():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    snap = load_early_snap()
    off_sim, _ = make_ballistic_sim()
    restore_offset(off_sim, snap)
    early = audit_early_state(off_sim)

    def restore_off():
        restore_offset(off_sim, snap)

    cen_sim = make_parent_sim()
    orig_masks = copy_masks(cen_sim)
    prep = prepare_high_parent(cen_sim, cfg, gains, orig_masks)
    if not prep.get("ok"):
        raise RuntimeError(prep)
    parent = prep["parent"]

    def restore_cen():
        apply_masks(cen_sim, orig_masks)
        restore(cen_sim, parent)
        disable_object_table_only(cen_sim)

    return {
        "cfg": cfg,
        "gains": gains,
        "early": early,
        "off_sim": off_sim,
        "cen_sim": cen_sim,
        "restore_off": restore_off,
        "restore_cen": restore_cen,
        "orig_masks": orig_masks,
        "snap": snap,
    }


def headless_phase_audit():
    RAW.mkdir(parents=True, exist_ok=True)
    env = make_restorers()
    gains = env["gains"]
    early = env["early"]
    dump(RAW / "early_state_audit.json", early)
    print("EARLY rhx_mm", early["rh_mm"][0], "match_doc", early["rhx_mm_match_doc"], flush=True)

    print("swing centered", flush=True)
    sw_c = constrained_swing(env["cen_sim"], gains, env["restore_cen"])
    print("swing offset", flush=True)
    sw_o = constrained_swing(env["off_sim"], gains, env["restore_off"])
    save_npz(RAW / "swing_centered.npz", sw_c["log"])
    save_npz(RAW / "swing_offset.npz", sw_o["log"])
    dump(
        RAW / "swing_marks.json",
        {
            "centered": sw_c["marks"],
            "offset": sw_o["marks"],
            "best_c": sw_c["best_supported_vz"],
            "best_o": sw_o["best_supported_vz"],
        },
    )

    trials = []
    for label, deg in OPEN_PHASES:
        print("centered", label, deg, flush=True)
        tr = phase_open_trial(env["cen_sim"], gains, env["restore_cen"], deg, f"centered_{label}")
        tr["phase_name"] = label
        dump(RAW / f"centered_{label}.json", slim_trial(tr))
        save_npz(RAW / f"centered_{label}.npz", tr["log"])
        print("  BOTH_OFF", tr.get("theta_first_both_off"), "vz", tr.get("v_obj_z_both_off"), "lat", tr.get("open_latency_s"), flush=True)
        trials.append(slim_trial(tr))
    for label, deg in OPEN_PHASES:
        print("offset", label, deg, flush=True)
        tr = phase_open_trial(env["off_sim"], gains, env["restore_off"], deg, f"offset_{label}")
        tr["phase_name"] = label
        dump(RAW / f"offset_{label}.json", slim_trial(tr))
        save_npz(RAW / f"offset_{label}.npz", tr["log"])
        print("  BOTH_OFF", tr.get("theta_first_both_off"), "vz", tr.get("v_obj_z_both_off"), "lat", tr.get("open_latency_s"), flush=True)
        trials.append(slim_trial(tr))

    oe = next(t for t in trials if t["name"] == "offset_early")
    ol = next(t for t in trials if t["name"] == "offset_late")
    ce = next(t for t in trials if t["name"] == "centered_early")
    offset_better = oe.get("upward") and oe.get("v_obj_z_both_off") is not None and (
        ce.get("v_obj_z_both_off") is None
        or float(oe["v_obj_z_both_off"]) > float(ce.get("v_obj_z_both_off") or -9) + 0.04
    )
    phase_matters = (
        oe.get("v_obj_z_both_off") is not None
        and ol.get("v_obj_z_both_off") is not None
        and abs(float(oe["v_obj_z_both_off"]) - float(ol["v_obj_z_both_off"])) > 0.04
    )
    recatch_row = None
    if oe.get("upward") and offset_better:
        print("ONE privileged recatch offset_early", flush=True)
        rec = phase_open_trial(env["off_sim"], gains, env["restore_off"], 15.0, "offset_early_recatch", recatch=True)
        recatch_row = slim_trial(rec)
        save_npz(RAW / "offset_early_recatch.npz", rec["log"])
        dump(RAW / "offset_early_recatch.json", recatch_row)
        print("recatch", recatch_row.get("ok_hold"), recatch_row.get("captured"), flush=True)

    dump(
        RAW / "phase_plan.json",
        {
            "early_open_deg": 15.0,
            "late_open_deg": 90.0,
            "omega_y": OMEGA,
            "offset_better": offset_better,
            "phase_matters": phase_matters,
            "recatch": bool(recatch_row and recatch_row.get("ok_hold")),
        },
    )
    verdict = (
        f"offset_early vz={oe.get('v_obj_z_both_off')} BOTH_OFF={oe.get('theta_first_both_off')} deg "
        f"(OPEN cmd 15); offset_late vz={ol.get('v_obj_z_both_off')} BOTH_OFF={ol.get('theta_first_both_off')} deg "
        f"(OPEN cmd 90); centered_early vz={ce.get('v_obj_z_both_off')} BOTH_OFF={ce.get('theta_first_both_off')} deg. "
        f"offset_better={offset_better} phase_matters={phase_matters} "
        f"recatch_ok={None if recatch_row is None else recatch_row.get('ok_hold')}."
    )
    print(verdict, flush=True)
    dump(RAW / "phase_trials.json", trials)
    append_offset_section(early, {"centered": sw_c, "offset": sw_o}, trials, recatch_row, verdict)


def run_viewer_trial(mode: str, env, gains, ctl):
    if mode == "centered_throw":
        return phase_open_trial(env["cen_sim"], gains, env["restore_cen"], 15.0, "centered_early", ctl=ctl)
    if mode == "offset_late_open":
        return phase_open_trial(env["off_sim"], gains, env["restore_off"], 90.0, "offset_late", ctl=ctl)
    if mode == "offset_early_open":
        return phase_open_trial(env["off_sim"], gains, env["restore_off"], 15.0, "offset_early", ctl=ctl)
    if mode == "offset_throw_recatch":
        return phase_open_trial(
            env["off_sim"], gains, env["restore_off"], 15.0, "offset_early_recatch", ctl=ctl, recatch=True
        )
    raise ValueError(mode)
