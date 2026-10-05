"""Disturbed vs nominal 10 s vertical hold; contact-softness and solver-accuracy sensitivity.

Diagnostic copies of the runtime MjModel only. Official XML is not written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.demo_teleport_recovery_state as demo
from controllers.jacobian_controller import gains_from_cfg, ori_error_deg
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.demo_teleport_recovery_state import (
    PAIR_MU,
    S_COUPLED,
    _row,
    advance_to_parent,
    load_or_measure_ref,
    make_parent_sim,
    pad_span_hand,
    run_teleport_case,
)
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "contact_creep_sensitivity"
RAW = OUT / "raw"
FIG = OUT / "figures"
HOLD_S = 10.0
LOG_STRIDE = 5  # 10 ms at dt=0.002
TAU = -18.0


def pack_full(sim) -> dict:
    snap = sim.snapshot()
    if hasattr(sim.data, "eq_active") and int(getattr(sim.model, "neq", 0) or 0):
        snap["eq_active"] = np.array(sim.data.eq_active, float).copy()
    if int(sim.model.nmocap):
        snap["mocap_pos"] = np.array(sim.data.mocap_pos, float).copy()
        snap["mocap_quat"] = np.array(sim.data.mocap_quat, float).copy()
    snap["w_cmd"] = np.asarray(sim.fsm.w_cmd, float).copy()
    return snap


def restore_full(sim, snap: dict) -> None:
    sim.load_snapshot(snap)
    if "eq_active" in snap and hasattr(sim.data, "eq_active"):
        sim.data.eq_active[:] = np.asarray(snap["eq_active"], float)
    if "mocap_pos" in snap and int(sim.model.nmocap):
        sim.data.mocap_pos[:] = snap["mocap_pos"]
        sim.data.mocap_quat[:] = snap["mocap_quat"]
    mujoco.mj_forward(sim.model, sim.data)


def capture_state_d() -> dict:
    holder = {}
    orig = demo._row

    def wrapped(sim, phase: str):
        r = orig(sim, phase)
        if (
            phase == "vertical_hold"
            and "snap" not in holder
            and float(r["t"]) >= 7.80
            and float(r["v_rel"]) < 0.010
        ):
            holder["snap"] = pack_full(sim)
            holder["t"] = float(r["t"])
            holder["row"] = {k: r[k] for k in r if k not in ("mech", "mech_contacts", "contacts")}
        return r

    demo._row = wrapped
    try:
        ref = load_or_measure_ref()
        run_teleport_case(
            S_COUPLED,
            ref,
            "coupled",
            interactive=False,
            playback=0.5,
            mode="REPOSITION_REGRASP",
            recovery_delay=0.10,
            diagnostic_hold=10.0,
            wrist_reposition={
                "omega_out": 1.2,
                "omega_return": 0.8,
                "theta_capture_deg": 60.0,
                "tau_secure": -18.0,
                "tau_reposition": -18.0,
                "reposition_s": 0.5,
                "capture_verify_s": 1.0,
                "vertical_s": 2.0,
                "fail_continue_s": 0.0,
                "do_return": True,
                "do_lift": False,
                "skip_unload": True,
                "secure_hold_s": 1.0,
                "inward_slide": True,
                "active_regrasp": False,
            },
        )
    finally:
        demo._row = orig
    if "snap" not in holder:
        raise RuntimeError("failed to capture STATE D")
    return holder


def capture_state_n() -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    parent = advance_to_parent(sim, cfg, mode="ZERO")
    if not parent.get("ok"):
        raise RuntimeError("failed STATE N parent")
    freeze(sim)
    return {"snap": pack_full(sim), "t": float(sim.data.time), "parent": parent.get("rel")}


def inspect_contact_defaults(sim) -> dict:
    ids = sim.ids
    live = []
    wr = np.zeros(6)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(sim.model.geom_bodyid[g1]), int(sim.model.geom_bodyid[g2])
        if ids.object_body not in (b1, b2):
            continue
        if ids.left_body not in (b1, b2) and ids.right_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(sim.model, sim.data, i, wr)
        live.append(
            {
                "solref": np.array(c.solref, float).tolist(),
                "solimp": np.array(c.solimp, float).tolist(),
                "mu": float(c.friction[0]),
                "dim": int(c.dim),
            }
        )
    obj = int(ids.object_geom)
    pads = [
        g
        for g in range(sim.model.ngeom)
        if int(sim.model.geom_bodyid[g]) in (int(ids.left_body), int(ids.right_body))
    ]
    return {
        "mujoco": mujoco.__version__,
        "timestep": float(sim.model.opt.timestep),
        "iterations": int(sim.model.opt.iterations),
        "tolerance": float(sim.model.opt.tolerance),
        "ls_iterations": int(getattr(sim.model.opt, "ls_iterations", 0) or 0),
        "solver": int(sim.model.opt.solver),
        "cone": int(sim.model.opt.cone),
        "impratio": float(sim.model.opt.impratio),
        "object_solref": np.array(sim.model.geom_solref[obj], float).tolist(),
        "object_solimp": np.array(sim.model.geom_solimp[obj], float).tolist(),
        "pad_solref_unique": sorted(
            {tuple(np.round(sim.model.geom_solref[g], 6)) for g in pads}
        ),
        "live_solref_unique": sorted({tuple(x["solref"]) for x in live}),
        "live_mu": sorted({x["mu"] for x in live}),
        "live_dim": sorted({x["dim"] for x in live}),
        "n_live": len(live),
    }


def apply_variant(sim, name: str) -> dict:
    """Runtime-only. Does not write XML. One family per name."""
    ids = sim.ids
    info = {"variant": name, "changes": []}
    if name == "baseline":
        return info
    if name == "harder":
        targets = [int(ids.object_geom)]
        targets += [
            g
            for g in range(sim.model.ngeom)
            if int(sim.model.geom_bodyid[g]) in (int(ids.left_body), int(ids.right_body))
        ]
        for g in targets:
            old = np.array(sim.model.geom_solref[g], float).copy()
            new = old.copy()
            new[0] = 0.005
            sim.model.geom_solref[g] = new
            info["changes"].append(
                {"geom": int(g), "solref_before": old.tolist(), "solref_after": new.tolist()}
            )
        info["family"] = "contact_softness_solref_timeconst_only"
        info["rationale"] = (
            "Live contacts used solref timeconst=0.015 s. Official object XML is 0.01. "
            "Diagnostic sets object+finger geom_solref[0]=0.005 (3x stiffer vs live 0.015) "
            "without changing solimp, condim, mu, cone, or solver."
        )
        mujoco.mj_forward(sim.model, sim.data)
        return info
    if name == "solver":
        old_it = int(sim.model.opt.iterations)
        old_tol = float(sim.model.opt.tolerance)
        old_ls = int(getattr(sim.model.opt, "ls_iterations", 0) or 0)
        sim.model.opt.iterations = max(old_it * 2, old_it + 50)
        sim.model.opt.tolerance = old_tol * 0.1 if old_tol > 0 else 1e-10
        if hasattr(sim.model.opt, "ls_iterations") and old_ls > 0:
            sim.model.opt.ls_iterations = old_ls * 2
        info["family"] = "solver_accuracy_only"
        info["rationale"] = (
            "Double Newton iterations and tighten tolerance 10x. No solref/solimp/mu/condim change."
        )
        info["changes"].append(
            {
                "iterations": [old_it, int(sim.model.opt.iterations)],
                "tolerance": [old_tol, float(sim.model.opt.tolerance)],
                "ls_iterations": [old_ls, int(getattr(sim.model.opt, "ls_iterations", 0) or 0)],
            }
        )
        mujoco.mj_forward(sim.model, sim.data)
        return info
    raise ValueError(name)


def _centroid(cons, side, ax):
    pts = [c for c in cons if c["side"] == side]
    if not pts:
        return float("nan")
    return float(np.mean([c["pos_h"][ax] for c in pts]))


def sample_row(sim, t_hold: float, phase: str) -> dict:
    r = _row(sim, phase)
    cons, mech = extract_contacts(sim)
    L = [c for c in cons if c["side"] == "L"]
    R = [c for c in cons if c["side"] == "R"]
    rhos = [c["rho"] for c in cons]
    pad = pad_span_hand(sim)
    physical_pack(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    vo = np.array(sim.data.cvel[sim.ids.object_body][3:6], float)
    return {
        "t": r["t"],
        "t_hold": t_hold,
        "e_x": r["e_x"],
        "e_y": r["e_y"],
        "e_z": r["e_z"],
        "v_rel_h": mech["v_rel_h"],
        "w_rel_h": mech["w_rel_h"],
        "aperture": r["aperture"],
        "tau": r["tau"],
        "nL": mech["nL"],
        "nR": mech["nR"],
        "Fn_L": float(sum(abs(c["Fn"]) for c in L)),
        "Fn_R": float(sum(abs(c["Fn"]) for c in R)),
        "rho_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "rho_max": float(np.nanmax(rhos)) if rhos else float("nan"),
        "cx_L": _centroid(cons, "L", 0),
        "cz_L": _centroid(cons, "L", 2),
        "cx_R": _centroid(cons, "R", 0),
        "cz_R": _centroid(cons, "R", 2),
        "ph": mech["ph"],
        "po": mech["po"],
        "vh": mech["vh"],
        "obj_linvel": vo.tolist(),
        "hand_linvel": r.get("hand_linvel"),
        "pos_err": mech["pos_err"],
        "ori_err_deg": float(ori_error_deg(r_des, Rh)),
        "residual_norm": mech["residual_norm"],
        "mu_live": sorted({c["mu"] for c in cons}),
        "dim_live": sorted({c["dim"] for c in cons}),
        "solref_live": sorted({tuple(c["solref"]) for c in cons if c.get("solref")}),
        "pad": pad,
        "obj_z": r["obj_z"],
        "clear": r["clear"],
        "wrist_act_deg": r.get("wrist_act_deg"),
    }


def robust_slope(t, y, t0=1.0, t1=10.0):
    m = (t >= t0) & (t <= t1)
    if int(m.sum()) < 10:
        return float("nan")
    tt, yy = t[m], y[m]
    n = len(tt)
    sl = []
    step = max(n // 400, 1)
    idx = np.arange(0, n, step)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dt = tt[idx[j]] - tt[idx[i]]
            if abs(dt) < 1e-6:
                continue
            sl.append((yy[idx[j]] - yy[idx[i]]) / dt)
    return float(np.median(sl)) if sl else float("nan")


def summarize(rows: list[dict]) -> dict:
    t = np.array([r["t_hold"] for r in rows], float)
    ez = np.array([r["e_z"] for r in rows], float)
    vz = np.array([r["v_rel_h"][2] for r in rows], float)

    def at(ts):
        i = int(np.argmin(np.abs(t - ts)))
        return float(ez[i] - ez[0])

    m = (t >= 1.0) & (t <= 10.0)
    dropped = any(r["obj_z"] < TABLE_DROP or r["clear"] < -0.005 for r in rows)
    uni = any((r["nL"] == 0) ^ (r["nR"] == 0) for r in rows)
    both = any(r["nL"] == 0 and r["nR"] == 0 for r in rows)
    res = [r["residual_norm"] for r in rows[:: max(len(rows) // 20, 1)]]
    invalid = float(np.nanmedian(res)) > 0.5
    return {
        "d_ez_2s_mm": 1e3 * at(2.0),
        "d_ez_5s_mm": 1e3 * at(5.0),
        "d_ez_10s_mm": 1e3 * at(10.0),
        "median_vz_1_10": float(np.median(vz[m])) if m.any() else float("nan"),
        "slope_ez_1_10_mm_s": 1e3 * robust_slope(t, ez, 1.0, 10.0),
        "max_vrel": float(max(np.linalg.norm(r["v_rel_h"]) for r in rows)),
        "max_wrel": float(max(np.linalg.norm(r["w_rel_h"]) for r in rows)),
        "nL_range": [min(r["nL"] for r in rows), max(r["nL"] for r in rows)],
        "nR_range": [min(r["nR"] for r in rows), max(r["nR"] for r in rows)],
        "Fn_L_range": [min(r["Fn_L"] for r in rows), max(r["Fn_L"] for r in rows)],
        "Fn_R_range": [min(r["Fn_R"] for r in rows), max(r["Fn_R"] for r in rows)],
        "rho_mean_med": float(np.nanmedian([r["rho_mean"] for r in rows])),
        "rho_max_max": float(np.nanmax([r["rho_max"] for r in rows])),
        "cz_L_drift_mm": 1e3 * (rows[-1]["cz_L"] - rows[0]["cz_L"]),
        "cz_R_drift_mm": 1e3 * (rows[-1]["cz_R"] - rows[0]["cz_R"]),
        "aperture_drift": float(rows[-1]["aperture"] - rows[0]["aperture"]),
        "hand_pos_err_end_mm": [1e3 * x for x in rows[-1]["pos_err"]],
        "ori_err_end_deg": rows[-1]["ori_err_deg"],
        "residual_median": float(np.nanmedian(res)),
        "dropped": dropped,
        "unilateral": uni,
        "both_lost": both,
        "invalid_balance": invalid,
        "ez0_mm": 1e3 * float(ez[0]),
        "ez10_mm": 1e3 * float(ez[-1]),
        "n_samples": len(rows),
        "mu_live": rows[0]["mu_live"],
        "dim_live": rows[0]["dim_live"],
        "solref_live": [list(x) for x in rows[0]["solref_live"]],
    }


def run_hold(snap, variant: str, label: str, interactive: bool = False, playback: float = 0.35):
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    restore_full(sim, snap)
    # Do NOT call set_finger_object_sliding_mu(..., data=sim.data) after restore.
    # That path uses mj_setConst and resets qpos to the home keyframe (verified).
    # Pads/object already have live mu=1.0 from make_parent_sim + snapshot mass/friction.
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, data=None)
    mujoco.mj_forward(sim.model, sim.data)
    defaults = inspect_contact_defaults(sim)
    vinfo = apply_variant(sim, variant)
    freeze(sim)
    if hasattr(sim.fsm, "v_des"):
        sim.fsm.v_des[:] = 0.0
    if hasattr(sim.fsm, "w_des"):
        sim.fsm.w_des[:] = 0.0
    freeze_note = {
        "p_des": np.array(sim.fsm.p_des, float).tolist(),
        "r_des": np.array(sim.fsm.r_des, float).reshape(9).tolist(),
        "v_cmd": np.array(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.array(sim.fsm.w_cmd, float).tolist(),
        "ph": np.array(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "rule": "p_des:=p_hand, r_des:=R_hand, v_cmd:=0, w_cmd:=0 once at hold start; then fixed",
    }
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    n = int(round(HOLD_S / dt))
    rows = []
    viewer = None
    pacer = None
    cm = None
    if interactive:
        import mujoco.viewer as mjviewer
        from training.demo_teleport_recovery_state import overlay, _cam
        from training.impact_demo_core import RENDER_HZ, RealtimePacer
        from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

        cm = mjviewer.launch_passive(sim.model, sim.data)
        viewer = cm.__enter__()
        apply_viewer_camera(viewer, _cam(sim))
        enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
        pacer = RealtimePacer(playback, RENDER_HZ, dt)
        pacer.start(float(sim.data.time))
        pacer.note_sync(float(sim.data.time))
    t0 = float(sim.data.time)
    rows.append(sample_row(sim, 0.0, "diagnostic_hold"))
    try:
        for k in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
            t_hold = (k + 1) * dt
            if k % LOG_STRIDE == 0 or k + 1 == n:
                rows.append(sample_row(sim, t_hold, "diagnostic_hold"))
            if viewer is not None and pacer is not None and pacer.should_render(float(sim.data.time)):
                extra = {"minimal": True, "tau": TAU, "aperture": rows[-1]["aperture"] if rows else 0}
                overlay(
                    viewer,
                    None,
                    "diagnostic_hold",
                    float(sim.data.time),
                    mode=label,
                    extra=extra,
                )
                try:
                    pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
                    viewer.add_overlay(pos, "STATE", label.split("/")[0])
                    viewer.add_overlay(pos, "CONTACT", variant.upper())
                    viewer.add_overlay(pos, "HOLD t", f"{t_hold:.2f} s")
                except Exception:
                    pass
                viewer.sync()
                pacer.note_sync(float(sim.data.time))
                pacer.wait_if_ahead(float(sim.data.time))
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
    finally:
        if interactive and cm is not None:
            cm.__exit__(None, None, None)
    after = inspect_contact_defaults(sim)
    summ = summarize(rows)
    summ["variant_info"] = vinfo
    summ["defaults_before"] = defaults
    summ["defaults_after"] = after
    summ["freeze"] = freeze_note
    summ["label"] = label
    summ["t_sim0"] = t0
    return rows, summ


def save_traj(tag: str, rows, summ):
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW / f"{tag}.npz",
        t_hold=np.array([r["t_hold"] for r in rows], float),
        e_x=np.array([r["e_x"] for r in rows], float),
        e_y=np.array([r["e_y"] for r in rows], float),
        e_z=np.array([r["e_z"] for r in rows], float),
        v_rel_h=np.array([r["v_rel_h"] for r in rows], float),
        w_rel_h=np.array([r["w_rel_h"] for r in rows], float),
        nL=np.array([r["nL"] for r in rows], float),
        nR=np.array([r["nR"] for r in rows], float),
        Fn_L=np.array([r["Fn_L"] for r in rows], float),
        Fn_R=np.array([r["Fn_R"] for r in rows], float),
        rho_mean=np.array([r["rho_mean"] for r in rows], float),
        rho_max=np.array([r["rho_max"] for r in rows], float),
        cz_L=np.array([r["cz_L"] for r in rows], float),
        cz_R=np.array([r["cz_R"] for r in rows], float),
        pos_err=np.array([r["pos_err"] for r in rows], float),
        aperture=np.array([r["aperture"] for r in rows], float),
        residual=np.array([r["residual_norm"] for r in rows], float),
        tau=np.array([r["tau"] for r in rows], float),
        obj_z=np.array([r["obj_z"] for r in rows], float),
    )
    (RAW / f"{tag}.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")


def figures(metrics: dict):
    FIG.mkdir(parents=True, exist_ok=True)

    def load(tag):
        return np.load(RAW / f"{tag}.npz")

    d0, n0 = load("D_baseline"), load("N_baseline")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], 1e3 * (d0["e_z"] - d0["e_z"][0]), label="D baseline")
    ax.plot(n0["t_hold"], 1e3 * (n0["e_z"] - n0["e_z"][0]), label="N baseline")
    ax.set_xlabel("hold t (s)")
    ax.set_ylabel("Delta r_h.z (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_rh_z.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], d0["v_rel_h"][:, 2], label="D")
    ax.plot(n0["t_hold"], n0["v_rel_h"][:, 2], label="N")
    ax.set_ylabel("v_rel_h.z (m/s)")
    ax.set_xlabel("hold t (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_vrel_z.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], d0["rho_mean"], label="D mean")
    ax.plot(d0["t_hold"], d0["rho_max"], ls=":", label="D max")
    ax.plot(n0["t_hold"], n0["rho_mean"], label="N mean")
    ax.plot(n0["t_hold"], n0["rho_max"], ls=":", label="N max")
    ax.set_ylabel("rho")
    ax.set_xlabel("hold t (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_rho.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], 1e3 * d0["cz_L"], label="D L")
    ax.plot(d0["t_hold"], 1e3 * d0["cz_R"], label="D R")
    ax.plot(n0["t_hold"], 1e3 * n0["cz_L"], label="N L")
    ax.plot(n0["t_hold"], 1e3 * n0["cz_R"], label="N R")
    ax.set_ylabel("centroid z mm")
    ax.set_xlabel("hold t (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_centroid_z.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], d0["Fn_L"], label="D L")
    ax.plot(d0["t_hold"], d0["Fn_R"], label="D R")
    ax.plot(n0["t_hold"], n0["Fn_L"], label="N L")
    ax.plot(n0["t_hold"], n0["Fn_R"], label="N R")
    ax.set_ylabel("sum |Fn| N")
    ax.set_xlabel("hold t (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_Fn.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d0["t_hold"], 1e3 * np.linalg.norm(d0["pos_err"], axis=1), label="D |pos_err|")
    ax.plot(n0["t_hold"], 1e3 * np.linalg.norm(n0["pos_err"], axis=1), label="N |pos_err|")
    ax.set_ylabel("hand tracking |p_des-p_hand| mm")
    ax.set_xlabel("hold t (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_track.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tag, sty in (
        ("D_baseline", "-"),
        ("N_baseline", "-"),
        ("D_harder", "--"),
        ("N_harder", "--"),
        ("D_solver", ":"),
        ("N_solver", ":"),
    ):
        z = load(tag)
        ax.plot(z["t_hold"], 1e3 * (z["e_z"] - z["e_z"][0]), sty, label=tag)
    ax.set_xlabel("hold t (s)")
    ax.set_ylabel("Delta r_h.z mm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_all_rh_z.png", dpi=130)
    plt.close(fig)

    names = ["baseline", "harder", "solver"]
    d_s = [metrics[f"D_{k}"]["slope_ez_1_10_mm_s"] for k in names]
    n_s = [metrics[f"N_{k}"]["slope_ez_1_10_mm_s"] for k in names]
    ex = [d_s[i] - n_s[i] for i in range(3)]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    ax.plot(x, d_s, "o-", label="D")
    ax.plot(x, n_s, "s-", label="N")
    ax.plot(x, ex, "^-", label="excess D-N")
    ax.set_xticks(x, names)
    ax.set_ylabel("robust slope r_h.z mm/s (1-10 s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_creep_rates.png", dpi=130)
    plt.close(fig)


def classify(metrics: dict) -> tuple[str, str]:
    db = metrics["D_baseline"]["slope_ez_1_10_mm_s"]
    nb = metrics["N_baseline"]["slope_ez_1_10_mm_s"]
    if db != db or nb != nb:
        return "CASE 5 UNRESOLVED", "NaN slope"
    if abs(db) < 0.15 and abs(nb) < 0.15:
        return (
            "CASE 4 CONTROLLER / HISTORY CONFOUND",
            "After freeze-here, neither state shows the previous ~1 mm/s distal creep.",
        )
    dh = metrics["D_harder"]["slope_ez_1_10_mm_s"]
    nh = metrics["N_harder"]["slope_ez_1_10_mm_s"]
    ds = metrics["D_solver"]["slope_ez_1_10_mm_s"]
    ns = metrics["N_solver"]["slope_ez_1_10_mm_s"]
    if any(metrics[k]["invalid_balance"] for k in metrics):
        return "CASE 5 UNRESOLVED", "A condition failed force-balance check."
    excess_b = db - nb
    excess_h = dh - nh
    both_large = abs(db) > 0.4 and abs(nb) > 0.4
    both_shrink = abs(dh) < 0.5 * abs(db) and abs(nh) < 0.5 * abs(nb)
    d_only = abs(db) > 0.4 and abs(nb) < 0.15
    if d_only and abs(dh) > 0.4:
        return (
            "CASE 1 ROBUST DISTURBED-STATE CREEP",
            "Nominal ~stationary; disturbed creeps and remains large under harder contact.",
        )
    if both_large and both_shrink:
        return (
            "CASE 2 CONTACT-MODEL / NUMERICAL CREEP DOMINATES",
            "D and N both creep at comparable baseline rates and both shrink when contact is hardened.",
        )
    if abs(nb) > 0.2 and abs(excess_b) > 0.3 and abs(excess_h) > 0.3:
        return (
            "CASE 3 MIXED",
            "Shared creep floor plus disturbed excess that survives the hardness/solver variants.",
        )
    if both_large and abs(db - nb) < 0.25 * max(abs(db), 1e-6):
        return (
            "CASE 2 CONTACT-MODEL / NUMERICAL CREEP DOMINATES",
            "D and N creep at similar rates (excess small vs absolute).",
        )
    if d_only:
        return (
            "CASE 1 ROBUST DISTURBED-STATE CREEP",
            "Nominal nearly still; disturbed creeps. Check hardness columns for robustness.",
        )
    _ = (ds, ns)
    return (
        "CASE 5 UNRESOLVED",
        f"baseline slopes D={db:.3f} N={nb:.3f} mm/s; harder D={dh:.3f} N={nh:.3f}; solver D={ds:.3f} N={ns:.3f}.",
    )


def write_md(metrics, classif, d_meta, n_meta, reproduced):
    p = OUT / "CONTACT_CREEP_SENSITIVITY_AUDIT.md"
    L = []
    a = L.append
    a("# Contact-creep sensitivity (disturbed vs nominal)")
    a("")
    a("Diagnostic only. Official XML/controller/recovery were not modified. Friction and condim unchanged.")
    a("")
    a("## Evidence classes")
    a("- VERIFIED CODE FACT")
    a("- RAW TRAJECTORY EVIDENCE")
    a("- USER VISUAL OBSERVATION (viewer commands below; not claimed here)")
    a("- INTERPRETATION")
    a("")
    a("## Freeze-here hold (VERIFIED CODE FACT)")
    a("")
    a("`p_des := p_hand`, `r_des := R_hand`, `v_cmd := 0`, `w_cmd := 0` once at diagnostic start, then `tick_vw(0,0,tau=-18)`. This is not a stale-target hold.")
    a("")
    a("## Live contact defaults at diagnostic start (RAW / CODE)")
    a("")
    a("```")
    a(json.dumps(metrics["D_baseline"]["defaults_before"], indent=2, default=str))
    a("```")
    a("")
    a("Harder-contact family: **solref time-constant only** on object + finger geoms → 0.005 s.")
    a("Solver family: **iterations and tolerance only**.")
    a("")
    a("## STATE D capture")
    a("")
    a(
        f"From 1.0 s inward-slide → return trajectory, first `vertical_hold` sample with t>=7.80 s and |v_rel|<0.01. Capture t={d_meta.get('t')}. Full `GraspSim.snapshot` (qpos/qvel/act/ctrl/time/p_des/r_des/v_cmd/FSM/meter/qacc_warmstart)."
    )
    a("")
    a("## STATE N capture")
    a("")
    a("Real nominal FSM to airborne bilateral hold (no pose teleport). m=0.20 kg, pair mu=1.0. Freeze-here at parent.")
    a("")
    a("## 6. D baseline replay vs prior audit")
    a("")
    a(
        f"Prior vertical creep ~1.2 mm/s in `v_rel_h.z`. This D baseline slope 1-10 s = **{metrics['D_baseline']['slope_ez_1_10_mm_s']:.3f} mm/s**, median vz={metrics['D_baseline']['median_vz_1_10']:.5f} m/s. Reproduced persistent +z: **{reproduced}**."
    )
    a("")
    a("## Matrix (RAW)")
    a("")
    a("| cell | d_ez 2/5/10 s mm | slope 1-10 mm/s | med vz 1-10 | rho mean/max | nL nR | drop/uni/both | |res| |")
    a("|---|---|---|---|---|---|---|---|")
    for k, s in metrics.items():
        a(
            f"| {k} | {s['d_ez_2s_mm']:.2f} / {s['d_ez_5s_mm']:.2f} / {s['d_ez_10s_mm']:.2f} | "
            f"{s['slope_ez_1_10_mm_s']:.3f} | {s['median_vz_1_10']:.5f} | "
            f"{s['rho_mean_med']:.3f}/{s['rho_max_max']:.3f} | {s['nL_range']}/{s['nR_range']} | "
            f"{s['dropped']}/{s['unilateral']}/{s['both_lost']} | {s['residual_median']:.3f} |"
        )
    a("")
    a("## Excess creep (D − N) mm/s")
    a("")
    for v in ("baseline", "harder", "solver"):
        ex = metrics[f"D_{v}"]["slope_ez_1_10_mm_s"] - metrics[f"N_{v}"]["slope_ez_1_10_mm_s"]
        a(f"- {v}: **{ex:.3f}**")
    a("")
    a("## INTERPRETATION")
    a("")
    a(f"**{classif[0]}**")
    a("")
    a(classif[1])
    a("")
    a("Not a hardware claim. Not a recovery-strategy change.")
    a("")
    a("## Viewer")
    a("")
    a("```")
    a("python training/contact_creep_sensitivity_audit.py --view D --variant baseline --playback-speed 0.35")
    a("python training/contact_creep_sensitivity_audit.py --view N --variant baseline --playback-speed 0.35")
    a("python training/contact_creep_sensitivity_audit.py --view D --variant harder --playback-speed 0.35")
    a("python training/contact_creep_sensitivity_audit.py --view N --variant harder --playback-speed 0.35")
    a("```")
    a("")
    a("Full 10 s hold. Overlay: DISTURBED/NOMINAL, variant, hold time. No early success exit.")
    p.write_text("\n".join(L), encoding="utf-8")


def _arrify(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            try:
                out[k] = np.array(v, float)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def dump_state(path: Path, snap: dict) -> None:
    ser = {}
    for k, v in snap.items():
        if isinstance(v, np.ndarray):
            ser[k] = v.tolist()
        else:
            ser[k] = v
    path.write_text(json.dumps(ser, indent=2, default=str), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=("D", "N"), default=None)
    p.add_argument("--variant", choices=("baseline", "harder", "solver"), default="baseline")
    p.add_argument("--playback-speed", type=float, default=0.35)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.view:
        sp = RAW / f"state_{args.view}.json"
        if not sp.exists():
            print("run the audit once headless first to save states")
            return 2
        snap = _arrify(json.loads(sp.read_text(encoding="utf-8")))
        run_hold(
            snap,
            args.variant,
            f"{'DISTURBED' if args.view == 'D' else 'NOMINAL'}/{args.variant}",
            interactive=True,
            playback=args.playback_speed,
        )
        return 0

    print("capturing D ...", flush=True)
    d_meta = capture_state_d()
    print("capturing N ...", flush=True)
    n_meta = capture_state_n()
    dump_state(RAW / "state_D.json", d_meta["snap"])
    dump_state(RAW / "state_N.json", n_meta["snap"])
    (RAW / "capture_meta.json").write_text(
        json.dumps({"D_t": d_meta.get("t"), "N_t": n_meta.get("t")}, indent=2),
        encoding="utf-8",
    )

    metrics = {}
    for st, snap, name in (("D", d_meta["snap"], "DISTURBED"), ("N", n_meta["snap"], "NOMINAL")):
        for var in ("baseline", "harder", "solver"):
            tag = f"{st}_{var}"
            print("hold", tag, "...", flush=True)
            rows, summ = run_hold(snap, var, f"{name}/{var}", interactive=False)
            save_traj(tag, rows, summ)
            metrics[tag] = summ
            print(
                " ",
                tag,
                "slope",
                round(summ["slope_ez_1_10_mm_s"], 3),
                "d10",
                round(summ["d_ez_10s_mm"], 2),
                "vz",
                round(summ["median_vz_1_10"], 5),
                flush=True,
            )

    db = metrics["D_baseline"]
    reproduced = bool(db["median_vz_1_10"] > 0.0003 and db["slope_ez_1_10_mm_s"] > 0.3)
    if not reproduced:
        print("STATE D did not reproduce persistent +z creep. STOP before over-interpreting sensitivity.", db)
    figures(metrics)
    classif = (
        classify(metrics)
        if reproduced
        else (
            "STOP: D replay did not reproduce creep",
            "Fix snapshot/hold semantics before contact-model conclusions.",
        )
    )
    slim = {k: {kk: vv for kk, vv in s.items() if kk not in ("freeze", "defaults_before", "defaults_after", "variant_info")} for k, s in metrics.items()}
    (RAW / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    write_md(metrics, classif, d_meta, n_meta, reproduced)
    print("class", classif[0])
    print("wrote", OUT / "CONTACT_CREEP_SENSITIVITY_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
