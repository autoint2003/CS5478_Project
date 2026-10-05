"""Calibrate MuJoCo noslip_iterations: kill secure-grasp creep, keep intended slip.

Diagnostic in-memory opt.noslip_iterations only. Official XML unchanged.
solref/solimp/mu/condim/solver/dt are the current baseline (not harder contact).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.box_vs_cylinder_creep import convert_object_to_box
from training.demo_teleport_recovery_state import (
    PAIR_MU,
    advance_to_parent,
    make_parent_sim,
    signed_rot_about_y_deg,
)
from training.nominal_grasp_creep_root_cause import HOLD_S, LOG_DT, TAU, dump_json, robust_slope
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "noslip_calibration"
RAW = OUT / "raw"
FIG = OUT / "figures"

# Predefined BEFORE any sweep results (simulation numerical-stability only).
PREFERRED_SLOPE_MM_S = 0.05
PREFERRED_D10_MM = 0.50
LOOSE_SLOPE_MM_S = 0.10
LOOSE_D10_MM = 1.00

CANDIDATES = [0, 1, 2, 3, 5]
GRIP_SWEEP = [-18.0, -12.0, -8.0, -4.0, -2.0, 0.0, 4.0, 8.0]
GRIP_SWEEP_S = 2.5
RELEASE_TAU = 8.0  # yaml open_tau
RELEASE_S = 1.5
PIVOT_HOLD_S = 2.0
PIVOT_OMEGA = 1.2
PIVOT_DEG = 60.0


def live_noslip(sim) -> dict:
    m = sim.model
    return {
        "mujoco": mujoco.__version__,
        "noslip_iterations": int(m.opt.noslip_iterations),
        "noslip_tolerance": float(m.opt.noslip_tolerance),
        "iterations": int(m.opt.iterations),
        "tolerance": float(m.opt.tolerance),
        "solver": int(m.opt.solver),
        "cone": int(m.opt.cone),
        "impratio": float(m.opt.impratio),
        "timestep": float(m.opt.timestep),
        "integrator": int(m.opt.integrator),
    }


def set_noslip(sim, n: int) -> None:
    sim.model.opt.noslip_iterations = int(n)
    # tolerance left at live default


def restore(sim, snap) -> None:
    sim.load_snapshot(snap)
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, data=None)
    mujoco.mj_forward(sim.model, sim.data)


def make_shape(shape: str):
    sim = make_parent_sim()
    if shape == "box":
        convert_object_to_box(sim)
    return sim


def sample(sim, t_hold: float, ez0: float, extra=None) -> dict:
    cons, mech = extract_contacts(sim)
    o = physical_pack(sim)
    rhos = [c["rho"] for c in cons]
    fts = [c["Ft"] for c in cons]
    d = {
        "t_hold": t_hold,
        "e_z": float(mech["rh"][2]),
        "e_x": float(mech["rh"][0]),
        "dez": float(mech["rh"][2] - ez0),
        "v_rel_z": float(mech["v_rel_h"][2]),
        "v_rel": float(np.linalg.norm(mech["v_rel_h"])),
        "aperture": float(o["aperture"]),
        "nL": mech["nL"],
        "nR": mech["nR"],
        "Fn_L": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "L")),
        "Fn_R": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "R")),
        "Ft_sum": float(sum(fts)) if fts else 0.0,
        "rho_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "rho_max": float(np.nanmax(rhos)) if rhos else float("nan"),
        "residual_norm": mech["residual_norm"],
        "pos_err": mech["pos_err"],
        "obj_z": float(o["obj_z"]),
        "tau": float(sim.data.ctrl[7]),
        "noslip": int(sim.model.opt.noslip_iterations),
        "solref_live": sorted({tuple(c["solref"]) for c in cons if c.get("solref")}),
    }
    if extra:
        d.update(extra)
    return d


def ez0_of(sim) -> float:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rh = Rh.T @ (np.array(sim.data.xpos[sim.ids.object_body]) - np.array(sim.data.xpos[sim.ids.hand_body]))
    return float(rh[2])


def hold_tau(sim, snap, ns: int, tau: float, seconds: float, shape="cylinder"):
    restore(sim, snap)
    set_noslip(sim, ns)
    freeze(sim)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    n = int(round(seconds / dt))
    stride = max(int(round(LOG_DT / dt)), 1)
    z0 = ez0_of(sim)
    rows = [sample(sim, 0.0, z0)]
    t0 = time.perf_counter()
    for k in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
        if k % stride == 0 or k + 1 == n:
            rows.append(sample(sim, (k + 1) * dt, z0))
    wall = time.perf_counter() - t0
    return rows, wall


def summarize_hold(rows, wall=None):
    t = np.array([r["t_hold"] for r in rows], float)
    ez = np.array([r["e_z"] for r in rows], float)
    vz = np.array([r["v_rel_z"] for r in rows], float)
    ap = np.array([r["aperture"] for r in rows], float)

    def at(ts):
        i = int(np.argmin(np.abs(t - ts)))
        return float(ez[i] - ez[0])

    dropped = any(r["obj_z"] < TABLE_DROP or (r["nL"] == 0 and r["nR"] == 0) for r in rows[5:])
    uni = any((r["nL"] == 0) ^ (r["nR"] == 0) for r in rows)
    return {
        "d_ez_2s_mm": 1e3 * at(min(2.0, t[-1])),
        "d_ez_5s_mm": 1e3 * at(min(5.0, t[-1])),
        "d_ez_10s_mm": 1e3 * at(min(10.0, t[-1])),
        "slope_ez_1_10_mm_s": 1e3 * robust_slope(t, ez, 1.0, min(10.0, float(t[-1]))),
        "median_vz": float(np.median(vz)),
        "max_vrel": float(max(r["v_rel"] for r in rows)),
        "d_aperture_mm": 1e3 * float(ap[-1] - ap[0]),
        "Fn_L_med": float(np.median([r["Fn_L"] for r in rows])),
        "Fn_R_med": float(np.median([r["Fn_R"] for r in rows])),
        "Ft_sum_med": float(np.median([r["Ft_sum"] for r in rows])),
        "rho_mean_med": float(np.nanmedian([r["rho_mean"] for r in rows])),
        "rho_max_max": float(np.nanmax([r["rho_max"] for r in rows])),
        "nL_range": [min(r["nL"] for r in rows), max(r["nL"] for r in rows)],
        "nR_range": [min(r["nR"] for r in rows), max(r["nR"] for r in rows)],
        "residual_median": float(np.nanmedian([r["residual_norm"] for r in rows])),
        "hand_pos_err_end_mm": [1e3 * x for x in rows[-1]["pos_err"]],
        "dropped_or_both_lost": dropped,
        "unilateral": uni,
        "d_obj_z_mm": 1e3 * (rows[-1]["obj_z"] - rows[0]["obj_z"]),
        "solref_live": [list(x) for x in rows[0]["solref_live"]],
        "tau": rows[-1]["tau"],
        "noslip": rows[0]["noslip"],
        "wall_s": wall,
        "n_samples": len(rows),
        "d_ex_mm": 1e3 * (rows[-1]["e_x"] - rows[0]["e_x"]),
    }


def save_npz(tag, rows):
    np.savez_compressed(
        RAW / f"{tag}.npz",
        t_hold=np.array([r["t_hold"] for r in rows], float),
        e_z=np.array([r["e_z"] for r in rows], float),
        e_x=np.array([r["e_x"] for r in rows], float),
        v_rel_z=np.array([r["v_rel_z"] for r in rows], float),
        v_rel=np.array([r["v_rel"] for r in rows], float),
        aperture=np.array([r["aperture"] for r in rows], float),
        Fn_L=np.array([r["Fn_L"] for r in rows], float),
        nL=np.array([r["nL"] for r in rows], float),
        nR=np.array([r["nR"] for r in rows], float),
        rho_mean=np.array([r["rho_mean"] for r in rows], float),
        residual=np.array([r["residual_norm"] for r in rows], float),
        obj_z=np.array([r["obj_z"] for r in rows], float),
    )


def capture_parent(shape: str):
    sim = make_shape(shape)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    parent = advance_to_parent(sim, cfg, mode="ZERO")
    if not parent.get("ok"):
        raise RuntimeError(f"parent failed {shape}")
    freeze(sim)
    return sim.snapshot(), live_noslip(sim)


def capture_tilt60(snap_cyl):
    """Rotate to ~60 deg at tau=-18, noslip=0, freeze. Privileged only as an IC for Test E."""
    sim = make_shape("cylinder")
    restore(sim, snap_cyl)
    set_noslip(sim, 0)
    freeze(sim)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    n_max = int(round(np.radians(PIVOT_DEG + 15.0) / PIVOT_OMEGA / dt))
    for _ in range(n_max):
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        w_w = Rd @ np.array([0.0, PIVOT_OMEGA, 0.0])
        tick_vw(sim, np.zeros(3), w_w, TAU, gains)
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        if abs(signed_rot_about_y_deg(Rh0, Rh)) >= PIVOT_DEG - 0.15:
            break
    tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
    freeze(sim)
    o = physical_pack(sim)
    meta = {
        "wrist_deg": float(signed_rot_about_y_deg(Rh0, np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3)))),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "rh": np.asarray(o["rh"], float).tolist(),
        "rho_note": "tilt IC captured at noslip=0",
    }
    return sim.snapshot(), meta


def impact_probe(ns: int) -> dict:
    from training.demo_teleport_recovery_state import measure_canonical_impact

    # Patch: measure_canonical_impact builds its own sim. Set noslip via wrapping make_demo_sim.
    import training.impact_demo_core as idc
    import training.demo_teleport_recovery_state as demo

    orig = idc.make_demo_sim

    def wrapped(ball_mass: float):
        sim, cfg = orig(ball_mass)
        set_noslip(sim, ns)
        return sim, cfg

    idc.make_demo_sim = wrapped
    demo.make_demo_sim = wrapped
    t0 = time.perf_counter()
    try:
        out = measure_canonical_impact()
    finally:
        idc.make_demo_sim = orig
        demo.make_demo_sim = orig
    wall = time.perf_counter() - t0
    # keep a compact subset
    slim = {k: out[k] for k in out if k in (
        "t_hit", "t_post", "pre", "post", "first_hit_rel", "ok"
    ) or k.startswith("t_")}
    # flatten useful scalars if present
    post = out.get("post") or {}
    pre = out.get("pre") or {}
    return {
        "noslip": ns,
        "wall_s": wall,
        "t_hit": out.get("t_hit") or out.get("t_first_hit"),
        "post_keys": list(post)[:40] if isinstance(post, dict) else None,
        "pre": {k: pre[k] for k in list(pre)[:12]} if isinstance(pre, dict) else pre,
        "post": {k: post[k] for k in list(post)[:20]} if isinstance(post, dict) else post,
        "raw_ok": bool(out.get("ok", True)),
    }


def overlay(viewer, title, phase, t_hold, dez, extra=""):
    if viewer is None:
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "TITLE", title)
        viewer.add_overlay(pos, "PHASE", str(phase))
        viewer.add_overlay(pos, "HOLD t", f"{t_hold:.2f} s")
        viewer.add_overlay(pos, "Delta r_h.z", f"{dez:.2f} mm")
        if extra:
            viewer.add_overlay(pos, "NOTE", extra)
    except Exception:
        pass


def run_viewer(kind: str, ns: int, playback: float):
    import mujoco.viewer as mjviewer
    from training.demo_teleport_recovery_state import PARENT_HOLD, Z_AIR, _cam
    from training.impact_demo_core import RENDER_HZ, RealtimePacer
    from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_shape("cylinder")
    dt = float(sim.model.opt.timestep)
    cm = mjviewer.launch_passive(sim.model, sim.data)
    viewer = cm.__enter__()
    apply_viewer_camera(viewer, _cam(sim))
    enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
    pacer = RealtimePacer(playback, RENDER_HZ, dt)
    pacer.start(0.0)
    pacer.note_sync(0.0)
    title = f"{kind} noslip={ns}"
    air_ok = 0.0
    ez0 = 0.0
    try:
        n_max = int(round(12.0 / dt))
        for _ in range(n_max):
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            o = physical_pack(sim)
            t = float(sim.data.time)
            if pacer.should_render(t):
                overlay(viewer, title, str(sim.fsm.phase), 0.0, 0.0, f"noslip={ns}")
                viewer.sync()
                pacer.note_sync(t)
                pacer.wait_if_ahead(t)
            if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
                if int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["v_rel"]) < 0.05:
                    air_ok += dt
                else:
                    air_ok = 0.0
                if air_ok >= PARENT_HOLD and abs(float(o["e_x"])) < 0.003:
                    freeze(sim)
                    ez0 = float(np.asarray(o["rh"])[2])
                    break
        set_noslip(sim, ns)
        freeze(sim)
        gains = gains_from_cfg(cfg)
        if kind == "secure":
            n = int(round(HOLD_S / dt))
            for k in range(n):
                tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
                o = physical_pack(sim)
                dez = 1e3 * (float(np.asarray(o["rh"])[2]) - ez0)
                t_hold = (k + 1) * dt
                if pacer.should_render(float(sim.data.time)):
                    overlay(viewer, title, "hold", t_hold, dez, f"noslip={ns} tau=-18")
                    viewer.sync()
                    pacer.note_sync(float(sim.data.time))
                    pacer.wait_if_ahead(float(sim.data.time))
        elif kind == "pivot":
            Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
            n_max = int(round(np.radians(PIVOT_DEG + 15.0) / PIVOT_OMEGA / dt))
            for _ in range(n_max):
                Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
                w_w = Rd @ np.array([0.0, PIVOT_OMEGA, 0.0])
                tick_vw(sim, np.zeros(3), w_w, TAU, gains)
                Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
                o = physical_pack(sim)
                if pacer.should_render(float(sim.data.time)):
                    overlay(viewer, title, "tilt", 0.0, 1e3 * float(np.asarray(o["rh"])[2]), f"noslip={ns}")
                    viewer.sync()
                    pacer.note_sync(float(sim.data.time))
                    pacer.wait_if_ahead(float(sim.data.time))
                if abs(signed_rot_about_y_deg(Rh0, Rh)) >= PIVOT_DEG - 0.15:
                    break
            freeze(sim)
            ez1 = float(physical_pack(sim)["rh"][2])
            n = int(round(PIVOT_HOLD_S / dt))
            for k in range(n):
                tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
                o = physical_pack(sim)
                if pacer.should_render(float(sim.data.time)):
                    overlay(viewer, title, "tilt-hold", (k + 1) * dt, 1e3 * (float(o["rh"][2]) - ez1), f"noslip={ns}")
                    viewer.sync()
                    pacer.note_sync(float(sim.data.time))
                    pacer.wait_if_ahead(float(sim.data.time))
    finally:
        cm.__exit__(None, None, None)


def meets_pref(s):
    return abs(s["slope_ez_1_10_mm_s"]) <= PREFERRED_SLOPE_MM_S and abs(s["d_ez_10s_mm"]) <= PREFERRED_D10_MM


def meets_loose(s):
    return abs(s["slope_ez_1_10_mm_s"]) <= LOOSE_SLOPE_MM_S and abs(s["d_ez_10s_mm"]) <= LOOSE_D10_MM


def select(metrics, slip, release, pivot):
    acceptable = []
    for ns in CANDIDATES:
        a = metrics[f"cyl_ns{ns}"]
        b = metrics[f"box_ns{ns}"]
        if not (meets_pref(a) and meets_pref(b)):
            continue
        sl = slip[ns]
        rel = release[ns]
        pv = pivot[ns]
        # C: still slips (dropped or large vz vs secure)
        still_slip = sl["dropped_or_both_lost"] or abs(sl["median_vz"]) > 0.01 or abs(sl["d_ez_10s_mm"]) > 5.0
        still_rel = rel["dropped_or_both_lost"] or rel["d_obj_z_mm"] < -20.0
        # E: not qualitatively stuck vs ns=0
        p0 = pivot[0]
        disp0 = abs(p0["d_ex_mm"]) + abs(p0["d_ez_10s_mm"])
        disp = abs(pv["d_ex_mm"]) + abs(pv["d_ez_10s_mm"])
        killed = disp0 > 1.0 and disp < 0.15 * disp0
        osc = pv["max_vrel"] > 0.5 and p0["max_vrel"] < 0.2
        if still_slip and still_rel and not killed and not osc and not a["dropped_or_both_lost"]:
            acceptable.append(ns)
    if not acceptable:
        # diagnose
        any_creep_ok = [ns for ns in CANDIDATES if ns > 0 and meets_pref(metrics[f"cyl_ns{ns}"])]
        return None, "no_acceptable", acceptable, any_creep_ok
    return min(acceptable), "smallest_acceptable", acceptable, []


def figures(metrics, slip, release, pivot, walls):
    FIG.mkdir(parents=True, exist_ok=True)
    xs = np.array(CANDIDATES, float)
    cyl = [metrics[f"cyl_ns{n}"]["slope_ez_1_10_mm_s"] for n in CANDIDATES]
    box = [metrics[f"box_ns{n}"]["slope_ez_1_10_mm_s"] for n in CANDIDATES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, cyl, "o-", label="cylinder")
    ax.plot(xs, box, "s-", label="box")
    ax.axhline(PREFERRED_SLOPE_MM_S, ls=":", color="k", label="preferred 0.05")
    ax.axhline(LOOSE_SLOPE_MM_S, ls="--", color="0.5", label="loose 0.10")
    ax.set_xlabel("noslip_iterations")
    ax.set_ylabel("secure creep mm/s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_creep_vs_noslip.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [abs(slip[n]["d_ez_10s_mm"]) for n in CANDIDATES], "o-", label="clear-slip |d ez| mm")
    ax.plot(xs, [abs(pivot[n]["d_ez_10s_mm"]) + abs(pivot[n]["d_ex_mm"]) for n in CANDIDATES], "s-", label="tilt |d e| mm")
    ax.set_xlabel("noslip_iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_slip_vs_noslip.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [walls[n] for n in CANDIDATES], "o-")
    ax.set_xlabel("noslip_iterations")
    ax.set_ylabel("10s cylinder hold wall-clock s")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_runtime.png", dpi=130)
    plt.close(fig)

    for ns in (0, 1, 2, 3, 5):
        if not (RAW / f"cyl_ns{ns}.npz").exists():
            continue
        z = np.load(RAW / f"cyl_ns{ns}.npz")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(z["t_hold"], 1e3 * (z["e_z"] - z["e_z"][0]), label=f"cyl ns={ns}")
        if (RAW / f"box_ns{ns}.npz").exists():
            b = np.load(RAW / f"box_ns{ns}.npz")
            ax.plot(b["t_hold"], 1e3 * (b["e_z"] - b["e_z"][0]), label=f"box ns={ns}")
        ax.set_ylabel("Delta r_h.z mm")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_traj_ns{ns}.png", dpi=120)
        plt.close(fig)


def write_md(api, grip, metrics, slip, release, pivot, impact, choice, walls, tilt_meta, clear_tau):
    p = OUT / "NOSLIP_CALIBRATION.md"
    L = []
    a = L.append
    a("# NoSlip calibration (MuJoCo 3.12.0)")
    a("")
    a("Official XML not modified. solref left at live baseline 0.015. noslip_tolerance not swept.")
    a("")
    a("## VERIFIED API / IMPLEMENTATION FACT")
    a("")
    a("Sources: installed MuJoCo **3.12.0**; `engine_forward.c` (constraint solve); `engine_solver.c` `solNoSlip`; modeling docs.")
    a("")
    a("- Live before intervention: " + json.dumps(api))
    a("- NoSlip runs **after** the main solver (Newton here). `mj_solNoSlip` then `mj_dualFinish` because noslip>0 forces the dual mapping.")
    a("- It updates **friction-related dual variables** only: dry friction rows `ne..ne+nf` and contact friction. Equality/limit (`ne`) are not the target of the friction sweeps. For **pyramidal** contacts (`condim=4` → `dim=4` → `2*(dim-1)=6` pyramid-edge rows) it holds the pair midpoint (`mid=(f0+f1)/2`, the mixed-normal content) and optimizes the difference `y` (friction) inside `[-mid,mid]`.")
    a("- Regularization is **ignored** in this step: `ARdiaginv(..., flg_subR=1)` and `extractBlock(..., flg_subR=1)` use `AR - R` on the diagonal (docs: “ignores constraint regularization”).")
    a("- Stop: `improvement < opt.noslip_tolerance` after scaling. Default live tolerance recorded above.")
    a("- Docs caveat: this cascade is **ad hoc**, not a well-defined optimization problem; possible instability with many interacting contacts. Cost: extra PGS-like sweeps over friction rows, linear in `noslip_iterations` × number of frictional constraints.")
    a("")
    a("## Predefined secure-creep targets (set before the sweep)")
    a("")
    a(f"- Preferred: |slope| ≤ {PREFERRED_SLOPE_MM_S} mm/s and |Δez| ≤ {PREFERRED_D10_MM} mm / 10 s, **both** geometries.")
    a(f"- Loose: ≤ {LOOSE_SLOPE_MM_S} mm/s and ≤ {LOOSE_D10_MM} mm / 10 s.")
    a("- These are simulation numerical floors, not robot tolerances.")
    a("")
    a("## Test C grip sweep at noslip=0 (RAW)")
    a("")
    a("| tau | slope mm/s | d ez mm | med vz | rho | nL nR | drop | class |")
    a("|---|---|---|---|---|---|---|---|")
    for g in grip:
        a(
            f"| {g['tau']} | {g['slope_ez_1_10_mm_s']:.3f} | {g['d_ez_10s_mm']:.2f} | {g['median_vz']:.4f} | "
            f"{g['rho_mean_med']:.3f} | {g['nL_range']}/{g['nR_range']} | {g['dropped_or_both_lost']} | {g.get('cls','')} |"
        )
    a(f"")
    a(f"Selected CLEAR-SLIP tau = **{clear_tau}**.")
    a("")
    a("## Summary table (RAW)")
    a("")
    a("| noslip | cyl slope / d10 | box slope / d10 | clear-slip d_ez / drop | release d_obj_z / drop | tilt |d e| mm | cyl 10s wall s |")
    a("|---|---|---|---|---|---|---|")
    for ns in CANDIDATES:
        a_ = metrics[f"cyl_ns{ns}"]
        b = metrics[f"box_ns{ns}"]
        s = slip[ns]
        r = release[ns]
        pv = pivot[ns]
        a(
            f"| {ns} | {a_['slope_ez_1_10_mm_s']:.4f} / {a_['d_ez_10s_mm']:.3f} | "
            f"{b['slope_ez_1_10_mm_s']:.4f} / {b['d_ez_10s_mm']:.3f} | "
            f"{s['d_ez_10s_mm']:.2f} / {s['dropped_or_both_lost']} | "
            f"{r['d_obj_z_mm']:.1f} / {r['dropped_or_both_lost']} | "
            f"{abs(pv['d_ex_mm'])+abs(pv['d_ez_10s_mm']):.2f} | {walls[ns]:.3f} |"
        )
    a("")
    a("Tilt IC (noslip=0 capture): " + json.dumps(tilt_meta, default=str))
    a("")
    a("## Impact probe (RAW)")
    a("")
    a("```")
    a(json.dumps(impact, indent=2, default=str)[:6000])
    a("```")
    a("")
    a("## INTERPRETATION / RECOMMENDATION")
    a("")
    a(f"**{choice}**")
    a("")
    a("Not written into official XML. Review before any model change.")
    a("")
    a("## Viewer")
    a("")
    a("```")
    a("python training/noslip_calibration.py --view secure --noslip 0 --playback-speed 0.35")
    a("python training/noslip_calibration.py --view secure --noslip CAND --playback-speed 0.35")
    a("python training/noslip_calibration.py --view pivot --noslip 0 --playback-speed 0.35")
    a("python training/noslip_calibration.py --view pivot --noslip CAND --playback-speed 0.35")
    a("```")
    p.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=("secure", "pivot"), default=None)
    p.add_argument("--noslip", type=int, default=0)
    p.add_argument("--playback-speed", type=float, default=0.35)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.view:
        run_viewer(args.view, int(args.noslip), args.playback_speed)
        return 0

    print("parents ...", flush=True)
    snap_c, api = capture_parent("cylinder")
    snap_b, _ = capture_parent("box")
    dump_json(RAW / "api_live.json", api)
    print("live", api, flush=True)

    sim_c = make_shape("cylinder")
    sim_b = make_shape("box")

    print("grip sweep ns=0 ...", flush=True)
    grip = []
    for tau in GRIP_SWEEP:
        rows, _ = hold_tau(sim_c, snap_c, 0, tau, GRIP_SWEEP_S)
        s = summarize_hold(rows)
        s["tau"] = tau
        if s["dropped_or_both_lost"] or s["nL_range"][1] == 0:
            s["cls"] = "released" if tau >= 0 else "clearly slipping"
        elif abs(s["median_vz"]) > 0.005 or abs(s["d_ez_10s_mm"]) > 8:
            s["cls"] = "clearly slipping"
        elif abs(s["d_ez_10s_mm"]) > 2:
            s["cls"] = "marginal"
        else:
            s["cls"] = "secure"
        grip.append(s)
        print("  tau", tau, s["cls"], "dez", round(s["d_ez_10s_mm"], 2), "vz", round(s["median_vz"], 4), flush=True)
    dump_json(RAW / "grip_sweep.json", grip)
    clear = next((g for g in grip if g["cls"] == "clearly slipping"), None)
    if clear is None:
        clear = next((g for g in reversed(grip) if g["cls"] != "secure"), grip[-2])
    clear_tau = float(clear["tau"])
    print("CLEAR-SLIP tau", clear_tau, flush=True)

    print("tilt IC ...", flush=True)
    snap_tilt, tilt_meta = capture_tilt60(snap_c)
    dump_json(RAW / "tilt_ic.json", tilt_meta)

    metrics = {}
    walls = {}
    slip = {}
    release = {}
    pivot = {}
    for ns in CANDIDATES:
        print("A cyl ns", ns, "...", flush=True)
        rows, wall = hold_tau(sim_c, snap_c, ns, TAU, HOLD_S)
        save_npz(f"cyl_ns{ns}", rows)
        metrics[f"cyl_ns{ns}"] = summarize_hold(rows, wall)
        walls[ns] = wall
        print("  slope", round(metrics[f"cyl_ns{ns}"]["slope_ez_1_10_mm_s"], 4), "wall", round(wall, 3), flush=True)

        print("B box ns", ns, "...", flush=True)
        rows, _ = hold_tau(sim_b, snap_b, ns, TAU, HOLD_S)
        save_npz(f"box_ns{ns}", rows)
        metrics[f"box_ns{ns}"] = summarize_hold(rows)
        print("  slope", round(metrics[f"box_ns{ns}"]["slope_ez_1_10_mm_s"], 4), flush=True)

        print("C slip ns", ns, "...", flush=True)
        rows, _ = hold_tau(sim_c, snap_c, ns, clear_tau, GRIP_SWEEP_S)
        save_npz(f"slip_ns{ns}", rows)
        slip[ns] = summarize_hold(rows)

        print("D release ns", ns, "...", flush=True)
        rows, _ = hold_tau(sim_c, snap_c, ns, RELEASE_TAU, RELEASE_S)
        save_npz(f"rel_ns{ns}", rows)
        release[ns] = summarize_hold(rows)

        print("E pivot ns", ns, "...", flush=True)
        rows, _ = hold_tau(sim_c, snap_tilt, ns, TAU, PIVOT_HOLD_S)
        save_npz(f"pivot_ns{ns}", rows)
        pivot[ns] = summarize_hold(rows)

    # maybe extend to 10
    c5, b5, c3, b3 = metrics["cyl_ns5"], metrics["box_ns5"], metrics["cyl_ns3"], metrics["box_ns3"]
    if abs(c5["slope_ez_1_10_mm_s"] - c3["slope_ez_1_10_mm_s"]) > 0.02 or abs(
        b5["slope_ez_1_10_mm_s"] - b3["slope_ez_1_10_mm_s"]
    ) > 0.02:
        ns = 10
        CANDIDATES.append(10)
        print("extend ns=10 ...", flush=True)
        rows, wall = hold_tau(sim_c, snap_c, ns, TAU, HOLD_S)
        save_npz("cyl_ns10", rows)
        metrics["cyl_ns10"] = summarize_hold(rows, wall)
        walls[10] = wall
        rows, _ = hold_tau(sim_b, snap_b, ns, TAU, HOLD_S)
        save_npz("box_ns10", rows)
        metrics["box_ns10"] = summarize_hold(rows)
        rows, _ = hold_tau(sim_c, snap_c, ns, clear_tau, GRIP_SWEEP_S)
        save_npz("slip_ns10", rows)
        slip[10] = summarize_hold(rows)
        rows, _ = hold_tau(sim_c, snap_c, ns, RELEASE_TAU, RELEASE_S)
        save_npz("rel_ns10", rows)
        release[10] = summarize_hold(rows)
        rows, _ = hold_tau(sim_c, snap_tilt, ns, TAU, PIVOT_HOLD_S)
        save_npz("pivot_ns10", rows)
        pivot[10] = summarize_hold(rows)

    chosen, why, acceptable, _ = select(metrics, slip, release, pivot)

    impact = {}
    f_ns = [0]
    if chosen is not None:
        f_ns.append(int(chosen))
        if chosen != 1 and 1 in CANDIDATES and 1 not in f_ns:
            pass
    else:
        f_ns.append(2 if 2 in CANDIDATES else CANDIDATES[-1])
    f_ns = sorted(set(f_ns))[:2]
    for ns in f_ns:
        print("F impact ns", ns, "...", flush=True)
        try:
            impact[str(ns)] = impact_probe(ns)
        except Exception as e:
            impact[str(ns)] = {"error": str(e), "noslip": ns}

    # case text
    if chosen is not None:
        rec = (
            f"CASE A — SMALL NOSLIP VALUE WORKS. Candidate **noslip_iterations = {chosen}** "
            f"(smallest among {acceptable} that meet preferred creep on cyl+box, still slip/release, tilt not qualitatively killed). "
            "NOT an official XML change."
        )
    else:
        # classify B/C/D/E
        cyl_ok = [n for n in CANDIDATES if n > 0 and meets_pref(metrics[f"cyl_ns{n}"])]
        box_ok = [n for n in CANDIDATES if n > 0 and meets_pref(metrics[f"box_ns{n}"])]
        if cyl_ok and not box_ok:
            rec = "CASE D — works for one geometry only. No recommendation of a large value to force the other."
        elif (cyl_ok or box_ok) and any(not slip[n]["dropped_or_both_lost"] and abs(slip[n]["median_vz"]) < 0.002 for n in (cyl_ok or box_ok)):
            rec = "CASE B — NoSlip also suppresses intended under-supported slip. Unsuitable as a clean recovery-physics setting."
        elif not cyl_ok and not box_ok:
            rec = "CASE C — tested noslip_iterations did not bring both geometries to the preferred floor. Do not raise solref in this audit; noslip_tolerance not swept."
        else:
            rec = f"CASE E / mixed. why={why} acceptable={acceptable}. No official recommendation."

    dump_json(RAW / "metrics_secure.json", metrics)
    dump_json(RAW / "metrics_slip.json", slip)
    dump_json(RAW / "metrics_release.json", release)
    dump_json(RAW / "metrics_pivot.json", pivot)
    dump_json(RAW / "impact.json", impact)
    dump_json(RAW / "choice.json", {"chosen": chosen, "acceptable": acceptable, "rec": rec, "clear_tau": clear_tau})

    figures(metrics, slip, release, pivot, walls)
    write_md(api, grip, metrics, slip, release, pivot, impact, rec, walls, tilt_meta, clear_tau)
    print("CHOICE", rec)
    print("wrote", OUT / "NOSLIP_CALIBRATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
