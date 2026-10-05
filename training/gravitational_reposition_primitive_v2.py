"""Gravitational reposition primitive v2 under frozen noslip_iterations=1.

Simple airborne centered grasp. No impact, teleport, s=2, RULE, or SAC.
Does not return-to-vertical or claim recovery success.
"""

from __future__ import annotations

import argparse
import json
import pickle
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
from controllers.nominal import grasp_orientation
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import TABLE_DROP, physical_pack
from training.demo_teleport_recovery_state import (
    PAIR_MU,
    CYL_MASS,
    advance_to_parent,
    make_parent_sim,
    signed_rot_about_y_deg,
)
from training.nominal_grasp_creep_root_cause import dump_json, robust_slope
from training.grav_reposition_v2_viz import (
    END_HOLD_S,
    PLAYBACK_SPEED,
    PRE_SLIP_PAUSE_S,
    PRE_VIEW_PAUSE_S,
    capture_hold_refs,
    init_camera_once,
    make_reset_cam_callback,
    viz_frame,
    wall_pause,
)
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "gravitational_reposition_primitive_v2"
RAW = OUT / "raw"
FIG = OUT / "figures"

OMEGA = 1.2
HOLD_S = 1.0
LOG_DT = 0.010
TAU_SECURE = -18.0
COARSE_TAUS = [-18.0, -8.0, -5.0, -3.0, -2.0, -1.0]
ANGLES = [30.0, 45.0, 60.0]
USEFUL_T = 0.30
USEFUL_MM = 1.0
STICK_MM = 0.50


def live_opt(sim) -> dict:
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
        "gravity": np.array(m.opt.gravity, float).tolist(),
        "solref_object": np.array(m.geom_solref[sim.ids.object_geom], float).tolist(),
    }


def restore(sim, snap) -> None:
    sim.load_snapshot(snap)
    set_finger_object_sliding_mu(sim.model, sim.ids, float(snap.get("friction", PAIR_MU)), data=None)
    mujoco.mj_forward(sim.model, sim.data)


def force_stats(rows) -> dict:
    if not rows:
        return {
            "rho_max": float("nan"),
            "rho_med": float("nan"),
            "Fn_L": 0.0,
            "Fn_R": 0.0,
            "Ft": 0.0,
        }
    rhos = [r["rho"] for r in rows if np.isfinite(r.get("rho", np.nan))]
    fnL = sum(abs(r["Fn"]) for r in rows if r["side"] == "L")
    fnR = sum(abs(r["Fn"]) for r in rows if r["side"] == "R")
    ft = sum(float(r["Ft"]) for r in rows)
    return {
        "rho_max": float(np.max(rhos)) if rhos else float("nan"),
        "rho_med": float(np.median(rhos)) if rhos else float("nan"),
        "Fn_L": float(fnL),
        "Fn_R": float(fnR),
        "Ft": float(ft),
    }


def sample(sim, tau: float) -> dict:
    rows, sm = extract_contacts(sim)
    o = physical_pack(sim)
    st = force_stats(rows)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rh = Rh.T @ (po - ph)
    g_w = np.array(sim.model.opt.gravity, float)
    g_h = Rh.T @ g_w
    return {
        "t": float(sim.data.time),
        "rh": rh,
        "v_rel_h": np.array(o["v_rel_h"], float),
        "w_rel_h": np.array(o["w_rel_h"], float),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "aperture": float(o["aperture"]),
        "ctrl": float(sim.data.ctrl[7]),
        "tau": float(tau),
        "g_h": g_h,
        "g_w": g_w,
        "wrist_deg": None,
        **st,
    }


def intended_dir(g_h: np.ndarray) -> np.ndarray:
    """Unit vector of measured g_h in the hand x–z plane.

    This is a *progress* direction: it is chosen per trial from that trial's
    g_h so that gravity-aligned motion makes progress_s positive. It is NOT a
    fixed-frame signed coordinate. Directional claims must use Delta r_h.
    """
    u = np.array([float(g_h[0]), 0.0, float(g_h[2])])
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return u / n


def _wrist_now(sim, Rh0) -> float:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return float(signed_rot_about_y_deg(Rh0, Rh))


def _emit_viz(viz, sim, *, phase, tau, Rh0, rh0, progress_mm, s_now):
    viz_frame(
        viz,
        sim,
        phase=phase,
        wrist_deg=_wrist_now(sim, Rh0),
        tau=tau,
        rho=float(s_now["rho_max"]),
        drh_mm=(np.array(s_now["rh"]) - rh0) * 1e3,
        progress_mm=progress_mm,
        v_rel_h=s_now["v_rel_h"],
        nL=s_now["nL"],
        nR=s_now["nR"],
    )


def rotate_wrist(sim, cfg, snap, *, target_deg: float, tau: float, gains, viewer=None, viz=None):
    restore(sim, snap)
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    dt = float(sim.model.opt.timestep)
    sign = 1.0 if target_deg >= 0 else -1.0
    goal = abs(float(target_deg))
    t0 = float(sim.data.time)
    timeout = t0 + goal / OMEGA + 1.5
    viz = viz if viz is not None else ({"viewer": viewer} if viewer is not None else None)
    while float(sim.data.time) < timeout:
        Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        ang = abs(signed_rot_about_y_deg(Rh0, Rh))
        if ang >= goal - 0.3:
            break
        w_w = Rh @ np.array([0.0, sign * OMEGA, 0.0])
        tick_vw(sim, np.zeros(3), w_w, tau, gains)
        if viz is not None:
            s = sample(sim, tau)
            rh = np.array(s["rh"], float)
            _emit_viz(viz, sim, phase="TILT", tau=tau, Rh0=Rh0, rh0=rh, progress_mm=0.0, s_now=s)
    freeze(sim)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return {
        "wrist_act_deg": float(signed_rot_about_y_deg(Rh0, Rh)),
        "Rh0": Rh0,
        "snap": sim.snapshot(),
        "tilt_s": float(sim.data.time - t0),
        "dt": dt,
    }


def hold_window(
    sim,
    cfg,
    *,
    tau: float,
    hold_s: float,
    Rh0: np.ndarray,
    gains,
    brake_after_mm: float | None = None,
    brake_tau: float = TAU_SECURE,
    viewer=None,
    viz=None,
    phase: str = "HOLD",
    post_brake_s: float | None = None,
):
    freeze(sim)
    dt = float(sim.model.opt.timestep)
    n = int(round(hold_s / dt))
    log_n = max(int(round(LOG_DT / dt)), 1)
    rows_log = []
    s0 = sample(sim, tau)
    g_h0 = np.array(s0["g_h"], float)
    u = intended_dir(g_h0)
    rh0 = np.array(s0["rh"], float)
    viz = viz if viz is not None else ({"viewer": viewer} if viewer is not None else None)
    if viz is not None:
        viz["refs"] = capture_hold_refs(sim, rh0, u)
    t_loss = None
    t_both_loss = None
    t_drop = None
    t_useful = None
    t_brake = None
    braked = False
    rh_brake = None
    smm_brake = None
    drh_brake = None
    cur_tau = float(tau)
    t_start = float(sim.data.time)
    i = 0
    while True:
        tick_vw(sim, np.zeros(3), np.zeros(3), cur_tau, gains)
        t = float(sim.data.time)
        s_now = sample(sim, cur_tau)
        smm = float((np.array(s_now["rh"]) - rh0) @ u * 1e3)
        if braked:
            if viz is not None:
                t0w = viz.get("_brake_wall0")
                if t0w is None:
                    viz["_brake_wall0"] = time.perf_counter()
                    t0w = viz["_brake_wall0"]
                phase_show = "BRAKE" if (time.perf_counter() - t0w) < 0.50 else "SECURE"
            else:
                dt_b = (t - t_start) - float(t_brake)
                phase_show = "BRAKE" if dt_b < 0.15 else "SECURE"
        else:
            phase_show = phase
        bilateral = int(s_now["nL"]) > 0 and int(s_now["nR"]) > 0
        if t_loss is None and not bilateral:
            t_loss = t - t_start
        if t_both_loss is None and int(s_now["nL"]) == 0 and int(s_now["nR"]) == 0:
            t_both_loss = t - t_start
        if t_drop is None and float(s_now["obj_z"]) < TABLE_DROP:
            t_drop = t - t_start
        if t_useful is None and bilateral and abs(smm) >= USEFUL_MM:
            t_useful = t - t_start
        if (
            brake_after_mm is not None
            and not braked
            and bilateral
            and abs(smm) >= float(brake_after_mm)
        ):
            braked = True
            t_brake = t - t_start
            rh_brake = np.array(s_now["rh"], float).copy()
            smm_brake = smm
            drh_brake = (rh_brake - rh0) * 1e3
            cur_tau = float(brake_tau)
            freeze(sim)
            if viz is not None:
                viz["_brake_wall0"] = time.perf_counter()
                if viz.get("refs") is not None:
                    viz["refs"]["rh_brake"] = rh_brake.copy()
            phase_show = "BRAKE"
        i += 1
        elapsed = t - t_start
        if braked and post_brake_s is not None:
            done = (elapsed - float(t_brake)) >= float(post_brake_s)
        else:
            done = i >= n
        if ((i - 1) % log_n == 0) or done:
            s = dict(s_now)
            s["wrist_deg"] = _wrist_now(sim, Rh0)
            s["s_mm"] = smm
            s["progress_mm"] = smm
            s["drh_mm"] = ((np.array(s["rh"]) - rh0) * 1e3)
            s["phase"] = phase_show
            rows_log.append(s)
        if viz is not None:
            _emit_viz(
                viz, sim, phase=phase_show, tau=cur_tau, Rh0=Rh0, rh0=rh0,
                progress_mm=smm, s_now=s_now,
            )
        if done or elapsed > 15.0:
            break
    t_arr = np.array([r["t"] - t_start for r in rows_log], float)
    s_arr = np.array([r["s_mm"] for r in rows_log], float)
    nL = np.array([r["nL"] for r in rows_log], int)
    nR = np.array([r["nR"] for r in rows_log], int)
    bil = (nL > 0) & (nR > 0)
    if np.any(bil):
        t_bil = float(t_arr[bil][-1] - t_arr[bil][0] + LOG_DT) if t_arr.size else 0.0
        # contiguous from start
        if bil[0]:
            k = int(np.argmin(bil) if not np.all(bil) else len(bil) - 1)
            if not bil[k]:
                k = max(k - 1, 0)
            t_bil_from_start = float(t_arr[k])
        else:
            t_bil_from_start = 0.0
    else:
        t_bil = 0.0
        t_bil_from_start = 0.0
    rho_med = float(np.nanmedian([r["rho_max"] for r in rows_log]))
    rho_max = float(np.nanmax([r["rho_max"] for r in rows_log]))
    v_med = float(np.median([np.linalg.norm(r["v_rel_h"]) for r in rows_log]))
    s_end = float(s_arr[-1]) if s_arr.size else 0.0
    s_at_bil = float(s_arr[bil][-1]) if np.any(bil) else float(s_end)
    dropped = t_drop is not None
    regime = classify(
        t_bil_from_start=t_bil_from_start,
        s_mm=s_at_bil,
        rho_med=rho_med,
        v_med=v_med,
        dropped=dropped,
        t_both_loss=t_both_loss,
        sign_ok=s_end * float(np.sign(s_end) or 1) >= 0,  # filled below
        u=u,
        s_end=s_end,
    )
    # gravity-consistent: positive s is along g_h in xz
    sign_ok = s_end >= -0.2  # allow tiny numerical reverse
    if regime == "CONTROLLED_SLIP" and s_end < 0.5:
        # motion opposite intended gravity direction
        regime = "LOSS"
        note = "motion_not_gravity_consistent"
    else:
        note = ""
    last = rows_log[-1] if rows_log else s0
    return {
        "rows": rows_log,
        "t_start": t_start,
        "g_h": g_h0.tolist(),
        "g_h_tangential_xz": float(np.linalg.norm([g_h0[0], g_h0[2]])),
        "g_hx": float(g_h0[0]),
        "g_hz": float(g_h0[2]),
        "u": u.tolist(),
        "s_end_mm": s_end,
        "s_at_bil_mm": s_at_bil,
        "t_bil_from_start": t_bil_from_start,
        "t_loss": t_loss,
        "t_both_loss": t_both_loss,
        "t_drop": t_drop,
        "t_useful": t_useful,
        "t_brake": t_brake,
        "progress_s_at_brake_mm": None if smm_brake is None else float(smm_brake),
        "drh_at_brake_mm": None if drh_brake is None else np.asarray(drh_brake, float).tolist(),
        "rh_brake": None if rh_brake is None else np.asarray(rh_brake, float).tolist(),
        "rho_med": rho_med,
        "rho_max": rho_max,
        "v_rel_med": v_med,
        "wrist_end_deg": float(last.get("wrist_deg", 0.0) or 0.0),
        "obj_z_end": float(last["obj_z"]),
        "nL_end": int(last["nL"]),
        "nR_end": int(last["nR"]),
        "Fn_L_end": float(last["Fn_L"]),
        "Fn_R_end": float(last["Fn_R"]),
        "aperture_end": float(last["aperture"]),
        "ctrl_end": float(last["ctrl"]),
        "regime": regime,
        "note": note,
        "sign_ok": bool(s_end >= -0.2),
        "dropped": bool(dropped),
        "rh0": rh0.tolist(),
        "drh_end_mm": ((np.array(last["rh"]) - rh0) * 1e3).tolist(),
        "progress_s_def": (
            "progress_s = dot(Delta r_h, u_hat)*1e3 mm; "
            "u_hat = g_h_xz/|g_h_xz| at hold start (trial-dependent progress, not a fixed signed axis)"
        ),
    }


def classify(*, t_bil_from_start, s_mm, rho_med, v_med, dropped, t_both_loss, sign_ok, u, s_end):
    if dropped or (t_both_loss is not None and t_both_loss < 0.15):
        return "LOSS"
    if t_bil_from_start >= USEFUL_T and abs(s_mm) >= USEFUL_MM and s_end > 0.5:
        return "CONTROLLED_SLIP"
    if t_bil_from_start < 0.15 and abs(s_mm) < USEFUL_MM:
        return "LOSS"
    if abs(s_mm) < STICK_MM and t_bil_from_start >= 0.8 * HOLD_S and (not np.isfinite(rho_med) or rho_med < 0.55):
        return "STICK"
    if abs(s_mm) < STICK_MM and t_bil_from_start >= 0.8 * HOLD_S:
        return "STICK"
    if t_bil_from_start >= USEFUL_T and abs(s_mm) >= USEFUL_MM:
        return "CONTROLLED_SLIP"
    if t_both_loss is not None or t_bil_from_start < USEFUL_T:
        return "LOSS"
    return "STICK"


def slim_hold(h: dict) -> dict:
    out = {k: v for k, v in h.items() if k != "rows"}
    rows = h.get("rows") or []
    out["n_log"] = len(rows)
    if rows:
        out["s_series_mm"] = [float(r["s_mm"]) for r in rows]
        out["t_series"] = [float(r["t"] - h["t_start"]) for r in rows]
        out["rho_series"] = [float(r["rho_max"]) for r in rows]
        out["nL_series"] = [int(r["nL"]) for r in rows]
        out["nR_series"] = [int(r["nR"]) for r in rows]
    return out


def save_traj(path: Path, h: dict) -> None:
    rows = h["rows"]
    if not rows:
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows]),
        rh=np.stack([r["rh"] for r in rows]),
        v_rel_h=np.stack([r["v_rel_h"] for r in rows]),
        w_rel_h=np.stack([r["w_rel_h"] for r in rows]),
        nL=np.array([r["nL"] for r in rows]),
        nR=np.array([r["nR"] for r in rows]),
        rho=np.array([r["rho_max"] for r in rows]),
        Fn_L=np.array([r["Fn_L"] for r in rows]),
        Fn_R=np.array([r["Fn_R"] for r in rows]),
        Ft=np.array([r["Ft"] for r in rows]),
        s_mm=np.array([r["s_mm"] for r in rows]),
        progress_mm=np.array([r.get("progress_mm", r["s_mm"]) for r in rows]),
        drh_mm=np.stack([np.asarray(r["drh_mm"], float) for r in rows]),
        obj_z=np.array([r["obj_z"] for r in rows]),
        aperture=np.array([r["aperture"] for r in rows]),
        ctrl=np.array([r["ctrl"] for r in rows]),
        g_h=np.stack([r["g_h"] for r in rows]),
        wrist_deg=np.array([r["wrist_deg"] for r in rows], float),
    )


def midpoint_taus(a: float, b: float) -> list[float]:
    lo, hi = (a, b) if a < b else (b, a)
    span = hi - lo
    if span < 0.35:
        return []
    if span <= 1.2:
        return [0.5 * (lo + hi)]
    return [lo + span / 3.0, lo + 2.0 * span / 3.0]


def run_cell(sim, cfg, parent, *, angle, tau, gains, g_scale=1.0, noslip=None, viewer=None, viz=None):
    g0 = np.array(sim.model.opt.gravity, float).copy()
    ns0 = int(sim.model.opt.noslip_iterations)
    try:
        if noslip is not None:
            sim.model.opt.noslip_iterations = int(noslip)
        sim.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81 * float(g_scale)])
        tilt = rotate_wrist(
            sim, cfg, parent, target_deg=angle, tau=TAU_SECURE, gains=gains, viewer=viewer, viz=viz
        )
        h = hold_window(
            sim, cfg, tau=tau, hold_s=HOLD_S, Rh0=tilt["Rh0"], gains=gains, viewer=viewer, viz=viz
        )
        h["angle_cmd"] = float(angle)
        h["tau"] = float(tau)
        h["g_scale"] = float(g_scale)
        h["noslip"] = int(sim.model.opt.noslip_iterations)
        h["wrist_act_deg"] = float(tilt["wrist_act_deg"])
        h["tilt_s"] = float(tilt["tilt_s"])
        return h, tilt
    finally:
        sim.model.opt.gravity[:] = g0
        sim.model.opt.noslip_iterations = ns0


def phase_plot(cells, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    cmap = {"STICK": "C0", "CONTROLLED_SLIP": "C2", "LOSS": "C3"}
    for c in cells:
        ax.scatter(
            -float(c["tau"]),
            abs(float(c["angle_cmd"])),
            c=cmap.get(c["regime"], "gray"),
            s=90,
            edgecolors="k",
            zorder=3,
        )
        ax.annotate(
            f"{c['regime'][:4]}\nρ={c['rho_med']:.2f}\nΔ={c['s_end_mm']:.1f}mm",
            (-float(c["tau"]), abs(float(c["angle_cmd"]))),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
        )
    ax.set_xlabel("|grip τ| (N·m command; larger = tighter)")
    ax.set_ylabel("commanded wrist tilt (deg)")
    ax.set_title("Gravitational reposition v2 phase diagram (noslip=1)")
    ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[k], markeredgecolor="k", markersize=10, label=k)
        for k in ("STICK", "CONTROLLED_SLIP", "LOSS")
    ]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def traj_plot(h: dict, path: Path, title: str) -> None:
    rows = h["rows"]
    t = np.array([r["t"] - h["t_start"] for r in rows])
    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, [r["s_mm"] for r in rows])
    ax[0].set_ylabel("s along g_h xz (mm)")
    ax[1].plot(t, [np.linalg.norm(r["v_rel_h"]) * 1e3 for r in rows])
    ax[1].set_ylabel("|v_rel| (mm/s)")
    ax[2].plot(t, [r["rho_max"] for r in rows])
    ax[2].set_ylabel("rho max")
    ax[3].plot(t, [r["nL"] for r in rows], label="nL")
    ax[3].plot(t, [r["nR"] for r in rows], label="nR")
    ax[3].legend()
    ax[3].set_ylabel("contacts")
    ax[3].set_xlabel("t (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(meta: dict) -> None:
    md = OUT / "GRAVITATIONAL_REPOSITION_PRIMITIVE_V2.md"
    lines = []
    a = lines.append
    a("# Gravitational reposition primitive v2")
    a("")
    a("**noslip_iterations = 1** (frozen contact-model correction).")
    a("")
    a("Old low-ρ secure-grip inward slide (60° + τ=-18, ρ≈0.10, ~1.8 mm / 2 s) is **SUPERSEDED BY NOSLIP CALIBRATION**. It is not physical recovery authority.")
    a("")
    a("This experiment does **not** apply the primitive to coupled teleport s=2.0. No SAC. No RULE retune.")
    a("")
    a("## VERIFIED CODE FACT")
    a("")
    a("```json")
    a(json.dumps(meta["live_opt"], indent=2))
    a("```")
    a("")
    a(f"- Canonical freeze: `config/sim.yaml` `noslip_iterations: {meta['cfg_noslip']}`, `noslip_tolerance: {meta['cfg_tol']}`.")
    a("- Applied in `envs/grasp_sim.py` `apply_solver_from_cfg` after XML load; XML option also set in `envs/xml_build.py` / `assets/panda_torque.xml`.")
    a(f"- Live `model.opt.noslip_iterations` at parent = **{meta['live_opt']['noslip_iterations']}**.")
    a("- Wrist rotation about live hand-y at 1.2 rad/s; `p_des := p_hand`; no table, impact, or teleport.")
    a("- Intended axis û is unit `(g_h.x, 0, g_h.z)` from **measured** `g_h = R_h^T g`, not commanded angle alone.")
    a("")
    a("## RAW TRAJECTORY EVIDENCE")
    a("")
    a("### Baseline (vertical, τ=-18)")
    a("")
    a("```json")
    a(json.dumps(meta["baseline"], indent=2))
    a("```")
    a("")
    a("### Phase table (angle × τ)")
    a("")
    a("| angle_cmd | wrist_act | g_hx | |g_xz| | τ | ρ_med | ρ_max | s_end mm | t_bil | t_loss | regime |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in meta["cells"]:
        a(
            f"| {c['angle_cmd']:.0f} | {c['wrist_act_deg']:.1f} | {c['g_hx']:.3f} | {c['g_h_tangential_xz']:.3f} | "
            f"{c['tau']:.2f} | {c['rho_med']:.3f} | {c['rho_max']:.3f} | {c['s_end_mm']:.2f} | "
            f"{c['t_bil_from_start']:.3f} | {c['t_loss']} | {c['regime']} |"
        )
    a("")
    a("Useful controlled-slip diagnostic (not task success): bilateral ≥ 0.30 s and |Δ| ≥ 1.0 mm, no drop.")
    a("")
    a(f"Controlled-slip cells: **{meta['n_slip']}**. Stick: **{meta['n_stick']}**. Loss: **{meta['n_loss']}**.")
    a(f"Authority: **{meta['authority']}**.")
    a("")
    a("### Directional control (±tilt)")
    a("")
    a("```json")
    a(json.dumps(meta["direction"], indent=2))
    a("```")
    a("")
    a("### Gravity causality (1g vs 0g)")
    a("")
    a("```json")
    a(json.dumps(meta["gravity"], indent=2))
    a("```")
    a("")
    a("### noslip=1 vs noslip=0 on selected candidate")
    a("")
    a("```json")
    a(json.dumps(meta["noslip_cmp"], indent=2))
    a("```")
    a("")
    a("### SLIP → SECURE brake")
    a("")
    a("```json")
    a(json.dumps(meta["brake"], indent=2))
    a("```")
    a("")
    a("### Small robustness (fixed τ, no per-condition retune)")
    a("")
    a("```json")
    a(json.dumps(meta["robust"], indent=2))
    a("```")
    a("")
    a("## USER VISUAL OBSERVATION")
    a("")
    a("Viewer commands (smooth playback, no early terminate):")
    a("")
    a("```text")
    for cmd in meta["viewer_cmds"]:
        a(cmd)
    a("```")
    a("")
    a("Overlay: wrist angle, τ, ρ, relative displacement along g_h xz, contact L/R, phase.")
    a("This section is a command list. Cursor text is not a substitute for watching the viewer.")
    a("")
    a("## INTERPRETATION")
    a("")
    a(meta["interpretation"])
    a("")
    a("Not a recovery-policy verdict. Disturbed s=2.0 was not tested.")
    a("")
    md.write_text("\n".join(lines), encoding="utf-8")


def pick_representatives(cells):
    stick = next((c for c in cells if c["regime"] == "STICK"), None)
    slip = None
    slips = [c for c in cells if c["regime"] == "CONTROLLED_SLIP"]
    if slips:
        slips.sort(key=lambda c: (c["t_bil_from_start"], abs(c["s_end_mm"])), reverse=True)
        slip = slips[0]
    loss = next((c for c in cells if c["regime"] == "LOSS"), None)
    return stick, slip, loss


def interpret(meta) -> str:
    n_slip = meta["n_slip"]
    angles_slip = sorted({c["angle_cmd"] for c in meta["cells"] if c["regime"] == "CONTROLLED_SLIP"})
    taus_slip = sorted({c["tau"] for c in meta["cells"] if c["regime"] == "CONTROLLED_SLIP"})
    d = meta["direction"]
    g = meta["gravity"]
    ns = meta["noslip_cmp"]
    bits = []
    if n_slip == 0:
        bits.append("No CONTROLLED_SLIP cell met the diagnostic thresholds under noslip=1.")
        meta["authority"] = "NO_CONTROLLED_SLIP_WINDOW"
    elif n_slip == 1:
        bits.append("CONTROLLED_SLIP appears at a single (angle, τ) cell. Treat as FRAGILE AUTHORITY until a region is shown.")
        meta["authority"] = "FRAGILE AUTHORITY"
    else:
        bits.append(
            f"CONTROLLED_SLIP occupies a region: angles {angles_slip}, τ {taus_slip}."
        )
        meta["authority"] = "REGION PRESENT"
    if d.get("ok") is False:
        bits.append("STOP: ±tilt relative motion did not track g_h. Not gravitational reposition.")
    elif d.get("ok"):
        bits.append("±tilt s_end signs follow g_hx (gravity-consistent direction).")
    if g.get("ok") is False:
        bits.append("STOP: substantial motion remains at 0g; not clean gravitational reposition.")
    elif g.get("ok"):
        bits.append("0g removes the 1g reposition (same causal standard as the creep audit).")
    if ns.get("ok") is False:
        bits.append("Candidate exists only with noslip=1; do not accept until investigated.")
    elif ns.get("ok"):
        bits.append("High-ρ slip remains qualitatively present at noslip=0 and noslip=1.")
    br = meta.get("brake") or {}
    if br.get("ok"):
        bits.append("SLIP→SECURE: relative motion stopped without ejection under the measured brake window.")
    elif br:
        bits.append("Brake transition did not cleanly arrest motion without contact loss.")
    return " ".join(bits)


def run(viewer_mode: str | None = None) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    opt0 = live_opt(sim)
    if int(opt0["noslip_iterations"]) != 1:
        raise RuntimeError(f"expected live noslip_iterations==1, got {opt0}")
    parent_pack = advance_to_parent(sim, cfg)
    if not parent_pack.get("ok"):
        raise RuntimeError(parent_pack)
    freeze(sim)
    parent = sim.snapshot()
    parent["friction"] = float(PAIR_MU)
    gains = gains_from_cfg(cfg)
    viewer = None
    if viewer_mode:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)

    # C1 baseline
    restore(sim, parent)
    freeze(sim)
    Rh0v = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    base = hold_window(sim, cfg, tau=TAU_SECURE, hold_s=HOLD_S, Rh0=Rh0v, gains=gains, viewer=viewer, phase="BASE")
    save_traj(RAW / "baseline_vertical_tau-18.npz", base)
    rh = np.stack([r["rh"] for r in base["rows"]])
    t = np.array([r["t"] for r in base["rows"]])
    slope = robust_slope(t, rh[:, 2] * 1e3, t0=t[0] + 0.2, t1=t[-1])
    baseline = {
        "slope_rh_z_mm_s": float(slope),
        "s_end_mm": base["s_end_mm"],
        "rho_med": base["rho_med"],
        "nL_end": base["nL_end"],
        "nR_end": base["nR_end"],
        "regime": base["regime"],
        "live_opt": live_opt(sim),
    }

    cells = []
    tilt_snaps = {}
    taus_by_angle = {ang: list(COARSE_TAUS) for ang in ANGLES}

    for ang in ANGLES:
        seen = set()
        queue = list(taus_by_angle[ang])
        regimes = {}
        while queue:
            tau = queue.pop(0)
            key = round(tau, 3)
            if key in seen:
                continue
            seen.add(key)
            h, tilt = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, viewer=viewer)
            tilt_snaps[ang] = tilt
            rec = slim_hold(h)
            rec.update(
                {
                    "angle_cmd": ang,
                    "tau": tau,
                    "wrist_act_deg": h["wrist_act_deg"],
                    "regime": h["regime"],
                    "g_hx": h["g_hx"],
                    "g_h_tangential_xz": h["g_h_tangential_xz"],
                    "s_end_mm": h["s_end_mm"],
                    "rho_med": h["rho_med"],
                    "rho_max": h["rho_max"],
                    "t_bil_from_start": h["t_bil_from_start"],
                    "t_loss": h["t_loss"],
                }
            )
            cells.append(rec)
            save_traj(RAW / f"ang{int(ang)}_tau{tau:.2f}.npz", h)
            dump_json(RAW / f"ang{int(ang)}_tau{tau:.2f}.json", rec)
            regimes[key] = h["regime"]
        # refine intervals that change class
        ordered = sorted(seen)
        for a, b in zip(ordered, ordered[1:]):
            if regimes[a] != regimes[b]:
                for mid in midpoint_taus(a, b):
                    if round(mid, 3) not in seen:
                        queue.append(mid)
        # one extra pass of queued refinements
        while queue:
            tau = queue.pop(0)
            key = round(tau, 3)
            if key in seen:
                continue
            seen.add(key)
            h, tilt = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, viewer=viewer)
            rec = slim_hold(h)
            rec.update(
                {
                    "angle_cmd": ang,
                    "tau": tau,
                    "wrist_act_deg": h["wrist_act_deg"],
                    "regime": h["regime"],
                    "g_hx": h["g_hx"],
                    "g_h_tangential_xz": h["g_h_tangential_xz"],
                    "s_end_mm": h["s_end_mm"],
                    "rho_med": h["rho_med"],
                    "rho_max": h["rho_max"],
                    "t_bil_from_start": h["t_bil_from_start"],
                    "t_loss": h["t_loss"],
                }
            )
            cells.append(rec)
            save_traj(RAW / f"ang{int(ang)}_tau{tau:.2f}.npz", h)
            dump_json(RAW / f"ang{int(ang)}_tau{tau:.2f}.json", rec)

    stick, slip, loss = pick_representatives(cells)
    selected = slip or stick

    # C6 ± tilt
    direction = {"ok": None}
    if selected is not None:
        ang = abs(float(selected["angle_cmd"]))
        tau = float(selected["tau"])
        hp, _ = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains)
        hm, _ = run_cell(sim, cfg, parent, angle=-ang, tau=tau, gains=gains)
        save_traj(RAW / "direction_plus.npz", hp)
        save_traj(RAW / "direction_minus.npz", hm)
        direction = {
            "angle": ang,
            "tau": tau,
            "plus": {
                "g_hx": hp["g_hx"],
                "s_end_mm": hp["s_end_mm"],
                "regime": hp["regime"],
                "wrist_act_deg": hp["wrist_act_deg"],
            },
            "minus": {
                "g_hx": hm["g_hx"],
                "s_end_mm": hm["s_end_mm"],
                "regime": hm["regime"],
                "wrist_act_deg": hm["wrist_act_deg"],
            },
        }
        # s is along each trial's own û, so both should be positive if gravity-driven
        direction["ok"] = bool(hp["s_end_mm"] > 0.5 and hm["s_end_mm"] > 0.5 and hp["g_hx"] * hm["g_hx"] < 0)
        direction["g_hx_flips"] = bool(hp["g_hx"] * hm["g_hx"] < 0)
        traj_plot(hp, FIG / "direction_plus.png", f"+{ang:.0f} deg τ={tau}")
        traj_plot(hm, FIG / "direction_minus.png", f"-{ang:.0f} deg τ={tau}")

    gravity = {"ok": None}
    noslip_cmp = {"ok": None}
    brake = {}
    if slip is not None:
        ang = float(slip["angle_cmd"])
        tau = float(slip["tau"])
        h1, _ = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, g_scale=1.0)
        h0, _ = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, g_scale=0.0)
        save_traj(RAW / "gravity_1g.npz", h1)
        save_traj(RAW / "gravity_0g.npz", h0)
        gravity = {
            "angle": ang,
            "tau": tau,
            "g1": {"s_end_mm": h1["s_end_mm"], "v_rel_med": h1["v_rel_med"], "rho_med": h1["rho_med"], "regime": h1["regime"]},
            "g0": {"s_end_mm": h0["s_end_mm"], "v_rel_med": h0["v_rel_med"], "rho_med": h0["rho_med"], "regime": h0["regime"]},
        }
        gravity["ok"] = bool(abs(h0["s_end_mm"]) < 0.4 * max(abs(h1["s_end_mm"]), 1.0) and abs(h0["s_end_mm"]) < 0.8)
        traj_plot(h1, FIG / "gravity_1g.png", "1g candidate")
        traj_plot(h0, FIG / "gravity_0g.png", "0g candidate")

        hns1, _ = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, noslip=1)
        hns0, _ = run_cell(sim, cfg, parent, angle=ang, tau=tau, gains=gains, noslip=0)
        save_traj(RAW / "noslip1_candidate.npz", hns1)
        save_traj(RAW / "noslip0_candidate.npz", hns0)
        noslip_cmp = {
            "angle": ang,
            "tau": tau,
            "ns1": {"s_end_mm": hns1["s_end_mm"], "rho_med": hns1["rho_med"], "regime": hns1["regime"], "v_rel_med": hns1["v_rel_med"]},
            "ns0": {"s_end_mm": hns0["s_end_mm"], "rho_med": hns0["rho_med"], "regime": hns0["regime"], "v_rel_med": hns0["v_rel_med"]},
        }
        noslip_cmp["ok"] = bool(
            hns1["regime"] == "CONTROLLED_SLIP"
            and abs(hns0["s_end_mm"]) >= 0.5 * abs(hns1["s_end_mm"])
            and hns0["rho_med"] > 0.5
        )

        # C10 brake
        g0 = np.array(sim.model.opt.gravity, float).copy()
        tilt = rotate_wrist(sim, cfg, parent, target_deg=ang, tau=TAU_SECURE, gains=gains)
        hb = hold_window(
            sim,
            cfg,
            tau=tau,
            hold_s=2.0,
            Rh0=tilt["Rh0"],
            gains=gains,
            brake_after_mm=1.2,
            brake_tau=TAU_SECURE,
            viewer=viewer,
            phase="SLIP",
        )
        save_traj(RAW / "brake.npz", hb)
        traj_plot(hb, FIG / "brake.png", "SLIP then BRAKE τ=-18")
        extra = 0.0
        v_after = float("nan")
        if hb["t_brake"] is not None:
            t0 = hb["t_start"]
            tb = hb["t_brake"]
            pre = [r for r in hb["rows"] if (r["t"] - t0) <= tb + 1e-6]
            post = [r for r in hb["rows"] if (r["t"] - t0) >= tb]
            if pre and post:
                extra = float(post[-1]["s_mm"] - pre[-1]["s_mm"])
                v_after = float(np.median([np.linalg.norm(r["v_rel_h"]) for r in post[-10:]]))
        brake = {
            "angle": ang,
            "tau_slip": tau,
            "t_brake": hb["t_brake"],
            "s_end_mm": hb["s_end_mm"],
            "additional_mm_after_brake": extra,
            "v_rel_med_late": v_after,
            "rho_med": hb["rho_med"],
            "nL_end": hb["nL_end"],
            "nR_end": hb["nR_end"],
            "dropped": hb["dropped"],
            "t_both_loss": hb["t_both_loss"],
        }
        brake["ok"] = bool(
            hb["t_brake"] is not None
            and not hb["dropped"]
            and hb["nL_end"] > 0
            and hb["nR_end"] > 0
            and extra < 8.0
        )
        dump_json(RAW / "brake.json", brake)
        sim.model.opt.gravity[:] = g0

    robust = []
    if slip is not None:
        ang = float(slip["angle_cmd"])
        tau = float(slip["tau"])
        for m in (0.18, 0.20, 0.22):
            sim_m = make_parent_sim()
            sim_m.reset(m, PAIR_MU, np.zeros(3))
            set_finger_object_sliding_mu(sim_m.model, sim_m.ids, PAIR_MU, sim_m.data)
            pk = advance_to_parent(sim_m, cfg)
            if not pk.get("ok"):
                robust.append({"mass": m, "mu": PAIR_MU, "ok": False, "reason": "parent_fail"})
                continue
            freeze(sim_m)
            ps = sim_m.snapshot()
            ps["mass"] = float(m)
            ps["friction"] = float(PAIR_MU)
            hm, _ = run_cell(sim_m, cfg, ps, angle=ang, tau=tau, gains=gains)
            robust.append(
                {
                    "mass": m,
                    "mu": PAIR_MU,
                    "regime": hm["regime"],
                    "s_end_mm": hm["s_end_mm"],
                    "rho_med": hm["rho_med"],
                    "t_bil_from_start": hm["t_bil_from_start"],
                }
            )
            save_traj(RAW / f"robust_m{m:.2f}.npz", hm)
        for mu in (0.9, 1.1):
            sim_m = make_parent_sim()
            sim_m.reset(0.20, mu, np.zeros(3))
            set_finger_object_sliding_mu(sim_m.model, sim_m.ids, mu, sim_m.data)
            pk = advance_to_parent(sim_m, cfg)
            if not pk.get("ok"):
                robust.append({"mass": 0.20, "mu": mu, "ok": False, "reason": "parent_fail"})
                continue
            freeze(sim_m)
            snap = sim_m.snapshot()
            snap["friction"] = float(mu)
            snap["mass"] = 0.20
            hm, _ = run_cell(sim_m, cfg, snap, angle=ang, tau=tau, gains=gains)
            robust.append(
                {
                    "mass": 0.20,
                    "mu": mu,
                    "regime": hm["regime"],
                    "s_end_mm": hm["s_end_mm"],
                    "rho_med": hm["rho_med"],
                    "t_bil_from_start": hm["t_bil_from_start"],
                }
            )
            save_traj(RAW / f"robust_mu{mu:.1f}.npz", hm)

    phase_plot(cells, FIG / "phase_diagram.png")
    if stick:
        # reload stick traj already saved
        pass
    n_slip = sum(1 for c in cells if c["regime"] == "CONTROLLED_SLIP")
    n_stick = sum(1 for c in cells if c["regime"] == "STICK")
    n_loss = sum(1 for c in cells if c["regime"] == "LOSS")

    viewer_cmds = [
        "python training/gravitational_reposition_primitive_v2.py --viewer slip",
        "python training/gravitational_reposition_primitive_v2.py --viewer brake",
        "python training/gravitational_reposition_primitive_v2.py --viewer slip --camera robot",
        "python training/gravitational_reposition_primitive_v2.py --viewer brake --camera robot",
    ]
    meta = {
        "live_opt": opt0,
        "cfg_noslip": int(cfg.get("noslip_iterations", -1)),
        "cfg_tol": float(cfg.get("noslip_tolerance", -1)),
        "baseline": baseline,
        "cells": cells,
        "n_slip": n_slip,
        "n_stick": n_stick,
        "n_loss": n_loss,
        "stick": stick,
        "slip": slip,
        "loss": loss,
        "direction": direction,
        "gravity": gravity,
        "noslip_cmp": noslip_cmp,
        "brake": brake,
        "robust": robust,
        "viewer_cmds": viewer_cmds,
        "authority": "UNSET",
        "interpretation": "",
    }
    meta["interpretation"] = interpret(meta)
    dump_json(RAW / "summary.json", meta)
    write_report(meta)
    if viewer is not None:
        time.sleep(0.5)
        viewer.close()
    return meta


PARENT_PKL = RAW / "parent_snapshot.pkl"


def make_viz(viewer, camera: str = "closeup"):
    if viewer is None:
        return None
    return {
        "viewer": viewer,
        "camera": camera,
        "playback_speed": PLAYBACK_SPEED,
        "pace": True,
        "refs": None,
    }


def get_parent(sim, cfg, *, recapture: bool = False):
    RAW.mkdir(parents=True, exist_ok=True)
    if PARENT_PKL.exists() and not recapture:
        with PARENT_PKL.open("rb") as f:
            snap = pickle.load(f)
        restore(sim, snap)
        freeze(sim)
        return snap
    pack = advance_to_parent(sim, cfg)
    if not pack.get("ok"):
        raise RuntimeError(pack)
    freeze(sim)
    snap = sim.snapshot()
    snap["friction"] = float(PAIR_MU)
    with PARENT_PKL.open("wb") as f:
        pickle.dump(snap, f)
    return snap


def signed_trial_from_npz(path: Path) -> dict:
    z = np.load(path)
    rh = np.asarray(z["rh"], float)
    gh = np.asarray(z["g_h"], float)
    rh0, rh1, g0 = rh[0], rh[-1], gh[0]
    drh_mm = (rh1 - rh0) * 1e3
    g_xz = np.array([g0[0], 0.0, g0[2]])
    n = float(np.linalg.norm(g_xz))
    u = g_xz / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    progress_mm = float(drh_mm @ u)
    align_mm_mps2 = float(drh_mm @ g_xz)
    drh_m = drh_mm * 1e-3
    align_cos = float(np.dot(drh_m, g_xz) / ((np.linalg.norm(drh_m) + 1e-12) * (n + 1e-12)))
    return {
        "file": str(path.name),
        "g_h": g0.tolist(),
        "Delta_r_h_mm": drh_mm.tolist(),
        "progress_s_mm": progress_mm,
        "dot_Delta_rh_mm_g_xz": align_mm_mps2,
        "cos_Delta_rh_g_xz": align_cos,
        "nL0": int(z["nL"][0]) if "nL" in z.files else None,
        "nL1": int(z["nL"][-1]) if "nL" in z.files else None,
        "nR0": int(z["nR"][0]) if "nR" in z.files else None,
        "nR1": int(z["nR"][-1]) if "nR" in z.files else None,
    }


def compare_hold(a: dict, b_npz: Path, *, kind: str) -> dict:
    z = np.load(b_npz)
    rows = a["rows"]
    s_a = float(a["s_end_mm"])
    s_b = float(z["s_mm"][-1])
    out = {
        "kind": kind,
        "replay_progress_s_end_mm": s_a,
        "canonical_progress_s_end_mm": s_b,
        "abs_diff_mm": abs(s_a - s_b),
        "replay_drh_end_mm": a.get("drh_end_mm"),
        "canonical_drh_end_mm": ((z["rh"][-1] - z["rh"][0]) * 1e3).tolist(),
        "replay_noslip": int(a.get("noslip", -1)) if "noslip" in a else None,
        "replay_t_brake": a.get("t_brake"),
        "dt": float(rows[0]["t"] - 0) if False else None,
    }
    if kind == "brake" and a.get("t_brake") is not None:
        t0 = a["t_start"]
        tb = a["t_brake"]
        pre = [r for r in rows if (r["t"] - t0) <= tb + 1e-6]
        post = [r for r in rows if (r["t"] - t0) >= tb]
        extra = float(post[-1]["s_mm"] - pre[-1]["s_mm"]) if pre and post else float("nan")
        s_trig = float(pre[-1]["s_mm"]) if pre else float("nan")
        out["replay_s_at_brake_mm"] = s_trig
        out["replay_extra_after_brake_mm"] = extra
    return out


def play_slip(sim, cfg, parent, gains, viz=None):
    h, tilt = run_cell(
        sim, cfg, parent, angle=30.0, tau=-3.0, gains=gains, viz=viz
    )
    h["angle_cmd"] = 30.0
    h["tau"] = -3.0
    h["noslip"] = int(sim.model.opt.noslip_iterations)
    return h, tilt


VISUAL_BRAKE_MM = 3.0


def play_brake(sim, cfg, parent, gains, viz=None):
    tilt = rotate_wrist(
        sim, cfg, parent, target_deg=30.0, tau=TAU_SECURE, gains=gains, viz=viz
    )
    if viz is not None:
        s = sample(sim, -3.0)
        wall_pause(
            viz, PRE_SLIP_PAUSE_S, sim,
            phase="PAUSE", wrist_deg=tilt["wrist_act_deg"], tau=TAU_SECURE,
            rho=float(s["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
            v_rel_h=s["v_rel_h"], nL=s["nL"], nR=s["nR"],
        )
    hb = hold_window(
        sim, cfg, tau=-3.0, hold_s=2.0, Rh0=tilt["Rh0"], gains=gains,
        brake_after_mm=1.2, viz=viz, phase="SLIP",
    )
    hb["angle_cmd"] = 30.0
    hb["tau"] = -3.0
    hb["noslip"] = int(sim.model.opt.noslip_iterations)
    hb["brake_threshold_mm"] = 1.2
    hb["visual_diagnostic"] = False
    return hb, tilt


def play_brake_visual(sim, cfg, parent, gains, viz=None):
    """Visualization-only delayed brake. NOT the canonical 1.2 mm threshold."""
    if viz is not None:
        viz["overlay_extra"] = {
            "visual_diag": True,
            "tau_from": -3.0,
            "tau_to": TAU_SECURE,
            "trigger_mm": VISUAL_BRAKE_MM,
        }
        viz["_brake_wall0"] = None
    tilt = rotate_wrist(
        sim, cfg, parent, target_deg=30.0, tau=TAU_SECURE, gains=gains, viz=viz
    )
    if viz is not None:
        s = sample(sim, -3.0)
        wall_pause(
            viz, PRE_SLIP_PAUSE_S, sim,
            phase="PAUSE", wrist_deg=tilt["wrist_act_deg"], tau=TAU_SECURE,
            rho=float(s["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
            v_rel_h=s["v_rel_h"], nL=s["nL"], nR=s["nR"],
        )
    hb = hold_window(
        sim, cfg, tau=-3.0, hold_s=8.0, Rh0=tilt["Rh0"], gains=gains,
        brake_after_mm=VISUAL_BRAKE_MM, post_brake_s=2.0, viz=viz, phase="SLIP",
    )
    hb["angle_cmd"] = 30.0
    hb["tau"] = -3.0
    hb["noslip"] = int(sim.model.opt.noslip_iterations)
    hb["brake_threshold_mm"] = VISUAL_BRAKE_MM
    hb["visual_diagnostic"] = True
    hb["label"] = "VISUAL DIAGNOSTIC — NOT CANONICAL BRAKE THRESHOLD"
    return hb, tilt


def nearest_row(rows, t_abs: float) -> dict:
    ts = np.array([r["t"] for r in rows], float)
    return rows[int(np.argmin(np.abs(ts - t_abs)))]


def summarize_brake_hold(h: dict) -> dict:
    rows = h["rows"]
    tb = h.get("t_brake")
    t0 = h["t_start"]
    out = {
        "label": h.get("label", "canonical"),
        "visual_diagnostic": bool(h.get("visual_diagnostic")),
        "brake_threshold_mm": h.get("brake_threshold_mm"),
        "t_brake": tb,
        "progress_s_at_brake_mm": h.get("progress_s_at_brake_mm"),
        "drh_at_brake_mm": h.get("drh_at_brake_mm"),
        "nL_end": h.get("nL_end"),
        "nR_end": h.get("nR_end"),
        "obj_z_end": h.get("obj_z_end"),
        "dropped": h.get("dropped"),
        "t_loss": h.get("t_loss"),
        "t_both_loss": h.get("t_both_loss"),
        "wrist_end_deg": h.get("wrist_end_deg"),
        "noslip": h.get("noslip"),
    }
    if tb is None or not rows:
        out["braked"] = False
        out["basin"] = "NO_TRIGGER"
        return out
    out["braked"] = True
    rb = nearest_row(rows, t0 + float(tb))
    s_b = float(rb["s_mm"])
    extra = {}
    v_after = {}
    for dt in (0.1, 0.5, 1.0, 2.0):
        r = nearest_row(rows, t0 + float(tb) + dt)
        extra[f"+{dt:.1f}s"] = float(r["s_mm"] - s_b)
        v_after[f"+{dt:.1f}s"] = float(np.linalg.norm(r["v_rel_h"]) * 1e3)
    post = [r for r in rows if (r["t"] - t0) >= float(tb)]
    late = post[-max(1, min(20, len(post))):]
    out["post_brake_extra_progress_mm"] = extra
    out["v_rel_mm_s_after"] = v_after
    out["v_rel_mm_s_late"] = float(np.median([np.linalg.norm(r["v_rel_h"]) * 1e3 for r in late]))
    extra_2 = extra["+2.0s"]
    bilateral_end = int(h["nL_end"]) > 0 and int(h["nR_end"]) > 0
    lost_after = h.get("t_loss") is not None and h["t_loss"] > float(tb)
    arrested = (
        bilateral_end
        and not h.get("dropped")
        and extra_2 < 1.0
        and out["v_rel_mm_s_late"] < 2.0
        and not (h.get("t_both_loss") is not None and h["t_both_loss"] > float(tb))
    )
    out["bilateral_end"] = bilateral_end
    out["lost_contact_after_brake"] = bool(lost_after)
    out["arrested"] = bool(arrested)
    if arrested:
        out["basin"] = "OK"
    else:
        out["basin"] = "BRAKING BASIN < 3 mm"
    return out


def viewer_only(kind: str, camera: str = "closeup") -> None:
    import mujoco.viewer

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    if int(sim.model.opt.noslip_iterations) != 1:
        raise RuntimeError(f"live noslip={sim.model.opt.noslip_iterations}")
    parent = get_parent(sim, cfg)
    gains = gains_from_cfg(cfg)
    key_holder = {}

    def _key(keycode):
        fn = key_holder.get("fn")
        if fn is not None:
            fn(keycode)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=_key) as viewer:
        viz = make_viz(viewer, camera)
        key_holder["fn"] = make_reset_cam_callback(viz)
        init_camera_once(viz, sim)
        s0 = sample(sim, TAU_SECURE)
        wall_pause(
            viz, PRE_VIEW_PAUSE_S, sim,
            phase="PAUSED — adjust camera",
            wrist_deg=0.0, tau=TAU_SECURE,
            rho=float(s0["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
            v_rel_h=s0["v_rel_h"], nL=s0["nL"], nR=s0["nR"],
        )
        if kind == "slip":
            tilt = rotate_wrist(
                sim, cfg, parent, target_deg=30.0, tau=TAU_SECURE, gains=gains, viz=viz
            )
            s = sample(sim, -3.0)
            wall_pause(
                viz, PRE_SLIP_PAUSE_S, sim,
                phase="PAUSE", wrist_deg=tilt["wrist_act_deg"], tau=TAU_SECURE,
                rho=float(s["rho_max"]), drh_mm=np.zeros(3), progress_mm=0.0,
                v_rel_h=s["v_rel_h"], nL=s["nL"], nR=s["nR"],
            )
            h = hold_window(
                sim, cfg, tau=-3.0, hold_s=HOLD_S, Rh0=tilt["Rh0"], gains=gains,
                viz=viz, phase="SLIP",
            )
            save_traj(RAW / "viewer_slip.npz", h)
            dump_json(RAW / "viewer_slip.json", {k: v for k, v in h.items() if k != "rows"})
            wall_pause(
                viz, END_HOLD_S, sim,
                phase="END HOLD", wrist_deg=h["wrist_end_deg"], tau=-3.0,
                rho=h["rho_med"], drh_mm=h["drh_end_mm"], progress_mm=h["s_end_mm"],
                v_rel_h=h["rows"][-1]["v_rel_h"], nL=h["nL_end"], nR=h["nR_end"],
            )
        elif kind == "brake":
            hb, tilt = play_brake(sim, cfg, parent, gains, viz=viz)
            save_traj(RAW / "viewer_brake.npz", hb)
            dump_json(RAW / "viewer_brake.json", {k: v for k, v in hb.items() if k != "rows"})
            wall_pause(
                viz, 2.0, sim,
                phase="SECURE", wrist_deg=hb["wrist_end_deg"], tau=TAU_SECURE,
                rho=hb["rho_med"], drh_mm=hb["drh_end_mm"], progress_mm=hb["s_end_mm"],
                v_rel_h=hb["rows"][-1]["v_rel_h"], nL=hb["nL_end"], nR=hb["nR_end"],
            )
        elif kind == "brake_visual":
            hb, tilt = play_brake_visual(sim, cfg, parent, gains, viz=viz)
            save_traj(RAW / "viewer_brake_visual.npz", hb)
            slim = {k: v for k, v in hb.items() if k != "rows"}
            slim["summary"] = summarize_brake_hold(hb)
            dump_json(RAW / "viewer_brake_visual.json", slim)
            wall_pause(
                viz, 2.0, sim,
                phase="SECURE", wrist_deg=hb["wrist_end_deg"], tau=TAU_SECURE,
                rho=hb["rho_med"], drh_mm=hb["drh_end_mm"], progress_mm=hb["s_end_mm"],
                v_rel_h=hb["rows"][-1]["v_rel_h"], nL=hb["nL_end"], nR=hb["nR_end"],
            )
        elif kind == "stick":
            h, _ = run_cell(sim, cfg, parent, angle=30.0, tau=-18.0, gains=gains, viz=viz)
            save_traj(RAW / "viewer_stick.npz", h)
        elif kind == "loss":
            h, _ = run_cell(sim, cfg, parent, angle=30.0, tau=-2.0, gains=gains, viz=viz)
            save_traj(RAW / "viewer_loss.npz", h)
        else:
            raise ValueError(kind)


def write_viewer_validation(payload: dict) -> None:
    lines = []
    a = lines.append
    a("# Viewer validation + signed relative motion (v2)")
    a("")
    a("Physics of the primitive was **not** changed (noslip, τ, wrist, μ, m, solref, dt, brake threshold).")
    a("")
    a("## USER VISUAL OBSERVATION")
    a("")
    a("**pending** — not confirmed. Cursor logs/screenshots are not visual validation.")
    a("Watch the viewer yourself.")
    a("")
    a("## 1. Viewer / headless trajectory equivalence")
    a("")
    a("Viewer and headless `--verify` use the same `rotate_wrist` / `hold_window` / `tick_vw` path,")
    a("the same parent snapshot (`raw/parent_snapshot.pkl`), noslip=1, 30°, 1.2 rad/s, τ=−3,")
    a("dt=0.002, and brake `|progress_s|>=1.2 mm` then τ=−18.")
    a("Viewer adds only: one-shot camera init, hand-fixed markers, overlay, and wall-clock pacing.")
    a("The camera is not written during playback.")
    a("")
    a("```json")
    a(json.dumps(payload["equivalence"], indent=2))
    a("```")
    a("")
    a("## 2. Exact definition of s / progress_s")
    a("")
    a("At the start of the hold (after tilt, before grip reduction):")
    a("")
    a("- `r_h` = `R_h^T (p_object - p_hand)`  (live hand frame)")
    a("- `g_h` = `R_h^T g`")
    a("- `u_hat` = `(g_h.x, 0, g_h.z) / ||...||`  **from that trial's measured g_h**")
    a("- `progress_s` [mm] = `dot(r_h(t)-r_h(0), u_hat) * 1000`")
    a("")
    a("`progress_s` is a **progress / magnitude coordinate along trial gravity in the pad x–z plane**.")
    a("It is constructed so gravity-aligned motion is positive. It is **not** a fixed signed axis.")
    a("Do not use `progress_s` to claim directional reversal.")
    a("")
    a("Legacy key `s_mm` in npz files is this same `progress_s`.")
    a("")
    a("## 3. Fixed-frame Δr_h for +30 / −30")
    a("")
    a("Same hand-frame convention both trials (live `R_h` of that trial):")
    a("")
    a("```json")
    a(json.dumps(payload["signed"], indent=2))
    a("```")
    a("")
    a("## 4. Gravity-direction consistency")
    a("")
    a(payload["signed_verdict"])
    a("")
    a("## 5. Viewer visualization changes (display only)")
    a("")
    a("- Default camera: one-shot safe oblique (lookat=cylinder, distance=0.55 m, azimuth=148, elevation=-25). `--camera robot` wider (0.85 m). No per-frame tracking. Mouse owns the camera. R resets the initial preset.")
    a("- Hand-fixed x (red) and z (blue) axes (thin).")
    a("- Yellow ruler ticks at 0,1,2,3,4,5 mm along `u_hat`, offset 12 mm in hand-y from the slip-start center.")
    a("- Sparse white wire ghost: start axis + radial ticks (outside cylinder radius).")
    a("- Green trail: last ~1 s of cylinder center in the **current** hand frame.")
    a("- 2.5 s wall-clock PAUSED after the viewer opens (no mj_step). Then playback ~0.28×; 0.8 s pre-slip pause; 1 s end freeze (slip) / 2 s freeze (brake).")
    a("- Brake overlay: SLIP → BRAKE (~0.15 s sim after trigger, physics not paused) → SECURE.")
    a("")
    a("## 6. Slip viewer command")
    a("")
    a("```text")
    a("python training/gravitational_reposition_primitive_v2.py --viewer slip")
    a("```")
    a("")
    a("## 7. Brake viewer command")
    a("")
    a("```text")
    a("python training/gravitational_reposition_primitive_v2.py --viewer brake")
    a("```")
    a("")
    (OUT / "VIEWER_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


def verify_viewer_physics() -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    live = live_opt(sim)
    if int(live["noslip_iterations"]) != 1:
        raise RuntimeError(live)
    parent = get_parent(sim, cfg, recapture=True)
    gains = gains_from_cfg(cfg)
    h1, _ = play_slip(sim, cfg, parent, gains, viz=None)
    save_traj(RAW / "verify_slip_a.npz", h1)
    h2, _ = play_slip(sim, cfg, parent, gains, viz=None)
    save_traj(RAW / "verify_slip_b.npz", h2)
    hb, _ = play_brake(sim, cfg, parent, gains, viz=None)
    save_traj(RAW / "verify_brake.npz", hb)
    canon_slip = RAW / "ang30_tau-3.00.npz"
    canon_brake = RAW / "brake.npz"
    eq = {
        "live_opt": live,
        "parent": str(PARENT_PKL.as_posix()),
        "repeatability_slip_progress_mm": [h1["s_end_mm"], h2["s_end_mm"]],
        "repeatability_abs_diff_mm": abs(h1["s_end_mm"] - h2["s_end_mm"]),
        "vs_published_slip": compare_hold(h1, canon_slip, kind="slip") if canon_slip.exists() else None,
        "vs_published_brake": compare_hold(hb, canon_brake, kind="brake") if canon_brake.exists() else None,
        "verify_slip_progress_end_mm": h1["s_end_mm"],
        "verify_slip_drh_end_mm": h1.get("drh_end_mm"),
        "verify_brake_t_brake": hb.get("t_brake"),
        "verify_brake_s_end_mm": hb.get("s_end_mm"),
        "dt": live["timestep"],
        "omega": OMEGA,
        "angle": 30.0,
        "tau_slip": -3.0,
        "tau_secure": TAU_SECURE,
        "brake_after_mm": 1.2,
        "hold_s_slip": HOLD_S,
        "hold_s_brake": 2.0,
        "playback_is_wallclock_only": True,
        "playback_speed": PLAYBACK_SPEED,
    }
    extra = None
    if hb.get("t_brake") is not None:
        t0, tb = hb["t_start"], hb["t_brake"]
        pre = [r for r in hb["rows"] if (r["t"] - t0) <= tb + 1e-6]
        post = [r for r in hb["rows"] if (r["t"] - t0) >= tb]
        extra = float(post[-1]["s_mm"] - pre[-1]["s_mm"]) if pre and post else None
        eq["verify_s_at_brake_mm"] = float(pre[-1]["s_mm"]) if pre else None
        eq["verify_extra_after_brake_mm"] = extra
    plus = signed_trial_from_npz(RAW / "direction_plus.npz")
    minus = signed_trial_from_npz(RAW / "direction_minus.npz")
    dx_p, dx_m = plus["Delta_r_h_mm"][0], minus["Delta_r_h_mm"][0]
    gx_p, gx_m = plus["g_h"][0], minus["g_h"][0]
    gx_flips = gx_p * gx_m < 0
    dx_flips = dx_p * dx_m < 0
    both_align = plus["dot_Delta_rh_mm_g_xz"] > 0 and minus["dot_Delta_rh_mm_g_xz"] > 0
    if gx_flips and dx_flips and both_align:
        verdict = (
            "VERIFIED CODE FACT + RAW TRAJECTORY: g_h.x flips with wrist sign, and Delta r_h.x "
            f"flips with it (Δx_+={dx_p:.2f} mm, Δx_-={dx_m:.2f} mm). "
            "dot(Δr_h, g_h_xz) is positive on both trials, so the fixed-frame motion tracks tangential gravity."
        )
    elif gx_flips and both_align and not dx_flips:
        verdict = (
            "STOP: g_h.x flips but Delta r_h.x does not reverse. Inspect geometry before calling "
            "the primitive directionally controlled. See JSON above."
        )
    else:
        verdict = "See raw JSON; directional statement is not forced."
    signed = {"plus30": plus, "minus30": minus, "g_hx_flips": gx_flips, "Delta_rh_x_flips": dx_flips}
    payload = {"equivalence": eq, "signed": signed, "signed_verdict": verdict}
    dump_json(RAW / "viewer_verify.json", payload)
    write_viewer_validation(payload)
    return payload


def dump_brake_visual() -> dict:
    """Headless visual-demo trajectory. Does not overwrite canonical brake.npz."""
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    if int(sim.model.opt.noslip_iterations) != 1:
        raise RuntimeError(live_opt(sim))
    parent = get_parent(sim, cfg, recapture=False)
    gains = gains_from_cfg(cfg)
    hb, _ = play_brake_visual(sim, cfg, parent, gains, viz=None)
    save_traj(RAW / "brake_visual.npz", hb)
    summary = summarize_brake_hold(hb)
    canon = json.loads((RAW / "brake.json").read_text(encoding="utf-8")) if (RAW / "brake.json").exists() else {}
    payload = {
        "label": "VISUAL DIAGNOSTIC — NOT CANONICAL BRAKE THRESHOLD",
        "summary": summary,
        "canonical_untouched": {
            "brake.json": canon,
            "files_not_written": ["brake.npz", "brake.json", "viewer_brake.npz"],
        },
        "compare": {
            "brake_threshold_mm": {"canonical": 1.2, "visual": 3.0},
            "tau_slip": {"canonical": -3, "visual": -3},
            "tau_brake": {"canonical": -18, "visual": -18},
            "wrist_deg": {"canonical": 30, "visual": 30},
            "noslip": {"canonical": 1, "visual": 1},
            "post_brake_extra_mm": {
                "canonical": canon.get("additional_mm_after_brake"),
                "visual_+2s": (summary.get("post_brake_extra_progress_mm") or {}).get("+2.0s"),
            },
            "bilateral_after": {
                "canonical": (canon.get("nL_end", 0) > 0 and canon.get("nR_end", 0) > 0),
                "visual": summary.get("bilateral_end"),
            },
        },
    }
    dump_json(RAW / "brake_visual.json", payload)
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", default="", help="slip|brake|brake_visual|stick|loss")
    p.add_argument(
        "--camera",
        default="closeup",
        help="closeup=safe oblique grasp (default, one-shot) | robot=wider arm view",
    )
    p.add_argument("--verify", action="store_true", help="headless viewer-path vs canonical logs + signed audit")
    p.add_argument("--dump-brake-visual", action="store_true", help="headless 3 mm visual-brake demo (not canonical)")
    args = p.parse_args()
    if args.verify:
        verify_viewer_physics()
        return
    if args.dump_brake_visual:
        dump_brake_visual()
        return
    if args.viewer:
        viewer_only(args.viewer, camera=args.camera)
        return
    run()


if __name__ == "__main__":
    main()
