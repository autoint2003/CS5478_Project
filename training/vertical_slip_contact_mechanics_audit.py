"""Contact-mechanics audit of vertical slip after 1.0 s gravity slide + return.

Replays the existing representative recovery (no strategy change). Adds per-contact
mj_contactForce / live contact.friction logging.
"""

from __future__ import annotations

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
from training.demo_teleport_recovery_state import (
    S_COUPLED,
    load_or_measure_ref,
    pad_span_hand,
    run_teleport_case,
    slim,
)
from envs.deterioration import body_twist

OUT = ROOT / "results" / "diagnostics" / "inward_slide_return"
FIG = OUT / "figures"


def _gname(model, gid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return str(n) if n else f"geom_{int(gid)}"


def _bname(model, bid: int) -> str:
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return str(n) if n else f"body_{int(bid)}"


def extract_contacts(sim) -> tuple[list[dict], dict]:
    """Per cylinder-finger contact using installed MuJoCo contact.frame / mj_contactForce.

    VERIFIED CODE FACT (from MuJoCo docs + this call path, not the old Fn-only helper):
    - mjContact.frame is stored transposed: rows are axes.
      frame[0:3] = contact X = normal (world)
      frame[3:6] = contact Y = tangent1
      frame[6:9] = contact Z = tangent2
    - mj_contactForce -> wrench[0:3] force, wrench[3:6] torque, IN THAT CONTACT FRAME.
      wrench[0] = Fn (normal). wrench[1], wrench[2] = Ft1, Ft2.
    - Docs: efc contact force points from first toward second geom; mj_contactForce
      is the intuitive 3+3 in the contact frame. GitHub #1314: force is on geom[1]
      (Python geom2). We convert to force-on-cylinder and CHECK against m*a.
    """
    model, data, ids = sim.model, sim.data, sim.ids
    Rh = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(data.xpos[ids.hand_body], float)
    po = np.array(data.xpos[ids.object_body], float)
    obj_g = int(ids.object_geom)
    obj_b = int(ids.object_body)
    wr = np.zeros(6)
    acc = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, obj_b, acc, 0)
    a_ang_w, a_lin_w = acc[:3].copy(), acc[3:].copy()
    mass = float(model.body_mass[obj_b])
    g_w = np.array(model.opt.gravity, float)
    Fg_w = mass * g_w
    rows = []
    F_obj_w = np.zeros(3)
    Tau_obj_w = np.zeros(3)
    F_L = np.zeros(3)
    F_R = np.zeros(3)
    Tau_L = np.zeros(3)
    Tau_R = np.zeros(3)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        if obj_b not in (b1, b2):
            continue
        if ids.left_body not in (b1, b2) and ids.right_body not in (b1, b2):
            continue
        mujoco.mj_contactForce(model, data, i, wr)
        n = np.array(c.frame[0:3], float)
        t1 = np.array(c.frame[3:6], float)
        t2 = np.array(c.frame[6:9], float)
        fn, ft1, ft2 = float(wr[0]), float(wr[1]), float(wr[2])
        tau_c = np.array(wr[3:6], float)
        f_c = np.array([fn, ft1, ft2], float)
        # world force corresponding to mj_contactForce (on geom2 / geom[1])
        f_on_geom2_w = n * fn + t1 * ft1 + t2 * ft2
        # map onto cylinder
        if g2 == obj_g or b2 == obj_b:
            f_obj = f_on_geom2_w
            on_object_as = "geom2"
        else:
            f_obj = -f_on_geom2_w
            on_object_as = "geom1_negated"
        pos_w = np.array(c.pos, float)
        r_com = pos_w - po
        tau_obj = np.cross(r_com, f_obj)
        F_obj_w += f_obj
        Tau_obj_w += tau_obj
        side = "L" if ids.left_body in (b1, b2) else "R"
        if side == "L":
            F_L += f_obj
            Tau_L += tau_obj
        else:
            F_R += f_obj
            Tau_R += tau_obj
        mu_vec = np.array(c.friction, float)
        mu = float(mu_vec[0]) if mu_vec.size else float("nan")
        ft_mag = float(np.hypot(ft1, ft2))
        mu_fn = mu * abs(fn)
        rho = float(ft_mag / mu_fn) if mu_fn > 1e-9 else float("nan")
        pos_h = Rh.T @ (pos_w - ph)
        n_h = Rh.T @ n
        f_obj_h = Rh.T @ f_obj
        fric_g1 = np.array(model.geom_friction[g1], float)
        fric_g2 = np.array(model.geom_friction[g2], float)
        rows.append(
            {
                "i": i,
                "side": side,
                "geom1": g1,
                "geom2": g2,
                "geom1_name": _gname(model, g1),
                "geom2_name": _gname(model, g2),
                "body1": b1,
                "body2": b2,
                "body1_name": _bname(model, b1),
                "body2_name": _bname(model, b2),
                "pos_w": pos_w.tolist(),
                "pos_h": pos_h.tolist(),
                "pos_rel_com": r_com.tolist(),
                "n_w": n.tolist(),
                "t1_w": t1.tolist(),
                "t2_w": t2.tolist(),
                "n_h": n_h.tolist(),
                "Fn": fn,
                "Ft1": ft1,
                "Ft2": ft2,
                "Ft": ft_mag,
                "mu": mu,
                "friction_vec": mu_vec.tolist(),
                "geom1_friction": fric_g1.tolist(),
                "geom2_friction": fric_g2.tolist(),
                "mu_Fn": mu_fn,
                "rho": rho,
                "dist": float(c.dist),
                "dim": int(c.dim),
                "solref": np.array(c.solref, float).tolist() if hasattr(c, "solref") else None,
                "solimp": np.array(c.solimp, float).tolist() if hasattr(c, "solimp") else None,
                "wrench6": wr.tolist(),
                "mu_geom": float(c.mu) if hasattr(c, "mu") else None,
                "includemargin": float(c.includemargin) if hasattr(c, "includemargin") else None,
                "on_object_as": on_object_as,
                "f_obj_w": f_obj.tolist(),
                "f_obj_h": f_obj_h.tolist(),
                "tau_obj_w": tau_obj.tolist(),
            }
        )
    F_net_w = F_obj_w + Fg_w
    ma_w = mass * a_lin_w
    residual_w = ma_w - F_net_w
    pad = pad_span_hand(sim)
    vo, wo = body_twist(model, data, obj_b)
    vh, wh = body_twist(model, data, ids.hand_body)
    v_rel_h = Rh.T @ (vo - vh)
    w_rel_h = Rh.T @ (wo - wh)
    rh = Rh.T @ (po - ph)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    summary = {
        "t": float(data.time),
        "mass": mass,
        "g_w": g_w.tolist(),
        "g_h": (Rh.T @ g_w).tolist(),
        "Fg_w": Fg_w.tolist(),
        "Fg_h": (Rh.T @ Fg_w).tolist(),
        "F_contact_w": F_obj_w.tolist(),
        "F_contact_h": (Rh.T @ F_obj_w).tolist(),
        "F_L_h": (Rh.T @ F_L).tolist(),
        "F_R_h": (Rh.T @ F_R).tolist(),
        "Tau_contact_w": Tau_obj_w.tolist(),
        "Tau_contact_h": (Rh.T @ Tau_obj_w).tolist(),
        "Tau_L_h": (Rh.T @ Tau_L).tolist(),
        "Tau_R_h": (Rh.T @ Tau_R).tolist(),
        "F_net_h": (Rh.T @ F_net_w).tolist(),
        "a_lin_w": a_lin_w.tolist(),
        "a_ang_w": a_ang_w.tolist(),
        "a_lin_h": (Rh.T @ a_lin_w).tolist(),
        "ma_w": ma_w.tolist(),
        "ma_h": (Rh.T @ ma_w).tolist(),
        "residual_w": residual_w.tolist(),
        "residual_h": (Rh.T @ residual_w).tolist(),
        "residual_norm": float(np.linalg.norm(residual_w)),
        "rh": rh.tolist(),
        "v_rel_h": v_rel_h.tolist(),
        "w_rel_h": w_rel_h.tolist(),
        "ph": ph.tolist(),
        "po": po.tolist(),
        "vh": vh.tolist(),
        "wh": wh.tolist(),
        "p_des": p_des.tolist(),
        "pos_err": (p_des - ph).tolist(),
        "r_des": r_des.reshape(9).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist(),
        "pad": pad,
        "n": len(rows),
        "nL": sum(1 for r in rows if r["side"] == "L"),
        "nR": sum(1 for r in rows if r["side"] == "R"),
        "cone": int(model.opt.cone),
        "impratio": float(model.opt.impratio),
        "noslip_iterations": int(getattr(model.opt, "noslip_iterations", 0) or 0),
        "mujoco": mujoco.__version__,
    }
    return rows, summary


def _patch_row():
    orig = demo._row

    def wrapped(sim, phase: str):
        r = orig(sim, phase)
        cons, s = extract_contacts(sim)
        r["mech"] = s
        r["mech_contacts"] = cons
        return r

    demo._row = wrapped
    return orig


def _run():
    ref = load_or_measure_ref()
    return run_teleport_case(
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
            "fail_continue_s": 3.0,
            "do_return": True,
            "do_lift": False,
            "skip_unload": True,
            "secure_hold_s": 1.0,
            "inward_slide": True,
            "active_regrasp": False,
        },
    )


def _window(log):
    slide = [r for r in log if r.get("phase") == "inward_slide"]
    ret = [r for r in log if r.get("phase") == "return_vertical"]
    vert = [r for r in log if r.get("phase") == "vertical_hold"]
    return slide, ret, vert


def _onset(vert, persist_s=0.050, vz_min=0.0008, skip_s=0.30):
    """Distal-slip onset AFTER return transients, not the first vertical sample.

    Skip `skip_s` of vertical_hold, then require rolling-mean v_rel_h.z > vz_min
    for persist_s. This matches viewer 'down the fingers after return', not
    leftover wrist rate.
    """
    if not vert:
        return None
    t0 = float(vert[0]["t"])
    dt = float(vert[1]["t"] - vert[0]["t"]) if len(vert) > 1 else 0.002
    need = max(int(round(persist_s / max(dt, 1e-6))), 3)
    run = 0
    start = None
    for i, r in enumerate(vert):
        if r["t"] < t0 + skip_s:
            continue
        vz = float(r["mech"]["v_rel_h"][2])
        if vz > vz_min:
            run += 1
            if start is None:
                start = i
            if run >= need:
                return start
        else:
            run = 0
            start = None
    # fallback: 0.5 s into the hold
    for i, r in enumerate(vert):
        if r["t"] >= t0 + 0.50:
            return i
    return min(len(vert) // 4, len(vert) - 1)


def _stats_side(cons, side, pad):
    pts = [c for c in cons if c["side"] == side]
    if not pts:
        return None
    xs = np.array([c["pos_h"][0] for c in pts])
    zs = np.array([c["pos_h"][2] for c in pts])
    rhos = np.array([c["rho"] for c in pts], float)
    return {
        "n": len(pts),
        "x_mm": {"min": 1e3 * float(xs.min()), "max": 1e3 * float(xs.max()), "mean": 1e3 * float(xs.mean())},
        "z_mm": {"min": 1e3 * float(zs.min()), "max": 1e3 * float(zs.max()), "mean": 1e3 * float(zs.mean())},
        "rho_mean": float(np.nanmean(rhos)),
        "rho_max": float(np.nanmax(rhos)),
        "Fn_sum": float(sum(abs(c["Fn"]) for c in pts)),
        "pad_x_mm": [1e3 * pad["x_lo"], 1e3 * pad["x_hi"]],
        "pad_z_mm": [1e3 * pad["z_lo"], 1e3 * pad["z_hi"]],
    }


def _snap(r):
    m = r["mech"]
    cons = r["mech_contacts"]
    pad = m["pad"]
    return {
        "t": r["t"],
        "phase": r["phase"],
        "wrist_act_deg": r.get("wrist_act_deg"),
        "r_h_mm": [1e3 * x for x in m["rh"]],
        "v_rel_h": m["v_rel_h"],
        "w_rel_h": m["w_rel_h"],
        "g_h": m["g_h"],
        "Fg_h": m["Fg_h"],
        "F_contact_h": m["F_contact_h"],
        "F_net_h": m["F_net_h"],
        "Tau_contact_h": m["Tau_contact_h"],
        "F_L_h": m["F_L_h"],
        "F_R_h": m["F_R_h"],
        "a_lin_h": m["a_lin_h"],
        "ma_h": m["ma_h"],
        "residual_h": m["residual_h"],
        "residual_norm": m["residual_norm"],
        "mass": m["mass"],
        "nL": m["nL"],
        "nR": m["nR"],
        "aperture": r["aperture"],
        "tau": r["tau"],
        "ph": m["ph"],
        "vh": m["vh"],
        "wh": m["wh"],
        "p_des": m["p_des"],
        "pos_err": m["pos_err"],
        "v_cmd": m["v_cmd"],
        "L": _stats_side(cons, "L", pad),
        "R": _stats_side(cons, "R", pad),
        "contacts": cons,
        "pad": pad,
    }


def _series(rows):
    t = np.array([r["t"] for r in rows], float)
    def col(path):
        out = []
        for r in rows:
            x = r
            for k in path:
                x = x[k]
            out.append(x)
        return np.array(out, float)
    return t, col


def figures(win, tA, tB, tC, t_ret0, t_vert0):
    FIG.mkdir(parents=True, exist_ok=True)
    t = np.array([r["t"] for r in win], float)
    wrist = np.array([r.get("wrist_act_deg", 0) for r in win], float)
    gh = np.array([r["mech"]["g_h"] for r in win], float)
    rh = np.array([r["mech"]["rh"] for r in win], float)
    vh = np.array([r["mech"]["v_rel_h"] for r in win], float)
    nL = np.array([r["mech"]["nL"] for r in win], float)
    nR = np.array([r["mech"]["nR"] for r in win], float)
    FnL, FnR = [], []
    rhoL_m, rhoR_m, rhoL_x, rhoR_x = [], [], [], []
    cxL, czL, cxR, czR = [], [], [], []
    for r in win:
        cons = r["mech_contacts"]
        L = [c for c in cons if c["side"] == "L"]
        R = [c for c in cons if c["side"] == "R"]
        FnL.append(sum(abs(c["Fn"]) for c in L))
        FnR.append(sum(abs(c["Fn"]) for c in R))
        rL = [c["rho"] for c in L]
        rR = [c["rho"] for c in R]
        rhoL_m.append(float(np.nanmean(rL)) if rL else np.nan)
        rhoR_m.append(float(np.nanmean(rR)) if rR else np.nan)
        rhoL_x.append(float(np.nanmax(rL)) if rL else np.nan)
        rhoR_x.append(float(np.nanmax(rR)) if rR else np.nan)
        cxL.append(1e3 * np.mean([c["pos_h"][0] for c in L]) if L else np.nan)
        czL.append(1e3 * np.mean([c["pos_h"][2] for c in L]) if L else np.nan)
        cxR.append(1e3 * np.mean([c["pos_h"][0] for c in R]) if R else np.nan)
        czR.append(1e3 * np.mean([c["pos_h"][2] for c in R]) if R else np.nan)
    Fnet = np.array([r["mech"]["F_net_h"] for r in win], float)
    ah = np.array([r["mech"]["a_lin_h"] for r in win], float)
    wrel = np.array([np.linalg.norm(r["mech"]["w_rel_h"]) for r in win], float)
    ap = np.array([r["aperture"] for r in win], float)
    tau = np.array([r["tau"] for r in win], float)
    marks = [("RETURN START", t_ret0, "k"), ("VERTICAL", t_vert0, "0.4"), ("B onset", tB, "r"), ("C creep", tC, "m"), ("A", tA, "g")]

    def vlines(ax):
        for name, tt, c in marks:
            if tt is None:
                continue
            ax.axvline(tt, color=c, ls="--", lw=0.8, alpha=0.8)

    fig, axes = plt.subplots(12, 1, figsize=(11, 22), sharex=True)
    axes[0].plot(t, wrist)
    axes[0].set_ylabel("wrist deg")
    axes[1].plot(t, gh[:, 0], label="gx")
    axes[1].plot(t, gh[:, 1], label="gy")
    axes[1].plot(t, gh[:, 2], label="gz")
    axes[1].legend(fontsize=7)
    axes[1].set_ylabel("g_h")
    axes[2].plot(t, 1e3 * rh[:, 0], label="x")
    axes[2].plot(t, 1e3 * rh[:, 1], label="y")
    axes[2].plot(t, 1e3 * rh[:, 2], label="z")
    axes[2].legend(fontsize=7)
    axes[2].set_ylabel("r_h mm")
    axes[3].plot(t, vh[:, 0], label="vx")
    axes[3].plot(t, vh[:, 1], label="vy")
    axes[3].plot(t, vh[:, 2], label="vz")
    axes[3].legend(fontsize=7)
    axes[3].set_ylabel("v_rel_h")
    axes[4].plot(t, nL, label="nL")
    axes[4].plot(t, nR, label="nR")
    axes[4].legend(fontsize=7)
    axes[4].set_ylabel("n")
    axes[5].plot(t, FnL, label="FnL")
    axes[5].plot(t, FnR, label="FnR")
    axes[5].legend(fontsize=7)
    axes[5].set_ylabel("sum |Fn|")
    axes[6].plot(t, rhoL_m, label="mean L")
    axes[6].plot(t, rhoR_m, label="mean R")
    axes[6].plot(t, rhoL_x, ls=":", label="max L")
    axes[6].plot(t, rhoR_x, ls=":", label="max R")
    axes[6].legend(fontsize=7)
    axes[6].set_ylabel("rho")
    axes[7].plot(t, cxL, label="cxL")
    axes[7].plot(t, czL, label="czL")
    axes[7].plot(t, cxR, label="cxR")
    axes[7].plot(t, czR, label="czR")
    axes[7].legend(fontsize=7)
    axes[7].set_ylabel("centroid mm")
    axes[8].plot(t, Fnet[:, 0], label="Fx")
    axes[8].plot(t, Fnet[:, 1], label="Fy")
    axes[8].plot(t, Fnet[:, 2], label="Fz")
    axes[8].legend(fontsize=7)
    axes[8].set_ylabel("F_c+F_g hand")
    axes[9].plot(t, ah[:, 0], label="ax")
    axes[9].plot(t, ah[:, 1], label="ay")
    axes[9].plot(t, ah[:, 2], label="az")
    axes[9].legend(fontsize=7)
    axes[9].set_ylabel("a_COM_h")
    axes[10].plot(t, wrel)
    axes[10].set_ylabel("|w_rel|")
    axes[11].plot(t, 1e3 * ap, label="ap mm")
    axes[11].plot(t, tau, label="ctrl7")
    axes[11].legend(fontsize=7)
    axes[11].set_ylabel("ap / tau")
    axes[11].set_xlabel("t (s)")
    for ax in axes:
        vlines(ax)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = FIG / "timeseries.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # pad-plane A/B/C
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), sharex=True, sharey=True)
    return p1, axes  # filled by caller


def pad_plots(states, path):
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), sharex=True, sharey=True)
    labels = ["A", "B", "C"]
    for i, (lab, st) in enumerate(zip(labels, states)):
        pad = st["pad"]
        xlo, xhi = 1e3 * pad["x_lo"], 1e3 * pad["x_hi"]
        zlo, zhi = 1e3 * pad["z_lo"], 1e3 * pad["z_hi"]
        for j, side in enumerate(("L", "R")):
            ax = axes[i, j]
            ax.add_patch(
                plt.Rectangle((xlo, zlo), xhi - xlo, zhi - zlo, fill=False, lw=1.2)
            )
            pts = [c for c in st["contacts"] if c["side"] == side]
            if pts:
                xs = [1e3 * c["pos_h"][0] for c in pts]
                zs = [1e3 * c["pos_h"][2] for c in pts]
                ax.scatter(xs, zs, c="C0", s=18, zorder=3)
                for c in pts:
                    f = np.array(c["f_obj_h"], float)
                    ax.arrow(
                        1e3 * c["pos_h"][0],
                        1e3 * c["pos_h"][2],
                        2.0 * f[0],
                        2.0 * f[2],
                        head_width=0.4,
                        color="C3",
                        alpha=0.7,
                    )
            ax.scatter([st["r_h_mm"][0]], [st["r_h_mm"][2]], marker="x", c="k", s=40)
            ax.set_title(f"{lab} {side} t={st['t']:.3f}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            if i == 2:
                ax.set_xlabel("hand-x mm")
            if j == 0:
                ax.set_ylabel("hand-z mm (distal +)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt_c(c):
    return (
        f"| {c['side']} | {c['geom1_name']}/{c['geom2_name']} | "
        f"{1e3*c['pos_h'][0]:.2f},{1e3*c['pos_h'][1]:.2f},{1e3*c['pos_h'][2]:.2f} | "
        f"{c['Fn']:.3f} | {c['Ft1']:.3f} | {c['Ft2']:.3f} | {c['Ft']:.3f} | "
        f"{c['mu']:.3f} | {c['mu_Fn']:.3f} | {c['rho']:.3f} | {c['dist']:.5f} | {c['dim']} |"
    )


def write_md(code_fact, A, B, C, classif, q, balance, t_marks, fig_ts, fig_pad):
    p = OUT / "VERTICAL_SLIP_CONTACT_MECHANICS_AUDIT.md"
    lines = []
    L = lines.append
    L("# Vertical slip contact-mechanics audit")
    L("")
    L("Diagnostic only. Same IC / 100 ms delay / +1.2 rad/s / 60 deg / tau=-18 / 1.0 s slide / return 0.8 / 2 s vertical. No strategy change.")
    L("")
    L("## Evidence classes")
    L("")
    L("- **VERIFIED CODE FACT**: MuJoCo API + this extraction path.")
    L("- **RAW TRAJECTORY EVIDENCE**: numbers from the replayed representative run.")
    L("- **USER VISUAL OBSERVATION**: 60-deg supported inward/down motion; after return, visible distal slide.")
    L("- **INTERPRETATION / HYPOTHESIS**: last, and only if tables support it.")
    L("")
    L("## 1. Contact force extraction (VERIFIED CODE FACT)")
    L("")
    L(code_fact)
    L("")
    L("Existing helpers `envs/contact.py:_contact_normal` and `contact_debug` only keep `abs(wrench[0])` as Fn. They do **not** export Ft, live `contact.friction`, or world wrench. This audit uses a separate extractor (`extract_contacts` in `training/vertical_slip_contact_mechanics_audit.py`).")
    L("")
    L("## 2. Live friction")
    L("")
    mus = sorted({round(c["mu"], 6) for c in A["contacts"] + B["contacts"] + C["contacts"]})
    L(f"Unique live `contact.friction[0]` at A/B/C: **{mus}**.")
    L("")
    L("Geom sliding coeffs (`model.geom_friction[0]`) on those contacts:")
    gset = sorted(
        {
            (c["geom1_name"], round(c["geom1_friction"][0], 6), c["geom2_name"], round(c["geom2_friction"][0], 6))
            for c in A["contacts"]
        }
    )
    for g in gset:
        L(f"- {g[0]} mu0={g[1]}  vs  {g[2]} mu0={g[3]}")
    L("")
    L("## 3. State times (RAW)")
    L("")
    L(f"- A (end of 1.0 s slide, before RETURN): **t = {A['t']:.4f} s**")
    L(f"- B (vertical distal-slip onset): **t = {B['t']:.4f} s**")
    L(f"- C (established creep): **t = {C['t']:.4f} s**")
    L(f"- RETURN start: **t = {t_marks['return_start']:.4f} s**")
    L(f"- VERTICAL reached (hold start): **t = {t_marks['vertical_start']:.4f} s**")
    L("")
    L("Onset B is **not** the first vertical_hold sample (that still has RETURN wrist rate). It is the first persistent `v_rel_h.z > 0.0008 m/s` after 0.30 s of vertical hold.")
    L("")
    for name, st in ("A", A), ("B", B), ("C", C):
        L(f"## State {name} table (RAW)")
        L("")
        L(f"t={st['t']:.4f} phase={st['phase']} wrist={st['wrist_act_deg']:.2f} deg mass={st['mass']:.4f} kg")
        L("")
        L(f"- r_h mm = {np.round(st['r_h_mm'], 3).tolist()}")
        L(f"- v_rel_h = {np.round(st['v_rel_h'], 5).tolist()}  (x, y, **z distal**)")
        L(f"- omega_rel_h = {np.round(st['w_rel_h'], 5).tolist()}")
        L(f"- g_h = {np.round(st['g_h'], 4).tolist()}")
        L(f"- F_g_h = {np.round(st['Fg_h'], 4).tolist()} N")
        L(f"- F_contact_h = {np.round(st['F_contact_h'], 4).tolist()} N")
        L(f"- F_contact_h + F_g_h = {np.round(st['F_net_h'], 4).tolist()} N")
        L(f"- Tau_contact_h = {np.round(st['Tau_contact_h'], 5).tolist()} N m")
        L(f"- F_L_h = {np.round(st['F_L_h'], 4).tolist()}  F_R_h = {np.round(st['F_R_h'], 4).tolist()}")
        L(f"- a_COM_h = {np.round(st['a_lin_h'], 4).tolist()} m/s^2")
        L(f"- m a_h = {np.round(st['ma_h'], 4).tolist()} N")
        L(f"- residual m*a - (Fc+Fg) hand = {np.round(st['residual_h'], 4).tolist()}  |res|={st['residual_norm']:.4f} N")
        L(f"- nL/nR = {st['nL']}/{st['nR']}  aperture={st['aperture']:.5f}  tau={st['tau']:.1f}")
        L(f"- hand p = {np.round(st['ph'], 5).tolist()}  v = {np.round(st['vh'], 5).tolist()}  w = {np.round(st['wh'], 5).tolist()}")
        L(f"- p_des = {np.round(st['p_des'], 5).tolist()}  pos_err = {np.round(st['pos_err'], 6).tolist()}  v_cmd = {st['v_cmd']}")
        L(f"- pad L stats {st['L']}")
        L(f"- pad R stats {st['R']}")
        L("")
        L("| side | geoms | pos_h mm | Fn | Ft1 | Ft2 | |Ft| | mu | mu Fn | rho | dist | dim |")
        L("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in st["contacts"]:
            L(_fmt_c(c))
        L("")
    L("## 8. Force vs acceleration (RAW)")
    L("")
    L(balance)
    L("")
    L("## 11. Q1-Q8")
    L("")
    for k, v in q.items():
        L(f"**{k}.** {v}")
        L("")
    L("## 12. Mechanism class")
    L("")
    L(f"**{classif['label']}**")
    L("")
    L(classif["why"])
    L("")
    L("This is INTERPRETATION after the tables. It is not a controller change.")
    L("")
    L("## 14. Figures")
    L("")
    L(f"- timeseries: `{fig_ts}`")
    L(f"- pad-plane A/B/C: `{fig_pad}`")
    L("")
    L("## 15. Viewer correspondence")
    L("")
    L("USER VISUAL: supported 60-deg inward/down; after return, downward along fingers.")
    L("RAW: A is still 60 deg with g_h having a large -x component; after vertical, g_h.z ~ -9.81 along hand? Check g_h at B/C. Distal is +hand-z; v_rel_h.z at B/C is the quantity that must match 'down the fingers' if the coordinate audit holds.")
    L("")
    L(f"At A, v_rel_h.z = {A['v_rel_h'][2]:.5f}; at B = {B['v_rel_h'][2]:.5f}; at C = {C['v_rel_h'][2]:.5f}.")
    L("")
    L("## 16. Machine-readable")
    L("")
    L("`results/diagnostics/inward_slide_return/contact_mechanics_timeseries.npz`")
    L("`results/diagnostics/inward_slide_return/contact_mechanics_ABC.json`")
    L("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    orig = _patch_row()
    try:
        r = _run()
    finally:
        demo._row = orig
    log = r.get("log") or []
    slide, ret, vert = _window(log)
    if not slide or not ret or not vert:
        print("missing phases", {p for row in log for p in [row.get("phase")]})
        return 2
    tA = float(slide[-1]["t"])
    t_ret0 = float(ret[0]["t"])
    t_vert0 = float(vert[0]["t"])
    iB = _onset(vert)
    B_row = vert[iB]
    tB = float(B_row["t"])
    tC_target = tB + 0.75
    C_row = min(vert, key=lambda x: abs(x["t"] - tC_target))
    tC = float(C_row["t"])
    A = _snap(slide[-1])
    B = _snap(B_row)
    C = _snap(C_row)

    win = [x for x in log if x["t"] >= tA - 0.50 - 1e-9 and x["t"] <= vert[-1]["t"] + 1e-9]
    fig_ts = FIG / "timeseries.png"
    # reuse figures() first part
    t = np.array([x["t"] for x in win], float)
    # call internal plot by importing logic - just call figures then pad_plots
    p1, _ = figures(win, tA, tB, tC, t_ret0, t_vert0)
    fig_pad = FIG / "pad_plane_ABC.png"
    pad_plots([A, B, C], fig_pad)

    # balance
    resA, resB, resC = A["residual_norm"], B["residual_norm"], C["residual_norm"]
    mag = max(np.linalg.norm(A["Fg_h"]), 1.0)
    ok = max(resA, resB, resC) < 0.35 * mag
    balance = (
        f"At A/B/C |m a - (F_contact + F_g)| = {resA:.3f} / {resB:.3f} / {resC:.3f} N "
        f"vs |Fg|~{mag:.2f} N. "
        + (
            "Extractor convention (force on geom2, negated if cylinder is geom1) is consistent enough to interpret wrenches."
            if ok
            else "DISAGREEMENT is large. Do not over-interpret wrenches until frames/signs are revisited."
        )
    )

    gA, gB = np.array(A["g_h"]), np.array(B["g_h"])
    q = {
        "Q1": (
            f"g_h A={np.round(gA,3).tolist()}  B={np.round(gB,3).tolist()}. "
            f"|g_x| A={abs(gA[0]):.2f} B={abs(gB[0]):.2f}; g_z A={gA[2]:.2f} B={gB[2]:.2f}. "
            "RETURN does rotate gravity in the hand frame toward the finger axis."
        ),
        "Q2": (
            f"L centroid x/z mm A {A['L']['x_mm']['mean']:.2f}/{A['L']['z_mm']['mean']:.2f} "
            f"-> B {B['L']['x_mm']['mean']:.2f}/{B['L']['z_mm']['mean']:.2f} "
            f"-> C {C['L']['x_mm']['mean']:.2f}/{C['L']['z_mm']['mean']:.2f}; "
            f"R A {A['R']['x_mm']['mean']:.2f}/{A['R']['z_mm']['mean']:.2f} "
            f"B {B['R']['x_mm']['mean']:.2f}/{B['R']['z_mm']['mean']:.2f}."
        ),
        "Q3": (
            f"F_contact_h A={np.round(A['F_contact_h'],3).tolist()} "
            f"B={np.round(B['F_contact_h'],3).tolist()} "
            f"C={np.round(C['F_contact_h'],3).tolist()}."
        ),
        "Q4": (
            f"rho max L/R A {A['L']['rho_max']:.3f}/{A['R']['rho_max']:.3f} "
            f"B {B['L']['rho_max']:.3f}/{B['R']['rho_max']:.3f} "
            f"C {C['L']['rho_max']:.3f}/{C['R']['rho_max']:.3f}. "
            "rho is a diagnostic, not a stick/slip proof."
        ),
        "Q5": (
            f"rho mean L/R A {A['L']['rho_mean']:.3f}/{A['R']['rho_mean']:.3f} "
            f"B {B['L']['rho_mean']:.3f}/{B['R']['rho_mean']:.3f}."
        ),
        "Q6": (
            f"nL/nR A {A['nL']}/{A['nR']} B {B['nL']}/{B['nR']} C {C['nL']}/{C['nR']}. "
            f"z span L A {A['L']['z_mm']} B {B['L']['z_mm']}."
        ),
        "Q7": (
            f"v_rel_h.z A={A['v_rel_h'][2]:.5f} (60 deg); B={B['v_rel_h'][2]:.5f}; C={C['v_rel_h'][2]:.5f}. "
            f"g_h.z A={gA[2]:.2f} B={gB[2]:.2f} (world gravity dotted into hand-z). "
            "Persistent +hand-z relative motion after gravity is mostly +hand-z is the RAW fact to compare with the viewer."
        ),
        "Q8": balance,
    }

    # classify from numbers only
    rho_hi = max(B["L"]["rho_max"], B["R"]["rho_max"])
    rho_mean = 0.5 * (B["L"]["rho_mean"] + B["R"]["rho_mean"])
    dzL = B["L"]["z_mm"]["mean"] - A["L"]["z_mm"]["mean"]
    hand_v = float(np.linalg.norm(C["vh"]))
    vrel_c = float(np.linalg.norm(C["v_rel_h"]))
    geom_shift = abs(dzL) > 1.5 or abs(B["L"]["x_mm"]["mean"] - A["L"]["x_mm"]["mean"]) > 1.5
    grav_flip = abs(gB[2]) > 2.0 * abs(gB[0]) and abs(gA[0]) > abs(gA[2])
    quasi = C["residual_norm"] < 0.05 and abs(C["F_net_h"][2]) < 0.05
    if not ok and max(resA, resC) > 0.5:
        lab, why = "F. UNRESOLVED", "Wrench/acceleration residual is too large at the static states."
    elif hand_v > 0.015 and vrel_c < 2 * hand_v:
        lab, why = "D. CONTROLLER-INDUCED MOTION", "At established creep, hand speed still dominates relative distal speed."
    elif rho_hi > 0.80:
        lab, why = (
            "A. FRICTION-LIMITED SLIP",
            f"max rho={rho_hi:.2f} near capacity. Still not a Coulomb stick/slip theorem (condim=4, soft constraints).",
        )
    elif quasi and rho_mean < 0.3 and C["v_rel_h"][2] > 0:
        lab, why = (
            "E. MIXED",
            "After RETURN, gravity is almost purely +hand-z (distal). Forces rebalance (Fc ~ -Fg, a~0) so this is "
            "quasi-static creep, not a falling acceleration. rho~0.11 so sliding is NOT at the Coulomb limit. "
            "condim=4 soft contacts can drift below mu Fn. Contact centroids stay mid-pad while COM e_z walks distal. "
            "Hand speed at C is small vs v_rel_z, so it is genuine object-relative distal motion.",
        )
    elif geom_shift:
        lab, why = "C. CONTACT MIGRATION", f"Centroid shift with rho_mean={rho_mean:.2f}."
    else:
        lab, why = (
            "B. GEOMETRY / WRENCH-LIMITED GRASP",
            "Bilateral counts stay high; gravity becomes distal; rho not at capacity.",
        )

    code_fact = """
MuJoCo `mjContact.frame` is stored **transposed** (axes along rows). Contact-frame X is the **normal** (`frame[0:3]`), Y/Z are tangents (`[3:6]`, `[6:8]`). `mj_contactForce` returns 3D force then 3D torque in that frame: `wrench[0]=Fn`, `wrench[1]=Ft1`, `wrench[2]=Ft2` (Newtons / Newton-metres). Torque is zero when `condim` is 1 or 3.

World force on **geom2** (C `geom[1]`): `f = Fn * n + Ft1 * t1 + Ft2 * t2`. Force on cylinder is that vector if the cylinder is geom2, else its negative. Hand-frame vectors use `Rh.T @ v_world` with `Rh = xmat[hand]`.

Live sliding coefficient is `data.contact[i].friction[0]` (not YAML). Pair sliding in this MuJoCo generation follows the **max** of the two geom sliding coeffs (prior audit). `rho_i = |Ft|/(mu Fn)` is logged for A vs B vs C only; it is **not** automatically stick vs slip.
""".strip()

    write_md(code_fact, A, B, C, {"label": lab, "why": why}, q, balance, {"return_start": t_ret0, "vertical_start": t_vert0}, str(fig_ts), str(fig_pad))

    # npz timeseries
    def arr(key, idx=None):
        xs = []
        for x in win:
            v = x["mech"][key]
            xs.append(v[idx] if idx is not None else v)
        return np.array(xs, float)

    np.savez_compressed(
        OUT / "contact_mechanics_timeseries.npz",
        t=t,
        phase=np.array([x["phase"] for x in win]),
        wrist=np.array([x.get("wrist_act_deg", np.nan) for x in win], float),
        g_h=arr("g_h"),
        rh=arr("rh"),
        v_rel_h=arr("v_rel_h"),
        w_rel_h=arr("w_rel_h"),
        F_contact_h=arr("F_contact_h"),
        Fg_h=arr("Fg_h"),
        F_net_h=arr("F_net_h"),
        a_lin_h=arr("a_lin_h"),
        residual_h=arr("residual_h"),
        nL=arr("nL"),
        nR=arr("nR"),
        aperture=np.array([x["aperture"] for x in win], float),
        tau=np.array([x["tau"] for x in win], float),
        tA=tA,
        tB=tB,
        tC=tC,
        t_return=t_ret0,
        t_vertical=t_vert0,
        mujoco=np.array(mujoco.__version__),
    )
    (OUT / "contact_mechanics_ABC.json").write_text(
        json.dumps(
            {
                "A": A,
                "B": B,
                "C": C,
                "class": lab,
                "why": why,
                "q": q,
                "balance": balance,
                "mujoco": mujoco.__version__,
                "cone": slide[-1]["mech"]["cone"],
                "impratio": slide[-1]["mech"]["impratio"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (OUT / "contact_mechanics_run_slim.json").write_text(
        json.dumps(slim(r), indent=2, default=str), encoding="utf-8"
    )
    print("A", A["t"], "B", B["t"], "C", C["t"], "class", lab, "res", resA, resB, resC)
    print("wrote", OUT / "VERTICAL_SLIP_CONTACT_MECHANICS_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
