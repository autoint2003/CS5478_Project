"""Audit why tau=-4 slip self-arrests on S_FAIL, and whether active brake is needed.

Does not retune RULE, SAC, reward, observation, noslip, or the disturbance family.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import RH_Z_NOM, TABLE_DROP
from training.demo_grav_reposition_recovery import (
    BRAKE_HOLD,
    HOLD_S,
    LOG_DT,
    OMEGA,
    OMEGA_RET,
    PAIR_MU,
    TAU_SEC,
    THETA,
    VERT_HOLD,
    V_LIFT,
    DZ_LIFT,
    _slim_m,
    apply_sfail,
    classify_zero,
    continue_nominal,
    dropped,
    freeze,
    hold_steps,
    live_opt,
    make_parent_sim,
    measure,
    restore,
    return_vertical,
    rotate_hy,
    run_zero,
    snap_now,
    useful_mm,
)
from training.demo_teleport_recovery_state import (
    coupled_pose_from_ref,
    load_or_measure_ref,
    pad_span_hand,
    signed_rot_about_y_deg,
)
from training.nominal_grasp_creep_root_cause import dump_json
from training.replay_core import tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "grav_reposition_self_arrest"
RAW = OUT / "raw"
FIG = OUT / "figures"
SFAIL_PKL = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction" / "raw" / "sfail.pkl"
PARENT_PKL = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction" / "raw" / "parent.pkl"
CONSTR_SUM = ROOT / "results" / "diagnostics" / "grav_reposition_recovery_construction" / "raw" / "summary.json"

TAU_WEAK = -4.0
SLIP_MAX_S = 3.0
HOLD5_S = 5.0
HOLD2_S = 2.0


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def wrist_deg(sim, Rh0) -> float:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    return float(signed_rot_about_y_deg(Rh0, Rh))


def ctrl_row(sim, phase: str, tau_cmd: float, Rh0) -> dict:
    return {
        "t": float(sim.data.time),
        "phase": phase,
        "tau_cmd": float(tau_cmd),
        "ctrl7": float(sim.data.ctrl[7]),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "r_des": np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist(),
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist(),
        "wrist_deg": wrist_deg(sim, Rh0),
        "noslip": int(sim.model.opt.noslip_iterations),
    }


def dump_snap(sim, tau: float, u, path_stem: Path, extra=None) -> dict:
    snap = snap_now(sim)
    with path_stem.with_suffix(".pkl").open("wb") as f:
        pickle.dump(snap, f)
    rec = rich_sample(sim, tau, u)
    rec["qpos"] = np.asarray(sim.data.qpos, float).tolist()
    rec["qvel"] = np.asarray(sim.data.qvel, float).tolist()
    rec["ctrl"] = np.asarray(sim.data.ctrl, float).tolist()
    rec["p_des"] = np.asarray(sim.fsm.p_des, float).tolist()
    rec["r_des"] = np.asarray(sim.fsm.r_des, float).reshape(3, 3).tolist()
    rec["v_cmd"] = np.asarray(sim.fsm.v_cmd, float).tolist()
    rec["w_cmd"] = np.asarray(sim.fsm.w_cmd, float).tolist()
    rec["live"] = live_opt(sim)
    rec["time"] = float(sim.data.time)
    if extra:
        rec.update(extra)
    dump_json(path_stem.with_suffix(".json"), rec)
    return rec


def contact_pack(rows: list) -> dict:
    out = {}
    for side in ("L", "R"):
        rs = [r for r in rows if r["side"] == side]
        if not rs:
            out[side] = {"n": 0, "Fn": 0.0, "Ft": 0.0, "rho_max": float("nan"),
                         "centroid_h": None, "span_h": None, "pos_h": [], "n_h": []}
            continue
        pos = np.array([r["pos_h"] for r in rs], float)
        nh = np.array([r["n_h"] for r in rs], float)
        rhos = [r["rho"] for r in rs if np.isfinite(r.get("rho", np.nan))]
        out[side] = {
            "n": len(rs),
            "Fn": float(sum(abs(r["Fn"]) for r in rs)),
            "Ft": float(sum(abs(r["Ft"]) for r in rs)),
            "rho_max": float(np.max(rhos)) if rhos else float("nan"),
            "rho_med": float(np.median(rhos)) if rhos else float("nan"),
            "centroid_h": pos.mean(axis=0).tolist(),
            "span_h": (pos.max(axis=0) - pos.min(axis=0)).tolist(),
            "pos_h": pos.tolist(),
            "n_h": nh.tolist(),
        }
    return out


def rich_sample(sim, tau: float, u) -> dict:
    rows, sm = extract_contacts(sim)
    m = measure(sim, tau)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    R_rel = Rh.T @ Ro
    g_h = np.asarray(m["g_h"], float)
    uu = np.asarray(u, float)
    uu = uu / (np.linalg.norm(uu) + 1e-12)
    v = np.asarray(m["v_rel_h"], float)
    cpack = contact_pack(rows)
    pad = pad_span_hand(sim)
    return {
        "measure": _slim_m(m),
        "g_h": g_h.tolist(),
        "g_dot_u": float(g_h @ uu),
        "g_xz": [float(g_h[0]), 0.0, float(g_h[2])],
        "g_xz_norm": float(np.hypot(g_h[0], g_h[2])),
        "v_along_u": float(v @ uu),
        "cyl_h": m["cyl_h"].tolist(),
        "R_rel": R_rel.tolist(),
        "contacts": cpack,
        "pad": pad,
        "ncon": int(sim.data.ncon),
        "wrist_from_g": float(np.degrees(np.arctan2(-g_h[0], g_h[2]))),
        "tau_cmd": float(tau),
        "ctrl7": float(sim.data.ctrl[7]),
    }


def log_to_npz(path: Path, log: list, u) -> None:
    if not log:
        return
    uu = np.asarray(u, float)
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in log]),
        rh=np.stack([r["rh"] for r in log]),
        v_rel_h=np.stack([r["v_rel_h"] for r in log]),
        w_rel_h=np.stack([r["w_rel_h"] for r in log]),
        nL=np.array([r["nL"] for r in log]),
        nR=np.array([r["nR"] for r in log]),
        rho=np.array([r["rho_max"] for r in log]),
        obj_z=np.array([r["obj_z"] for r in log]),
        g_h=np.stack([r["g_h"] for r in log]),
        cyl_h=np.stack([r["cyl_h"] for r in log]),
        tau=np.array([r.get("tau", np.nan) for r in log]),
        ctrl=np.array([r.get("ctrl", np.nan) for r in log]),
        wrist=np.array([r.get("wrist_deg", np.nan) if r.get("wrist_deg") is not None else np.nan for r in log]),
        phase=np.array([str(r.get("phase", "")) for r in log]),
        v_along=np.array([float(np.asarray(r["v_rel_h"]) @ uu) for r in log]),
    )


def find_saturation(t, v_along, persist_s=0.15, v_thr=5e-4):
    if len(t) < 3:
        return None, None
    absv = np.abs(np.asarray(v_along, float))
    dt = float(np.median(np.diff(t)))
    nneed = max(int(round(persist_s / max(dt, 1e-6))), 3)
    for i in range(len(t) - nneed):
        if np.all(absv[i : i + nneed] < v_thr):
            return float(t[i]), int(i)
    return None, None


def hold_tau(sim, gains, tau, seconds, viz, Rh0, rh0, phase, log, u, ctrl_log):
    dt = float(sim.model.opt.timestep)
    n = int(round(seconds / dt))
    log_n = max(int(round(LOG_DT / dt)), 1)
    t_drop = None
    for i in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
        m = measure(sim, tau)
        m["phase"] = phase
        m["wrist_deg"] = wrist_deg(sim, Rh0)
        if t_drop is None and dropped(sim, m):
            t_drop = float(sim.data.time)
        if i % log_n == 0 or i == n - 1:
            log.append(m)
            if i % (5 * log_n) == 0:
                ctrl_log.append(ctrl_row(sim, phase, tau, Rh0))
        if t_drop is not None:
            break
    return {"t_drop": t_drop, "end": measure(sim, tau)}


def go_slip_start(sim, gains, sfail, theta, log, ctrl_log, viz=None):
    restore(sim, sfail["snap"])
    freeze(sim)
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    m0 = measure(sim, TAU_SEC)
    rh0 = m0["rh"].copy()
    ctrl_log.append(ctrl_row(sim, "SFAIL", TAU_SEC, Rh0))
    hold_steps(sim, gains, TAU_SEC, 0.100, viz, Rh0, rh0, "OBSERVE", log)
    ctrl_log.append(ctrl_row(sim, "OBSERVE_END", TAU_SEC, Rh0))
    rotate_hy(sim, gains, theta, TAU_SEC, viz, Rh0, rh0, "ROTATE", log, omega=OMEGA)
    freeze(sim)
    ctrl_log.append(ctrl_row(sim, "SLIP_START", TAU_SEC, Rh0))
    return Rh0, rh0


def slip_window(sim, gains, tau, u, seconds, viz, Rh0, rh_s, log, ctrl_log, phase="SLIP"):
    dt = float(sim.model.opt.timestep)
    n = int(round(seconds / dt))
    log_n = max(int(round(LOG_DT / dt)), 1)
    lost = False
    for i in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), tau, gains)
        m = measure(sim, tau)
        m["phase"] = phase
        m["wrist_deg"] = wrist_deg(sim, Rh0)
        if i % log_n == 0 or i == n - 1:
            log.append(m)
        if i % (5 * log_n) == 0:
            ctrl_log.append(ctrl_row(sim, phase, tau, Rh0))
        if dropped(sim, m) or m["nL"] == 0 or m["nR"] == 0:
            lost = True
            break
    return lost


def samples_at_frac(log_slip, rh_s, u, fracs):
    if not log_slip:
        return []
    prog = np.array([useful_mm(r["rh"], rh_s, u) for r in log_slip])
    pmax = float(np.max(prog)) if len(prog) else 0.0
    out = []
    for f in fracs:
        target = f * pmax
        i = int(np.argmin(np.abs(prog - target)))
        out.append((f, i, log_slip[i], float(prog[i]), pmax))
    return out


def task_ok(cont) -> bool:
    if not cont:
        return False
    end = cont.get("end") or {}
    return bool(
        cont.get("hold_survived")
        and not cont.get("drop")
        and int(end.get("nL") or 0) > 0
        and int(end.get("nR") or 0) > 0
    )


def vert_then_cont(sim, cfg, gains, tau_hold, viz, Rh0, rh0):
    freeze(sim)
    vh = hold_tau(sim, gains, tau_hold, VERT_HOLD, viz, Rh0, rh0, "VERT_HOLD", [], np.array([-1.0, 0, 0]), [])
    freeze(sim)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.p_des = ph.copy()
    cont = continue_nominal(sim, cfg, gains, viz=viz, Rh0=Rh0, rh0=rh0)
    return vh, cont


def write_report(meta: dict) -> None:
    p = OUT / "GRAV_REPOSITION_SELF_ARREST.md"
    a = []
    ap = a.append
    ap("# Gravitational reposition self-arrest (s=2.0, noslip=1)")
    ap("")
    ap("Privileged diagnostic. RULE / SAC / reward / observation / noslip / disturbance family were not changed.")
    ap("")
    ap("## USER VISUAL OBSERVATION")
    ap("")
    ap("- Simple CONTROLLED SLIP: CONFIRMED (prior).")
    ap("- Self-arrest viewer (`--mode self_arrest`): **PENDING** until the user watches it.")
    ap("")
    ap("## 1. Exact NO_BRAKE semantics")
    ap("")
    ap("Source: `run_full(..., do_slip=True, do_brake=False)` in `training/demo_grav_reposition_recovery.py`.")
    ap("Name is **not** the control law. Measured `ctrl[7]` is in `raw/no_brake_ctrl_trace.json`.")
    ap("")
    ap("| Phase | duration (design) | wrist | p_des | r_des | v_cmd | tau / ctrl[7] |")
    ap("|-------|-------------------|-------|-------|-------|-------|----------------|")
    ap("| OBSERVE | 0.10 s | hold | freeze-here | freeze-here | 0 | **-18** |")
    ap("| ROTATE | ~0.44 s (30° / 1.2 rad/s) | +hand-y | freeze translation | integrate ω | 0 | **-18** |")
    ap("| freeze() | instant | current pose snapped to p_des/r_des | | | 0 | (ctrl last) |")
    ap("| SLIP | **fixed 1.50 s** (not saturation) | held +30° | frozen | frozen | 0 | **-4** |")
    ap("| RETURN | ~0.52 s (1.0 rad/s) | back to 0 | frozen translation | integrate −ω | 0 | **-4** (tau_ret = slip tau) |")
    ap("| VERT_HOLD | 2.00 s | held 0° | freeze-here | freeze-here | 0 | **-18** (hardcoded TAU_SEC) |")
    ap("| LIFT + HOLD10 | 2.25 + 10 s | 0° | p_des := p_hand then +z | frozen ori | (0,0,0.08) then 0 | **-18** |")
    ap("")
    ap("Critical facts:")
    ap("")
    ap("1. NO_BRAKE does **not** keep τ=−4 through vertical hold or continuation.")
    ap("2. VERT_HOLD and `continue_nominal` always command τ=−18.")
    ap("3. What NO_BRAKE actually skips is the **tilted 0.5 s τ=−18 dwell**.")
    ap("4. Return-to-vertical in NO_BRAKE is a **weak-grip return** (τ=−4).")
    ap("5. Slip duration is a **1.5 s timeout**, not a saturation detector.")
    ap("")
    sem = meta.get("semantics_measured") or {}
    ap("Measured unique (phase, tau_cmd, ctrl7) pairs:")
    ap("")
    ap("```json")
    ap(json.dumps(sem, indent=2, default=str))
    ap("```")
    ap("")
    ap("## 2. Continuous slip-decay trajectory")
    ap("")
    sat = meta.get("saturation") or {}
    ap("```json")
    ap(json.dumps(sat, indent=2, default=str))
    ap("```")
    ap("")
    ap("Traces: `figures/fig_slip_decay.png`, `raw/slip_tau4.npz`.")
    ap("")
    ap("## 3. Force / contact / geometry evolution")
    ap("")
    ap("Samples at slip start, 25/50/75% of peak useful displacement, and saturation:")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("evolution"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 4. H1–H4")
    ap("")
    ap(meta.get("hypotheses_text", ""))
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("hypotheses"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 5. 5 s tilted weak-grip hold from saturation")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("hold5"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 6. Weak vs secure 2 s hold from identical saturated snapshot")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("hold2"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 7. WEAK_RETURN vs SECURE_RETURN vs BRAKE_DWELL_RETURN")
    ap("")
    ap("Vertical 2 s hold after return, **no lift** in this comparison:")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("returns_vert"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 8. Task continuation")
    ap("")
    ap("Same nominal lift + 10 s hold. Physical outcomes only.")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("continuation"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## 9. Does brake dwell have causal value?")
    ap("")
    ap(meta.get("brake_verdict", ""))
    ap("")
    ap("## 10. Viewer command")
    ap("")
    ap("```text")
    ap("python training/demo_grav_reposition_recovery.py --mode self_arrest")
    ap("```")
    ap("")
    ap("Sequence: S_FAIL → rotate +30° at τ=−18 → τ=−4 at +30° for ≥5 s. τ does **not** return to −18.")
    ap("Camera: one-shot preset, then mouse-owned. Overlay: tau, progress, v_rel, rho, nL/nR.")
    ap("")
    ap("## 11. USER VISUAL OBSERVATION: pending")
    ap("")
    ap("## Local s-perturbation (after mechanism)")
    ap("")
    ap("```json")
    ap(json.dumps(meta.get("perturbation"), indent=2, default=str))
    ap("```")
    ap("")
    ap("## Snapshots")
    ap("")
    ap("- `raw/slip_start.*`  wrist +30°, immediately before τ weakening")
    ap("- `raw/saturation.*`  self-arrested relative pose, still τ=−4, +30°")
    ap("- `raw/pre_return_nobrake.*`  after 1.5 s slip (legacy NO_BRAKE pre-return)")
    ap("- `raw/post_active_brake.*`  FULL 1 mm + 0.5 s τ=−18")
    p.write_text("\n".join(a), encoding="utf-8")


def construction():
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    opt = live_opt(sim)
    if int(opt["noslip_iterations"]) != 1:
        raise RuntimeError(opt)
    gains = gains_from_cfg(cfg)
    with SFAIL_PKL.open("rb") as f:
        sfail_snap = pickle.load(f)
    sfail = {"snap": sfail_snap, "ok": True, "s": 2.0}
    theta = THETA
    u = np.array([-1.0, 0.0, 0.0])
    if CONSTR_SUM.is_file():
        sm = json.loads(CONSTR_SUM.read_text(encoding="utf-8"))
        theta = float((sm.get("wrist") or {}).get("theta_cmd_deg", THETA))
        ex = float((sm.get("sfail_state") or {}).get("rh", [1, 0, 0])[0])
        u = np.array([-np.sign(ex) if abs(ex) > 1e-6 else 1.0, 0.0, 0.0])

    # ----- instrumented NO_BRAKE -----
    log_nb = []
    ctrl_nb = []
    Rh0, rh0 = go_slip_start(sim, gains, sfail, theta, log_nb, ctrl_nb)
    rec_start = dump_snap(sim, TAU_SEC, u, RAW / "slip_start", extra={"note": "before tau weaken"})
    rh_s = measure(sim, TAU_WEAK)["rh"].copy()
    u_g = np.asarray(rec_start["g_h"], float).copy()
    u_g[1] = 0.0
    nrm = np.linalg.norm(u_g)
    if nrm > 1e-9:
        u_g /= nrm
    else:
        u_g = u.copy()

    log_slip = []
    lost = slip_window(sim, gains, TAU_WEAK, u, SLIP_MAX_S, None, Rh0, rh_s, log_slip, ctrl_nb, "SLIP")
    for r in log_slip:
        r["v_along"] = float(np.asarray(r["v_rel_h"]) @ u)
        r["v_along_g"] = float(np.asarray(r["v_rel_h"]) @ u_g)
        r["prog_mm"] = useful_mm(r["rh"], rh_s, u)
    log_nb.extend(log_slip)
    log_to_npz(RAW / "slip_tau4.npz", log_slip, u)

    t = np.array([r["t"] for r in log_slip])
    t0 = float(t[0]) if len(t) else 0.0
    v_u = np.array([r["v_along"] for r in log_slip])
    v_g = np.array([r["v_along_g"] for r in log_slip])
    prog = np.array([r["prog_mm"] for r in log_slip])
    rh_tr = np.stack([r["rh"] for r in log_slip]) if log_slip else np.zeros((0, 3))
    g_dot = np.array([float(np.asarray(r["g_h"]) @ u) for r in log_slip])
    rho = np.array([r["rho_max"] for r in log_slip])

    sat_tab = {}
    sat_t, sat_i = None, None
    for thr, persist in ((1e-3, 0.15), (5e-4, 0.15), (2e-4, 0.20)):
        ts, i = find_saturation(t, v_u, persist_s=persist, v_thr=thr)
        sat_tab[f"v_along_u<{thr}_for_{persist}s"] = None if ts is None else float(ts - t0)
        if sat_t is None and ts is not None:
            sat_t, sat_i = ts, i
    if sat_i is None and len(log_slip):
        sat_i = int(np.argmin(np.abs(v_u)))
        sat_t = float(t[sat_i])

    # restore slip_start and replay to saturation to snapshot
    restore(sim, sfail["snap"])
    freeze(sim)
    log_tmp, ctrl_tmp = [], []
    Rh0, rh0 = go_slip_start(sim, gains, sfail, theta, log_tmp, ctrl_tmp)
    rh_s = measure(sim, TAU_WEAK)["rh"].copy()
    dt = float(sim.model.opt.timestep)
    n_sat = int(round(max(sat_t - t0, 0.05) / dt)) if sat_t is not None else int(round(1.2 / dt))
    for i in range(n_sat):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_WEAK, gains)
        m = measure(sim, TAU_WEAK)
        if dropped(sim, m) or m["nL"] == 0 or m["nR"] == 0:
            break
    freeze(sim)
    rec_sat = dump_snap(sim, TAU_WEAK, u, RAW / "saturation", extra={"note": "self-arrest under tau=-4, wrist +30"})
    sat_snap = snap_now(sim)

    # legacy NO_BRAKE pre-return = 1.5 s slip
    restore(sim, sfail["snap"])
    freeze(sim)
    log_pr, ctrl_pr = [], []
    Rh0, rh0 = go_slip_start(sim, gains, sfail, theta, log_pr, ctrl_pr)
    slip_window(sim, gains, TAU_WEAK, u, 1.5, None, Rh0, rh_s, log_pr, ctrl_pr, "SLIP")
    rec_pr = dump_snap(sim, TAU_WEAK, u, RAW / "pre_return_nobrake", extra={"note": "after 1.5s slip, before weak return"})
    dump_json(RAW / "no_brake_ctrl_trace.json", jsonable(ctrl_nb + ctrl_pr[-8:]))

    # FULL post-brake
    restore(sim, sfail["snap"])
    freeze(sim)
    log_f, ctrl_f = [], []
    Rh0, rh0 = go_slip_start(sim, gains, sfail, theta, log_f, ctrl_f)
    rh_s_f = measure(sim, TAU_WEAK)["rh"].copy()
    nmax = int(round(1.5 / dt))
    for i in range(nmax):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU_WEAK, gains)
        m = measure(sim, TAU_WEAK)
        if useful_mm(m["rh"], rh_s_f, u) >= 1.0 or dropped(sim, m) or m["nL"] == 0 or m["nR"] == 0:
            break
    freeze(sim)
    hold_steps(sim, gains, TAU_SEC, BRAKE_HOLD, None, Rh0, rh_s_f, "BRAKE", log_f)
    rec_br = dump_snap(sim, TAU_SEC, u, RAW / "post_active_brake", extra={"note": "1mm + 0.5s tau=-18"})

    # evolution samples from log_slip
    fracs = samples_at_frac(log_slip, rh_s, u, (0.0, 0.25, 0.50, 0.75, 1.0))
    evo = []
    # rebuild rich samples by time: restore slip_start and step
    restore(sim, sfail["snap"])
    freeze(sim)
    go_slip_start(sim, gains, sfail, theta, [], [])
    times_need = []
    if log_slip:
        times_need.append(("start", float(log_slip[0]["t"])))
        for f, i, row, pv, pmax in fracs[1:]:
            times_need.append((f"{int(f*100)}pct", float(row["t"])))
        times_need.append(("saturation", float(sat_t)))
    # unique increasing
    seen_t = set()
    restore(sim, sfail["snap"])
    freeze(sim)
    go_slip_start(sim, gains, sfail, theta, [], [])
    t_now = float(sim.data.time)
    targets = []
    if log_slip:
        idxs = [0]
        for f, i, row, pv, pmax in fracs:
            if i not in idxs:
                idxs.append(i)
        if sat_i not in idxs:
            idxs.append(sat_i)
        idxs = sorted(set(idxs))
        # step through slip logging rich at those indices
        k = 0
        nall = int(round(SLIP_MAX_S / dt))
        log_i = 0
        want = set(idxs)
        for i in range(nall + 1):
            if log_i in want:
                lab = "start" if log_i == 0 else ("saturation" if log_i == sat_i else f"idx{log_i}")
                for f, ii, row, pv, pmax in fracs:
                    if ii == log_i:
                        lab = f"{int(round(f*100))}pct_prog"
                if log_i == sat_i:
                    lab = "saturation"
                evo.append({
                    "label": lab,
                    "log_i": log_i,
                    "prog_mm": useful_mm(measure(sim, TAU_WEAK)["rh"], rh_s, u),
                    **rich_sample(sim, TAU_WEAK, u),
                })
                want.discard(log_i)
                if not want:
                    break
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_WEAK, gains)
            if i % max(int(round(LOG_DT / dt)), 1) == 0:
                log_i += 1
        dump_json(RAW / "evolution_samples.json", jsonable(evo))

    # H1-H4 numbers
    g0 = np.asarray(rec_start["g_h"], float)
    gs = np.asarray(rec_sat["g_h"], float)
    h = {
        "g_h_slip_start": g0.tolist(),
        "g_h_saturation": gs.tolist(),
        "g_h_delta": (gs - g0).tolist(),
        "g_dot_u_start": float(g0 @ u),
        "g_dot_u_sat": float(gs @ u),
        "g_xz_norm_start": float(np.hypot(g0[0], g0[2])),
        "g_xz_norm_sat": float(np.hypot(gs[0], gs[2])),
        "rho_start": rec_start["contacts"]["L"]["rho_max"] if rec_start["contacts"]["L"]["n"] else rec_start["measure"]["rho_max"],
        "rho_sat": rec_sat["measure"]["rho_max"],
        "Fn_L_start": rec_start["contacts"]["L"]["Fn"],
        "Fn_R_start": rec_start["contacts"]["R"]["Fn"],
        "Fn_L_sat": rec_sat["contacts"]["L"]["Fn"],
        "Fn_R_sat": rec_sat["contacts"]["R"]["Fn"],
        "centroid_L_start": rec_start["contacts"]["L"]["centroid_h"],
        "centroid_R_start": rec_start["contacts"]["R"]["centroid_h"],
        "centroid_L_sat": rec_sat["contacts"]["L"]["centroid_h"],
        "centroid_R_sat": rec_sat["contacts"]["R"]["centroid_h"],
        "span_L_start": rec_start["contacts"]["L"]["span_h"],
        "span_L_sat": rec_sat["contacts"]["L"]["span_h"],
        "nL_start": rec_start["measure"]["nL"],
        "nR_start": rec_start["measure"]["nR"],
        "nL_sat": rec_sat["measure"]["nL"],
        "nR_sat": rec_sat["measure"]["nR"],
        "cyl_h_start": rec_start["cyl_h"],
        "cyl_h_sat": rec_sat["cyl_h"],
        "rh_start": rec_start["measure"]["rh"],
        "rh_sat": rec_sat["measure"]["rh"],
        "v_along_start": rec_start["v_along_u"],
        "v_along_sat": rec_sat["v_along_u"],
        "wrist_g_start": rec_start["wrist_from_g"],
        "wrist_g_sat": rec_sat["wrist_from_g"],
    }
    h["H1_g_tangent_drop_frac"] = abs(h["g_dot_u_sat"] - h["g_dot_u_start"]) / (abs(h["g_dot_u_start"]) + 1e-12)
    h["H2_rho_drop"] = float(h["rho_start"] - h["rho_sat"]) if np.isfinite(h["rho_start"]) and np.isfinite(h["rho_sat"]) else None
    cL0 = np.array(h["centroid_L_start"] or [0, 0, 0], float)
    cLs = np.array(h["centroid_L_sat"] or [0, 0, 0], float)
    cR0 = np.array(h["centroid_R_start"] or [0, 0, 0], float)
    cRs = np.array(h["centroid_R_sat"] or [0, 0, 0], float)
    h["H3_centroid_L_mm"] = (1e3 * (cLs - cL0)).tolist()
    h["H3_centroid_R_mm"] = (1e3 * (cRs - cR0)).tolist()
    h["H3_cyl_delta"] = (np.array(h["cyl_h_sat"]) - np.array(h["cyl_h_start"])).tolist()

    # ----- 5 s hold tau=-4 tilted -----
    restore(sim, sat_snap)
    freeze(sim)
    m_a = measure(sim, TAU_WEAK)
    wdeg0 = wrist_deg(sim, Rh0)
    log5 = []
    hold_tau(sim, gains, TAU_WEAK, HOLD5_S, None, Rh0, m_a["rh"], "TILT_WEAK_5S", log5, u, [])
    m_b = measure(sim, TAU_WEAK)
    hold5 = {
        "tau": TAU_WEAK,
        "wrist_start": wdeg0,
        "wrist_end": wrist_deg(sim, Rh0),
        "rh0": m_a["rh"].tolist(),
        "rh1": m_b["rh"].tolist(),
        "d_rh_mm": (1e3 * (m_b["rh"] - m_a["rh"])).tolist(),
        "v0": m_a["v_rel_h"].tolist(),
        "v1": m_b["v_rel_h"].tolist(),
        "rho0": m_a["rho_max"],
        "rho1": m_b["rho_max"],
        "n0": [m_a["nL"], m_a["nR"]],
        "n1": [m_b["nL"], m_b["nR"]],
        "dropped": dropped(sim, m_b),
        "ctrl7_end": float(sim.data.ctrl[7]),
        "live": live_opt(sim),
    }
    log_to_npz(RAW / "hold5_tau4.npz", log5, u)
    dump_json(RAW / "hold5.json", jsonable(hold5))

    # ----- 2 s A/B from sat -----
    hold2 = {}
    for name, tau in (("A_tau-4", TAU_WEAK), ("B_tau-18", TAU_SEC)):
        restore(sim, sat_snap)
        freeze(sim)
        ma = measure(sim, tau)
        lg = []
        hold_tau(sim, gains, tau, HOLD2_S, None, Rh0, ma["rh"], f"HOLD2_{name}", lg, u, [])
        mb = measure(sim, tau)
        hold2[name] = {
            "tau": tau,
            "ctrl7_end": float(sim.data.ctrl[7]),
            "rh0": ma["rh"].tolist(),
            "rh1": mb["rh"].tolist(),
            "d_rh_mm": (1e3 * (mb["rh"] - ma["rh"])).tolist(),
            "v0_norm": float(np.linalg.norm(ma["v_rel_h"])),
            "v1_norm": float(np.linalg.norm(mb["v_rel_h"])),
            "w0_norm": float(np.linalg.norm(ma["w_rel_h"])),
            "w1_norm": float(np.linalg.norm(mb["w_rel_h"])),
            "n0": [ma["nL"], ma["nR"]],
            "n1": [mb["nL"], mb["nR"]],
            "Fn0": [ma["Fn_L"], ma["Fn_R"]],
            "Fn1": [mb["Fn_L"], mb["Fn_R"]],
            "rho0": ma["rho_max"],
            "rho1": mb["rho_max"],
        }
        log_to_npz(RAW / f"hold2_{name}.npz", lg, u)
    dump_json(RAW / "hold2.json", jsonable(hold2))

    # ----- return branches from sat -----
    returns_vert = {}
    continuation = {}
    branch_logs = {}

    def _sum_vert(m):
        return {
            "rh": m["rh"].tolist(),
            "e_x_mm": 1e3 * float(m["e_x"]),
            "e_z_mm": 1e3 * float(m["e_z"]),
            "nL": m["nL"],
            "nR": m["nR"],
            "rho": m["rho_max"],
            "v_norm": float(np.linalg.norm(m["v_rel_h"])),
            "w_norm": float(np.linalg.norm(m["w_rel_h"])),
            "cyl_h": m["cyl_h"].tolist(),
            "wrist": None,
            "ctrl7": float(sim.data.ctrl[7]),
            "obj_z": m["obj_z"],
        }

    def _sum_cont(c):
        end = c.get("end") or {}
        return {
            "lift": c.get("lift_completed"),
            "drop": c.get("drop"),
            "t_drop": c.get("t_drop"),
            "t_both_loss": c.get("t_both_loss"),
            "hold_survived": c.get("hold_survived"),
            "end_nL": end.get("nL"),
            "end_nR": end.get("nR"),
            "end_e_x_mm": None if end.get("e_x") is None else 1e3 * end["e_x"],
            "end_obj_z": end.get("obj_z"),
            "end_v": None if end.get("v_rel_h") is None else float(np.linalg.norm(end["v_rel_h"])),
            "success_physical": task_ok(c),
            "ctrl7_end": end.get("ctrl"),
        }

    # WEAK_RETURN
    restore(sim, sat_snap)
    freeze(sim)
    lg = []
    return_vertical(sim, gains, TAU_WEAK, None, Rh0, rh0, lg, omega=OMEGA_RET)
    freeze(sim)
    vh = hold_tau(sim, gains, TAU_WEAK, VERT_HOLD, None, Rh0, rh0, "VERT_WEAK", lg, u, [])
    returns_vert["WEAK_RETURN"] = _sum_vert(measure(sim, TAU_WEAK))
    freeze(sim)
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    # continuation uses TAU_SEC inside continue_nominal — document that
    cont = continue_nominal(sim, cfg, gains, viz=None, Rh0=Rh0, rh0=rh0)
    continuation["WEAK_RETURN"] = _sum_cont(cont)
    continuation["WEAK_RETURN"]["note"] = "vertical hold was tau=-4; continuation function always uses tau=-18"
    log_to_npz(RAW / "branch_weak_return.npz", lg + cont.get("log", []), u)

    # SECURE_RETURN: tau=-18 immediately, no dwell
    restore(sim, sat_snap)
    freeze(sim)
    lg = []
    return_vertical(sim, gains, TAU_SEC, None, Rh0, rh0, lg, omega=OMEGA_RET)
    freeze(sim)
    hold_tau(sim, gains, TAU_SEC, VERT_HOLD, None, Rh0, rh0, "VERT_SEC", lg, u, [])
    returns_vert["SECURE_RETURN"] = _sum_vert(measure(sim, TAU_SEC))
    freeze(sim)
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    cont = continue_nominal(sim, cfg, gains, viz=None, Rh0=Rh0, rh0=rh0)
    continuation["SECURE_RETURN"] = _sum_cont(cont)
    log_to_npz(RAW / "branch_secure_return.npz", lg + cont.get("log", []), u)

    # BRAKE_DWELL_RETURN
    restore(sim, sat_snap)
    freeze(sim)
    lg = []
    hold_tau(sim, gains, TAU_SEC, BRAKE_HOLD, None, Rh0, rh0, "BRAKE_DWELL", lg, u, [])
    return_vertical(sim, gains, TAU_SEC, None, Rh0, rh0, lg, omega=OMEGA_RET)
    freeze(sim)
    hold_tau(sim, gains, TAU_SEC, VERT_HOLD, None, Rh0, rh0, "VERT_BRK", lg, u, [])
    returns_vert["BRAKE_DWELL_RETURN"] = _sum_vert(measure(sim, TAU_SEC))
    freeze(sim)
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    cont = continue_nominal(sim, cfg, gains, viz=None, Rh0=Rh0, rh0=rh0)
    continuation["BRAKE_DWELL_RETURN"] = _sum_cont(cont)
    log_to_npz(RAW / "branch_brake_dwell.npz", lg + cont.get("log", []), u)

    # ZERO / ROTATE_ONLY / SLIP_SELF_ARREST summaries
    z = run_zero(sim, cfg, gains, sfail, viz=None)
    continuation["ZERO"] = _sum_cont(z)
    continuation["ZERO"]["class"] = z.get("class")
    log_to_npz(RAW / "zero.npz", z.get("log") or [], u)

    restore(sim, sfail["snap"])
    freeze(sim)
    lg = []
    Rh0, rh0 = go_slip_start(sim, gains, sfail, theta, lg, [])
    ang = return_vertical(sim, gains, TAU_SEC, None, Rh0, rh0, lg, omega=OMEGA_RET)
    freeze(sim)
    hold_tau(sim, gains, TAU_SEC, VERT_HOLD, None, Rh0, rh0, "VERT", lg, u, [])
    freeze(sim)
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    cont = continue_nominal(sim, cfg, gains, viz=None, Rh0=Rh0, rh0=rh0)
    continuation["ROTATE_ONLY"] = _sum_cont(cont)
    log_to_npz(RAW / "rotate_only.npz", lg + cont.get("log", []), u)

    continuation["SLIP_SELF_ARREST"] = {
        "note": "remain +30 deg, tau=-4 after saturation; no return, no lift",
        "hold5": hold5,
        "long_lived_tilted": (not hold5["dropped"] and abs(hold5["d_rh_mm"][0]) < 0.5 and hold5["n1"][0] > 0 and hold5["n1"][1] > 0),
    }

    dump_json(RAW / "returns_vert.json", jsonable(returns_vert))
    dump_json(RAW / "continuation.json", jsonable(continuation))

    # figures
    if len(t):
        tr = t - t0
        fig, ax = plt.subplots(5, 1, figsize=(8.2, 10.2), sharex=True)
        ax[0].plot(tr, 1e3 * rh_tr[:, 0], label="r_h.x")
        ax[0].plot(tr, 1e3 * rh_tr[:, 1], label="r_h.y")
        ax[0].plot(tr, 1e3 * (rh_tr[:, 2] - RH_Z_NOM), label="r_h.z-nom")
        ax[0].set_ylabel("mm")
        ax[0].legend(fontsize=7)
        ax[0].set_title("tau=-4 slip at +30 deg (wrist frozen)")
        ax[1].plot(tr, 1e3 * v_u, label="v·û_x")
        ax[1].plot(tr, 1e3 * v_g, label="v·ĝ_xz0")
        if sat_t is not None:
            ax[1].axvline(sat_t - t0, c="k", ls="--", lw=0.9, label="sat")
        ax[1].set_ylabel("v along (mm/s)")
        ax[1].legend(fontsize=7)
        ax[2].plot(tr, prog)
        ax[2].set_ylabel("useful mm")
        ax[3].plot(tr, rho)
        ax[3].set_ylabel("rho_max")
        ax[4].plot(tr, g_dot)
        ax[4].set_ylabel("g_h·û")
        ax[4].set_xlabel("t from slip start (s)")
        for a in ax:
            a.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "fig_slip_decay.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        names = ["ZERO", "ROTATE_ONLY", "WEAK_RETURN", "SECURE_RETURN", "BRAKE_DWELL_RETURN"]
        xs = np.arange(len(names))
        survived = [1.0 if continuation.get(n, {}).get("success_physical") or continuation.get(n, {}).get("hold_survived") else 0.0 for n in names]
        ax.bar(xs, survived, color=["#c44" if s < 0.5 else "#4a4" for s in survived])
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("physical survive (0/1)")
        ax.set_title("Continuation after matched S_FAIL")
        ax.set_ylim(0, 1.2)
        fig.tight_layout()
        fig.savefig(FIG / "fig_continuation.png", dpi=140)
        plt.close(fig)

    # perturbation s=1.9, 2.1
    pert = []
    if PARENT_PKL.is_file():
        with PARENT_PKL.open("rb") as f:
            parent = pickle.load(f)
        ref = load_or_measure_ref()
        for s in (1.9, 2.0, 2.1):
            ic = apply_sfail(sim, parent, ref, s)
            zz = run_zero(sim, cfg, gains, ic, viz=None)
            cls = classify_zero(zz, bool(ic["ok"]))
            rec = {"s": s, "t0_ok": ic["ok"], "class": cls, "drop": zz.get("drop"), "t_drop": zz.get("t_drop"),
                   "hold_survived": zz.get("hold_survived")}
            if ic["ok"] and cls == "DELAYED_FAIL":
                # tau=-4 slip 1.5s from this IC
                lg = []
                go_slip_start(sim, gains, ic, theta, lg, [])
                rh_i = measure(sim, TAU_WEAK)["rh"].copy()
                lost_p = slip_window(sim, gains, TAU_WEAK, u, 1.5, None, Rh0, rh_i, lg, [], "SLIP")
                last = lg[-1] if lg else measure(sim, TAU_WEAK)
                rec["tau4_lost"] = bool(lost_p)
                rec["tau4_useful_mm"] = useful_mm(last["rh"], rh_i, u)
                rec["tau4_end_n"] = [last["nL"], last["nR"]]
                rec["tau4_end_rho"] = last["rho_max"]
                rec["tau4_kind"] = "LOSS" if lost_p else ("STICK" if abs(rec["tau4_useful_mm"]) < 0.35 else "CONTROLLED_SLIP")
            pert.append(rec)
    dump_json(RAW / "perturbation.json", jsonable(pert))

    # hypothesis text from numbers
    h1_support = h["H1_g_tangent_drop_frac"] > 0.25
    h2_support = h["H2_rho_drop"] is not None and h["H2_rho_drop"] > 0.08
    h3_support = float(np.linalg.norm(h["H3_centroid_L_mm"])) > 0.4 or float(np.linalg.norm(h["H3_centroid_R_mm"])) > 0.4
    geom_still = h["H1_g_tangent_drop_frac"] < 0.08
    load_still = (h["H2_rho_drop"] is not None and abs(h["H2_rho_drop"]) < 0.05)
    h4_support = geom_still and (load_still or not h2_support) and abs(h["v_along_sat"]) < 8e-4
    hyp_text = []
    hyp_text.append(
        f"H1 GRAVITY ALIGNMENT: g_h·û start={h['g_dot_u_start']:.3f} sat={h['g_dot_u_sat']:.3f} "
        f"(relative change {h['H1_g_tangent_drop_frac']:.3f}). Wrist is frozen so g_h should be nearly constant. "
        + ("Supported if g tangent collapsed. " if h1_support else "**Contradicted** if g_h driving component stays large. ")
    )
    hyp_text.append(
        f"H2 INCREASED FRICTIONAL SUPPORT: ρ start={h['rho_start']} sat={h['rho_sat']} drop={h['H2_rho_drop']}. "
        + ("Supported: ρ falls as motion arrests. " if h2_support else "Not a large ρ collapse. ")
    )
    hyp_text.append(
        f"H3 GEOMETRIC CONTACT EQUILIBRIUM: Δcentroid L {h['H3_centroid_L_mm']} mm, R {h['H3_centroid_R_mm']} mm; "
        f"Δcyl {h['H3_cyl_delta']}. "
        + ("Supported: contact geometry moved while g tangent remains. " if h3_support and not h1_support else "Geometry change is small or mixed with other hypotheses. ")
    )
    hyp_text.append(
        "H4 NUMERICAL/NOSLIP ARREST: inferred only if gravity load and contact geometry stay essentially "
        "unchanged while velocity goes to ~0. noslip was **not** toggled (forbidden). "
        + ("Compatible with H4 (load/geometry static). " if h4_support else "Not the primary reading if ρ or contacts evolved. ")
    )
    if hold5["dropped"]:
        hyp_text.append("5 s tilted hold **dropped** — saturation was transient.")
    else:
        dx = abs(hold5["d_rh_mm"][0])
        dx_mm = float(hold5['d_rh_mm'][0])
        extra = 'Long-lived tilted equilibrium under this contact model.' if dx < 0.5 else 'Continued creep after the short-window saturation.'
        hyp_text.append('5 s tilted tau=-4 hold: d r_h.x=%.3f mm, dropped=%s. %s' % (dx_mm, hold5['dropped'], extra))

    wr = continuation.get("WEAK_RETURN", {})
    sr = continuation.get("SECURE_RETURN", {})
    br = continuation.get("BRAKE_DWELL_RETURN", {})
    verdict_lines = [
        "Legacy FULL vs NO_BRAKE is ambiguous: NO_BRAKE already restores τ=−18 at VERT_HOLD and lift.",
        f"From the **same saturation snapshot**: WEAK_RETURN survive={wr.get('success_physical')} "
        f"(vert hold was τ=−4; lift still τ=−18), SECURE_RETURN survive={sr.get('success_physical')}, "
        f"BRAKE_DWELL_RETURN survive={br.get('success_physical')}.",
        f"2 s tilted hold A (τ=−4) Δr_h.x={hold2['A_tau-4']['d_rh_mm'][0]:.3f} mm vs B (τ=−18) "
        f"Δr_h.x={hold2['B_tau-18']['d_rh_mm'][0]:.3f} mm.",
    ]
    if sr.get("success_physical") and br.get("success_physical"):
        if wr.get("success_physical"):
            verdict_lines.append(
                "Brake dwell is **not** required for this S_FAIL: restoring secure grip at/after saturation, "
                "or even returning weak and then securing at vertical/lift, already survives. "
                "Do not keep BRAKE as a mandatory FSM state from this IC."
            )
        else:
            verdict_lines.append(
                "Immediate τ=−18 return matches dwell+return at task level. The useful extra vs WEAK_RETURN "
                "is **restoring secure grip**, not the 0.5 s tilted dwell."
            )
    elif not wr.get("success_physical") and sr.get("success_physical"):
        verdict_lines.append("Secure grip before/during return matters; extra 0.5 s dwell may not.")
    else:
        verdict_lines.append("See continuation JSON; do not force a BRAKE-mandatory conclusion.")

    unique_ctrl = []
    seen = set()
    for row in ctrl_nb:
        key = (row["phase"], round(row["tau_cmd"], 2), round(row["ctrl7"], 2))
        if key in seen:
            continue
        seen.add(key)
        unique_ctrl.append({"phase": row["phase"], "tau_cmd": row["tau_cmd"], "ctrl7": row["ctrl7"],
                            "v_cmd": row["v_cmd"], "w_cmd": row["w_cmd"], "wrist_deg": row["wrist_deg"]})

    sat_report = {
        "lost_during_3s_slip": bool(lost),
        "t_slip0": t0,
        "n_samples": int(len(log_slip)),
        "v_along_u_initial": float(v_u[0]) if len(v_u) else None,
        "v_along_u_peak": float(v_u[np.argmax(np.abs(v_u))]) if len(v_u) else None,
        "v_along_u_end": float(v_u[-1]) if len(v_u) else None,
        "peak_useful_mm": float(np.max(prog)) if len(prog) else None,
        "useful_at_sat_mm": float(prog[sat_i]) if sat_i is not None and sat_i < len(prog) else None,
        "sat_time_from_slip_s": None if sat_t is None else float(sat_t - t0),
        "thresholds_s": sat_tab,
        "rh0": rh_tr[0].tolist() if len(rh_tr) else None,
        "rh_sat": rh_tr[sat_i].tolist() if sat_i is not None and sat_i < len(rh_tr) else None,
        "u": u.tolist(),
        "u_g_xz0": u_g.tolist(),
    }

    meta = {
        "live": opt,
        "semantics_measured": unique_ctrl,
        "saturation": sat_report,
        "evolution": [{
            "label": e["label"],
            "prog_mm": e["prog_mm"],
            "rh": e["measure"]["rh"],
            "cyl_h": e["cyl_h"],
            "g_h": e["g_h"],
            "g_dot_u": e["g_dot_u"],
            "nL": e["measure"]["nL"],
            "nR": e["measure"]["nR"],
            "Fn_L": e["contacts"]["L"]["Fn"],
            "Fn_R": e["contacts"]["R"]["Fn"],
            "Ft_L": e["contacts"]["L"]["Ft"],
            "Ft_R": e["contacts"]["R"]["Ft"],
            "rho_L": e["contacts"]["L"]["rho_max"],
            "rho_R": e["contacts"]["R"]["rho_max"],
            "centroid_L": e["contacts"]["L"]["centroid_h"],
            "centroid_R": e["contacts"]["R"]["centroid_h"],
            "span_L": e["contacts"]["L"]["span_h"],
            "span_R": e["contacts"]["R"]["span_h"],
            "v_rel_h": e["measure"]["v_rel_h"],
            "w_rel_h": e["measure"]["w_rel_h"],
            "ctrl7": e["ctrl7"],
        } for e in evo],
        "hypotheses": h,
        "hypotheses_text": "\n\n".join(hyp_text),
        "hold5": hold5,
        "hold2": hold2,
        "returns_vert": returns_vert,
        "continuation": continuation,
        "brake_verdict": " ".join(verdict_lines),
        "perturbation": pert,
        "post_active_brake_rh": rec_br["measure"]["rh"],
        "pre_return_nobrake_rh": rec_pr["measure"]["rh"],
    }
    dump_json(RAW / "summary.json", jsonable(meta))
    write_report(meta)
    return meta


if __name__ == "__main__":
    construction()
