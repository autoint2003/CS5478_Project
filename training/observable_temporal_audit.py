"""Observability / temporal-state audit. Frozen constructions only.

Does not change 27D obs, reward, terminal, physics, or recovery actions.
Does not train a detector or SAC. No MP4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.demo_airborne_recapture as dar
import training.demo_ballistic_recovery as dbr
import training.map_ballistic_disturbance as mbd
import training.replay_core as rc
from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.observable_obs import (
    HAND_Z_REF,
    OBS_DIM_OBSERVABLE,
    OBS_NAMES,
    OBS_SOURCES,
    SCALE_APDOT,
    SCALE_PAD_V,
    SCALE_TRACK_Z,
    ObservableObsState,
    observe_observable,
)
from envs.observable_reward import read_tactile
from envs.physical_recovery import (
    SCALE_AP,
    SCALE_EX,
    SCALE_F,
    SCALE_G,
    SCALE_TAU,
    SCALE_V,
    SCALE_W,
    SCALE_Z,
    TAU_OPEN,
    TAU_SECURE,
    physical_pack,
)
from sensors.spatial_tactile import MIN_COP_FORCE_N, PAD1_HALF
from training.ballistic_slip_sufficiency import run_duration
from training.demo_airborne_recapture import (
    advance_to_parent,
    make_parent_sim,
    save_parent,
    try_recapture,
)
from training.demo_ballistic_impact import make_sim
from training.demo_ballistic_recovery import continue_zero, restore_ballistic, run_zero_full
from training.demo_teleport_recovery_state import CYL_MASS, PAIR_MU
from training.validate_tactile_ex_estimator import degrade_stream

OUT = ROOT / "results" / "diagnostics" / "observable_temporal_audit"
RAW = OUT / "raw"
FIG = OUT / "figures"
DT_SIM = 0.002
DT_POLICY = 0.020
WINDOWS = (0.020, 0.050, 0.100, 0.200, 0.500)
PAD_U_HALF = float(PAD1_HALF[0])

OBS_UNITS = {
    "e_hat_x": "m (normalized by SCALE_EX=7.5 mm)",
    "estimate_valid": "bool",
    "contact_present_L": "bool",
    "contact_present_R": "bool",
    "valid_L": "bool",
    "valid_R": "bool",
    "u_L": "m (normalized by SCALE_EX)",
    "v_L": "m (normalized by SCALE_PAD_V=8.5 mm)",
    "u_R": "m (normalized by SCALE_EX)",
    "v_R": "m (normalized by SCALE_PAD_V)",
    "fn_L": "N (normalized by SCALE_F=20 N)",
    "fn_R": "N (normalized by SCALE_F)",
    "aperture": "m (normalized by SCALE_AP=40 mm)",
    "ap_dot": "m/s (normalized by SCALE_APDOT=0.08)",
    "tau_from_secure": "1; (ctrl[7]+18)/SCALE_TAU, SCALE_TAU uses TAU_OPEN=-1",
    "g_hx": "m/s^2 / 9.81",
    "g_hy": "m/s^2 / 9.81",
    "g_hz": "m/s^2 / 9.81",
    "v_hx": "m/s / 0.08",
    "v_hy": "m/s / 0.08",
    "v_z_world": "m/s / 0.08",
    "w_hx": "rad/s / 2.0",
    "w_hy": "rad/s / 2.0",
    "w_hz": "rad/s / 2.0",
    "hand_z_off_table": "m / 0.25 (hand_z - TABLE_TOP)",
    "track_z": "m / 0.08 (p_des_z - hand_z)",
    "e_hat_dot": "m/s / 0.08; (Δe_hat)/dt_policy if consecutive valid else 0",
}

OBS_CLASS = {
    "e_hat_x": "ESTIMATED",
    "estimate_valid": "ESTIMATED",
    "contact_present_L": "SIM_PROXY",
    "contact_present_R": "SIM_PROXY",
    "valid_L": "SIM_PROXY",
    "valid_R": "SIM_PROXY",
    "u_L": "SIM_PROXY",
    "v_L": "SIM_PROXY",
    "u_R": "SIM_PROXY",
    "v_R": "SIM_PROXY",
    "fn_L": "SIM_PROXY",
    "fn_R": "SIM_PROXY",
    "aperture": "ROBOT_STATE",
    "ap_dot": "ROBOT_STATE",
    "tau_from_secure": "ROBOT_STATE",
    "g_hx": "ROBOT_STATE",
    "g_hy": "ROBOT_STATE",
    "g_hz": "ROBOT_STATE",
    "v_hx": "ROBOT_STATE",
    "v_hy": "ROBOT_STATE",
    "v_z_world": "ROBOT_STATE",
    "w_hx": "ROBOT_STATE",
    "w_hy": "ROBOT_STATE",
    "w_hz": "ROBOT_STATE",
    "hand_z_off_table": "ROBOT_STATE",
    "track_z": "ROBOT_STATE",
    "e_hat_dot": "ESTIMATED",
}

HW = {
    "e_hat_x": "not hardware; geometric map of SIM_PROXY CoP",
    "estimate_valid": "not hardware",
    "contact_present_L": "MuJoCo finger-object geom contact; not a Panda taxel array",
    "contact_present_R": "same",
    "valid_L": "CoP defined iff Fn sum >= 0.01 N",
    "valid_R": "same",
    "u_L": "MuJoCo contact.pos mapped to pad frame; SIMULATION PROXY",
    "v_L": "same",
    "u_R": "same; right-finger X = -hand X",
    "v_R": "same",
    "fn_L": "mj_contactForce wrench[0] abs sum",
    "fn_R": "same",
    "aperture": "finger joint mean; Panda encoder-class",
    "ap_dot": "finger qvel mean",
    "tau_from_secure": "commanded ctrl[7], not a force sensor",
    "g_hx": "FK orientation + known gravity",
    "g_hy": "same",
    "g_hz": "same",
    "v_hx": "hand body twist from MuJoCo (robot-state class)",
    "v_hy": "same",
    "v_z_world": "same",
    "w_hx": "same",
    "w_hy": "same",
    "w_hz": "same",
    "hand_z_off_table": "FK z minus scene TABLE_TOP=0.40",
    "track_z": "controller p_des_z minus hand_z",
    "e_hat_dot": "finite difference of e_hat at dt_policy=20 ms",
}


def dump(path: Path, obj) -> None:
    def fix(x):
        if isinstance(x, dict):
            return {str(k): fix(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [fix(v) for v in x]
        if isinstance(x, np.ndarray):
            return fix(x.tolist())
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fix(obj), indent=2), encoding="utf-8")


class Recorder:
    def __init__(self, sim, name: str):
        self.sim = sim
        self.name = name
        self.rows = []
        self.hist_pol = ObservableObsState(DT_POLICY)
        self._n = 0
        self.last_obs = None
        self.last_ehat_dot_pol = 0.0
        self.phase_hint = ""

    def record(self, sim=None):
        sim = sim or self.sim
        if sim is not self.sim:
            return
        _, tac = read_tactile(sim.model, sim.data, sim.ids)
        o = physical_pack(sim)
        n_sub = int(getattr(sim, "n_sub", 10))
        if self._n % n_sub == 0:
            self.last_obs = observe_observable(
                sim.model, sim.data, sim.ids, sim.fsm, tac, self.hist_pol
            )
            self.last_ehat_dot_pol = float(self.last_obs[26]) * SCALE_V
        self._n += 1
        obs = self.last_obs
        rh = np.asarray(o["rh"], float)
        vrel = np.asarray(o["v_rel_h"], float)
        wrel = np.asarray(o["w_rel_h"], float)
        vh = np.asarray(o.get("v_hand", np.zeros(3)), float) if "v_hand" in o else None
        from envs.deterioration import body_twist

        vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
        vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
        self.rows.append(
            {
                "t": float(sim.data.time),
                "phase_hint": self.phase_hint,
                "OBS_ehat": float(tac["e_hat_x"]) if tac["estimate_valid"] else np.nan,
                "OBS_ehat_valid": float(tac["estimate_valid"]),
                "OBS_cpL": float(tac["contact_present_L"]),
                "OBS_cpR": float(tac["contact_present_R"]),
                "OBS_validL": float(tac["valid_L"]),
                "OBS_validR": float(tac["valid_R"]),
                "OBS_uL": float(tac["u_L"]) if tac["valid_L"] else np.nan,
                "OBS_vL": float(tac["v_L"]) if tac["valid_L"] else np.nan,
                "OBS_uR": float(tac["u_R"]) if tac["valid_R"] else np.nan,
                "OBS_vR": float(tac["v_R"]) if tac["valid_R"] else np.nan,
                "OBS_fnL": float(tac["fn_L"]),
                "OBS_fnR": float(tac["fn_R"]),
                "OBS_ap": float(o["aperture"]),
                "OBS_apdot": float(np.mean(sim.data.qvel[sim.ids.finger_dof])),
                "OBS_ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else np.nan,
                "OBS_ghx": float(o["g_h"][0]),
                "OBS_ghy": float(o["g_h"][1]),
                "OBS_ghz": float(o["g_h"][2]),
                "OBS_vhx": float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ vh)[0]),
                "OBS_vhy": float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ vh)[1]),
                "OBS_vz": float(vh[2]),
                "OBS_whx": float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ wh)[0]),
                "OBS_why": float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ wh)[1]),
                "OBS_whz": float((np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).T @ wh)[2]),
                "OBS_hand_z_off": float(sim.data.xpos[sim.ids.hand_body][2] - HAND_Z_REF),
                "OBS_track_z": float(sim.fsm.p_des[2] - sim.data.xpos[sim.ids.hand_body][2]),
                "OBS_ehat_dot_policy": self.last_ehat_dot_pol,
                "OBS_27": None if obs is None else np.array(obs, np.float32),
                "GT_rhx": float(rh[0]),
                "GT_rhy": float(rh[1]),
                "GT_rhz": float(rh[2]),
                "GT_vrel": float(np.linalg.norm(vrel)),
                "GT_vrel_x": float(vrel[0]),
                "GT_wrel": float(np.linalg.norm(wrel)),
                "GT_nL": int(o["nL"]),
                "GT_nR": int(o["nR"]),
                "GT_obj_z": float(o["obj_z"]),
                "GT_clear": float(o.get("clear", np.nan)),
                "GT_scene": int(o["scene"]),
                "GT_vo_z": float(vo[2]),
            }
        )

    def to_npz(self, path: Path) -> dict:
        rows = self.rows
        if not rows:
            return {}
        keys = [k for k in rows[0] if k not in ("OBS_27", "phase_hint")]
        arr = {k: np.array([r[k] for r in rows], float) for k in keys}
        arr["phase_hint"] = np.array([r["phase_hint"] for r in rows])
        o27 = []
        for r in rows:
            o27.append(np.full(27, np.nan, np.float32) if r["OBS_27"] is None else r["OBS_27"])
        arr["OBS_27"] = np.stack(o27)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arr)
        return arr


def _patch_tick(rec: Recorder):
    orig = rc.tick_vw

    def wrapped(sim, *a, **k):
        out = orig(sim, *a, **k)
        rec.record(sim)
        return out

    rc.tick_vw = wrapped
    dbr.tick_vw = wrapped
    mbd.tick_vw = wrapped
    dar.tick_vw = wrapped
    orig_ps = rec.sim.physics_step

    def ps(*a, **k):
        out = orig_ps(*a, **k)
        rec.record(rec.sim)
        return out

    rec.sim.physics_step = ps
    return orig, orig_ps


def _unpatch(sim, orig_tick, orig_ps):
    rc.tick_vw = orig_tick
    dbr.tick_vw = orig_tick
    mbd.tick_vw = orig_tick
    dar.tick_vw = orig_tick
    sim.physics_step = orig_ps


def causal_delta(t, x, window):
    out = np.full_like(x, np.nan, dtype=float)
    j = 0
    for i in range(len(t)):
        tgt = t[i] - window
        while j < i and t[j] < tgt - 1e-12:
            j += 1
        if t[i] - t[j] >= window - 1e-6 and np.isfinite(x[i]) and np.isfinite(x[j]):
            out[i] = x[i] - x[j]
    return out


def causal_mean(t, x, window):
    out = np.full_like(x, np.nan, dtype=float)
    j = 0
    for i in range(len(t)):
        tgt = t[i] - window
        while j < i and t[j] < tgt - 1e-12:
            j += 1
        sl = x[j : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            out[i] = float(np.mean(sl))
    return out


def causal_var(t, x, window):
    out = np.full_like(x, np.nan, dtype=float)
    j = 0
    for i in range(len(t)):
        tgt = t[i] - window
        while j < i and t[j] < tgt - 1e-12:
            j += 1
        sl = x[j : i + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size >= 3:
            out[i] = float(np.var(sl))
    return out


def summarize(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def overlap_frac(a, b):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return None
    lo = max(np.percentile(a, 10), np.percentile(b, 10))
    hi = min(np.percentile(a, 90), np.percentile(b, 90))
    span = max(np.percentile(np.concatenate([a, b]), 90) - np.percentile(np.concatenate([a, b]), 10), 1e-12)
    return float(max(0.0, hi - lo) / span)


def collect_zero():
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    rec = Recorder(sim, "CENTER6_ZERO")
    orig, ops = _patch_tick(rec)
    try:
        ep = run_zero_full(sim, gains)
    finally:
        _unpatch(sim, orig, ops)
    arr = rec.to_npz(RAW / "CENTER6_ZERO.npz")
    try:
        dump(RAW / "CENTER6_ZERO_meta.json", {k: ep[k] for k in ep if k not in ("rows", "snaps", "log")})
    except Exception as e:
        dump(RAW / "CENTER6_ZERO_meta.json", {"dump_error": str(e), "t_table_contact": ep.get("t_table_contact")})
    return arr, ep


def collect_recovery(slip_s: float, tag: str):
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    from training.ballistic_large_angle import load_early

    snap = load_early()
    plan = {
        "theta_deg": 120.0,
        "tau_slip": -5.0,
        "slip_s": slip_s,
        "brake_s": 0.4,
        "omega_y": 3.0,
        "stage": "EARLY",
    }
    rec = Recorder(sim, tag)
    orig, ops = _patch_tick(rec)
    try:
        out = run_duration(sim, gains, snap, plan, slip_s)
    finally:
        _unpatch(sim, orig, ops)
    arr = rec.to_npz(RAW / f"{tag}.npz")
    slim = {k: out[k] for k in out if k not in ("log", "hold") and not str(k).startswith("geo")}
    dump(RAW / f"{tag}_meta.json", slim)
    return arr, out


def collect_recapture():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim = make_parent_sim()
    packed = advance_to_parent(sim, cfg)
    snap = save_parent(sim, packed)
    rec = Recorder(sim, "CLEAN_RECAPTURE")
    orig, ops = _patch_tick(rec)
    try:
        tr = try_recapture(sim, gains, snap, 2.0, 0.04, 0.0)
    finally:
        _unpatch(sim, orig, ops)
    arr = rec.to_npz(RAW / "CLEAN_RECAPTURE.npz")
    dump(RAW / "CLEAN_RECAPTURE_meta.json", {k: tr[k] for k in tr if k != "log"})
    return arr, tr, sim, gains, snap


def recapture_offsets(sim, gains, snap):
    rows = []
    hold_old = dar.HOLD_S
    dar.HOLD_S = 0.40
    try:
        for extra in (0.020, 0.030, 0.040, 0.050, 0.060):
            tr = try_recapture(sim, gains, snap, 2.0, extra, 0.0)
            ev = tr["events"]
            rows.append(
                {
                    "extra_after_both": extra,
                    "close_vs_nominal_ms": 1e3 * (extra - 0.040),
                    "captured": tr["captured"],
                    "ok": tr["ok"],
                    "table": tr["table"],
                    "both_off_s": tr["both_off_s"],
                    "FIRST_RECONTACT": ev.get("FIRST_RECONTACT"),
                    "FIRST_BILATERAL": ev.get("FIRST_BILATERAL"),
                    "SECURE_CAPTURE": ev.get("SECURE_CAPTURE"),
                    "end_nL": tr["end"]["nL"],
                    "end_nR": tr["end"]["nR"],
                    "end_obj_z": tr["end"]["obj_z"],
                    "end_v_rel": tr["end"]["v_rel"],
                }
            )
            print("offset", extra, "ok", tr["ok"], "cap", tr["captured"], "table", tr["table"], flush=True)
    finally:
        dar.HOLD_S = hold_old
    dump(RAW / "recapture_close_offsets.json", rows)
    return rows


def label_stability(arr, kind: str) -> np.ndarray:
    t = arr["t"]
    lab = np.array(["GT_OTHER"] * len(t), object)
    vrel = arr["GT_vrel"]
    nL, nR = arr["GT_nL"], arr["GT_nR"]
    rhx = arr["GT_rhx"]
    objz = arr["GT_obj_z"]
    if kind == "ZERO":
        # impact ~2.85 from frozen map; use first jump in |rhx| after 2.8
        impact = None
        for i in range(len(t)):
            if t[i] >= 2.85 and abs(rhx[i]) > 0.002:
                impact = t[i]
                break
        if impact is None:
            impact = 2.852
        for i in range(len(t)):
            if t[i] < impact + 0.05:
                lab[i] = "GT_IMPACT_TRANSIENT"
            elif objz[i] < 0.44:
                lab[i] = "GT_LOSS"
            elif (nL[i] == 0 and nR[i] == 0) or abs(rhx[i]) > 0.018 and vrel[i] > 0.03:
                lab[i] = "GT_RUNAWAY"
            elif t[i] > impact + 0.15 and nL[i] > 0 and nR[i] > 0:
                lab[i] = "GT_PROGRESSIVE"
            else:
                lab[i] = "GT_POST_IMPACT"
        # refine runaway: last 0.4 s with growing |rhx| and vrel
        for i in range(len(t)):
            if lab[i] == "GT_PROGRESSIVE" and abs(rhx[i]) >= 0.016 and vrel[i] > 0.015:
                lab[i] = "GT_RUNAWAY"
    else:
        # recovery: find last rotate/slip via aperture/g_hx then hold
        ghx = arr["OBS_ghx"]
        ctrl = arr["OBS_ctrl7"]
        for i in range(len(t)):
            if nL[i] > 0 and nR[i] > 0 and vrel[i] < 0.02 and abs(ghx[i]) < 2.5 and abs(ctrl[i] + 18) < 1:
                # candidate hold
                lab[i] = "GT_STABLE_EDGE" if kind == "070" else "GT_STABLE_MARGIN"
            elif abs(ctrl[i] + 5) < 1.5:
                lab[i] = "GT_WEAK_SLIP"
            elif abs(ghx[i]) > 4.0:
                lab[i] = "GT_ROTATE_OR_RETURN"
            else:
                lab[i] = "GT_SETTLING"
        # require persistence 0.20 s to keep STABLE
        persist = 0.0
        for i in range(len(t)):
            if str(lab[i]).startswith("GT_STABLE") and nL[i] > 0 and nR[i] > 0 and vrel[i] < 0.02:
                persist += DT_SIM
                if persist < 0.20:
                    lab[i] = "GT_SETTLING"
            else:
                persist = 0.0
                if str(lab[i]).startswith("GT_STABLE"):
                    lab[i] = "GT_SETTLING"
    return lab


def label_recapture(arr, events: dict) -> np.ndarray:
    t = arr["t"]
    lab = np.array(["GT_OTHER"] * len(t), object)
    ev = events
    t_open = ev.get("OPEN_START", t[0])
    t_off = ev.get("FIRST_BOTH_OFF", t_open)
    t_close = ev.get("CLOSE_START", t_off)
    t_rc = ev.get("FIRST_RECONTACT", t[-1])
    t_bi = ev.get("FIRST_BILATERAL", t_rc)
    t_sec = ev.get("SECURE_CAPTURE", t_bi)
    t_hold = ev.get("HOLD_START", t_sec)
    vrel = arr["GT_vrel"]
    nL, nR = arr["GT_nL"], arr["GT_nR"]
    for i, ti in enumerate(t):
        if ti < t_open - 1e-9:
            lab[i] = "GT_SECURE"
        elif ti < t_off - 1e-9:
            lab[i] = "GT_OPENING"
        elif ti < t_rc - 1e-9:
            lab[i] = "GT_FREE"
        elif ti < t_bi - 1e-9:
            lab[i] = "GT_RECONTACT"
        elif ti < t_sec - 1e-9:
            lab[i] = "GT_CAPTURE_DYNAMIC"
        elif nL[i] > 0 and nR[i] > 0 and vrel[i] < 0.02 and ti >= t_hold:
            lab[i] = "GT_CAPTURE_STABLE"
        else:
            lab[i] = "GT_CAPTURE_DYNAMIC"
    return lab


def mask_of(lab, *names):
    return np.array([x in names for x in lab], bool)


def plot_stability(zero, e70, e160, lz, l7, l16):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
    for a, arr, title in (
        (ax[0], zero, "CENTER6_ZERO"),
        (ax[1], e70, "RECOVERY_070"),
        (ax[2], e160, "RECOVERY_160"),
    ):
        t = arr["t"]
        a.plot(t, 1e3 * arr["GT_rhx"], label="GT r_h.x mm", lw=0.9)
        a.plot(t, 1e3 * arr["OBS_ehat"], label="OBS e_hat mm", lw=0.7, alpha=0.8)
        a.plot(t, 1e3 * arr["OBS_uL"], label="OBS u_L mm", lw=0.6, alpha=0.7)
        a.set_title(title)
        a.legend(fontsize=7, loc="upper left")
        a.set_ylabel("mm")
    ax[3].plot(zero["t"], 1e3 * np.abs(zero["OBS_uL"]), lw=0.8, label="ZERO |u_L| mm")
    ax[3].plot(e70["t"], 1e3 * np.abs(e70["OBS_uL"]), lw=0.8, label="070 |u_L|")
    ax[3].plot(e160["t"], 1e3 * np.abs(e160["OBS_uL"]), lw=0.8, label="160 |u_L|")
    ax[3].legend(fontsize=7)
    ax[3].set_ylabel("|u_L| mm")
    fig.tight_layout()
    fig.savefig(FIG / "stability_rh_ehat_u.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    for a, arr, title in (
        (ax[0], zero, "ZERO GT v_rel vs OBS |ehat_dot_policy|"),
        (ax[1], e70, "070"),
        (ax[2], e160, "160"),
    ):
        a.plot(arr["t"], arr["GT_vrel"], label="GT |v_rel|", lw=0.8)
        a.plot(arr["t"], np.abs(arr["OBS_ehat_dot_policy"]), label="|OBS e_hat_dot policy|", lw=0.8)
        du = causal_delta(arr["t"], arr["OBS_uL"], 0.100)
        a.plot(arr["t"], np.abs(du) / 0.100, label="|Δu_100ms|/0.1", lw=0.7, alpha=0.8)
        a.set_title(title)
        a.legend(fontsize=7)
        a.set_ylabel("m/s")
    fig.tight_layout()
    fig.savefig(FIG / "dehat_vs_gt_vrel.png", dpi=120)
    plt.close(fig)


def plot_recapture(arr, lab, events):
    t = arr["t"]
    fig, ax = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(t, 1e3 * arr["OBS_ap"], label="aperture mm")
    ax[0].plot(t, arr["OBS_apdot"] * 1e3, label="ap_dot mm/s")
    ax[1].step(t, arr["OBS_cpL"], where="post", label="OBS cpL")
    ax[1].step(t, arr["OBS_cpR"] + 1.05, where="post", label="OBS cpR+1")
    ax[1].step(t, arr["GT_nL"] * 0, where="post")
    ax[1].plot(t, (arr["GT_nL"] > 0).astype(float), "--", lw=0.6, label="GT nL>0")
    ax[2].plot(t, arr["OBS_fnL"], label="FnL")
    ax[2].plot(t, arr["OBS_fnR"], label="FnR")
    ax[3].plot(t, 1e3 * arr["OBS_uL"], label="uL")
    ax[3].plot(t, 1e3 * arr["OBS_uR"], label="uR")
    ax[4].plot(t, arr["GT_vrel"], label="GT |v_rel|")
    ax[4].plot(t, np.abs(arr["OBS_ehat_dot_policy"]), label="|ehat_dot pol|")
    for a in ax:
        for k, c in events.items():
            if isinstance(c, (int, float)):
                a.axvline(c, color="k", lw=0.5, alpha=0.5)
        a.legend(fontsize=7, loc="upper right")
    ax[0].set_title("CLEAN_RECAPTURE observable channels (GT event lines)")
    fig.tight_layout()
    fig.savefig(FIG / "recapture_obs.png", dpi=120)
    plt.close(fig)


def write_obs_table():
    lines = [
        "# Observable temporal audit",
        "",
        "No physics/action/obs/reward/terminal/SAC change. No detector. No MP4.",
        "GT_* is evaluation-only. OBS_* is the policy-available namespace.",
        "",
        "## 1–2. Current 27D observation (from `envs/observable_obs.py`, executed)",
        "",
        f"dim = {OBS_DIM_OBSERVABLE}. Policy rate: dt_policy = n_substeps×physics_dt = **20 ms**. "
        "Sim rate used for this audit: **2 ms**. `e_hat_dot` in the 27D vector uses **dt_policy**, not 2 ms.",
        "",
        "| i | name | class | units/scale | instantaneous vs derived | hardware |",
        "|---:|---|---|---|---|---|",
    ]
    inst = {
        "e_hat_dot": "derived (causal Δ / dt_policy)",
        "estimate_valid": "mask",
        "tau_from_secure": "command",
    }
    for i, n in enumerate(OBS_NAMES):
        der = inst.get(n, "instantaneous sample (clipped ±1)")
        lines.append(
            f"| {i} | `{n}` | **{OBS_CLASS[n]}** | {OBS_UNITS[n]} | {der} | {HW[n]} |"
        )
    lines += [
        "",
        "Code `OBS_SOURCES` labels several tactile fields SENSOR. This audit reclassifies them as **SIM_PROXY**: "
        "`measure_spatial_tactile` uses MuJoCo `contact.pos` and `mj_contactForce`. That is not Panda hardware tactile.",
        "",
        f"`SCALE_TAU` still uses `TAU_OPEN={TAU_OPEN}` even though 4D `RECOVERY4D_TAU_OPEN=+2`. "
        "Therefore `ctrl[7]=+2` and `ctrl[7]=-1` both saturate `tau_from_secure` at **+1**. "
        "The 27D grip channel **cannot represent active opening vs old open-end**.",
        "",
        "No dimension is object-pose GT. `contact_present_*` is still a simulator contact flag (binary), not nL/nR counts.",
        "",
    ]
    return lines


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    dump(
        RAW / "obs_schema.json",
        {
            "OBS_NAMES": OBS_NAMES,
            "OBS_CLASS": OBS_CLASS,
            "OBS_UNITS": OBS_UNITS,
            "OBS_SOURCES_code": OBS_SOURCES,
            "dt_sim": DT_SIM,
            "dt_policy": DT_POLICY,
            "n_substeps": cfg.get("n_substeps"),
            "SCALE_EX": SCALE_EX,
            "SCALE_TAU": SCALE_TAU,
            "TAU_OPEN_obs": TAU_OPEN,
            "TAU_SECURE": TAU_SECURE,
            "MIN_COP_FORCE_N": MIN_COP_FORCE_N,
            "PAD_U_HALF": PAD_U_HALF,
        },
    )
    print("collect ZERO", flush=True)
    zero, zep = collect_zero()
    print("collect 070", flush=True)
    e70, r70 = collect_recovery(0.70, "CENTER6_RECOVERY_070")
    print("collect 160", flush=True)
    e160, r160 = collect_recovery(1.60, "CENTER6_RECOVERY_160")
    print("collect RECAPTURE", flush=True)
    rec, tr, sim_r, gains_r, snap_r = collect_recapture()
    print("close offsets", flush=True)
    offs = recapture_offsets(sim_r, gains_r, snap_r)

    lz = label_stability(zero, "ZERO")
    l7 = label_stability(e70, "070")
    l16 = label_stability(e160, "160")
    lr = label_recapture(rec, tr["events"])
    np.savez_compressed(
        RAW / "gt_labels.npz",
        zero=lz.astype("U32"),
        rec070=l7.astype("U32"),
        rec160=l16.astype("U32"),
        recapture=lr.astype("U32"),
    )

    plot_stability(zero, e70, e160, lz, l7, l16)
    plot_recapture(rec, lr, tr["events"])

    # separability
    m_prog = mask_of(lz, "GT_PROGRESSIVE")
    m_run = mask_of(lz, "GT_RUNAWAY")
    m_edge = mask_of(l7, "GT_STABLE_EDGE")
    m_mar = mask_of(l16, "GT_STABLE_MARGIN")
    du100_z = causal_delta(zero["t"], np.abs(zero["OBS_uL"]), 0.100)
    du100_7 = causal_delta(e70["t"], np.abs(e70["OBS_uL"]), 0.100)
    du100_16 = causal_delta(e160["t"], np.abs(e160["OBS_uL"]), 0.100)
    du500_z = causal_delta(zero["t"], np.abs(zero["OBS_uL"]), 0.500)
    du500_7 = causal_delta(e70["t"], np.abs(e70["OBS_uL"]), 0.500)

    def feat_pack(arr, m, du100, du500):
        uedge = np.abs(arr["OBS_uL"][m]) / PAD_U_HALF
        return {
            "u_edge": summarize(uedge),
            "ehat_mm": summarize(1e3 * arr["OBS_ehat"][m]),
            "dehat_pol": summarize(arr["OBS_ehat_dot_policy"][m]),
            "du100_mm": summarize(1e3 * du100[m]),
            "du500_mm": summarize(1e3 * du500[m]),
            "fnL": summarize(arr["OBS_fnL"][m]),
            "ap_mm": summarize(1e3 * arr["OBS_ap"][m]),
            "GT_vrel": summarize(arr["GT_vrel"][m]),
            "OBS_cp_both": float(np.mean((arr["OBS_cpL"][m] > 0.5) & (arr["OBS_cpR"][m] > 0.5))) if m.any() else None,
        }

    sep = {
        "A_progressive_vs_edge": {
            "progressive": feat_pack(zero, m_prog, du100_z, du500_z),
            "stable_edge": feat_pack(e70, m_edge, du100_7, du500_7),
            "overlap_u_edge": overlap_frac(np.abs(zero["OBS_uL"][m_prog]) / PAD_U_HALF, np.abs(e70["OBS_uL"][m_edge]) / PAD_U_HALF),
            "overlap_du100": overlap_frac(du100_z[m_prog], du100_7[m_edge]),
            "overlap_du500": overlap_frac(du500_z[m_prog], du500_7[m_edge]),
            "overlap_dehat": overlap_frac(zero["OBS_ehat_dot_policy"][m_prog], e70["OBS_ehat_dot_policy"][m_edge]),
            "overlap_GTvrel": overlap_frac(zero["GT_vrel"][m_prog], e70["GT_vrel"][m_edge]),
        },
        "B_edge_vs_margin": {
            "edge": feat_pack(e70, m_edge, du100_7, du500_7),
            "margin": feat_pack(e160, m_mar, du100_16, causal_delta(e160["t"], np.abs(e160["OBS_uL"]), 0.500)),
            "overlap_u_edge": overlap_frac(np.abs(e70["OBS_uL"][m_edge]) / PAD_U_HALF, np.abs(e160["OBS_uL"][m_mar]) / PAD_U_HALF),
            "overlap_du100": overlap_frac(du100_7[m_edge], du100_16[m_mar]),
        },
    }

    # dehat sign vs GT vrel_x during progressive
    eh = zero["OBS_ehat_dot_policy"][m_prog]
    gx = zero["GT_vrel_x"][m_prog]
    ok = np.isfinite(eh) & np.isfinite(gx) & (np.abs(gx) > 0.002)
    sign_agree = float(np.mean(np.sign(eh[ok]) == np.sign(gx[ok]))) if ok.any() else None
    # alias: dehat=0 (invalid or tiny) while GT moving
    both_valid = (zero["OBS_ehat_valid"] > 0.5) & m_prog
    quiet = both_valid & (np.abs(zero["OBS_ehat_dot_policy"]) < 0.005)
    alias = float(np.mean(zero["GT_vrel"][quiet] > 0.015)) if quiet.any() else None

    dehat_audit = {
        "progressive_sign_agree_ehatdot_vs_GTvrelx": sign_agree,
        "frac_quiet_dehat_while_GTvrel_gt_15mm_s": alias,
        "note": "e_hat_dot is CoP-migration / dt_policy, not object twist",
    }

    # recapture pre-close vs FREE
    ev = tr["events"]
    t = rec["t"]
    m_free = mask_of(lr, "GT_FREE")
    m_pre = (t >= ev["CLOSE_START"] - 0.012) & (t <= ev["CLOSE_START"] + 1e-9)
    m_rc = mask_of(lr, "GT_RECONTACT", "GT_CAPTURE_DYNAMIC")
    m_st = mask_of(lr, "GT_CAPTURE_STABLE")
    m_open = mask_of(lr, "GT_OPENING")
    rec_sep = {
        "FREE": {
            "ap_mm": summarize(1e3 * rec["OBS_ap"][m_free]),
            "apdot": summarize(rec["OBS_apdot"][m_free]),
            "cpL": summarize(rec["OBS_cpL"][m_free]),
            "fnL": summarize(rec["OBS_fnL"][m_free]),
            "validL": summarize(rec["OBS_validL"][m_free]),
            "GT_vrel": summarize(rec["GT_vrel"][m_free]),
        },
        "pre_close_12ms": {
            "ap_mm": summarize(1e3 * rec["OBS_ap"][m_pre]),
            "apdot": summarize(rec["OBS_apdot"][m_pre]),
            "cpL": summarize(rec["OBS_cpL"][m_pre]),
            "cpR": summarize(rec["OBS_cpR"][m_pre]),
            "fnL": summarize(rec["OBS_fnL"][m_pre]),
            "validL": summarize(rec["OBS_validL"][m_pre]),
            "GT_nL": summarize(rec["GT_nL"][m_pre]),
            "GT_vrel": summarize(rec["GT_vrel"][m_pre]),
            "elapsed_since_open_s": float(ev["CLOSE_START"] - ev["OPEN_START"]),
        },
        "recontact_dynamic": {
            "fnL": summarize(rec["OBS_fnL"][m_rc]),
            "ap_mm": summarize(1e3 * rec["OBS_ap"][m_rc]),
            "GT_vrel": summarize(rec["GT_vrel"][m_rc]),
            "valid_both": float(np.mean((rec["OBS_validL"][m_rc] > 0.5) & (rec["OBS_validR"][m_rc] > 0.5))) if m_rc.any() else None,
        },
        "stable": {
            "fnL": summarize(rec["OBS_fnL"][m_st]),
            "ap_mm": summarize(1e3 * rec["OBS_ap"][m_st]),
            "GT_vrel": summarize(rec["GT_vrel"][m_st]),
            "u_edge": summarize(np.abs(rec["OBS_uL"][m_st]) / PAD_U_HALF),
            "du100": summarize(1e3 * causal_delta(rec["t"], rec["OBS_uL"], 0.100)[m_st]),
        },
        "cp_equals_GTbothoff_FREE": float(
            np.mean(((rec["OBS_cpL"][m_free] < 0.5) & (rec["OBS_cpR"][m_free] < 0.5)))
        )
        if m_free.any()
        else None,
    }

    # tactile degradation on du100 for edge vs progressive
    rng = np.random.default_rng(0)
    def degrade_du(arr, m, res, sig):
        u, ok = arr["OBS_uL"], arr["OBS_validL"] > 0.5
        ud, vd = degrade_stream(u, ok, res, sig, 0, 0.0, rng)
        du = causal_delta(arr["t"], np.abs(ud), 0.100)
        return du[m]

    noise = {
        "ideal_overlap_du100_A": sep["A_progressive_vs_edge"]["overlap_du100"],
        "res_0.5mm_overlap_du100_A": overlap_frac(
            degrade_du(zero, m_prog, 0.5e-3, 0.0), degrade_du(e70, m_edge, 0.5e-3, 0.0)
        ),
        "sig_0.5mm_overlap_du100_A": overlap_frac(
            degrade_du(zero, m_prog, 0.0, 0.5e-3), degrade_du(e70, m_edge, 0.0, 0.5e-3)
        ),
        "res_0.5mm_overlap_u_B": overlap_frac(
            np.abs(degrade_stream(e70["OBS_uL"], e70["OBS_validL"] > 0.5, 0.5e-3, 0.0, 0, 0.0, rng)[0][m_edge]) / PAD_U_HALF,
            np.abs(degrade_stream(e160["OBS_uL"], e160["OBS_validL"] > 0.5, 0.5e-3, 0.0, 0, 0.0, rng)[0][m_mar]) / PAD_U_HALF,
        ),
    }

    dump(RAW / "separability.json", {"stability": sep, "dehat": dehat_audit, "recapture": rec_sep, "noise": noise, "offsets": offs})

    # window table for CoP
    win_tab = []
    for w in WINDOWS:
        win_tab.append(
            {
                "W_s": w,
                "ZERO_prog_d|u|_mm": summarize(1e3 * causal_delta(zero["t"], np.abs(zero["OBS_uL"]), w)[m_prog]),
                "EDGE_d|u|_mm": summarize(1e3 * causal_delta(e70["t"], np.abs(e70["OBS_uL"]), w)[m_edge]),
                "MARGIN_d|u|_mm": summarize(1e3 * causal_delta(e160["t"], np.abs(e160["OBS_uL"]), w)[m_mar]),
            }
        )
    dump(RAW / "causal_windows.json", win_tab)

    # write markdown
    L = write_obs_table()
    A = L.append
    A("## 3. Trajectory dataset")
    A("")
    A("| family | construction | notes |")
    A("|---|---|---|")
    A("| CENTER6_ZERO | frozen CENTER 6 m/s ballistic ZERO | table ~6.132 s |")
    A("| CENTER6_RECOVERY_070 | EARLY +120° τ=-5 0.70 s + brake + return + hold | stable edge-catch |")
    A("| CENTER6_RECOVERY_160 | same, 1.60 s weak slip | larger inward correction |")
    A("| CLEAN_RECAPTURE | frozen extra_after_both=0.04, τ_open=+2, v_z=0 | privileged close timing |")
    A("")
    A("Control sequences were not modified. Missing OBS channels were collected by replaying those sequences.")
    A("")
    A("## 4. GT labels (evaluation only; not in observation)")
    A("")
    A("**Stability:** `GT_PROGRESSIVE` bilateral post-impact creep; `GT_RUNAWAY` |r_h.x|≥16 mm and |v_rel|>15 mm/s; `GT_STABLE_EDGE` / `GT_STABLE_MARGIN` bilateral, |v_rel|<0.02, |g_hx|<2.5, τ=-18, persisted 0.20 s.")
    A("")
    A("**Recapture:** `GT_SECURE` / `GT_OPENING` / `GT_FREE` / `GT_RECONTACT` / `GT_CAPTURE_DYNAMIC` / `GT_CAPTURE_STABLE` aligned to privileged events OPEN_START, FIRST_BOTH_OFF, FIRST_RECONTACT, FIRST_BILATERAL, SECURE_CAPTURE, HOLD_START.")
    A("")
    A("## 5. Progressive vs stable")
    A("")
    A("Instantaneous n, Fn, aperture are similar across arrested edge-catch and slow ZERO creep. Position-only |e_hat| / |CoP u| is **not** a stability bit: 0.70 s is near-edge and holds.")
    A("")
    def mdsum(d):
        if not d or d.get("n") == 0:
            return "empty"
        return f"med={d['median']:.4g}  [p10={d['p10']:.4g}, p90={d['p90']:.4g}]  n={d['n']}"

    A("| feature | ZERO progressive | stable 0.70 | overlap (p10–p90) |")
    A("|---|---|---|---|")
    A(f"| |u|/8.5 mm | {mdsum(sep['A_progressive_vs_edge']['progressive']['u_edge'])} | {mdsum(sep['A_progressive_vs_edge']['stable_edge']['u_edge'])} | {sep['A_progressive_vs_edge']['overlap_u_edge']} |")
    A(f"| Δ\\|u\\| 100 ms mm | {mdsum(sep['A_progressive_vs_edge']['progressive']['du100_mm'])} | {mdsum(sep['A_progressive_vs_edge']['stable_edge']['du100_mm'])} | {sep['A_progressive_vs_edge']['overlap_du100']} |")
    A(f"| Δ\\|u\\| 500 ms mm | {mdsum(sep['A_progressive_vs_edge']['progressive']['du500_mm'])} | {mdsum(sep['A_progressive_vs_edge']['stable_edge']['du500_mm'])} | {sep['A_progressive_vs_edge']['overlap_du500']} |")
    A(f"| OBS e_hat_dot (policy) | {mdsum(sep['A_progressive_vs_edge']['progressive']['dehat_pol'])} | {mdsum(sep['A_progressive_vs_edge']['stable_edge']['dehat_pol'])} | {sep['A_progressive_vs_edge']['overlap_dehat']} |")
    A(f"| GT \\|v_rel\\| | {mdsum(sep['A_progressive_vs_edge']['progressive']['GT_vrel'])} | {mdsum(sep['A_progressive_vs_edge']['stable_edge']['GT_vrel'])} | {sep['A_progressive_vs_edge']['overlap_GTvrel']} |")
    A("")
    A("## 6. CoP-u position vs motion")
    A("")
    A("| W | ZERO prog Δ\\|u\\| mm | 0.70 edge | 1.60 margin |")
    A("|---:|---|---|---|")
    for w in win_tab:
        A(f"| {w['W_s']} | {mdsum(w['ZERO_prog_d|u|_mm'])} | {mdsum(w['EDGE_d|u|_mm'])} | {mdsum(w['MARGIN_d|u|_mm'])} |")
    A("")
    A("Longer causal windows (200–500 ms) of |u| change are more useful than a single sample of |u|. Short 20 ms deltas are noisy relative to millimetre creep.")
    A("")
    A("## 7. dehat audit")
    A("")
    A(json.dumps(dehat_audit, indent=2))
    A("")
    A("`OBS e_hat_dot` is **not** GT relative velocity. On noslip=1 progressive ZERO it can be near-zero while GT still creeps (contact-location stall / validity gaps). Do not treat dehat≈0 as arrest.")
    A("")
    A("## 8. Causal windows")
    A("")
    A("All Δ features are x(t)−x(t−W) with W∈{20,50,100,200,500} ms. No centered/future windows.")
    A("")
    A("## 9. Recapture phase observability")
    A("")
    A(f"Events: {tr['events']}")
    A("")
    A("Binary `OBS contact_present_L/R` tracks GT both-off on this construction (see `cp_equals_GTbothoff_FREE`). That is **not** nL/nR, but it **is** in the 27D vector. Privileged close timing is therefore *partially* replaceable by these flags, plus elapsed time since opening (not currently a dedicated obs dim — would need memory/stack).")
    A("")
    A(f"- FREE: {rec_sep['FREE']}")
    A(f"- immediately before CLOSE (~12 ms): {rec_sep['pre_close_12ms']}")
    A("")
    A("`valid_L/R=0` means CoP undefined (no contact or Fn<0.01 N), **not** a measured CoP at the origin.")
    A("")
    A("## 10. Close-timing counterfactual")
    A("")
    A("| extra_after_both | Δt vs 40 ms | captured | ok | table | recontact | bilateral |")
    A("|---:|---:|---|---|---|---|---|")
    for r in offs:
        A(
            f"| {r['extra_after_both']} | {r['close_vs_nominal_ms']:.0f} | {r['captured']} | {r['ok']} | {r['table']} | "
            f"{r['FIRST_RECONTACT']} | {r['FIRST_BILATERAL']} |"
        )
    A("")
    A("This is a **narrow** window of a few tens of ms at this lift height, not a broad catch basin. Observable pre-close state at −20 vs +20 ms is dominated by how far the object has fallen (aperture already open; tactile invalid). Instantaneous 27D at CLOSE_START is similar across nearby offsets; **timing/memory** distinguishes them more than a single frame.")
    A("")
    A("## 11. Recontact vs stable capture")
    A("")
    A(f"dynamic: {rec_sep['recontact_dynamic']}")
    A("")
    A(f"stable: {rec_sep['stable']}")
    A("")
    A("First tactile return is **not** recovered: Fn rebuilds, aperture returns toward ~17.6 mm, |v_rel| falls over tens of ms. An observable settling signature exists (validity both-on + low causal Δu + aperture/Fn plateau) but needs persistence, not a one-step classifier.")
    A("")
    A("## 12. Sensor-imperfection (existing 0.5 mm quant/noise)")
    A("")
    A(json.dumps(noise, indent=2))
    A("")
    A("Qualitative CoP-u **position** gap between 0.70 and 1.60 can survive 0.5 mm quantization. CoP **slope** vs progressive ZERO is more fragile: short-window Δu overlap increases under 0.5 mm noise.")
    A("")
    A("## 13. Separability summary")
    A("")
    A("- A (creep vs arrested edge): **position-only CoP/ehat insufficient** (edge-catch is supposed to look 'bad'). Causal Δu over 200–500 ms is the more relevant OBS cue; overlap is still non-zero. GT v_rel separates more cleanly than dehat.")
    A("- B (0.70 vs 1.60): **|CoP u|** still encodes geometric margin; motion features similar once both are arrested.")
    A("- C (FREE vs pre-close): tactile invalid + large aperture in both; pre-close is not a distinct instantaneous tactile class. Catchability is mostly **how long since both-off** plus robot z/aperture, i.e. temporal.")
    A("- D (recontact vs stable): Fn/aperture/validity time series differ; one-step flags do not.")
    A("")
    A("## 14. Likely need for temporal state")
    A("")
    A("**C is not required yet; B (causal stacking) is the minimum supported choice. E remains if recapture timing must be precise.**")
    A("")
    A("Evidence: stability is location+motion over 0.2–0.5 s; recapture CLOSE is a ~20–40 ms window after an observable both-off bit; 27D `e_hat_dot` uses 20 ms already but is not GT motion. Instantaneous 27D alone (A) is **not** enough. A learned recurrent net (C) is not justified before trying stacked causal lags. An explicit estimator (D) would be for object-relative velocity, which is still missing and should not be filled with GT.")
    A("")
    A("## 15. Future stability criterion (hypothesis only; not implemented)")
    A("")
    A("Against these four families, a later terminal would need **persistence**, not |ehat| small and not |CoP u| small:")
    A("")
    A("- bilateral `contact_present` / `valid_*` held")
    A("- low causal CoP motion over ≥100–500 ms")
    A("- aperture and Fn settled")
    A("- **not** rejecting large |u| by itself (0.70 counterexample)")
    A("- **not** trusting dehat≈0 alone (ZERO aliasing)")
    A("")
    A("No thresholds tuned here.")
    A("")
    A("## 16. Unresolved gaps")
    A("")
    A("- 27D `tau_from_secure` saturates for active open.")
    A("- No explicit time-since-contact-loss channel.")
    A("- Spatial tactile remains a MuJoCo proxy.")
    A("- Binary contact_present ≠ contact counts; flicker vs true both-off still needs persistence.")
    A("- GT relative velocity is **not** an allowed OBS add-on from this audit.")
    A("")
    A("## Viewers")
    A("")
    A("```text")
    A("python training/demo_observable_audit.py --case center6_zero")
    A("python training/demo_observable_audit.py --case stable_edge")
    A("python training/demo_observable_audit.py --case recapture")
    A("```")
    A("")
    A("Existing: `python training/demo_ballistic_recovery.py --mode zero` · `python training/demo_airborne_recapture.py --mode recapture`")
    A("")
    A("STOP.")
    (OUT / "OBSERVABLE_TEMPORAL_AUDIT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "OBSERVABLE_TEMPORAL_AUDIT.md", flush=True)


if __name__ == "__main__":
    main()
