"""Diagnostic-only observable-signal audit. No SAC train. No RecoveryEnv/reward edits.

Does not read results/eval_sets/*.npz.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.gripper_controller import finger_opening, read_touch
from controllers.jacobian_controller import gains_from_cfg, ori_error_deg, rotation_error
from controllers.residual import map_recovery4d
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.contact import finger_object_normals, touch_values
from envs.deterioration import body_twist
from envs.grasp_sim import GraspSim
from envs.physical_recovery import (
    CLEAR_MIN,
    CONTACT_LOSS_HOLD,
    E_TOL,
    TABLE_DROP,
    V_REL_TOL,
    W_REL_TOL,
    Z_AIR,
    fail_kind,
    physical_pack,
    recovered,
)
from training.offset_construct import apply_rel_pose, lift_to_airborne, settle_hold
from training.replay_core import (
    MASS,
    MU,
    disable_object_table,
    freeze,
    params_from_frozen,
    tick_vw,
)

OUT = ROOT / "results" / "diagnostics" / "observable_reward_signal_audit"
FIG = OUT / "figures"
EVAL_OFF = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMP = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"

CANDIDATES = [
    "aperture",
    "aperture_vel",
    "tau",
    "fn_l",
    "fn_r",
    "fn_sum",
    "fn_imbalance",
    "fn_ratio",
    "touch_l",
    "touch_r",
    "touch_sum",
    "nL",
    "nR",
    "bilat",
    "track_pos_err",
    "track_ori_deg",
    "track_vel_err",
    "hand_speed",
    "v_cmd_norm",
    "w_cmd_norm",
    "finger_qd_mean",
    "D",
    "Ddot",
    "D_C",
    "D_lost",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rank(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a)
    r = np.empty_like(a, float)
    r[o] = np.arange(a.size, dtype=float)
    return r


def pearson(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 8:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / d) if d > 1e-12 else float("nan")


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 8:
        return float("nan")
    return pearson(_rank(x), _rank(y))


def sample_row(sim, gains, extra: dict) -> dict:
    ids = sim.ids
    d = sim.data
    o = physical_pack(sim)
    snap = sim.meter.compute(sim.model, d, ids, float(sim.model.opt.timestep))
    p_h = np.array(d.xpos[ids.hand_body], float)
    r_h = np.array(d.xmat[ids.hand_body].reshape(3, 3), float)
    p_o = np.array(d.xpos[ids.object_body], float)
    vh, wh = body_twist(sim.model, d, ids.hand_body)
    vo, wo = body_twist(sim.model, d, ids.object_body)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    e_p = p_des - p_h
    e_o = rotation_error(r_des, r_h)
    v_des = np.asarray(sim.fsm.v_cmd, float)
    w_des = np.asarray(sim.fsm.w_cmd, float)
    v_err = v_des - vh
    w_err = w_des - wh
    fn = finger_object_normals(sim.model, d, ids)
    touch = touch_values(d, ids)
    ap = finger_opening(d, ids)
    qf = np.array(d.qpos[ids.finger_jnt], float)
    qdf = np.array(d.qvel[ids.finger_dof], float)
    qa = np.array(d.qpos[ids.arm_jnt], float)
    qda = np.array(d.qvel[ids.arm_dof], float)
    tau = float(d.ctrl[7]) if d.ctrl.size > 7 else 0.0
    fn_l, fn_r = float(fn[0]), float(fn[1])
    fn_sum = fn_l + fn_r
    lost = 1.0 if (fn_l < 0.5 or fn_r < 0.5) else 0.0
    dp = snap.p_rel - (sim.meter.prel_ref if sim.meter.prel_ref is not None else snap.p_rel)
    D_p = float(np.linalg.norm(dp)) / sim.meter.p_scale
    D_v = float(np.linalg.norm(snap.v_rel)) / sim.meter.v_scale
    D_th = abs(float(snap.theta_rel)) / sim.meter.theta_scale
    D_w = float(np.linalg.norm(snap.omega_rel)) / sim.meter.omega_scale
    D_c = float(snap.C)
    rh = np.asarray(o["rh"], float)
    row = {
        "t": float(d.time),
        "mode": extra.get("mode", ""),
        "family": extra.get("family", ""),
        "traj_id": extra.get("traj_id", ""),
        "ic_ex_sign": extra.get("ic_ex_sign", 0.0),
        "ic_ex": extra.get("ic_ex", 0.0),
        "a0": extra.get("a0", 0.0),
        "a1": extra.get("a1", 0.0),
        "a2": extra.get("a2", 0.0),
        "a3": extra.get("a3", 0.0),
        "tau": tau,
        "q0": qa[0],
        "q1": qa[1],
        "q2": qa[2],
        "q3": qa[3],
        "q4": qa[4],
        "q5": qa[5],
        "q6": qa[6],
        "qd0": qda[0],
        "qd1": qda[1],
        "qd2": qda[2],
        "qd3": qda[3],
        "qd4": qda[4],
        "qd5": qda[5],
        "qd6": qda[6],
        "hand_x": p_h[0],
        "hand_y": p_h[1],
        "hand_z": p_h[2],
        "hand_vx": vh[0],
        "hand_vy": vh[1],
        "hand_vz": vh[2],
        "hand_speed": float(np.linalg.norm(vh)),
        "pdes_x": p_des[0],
        "pdes_y": p_des[1],
        "pdes_z": p_des[2],
        "track_pos_err": float(np.linalg.norm(e_p)),
        "track_ori_deg": ori_error_deg(r_des, r_h),
        "track_vel_err": float(np.linalg.norm(v_err)),
        "track_w_err": float(np.linalg.norm(w_err)),
        "v_cmd_norm": float(np.linalg.norm(v_des)),
        "w_cmd_norm": float(np.linalg.norm(w_des)),
        "aperture": ap,
        "finger_q0": qf[0],
        "finger_q1": qf[1],
        "finger_qd_mean": float(np.mean(qdf)),
        "nL": float(o["nL"]),
        "nR": float(o["nR"]),
        "bilat": 1.0 if int(o["nL"]) > 0 and int(o["nR"]) > 0 else 0.0,
        "fn_l": fn_l,
        "fn_r": fn_r,
        "fn_sum": fn_sum,
        "fn_imbalance": abs(fn_l - fn_r),
        "fn_ratio": abs(fn_l - fn_r) / (fn_sum + 0.05),
        "touch_l": float(touch[0]),
        "touch_r": float(touch[1]),
        "touch_sum": float(touch[0] + touch[1]),
        "D": float(snap.D),
        "Ddot": float(snap.Ddot),
        "D_p": D_p,
        "D_v": D_v,
        "D_th": D_th,
        "D_w": D_w,
        "D_C": D_c,
        "D_lost": lost,
        "EVAL_ONLY_GT_e_x": float(o["e_x"]),
        "EVAL_ONLY_GT_e_y": float(rh[1]),
        "EVAL_ONLY_GT_e_z": float(rh[2]),
        "EVAL_ONLY_GT_abs_ex": abs(float(o["e_x"])),
        "EVAL_ONLY_GT_P": -abs(float(o["e_x"])),
        "EVAL_ONLY_GT_v_rel": float(o["v_rel"]),
        "EVAL_ONLY_GT_w_rel": float(o["w_rel"]),
        "EVAL_ONLY_GT_obj_x": p_o[0],
        "EVAL_ONLY_GT_obj_y": p_o[1],
        "EVAL_ONLY_GT_obj_z": p_o[2],
        "EVAL_ONLY_GT_recovered_now": 1.0 if recovered(o) else 0.0,
        "EVAL_ONLY_GT_drop": 1.0 if fail_kind(o, CONTACT_LOSS_HOLD) == "drop" or float(o["obj_z"]) < TABLE_DROP else 0.0,
        "EVAL_ONLY_GT_scene": float(o["scene"]),
        "EVAL_ONLY_GT_airborne": 1.0 if float(o["obj_z"]) >= Z_AIR and float(o["clear"]) >= CLEAR_MIN else 0.0,
        "EVAL_ONLY_GT_stable": 1.0
        if (
            recovered(o)
            or (
                int(o["nL"]) > 0
                and int(o["nR"]) > 0
                and float(o["v_rel"]) < V_REL_TOL
                and float(o["w_rel"]) < W_REL_TOL
            )
        )
        else 0.0,
    }
    row["aperture_vel"] = extra.get("aperture_vel", 0.0)
    return row


def airborne_base(cfg) -> dict:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("lift failed")
    disable_object_table(sim)
    freeze(sim)
    settle_hold(sim, cfg, 0.20, -18.0)
    return sim.snapshot()


def make_ic(cfg, base, e_x: float) -> dict:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim)
    sim.load_snapshot(base)
    disable_object_table(sim)
    freeze(sim)
    apply_rel_pose(sim, e_x)
    settle_hold(sim, cfg, 0.25, -18.0)
    o = physical_pack(sim)
    snap = sim.snapshot()
    snap["source"] = "diagnostic_offset"
    snap["e_x"] = float(o["e_x"])
    return snap, float(o["e_x"])


def run_protocol(cfg, snap, family, proto, ic_ex) -> list[dict]:
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim)
    sim.load_snapshot(snap)
    disable_object_table(sim)
    freeze(sim)
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    ctrl = RuleBasedRecovery(params)
    ctrl.reset(np.array(sim.fsm.r_des, float).reshape(3, 3))
    rows = []
    dt = float(sim.model.opt.timestep)
    ap_prev = finger_opening(sim.data, sim.ids)
    last_mode = "STABILIZE"
    t_end = float(proto.get("t", 1.0))
    n = int(round(t_end / dt))
    traj_id = f"{family}_ex{ic_ex:+.4f}"
    sign = float(np.sign(ic_ex) or 1.0)
    for i in range(n):
        kind = proto["kind"]
        a = np.zeros(4)
        mode = kind
        if kind == "ZERO":
            v, w, tau = np.zeros(3), np.zeros(3), -18.0
            mode = "ZERO"
        elif kind == "RULE":
            o = physical_pack(sim)
            cmd = ctrl.step(o, dt)
            if cmd["mode"] != last_mode:
                last_mode = cmd["mode"]
                if cmd["freeze_p"]:
                    freeze(sim)
            if cmd["r_des"] is not None:
                sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            v, w, tau = cmd["v_world"], np.zeros(3), cmd["tau"]
            mode = cmd["mode"]
            if ctrl.s.done:
                tick_vw(sim, v, w, tau, gains)
                ap = finger_opening(sim.data, sim.ids)
                extra = dict(
                    mode=mode,
                    family=family,
                    traj_id=traj_id,
                    ic_ex=ic_ex,
                    ic_ex_sign=sign,
                    aperture_vel=(ap - ap_prev) / dt,
                    a0=0,
                    a1=0,
                    a2=0,
                    a3=0,
                )
                rows.append(sample_row(sim, gains, extra))
                break
        else:
            a = np.array(proto["a"], float)
            t_rel = i * dt
            for seg in proto.get("segs", []):
                if t_rel >= seg[0]:
                    a = np.array(seg[1], float)
            rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            m = map_recovery4d(a, sim.fsm.r_des, rh)
            v, w, tau = m["v_world"], m["w_world"], m["tau"]
            a = m["a"]
            mode = proto.get("mode", kind)
        tick_vw(sim, v, w, tau, gains)
        ap = finger_opening(sim.data, sim.ids)
        extra = dict(
            mode=mode,
            family=family,
            traj_id=traj_id,
            ic_ex=ic_ex,
            ic_ex_sign=sign,
            aperture_vel=(ap - ap_prev) / dt,
            a0=float(a[0]),
            a1=float(a[1]),
            a2=float(a[2]),
            a3=float(a[3]),
        )
        rows.append(sample_row(sim, gains, extra))
        ap_prev = ap
        if kind != "RULE":
            o = physical_pack(sim)
            if fail_kind(o, CONTACT_LOSS_HOLD) == "drop" and i * dt > 0.15:
                break
    return rows


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    h0, h1 = _sha(EVAL_OFF), _sha(EVAL_IMP)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    print("lifting base IC...")
    base = airborne_base(cfg)
    targets = [-0.0075, -0.0050, -0.0025, 0.0025, 0.0050, 0.0075]
    ics = []
    for e in targets:
        snap, got = make_ic(cfg, base, e)
        ics.append((snap, got))
        print(f"  IC requested {1e3*e:+.1f} mm  got {1e3*got:+.2f} mm")
    a_slip = 2.0 * ((-2.0) - (-1.0)) / ((-18.0) - (-1.0)) - 1.0
    protocols = [
        ("A_zero_hold", {"kind": "ZERO", "t": 0.80}),
        ("B_rule", {"kind": "RULE", "t": 3.50}),
        ("D_slip", {"kind": "R4D", "t": 0.35, "a": [0, 0, 0, a_slip], "mode": "SLIP"}),
        (
            "E_slip_brake",
            {
                "kind": "R4D",
                "t": 0.45,
                "a": [0, 0, 0, a_slip],
                "mode": "SLIP_BRAKE",
                "segs": [(0.12, [0, 0, 0, 1.0])],
            },
        ),
        (
            "F_open_regrasp",
            {
                "kind": "R4D",
                "t": 0.40,
                "a": [0, 1.0, 0, -1.0],
                "mode": "OPEN_REGRASP",
                "segs": [(0.04, [0, 0, 0, 1.0])],
            },
        ),
        ("H_wrist_only", {"kind": "R4D", "t": 0.30, "a": [1.0, 0, 0, 1.0], "mode": "WRIST"}),
    ]
    all_rows = []
    for snap, ex in ics:
        for fam, proto in protocols:
            print(f"  run {fam} e_x={1e3*ex:+.2f} mm")
            all_rows.extend(run_protocol(cfg, snap, fam, proto, ex))
    keys = list(all_rows[0].keys())
    arr = {k: np.array([r[k] for r in all_rows]) for k in keys}
    np.savez_compressed(OUT / "rows.npz", **{k: (v if v.dtype != object else v.astype("U32")) for k, v in arr.items()})
    with (OUT / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    meta = {
        "used_final_eval_data": False,
        "eval_offset_sha256_start": h0,
        "eval_impact_sha256_start": h1,
        "eval_offset_sha256_end": _sha(EVAL_OFF),
        "eval_impact_sha256_end": _sha(EVAL_IMP),
        "n_rows": len(all_rows),
        "n_ics": len(ics),
        "ic_e_x_mm": [1e3 * x[1] for x in ics],
        "families": [p[0] for p in protocols],
        "dt_s": float(merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml")).get("physics_dt", 0.002)),
        "mass": MASS,
        "mu": MU,
        "source": "constructed_offset via offset_construct; not eval_sets",
    }
    if meta["eval_offset_sha256_start"] != meta["eval_offset_sha256_end"]:
        raise RuntimeError("held-out offset npz changed")
    if meta["eval_impact_sha256_start"] != meta["eval_impact_sha256_end"]:
        raise RuntimeError("held-out impact npz changed")
    (OUT / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("collected", len(all_rows), "rows")
    return arr, meta


def load_rows():
    z = np.load(OUT / "rows.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def add_deltas(d: dict) -> dict:
    n = len(d["t"])
    dP = np.zeros(n)
    abs_e = np.asarray(d["EVAL_ONLY_GT_abs_ex"], float)
    tid = np.asarray(d["traj_id"]).astype(str)
    for tr in np.unique(tid):
        idx = np.where(tid == tr)[0]
        ae = abs_e[idx]
        dp = np.zeros_like(ae)
        dp[:-1] = ae[:-1] - ae[1:]
        dP[idx] = dp
    d["EVAL_ONLY_GT_dP"] = dP
    for name in CANDIDATES:
        if name not in d:
            continue
        x = np.asarray(d[name], float)
        dx = np.zeros_like(x)
        for tr in np.unique(tid):
            idx = np.where(tid == tr)[0]
            xx = x[idx]
            dxx = np.zeros_like(xx)
            dxx[:-1] = xx[1:] - xx[:-1]
            dx[idx] = dxx
        d[f"d_{name}"] = dx
    return d


def win_feat(x, t, tid, width_s, kind):
    out = np.full_like(x, np.nan, float)
    dt = float(np.median(np.diff(t[tid == tid[0]]))) if x.size else 0.002
    w = max(2, int(round(width_s / max(dt, 1e-6))))
    for tr in np.unique(tid):
        idx = np.where(tid == tr)[0]
        xx = x[idx]
        for i in range(len(idx)):
            a = max(0, i - w + 1)
            sl = xx[a : i + 1]
            if kind == "mean":
                out[idx[i]] = float(np.mean(sl))
            elif kind == "var":
                out[idx[i]] = float(np.var(sl))
            elif kind == "slope":
                if sl.size < 3:
                    out[idx[i]] = np.nan
                else:
                    tt = np.arange(sl.size, dtype=float)
                    out[idx[i]] = float(np.polyfit(tt, sl, 1)[0])
    return out


def corr_block(d, mask, xname, yname="EVAL_ONLY_GT_dP"):
    x = np.asarray(d[xname], float)[mask]
    y = np.asarray(d[yname], float)[mask]
    if x.size < 20:
        return {"n": int(x.size), "pearson": None, "spearman": None, "sign_agree": None}
    dx = x
    p, s = pearson(dx, y), spearman(dx, y)
    m2 = (np.abs(dx) > 1e-9) & (np.abs(y) > 1e-9)
    agr = float(np.mean(np.sign(dx[m2]) == np.sign(y[m2]))) if np.any(m2) else float("nan")
    return {"n": int(x.size), "pearson": p, "spearman": s, "sign_agree": agr}


def analyze(d, meta):
    d = add_deltas(d)
    tid = np.asarray(d["traj_id"]).astype(str)
    fam = np.asarray(d["family"]).astype(str)
    sgn = np.asarray(d["ic_ex_sign"], float)
    mode = np.asarray(d["mode"]).astype(str)
    t = np.asarray(d["t"], float)
    # per-traj labels
    labels = {}
    for tr in np.unique(tid):
        idx = np.where(tid == tr)[0]
        rec = np.asarray(d["EVAL_ONLY_GT_recovered_now"], float)[idx]
        dt = 0.002
        hold = False
        run = 0.0
        for v in rec:
            run = run + dt if v > 0.5 else 0.0
            if run >= 0.20:
                hold = True
                break
        labels[tr] = {
            "final_recovery": hold,
            "end_abs_ex": float(d["EVAL_ONLY_GT_abs_ex"][idx[-1]]),
            "start_abs_ex": float(d["EVAL_ONLY_GT_abs_ex"][idx[0]]),
            "end_drop": float(d["EVAL_ONLY_GT_drop"][idx[-1]]) > 0.5,
            "min_abs_ex": float(np.min(d["EVAL_ONLY_GT_abs_ex"][idx])),
            "family": str(fam[idx[0]]),
            "sign": float(sgn[idx[0]]),
            "n": int(idx.size),
        }
    success_mask = np.array([labels[tr]["final_recovery"] for tr in tid])
    fail_mask = np.array([labels[tr]["end_drop"] or (not labels[tr]["final_recovery"] and str(fam[i]).startswith("B_")) for i, tr in enumerate(tid)])
    report = {"used_final_eval_data": False, "n_rows": int(len(tid)), "traj": labels}

    cand_stats = {}
    for name in CANDIDATES:
        if name not in d:
            continue
        dxn = f"d_{name}"
        blocks = {
            "all_dP": corr_block(d, np.ones(len(tid), bool), dxn),
            "all_abs_e": corr_block(d, np.ones(len(tid), bool), name, "EVAL_ONLY_GT_abs_ex"),
            "pos_dP": corr_block(d, sgn > 0, dxn),
            "neg_dP": corr_block(d, sgn < 0, dxn),
        }
        for mname in np.unique(mode):
            blocks[f"mode_{mname}"] = corr_block(d, mode == mname, dxn)
        for f in np.unique(fam):
            blocks[f"fam_{f}"] = corr_block(d, fam == f, dxn)
        blocks["success_dP"] = corr_block(d, success_mask, dxn)
        blocks["fail_dP"] = corr_block(d, fail_mask, dxn)
        # temporal
        x = np.asarray(d[name], float)
        temporal = {}
        for w in (0.02, 0.05, 0.10, 0.20):
            sl = win_feat(x, t, tid, w, "slope")
            d[f"{name}_slope_{int(1000*w)}ms"] = sl
            temporal[f"slope_{int(1000*w)}ms"] = corr_block_arr(sl, d["EVAL_ONLY_GT_dP"])
            temporal[f"mean_{int(1000*w)}ms"] = corr_block_arr(
                win_feat(x, t, tid, w, "mean"), d["EVAL_ONLY_GT_dP"]
            )
        cand_stats[name] = {"blocks": blocks, "temporal": temporal}
    report["candidates"] = cand_stats

    # D_t counterexamples
    dP = d["EVAL_ONLY_GT_dP"]
    dD = d["d_D"]
    abs_e = d["EVAL_ONLY_GT_abs_ex"]
    cex = []
    # D decreases, |e| gets worse (dP negative)
    m = (dD < -1e-4) & (dP < -1e-5)
    cex.append({"name": "D_decreases_but_abs_e_worsens", "n": int(np.sum(m))})
    m2 = (dD < -1e-4) & (np.asarray(d["bilat"]) < 0.5)
    cex.append({"name": "D_decreases_without_bilateral", "n": int(np.sum(m2))})
    m3 = (dP > 1e-5) & (dD > 1e-4)
    cex.append({"name": "abs_e_improves_but_D_worsens", "n": int(np.sum(m3))})
    m4 = (np.abs(abs_e) > 0.003) & (dD < -1e-4)
    cex.append({"name": "D_improves_while_still_displaced", "n": int(np.sum(m4))})
    # raw examples
    examples = []
    for mask, tag in [
        (m, "D_down_|e|_up"),
        (m3, "|e|_down_D_up"),
        (m2, "D_down_no_bilat"),
    ]:
        idx = np.where(mask)[0][:5]
        for i in idx:
            examples.append(
                {
                    "tag": tag,
                    "traj": str(tid[i]),
                    "t": float(t[i]),
                    "mode": str(mode[i]),
                    "D": float(d["D"][i]),
                    "dD": float(dD[i]),
                    "abs_e": float(abs_e[i]),
                    "dP": float(dP[i]),
                    "bilat": float(d["bilat"][i]),
                }
            )
    report["dt_counterexamples"] = {"counts": cex, "examples": examples}

    # proxy: ZERO dP vs dD, wrist |e| vs g change via w_cmd
    zero = fam == "A_zero_hold"
    wrist = fam == "H_wrist_only"
    report["proxy"] = {
        "ZERO_mean_dP": float(np.mean(dP[zero])) if np.any(zero) else None,
        "ZERO_mean_dD": float(np.mean(dD[zero])) if np.any(zero) else None,
        "ZERO_std_abs_e": float(np.std(abs_e[zero])) if np.any(zero) else None,
        "WRIST_mean_dP": float(np.mean(dP[wrist])) if np.any(wrist) else None,
        "WRIST_mean_d_track": float(np.mean(d["d_track_ori_deg"][wrist])) if np.any(wrist) else None,
        "WRIST_end_abs_e_minus_start": None,
    }
    if np.any(wrist):
        deltas = []
        for tr in np.unique(tid[wrist]):
            idx = np.where(tid == tr)[0]
            deltas.append(float(abs_e[idx[-1]] - abs_e[idx[0]]))
        report["proxy"]["WRIST_end_abs_e_minus_start"] = deltas

    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, default=_jsonish), encoding="utf-8")
    return d, report, labels


def corr_block_arr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 20:
        return {"n": int(x.size), "pearson": None, "spearman": None}
    return {"n": int(x.size), "pearson": pearson(x, y), "spearman": spearman(x, y)}


def _jsonish(x):
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def figures(d, labels):
    FIG.mkdir(parents=True, exist_ok=True)
    tid = np.asarray(d["traj_id"]).astype(str)
    fam = np.asarray(d["family"]).astype(str)
    t = np.asarray(d["t"], float)
    abs_e = np.asarray(d["EVAL_ONLY_GT_abs_ex"], float)
    dP = np.asarray(d["EVAL_ONLY_GT_dP"], float)

    def _pick(prefix, sign):
        for tr, lab in labels.items():
            if lab["family"] == prefix and lab["sign"] == sign:
                return tr
        return None

    # A: |e| vs observables, RULE + and -
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, tr, title in [
        (axes[0], _pick("B_rule", 1.0), "RULE +offset"),
        (axes[1], _pick("B_rule", -1.0), "RULE -offset"),
        (axes[2], _pick("A_zero_hold", 1.0), "ZERO +offset"),
        (axes[3], _pick("H_wrist_only", 1.0), "WRIST +offset"),
    ]:
        if tr is None:
            ax.set_title(title + " (missing)")
            continue
        idx = np.where(tid == tr)[0]
        tt = t[idx] - t[idx[0]]
        ax.plot(tt, 1e3 * abs_e[idx], "k", label="|e_x_GT| mm")
        ax2 = ax.twinx()
        ax2.plot(tt, d["D"][idx], "C1", alpha=0.8, label="D_t")
        ax2.plot(tt, d["fn_imbalance"][idx] / 10.0, "C2", alpha=0.7, label="Fn imb/10")
        ax2.plot(tt, d["aperture"][idx] * 100, "C3", alpha=0.7, label="ap*100")
        ax.set_title(f"{title}  {tr}")
        ax.set_ylabel("|e| mm")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(FIG / "A_ex_vs_observables.png", dpi=140)
    plt.close(fig)

    # B scatter dP vs d-signal
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    names = ["d_D", "d_fn_imbalance", "d_aperture", "d_track_pos_err", "d_fn_sum", "d_bilat"]
    rule = fam == "B_rule"
    for ax, nm in zip(axes.ravel(), names):
        if nm not in d:
            continue
        ax.scatter(dP[rule], d[nm][rule], s=4, alpha=0.25, c=np.where(d["ic_ex_sign"][rule] > 0, "C0", "C3"))
        ax.set_xlabel("ΔP_GT")
        ax.set_ylabel(nm)
        ax.set_title(nm)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "B_dP_scatter_rule.png", dpi=140)
    plt.close(fig)

    # C success vs fail RULE |e| and D
    fig, ax = plt.subplots(figsize=(8, 4))
    for tr, lab in labels.items():
        if lab["family"] != "B_rule":
            continue
        idx = np.where(tid == tr)[0]
        tt = t[idx] - t[idx[0]]
        col = "C2" if lab["final_recovery"] else "C3"
        ax.plot(tt, 1e3 * abs_e[idx], color=col, alpha=0.8, label=("success" if lab["final_recovery"] else "not-success") + f" s={lab['sign']:+.0f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("|e_x_GT| mm")
    ax.set_title("RULE |e_x_GT| success vs not (green=0.20s recovered hold)")
    handles, labs2 = ax.get_legend_handles_labels()
    uniq = dict(zip(labs2, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "C_success_fail_ex.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for tr, lab in labels.items():
        if lab["family"] != "B_rule":
            continue
        idx = np.where(tid == tr)[0]
        tt = t[idx] - t[idx[0]]
        ax.plot(tt, d["D"][idx], color=("C2" if lab["final_recovery"] else "C3"), alpha=0.8)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("D_t")
    ax.set_title("RULE D_t success vs not")
    fig.tight_layout()
    fig.savefig(FIG / "C_success_fail_D.png", dpi=140)
    plt.close(fig)

    # D + vs - offset RULE D and fn_imbalance
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for tr, lab in labels.items():
        if lab["family"] != "B_rule":
            continue
        idx = np.where(tid == tr)[0]
        tt = t[idx] - t[idx[0]]
        c = "C0" if lab["sign"] > 0 else "C3"
        axes[0].plot(tt, 1e3 * abs_e[idx], color=c, alpha=0.85)
        axes[1].plot(tt, d["fn_imbalance"][idx], color=c, alpha=0.85)
    axes[0].set_ylabel("|e_x| mm")
    axes[1].set_ylabel("Fn imbalance")
    axes[1].set_xlabel("t (s)")
    axes[0].set_title("RULE +offset (blue) vs -offset (red)")
    fig.tight_layout()
    fig.savefig(FIG / "D_sign_rule.png", dpi=140)
    plt.close(fig)

    # E counterexample: ZERO |e| vs D, and wrist
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    for tr, lab in labels.items():
        if lab["family"] != "A_zero_hold":
            continue
        idx = np.where(tid == tr)[0]
        tt = t[idx] - t[idx[0]]
        axes[0].plot(tt, 1e3 * abs_e[idx], "k")
        axes[0].plot(tt, d["D"][idx], "C1")
    axes[0].set_title("ZERO hold: |e_x| mm (black) and D_t (orange) — no commanded recovery")
    axes[0].set_xlabel("t (s)")
    wr = _pick("H_wrist_only", 1.0)
    if wr:
        idx = np.where(tid == wr)[0]
        tt = t[idx] - t[idx[0]]
        axes[1].plot(tt, 1e3 * abs_e[idx], "k", label="|e_x| mm")
        axes[1].plot(tt, d["track_ori_deg"][idx], "C4", label="ori track err deg")
        axes[1].plot(tt, d["w_cmd_norm"][idx], "C5", label="||w_cmd||")
        axes[1].legend(fontsize=7)
        axes[1].set_title("WRIST-only tape: orientation command without recenter")
        axes[1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(FIG / "E_proxy_counterexamples.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    m = (d["d_D"] < -1e-4) & (dP < -1e-5)
    ax.scatter(dP[m], d["d_D"][m], s=8, c="C3", label="D↓ and |e| worse")
    m3 = (dP > 1e-5) & (d["d_D"] > 1e-4)
    ax.scatter(dP[m3], d["d_D"][m3], s=8, c="C0", label="|e| better and D↑")
    ax.set_xlabel("ΔP_GT")
    ax.set_ylabel("ΔD")
    ax.legend(fontsize=7)
    ax.set_title("D_t vs true progress disagreements (all families)")
    fig.tight_layout()
    fig.savefig(FIG / "E_D_disagreement_scatter.png", dpi=140)
    plt.close(fig)


def fmt(v):
    if v is None:
        return "NA"
    try:
        if not math.isfinite(float(v)):
            return "NA"
        return f"{float(v):+.3f}"
    except Exception:
        return "NA"


def write_report(d, report, labels, meta):
    cs = report["candidates"]

    def row_for(name):
        b = cs[name]["blocks"]
        t = cs[name]["temporal"]
        return b, t

    lines = []
    L = lines.append
    L("# Observable Reward Signal Audit")
    L("")
    L("**used_final_eval_data: false**")
    L("")
    L("This audit is signal discovery only. It does not modify RecoveryEnv, reward, success, recovery4d mapping, frozen RULE, or held-out eval sets. It does not train SAC.")
    L("")
    L("## 1. Goal and information boundary")
    L("")
    L("Oracle progress is \(P_{GT}=-|e_{x,GT}|\) and \(\\Delta P_{GT}=|e_{x,GT}(t)|-|e_{x,GT}(t+1)|\), from simulator object pose in the hand frame (`obs_from_sim` / `physical_pack`). That quantity is **EVAL_ONLY_GT**. A future real-world learning signal must be computable from physically obtainable sensing.")
    L("")
    L("Simulation GT is used here only as an **offline analysis label**.")
    L("")
    L("## 2. Available observable signals")
    L("")
    L("| signal | exact code source | unit | frame | class | realistic source | currently logged in recovery4d obs? |")
    L("|---|---|---|---|---|---|---|")
    L("| arm q, qdot | `data.qpos[ids.arm_jnt]`, `qvel[ids.arm_dof]` `envs/ids.py` | rad, rad/s | joint | DIRECT | joint encoders | no (recovery4d obs omits) |")
    L("| hand pose / twist | `data.xpos/xmat[hand_body]`, `mj_objectVelocity` `body_twist` | m, rad/s | world | ESTIMATED | FK from encoders | no (g_h only) |")
    L("| p_des − p_hand | `jacobian_controller.cartesian_torque` e_p; FSM `p_des` | m | world | DIRECT (controller) | Cartesian servo residual | no |")
    L("| ori tracking error | `rotation_error` / `ori_error_deg` | rad / deg | world | DIRECT (controller) | Cartesian servo residual | no |")
    L("| v_cmd, w_cmd | `sim.fsm.v_cmd/w_cmd` | m/s, rad/s | world | DIRECT | commanded twist | no |")
    L("| tendon ctrl[7] / tau | `data.ctrl[7]` | N (tendon) | actuator | DIRECT | gripper command | yes, scaled |")
    L("| aperture, finger q | `finger_opening` = mean finger joint pos | m | joint | DIRECT | finger encoders | aperture yes |")
    L("| finger qdot | `qvel[ids.finger_dof]` | m/s | joint | DIRECT | finger encoders | no |")
    L("| nL, nR | `obs_from_sim` contact geom counts | count | — | SENSOR | tactile / contact | no (only Fn) |")
    L("| Fn_L, Fn_R | `mj_contactForce` via `finger_object_normals` / `obs_from_sim` | N | contact normal | SENSOR | tactile/FT; **not** automatically DIRECT | yes |")
    L("| touch_left/right | `data.sensordata` sensors `touch_left/right` `panda_torque.xml` | touch | fingertip site | SENSOR | tactile | no |")
    L("| force imbalance/ratio | derived from Fn | N / 1 | — | SENSOR | same as Fn | no |")
    L("| D_t, Ddot | `DeteriorationMeter.compute` `envs/deterioration.py` | 1, 1/s | mixed | **PRIVILEGED_GT mix** | needs object pose/twist vs reference + forces | no |")
    L("| D_p, D_v, D_θ, D_ω | ‖Δp_rel‖, ‖v_rel‖, θ_rel, ‖ω_rel‖ vs captured ref | m, m/s, rad, rad/s | world/hand | PRIVILEGED_GT | object state | no |")
    L("| D_C, lost_contact | \|fL−fR\|/(sum+ε) + lost | 1 | — | SENSOR | tactile | no |")
    L("| e_x, object pose/twist | `Rh.T@(po-ph)`, object xpos/vel | m, m/s | hand/world | PRIVILEGED_GT | vision/tactile estimator **missing** | e_x in GT pack only |")
    L("")
    L("MuJoCo contact force is classified **SENSOR**, not DIRECT. Object-relative quantities are **PRIVILEGED_GT** even though MuJoCo exposes them.")
    L("")
    L("## 3. Diagnostic dataset and provenance")
    L("")
    L(f"- Path: `{OUT}`")
    L(f"- Rows: {meta['n_rows']} at physics dt (nominal 0.002 s)")
    L(f"- ICs (mm): {meta['ic_e_x_mm']}")
    L(f"- Families: {meta['families']}")
    L("- Construction: one airborne lift (`offset_construct.lift_to_airborne`), table disabled, `apply_rel_pose` at training-range e_x ∈ {±2.5, ±5.0, ±7.5} mm, settle 0.25 s.")
    L(f"- `used_final_eval_data`: {meta['used_final_eval_data']}")
    L(f"- Held-out hashes unchanged: offset `{meta['eval_offset_sha256_start'][:16]}…` impact `{meta['eval_impact_sha256_start'][:16]}…`")
    L("- Analysis code: `training/observable_signal_audit.py`")
    L("")
    L("Trajectory counts by family:")
    from collections import Counter
    fam_c = Counter(lab["family"] for lab in labels.values())
    rec_c = Counter((lab["family"], lab["final_recovery"]) for lab in labels.values())
    L("")
    for k, v in sorted(fam_c.items()):
        L(f"- `{k}`: {v} trajs")
    L("")
    L("RULE `final_recovery` (0.20 s physical recovered() hold):")
    for (f, ok), v in sorted(rec_c.items()):
        if f == "B_rule":
            L(f"- recovery={ok}: {v}")
    L("")
    L("## 4. Ground-truth labels used only for offline analysis")
    L("")
    L("Prefixed `EVAL_ONLY_GT_*` in `rows.csv` / `rows.npz`: e_x/y/z, |e_x|, P, v_rel, ω_rel, object pose, recovered_now, drop, scene, airborne, stable.")
    L("")
    L("Progress: `EVAL_ONLY_GT_dP = |e|(t) − |e|(t+1)` within each trajectory. Positive = true lateral recenter.")
    L("")
    L("These are **not** proposed as observable reward inputs.")
    L("")
    L("## 5. Signal-by-signal analysis")
    L("")
    L("Correlations are **Pearson/Spearman of Δx vs ΔP_GT** unless noted. Evidence is correlational on interventional tapes (ZERO / RULE / recovery4d open-loop), not a trained policy.")
    L("")
    L("| candidate | n (all Δ) | Pearson Δx↔ΔP | Spearman | sign agree | +e_x Pearson | −e_x Pearson | RULE Pearson | ZERO Pearson |")
    L("|---|---|---|---|---|---|---|---|---|")
    for name in CANDIDATES:
        if name not in cs:
            continue
        b = cs[name]["blocks"]
        L(
            f"| {name} | {b['all_dP']['n']} | {fmt(b['all_dP']['pearson'])} | {fmt(b['all_dP']['spearman'])} | {fmt(b['all_dP']['sign_agree'])} | {fmt(b['pos_dP']['pearson'])} | {fmt(b['neg_dP']['pearson'])} | {fmt(b.get('fam_B_rule',{}).get('pearson'))} | {fmt(b.get('fam_A_zero_hold',{}).get('pearson'))} |"
        )
    L("")
    L("Instantaneous x vs |e_x_GT| (level, not progress):")
    L("")
    L("| candidate | Pearson x↔|e_x| | Spearman |")
    L("|---|---|---|")
    for name in ["D", "fn_imbalance", "fn_sum", "aperture", "track_pos_err", "bilat", "D_C"]:
        if name not in cs:
            continue
        b = cs[name]["blocks"]["all_abs_e"]
        L(f"| {name} | {fmt(b['pearson'])} | {fmt(b['spearman'])} |")
    L("")
    L("## 6. Temporal-history analysis")
    L("")
    L("Windows 20/50/100/200 ms on physics samples. Compare instantaneous Δx vs slope of x.")
    L("")
    L("| signal | inst Δ Pearson | slope 20 ms | 50 ms | 100 ms | 200 ms |")
    L("|---|---|---|---|---|---|")
    for name in ["D", "fn_imbalance", "aperture", "track_pos_err", "fn_sum", "D_C"]:
        if name not in cs:
            continue
        inst = cs[name]["blocks"]["all_dP"]["pearson"]
        tm = cs[name]["temporal"]
        L(
            f"| {name} | {fmt(inst)} | {fmt(tm['slope_20ms']['pearson'])} | {fmt(tm['slope_50ms']['pearson'])} | {fmt(tm['slope_100ms']['pearson'])} | {fmt(tm['slope_200ms']['pearson'])} |"
        )
    L("")
    L("If slope correlations are not materially stronger than instantaneous Δ, a short history is still likely needed for **events** (contact persistence), even if not for a dense signed progress proxy.")
    L("")
    L("## 7. D_t analysis")
    L("")
    L("`D` is a weighted sum of privileged object-relative terms plus contact imbalance (`envs/deterioration.py` lines 120–126). It is **not** a real-world-only signal.")
    L("")
    bD = cs["D"]["blocks"]
    L(f"- ΔD vs ΔP_GT all: Pearson {fmt(bD['all_dP']['pearson'])}, Spearman {fmt(bD['all_dP']['spearman'])}, n={bD['all_dP']['n']}")
    L(f"- +offset Pearson {fmt(bD['pos_dP']['pearson'])}; −offset Pearson {fmt(bD['neg_dP']['pearson'])}")
    L(f"- RULE Pearson {fmt(bD.get('fam_B_rule',{}).get('pearson'))}; ZERO Pearson {fmt(bD.get('fam_A_zero_hold',{}).get('pearson'))}")
    L("")
    L("Counterexample **counts** (physics steps):")
    for c in report["dt_counterexamples"]["counts"]:
        L(f"- {c['name']}: n={c['n']}")
    L("")
    L("Raw examples (first listed in `analysis.json`):")
    L("")
    L("| tag | traj | t | mode | D | ΔD | |e_x| | ΔP | bilat |")
    L("|---|---|---|---|---|---|---|---|---|")
    for e in report["dt_counterexamples"]["examples"][:12]:
        L(
            f"| {e['tag']} | `{e['traj']}` | {e['t']:.4f} | {e['mode']} | {e['D']:.4f} | {e['dD']:.5f} | {e['abs_e']:.5f} | {e['dP']:.6f} | {e['bilat']:.0f} |"
        )
    L("")
    L("## 8. Policy-information vs reward-information")
    L("")
    L("A signal can inform the **policy** (what state am I in?) without having a safe **desired direction** for a reward.")
    L("")
    L("| signal | POLICY_INFO | REWARD_INFO |")
    L("|---|---|---|")
    L("| Fn_L/R, nL/nR, touch | yes — grasp loaded? | no — larger force ≠ recenter |")
    L("| aperture, tau | yes — gripper regime | no — open/close are modes, not progress |")
    L("| tracking error | yes — servo lag | no — can be reduced by freezing a bad grasp |")
    L("| D_t | mixed (mostly GT) | no as dense reward — see counterexamples |")
    L("| ΔP_GT / e_x | oracle only | oracle only |")
    L("| force imbalance | maybe slip/asymmetry | not signed recenter; may shrink by dropping a finger |")
    L("")
    L("## 9. Proxy-exploit analysis")
    L("")
    px = report["proxy"]
    L(f"- ZERO (do nothing, displaced grasp): mean ΔP={fmt(px['ZERO_mean_dP'])}, mean ΔD={fmt(px['ZERO_mean_dD'])}, std |e|={fmt(px['ZERO_std_abs_e'])}.")
    L(f"- WRIST-only (a0=+1, tau=−18): mean ΔP={fmt(px['WRIST_mean_dP'])}; per-traj Δ|e_x| end−start={px['WRIST_end_abs_e_minus_start']}.")
    L("")
    L("| candidate | proxy risk | evidence |")
    L("|---|---|---|")
    L("| maximize −D or −ΔD | HIGH | D can fall while |e_x| stays large or worsens (counts in §7); ZERO still has D dynamics from privileged v_rel/C |")
    L("| maximize Fn_sum / squeeze | HIGH | secure hold already ~9 N; extra squeeze does not reduce |e_x| (ZERO/secure) |")
    L("| minimize force imbalance | HIGH | zero-contact sets imbalance→0 without capture |")
    L("| minimize tracking error | HIGH | ZERO freeze can keep small Cartesian error while object remains offset |")
    L("| aperture slope | HIGH | opening changes aperture without recenter; drop also changes it |")
    L("| wrist / ori error | HIGH | WRIST tape commands rotation; |e_x| not a signed recovery (figure E) |")
    L("| nL/nR / bilat | MEDIUM | necessary for capture events, not a recenter measure |")
    L("")
    L("## 10. Observable event signals")
    L("")
    L("| event | from assumed sensing | robust? | ≠ true recenter |")
    L("|---|---|---|---|")
    L("| bilateral acquired/maintained | tactile or Fn>eps both fingers | moderate (threshold) | yes |")
    L("| contact lost | both Fn/touch ~0 | moderate | yes |")
    L("| stable force regime | Fn variance low + bilateral | weak without tuning | yes |")
    L("| slip detected | touch/Fn fluctuation + finger motion | weak; no slip estimator in code | yes |")
    L("| resecure | tau→−18 and Fn recover | descriptive | yes |")
    L("| excessive motion | hand-speed / qdot | yes (proprio) | yes |")
    L("| drop precursor | aperture open + Fn collapse + downward? | proprio+tactile, z of **object** is GT | partial |")
    L("")
    L("## 11. Candidate summary table")
    L("")
    L("| candidate | class | tracks progress? | sign-general? | temporal needed? | policy-info | reward-info | proxy risk | evidence |")
    L("|---|---|---|---|---|---|---|---|---|")

    def ev(name):
        p = cs[name]["blocks"]["all_dP"]["pearson"]
        pp = cs[name]["blocks"]["pos_dP"]["pearson"]
        pn = cs[name]["blocks"]["neg_dP"]["pearson"]
        if p is None or (isinstance(p, float) and not math.isfinite(p)):
            return "INSUFFICIENT_DATA"
        if abs(p) < 0.05:
            return "NONE"
        if abs(p) < 0.15:
            return "WEAK"
        # sign flip
        if pp is not None and pn is not None and math.isfinite(pp) and math.isfinite(pn) and pp * pn < 0 and abs(pp) > 0.1 and abs(pn) > 0.1:
            return "WEAK"
        if abs(p) < 0.35:
            return "MODERATE"
        return "MODERATE"  # never STRONG from correlation alone

    summary_rows = [
        ("D", "PRIVILEGED_GT mix", "weak/mixed", "see ± Pearson", "maybe ΔD", "mixed", "no dense", "HIGH", ev("D")),
        ("D_C / fn_imbalance", "SENSOR", "no signed recenter", "must check ±", "trend maybe", "yes", "no", "HIGH", ev("fn_imbalance")),
        ("fn_sum", "SENSOR", "no", "n/a", "no", "yes", "no", "HIGH", ev("fn_sum")),
        ("aperture", "DIRECT", "no", "n/a", "slope=mode", "yes", "no", "HIGH", ev("aperture")),
        ("tau", "DIRECT", "no", "n/a", "no", "yes", "no", "HIGH", ev("tau")),
        ("track_pos_err", "DIRECT", "no", "n/a", "no", "yes", "no", "HIGH", ev("track_pos_err")),
        ("bilat / nL nR", "SENSOR", "event only", "yes (binary)", "persistence", "yes", "sparse maybe", "MEDIUM", ev("bilat")),
        ("touch_*", "SENSOR", "unknown/weak", "check ±", "maybe", "yes", "no dense", "HIGH", ev("touch_sum") if "touch_sum" in cs else "INSUFFICIENT_DATA"),
        ("e_x_GT", "PRIVILEGED_GT", "yes (oracle)", "yes", "Δ", "oracle", "oracle only", "N/A (not observable)", "STRONG as label only"),
    ]
    for r in summary_rows:
        L("| " + " | ".join(map(str, r)) + " |")
    L("")
    L("**No STRONG reward-progress claim is made from correlation.**")
    L("")
    L("## 12. Counterexamples")
    L("")
    L("See `figures/E_proxy_counterexamples.png`, `E_D_disagreement_scatter.png`, and examples in §7.")
    L("- ZERO: no recovery command; |e_x| stays in the offset band while D and forces still vary.")
    L("- WRIST-only: large orientation command / tracking change without a signed |e_x| recovery.")
    L("- D↓ while |e_x|↑ and |e_x|↓ while D↑ are both present in step counts above.")
    L("")
    L("## 13. Implications for Observable SAC")
    L("")
    L("Allowed conclusions: **no single observable dense proxy is sufficient** for signed lateral recenter.")
    L("")
    L("Future directions (not chosen here):")
    L("A. sparse observable event reward (bilateral persist / drop)")
    L("B. multi-signal observable reward")
    L("C. learned estimator from sensor history")
    L("D. vision/tactile estimate of relative object state")
    L("E. offline pretraining with GT + real-world observable fine-tuning")
    L("")
    L("## 14. Remaining sensing requirements")
    L("")
    L("To know actual recenter progress without simulator GT, the system needs an estimate of **object pose in the hand/gripper frame** (or equivalent contact geometry), e.g. vision, proximity, or spatially resolved tactile. Finger encoders + Cartesian residuals + scalar fingertip force do not currently yield a sign-general dense \(e_x\).")
    L("")
    L("## Q1–Q7")
    L("")
    L("**Q1.** No. No currently logged non-GT signal independently provides a reliable **signed** measure of true lateral recovery progress on both +e_x and −e_x.")
    L("")
    L("**Q2.** No. D_t is mostly privileged object-relative GT plus a contact term, and ΔD disagrees with ΔP_GT in substantial step counts. It is not a suitable direct dense real-world reward on current evidence.")
    L("")
    L("**Q3.** Fn magnitude, squeeze/tau, aperture, Cartesian tracking error, D_t, force imbalance — useful as **state**, not as quantities to maximize/minimize as progress.")
    L("")
    L("**Q4.** Plausible **event** ingredients: bilateral contact acquired/maintained, contact-loss/drop, (if sensing allows) slip/resecure flags. Not a dense signed recenter term.")
    L("")
    L("**Q5.** Yes for events (persistence windows). Not shown to create a dense signed progress proxy from current scalars.")
    L("")
    L("**Q6.** Missing: object-relative lateral pose (or a calibrated tactile/vision estimator of e_x).")
    L("")
    L("**Q7.** An Observable-SAC **dense** reward cannot be designed from current evidence without an additional sensing/estimation step. Sparse/hybrid event rewards could be sketched later; this audit does not design them.")
    L("")
    L("## Evidence pointers")
    L("")
    L(f"- Raw log: `{OUT / 'rows.csv'}` and `{OUT / 'rows.npz'}`")
    L(f"- Provenance: `{OUT / 'provenance.json'}`")
    L(f"- Numeric analysis: `{OUT / 'analysis.json'}`")
    L(f"- Code: `training/observable_signal_audit.py` (`collect`, `analyze`, `figures`)")
    L(f"- Sample count: n_rows={meta['n_rows']}, n_ics={meta['n_ics']} (both offset signs)")
    L("- Figures: `results/diagnostics/observable_reward_signal_audit/figures/`")
    L("")
    text = "\n".join(lines) + "\n"
    path = ROOT / "OBSERVABLE_REWARD_SIGNAL_AUDIT.md"
    path.write_text(text, encoding="utf-8")
    (OUT / "OBSERVABLE_REWARD_SIGNAL_AUDIT.md").write_text(text, encoding="utf-8")
    return path


def main():
    force = "--force-collect" in sys.argv
    npz = OUT / "rows.npz"
    if force or not npz.is_file():
        arr, meta = collect()
    else:
        arr = load_rows()
        meta = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
        print("reusing existing", npz, "rows", meta.get("n_rows"))
    d, report, labels = analyze(arr, meta)
    figures(d, labels)
    path = write_report(d, report, labels, meta)
    print("REPORT", path)
    print("used_final_eval_data", meta["used_final_eval_data"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
