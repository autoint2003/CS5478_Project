"""Recenter-state observability / estimation feasibility. No SAC. No reward edits.

Does not read results/eval_sets. Diagnostic ICs only.
"""
from __future__ import annotations

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

from controllers.gripper_controller import finger_opening
from controllers.jacobian_controller import gains_from_cfg, ori_error_deg
from controllers.residual import map_recovery4d
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.contact import finger_object_normals, touch_values
from envs.deterioration import body_twist
from envs.grasp_sim import GraspSim
from envs.physical_recovery import physical_pack
from training.offset_construct import apply_rel_pose, lift_to_airborne, settle_hold
from training.replay_core import MASS, MU, disable_object_table, freeze, params_from_frozen, tick_vw

import mujoco

OUT = ROOT / "results" / "diagnostics" / "recenter_observability"
FIG = OUT / "figures"
EVAL_OFF = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMP = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"

EX_MM = np.array([-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5])
SETTLES = (0.20, 0.25, 0.30)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def spatial_contacts(sim) -> dict:
    """MuJoCo contact points as a spatial-tactile PROXY (not DIRECT deployable)."""
    ids = sim.ids
    d = sim.data
    m = sim.model
    Rh = np.array(d.xmat[ids.hand_body].reshape(3, 3), float)
    ph = np.array(d.xpos[ids.hand_body], float)
    acc = {
        ids.left_body: {"w": 0.0, "p_h": np.zeros(3), "p_f": np.zeros(3), "n": 0},
        ids.right_body: {"w": 0.0, "p_h": np.zeros(3), "p_f": np.zeros(3), "n": 0},
    }
    wr = np.zeros(6)
    for i in range(int(d.ncon)):
        c = d.contact[i]
        b1 = int(m.geom_bodyid[c.geom1])
        b2 = int(m.geom_bodyid[c.geom2])
        if ids.object_body not in (b1, b2):
            continue
        fb = b1 if b1 in acc else (b2 if b2 in acc else None)
        if fb is None:
            continue
        mujoco.mj_contactForce(m, d, i, wr)
        fn = abs(float(wr[0]))
        pw = np.array(c.pos, float)
        p_h = Rh.T @ (pw - ph)
        Rf = np.array(d.xmat[fb].reshape(3, 3), float)
        pf = np.array(d.xpos[fb], float)
        p_f = Rf.T @ (pw - pf)
        a = acc[fb]
        a["w"] += fn
        a["p_h"] += fn * p_h
        a["p_f"] += fn * p_f
        a["n"] += 1
    out = {}
    for name, bid in (("L", ids.left_body), ("R", ids.right_body)):
        a = acc[bid]
        if a["w"] > 1e-9:
            ch = a["p_h"] / a["w"]
            cf = a["p_f"] / a["w"]
        else:
            ch = cf = np.zeros(3)
        out[f"cop_h_{name}_x"] = float(ch[0])
        out[f"cop_h_{name}_y"] = float(ch[1])
        out[f"cop_h_{name}_z"] = float(ch[2])
        out[f"cop_f_{name}_x"] = float(cf[0])
        out[f"cop_f_{name}_y"] = float(cf[1])
        out[f"cop_f_{name}_z"] = float(cf[2])
        out[f"cop_n_{name}"] = float(a["n"])
        out[f"cop_w_{name}"] = float(a["w"])
    out["cop_h_x_mean"] = 0.5 * (out["cop_h_L_x"] + out["cop_h_R_x"])
    out["cop_h_x_diff"] = out["cop_h_L_x"] - out["cop_h_R_x"]
    return out


def sample(sim, extra: dict) -> dict:
    ids = sim.ids
    d = sim.data
    o = physical_pack(sim)
    p_h = np.array(d.xpos[ids.hand_body], float)
    r_h = np.array(d.xmat[ids.hand_body].reshape(3, 3), float)
    vh, _ = body_twist(sim.model, d, ids.hand_body)
    p_des = np.asarray(sim.fsm.p_des, float)
    r_des = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    fn = finger_object_normals(sim.model, d, ids)
    touch = touch_values(d, ids)
    qa = np.array(d.qpos[ids.arm_jnt], float)
    qda = np.array(d.qvel[ids.arm_dof], float)
    qf = np.array(d.qpos[ids.finger_jnt], float)
    qdf = np.array(d.qvel[ids.finger_dof], float)
    row = dict(extra)
    row.update(
        {
            "t": float(d.time),
            "EVAL_ONLY_GT_e_x": float(o["e_x"]),
            "EVAL_ONLY_GT_e_y": float(o["rh"][1]),
            "EVAL_ONLY_GT_e_z": float(o["rh"][2]),
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
            "track_pos_err": float(np.linalg.norm(p_des - p_h)),
            "track_ori_deg": ori_error_deg(r_des, r_h),
            "tau": float(d.ctrl[7]),
            "aperture": finger_opening(d, ids),
            "finger_q0": qf[0],
            "finger_q1": qf[1],
            "finger_qd0": qdf[0],
            "finger_qd1": qdf[1],
            "v_cmd_n": float(np.linalg.norm(sim.fsm.v_cmd)),
            "w_cmd_n": float(np.linalg.norm(sim.fsm.w_cmd)),
            "nL": float(o["nL"]),
            "nR": float(o["nR"]),
            "cL": 1.0 if int(o["nL"]) > 0 else 0.0,
            "cR": 1.0 if int(o["nR"]) > 0 else 0.0,
            "bilat": 1.0 if int(o["nL"]) > 0 and int(o["nR"]) > 0 else 0.0,
            "fn_l": float(fn[0]),
            "fn_r": float(fn[1]),
            "touch_l": float(touch[0]),
            "touch_r": float(touch[1]),
            "obj_z": float(o["obj_z"]),
        }
    )
    row.update(spatial_contacts(sim))
    return row


P0 = [
    "q0",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "qd0",
    "qd1",
    "qd2",
    "qd3",
    "qd4",
    "qd5",
    "qd6",
    "hand_x",
    "hand_y",
    "hand_z",
    "hand_vx",
    "hand_vy",
    "hand_vz",
    "track_pos_err",
    "track_ori_deg",
    "tau",
    "aperture",
    "finger_q0",
    "finger_q1",
    "finger_qd0",
    "finger_qd1",
    "v_cmd_n",
    "w_cmd_n",
]
P1 = P0 + ["cL", "cR", "bilat", "nL", "nR"]
P2 = P1 + ["fn_l", "fn_r", "touch_l", "touch_r"]
P3 = P2 + [
    "cop_h_L_x",
    "cop_h_L_y",
    "cop_h_L_z",
    "cop_h_R_x",
    "cop_h_R_y",
    "cop_h_R_z",
    "cop_h_x_mean",
    "cop_h_x_diff",
    "cop_f_L_x",
    "cop_f_L_y",
    "cop_f_L_z",
    "cop_f_R_x",
    "cop_f_R_y",
    "cop_f_R_z",
]
P2S = ["fn_l", "fn_r", "touch_l", "touch_r", "aperture", "tau", "cL", "cR", "bilat"]
P3S = P2S + ["cop_h_x_mean", "cop_h_L_x", "cop_h_R_x", "cop_h_x_diff"]


def airborne_base(cfg):
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("lift failed")
    disable_object_table(sim)
    freeze(sim)
    settle_hold(sim, cfg, 0.20, -18.0)
    return sim.snapshot()


def make_ic(cfg, base, e_x, settle):
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    disable_object_table(sim)
    sim.load_snapshot(base)
    disable_object_table(sim)
    freeze(sim)
    apply_rel_pose(sim, e_x)
    settle_hold(sim, cfg, settle, -18.0)
    o = physical_pack(sim)
    snap = sim.snapshot()
    snap["e_x"] = float(o["e_x"])
    return snap, float(o["e_x"])


def run_family(cfg, snap, family, proto, ic_ex, settle, rep):
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
    dt = float(sim.model.opt.timestep)
    n = int(round(float(proto["t"]) / dt))
    rows = []
    last = "STABILIZE"
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
            if cmd["mode"] != last:
                last = cmd["mode"]
                if cmd["freeze_p"]:
                    freeze(sim)
            if cmd["r_des"] is not None:
                sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
            v, w, tau = cmd["v_world"], np.zeros(3), cmd["tau"]
            mode = cmd["mode"]
            if ctrl.s.done:
                tick_vw(sim, v, w, tau, gains)
                rows.append(
                    sample(
                        sim,
                        dict(
                            family=family,
                            mode=mode,
                            ic_ex=ic_ex,
                            settle=settle,
                            rep=rep,
                            traj=f"{family}_ex{ic_ex:+.4f}_s{settle:.2f}_r{rep}",
                        ),
                    )
                )
                break
        else:
            a = np.array(proto["a"], float)
            rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            m = map_recovery4d(a, sim.fsm.r_des, rh)
            v, w, tau = m["v_world"], m["w_world"], m["tau"]
            mode = proto.get("mode", kind)
        tick_vw(sim, v, w, tau, gains)
        rows.append(
            sample(
                sim,
                dict(
                    family=family,
                    mode=mode,
                    ic_ex=ic_ex,
                    settle=settle,
                    rep=rep,
                    traj=f"{family}_ex{ic_ex:+.4f}_s{settle:.2f}_r{rep}",
                ),
            )
        )
    return rows


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    h0, h1 = _sha(EVAL_OFF), _sha(EVAL_IMP)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    print("lift base")
    base = airborne_base(cfg)
    a_slip = 2.0 * ((-2.0) - (-1.0)) / ((-18.0) - (-1.0)) - 1.0
    # Small excitations: fractions of previously demonstrated recovery4d actions.
    protocols = [
        ("ZERO", {"kind": "ZERO", "t": 0.12}),
        ("HX_SMALL", {"kind": "R4D", "t": 0.10, "a": [0.0, 0.30, 0.0, 1.0], "mode": "HX"}),
        ("WRIST_SMALL", {"kind": "R4D", "t": 0.10, "a": [0.30, 0.0, 0.0, 1.0], "mode": "WRIST"}),
        ("GRIP_SMALL", {"kind": "R4D", "t": 0.10, "a": [0.0, 0.0, 0.0, a_slip], "mode": "GRIP"}),
        ("RULE", {"kind": "RULE", "t": 1.20}),
    ]
    all_rows = []
    for e_mm in EX_MM:
        e = float(e_mm) * 1e-3
        for si, settle in enumerate(SETTLES):
            snap, got = make_ic(cfg, base, e, settle)
            print(f"IC {1e3*e:+.1f} mm settle={settle} got {1e3*got:+.2f}")
            for fam, proto in protocols:
                if fam == "RULE" and abs(e) < 1e-4:
                    continue
                all_rows.extend(run_family(cfg, snap, fam, proto, got, settle, si))
    keys = list(all_rows[0].keys())
    arr = {}
    for k in keys:
        v = [r[k] for r in all_rows]
        if isinstance(v[0], str):
            arr[k] = np.array(v, dtype="U64")
        else:
            arr[k] = np.array(v, float)
    np.savez_compressed(OUT / "rows.npz", **arr)
    feat_def = {"P0": P0, "P1": P1, "P2": P2, "P3": P3, "P2S": P2S, "P3S": P3S}
    (OUT / "feature_definitions.json").write_text(json.dumps(feat_def, indent=2), encoding="utf-8")
    meta = {
        "used_final_eval_data": False,
        "eval_offset_sha256": h0,
        "eval_impact_sha256": h1,
        "eval_offset_sha256_end": _sha(EVAL_OFF),
        "eval_impact_sha256_end": _sha(EVAL_IMP),
        "n_rows": len(all_rows),
        "e_x_mm_targets": EX_MM.tolist(),
        "settles_s": list(SETTLES),
        "families": [p[0] for p in protocols],
        "source": "constructed_offset; not eval_sets",
        "excitation_note": "HX a1=0.30, WRIST a0=0.30, GRIP tau=-2; all <= previously demonstrated recovery4d amplitudes",
    }
    if meta["eval_offset_sha256"] != meta["eval_offset_sha256_end"]:
        raise RuntimeError("held-out offset changed")
    if meta["eval_impact_sha256"] != meta["eval_impact_sha256_end"]:
        raise RuntimeError("held-out impact changed")
    (OUT / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rows", len(all_rows))
    return arr, meta


def load_rows():
    z = np.load(OUT / "rows.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def X_of(d, cols, idx):
    return np.column_stack([np.asarray(d[c], float)[idx] for c in cols])


def standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def ridge(X, y, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def logreg(X, y, lam=1.0, iters=80):
    w = np.zeros(X.shape[1])
    y = y.astype(float)
    for _ in range(iters):
        z = np.clip(X @ w, -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + lam * w
        s = p * (1 - p) + 1e-6
        H = X.T @ (s[:, None] * X) + lam * np.eye(len(w))
        try:
            w = w - np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return w


def predict_log(X, w):
    z = np.clip(X @ w, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def metrics_sign(y, p):
    y = (np.asarray(y) > 0).astype(int)
    yhat = (np.asarray(p) >= 0.5).astype(int)
    acc = float(np.mean(y == yhat)) if y.size else float("nan")
    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    rec1 = tp / max(tp + fn, 1)
    rec0 = tn / max(tn + fp, 1)
    return {
        "n": int(y.size),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "accuracy": acc,
        "balanced_accuracy": 0.5 * (rec0 + rec1),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def metrics_reg(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    err = yhat - y
    sign_err = float(np.mean(np.sign(yhat) != np.sign(y))) if y.size else float("nan")
    # ignore ~0 labels for sign error
    m = np.abs(y) >= 0.001
    se = float(np.mean(np.sign(yhat[m]) != np.sign(y[m]))) if np.any(m) else float("nan")
    return {
        "n": int(y.size),
        "MAE_mm": float(np.mean(np.abs(err)) * 1e3),
        "RMSE_mm": float(np.sqrt(np.mean(err**2)) * 1e3),
        "sign_error_rate_all": sign_err,
        "sign_error_rate_|e|>=1mm": se,
        "MAE_mm_pos": float(np.mean(np.abs(err[y > 0.001])) * 1e3) if np.any(y > 0.001) else None,
        "MAE_mm_neg": float(np.mean(np.abs(err[y < -0.001])) * 1e3) if np.any(y < -0.001) else None,
    }


def last_idx_per_traj(d, mask):
    tid = np.asarray(d["traj"]).astype(str)
    out = []
    for tr in np.unique(tid[mask]):
        idx = np.where((tid == tr) & mask)[0]
        out.append(idx[-1])
    return np.array(out, int)


def window_feats(d, cols, idx_end, n_hist):
    """mean, slope, last of each col over n_hist samples ending at idx_end."""
    feats = []
    tid = np.asarray(d["traj"]).astype(str)
    for i in idx_end:
        tr = tid[i]
        # find start of traj
        all_i = np.where(tid == tr)[0]
        pos = int(np.where(all_i == i)[0][0])
        sl = all_i[max(0, pos + 1 - n_hist) : pos + 1]
        row = []
        for c in cols:
            x = np.asarray(d[c], float)[sl]
            row.append(float(x[-1]))
            row.append(float(np.mean(x)))
            if x.size >= 3:
                t = np.arange(x.size, dtype=float)
                row.append(float(np.polyfit(t, x, 1)[0]))
            else:
                row.append(0.0)
        feats.append(row)
    return np.array(feats, float)


def fit_eval(Xtr, ytr, Xte, yte, task):
    Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
    Xtr_b, Xte_b = add_bias(Xtr_s), add_bias(Xte_s)
    if task == "sign":
        yb_tr = (ytr > 0).astype(float)
        w = logreg(Xtr_b, yb_tr)
        p = predict_log(Xte_b, w)
        return metrics_sign(yte, p), p
    w = ridge(Xtr_b, ytr, lam=1.0)
    yhat = Xte_b @ w
    return metrics_reg(yte, yhat), yhat


def analyze(d, meta):
    fam = np.asarray(d["family"]).astype(str)
    y = np.asarray(d["EVAL_ONLY_GT_e_x"], float)
    settle = np.asarray(d["settle"], float)
    groups = {"P0": P0, "P1": P1, "P2": P2, "P3": P3, "P2S": P2S, "P3S": P3S}

    # --- identifiability: settled ZERO last sample, matched |e| ---
    zmask = fam == "ZERO"
    z_last = last_idx_per_traj(d, zmask)
    ident = []
    for mag in (0.0025, 0.005, 0.0075):
        for s in SETTLES:
            ip = [i for i in z_last if abs(y[i] - mag) < 0.0008 and abs(settle[i] - s) < 1e-9]
            im = [i for i in z_last if abs(y[i] + mag) < 0.0008 and abs(settle[i] - s) < 1e-9]
            if not ip or not im:
                continue
            i, j = ip[0], im[0]
            rec = {"mag_mm": mag * 1e3, "settle": s, "e_plus": float(y[i]), "e_minus": float(y[j])}
            for g, cols in groups.items():
                xp, xm = X_of(d, cols, [i])[0], X_of(d, cols, [j])[0]
                rec[f"{g}_l2"] = float(np.linalg.norm(xp - xm))
                rec[f"{g}_maxabs"] = float(np.max(np.abs(xp - xm)))
            rec["cop_h_x_mean_plus"] = float(d["cop_h_x_mean"][i])
            rec["cop_h_x_mean_minus"] = float(d["cop_h_x_mean"][j])
            rec["fn_l_plus"], rec["fn_l_minus"] = float(d["fn_l"][i]), float(d["fn_l"][j])
            rec["fn_r_plus"], rec["fn_r_minus"] = float(d["fn_r"][i]), float(d["fn_r"][j])
            rec["ap_plus"], rec["ap_minus"] = float(d["aperture"][i]), float(d["aperture"][j])
            ident.append(rec)

    # within-sign P2 distances (same mag, different settle) vs across-sign
    p2_within, p2_across, p3_across, p3_within = [], [], [], []
    for mag in (0.0025, 0.005, 0.0075):
        plus = [i for i in z_last if abs(y[i] - mag) < 0.0008]
        minus = [i for i in z_last if abs(y[i] + mag) < 0.0008]
        Xp, Xm = X_of(d, P2, plus), X_of(d, P2, minus)
        Yp, Ym = X_of(d, P3, plus), X_of(d, P3, minus)
        if len(plus) >= 2:
            p2_within.append(float(np.linalg.norm(Xp[0] - Xp[1])))
            p3_within.append(float(np.linalg.norm(Yp[0] - Yp[1])))
        if plus and minus:
            p2_across.append(float(np.linalg.norm(Xp[0] - Xm[0])))
            p3_across.append(float(np.linalg.norm(Yp[0] - Ym[0])))

    # --- splits (documented a priori in report) ---
    # Static split: ZERO last-of-traj; train |e| in {2.5,7.5}mm, test {0,5}mm
    mag = np.round(np.abs(y) * 1e3, 1)
    tr_m = np.isin(mag, (2.5, 7.5))
    te_m = np.isin(mag, (0.0, 5.0))
    # Sign labels: exclude |e_x|<1 mm (near-zero is not a sign class).
    signed = np.abs(y) >= 0.001
    static_idx = z_last
    i_tr = static_idx[tr_m[static_idx] & signed[static_idx]]
    i_te_all = static_idx[te_m[static_idx]]
    i_te = static_idx[te_m[static_idx] & signed[static_idx]]
    i_te_incl0 = i_te_all
    # also evaluate 5mm and 0 separately

    splits = {
        "static_mag_holdout": {
            "definition": "ZERO last sample/traj; train |e_x|~2.5 and 7.5 mm both signs; test |e_x|~5.0 mm both signs. |e_x|<1 mm excluded from SIGN metrics (zeros are a third class).",
            "train_n": int(i_tr.size),
            "test_n_signed": int(i_te.size),
            "test_n_incl_near_zero": int(i_te_all.size),
        },
        "lolo_family": {
            "definition": "all timesteps |e_x|>=1mm; train families ZERO+HX_SMALL+WRIST_SMALL+GRIP_SMALL; test RULE",
        },
    }

    results = {}
    ytr, yte = y[i_tr], y[i_te]
    for g, cols in groups.items():
        Xtr, Xte = X_of(d, cols, i_tr), X_of(d, cols, i_te)
        ms, _ = fit_eval(Xtr, ytr, Xte, yte, "sign")
        mr, yhat = fit_eval(Xtr, ytr, Xte, yte, "reg")
        # per magnitude on test
        by = {}
        for lab, mm in (("zero", 0.0), ("5mm", 5.0)):
            sel = np.abs(np.abs(yte) * 1e3 - mm) < 0.8
            if np.any(sel):
                by[lab] = {
                    "sign": metrics_sign(yte[sel], (yhat[sel] > 0).astype(float)),
                    "reg": metrics_reg(yte[sel], yhat[sel]),
                }
        pos = yte > 0.001
        neg = yte < -0.001
        results[f"static_{g}"] = {
            "sign": ms,
            "reg": mr,
            "reg_pos": metrics_reg(yte[pos], yhat[pos]) if np.any(pos) else None,
            "reg_neg": metrics_reg(yte[neg], yhat[neg]) if np.any(neg) else None,
            "by_mag": by,
        }
        np.savez(
            OUT / f"pred_static_{g}.npz",
            y_true=yte,
            y_hat=yhat,
            family=np.asarray(d["family"])[i_te],
        )

    # LOLO family, subsample every 5 steps to reduce dependence
    step = np.arange(len(y))
    keep = (step % 5 == 0) & (np.abs(y) >= 0.001)
    train_f = np.isin(fam, ["ZERO", "HX_SMALL", "WRIST_SMALL", "GRIP_SMALL"]) & keep
    test_f = (fam == "RULE") & keep
    splits["lolo_family"]["train_n"] = int(np.sum(train_f))
    splits["lolo_family"]["test_n"] = int(np.sum(test_f))
    ytr, yte = y[train_f], y[test_f]
    for g, cols in groups.items():
        Xtr, Xte = X_of(d, cols, train_f), X_of(d, cols, test_f)
        ms, _ = fit_eval(Xtr, ytr, Xte, yte, "sign")
        mr, yhat = fit_eval(Xtr, ytr, Xte, yte, "reg")
        results[f"lolo_{g}"] = {"sign": ms, "reg": mr}
        np.savez(OUT / f"pred_lolo_{g}.npz", y_true=yte, y_hat=yhat)

    # temporal: ZERO windows at last index, same mag split
    dt = 0.002
    temporal = {}
    for ms_w, n_hist in ((20, 10), (50, 25), (100, 50), (200, 100)):
        temporal[ms_w] = {}
        for g, cols in (("P2", P2), ("P3", P3)):
            try:
                Xtr = window_feats(d, cols, i_tr, n_hist)
                Xte = window_feats(d, cols, i_te, n_hist)
                ms, _ = fit_eval(Xtr, y[i_tr], Xte, y[i_te], "sign")
                mr, _ = fit_eval(Xtr, y[i_tr], Xte, y[i_te], "reg")
                temporal[ms_w][g] = {"sign": ms, "reg": mr}
            except Exception as e:
                temporal[ms_w][g] = {"error": str(e)}

    # excitation last samples, same mag split
    excit = {}
    for fam_name in ("HX_SMALL", "WRIST_SMALL", "GRIP_SMALL"):
        idx = last_idx_per_traj(d, fam == fam_name)
        itr = idx[tr_m[idx] & (np.abs(y[idx]) >= 0.001)]
        ite = idx[te_m[idx]]
        excit[fam_name] = {"train_n": int(itr.size), "test_n": int(ite.size)}
        if itr.size < 4 or ite.size < 2:
            continue
        for g, cols in (("P2", P2), ("P3", P3)):
            ms, _ = fit_eval(X_of(d, cols, itr), y[itr], X_of(d, cols, ite), y[ite], "sign")
            mr, _ = fit_eval(X_of(d, cols, itr), y[itr], X_of(d, cols, ite), y[ite], "reg")
            excit[fam_name][g] = {"sign": ms, "reg": mr}

    # cop vs e_x correlation on ZERO last (label analysis only)
    cop = {
        "pearson_cop_h_x_mean": _pearson(d["cop_h_x_mean"][z_last], y[z_last]),
        "pearson_fn_l": _pearson(d["fn_l"][z_last], y[z_last]),
        "pearson_fn_imbalance": _pearson(
            np.abs(d["fn_l"][z_last] - d["fn_r"][z_last]), np.abs(y[z_last])
        ),
        "n_static_zero": int(z_last.size),
    }

    report = {
        "used_final_eval_data": False,
        "n_rows": int(len(y)),
        "identifiability": ident,
        "p2_within_l2_mean": float(np.mean(p2_within)) if p2_within else None,
        "p2_across_l2_mean": float(np.mean(p2_across)) if p2_across else None,
        "p3_within_l2_mean": float(np.mean(p3_within)) if p3_within else None,
        "p3_across_l2_mean": float(np.mean(p3_across)) if p3_across else None,
        "splits": splits,
        "estimators": results,
        "temporal": temporal,
        "excitation": excit,
        "cop": cop,
    }
    (OUT / "estimator_metrics.json").write_text(json.dumps(report, indent=2, default=_j), encoding="utf-8")
    (OUT / "split_definitions.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return report, z_last


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / d) if d > 1e-12 else float("nan")


def _j(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    return str(x)


def figures(d, report, z_last):
    FIG.mkdir(parents=True, exist_ok=True)
    y = np.asarray(d["EVAL_ONLY_GT_e_x"], float)
    # A: cop vs e_x
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(1e3 * y[z_last], 1e3 * np.asarray(d["cop_h_x_mean"])[z_last], c=np.sign(y[z_last]), cmap="coolwarm", s=40)
    ax.set_xlabel("e_x_GT (mm)")
    ax.set_ylabel("CoP_hand_x mean (mm)  [tactile PROXY]")
    ax.set_title("Spatial contact CoP vs e_x (ZERO settled)")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "A_cop_vs_ex.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(1e3 * y[z_last], np.asarray(d["fn_l"])[z_last], label="Fn_L", s=30)
    ax.scatter(1e3 * y[z_last], np.asarray(d["fn_r"])[z_last], label="Fn_R", s=30, marker="x")
    ax.set_xlabel("e_x_GT (mm)")
    ax.set_ylabel("Fn (N)")
    ax.legend()
    ax.set_title("Scalar finger forces vs e_x (ZERO settled)")
    fig.tight_layout()
    fig.savefig(FIG / "B_fn_vs_ex.png", dpi=140)
    plt.close(fig)

    # identifiability bars
    fig, ax = plt.subplots(figsize=(7, 4))
    labs, p2, p3 = [], [], []
    for rec in report["identifiability"]:
        labs.append(f"{rec['mag_mm']:.1f}/{rec['settle']}")
        p2.append(rec["P2_l2"])
        p3.append(rec["P3_l2"])
    x = np.arange(len(labs))
    ax.bar(x - 0.2, p2, 0.4, label="P2 L2(+ vs -)")
    ax.bar(x + 0.2, p3, 0.4, label="P3 L2(+ vs -)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("||obs(+) - obs(-)||")
    ax.legend()
    ax.set_title("Matched-pair observable distance +e_x vs -e_x")
    fig.tight_layout()
    fig.savefig(FIG / "C_pair_distances.png", dpi=140)
    plt.close(fig)

    # predictions P2 vs P3 static
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, g in zip(axes, ("P2", "P3")):
        z = np.load(OUT / f"pred_static_{g}.npz")
        ax.scatter(1e3 * z["y_true"], 1e3 * z["y_hat"], s=40)
        lim = [-8, 8]
        ax.plot(lim, lim, "k", lw=0.6)
        ax.set_title(f"static {g} ridge (test)")
        ax.set_xlabel("e_x_GT mm")
        ax.set_ylabel("e_hat mm")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "D_static_regression.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, g in zip(axes, ("P2", "P3")):
        z = np.load(OUT / f"pred_lolo_{g}.npz")
        ax.scatter(1e3 * z["y_true"], 1e3 * z["y_hat"], s=6, alpha=0.3)
        ax.plot([-8, 8], [-8, 8], "k", lw=0.6)
        ax.set_title(f"LOLO RULE {g}")
        ax.set_xlabel("e_x_GT mm")
        ax.set_ylabel("e_hat mm")
    fig.tight_layout()
    fig.savefig(FIG / "E_lolo_rule.png", dpi=140)
    plt.close(fig)


def fmt(v, nd=3):
    if v is None:
        return "NA"
    try:
        v = float(v)
        if not math.isfinite(v):
            return "NA"
        return f"{v:.{nd}f}"
    except Exception:
        return str(v)


def write_md(d, report, meta):
    E = report["estimators"]
    lines = []
    A = lines.append
    A("# Recenter State Observability Audit")
    A("")
    A("**used_final_eval_data: false**")
    A("")
    A("Frozen prior: `OBSERVABLE_REWARD_SIGNAL_AUDIT.md` — no scalar non-GT signal independently gives sign-general lateral **progress**. This study asks whether **state** \(e_x\) is identifiable from sensors. Weak correlations are **not** reinterpreted as a reward.")
    A("")
    A("No SAC, no reward change, no RecoveryEnv physics change, no frozen RULE/eval-set use.")
    A("")
    A("## 1. Minimum task state")
    A("")
    A("| quantity | role |")
    A("|---|---|")
    A("| \(e_x = [R_h^T(p_{obj}-p_{hand})]_x\) | **NECESSARY FOR RECENTER CONTROL** (signed lateral error) |")
    A("| \(\\dot e_x\) | **USEFUL BUT NOT NECESSARY** for a first recenter law; needed for damping/slip timing |")
    A("| bilateral / capture | **NECESSARY FOR SAFETY** (hold vs drop) |")
    A("| relative angular rate | **NECESSARY FOR SAFETY** of the current success definition; not required to *define* recenter |")
    A("| full 6D object pose | **EVALUATION ONLY** / not shown necessary for 1-D hand-x recenter |")
    A("| 24-D privileged recovery4d obs | **NOT ALL NECESSARY**; many channels are relative/GT |")
    A("")
    A("## 2. Sensor inventory")
    A("")
    A("| modality | in sim? | logged here? | realistic Panda? | extra HW? | signed lateral? | slip in time? |")
    A("|---|---|---|---|---|---|---|")
    A("| A proprio q/qdot, FK hand, cmds, tracking residual | yes | yes | yes (encoders+FK) | no | **no in principle** (hand state independent of object x) | hand motion only |")
    A("| B gripper aperture/finger/tau | yes | yes | yes | no | not signed object x | finger motion |")
    A("| C binary contact L/R | yes (geom count) | yes | possible (contact/touch) | maybe | not signed | persistence |")
    A("| D scalar Fn L/R, touch sites | yes `mj_contactForce` / `touch_*` | yes | needs F/T or tactile | **yes if not already on gripper** | not shown signed | fluctuation |")
    A("| E spatial tactile / CoP | MuJoCo `contact.pos` **proxy only** | yes as PROXY | **not** in default panda_torque interface | **yes** | candidate | contact migration |")
    A("| F vision relative pose | not implemented | no | camera | **yes** | **yes if it outputs e_x** | via tracking |")
    A("")
    A("VERIFIED FROM CODE: `assets/panda_torque.xml` exposes `touch_left/right` (scalar touch sites), not a taxel array. Panda joint encoders are assumed. Project has **no** camera observation pipeline.")
    A("")
    A("## 3. Dataset")
    A("")
    A(f"- `{OUT}`  n_rows={meta['n_rows']}  ICs at {meta['e_x_mm_targets']} mm × settles {meta['settles_s']}")
    A(f"- Families: {meta['families']}")
    A(f"- `{meta['excitation_note']}`")
    A(f"- used_final_eval_data: {meta['used_final_eval_data']}")
    A("- Construction: `training/offset_construct.apply_rel_pose` after shared airborne lift (not eval npz).")
    A("")
    A("## 4. Split definitions (fixed before interpreting scores)")
    A("")
    A(json.dumps(report["splits"], indent=2))
    A("")
    A("Ridge λ=1, logistic IRLS λ=1, features z-scored on **train only**. Diagnostic models, not SOTA.")
    A("")
    A("## 5. Identifiability / symmetry (ZERO settled, matched |e_x|)")
    A("")
    A(f"- Mean P2 L2 **within** sign (different settle): {fmt(report['p2_within_l2_mean'])}")
    A(f"- Mean P2 L2 **across** + vs −: {fmt(report['p2_across_l2_mean'])}")
    A(f"- Mean P3 L2 within: {fmt(report['p3_within_l2_mean'])}")
    A(f"- Mean P3 L2 across + vs −: {fmt(report['p3_across_l2_mean'])}")
    A("")
    A("| mag mm | settle | P0 L2 | P2 L2 | P3 L2 | CoP_x + | CoP_x − | FnL + | FnL − |")
    A("|---|---|---|---|---|---|---|---|---|")
    for rec in report["identifiability"]:
        A(
            f"| {rec['mag_mm']:.1f} | {rec['settle']:.2f} | {fmt(rec['P0_l2'])} | {fmt(rec['P2_l2'])} | {fmt(rec['P3_l2'])} | {1e3*rec['cop_h_x_mean_plus']:.2f} | {1e3*rec['cop_h_x_mean_minus']:.2f} | {fmt(rec['fn_l_plus'],2)} | {fmt(rec['fn_l_minus'],2)} |"
        )
    A("")
    A("If P0/P2 across-sign distance ≲ within-sign distance, **+e_x and −e_x are not identifiable** from those observations (parallel-jaw + scalar force **symmetry** about the grasp midline).")
    A("")
    A("## 6. Static estimator metrics")
    A("")
    A("Split: `static_mag_holdout`.")
    A("")
    A("| group | dim note | sign acc | bal acc | tn,fp,fn,tp | MAE mm | RMSE mm | sign err≥1mm | MAE+ | MAE− |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for g in ("P0", "P1", "P2", "P3", "P2S", "P3S"):
        r = E[f"static_{g}"]
        c = r["sign"]["confusion"]
        note = "p>>n" if g in ("P0", "P1", "P2", "P3") else "reduced"
        A(
            f"| {g} | {note} | {fmt(r['sign']['accuracy'])} | {fmt(r['sign']['balanced_accuracy'])} | {c['tn']},{c['fp']},{c['fn']},{c['tp']} | {fmt(r['reg']['MAE_mm'])} | {fmt(r['reg']['RMSE_mm'])} | {fmt(r['reg']['sign_error_rate_|e|>=1mm'])} | {fmt((r['reg_pos'] or {}).get('MAE_mm') if r['reg_pos'] else None)} | {fmt((r['reg_neg'] or {}).get('MAE_mm') if r['reg_neg'] else None)} |"
        )
    A("")
    A("Full P0–P3 static ridge has **more features than train samples**; wild MAE is ill-posed, not observability. Trust matched-pair L2 and reduced P2S/P3S.")
    A("")
    for lab in ("zero", "5mm"):
        if lab in E["static_P3"]["by_mag"]:
            b = E["static_P3"]["by_mag"][lab]
            A(f"- {lab}: sign acc {fmt(b['sign']['accuracy'])}, MAE {fmt(b['reg']['MAE_mm'])} mm, n={b['sign']['n']}")
    A("")
    A("## 7. Temporal histories (static mag split, ZERO windows)")
    A("")
    A("| window | P2 sign acc | P2 MAE mm | P3 sign acc | P3 MAE mm |")
    A("|---|---|---|---|---|")
    for w in (20, 50, 100, 200):
        tw = report["temporal"].get(str(w)) or report["temporal"].get(w) or {}
        p2, p3 = tw.get("P2", {}), tw.get("P3", {})
        A(
            f"| {w} ms | {fmt((p2.get('sign') or {}).get('accuracy'))} | {fmt((p2.get('reg') or {}).get('MAE_mm'))} | {fmt((p3.get('sign') or {}).get('accuracy'))} | {fmt((p3.get('reg') or {}).get('MAE_mm'))} |"
        )
    A("")
    A("## 8. Controlled excitation (last sample, same mag split)")
    A("")
    for name, blk in report["excitation"].items():
        A(f"### {name} train_n={blk.get('train_n')} test_n={blk.get('test_n')}")
        for g in ("P2", "P3"):
            if g not in blk:
                continue
            r = blk[g]
            A(
                f"- {g}: sign acc {fmt(r['sign']['accuracy'])} bal {fmt(r['sign']['balanced_accuracy'])} MAE {fmt(r['reg']['MAE_mm'])} mm signerr {fmt(r['reg']['sign_error_rate_|e|>=1mm'])}"
            )
    A("")
    A("## 9. Cross-family generalization (train ZERO+small probes, test RULE)")
    A("")
    A("| group | sign acc | bal acc | MAE mm | RMSE mm | sign err ≥1mm | n_test |")
    A("|---|---|---|---|---|---|---|")
    for g in ("P0", "P1", "P2", "P3", "P2S", "P3S"):
        r = E[f"lolo_{g}"]
        A(
            f"| {g} | {fmt(r['sign']['accuracy'])} | {fmt(r['sign']['balanced_accuracy'])} | {fmt(r['reg']['MAE_mm'])} | {fmt(r['reg']['RMSE_mm'])} | {fmt(r['reg']['sign_error_rate_|e|>=1mm'])} | {r['sign']['n']} |"
        )
    A("")
    A("If P2 looks good on static ICs but collapses on RULE, it is **CONTROLLER-CONFOUNDED**.")
    A("")
    A(f"CoP vs e_x Pearson (ZERO last, n={report['cop']['n_static_zero']}): {fmt(report['cop']['pearson_cop_h_x_mean'])}. Fn_L vs e_x: {fmt(report['cop']['pearson_fn_l'])}.")
    A("")
    A("## 10. Sensing-class verdicts")
    A("")
    A("| config | verdict |")
    A("|---|---|")
    A("| S0 proprioception only | **FUNDAMENTALLY_AMBIGUOUS** — object x is not in the robot kinematic state (VERIFIED FROM CODE + matched-pair P0 distances). |")
    A("| S1 + binary contact | **INSUFFICIENT** — bilateral capture is common to ±e_x. |")
    A("| S2 + scalar force | **FUNDAMENTALLY_AMBIGUOUS / INSUFFICIENT** if across-sign P2 distance ≲ within-sign; parallel-jaw scalar Fn does not encode hand-x sign. |")
    A("| S3 + spatial CoP proxy | **PROMISING_BUT_UNPROVEN** if CoP tracks e_x and P3 beats P2 on sign; still **not deployable** without tactile hardware (MuJoCo contact.pos ≠ Panda sensor). |")
    A("| S4 vision of e_x | **NOT_TESTED** in sim; **INFERENCE**: a camera estimate of signed lateral relative displacement would make e_x **directly** the observation (reward + policy + trigger). Full 6D not required. |")
    A("")
    A("## 11. Reward implication (no design)")
    A("")
    A("If a **deployable** \(\\hat e_x\) existed, a future dense term could be \(\\Delta|\\hat e_x|\) analogously to \(\\Delta|e_{x,GT}|\). Expected exploits: estimator bias that shrinks \(|\\hat e|\) by changing contacts/lighting/force without moving the object; sign flips causing the policy to drive the wrong way. Policy-state \(\\hat e\) and reward \(\\hat e\) need not be the same network, but both require identifiability first.")
    A("")
    A("## 12. Additional sensing")
    A("")
    A("If S0–S2 are ambiguous, the result is **ADDITIONAL SENSING REQUIRED**. Minimum plausible missing modality: **spatial tactile** (contact location / CoP on finger pads) **or** **vision** of signed relative lateral displacement. Do not hide this behind a larger neural net on S2.")
    A("")
    A("## Q1–Q9")
    A("")
    A("See the metrics above; narrative answers are in the repository return message and rest on those numbers plus matched-pair L2.")
    A("")
    A("## Paths")
    A("")
    A(f"- raw: `{OUT / 'rows.npz'}`")
    A(f"- provenance: `{OUT / 'provenance.json'}`")
    A(f"- metrics: `{OUT / 'estimator_metrics.json'}`")
    A(f"- splits: `{OUT / 'split_definitions.json'}`")
    A(f"- features: `{OUT / 'feature_definitions.json'}`")
    A(f"- figures: `{FIG}`")
    A("- code: `training/recenter_observability_audit.py`")
    A("")
    text = "\n".join(lines) + "\n"
    (ROOT / "RECENTER_OBSERVABILITY_AUDIT.md").write_text(text, encoding="utf-8")
    (OUT / "RECENTER_OBSERVABILITY_AUDIT.md").write_text(text, encoding="utf-8")
    return ROOT / "RECENTER_OBSERVABILITY_AUDIT.md"


def main():
    force = "--force-collect" in sys.argv
    if force or not (OUT / "rows.npz").is_file():
        arr, meta = collect()
    else:
        arr = load_rows()
        meta = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
        print("reuse", meta.get("n_rows"))
    report, z_last = analyze(arr, meta)
    figures(arr, report, z_last)
    p = write_md(arr, report, meta)
    print("REPORT", p)
    print("P2 within/across", report["p2_within_l2_mean"], report["p2_across_l2_mean"])
    print("P3 within/across", report["p3_within_l2_mean"], report["p3_across_l2_mean"])
    print("static P2", report["estimators"]["static_P2"]["sign"], report["estimators"]["static_P2"]["reg"])
    print("static P3", report["estimators"]["static_P3"]["sign"], report["estimators"]["static_P3"]["reg"])
    print("lolo P2", report["estimators"]["lolo_P2"]["sign"], report["estimators"]["lolo_P2"]["reg"])
    print("lolo P3", report["estimators"]["lolo_P3"]["sign"], report["estimators"]["lolo_P3"]["reg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
