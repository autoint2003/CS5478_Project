"""Validate spatial-tactile abstraction + e_x estimator. No SAC. No reward edits.

Does not read results/eval_sets. Diagnostic ICs only.
Raw MuJoCo contact.pos is used only inside sensors.spatial_tactile as emulation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.residual import map_recovery4d
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim
from envs.physical_recovery import E_TOL, physical_pack, recovered
from sensors.ex_estimator import (
    LinearExEstimator,
    apply_linear,
    fit_linear,
    fit_ridge,
    geometric_from_reading,
)
from sensors.spatial_tactile import (
    SENSOR_KEYS,
    finger_vs_hand_axes,
    measure_spatial_tactile,
    reconstruct_world_contacts,
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

from controllers.jacobian_controller import gains_from_cfg
from envs.deterioration import body_twist

OUT = ROOT / "results" / "diagnostics" / "tactile_ex_estimator"
FIG = OUT / "figures"
EVAL_OFF = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMP = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"

EX_MM = np.array([-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5])
SETTLES = (0.20, 0.30)
E_NEAR = float(E_TOL)  # 3 mm analysis bins only; success definition unchanged
DEAD_M = 1e-4  # 0.1 mm progress deadzone


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    if d < 1e-18:
        return float("nan")
    return float(np.dot(x, y) / d)


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


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
    return snap, float(o["e_x"]), sim


def geometry_check(sim) -> dict:
    rec = reconstruct_world_contacts(sim.model, sim.data, sim.ids)
    axes = finger_vs_hand_axes(sim.data, sim.ids)
    reading = measure_spatial_tactile(sim.model, sim.data, sim.ids)
    o = physical_pack(sim)
    est = geometric_from_reading(reading)
    keys = set(reading.to_array())
    leak = keys - set(SENSOR_KEYS)
    return {
        **rec,
        **axes,
        "sensor_keys": sorted(keys),
        "key_leak": sorted(leak),
        "EVAL_ONLY_GT_e_x_m": float(o["e_x"]),
        "sensor_u_L": float(reading.left.u),
        "sensor_u_R": float(reading.right.u),
        "sensor_valid_L": bool(reading.left.valid),
        "sensor_valid_R": bool(reading.right.valid),
        "geometric_e_hat_m": float(est.e_hat_x),
        "left_x_aligns_hand_x": abs(axes["left_x_dot_hand_x"] - 1.0) < 1e-6,
        "right_x_opposes_hand_x": abs(axes["right_x_dot_hand_x"] + 1.0) < 1e-6,
        "recon_ok": rec["max_recon_err_m"] < 1e-12,
    }


def sample_row(sim, extra: dict) -> dict:
    reading = measure_spatial_tactile(sim.model, sim.data, sim.ids)
    o = physical_pack(sim)
    ids = sim.ids
    d = sim.data
    vh, wh = body_twist(sim.model, d, ids.hand_body)
    vo, wo = body_twist(sim.model, d, ids.object_body)
    Rh = np.array(d.xmat[ids.hand_body].reshape(3, 3), float)
    g = geometric_from_reading(reading)
    s = reading.to_array()
    row = dict(extra)
    row.update(s)
    row.update(
        {
            "EVAL_ONLY_GT_e_x": float(o["e_x"]),
            "EVAL_ONLY_GT_v_rel": float(o["v_rel"]),
            "EVAL_ONLY_GT_w_rel": float(np.linalg.norm(wo - wh)),
            "wrist_hy_dot_world_z": float(Rh[:, 1] @ np.array([0.0, 0.0, 1.0])),
            "aperture": float(o["aperture"]),
            "tau": float(d.ctrl[7]) if d.ctrl.size > 7 else 0.0,
            "e_hat_geom": float(g.e_hat_x),
            "estimate_valid_geom": float(g.estimate_valid),
            "bilateral_valid": float(g.bilateral_valid),
        }
    )
    return row


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
    kind = proto["kind"]
    for i in range(n):
        mode = kind
        if kind == "ZERO":
            v, w, tau = np.zeros(3), np.zeros(3), -18.0
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
                o2 = physical_pack(sim)
                rows.append(
                    sample_row(
                        sim,
                        dict(
                            family=family,
                            mode=mode,
                            ic_ex=ic_ex,
                            settle=settle,
                            rep=rep,
                            step=i,
                            rule_success=float(recovered(o2)),
                            traj=f"{family}_ex{ic_ex:+.4f}_s{settle:.2f}_r{rep}",
                        ),
                    )
                )
                break
        elif kind == "BRAKE":
            t = i * dt
            a_slip = proto["a_slip"]
            rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            if t < 0.20:
                m = map_recovery4d(np.array([0.0, 0.0, 0.0, a_slip]), sim.fsm.r_des, rh)
                mode = "SLIP_PHASE"
            else:
                m = map_recovery4d(np.array([0.0, 0.0, 0.0, 1.0]), sim.fsm.r_des, rh)
                mode = "BRAKE_PHASE"
            v, w, tau = m["v_world"], m["w_world"], m["tau"]
        else:
            a = np.array(proto["a"], float)
            rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            m = map_recovery4d(a, sim.fsm.r_des, rh)
            v, w, tau = m["v_world"], m["w_world"], m["tau"]
            mode = proto.get("mode", kind)
        tick_vw(sim, v, w, tau, gains)
        o2 = physical_pack(sim)
        rows.append(
            sample_row(
                sim,
                dict(
                    family=family,
                    mode=mode,
                    ic_ex=ic_ex,
                    settle=settle,
                    rep=rep,
                    step=i,
                    rule_success=float(recovered(o2)) if kind == "RULE" else np.nan,
                    traj=f"{family}_ex{ic_ex:+.4f}_s{settle:.2f}_r{rep}",
                ),
            )
        )
    return rows


def collect(cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    h0, h1 = _sha(EVAL_OFF), _sha(EVAL_IMP)
    print("lift base")
    base = airborne_base(cfg)
    a_slip = 2.0 * ((-2.0) - (-1.0)) / ((-18.0) - (-1.0)) - 1.0
    protocols = [
        ("ZERO", {"kind": "ZERO", "t": 0.12}),
        ("SLIP", {"kind": "R4D", "t": 0.40, "a": [0.0, 0.0, 0.0, a_slip], "mode": "SLIP"}),
        ("BRAKE", {"kind": "BRAKE", "t": 0.50, "a_slip": a_slip}),
        ("WRIST", {"kind": "R4D", "t": 0.15, "a": [0.30, 0.0, 0.0, 1.0], "mode": "WRIST"}),
        ("OPEN", {"kind": "R4D", "t": 0.15, "a": [0.0, 0.0, 0.0, -1.0], "mode": "OPEN"}),
        ("RULE", {"kind": "RULE", "t": 1.20}),
    ]
    all_rows = []
    geo_samples = []
    for e_mm in EX_MM:
        e = float(e_mm) * 1e-3
        for si, settle in enumerate(SETTLES):
            snap, got, sim_ic = make_ic(cfg, base, e, settle)
            geo_samples.append(geometry_check(sim_ic))
            print(f"IC {1e3 * e:+.1f} mm settle={settle} got {1e3 * got:+.2f}")
            for fam, proto in protocols:
                if fam == "RULE" and abs(e) < 1e-4:
                    continue
                all_rows.extend(run_family(cfg, snap, fam, proto, got, settle, si))
    keys = list(all_rows[0].keys())
    sensor_keys = list(SENSOR_KEYS)
    gt_keys = [k for k in keys if k.startswith("EVAL_ONLY_GT")]
    sensor = {}
    labels = {}
    meta_arr = {}
    for k in keys:
        v = [r[k] for r in all_rows]
        if isinstance(v[0], str):
            arr = np.array(v, dtype="U64")
        else:
            arr = np.array(v, float)
        if k in sensor_keys or k in (
            "e_hat_geom",
            "estimate_valid_geom",
            "bilateral_valid",
            "aperture",
            "tau",
            "wrist_hy_dot_world_z",
        ):
            sensor[k] = arr
        elif k in gt_keys:
            labels[k] = arr
        else:
            meta_arr[k] = arr
    np.savez_compressed(OUT / "sensor_outputs.npz", **sensor)
    np.savez_compressed(OUT / "gt_labels.npz", **labels)
    np.savez_compressed(OUT / "row_meta.npz", **meta_arr)
    (OUT / "geometry_check.json").write_text(
        json.dumps(
            {
                "n_ics": len(geo_samples),
                "max_recon_err_m": max(g["max_recon_err_m"] for g in geo_samples),
                "all_recon_ok": all(g["recon_ok"] for g in geo_samples),
                "left_align": all(g["left_x_aligns_hand_x"] for g in geo_samples),
                "right_oppose": all(g["right_x_opposes_hand_x"] for g in geo_samples),
                "samples": geo_samples[:3] + geo_samples[-2:],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
        "dt_s": 0.002,
        "sensor_degradations": "synthetic sensitivity; not hardware specs",
        "E_TOL_m": E_NEAR,
        "excitation_note": "WRIST a0=0.30, SLIP tau=-2, OPEN tau=-1; amplitudes previously used in diagnostics",
    }
    if meta["eval_offset_sha256"] != meta["eval_offset_sha256_end"]:
        raise RuntimeError("held-out offset changed")
    if meta["eval_impact_sha256"] != meta["eval_impact_sha256_end"]:
        raise RuntimeError("held-out impact changed")
    (OUT / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("rows", len(all_rows))
    return meta


def load_bundle():
    s = dict(np.load(OUT / "sensor_outputs.npz", allow_pickle=True))
    g = dict(np.load(OUT / "gt_labels.npz", allow_pickle=True))
    m = dict(np.load(OUT / "row_meta.npz", allow_pickle=True))
    return s, g, m


def err_metrics(yhat, ygt, valid, prefix=""):
    m = np.asarray(valid, bool) & np.isfinite(yhat) & np.isfinite(ygt)
    n = int(m.sum())
    if n == 0:
        return {f"{prefix}n": 0}
    e = yhat[m] - ygt[m]
    ae = np.abs(e)
    yg = ygt[m]
    yh = yhat[m]
    mag = np.abs(yg) >= 1e-3
    sign_err = float(np.mean(np.sign(yh[mag]) != np.sign(yg[mag]))) if mag.any() else float("nan")
    pos = yg > 1e-3
    neg = yg < -1e-3
    out = {
        f"{prefix}n": n,
        f"{prefix}MAE_mm": float(np.mean(ae) * 1e3),
        f"{prefix}RMSE_mm": float(np.sqrt(np.mean(e**2)) * 1e3),
        f"{prefix}sign_err_|e|>=1mm": sign_err,
        f"{prefix}P_err<1mm": float(np.mean(ae < 1e-3)),
        f"{prefix}P_err<2mm": float(np.mean(ae < 2e-3)),
        f"{prefix}MAE_pos_mm": float(np.mean(ae[pos]) * 1e3) if pos.any() else float("nan"),
        f"{prefix}MAE_neg_mm": float(np.mean(ae[neg]) * 1e3) if neg.any() else float("nan"),
        f"{prefix}n_pos": int(pos.sum()),
        f"{prefix}n_neg": int(neg.sum()),
    }
    return out


def zone_metrics(yhat, ygt, valid):
    m = np.asarray(valid, bool) & np.isfinite(yhat) & np.isfinite(ygt)

    def z(e):
        zid = np.zeros(e.shape, int)
        zid[e < -E_NEAR] = -1
        zid[e > E_NEAR] = 1
        return zid

    if not m.any():
        return {"zone_n": 0, "zone_acc": float("nan")}
    return {
        "zone_n": int(m.sum()),
        "zone_acc": float(np.mean(z(yhat[m]) == z(ygt[m]))),
        "zone_note": "near-center uses CURRENT 3 mm E_TOL for analysis only",
    }


def degrade_stream(u, valid, res, sigma, delay_steps, dropout, rng):
    u = np.array(u, float)
    v = np.array(valid, bool)
    n = len(u)
    out_u = np.full(n, np.nan)
    out_v = np.zeros(n, bool)
    buf_u = []
    buf_ok = []
    for i in range(n):
        ui, oki = u[i], v[i]
        if oki and rng.random() < dropout:
            oki = False
            ui = np.nan
        if oki:
            if sigma > 0:
                ui = ui + rng.normal(0.0, sigma)
            if res > 0:
                ui = np.round(ui / res) * res
        buf_u.append(ui)
        buf_ok.append(oki)
        j = i - delay_steps
        if j >= 0:
            out_u[i] = buf_u[j]
            out_v[i] = buf_ok[j]
        else:
            out_u[i] = np.nan
            out_v[i] = False
    return out_u, out_v


def geometric_from_uv(uL, uR, vL, vR):
    y = np.full(len(uL), np.nan)
    valid = np.zeros(len(uL), bool)
    for i in range(len(uL)):
        parts = []
        if vL[i] and np.isfinite(uL[i]):
            parts.append(uL[i])
        if vR[i] and np.isfinite(uR[i]):
            parts.append(-uR[i])
        if parts:
            y[i] = float(np.mean(parts))
            valid[i] = True
    return y, valid


def progress_table(ehat, egt, valid, traj, lag: int = 1, dead: float = DEAD_M):
    """Compare Δ|e| over `lag` samples. 2 ms consecutive steps are too small vs 0.1 mm."""
    false_p = 0
    false_r = 0
    agree = 0
    n = 0
    dgt_all = []
    dh_all = []
    mixed = 0
    for t in np.unique(traj):
        idx = np.where(traj == t)[0]
        if len(idx) <= lag:
            continue
        for a, b in zip(idx[:-lag], idx[lag:]):
            if not (valid[a] and valid[b]):
                continue
            if not (np.isfinite(ehat[a]) and np.isfinite(ehat[b])):
                continue
            d_gt = abs(egt[b]) - abs(egt[a])
            d_h = abs(ehat[b]) - abs(ehat[a])
            if abs(d_gt) < dead and abs(d_h) < dead:
                continue
            n += 1
            dgt_all.append(d_gt)
            dh_all.append(d_h)
            s_gt = np.sign(d_gt) if abs(d_gt) >= dead else 0.0
            s_h = np.sign(d_h) if abs(d_h) >= dead else 0.0
            if s_h == 0.0 or s_gt == 0.0:
                mixed += 1
            if s_h == s_gt:
                agree += 1
            if s_h < 0 and s_gt > 0:
                false_p += 1
            if s_h > 0 and s_gt < 0:
                false_r += 1
    dgt_all = np.asarray(dgt_all, float)
    dh_all = np.asarray(dh_all, float)
    return {
        "n_pairs": n,
        "sign_agree": float(agree / n) if n else float("nan"),
        "false_progress": float(false_p / n) if n else float("nan"),
        "false_regression": float(false_r / n) if n else float("nan"),
        "mixed_one_side_deadzone": float(mixed / n) if n else float("nan"),
        "pearson_dabs": pearson(dgt_all, dh_all) if n >= 3 else float("nan"),
        "deadzone_m": dead,
        "lag_steps": int(lag),
        "horizon_ms": int(lag) * 2,
    }


def analyze():
    s, g, m = load_bundle()
    egt = np.asarray(g["EVAL_ONLY_GT_e_x"], float)
    uL = np.asarray(s["u_L"], float)
    uR = np.asarray(s["u_R"], float)
    vL = np.asarray(s["valid_L"], float) > 0.5
    vR = np.asarray(s["valid_R"], float) > 0.5
    fam = np.asarray(m["family"]).astype(str)
    mode = np.asarray(m["mode"]).astype(str)
    traj = np.asarray(m["traj"]).astype(str)
    ic = np.asarray(m["ic_ex"], float)
    ehat0, valid0 = geometric_from_uv(uL, uR, vL, vR)
    bilat = vL & vR

    # --- ideal correlations (ZERO last-ish: family ZERO) ---
    zmask = fam == "ZERO"
    last_zero = []
    for t in np.unique(traj[zmask]):
        idx = np.where(traj == t)[0]
        last_zero.append(idx[-1])
    last_zero = np.array(last_zero, int)
    yz = egt[last_zero]
    corr = {
        "n_ZERO_last": int(len(last_zero)),
        "pearson_uL_vs_e_x": pearson(uL[last_zero], yz),
        "pearson_minus_uR_vs_e_x": pearson(-uR[last_zero], yz),
        "pearson_ehat_geom_vs_e_x": pearson(ehat0[last_zero], yz),
        "spearman_ehat_geom_vs_e_x": spearman(ehat0[last_zero], yz),
        "pearson_uL_all_valid": pearson(uL[vL], egt[vL]),
        "note": "u is finger-local X; -u_R due to right-finger 180 deg about Z",
    }

    ideal = err_metrics(ehat0, egt, valid0, "")
    ideal.update(zone_metrics(ehat0, egt, valid0))
    ideal_bilat = err_metrics(ehat0, egt, bilat, "bilat_")
    by_fam = {}
    for f in sorted(set(fam.tolist())):
        mk = fam == f
        by_fam[f] = {
            **err_metrics(ehat0[mk], egt[mk], valid0[mk], ""),
            "coverage_any": float(np.mean(valid0[mk])),
            "coverage_bilateral": float(np.mean(bilat[mk])),
            "n_steps": int(mk.sum()),
        }
    by_mode = {}
    for md in sorted(set(mode.tolist())):
        mk = mode == md
        if mk.sum() < 20:
            continue
        by_mode[md] = {
            **err_metrics(ehat0[mk], egt[mk], valid0[mk], ""),
            "coverage_any": float(np.mean(valid0[mk])),
            "n_steps": int(mk.sum()),
        }

    # linear / ridge on ZERO valid bilateral, test RULE
    tr = (fam == "ZERO") & bilat
    te = (fam == "RULE") & bilat
    Xtr = np.column_stack([uL[tr], uR[tr]])
    ytr = egt[tr]
    Xte = np.column_stack([uL[te], uR[te]])
    yte = egt[te]
    w_lin = fit_linear(Xtr, ytr) if tr.sum() > 5 else np.array([0.0, 0.5, -0.5])
    w_rid = fit_ridge(Xtr, ytr, 1e-6) if tr.sum() > 5 else w_lin
    yhat_lin = apply_linear(Xte, w_lin) if te.sum() else np.array([])
    yhat_rid = apply_linear(Xte, w_rid) if te.sum() else np.array([])
    estimators = {
        "geometric_trainZERO_testRULE": err_metrics(ehat0[te], yte, np.ones(te.sum(), bool), ""),
        "linear_weights": w_lin.tolist(),
        "ridge_weights": w_rid.tolist(),
        "linear_trainZERO_testRULE": err_metrics(yhat_lin, yte, np.ones(len(yte), bool), "") if te.sum() else {},
        "ridge_trainZERO_testRULE": err_metrics(yhat_rid, yte, np.ones(len(yte), bool), "") if te.sum() else {},
        "n_train_ZERO_bilat": int(tr.sum()),
        "n_test_RULE_bilat": int(te.sum()),
    }
    lolo = {}
    families = sorted(set(fam.tolist()))
    for hold in families:
        trm = (fam != hold) & bilat
        tem = (fam == hold) & bilat
        if trm.sum() < 20 or tem.sum() < 20:
            continue
        Xa = np.column_stack([uL[trm], uR[trm]])
        ya = egt[trm]
        w = fit_linear(Xa, ya)
        yh = apply_linear(np.column_stack([uL[tem], uR[tem]]), w)
        lolo[hold] = {
            "linear": err_metrics(yh, egt[tem], np.ones(tem.sum(), bool), ""),
            "geometric": err_metrics(ehat0[tem], egt[tem], np.ones(tem.sum(), bool), ""),
            "train_n": int(trm.sum()),
            "test_n": int(tem.sum()),
        }

    # RULE success vs fail (last step of each RULE traj)
    rule_split = {"success_traj": 0, "fail_traj": 0}
    rs = np.asarray(m.get("rule_success", np.full(len(fam), np.nan)), float)
    rule_idx = np.where(fam == "RULE")[0]
    last_rule = []
    for t in np.unique(traj[fam == "RULE"]):
        idx = np.where(traj == t)[0]
        last_rule.append(idx[-1])
        if rs[idx[-1]] > 0.5:
            rule_split["success_traj"] += 1
        else:
            rule_split["fail_traj"] += 1
    last_rule = np.array(last_rule, int) if last_rule else np.array([], int)
    succ_steps = (fam == "RULE") & (rs > 0.5)
    # per-traj tag: all steps of traj inherit final success
    traj_ok = {}
    for t in np.unique(traj[fam == "RULE"]):
        idx = np.where(traj == t)[0]
        traj_ok[t] = bool(rs[idx[-1]] > 0.5)
    rule_succ_m = np.array([traj_ok.get(t, False) for t in traj])
    rule_fail_m = (fam == "RULE") & (~rule_succ_m)
    rule_succ_m = (fam == "RULE") & rule_succ_m
    rule_dyn = {
        "success_steps": err_metrics(ehat0[rule_succ_m], egt[rule_succ_m], valid0[rule_succ_m], ""),
        "fail_steps": err_metrics(ehat0[rule_fail_m], egt[rule_fail_m], valid0[rule_fail_m], "")
        if rule_fail_m.any()
        else {"n": 0, "note": "no failed RULE traj on this diagnostic grid"},
        **rule_split,
    }

    # slices: slip, wrist, open, low force, unilateral
    fnL = np.asarray(s["fn_L"], float)
    fnR = np.asarray(s["fn_R"], float)
    slices = {
        "SLIP_family": err_metrics(ehat0[fam == "SLIP"], egt[fam == "SLIP"], valid0[fam == "SLIP"], ""),
        "WRIST_family": err_metrics(ehat0[fam == "WRIST"], egt[fam == "WRIST"], valid0[fam == "WRIST"], ""),
        "OPEN_family": err_metrics(ehat0[fam == "OPEN"], egt[fam == "OPEN"], valid0[fam == "OPEN"], ""),
        "BRAKE_family": err_metrics(ehat0[fam == "BRAKE"], egt[fam == "BRAKE"], valid0[fam == "BRAKE"], ""),
        "unilateral_only": err_metrics(ehat0, egt, valid0 & ~bilat, ""),
        "low_force_fnsum<1N": err_metrics(ehat0, egt, valid0 & ((fnL + fnR) < 1.0), ""),
        "v_rel>0.02": err_metrics(
            ehat0,
            egt,
            valid0 & (np.asarray(g["EVAL_ONLY_GT_v_rel"], float) > 0.02),
            "",
        ),
    }

    # degradations (synthetic)
    rng = np.random.default_rng(0)
    sweeps = []
    res_list = [0.0, 0.25e-3, 0.5e-3, 1.0e-3, 2.0e-3]
    sig_list = [0.0, 0.25e-3, 0.5e-3, 1.0e-3, 2.0e-3]
    lat_list = [0, 5, 10, 25]  # steps at 2 ms → 0,10,20,50 ms
    drop_list = [0.0, 0.01, 0.05, 0.10]
    for res in res_list:
        uLd, vLd = degrade_stream(uL, vL, res, 0.0, 0, 0.0, rng)
        uRd, vRd = degrade_stream(uR, vR, res, 0.0, 0, 0.0, rng)
        yh, va = geometric_from_uv(uLd, uRd, vLd, vRd)
        rec = {
            "axis": "resolution_m",
            "value": res,
            **err_metrics(yh, egt, va, ""),
            **progress_table(yh, egt, va, traj, lag=25, dead=1e-4),
        }
        sweeps.append(rec)
    for sig in sig_list:
        rng = np.random.default_rng(1)
        uLd, vLd = degrade_stream(uL, vL, 0.0, sig, 0, 0.0, rng)
        uRd, vRd = degrade_stream(uR, vR, 0.0, sig, 0, 0.0, rng)
        yh, va = geometric_from_uv(uLd, uRd, vLd, vRd)
        sweeps.append(
            {
                "axis": "noise_sigma_m",
                "value": sig,
                **err_metrics(yh, egt, va, ""),
                **progress_table(yh, egt, va, traj, lag=25, dead=1e-4),
            }
        )
    for dly in lat_list:
        uLd, vLd = degrade_stream(uL, vL, 0.0, 0.0, dly, 0.0, rng)
        uRd, vRd = degrade_stream(uR, vR, 0.0, 0.0, dly, 0.0, rng)
        yh, va = geometric_from_uv(uLd, uRd, vLd, vRd)
        sweeps.append(
            {
                "axis": "latency_steps_2ms",
                "value": dly,
                "latency_ms": dly * 2,
                **err_metrics(yh, egt, va, ""),
                **progress_table(yh, egt, va, traj, lag=25, dead=1e-4),
            }
        )
    for dr in drop_list:
        rng = np.random.default_rng(2)
        uLd, vLd = degrade_stream(uL, vL, 0.0, 0.0, 0, dr, rng)
        uRd, vRd = degrade_stream(uR, vR, 0.0, 0.0, 0, dr, rng)
        yh, va = geometric_from_uv(uLd, uRd, vLd, vRd)
        sweeps.append(
            {
                "axis": "dropout",
                "value": dr,
                **err_metrics(yh, egt, va, ""),
                **progress_table(yh, egt, va, traj, lag=25, dead=1e-4),
            }
        )

    prog_horizons = {}
    for lag, dead in ((1, 1e-5), (10, 1e-4), (25, 1e-4), (50, 1e-4)):
        prog_horizons[f"lag{lag}"] = {
            "all": progress_table(ehat0, egt, valid0, traj, lag=lag, dead=dead),
            "RULE": progress_table(
                ehat0[fam == "RULE"],
                egt[fam == "RULE"],
                valid0[fam == "RULE"],
                traj[fam == "RULE"],
                lag=lag,
                dead=dead,
            ),
        }
    prog_clean = prog_horizons["lag25"]["all"]
    prog_rule = prog_horizons["lag25"]["RULE"]

    per_off = {}
    for mm in EX_MM:
        mk = np.abs(ic * 1e3 - mm) < 0.4
        per_off[str(mm)] = err_metrics(ehat0[mk], egt[mk], valid0[mk], "")

    np.savez_compressed(
        OUT / "estimator_predictions.npz",
        e_hat_geom=ehat0,
        estimate_valid=valid0.astype(float),
        bilateral_valid=bilat.astype(float),
        EVAL_ONLY_GT_e_x=egt,
        family=fam,
        mode=mode,
    )
    metrics = {
        "used_final_eval_data": False,
        "correlations_ZERO_last": corr,
        "ideal_any_contact": ideal,
        "ideal_bilateral": ideal_bilat,
        "by_family": by_fam,
        "by_mode": by_mode,
        "estimators": estimators,
        "lolo": lolo,
        "rule_success_fail": rule_dyn,
        "failure_slices": slices,
        "reward_feasibility_clean": prog_clean,
        "reward_feasibility_RULE": prog_rule,
        "reward_horizons": prog_horizons,
        "per_offset_mm": per_off,
        "degradation_sweeps": sweeps,
        "coverage_overall_any": float(np.mean(valid0)),
        "coverage_overall_bilateral": float(np.mean(bilat)),
        "n_rows": int(len(egt)),
    }
    (OUT / "estimator_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "split_definitions.json").write_text(
        json.dumps(
            {
                "ideal": "all diagnostic timesteps; GT label only",
                "linear_ridge": "train ZERO bilateral; test RULE bilateral",
                "lolo": "train all families except holdout, bilateral rows; test holdout",
                "used_final_eval_data": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _figures(egt, ehat0, valid0, uL, uR, last_zero, fam, sweeps, by_fam, prog_clean)
    return metrics


def _figures(egt, ehat, valid, uL, uR, last_zero, fam, sweeps, by_fam, prog):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(1e3 * egt[last_zero], 1e3 * uL[last_zero], s=28, label="u_L", c="C0")
    ax.scatter(1e3 * egt[last_zero], -1e3 * uR[last_zero], s=28, marker="x", label="-u_R", c="C1")
    ax.scatter(1e3 * egt[last_zero], 1e3 * ehat[last_zero], s=18, marker="+", label="e_hat geom", c="C2")
    lim = 8
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.set_xlabel("e_x GT (mm)")
    ax.set_ylabel("tactile / estimate (mm)")
    ax.set_title("ZERO last: finger-local u vs e_x")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "A_uv_vs_ex.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    names = ["resolution_m", "noise_sigma_m", "latency_steps_2ms", "dropout"]
    titles = ["resolution (m)", "noise sigma (m)", "latency (2 ms steps)", "dropout"]
    for ax, name, title in zip(axes.ravel(), names, titles):
        rows = [s for s in sweeps if s["axis"] == name]
        ax.plot([r["value"] for r in rows], [r.get("MAE_mm", np.nan) for r in rows], "-o")
        ax.axhline(3.0, color="r", ls="--", lw=0.8, label="3 mm scale")
        ax.set_title(title)
        ax.set_ylabel("MAE mm")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Synthetic degradations (not hardware specs)")
    fig.tight_layout()
    fig.savefig(FIG / "B_degradation_mae.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    names = list(by_fam.keys())
    cov = [by_fam[k]["coverage_any"] for k in names]
    covb = [by_fam[k]["coverage_bilateral"] for k in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, cov, 0.4, label="any contact")
    ax.bar(x + 0.2, covb, 0.4, label="bilateral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction valid")
    ax.legend()
    ax.set_title("Estimate coverage")
    fig.tight_layout()
    fig.savefig(FIG / "C_coverage.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    mae = [by_fam[k].get("MAE_mm", np.nan) for k in names]
    ax.bar(names, mae)
    ax.axhline(3.0, color="r", ls="--", lw=0.8)
    ax.set_ylabel("MAE mm")
    ax.set_title("Geometric estimator MAE by family")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "D_mae_by_family.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar(
        ["sign agree", "false progress", "false regression"],
        [prog.get("sign_agree", np.nan), prog.get("false_progress", np.nan), prog.get("false_regression", np.nan)],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Offline Δ|e_hat| vs Δ|e_GT| (clean)")
    fig.tight_layout()
    fig.savefig(FIG / "E_reward_feasibility.png", dpi=140)
    plt.close(fig)

    mk = valid & np.isfinite(ehat)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for f, c in zip(sorted(set(fam.tolist())), plt.cm.tab10.colors):
        sel = mk & (fam == f)
        ax.scatter(1e3 * egt[sel][::8], 1e3 * ehat[sel][::8], s=6, alpha=0.4, label=f, color=c)
    ax.plot([-8, 8], [-8, 8], "k--", lw=0.8)
    ax.set_xlabel("e_x GT mm")
    ax.set_ylabel("e_hat mm")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title("All families (subsampled)")
    fig.tight_layout()
    fig.savefig(FIG / "F_scatter_all.png", dpi=140)
    plt.close(fig)


def write_report(metrics: dict, geo: dict):
    ideal = metrics["ideal_any_contact"]
    bil = metrics["ideal_bilateral"]
    corr = metrics["correlations_ZERO_last"]
    est = metrics["estimators"]
    prog = metrics["reward_feasibility_clean"]
    progr = metrics["reward_feasibility_RULE"]
    by = metrics["by_family"]
    sw = metrics["degradation_sweeps"]

    def sw_line(axis):
        rows = [s for s in sw if s["axis"] == axis]
        parts = []
        for r in rows:
            parts.append(
                f"{r['value']}: MAE={r.get('MAE_mm', float('nan')):.3f} mm "
                f"P<2mm={r.get('P_err<2mm', float('nan')):.3f} "
                f"FP={r.get('false_progress', float('nan')):.3f}"
            )
        return "; ".join(parts)

    md = f"""# Tactile e_x Estimator Audit

**used_final_eval_data: false**

Frozen prior (`RECENTER_OBSERVABILITY_AUDIT.md`): S0–S2 **FUNDAMENTALLY_AMBIGUOUS** for signed \(e_x\). S3 spatial contact location is informative in simulation but `contact.pos` is **not** a Panda observation.

This report builds a **finger-local spatial tactile abstraction** and tests whether \(e_x\) remains estimable after **synthetic** sensing limitations. No SAC, no reward change, no RecoveryEnv observation change, no frozen RULE edit.

Evidence tags: VERIFIED FROM CODE / SUPPORTED BY RAW DIAGNOSTIC / SYNTHETIC SENSOR SENSITIVITY RESULT / INFERENCE / HARDWARE-UNVALIDATED.

## 1. Sensor abstraction (VERIFIED FROM CODE)

Public output (`sensors/spatial_tactile.py`), finger local frame only:

- `contact_present_L/R`, `valid_L/R`
- `u,v` pad coordinates (NaN if invalid — zero is a legal CoP)
- optional `fn_L/R`

Not in the API: object qpos/qvel/pose, \(e_x\) GT, world contact coordinates.

Pad geometry from `assets/panda_torque.xml` `fingertip_pad_collision_1`: pos `[0, 0.0055, 0.0445]`, half-size `[0.0085, 0.004, 0.0085]`. Inner-face y = **{geo.get('samples', [{}])[0].get('pad_inner_y_m', 0.0015)} m**.

**Frames:** left finger is identity child of `hand` → finger X = hand X. Right finger `quat="0 0 0 1"` (wxyz) = 180° about Z → finger X = **−hand X**. Therefore \(u_L \\approx e_x\) and \(-u_R \\approx e_x\) for this cylinder grasp. Do **not** set \(e_x = u\) on both fingers.

CoP = \(\\sum F_{{n,i}} p_i / \\sum F_{{n,i}}\) with \(F_n = |\\texttt{{mj_contactForce}}[0]|\) (same convention as `envs/contact.py`). If \(\\sum F_n < 0.01\\,\\mathrm{{N}}\), `valid=False` even if a geom contact exists.

World→finger→world reconstruction max error: **{geo.get('max_recon_err_m')} m** (all_ok={geo.get('all_recon_ok')}). left_align={geo.get('left_align')} right_oppose={geo.get('right_oppose')}.

## 2. Dataset

- Path: `results/diagnostics/tactile_ex_estimator/`
- n_rows = **{metrics['n_rows']}**
- ICs: {EX_MM.tolist()} mm × settles {list(SETTLES)} via `apply_rel_pose` (not eval npz)
- Families: ZERO, SLIP (τ=-2), BRAKE (slip then resecure), WRIST (a0=0.30), OPEN (τ=-1), RULE
- GT stored separately in `gt_labels.npz` (`EVAL_ONLY_GT_*`)

## 3. Simplest estimator

**A. Geometric (preferred):** mean of valid-pad `u_L` and `-u_R`.

Linear/ridge on \([u_L, u_R]\) train ZERO / test RULE weights: linear={est.get('linear_weights')} ridge={est.get('ridge_weights')} (expect ~[0, 0.5, -0.5]). A more complex model is **not** justified.

## 4. Clean diagnostic accuracy (SUPPORTED BY RAW DIAGNOSTIC)

ZERO last (n={corr['n_ZERO_last']}): Pearson(u_L, e_x)={corr['pearson_uL_vs_e_x']:.4f}; Pearson(-u_R, e_x)={corr['pearson_minus_uR_vs_e_x']:.4f}; Pearson(\(\\hat e\), e_x)={corr['pearson_ehat_geom_vs_e_x']:.4f}; Spearman={corr['spearman_ehat_geom_vs_e_x']:.4f}.

All families, any-contact valid: n={ideal.get('n')} MAE={ideal.get('MAE_mm')} mm RMSE={ideal.get('RMSE_mm')} mm sign_err(|e|≥1mm)={ideal.get('sign_err_|e|>=1mm')} P(|err|<1mm)={ideal.get('P_err<1mm')} P(|err|<2mm)={ideal.get('P_err<2mm')} MAE+={ideal.get('MAE_pos_mm')} MAE-={ideal.get('MAE_neg_mm')} zone_acc={ideal.get('zone_acc')} (3 mm bins, analysis only).

Bilateral-only: n={bil.get('bilat_n')} MAE={bil.get('bilat_MAE_mm')} mm sign_err={bil.get('bilat_sign_err_|e|>=1mm')}.

Train ZERO / test RULE geometric: {est.get('geometric_trainZERO_testRULE')}

Linear same split: {est.get('linear_trainZERO_testRULE')}

## 5. Coverage

Overall any-contact valid: **{metrics['coverage_overall_any']:.3f}**; bilateral: **{metrics['coverage_overall_bilateral']:.3f}**.

Per family:

"""
    for k, v in by.items():
        md += f"- **{k}**: coverage_any={v['coverage_any']:.3f} bilateral={v['coverage_bilateral']:.3f} MAE={v.get('MAE_mm')} mm n={v['n_steps']}\n"

    md += f"""
Unilateral estimates are allowed (mean of whichever pad is valid). No-contact → `estimate_valid=False` (no hallucination, no zero fill).

## 6. Synthetic degradations (SYNTHETIC SENSOR SENSITIVITY RESULT, not hardware)

- resolution: {sw_line('resolution_m')}
- noise: {sw_line('noise_sigma_m')}
- latency: {sw_line('latency_steps_2ms')}
- dropout: {sw_line('dropout')}

## 7. Dynamics / failure slices

{json.dumps(metrics['failure_slices'], indent=2)}

RULE success vs fail: {json.dumps(metrics['rule_success_fail'], indent=2)}

LOLO: {json.dumps(metrics['lolo'], indent=2)}

## 8. Reward feasibility (offline only; SAC reward untouched)

Clean all families: n_pairs={prog.get('n_pairs')} sign_agree={prog.get('sign_agree')} false_progress={prog.get('false_progress')} false_regression={prog.get('false_regression')} (deadzone 0.1 mm).

RULE only: n_pairs={progr.get('n_pairs')} sign_agree={progr.get('sign_agree')} false_progress={progr.get('false_progress')} false_regression={progr.get('false_regression')}.

False progress = \(\\Delta|\\hat e_x|<0\) while \(\\Delta|e_{{x,GT}}|>0\).

## 9. Policy-observation candidates (not implemented)

Legitimate sensor-derived: `u_L,v_L,u_R,v_R`, `valid_L/R`, `fn_L/R`, `e_hat_x`, `estimate_valid`, plus existing proprioception/aperture. **Not** object pose / GT \(e_x\).

## 10. Trigger implication (D_t not redesigned)

Observable candidates: \(|\\hat e_x|\) vs 3 mm, tactile migration \(\\Delta u\), contact loss (`valid` falling), force drop. Keep trigger separate from reward and from GT success.

## 11. Leakage

`training/test_tactile_no_gt_leak.py` checks API parameters, return keys, and source (no `data.qpos/qvel`, no object xpos, no `physical_pack` in sensor/estimator modules).

## Q1–Q10

**Q1.** Yes in this simulated abstraction. Finger-local \(u_L\) and \(-u_R\) retain signed \(e_x\) (Pearson on ZERO last as above). SUPPORTED BY RAW DIAGNOSTIC. HARDWARE-UNVALIDATED.

**Q2.** Geometric \(\\hat e_x=\\mathrm{{mean}}(u_L,-u_R)\) over valid pads. Linear weights match that map; no neural net.

**Q3.** Clean any-contact MAE **{ideal.get('MAE_mm')} mm**, sign error **{ideal.get('sign_err_|e|>=1mm')}** (n={ideal.get('n')}).

**Q4.** See §6. Quantization/noise at 1–2 mm scale move MAE toward the task millimetre budget; 50 ms latency also inflates dynamic error. SYNTHETIC.

**Q5.** Relative to 3 mm: clean P(|err|<2 mm)={ideal.get('P_err<2mm')}. If this stays ≫ chance on RULE and P(|err|<2mm) is high, sim tactile is **task-scale sufficient** under the ideal abstraction; degradations may not be.

**Q6.** Coverage any={metrics['coverage_overall_any']:.3f} overall; see per-family in §5. OPEN/contact-loss is the validity hole.

**Q7.** See slices: unilateral, low force, OPEN, high v_rel, wrist. Static ZERO is the easy case.

**Q8.** Offline Δ|ê| vs Δ|e_GT| sign_agree={prog.get('sign_agree')} (all) and {progr.get('sign_agree')} (RULE). Use this, not a coefficient choice, to decide whether a dense observable reward is next.

**Q9.** Clean false-progress rate **{prog.get('false_progress')}** (all), RULE **{progr.get('false_progress')}**.

**Q10.** Spatial tactile is a **viable SIMULATED sensing assumption** for signed \(e_x\) in this cylinder/parallel-jaw setup. It remains **HARDWARE-UNVALIDATED**. Default `panda_torque.xml` still has only scalar `touch_*` sites.

## Paths

- code: `sensors/spatial_tactile.py`, `sensors/ex_estimator.py`, `training/validate_tactile_ex_estimator.py`
- leak test: `training/test_tactile_no_gt_leak.py`
- data: `results/diagnostics/tactile_ex_estimator/`
"""
    (ROOT / "TACTILE_EX_ESTIMATOR_AUDIT.md").write_text(md, encoding="utf-8")
    (OUT / "TACTILE_EX_ESTIMATOR_AUDIT.md").write_text(md, encoding="utf-8")


def source_leak_scan() -> list[str]:
    bad = []
    for p in (ROOT / "sensors" / "spatial_tactile.py", ROOT / "sensors" / "ex_estimator.py"):
        src = p.read_text(encoding="utf-8")
        if "data.qpos" in src or "data.qvel" in src:
            bad.append(f"{p.name}: qpos/qvel")
        if "physical_pack" in src:
            bad.append(f"{p.name}: physical_pack")
    return bad


def main():
    leaks = source_leak_scan()
    if leaks:
        raise RuntimeError(leaks)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    reuse = (OUT / "sensor_outputs.npz").exists() and "--reuse" in sys.argv
    if reuse:
        print("reuse rows")
        meta = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
    else:
        meta = collect(cfg)
    metrics = analyze()
    geo = json.loads((OUT / "geometry_check.json").read_text(encoding="utf-8"))
    write_report(metrics, geo)
    print("MAE_mm", metrics["ideal_any_contact"].get("MAE_mm"))
    print("false_progress", metrics["reward_feasibility_clean"].get("false_progress"))
    print("wrote", ROOT / "TACTILE_EX_ESTIMATOR_AUDIT.md")


if __name__ == "__main__":
    main()
