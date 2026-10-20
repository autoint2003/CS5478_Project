"""Diagnose axial slip during K49 tok48 reverse wrist rotation (hard_910263).

Legal tests use unchanged step_legal + tok48.
Counterfactuals (speed, hold, gravity, friction) are explicitly labeled.
No expert generation, SFT, or production physics edits.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action
from controllers.recovery.geometry_switch import rot_angle
from envs.contacts.frames import extract_contacts
from envs.outcomes.table_drop import object_vz, observe
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.grasp_quality import body_twist
from envs.simulation.object_dynamics import set_finger_object_sliding_mu
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.full_objective import P_TGT, U_TGT
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table, decode_action as decode_k49
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env

OUT = ROOT / "results" / "diagnostics" / "k49_return_axial_slip_mechanism"
PRIOR = ROOT / "results" / "expert_generation" / "return_aware_sliding_k49"
SNAP_IC = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
SEED = 20261018

CPP = [4] * 20 + [12] * 5 + [0] * 3
SLIDE = [17] * 16 + [12] * 5 + [0] * 5
RETIGHTEN = [0, 0, 0]
PREFIX = CPP + SLIDE + RETIGHTEN
N_RETURN = 19  # matches prior converge count


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def decode(table, book48, tok):
    tok = int(tok)
    if tok <= 46:
        return decode_k48(book48, tok, np.zeros(7, np.float32))[1]
    return decode_k49(table, tok, np.zeros(7, np.float32))[1]


def axial_decomp(sim):
    m = measure(sim, U_TGT, P_TGT)
    p = np.asarray(m["p_c_h"], float)
    u = np.asarray(m["u_c_h"], float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    ut = U_TGT / max(float(np.linalg.norm(U_TGT)), 1e-12)
    if float(np.dot(u, ut)) < 0:
        u = -u
    err = p - P_TGT
    e_ax = float(np.dot(err, u))
    e_lat = float(np.linalg.norm(err - e_ax * u))
    return {
        "e_ax_mm": e_ax * 1e3,
        "e_lat_mm": e_lat * 1e3,
        "e_pos_mm": m["e_pos_mm"],
        "p_c_h": p,
        "u_c_h": u,
        "u_c_w": np.asarray(m["u_c_w"], float),
    }


def contact_axial(sim, u_axis_w):
    rows, _ = extract_contacts(sim)
    u = np.asarray(u_axis_w, float)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    Fn_L = Ft_L = Fn_R = Ft_R = 0.0
    F_ax_L = F_ax_R = 0.0
    nL = nR = 0
    slip_ax = []
    pos_h = []
    for r in rows:
        fn = abs(float(r["Fn"]))
        ft = float(r["Ft"])
        f_obj_w = np.asarray(r["f_obj_w"], float)
        fax = float(np.dot(f_obj_w, u))
        if r["side"] == "L":
            Fn_L += fn
            Ft_L += ft
            F_ax_L += fax
            nL += 1
        else:
            Fn_R += fn
            Ft_R += ft
            F_ax_R += fax
            nR += 1
        pos_h.append(r["pos_h"])
        # approximate contact slip: relative vel of object vs hand at contact along axis
    return {
        "nL": nL, "nR": nR, "n_contacts": nL + nR,
        "Fn_L": Fn_L, "Ft_L": Ft_L, "Fn_R": Fn_R, "Ft_R": Ft_R,
        "F_ax_contact_L": F_ax_L, "F_ax_contact_R": F_ax_R,
        "F_ax_contact_total": F_ax_L + F_ax_R,
        "Ft_total": Ft_L + Ft_R, "Fn_total": Fn_L + Fn_R,
    }


def sample_substep(sim, blob, prev=None):
    """One physics-substep diagnostic sample. prev used for finite-diff accel."""
    dt = float(sim.model.opt.timestep)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    ax = axial_decomp(sim)
    u_w = np.asarray(ax["u_c_w"], float)
    u_w = u_w / max(float(np.linalg.norm(u_w)), 1e-12)
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g = np.asarray(sim.model.opt.gravity, float)
    F_g_ax = mass * float(np.dot(g, u_w))
    cax = contact_axial(sim, u_w)
    # relative axial displacement along instantaneous cylinder axis in hand frame
    # (matches prior e_ax metric)
    p_rel_h = Rh.T @ (po - ph)
    v_rel_w = vo - vh
    v_rel_h = Rh.T @ v_rel_w
    v_ax_h = float(np.dot(v_rel_h, ax["u_c_h"]))
    v_ax_w = float(np.dot(v_rel_w, u_w))
    # accelerations via finite difference when prev available
    if prev is not None:
        ah = (vh - np.asarray(prev["vh"], float)) / dt
        ao = (vo - np.asarray(prev["vo"], float)) / dt
        a_wh = (wh - np.asarray(prev["wh"], float)) / dt
    else:
        ah = ao = a_wh = np.zeros(3)
    a_ax_obj_w = float(np.dot(ao, u_w))
    a_ax_hand_w = float(np.dot(ah, u_w))
    # residual force balance along axis (world): m*a - (F_g + F_contact) 
    # (other forces: unmodeled if any)
    F_net_measured = cax["F_ax_contact_total"] + F_g_ax
    F_required = mass * a_ax_obj_w
    imbalance = F_required - F_net_measured
    Rref = np.asarray(blob["ref"]["r_des"], float).reshape(3, 3)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    # grip actuator
    tau_ctrl = float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan")
    tau_cmd = float(getattr(sim.fsm, "fg_cmd", np.nan))
    # finger joint forces if available
    try:
        qfrc_L = float(sim.data.qfrc_actuator[sim.ids.finger_dof[0]])
        qfrc_R = float(sim.data.qfrc_actuator[sim.ids.finger_dof[1]])
    except Exception:
        qfrc_L = qfrc_R = float("nan")
    return {
        "t": float(sim.data.time),
        "e_ax_mm": ax["e_ax_mm"],
        "e_lat_mm": ax["e_lat_mm"],
        "e_pos_mm": ax["e_pos_mm"],
        "p_c_h": ax["p_c_h"].tolist(),
        "u_c_h": ax["u_c_h"].tolist(),
        "u_c_w": u_w.tolist(),
        "ph": ph.tolist(),
        "po": po.tolist(),
        "vh": vh.tolist(),
        "vo": vo.tolist(),
        "wh": wh.tolist(),
        "wo": wo.tolist(),
        "ah": ah.tolist(),
        "ao": ao.tolist(),
        "a_wh": a_wh.tolist(),
        "v_ax_h": v_ax_h,
        "v_ax_w": v_ax_w,
        "a_ax_obj_w": a_ax_obj_w,
        "a_ax_hand_w": a_ax_hand_w,
        "mass": mass,
        "F_g_ax": F_g_ax,
        "F_ax_contact_total": cax["F_ax_contact_total"],
        "F_net_ax": F_net_measured,
        "F_ma_ax": F_required,
        "F_imbalance_ax": imbalance,
        "Fn_L": cax["Fn_L"], "Fn_R": cax["Fn_R"], "Fn_total": cax["Fn_total"],
        "Ft_L": cax["Ft_L"], "Ft_R": cax["Ft_R"], "Ft_total": cax["Ft_total"],
        "nL": cax["nL"], "nR": cax["nR"], "n_contacts": cax["n_contacts"],
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "p_des": np.asarray(sim.fsm.p_des, float).tolist(),
        "v_cmd": np.asarray(sim.fsm.v_cmd, float).tolist() if sim.fsm.v_cmd is not None else None,
        "w_cmd": np.asarray(sim.fsm.w_cmd, float).tolist() if sim.fsm.w_cmd is not None else None,
        "w_cmd_norm": float(np.linalg.norm(sim.fsm.w_cmd)) if sim.fsm.w_cmd is not None else None,
        "w_hand_norm": float(np.linalg.norm(wh)),
        "tau_ctrl": tau_ctrl,
        "tau_fg_cmd": tau_cmd,
        "qfrc_finger_L": qfrc_L,
        "qfrc_finger_R": qfrc_R,
        "gravity": g.tolist(),
    }


def step_instrumented(sim, action, blob, rows, label_prefix, scale_w=1.0):
    """Like step_legal but logs every mj_step. scale_w scales angular cmd (counterfactual)."""
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel = cmd["v_world"].copy()
    ang = cmd["w_world"].copy() * float(scale_w)
    tau = float(np.clip(float(cmd["tau"]), sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    prev = None
    for s in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        samp = sample_substep(sim, blob, prev)
        samp["label"] = f"{label_prefix}_s{s}"
        samp["scale_w"] = float(scale_w)
        rows.append(samp)
        prev = samp
    return action


def run_seq_legal(env, table, book48, tokens):
    for tok in tokens:
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)


def reach_prereturn(env, table, book48, blob_ic):
    restore_capture(env, blob_ic["capture"])
    run_seq_legal(env, table, book48, PREFIX)
    return take_capture(env)


def summarize_trace(rows, t0=None):
    if not rows:
        return {}
    e0 = rows[0]["e_ax_mm"]
    e1 = rows[-1]["e_ax_mm"]
    # onset: first sustained |v_ax| > thresh with continuing e_ax drift
    onset = None
    for r in rows:
        if abs(r["v_ax_h"]) > 0.005 and abs(r["e_ax_mm"] - e0) > 0.15:
            onset = r
            break
    wh_mean = float(np.mean([r["w_hand_norm"] for r in rows]))
    wh_max = float(np.max([r["w_hand_norm"] for r in rows]))
    fn0 = rows[0]["Fn_total"]
    fn_min = float(np.min([r["Fn_total"] for r in rows]))
    fn_mean = float(np.mean([r["Fn_total"] for r in rows]))
    fg_mean = float(np.mean([r["F_g_ax"] for r in rows]))
    # correlation proxies
    d_e = np.array([r["e_ax_mm"] - e0 for r in rows])
    v_ax = np.array([r["v_ax_h"] for r in rows])
    return {
        "n_samples": len(rows),
        "t0": rows[0]["t"],
        "t1": rows[-1]["t"],
        "dt": rows[-1]["t"] - rows[0]["t"],
        "e_ax0_mm": e0,
        "e_ax1_mm": e1,
        "d_e_ax_mm": e1 - e0,
        "e_lat0_mm": rows[0]["e_lat_mm"],
        "e_lat1_mm": rows[-1]["e_lat_mm"],
        "e_pos0_mm": rows[0]["e_pos_mm"],
        "e_pos1_mm": rows[-1]["e_pos_mm"],
        "hand0_deg": rows[0]["hand_to_ref_deg"],
        "hand1_deg": rows[-1]["hand_to_ref_deg"],
        "w_hand_mean": wh_mean,
        "w_hand_max": wh_max,
        "Fn0": fn0,
        "Fn_min": fn_min,
        "Fn_mean": fn_mean,
        "Fn_delta": fn_min - fn0,
        "F_g_ax_mean": fg_mean,
        "onset_t": None if onset is None else onset["t"],
        "onset_hand_deg": None if onset is None else onset["hand_to_ref_deg"],
        "onset_e_ax_mm": None if onset is None else onset["e_ax_mm"],
        "mean_imbalance": float(np.mean([r["F_imbalance_ax"] for r in rows])),
        "mean_abs_v_ax": float(np.mean(np.abs(v_ax))),
        "max_abs_v_ax": float(np.max(np.abs(v_ax))),
        "integrated_v_ax_mm": float(
            (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(v_ax, [r["t"] for r in rows]) * 1e3
        ) if len(rows) > 2 else None,
    }


def plot_traces(baselines: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "PLOT_ERROR.txt").write_text(str(exc), encoding="utf-8")
        return
    # multi-panel for baseline A
    rows = baselines.get("A_original") or []
    if not rows:
        return
    t0 = rows[0]["t"]
    t = np.array([r["t"] - t0 for r in rows])
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(t, [r["e_ax_mm"] for r in rows], label="e_ax")
    axes[0].set_ylabel("e_ax (mm)")
    axes[0].set_title("A original tok48 Return — substep traces")
    axes[1].plot(t, [r["v_ax_h"] for r in rows], label="v_ax_h")
    axes[1].set_ylabel("v_ax hand (m/s)")
    axes[2].plot(t, [r["F_g_ax"] for r in rows], label="m g·u")
    axes[2].plot(t, [r["F_ax_contact_total"] for r in rows], label="F_contact·u")
    axes[2].plot(t, [r["F_ma_ax"] for r in rows], label="m a·u", linestyle="--")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("axial force (N)")
    axes[3].plot(t, [r["Fn_total"] for r in rows], label="Fn")
    axes[3].plot(t, [r["Ft_total"] for r in rows], label="Ft")
    axes[3].legend(fontsize=8)
    axes[3].set_ylabel("contact (N)")
    axes[4].plot(t, [r["w_hand_norm"] for r in rows], label="|w_hand|")
    axes[4].plot(t, [r["w_cmd_norm"] or 0 for r in rows], label="|w_cmd|", linestyle="--")
    axes[4].legend(fontsize=8)
    axes[4].set_ylabel("ω (rad/s)")
    axes[4].set_xlabel("t − t0 (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "A_baseline_panels.png", dpi=140)
    plt.close(fig)

    # comparison of e_ax across conditions
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rs in baselines.items():
        if not rs:
            continue
        t0 = rs[0]["t"]
        ax.plot([r["t"] - t0 for r in rs], [r["e_ax_mm"] for r in rs], label=name)
    ax.set_xlabel("t − t0 (s)")
    ax.set_ylabel("e_ax (mm)")
    ax.set_title("Axial error across conditions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "e_ax_comparison.png", dpi=140)
    plt.close(fig)


def hold_at_state(sim, blob, rows, label, n_sub=50):
    """Quasi-static hold: p_des=hand, r_des=hand R, zero cmd. Counterfactual."""
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.r_des = rh.copy()
    vel = np.zeros(3)
    ang = np.zeros(3)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    # keep secure grip torque
    tau = float(np.clip(-18.0, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.fg_cmd = float(-tau)
    prev = None
    for s in range(n_sub):
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        samp = sample_substep(sim, blob, prev)
        samp["label"] = f"{label}_s{s}"
        samp["scale_w"] = 0.0
        rows.append(samp)
        prev = samp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-return", type=int, default=N_RETURN)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "plots", "SNAPSHOTS"):
        (OUT / sub).mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    blob_ic = pickle.loads(SNAP_IC.read_bytes())
    shutil.copy2(SNAP_IC, OUT / "SNAPSHOTS" / "hard_910263_ic.pkl")
    # also copy prior if present
    prior_snap = PRIOR / "SNAPSHOTS" / "hard_910263.pkl"
    if prior_snap.exists():
        shutil.copy2(prior_snap, OUT / "SNAPSHOTS" / "hard_910263_prior_copy.pkl")

    print("REACH_PRERETURN", flush=True)
    pre_cap = reach_prereturn(env, table, book48, blob_ic)
    # wrap as restoreable blob
    pre_blob = {"capture": pre_cap, "ref": blob_ic["ref"]}
    write_json(OUT / "RAW" / "prereturn_state.json", {
        **axial_decomp(env.sim),
        "hand_to_ref_deg": float(np.degrees(rot_angle(
            np.asarray(blob_ic["ref"]["r_des"], float).reshape(3, 3),
            np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float),
        ))),
        "t": float(env.sim.data.time),
        "note": "complete capture after CPP+SLIDE+RETIGHTEN; controller live",
    })
    # persist pre-return capture
    with (OUT / "SNAPSHOTS" / "hard_910263_prereturn.pkl").open("wb") as f:
        pickle.dump(pre_blob, f)

    action48 = decode(table, book48, SECURE_REVERSE_WRIST)
    print("tok48 action", action48, flush=True)

    traces = {}
    summaries = {}
    cmp_rows = []

    # ----- A: Original legal Return -----
    print("A_ORIGINAL", flush=True)
    restore_capture(env, pre_cap)
    rows_a = []
    # sample at t0
    rows_a.append({**sample_substep(env.sim, pre_blob), "label": "A_t0", "scale_w": 1.0})
    for i in range(args.n_return):
        step_instrumented(env.sim, action48, pre_blob, rows_a, f"A_tok{i}", scale_w=1.0)
    traces["A_original"] = rows_a
    summaries["A_original"] = summarize_trace(rows_a)
    write_csv(OUT / "RAW" / "A_original_substeps.csv", rows_a)
    print("  d_e_ax", summaries["A_original"]["d_e_ax_mm"], flush=True)

    # ----- B: Reduced angular speed (COUNTERFACTUAL) -----
    # K49 has no legal slower secure -wy; scale w_cmd via interface.
    print("B_SLOW_COUNTERFACTUAL", flush=True)
    for scale, n_steps in [(0.5, args.n_return * 2), (0.25, args.n_return * 4)]:
        restore_capture(env, pre_cap)
        rows = [{**sample_substep(env.sim, pre_blob), "label": f"B{scale}_t0", "scale_w": scale}]
        for i in range(n_steps):
            step_instrumented(env.sim, action48, pre_blob, rows, f"B{scale}_tok{i}", scale_w=scale)
            # stop when hand near same orientation as A end
            if rows[-1]["hand_to_ref_deg"] <= summaries["A_original"]["hand1_deg"] + 1.0:
                break
        key = f"B_slow_x{scale}"
        traces[key] = rows
        summaries[key] = summarize_trace(rows)
        write_csv(OUT / "RAW" / f"{key}_substeps.csv", rows)
        print(f"  {key} d_e_ax={summaries[key]['d_e_ax_mm']:.3f} hand1={summaries[key]['hand1_deg']:.2f}", flush=True)

    # matched angular displacement: integrate until same Δ hand angle as A
    print("B_MATCHED_ANGLE", flush=True)
    target_hand = summaries["A_original"]["hand1_deg"]
    for scale in (0.5, 0.25):
        restore_capture(env, pre_cap)
        rows = [{**sample_substep(env.sim, pre_blob), "label": f"Bm{scale}_t0", "scale_w": scale}]
        for i in range(args.n_return * 6):
            step_instrumented(env.sim, action48, pre_blob, rows, f"Bm{scale}_tok{i}", scale_w=scale)
            if rows[-1]["hand_to_ref_deg"] <= target_hand + 0.5:
                break
        key = f"B_matched_x{scale}"
        traces[key] = rows
        summaries[key] = summarize_trace(rows)
        write_csv(OUT / "RAW" / f"{key}_substeps.csv", rows)
        print(f"  {key} d_e_ax={summaries[key]['d_e_ax_mm']:.3f} n={len(rows)}", flush=True)

    # ----- C: Quasi-static holds (COUNTERFACTUAL) -----
    print("C_HOLDS", flush=True)
    # take orientations at 25/50/75% through A by restoring pre and replaying
    hold_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    hold_summaries = []
    for frac in hold_fracs:
        restore_capture(env, pre_cap)
        n_pre = int(round(frac * args.n_return))
        for i in range(n_pre):
            step_legal(env.sim, action48)
        # capture mid state
        mid_cap = take_capture(env)
        e_before = axial_decomp(env.sim)["e_ax_mm"]
        hand_before = float(np.degrees(rot_angle(
            np.asarray(blob_ic["ref"]["r_des"], float).reshape(3, 3),
            np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float),
        )))
        rows = []
        hold_at_state(env.sim, pre_blob, rows, f"C_hold_f{frac}", n_sub=50)  # 0.1s
        # split transient (first 10 substeps) vs sustained (last 20)
        e_early = rows[9]["e_ax_mm"] if len(rows) > 9 else rows[-1]["e_ax_mm"]
        e_late0 = rows[-20]["e_ax_mm"] if len(rows) >= 20 else rows[0]["e_ax_mm"]
        e_late1 = rows[-1]["e_ax_mm"]
        hs = {
            "frac": frac,
            "n_pre_tok48": n_pre,
            "hand_deg": hand_before,
            "e_ax_at_hold_start_mm": e_before,
            "e_ax_after_20ms_mm": e_early,
            "transient_d_e_ax_mm": e_early - e_before,
            "sustained_d_e_ax_mm": e_late1 - e_late0,
            "total_hold_d_e_ax_mm": e_late1 - e_before,
            "mean_abs_v_ax_late": float(np.mean(np.abs([r["v_ax_h"] for r in rows[-20:]]))),
            "label": "COUNTERFACTUAL_HOLD",
        }
        hold_summaries.append(hs)
        write_csv(OUT / "RAW" / f"C_hold_frac{frac}_substeps.csv", rows)
        print(f"  hold frac={frac} hand={hand_before:.1f} transient={hs['transient_d_e_ax_mm']:.3f} "
              f"sustained={hs['sustained_d_e_ax_mm']:.3f}", flush=True)
    write_json(OUT / "RAW" / "C_hold_summaries.json", hold_summaries)
    summaries["C_holds"] = hold_summaries

    # ----- D: Gravity counterfactual -----
    print("D_GRAVITY", flush=True)
    g0 = np.asarray(env.sim.model.opt.gravity, float).copy()
    for gscale, tag in [(0.0, "g0"), (0.5, "g0p5")]:
        restore_capture(env, pre_cap)
        env.sim.model.opt.gravity[:] = g0 * gscale
        mujoco.mj_forward(env.sim.model, env.sim.data)
        rows = [{**sample_substep(env.sim, pre_blob), "label": f"D{tag}_t0", "scale_w": 1.0}]
        for i in range(args.n_return):
            step_instrumented(env.sim, action48, pre_blob, rows, f"D{tag}_tok{i}", scale_w=1.0)
        key = f"D_gravity_{tag}"
        traces[key] = rows
        summaries[key] = summarize_trace(rows)
        write_csv(OUT / "RAW" / f"{key}_substeps.csv", rows)
        print(f"  {key} d_e_ax={summaries[key]['d_e_ax_mm']:.3f} hand_track_w={summaries[key]['w_hand_mean']:.3f}", flush=True)
    env.sim.model.opt.gravity[:] = g0  # restore

    # ----- E: Friction counterfactual -----
    print("E_FRICTION", flush=True)
    # read current mu
    mu0 = float(env.sim.model.geom_friction[env.sim.ids.object_geom][0])
    for mu, tag in [(mu0 * 2, "x2"), (mu0 * 5, "x5"), (1.5, "abs1p5")]:
        restore_capture(env, pre_cap)
        info = set_finger_object_sliding_mu(env.sim.model, env.sim.ids, mu, env.sim.data)
        rows = [{**sample_substep(env.sim, pre_blob), "label": f"E{tag}_t0", "scale_w": 1.0}]
        for i in range(args.n_return):
            step_instrumented(env.sim, action48, pre_blob, rows, f"E{tag}_tok{i}", scale_w=1.0)
        key = f"E_friction_{tag}"
        traces[key] = rows
        summaries[key] = summarize_trace(rows)
        summaries[key]["mu"] = mu
        summaries[key]["mu0"] = mu0
        write_csv(OUT / "RAW" / f"{key}_substeps.csv", rows)
        write_json(OUT / "RAW" / f"{key}_mu_info.json", info)
        print(f"  {key} mu={mu:.3f} d_e_ax={summaries[key]['d_e_ax_mm']:.3f} Fn_mean={summaries[key]['Fn_mean']:.2f}", flush=True)
    # restore friction
    set_finger_object_sliding_mu(env.sim.model, env.sim.ids, mu0, env.sim.data)

    # comparison table
    for name, s in summaries.items():
        if name == "C_holds":
            continue
        cmp_rows.append({
            "condition": name,
            "kind": "LEGAL" if name == "A_original" else "COUNTERFACTUAL",
            "d_e_ax_mm": s.get("d_e_ax_mm"),
            "d_e_pos_mm": (s.get("e_pos1_mm") or 0) - (s.get("e_pos0_mm") or 0),
            "hand0_deg": s.get("hand0_deg"),
            "hand1_deg": s.get("hand1_deg"),
            "w_hand_mean": s.get("w_hand_mean"),
            "Fn0": s.get("Fn0"),
            "Fn_min": s.get("Fn_min"),
            "Fn_mean": s.get("Fn_mean"),
            "F_g_ax_mean": s.get("F_g_ax_mean"),
            "onset_t": s.get("onset_t"),
            "onset_hand_deg": s.get("onset_hand_deg"),
            "mean_abs_v_ax": s.get("mean_abs_v_ax"),
            "integrated_v_ax_mm": s.get("integrated_v_ax_mm"),
            "dt": s.get("dt"),
            "mu": s.get("mu"),
        })
    write_csv(OUT / "COUNTERFACTUAL_COMPARISON.csv", cmp_rows)
    write_json(OUT / "RAW" / "summaries.json", summaries)

    if not args.skip_plots:
        plot_traces(traces, OUT)

    # reports
    _write_reports(summaries, cmp_rows, hold_summaries, mu0)

    print("DIAG_DONE", OUT, "d_e_ax_A", summaries["A_original"]["d_e_ax_mm"], flush=True)


def _write_reports(summaries, cmp_rows, hold_summaries, mu0):
    A = summaries["A_original"]
    prior_target = -4.84
    repro_ok = abs(A["d_e_ax_mm"] - prior_target) < 0.6  # tolerance

    # pick gravity / friction / slow effects
    Dg0 = summaries.get("D_gravity_g0", {})
    Dg05 = summaries.get("D_gravity_g0p5", {})
    Bm05 = summaries.get("B_matched_x0.5", {})
    Bm025 = summaries.get("B_matched_x0.25", {})
    Ex2 = summaries.get("E_friction_x2", {})
    Ex5 = summaries.get("E_friction_x5", {})

    # hold: any sustained slip?
    sust = [h["sustained_d_e_ax_mm"] for h in hold_summaries]
    max_sust = max(abs(x) for x in sust) if sust else None

    (OUT / "BASELINE_REPRODUCTION.md").write_text(f"""# Baseline Reproduction

## Setup

IC: `hard_910263` from hard_ic_mining.
Prefix (legal K49): CPP `[4]×20+[12]×5+[0]×3` + Sliding `[17]×16+[12]×5+[0]×5` + `[0]×3`.
Return: **token 48** × {N_RETURN} via instrumented `step_legal` (N_SUB=10).

Pre-return snapshot saved: `SNAPSHOTS/hard_910263_prereturn.pkl` (full capture: qpos/qvel/ctrl/FSM).

## Prior reference

From `return_aware_sliding_k49`: e_ax −7.11 → −11.94 mm (**Δ −4.84 mm**).

## This run

| Metric | Value |
|---|---:|
| e_ax0 (mm) | {A['e_ax0_mm']:.4f} |
| e_ax1 (mm) | {A['e_ax1_mm']:.4f} |
| **Δ e_ax (mm)** | **{A['d_e_ax_mm']:.4f}** |
| e_lat0 → e_lat1 | {A['e_lat0_mm']:.3f} → {A['e_lat1_mm']:.3f} |
| hand↔ref | {A['hand0_deg']:.2f}° → {A['hand1_deg']:.2f}° |
| duration (s) | {A['dt']:.4f} |

Reproduction vs −4.84 mm: **{"YES" if repro_ok else "PARTIAL"}** (|Δ−(−4.84)|={abs(A['d_e_ax_mm']-prior_target):.3f}).

Raw: `RAW/A_original_substeps.csv`.
""", encoding="utf-8")

    (OUT / "FORCE_BALANCE.md").write_text(f"""# Axial Force Balance (world frame along cylinder axis)

## Formulation (measured)

Along instantaneous cylinder axis `u` (world):

- `F_g_ax = m * dot(g, u)` — **measured** from model gravity and axis.
- `F_contact_ax = Σ f_obj·u` — **measured** via `mj_contactForce` → world (`extract_contacts`).
- `a_ax = dot(a_com, u)` — **inferred** by finite-differencing object COM velocity across MuJoCo substeps (`model.opt.timestep`).
- Residual: `imbalance = m*a_ax − (F_g_ax + F_contact_ax)`.

Hand-frame rotating terms (Euler/Coriolis) are **not** claimed as a closed quantitative balance here; primary analysis is world-axis projection. Relative axial slip `e_ax`, `v_ax_h` are reported in the hand-aligned metric used by prior pilots.

## Baseline (A) averages

| Quantity | Mean / note |
|---|---|
| F_g_ax | {A.get('F_g_ax_mean'):.4f} N |
| Fn_total (start → min) | {A.get('Fn0'):.3f} → {A.get('Fn_min'):.3f} N |
| mean |imbalance| | {A.get('mean_imbalance'):.4f} N |
| mean |v_ax_h| | {A.get('mean_abs_v_ax'):.5f} m/s |
| ∫ v_ax dt (mm) | {A.get('integrated_v_ax_mm')} |

Imbalance ≠ 0 is expected: finite-diff accel noise, contact force sampling, and unaccounted constraints (tendon/joint) mean this is a **consistency check**, not a proof of a single missing force.

## What the counterfactuals imply

- **Gravity off (D g=0):** Δe_ax = {Dg0.get('d_e_ax_mm')} mm (vs A {A['d_e_ax_mm']:.3f}). If slip collapses, gravity loading is necessary; if residual remains, rotation/contact dynamics contribute.
- **Half gravity:** Δe_ax = {Dg05.get('d_e_ax_mm')} mm.
- **Matched-angle slow ×0.5:** Δe_ax = {Bm05.get('d_e_ax_mm')} mm at mean |w|={Bm05.get('w_hand_mean')}.
- **Friction ×2 / ×5:** Δe_ax = {Ex2.get('d_e_ax_mm')} / {Ex5.get('d_e_ax_mm')} mm (μ0={mu0}).

See `COUNTERFACTUAL_COMPARISON.csv` and plots.
""", encoding="utf-8")

    # mechanism verdict
    g_frac = None
    if Dg0.get("d_e_ax_mm") is not None and abs(A["d_e_ax_mm"]) > 1e-6:
        g_frac = 1.0 - abs(Dg0["d_e_ax_mm"]) / abs(A["d_e_ax_mm"])

    slow_helps = (Bm05.get("d_e_ax_mm") is not None and abs(Bm05["d_e_ax_mm"]) < abs(A["d_e_ax_mm"]) - 0.3)
    fric_helps = (Ex5.get("d_e_ax_mm") is not None and abs(Ex5["d_e_ax_mm"]) < abs(A["d_e_ax_mm"]) - 0.3)
    hold_sustained = max_sust is not None and max_sust > 0.3

    if g_frac is not None and g_frac > 0.6 and not slow_helps:
        mech = "Primarily gravity-driven axial loading under secure contact during reorientation"
    elif g_frac is not None and g_frac > 0.35 and slow_helps:
        mech = "Combination: gravity axial load + rate-dependent / inertial contribution during rotation"
    elif slow_helps and (g_frac is None or g_frac < 0.35):
        mech = "Primarily rotation-rate / inertial; gravity secondary"
    else:
        mech = "Mixed / friction-limited contact sliding under changing orientation (see numbers)"

    next_legal = (
        "Interleave short secure stabilize `[0]` or mid `[12]` **without** long unload bursts; "
        "or partial return to an orientation where |F_g_ax| is smaller before HANDOFF; "
        "or slide axially **after** partial return when |g·u| projection is reduced. "
        "Do not add tokens unless a legal slower −wy is proven necessary after these fail."
    )
    if slow_helps:
        next_legal = (
            "Priority legal test: approximate slower return by interleaving tok48 with secure zero-motion `[0]` "
            "(same total angle, lower mean |w|), then re-measure Δe_ax. "
            "If still insufficient, consider a documented K50 lower-|wy| mirror — only after this legal test."
        )

    (OUT / "REPORT.md").write_text(f"""# K49 Return Axial Slip Mechanism — Diagnosis

**Legal baseline:** unchanged K49 tok48 via `step_legal`.  
**Counterfactuals** (speed scale, holds, gravity, friction) are labeled and do **not** modify production physics or the vocabulary.

```text
python -u scripts/diagnose_k49_return_axial_slip.py --device cuda
```

## Q1 — Reproduce −4.84 mm?

**{"Yes" if repro_ok else "Close / partial"}** — measured Δe_ax = **{A['d_e_ax_mm']:.3f} mm** (prior −4.84).  
See `BASELINE_REPRODUCTION.md`.

## Q2 — When does sliding begin?

Onset (first |v_ax_h|>5 mm/s with >0.15 mm drift): t={A.get('onset_t')}, hand↔ref≈{A.get('onset_hand_deg')}°, e_ax≈{A.get('onset_e_ax_mm')} mm.  
Full series: `RAW/A_original_substeps.csv`, `plots/A_baseline_panels.png`.

## Q3 — Gravity vs inertia vs combination?

**{mech}**

| Condition | Δe_ax (mm) | mean |w_hand| | notes |
|---|---:|---:|---|
| A original (legal) | {A['d_e_ax_mm']:.3f} | {A.get('w_hand_mean'):.3f} | |
| D gravity=0 (CF) | {Dg0.get('d_e_ax_mm')} | {Dg0.get('w_hand_mean')} | physics CF |
| D gravity×0.5 (CF) | {Dg05.get('d_e_ax_mm')} | {Dg05.get('w_hand_mean')} | |
| B matched ×0.5 (CF) | {Bm05.get('d_e_ax_mm')} | {Bm05.get('w_hand_mean')} | same angle, slower |
| B matched ×0.25 (CF) | {Bm025.get('d_e_ax_mm')} | {Bm025.get('w_hand_mean')} | |
| E friction×2 (CF) | {Ex2.get('d_e_ax_mm')} | {Ex2.get('w_hand_mean')} | μ0={mu0} |
| E friction×5 (CF) | {Ex5.get('d_e_ax_mm')} | {Ex5.get('w_hand_mean')} | |

Gravity-removal fraction of |Δe_ax| (rough): **{None if g_frac is None else f"{100*g_frac:.0f}%"}**.

## Q4 — Does slowing reduce slip?

{"**Yes** (counterfactual w-scale)." if slow_helps else "**Weak / no** clear reduction at matched angle under tested scales."}  
K49 has **no legal** reduced-|wy| secure reverse; slow tests scaled `w_cmd` only.

## Q5 — Slip during quasi-static holds?

Max |sustained| Δe_ax over last 40 ms of 0.1 s holds: **{max_sust} mm**.  
{"Sustained creep present at some orientations." if hold_sustained else "Mostly transient settling; little sustained quasi-static creep."}  
Details: `RAW/C_hold_summaries.json`.

## Q6 — Does more friction suppress slip?

{"**Partially / yes** in CF." if fric_helps else "**No strong suppression** at tested μ multipliers."}  
Diagnostic only — production μ unchanged.

## Q7 — Does finger normal force decrease?

A: Fn_total {A.get('Fn0'):.2f} → min {A.get('Fn_min'):.2f} (Δ {A.get('Fn_delta'):.2f} N).  
{"Yes, measurable drop." if (A.get('Fn_delta') or 0) < -0.5 else "No large sustained drop (or slight change only)."}

## Q8 — Supported mechanism

**{mech}**

Force/motion evidence: `FORCE_BALANCE.md`, `plots/`, `COUNTERFACTUAL_COMPARISON.csv`.

## Q9 — Smallest justified next **legal** K49 test

{next_legal}

## Stop

Diagnosis complete. No expert dataset, no SFT/GRPO, no vocab/physics production changes.
""", encoding="utf-8")

    write_json(OUT / "summary.json", {
        "A_d_e_ax_mm": A["d_e_ax_mm"],
        "reproduces_prior": repro_ok,
        "mechanism": mech,
        "gravity_fraction_rough": g_frac,
        "slow_helps": slow_helps,
        "friction_helps": fric_helps,
        "hold_sustained": hold_sustained,
        "next_legal": next_legal,
    })


if __name__ == "__main__":
    main()
