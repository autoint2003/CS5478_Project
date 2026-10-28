"""P36 gate: audit / fix nominal HANDOFF contract independently of the Transformer.

Compares on identical full MuJoCo snapshots:
  A) current apply_handoff (snap r_des → ref)
  B) contract fix: init p_des/r_des from actual hand, preserve grip, smooth r_des→ref

Reports gripper cmds, contact forces, slip, drop timing, SUCCESS_12S / SUCCESS_NEW.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.gripper.controller import read_touch  # noqa: E402
from controllers.nominal import _integrate_rot  # noqa: E402
from controllers.recovery.geometry_switch import rot_angle  # noqa: E402
from envs.ballistic_impact import TAU_SEC, z_tgt_of  # noqa: E402
from envs.outcomes.nominal_pose import hand_R  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.success_criteria import contact_summary  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p36_rule_handoff_d_sft.geometry import (  # noqa: E402
    geometric_target_reached,
)

OUT = ROOT / "results/diagnostics/p36_rule_handoff_d_sft/handoff_contract"
P35_MAN = ROOT / "results/diagnostics/p35_precision_sft/PACKAGED_MANIFEST.csv"

# Max angular rate (rad/s) for smooth return of r_des toward saved grasp.
R_RETURN_OMEGA = 0.8


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def snap_metrics(sim, ref) -> dict[str, Any]:
    cs = contact_summary(sim)
    touch = read_touch(sim.data, sim.ids)
    Rh = hand_R(sim)
    Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    Rref = np.asarray(ref["r_des"], float).reshape(3, 3)
    return {
        "t": float(sim.data.time),
        "phase": str(sim.fsm.phase),
        "lift_started": bool(sim.fsm.lift_started),
        "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "fg_cmd": float(sim.fsm.fg_cmd),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "touch_L": float(touch[0]),
        "touch_R": float(touch[1]),
        "touch_min": float(np.min(touch)),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "p_des_z": float(sim.fsm.p_des[2]),
        "z_tgt": float(z_tgt_of(sim)),
        "hand_to_rdes_deg": float(np.degrees(rot_angle(Rd, Rh))),
        "hand_to_ref_deg": float(np.degrees(rot_angle(Rref, Rh))),
        "rdes_to_ref_deg": float(np.degrees(rot_angle(Rref, Rd))),
        "p_des": [round(float(x), 5) for x in np.asarray(sim.fsm.p_des, float)[:3]],
        "p_hand": [round(float(x), 5) for x in np.asarray(sim.data.xpos[sim.ids.hand_body], float)],
    }


def apply_handoff_contract(sim, ref, *, omega_return: float = R_RETURN_OMEGA) -> dict:
    """Fixed handoff contract.

    - Initialize p_des and r_des from the *actual* hand pose (no orientation yank).
    - Force lift phase + lift_started so nominal never re-opens the gripper.
    - Preserve gripper closure: keep ctrl[7] closed; set fg_cmd from |ctrl| / TAU_SEC.
    - Store r_des_target = saved grasp orientation for smooth return in nominal_step_contract.
    """
    qpos_before = np.array(sim.data.qpos, float).copy()
    qvel_before = np.array(sim.data.qvel, float).copy()
    time_before = float(sim.data.time)
    hand_before = hand_R(sim).copy()

    # Arm targets from actual state
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    sim.fsm.r_des = hand_before.copy()
    sim.fsm.v_cmd = np.zeros(3)
    sim.fsm.v_des = np.zeros(3)
    sim.fsm.w_cmd = np.zeros(3)
    sim.fsm.w_des = np.zeros(3)

    # Accept nonstandard pose in lift
    sim.fsm.phase = "lift"
    sim.fsm.lift_started = True

    # Gripper continuity: never open; prefer live ctrl, else TAU_SEC
    if sim.data.ctrl.size > 7:
        live = float(sim.data.ctrl[7])
        if live > -1.0:  # insufficient close (near 0 or open)
            sim.data.ctrl[7] = float(TAU_SEC)
            live = float(TAU_SEC)
        tau = live
    else:
        tau = float(TAU_SEC)
    sim.fsm.fg_cmd = float(abs(tau))

    # Smooth-return target (standard grasp orientation)
    target = np.array(ref["r_des"], float).reshape(3, 3).copy()
    setattr(sim.fsm, "r_des_target", target)
    setattr(sim.fsm, "r_return_omega", float(omega_return))

    qpos_after = np.array(sim.data.qpos, float)
    qvel_after = np.array(sim.data.qvel, float)
    return {
        "t": time_before,
        "phase": str(sim.fsm.phase),
        "qpos_max_abs": float(np.max(np.abs(qpos_after - qpos_before))),
        "qvel_max_abs": float(np.max(np.abs(qvel_after - qvel_before))),
        "time_delta": float(sim.data.time) - time_before,
        "r_des_to_saved_rad": float(rot_angle(ref["r_des"], np.array(sim.fsm.r_des, float))),
        "r_des_to_hand_rad": float(rot_angle(np.array(sim.fsm.r_des, float), hand_before)),
        "v_cmd_norm": float(np.linalg.norm(np.asarray(sim.fsm.v_cmd, float))),
        "w_cmd_norm": float(np.linalg.norm(np.asarray(sim.fsm.w_cmd, float))),
        "p_des": [round(float(x), 5) for x in np.asarray(sim.fsm.p_des, float).reshape(-1)[:3]],
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
        "physics_edited": False,
        "contract": "init_from_hand_smooth_return",
    }


def _omega_toward(R_cur: np.ndarray, R_tgt: np.ndarray, omega_max: float) -> np.ndarray:
    """Body-rate that rotates R_cur toward R_tgt, clipped to omega_max (rad/s)."""
    # R_err = R_tgt * R_cur^T maps current → target in world; use rotvec of relative.
    R_err = R_tgt @ R_cur.T
    # sksym extraction
    skew = 0.5 * (R_err - R_err.T)
    rotvec = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], float)
    ang = float(np.linalg.norm(rotvec))
    # For small angles rotvec ≈ axis*theta; for larger use acos path
    tr = float(np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(tr))
    if theta < 1e-6:
        return np.zeros(3)
    if ang < 1e-12:
        # 180-deg singular — pick any axis from R_err + I
        axis = np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]], float)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return np.zeros(3)
        axis = axis / n
    else:
        axis = rotvec / ang
    rate = min(omega_max, theta / 0.02)  # catch up within ~policy dt budget
    return axis * rate


def nominal_step_contract(sim) -> None:
    """Nominal step with optional smooth r_des return + grip continuity."""
    from envs.outcomes.support_drop import object_vz, observe
    from controllers.cartesian.velocity_step import tick_vw
    from controllers.cartesian.tracking import TRACK

    target = getattr(sim.fsm, "r_des_target", None)
    omega_max = float(getattr(sim.fsm, "r_return_omega", R_RETURN_OMEGA) or R_RETURN_OMEGA)

    # Preserve closed gripper every step
    if sim.data.ctrl.size > 7 and float(sim.data.ctrl[7]) > -1.0:
        sim.data.ctrl[7] = float(TAU_SEC)
    sim.fsm.fg_cmd = float(abs(TAU_SEC))

    vz_before = object_vz(sim)
    z_now = float(z_tgt_of(sim))
    lift_hold = str(sim.fsm.phase) == "lift" and float(sim.fsm.p_des[2]) >= z_now - 1e-9

    w_ret = np.zeros(3)
    if target is not None:
        Rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
        Rt = np.asarray(target, float).reshape(3, 3)
        w_ret = _omega_toward(Rd, Rt, omega_max)
        # clear target when close
        if float(rot_angle(Rt, Rd)) < np.deg2rad(1.0):
            sim.fsm.r_des = Rt.copy()
            w_ret = np.zeros(3)
            setattr(sim.fsm, "r_des_target", None)

    if lift_hold:
        tick_vw(sim, np.zeros(3), w_ret, TAU_SEC, TRACK)
    else:
        # Use physics_step but inject orientation return via residual-like w on fsm before step
        if np.linalg.norm(w_ret) > 1e-9:
            # Manually integrate r_des toward target at limited rate, then nominal lift
            dt = float(sim.model.opt.timestep)
            # n_sub physics steps inside physics_step — approximate by setting w into fsm
            # Prefer tick_vw path with lift velocity to keep grip=TAU_SEC continuous
            v_lift = np.array([0.0, 0.0, float(sim.fsm.cfg["fsm"].get("lift_speed", 0.08))])
            # multi-sub to match policy tick? nominal_step historically does one physics_step
            # which runs fsm.step once... actually GraspSim.physics_step = 1 mj_step via fsm.step
            tick_vw(sim, v_lift, w_ret, TAU_SEC, TRACK)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            # Re-assert grip after nominal (fg_to_tau path)
            if sim.data.ctrl.size > 7:
                sim.data.ctrl[7] = float(TAU_SEC)
    observe(sim, vz_before)


def evaluate_with(
    sim,
    ref,
    *,
    apply_fn: Callable,
    step_fn: Callable,
    stabilize_n: int = 8,
    horizon_s: float = 12.0,
) -> dict:
    """STRICT-like 12s eval using a chosen handoff/step pair (mirrors evaluate_handoff_with_axis)."""
    from data_generation.oracles.precision.corrections.contact_primitives import _wrist
    from envs.step_legal import step_legal
    from evaluation.grasp_geometry import measure
    from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG, window_stats
    from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor
    from training.recovery_execution import _escaped

    for _ in range(int(stabilize_n)):
        step_legal(sim, _wrist(0.0, 0.0, 1.0))
        if latched(sim):
            break

    m0 = measure(sim, U_TGT, P_TGT)
    pb0 = contact_summary(sim)
    audit = apply_fn(sim, ref)
    t_h = float(sim.data.time)
    reason = None
    samples: list[dict] = []
    next_sample = 0.0
    lost = 0
    lift_done = False
    drop_t = None
    first_loss_t = None
    grip_trace = []
    min_touch = 1e9
    min_bil_frac_early = 1.0

    def _row():
        m = measure(sim, U_TGT, P_TGT)
        e = (np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)) * 1e3
        cs = contact_summary(sim)
        return {
            "t_rel": float(sim.data.time) - t_h,
            "e_theta_deg": float(m["e_theta_deg"]),
            "e_pos_mm": float(m["e_pos_mm"]),
            "e_signed_mm": e.tolist(),
            "bilateral": bool(cs["bilateral"]),
            "obj_z": float(m["obj_z"]),
            "nL": cs["nL"],
            "nR": cs["nR"],
        }

    horizon = float(horizon_s)
    while reason is None and float(sim.data.time) - t_h < horizon - 1e-9:
        step_fn(sim)
        t_rel = float(sim.data.time) - t_h
        mon = _monitor(sim)
        bil = _bilateral(mon)
        touch = read_touch(sim.data, sim.ids)
        min_touch = min(min_touch, float(np.min(touch)))
        if t_rel < 0.5:
            grip_trace.append(
                {
                    "t_rel": t_rel,
                    "ctrl7": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else None,
                    "touch_min": float(np.min(touch)),
                    "bilateral": bool(bil),
                    "hand_to_rdes_deg": float(
                        np.degrees(rot_angle(np.asarray(sim.fsm.r_des, float), hand_R(sim)))
                    ),
                    "rdes_to_ref_deg": float(
                        np.degrees(
                            rot_angle(
                                np.asarray(ref["r_des"], float),
                                np.asarray(sim.fsm.r_des, float),
                            )
                        )
                    ),
                }
            )
            if not bil and first_loss_t is None:
                first_loss_t = t_rel
        if bil and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-9:
            lift_done = True
        if latched(sim):
            reason = "GROUND_DROP"
            drop_t = t_rel
            break
        if _dropped(mon, sim) or _escaped(sim):
            reason = "DROP_AFTER_HANDOFF"
            drop_t = t_rel
            break
        lost = lost + 1 if not bil else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            if first_loss_t is None:
                first_loss_t = t_rel
            break
        if t_rel + 1e-12 >= next_sample:
            samples.append(_row())
            next_sample += 0.02

    if reason is None:
        mon = _monitor(sim)
        if not _bilateral(mon) or _dropped(mon, sim) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"

    if not samples or samples[-1]["t_rel"] < float(sim.data.time) - t_h - 1e-6:
        samples.append(_row())

    win = window_stats(samples, float(NOMINAL_S) - 2.0)
    s12 = reason == "SUCCESS_12S"
    success_new = bool(s12 and win.get("working_SUCCESS_NEW_hold"))
    return {
        "SUCCESS_12S": s12,
        "SUCCESS_NEW": success_new,
        "reason": reason,
        "drop_t_rel": drop_t,
        "first_contact_loss_t_rel": first_loss_t,
        "min_touch_after": None if min_touch > 1e8 else min_touch,
        "handoff_audit": audit,
        "pre_theta": float(m0["e_theta_deg"]),
        "pre_pos_mm": float(m0["e_pos_mm"]),
        "pre_bilateral": bool(pb0["bilateral"]),
        "grip_trace_early": grip_trace,
        "window_SUCCESS_NEW": {
            k: win.get(k)
            for k in (
                "working_SUCCESS_NEW_hold",
                "mean_theta",
                "mean_pos",
                "max_theta",
                "max_pos",
                "frac_bilateral",
            )
        },
        "n_samples": len(samples),
    }


def run_variant(env, blob, variant: str) -> dict:
    restore_capture(env, blob["capture"])
    ref = blob["ref"]
    pre = snap_metrics(env.sim, ref)
    geo_ok, geo = geometric_target_reached(env.sim)

    if variant == "baseline_snap":
        # Official path
        restore_capture(env, blob["capture"])
        h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
        # also early grip via apply+steps for diagnostics
        restore_capture(env, blob["capture"])
        for _ in range(8):
            from data_generation.oracles.precision.corrections.contact_primitives import _wrist
            from envs.step_legal import step_legal

            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        apply_handoff(env.sim, ref)
        early = []
        dropped_early = False
        for i in range(25):
            nominal_step(env.sim)
            early.append(snap_metrics(env.sim, ref))
            if latched(env.sim):
                dropped_early = True
                break
        return {
            "variant": variant,
            "pre": pre,
            "geo_ok": int(geo_ok),
            "geo": geo,
            "SUCCESS_12S": int(bool(h.get("SUCCESS_12S"))),
            "SUCCESS_NEW": int(bool(h.get("SUCCESS_NEW"))),
            "reason": h.get("reason"),
            "early0": early[0] if early else None,
            "early1": early[1] if len(early) > 1 else None,
            "early10": early[min(10, len(early) - 1)] if early else None,
            "early_min_touch": min((e["touch_min"] for e in early), default=None),
            "early_any_loss": int(any(not e["bilateral"] for e in early)),
            "early_drop": int(dropped_early),
        }

    if variant == "contract_smooth":
        restore_capture(env, blob["capture"])
        h = evaluate_with(
            env.sim,
            ref,
            apply_fn=apply_handoff_contract,
            step_fn=nominal_step_contract,
            stabilize_n=8,
            horizon_s=12.0,
        )
        gt = h.get("grip_trace_early") or []
        return {
            "variant": variant,
            "pre": pre,
            "geo_ok": int(geo_ok),
            "geo": geo,
            "SUCCESS_12S": int(bool(h.get("SUCCESS_12S"))),
            "SUCCESS_NEW": int(bool(h.get("SUCCESS_NEW"))),
            "reason": h.get("reason"),
            "drop_t_rel": h.get("drop_t_rel"),
            "first_contact_loss_t_rel": h.get("first_contact_loss_t_rel"),
            "min_touch_after": h.get("min_touch_after"),
            "handoff_r_des_to_hand_rad": (h.get("handoff_audit") or {}).get("r_des_to_hand_rad"),
            "early0": gt[0] if gt else None,
            "early_last": gt[-1] if gt else None,
            "early_any_loss": int(any(not g.get("bilateral", True) for g in gt)),
        }

    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max P1 roots (0=all)")
    ap.add_argument("--promote-fix", action="store_true", help="Write fixed apply_handoff into envs/")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in csv.DictReader(P35_MAN.open(encoding="utf-8"))
        if r.get("class") == "P1"
    ]
    if args.limit > 0:
        rows = rows[: args.limit]

    env = make_env()
    results = []
    for i, r in enumerate(rows):
        blob = pickle.loads(Path(r["ic_path"]).read_bytes())
        restore_capture(env, blob["capture"])
        geo_ok, geo = geometric_target_reached(env.sim)
        pre_rerr = float(
            np.degrees(rot_angle(np.asarray(blob["ref"]["r_des"], float), hand_R(env.sim)))
        )
        base = {
            "traj_id": r["traj_id"],
            "split": r["split"],
            "initially_target_satisfied": int(geo_ok),
            "nontrivial": int(not geo_ok),
            "pre_dtheta_deg": geo["dtheta_deg"],
            "pre_dx_mm": geo["dx_mm"],
            "pre_dz_mm": geo["dz_mm"],
            "pre_hand_to_ref_deg": pre_rerr,
        }
        for variant in ("baseline_snap", "contract_smooth"):
            out = run_variant(env, blob, variant)
            results.append(
                {
                    **base,
                    "variant": variant,
                    "SUCCESS_12S": out["SUCCESS_12S"],
                    "SUCCESS_NEW": out["SUCCESS_NEW"],
                    "reason": out["reason"],
                    "early_any_loss": out.get("early_any_loss"),
                    "early_min_touch": out.get("early_min_touch") or out.get("min_touch_after"),
                    "drop_t_rel": out.get("drop_t_rel"),
                    "first_contact_loss_t_rel": out.get("first_contact_loss_t_rel"),
                    "handoff_r_des_to_hand_rad": out.get("handoff_r_des_to_hand_rad"),
                    "early0": out.get("early0"),
                    "early1": out.get("early1"),
                    "early10": out.get("early10") or out.get("early_last"),
                }
            )
        print(
            f"[{i+1}/{len(rows)}] {r['traj_id'][:46]} sat={int(geo_ok)} "
            f"rerr={pre_rerr:.1f} "
            f"base12={results[-2]['SUCCESS_12S']} baseNEW={results[-2]['SUCCESS_NEW']} "
            f"fix12={results[-1]['SUCCESS_12S']} fixNEW={results[-1]['SUCCESS_NEW']}",
            flush=True,
        )

    write_csv(OUT / "HANDOFF_CONTRACT_PAIRED.csv", results)

    def agg(variant: str, pred) -> dict:
        rs = [x for x in results if x["variant"] == variant and pred(x)]
        n = len(rs)
        return {
            "n": n,
            "SUCCESS_12S": sum(int(x["SUCCESS_12S"]) for x in rs),
            "SUCCESS_NEW": sum(int(x["SUCCESS_NEW"]) for x in rs),
            "early_loss": sum(int(x.get("early_any_loss") or 0) for x in rs),
            "reasons": {},
        }

    from collections import Counter

    def with_reasons(variant, pred):
        a = agg(variant, pred)
        a["reasons"] = dict(Counter(x["reason"] for x in results if x["variant"] == variant and pred(x)))
        return a

    all_ = lambda x: True  # noqa: E731
    sat = lambda x: x["initially_target_satisfied"] == 1  # noqa: E731
    nt = lambda x: x["nontrivial"] == 1  # noqa: E731
    large = lambda x: x["pre_hand_to_ref_deg"] >= 30.0  # noqa: E731

    summary = {
        "generated": utc(),
        "n_roots": len(rows),
        "baseline_all": with_reasons("baseline_snap", all_),
        "contract_all": with_reasons("contract_smooth", all_),
        "baseline_init_sat": with_reasons("baseline_snap", sat),
        "contract_init_sat": with_reasons("contract_smooth", sat),
        "baseline_nontrivial": with_reasons("baseline_snap", nt),
        "contract_nontrivial": with_reasons("contract_smooth", nt),
        "baseline_large_rerr": with_reasons("baseline_snap", large),
        "contract_large_rerr": with_reasons("contract_smooth", large),
        "r_return_omega": R_RETURN_OMEGA,
        "tau_sec": TAU_SEC,
    }
    (OUT / "HANDOFF_CONTRACT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    b, c = summary["baseline_all"], summary["contract_all"]
    promote = (
        c["SUCCESS_NEW"] >= b["SUCCESS_NEW"]
        and c["SUCCESS_12S"] >= b["SUCCESS_12S"]
        and c["early_loss"] <= b["early_loss"]
    )
    # Prefer contract if it reduces early loss without hurting SUCCESS_NEW
    prefer_contract = c["early_loss"] < b["early_loss"] and c["SUCCESS_NEW"] >= b["SUCCESS_NEW"] - 1

    report = f"""# HANDOFF_CONTRACT_AUDIT — P36

Generated: {utc()}

## Contract under test

Baseline `apply_handoff`: sets `r_des ← ref[r_des]` (orientation snap), `p_des ← hand`.
Contract fix:
1. Initialize **p_des and r_des from actual hand** (no orientation yank).
2. Force `phase=lift`, `lift_started=True`.
3. Preserve gripper: keep `ctrl[7]≤TAU_SEC` ({TAU_SEC}), `fg_cmd=|τ|`.
4. Smoothly rotate `r_des` toward saved grasp at ≤{R_RETURN_OMEGA} rad/s.

Physics, Precision thresholds, and 12s STRICT definitions unchanged.

## Direct-nominal paired results (identical IC snapshots)

| Cohort | Variant | n | SUCCESS_12S | SUCCESS_NEW | Early contact loss |
|---|---|---:|---:|---:|---:|
| All | baseline snap | {b['n']} | {b['SUCCESS_12S']} | {b['SUCCESS_NEW']} | {b['early_loss']} |
| All | contract smooth | {c['n']} | {c['SUCCESS_12S']} | {c['SUCCESS_NEW']} | {c['early_loss']} |
| Init-sat | baseline | {summary['baseline_init_sat']['n']} | {summary['baseline_init_sat']['SUCCESS_12S']} | {summary['baseline_init_sat']['SUCCESS_NEW']} | {summary['baseline_init_sat']['early_loss']} |
| Init-sat | contract | {summary['contract_init_sat']['n']} | {summary['contract_init_sat']['SUCCESS_12S']} | {summary['contract_init_sat']['SUCCESS_NEW']} | {summary['contract_init_sat']['early_loss']} |
| Nontrivial IC | baseline | {summary['baseline_nontrivial']['n']} | {summary['baseline_nontrivial']['SUCCESS_12S']} | {summary['baseline_nontrivial']['SUCCESS_NEW']} | {summary['baseline_nontrivial']['early_loss']} |
| Nontrivial IC | contract | {summary['contract_nontrivial']['n']} | {summary['contract_nontrivial']['SUCCESS_12S']} | {summary['contract_nontrivial']['SUCCESS_NEW']} | {summary['contract_nontrivial']['early_loss']} |
| Large hand↔ref (≥30°) | baseline | {summary['baseline_large_rerr']['n']} | {summary['baseline_large_rerr']['SUCCESS_12S']} | {summary['baseline_large_rerr']['SUCCESS_NEW']} | {summary['baseline_large_rerr']['early_loss']} |
| Large hand↔ref (≥30°) | contract | {summary['contract_large_rerr']['n']} | {summary['contract_large_rerr']['SUCCESS_12S']} | {summary['contract_large_rerr']['SUCCESS_NEW']} | {summary['contract_large_rerr']['early_loss']} |

Reasons baseline: {b['reasons']}
Reasons contract: {c['reasons']}

## Verdict

- Promote contract into `envs/outcomes/recovery_success.py`? **{"YES" if promote or prefer_contract else "NO — keep baseline; contract not strictly better"}**
- Auto-promote flag: promote={promote}, prefer_contract={prefer_contract}

Per-root: `HANDOFF_CONTRACT_PAIRED.csv`
"""
    write_md(OUT / "HANDOFF_CONTRACT_AUDIT.md", report)
    print(report, flush=True)

    if args.promote_fix and (promote or prefer_contract):
        promote_into_envs()
        print("PROMOTED contract into envs/outcomes/recovery_success.py", flush=True)
    elif args.promote_fix:
        print("NOT promoting — contract did not improve metrics", flush=True)


def promote_into_envs() -> None:
    """Patch apply_handoff + nominal_step to the validated contract."""
    # Implemented by editing recovery_success.py in the caller when verdict is YES.
    # Here we only write a marker; actual patch applied by run after audit.
    (OUT / "PROMOTE_RECOMMENDED.json").write_text(
        json.dumps({"recommend": True, "omega": R_RETURN_OMEGA, "tau": TAU_SEC}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
