#!/usr/bin/env python
"""Diagnose Seq4 to_UP overshoot on hard_910084 post_b1_v2 (dense series)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import W_MAX
from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from data_generation.oracles.precision.mechanics.open_axis import (
    gates,
    project_out_roll,
    u_open_world,
    world_omega_to_body,
)
from data_generation.oracles.precision.mechanics.finger_vertical import Z_ABORT
from data_generation.oracles.precision.mechanics.dual_loop import DT, make_action, sample_state
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand
from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.mechanics.reverse_rotation import HoldModeGrip
from data_generation.oracles.precision.mechanics.rotation_braking import (
    contact_rate_guard,
    scheduled_rate,
)
from data_generation.oracles.precision.mechanics import protected_rotation as sar
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from models.transformer_policy import DT_POLICY

OUT = ROOT / "results/diagnostics/seq4_to_up_overshoot"
IC = ROOT / "results/data_generation/stage2_abc_final_v2/ics/B__hard_910084__post_b1_v2__71c154bb9a00bf73.pkl"
FN_HOLD = sar.FN_HOLD


def dense_fixed_axis_tilt(env, blob, **kw):
    axis0 = sar._unit(kw["axis_w"])
    t = sar._unit(kw["target_dir_w"])
    sign = float(kw["sign"])
    phase = kw["phase"]
    rate_scale = float(kw.get("rate_scale", 0.14))
    fn_hold = float(kw.get("fn_hold", FN_HOLD))
    max_time_s = float(kw.get("max_time_s", 18.0))
    ang_tol_deg = float(kw.get("ang_tol_deg", 6.0))
    z_ref_boost = float(kw.get("z_ref_boost", 0.08))
    height_kp = float(kw.get("height_kp", 6.5))
    support_side_hint = kw.get("support_side_hint")
    pre_lift_s = float(kw.get("pre_lift_s", 0.0))
    cruise_r = float(kw.get("cruise") if kw.get("cruise") is not None else rate_scale)
    peak_r = float(kw.get("peak") if kw.get("peak") is not None else max(cruise_r * 1.25, cruise_r))
    floor_r = float(kw.get("floor") if kw.get("floor") is not None else min(0.06, cruise_r))
    accel = float(kw.get("accel_s") if kw.get("accel_s") is not None else 0.35)
    use_speed_profile = bool(kw.get("use_speed_profile", True))

    restore_capture(env, blob["capture"])
    u0 = u_open_world(env.sim)
    ang0 = float(np.degrees(np.arccos(np.clip(np.dot(u0, t), -1.0, 1.0))))
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    hold = HoldModeGrip(fn_hold=fn_hold)
    hold.reset(sample_state(env.sim)["fn_tot"])
    actions, series = [], []
    fail = None
    path_len = 0.0
    prev_u = u0.copy()
    side = support_side_hint or classify_contact_mode(env.sim).get("support_side")
    z_cmd = z0 + float(z_ref_boost)
    for _ in range(int(max(0.0, pre_lift_s) / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_cmd, kp=height_kp), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
    z0 = float(env.sim.data.xpos[env.sim.ids.object_body][2])

    first_in_tol_t = None
    min_ang = ang0
    min_ang_t = 0.0
    min_alpha = float(gates(env.sim)["alpha_up_deg"])
    stop_cmd_t = None

    for i in range(int(max_time_s / DT)):
        u = u_open_world(env.sim)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(u, t), -1.0, 1.0))))
        path_len += float(np.linalg.norm(u - prev_u))
        prev_u = u.copy()
        alpha = float(gates(env.sim)["alpha_up_deg"])
        t_s = i * DT
        if ang < min_ang:
            min_ang = ang
            min_ang_t = t_s
            min_alpha = alpha
        if first_in_tol_t is None and ang <= ang_tol_deg:
            first_in_tol_t = t_s

        if use_speed_profile:
            rate = scheduled_rate(
                t_s=t_s, ang_err_deg=ang, ang0_deg=ang0,
                cruise=cruise_r, peak=peak_r, floor=floor_r, accel_s=accel,
            )
            rate, guard = contact_rate_guard(env.sim, rate, support_side=side, floor=floor_r)
        else:
            rate = float(rate_scale)
            guard = {"guard": "legacy"}

        zb = float(z_ref_boost)
        if ang < 100.0:
            zb = max(zb, 0.08)
        if ang < 40.0:
            zb = max(zb, 0.10)
        z_tgt = max(z0 + zb, Z_ABORT + 0.05)
        ow = project_out_roll(axis0 * (float(sign) * float(W_MAX) * rate), u)
        wb = world_omega_to_body(env.sim, ow)
        omega_cmd_norm = float(np.linalg.norm(wb))
        w_act = np.asarray(env.sim.data.cvel[env.sim.ids.hand_body][3:6], float)
        omega_act_norm = float(np.linalg.norm(w_act))

        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_tgt, kp=height_kp), wb, a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        st2 = sample_state(env.sim)
        m = classify_contact_mode(env.sim)
        series.append({
            "t": (i + 1) * DT,
            "ang_err_deg": ang,
            "alpha_up_deg": alpha,
            "rate_cmd": rate,
            "omega_cmd_norm": omega_cmd_norm,
            "omega_act_norm": omega_act_norm,
            "Fn_z_L": m["Fn_z_L"],
            "Fn_z_R": m["Fn_z_R"],
            "fn_L": st2["fn_L"],
            "fn_R": st2["fn_R"],
            "support_side": m["support_side"],
            "one_sided": m["one_sided_mechanical"],
            "v_rel_norm": st2["v_rel_norm"],
            "bilateral": st2["bilateral"],
            "guard": guard.get("guard"),
        })
        if latched(env.sim) or not st2.get("bilateral"):
            fail = "contact_loss"
            break
        if ang <= ang_tol_deg:
            stop_cmd_t = (i + 1) * DT
            break
    else:
        fail = fail or "timeout"

    for _ in range(6):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_cmd, kp=height_kp), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        alpha = float(gates(env.sim)["alpha_up_deg"])
        u = u_open_world(env.sim)
        ang = float(np.degrees(np.arccos(np.clip(np.dot(u, t), -1.0, 1.0))))
        series.append({
            "t": (series[-1]["t"] + DT) if series else DT,
            "ang_err_deg": ang,
            "alpha_up_deg": alpha,
            "rate_cmd": 0.0,
            "omega_cmd_norm": 0.0,
            "omega_act_norm": float(np.linalg.norm(np.asarray(env.sim.data.cvel[env.sim.ids.hand_body][3:6], float))),
            "settle": True,
            "Fn_z_L": classify_contact_mode(env.sim)["Fn_z_L"],
            "Fn_z_R": classify_contact_mode(env.sim)["Fn_z_R"],
            "bilateral": bool(sample_state(env.sim)["bilateral"]),
        })

    u1 = u_open_world(env.sim)
    ang1 = float(np.degrees(np.arccos(np.clip(np.dot(u1, t), -1.0, 1.0))))
    alpha1 = float(gates(env.sim)["alpha_up_deg"])
    post_min = [s for s in series if s["t"] >= min_ang_t]
    max_ang_after_min = max((s["ang_err_deg"] for s in post_min), default=ang1)
    min_alpha_all = min((s["alpha_up_deg"] for s in series), default=alpha1)
    summary = {
        "phase": phase,
        "ang0_deg": ang0,
        "ang1_deg": ang1,
        "alpha_final": alpha1,
        "ang_tol_deg": ang_tol_deg,
        "first_in_tol_t": first_in_tol_t,
        "stop_cmd_t": stop_cmd_t,
        "min_ang_err": min_ang,
        "min_ang_t": min_ang_t,
        "min_alpha": min_alpha,
        "min_alpha_all": min_alpha_all,
        "max_ang_after_min": max_ang_after_min,
        "overshoot_ang_deg": float(max(0.0, max_ang_after_min - min_ang)),
        "n_actions": len(actions),
        "fail": fail,
        "floor_r": floor_r,
        "cruise_r": cruise_r,
    }
    dense_fixed_axis_tilt.last = {"summary": summary, "series": series}
    ok = fail is None and ang1 <= ang_tol_deg + 2.0 and bool(sample_state(env.sim).get("bilateral"))
    return {
        "ok": ok,
        "reason": "ok" if ok else fail,
        "phase": phase,
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
        "axis_w": axis0.tolist(),
        "sign": sign,
        "ang0_deg": ang0,
        "ang1_deg": ang1,
        "path_length_u": path_len,
        "sim_s": len(actions) * DT,
        "support_side": side,
        "mode_exit": classify_contact_mode(env.sim),
        "selection": {},
        "rate_profile": {"cruise": cruise_r, "peak": peak_r, "floor": floor_r},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    cfg = PrecisionConfig()
    blob = pickle.loads(IC.read_bytes())

    sar.run_fixed_axis_tilt = dense_fixed_axis_tilt
    restore_capture(env, blob["capture"])
    rz = correct_dz_inward(env, {"capture": blob["capture"], "ref": blob["ref"]}, cfg, s1_state=None)
    data = getattr(dense_fixed_axis_tilt, "last", None)

    cause = []
    out = {
        "dz_ok": rz.get("ok"),
        "dz_branch": rz.get("branch"),
        "dz_phase": rz.get("phase"),
        "dz_reason": rz.get("reason"),
    }
    if data:
        s = data["summary"]
        series = data["series"]
        out["to_up"] = s
        if s["floor_r"] > 0 and s["first_in_tol_t"] is not None:
            near = [x for x in series if abs(x["t"] - s["first_in_tol_t"]) < 0.06 and not x.get("settle")]
            if near and float(near[0].get("rate_cmd") or 0) >= s["floor_r"] * 0.85:
                cause.append("floor_rate_still_commanded_near_tol")
        if s["overshoot_ang_deg"] > 2.0:
            cause.append("ang_err_increased_after_minimum")
        if s["min_alpha_all"] < 2.0 and s["ang1_deg"] > s["ang_tol_deg"] + 1.0:
            cause.append("passed_near_or_through_vertical_with_exit_error")
        if s["min_alpha_all"] < -1.0:
            cause.append("alpha_went_negative_past_UP")
        settle = [x for x in series if x.get("settle")]
        if settle and float(settle[0].get("omega_act_norm") or 0) > 0.2:
            cause.append("residual_angular_inertia_at_settle_start")
        # Same-direction command while ang already rising after min
        after = [x for x in series if x["t"] > s["min_ang_t"] + 0.02 and not x.get("settle")]
        if after and any(float(x.get("rate_cmd") or 0) >= s["floor_r"] * 0.85 for x in after):
            cause.append("continued_same_sign_command_after_closest_approach")

        out["cause_tags"] = cause
        keys = [
            "t", "ang_err_deg", "alpha_up_deg", "rate_cmd", "omega_cmd_norm", "omega_act_norm",
            "Fn_z_L", "Fn_z_R", "fn_L", "fn_R", "v_rel_norm", "bilateral", "guard",
        ]
        lines = [",".join(keys)]
        for row in series:
            lines.append(",".join(str(row.get(k, "")) for k in keys))
        (OUT / "to_up_series.csv").write_text("\n".join(lines), encoding="utf-8")
        (OUT / "to_up_series.json").write_text(json.dumps(series, indent=2, default=float), encoding="utf-8")
        print(json.dumps(s, indent=2, default=float))
        print("cause_tags", cause)
    else:
        out["error"] = "no_series"
        print("no series; dz", rz.get("ok"), rz.get("phase"), rz.get("reason"))

    (OUT / "DIAGNOSIS.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
