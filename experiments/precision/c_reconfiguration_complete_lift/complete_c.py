"""Complete experimental C: release → recatch → stabilize → lift → 12S.

Reuses production ``fast_world_lift_sequence`` (no new lift controller).
Does not modify production Oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from controllers.recovery.action_map import TAU_SECURE
from data_generation.oracles.plan_b.release_recatch import _action_from
from data_generation.oracles.plan_c.catch_stabilization import fast_world_lift_sequence, grasp_geometry
from data_generation.oracles.plan_c.dynamic_catch import jaw_state
from data_generation.oracles.plan_c.future_trajectory import grasp_point, hold_action, pad_bilateral, sample_kin
from data_generation.oracles.plan_c.interception import FULL_DETACH_ACTIVE_RECATCH
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.representability import representability
from envs.outcomes.ground_drop import arm, latched as ground_latched
from envs.outcomes.ground_drop import restore_state as restore_drop
from envs.outcomes.horizon_outcome import NOMINAL_S, nominal_verify
from envs.outcomes.table_drop import latched as table_latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from training.recovery_execution import make_snap

DT = 0.02
PINCH_Z = 0.0
# Production Plan C / A-B airborne nominal elevation
DEFAULT_TARGET_Z = 0.55


def err_pack(sim) -> dict:
    e = measure_errors(sim)
    return {
        "dx": float(e["dx_mm"]),
        "dy": float(e["dy_mm"]),
        "dz": float(e["dz_mm"]),
        "dtheta": float(e["dtheta_deg"]),
        "e_pos": float(e["e_pos_mm"]),
        "bilateral": bool(e.get("bilateral")),
        "fn_tot": float(e.get("fn_tot") or 0),
        "obj_z": float(e.get("obj_z") or sim.data.xpos[sim.ids.object_body][2]),
        "alpha_up_deg": float(e.get("alpha_up_deg") or 0),
        "v_rel_norm": float(e.get("v_rel_norm") or 0),
    }


def _quantize(raw, book):
    packed = representability(np.asarray(raw, np.float32).reshape(1, 7), book)
    return np.asarray(packed["quantized"][0], np.float32)


def reconfig_improved(e0: dict, e1: dict, *, t_dx: float = 6.0, t_dz: float = 12.0) -> dict:
    ax0, az0 = abs(float(e0["dx"])), abs(float(e0["dz"]))
    ax1, az1 = abs(float(e1["dx"])), abs(float(e1["dz"]))
    dth0, dth1 = abs(float(e0["dtheta"])), abs(float(e1["dtheta"]))
    dx_ok = ax1 <= t_dx
    dz_ok = az1 <= t_dz
    other_blown = False
    if dx_ok and not dz_ok:
        other_blown = az1 > az0 + 10.0 or dth1 > dth0 + 20.0
    if dz_ok and not dx_ok:
        other_blown = ax1 > ax0 + 10.0 or dth1 > dth0 + 20.0
    useful = (dx_ok or dz_ok) and (not other_blown) and bool(e1.get("bilateral"))
    return {
        "dx_in_band": dx_ok,
        "dz_in_band": dz_ok,
        "other_deteriorated": other_blown,
        "RECONFIG_IMPROVED": useful,
        "delta_abs_dx": ax1 - ax0,
        "delta_abs_dz": az1 - az0,
        "delta_abs_dtheta": dth1 - dth0,
        "delta_e_pos": float(e1["e_pos"]) - float(e0["e_pos"]),
    }


def _append_trace(traces, sim, t0, phase, **extra):
    e = err_pack(sim)
    pb = pad_bilateral(sim)
    traces.append(
        {
            "t": float(sim.data.time) - t0,
            "phase": phase,
            "obj_z": e["obj_z"],
            "hand_z": float(sim.data.xpos[sim.ids.hand_body][2]),
            "bilateral": bool(pb.get("bilateral")),
            "nL": int(pb.get("nL") or 0),
            "nR": int(pb.get("nR") or 0),
            "dx": e["dx"],
            "dz": e["dz"],
            "dtheta": e["dtheta"],
            "fn_tot": e["fn_tot"],
            **extra,
        }
    )


def release_and_recatch(
    env,
    blob: dict,
    book,
    *,
    open_steps: int = 5,
    t_c: float = 0.14,
    s_axial: float = 0.0,
    force_close_s: float = 0.34,
    horizon_s: float = 0.90,
    label: str = "mid_body",
) -> dict[str, Any]:
    """Legal controlled release + large recatch (no lift yet)."""
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    actions: list = []
    traces: list = []
    t0 = float(env.sim.data.time)
    _append_trace(traces, env.sim, t0, "init")

    closing = False
    first_bil_t = None
    saw_detach = False
    max_free = 0.0
    free_run = 0.0
    t_release = t_reacq = None
    release_state = None

    k0 = sample_kin(env.sim)
    p_tgt = grasp_point(k0["p_obj"], k0["u_obj"], s_axial)
    v_tgt = np.asarray(k0["v_obj"], float).copy()
    match, approach = 1.05, 4.5

    for i in range(int(horizon_s / DT)):
        t_rel = float(env.sim.data.time) - t0
        k = sample_kin(env.sim)
        js = jaw_state(env.sim)
        pb = pad_bilateral(env.sim)

        if t_rel >= t_c - 0.02:
            p_tgt = grasp_point(k["p_obj"], k["u_obj"], s_axial)
            v_tgt = np.asarray(k["v_obj"], float).copy()

        if i < open_steps:
            tau, grip_a = TAU_SECURE, -1.0  # open via grip; tau unused for grip_a override
            from controllers.recovery.action_map import TAU_OPEN

            tau = TAU_OPEN
        else:
            opening = float(js.get("opening_m") or 0)
            if (not closing) and ((opening >= 0.026 and bool(js.get("inside_jaw"))) or t_rel >= force_close_s):
                closing = True
            if closing:
                tau, grip_a = TAU_SECURE, 1.0
            else:
                from controllers.recovery.action_map import TAU_OPEN

                tau, grip_a = TAU_OPEN, -1.0

        ph, Rh = k["p_hand"], k["R_hand"]
        pinch_w = ph + Rh @ np.array([0.0, 0.0, PINCH_Z], float)
        err = p_tgt - pinch_w
        remain = max(t_c - t_rel, 0.02)
        brake_active = first_bil_t is not None and t_rel >= first_bil_t + 0.04
        if brake_active and pb.get("bilateral"):
            v_cmd = 0.5 * match * v_tgt + 0.3 * err
            if t_rel >= first_bil_t + 0.04 + 0.28:
                v_cmd = 0.1 * v_tgt + 0.2 * err
        else:
            v_cmd = approach * (err / remain) + match * v_tgt

        a = _action_from(env.sim, v_cmd, np.zeros(3), tau)
        a[6] = grip_a
        a = _quantize(a, book)
        step_legal(env.sim, a)
        actions.append(a.copy())

        pb2 = pad_bilateral(env.sim)
        nL, nR = int(pb2.get("nL") or 0), int(pb2.get("nR") or 0)
        if nL == 0 and nR == 0:
            if not saw_detach:
                t_release = t_rel
                release_state = {
                    "t": t_rel,
                    "p_obj": np.asarray(k["p_obj"], float).tolist(),
                    "v_obj": np.asarray(k["v_obj"], float).tolist(),
                    "obj_z": float(k["p_obj"][2]),
                }
            saw_detach = True
            free_run += DT
            max_free = max(max_free, free_run)
            phase = "free_motion"
        else:
            free_run = 0.0
            if saw_detach and nL > 0 and nR > 0 and t_reacq is None:
                t_reacq = t_rel
                first_bil_t = first_bil_t or t_rel
            if closing and saw_detach:
                phase = "brake" if brake_active else "close"
            elif not closing:
                phase = "open" if i < open_steps else "chase"
            else:
                phase = "close"
            if first_bil_t is not None and t_rel >= first_bil_t + 0.04:
                phase = "brake"

        if i % 2 == 0:
            _append_trace(traces, env.sim, t0, phase, nL=nL, nR=nR)

        if ground_latched(env.sim) or table_latched(env.sim):
            break
        if first_bil_t is not None and t_rel >= first_bil_t + 0.04 + 0.28 + 0.20:
            if pb2.get("bilateral"):
                break

    RELEASE_EXECUTED = bool(saw_detach and max_free >= DT - 1e-9)
    REACQUIRE_EXECUTED = bool(t_reacq is not None and pad_bilateral(env.sim).get("bilateral"))
    return {
        "label": label,
        "actions": actions,
        "traces": traces,
        "t0": t0,
        "errors0": e0,
        "errors_after_recatch": err_pack(env.sim),
        "RELEASE_EXECUTED": RELEASE_EXECUTED,
        "REACQUIRE_EXECUTED": REACQUIRE_EXECUTED,
        "t_release": t_release,
        "t_reacquire": t_reacq,
        "contact_free_s": max_free,
        "release_state": release_state,
        "params": {"open_steps": open_steps, "t_c": t_c, "s_axial": s_axial, "force_close_s": force_close_s},
        "capture": take_capture(env),
    }


def stabilize_contact(env, book, traces, t0, *, hold_s: float = 0.40) -> dict:
    actions = []
    ok = True
    for i in range(int(hold_s / DT)):
        if not pad_bilateral(env.sim).get("bilateral") or ground_latched(env.sim):
            ok = False
            break
        a = _action_from(env.sim, np.zeros(3), np.zeros(3), TAU_SECURE)
        a[6] = 1.0
        a = _quantize(a, book)
        step_legal(env.sim, a)
        actions.append(a.copy())
        if i % 2 == 0:
            _append_trace(traces, env.sim, t0, "stabilize")
    return {
        "CONTACT_STABILIZED": bool(ok and pad_bilateral(env.sim).get("bilateral")),
        "actions": actions,
        "errors": err_pack(env.sim),
    }


def lift_to_nominal(env, book, traces, t0, *, target_z: float, vz_up: float = 0.12) -> dict:
    """Reuse production ``fast_world_lift_sequence`` from current post-recatch snap."""
    z_before = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    xy_before = np.asarray(env.sim.data.xpos[env.sim.ids.object_body][:2], float).copy()
    dth_before = float(err_pack(env.sim)["dtheta"])

    # Arm ground-drop monitor like production catch path
    restore_drop(env.sim, None)
    arm(env.sim)
    brake_snap = make_snap(env.sim)
    lift = fast_world_lift_sequence(
        env.sim,
        brake_snap,
        vz_up=vz_up,
        target_z=float(target_z),
        max_s=8.0,
        grip=1.0,
    )
    actions = [np.asarray(a, np.float32) for a in (lift.get("actions") or [])]
    # Trace subsample during lift by replaying is unnecessary — lift already ran.
    # Log end state; also mid checkpoints from lift g0/g1
    _append_trace(
        traces,
        env.sim,
        t0,
        "lift_end",
        reached=bool(lift.get("reached_target_z")),
        target_z=target_z,
    )
    # Update FSM grasp height for nominal handoff (production raise_hand_airborne does this)
    env.sim.fsm.grasp_z = float(env.sim.data.xpos[env.sim.ids.hand_body][2])

    z_after = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    xy_after = np.asarray(env.sim.data.xpos[env.sim.ids.object_body][:2], float)
    e1 = err_pack(env.sim)
    reached = bool(lift.get("reached_target_z")) and bool(e1.get("bilateral")) and not ground_latched(env.sim)
    return {
        "LIFT_COMPLETED": reached,
        "actions": actions,
        "lift_meta": {
            k: lift.get(k)
            for k in (
                "reached_target_z",
                "obj_z_end",
                "hand_z_end",
                "end_bilateral",
                "ground_drop",
                "vz_up",
                "target_z",
                "continuous_150",
            )
        },
        "z_before": z_before,
        "z_after": z_after,
        "xy_drift_mm": float(np.linalg.norm(xy_after - xy_before)) * 1e3,
        "dtheta_delta": float(e1["dtheta"]) - dth_before,
        "errors": e1,
        "target_z": float(target_z),
    }


def verify_12s(env, ref) -> dict:
    cap = take_capture(env)
    restore_capture(env, cap)
    try:
        out = nominal_verify(env, ref, seconds=NOMINAL_S)
        ok12 = bool(out.get("success"))
        reason = out.get("reason")
    except Exception as exc:
        ok12, reason, out = False, f"verify_exception:{exc}", {}
    restore_capture(env, cap)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    restore_capture(env, cap)
    return {
        "SUCCESS_12S": ok12 or bool(hand.get("SUCCESS_12S")),
        "SUCCESS_NEW_probe": bool(hand.get("SUCCESS_NEW")),
        "nominal_reason": reason,
        "nominal_success": bool(out.get("success")) if out else False,
    }


def run_complete_c(
    env,
    blob: dict,
    book,
    *,
    variant: dict,
    t_dx: float = 6.0,
    t_dz: float = 12.0,
) -> dict[str, Any]:
    """Full sequence with explicit stage flags."""
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    # Nominal height: production C uses 0.55; if original exit already higher, match exit hand/obj
    z_exit = float(e0["obj_z"])
    target_z = max(DEFAULT_TARGET_Z, min(z_exit, 1.25))
    # If exit is already high (lifted B2), recatch may land lower — lift back toward exit height
    if z_exit >= 0.70:
        target_z = float(z_exit)

    rec = release_and_recatch(
        env,
        blob,
        book,
        open_steps=int(variant.get("open_steps", 5)),
        t_c=float(variant.get("t_c", 0.14)),
        s_axial=float(variant.get("s_axial", 0.0)),
        force_close_s=float(variant.get("force_close_s", 0.34)),
        label=str(variant.get("label", "default")),
    )
    actions = list(rec["actions"])
    traces = list(rec["traces"])
    t0 = float(rec["t0"])

    out = {
        "variant": variant.get("label"),
        "variant_params": rec["params"],
        "errors0": e0,
        "target_z": target_z,
        "RELEASE_EXECUTED": rec["RELEASE_EXECUTED"],
        "REACQUIRE_EXECUTED": rec["REACQUIRE_EXECUTED"],
        "CONTACT_STABILIZED": False,
        "LIFT_COMPLETED": False,
        "SUCCESS_12S": False,
        "RECONFIG_IMPROVED": False,
        "SUCCESS_NEW": False,
        "t_release": rec["t_release"],
        "t_reacquire": rec["t_reacquire"],
        "contact_free_s": rec["contact_free_s"],
        "release_state": rec["release_state"],
        "errors_after_recatch": rec["errors_after_recatch"],
        "stage_fail": None,
    }

    if not rec["RELEASE_EXECUTED"]:
        out["stage_fail"] = "NO_RELEASE"
        out["actions"] = actions
        out["traces"] = traces
        out["capture"] = take_capture(env)
        return out
    if not rec["REACQUIRE_EXECUTED"]:
        out["stage_fail"] = "NO_REACQUIRE"
        out["actions"] = actions
        out["traces"] = traces
        out["capture"] = take_capture(env)
        return out

    stab = stabilize_contact(env, book, traces, t0)
    actions += stab["actions"]
    out["CONTACT_STABILIZED"] = stab["CONTACT_STABILIZED"]
    out["errors_after_stabilize"] = stab["errors"]
    if not stab["CONTACT_STABILIZED"]:
        out["stage_fail"] = "STABILIZE_FAILED"
        out["actions"] = actions
        out["traces"] = traces
        out["capture"] = take_capture(env)
        return out

    z_pre_lift = float(err_pack(env.sim)["obj_z"])
    out["obj_z_pre_lift"] = z_pre_lift
    lift = lift_to_nominal(env, book, traces, t0, target_z=target_z)
    actions += lift["actions"]
    out["LIFT_COMPLETED"] = lift["LIFT_COMPLETED"]
    out["lift"] = {k: lift[k] for k in lift if k != "actions"}
    out["errors_after_lift"] = lift["errors"]
    if not lift["LIFT_COMPLETED"]:
        out["stage_fail"] = "LIFT_FAILED"
        out["actions"] = actions
        out["traces"] = traces
        out["capture"] = take_capture(env)
        return out

    # Extra settle after lift
    for i in range(15):
        a = hold_action(1.0)
        step_legal(env.sim, a)
        actions.append(np.asarray(a, np.float32).copy())
        if i % 3 == 0:
            _append_trace(traces, env.sim, t0, "post_lift_settle")

    e_lift = err_pack(env.sim)
    v12 = verify_12s(env, blob["ref"])
    out["SUCCESS_12S"] = bool(v12["SUCCESS_12S"])
    out["nominal_reason"] = v12.get("nominal_reason")
    out["verify_12s"] = v12
    imp = reconfig_improved(e0, e_lift, t_dx=t_dx, t_dz=t_dz)
    out.update(imp)
    out["RECONFIG_IMPROVED"] = bool(imp["RECONFIG_IMPROVED"] and out["SUCCESS_12S"])
    out["errors_final"] = e_lift
    out["actions"] = actions
    out["traces"] = traces
    out["capture"] = take_capture(env)
    out["mechanism"] = FULL_DETACH_ACTIVE_RECATCH
    if not out["SUCCESS_12S"]:
        out["stage_fail"] = f"HANDOFF_FAIL:{v12.get('nominal_reason')}"
    return out
