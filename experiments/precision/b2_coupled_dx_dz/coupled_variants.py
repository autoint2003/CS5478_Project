"""Experimental Coupled DX+DZ variants (V1 prior mechanism, V2 return-aware stop).

Does not modify production ``correct_dx`` / ``correct_dz`` / ``correct_dtheta``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.support_transfer import restore_fingertips_down
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig, SlipGoal, WORKING_POS_MM
from envs.simulation.capture import restore_capture, take_capture
from experiments.precision.coupled_feasibility.coupled_dx_dz import (
    coupled_desire_hand,
    coupled_dx_dz,
)


def _err_pack(err: dict) -> dict[str, Any]:
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    return {
        "dx_mm": float(e[0]),
        "dy_mm": float(e[1]),
        "dz_mm": float(e[2]),
        "dtheta_deg": float(err.get("dtheta_deg") or 0.0),
        "e_pos_mm": float(err.get("e_pos_mm") or 0.0),
        "bilateral": bool(err.get("bilateral")),
        "fn_L": float(err.get("fn_L") or 0.0),
        "fn_R": float(err.get("fn_R") or 0.0),
        "fn_tot": float(err.get("fn_tot") or 0.0),
        "v_rel_norm": float(err.get("v_rel_norm") or 0.0),
        "alpha_up_deg": float(err.get("alpha_up_deg") or 0.0),
        "fingertips_down": bool(err.get("fingertips_down")),
    }


def coupled_v1(env, blob: dict, *, cfg: PrecisionConfig | None = None, max_safe_travel_mm: float = 18.0) -> dict[str, Any]:
    """Prior Coupled DX+DZ mechanism (recovered from precision_coupled_feasibility)."""
    cfg = cfg or PrecisionConfig(use_verified_lateral_fallback=False)
    # Temporarily tighten travel if ablation requests it via monkeypatch of SlipGoal defaults —
    # we wrap by editing cfg-like fields after calling base, then re-implement lightly.
    if abs(float(max_safe_travel_mm) - 18.0) < 1e-9:
        out = coupled_dx_dz(env, blob, cfg=cfg)
        out["variant"] = "V1"
        out["errors0_pack"] = _err_pack(out.get("errors0") or {})
        out["errors1_pack"] = _err_pack(out.get("errors1") or {})
        return out

    # Smaller-travel ablation: same pipeline with reduced max_safe_travel.
    from data_generation.oracles.precision.corrections.orientation import (
        execute_orientation_plan,
        plan_orientation,
        score_gravity_alignment,
    )
    from data_generation.oracles.precision.mechanics.axial_sliding import run_goal_aware_axial_slip
    from data_generation.oracles.precision.mechanics.dual_loop import sample_state
    from data_generation.oracles.precision.mechanics.open_axis import choose_roll_sign_for_support, run_roll_phase
    from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
    from data_generation.oracles.precision.state import FN_HOLD
    from evaluation.grasp_geometry import hand_R

    G_WORLD = np.array([0.0, 0.0, -9.81], float)
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    if not err0.get("bilateral"):
        return {"ok": False, "reason": "no_bilateral_contact", "actions": [], "capture": blob["capture"], "variant": "V1_small"}
    desire = coupled_desire_hand(err0["e_signed_mm"])
    if not desire.get("ok"):
        return {
            "ok": False,
            "reason": desire.get("reason") or "desire_infeasible",
            "desire": desire,
            "actions": [],
            "capture": blob["capture"],
            "infeasible": True,
            "variant": "V1_small",
            "errors0": err0,
        }
    d = np.asarray(desire["d"], float)
    actions: list = []
    logs: dict[str, Any] = {"desire": desire, "max_safe_travel_mm": max_safe_travel_mm}
    plan = plan_orientation(env, blob, d=d, g_useful_min=6.0, max_candidates=36)
    logs["orientation_plan"] = {k: v for k, v in plan.items() if k not in ("actions",)}
    if not plan.get("ok") and not plan.get("already_sufficient"):
        return {
            "ok": False,
            "reason": "no_feasible_gravity_orientation",
            "actions": [],
            "capture": blob["capture"],
            "infeasible": True,
            "variant": "V1_small",
            "logs": logs,
            "errors0": err0,
        }
    cur = {"capture": blob["capture"], "ref": blob.get("ref")}
    if not plan.get("already_sufficient"):
        eo = execute_orientation_plan(env, cur, plan)
        actions += eo.get("actions") or []
        if not eo.get("ok"):
            return {
                "ok": False,
                "reason": "orientation_exec_failed",
                "actions": actions,
                "capture": take_capture(env),
                "variant": "V1_small",
                "errors0": err0,
            }
        cur = {"capture": eo.get("capture") or take_capture(env), "ref": blob.get("ref")}
    restore_capture(env, cur["capture"])
    Rh = hand_R(env.sim)
    g_h = Rh.T @ G_WORLD
    sc = score_gravity_alignment(g_h, d)
    if float(sc["useful"]) < 4.0:
        return {
            "ok": False,
            "reason": "gravity_alignment_insufficient_after_orient",
            "actions": actions,
            "capture": take_capture(env),
            "infeasible": True,
            "variant": "V1_small",
            "errors0": err0,
        }
    roll_sign = float(choose_roll_sign_for_support(env.sim))
    try:
        from data_generation.oracles.precision.mechanics.open_axis import reference_pad_vector

        y0 = reference_pad_vector(env.sim)
    except Exception:
        y0 = np.array([0.0, 1.0, 0.0])
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    pr = run_roll_phase(
        env,
        cur,
        sign=roll_sign,
        angle_deg=10.0,
        phase="coupled_S1_bias",
        rate_scale=0.12,
        fn_hold=FN_HOLD,
        require_down=False,
        y0_w=y0,
        e_ref=e_ref,
    )
    actions += pr.get("actions") or []
    if not sample_state(env.sim).get("bilateral"):
        return {
            "ok": False,
            "reason": "contact_loss_during_support_bias",
            "actions": actions,
            "capture": take_capture(env),
            "variant": "V1_small",
        }
    cur = {"capture": take_capture(env), "ref": blob.get("ref")}
    theta0 = float(measure_errors(env.sim)["dtheta_deg"])
    e0 = np.asarray(measure_errors(env.sim)["e_signed_mm"], float)
    wp = float(cfg.working_pos_mm)
    axis_tol = float(np.clip(np.sqrt(max(wp * wp - float(e0[1]) ** 2, 0.0)) * 0.5, 2.0, 6.0))
    goal = SlipGoal(
        axis="dx",
        desired_direction=d,
        max_safe_travel_mm=float(max_safe_travel_mm),
        max_duration_s=10.0,
        axis_tol_mm=axis_tol,
        working_pos_mm=wp,
        working_theta_deg=float(cfg.working_theta_deg),
        ey_tol_mm=float(cfg.ey_tol_mm),
        use_euclidean_stop=True,
        brake_margin_mm=1.5,
    )
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    slip = run_goal_aware_axial_slip(env, cur, goal, z_ref=z_ref, guard_lateral=False, guard_fingertip_margin=False)
    actions += slip.get("actions") or []
    err1 = measure_errors(env.sim)
    e1 = np.asarray(err1["e_signed_mm"], float)
    theta1 = float(err1["dtheta_deg"])
    if theta1 > theta0 + 5.0:
        return {
            "ok": False,
            "reason": "induced_dtheta_increase",
            "theta0": theta0,
            "theta1": theta1,
            "actions": actions,
            "capture": take_capture(env),
            "variant": "V1_small",
            "errors0": err0,
            "errors1": err1,
        }
    r0 = float(np.linalg.norm([e0[0], e0[2]]))
    r1 = float(np.linalg.norm([e1[0], e1[2]]))
    improved = r1 < r0 - 1.0 or float(err1["e_pos_mm"]) <= wp
    ok = bool(slip.get("ok") and err1.get("bilateral") and improved and not slip.get("infeasible"))
    return {
        "ok": bool(ok),
        "mechanism": "coupled_dx_dz_experimental_small_travel",
        "variant": "V1_small",
        "reason": None if ok else (slip.get("stop_reason") or slip.get("reason") or "coupled_slip_failed"),
        "desire": desire,
        "errors0": err0,
        "errors1": err1,
        "residual_xz0_mm": r0,
        "residual_xz1_mm": r1,
        "d_theta_deg": theta1 - theta0,
        "actions": actions,
        "capture": take_capture(env),
        "logs": logs,
        "errors0_pack": _err_pack(err0),
        "errors1_pack": _err_pack(err1),
    }


def _probe_return_dz_drift(env, blob: dict, cfg: PrecisionConfig) -> dict[str, Any]:
    """After a V1 coupled slip, measure DZ change across restore_DOWN (no second slip)."""
    v1 = coupled_dx_dz(env, blob, cfg=cfg)
    if not v1.get("ok") or not v1.get("capture"):
        return {
            "ok": False,
            "reason": v1.get("reason") or "coupled_failed_before_return_probe",
            "v1": {k: v for k, v in v1.items() if k not in ("actions", "capture")},
            "dz_pre_return": None,
            "dz_post_return": None,
            "dz_drift_mm": None,
            "actions": v1.get("actions") or [],
            "capture": v1.get("capture") or blob["capture"],
            "dropped": not bool((v1.get("errors1") or {}).get("bilateral", True)),
        }
    restore_capture(env, v1["capture"])
    err_pre = measure_errors(env.sim)
    dz_pre = float(err_pre["dz_mm"])
    rd = restore_fingertips_down(
        env,
        {"capture": v1["capture"], "ref": blob.get("ref")},
        cfg,
        rate_scale=float(cfg.rate_restore_down),
    )
    acts = list(v1.get("actions") or []) + list(rd.get("actions") or [])
    if not rd.get("ok"):
        return {
            "ok": False,
            "reason": "restore_DOWN_failed_in_drift_probe",
            "dz_pre_return": dz_pre,
            "dz_post_return": None,
            "dz_drift_mm": None,
            "actions": acts,
            "capture": take_capture(env),
            "dropped": not bool(measure_errors(env.sim).get("bilateral")),
            "v1_ok": True,
        }
    err_post = measure_errors(env.sim)
    dz_post = float(err_post["dz_mm"])
    drift = dz_post - dz_pre  # signed; positive = DZ grew during return
    return {
        "ok": True,
        "dz_pre_return": dz_pre,
        "dz_post_return": dz_post,
        "dz_drift_mm": float(drift),
        "errors_pre_return": _err_pack(err_pre),
        "errors_post_return": _err_pack(err_post),
        "actions": acts,
        "capture": rd.get("capture") or take_capture(env),
        "dropped": not bool(err_post.get("bilateral")),
        "v1_ok": True,
        "v1_residual_xz1": v1.get("residual_xz1_mm"),
        "v1_stop_reason": v1.get("stop_reason"),
    }


def coupled_v2(env, blob: dict, *, cfg: PrecisionConfig | None = None) -> dict[str, Any]:
    """Coupled DX+DZ with final-error-aware stopping via measured return/regrip DZ drift.

    Does **not** arbitrarily lower the DZ threshold. Procedure:
    1. Probe V1 + restore_DOWN on a clone of the IC; measure signed DZ drift.
    2. If grasp is lost in the probe, reject compensation (return V1 result / fail).
    3. Re-run coupled with a tighter Euclidean working band only when drift is
       outward (worsens |DZ| toward failure) and a safe pre-return target exists:
       ``working_pos_eff = clip(WORKING_POS - |drift| - margin, 4, WORKING_POS)``.
    4. Reject any adjusted run that drops the object or worsens bilateral contact.
    """
    cfg = cfg or PrecisionConfig(use_verified_lateral_fallback=False)
    restore_capture(env, blob["capture"])
    entry = measure_errors(env.sim)

    # Probe on deep-copied capture identity (same physics; restore after).
    probe_blob = {"capture": blob["capture"], "ref": blob.get("ref")}
    probe = _probe_return_dz_drift(env, probe_blob, cfg)
    restore_capture(env, blob["capture"])

    if probe.get("dropped"):
        return {
            "ok": False,
            "variant": "V2",
            "reason": "return_compensation_rejected_grasp_loss_in_probe",
            "probe": {k: v for k, v in probe.items() if k != "actions"},
            "actions": [],
            "capture": blob["capture"],
            "errors0": entry,
            "compensation_applied": False,
        }

    drift = probe.get("dz_drift_mm")
    if drift is None or not probe.get("ok"):
        # Fall back to V1 (no compensation available)
        v1 = coupled_dx_dz(env, blob, cfg=cfg)
        v1["variant"] = "V2"
        v1["compensation_applied"] = False
        v1["probe"] = {k: v for k, v in probe.items() if k != "actions"}
        v1["reason"] = v1.get("reason") or probe.get("reason")
        v1["errors0_pack"] = _err_pack(v1.get("errors0") or entry)
        v1["errors1_pack"] = _err_pack(v1.get("errors1") or {})
        return v1

    # Only compensate when return increases |DZ| (signed drift same sign as residual and grows magnitude)
    dz_pre = float(probe["dz_pre_return"])
    dz_post = float(probe["dz_post_return"])
    worsens = abs(dz_post) > abs(dz_pre) + 0.5
    if not worsens:
        # Return does not inflate DZ; V1 stop is already final-error-aware enough.
        v1 = coupled_dx_dz(env, blob, cfg=cfg)
        # Still append restore for parity with full path expectation of post-return state
        acts = list(v1.get("actions") or [])
        cap = v1.get("capture") or blob["capture"]
        if v1.get("ok") and cap:
            rd = restore_fingertips_down(env, {"capture": cap, "ref": blob.get("ref")}, cfg, rate_scale=float(cfg.rate_restore_down))
            acts += rd.get("actions") or []
            cap = rd.get("capture") or cap
            err_f = measure_errors(env.sim)
        else:
            err_f = v1.get("errors1") or measure_errors(env.sim)
        return {
            **{k: v for k, v in v1.items() if k not in ("actions", "capture")},
            "ok": bool(v1.get("ok") and err_f.get("bilateral")),
            "variant": "V2",
            "compensation_applied": False,
            "compensation_reason": "return_dz_drift_not_worsening",
            "probe": {k: v for k, v in probe.items() if k != "actions"},
            "dz_pre_return": dz_pre,
            "dz_post_return": float(err_f.get("dz_mm") or dz_post),
            "actions": acts,
            "capture": cap,
            "errors0": v1.get("errors0") or entry,
            "errors1": err_f,
            "errors0_pack": _err_pack(v1.get("errors0") or entry),
            "errors1_pack": _err_pack(err_f),
        }

    # Safe pre-return Euclidean target: leave headroom for observed drift magnitude.
    wp0 = float(cfg.working_pos_mm or WORKING_POS_MM)
    margin = 1.0
    drift_mag = abs(float(drift))
    wp_eff = float(np.clip(wp0 - drift_mag - margin, 4.0, wp0))
    if wp_eff >= wp0 - 0.25:
        # No meaningful legal headroom
        v1 = coupled_dx_dz(env, blob, cfg=cfg)
        v1["variant"] = "V2"
        v1["compensation_applied"] = False
        v1["compensation_reason"] = "no_safe_headroom_for_drift"
        v1["probe"] = {k: v for k, v in probe.items() if k != "actions"}
        return v1

    cfg2 = deepcopy(cfg)
    cfg2.working_pos_mm = wp_eff
    v2 = coupled_dx_dz(env, blob, cfg=cfg2)
    acts = list(v2.get("actions") or [])
    if not v2.get("ok"):
        # Reject adjustment that fails primitive; do not keep a worse drop.
        restore_capture(env, blob["capture"])
        return {
            "ok": False,
            "variant": "V2",
            "reason": "compensated_coupled_failed",
            "compensation_applied": True,
            "working_pos_eff_mm": wp_eff,
            "probe": {k: v for k, v in probe.items() if k != "actions"},
            "coupled": {k: v for k, v in v2.items() if k not in ("actions", "capture")},
            "actions": acts,
            "capture": v2.get("capture") or blob["capture"],
            "errors0": entry,
            "errors1": v2.get("errors1"),
            "dropped": not bool((v2.get("errors1") or {}).get("bilateral", True)),
        }

    restore_capture(env, v2["capture"])
    err_pre_r = measure_errors(env.sim)
    rd = restore_fingertips_down(
        env,
        {"capture": v2["capture"], "ref": blob.get("ref")},
        cfg,
        rate_scale=float(cfg.rate_restore_down),
    )
    acts += rd.get("actions") or []
    if not rd.get("ok") or not measure_errors(env.sim).get("bilateral"):
        restore_capture(env, blob["capture"])
        return {
            "ok": False,
            "variant": "V2",
            "reason": "compensation_rejected_grasp_loss_after_return",
            "compensation_applied": True,
            "working_pos_eff_mm": wp_eff,
            "probe": {k: v for k, v in probe.items() if k != "actions"},
            "actions": [],  # do not keep destabilizing tape for SM injection
            "capture": blob["capture"],
            "dropped": True,
            "dz_pre_return": float(err_pre_r["dz_mm"]),
        }

    err_f = measure_errors(env.sim)
    return {
        "ok": True,
        "variant": "V2",
        "mechanism": "coupled_dx_dz_final_error_aware",
        "compensation_applied": True,
        "working_pos_eff_mm": wp_eff,
        "working_pos_nominal_mm": wp0,
        "probe": {k: v for k, v in probe.items() if k != "actions"},
        "dz_pre_return": float(err_pre_r["dz_mm"]),
        "dz_post_return": float(err_f["dz_mm"]),
        "dz_drift_observed_mm": float(drift),
        "desire": v2.get("desire"),
        "errors0": v2.get("errors0") or entry,
        "errors1": err_f,
        "residual_xz0_mm": v2.get("residual_xz0_mm"),
        "residual_xz1_mm": float(np.linalg.norm([err_f["e_signed_mm"][0], err_f["e_signed_mm"][2]])),
        "d_theta_deg": v2.get("d_theta_deg"),
        "stop_reason": v2.get("stop_reason"),
        "actions": acts,
        "capture": rd.get("capture") or take_capture(env),
        "logs": v2.get("logs"),
        "errors0_pack": _err_pack(v2.get("errors0") or entry),
        "errors1_pack": _err_pack(err_f),
        "dropped": False,
    }


def should_attempt_coupled(err: dict, cfg: PrecisionConfig) -> bool:
    """True when both DX and DZ contribute to position residual (coupled relevant)."""
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    dx, dz = abs(float(e[0])), abs(float(e[2]))
    return bool(dx > 0.25 * float(cfg.ex_tol_mm) and dz > 0.25 * float(cfg.ez_tol_mm))
