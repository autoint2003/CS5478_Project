"""Coordinate conventions + favorable orientation for DZ (legal, non-Coupled)."""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.precision.mechanics.dual_loop import sample_state, DT, make_action
from data_generation.oracles.precision.mechanics.open_axis import (
    choose_roll_sign_for_support,
    reference_pad_vector,
    run_roll_phase,
)
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import FN_HOLD
from envs.simulation.capture import restore_capture, take_capture
from evaluation.grasp_geometry import hand_R

G_WORLD = np.array([0.0, 0.0, -9.81], float)


def analyze_dx_gravity(sim) -> dict[str, Any]:
    """Signed DX vs gravity in hand/world frames.

    Conventions (Precision residuals):
    - ``e_signed_mm[0]`` = DX = hand-frame lateral COM residual (mm).
    - Hand X ≈ open/lateral pad axis used by DualLoop lateral; Z ≈ axial (open).
    - World gravity = (0,0,-9.81).
    - ``u_dx_offset_hand = sign(DX) * e_x_hat`` — direction of the existing lateral offset.
    - Adverse gravity loading: ``adverse = g_h · u_dx_offset_hand`` (m/s²).
      Positive ⇒ gravity pulls further along the already-offset side (unsafe edge load risk).
    - World downwardness of offset: ``offset_down = -u_dx_world[2]`` (>0 ⇒ offset points down).
    """
    err = measure_errors(sim)
    e = np.asarray(err["e_signed_mm"], float)
    dx = float(e[0])
    Rh = hand_R(sim)
    g_h = Rh.T @ G_WORLD
    sign = 0.0 if abs(dx) < 1e-6 else float(np.sign(dx))
    u_dx_h = np.array([sign, 0.0, 0.0], float)
    u_dx_w = Rh @ u_dx_h
    adverse = float(np.dot(g_h, u_dx_h))  # m/s^2
    offset_down = float(-u_dx_w[2])  # dimensionless direction cosine toward -world Z
    # Favorable region: gravity does not strongly load the offset side, and offset not steeply down
    favorable = bool(adverse <= 2.0 and offset_down <= 0.35)
    return {
        "dx_mm": dx,
        "dz_mm": float(e[2]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "g_h": g_h.tolist(),
        "sign_dx": sign,
        "u_dx_hand": u_dx_h.tolist(),
        "u_dx_world": u_dx_w.tolist(),
        "adverse_g_along_dx_offset_mps2": adverse,
        "offset_points_down_score": offset_down,
        "favorable": favorable,
        "bilateral": bool(err.get("bilateral")),
        "note": (
            "favorable := (g_h·sign(dx)·x̂ ≤ 2 m/s²) and (−u_dx_world·ẑ ≤ 0.35); "
            "not Coupled-style g·d≥6 axial alignment"
        ),
    }


def prepare_favorable_orientation(env, blob: dict, *, max_roll_deg: float = 25.0) -> dict[str, Any]:
    """Modest legal roll to reduce adverse gravity×DX loading. Abort if contact lost.

    Does **not** require Coupled ``g_useful_min`` along a 2D desire.
    """
    restore_capture(env, blob["capture"])
    before = analyze_dx_gravity(env.sim)
    actions: list = []
    logs: dict[str, Any] = {"before": before}
    if before["favorable"] or abs(before["dx_mm"]) < 1.5:
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_favorable_or_dx_small",
            "actions": [],
            "capture": blob["capture"],
            "before": before,
            "after": before,
            "dtheta_delta": 0.0,
        }

    # Roll sense: try both signs; pick the one that reduces adverse without losing contact
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    best = None
    for sign in (+1.0, -1.0):
        restore_capture(env, blob["capture"])
        snap = take_capture(env)
        pr = run_roll_phase(
            env,
            {"capture": snap, "ref": blob.get("ref")},
            sign=float(sign),
            angle_deg=float(max_roll_deg),
            phase="dz_first_favorable_roll",
            rate_scale=0.10,
            fn_hold=FN_HOLD,
            require_down=False,
            y0_w=y0,
            e_ref=e_ref,
        )
        if not pr.get("ok") or not sample_state(env.sim).get("bilateral"):
            restore_capture(env, blob["capture"])
            continue
        after = analyze_dx_gravity(env.sim)
        score = (
            float(after["adverse_g_along_dx_offset_mps2"]),
            float(after["offset_points_down_score"]),
            abs(float(after["dtheta_deg"]) - float(before["dtheta_deg"])),
        )
        cand = {
            "sign": sign,
            "after": after,
            "actions": list(pr.get("actions") or []),
            "capture": take_capture(env),
            "score": score,
            "favorable": bool(after["favorable"]),
        }
        if best is None or cand["score"] < best["score"]:
            best = cand
        restore_capture(env, blob["capture"])

    if best is None:
        return {
            "ok": False,
            "reason": "no_safe_roll_candidate",
            "actions": [],
            "capture": blob["capture"],
            "before": before,
            "after": before,
            "unreachable_favorable": True,
        }

    # Accept only if adverse improved and grasp retained; do not force favorable flag
    improved = float(best["after"]["adverse_g_along_dx_offset_mps2"]) < float(
        before["adverse_g_along_dx_offset_mps2"]
    ) - 0.5
    if not improved and not best["favorable"]:
        return {
            "ok": True,
            "skipped": True,
            "reason": "roll_would_not_improve_adverse_skip",
            "actions": [],
            "capture": blob["capture"],
            "before": before,
            "after": before,
            "unreachable_favorable": not before["favorable"],
            "tried_best": {k: v for k, v in best.items() if k != "actions"},
        }

    restore_capture(env, best["capture"])
    actions = best["actions"]
    after = best["after"]
    return {
        "ok": True,
        "skipped": False,
        "reason": None,
        "actions": actions,
        "capture": best["capture"],
        "before": before,
        "after": after,
        "roll_sign": best["sign"],
        "dtheta_delta": float(after["dtheta_deg"]) - float(before["dtheta_deg"]),
        "favorable_after": bool(after["favorable"]),
        "unreachable_favorable": not bool(after["favorable"]),
        "logs": logs,
    }
