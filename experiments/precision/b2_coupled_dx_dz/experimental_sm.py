"""Full-trajectory Precision variants that inject Coupled DX+DZ without editing production."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

import data_generation.oracles.precision.state_machine as sm
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.state_machine import run_precision
from envs.simulation.capture import restore_capture, take_capture
from experiments.precision.b2_coupled_dx_dz.coupled_variants import (
    coupled_v1,
    coupled_v2,
    should_attempt_coupled,
)

_ORIG_ST = sm.correct_dx_support_transfer
_ORIG_DZ_IN = sm.correct_dz_inward
_ORIG_DZ_OUT = sm.correct_dz_outward


@contextmanager
def _patched_coupled(variant: str, *, max_safe_travel_mm: float = 18.0, force_coupled: bool = False):
    """Replace ST DX (+ subsequent DZ) with coupled when both residuals are large."""
    ctx: dict[str, Any] = {
        "coupled_done": False,
        "coupled_result": None,
        "variant": variant,
        "skipped_dz": False,
        "attempted": False,
        "force": force_coupled,
    }

    def _run_coupled(env, cur, cfg):
        blob = {"capture": cur["capture"], "ref": cur.get("ref")}
        if variant == "V2":
            return coupled_v2(env, blob, cfg=cfg)
        return coupled_v1(env, blob, cfg=cfg, max_safe_travel_mm=max_safe_travel_mm)

    def patched_st(env, cur, cfg: PrecisionConfig):
        restore_capture(env, cur["capture"])
        err = measure_errors(env.sim)
        use = force_coupled or should_attempt_coupled(err, cfg)
        if not use or ctx["coupled_done"]:
            return _ORIG_ST(env, cur, cfg)
        ctx["attempted"] = True
        res = _run_coupled(env, cur, cfg)
        ctx["coupled_result"] = {k: v for k, v in res.items() if k not in ("actions", "capture")}
        actions = list(res.get("actions") or [])
        if not res.get("ok"):
            # Match ST failure contract: SM restores snap and fails without silent fallback.
            return {
                "ok": False,
                "skipped": False,
                "reason": res.get("reason") or "coupled_dx_dz_failed",
                "failure_code": "COUPLED_DX_DZ_FAILED",
                "phase": f"coupled_{variant}",
                "actions": actions,
                "capture": res.get("capture") or cur["capture"],
                "errors": res.get("errors1") or measure_errors(env.sim),
                "experimental_coupled": ctx["coupled_result"],
            }
        ctx["coupled_done"] = True
        # Present as successful ST so SM keeps S1 path / skips redundant DZ via our dz patch.
        return {
            "ok": True,
            "skipped": False,
            "actions": actions,
            "capture": res.get("capture") or take_capture(env),
            "errors": res.get("errors1") or measure_errors(env.sim),
            "s1_state": {"experimental_coupled": True, "variant": variant},
            "timing_s": None,
            "experimental_coupled": ctx["coupled_result"],
            "phase": f"coupled_{variant}",
        }

    def patched_dz_in(env, cur, cfg: PrecisionConfig, s1_state=None):
        if ctx["coupled_done"]:
            ctx["skipped_dz"] = True
            restore_capture(env, cur["capture"])
            err = measure_errors(env.sim)
            return {
                "ok": True,
                "skipped": True,
                "reason": "coupled_already_corrected_xz",
                "actions": [],
                "capture": cur["capture"],
                "errors": err,
            }
        return _ORIG_DZ_IN(env, cur, cfg, s1_state=s1_state)

    def patched_dz_out(env, cur, cfg: PrecisionConfig, s1_state=None):
        if ctx["coupled_done"]:
            ctx["skipped_dz"] = True
            restore_capture(env, cur["capture"])
            err = measure_errors(env.sim)
            return {
                "ok": True,
                "skipped": True,
                "reason": "coupled_already_corrected_xz",
                "actions": [],
                "capture": cur["capture"],
                "errors": err,
            }
        return _ORIG_DZ_OUT(env, cur, cfg, s1_state=s1_state)

    sm.correct_dx_support_transfer = patched_st
    sm.correct_dz_inward = patched_dz_in
    sm.correct_dz_outward = patched_dz_out
    try:
        yield ctx
    finally:
        sm.correct_dx_support_transfer = _ORIG_ST
        sm.correct_dz_inward = _ORIG_DZ_IN
        sm.correct_dz_outward = _ORIG_DZ_OUT


def run_variant(
    env,
    blob: dict,
    *,
    plan_id: str = "B",
    variant: str = "V0",
    cfg: PrecisionConfig | None = None,
    max_safe_travel_mm: float = 18.0,
    force_coupled: bool = False,
) -> dict[str, Any]:
    cfg = cfg or PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)
    state = {
        "capture": blob["capture"],
        "ref": blob["ref"],
        "sample_id": blob.get("sample_id") or blob.get("hard_id"),
    }
    restore_capture(env, blob["capture"])
    if variant == "V0":
        res = run_precision(env, state, plan_id, cfg)
        return _pack(res, variant="V0", coupled_ctx=None)

    with _patched_coupled(variant, max_safe_travel_mm=max_safe_travel_mm, force_coupled=force_coupled) as ctx:
        res = run_precision(env, state, plan_id, cfg)
        return _pack(res, variant=variant, coupled_ctx=ctx)


def _pack(res, *, variant: str, coupled_ctx: dict | None) -> dict[str, Any]:
    hand = res.handoff or {}
    err_e = res.errors_entry or {}
    err_x = res.errors_exit or {}
    e0 = np.asarray(err_e.get("e_signed_mm") or [0, 0, 0], float)
    e1 = np.asarray(err_x.get("e_signed_mm") or [0, 0, 0], float)
    logs = res.primitive_logs or {}
    return {
        "variant": variant,
        "status": res.status,
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "dropped": not bool(err_x.get("bilateral", True)),
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "n_actions": int(len(res.actions or [])),
        "correction_duration_s": float(len(res.actions or [])) * 0.02,
        "errors_entry": {
            "dx_mm": float(e0[0]),
            "dy_mm": float(e0[1]),
            "dz_mm": float(e0[2]),
            "dtheta_deg": float(err_e.get("dtheta_deg") or 0.0),
            "e_pos_mm": float(err_e.get("e_pos_mm") or 0.0),
            "bilateral": bool(err_e.get("bilateral")),
        },
        "errors_exit": {
            "dx_mm": float(e1[0]),
            "dy_mm": float(e1[1]),
            "dz_mm": float(e1[2]),
            "dtheta_deg": float(err_x.get("dtheta_deg") or 0.0),
            "e_pos_mm": float(err_x.get("e_pos_mm") or 0.0),
            "bilateral": bool(err_x.get("bilateral")),
        },
        "handoff": {
            "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
            "reason": hand.get("std_reason") or hand.get("reason"),
        },
        "coupled_attempted": bool(coupled_ctx and coupled_ctx.get("attempted")),
        "coupled_done": bool(coupled_ctx and coupled_ctx.get("coupled_done")),
        "coupled_result": (coupled_ctx or {}).get("coupled_result"),
        "dx_mechanism": (logs.get("dx_mechanism") if isinstance(logs, dict) else None),
        "dx_support_transfer": logs.get("dx_support_transfer") if isinstance(logs, dict) else None,
        "primitive_logs_keys": sorted(list(logs.keys())) if isinstance(logs, dict) else [],
        "actions": list(res.actions or []),
        "capture_exit": res.capture,
        "note": res.note,
        "stage1_eligibility": logs.get("stage1_eligibility") if isinstance(logs, dict) else None,
    }
