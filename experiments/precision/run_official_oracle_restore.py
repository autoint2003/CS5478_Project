#!/usr/bin/env python
"""Official Precision Oracle restore diagnostics.

Config A: broken DX1 ang_tol=6, fallback off
Config B: corrected DX1, fallback off  (production)
Config C: corrected DX1, fallback on   (diagnosis only)

Mechanism cases: hard_910026, hard_910220, Path07 (Plan B ST).
Also re-runs the 20 A Precision ICs under B.
"""

from __future__ import annotations

import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.expert_pipeline.sources import discover_branch_ics, load_generation_config  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/official_oracle_restore"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
IC_910026 = ROOT / "results/data_generation/stage2_abc_final_v2/ics/B__hard_910026__post_b1_v2__e3a32a342c39f52b.pkl"
IC_910220 = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"


def jsafe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def load_blob(path: Path, plan: str | None = None) -> dict:
    blob = pickle.loads(path.read_bytes())
    if plan:
        blob["plan"] = plan
    return blob


def snap_physics(env) -> dict:
    err = measure_errors(env.sim)
    mode = classify_contact_mode(env.sim)
    g = gates(env.sim)
    return {
        "errors": err,
        "contact_mode": mode,
        "fingertips_down": bool(g["fingertips_down"]),
        "alpha_up_deg": float(g["alpha_up_deg"]),
        "bilateral": bool(err.get("bilateral")),
        "Fn_z_L": mode.get("Fn_z_L"),
        "Fn_z_R": mode.get("Fn_z_R"),
        "support_side": mode.get("support_side"),
        "one_sided_mechanical": bool(mode.get("one_sided_mechanical")),
    }


def cfg_b() -> PrecisionConfig:
    return PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=False,
    )


def cfg_c() -> PrecisionConfig:
    return PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=True,
    )


def run_st_with_forced_dx1_tol(env, blob: dict, ang_tol: float) -> dict:
    """Monkey-patch DX1 prep tilt tolerance to emulate broken vs fixed DX1."""
    import data_generation.oracles.precision.mechanics.support_transfer as st_mod
    from data_generation.oracles.precision.mechanics import open_axis as oa

    orig = oa.run_tilt_phase

    def wrapped(*args, **kwargs):
        phase = kwargs.get("phase") or (args[3] if len(args) > 3 else None)
        # Only force on DX1 prep down phase name used in support_transfer
        if phase == "dxST_P0":
            kwargs = dict(kwargs)
            kwargs["ang_tol_deg"] = float(ang_tol)
        return orig(*args, **kwargs)

    oa.run_tilt_phase = wrapped
    try:
        restore_capture(env, blob["capture"])
        return correct_dx_support_transfer(env, blob, cfg_b())
    finally:
        oa.run_tilt_phase = orig


def summarize_precision(res) -> dict:
    logs = res.primitive_logs or {}
    handoff = res.handoff or {}
    return {
        "status": res.status,
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "dx_mechanism": logs.get("dx_mechanism"),
        "legacy_fallback_explicit": bool(logs.get("legacy_fallback_explicit")),
        "dz_s1_reused": logs.get("dz_s1_reused"),
        "dx_keep_s1_for_dz": logs.get("dx_keep_s1_for_dz"),
        "SUCCESS_NEW": bool(handoff.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(handoff.get("SUCCESS_12S")),
        "handoff_reason": handoff.get("reason"),
        "note": res.note,
    }


def first_failure_locus(res) -> str:
    fails = list(res.failures or [])
    phases = list(res.phases or [])
    status = res.status or ""
    joined = " ".join(fails) + " " + " ".join(phases) + " " + status
    for key in (
        "DX1",
        "support_transfer",
        "dtheta",
        "protected",
        "lateral",
        "dz",
        "FINAL_DOWN",
        "HANDOFF",
        "LOST_GRASP",
    ):
        if key.lower() in joined.lower():
            return key
    if "PRECISION_PRIMITIVE_FAILED" in status:
        return "primitive"
    if "HANDOFF" in status or "handoff" in joined.lower():
        return "HANDOFF"
    return status or "unknown"


def compare_abc(env, name: str, blob: dict) -> dict:
    out: dict[str, Any] = {"ic": name}
    # A: broken DX1 ST-only probe
    try:
        rst_a = run_st_with_forced_dx1_tol(env, blob, ang_tol=6.0)
        g_a = gates(env.sim) if rst_a.get("ok") else None
        out["A_broken_dx1_st"] = {
            "ok": bool(rst_a.get("ok")),
            "phase": rst_a.get("phase"),
            "reason": rst_a.get("reason"),
            "alpha_up_deg": rst_a.get("alpha_up_deg") or (g_a or {}).get("alpha_up_deg"),
            "down_ang_tol_deg": rst_a.get("down_ang_tol_deg"),
            "fingertips_down_after": bool((g_a or {}).get("fingertips_down")) if g_a else None,
            "physics": snap_physics(env) if rst_a.get("ok") else None,
        }
    except Exception as exc:
        out["A_broken_dx1_st"] = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}

    # B: official full precision
    try:
        res_b = run_precision(env, blob, "B", cfg_b())
        out["B_official"] = summarize_precision(res_b)
        out["B_official"]["failure_locus"] = first_failure_locus(res_b)
        out["B_official"]["physics_exit"] = snap_physics(env)
    except Exception as exc:
        out["B_official"] = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}

    # C: diagnosis fallback
    try:
        res_c = run_precision(env, blob, "B", cfg_c())
        out["C_fallback_diag"] = summarize_precision(res_c)
        out["C_fallback_diag"]["failure_locus"] = first_failure_locus(res_c)
    except Exception as exc:
        out["C_fallback_diag"] = {"error": str(exc), "traceback": traceback.format_exc(limit=3)}
    return out


def verify_dx1_gate(env, blob: dict) -> dict:
    """Confirm fixed DX1 stopped pose passes DOWN gate before DX2."""
    restore_capture(env, blob["capture"])
    rst = correct_dx_support_transfer(env, blob, cfg_b())
    pl = rst.get("phase_log") or []
    # After DX1 prep, check gate from returned metadata / current if still in mid-ST
    return {
        "st_ok": bool(rst.get("ok")),
        "st_phase": rst.get("phase"),
        "st_reason": rst.get("reason"),
        "down_ang_tol_deg": rst.get("down_ang_tol_deg"),
        "alpha_up_deg": rst.get("alpha_up_deg"),
        "mechanism": rst.get("mechanism"),
        "phase_log_phases": [p.get("phase") for p in pl],
        "physics": snap_physics(env),
        "s1_state": {k: v for k, v in (rst.get("s1_state") or {}).items() if k != "capture"},
    }


def run_a20(env) -> dict:
    cfg = load_generation_config()
    sources = discover_branch_ics("A_STABLE_PRECISION", cfg)
    rows = []
    for src in sources:
        blob = load_blob(src.ic_path, "A")
        state = {"capture": blob["capture"], "ref": blob["ref"]}
        res = run_precision(env, state, "A", cfg_b())
        row = {
            "sample_id": src.sample_id,
            **summarize_precision(res),
            "failure_locus": first_failure_locus(res),
        }
        rows.append(row)
        print(
            f"A20 {src.sample_id}: {row['status']} NEW={row['SUCCESS_NEW']} "
            f"mech={row['dx_mechanism']} locus={row['failure_locus']}",
            flush=True,
        )
    n_new = sum(1 for r in rows if r["SUCCESS_NEW"])
    return {
        "n": len(rows),
        "SUCCESS_NEW": n_new,
        "SUCCESS_12S": sum(1 for r in rows if r["SUCCESS_12S"]),
        "by_status": {s: sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "by_locus": {s: sum(1 for r in rows if r["failure_locus"] == s) for s in sorted({r["failure_locus"] for r in rows})},
        "fallback_entered": sum(1 for r in rows if r.get("legacy_fallback_explicit")),
        "rows": rows,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "production_fallback": False,
        "PrecisionConfig_default_fallback": PrecisionConfig().use_verified_lateral_fallback,
    }

    print("DX1 gate verify on hard_910220...", flush=True)
    blob220 = load_blob(IC_910220, "B")
    report["dx1_gate_hard_910220"] = verify_dx1_gate(env, blob220)

    print("ABC compare hard_910220...", flush=True)
    report["abc_hard_910220"] = compare_abc(env, "hard_910220", blob220)

    print("ABC compare hard_910026...", flush=True)
    if IC_910026.is_file():
        blob026 = load_blob(IC_910026, "B")
        report["dx1_gate_hard_910026"] = verify_dx1_gate(env, blob026)
        report["abc_hard_910026"] = compare_abc(env, "hard_910026", blob026)
    else:
        report["abc_hard_910026"] = {"error": f"missing {IC_910026}"}

    print("Path07 full Plan B (official B)...", flush=True)
    res07 = run_precision(env, blob220, "B", cfg_b())
    report["path07_official"] = {
        **summarize_precision(res07),
        "failure_locus": first_failure_locus(res07),
        "physics": snap_physics(env),
        "cycles": jsafe((res07.primitive_logs or {}).get("cycles")),
    }

    print("Re-run 20 A Precision ICs (official)...", flush=True)
    report["a20_official"] = run_a20(env)

    path = OUT / "OFFICIAL_ORACLE_RESTORE.json"
    path.write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    a20 = report["a20_official"]
    print(
        f"A20 official SUCCESS_NEW={a20['SUCCESS_NEW']}/{a20['n']} "
        f"fallback_entered={a20['fallback_entered']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
