#!/usr/bin/env python
"""MuJoCo replay: five sequences proving FINAL_DOWN_GATE before HANDOFF.

1. DX → DZ inward (S1 reuse) → DOWN → HANDOFF
2. DX → restore DOWN → DZ outward → HANDOFF
3. DX-only → restore DOWN → HANDOFF
4. DZ-only inward → restore DOWN → HANDOFF
5. DZ-only outward (already DOWN) → verify DOWN → HANDOFF

Does not weaken SUCCESS_NEW / SUCCESS_12S. No Stage-1 / GRPO / M111 changes.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
    restore_fingertips_down,
)
from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward
from data_generation.oracles.precision.corrections.axial_outward import correct_dz_outward
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.handoff import (  # noqa: E402
    posture_is_fingertips_down,
    run_final_down_handoff_gate,
)
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/final_down_handoff_sequences"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"


def jsafe(o):
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


def load(plan: str) -> dict:
    p = {
        "A": STATES / "plan_a__ball_train_v2_0070__stable.pkl",
        "B": STATES / "plan_b__hard_910220__stable.pkl",
        "C": STATES / "plan_c__gbi_010__stable.pkl",
    }[plan]
    return pickle.loads(p.read_bytes())


def posture_snap(env) -> dict:
    g = gates(env.sim)
    err = measure_errors(env.sim)
    return {
        "fingertips_down": posture_is_fingertips_down(env.sim),
        "alpha_up_deg": float(g["alpha_up_deg"]),
        "bilateral": bool(err["bilateral"]),
        "v_rel_norm": float(err["v_rel_norm"]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dx_mm": float(err["dx_mm"]),
        "dz_mm": float(err["dz_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
    }


def assert_no_horiz_handoff(phases: list, row: dict) -> list[str]:
    """Fail if early HANDOFF claimed without a final_down_gate phase nearby."""
    issues = []
    ph = list(phases or [])
    if "PRECISION_EARLY_HANDOFF" in ph or "handoff_nominal_12s" in ph:
        if not any("final_down_gate" in str(p) for p in ph):
            issues.append("HANDOFF_without_final_down_gate")
    # Horizontal working alpha should not be the exit posture on success
    if row.get("SUCCESS_NEW") and row.get("exit_posture"):
        a = float((row["exit_posture"] or {}).get("alpha_up_deg") or 180)
        if 60.0 <= a <= 120.0:
            issues.append(f"SUCCESS_NEW_at_horizontal_alpha={a:.1f}")
        if not (row["exit_posture"] or {}).get("fingertips_down"):
            issues.append("SUCCESS_NEW_without_fingertips_down")
    return issues


def seq1_dx_dz_inward_s1(env, cfg) -> dict:
    st = load("B")
    res = run_precision(env, st, "B", cfg)
    restore_capture(env, res.capture)
    exit_p = posture_snap(env)
    row = {
        "sequence": "1_DX_DZ_inward_S1_reuse",
        "phases": list(res.phases or []),
        "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
        "dz_s1_reused": (res.primitive_logs or {}).get("dz_s1_reused"),
        "status": res.status,
        "SUCCESS_NEW": (res.handoff or {}).get("SUCCESS_NEW"),
        "SUCCESS_12S": (res.handoff or {}).get("SUCCESS_12S"),
        "failures": list(res.failures or []),
        "exit_posture": exit_p,
        "contract": {
            "s1_reuse_preserved": bool((res.primitive_logs or {}).get("dz_s1_reused")),
            "final_down_gate_present": any("final_down_gate" in str(p) for p in (res.phases or [])),
            "no_restore_between_dx_and_inward": "restore_DOWN_before_dz_outward" not in (res.phases or []),
        },
    }
    row["issues"] = assert_no_horiz_handoff(res.phases, row)
    return row


def seq3_dx_only(env, cfg) -> dict:
    """Forced DX-only: ST then FINAL_DOWN_GATE then HANDOFF (no dz)."""
    st = load("B")
    restore_capture(env, st["capture"])
    blob = {"capture": st["capture"], "ref": st["ref"]}
    actions = []
    states = ["ENTRY", "DX1_6"]
    rdx = correct_dx_support_transfer(env, blob, cfg)
    actions.extend(rdx.get("actions") or [])
    if not rdx.get("ok"):
        return {
            "sequence": "3_DX_only_restore_DOWN_HANDOFF",
            "ok": False,
            "failure_reason": f"ST:{rdx.get('reason')}:{rdx.get('phase')}",
            "phases": states + ["FAIL_ST"],
            "SUCCESS_NEW": False,
            "SUCCESS_12S": None,
        }
    cur = {"capture": rdx["capture"], "ref": blob["ref"]}
    # Must not HANDOFF from horizontal — gate restores DOWN
    states += ["DX6_done", "FINAL_DOWN_GATE"]
    gate = run_final_down_handoff_gate(env, cur, cfg, actions)
    if not gate.get("ok"):
        return {
            "sequence": "3_DX_only_restore_DOWN_HANDOFF",
            "ok": False,
            "failure_reason": f"{gate.get('failure_code')}:{gate.get('reason')}",
            "phases": states + ["FAIL_GATE"],
            "gate": jsafe({k: v for k, v in gate.items() if k != "capture"}),
            "SUCCESS_NEW": False,
            "SUCCESS_12S": None,
            "exit_posture": posture_snap(env),
        }
    cur = {"capture": gate["capture"], "ref": cur["ref"]}
    restore_capture(env, cur["capture"])
    assert posture_is_fingertips_down(env.sim)
    states.append("HANDOFF")
    handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    exit_p = posture_snap(env)
    row = {
        "sequence": "3_DX_only_restore_DOWN_HANDOFF",
        "ok": True,
        "phases": states,
        "skipped_restore_in_gate": gate.get("skipped_restore"),
        "gate_phase": gate.get("phase"),
        "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
        "SUCCESS_12S": handoff.get("SUCCESS_12S"),
        "exit_posture": exit_p,
        "ex0": rdx.get("ex0"),
        "ex1": rdx.get("ex1"),
        "contract": {
            "down_before_handoff": exit_p["fingertips_down"],
            "not_horizontal": not (60 <= exit_p["alpha_up_deg"] <= 120),
        },
    }
    row["issues"] = assert_no_horiz_handoff(states, row)
    return row


def seq2_dx_then_outward(env, cfg) -> dict:
    st = load("B")
    restore_capture(env, st["capture"])
    blob = {"capture": st["capture"], "ref": st["ref"]}
    actions = []
    states = ["ENTRY", "DX1_6"]
    rdx = correct_dx_support_transfer(env, blob, cfg)
    actions.extend(rdx.get("actions") or [])
    if not rdx.get("ok"):
        return {"sequence": "2_DX_restore_DOWN_DZ_outward", "ok": False, "failure_reason": rdx.get("reason")}
    cur = {"capture": rdx["capture"], "ref": blob["ref"]}
    states.append("RESTORE_DOWN_before_outward")
    rd = restore_fingertips_down(env, cur, cfg, rate_scale=float(cfg.rate_restore_down))
    actions.extend(rd.get("actions") or [])
    if not rd.get("ok"):
        return {
            "sequence": "2_DX_restore_DOWN_DZ_outward",
            "ok": False,
            "failure_reason": f"POSTURE_DOWN_RESTORE_FAILED:{rd.get('reason')}",
            "phases": states,
        }
    cur = {"capture": rd["capture"], "ref": cur["ref"]}
    assert posture_is_fingertips_down(env.sim)
    states.append("DZ_outward_Case_A")
    rz = correct_dz_outward(env, cur, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    if not rz.get("ok") and not rz.get("skipped"):
        # Outward may be unnecessary (ez already small); still proceed to gate
        states.append(f"dz_out_status:{rz.get('reason') or rz.get('branch')}")
    else:
        cur = {"capture": rz.get("capture") or cur["capture"], "ref": cur["ref"]}
        states.append(str(rz.get("branch") or "outward_ok"))
    states.append("FINAL_DOWN_GATE")
    gate = run_final_down_handoff_gate(env, cur, cfg, actions)
    if not gate.get("ok"):
        return {
            "sequence": "2_DX_restore_DOWN_DZ_outward",
            "ok": False,
            "failure_reason": f"{gate.get('failure_code')}:{gate.get('reason')}",
            "phases": states,
            "SUCCESS_NEW": False,
        }
    cur = {"capture": gate["capture"], "ref": cur["ref"]}
    restore_capture(env, cur["capture"])
    handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    exit_p = posture_snap(env)
    row = {
        "sequence": "2_DX_restore_DOWN_DZ_outward",
        "ok": True,
        "phases": states + ["HANDOFF"],
        "dz_branch": rz.get("branch"),
        "skipped_rotation": rz.get("skipped_rotation"),
        "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
        "SUCCESS_12S": handoff.get("SUCCESS_12S"),
        "exit_posture": exit_p,
        "contract": {
            "restored_down_before_outward": True,
            "down_at_handoff": exit_p["fingertips_down"],
        },
    }
    row["issues"] = assert_no_horiz_handoff(row["phases"], row)
    return row


def seq4_dz_only_inward(env, cfg) -> dict:
    st = load("B")
    restore_capture(env, st["capture"])
    blob = {"capture": st["capture"], "ref": st["ref"]}
    actions = []
    states = ["ENTRY", "DZ_inward"]
    rz = correct_dz_inward(env, blob, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    states.append(str(rz.get("branch") or "B"))
    if not rz.get("ok"):
        return {
            "sequence": "4_DZ_only_inward",
            "ok": False,
            "failure_reason": rz.get("reason") or rz.get("phase"),
            "phases": states,
            "SUCCESS_NEW": None,
            "SUCCESS_12S": None,
            "notes": "Cold upright inward may fail (unresolved); gate not reached.",
            "exit_posture": posture_snap(env),
        }
    cur = {"capture": rz["capture"], "ref": blob["ref"]}
    states.append("FINAL_DOWN_GATE")
    gate = run_final_down_handoff_gate(env, cur, cfg, actions)
    if not gate.get("ok"):
        return {
            "sequence": "4_DZ_only_inward",
            "ok": False,
            "failure_reason": f"{gate.get('failure_code')}:{gate.get('reason')}",
            "phases": states,
            "SUCCESS_NEW": False,
        }
    cur = {"capture": gate["capture"], "ref": cur["ref"]}
    restore_capture(env, cur["capture"])
    handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    exit_p = posture_snap(env)
    row = {
        "sequence": "4_DZ_only_inward",
        "ok": True,
        "phases": states + ["HANDOFF"],
        "branch": rz.get("branch"),
        "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
        "SUCCESS_12S": handoff.get("SUCCESS_12S"),
        "exit_posture": exit_p,
        "outcomes": jsafe(rz.get("outcomes")),
        "contract": {"down_at_handoff": exit_p["fingertips_down"]},
    }
    row["issues"] = assert_no_horiz_handoff(row["phases"], row)
    return row


def seq5_dz_outward_already_down(env, cfg) -> dict:
    st = load("B")
    restore_capture(env, st["capture"])
    assert posture_is_fingertips_down(env.sim) or float(gates(env.sim)["alpha_up_deg"]) >= 160.0
    blob = {"capture": st["capture"], "ref": st["ref"]}
    actions = []
    states = ["ENTRY", "already_DOWN", "DZ_outward_direct"]
    rz = correct_dz_outward(env, blob, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    states.append(str(rz.get("branch") or "?"))
    cur = {"capture": rz.get("capture") or take_capture(env), "ref": blob["ref"]}
    states.append("FINAL_DOWN_GATE")
    gate = run_final_down_handoff_gate(env, cur, cfg, actions)
    if not gate.get("ok"):
        return {
            "sequence": "5_DZ_outward_already_DOWN",
            "ok": False,
            "failure_reason": f"{gate.get('failure_code')}:{gate.get('reason')}",
            "phases": states,
            "SUCCESS_NEW": False,
        }
    cur = {"capture": gate["capture"], "ref": cur["ref"]}
    restore_capture(env, cur["capture"])
    handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    exit_p = posture_snap(env)
    row = {
        "sequence": "5_DZ_outward_already_DOWN",
        "ok": True,
        "phases": states + ["HANDOFF"],
        "skipped_rotation": rz.get("skipped_rotation"),
        "branch": rz.get("branch"),
        "dz_ok": rz.get("ok"),
        "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
        "SUCCESS_12S": handoff.get("SUCCESS_12S"),
        "exit_posture": exit_p,
        "contract": {
            "skipped_unnecessary_S1": bool(rz.get("skipped_rotation")),
            "down_at_handoff": exit_p["fingertips_down"],
        },
    }
    row["issues"] = assert_no_horiz_handoff(row["phases"], row)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=False,
    )
    t0 = time.time()
    results = {}
    print("seq1 DX→DZ inward S1...", flush=True)
    results["1"] = seq1_dx_dz_inward_s1(env, cfg)
    print("seq2 DX→DOWN→outward...", flush=True)
    results["2"] = seq2_dx_then_outward(env, cfg)
    print("seq3 DX-only...", flush=True)
    results["3"] = seq3_dx_only(env, cfg)
    print("seq4 DZ-only inward...", flush=True)
    results["4"] = seq4_dz_only_inward(env, cfg)
    print("seq5 DZ outward already DOWN...", flush=True)
    results["5"] = seq5_dz_outward_already_down(env, cfg)

    lines = [
        "# Final DOWN HANDOFF Sequences",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| # | Sequence | SUCCESS_NEW | SUCCESS_12S | DOWN@exit | Issues |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in results.items():
        ep = v.get("exit_posture") or {}
        lines.append(
            f"| {k} | {v.get('sequence')} | {v.get('SUCCESS_NEW')} | {v.get('SUCCESS_12S')} | "
            f"{ep.get('fingertips_down')} | {v.get('issues') or v.get('failure_reason') or ''} |"
        )
    lines += [
        "",
        "## Contract",
        "",
        "- All successful HANDOFF paths pass `final_down_gate` / `FINAL_DOWN_GATE`.",
        "- DX→DZ inward preserves S1 reuse (no restore between DX6 and inward).",
        "- DX→DZ outward restores DOWN before DualLoop.",
        "- SUCCESS_* from frozen nominal continuation only; in-band residuals alone ≠ success.",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "SEQUENCES.json").write_text(json.dumps(jsafe(results), indent=2), encoding="utf-8")
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wall {time.time()-t0:.1f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
