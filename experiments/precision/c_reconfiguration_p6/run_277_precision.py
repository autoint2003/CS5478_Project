#!/usr/bin/env python
"""P6 Part B: diagnose Precision failure on frozen 910277 C output; one paired P1."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import correct_dtheta, run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.c_reconfiguration_p6.geometry import snapshot_geometry  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/c_reconfiguration_p6"
PRIOR_CAP = ROOT / "results/diagnostics/c_reconfiguration_complete_lift/raw/hard_910277__mid_body_early__capture.pkl"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
HID = "hard_910277"


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


def load_frozen(env) -> dict:
    """Freeze post-C lift snapshot; verify it is the continuation start state."""
    cap = pickle.loads(PRIOR_CAP.read_bytes())
    blob = pickle.loads((EXIT / f"{HID}.pkl").read_bytes())
    meta = json.loads((EXIT / f"{HID}__meta.json").read_text(encoding="utf-8"))
    restore_capture(env, cap)
    geo = snapshot_geometry(env.sim)
    need = assess_need(measure_errors(env.sim), PrecisionConfig())
    return {
        "capture": cap,
        "ref": blob["ref"],
        "sample_id": meta.get("raw_id"),
        "frozen_geometry": geo,
        "assess_need": need,
        "note": "Continuation starts from mid_body_early capture after lift+12S (not a pose reset).",
    }


def run_p0(env, frozen: dict, pcfg: PrecisionConfig) -> dict:
    restore_capture(env, frozen["capture"])
    e0 = snapshot_geometry(env.sim)
    need0 = assess_need(measure_errors(env.sim), pcfg)
    res = run_precision(
        env,
        {"capture": frozen["capture"], "ref": frozen["ref"], "sample_id": frozen["sample_id"]},
        "B",
        pcfg,
    )
    hand = res.handoff or {}
    restore_capture(env, res.capture or take_capture(env)) if res.capture is not None else None
    e1 = snapshot_geometry(env.sim) if res.capture is not None else e0
    # Trace phases with residual snapshots by re-running need at start only + phase list
    phases = list(res.phases or [])
    # Detect first worsening: compare e0 vs exit
    worse = None
    if abs(float(e1.get("e_pos") or 0)) > abs(float(e0.get("e_pos") or 0)) + 1.0:
        worse = "e_pos_increased"
    if abs(float(e1.get("dtheta") or 0)) > abs(float(e0.get("dtheta") or 0)) + 2.0:
        worse = (worse + "+dtheta") if worse else "dtheta_increased"
    if not e1.get("bilateral"):
        worse = "contact_loss"

    skills_requested = {
        "dtheta": bool(need0.get("dtheta_needed")),
        "dx": bool(need0.get("dx_needed")),
        "dz": bool(need0.get("dz_needed")),
    }
    skills_executed = {
        "dtheta": any("dtheta" in p for p in phases),
        "dx": any("correct_dx" in p or "dx_" in p for p in phases),
        "dz": any("correct_dz" in p or "dz_" in p for p in phases),
        "support_transfer": any("support_transfer" in p for p in phases),
    }

    return {
        "label": "P0",
        "errors0": e0,
        "errors1": e1,
        "need0": need0,
        "skills_requested": skills_requested,
        "skills_executed": skills_executed,
        "PRECISION_SKILLS_EXECUTED": [k for k, v in skills_executed.items() if v],
        "phases": phases,
        "status": res.status,
        "failures": list(res.failures or []),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "PRECISION_FAILURE_REASON": (
            None
            if hand.get("SUCCESS_NEW")
            else (";".join(res.failures or []) or res.status or "HANDOFF_NEW_FAIL")
        ),
        "FINAL_POSE_ERROR": {
            "dx": e1.get("dx"),
            "dy": e1.get("dy"),
            "dz": e1.get("dz"),
            "dtheta": e1.get("dtheta"),
            "e_pos": e1.get("e_pos"),
        },
        "first_worsening": worse,
        "n_actions": len(res.actions or []),
        "actions": list(res.actions or []),
        "unnecessary_dx": bool(skills_executed["dx"] and not skills_requested["dx"]),
        "dx_already_small": abs(float(e0.get("dx") or 0)) <= 6.0,
    }


def run_p1_dtheta_only(env, frozen: dict, pcfg: PrecisionConfig) -> dict:
    """Targeted: only orientation correction + final down/handoff — skip dx/dz entirely.

    Justified because assess_need shows dx_needed=False, dz_needed=False, dtheta_needed=True.
    """
    from data_generation.oracles.precision.handoff import run_final_down_handoff_gate
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible

    restore_capture(env, frozen["capture"])
    e0 = snapshot_geometry(env.sim)
    actions: list = []
    phases = ["entry"]
    failures = []

    elig = check_stage1_eligible(
        env,
        {"capture": frozen["capture"], "ref": frozen["ref"], "sample_id": frozen["sample_id"]},
        allow_live_verify=True,
    )
    # Soft: if registry miss, continue from frozen bilateral state (experimental)
    cur = {"capture": frozen["capture"], "ref": frozen["ref"]}
    phases.append("correct_dtheta_only")
    rd = correct_dtheta(env, cur, pcfg, "B")
    actions += list(rd.get("actions") or [])
    if not rd.get("ok"):
        failures.append(f"dtheta:{rd.get('reason')}")
        restore_capture(env, take_capture(env))
        e1 = snapshot_geometry(env.sim)
        return {
            "label": "P1",
            "errors0": e0,
            "errors1": e1,
            "phases": phases,
            "failures": failures,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "PRECISION_FAILURE_REASON": failures[0],
            "FINAL_POSE_ERROR": {k: e1.get(k) for k in ("dx", "dy", "dz", "dtheta", "e_pos")},
            "PRECISION_SKILLS_EXECUTED": ["dtheta"],
            "n_actions": len(actions),
            "actions": actions,
            "hypothesis": "dtheta_only_skip_dx_dz",
            "stage1_eligible": elig.to_dict(),
        }
    cur = {"capture": rd["capture"], "ref": frozen["ref"]}
    phases.append("final_down_gate")
    gate = run_final_down_handoff_gate(env, cur, pcfg, actions)
    actions += []  # gate appends to actions list in-place
    if not gate.get("ok"):
        failures.append(f"final_down:{gate.get('reason')}")
    else:
        cur = {"capture": gate["capture"], "ref": frozen["ref"]}
    restore_capture(env, cur["capture"])
    hand = evaluate_handoff_with_axis(env.sim, frozen["ref"], stabilize_n=pcfg.stabilize_handoff_n)
    restore_capture(env, cur["capture"])
    e1 = snapshot_geometry(env.sim)
    return {
        "label": "P1",
        "errors0": e0,
        "errors1": e1,
        "phases": phases,
        "failures": failures,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "PRECISION_FAILURE_REASON": None
        if hand.get("SUCCESS_NEW")
        else (";".join(failures) or f"handoff_NEW_fail:12S={hand.get('SUCCESS_12S')}"),
        "FINAL_POSE_ERROR": {k: e1.get(k) for k in ("dx", "dy", "dz", "dtheta", "e_pos")},
        "PRECISION_SKILLS_EXECUTED": ["dtheta", "final_down_gate"],
        "n_actions": len(actions),
        "actions": actions,
        "hypothesis": "dtheta_only_skip_dx_dz",
        "handoff": {k: hand.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW")},
        "stage1_eligible": elig.to_dict(),
    }


def write_analysis(frozen: dict, p0: dict, p1: dict) -> None:
    lines = [
        "# 910277_PRECISION_FAILURE_ANALYSIS",
        "",
        "## Frozen C output (continuation start)",
        "",
        f"- Snapshot: `hard_910277__mid_body_early__capture.pkl` after complete C lift+12S.",
        f"- DX={frozen['frozen_geometry'].get('dx'):.3f} mm, DZ={frozen['frozen_geometry'].get('dz'):.3f} mm, "
        f"dθ={frozen['frozen_geometry'].get('dtheta'):.3f}°, e_pos={frozen['frozen_geometry'].get('e_pos'):.3f} mm",
        f"- grasp_kind=`{frozen['frozen_geometry'].get('grasp_kind')}`, "
        f"contact_s_mean={frozen['frozen_geometry'].get('contact_s_mean')}",
        f"- assess_need: `{json.dumps(jsafe(frozen['assess_need']), indent=2)}`",
        "",
        "## Skills requested vs executed (P0)",
        "",
        f"- Requested: {p0['skills_requested']}",
        f"- Executed: {p0['skills_executed']}",
        f"- Unnecessary DX triggered: **{p0.get('unnecessary_dx')}** "
        f"(dx already small: {p0.get('dx_already_small')})",
        f"- Phases: `{' | '.join(p0.get('phases') or [])}`",
        "",
        "## Failure",
        "",
        f"- P0 SUCCESS_12S={p0.get('SUCCESS_12S')} SUCCESS_NEW={p0.get('SUCCESS_NEW')}",
        f"- PRECISION_FAILURE_REASON: `{p0.get('PRECISION_FAILURE_REASON')}`",
        f"- FINAL_POSE_ERROR: `{p0.get('FINAL_POSE_ERROR')}`",
        f"- First worsening signal: `{p0.get('first_worsening')}`",
        "",
        "## Hypothesis for P1",
        "",
        "Only **dθ** is needed (dx/dz already inside AXIS bands; e_pos≈2.5 mm). "
        "P1 runs `correct_dtheta` + final DOWN/handoff only — no DX/DZ primitives.",
        "",
        f"- P1 SUCCESS_12S={p1.get('SUCCESS_12S')} SUCCESS_NEW={p1.get('SUCCESS_NEW')}",
        f"- P1 reason: `{p1.get('PRECISION_FAILURE_REASON')}`",
        f"- P1 FINAL_POSE_ERROR: `{p1.get('FINAL_POSE_ERROR')}`",
        "",
    ]
    (OUT / "910277_PRECISION_FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    pcfg = PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)

    frozen = load_frozen(env)
    (OUT / "raw" / "910277_frozen_state.json").write_text(json.dumps(jsafe({k: v for k, v in frozen.items() if k != "capture"}), indent=2), encoding="utf-8")
    print("frozen need", frozen["assess_need"], flush=True)
    print("frozen geo", {k: frozen["frozen_geometry"].get(k) for k in ("dx", "dz", "dtheta", "e_pos", "grasp_kind")}, flush=True)

    print("P0 full Precision...", flush=True)
    p0 = run_p0(env, frozen, pcfg)
    np.savez_compressed(OUT / "raw" / "910277__P0__actions.npz", actions=np.asarray(p0.pop("actions"), np.float32))
    print("P0", p0.get("SUCCESS_NEW"), p0.get("PRECISION_FAILURE_REASON"), p0.get("phases")[:12], flush=True)

    print("P1 dtheta-only...", flush=True)
    p1 = run_p1_dtheta_only(env, frozen, pcfg)
    np.savez_compressed(OUT / "raw" / "910277__P1__actions.npz", actions=np.asarray(p1.pop("actions"), np.float32))
    print("P1", p1.get("SUCCESS_NEW"), p1.get("PRECISION_FAILURE_REASON"), flush=True)

    # Trace CSV
    trace_rows = []
    for label, pack in (("P0", p0), ("P1", p1)):
        for stage, err in (("start", pack["errors0"]), ("end", pack["errors1"])):
            trace_rows.append(
                {
                    "hard_id": HID,
                    "variant": label,
                    "stage": stage,
                    "dx": err.get("dx"),
                    "dy": err.get("dy"),
                    "dz": err.get("dz"),
                    "dtheta": err.get("dtheta"),
                    "e_pos": err.get("e_pos"),
                    "bilateral": err.get("bilateral"),
                    "SUCCESS_12S": pack.get("SUCCESS_12S") if stage == "end" else None,
                    "SUCCESS_NEW": pack.get("SUCCESS_NEW") if stage == "end" else None,
                    "PRECISION_FAILURE_REASON": pack.get("PRECISION_FAILURE_REASON") if stage == "end" else None,
                    "phases": "|".join(pack.get("phases") or []) if stage == "end" else None,
                    "skills": "|".join(pack.get("PRECISION_SKILLS_EXECUTED") or []) if stage == "end" else None,
                }
            )
    keys = []
    for r in trace_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / "910277_PRECISION_TRACE.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in trace_rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})

    cmp_rows = [
        {
            "hard_id": HID,
            "variant": "P0",
            "hypothesis": "unchanged_precision",
            "SUCCESS_12S": p0.get("SUCCESS_12S"),
            "SUCCESS_NEW": p0.get("SUCCESS_NEW"),
            "PRECISION_FAILURE_REASON": p0.get("PRECISION_FAILURE_REASON"),
            "final_dx": (p0.get("FINAL_POSE_ERROR") or {}).get("dx"),
            "final_dz": (p0.get("FINAL_POSE_ERROR") or {}).get("dz"),
            "final_dtheta": (p0.get("FINAL_POSE_ERROR") or {}).get("dtheta"),
            "final_e_pos": (p0.get("FINAL_POSE_ERROR") or {}).get("e_pos"),
            "skills": "|".join(p0.get("PRECISION_SKILLS_EXECUTED") or []),
            "unnecessary_dx": p0.get("unnecessary_dx"),
            "n_actions": p0.get("n_actions"),
        },
        {
            "hard_id": HID,
            "variant": "P1",
            "hypothesis": p1.get("hypothesis"),
            "SUCCESS_12S": p1.get("SUCCESS_12S"),
            "SUCCESS_NEW": p1.get("SUCCESS_NEW"),
            "PRECISION_FAILURE_REASON": p1.get("PRECISION_FAILURE_REASON"),
            "final_dx": (p1.get("FINAL_POSE_ERROR") or {}).get("dx"),
            "final_dz": (p1.get("FINAL_POSE_ERROR") or {}).get("dz"),
            "final_dtheta": (p1.get("FINAL_POSE_ERROR") or {}).get("dtheta"),
            "final_e_pos": (p1.get("FINAL_POSE_ERROR") or {}).get("e_pos"),
            "skills": "|".join(p1.get("PRECISION_SKILLS_EXECUTED") or []),
            "unnecessary_dx": False,
            "n_actions": p1.get("n_actions"),
        },
    ]
    keys2 = list(cmp_rows[0].keys())
    with (OUT / "P0_P1_COMPARISON.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys2)
        w.writeheader()
        for r in cmp_rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys2})

    write_analysis(frozen, p0, p1)
    (OUT / "raw" / "910277_p0_p1.json").write_text(json.dumps(jsafe({"P0": p0, "P1": p1}), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
