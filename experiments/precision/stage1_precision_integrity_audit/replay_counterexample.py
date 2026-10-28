#!/usr/bin/env python
"""Reproduce PILOT_A__ball_train_v2_0012 counterexample with full contact/state trace."""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.entry_conditions import entry_gate  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import DT, sample_state  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _monitor  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.recovery_execution import _escaped  # noqa: E402

OUT = ROOT / "results/diagnostics/stage1_precision_integrity_audit"
SID = "PILOT_A__ball_train_v2_0012__93a263b2b1a49030"
IC = ROOT / f"results/data_generation/stage2_abc_final_v2/ics/{SID}.pkl"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = pickle.loads(IC.read_bytes())
    prefix = blob.get("stabilization_prefix_actions") or []

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    # --- A: capture as stored ---
    restore_capture(env, blob["capture"])
    cap0 = take_capture(env)
    phys0 = {
        **{k: measure_errors(env.sim).get(k) for k in ("dtheta_deg", "e_pos_mm", "e_signed_mm", "bilateral")},
        "fn_tot": sample_state(env.sim)["fn_tot"],
        "mode": classify_contact_mode(env.sim).get("mode"),
        "gates": gates(env.sim),
    }
    restore_capture(env, blob["capture"])
    gate = entry_gate(env)

    # --- B: does replaying prefix duplicate / change state? ---
    restore_capture(env, blob["capture"])
    before = np.asarray(env.sim.data.qpos, float).copy()
    for a in prefix:
        step_legal(env.sim, np.asarray(a, np.float32))
    after = np.asarray(env.sim.data.qpos, float).copy()
    prefix_delta = float(np.max(np.abs(after - before)))

    # --- C: 12S from exact capture, stabilize_n=0, instrumented ---
    restore_capture(env, blob["capture"])
    apply_handoff(env.sim, blob["ref"])
    t0 = float(env.sim.data.time)
    series = []
    lost = 0
    reason = None
    while reason is None and float(env.sim.data.time) - t0 < float(NOMINAL_S) - 1e-9:
        nominal_step(env.sim)
        mon = _monitor(env.sim)
        bil = _bilateral(mon)
        st = sample_state(env.sim)
        mode = classify_contact_mode(env.sim)
        t_rel = float(env.sim.data.time) - t0
        series.append(
            {
                "t_rel": t_rel,
                "bilateral": bil,
                "nL": mon["nL"],
                "nR": mon["nR"],
                "z": mon["z"],
                "fn_tot": st["fn_tot"],
                "fn_L": st["fn_L"],
                "fn_R": st["fn_R"],
                "dtheta_deg": measure_errors(env.sim)["dtheta_deg"],
                "e_signed_mm": measure_errors(env.sim)["e_signed_mm"],
                "v_rel": st["v_rel_norm"],
                "w_rel": st["w_rel_norm"],
                "mode": mode.get("mode"),
                "Fn_z_L": mode.get("Fn_z_L"),
                "Fn_z_R": mode.get("Fn_z_R"),
                "latched": bool(latched(env.sim)),
            }
        )
        if latched(env.sim):
            reason = "GROUND_DROP"
            break
        if _escaped(env.sim):
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bil else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
    if reason is None:
        mon = _monitor(env.sim)
        reason = "SUCCESS_12S" if _bilateral(mon) else "LOST_GRASP"

    fail_t = next((s["t_rel"] for s in series if not s["bilateral"]), None)
    lost_start = None
    streak = 0
    for s in series:
        if not s["bilateral"]:
            streak += 1
            if lost_start is None:
                lost_start = s["t_rel"]
            if streak >= LOST_STEPS:
                break
        else:
            streak = 0
            lost_start = None

    # Official evaluator cross-check
    restore_capture(env, blob["capture"])
    h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=0)

    report = {
        "sample_id": SID,
        "ic_path": str(IC.relative_to(ROOT)).replace("\\", "/"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "blob_keys": sorted(blob.keys()),
        "n_stabilization_prefix_actions": len(prefix),
        "prefix_replay_max_abs_qpos_delta": prefix_delta,
        "prefix_already_in_capture_likely": bool(prefix_delta < 1e-4 or len(prefix) == 0),
        "note_prefix": (
            "If delta is large, prefix is NOT already applied in capture; "
            "OAR/Precision restore capture alone and do not replay prefix."
        ),
        "physics_at_capture": phys0,
        "entry_gate": {
            "ok": bool(gate.get("ok")),
            "stability": gate.get("stability"),
        },
        "instrumented_12s": {
            "NOMINAL_S": NOMINAL_S,
            "LOST_STEPS": LOST_STEPS,
            "reason": reason,
            "SUCCESS_12S": reason == "SUCCESS_12S",
            "first_nonbilateral_t": fail_t,
            "lost_grasp_streak_start_t": lost_start,
            "n_series": len(series),
        },
        "evaluate_handoff_with_axis": {
            "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
            "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
            "reason": h.get("reason"),
        },
        "why_admitted_to_oar": (
            "OAR/LAP/Precision diagnostics select ICs from precision_candidate_audit / "
            "stage2_abc_final_v2 where grasp_stable:=entry_gate(~150-200ms) ∧ bilateral, "
            "NOT SUCCESS_12S. This IC has grasp_stable=True but direct/revalidate SUCCESS_12S=False."
        ),
    }
    (OUT / "COUNTEREXAMPLE_RAW.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (OUT / "logs" / f"counterexample_{SID}_series.json").write_text(
        json.dumps(series, indent=2, default=float), encoding="utf-8"
    )

    md = f"""# COUNTEREXAMPLE_REPLAY — `{SID}`

Generated: {report['generated_utc']}

## Snapshot provenance

- Path: `{report['ic_path']}`
- Blob keys: `{report['blob_keys']}`
- `stabilization_prefix_actions`: **{report['n_stabilization_prefix_actions']}** steps
- Max |Δqpos| if prefix replayed after restore: **{report['prefix_replay_max_abs_qpos_delta']:.6g}**
- Interpretation: {"prefix already reflected / no-op" if report['prefix_replay_max_abs_qpos_delta'] < 1e-3 else "prefix would change state if replayed — capture is pre- or mid-prefix"}

## Stage-1 gate actually used at Precision admission

- `entry_gate` / ~150–200 ms hold: **ok={report['entry_gate']['ok']}**
- Contact at capture: mode=`{phys0.get('mode')}`, bilateral=`{phys0.get('bilateral')}`, fn_tot=`{phys0.get('fn_tot')}`, dθ=`{phys0.get('dtheta_deg')}`

## Independent 12 s nominal test (stabilize_n=0)

| Evaluator | SUCCESS_12S | reason | first non-bilateral t |
|---|---|---|---|
| Instrumented nominal loop | {report['instrumented_12s']['SUCCESS_12S']} | {report['instrumented_12s']['reason']} | {report['instrumented_12s']['first_nonbilateral_t']} |
| `evaluate_handoff_with_axis` | {report['evaluate_handoff_with_axis']['SUCCESS_12S']} | {report['evaluate_handoff_with_axis']['reason']} | — |

LOST_STEPS={LOST_STEPS} (75×20 ms = 1.5 s continuous non-bilateral). Lost-streak start t≈`{lost_start}`.

## Why OAR admitted it

{report['why_admitted_to_oar']}

## Artifacts

- `COUNTEREXAMPLE_RAW.json`
- `logs/counterexample_{SID}_series.json`
"""
    (OUT / "COUNTEREXAMPLE_REPLAY.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
