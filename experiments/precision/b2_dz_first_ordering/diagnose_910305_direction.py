#!/usr/bin/env python
"""Sign probes + instrumented slip trace for hard_910305 wrong-direction DZ."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    DT,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.mechanics.finger_vertical import U_OPEN_HAND  # noqa: E402
from data_generation.oracles.precision.corrections.orientation import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import (  # noqa: E402
    A0_SLIP,
    FN_INIT,
    FN_MAX,
    FN_MIN,
    FORCE_DA,
    FORCE_KP,
    PrecisionConfig,
    SLIP_KN,
    V_SLIP,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.b2_dz_first_ordering.dz_controller import run_zero_target_dz  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/dz_wrong_direction_910305"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits/hard_910305.pkl"


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def probe_slip_direction(env, blob, *, d_hand: np.ndarray, n_steps: int = 25) -> dict:
    """Short legal DualLoop with fixed desire d; measure Δez and disp along d."""
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    st0 = sample_state(env.sim)
    p0 = np.asarray(st0["p_c_h"], float)
    d = np.asarray(d_hand, float)
    d = d / float(np.linalg.norm(d))
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    # Match production / experimental: slip_axis = -d
    ctl.reset(a0=A0_SLIP, fn0=st0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))
    for _ in range(n_steps):
        st = sample_state(env.sim)
        _, meta = ctl.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        if not sample_state(env.sim).get("bilateral"):
            break
    err1 = measure_errors(env.sim)
    p1 = np.asarray(sample_state(env.sim)["p_c_h"], float)
    disp = float(np.dot(p1 - p0, d)) * 1e3
    return {
        "d_hand": d.tolist(),
        "slip_axis": (-d).tolist(),
        "dz0": float(err0["dz_mm"]),
        "dz1": float(err1["dz_mm"]),
        "delta_dz": float(err1["dz_mm"] - err0["dz_mm"]),
        "disp_along_d_mm": disp,
        "abs_err0": abs(float(err0["dz_mm"])),
        "abs_err1": abs(float(err1["dz_mm"])),
        "improved_abs": abs(float(err1["dz_mm"])) < abs(float(err0["dz_mm"])) - 0.05,
        "bilateral": bool(err1.get("bilateral")),
        "p_c_h0": p0.tolist(),
        "p_c_h1": p1.tolist(),
        "e_signed0": np.asarray(err0["e_signed_mm"], float).tolist(),
        "U_OPEN_HAND": np.asarray(U_OPEN_HAND, float).tolist(),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    blob = pickle.loads(EXIT.read_bytes())
    pcfg = PrecisionConfig(use_verified_lateral_fallback=False)
    u = np.asarray(U_OPEN_HAND, float)
    u = u / float(np.linalg.norm(u))

    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    st0 = sample_state(env.sim)
    conventions = {
        "dz_mm_entry": float(err0["dz_mm"]),
        "dx_mm_entry": float(err0["dx_mm"]),
        "e_signed_mm": np.asarray(err0["e_signed_mm"], float).tolist(),
        "p_c_h": st0["p_c_h"],
        "U_OPEN_HAND": u.tolist(),
        "production_inward_desire_d": (-u).tolist(),
        "experimental_choice_when_ez_gt_0": "d=-U_OPEN (same as production inward)",
        "slip_axis_set_to": "(-d) so for inward d=-U_OPEN => slip_axis=+U_OPEN",
        "vx_target": float(V_SLIP),
        "vx_target_meaning": "DualLoop seeks v_rel·slip_axis ≈ vx_target (<0)",
        "disp_definition": "disp=(p_c_h-p0)·d [mm]",
        "hypothesis": (
            "If Δez and disp_along_d have opposite implications for error reduction, "
            "either e_z sign vs p_c_h[2] mismatch or slip_axis=-d double-flip is wrong for this frame."
        ),
    }

    probes = {
        "desire_minus_U_OPEN_slip_axis_plus_U": probe_slip_direction(env, blob, d_hand=-u),
        "desire_plus_U_OPEN_slip_axis_minus_U": probe_slip_direction(env, blob, d_hand=+u),
    }
    # Which desire actually reduces |dz|?
    best = min(probes.items(), key=lambda kv: kv[1]["abs_err1"] if kv[1]["bilateral"] else 1e9)
    conventions["probe_winner"] = best[0]
    conventions["probes"] = probes

    # Full experimental slip with dense logging (current controller)
    restore_capture(env, blob["capture"])
    tl: list = []
    res = run_zero_target_dz(env, blob, pcfg, dz_target_mm=0.0, action_budget_left=2500, timeline=tl, label="diagnose")
    # Enrich timeline with desire direction used at start
    ez0 = float(err0["dz_mm"])
    d_used = (-u if ez0 >= 0 else u).tolist()
    for row in tl:
        row["desired_d"] = d_used
        row["abs_dz"] = abs(float(row.get("dz_mm") or 0))

    with (OUT / "SIGNED_MOTION_TRACE.csv").open("w", encoding="utf-8", newline="") as f:
        if tl:
            keys = list(tl[0].keys())
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in tl:
                w.writerow(row)

    # Analyze when error grows
    growth_phases = []
    prev = None
    for row in tl:
        if prev is not None:
            ddz = float(row["dz_mm"]) - float(prev["dz_mm"])
            if abs(float(row["dz_mm"])) > abs(float(prev["dz_mm"])) + 0.05:
                growth_phases.append(
                    {
                        "t": row["t"],
                        "phase": row["phase"],
                        "dz": row["dz_mm"],
                        "ddz": ddz,
                        "disp": row.get("disp_along_mm"),
                    }
                )
        prev = row

    diagnosis = {
        "conventions": conventions,
        "slip_result": {k: v for k, v in res.items() if k not in ("actions", "timeline")},
        "n_timeline": len(tl),
        "dz_start": float(err0["dz_mm"]),
        "dz_end": float((res.get("errors1") or measure_errors(env.sim))["dz_mm"]),
        "n_attempts": res.get("n_attempts"),
        "stop_reason": res.get("stop_reason") or res.get("reason"),
        "error_growth_events": growth_phases[:40],
        "n_growth_events": len(growth_phases),
        "root_cause_candidates": [],
    }

    # Root cause classification from probes
    p_in = probes["desire_minus_U_OPEN_slip_axis_plus_U"]
    p_out = probes["desire_plus_U_OPEN_slip_axis_minus_U"]
    if p_in["improved_abs"] and not p_out["improved_abs"]:
        diagnosis["root_cause_candidates"].append(
            "SIGN_OK_production_inward_desire_reduces_abs_dz; worsening elsewhere (brake/retry/order)"
        )
    elif p_out["improved_abs"] and not p_in["improved_abs"]:
        diagnosis["root_cause_candidates"].append(
            "SIGN_INVERTED: experimental d=-U_OPEN when ez>0 moves error wrong; should use d=+U_OPEN"
        )
    elif not p_in["improved_abs"] and not p_out["improved_abs"]:
        diagnosis["root_cause_candidates"].append(
            "NEITHER_SHORT_PROBE_IMPROVES — contact/geometry limited or DualLoop not producing axial slip"
        )
    else:
        diagnosis["root_cause_candidates"].append("BOTH_DIRECTIONS_IMPROVE_SHORT — inconclusive")

    # Correlation: disp_along_d vs delta_ez during slip rows
    slip_rows = [r for r in tl if str(r.get("phase")).startswith("slip")]
    if len(slip_rows) >= 3:
        dzs = np.array([float(r["dz_mm"]) for r in slip_rows])
        disps = np.array([float(r.get("disp_along_mm") or 0) for r in slip_rows])
        # If disp increases while |dz| increases with same-sign ez, desire points wrong
        diagnosis["slip_span"] = {
            "dz0": float(dzs[0]),
            "dz1": float(dzs[-1]),
            "disp0": float(disps[0]),
            "disp1": float(disps[-1]),
            "corr_disp_dz": float(np.corrcoef(disps, dzs)[0, 1]) if np.std(disps) > 1e-6 else None,
        }
        if abs(dzs[-1]) > abs(dzs[0]) + 1.0 and disps[-1] > disps[0] + 1.0:
            diagnosis["root_cause_candidates"].append(
                "ACTIVE_SLIP_GROWS_ABS_ERROR_WHILE_DISP_ALONG_d_INCREASES — wrong desire or wrong ez frame"
            )

    # Why single attempt
    diagnosis["retry_analysis"] = {
        "code_path": (
            "run_zero_target_dz returns immediately if attempt ok=False (e.g. contact_loss); "
            "only retries when stop_reason==NO_PROGRESS; TIME_BUDGET/GOAL/WRONG_DIR end the loop"
        ),
        "this_run_n_attempts": res.get("n_attempts"),
        "this_run_ok": res.get("ok"),
        "this_run_reason": res.get("reason") or res.get("stop_reason"),
    }

    (OUT / "DIAGNOSIS.json").write_text(json.dumps(diagnosis, indent=2, default=str), encoding="utf-8")

    # Markdown
    md = [
        "# DZ Wrong-Direction Diagnosis — hard_910305",
        "",
        "## S1 ordering fact",
        "",
        "Prior S1 phases were `entry_settle → dz_coarse_zero_target` then `dz_coarse:contact_loss`.",
        "**DZ-first did execute** (no `fallback_dx_before_dz_due_to_safety`).",
        "",
        "## Sign probes (identical Stage-1 exit)",
        "",
        f"- Entry dz = {conventions['dz_mm_entry']:.3f} mm (signed; HUD may match this, not −23..−33 unless different frame).",
        f"- Probe winner: **{best[0]}**",
        f"- d=-U_OPEN: Δdz={p_in['delta_dz']:.3f}, disp={p_in['disp_along_d_mm']:.3f}, improved={p_in['improved_abs']}",
        f"- d=+U_OPEN: Δdz={p_out['delta_dz']:.3f}, disp={p_out['disp_along_d_mm']:.3f}, improved={p_out['improved_abs']}",
        "",
        "## Instrumented slip",
        "",
        f"- stop/reason: `{diagnosis['stop_reason']}`",
        f"- n_attempts: {diagnosis['n_attempts']}",
        f"- dz {diagnosis['dz_start']:.3f} → {diagnosis['dz_end']:.3f}",
        f"- growth events: {diagnosis['n_growth_events']}",
        f"- slip_span: {diagnosis.get('slip_span')}",
        "",
        "## Root-cause candidates",
        "",
    ]
    for c in diagnosis["root_cause_candidates"]:
        md.append(f"- {c}")
    md += [
        "",
        "## No-progress / wrong-way continuation",
        "",
        "Stall counter only resets when `|ez-tgt|` improves by >0.05 mm. Error **growth** still",
        "counts as stall, but slip continues up to ~2.5 s while `disp>0.5`. That permits prolonged",
        "motion away from zero before `STOP_NO_PROGRESS`.",
        "",
        "## Retry",
        "",
        str(diagnosis["retry_analysis"]),
        "",
        "Artifacts: `SIGNED_MOTION_TRACE.csv`, `DIAGNOSIS.json`.",
        "",
    ]
    (OUT / "ROOT_CAUSE.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({k: diagnosis[k] for k in ("dz_start", "dz_end", "stop_reason", "n_attempts", "root_cause_candidates", "slip_span")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
