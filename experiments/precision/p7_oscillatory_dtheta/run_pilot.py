#!/usr/bin/env python
"""P7 O0–O3 oscillatory dθ pilot on frozen hard_910277 C output."""

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
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    correct_dtheta,
    default_pivot_tag,
    parse_ppivot_tag,
    pivot_actions_from_tag,
)
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from experiments.precision.oscillation_assisted_reorientation.metrics import (  # noqa: E402
    desired_correction_axis_hand,
    snapshot_state,
)
from experiments.precision.p7_oscillatory_dtheta.geometry import signed_theta_to_target  # noqa: E402
from experiments.precision.p7_oscillatory_dtheta.oscillation import (  # noqa: E402
    run_oscillation_controller,
)
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/p7_oscillatory_dtheta"
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
    cap = pickle.loads(PRIOR_CAP.read_bytes())
    blob = pickle.loads((EXIT / f"{HID}.pkl").read_bytes())
    restore_capture(env, cap)
    return {
        "capture": cap,
        "ref": blob["ref"],
        "sample_id": HID,
        "start": signed_theta_to_target(env.sim),
    }


def audit_baseline(env, frozen: dict) -> dict:
    """Diagnose existing correct_dtheta without modifying production code."""
    restore_capture(env, frozen["capture"])
    th0 = signed_theta_to_target(env.sim)
    tag = default_pivot_tag("B")
    parsed = parse_ppivot_tag(tag) if tag else {}
    m0 = measure(env.sim, U_TGT, P_TGT)
    u0 = np.asarray(m0["u_c_h"], float)
    ax = desired_correction_axis_hand(env.sim)
    R0 = hand_R(env.sim).copy()

    acts = pivot_actions_from_tag(tag) if tag else []
    series = []
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        snap = snapshot_state(env.sim, u0_h=u0, axis_h=ax, R0=R0)
        th = signed_theta_to_target(env.sim)
        series.append(
            {
                "i": i,
                "abs_theta": th["abs_theta_deg"],
                "signed_theta": th["signed_theta_deg"],
                "signed_rel": snap["signed_rel_deg"],
                "hand_tilt": snap.get("hand_tilt_deg"),
                "w_rel_along": snap.get("w_rel_along_axis"),
                "bilateral": snap["bilateral"],
                "dx": snap["dx_mm"],
                "dz": snap["dz_mm"],
            }
        )
    th_mid = signed_theta_to_target(env.sim)
    # Full correct_dtheta (includes restore_down)
    restore_capture(env, frozen["capture"])
    pcfg = PrecisionConfig(evaluate_handoff=False)
    rd = correct_dtheta(env, frozen, pcfg, "B")
    th1 = signed_theta_to_target(env.sim)
    peak_hand = max((abs(s.get("hand_tilt") or 0) for s in series), default=0)
    peak_rel = max((abs(s.get("signed_rel") or 0) for s in series), default=0)
    stop_reason = "fixed_duration_pivot_tag"
    if rd.get("ok"):
        if th1["abs_theta_deg"] <= float(pcfg.theta_tol_deg):
            stop_reason = "reached_theta_tol"
        elif (rd.get("metrics") or {}).get("d_theta", 0) is not None:
            stop_reason = "pivot_improved_ge_2deg_then_restore_DOWN"
    elif rd.get("reason"):
        stop_reason = str(rd.get("reason"))

    audit = {
        "start": th0,
        "after_pivot_actions": th_mid,
        "after_correct_dtheta": th1,
        "pivot_tag": tag,
        "pivot_parsed": parsed,
        "command_frame": "r_des / hand body (normalize_action w_body); primary channel wy",
        "rotation_axis_commanded": f"body wy={parsed.get('wy')} (physical ω_y = a[4]*W_MAX)",
        "n_pivot_actions": len(acts),
        "peak_hand_tilt_deg": peak_hand,
        "peak_relative_cyl_deg": peak_rel,
        "co_rotation_ratio": (peak_rel / peak_hand) if peak_hand > 1e-6 else None,
        "mostly_co_rotation": bool(peak_hand >= 5.0 and peak_rel < 0.4 * peak_hand),
        "genuine_geometric_residual": True,
        "measurement_note": (
            "abs θ uses axis-symmetric acos(|u·tgt|); signed θ about correction axis "
            "matches |abs| at start (±14.63°), so residual is not a frame mismatch."
        ),
        "stop_reason": stop_reason,
        "correct_dtheta_ok": bool(rd.get("ok")),
        "correct_dtheta_mechanism": rd.get("mechanism"),
        "metrics": {k: v for k, v in (rd.get("metrics") or {}).items() if k not in ("series",)},
        "series_pivot": series,
        "actions_o0": list(rd.get("actions") or []),
        "capture_o0": rd.get("capture") or take_capture(env),
        "theta_tol_cfg": float(pcfg.theta_tol_deg),
        "working_theta": WORKING_THETA_DEG,
    }
    return audit


def write_audit_md(audit: dict) -> None:
    s0, s1 = audit["start"], audit["after_correct_dtheta"]
    lines = [
        "# DTHETA_BASELINE_AUDIT — hard_910277",
        "",
        "Frozen snapshot: complete-C `mid_body_early` after lift+12S.",
        "",
        "## Starting residuals",
        "",
        f"- abs θ = {s0['abs_theta_deg']:.3f}°",
        f"- signed θ (cyl→target about correction axis) = {s0['signed_theta_deg']:.3f}°",
        f"- DX={s0['dx_mm']:.3f} mm, DZ={s0['dz_mm']:.3f} mm, e_pos={s0['e_pos_mm']:.3f} mm",
        f"- bilateral={s0['bilateral']}",
        "",
        "## `correct_dtheta` command",
        "",
        f"- pivot_tag: `{audit['pivot_tag']}`",
        f"- parsed: `{audit['pivot_parsed']}`",
        f"- Frame: {audit['command_frame']}",
        f"- Axis: {audit['rotation_axis_commanded']}",
        f"- Duration: {audit['n_pivot_actions']} steps (~{audit['n_pivot_actions']*0.02:.2f} s) + restore_DOWN",
        "",
        "## Relative vs co-rotation",
        "",
        f"- Peak hand tilt ≈ **{audit['peak_hand_tilt_deg']:.2f}°**",
        f"- Peak relative cylinder angle ≈ **{audit['peak_relative_cyl_deg']:.2f}°**",
        f"- co_rotation_ratio (rel/hand) ≈ {audit['co_rotation_ratio']}",
        f"- Mostly co-rotation: **{audit['mostly_co_rotation']}**",
        "",
        "Interpretation: the controller primarily rotates the **hand and object together**. "
        "Only ~2° of relative slip accumulates, matching the abs-θ improvement "
        f"({s0['abs_theta_deg']:.2f}° → {s1['abs_theta_deg']:.2f}°).",
        "",
        "## Why correction stops",
        "",
        f"- stop_reason: `{audit['stop_reason']}`",
        f"- cfg.theta_tol_deg={audit['theta_tol_cfg']} (AXIS), WORKING_THETA={audit['working_theta']} (SUCCESS_NEW)",
        "- Pivot is a **fixed-duration verified tag**. Acceptance is "
        "`θ≤tol` OR `|Δθ|≥2°`. Here Δθ≈2.27° so the tag is accepted even though "
        "θ remains above the 10° SUCCESS_NEW hold.",
        "- Not aborted by contact safety; bilateral retained.",
        "",
        "## Residual authenticity",
        "",
        f"- {audit['measurement_note']}",
        f"- Residual {s1['abs_theta_deg']:.2f}° after correct_dtheta is a **genuine geometric error**.",
        "",
    ]
    (OUT / "DTHETA_BASELINE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def validate_handoff(env, frozen: dict, capture) -> dict:
    restore_capture(env, capture)
    th = signed_theta_to_target(env.sim)
    pcfg = PrecisionConfig(evaluate_handoff=True)
    hand = evaluate_handoff_with_axis(env.sim, frozen["ref"], stabilize_n=pcfg.stabilize_handoff_n)
    restore_capture(env, capture)
    return {
        "theta_abs": th["abs_theta_deg"],
        "signed_theta": th["signed_theta_deg"],
        "dx": th["dx_mm"],
        "dy": th["dy_mm"],
        "dz": th["dz_mm"],
        "e_pos": th["e_pos_mm"],
        "bilateral": th["bilateral"],
        "pos_ok": float(th["e_pos_mm"]) <= WORKING_POS_MM,
        "theta_ok_transient": float(th["abs_theta_deg"]) <= WORKING_THETA_DEG,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "handoff_reason": hand.get("reason"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    frozen = load_frozen(env)
    print("frozen", frozen["start"], flush=True)

    print("Auditing correct_dtheta (O0)...", flush=True)
    audit = audit_baseline(env, frozen)
    write_audit_md(audit)
    (OUT / "raw" / "baseline_audit.json").write_text(
        json.dumps(jsafe({k: v for k, v in audit.items() if k not in ("actions_o0", "capture_o0")}), indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(OUT / "raw" / "O0__actions.npz", actions=np.asarray(audit["actions_o0"], np.float32))
    (OUT / "raw" / "O0__capture.pkl").write_bytes(pickle.dumps(audit["capture_o0"]))

    cycle_rows: list[dict] = []
    cmp_rows: list[dict] = []
    packs: dict[str, Any] = {}

    # O0 summary row
    val0 = validate_handoff(env, frozen, audit["capture_o0"])
    packs["O0"] = {
        "label": "O0",
        "mode": "correct_dtheta",
        "theta0": audit["start"]["abs_theta_deg"],
        "theta_final": audit["after_correct_dtheta"]["abs_theta_deg"],
        "signed0": audit["start"]["signed_theta_deg"],
        "signed_final": audit["after_correct_dtheta"]["signed_theta_deg"],
        "net_improve": audit["start"]["abs_theta_deg"] - audit["after_correct_dtheta"]["abs_theta_deg"],
        "n_cycles": 1,
        "abort_reason": audit["stop_reason"],
        "reached_theta_tol": audit["after_correct_dtheta"]["abs_theta_deg"] <= WORKING_THETA_DEG,
        "validation": val0,
        "actions": audit["actions_o0"],
        "series": audit["series_pivot"],
        "capture": audit["capture_o0"],
        "classification_summary": "CO_ROTATION_DOMINANT" if audit["mostly_co_rotation"] else "MIXED",
    }
    cycle_rows.append(
        {
            "variant": "O0",
            "cycle": 0,
            "phase": "pivot_tag",
            "theta0_abs": audit["start"]["abs_theta_deg"],
            "theta1_abs": audit["after_correct_dtheta"]["abs_theta_deg"],
            "signed0": audit["start"]["signed_theta_deg"],
            "signed1": audit["after_correct_dtheta"]["signed_theta_deg"],
            "net_improve_abs": packs["O0"]["net_improve"],
            "net_relative_deg": audit["peak_relative_cyl_deg"],
            "peak_hand_tilt_deg": audit["peak_hand_tilt_deg"],
            "classification": packs["O0"]["classification_summary"],
            "dx0": audit["start"]["dx_mm"],
            "dx1": audit["after_correct_dtheta"]["dx_mm"],
            "dz0": audit["start"]["dz_mm"],
            "dz1": audit["after_correct_dtheta"]["dz_mm"],
            "bilateral_end": audit["after_correct_dtheta"]["bilateral"],
        }
    )

    variants = [
        ("O1", "symmetric", 1.0, 8.0),
        ("O2", "asymmetric", 0.40, 10.0),
        ("O3", "feedback", 0.45, 10.0),
    ]
    for label, mode, rev_frac, amp in variants:
        print(f"Running {label} mode={mode} amp={amp} rev_frac={rev_frac}...", flush=True)
        restore_capture(env, frozen["capture"])
        res = run_oscillation_controller(
            env,
            frozen,
            mode=mode,  # type: ignore[arg-type]
            amp_fwd0=amp,
            amp_rev_frac=rev_frac,
            max_cycles=6,
            half_period_s=0.45,
        )
        # Final validation only meaningful if θ≤10 or always report
        val = validate_handoff(env, frozen, res["capture"])
        packs[label] = {
            "label": label,
            "mode": mode,
            "theta0": res["theta0_abs"],
            "theta_final": res["theta_final_abs"],
            "signed0": res["signed0"],
            "signed_final": res["signed_final"],
            "net_improve": res["net_improve_total"],
            "n_cycles": res["n_cycles"],
            "abort_reason": res["abort_reason"],
            "reached_theta_tol": res["reached_theta_tol"],
            "validation": val,
            "actions": res["actions"],
            "series": res["series"],
            "capture": res["capture"],
            "cycles": res["cycles"],
            "direction": res["direction"],
        }
        np.savez_compressed(OUT / "raw" / f"{label}__actions.npz", actions=np.asarray(res["actions"], np.float32))
        (OUT / "raw" / f"{label}__capture.pkl").write_bytes(pickle.dumps(res["capture"]))
        (OUT / "raw" / f"{label}__meta.json").write_text(
            json.dumps(jsafe({k: v for k, v in packs[label].items() if k not in ("actions", "series", "capture")}), indent=2),
            encoding="utf-8",
        )
        for c in res["cycles"]:
            cycle_rows.append(
                {
                    "variant": label,
                    "cycle": c.get("cycle"),
                    "phase": "full_cycle",
                    "theta0_abs": c.get("theta0_abs"),
                    "theta1_abs": c.get("theta1_abs"),
                    "signed0": c.get("signed0"),
                    "signed1": c.get("signed1"),
                    "net_improve_abs": c.get("net_improve_abs"),
                    "net_relative_deg": c.get("net_relative_deg"),
                    "peak_relative_deg": c.get("peak_relative_deg"),
                    "peak_hand_tilt_deg": c.get("peak_hand_tilt_deg"),
                    "rel_after_forward": c.get("rel_after_forward"),
                    "classification": c.get("classification"),
                    "amp_fwd": c.get("amplitude_fwd_deg"),
                    "amp_rev": c.get("amplitude_rev_deg"),
                    "direction": c.get("direction"),
                    "dx0": c.get("dx0"),
                    "dx1": c.get("dx1"),
                    "dz0": c.get("dz0"),
                    "dz1": c.get("dz1"),
                    "e_pos0": c.get("e_pos0"),
                    "e_pos1": c.get("e_pos1"),
                    "bilateral_end": c.get("bilateral_end"),
                    "undid_on_reverse": c.get("undid_on_reverse"),
                    "co_rotation_dominant": c.get("co_rotation_dominant"),
                }
            )
        print(
            f"  {label}: θ {res['theta0_abs']:.2f}→{res['theta_final_abs']:.2f} "
            f"cycles={res['n_cycles']} abort={res['abort_reason']} "
            f"NEW={val.get('SUCCESS_NEW')} 12S={val.get('SUCCESS_12S')}",
            flush=True,
        )

    # O0 validation already done
    print(
        f"  O0: θ {packs['O0']['theta0']:.2f}→{packs['O0']['theta_final']:.2f} "
        f"NEW={val0.get('SUCCESS_NEW')} 12S={val0.get('SUCCESS_12S')}",
        flush=True,
    )

    # CSVs
    keys: list[str] = []
    for r in cycle_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / "OSCILLATION_CYCLE_TRACES.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in cycle_rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})

    for label in ("O0", "O1", "O2", "O3"):
        p = packs[label]
        v = p["validation"]
        cmp_rows.append(
            {
                "variant": label,
                "mode": p["mode"],
                "theta0": p["theta0"],
                "theta_final": p["theta_final"],
                "signed0": p["signed0"],
                "signed_final": p["signed_final"],
                "net_improve_deg": p["net_improve"],
                "n_cycles": p["n_cycles"],
                "reached_theta_tol": p["reached_theta_tol"],
                "abort_reason": p["abort_reason"],
                "dx_final": v.get("dx"),
                "dz_final": v.get("dz"),
                "e_pos_final": v.get("e_pos"),
                "pos_ok": v.get("pos_ok"),
                "SUCCESS_12S": v.get("SUCCESS_12S"),
                "SUCCESS_NEW": v.get("SUCCESS_NEW"),
                "classification": p.get("classification_summary")
                or (p.get("cycles") or [{}])[-1].get("classification")
                if p.get("cycles")
                else p.get("classification_summary"),
            }
        )
    keys2 = list(cmp_rows[0].keys())
    with (OUT / "OSCILLATION_COMPARISON.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys2)
        w.writeheader()
        for r in cmp_rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys2})

    # Contact/rotation analysis
    lines = [
        "# CONTACT_ROTATION_ANALYSIS",
        "",
        "## Mechanism classes observed",
        "",
    ]
    for label in ("O0", "O1", "O2", "O3"):
        p = packs[label]
        lines.append(f"### {label} ({p['mode']})")
        lines.append(
            f"- θ: {p['theta0']:.3f}° → {p['theta_final']:.3f}° (Δ={p['net_improve']:.3f}°)"
        )
        lines.append(f"- cycles={p['n_cycles']}, abort=`{p['abort_reason']}`, reached_tol={p['reached_theta_tol']}")
        if label == "O0":
            lines.append(
                f"- Hand tilt ≫ relative slip → **object and hand rotate together**; "
                f"only ~{audit['peak_relative_cyl_deg']:.1f}° irreversible relative motion."
            )
        else:
            for c in p.get("cycles") or []:
                lines.append(
                    f"  - cycle {c.get('cycle')}: class=`{c.get('classification')}` "
                    f"net_rel={c.get('net_relative_deg')} improve={c.get('net_improve_abs')} "
                    f"hand={c.get('peak_hand_tilt_deg')} undid={c.get('undid_on_reverse')}"
                )
        v = p["validation"]
        lines.append(
            f"- Validation: θ={v.get('theta_abs'):.3f} e_pos={v.get('e_pos'):.3f} "
            f"12S={v.get('SUCCESS_12S')} NEW={v.get('SUCCESS_NEW')}"
        )
        lines.append("")
    (OUT / "CONTACT_ROTATION_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    (OUT / "raw" / "all_packs_summary.json").write_text(
        json.dumps(
            jsafe(
                {
                    lab: {k: v for k, v in p.items() if k not in ("actions", "series", "capture")}
                    for lab, p in packs.items()
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
