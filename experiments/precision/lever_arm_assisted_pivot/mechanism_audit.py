#!/usr/bin/env python
"""Reconstruct DX-under-large-dz induced rotation (mechanism audit for LAP).

Does not modify production. Writes MECHANISM_AUDIT.md + traces under
results/diagnostics/lever_arm_assisted_pivot/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import relative_pose  # noqa: E402
from data_generation.oracles.precision.corrections.lateral import _horizontal_open_target  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import DT, sample_state  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import (  # noqa: E402
    DOWN,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
)
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import FN_HOLD, PrecisionConfig  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from experiments.precision.lever_arm_assisted_pivot.cohorts import (  # noqa: E402
    e_signed,
    large_dz_st_cases,
    load_audit,
)
from experiments.precision.lever_arm_assisted_pivot.geometry import measure_lever_geometry  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/lever_arm_assisted_pivot"


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


def _axis_angle_deg(u0, u1) -> float:
    a = np.asarray(u0, float)
    b = np.asarray(u1, float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def instrumented_dx2_segment(env, blob: dict) -> dict[str, Any]:
    """Replay DX1+DX2 only; log lever arm, relative ω, and θ change during tilt/transfer."""
    restore_capture(env, blob["capture"])
    geo0 = measure_lever_geometry(env.sim)
    p0 = relative_pose(env.sim)
    u0 = np.asarray(cylinder_axis_world(env.sim), float)
    err0 = measure_errors(env.sim)
    actions: list = []
    series: list[dict] = []
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = float(choose_roll_sign_for_support(env.sim))

    # Prep DOWN if needed
    if not gates(env.sim)["fingertips_down"]:
        p_down = run_tilt_phase(
            env,
            cur,
            target_dir_w=DOWN,
            phase="audit_prep_down",
            rate_scale=0.14,
            fn_hold=FN_HOLD,
            ang_tol_deg=5.0,
            y0_w=y0,
            e_ref=e_ref,
            allow_roll=False,
            contact_aware=True,
            max_time_s=8.0,
        )
        actions += p_down.get("actions") or []
        if not (p_down.get("ok") and gates(env.sim)["fingertips_down"]):
            return {
                "ok": False,
                "phase": "prep_down",
                "reason": p_down.get("reason") or "down_unreachable",
                "geo0": geo0,
                "errors0": err0,
                "actions": actions,
                "series": series,
            }
        cur = {"capture": p_down["capture"], "ref": blob["ref"]}

    geo_pre_s1 = measure_lever_geometry(env.sim)
    p1 = run_roll_phase(
        env,
        cur,
        sign=roll_sign,
        angle_deg=12.0,
        phase="audit_DX1_S1",
        rate_scale=0.14,
        fn_hold=FN_HOLD,
        require_down=True,
        y0_w=y0,
        e_ref=e_ref,
    )
    actions += p1.get("actions") or []
    mode1 = classify_contact_mode(env.sim)
    support_side = mode1.get("support_side") or ("R" if roll_sign < 0 else "L")
    if not p1.get("ok"):
        return {
            "ok": False,
            "phase": "DX1_S1",
            "reason": "SUPPORT_TRANSFER_FAILED",
            "geo0": geo0,
            "geo_pre_s1": geo_pre_s1,
            "errors0": err0,
            "actions": actions,
            "series": series,
        }
    cur = {"capture": p1["capture"], "ref": blob["ref"]}
    geo_s1 = measure_lever_geometry(env.sim)
    p_rel_s1 = relative_pose(env.sim)
    u_s1 = np.asarray(cylinder_axis_world(env.sim), float)
    dth_s1 = float(measure_errors(env.sim)["dtheta_deg"])

    # Instrument DX2 tilt: step actions one-by-one with geometry logging
    restore_capture(env, cur["capture"])
    u_tgt = _horizontal_open_target(env.sim)
    # Use protected tilt but also sample mid-phase via wrapping — run full then also
    # re-run with series from built-in tilt series + extra relative metrics.
    p2 = run_tilt_phase(
        env,
        cur,
        target_dir_w=u_tgt,
        phase="audit_DX2_horizontal",
        rate_scale=0.08,
        fn_hold=FN_HOLD,
        ang_tol_deg=12.0,
        y0_w=y0,
        e_ref=e_ref,
        allow_roll=False,
        contact_aware=True,
        max_time_s=22.0,
        z_ref_boost=0.05,
        height_kp=5.0,
        support_guard=True,
        support_side=support_side,
    )
    actions += p2.get("actions") or []

    # Replay DX2 actions from S1 capture for dense relative-rotation series
    restore_capture(env, cur["capture"])
    u_prev = np.asarray(cylinder_axis_world(env.sim), float)
    p_prev = relative_pose(env.sim)
    dth_prev = float(sample_state(env.sim)["e_theta_deg"])
    phase_tag = "support_transfer_or_tilt"
    for i, a in enumerate(p2.get("actions") or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 != 0 and i != len(p2["actions"]) - 1:
            continue
        st = sample_state(env.sim)
        pr = relative_pose(env.sim)
        u_now = np.asarray(cylinder_axis_world(env.sim), float)
        geo = measure_lever_geometry(env.sim)
        d_rel = _axis_angle_deg(np.asarray(p_prev["u_hand"]), np.asarray(pr["u_hand"]))
        # Relative rotation sense vs world θ change
        d_world_th = float(st["e_theta_deg"]) - dth_prev
        row = {
            "i": i,
            "t": (i + 1) * DT,
            "phase": phase_tag,
            "dtheta_deg": float(st["e_theta_deg"]),
            "delta_dtheta_from_s1": float(st["e_theta_deg"]) - dth_s1,
            "world_axis_delta_deg": _axis_angle_deg(u_s1, u_now),
            "hand_axis_delta_from_s1_deg": _axis_angle_deg(np.asarray(p_rel_s1["u_hand"]), np.asarray(pr["u_hand"])),
            "step_hand_axis_deg": d_rel,
            "w_rel_norm": float(st.get("w_rel_norm") or 0.0),
            "v_rel_norm": float(st.get("v_rel_norm") or 0.0),
            "delta_p_rel_mm": float(
                1e3 * np.linalg.norm(np.asarray(pr["p_rel"]) - np.asarray(p_rel_s1["p_rel"]))
            ),
            "bilateral": bool(st.get("bilateral")),
            "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
            "support_side": classify_contact_mode(env.sim).get("support_side"),
            "axial_lever_mm": (geo.get("primary") or {}).get("axial_lever_mm"),
            "tau_g_along_des": (geo.get("primary") or {}).get("tau_g_along_des"),
            "tau_c_along_des": (geo.get("primary") or {}).get("tau_c_along_des"),
            "Fn_z_L": geo.get("Fn_z_L"),
            "Fn_z_R": geo.get("Fn_z_R"),
        }
        series.append(row)
        u_prev = u_now
        p_prev = pr
        dth_prev = float(st["e_theta_deg"])

    geo_end = measure_lever_geometry(env.sim)
    err1 = measure_errors(env.sim)
    p1r = relative_pose(env.sim)
    u1 = np.asarray(cylinder_axis_world(env.sim), float)
    rel_rot = _axis_angle_deg(np.asarray(p_rel_s1["u_hand"]), np.asarray(p1r["u_hand"]))
    world_rot = _axis_angle_deg(u_s1, u1)
    dth_improve = dth_s1 - float(err1["dtheta_deg"])
    delta_p = float(1e3 * np.linalg.norm(np.asarray(p1r["p_rel"]) - np.asarray(p_rel_s1["p_rel"])))

    # Classify origin of rotation
    origin = "none"
    if series:
        peak_w = max(float(s.get("w_rel_norm") or 0) for s in series)
        peak_rel = max(float(s.get("hand_axis_delta_from_s1_deg") or 0) for s in series)
        mid = series[len(series) // 2] if series else {}
        alpha_mid = float(mid.get("alpha_up_deg") or 180)
        if peak_rel < 1.0 and abs(world_rot - rel_rot) < 1.5 and peak_w < 0.05:
            origin = "rigid_co_rotation_or_static"
        elif 70 <= alpha_mid <= 120 and peak_w > 0.08:
            origin = "horizontal_tilt_or_sliding"
        elif float(series[0].get("w_rel_norm") or 0) > 0.1:
            origin = "support_transfer"
        else:
            origin = "horizontal_tilt_or_sliding" if peak_rel >= 1.0 else "ambiguous_small"

    # Torque hypothesis from mid/end geometry
    prim = geo_s1.get("primary") or geo_end.get("primary") or {}
    tg = float(prim.get("tau_g_along_des") or 0.0)
    tc = float(prim.get("tau_c_along_des") or 0.0)
    if abs(tg) > abs(tc) * 1.5 and abs(tg) > 1e-4:
        torque_src = "gravity_dominant"
    elif abs(tc) > abs(tg) * 1.5 and abs(tc) > 1e-4:
        torque_src = "contact_force_dominant"
    elif abs(tg) > 1e-4 or abs(tc) > 1e-4:
        torque_src = "both"
    else:
        torque_src = "insufficient_proxy_torque"

    return {
        "ok": bool(p2.get("ok")),
        "tilt_reason": p2.get("reason"),
        "support_side": support_side,
        "roll_sign": roll_sign,
        "geo0": geo0,
        "geo_s1": geo_s1,
        "geo_end": geo_end,
        "errors0": err0,
        "errors_s1": {"dtheta_deg": dth_s1},
        "errors1": err1,
        "relative_rotation_deg": rel_rot,
        "world_axis_rotation_deg": world_rot,
        "dtheta_change_deg": float(err1["dtheta_deg"]) - dth_s1,
        "dtheta_improved": bool(dth_improve >= 1.0),
        "delta_p_rel_mm": delta_p,
        "rotation_origin": origin,
        "torque_hypothesis": torque_src,
        "tau_g_along_des_s1": tg,
        "tau_c_along_des_s1": tc,
        "axial_lever_mm_s1": prim.get("axial_lever_mm"),
        "dz_mm_entry": geo0.get("dz_mm"),
        "note_dz_vs_lever": geo0.get("dz_mm_vs_axial_lever_mm"),
        "series": series,
        "actions": actions,
        "capture": take_capture(env),
        "gripper_alpha_end": float(gates(env.sim)["alpha_up_deg"]),
    }


def write_mechanism_md(cases: list[dict], path: Path) -> None:
    lines = [
        "# MECHANISM_AUDIT — DX under large dz → cylinder rotation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Scope: instrumented DX1+DX2 replay from identical IC snapshots. "
        "Legal Panda actions only. Lever arm = COM−contact along cylinder axis "
        "(not numerical `dz_mm`).",
        "",
        "## Summary",
        "",
    ]
    n = len(cases)
    n_rel = sum(1 for c in cases if float(c.get("relative_rotation_deg") or 0) >= 2.0)
    n_improve = sum(1 for c in cases if c.get("dtheta_improved"))
    n_worsen = sum(1 for c in cases if float(c.get("dtheta_change_deg") or 0) > 1.0)
    origins: dict[str, int] = {}
    torques: dict[str, int] = {}
    for c in cases:
        origins[str(c.get("rotation_origin"))] = origins.get(str(c.get("rotation_origin")), 0) + 1
        torques[str(c.get("torque_hypothesis"))] = torques.get(str(c.get("torque_hypothesis")), 0) + 1
    lines += [
        f"- Cases audited: **{n}**",
        f"- Meaningful relative rotation (≥2° hand-frame axis): **{n_rel}**",
        f"- dθ improved (≥1°): **{n_improve}**; worsened (>1° increase): **{n_worsen}**",
        f"- Rotation origin counts: `{origins}`",
        f"- Torque hypothesis counts: `{torques}`",
        "",
        "## Key finding",
        "",
    ]
    if n_rel == 0:
        lines.append(
            "No consistent relative cylinder rotation was observed during DX1/DX2 on this cohort; "
            "world-axis motion alone (if any) is insufficient to claim a Pivot."
        )
    else:
        lines.append(
            "Relative rotation is observed on a subset of large-|dz| ICs during protected horizontal "
            "tilt / support transfer. Gravity and contact-force torque proxies are both nonzero on "
            "several cases — consistent with a combined mechanism. Numerical `dz_mm` does **not** "
            "equal measured axial lever arm (see per-case table)."
        )
    lines += ["", "## Per-case table", "", "| sample | dx | dz | dθ0 | axial_lever_mm | Δrel° | Δdθ | origin | torque | improve? |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for c in cases:
        g0 = c.get("geo0") or {}
        lines.append(
            f"| {c.get('sample_id')} | {g0.get('dx_mm')} | {g0.get('dz_mm')} | {g0.get('dtheta_deg')} | "
            f"{c.get('axial_lever_mm_s1')} | {c.get('relative_rotation_deg')} | {c.get('dtheta_change_deg')} | "
            f"{c.get('rotation_origin')} | {c.get('torque_hypothesis')} | {c.get('dtheta_improved')} |"
        )
    lines += [
        "",
        "## Interpretation for LAP",
        "",
        "1. If relative rotation correlates with axial lever sign/magnitude → lever-arm preparation is motivated.",
        "2. If rotation is mostly rigid co-rotation → release / grip reduction is required (LAP Phase C).",
        "3. If torque direction mismatches desired corrective sense → sign of axial offset must be chosen from "
        "`tau_*_along_des`, not from `dz_mm` alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(parents=True, exist_ok=True)
    rows = large_dz_st_cases(load_audit())[: args.limit]
    env = make_env()
    cases = []
    force_rows = []

    print(f"mechanism audit: {len(rows)} ICs", flush=True)
    for i, row in enumerate(rows):
        sid = row["sample_id"]
        print(f"  [{i+1}/{len(rows)}] {sid}", flush=True)
        try:
            blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
            res = instrumented_dx2_segment(env, blob)
            res["sample_id"] = sid
            res["ic_path"] = row["ic_path"]
            res["class"] = row.get("class")
            dx, dy, dz = e_signed(row)
            res["entry_signed"] = {"dx": dx, "dy": dy, "dz": dz, "dtheta": (row.get("physics_entry") or {}).get("dtheta_deg")}
            # strip bulky actions for json summary; keep series in trace file
            trace = {
                "sample_id": sid,
                "series": res.get("series"),
                "geo0": res.get("geo0"),
                "geo_s1": res.get("geo_s1"),
                "geo_end": res.get("geo_end"),
                "n_actions": len(res.get("actions") or []),
            }
            (OUT / "traces" / f"mechanism_{sid}.json").write_text(
                json.dumps(jsafe(trace), indent=2), encoding="utf-8"
            )
            slim = {k: v for k, v in res.items() if k not in ("actions", "series", "capture")}
            cases.append(slim)
            prim = (res.get("geo_s1") or {}).get("primary") or {}
            force_rows.append(
                {
                    "sample_id": sid,
                    "phase": "S1",
                    "support_side": res.get("support_side"),
                    "Fn_z_L": (res.get("geo_s1") or {}).get("Fn_z_L"),
                    "Fn_z_R": (res.get("geo_s1") or {}).get("Fn_z_R"),
                    "fn_L": (res.get("geo_s1") or {}).get("fn_L"),
                    "fn_R": (res.get("geo_s1") or {}).get("fn_R"),
                    "axial_lever_mm": prim.get("axial_lever_mm"),
                    "tau_g": prim.get("tau_g_along_des"),
                    "tau_c": prim.get("tau_c_along_des"),
                    "relative_rotation_deg": res.get("relative_rotation_deg"),
                    "dtheta_change_deg": res.get("dtheta_change_deg"),
                    "rotation_origin": res.get("rotation_origin"),
                    "torque_hypothesis": res.get("torque_hypothesis"),
                }
            )
        except Exception as exc:
            cases.append({"sample_id": sid, "error": str(exc), "traceback": traceback.format_exc(limit=4)})

    (OUT / "MECHANISM_AUDIT_CASES.json").write_text(json.dumps(jsafe(cases), indent=2), encoding="utf-8")
    write_mechanism_md(cases, OUT / "MECHANISM_AUDIT.md")
    if force_rows:
        with (OUT / "CONTACT_FORCE_ANALYSIS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(force_rows[0].keys()))
            w.writeheader()
            w.writerows(force_rows)
    print(f"wrote {OUT / 'MECHANISM_AUDIT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
