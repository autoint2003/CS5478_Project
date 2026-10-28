#!/usr/bin/env python
"""P8 inertial stick-slip dθ pilot on frozen hard_910277 C output."""

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
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.simulation.ground_scene import CYL_R  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.success_criteria import WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from experiments.precision.p7_oscillatory_dtheta.geometry import signed_theta_to_target  # noqa: E402
from experiments.precision.p8_inertial_dtheta.authority import (  # noqa: E402
    authority_static_audit,
    friction_torque_estimate,
    inertial_torque_estimate,
    run_accel_probe,
)
from experiments.precision.p8_inertial_dtheta.stick_slip import run_controller  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/p8_inertial_dtheta"
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
    return {"capture": cap, "ref": blob["ref"], "sample_id": HID, "start": signed_theta_to_target(env.sim)}


def write_authority_md(audit: dict, probes: list[dict]) -> None:
    best = max(probes, key=lambda p: p.get("peak_w_rel") or 0)
    lines = [
        "# ROTATIONAL_AUTHORITY_AUDIT — hard_910277",
        "",
        "Frozen complete-C mid_body_early snapshot.",
        "",
        "## Commanded vs achieved",
        "",
        f"- Legal `W_MAX` = **{audit['W_MAX_rad_s']} rad/s** (body frame of `r_des`).",
        f"- Control step = {audit['control_dt_s']} s ({audit['n_substeps_per_cmd']} physics substeps @ {audit['sim_timestep_s']} s).",
        f"- Orientation tracking: kp_ori={audit['ori_tracking']['kp_ori']}, kd_ori={audit['ori_tracking']['kd_ori']}.",
        f"- {audit['note_command']}",
        "",
        "## Geometry / inertia / friction",
        "",
        f"- Correction body axis: `{audit['body_axis_name']}` {audit['body_axis']}",
        f"- Signed θ = {audit['signed_theta_deg']:.3f}°, abs θ = {audit['abs_theta_deg']:.3f}°",
        f"- Cylinder m={audit['cyl_mass_kg']} kg, R={audit['cyl_radius_m']} m, I_xx={audit['cyl_Ixx']:.3e} kg·m²",
        f"- Contact Fn_tot ≈ {audit['fn_tot']:.2f} N (L={audit['fn_L']:.2f}, R={audit['fn_R']:.2f})",
        f"- τ_friction ≈ μ N R ≈ **{audit['tau_friction_est_Nm']:.4f} N·m** (μ=1)",
        f"- α to match friction with object inertia alone: **{audit['alpha_needed_to_match_friction_rad_s2']:.1f} rad/s²**",
        "",
        "Interpretation: object inertia is tiny; rotational friction at current pad load dominates. "
        "Useful slip requires reducing normal force and/or producing hand α large enough that "
        "inertial lag appears — but α is limited by orientation tracking, not by writing a larger ω command.",
        "",
        "## Probe highlights",
        "",
        f"- Highest peak relative ω: `{best['label']}` → w_rel={best['peak_w_rel']:.4f} rad/s, "
        f"α_hand={best['peak_alpha_hand']:.2f}, Δθ={best['d_theta_abs']:.3f}°, "
        f"cmd/achieved ω ratio={best['cmd_vs_achieved_w_ratio']:.3f}",
        "",
    ]
    for p in probes:
        lines.append(
            f"- `{p['label']}`: cmd={p['w_cmd_rad_s']} grip={p['grip_cmd']} profile={p['profile']} "
            f"peak_w={p['peak_w_hand']:.3f} peak_α={p['peak_alpha_hand']:.2f} "
            f"peak_wrel={p['peak_w_rel']:.4f} Δθ={p['d_theta_abs']:.3f} "
            f"ratio={p['cmd_vs_achieved_w_ratio']:.3f} bil={p['bilateral_end']} drop={p['dropped']}"
        )
    lines.append("")
    (OUT / "ROTATIONAL_AUTHORITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def validate(env, frozen, capture) -> dict:
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
    }


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    frozen = load_frozen(env)
    print("frozen", {k: frozen["start"].get(k) for k in ("abs_theta_deg", "signed_theta_deg", "dx_mm", "dz_mm")}, flush=True)

    # --- 1. Authority audit ---
    restore_capture(env, frozen["capture"])
    audit = authority_static_audit(env.sim)
    # attach measured geom friction if available
    try:
        gid = int(env.sim.ids.object_geom) if hasattr(env.sim.ids, "object_geom") else None
        if gid is None:
            # find object geom
            for i in range(env.sim.model.ngeom):
                if env.sim.model.geom_bodyid[i] == env.sim.ids.object_body:
                    gid = i
                    break
        if gid is not None:
            audit["geom_friction_object"] = env.sim.model.geom_friction[gid].tolist()
    except Exception:
        pass
    (OUT / "raw" / "authority_static.json").write_text(json.dumps(jsafe(audit), indent=2), encoding="utf-8")

    # --- 2. Acceleration probes (grip held secure) ---
    print("Acceleration probes...", flush=True)
    accel_specs = [
        {"label": "slow_step_1", "w_cmd_rad_s": 1.0, "grip": 0.90, "n_steps": 12, "profile": "step"},
        {"label": "med_step_2", "w_cmd_rad_s": 2.0, "grip": 0.90, "n_steps": 12, "profile": "step"},
        {"label": "fast_step_4", "w_cmd_rad_s": 4.0, "grip": 0.90, "n_steps": 12, "profile": "step"},
        {"label": "fast_ramp_4", "w_cmd_rad_s": 4.0, "grip": 0.90, "n_steps": 12, "profile": "ramp"},
        {"label": "fast_step_4_dir-", "w_cmd_rad_s": 4.0, "grip": 0.90, "n_steps": 12, "profile": "step", "direction": -1.0},
    ]
    accel_rows = []
    accel_packs = []
    for sp in accel_specs:
        r = run_accel_probe(
            env,
            frozen,
            label=sp["label"],
            w_cmd_rad_s=sp["w_cmd_rad_s"],
            grip=sp["grip"],
            n_steps=sp["n_steps"],
            profile=sp["profile"],
            direction=float(sp.get("direction", 1.0)),
        )
        accel_packs.append(r)
        row = {k: v for k, v in r.items() if k not in ("series", "actions", "capture")}
        # torque comparison at peak alpha
        row["tau_inertial_est"] = inertial_torque_estimate(audit["I_relevant"], r["peak_alpha_hand"])
        row["tau_friction_est"] = friction_torque_estimate(r.get("fn0") or audit["fn_tot"])
        row["inertial_over_friction"] = row["tau_inertial_est"] / max(row["tau_friction_est"], 1e-9)
        accel_rows.append(row)
        print(
            f"  {r['label']}: peak_w={r['peak_w_hand']:.3f} peak_a={r['peak_alpha_hand']:.2f} "
            f"wrel={r['peak_w_rel']:.4f} dth={r['d_theta_abs']:.3f} ratio={r['cmd_vs_achieved_w_ratio']:.3f}",
            flush=True,
        )
    csv_write(OUT / "ACCELERATION_PROBES.csv", accel_rows)
    write_authority_md(audit, accel_rows)

    # Pick motion profile with highest useful relative ω without drop
    safe_accel = [p for p in accel_packs if p["bilateral_end"] and not p["dropped"] and not p["contact_loss"]]
    motion = max(safe_accel, key=lambda p: (p["peak_w_rel"], p["peak_alpha_hand"])) if safe_accel else accel_packs[2]
    print(f"Selected motion profile for grip tests: {motion['label']}", flush=True)

    # --- 3. Grip modulation probes (fixed fast step profile) ---
    print("Grip modulation probes...", flush=True)
    grip_values = [0.90, 0.50, 0.20, 0.05, -0.15, -0.30]
    grip_rows = []
    grip_packs = []
    for g in grip_values:
        r = run_accel_probe(
            env,
            frozen,
            label=f"fast4_g{g:+.2f}",
            w_cmd_rad_s=4.0,
            grip=g,
            n_steps=12,
            profile="step",
            direction=float(motion.get("direction") or 1.0),
        )
        grip_packs.append(r)
        row = {k: v for k, v in r.items() if k not in ("series", "actions", "capture")}
        # classify
        if r["dropped"] or r["contact_loss"] or not r["bilateral_end"]:
            row["behavior"] = "CONTACT_LOSS"
        elif abs(r["dx1"] - r["dx0"]) > 4.0 or (r["e_pos1"] - r["e_pos0"]) > 5.0:
            row["behavior"] = "TRANSLATIONAL_SLIP"
        elif r["peak_w_rel"] >= 0.15 and abs(r["d_theta_abs"]) >= 0.5:
            row["behavior"] = "USEFUL_ROTATIONAL_SLIP"
        elif r["peak_w_rel"] < 0.08:
            row["behavior"] = "CO_ROTATION"
        else:
            row["behavior"] = "WEAK_RELATIVE"
        grip_rows.append(row)
        print(
            f"  grip={g:+.2f}: fn0={r['fn0']:.1f}→{r['fn1']:.1f} wrel={r['peak_w_rel']:.4f} "
            f"dth={r['d_theta_abs']:.3f} beh={row['behavior']} bil={r['bilateral_end']}",
            flush=True,
        )
    csv_write(OUT / "GRIP_MODULATION_PROBES.csv", grip_rows)

    # Feasible relax: require bilateral, no contact loss, and abs-θ must not worsen by >0.5°.
    def _safe(pair):
        pack, row = pair
        if row["behavior"] in ("CONTACT_LOSS", "TRANSLATIONAL_SLIP"):
            return False
        if not pack["bilateral_end"] or pack["dropped"]:
            return False
        if float(pack["d_theta_abs"]) < -0.5:  # θ increased
            return False
        return True

    feasible = [p for p in zip(grip_packs, grip_rows) if _safe(p)]
    useful = [p for p in feasible if p[1]["behavior"] == "USEFUL_ROTATIONAL_SLIP"]
    if useful:
        best_grip = max(useful, key=lambda p: float(p[0]["d_theta_abs"]))[0]["grip_cmd"]
        grip_note = f"useful slip at grip_cmd={best_grip}"
    elif feasible:
        # Prefer mild relaxation that stays co-rotation/weak without harming θ
        best_grip = min(feasible, key=lambda p: abs(float(p[0]["grip_cmd"]) - 0.05))[0]["grip_cmd"]
        grip_note = (
            "NO_FEASIBLE_USEFUL_SLIP: all open-grip probes either co-rotate, worsen θ, "
            f"or lose contact; using safest relax grip_cmd={best_grip} for I2/I3 documentation"
        )
    else:
        best_grip = 0.90
        grip_note = (
            "NO_FEASIBLE_GRIP_RANGE: every reduced-grip probe lost contact, translated, "
            "or increased |θ|; I2/I3 use hold grip only to document failure mode"
        )
    print("Grip selection:", grip_note, flush=True)

    # --- 4/5. I0–I3 controllers ---
    variants = [
        ("I0", "p7_baseline", 0.35, 0.85, 0.85),
        ("I1", "fast_hold_grip", 4.0, 0.90, 0.90),
        ("I2", "fast_reduced_grip", 4.0, 0.90, float(best_grip)),
        ("I3", "phase_grip", 4.0, 0.90, float(best_grip)),
    ]
    cycle_rows = []
    val_rows = []
    packs = {}

    for label, mode, wcru, ghold, grel in variants:
        print(f"Running {label} mode={mode} w={wcru} hold={ghold} relax={grel}...", flush=True)
        res = run_controller(
            env,
            frozen,
            mode=mode,  # type: ignore[arg-type]
            w_cruise=wcru,
            grip_hold=ghold,
            grip_relax=grel,
            max_cycles=5,
        )
        val = validate(env, frozen, res["capture"])
        packs[label] = {**res, "validation": val, "grip_note": grip_note}
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
                    "classification": c.get("classification"),
                    "theta0_abs": c.get("theta0_abs"),
                    "theta1_abs": c.get("theta1_abs"),
                    "signed0": c.get("signed0"),
                    "signed1": c.get("signed1"),
                    "net_improve_abs": c.get("net_improve_abs"),
                    "net_relative_deg": c.get("net_relative_deg"),
                    "peak_relative_deg": c.get("peak_relative_deg"),
                    "peak_w_hand": c.get("peak_w_hand"),
                    "peak_alpha_hand": c.get("peak_alpha_hand"),
                    "peak_w_rel": c.get("peak_w_rel"),
                    "dx0": c.get("dx0"),
                    "dx1": c.get("dx1"),
                    "dz0": c.get("dz0"),
                    "dz1": c.get("dz1"),
                    "e_pos0": c.get("e_pos0"),
                    "e_pos1": c.get("e_pos1"),
                    "fn_min": c.get("fn_min"),
                    "fn_max": c.get("fn_max"),
                    "bilateral_end": c.get("bilateral_end"),
                    "w_cruise": c.get("w_cruise"),
                    "grip_hold": c.get("grip_hold"),
                    "grip_relax": c.get("grip_relax"),
                    "direction": c.get("direction"),
                }
            )
        val_rows.append(
            {
                "variant": label,
                "mode": mode,
                "theta0": res["theta0_abs"],
                "theta_final": res["theta_final_abs"],
                "signed0": res["signed0"],
                "signed_final": res["signed_final"],
                "net_improve_deg": res["net_improve_total"],
                "n_cycles": res["n_cycles"],
                "reached_theta_tol": res["reached_theta_tol"],
                "abort_reason": res["abort_reason"],
                "peak_alpha": res["peak_alpha_all"],
                "peak_w_rel": res["peak_w_rel_all"],
                "dx_final": val["dx"],
                "dz_final": val["dz"],
                "e_pos_final": val["e_pos"],
                "pos_ok": val["pos_ok"],
                "SUCCESS_12S": val["SUCCESS_12S"],
                "SUCCESS_NEW": val["SUCCESS_NEW"],
                "grip_relax_used": grel,
                "classification_last": (res["cycles"][-1]["classification"] if res["cycles"] else None),
            }
        )
        print(
            f"  {label}: θ {res['theta0_abs']:.2f}→{res['theta_final_abs']:.2f} "
            f"α={res['peak_alpha_all']:.1f} wrel={res['peak_w_rel_all']:.3f} "
            f"NEW={val['SUCCESS_NEW']} abort={res['abort_reason']}",
            flush=True,
        )

    csv_write(OUT / "STICK_SLIP_CYCLE_TRACES.csv", cycle_rows)
    csv_write(OUT / "FINAL_VALIDATION.csv", val_rows)
    (OUT / "raw" / "grip_note.txt").write_text(grip_note + f"\nCYL_R={CYL_R}\n", encoding="utf-8")
    (OUT / "raw" / "all_summary.json").write_text(
        json.dumps(
            jsafe({lab: {k: v for k, v in p.items() if k not in ("actions", "series", "capture")} for lab, p in packs.items()}),
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
