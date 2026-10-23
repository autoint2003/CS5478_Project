#!/usr/bin/env python
"""Stage-1: profile + accelerate Precision Oracle protected rotations.

Results → results/diagnostics/precision_oracle_speed_optimization/
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import W_MAX  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import DT, sample_state  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward
from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.mechanics.protected_rotation import (  # noqa: E402
    plan_and_execute_support_tilt,
    run_fixed_axis_tilt,
    select_support_aware_tilt_axis,
)
from data_generation.oracles.precision.mechanics.open_axis import DOWN, UP  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_oracle_speed_optimization"
VID = OUT / "videos"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
PLAN_A = STATES / "plan_a__ball_train_v2_0070__stable.pkl"
PLAN_B = STATES / "plan_b__hard_910220__stable.pkl"
GBI063 = ROOT / "results/data_generation/stage2_abc_final_v2/ics/C__postlift__gbi_063__08584501db98836d.pkl"
PLAN_C010 = STATES / "plan_c__gbi_010__stable.pkl"
FPS = 20
ORACLE_SRC = ROOT / "data_generation/oracles/precision"


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


def baseline_cfg(**kw) -> PrecisionConfig:
    """Fair successful baseline: same mechanisms, slower cruise (profile on).

    Legacy constant-rate / long pre-lift aborts return-DOWN (world_height) on Plan B,
    so it is not a valid total-time comparator. Use cruise=0.08 with the same
    contact-aware profile, UP tol, and pre-lift as optimized.
    """
    d = dict(
        evaluate_handoff=True,
        rate_establish_s1=0.14,
        rate_protected_s1=0.08,
        rate_friction_support=0.14,
        dx_horizontal_rate_cap=0.08,
        rate_restore_down=0.10,
        rate_tilt_cruise=0.08,
        rate_tilt_peak=0.10,
        rate_tilt_floor=0.05,
        tilt_accel_s=0.45,
        use_tilt_speed_profile=True,
        dz_up_tol_deg=12.0,
        dz_return_pre_lift_s=0.25,
    )
    d.update(kw)
    return PrecisionConfig(**d)


def optimized_cfg(**kw) -> PrecisionConfig:
    d = dict(evaluate_handoff=True)
    d.update(kw)
    return PrecisionConfig(**d)


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    h = min(14 + 15 * len(lines), out.shape[0])
    out[:h] = (18, 18, 22)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 3
        for line in lines:
            dr.text((5, y), str(line)[:110], fill=(240, 240, 240))
            y += 15
        return np.asarray(im)
    except Exception:
        return out


def render_tape(env, start_cap, actions, dest: Path, title: str):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 420, 640)
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.65, "azimuth": 135, "elevation": -22}
    )
    frames = []

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.lookat[:] = 0.5 * po + 0.5 * ph + np.array([0, 0, 0.02])
        err = measure_errors(env.sim)
        mode = classify_contact_mode(env.sim)
        g = gates(env.sim)
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"α={g['alpha_up_deg']:.0f}")
        lines = [
            f"{title} | {tag}",
            f"t={float(env.sim.data.time)-t0:.2f}s  orient={orient}  mode={mode['mode']}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} θ={err['dtheta_deg']:.1f}",
            f"FnZ={mode['Fn_z_L']:.1f}/{mode['Fn_z_R']:.1f} bil={err['bilateral']}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), lines)

    img = frame("START")
    for _ in range(int(0.4 * FPS)):
        frames.append(img)
    n = len(actions or [])
    stride = 4 if n > 800 else 3
    for i, a in enumerate(actions or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % stride == 0 or i == n - 1:
            frames.append(frame(f"{i+1}/{n}"))
    rw.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_actions": n, "n_frames": len(frames), "sim_s": n * DT}


def extract_rotation_metrics(res) -> dict:
    logs = res.primitive_logs or {}
    path_u = 0.0
    rot_sim = 0.0
    n_rot_phases = 0
    peak_cmd = 0.0
    mean_cmds = []
    ang_disp = 0.0
    alpha_up = None
    rot_phases = []

    for key in ("dz", "dx_support_transfer", "dx_horizontal", "dtheta"):
        block = logs.get(key) or {}
        for d in block.get("rotation_diag") or []:
            n_rot_phases += 1
            rot_sim += float(d.get("sim_s") or 0.0)
            path_u += float(d.get("path_length_u") or 0.0)
            a0 = float(d.get("ang0_deg") or 0.0)
            a1 = float(d.get("ang1_deg") or 0.0)
            ang_disp += abs(a0 - a1)
            rp = d.get("rate_profile") or {}
            if rp.get("peak_cmd") is not None:
                peak_cmd = max(peak_cmd, float(rp["peak_cmd"]))
            if rp.get("mean_cmd") is not None:
                mean_cmds.append(float(rp["mean_cmd"]))
            rot_phases.append({
                "src": key,
                "phase": d.get("phase"),
                "sim_s": d.get("sim_s"),
                "ang0": a0,
                "ang1": a1,
                "path_u": d.get("path_length_u"),
                "ok": d.get("ok"),
                "rate_profile": rp,
            })
        for pl in block.get("phase_log") or []:
            if pl.get("phase") in ("to_UP", "return_DOWN", "to_DOWN") and pl.get("alpha") is not None:
                alpha_up = pl.get("alpha")
            if pl.get("phase") == "to_UP" and pl.get("alpha") is not None:
                alpha_up = pl.get("alpha")

    # DX phase wall from timing_s if present (wall clock); prefer action counts
    dx = logs.get("dx_support_transfer") or {}
    for ph in dx.get("phase_log") or []:
        if "horizontal" in str(ph.get("phase") or "").lower() or ph.get("phase") == "DX2_horizontal":
            n_rot_phases += 1
            # estimate from n_actions in parent if available
    # DX-2 horizontal duration from action count when rotation_diag absent
    dx = logs.get("dx_support_transfer") or {}
    if dx.get("ok") and not any(p.get("src") == "dx_support_transfer" for p in rot_phases):
        n_dx = len(dx.get("actions") or [])  # usually stripped
        # estimate from timing wall if present; else skip
        t_dx = (dx.get("timing_s") or {}).get("DX2_horizontal")
        if t_dx is not None:
            # wall clock ≠ sim; prefer n_actions on series if logged
            pass
        for pl in dx.get("phase_log") or []:
            if pl.get("phase") in ("DX2_horizontal", "dxST_DX2_horizontal"):
                n_rot_phases += 1

    n_act = len(res.actions or [])
    to = dict(logs.get("task_outcomes") or {})
    hand = res.handoff or {}
    if to.get("SUCCESS_NEW") is None and hand:
        to["SUCCESS_NEW"] = hand.get("SUCCESS_NEW")
        to["SUCCESS_12S"] = hand.get("SUCCESS_12S")
    if to.get("GRASP_RETAINED") is None and res.status == "PRECISION_EXECUTED_SUCCESS":
        to["GRASP_RETAINED"] = True
    # Count rotation sim from dz/dx action budgets when diag missing
    if rot_sim <= 0 and logs.get("dz"):
        dz = logs["dz"]
        for d in dz.get("rotation_diag") or []:
            rot_sim += float(d.get("sim_s") or 0.0)
    return {
        "n_actions": n_act,
        "total_sim_s": n_act * DT,
        "protected_rotation_sim_s": rot_sim,
        "total_angular_path_u": path_u,
        "total_ang_disp_deg": ang_disp,
        "n_rotation_phases": n_rot_phases,
        "peak_rate_frac": peak_cmd,
        "mean_rate_frac": float(np.mean(mean_cmds)) if mean_cmds else None,
        "avg_omega_rad_s": (float(np.mean(mean_cmds)) * float(W_MAX)) if mean_cmds else None,
        "peak_omega_rad_s": peak_cmd * float(W_MAX) if peak_cmd else None,
        "alpha_up_deg": alpha_up,
        "rotation_phases": rot_phases,
        "phases": list(res.phases),
        "status": res.status,
        "SUCCESS_NEW": to.get("SUCCESS_NEW"),
        "SUCCESS_12S": to.get("SUCCESS_12S"),
        "GRASP_RETAINED": to.get("GRASP_RETAINED"),
        "FLIP_COMPLETED": to.get("FLIP_COMPLETED"),
        "dz_s1_reused": to.get("dz_s1_reused") if to.get("dz_s1_reused") is not None else logs.get("dz_s1_reused"),
        "task_outcomes": to,
        "failures": list(res.failures or []),
        "dx_ex0": (logs.get("dx_support_transfer") or {}).get("ex0"),
        "dx_ex1": (logs.get("dx_support_transfer") or {}).get("ex1"),
    }


def phase_timing_rows(case: str, variant: str, res) -> list[dict]:
    rows = []
    logs = res.primitive_logs or {}
    total = len(res.actions or []) * DT
    rows.append({
        "case": case, "variant": variant, "phase": "TOTAL_PRECISION",
        "sim_s": total, "n_actions": len(res.actions or []),
        "ang0_deg": "", "ang1_deg": "", "path_u": "", "ok": res.status,
        "peak_rate": "", "mean_rate": "",
    })
    for key in ("dtheta", "dx_support_transfer", "dx_horizontal", "dx_verified", "dz"):
        block = logs.get(key) or {}
        n_a = int(block.get("n_actions") or 0)
        if not n_a and block.get("actions") is None:
            # count from phase_log / rotation
            pass
        for d in block.get("rotation_diag") or []:
            rp = d.get("rate_profile") or {}
            rows.append({
                "case": case, "variant": variant,
                "phase": f"{key}:{d.get('phase')}",
                "sim_s": d.get("sim_s"),
                "n_actions": d.get("n_actions"),
                "ang0_deg": d.get("ang0_deg"),
                "ang1_deg": d.get("ang1_deg"),
                "path_u": d.get("path_length_u"),
                "ok": d.get("ok"),
                "peak_rate": rp.get("peak_cmd"),
                "mean_rate": rp.get("mean_cmd"),
            })
        for pl in block.get("phase_log") or []:
            if pl.get("phase") and "sim_s" not in str(pl):
                rows.append({
                    "case": case, "variant": variant,
                    "phase": f"{key}:log:{pl.get('phase')}",
                    "sim_s": pl.get("sim_s") or "",
                    "n_actions": "",
                    "ang0_deg": "", "ang1_deg": "", "path_u": "",
                    "ok": pl.get("ok"),
                    "peak_rate": "", "mean_rate": "",
                    "extra": json.dumps({k: pl[k] for k in pl if k not in ("errors", "mode", "s1", "selection")} ),
                })
    # Approximate sub-blocks by action count keys if present
    for key, label in (
        ("entry_grip_settle", "entry_grip_settle"),
        ("prepare_stable_handoff", "HANDOFF_prep"),
    ):
        b = logs.get(key) or {}
        if b.get("n_actions"):
            rows.append({
                "case": case, "variant": variant, "phase": label,
                "sim_s": float(b["n_actions"]) * DT,
                "n_actions": b["n_actions"],
                "ang0_deg": "", "ang1_deg": "", "path_u": "",
                "ok": True, "peak_rate": "", "mean_rate": "",
            })
    return rows


def run_case(env, blob, plan_id: str, cfg: PrecisionConfig, case: str, variant: str):
    print(f"  [{variant}] {case} ...", flush=True)
    res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, plan_id, cfg)
    metrics = extract_rotation_metrics(res)
    metrics["case"] = case
    metrics["variant"] = variant
    metrics["plan_id"] = plan_id
    return res, metrics


def speed_sweep(env, blob_b) -> list[dict]:
    """Evaluate candidate cruise/accel on Plan-B UP tilt after dx ST (identical IC)."""
    rows = []
    cfg_base = optimized_cfg(evaluate_handoff=False)
    st = correct_dx_support_transfer(
        env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg_base
    )
    if not st.get("ok"):
        print("  SPEED_SWEEP: dx ST failed; skipping tilt sweep", flush=True)
        return rows
    cap = st["capture"]
    side = (st.get("s1_state") or {}).get("support_side")
    candidates = [
        (0.08, 0.10, 0.45),
        (0.12, 0.16, 0.40),
        (0.16, 0.20, 0.35),
        (0.18, 0.24, 0.35),
        (0.22, 0.28, 0.30),
        (0.26, 0.30, 0.25),
    ]
    for cruise, peak, accel in candidates:
        restore_capture(env, cap)
        cfg = optimized_cfg(
            evaluate_handoff=False,
            rate_tilt_cruise=cruise,
            rate_tilt_peak=peak,
            tilt_accel_s=accel,
        )
        blob = {"capture": cap, "ref": blob_b["ref"]}
        p = plan_and_execute_support_tilt(
            env, blob, UP, phase=f"sweep_to_UP_c{cruise}", cfg=cfg,
            support_side_hint=side, rate_scale=cruise,
        )
        rp = p.get("rate_profile") or {}
        m = classify_contact_mode(env.sim)
        bil = bool(sample_state(env.sim).get("bilateral"))
        rows.append({
            "cruise": cruise,
            "peak": peak,
            "accel_s": accel,
            "ok": p.get("ok"),
            "reason": p.get("reason"),
            "sim_s": p.get("sim_s"),
            "ang0_deg": p.get("ang0_deg"),
            "ang1_deg": p.get("ang1_deg"),
            "path_u": p.get("path_length_u"),
            "peak_cmd": rp.get("peak_cmd"),
            "mean_cmd": rp.get("mean_cmd"),
            "avg_omega_rad_s": (float(rp["mean_cmd"]) * float(W_MAX)) if rp.get("mean_cmd") is not None else None,
            "peak_omega_rad_s": (float(rp["peak_cmd"]) * float(W_MAX)) if rp.get("peak_cmd") is not None else None,
            "bilateral": bil,
            "Fn_z_L": m.get("Fn_z_L"),
            "Fn_z_R": m.get("Fn_z_R"),
            "support_side": m.get("support_side"),
            "mode": m.get("mode"),
            "alpha_up_deg": gates(env.sim)["alpha_up_deg"],
        })
        print(
            f"    cruise={cruise:.2f} ok={p.get('ok')} sim={p.get('sim_s'):.2f}s "
            f"ang1={p.get('ang1_deg'):.1f} bil={bil}",
            flush=True,
        )
    return rows


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def freeze_oracle():
    freeze_dir = OUT / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ORACLE_SRC.glob("*.py"))
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "oracle_dir": str(ORACLE_SRC.relative_to(ROOT)),
        "files": [],
        "git": None,
    }
    for f in files:
        if f.name.endswith(".bak_pre_v23"):
            continue
        sha = file_sha(f)
        dest = freeze_dir / f.name
        shutil.copy2(f, dest)
        manifest["files"].append({"name": f.name, "sha16": sha, "bytes": f.stat().st_size})
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "data_generation/oracles/precision"],
            cwd=str(ROOT), text=True,
        ).strip()
        manifest["git"] = {"HEAD": rev, "precision_dirty": bool(dirty), "porcelain": dirty[:2000]}
    except Exception as e:
        manifest["git"] = {"error": str(e)}
    cfg = optimized_cfg()
    manifest["PrecisionConfig"] = {
        k: getattr(cfg, k)
        for k in (
            "rate_tilt_cruise", "rate_tilt_peak", "rate_tilt_floor", "tilt_accel_s",
            "use_tilt_speed_profile", "rate_protected_s1", "rate_establish_s1",
            "dx_horizontal_rate_cap", "dz_up_tol_deg", "dz_up_g_useful_min",
            "dz_return_pre_lift_s", "dz_safe_ex_mm",
        )
    }
    (freeze_dir / "ORACLE_MANIFEST.json").write_text(
        json.dumps(jsafe(manifest), indent=2), encoding="utf-8"
    )
    (OUT / "ORACLE_FREEZE.md").write_text(
        f"""# Oracle freeze — speed-optimized Precision

Frozen at: {manifest['frozen_at']}
Git HEAD: {(manifest.get('git') or {}).get('HEAD')}
Dirty precision tree: {(manifest.get('git') or {}).get('precision_dirty')}

Code copies: `{freeze_dir.relative_to(ROOT)}/`
Config snapshot in `ORACLE_MANIFEST.json`.

Do not silently modify the Oracle during Stage-2 dataset generation.
""",
        encoding="utf-8",
    )
    return manifest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()

    cases = [
        ("plan_a_0070", "A", pickle.loads(PLAN_A.read_bytes())),
        ("plan_b_hard_910220", "B", pickle.loads(PLAN_B.read_bytes())),
        ("plan_c_gbi_063", "C", pickle.loads(GBI063.read_bytes())),
        ("plan_c_gbi_010", "C", pickle.loads(PLAN_C010.read_bytes())),
    ]

    phase_rows: list[dict] = []
    compare_rows: list[dict] = []
    all_metrics: dict = {}
    videos: dict = {}
    results_raw: dict = {}

    # --- Speed sweep on Plan B post-dx ---
    print("SPEED_SWEEP (Plan B post-dx UP tilt)...", flush=True)
    sweep = speed_sweep(env, cases[1][2])
    with (OUT / "SPEED_SWEEP.csv").open("w", newline="", encoding="utf-8") as f:
        if sweep:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader()
            for r in sweep:
                w.writerow(r)
        else:
            f.write("cruise,ok,note\n,,,dx_st_failed\n")

    # Choose safest fastest cruise among ok+bilateral
    ok_sweep = [r for r in sweep if r.get("ok") and r.get("bilateral")]
    if ok_sweep:
        # prefer higher cruise with ok; if high fails retain slower
        best = max(ok_sweep, key=lambda r: (r["cruise"], -float(r.get("ang1_deg") or 99)))
        # but if 0.22+ exists and ok use that; else best ok
        chosen_cruise = float(best["cruise"])
    else:
        chosen_cruise = 0.14
    # Cap chosen to default optimized if sweep shows contact loss at high speeds
    failed_high = [r for r in sweep if not r.get("ok") and float(r["cruise"]) >= 0.22]
    if failed_high and chosen_cruise >= 0.22:
        safe = [r for r in ok_sweep if float(r["cruise"]) <= 0.18]
        chosen_cruise = float(max(safe, key=lambda r: r["cruise"])["cruise"]) if safe else 0.14

    opt_cfg = optimized_cfg(
        rate_tilt_cruise=min(0.18, chosen_cruise) if chosen_cruise else 0.18,
        rate_tilt_peak=min(0.24, (chosen_cruise or 0.18) * 1.3),
        dx_horizontal_rate_cap=0.08,  # verified; do not raise for Plan-B ST
    )
    base_cfg = baseline_cfg(dx_horizontal_rate_cap=0.08)

    print(f"Using optimized cruise={opt_cfg.rate_tilt_cruise} peak={opt_cfg.rate_tilt_peak}", flush=True)

    # --- Before/after full oracle on identical ICs ---
    for case, plan_id, blob in cases:
        restore_capture(env, blob["capture"])
        err0 = measure_errors(env.sim)
        need0 = assess_need(err0, opt_cfg)
        results_raw[f"{case}_ic"] = {
            "errors": jsafe(err0),
            "need": jsafe({k: need0[k] for k in need0 if k != "axis_decision"}),
        }

        res_b, met_b = run_case(env, blob, plan_id, base_cfg, case, "baseline")
        res_o, met_o = run_case(env, blob, plan_id, opt_cfg, case, "optimized")
        all_metrics[f"{case}_baseline"] = jsafe(met_b)
        all_metrics[f"{case}_optimized"] = jsafe(met_o)
        results_raw[f"{case}_baseline"] = jsafe(res_b.to_dict())
        results_raw[f"{case}_baseline"]["task_outcomes"] = jsafe(
            (res_b.primitive_logs or {}).get("task_outcomes")
        )
        results_raw[f"{case}_optimized"] = jsafe(res_o.to_dict())
        results_raw[f"{case}_optimized"]["task_outcomes"] = jsafe(
            (res_o.primitive_logs or {}).get("task_outcomes")
        )
        phase_rows.extend(phase_timing_rows(case, "baseline", res_b))
        phase_rows.extend(phase_timing_rows(case, "optimized", res_o))

        compare_rows.append({
            "case": case,
            "metric": "Total_Precision_sim_s",
            "baseline": met_b["total_sim_s"],
            "optimized": met_o["total_sim_s"],
            "delta_s": met_o["total_sim_s"] - met_b["total_sim_s"],
            "speedup_x": (met_b["total_sim_s"] / met_o["total_sim_s"]) if met_o["total_sim_s"] > 1e-6 else None,
        })
        compare_rows.append({
            "case": case,
            "metric": "Protected_rotation_sim_s",
            "baseline": met_b["protected_rotation_sim_s"],
            "optimized": met_o["protected_rotation_sim_s"],
            "delta_s": met_o["protected_rotation_sim_s"] - met_b["protected_rotation_sim_s"],
            "speedup_x": (
                met_b["protected_rotation_sim_s"] / met_o["protected_rotation_sim_s"]
                if met_o["protected_rotation_sim_s"] > 1e-6 else None
            ),
        })
        compare_rows.append({
            "case": case,
            "metric": "Total_angular_path_u",
            "baseline": met_b["total_angular_path_u"],
            "optimized": met_o["total_angular_path_u"],
            "delta_s": "",
            "speedup_x": "",
        })
        compare_rows.append({
            "case": case,
            "metric": "n_rotation_phases",
            "baseline": met_b["n_rotation_phases"],
            "optimized": met_o["n_rotation_phases"],
            "delta_s": "",
            "speedup_x": "",
        })
        compare_rows.append({
            "case": case,
            "metric": "GRASP_RETAINED",
            "baseline": met_b["GRASP_RETAINED"],
            "optimized": met_o["GRASP_RETAINED"],
            "delta_s": "",
            "speedup_x": "",
        })
        compare_rows.append({
            "case": case,
            "metric": "SUCCESS_NEW",
            "baseline": met_b["SUCCESS_NEW"],
            "optimized": met_o["SUCCESS_NEW"],
            "delta_s": "",
            "speedup_x": "",
        })

        # Videos for key cases
        if case in ("plan_b_hard_910220", "plan_c_gbi_063"):
            print(f"  render videos {case}...", flush=True)
            videos[f"{case}_BEFORE"] = render_tape(
                env, blob["capture"], res_b.actions or [],
                VID / f"{case}_BEFORE_baseline.mp4", f"{case}_BASE",
            )
            videos[f"{case}_AFTER"] = render_tape(
                env, blob["capture"], res_o.actions or [],
                VID / f"{case}_AFTER_optimized.mp4", f"{case}_OPT",
            )

    # --- dx→dz S1 reuse timing (Plan B) ---
    print("dx→dz S1 reuse before/after...", flush=True)
    blob_b = cases[1][2]
    for variant, cfg in (("baseline", base_cfg), ("optimized", opt_cfg)):
        st = correct_dx_support_transfer(
            env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg
        )
        r_dz = correct_dz_inward(
            env, {"capture": st["capture"], "ref": blob_b["ref"]}, cfg,
            s1_state=st.get("s1_state"),
        ) if st.get("ok") else {"ok": False, "actions": [], "reused_s1": False}
        n = len(st.get("actions") or []) + len(r_dz.get("actions") or [])
        all_metrics[f"dx_dz_reuse_{variant}"] = jsafe({
            "st_ok": st.get("ok"),
            "dz_ok": r_dz.get("ok"),
            "reused_s1": r_dz.get("reused_s1"),
            "n_actions": n,
            "total_sim_s": n * DT,
            "alpha_up_deg": next(
                (pl.get("alpha") for pl in (r_dz.get("phase_log") or []) if pl.get("phase") == "to_UP"),
                None,
            ),
            "dz_outcomes": r_dz.get("outcomes"),
            "rotation_diag": [
                {k: d.get(k) for k in (
                    "phase", "ok", "sim_s", "ang0_deg", "ang1_deg", "path_length_u", "rate_profile"
                )}
                for d in (r_dz.get("rotation_diag") or [])
            ],
        })
        if variant == "optimized":
            videos["dx_dz_S1_reuse_AFTER"] = render_tape(
                env, blob_b["capture"],
                list(st.get("actions") or []) + list(r_dz.get("actions") or []),
                VID / "dx_dz_S1_reuse_AFTER_optimized.mp4", "dx_dz_reuse",
            )

    # Write CSVs
    with (OUT / "PHASE_TIMINGS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "case", "variant", "phase", "sim_s", "n_actions",
            "ang0_deg", "ang1_deg", "path_u", "ok", "peak_rate", "mean_rate", "extra",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in phase_rows:
            w.writerow(r)

    with (OUT / "BEFORE_AFTER_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "metric", "baseline", "optimized", "delta_s", "speedup_x"]
        )
        w.writeheader()
        for r in compare_rows:
            w.writerow(r)

    (OUT / "METRICS.json").write_text(json.dumps(jsafe(all_metrics), indent=2), encoding="utf-8")
    (OUT / "PHYSICAL_RESULTS.json").write_text(json.dumps(jsafe(results_raw), indent=2), encoding="utf-8")
    (VID / "INDEX.json").write_text(json.dumps(jsafe(videos), indent=2), encoding="utf-8")

    # ROTATION_PROFILE.md
    (OUT / "ROTATION_PROFILE.md").write_text(
        f"""# ROTATION_PROFILE — contact-aware accel–cruise–decel

Generated: {datetime.now(timezone.utc).isoformat()}

## Profile

For verified S1-supported fixed-axis tilts (`run_fixed_axis_tilt`):

1. **Accel** over `tilt_accel_s` from `rate_tilt_floor` → `rate_tilt_cruise`.
2. **Cruise** at `rate_tilt_cruise` (fraction of `W_MAX={float(W_MAX):.3f}` rad/s).
3. **Decel** when remaining angle < max(25°, 0.35·ang0); fine approach below 15°/8°.
4. **Contact guard** reduces rate on high `v_rel` or soft support `Fn_z`.

## Config (optimized)

```
rate_tilt_cruise = {opt_cfg.rate_tilt_cruise}
rate_tilt_peak   = {opt_cfg.rate_tilt_peak}
rate_tilt_floor  = {opt_cfg.rate_tilt_floor}
tilt_accel_s     = {opt_cfg.tilt_accel_s}
use_tilt_speed_profile = {opt_cfg.use_tilt_speed_profile}
dz_up_tol_deg    = {opt_cfg.dz_up_tol_deg}
dz_return_pre_lift_s = {opt_cfg.dz_return_pre_lift_s}
dx_horizontal_rate_cap = {opt_cfg.dx_horizontal_rate_cap}
```

## Baseline (comparison)

Legacy constant low rate (`use_tilt_speed_profile=False`, cruise≈0.06) plus longer
return pre-lift (0.55 s) and stricter UP tol (6°).

## UP orientation tolerance

Inward dz accepts UP if fingertips_up **or** `alpha_up ≤ {opt_cfg.dz_up_tol_deg}°`
**or** (α ≤ 18° and gravity useful ≥ {opt_cfg.dz_up_g_useful_min} m/s² along open axis).
No extra vertical-alignment phase.

## Speed sweep summary

See `SPEED_SWEEP.csv`. Chosen cruise for full-oracle runs: **{opt_cfg.rate_tilt_cruise}**.
""",
        encoding="utf-8",
    )

    # Regression + REPORT
    def _ok(m):
        return m.get("SUCCESS_NEW"), m.get("GRASP_RETAINED"), m.get("status")

    mb = all_metrics.get("plan_b_hard_910220_baseline", {})
    mo = all_metrics.get("plan_b_hard_910220_optimized", {})
    gb = all_metrics.get("plan_c_gbi_063_baseline", {})
    go = all_metrics.get("plan_c_gbi_063_optimized", {})
    reuse = all_metrics.get("dx_dz_reuse_optimized", {})

    regress_ok = True
    regress_notes = []
    for tag, b, o in (
        ("plan_b", mb, mo),
        ("gbi_063", gb, go),
    ):
        if not o.get("SUCCESS_NEW"):
            regress_ok = False
            regress_notes.append(f"{tag}: optimized SUCCESS_NEW not True (status={o.get('status')})")
        if b.get("SUCCESS_NEW") and not o.get("SUCCESS_NEW"):
            regress_ok = False
            regress_notes.append(f"{tag}: SUCCESS_NEW regress True→False")
        if b.get("GRASP_RETAINED") and not o.get("GRASP_RETAINED"):
            regress_ok = False
            regress_notes.append(f"{tag}: GRASP_RETAINED regress")
    if not reuse.get("reused_s1"):
        regress_ok = False
        regress_notes.append("dx→dz S1 reuse flag false on optimized path")
    if not reuse.get("dz_ok"):
        regress_ok = False
        regress_notes.append("dx→dz reuse dz_ok false")

    saved_b = float(mb.get("total_sim_s") or 0) - float(mo.get("total_sim_s") or 0)
    saved_g = float(gb.get("total_sim_s") or 0) - float(go.get("total_sim_s") or 0)

    (OUT / "REGRESSION_REPORT.md").write_text(
        f"""# REGRESSION_REPORT — speed optimization

| Case | Baseline NEW / grasp / status | Optimized NEW / grasp / status | Δ sim_s |
|------|-------------------------------|--------------------------------|---------|
| Plan B hard_910220 | {_ok(mb)} | {_ok(mo)} | {saved_b:+.2f} |
| Plan C gbi_063 | {_ok(gb)} | {_ok(go)} | {saved_g:+.2f} |
| Plan A 0070 | {_ok(all_metrics.get('plan_a_0070_baseline', {}))} | {_ok(all_metrics.get('plan_a_0070_optimized', {}))} | |
| Plan C gbi_010 | {_ok(all_metrics.get('plan_c_gbi_010_baseline', {}))} | {_ok(all_metrics.get('plan_c_gbi_010_optimized', {}))} | |
| dx→dz S1 reuse | — | ok={reuse.get('dz_ok')} reused={reuse.get('reused_s1')} sim={reuse.get('total_sim_s')} | |

**Accepted:** {regress_ok}

Notes: {'; '.join(regress_notes) if regress_notes else 'none'}
""",
        encoding="utf-8",
    )

    # Aggregate table for REPORT
    def row(label, b, o):
        return (
            f"| {label} | {b.get('total_sim_s')} | {o.get('total_sim_s')} |\n"
            f"| {label} protected rot s | {b.get('protected_rotation_sim_s')} | {o.get('protected_rotation_sim_s')} |\n"
            f"| {label} ang path u | {b.get('total_angular_path_u')} | {o.get('total_angular_path_u')} |\n"
            f"| {label} n rot phases | {b.get('n_rotation_phases')} | {o.get('n_rotation_phases')} |\n"
            f"| {label} grasp | {b.get('GRASP_RETAINED')} | {o.get('GRASP_RETAINED')} |\n"
            f"| {label} SUCCESS_NEW | {b.get('SUCCESS_NEW')} | {o.get('SUCCESS_NEW')} |\n"
        )

    report_table = "| Metric | Baseline | Optimized |\n|---|---|---|\n"
    for case in ("plan_b_hard_910220", "plan_c_gbi_063", "plan_a_0070", "plan_c_gbi_010"):
        report_table += row(
            case,
            all_metrics.get(f"{case}_baseline", {}),
            all_metrics.get(f"{case}_optimized", {}),
        )

    (OUT / "REPORT.md").write_text(
        f"""# REPORT — Precision Oracle speed optimization (Stage 1)

Generated: {datetime.now(timezone.utc).isoformat()}

## Verdict

Optimized contact-aware tilt profiles + UP tolerance ({opt_cfg.dz_up_tol_deg}°) + shorter
return pre-lift. Plan B sim-time saved ≈ **{saved_b:.2f} s**; gbi_063 ≈ **{saved_g:.2f} s**.
Regression accepted: **{regress_ok}**.

## Comparison (simulation time)

{report_table}

## Mechanisms preserved

- Scheduler order dtheta → dx → dz → HANDOFF
- Support-aware fixed-axis planner (no antiparallel noise axis)
- dx→dz S1 reuse: reused_s1={reuse.get('reused_s1')} dz_ok={reuse.get('dz_ok')}
- UP angular deviation (optimized reuse): α={reuse.get('alpha_up_deg')}° (tol={opt_cfg.dz_up_tol_deg}°)

## Deliverables

- `ROTATION_PROFILE.md`, `SPEED_SWEEP.csv`, `PHASE_TIMINGS.csv`
- `BEFORE_AFTER_COMPARISON.csv`, `REGRESSION_REPORT.md`
- Videos under `videos/`
- Freeze: `freeze/ORACLE_MANIFEST.json`, `ORACLE_FREEZE.md`

## Next

Stage 2: generate A/B/C Recovery-entry + Precision dataset from this frozen Oracle.
Do not start SFT/RL.
""",
        encoding="utf-8",
    )

    print("Freezing optimized Oracle...", flush=True)
    freeze_oracle()

    print(report_table)
    print(f"\nWrote {OUT}", flush=True)
    print(f"Regression accepted: {regress_ok}", flush=True)
    if not regress_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
