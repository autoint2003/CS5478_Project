#!/usr/bin/env python
"""Same-IC Seq4 to_UP overshoot regression (before metrics vs after repair) + MP4."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode
from data_generation.oracles.precision.mechanics.open_axis import gates, u_open_world
from data_generation.oracles.precision.mechanics.dual_loop import DT
from data_generation.oracles.precision.corrections.axial_inward import correct_dz_inward
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.handoff import (
    posture_is_fingertips_down,
    run_final_down_handoff_gate,
)
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.mechanics import protected_rotation as sar
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from evaluation.grasp_geometry import cylinder_axis_world
from models.transformer_policy import DT_POLICY
from visualization.camera import mjv_camera_from_preset
from visualization.impact_view import add_capsule, add_sphere
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results/diagnostics/seq4_to_up_overshoot"
IC = ROOT / "results/data_generation/stage2_abc_final_v2/ics/B__hard_910084__post_b1_v2__71c154bb9a00bf73.pkl"
BEFORE = OUT / "DIAGNOSIS.json"
FPS = 20


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    h = min(16 + 15 * len(lines), out.shape[0])
    out[:h] = (18, 18, 22)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 3
        for line in lines:
            dr.text((5, y), str(line)[:105], fill=(240, 240, 240))
            y += 14
        return np.asarray(im)
    except Exception:
        return out


def render(env, start_cap, actions, dest, title):
    import mujoco

    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 480, 640)
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
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"a={g['alpha_up_deg']:.0f}")
        lines = [
            f"{title} | {tag}",
            f"t={float(env.sim.data.time)-t0:.2f}s orient={orient} mode={mode['mode']}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} th={err['dtheta_deg']:.1f} epos={err['e_pos_mm']:.1f}",
            f"alpha={g['alpha_up_deg']:.1f} vrel={err['v_rel_norm']:.3f} fn={err['fn_L']:.1f}/{err['fn_R']:.1f}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), lines)

    img0 = frame("START")
    for _ in range(int(0.4 * FPS)):
        frames.append(img0)
    n = len(actions)
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == n - 1:
            frames.append(frame(f"{i+1}/{n}"))
    for _ in range(int(0.4 * FPS)):
        frames.append(frame("END"))
    rw.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return str(path)


def summarize_series(series, ang_tol=12.0):
    if not series:
        return {}
    ang = [float(s["ang_err_deg"]) for s in series]
    alpha = [float(s["alpha_up_deg"]) for s in series]
    rate = [float(s.get("rate_cmd", s.get("rate", 0)) or 0) for s in series]
    t = [float(s["t"]) for s in series]
    i_min = int(np.argmin(ang))
    first_tol = next((t[i] for i, a in enumerate(ang) if a <= ang_tol), None)
    # duration commanding nonzero rate while alpha already < 25
    near = [
        (t[i], rate[i], alpha[i])
        for i in range(len(series))
        if alpha[i] < 25.0 and rate[i] > 1e-4 and not series[i].get("settle")
    ]
    return {
        "ang0": ang[0],
        "ang1": ang[-1],
        "alpha_final": alpha[-1],
        "min_ang": float(min(ang)),
        "min_ang_t": t[i_min],
        "min_alpha": float(min(alpha)),
        "max_ang_after_min": float(max(ang[i_min:])),
        "overshoot_ang_deg": float(max(0.0, max(ang[i_min:]) - min(ang))),
        "first_in_tol_t": first_tol,
        "n_steps": len(series),
        "duration_s": t[-1] if t else 0.0,
        "mean_rate_alpha_lt_25": float(np.mean([r for _, r, _ in near])) if near else 0.0,
        "duration_cmd_alpha_lt_25_s": float(len(near) * DT),
        "max_rate_alpha_lt_25": float(max([r for _, r, _ in near])) if near else 0.0,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    before = json.loads(BEFORE.read_text(encoding="utf-8")) if BEFORE.exists() else {}

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    cfg = PrecisionConfig()
    blob = pickle.loads(IC.read_bytes())

    # Dense wrap via recording series from actual run_fixed_axis_tilt return
    captured = {}
    _orig = sar.run_fixed_axis_tilt

    def _rec(env, blob, **kw):
        out = _orig(env, blob, **kw)
        # Rebuild dense-ish from returned series + add stop meta
        series = list(out.get("series") or [])
        # Also walk through with denser sampling by replaying actions? Use series as-is
        # Enrich: interpolate not needed; series every 5 steps. Re-run dense diagnostic path:
        captured[kw.get("phase", "?")] = {
            "ang0": out.get("ang0_deg"),
            "ang1": out.get("ang1_deg"),
            "stop_reason": out.get("stop_reason"),
            "min_ang_err_deg": out.get("min_ang_err_deg"),
            "n_actions": out.get("n_actions"),
            "sim_s": out.get("sim_s"),
            "rate_profile": out.get("rate_profile"),
            "series": series,
            "ok": out.get("ok"),
            "reason": out.get("reason"),
        }
        return out

    sar.run_fixed_axis_tilt = _rec

    restore_capture(env, blob["capture"])
    start = blob["capture"]
    actions = []
    rz = correct_dz_inward(env, {"capture": start, "ref": blob["ref"]}, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    up = captured.get("dzIn_to_UP") or {}

    # densify metrics from series if sparse: also load AFTER csv by re-running diag helper
    # Convert series rate key
    series = []
    for s in up.get("series") or []:
        row = dict(s)
        if "rate" in row and "rate_cmd" not in row:
            row["rate_cmd"] = row["rate"]
        series.append(row)
    after_sum = summarize_series(series, ang_tol=float(cfg.dz_up_tol_deg))
    after_sum.update({
        "stop_reason": up.get("stop_reason"),
        "min_ang_err_deg": up.get("min_ang_err_deg"),
        "tilt_n_actions": up.get("n_actions"),
        "tilt_sim_s": up.get("sim_s"),
        "tilt_ok": up.get("ok"),
    })

    # Full handoff path
    gate = None
    handoff = None
    if rz.get("ok") and rz.get("capture") is not None:
        cur = {"capture": rz["capture"], "ref": blob["ref"]}
        gate = run_final_down_handoff_gate(env, cur, cfg, actions)
        if gate.get("ok"):
            cur = {"capture": gate["capture"], "ref": cur["ref"]}
            restore_capture(env, cur["capture"])
            handoff = evaluate_handoff_with_axis(env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n)
            restore_capture(env, cur["capture"])

    video = render(
        env, start, actions,
        OUT / "videos" / "seq4_to_up_after_brake_fix.mp4",
        "SEQ4 after UP brake fix",
    )

    before_to = (before.get("to_up") or {})
    before_sum = {
        "ang0": before_to.get("ang0_deg"),
        "ang1": before_to.get("ang1_deg"),
        "alpha_final": before_to.get("alpha_final"),
        "min_ang": before_to.get("min_ang_err"),
        "min_ang_t": before_to.get("min_ang_t"),
        "min_alpha": before_to.get("min_alpha_all") or before_to.get("min_alpha"),
        "max_ang_after_min": before_to.get("max_ang_after_min"),
        "overshoot_ang_deg": before_to.get("overshoot_ang_deg"),
        "first_in_tol_t": before_to.get("first_in_tol_t"),
        "duration_s": before_to.get("n_actions", 0) * DT if before_to else None,
        "tilt_n_actions": before_to.get("n_actions"),
        "cause_tags": before.get("cause_tags"),
    }
    # from before CSV for alpha<25 commanding
    csv_path = OUT / "to_up_series.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open()))
        near = [
            r for r in rows
            if float(r["alpha_up_deg"]) < 25 and float(r["rate_cmd"]) > 1e-4
            and r.get("settle", "") == ""
        ]
        before_sum["duration_cmd_alpha_lt_25_s"] = len(near) * DT
        before_sum["mean_rate_alpha_lt_25"] = float(np.mean([float(r["rate_cmd"]) for r in near])) if near else 0.0
        before_sum["max_rate_alpha_lt_25"] = float(max([float(r["rate_cmd"]) for r in near])) if near else 0.0

    restore_capture(env, rz.get("capture") or start)
    exit_p = {
        "fingertips_down": posture_is_fingertips_down(env.sim),
        "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
        "errors": measure_errors(env.sim),
    }

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "ic": IC.name,
        "diagnosis_before": before_sum,
        "after": after_sum,
        "dz_ok": rz.get("ok"),
        "dz_branch": rz.get("branch"),
        "dz_n_actions": len(rz.get("actions") or []),
        "total_n_actions": len(actions),
        "outcomes": rz.get("outcomes"),
        "reused_s1": rz.get("reused_s1"),
        "gate_ok": (gate or {}).get("ok"),
        "SUCCESS_NEW": (handoff or {}).get("SUCCESS_NEW"),
        "SUCCESS_12S": (handoff or {}).get("SUCCESS_12S"),
        "exit_posture": exit_p,
        "video": video,
        "cause_before": before.get("cause_tags"),
        "repair": [
            "scheduled_rate: brake band allows rate→0 (no floor clamp near target)",
            "run_fixed_axis_tilt: stop before command when in tolerance; overshoot latch",
            "settle: light angular damping instead of only zero-hold",
            "contact_rate_guard: do not reinflate brake-zero to floor",
        ],
    }
    (OUT / "REGRESSION.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    # Save after series csv from captured (sparse)
    if series:
        keys = ["t", "ang_err_deg", "alpha_up_deg", "rate_cmd", "Fn_z_L", "Fn_z_R", "v_rel_norm", "bilateral", "guard"]
        lines = [",".join(keys)]
        for s in series:
            lines.append(",".join(str(s.get(k, "")) for k in keys))
        (OUT / "to_up_series_after.csv").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "# Seq4 to_UP Overshoot — Diagnosis & Regression",
        "",
        f"Generated: {result['generated']}",
        "",
        "## Diagnosis (pre-repair)",
        "",
        f"- Cause tags: `{before.get('cause_tags')}`",
        "- Root cause: **floor-clamped same-sign rate (0.06)** continued for ~0.8s while α already <25°,",
        "  then **inertial coast** during zero-ω settle (α 11.7→8.7). Not a frame/sign bug;",
        "  not a large past-vertical overshoot (α_min≈8.7°).",
        "",
        "## Before → After (same IC)",
        "",
        "| Metric | Before | After |",
        "|---|---|---|",
        f"| overshoot_ang_deg | {before_sum.get('overshoot_ang_deg')} | {after_sum.get('overshoot_ang_deg')} |",
        f"| ang1 / alpha_final | {before_sum.get('ang1')} / {before_sum.get('alpha_final')} | {after_sum.get('ang1')} / {after_sum.get('alpha_final')} |",
        f"| duration_cmd while α<25 | {before_sum.get('duration_cmd_alpha_lt_25_s')} s | {after_sum.get('duration_cmd_alpha_lt_25_s')} s |",
        f"| mean rate while α<25 | {before_sum.get('mean_rate_alpha_lt_25')} | {after_sum.get('mean_rate_alpha_lt_25')} |",
        f"| max rate while α<25 | {before_sum.get('max_rate_alpha_lt_25')} | {after_sum.get('max_rate_alpha_lt_25')} |",
        f"| to_UP n_actions / sim_s | {before_sum.get('tilt_n_actions')} | {after_sum.get('tilt_n_actions')} / {after_sum.get('tilt_sim_s')} |",
        f"| stop_reason | (post-command tol) | {after_sum.get('stop_reason')} |",
        f"| dz ok / branch | True / B_full | {rz.get('ok')} / {rz.get('branch')} |",
        f"| SUCCESS_12S / NEW | True / False | {result['SUCCESS_12S']} / {result['SUCCESS_NEW']} |",
        "",
        f"Video: `{video}`",
        "",
        "## Repair (minimal)",
        "",
    ]
    for r in result["repair"]:
        lines.append(f"- {r}")
    lines += [
        "",
        "No Stage-1 / GRPO / M111 / dataset / SUCCESS_* definition changes.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
