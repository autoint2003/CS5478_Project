#!/usr/bin/env python
"""Regression: velocity-based brake-before-vertical on Seq4 IC hard_910084."""
from __future__ import annotations

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
from data_generation.oracles.precision.mechanics.open_axis import gates
from data_generation.oracles.precision.mechanics.dual_loop import DT
from data_generation.oracles.precision.corrections import axial_inward as pdz
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

OUT = ROOT / "results/diagnostics/seq4_to_up_brake_before_vertical"
IC = ROOT / "results/data_generation/stage2_abc_final_v2/ics/B__hard_910084__post_b1_v2__71c154bb9a00bf73.pkl"
PRIOR = ROOT / "results/diagnostics/seq4_to_up_overshoot/DIAGNOSIS.json"
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

    dest.parent.mkdir(parents=True, exist_ok=True)
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
            f"t={float(env.sim.data.time)-t0:.2f}s orient={orient} mode={mode['mode']} S1={mode.get('one_sided_mechanical')}",
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
    for _ in range(int(0.5 * FPS)):
        frames.append(frame("END"))
    rw.close()
    return str(write_frames_mp4(frames, dest, fps=FPS))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    prior = json.loads(PRIOR.read_text(encoding="utf-8")) if PRIOR.exists() else {}

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    cfg = PrecisionConfig()
    blob = pickle.loads(IC.read_bytes())

    captured = {}
    _orig = sar.run_fixed_axis_tilt

    def _rec(env, blob, **kw):
        out = _orig(env, blob, **kw)
        captured[kw.get("phase", "?")] = {
            k: out.get(k)
            for k in (
                "ok", "reason", "ang0_deg", "ang1_deg", "alpha_final_deg",
                "stop_reason", "min_ang_err_deg", "min_alpha_up_deg",
                "crossed_vertical", "brake_begin", "short_bias_deg",
                "braking_latched", "n_actions", "sim_s", "series", "mode_exit",
            )
        }
        return out

    # Patch both module and precision_dz's bound import.
    sar.run_fixed_axis_tilt = _rec
    pdz.run_fixed_axis_tilt = _rec

    restore_capture(env, blob["capture"])
    start = blob["capture"]
    actions: list = []
    rz = correct_dz_inward(env, {"capture": start, "ref": blob["ref"]}, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    up = captured.get("dzIn_to_UP") or {}

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
        OUT / "videos" / "seq4_to_up_brake_before_vertical.mp4",
        "SEQ4 to_UP brake-before-vertical",
    )

    series = up.get("series") or []
    # settling time: from brake_begin to end of tilt sim
    bb = up.get("brake_begin") or {}
    settle_s = None
    if bb.get("t") is not None and up.get("sim_s") is not None:
        settle_s = float(up["sim_s"]) - float(bb["t"])

    # S1 continuity during to_UP series
    s1_frac = None
    s1_mode_frac = None
    bilateral_frac = None
    if series:
        s1_frac = float(np.mean([
            1.0 if max(float(s.get("Fn_z_L") or 0), float(s.get("Fn_z_R") or 0)) > 1.0 else 0.0
            for s in series
        ]))
        s1_mode_frac = float(np.mean([
            1.0 if (s.get("mode") == "ONE_SIDED_MECHANICAL_SUPPORT" or s.get("support_side")) else 0.0
            for s in series
        ]))
        bilateral_frac = float(np.mean([1.0 if s.get("bilateral") else 0.0 for s in series]))

    restore_capture(env, rz.get("capture") or start)
    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "ic": IC.name,
        "prior_cause_tags": prior.get("cause_tags"),
        "prior_to_up": prior.get("to_up"),
        "to_up": {
            "ok": up.get("ok"),
            "reason": up.get("reason"),
            "stop_reason": up.get("stop_reason"),
            "brake_begin": bb,
            "ang0_deg": up.get("ang0_deg"),
            "ang1_deg": up.get("ang1_deg"),
            "alpha_final_deg": up.get("alpha_final_deg"),
            "min_ang_err_deg": up.get("min_ang_err_deg"),
            "min_alpha_up_deg": up.get("min_alpha_up_deg"),
            "crossed_vertical": up.get("crossed_vertical"),
            "short_bias_deg": up.get("short_bias_deg"),
            "braking_latched": up.get("braking_latched"),
            "n_actions": up.get("n_actions"),
            "sim_s": up.get("sim_s"),
            "settle_s_after_brake": settle_s,
            "s1_force_proxy_frac": s1_frac,
            "s1_mode_frac": s1_mode_frac,
            "bilateral_frac": bilateral_frac,
            "mode_exit": up.get("mode_exit"),
        },
        "dz_ok": rz.get("ok"),
        "dz_branch": rz.get("branch"),
        "reused_s1": rz.get("reused_s1"),
        "outcomes": rz.get("outcomes"),
        "gate_ok": (gate or {}).get("ok"),
        "SUCCESS_12S": (handoff or {}).get("SUCCESS_12S"),
        "SUCCESS_NEW": (handoff or {}).get("SUCCESS_NEW"),
        "exit_down": posture_is_fingertips_down(env.sim),
        "exit_alpha": float(gates(env.sim)["alpha_up_deg"]),
        "video": video,
        "acceptance": {
            "no_cross_vertical": not bool(up.get("crossed_vertical")),
            "stop_short_preferred": bool(
                up.get("min_ang_err_deg") is not None and float(up.get("min_ang_err_deg")) > 0.5
            ),
            "mechanism_ok": bool(rz.get("ok")),
            "s1_contact_retained": bool(bilateral_frac is not None and bilateral_frac > 0.95),
        },
    }
    # Save series csv
    if series:
        keys = [
            "t", "ang_err_deg", "alpha_up_deg", "rate", "omega_closing", "coast_deg",
            "braking_latched", "Fn_z_L", "Fn_z_R", "v_rel_norm", "bilateral", "guard", "stop_reason",
        ]
        lines = [",".join(keys)]
        for s in series:
            lines.append(",".join(str(s.get(k, "")) for k in keys))
        (OUT / "to_up_series.csv").write_text("\n".join(lines), encoding="utf-8")

    (OUT / "RESULT.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    bb = result["to_up"]["brake_begin"] or {}
    lines = [
        "# to_UP Brake-Before-Vertical Regression",
        "",
        f"IC: `{IC.name}`",
        f"Generated: {result['generated']}",
        "",
        "## Brake begin",
        "",
        f"- t = {bb.get('t')}",
        f"- ang_err = {bb.get('ang_err_deg')}",
        f"- alpha = {bb.get('alpha_up_deg')}",
        f"- omega_closing = {bb.get('omega_closing_rad_s')} rad/s",
        f"- estimated coast = {bb.get('coast_deg')}°",
        f"- short_bias = {bb.get('short_bias_deg')}°",
        "",
        "## Approach / settle",
        "",
        f"- min remaining ang to vertical during motion: **{result['to_up']['min_ang_err_deg']}°**",
        f"- min alpha: **{result['to_up']['min_alpha_up_deg']}°**",
        f"- crossed vertical: **{result['to_up']['crossed_vertical']}**",
        f"- final UP ang / alpha: **{result['to_up']['ang1_deg']}°** / **{result['to_up']['alpha_final_deg']}°**",
        f"- stop_reason: `{result['to_up']['stop_reason']}`",
        f"- settle time after brake begin: {settle_s} s",
        "",
        "## Mechanism / HANDOFF",
        "",
        f"- S1 reuse: {result['reused_s1']} (cold establish)",
        f"- S1/bilateral during to_UP: force_proxy={s1_frac} mode={s1_mode_frac} bilateral={bilateral_frac}",
        f"- DZ ok / branch: {result['dz_ok']} / {result['dz_branch']}",
        f"- DualLoop / outcomes: {result['outcomes']}",
        f"- SUCCESS_12S / SUCCESS_NEW: **{result['SUCCESS_12S']}** / **{result['SUCCESS_NEW']}**",
        f"- exit DOWN: {result['exit_down']} α={result['exit_alpha']:.1f}",
        "",
        f"Video: `{video}`",
        "",
        f"Acceptance: {result['acceptance']}",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
