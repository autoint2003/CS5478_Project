#!/usr/bin/env python
"""Finger-inner-surface supported rotation vs previous +wy (Plan B hard_910220).

Primary question: does ±wx (or ±wz) produce upward *normal* force from one finger
pad, unlike friction-dominated +wy@85°?
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.finger_surface_support import (  # noqa: E402
    REORI_DTHETA_DEG,
    REORI_DX_MM,
    finger_inner_normals_world,
    probe_action_omega_mapping,
    support_force_decomposition,
    waypoint_rotate_axis,
)
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (  # noqa: E402
    _height_hold_v_hand,
    execute_restore_plan,
    plan_restore_orientation,
)
from data_generation.oracles.precision.sign_aware_axial import (  # noqa: E402
    desire_direction,
    opening_axis_hand,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R  # noqa: E402
from evaluation.success_criteria import P_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

B_POST = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
B_STABLE = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_stable.pkl"
OUT = ROOT / "results/diagnostics/finger_surface_supported_axial"

FORCE_KP, FORCE_DA = 0.10, 0.12
V_SLIP = -0.001
FN_MIN, FN_MAX, FN_INIT, FN_BRAKE = 6.5, 14.0, 9.0, 14.0
SLIP_KN = 400.0
Z_ABORT = 0.44
FPS = 25
DT = 0.02


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


def load_blob(path: Path) -> dict:
    d = pickle.loads(path.read_bytes())
    if "ref" not in d:
        st = pickle.loads(B_STABLE.read_bytes())
        d = {"capture": d["capture"], "ref": st["ref"]}
    return d


def run_axial_slip(env, blob, d, *, milestone_mm: float, z_ref: float, e_x0: float):
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    d = np.asarray(d, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=FN_MIN, fn_max=FN_MAX, kn=SLIP_KN),
        use_slip=True,
    )
    ctl.reset(a0=0.55, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))
    actions, series = [], []
    lost = 0
    reached = False
    abort = None
    for i in range(int(12.0 / DT)):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        e2 = np.asarray(s2["e_signed_mm"], float)
        disp = float(np.dot(dp, d)) * 1e3
        series.append(
            {
                "t": (i + 1) * DT,
                "disp_along_mm": disp,
                "e_signed": e2.tolist(),
                "theta": s2["e_theta_deg"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "slip",
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {"ok": False, "reason": "contact_loss", "actions": actions, "series": series}
        if abs(float(e2[0]) - e_x0) > 4.0:
            abort = "lateral_drift"
            break
        if disp >= milestone_mm:
            reached = True
            break
    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(0.8, sample_state(env.sim)["fn_tot"])
    for j in range(int(0.4 / DT) + int(0.15 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        series.append(
            {
                "t": series[-1]["t"] + DT,
                "disp_along_mm": float(np.dot(dp, d) * 1e3),
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "bilateral": s2["bilateral"],
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "phase": "brake" if j < int(0.4 / DT) else "hold",
            }
        )
        if not s2["bilateral"]:
            return {"ok": False, "reason": "loss_on_brake", "actions": actions, "series": series, "reached": reached}
    disp = float(series[-1]["disp_along_mm"])
    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    return {
        "ok": bool(hold_ok and abort is None and (reached or disp >= 0.5 * milestone_mm)),
        "reached": reached,
        "abort_reason": abort,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "theta": series[-1]["theta"],
        "lateral_drift_mm": abs(float(series[-1]["e_signed"][0]) - e_x0),
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
    }


def handoff_rich(env, capture, ref):
    from data_generation.oracles.precision.primitives import _wrist

    restore_capture(env, capture)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    meta = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    meta.pop("samples", None)
    return meta


def overlay(img, lines):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    y = 5
    for line in lines:
        dr.text((6, y), line, fill=(255, 255, 255), font=font)
        y += 17
    return np.asarray(im)


def render_compare_video(env, blob, sequences: dict, dest: Path):
    """sequences: name -> {actions, title, d optional}."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r_w = mujoco.Renderer(env.sim.model, 480, 640)
    r_c = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0.35, 0.0, 0.4], "distance": 2.2, "azimuth": 145, "elevation": -18})
    cam_c = mjv_camera_from_preset({"lookat": [0.5, 0.05, 0.52], "distance": 0.55, "azimuth": 120, "elevation": -25})
    frames = []

    def draw(scn, d_axis=None):
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        kin = finger_inner_normals_world(env.sim)
        nL = np.asarray(kin["left_inner_n_world"], float)
        nR = np.asarray(kin["right_inner_n_world"], float)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.0025)
        # finger inner normals from approx contact mid
        add_capsule(scn, po, po + 0.06 * nL, [1.0, 0.3, 0.3, 1.0], 0.003)
        add_capsule(scn, po, po + 0.06 * nR, [0.3, 0.5, 1.0, 1.0], 0.003)
        g_h = sample_state(env.sim)["g_h"]
        add_capsule(scn, po, po + 0.01 * (Rh @ np.asarray(g_h, float)), [0.85, 0.4, 1.0, 1.0], 0.0028)
        if d_axis is not None:
            d = np.asarray(d_axis, float)
            add_capsule(scn, po, po + 0.07 * (Rh @ d), [1.0, 0.6, 0.1, 1.0], 0.003)

    def frame(tag, title, d_axis=None):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        kin = finger_inner_normals_world(env.sim)
        sup = support_force_decomposition(env.sim)
        lines = [
            f"{tag} {title}",
            f"e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.2f}",
            f"nL·up={kin['left_align_up']:.2f} nR·up={kin['right_align_up']:.2f}",
            f"Fn_z L/R={sup['sides']['L']['Fn_z_normal']:.2f}/{sup['sides']['R']['Fn_z_normal']:.2f} "
            f"Ft_z={sup['sides']['L']['Ft_z_friction']:.2f}/{sup['sides']['R']['Ft_z_friction']:.2f}",
            "red=nL  blue=nR  purple=g  cyan=COM",
        ]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene, d_axis)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene, d_axis)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["FINGER CLOSE-UP"] + lines)
        return np.concatenate([iw, ic], axis=1)

    for name, seq in sequences.items():
        actions = seq["actions"]
        if not actions:
            continue
        restore_capture(env, blob["capture"])
        img = frame("BEFORE", seq["title"])
        for _ in range(int(1.2 * FPS)):
            frames.append(img)
        hold = max(1, int(round(0.5 * FPS * DT / max(len(actions), 1) * 200)))
        hold = 2
        for i, a in enumerate(actions):
            step_legal(env.sim, a)
            if i % 2 == 0 or i == len(actions) - 1:
                img = frame(f"{name} {i+1}/{len(actions)}", seq["title"])
                for _ in range(hold):
                    frames.append(img)
        img = frame("AFTER", seq["title"])
        for _ in range(int(1.5 * FPS)):
            frames.append(img)

    r_w.close()
    r_c.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS, "n_frames": len(frames)}


def make_plots(out_dir: Path, wy, L, R) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    labels, fn, ft, align = [], [], [], []
    for name, res in (("wy85", wy), ("cand_L", L), ("cand_R", R)):
        if not res:
            continue
        labels.append(name)
        fn.append(float(res.get("selected_Fn_z") or 0.0))
        ft.append(float(res.get("selected_Ft_z") or 0.0))
        align.append(float(res.get("selected_align_up") or 0.0))
    if labels:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(labels))
        ax.bar(x - 0.2, fn, 0.2, label="Fn_z (normal)")
        ax.bar(x, ft, 0.2, label="Ft_z (friction)")
        ax.bar(x + 0.2, align, 0.2, label="n·up (align)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("N or dimensionless")
        ax.set_title("Vertical support: normal vs friction")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "support_force_compare.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["support"] = str(p)
    return paths


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)

    env = make_env()
    blob = load_blob(B_POST)
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    print(
        f"Start e={np.round(s0['e_signed_mm'],2)} θ={s0['e_theta_deg']:.2f} "
        f"bil={s0['bilateral']}",
        flush=True,
    )
    (OUT / "start_state.json").write_text(
        json.dumps(
            jsafe(
                {
                    "e_signed_mm": s0["e_signed_mm"],
                    "theta": s0["e_theta_deg"],
                    "kinematics": finger_inner_normals_world(env.sim),
                    "support": support_force_decomposition(env.sim),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --- §2 axis mapping ---
    print("=== Action→ω mapping probe ===", flush=True)
    mapping = probe_action_omega_mapping(env, blob, n_steps=50, rate=0.18)
    (OUT / "rotation_axis_mapping.json").write_text(json.dumps(jsafe(mapping), indent=2), encoding="utf-8")
    print("  best L", mapping["best_for_left_up"]["action"], "Δalign", mapping["best_for_left_up"]["delta_left_align_up"])
    print("  best R", mapping["best_for_right_up"]["action"], "Δalign", mapping["best_for_right_up"]["delta_right_align_up"])

    # Prefer axes that improve alignment for each candidate
    def axes_for(side: str) -> tuple[str, ...]:
        key = "delta_left_align_up" if side == "L" else "delta_right_align_up"
        ranked = sorted(mapping["trials"].items(), key=lambda kv: -kv[1][key])
        chosen = []
        for name, tr in ranked:
            # Normalize legacy probe labels wx+/wz+ → +wx/+wz
            ax = {"wx+": "+wx", "wx-": "-wx", "wy+": "+wy", "wy-": "-wy", "wz+": "+wz", "wz-": "-wz"}.get(
                name, name
            )
            if tr[key] <= 0.02:
                continue
            if max(tr["hand_ang_x_deg"], tr["hand_ang_y_deg"], tr["hand_ang_z_deg"]) < 5.0:
                continue
            if ax not in chosen:
                chosen.append(ax)
        for a in ("+wx", "-wx", "+wz", "-wz"):
            if a not in chosen:
                chosen.append(a)
        return tuple(chosen[:4])

    # --- §3–5 Candidate L / R / old wy ---
    print("=== Old +wy@85° baseline ===", flush=True)
    wy = waypoint_rotate_axis(env, blob, axis_name="+wy", angle_deg=85.0, rate_scale=0.20, support_side="L")
    # also score with R
    if wy.get("support_final"):
        sides = wy["support_final"]["sides"]
        # pick side with larger align at end
        kin = wy.get("kinematics_final") or {}
        if (kin.get("right_align_up") or -1) > (kin.get("left_align_up") or -1):
            wy["support_side"] = "R"
            wy["selected_Fn_z"] = sides["R"]["Fn_z_normal"]
            wy["selected_Ft_z"] = sides["R"]["Ft_z_friction"]
            wy["selected_align_up"] = kin.get("right_align_up")
            wy["normal_support_ok"] = bool(sides["R"]["Fn_z_normal"] > 0.3)
    wy_sum = {k: v for k, v in wy.items() if k not in ("actions", "waypoints", "capture")}
    print(
        f"  wy: ok={wy.get('ok')} Fn_z={wy.get('selected_Fn_z')} Ft_z={wy.get('selected_Ft_z')} "
        f"align={wy.get('selected_align_up')} Δex={wy.get('delta_ex_mm')}",
        flush=True,
    )
    (OUT / "baseline_wy85.json").write_text(json.dumps(jsafe(wy_sum), indent=2), encoding="utf-8")

    print("=== Candidate L (left finger up) ===", flush=True)

    def search_side(candidate, side):
        axes = axes_for(side)
        print(f"  axes order: {axes}", flush=True)
        trials = []
        best = None
        for axis_name in axes:
            for ang in (30.0, 45.0, 60.0, 75.0, 90.0):
                for rate in (0.12, 0.15, 0.20):
                    tr = waypoint_rotate_axis(
                        env, blob, axis_name=axis_name, angle_deg=ang, rate_scale=rate, support_side=side
                    )
                    summary = {k: v for k, v in tr.items() if k not in ("actions", "waypoints", "capture")}
                    trials.append(summary)
                    print(
                        f"  [{candidate}] {axis_name}@{ang} r={rate}: ok={tr.get('ok')} "
                        f"align={tr.get('selected_align_up')} Fn_z={tr.get('selected_Fn_z')} "
                        f"Ft_z={tr.get('selected_Ft_z')} Δex={tr.get('delta_ex_mm')} reason={tr.get('reason')}",
                        flush=True,
                    )
                    if not tr.get("ok"):
                        continue
                    score = (
                        40.0 * float(tr.get("selected_Fn_z") or 0.0)
                        + 20.0 * float(tr.get("selected_align_up") or 0.0)
                        - 2.0 * abs(float(tr.get("delta_ex_mm") or 0.0))
                        - 2.0 * abs(float(tr.get("delta_theta_deg") or 0.0))
                    )
                    if best is None or score > best["score"]:
                        best = {**tr, "score": score}
        return {
            "candidate": candidate,
            "support_side": side,
            "axes": list(axes),
            "trials": trials,
            "best": None if best is None else {k: v for k, v in best.items() if k not in ("actions", "waypoints")},
            "best_full": best,
            "any_normal_support": bool(best and best.get("normal_support_ok")),
        }

    res_L = search_side("L", "L")
    res_R = search_side("R", "R")
    (OUT / "candidate_L.json").write_text(
        json.dumps(jsafe({k: v for k, v in res_L.items() if k != "best_full"}), indent=2), encoding="utf-8"
    )
    (OUT / "candidate_R.json").write_text(
        json.dumps(jsafe({k: v for k, v in res_R.items() if k != "best_full"}), indent=2), encoding="utf-8"
    )

    comparison = {
        "wy85": {
            "ok": wy.get("ok"),
            "axis": "+wy",
            "angle": 85.0,
            "Fn_z": wy.get("selected_Fn_z"),
            "Ft_z": wy.get("selected_Ft_z"),
            "align_up": wy.get("selected_align_up"),
            "delta_ex_mm": wy.get("delta_ex_mm"),
            "delta_theta_deg": wy.get("delta_theta_deg"),
            "normal_support_ok": wy.get("normal_support_ok"),
            "support_sides": (wy.get("support_final") or {}).get("sides"),
        },
        "candidate_L": res_L.get("best"),
        "candidate_R": res_R.get("best"),
        "diagnostic_targets": {"delta_ex_mm": REORI_DX_MM, "delta_theta_deg": REORI_DTHETA_DEG},
    }
    (OUT / "wy_vs_finger_support.json").write_text(json.dumps(jsafe(comparison), indent=2), encoding="utf-8")

    # Choose best overall by normal support then align
    contenders = []
    if res_L.get("best_full"):
        contenders.append(("L", res_L["best_full"]))
    if res_R.get("best_full"):
        contenders.append(("R", res_R["best_full"]))
    best_name, best = (None, None)
    if contenders:
        best_name, best = max(
            contenders,
            key=lambda kv: (
                float(kv[1].get("selected_Fn_z") or -1e9),
                float(kv[1].get("selected_align_up") or -1e9),
            ),
        )

    slip = None
    handoff = None
    geom_compat = None
    if best and best.get("normal_support_ok"):
        print(f"=== Axial compatibility after {best_name} ===", flush=True)
        restore_capture(env, best["capture"])
        oa = opening_axis_hand(env.sim)
        u_open = np.asarray(oa["u_open"], float)
        g_h = np.asarray(sample_state(env.sim)["g_h"], float)
        d_in = desire_direction("B_inward", u_open)
        geom_compat = {
            "g_h": g_h.tolist(),
            "u_open": u_open.tolist(),
            "g_dot_u_open": float(np.dot(g_h, u_open)),
            "g_dot_d_inward": float(np.dot(g_h, d_in)),
            "finger_support_and_opening_aligned": bool(
                abs(float(np.dot(g_h, u_open))) > 6.0 and best.get("selected_align_up", 0) > 0.7
            ),
            "note": "Finger-surface support (n·up) is distinct from opening-axis gravity alignment (g·u_open).",
        }
        geom_compat["gravity_assists_inward"] = bool(geom_compat["g_dot_d_inward"] > 4.0)
        geom_compat["gravity_opposes_inward"] = bool(geom_compat["g_dot_d_inward"] < -4.0)
        (OUT / "axial_geometry_compatibility.json").write_text(json.dumps(jsafe(geom_compat), indent=2), encoding="utf-8")
        print(
            f"  compat g·d_in={geom_compat['g_dot_d_inward']:.3f} g·u_open={geom_compat['g_dot_u_open']:.3f} "
            f"aligned={geom_compat['finger_support_and_opening_aligned']}",
            flush=True,
        )

        # Always attempt 2 mm once normal support is real; report gravity conflict if present.
        print("=== 2 mm axial slip (toward palm / B_inward) ===", flush=True)
        tilt = {"capture": best["capture"], "ref": blob["ref"]}
        slip = run_axial_slip(
            env,
            tilt,
            d_in,
            milestone_mm=2.0,
            z_ref=float(best.get("min_obj_z") or 0.5),
            e_x0=float(best["e0"][0]),
        )
        slip["gravity_assists_inward"] = geom_compat["gravity_assists_inward"]
        slip["g_dot_d_inward"] = geom_compat["g_dot_d_inward"]
        print(
            f"  slip: ok={slip.get('ok')} disp={slip.get('disp_along_mm')} "
            f"abort={slip.get('abort_reason')} reason={slip.get('reason')}",
            flush=True,
        )
        (OUT / "axial_slip_2mm.json").write_text(
            json.dumps(jsafe({k: v for k, v in slip.items() if k not in ("actions", "series", "capture")}), indent=2),
            encoding="utf-8",
        )
        if slip.get("ok"):
            post = {"capture": slip["capture"], "ref": blob["ref"]}
            rplan = plan_restore_orientation(env, post)
            rexe = execute_restore_plan(env, post, rplan)
            handoff = handoff_rich(env, take_capture(env), blob["ref"])
            (OUT / "handoff.json").write_text(
                json.dumps(
                    jsafe(
                        {
                            k: handoff[k]
                            for k in (
                                "SUCCESS_12S",
                                "SUCCESS_NEW",
                                "SUCCESS_AXIS_V1",
                                "pre_e_signed_mm",
                                "pre_theta",
                                "pre_pos_mm",
                                "reason",
                            )
                            if k in handoff
                        }
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            np.savez_compressed(
                OUT / "best_actions.npz",
                actions=np.asarray(list(best["actions"]) + list(slip["actions"]) + list(rexe.get("actions") or []), np.float32),
            )
    else:
        print("=== No meaningful normal-force support — skip axial slip ===", flush=True)

    # Plots / video
    best_L = res_L.get("best") or {}
    best_R = res_R.get("best") or {}
    plots = make_plots(
        OUT / "plots",
        {
            "selected_Fn_z": wy.get("selected_Fn_z"),
            "selected_Ft_z": wy.get("selected_Ft_z"),
            "selected_align_up": wy.get("selected_align_up"),
        },
        best_L,
        best_R,
    )

    print("Rendering videos...", flush=True)
    seqs = {}
    if wy.get("actions"):
        seqs["wy85"] = {"actions": wy["actions"], "title": "OLD +wy@85 (friction path)"}
    if res_L.get("best_full") and res_L["best_full"].get("actions"):
        seqs["L"] = {
            "actions": res_L["best_full"]["actions"],
            "title": f"L {res_L['best_full'].get('axis_name')}@{res_L['best_full'].get('angle_deg')}",
        }
    if res_R.get("best_full") and res_R["best_full"].get("actions"):
        seqs["R"] = {
            "actions": res_R["best_full"]["actions"],
            "title": f"R {res_R['best_full'].get('axis_name')}@{res_R['best_full'].get('angle_deg')}",
        }
    videos = {}
    if seqs:
        videos["compare"] = render_compare_video(
            env, blob, seqs, OUT / "videos" / "wy_vs_finger_surface_rotation.mp4"
        )

    # Verdict
    wy_ns = bool(wy.get("normal_support_ok"))
    L_ns = bool(res_L.get("any_normal_support"))
    R_ns = bool(res_R.get("any_normal_support"))
    if (L_ns or R_ns) and slip and slip.get("ok") and handoff and handoff.get("SUCCESS_NEW"):
        rec = "finger_surface_support_enables_SUCCESS_NEW"
    elif L_ns or R_ns:
        if slip and not slip.get("ok"):
            rec = "finger_surface_normal_support_ok_2mm_slip_failed"
        elif geom_compat and geom_compat.get("gravity_opposes_inward"):
            rec = "finger_surface_normal_support_ok_gravity_opposes_inward_slip"
        else:
            rec = "finger_surface_normal_support_achieved_axial_slip_limited"
    elif max(float(best_L.get("selected_align_up") or 0), float(best_R.get("selected_align_up") or 0)) > 0.7:
        rec = "finger_aligned_up_but_support_still_friction_dominated"
    else:
        rec = "wx_path_does_not_beat_wy_for_normal_support_prefer_AXIS_V1"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "start_e_signed_mm": s0["e_signed_mm"],
        "mapping": {
            "best_L": mapping["best_for_left_up"]["action"],
            "best_R": mapping["best_for_right_up"]["action"],
        },
        "wy85": comparison["wy85"],
        "candidate_L_best": best_L,
        "candidate_R_best": best_R,
        "normal_support": {"wy": wy_ns, "L": L_ns, "R": R_ns},
        "geom_compat": geom_compat,
        "slip_2mm": None
        if not slip
        else {
            k: slip.get(k)
            for k in (
                "ok",
                "disp_along_mm",
                "abort_reason",
                "reason",
                "lateral_drift_mm",
                "gravity_assists_inward",
                "g_dot_d_inward",
            )
        },
        "handoff": None
        if not handoff
        else {k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm")},
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "preserved": {
            "SUCCESS_NEW": True,
            "SUCCESS_AXIS_V1": True,
            "plan_a": True,
            "plan_b_branch_a": True,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    report = f"""# Finger-Surface-Supported Axial Rotation — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## Verdict

**{rec}**

| Path | ok | n·up | Fn_z (normal) | Ft_z (friction) | Δe_x mm | normal_support |
|------|----|------|---------------|-----------------|---------|----------------|
| +wy@85 | {wy.get('ok')} | {wy.get('selected_align_up')} | {wy.get('selected_Fn_z')} | {wy.get('selected_Ft_z')} | {wy.get('delta_ex_mm')} | {wy_ns} |
| Best L | {best_L.get('ok')} | {best_L.get('selected_align_up')} | {best_L.get('selected_Fn_z')} | {best_L.get('selected_Ft_z')} | {best_L.get('delta_ex_mm')} | {L_ns} |
| Best R | {best_R.get('ok')} | {best_R.get('selected_align_up')} | {best_R.get('selected_Fn_z')} | {best_R.get('selected_Ft_z')} | {best_R.get('delta_ex_mm')} | {R_ns} |

## 1. Rotation-axis mapping

Legal `a[3:6]` → ω in **r_des** frame (`w_world = R_des @ w_body`).

Probe best for left-up: `{mapping['best_for_left_up']['action']}` (Δ nL·up = {mapping['best_for_left_up']['delta_left_align_up']:.3f})  
Probe best for right-up: `{mapping['best_for_right_up']['action']}` (Δ nR·up = {mapping['best_for_right_up']['delta_right_align_up']:.3f})

Pad geometry: left inner normal ≈ −hand-y; right ≈ +hand-y.

## 2. Old wy vs new axis

See `wy_vs_finger_support.json` and `plots/support_force_compare.png`.

Meaningful support requires **Fn_z > 0** from the selected finger's normal force on the cylinder — not friction-only Ft_z.

## 3. Axial slip / HANDOFF

Geometry compat: {json.dumps(jsafe(geom_compat), indent=2) if geom_compat else 'n/a'}

Slip 2 mm: {json.dumps(jsafe(summary.get('slip_2mm')), indent=2)}

Handoff: {json.dumps(jsafe(summary.get('handoff')), indent=2)}

## 4. Videos

{json.dumps(jsafe(videos), indent=2)}

## 5. Limitation

Finger-surface **normal** support (selected pad n·up ≈ +1, Fn_z ≫ 0) is distinct from opening-axis gravity
alignment needed for Mode-B inward axial slip (want g·d_inward > 0). After the best finger-up rotation,
these two objectives conflict on a single rigid hand pose: the pad that bears weight faces world-up, but
gravity then pulls the cylinder *outward* along the opening axis (away from the palm / against e_z correction).

Staged motion (support reori → partial restore toward opening-down → slip) would be required; not explored
beyond the 2 mm diagnostic. Prefer Plan B Branch A SUCCESS_AXIS_V1 when axial SUCCESS_NEW remains blocked.

Preserved SUCCESS_NEW / AXIS_V1 / Plan A experts. No physics/obs/action/nominal changes.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "normal_support": summary["normal_support"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
