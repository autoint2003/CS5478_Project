#!/usr/bin/env python
"""Finger-support → fingertips-up → inward axial slip (Plan B hard_910220).

Correct sequence: do NOT slip at intermediate ±wz@90° pad-up pose.
Slip only after opening axis aligns with world-up (α ≤ 5°) and g·d_inward > 0.
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
from data_generation.oracles.precision.mechanics.finger_vertical import (  # noqa: E402
    ALPHA_GATE_DEG,
    U_OPEN_HAND,
    opening_alignment,
    run_support_then_vertical,
    vertical_gate_report,
)
from data_generation.oracles.precision.mechanics.finger_surface import (  # noqa: E402
    finger_inner_normals_world,
    support_force_decomposition,
)
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.corrections.orientation import (  # noqa: E402
    _height_hold_v_hand,
    execute_restore_plan,
    plan_restore_orientation,
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
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"
STABLE_JSON = ROOT / "results/diagnostics/axis_aware_precision/stable_state.json"
OUT = ROOT / "results/diagnostics/finger_supported_vertical_axial"

FORCE_KP, FORCE_DA = 0.10, 0.12
V_SLIP = -0.001
# Calibrated DualLoop (sign_aware / force_slip defaults). Stiff FN_MIN≥6.5 stuck at vertical.
FN_MIN, FN_MAX, FN_INIT, FN_BRAKE = 3.8, 11.0, 5.2, 14.0
SLIP_KN = 900.0
Z_ABORT = 0.44
DT = 0.02
FPS = 25
A0_SLIP = 0.55


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
    return env


def load_blob(path: Path):
    return pickle.loads(path.read_bytes())


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
    ctl.reset(a0=A0_SLIP, fn0=s0["fn_tot"], fn_target=FN_INIT, slip_axis=tuple((-d).tolist()))
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
                "v_rel_norm": s2["v_rel_norm"],
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
                "v_rel_norm": s2["v_rel_norm"],
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
    from data_generation.oracles.precision.corrections.contact_primitives import _wrist

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
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    y = 4
    for line in lines:
        dr.text((5, y), line, fill=(255, 255, 255), font=font)
        y += 16
    return np.asarray(im)


def render_path_video(env, blob, actions, dest: Path, title: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r_w = mujoco.Renderer(env.sim.model, 480, 640)
    r_c = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0.35, 0.0, 0.4], "distance": 2.2, "azimuth": 145, "elevation": -18})
    cam_c = mjv_camera_from_preset({"lookat": [0.5, 0.05, 0.52], "distance": 0.55, "azimuth": 120, "elevation": -25})
    frames = []

    def draw(scn):
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u_cyl = cylinder_axis_world(env.sim)
        oa = opening_alignment(env.sim)
        u_open = np.asarray(oa["u_open_world"], float)
        kin = finger_inner_normals_world(env.sim)
        nL = np.asarray(kin["left_inner_n_world"], float)
        nR = np.asarray(kin["right_inner_n_world"], float)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u_cyl, po + 0.05 * u_cyl, [0.2, 0.95, 0.25, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.08 * u_open, [1.0, 0.85, 0.1, 1.0], 0.0035)  # opening / fingertips
        add_capsule(scn, po, po + 0.06 * nL, [1.0, 0.3, 0.3, 1.0], 0.003)
        add_capsule(scn, po, po + 0.06 * nR, [0.3, 0.5, 1.0, 1.0], 0.003)
        add_capsule(scn, po, po + 0.07 * np.array([0, 0, 1.0]), [0.85, 0.4, 1.0, 1.0], 0.0028)  # world up

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        oa = opening_alignment(env.sim)
        sup = support_force_decomposition(env.sim)
        lines = [
            f"{tag} {title}",
            f"α_open_up={oa['alpha_open_up_deg']:.1f}°  g·d_in={oa['g_dot_d_inward']:.2f}",
            f"e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.2f}",
            f"Fn_z L/R={sup['sides']['L']['Fn_z_normal']:.2f}/{sup['sides']['R']['Fn_z_normal']:.2f}",
            "yellow=u_open  purple=world_up  red/blue=pad normals",
        ]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, ic], axis=1)

    restore_capture(env, blob["capture"])
    img = frame("START")
    for _ in range(int(1.0 * FPS)):
        frames.append(img)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % 2 == 0 or i == len(actions) - 1:
            img = frame(f"{i+1}/{len(actions)}")
            frames.append(img)
            frames.append(img)
    img = frame("END")
    for _ in range(int(1.5 * FPS)):
        frames.append(img)
    r_w.close()
    r_c.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS, "n_frames": len(frames)}


def make_plots(out_dir: Path, paths: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, res in paths.items():
        series = res.get("stage_B_series") or []
        if not series:
            continue
        t = [r["t"] for r in series]
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axs[0].plot(t, [r["alpha_open_up_deg"] for r in series], label="α_open_up")
        axs[0].axhline(ALPHA_GATE_DEG, color="r", ls="--", label="gate 5°")
        axs[0].set_ylabel("deg")
        axs[0].legend()
        axs[0].set_title(f"{name}: opening alignment & support")
        axs[1].plot(t, [r["Fn_z_L"] for r in series], label="Fn_z L")
        axs[1].plot(t, [r["Fn_z_R"] for r in series], label="Fn_z R")
        axs[1].plot(t, [r["g_dot_d_inward"] for r in series], label="g·d_in")
        axs[1].axhline(0, color="k", lw=0.5)
        axs[1].set_ylabel("N or m/s²")
        axs[1].legend()
        axs[2].plot(t, [r["delta_ex_mm"] for r in series], label="Δe_x")
        axs[2].plot(t, [r.get("theta", 0) for r in series], label="θ")
        axs[2].set_xlabel("t (s)")
        axs[2].legend()
        fig.tight_layout()
        p = out_dir / f"trace_{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out[name] = str(p)
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "plots", "videos"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    blob = load_blob(B_POST)
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    oa0 = opening_alignment(env.sim)
    gate0 = vertical_gate_report(env.sim)
    print(
        f"Start e={np.round(s0['e_signed_mm'],2)} θ={s0['e_theta_deg']:.2f} "
        f"α={oa0['alpha_open_up_deg']:.1f}° g·d_in={oa0['g_dot_d_inward']:.2f}",
        flush=True,
    )
    (OUT / "start_state.json").write_text(
        json.dumps(jsafe({"e_signed_mm": s0["e_signed_mm"], "opening": oa0, "gate": gate0}), indent=2),
        encoding="utf-8",
    )

    path_results = {}
    for axis in ("+wz", "-wz"):
        print(f"=== Path {axis}: Stage A pad-support → Stage B fingertips-up ===", flush=True)
        res = run_support_then_vertical(
            env, blob, support_axis=axis, support_angle_deg=90.0, support_rate=0.18, tip_rate=0.15
        )
        # strip bulky nested fulls for json
        slim = {
            k: v
            for k, v in res.items()
            if k not in ("actions", "stage_A_full", "stage_B_full", "stage_B_series", "capture")
        }
        slim["n_actions"] = len(res.get("actions") or [])
        slim["stage_B_series_len"] = len(res.get("stage_B_series") or [])
        path_results[axis] = res
        (OUT / f"path_{axis.replace('+','p').replace('-','m')}.json").write_text(
            json.dumps(jsafe(slim), indent=2), encoding="utf-8"
        )
        print(
            f"  {axis}: ok={res.get('ok')} reason={res.get('reason')} "
            f"α_final={((res.get('opening_final') or {}).get('alpha_open_up_deg'))} "
            f"g·d={(res.get('opening_final') or {}).get('g_dot_d_inward')}",
            flush=True,
        )
        if res.get("diagnosis"):
            print(f"  diagnosis: {res['diagnosis']}", flush=True)

    # Choose best path: prefer ok, then lowest alpha, then highest g·d_inward
    def score(kv):
        r = kv[1]
        oa = r.get("opening_final") or {}
        return (
            1 if r.get("ok") else 0,
            -float(oa.get("alpha_open_up_deg") or 999),
            float(oa.get("g_dot_d_inward") or -999),
        )

    best_axis, best = max(path_results.items(), key=score)
    (OUT / "snapshots" / "best_vertical.pkl").write_bytes(
        pickle.dumps({"capture": best.get("capture"), "ref": blob["ref"], "axis": best_axis})
    )

    slip = None
    slip5 = None
    handoff = None
    restore_meta = None
    gate_pre_slip = best.get("gate_final")

    vertical_ok = bool(
        best.get("ok")
        and gate_pre_slip
        and gate_pre_slip.get("fingertips_up_ok")
        and gate_pre_slip.get("gravity_assists_inward")
        and gate_pre_slip.get("bilateral")
        and gate_pre_slip.get("ground_clear")
    )

    if vertical_ok:
        print("=== Vertical gate PASSED — 2 mm inward slip ===", flush=True)
        (OUT / "vertical_gate.json").write_text(json.dumps(jsafe(gate_pre_slip), indent=2), encoding="utf-8")
        d_in_h = -U_OPEN_HAND  # inward in hand frame (= −hand_z)
        z_ref = float(gate_pre_slip.get("obj_z") or 0.5)
        e_x0 = float(gate_pre_slip["e_signed_mm"][0])
        tilt = {"capture": best["capture"], "ref": blob["ref"]}
        slip = run_axial_slip(env, tilt, d_in_h, milestone_mm=2.0, z_ref=z_ref, e_x0=e_x0)
        print(
            f"  slip2: ok={slip.get('ok')} disp={slip.get('disp_along_mm')} "
            f"abort={slip.get('abort_reason')} reason={slip.get('reason')}",
            flush=True,
        )
        (OUT / "axial_slip_2mm.json").write_text(
            json.dumps(jsafe({k: v for k, v in slip.items() if k not in ("actions", "series", "capture")}), indent=2),
            encoding="utf-8",
        )
        if slip.get("ok"):
            tilt5 = {"capture": slip["capture"], "ref": blob["ref"]}
            print("=== Optional 5 mm inward slip ===", flush=True)
            slip5 = run_axial_slip(
                env, tilt5, d_in_h, milestone_mm=5.0, z_ref=z_ref, e_x0=float(slip["e_signed"][0])
            )
            print(f"  slip5: ok={slip5.get('ok')} disp={slip5.get('disp_along_mm')}", flush=True)
            post_cap = slip5["capture"] if slip5.get("ok") else slip["capture"]
            post = {"capture": post_cap, "ref": blob["ref"]}
            rplan = plan_restore_orientation(env, post)
            rexe = execute_restore_plan(env, post, rplan)
            restore_meta = {
                "plan_ok": rplan.get("ok"),
                "already_upright": rplan.get("already_upright"),
                "axis_name": rplan.get("axis_name"),
                "angle_deg": rplan.get("angle_deg"),
                "exe_ok": rexe.get("ok"),
            }
            handoff = handoff_rich(env, take_capture(env), blob["ref"])
            (OUT / "handoff.json").write_text(
                json.dumps(
                    jsafe(
                        {
                            k: handoff.get(k)
                            for k in (
                                "SUCCESS_12S",
                                "SUCCESS_NEW",
                                "SUCCESS_AXIS_V1",
                                "pre_e_signed_mm",
                                "pre_theta",
                                "pre_pos_mm",
                                "reason",
                            )
                        }
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            acts = list(best.get("actions") or []) + list(slip.get("actions") or [])
            if slip5 and slip5.get("ok"):
                acts += list(slip5.get("actions") or [])
            acts += list(rexe.get("actions") or [])
            np.savez_compressed(OUT / "best_actions.npz", actions=np.asarray(acts, np.float32))
    else:
        print("=== Vertical gate FAILED — skip axial slip (not a slip failure) ===", flush=True)
        (OUT / "vertical_gate.json").write_text(
            json.dumps(
                jsafe(
                    {
                        "passed": False,
                        "best_axis": best_axis,
                        "gate": gate_pre_slip,
                        "diagnosis": best.get("diagnosis") or best.get("reason"),
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    # Replay IC if SUCCESS_NEW
    replay = None
    if handoff and handoff.get("SUCCESS_NEW") and (OUT / "best_actions.npz").exists():
        print("=== Original-IC 3× frozen-action replay ===", flush=True)
        prefix = []
        if STABLE_JSON.exists():
            stj = json.loads(STABLE_JSON.read_text(encoding="utf-8"))
            prefix = [np.asarray(a, np.float32) for a in stj.get("stabilization_prefix_actions") or []]
        lat = np.load(AXIS_PREFIX)["actions"] if AXIS_PREFIX.exists() else []
        mb = np.load(OUT / "best_actions.npz")["actions"]
        trials = []
        for _ in range(3):
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for a in prefix:
                step_legal(env.sim, a)
            for a in lat:
                step_legal(env.sim, a)
            for a in mb:
                step_legal(env.sim, a)
            hr = handoff_rich(env, take_capture(env), blob["ref"])
            trials.append(
                {
                    "SUCCESS_12S": hr.get("SUCCESS_12S"),
                    "SUCCESS_NEW": hr.get("SUCCESS_NEW"),
                    "SUCCESS_AXIS_V1": hr.get("SUCCESS_AXIS_V1"),
                    "pre_e_signed_mm": hr.get("pre_e_signed_mm"),
                }
            )
        replay = {
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
        }
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")

    plots = make_plots(OUT / "plots", {k: path_results[k] for k in path_results})

    print("Rendering videos...", flush=True)
    videos = {}
    for axis, res in path_results.items():
        acts = res.get("actions") or []
        if not acts:
            continue
        tag = axis.replace("+", "p").replace("-", "m")
        videos[axis] = render_path_video(
            env, blob, acts, OUT / "videos" / f"complete_rotation_{tag}.mp4", f"{axis} A→B complete"
        )
    # Combined best
    if best.get("actions"):
        videos["best"] = render_path_video(
            env,
            blob,
            best["actions"],
            OUT / "videos" / "complete_rotation_best.mp4",
            f"BEST {best_axis} support→fingertips-up",
        )

    # Verdict
    if vertical_ok and slip and slip.get("ok") and handoff and handoff.get("SUCCESS_NEW"):
        rec = "fingertips_up_slip_handoff_SUCCESS_NEW"
    elif vertical_ok and slip and slip.get("ok"):
        rec = "fingertips_up_2mm_slip_ok_handoff_limited"
    elif vertical_ok:
        rec = "fingertips_up_reached_slip_failed"
    elif best.get("opening_final") and float(best["opening_final"].get("alpha_open_up_deg") or 99) < 30:
        rec = "partial_tip_toward_vertical_contact_or_drift_limit"
    else:
        rec = "stage_B_transition_failed_before_slip"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "start": {"e_signed_mm": s0["e_signed_mm"], "opening": oa0},
        "paths": {
            ax: {
                "ok": path_results[ax].get("ok"),
                "reason": path_results[ax].get("reason"),
                "alpha_final": (path_results[ax].get("opening_final") or {}).get("alpha_open_up_deg"),
                "g_dot_d_inward": (path_results[ax].get("opening_final") or {}).get("g_dot_d_inward"),
                "stage_A_Fn_z": (path_results[ax].get("stage_A") or {}).get("selected_Fn_z"),
                "stage_A_align": (path_results[ax].get("stage_A") or {}).get("selected_align_up"),
                "delta_ex_B": (path_results[ax].get("stage_B") or {}).get("delta_ex_mm"),
                "diagnosis": path_results[ax].get("diagnosis"),
            }
            for ax in path_results
        },
        "best_axis": best_axis,
        "vertical_gate_passed": vertical_ok,
        "gate_pre_slip": gate_pre_slip,
        "slip_2mm": None
        if not slip
        else {k: slip.get(k) for k in ("ok", "disp_along_mm", "abort_reason", "reason", "lateral_drift_mm", "e_signed")},
        "slip_5mm": None
        if not slip5
        else {k: slip5.get(k) for k in ("ok", "disp_along_mm", "abort_reason", "reason")},
        "restore": restore_meta,
        "handoff": None
        if not handoff
        else {k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta")},
        "replay_ic": replay,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "sequence_note": (
            "Stage A ±wz@90° is intermediate pad support only. "
            "Axial slip is gated on Stage B fingertips-up (α≤5°) and g·d_inward>0."
        ),
        "preserved": {
            "SUCCESS_NEW": True,
            "SUCCESS_AXIS_V1": True,
            "plan_a": True,
            "plan_b_branch_a": True,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    report = f"""# Finger-Supported → Fingertips-Up → Axial Slip — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## Verdict

**{rec}**

Correct sequence enforced: Stage A pad-up (`±wz`) → Stage B tips to `α_open_up ≤ {ALPHA_GATE_DEG}°` → **then** slip.
Do **not** slip at the intermediate 90° pad-up pose.

| Path | Stage A→B ok | α_final (°) | g·d_inward | Stage A Fn_z | Δe_x Stage B |
|------|--------------|-------------|------------|--------------|--------------|
| +wz | {summary['paths']['+wz']['ok']} | {summary['paths']['+wz']['alpha_final']} | {summary['paths']['+wz']['g_dot_d_inward']} | {summary['paths']['+wz']['stage_A_Fn_z']} | {summary['paths']['+wz']['delta_ex_B']} |
| -wz | {summary['paths']['-wz']['ok']} | {summary['paths']['-wz']['alpha_final']} | {summary['paths']['-wz']['g_dot_d_inward']} | {summary['paths']['-wz']['stage_A_Fn_z']} | {summary['paths']['-wz']['delta_ex_B']} |

Best path: `{best_axis}`  Vertical gate: `{vertical_ok}`

## Start

e ≈ {np.round(s0['e_signed_mm'], 2).tolist()}  θ ≈ {s0['e_theta_deg']:.2f}°
α_open_up ≈ {oa0['alpha_open_up_deg']:.1f}°  g·d_inward ≈ {oa0['g_dot_d_inward']:.2f}
u_open = +hand_z

## Vertical gate (pre-slip)

{json.dumps(jsafe(gate_pre_slip), indent=2) if gate_pre_slip else 'n/a'}

## Axial slip / HANDOFF

Slip 2 mm: {json.dumps(jsafe(summary.get('slip_2mm')), indent=2)}

Slip 5 mm: {json.dumps(jsafe(summary.get('slip_5mm')), indent=2)}

Handoff: {json.dumps(jsafe(summary.get('handoff')), indent=2)}

Replay IC: {json.dumps(jsafe(replay), indent=2) if replay else 'not run'}

## Transition diagnosis (if Stage B failed)

{json.dumps(jsafe(best.get('diagnosis')), indent=2) if best.get('diagnosis') else 'n/a — or see path_*.json'}

## Videos

{json.dumps(jsafe(videos), indent=2)}

## Invariants

Preserved SUCCESS_NEW / AXIS_V1 / Plan A / Branch A. No physics/obs/action/nominal changes.
No bulk Oracle / SFT / GRPO.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "vertical_ok": vertical_ok}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return summary


if __name__ == "__main__":
    main()
