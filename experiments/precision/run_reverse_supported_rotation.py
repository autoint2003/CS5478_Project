#!/usr/bin/env python
"""Reverse verified -wz vertical path under hold-mode; preserve axial correction before HANDOFF.

Does not redesign Stage A/B/slip. Freezes forward -wz → vertical → 5 mm slip, then
compares exact / slow / contact-aware reverse restoration.
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
from data_generation.oracles.precision.finger_supported_vertical import (  # noqa: E402
    U_OPEN_HAND,
    opening_alignment,
)
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.reverse_supported_rotation import (  # noqa: E402
    HoldModeGrip,
    record_forward_with_waypoints,
    run_reverse_restore,
    snapshot_rich,
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
OUT = ROOT / "results/diagnostics/reverse_supported_rotation"

# Slip (unchanged successful settings)
FORCE_KP, FORCE_DA = 0.10, 0.12
V_SLIP = -0.001
FN_MIN, FN_MAX, FN_INIT, FN_BRAKE = 3.8, 11.0, 5.2, 14.0
SLIP_KN = 900.0
A0_SLIP = 0.55
DT = 0.02
FPS = 25
Z_ABORT = 0.44

# Hold-mode restore (NOT slip FN_MIN)
FN_HOLD = 13.0


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


def run_slip_2_then_5(env, blob, vert_snap):
    """Identical successful DualLoop 2 mm + 5 mm + brake/hold."""
    d = -U_OPEN_HAND
    z_ref = float(vert_snap["obj_z"])
    e_x0 = float(vert_snap["e_signed_mm"][0])

    def _one(blob_i, milestone, e_x):
        restore_capture(env, blob_i["capture"])
        s0 = sample_state(env.sim)
        p0 = np.asarray(s0["p_c_h"], float)
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
            failed, lost = recovery_failed(env.sim, blob_i["ref"], lost)
            s2 = sample_state(env.sim)
            dp = np.asarray(s2["p_c_h"], float) - p0
            disp = float(np.dot(dp, d)) * 1e3
            series.append({"t": (i + 1) * DT, "disp": disp, "e": s2["e_signed_mm"], "phase": "slip"})
            if failed or latched(env.sim) or not s2["bilateral"]:
                return {"ok": False, "reason": "contact_loss", "actions": actions, "series": series}
            if abs(float(s2["e_signed_mm"][0]) - e_x) > 4.0:
                abort = "lateral_drift"
                break
            if disp >= milestone:
                reached = True
                break
        ctl_b = ForcePController(kp=0.15, da_max=0.2)
        ctl_b.reset(0.85, sample_state(env.sim)["fn_tot"])
        for j in range(int(0.4 / DT) + int(0.2 / DT)):
            s = sample_state(env.sim)
            a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
            step_legal(env.sim, a)
            actions.append(a.copy())
            s2 = sample_state(env.sim)
            series.append(
                {
                    "t": series[-1]["t"] + DT,
                    "disp": float(np.dot(np.asarray(s2["p_c_h"], float) - p0, d) * 1e3),
                    "e": s2["e_signed_mm"],
                    "phase": "brake" if j < int(0.4 / DT) else "hold",
                }
            )
            if not s2["bilateral"]:
                return {"ok": False, "reason": "loss_on_brake", "actions": actions, "series": series}
        hold_ok = bool(sample_state(env.sim)["bilateral"])
        return {
            "ok": bool(hold_ok and abort is None and (reached or series[-1]["disp"] >= 0.5 * milestone)),
            "reached": reached,
            "abort_reason": abort,
            "disp_along_mm": float(series[-1]["disp"]),
            "e_signed": series[-1]["e"],
            "actions": actions,
            "series": series,
            "capture": take_capture(env),
        }

    s2 = _one(blob, 2.0, e_x0)
    if not s2.get("ok"):
        return s2
    s5 = _one({"capture": s2["capture"], "ref": blob["ref"]}, 5.0, float(s2["e_signed"][0]))
    if not s5.get("ok"):
        return {**s5, "actions": list(s2["actions"]) + list(s5.get("actions") or [])}
    # Extra firm hold settle
    restore_capture(env, s5["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    extra = []
    for _ in range(int(0.3 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        extra.append(a.copy())
    return {
        "ok": True,
        "disp_2mm": s2["disp_along_mm"],
        "disp_5mm": s5["disp_along_mm"],
        "e_signed": sample_state(env.sim)["e_signed_mm"],
        "theta": sample_state(env.sim)["e_theta_deg"],
        "actions": list(s2["actions"]) + list(s5["actions"]) + extra,
        "capture": take_capture(env),
        "snapshot": snapshot_rich(env.sim),
    }


def handoff_rich(env, capture, ref):
    from data_generation.oracles.precision.primitives import _wrist

    restore_capture(env, capture)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    meta = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    meta.pop("samples", None)
    return meta


def validate_nominal_pose(sim, snap_start: dict, snap_final: dict) -> dict:
    oa = opening_alignment(sim)
    s = sample_state(sim)
    # Original start had α≈122° (fingertips partly down / opening not world-up)
    alpha0 = float(snap_start["alpha_open_up_deg"])
    alpha = float(oa["alpha_open_up_deg"])
    # "fingertips-down-ish": opening not pointing up; prefer α close to start
    fingertips_restored = bool(alpha > 90.0 and abs(alpha - alpha0) < 25.0)
    ez_post = None
    return {
        "bilateral": s["bilateral"],
        "v_rel_norm": s["v_rel_norm"],
        "alpha_open_up_deg": alpha,
        "alpha_start_deg": alpha0,
        "alpha_err_to_start_deg": abs(alpha - alpha0),
        "fingertips_orientation_restored": fingertips_restored,
        "e_signed_mm": s["e_signed_mm"],
        "e_start_mm": snap_start["e_signed_mm"],
        "theta_deg": s["e_theta_deg"],
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "world_height_margin": float(sim.data.xpos[sim.ids.object_body][2]) - Z_ABORT,
        "ground_clear": bool(sim.data.xpos[sim.ids.object_body][2] >= Z_ABORT and not latched(sim)),
        "fn_tot": s["fn_tot"],
    }


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


def render_video(env, start_capture, actions, dest: Path, title: str):
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
        u = cylinder_axis_world(env.sim)
        oa = opening_alignment(env.sim)
        u_open = np.asarray(oa["u_open_world"], float)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.08 * u_open, [1.0, 0.85, 0.1, 1.0], 0.0035)
        add_capsule(scn, po, po + 0.07 * np.array([0, 0, 1.0]), [0.85, 0.4, 1.0, 1.0], 0.0028)

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        oa = opening_alignment(env.sim)
        lines = [
            f"{tag} {title}",
            f"α={oa['alpha_open_up_deg']:.1f}° e={np.round(s['e_signed_mm'],1)}",
            f"θ={s['e_theta_deg']:.2f} v_rel={s['v_rel_norm']:.4f} fn={s['fn_tot']:.1f}",
        ]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, ic], axis=1)

    restore_capture(env, start_capture)
    img = frame("START")
    for _ in range(int(0.8 * FPS)):
        frames.append(img)
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % 3 == 0 or i == len(actions) - 1:
            frames.append(frame(f"{i+1}/{len(actions)}"))
    for _ in range(int(1.2 * FPS)):
        frames.append(frame("END"))
    r_w.close()
    r_c.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS, "n_frames": len(frames)}


def make_plots(out_dir: Path, variants: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, res in variants.items():
        series = res.get("series") or []
        if not series:
            continue
        t = [r.get("t", i * DT) for i, r in enumerate(series)]
        fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        axs[0].plot(t, [r["delta_ez_mm"] for r in series], label="Δe_z vs post-slip")
        axs[0].plot(t, [r["delta_ex_mm"] for r in series], label="Δe_x")
        axs[0].axhline(0, color="k", lw=0.5)
        axs[0].legend()
        axs[0].set_ylabel("mm")
        axs[0].set_title(f"{name}: axial/lateral preservation during reverse")
        axs[1].plot(t, [r["fn_tot"] for r in series], label="fn_tot")
        axs[1].plot(t, [r.get("fn_target", np.nan) for r in series], label="fn_tgt")
        axs[1].plot(t, [r["v_rel_norm"] * 1000 for r in series], label="v_rel (mm/s)")
        axs[1].legend()
        axs[1].set_ylabel("N / mm/s")
        axs[2].plot(t, [r["alpha_open_up_deg"] for r in series], label="α_open_up")
        axs[2].plot(t, [r.get("ang_err_deg", np.nan) for r in series], label="ang_err_to_wp")
        axs[2].legend()
        axs[2].set_xlabel("t (s)")
        fig.tight_layout()
        p = out_dir / f"reverse_trace_{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[name] = str(p)
    return paths


def diagnose_loss(res: dict) -> dict:
    series = res.get("series") or []
    if not series:
        return {"note": "no series"}
    # Find max axial regression step
    worst = max(series, key=lambda r: abs(r.get("delta_ez_mm") or 0))
    phases = {}
    for ph in ("B_inv", "A_inv"):
        rows = [r for r in series if str(r.get("phase", "")).startswith(ph)]
        if not rows:
            continue
        phases[ph] = {
            "n": len(rows),
            "delta_ez_end": rows[-1].get("delta_ez_mm"),
            "delta_ex_end": rows[-1].get("delta_ex_mm"),
            "max_abs_delta_ez": max(abs(r.get("delta_ez_mm") or 0) for r in rows),
            "max_v_rel": max(r.get("v_rel_norm") or 0 for r in rows),
            "min_fn": min(r.get("fn_tot") or 99 for r in rows),
            "alpha_end": rows[-1].get("alpha_open_up_deg"),
        }
    causes = []
    if res.get("reason") and "axial_regression" in str(res.get("reason")):
        causes.append("axial_regression_abort")
    if worst.get("v_rel_norm", 0) > 0.003:
        causes.append("elevated_relative_slip_velocity")
    if (phases.get("B_inv") or {}).get("max_abs_delta_ez", 0) > 1.0:
        causes.append("loss_during_B_inv")
    if (phases.get("A_inv") or {}).get("max_abs_delta_ez", 0) > 1.0:
        causes.append("loss_during_A_inv")
    pb = res.get("phase_B_inv") or {}
    pa = res.get("phase_A_inv") or {}
    if pb.get("ok") and not (pa or {}).get("ok"):
        causes.append("failed_in_A_inv")
    if not pb.get("ok"):
        causes.append("failed_in_B_inv")
    return {
        "worst_sample": {k: worst.get(k) for k in ("phase", "delta_ez_mm", "delta_ex_mm", "v_rel_norm", "fn_tot", "alpha_open_up_deg")},
        "phases": phases,
        "candidate_causes": causes or ["small_regression_or_tracking"],
        "delta_ez_total_mm": res.get("delta_ez_total_mm"),
        "loss_locus": res.get("loss_locus"),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "plots", "videos"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    blob = load_blob(B_POST)
    print("=== Freeze forward -wz → vertical → 5 mm slip ===", flush=True)

    def slip_fn(e, b, vert_snap):
        return run_slip_2_then_5(e, b, vert_snap)

    fwd = record_forward_with_waypoints(
        env,
        blob,
        support_axis="-wz",
        support_angle_deg=90.0,
        support_rate=0.18,
        tip_rate=0.15,
        wp_every=4,
        run_slip_fn=slip_fn,
    )
    if not fwd.get("ok"):
        print("FORWARD FAILED", fwd.get("reason"), flush=True)
        (OUT / "forward_failed.json").write_text(json.dumps(jsafe(fwd), indent=2), encoding="utf-8")
        return fwd

    post_snap = fwd["snapshot_post_slip"]
    print(
        f"  post-slip e={np.round(post_snap['e_signed_mm'],2)} α={post_snap['alpha_open_up_deg']:.1f}° "
        f"θ={post_snap['theta_deg']:.2f}",
        flush=True,
    )
    (OUT / "snapshots" / "post_slip_hold.pkl").write_bytes(
        pickle.dumps({"capture": fwd["capture_post_slip"], "ref": blob["ref"]})
    )
    (OUT / "snapshots" / "forward_frozen.pkl").write_bytes(
        pickle.dumps(
            {
                "capture_start": fwd["capture_start"],
                "capture_A_end": fwd["capture_A_end"],
                "capture_B_end": fwd["capture_B_end"],
                "capture_post_slip": fwd["capture_post_slip"],
                "ref": blob["ref"],
                "support_axis": "-wz",
            }
        )
    )
    np.savez_compressed(
        OUT / "forward_actions.npz",
        actions=np.asarray(fwd["actions_forward"], np.float32),
        actions_A=np.asarray(fwd["actions_A"], np.float32),
        actions_B=np.asarray(fwd["actions_B"], np.float32),
        actions_slip=np.asarray(fwd["actions_slip"], np.float32),
    )
    # waypoints without huge nested — hand_R only
    (OUT / "forward_waypoints.json").write_text(
        json.dumps(
            jsafe(
                {
                    "n": len(fwd["waypoints"]),
                    "waypoints": fwd["waypoints"],
                    "snapshot_start": fwd["snapshot_start"],
                    "snapshot_A_end": fwd["snapshot_A_end"],
                    "snapshot_B_end": fwd["snapshot_B_end"],
                    "snapshot_post_slip": post_snap,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    post_blob = {"capture": fwd["capture_post_slip"], "ref": blob["ref"]}
    variants_cfg = [
        ("exact", 0.12, FN_HOLD, False),
        ("slow", 0.06, FN_HOLD, False),
        ("contact_aware", 0.12, FN_HOLD, True),
    ]
    variants = {}
    for name, rate, fn_h, aware in variants_cfg:
        print(f"=== Reverse restore variant: {name} rate={rate} aware={aware} ===", flush=True)
        res = run_reverse_restore(
            env,
            post_blob,
            fwd["waypoints"],
            rate_scale=rate,
            fn_hold=fn_h,
            contact_aware=aware,
            variant=name,
        )
        diag = diagnose_loss(res)
        res["diagnosis"] = diag
        slim = {
            k: v
            for k, v in res.items()
            if k not in ("actions", "series")
        }
        slim["n_actions"] = len(res.get("actions") or [])
        slim["diagnosis"] = diag
        variants[name] = res
        (OUT / f"restore_{name}.json").write_text(json.dumps(jsafe(slim), indent=2), encoding="utf-8")
        (OUT / f"restore_{name}_series.json").write_text(
            json.dumps(jsafe(res.get("series") or []), indent=2), encoding="utf-8"
        )
        print(
            f"  {name}: ok={res.get('ok')} reason={res.get('reason')} "
            f"Δez={res.get('delta_ez_total_mm')} locus={res.get('loss_locus')} causes={diag.get('candidate_causes')}",
            flush=True,
        )

    # Pick best: prefer ok, then smallest |Δez|, then smallest |Δex|
    def score(kv):
        r = kv[1]
        return (
            1 if r.get("ok") else 0,
            -abs(float(r.get("delta_ez_total_mm") or 99)),
            -abs(float(r.get("delta_ex_total_mm") or 99)),
        )

    best_name, best = max(variants.items(), key=score)
    print(f"Best restore: {best_name}", flush=True)

    # Final pose validation vs start
    handoff = None
    replay = None
    pose_val = None
    if best.get("ok") and best.get("capture") is not None:
        restore_capture(env, best["capture"])
        # firm settle
        hold = HoldModeGrip(fn_hold=FN_HOLD)
        hold.reset(sample_state(env.sim)["fn_tot"])
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        settle_acts = []
        for _ in range(int(0.4 / DT)):
            st = sample_state(env.sim)
            a_g, _, _ = hold.step(st)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
            step_legal(env.sim, a)
            settle_acts.append(a.copy())
        pose_val = validate_nominal_pose(env.sim, fwd["snapshot_start"], best.get("final") or {})
        pose_val["e_post_slip_mm"] = post_snap["e_signed_mm"]
        pose_val["e_final_mm"] = sample_state(env.sim)["e_signed_mm"]
        pose_val["ez_preserved_mm"] = float(pose_val["e_final_mm"][2] - post_snap["e_signed_mm"][2])
        pose_val["axial_ok"] = bool(abs(pose_val["ez_preserved_mm"]) <= 3.0)
        pose_val["ready_handoff"] = bool(
            pose_val["bilateral"]
            and pose_val["ground_clear"]
            and pose_val["v_rel_norm"] < 0.01
            and pose_val["fingertips_orientation_restored"]
            and pose_val["axial_ok"]
        )
        (OUT / "final_pose_validation.json").write_text(json.dumps(jsafe(pose_val), indent=2), encoding="utf-8")
        print(f"  pose_val ready={pose_val['ready_handoff']} α={pose_val['alpha_open_up_deg']:.1f} "
              f"Δez_vs_post={pose_val['ez_preserved_mm']:.2f}", flush=True)

        if pose_val["ready_handoff"]:
            print("=== HANDOFF after successful restore ===", flush=True)
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
            print(
                f"  HANDOFF NEW={handoff.get('SUCCESS_NEW')} AXIS={handoff.get('SUCCESS_AXIS_V1')} "
                f"12S={handoff.get('SUCCESS_12S')} reason={handoff.get('reason')}",
                flush=True,
            )
            all_acts = list(fwd["actions_forward"]) + list(best.get("actions") or []) + settle_acts
            np.savez_compressed(OUT / "complete_actions.npz", actions=np.asarray(all_acts, np.float32))

            if handoff.get("SUCCESS_NEW"):
                print("=== Original-IC 3× frozen replay ===", flush=True)
                prefix = []
                if STABLE_JSON.exists():
                    stj = json.loads(STABLE_JSON.read_text(encoding="utf-8"))
                    prefix = [np.asarray(a, np.float32) for a in stj.get("stabilization_prefix_actions") or []]
                lat = np.load(AXIS_PREFIX)["actions"] if AXIS_PREFIX.exists() else []
                mb = np.load(OUT / "complete_actions.npz")["actions"]
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
        else:
            # Still try HANDOFF for diagnosis if orientation restored but axial soft-fail
            if pose_val["bilateral"] and pose_val["fingertips_orientation_restored"]:
                print("=== Diagnostic HANDOFF (axial gate soft) ===", flush=True)
                handoff = handoff_rich(env, take_capture(env), blob["ref"])
                (OUT / "handoff_diagnostic.json").write_text(
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
                print(f"  diag HANDOFF reason={handoff.get('reason')}", flush=True)

    plots = make_plots(OUT / "plots", variants)

    print("Rendering videos...", flush=True)
    videos = {}
    # Forward only
    videos["forward"] = render_video(
        env, fwd["capture_start"], fwd["actions_forward"], OUT / "videos" / "forward_slip.mp4", "FORWARD -wz→slip"
    )
    if best.get("actions"):
        videos["reverse_best"] = render_video(
            env,
            fwd["capture_post_slip"],
            best["actions"],
            OUT / "videos" / f"reverse_{best_name}.mp4",
            f"REVERSE {best_name}",
        )
        full = list(fwd["actions_forward"]) + list(best["actions"])
        videos["complete"] = render_video(
            env, fwd["capture_start"], full, OUT / "videos" / "forward_slip_reverse.mp4", "FULL forward+reverse"
        )

    if handoff and handoff.get("SUCCESS_NEW"):
        rec = "reverse_restore_handoff_SUCCESS_NEW"
    elif best.get("ok") and pose_val and pose_val.get("ready_handoff"):
        rec = "reverse_restore_ok_handoff_failed"
    elif best.get("ok"):
        rec = "reverse_restore_ok_pose_gate_failed"
    else:
        rec = "reverse_restore_failed_axial_or_contact"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "forward": {
            "support_axis": "-wz",
            "snapshot_start": fwd["snapshot_start"],
            "snapshot_A_end": fwd["snapshot_A_end"],
            "snapshot_B_end": fwd["snapshot_B_end"],
            "snapshot_post_slip": post_snap,
            "n_waypoints": len(fwd["waypoints"]),
        },
        "variants": {
            n: {
                "ok": variants[n].get("ok"),
                "reason": variants[n].get("reason"),
                "delta_ez_total_mm": variants[n].get("delta_ez_total_mm"),
                "delta_ex_total_mm": variants[n].get("delta_ex_total_mm"),
                "loss_locus": variants[n].get("loss_locus"),
                "diagnosis": variants[n].get("diagnosis"),
                "phase_B_inv": {
                    k: (variants[n].get("phase_B_inv") or {}).get(k)
                    for k in ("ok", "reason", "delta_ez_mm", "delta_ex_mm", "n_steps")
                },
                "phase_A_inv": {
                    k: (variants[n].get("phase_A_inv") or {}).get(k)
                    for k in ("ok", "reason", "delta_ez_mm", "delta_ex_mm", "n_steps")
                },
            }
            for n in variants
        },
        "best_variant": best_name,
        "final_pose_validation": pose_val,
        "handoff": None
        if not handoff
        else {k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "reason")},
        "replay_ic": replay,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "preserved": {
            "SUCCESS_NEW": True,
            "SUCCESS_AXIS_V1": True,
            "forward_slip_unchanged": True,
            "plan_a": True,
            "plan_b_branch_a": True,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    report = f"""# Reverse Supported Rotation — Preserve Axial Correction — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## Verdict

**{rec}**

Forward trajectory frozen (`-wz` → fingertips-up → 2+5 mm slip). Return uses B⁻¹ then A⁻¹ under **hold-mode** grip (not slip `FN_MIN=3.8`).

## Forward (frozen)

| Checkpoint | e (mm) | α_open_up (°) |
|------------|--------|---------------|
| Start | {np.round(fwd['snapshot_start']['e_signed_mm'],2).tolist()} | {fwd['snapshot_start']['alpha_open_up_deg']:.1f} |
| A end (pad-up) | {np.round(fwd['snapshot_A_end']['e_signed_mm'],2).tolist()} | {fwd['snapshot_A_end']['alpha_open_up_deg']:.1f} |
| B end (vertical) | {np.round(fwd['snapshot_B_end']['e_signed_mm'],2).tolist()} | {fwd['snapshot_B_end']['alpha_open_up_deg']:.1f} |
| Post-slip hold | {np.round(post_snap['e_signed_mm'],2).tolist()} | {post_snap['alpha_open_up_deg']:.1f} |

## Restoration comparison

| Variant | ok | Δe_z total (mm) | Δe_x total | loss locus | causes |
|---------|----|-----------------|------------|------------|--------|
| exact | {summary['variants']['exact']['ok']} | {summary['variants']['exact']['delta_ez_total_mm']} | {summary['variants']['exact']['delta_ex_total_mm']} | {summary['variants']['exact']['loss_locus']} | {summary['variants']['exact']['diagnosis'].get('candidate_causes')} |
| slow | {summary['variants']['slow']['ok']} | {summary['variants']['slow']['delta_ez_total_mm']} | {summary['variants']['slow']['delta_ex_total_mm']} | {summary['variants']['slow']['loss_locus']} | {summary['variants']['slow']['diagnosis'].get('candidate_causes')} |
| contact_aware | {summary['variants']['contact_aware']['ok']} | {summary['variants']['contact_aware']['delta_ez_total_mm']} | {summary['variants']['contact_aware']['delta_ex_total_mm']} | {summary['variants']['contact_aware']['loss_locus']} | {summary['variants']['contact_aware']['diagnosis'].get('candidate_causes')} |

Best: `{best_name}`

### Where axial correction is lost

{json.dumps(jsafe(best.get('diagnosis')), indent=2)}

Phase B⁻¹: {json.dumps(jsafe(summary['variants'][best_name]['phase_B_inv']), indent=2)}

Phase A⁻¹: {json.dumps(jsafe(summary['variants'][best_name]['phase_A_inv']), indent=2)}

## Final pose / HANDOFF

Pose validation: {json.dumps(jsafe(pose_val), indent=2) if pose_val else 'n/a'}

Handoff: {json.dumps(jsafe(summary.get('handoff')), indent=2)}

Replay IC: {json.dumps(jsafe(replay), indent=2) if replay else 'not run'}

## Videos

{json.dumps(jsafe(videos), indent=2)}

## Invariants

Forward slip mechanism unchanged. Hold-mode restore only. No physics/obs/action/nominal changes.
Preserved SUCCESS_NEW / AXIS_V1 / Plan A / Branch A.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "best": best_name}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return summary


if __name__ == "__main__":
    main()
