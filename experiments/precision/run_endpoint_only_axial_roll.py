#!/usr/bin/env python
"""Endpoint-only axial roll maneuver — Plan B hard_910220.

Roll about u_open ONLY when α_up ≥ 175° (fingertips-down).
Forward/reverse tilt uses pure tilt (ω ⊥ u_open). Slip controller unchanged.
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
from data_generation.oracles.precision.endpoint_only_axial_roll import (  # noqa: E402
    ALPHA_DOWN_GATE_DEG,
    ALPHA_UP_GATE_DEG,
    DOWN,
    UP,
    choose_roll_sign_for_support,
    gates,
    phase_boundary_log,
    reference_pad_vector,
    roll_angle_about_open_deg,
    run_roll_phase,
    run_tilt_phase,
    u_open_world,
)
from data_generation.oracles.precision.finger_supported_vertical import U_OPEN_HAND  # noqa: E402
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip  # noqa: E402
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
OUT = ROOT / "results/diagnostics/endpoint_only_axial_roll"

# Unchanged successful DualLoop slip
FORCE_KP, FORCE_DA = 0.10, 0.12
V_SLIP = -0.001
FN_MIN, FN_MAX, FN_INIT, FN_BRAKE = 3.8, 11.0, 5.2, 14.0
SLIP_KN = 900.0
A0_SLIP = 0.55
FN_HOLD = 13.0
DT = 0.02
FPS = 25
Z_ABORT = 0.44
ROLL_ANGLE_DEG = 90.0


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


def run_slip_5mm(env, blob, z_ref: float, e_x0: float):
    d = -U_OPEN_HAND

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
        actions = []
        lost = 0
        reached = False
        abort = None
        last_disp = 0.0
        for _ in range(int(12.0 / DT)):
            s = sample_state(env.sim)
            _, meta = ctl.step(s)
            a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
            step_legal(env.sim, a)
            actions.append(a.copy())
            failed, lost = recovery_failed(env.sim, blob_i["ref"], lost)
            s2 = sample_state(env.sim)
            disp = float(np.dot(np.asarray(s2["p_c_h"], float) - p0, d)) * 1e3
            last_disp = disp
            if failed or latched(env.sim) or not s2["bilateral"]:
                return {"ok": False, "reason": "contact_loss", "actions": actions}
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
            if not sample_state(env.sim)["bilateral"]:
                return {"ok": False, "reason": "loss_on_brake", "actions": actions}
        return {
            "ok": bool(abort is None and (reached or last_disp >= 0.5 * milestone)),
            "disp": last_disp,
            "e_signed": sample_state(env.sim)["e_signed_mm"],
            "actions": actions,
            "capture": take_capture(env),
        }

    s2 = _one(blob, 2.0, e_x0)
    if not s2.get("ok"):
        return s2
    s5 = _one({"capture": s2["capture"], "ref": blob["ref"]}, 5.0, float(s2["e_signed"][0]))
    if not s5.get("ok"):
        return {**s5, "actions": list(s2["actions"]) + list(s5.get("actions") or [])}
    restore_capture(env, s5["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    extra = []
    for _ in range(int(0.25 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        extra.append(a.copy())
    return {
        "ok": True,
        "disp_2": s2["disp"],
        "disp_5": s5["disp"],
        "e_signed": sample_state(env.sim)["e_signed_mm"],
        "actions": list(s2["actions"]) + list(s5["actions"]) + extra,
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
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    y = 3
    for line in lines:
        dr.text((4, y), line, fill=(255, 255, 255), font=font)
        y += 15
    return np.asarray(im)


def render_video(env, start_cap, actions, phase_labels, dest: Path, title: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    r_w = mujoco.Renderer(env.sim.model, 480, 640)
    r_c = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0.35, 0.0, 0.4], "distance": 2.2, "azimuth": 145, "elevation": -18})
    cam_c = mjv_camera_from_preset({"lookat": [0.5, 0.05, 0.52], "distance": 0.55, "azimuth": 120, "elevation": -25})
    frames = []
    # phase_labels: list of (end_index_exclusive, name)
    phase_labels = sorted(phase_labels, key=lambda x: x[0])

    def cur_phase(i):
        for end, name in phase_labels:
            if i < end:
                return name
        return phase_labels[-1][1] if phase_labels else "?"

    def draw(scn):
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        uo = u_open_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ np.asarray(P_TGT, float), [1.0, 0.9, 0.15, 1.0], 0.005)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.09 * uo, [1.0, 0.85, 0.1, 1.0], 0.0035)
        add_capsule(scn, po, po + 0.07 * UP, [0.85, 0.4, 1.0, 1.0], 0.0025)
        add_capsule(scn, po, po + 0.07 * DOWN, [0.4, 0.9, 0.9, 1.0], 0.0025)

    def frame(tag, pname):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.35 * ph + 0.35 * po + np.array([0, 0, 0.05])
        cam_c.lookat[:] = 0.5 * ph + 0.5 * po
        s = sample_state(env.sim)
        g = gates(env.sim)
        lines = [
            f"{tag} [{pname}] {title}",
            f"α_up={g['alpha_up_deg']:.1f}° down={g['fingertips_down']} up={g['fingertips_up']}",
            f"e={np.round(s['e_signed_mm'],1)} θ={s['e_theta_deg']:.2f}",
            "yellow=u_open purple=+Z cyan=-Z",
        ]
        r_w.update_scene(env.sim.data, cam_w)
        draw(r_w.scene)
        iw = overlay(np.asarray(r_w.render(), np.uint8), ["WIDE"] + lines)
        r_c.update_scene(env.sim.data, cam_c)
        draw(r_c.scene)
        ic = overlay(np.asarray(r_c.render(), np.uint8), ["CLOSE"] + lines)
        return np.concatenate([iw, ic], axis=1)

    restore_capture(env, start_cap)
    frames.append(frame("0", "START"))
    for _ in range(int(0.6 * FPS)):
        frames.append(frames[-1])
    for i, a in enumerate(actions):
        step_legal(env.sim, a)
        if i % 3 == 0 or i == len(actions) - 1:
            frames.append(frame(f"{i+1}/{len(actions)}", cur_phase(i + 1)))
    for _ in range(int(1.0 * FPS)):
        frames.append(frame("END", cur_phase(len(actions))))
    r_w.close()
    r_c.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_frames": len(frames), "duration_s": len(frames) / FPS}


def make_plots(out_dir: Path, series_all: list, boundaries: list) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if not series_all:
        return {}
    # stitch time
    t_off = 0.0
    T, A, ER, ET, EZ, EX, FN, VR, ROLL = [], [], [], [], [], [], [], [], []
    for block in series_all:
        for r in block:
            T.append(t_off + float(r.get("t", 0)))
            A.append(r.get("alpha_up_deg"))
            ER.append(r.get("omega_roll_norm") or abs(r.get("omega_roll_signed") or 0))
            ET.append(r.get("omega_tilt_norm") or 0)
            EZ.append(r.get("delta_ez_mm") if r.get("delta_ez_mm") is not None else np.nan)
            EX.append(r.get("delta_ex_mm") if r.get("delta_ex_mm") is not None else np.nan)
            FN.append(r.get("fn_tot"))
            VR.append((r.get("v_rel_norm") or 0) * 1000)
            ROLL.append(r.get("roll_angle_deg"))
        if block:
            t_off = T[-1] + 0.05

    paths = {}
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(T, A, label="α_up")
    axs[0].axhline(ALPHA_UP_GATE_DEG, color="g", ls="--", label="up gate")
    axs[0].axhline(ALPHA_DOWN_GATE_DEG, color="c", ls="--", label="down gate")
    axs[0].axhline(122, color="orange", ls=":", label="old 122° (NOT down)")
    axs[0].legend(fontsize=8)
    axs[0].set_ylabel("deg")
    axs[0].set_title("Opening axis vs world-up (122° ≠ fingertips-down)")
    axs[1].plot(T, ER, label="|ω_roll|")
    axs[1].plot(T, ET, label="|ω_tilt|")
    axs[1].legend()
    axs[1].set_ylabel("rad/s")
    axs[2].plot(T, EZ, label="Δe_z")
    axs[2].plot(T, EX, label="Δe_x")
    axs[2].legend()
    axs[2].set_ylabel("mm")
    axs[3].plot(T, FN, label="fn_tot")
    axs[3].plot(T, VR, label="v_rel mm/s")
    axs[3].legend()
    axs[3].set_xlabel("t (s)")
    fig.tight_layout()
    p = out_dir / "phase_traces.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["traces"] = str(p)

    # boundary table figure as text file already; save alpha bar
    fig, ax = plt.subplots(figsize=(8, 3))
    names = [b["phase"] for b in boundaries]
    alphas = [b.get("alpha_up_deg") for b in boundaries]
    ax.bar(range(len(names)), alphas)
    ax.axhline(ALPHA_DOWN_GATE_DEG, color="c", ls="--")
    ax.axhline(ALPHA_UP_GATE_DEG, color="g", ls="--")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("α_up deg")
    ax.set_title("Phase-boundary α_up")
    fig.tight_layout()
    p2 = out_dir / "boundary_alpha.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths["boundary"] = str(p2)
    return paths


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("snapshots", "plots", "videos"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    blob = load_blob(B_POST)
    restore_capture(env, blob["capture"])
    start_log = phase_boundary_log(env.sim, phase="IC_start")
    y0 = reference_pad_vector(env.sim)
    u0 = u_open_world(env.sim)
    e_start = np.asarray(start_log["e_signed_mm"], float)
    print(
        f"IC α_up={start_log['alpha_up_deg']:.1f}° (NOT down; gate={ALPHA_DOWN_GATE_DEG}°) "
        f"e={np.round(e_start,2)}",
        flush=True,
    )
    assert start_log["alpha_up_deg"] < ALPHA_DOWN_GATE_DEG, "IC unexpectedly already down"

    boundaries = [start_log]
    series_all = []
    all_actions = []
    phase_ends = []  # (end_idx, name)
    assertions = []
    e_ref = e_start.copy()
    cur = {"capture": blob["capture"], "ref": blob["ref"]}
    roll_sign = choose_roll_sign_for_support(env.sim)
    nominal_u_open = u0.copy()  # for Phase 6 target
    failed_phase = None
    phases = {}

    def commit(res, name):
        nonlocal cur, failed_phase
        phases[name] = {k: v for k, v in res.items() if k not in ("actions", "series", "capture")}
        acts = list(res.get("actions") or [])
        all_actions.extend(acts)
        phase_ends.append((len(all_actions), name))
        if res.get("series"):
            series_all.append(res["series"])
        if res.get("boundary"):
            boundaries.append(res["boundary"])
        if res.get("capture") is not None:
            cur = {"capture": res["capture"], "ref": blob["ref"]}
        ok = bool(res.get("ok"))
        assertions.append({"phase": name, "ok": ok, "reason": res.get("reason")})
        print(
            f"  [{name}] ok={ok} reason={res.get('reason')} α={res.get('alpha_final') or (res.get('boundary') or {}).get('alpha_up_deg')}",
            flush=True,
        )
        if not ok:
            failed_phase = name
        return ok

    # ----- Phase 0: fingertips-down preparation (tilt only) -----
    print("=== Phase 0: tilt to fingertips-down (α≥175°) — NO roll ===", flush=True)
    # contact_aware off for prep: relative motion during large tip is expected; zeroing ω stalled at ~145°.
    p0 = run_tilt_phase(
        env,
        cur,
        target_dir_w=DOWN,
        phase="P0_prep_down",
        rate_scale=0.20,
        fn_hold=FN_HOLD,
        ang_tol_deg=4.0,
        y0_w=y0,
        e_ref=e_ref,
        allow_roll=False,
        contact_aware=False,
        max_time_s=8.0,
    )
    commit(p0, "P0_prep_down")
    if not gates(env.sim)["fingertips_down"]:
        print("  P0 gate miss — one targeted retry rate=0.25", flush=True)
        p0b = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="P0_retry", rate_scale=0.25, fn_hold=FN_HOLD,
            ang_tol_deg=3.5, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=False, max_time_s=8.0,
        )
        if not commit(p0b, "P0_retry") or not gates(env.sim)["fingertips_down"]:
            failed_phase = "P0_prep_down"
            assertions.append({"phase": "P0_down_gate", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"]})
            _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
            return
    elif failed_phase == "P0_prep_down":
        # commit marked fail but gate actually passed after settle — clear
        failed_phase = None

    assertions.append({"phase": "no_roll_before_down", "ok": True, "note": "P1 not started before down gate"})

    # ----- Phase 1: initial axial roll -----
    print(f"=== Phase 1: axial roll sign={roll_sign} @ {ROLL_ANGLE_DEG}° (down only) ===", flush=True)
    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="P1_roll",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    if not commit(p1, "P1_roll"):
        _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
        return
    roll_after_p1 = float(p1.get("roll_final_deg") or 0.0)

    # ----- Phase 2: tilt to fingertips-up (no roll) -----
    print("=== Phase 2: tilt to fingertips-up (α≤5°) — NO roll ===", flush=True)
    # Validated: rate≈0.22 + mild contact_aware reaches up; faster open-loop loses contact near α~90°.
    p2 = run_tilt_phase(
        env, cur, target_dir_w=UP, phase="P2_tilt_up", rate_scale=0.22, fn_hold=FN_HOLD,
        ang_tol_deg=ALPHA_UP_GATE_DEG, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True,
        max_time_s=16.0,
    )
    if not commit(p2, "P2_tilt_up") or not gates(env.sim)["fingertips_up"]:
        print("  P2 retry rate=0.20", flush=True)
        p2b = run_tilt_phase(
            env, cur, target_dir_w=UP, phase="P2_retry", rate_scale=0.20, fn_hold=FN_HOLD,
            ang_tol_deg=4.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=18.0,
        )
        if not commit(p2b, "P2_retry") or not gates(env.sim)["fingertips_up"]:
            failed_phase = "P2_tilt_up"
            _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
            return
        p2 = p2b
        failed_phase = None
    assertions.append({"phase": "P2_no_roll", "ok": p2.get("max_omega_roll_cmd", 0) <= 0.2, "max_roll": p2.get("max_omega_roll_cmd")})

    # ----- Phase 3: slip -----
    print("=== Phase 3: DualLoop 2+5 mm inward slip (unchanged) ===", flush=True)
    restore_capture(env, cur["capture"])
    # Targeted fix: firm settle after tilt-up before unloading for slip (avoids contact_loss).
    hold = HoldModeGrip(fn_hold=14.0)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    settle_pre = []
    for _ in range(int(0.4 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        settle_pre.append(a.copy())
    all_actions.extend(settle_pre)
    phase_ends.append((len(all_actions), "P3_presettle"))
    cur = {"capture": take_capture(env), "ref": blob["ref"]}
    boundaries.append(phase_boundary_log(env.sim, phase="P3_presettle"))
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    e_x0 = float(sample_state(env.sim)["e_signed_mm"][0])
    e_before_slip = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    slip = run_slip_5mm(env, cur, z_ref, e_x0)
    slip_bound = phase_boundary_log(env.sim, phase="P3_slip_end", e_ref=e_before_slip)
    boundaries.append(slip_bound)
    phases["P3_slip"] = {k: v for k, v in slip.items() if k not in ("actions", "capture")}
    all_actions.extend(list(slip.get("actions") or []))
    phase_ends.append((len(all_actions), "P3_slip"))
    assertions.append({"phase": "P3_slip", "ok": bool(slip.get("ok")), "reason": slip.get("reason")})
    print(f"  [P3_slip] ok={slip.get('ok')} e={np.round(slip.get('e_signed'),2) if slip.get('e_signed') else None}", flush=True)
    if not slip.get("ok"):
        failed_phase = "P3_slip"
        _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
        return
    cur = {"capture": slip["capture"], "ref": blob["ref"]}
    e_post_slip = np.asarray(slip["e_signed"], float)
    e_ref = e_post_slip.copy()  # preserve axial from here

    # ----- Phase 4: reverse tilt to down (no roll) -----
    print("=== Phase 4: reverse tilt to fingertips-down — NO roll (keep Phase-1 roll) ===", flush=True)
    # Targeted: boost height hold — reverse tip after slip dips below Z_ABORT without lift bias.
    p4 = run_tilt_phase(
        env, cur, target_dir_w=DOWN, phase="P4_tilt_down", rate_scale=0.18, fn_hold=FN_HOLD,
        ang_tol_deg=4.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=16.0,
        z_ref_boost=0.04, height_kp=4.0,
    )
    if (not commit(p4, "P4_tilt_down")) or (not gates(env.sim)["fingertips_down"]):
        print("  P4 retry + lift", flush=True)
        p4b = run_tilt_phase(
            env, cur, target_dir_w=DOWN, phase="P4_retry", rate_scale=0.16, fn_hold=14.0,
            ang_tol_deg=3.5, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True, max_time_s=18.0,
            z_ref_boost=0.06, height_kp=5.0,
        )
        if not commit(p4b, "P4_retry") or not gates(env.sim)["fingertips_down"]:
            failed_phase = "P4_tilt_down"
            assertions.append({"phase": "P4_down_gate", "ok": False, "alpha": gates(env.sim)["alpha_up_deg"]})
            _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
            return
        p4 = p4b
        failed_phase = None
    assertions.append({"phase": "P4_no_roll", "ok": p4.get("max_omega_roll_cmd", 0) <= 0.2, "max_roll": p4.get("max_omega_roll_cmd")})
    a4 = gates(env.sim)["alpha_up_deg"]
    assertions.append({"phase": "P4_not_122", "ok": a4 >= ALPHA_DOWN_GATE_DEG, "alpha": a4})

    # ----- Phase 5: inverse axial roll -----
    print("=== Phase 5: inverse axial roll (down only) ===", flush=True)
    p5 = run_roll_phase(
        env, cur, sign=-roll_sign, angle_deg=ROLL_ANGLE_DEG, phase="P5_unroll",
        rate_scale=0.15, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    if not commit(p5, "P5_unroll"):
        _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
        return

    # ----- Phase 6: residual tilt to nominal-compatible u_open (start IC) -----
    print("=== Phase 6: residual tilt to nominal-compatible (start u_open) — NO roll ===", flush=True)
    # Only if current u_open differs from nominal start
    ang_to_nom = float(np.degrees(np.arccos(np.clip(np.dot(u_open_world(env.sim), nominal_u_open), -1, 1))))
    p6 = None
    if ang_to_nom > 5.0:
        p6 = run_tilt_phase(
            env, cur, target_dir_w=nominal_u_open, phase="P6_nominal_tilt", rate_scale=0.10,
            fn_hold=FN_HOLD, ang_tol_deg=5.0, y0_w=y0, e_ref=e_ref, allow_roll=False, contact_aware=True,
        )
        if not commit(p6, "P6_nominal_tilt"):
            _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, None, None, None, y0)
            return
    else:
        boundaries.append(phase_boundary_log(env.sim, phase="P6_skip_already_nominal", e_ref=e_ref))
        assertions.append({"phase": "P6_skip", "ok": True, "ang_to_nom": ang_to_nom})
        print(f"  [P6] skip ang_to_nom={ang_to_nom:.1f}°", flush=True)

    # Settle
    restore_capture(env, cur["capture"])
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(sample_state(env.sim)["fn_tot"])
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
    settle = []
    for _ in range(int(0.4 / DT)):
        st = sample_state(env.sim)
        a_g, _, _ = hold.step(st)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        settle.append(a.copy())
    all_actions.extend(settle)
    phase_ends.append((len(all_actions), "settle"))
    final_bound = phase_boundary_log(env.sim, phase="P6_final_settle", e_ref=e_post_slip)
    boundaries.append(final_bound)
    (OUT / "snapshots" / "pre_handoff.pkl").write_bytes(pickle.dumps({"capture": take_capture(env), "ref": blob["ref"]}))

    ez_pres = float(final_bound["e_signed_mm"][2] - e_post_slip[2])
    ready = bool(
        final_bound["bilateral"]
        and final_bound["world_height_margin"] > 0.01
        and final_bound["v_rel_norm"] < 0.01
        and abs(ez_pres) <= 3.0
    )
    assertions.append({"phase": "ready_handoff", "ok": ready, "ez_preserved_mm": ez_pres, "alpha": final_bound["alpha_up_deg"]})
    print(f"  final α={final_bound['alpha_up_deg']:.1f} e={np.round(final_bound['e_signed_mm'],2)} Δez_vs_slip={ez_pres:.2f} ready={ready}", flush=True)

    handoff = None
    replay = None
    if ready:
        print("=== Phase 7: HANDOFF ===", flush=True)
        assertions.append({"phase": "handoff_before_full_restore", "ok": True})
        handoff = handoff_rich(env, take_capture(env), blob["ref"])
        (OUT / "handoff.json").write_text(
            json.dumps(
                jsafe({k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "pre_pos_mm", "reason")}),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  HANDOFF NEW={handoff.get('SUCCESS_NEW')} reason={handoff.get('reason')}", flush=True)
        np.savez_compressed(OUT / "complete_actions.npz", actions=np.asarray(all_actions, np.float32))
        if handoff.get("SUCCESS_NEW"):
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
                trials.append({k: hr.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm")})
            replay = {
                "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
                "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
                "trials": trials,
            }
            (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")
    else:
        # diagnostic handoff
        print("=== Diagnostic HANDOFF (ready=False) ===", flush=True)
        handoff = handoff_rich(env, take_capture(env), blob["ref"])
        (OUT / "handoff_diagnostic.json").write_text(
            json.dumps(jsafe({k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "reason")}), indent=2),
            encoding="utf-8",
        )

    _finish(
        env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases,
        failed_phase, t0, handoff, replay, e_post_slip, y0, ready=ready, ez_pres=ez_pres,
    )


def _finish(env, blob, boundaries, series_all, all_actions, phase_ends, assertions, phases, failed_phase, t0, handoff, replay, e_post_slip, y0, ready=False, ez_pres=None):
    plots = make_plots(OUT / "plots", series_all, boundaries)
    print("Rendering video...", flush=True)
    videos = {}
    if all_actions:
        videos["complete"] = render_video(
            env, blob["capture"], all_actions, phase_ends,
            OUT / "videos" / "endpoint_only_axial_roll.mp4",
            "endpoint-only axial roll",
        )

    if handoff and handoff.get("SUCCESS_NEW"):
        rec = "endpoint_roll_SUCCESS_NEW"
    elif ready and handoff and handoff.get("reason") == "DROP_AFTER_HANDOFF":
        rec = "endpoint_roll_restore_ok_handoff_drop"
    elif failed_phase:
        rec = f"failed_at_{failed_phase}"
    elif ready:
        rec = "endpoint_roll_ready_handoff_limited"
    else:
        rec = "endpoint_roll_incomplete"

    # Phase boundary table markdown
    rows = []
    for b in boundaries:
        rows.append(
            f"| {b.get('phase')} | {b.get('alpha_up_deg')} | {b.get('fingertips_down')} | {b.get('fingertips_up')} | "
            f"{np.round(b.get('e_signed_mm'),2).tolist() if b.get('e_signed_mm') else None} | "
            f"{b.get('fn_tot')} | {b.get('v_rel_norm')} | {b.get('world_height_margin')} |"
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "gates": {"up_deg": ALPHA_UP_GATE_DEG, "down_deg": ALPHA_DOWN_GATE_DEG},
        "ic_alpha_up_deg": boundaries[0].get("alpha_up_deg") if boundaries else None,
        "ic_note": "122° IC is NOT fingertips-down; Phase 0 required",
        "boundaries": boundaries,
        "assertions": assertions,
        "phases": phases,
        "failed_phase": failed_phase,
        "ready_handoff": ready,
        "ez_preserved_mm": ez_pres,
        "handoff": None if not handoff else {k: handoff.get(k) for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "reason")},
        "replay_ic": replay,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "n_actions": len(all_actions),
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "assertions.json").write_text(json.dumps(jsafe(assertions), indent=2), encoding="utf-8")
    (OUT / "phase_boundaries.json").write_text(json.dumps(jsafe(boundaries), indent=2), encoding="utf-8")
    if all_actions:
        np.savez_compressed(OUT / "complete_actions.npz", actions=np.asarray(all_actions, np.float32))

    report = f"""# Endpoint-Only Axial Roll — Plan B hard_910220

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## Verdict

**{rec}**

IC α_up ≈ {summary['ic_alpha_up_deg']:.1f}° — **not** vertically downward. Phase 0 tilts to α≥{ALPHA_DOWN_GATE_DEG}° before any axial roll.
122° is explicitly rejected as a fingertips-down state.

## Phase-boundary table

| Phase | α_up | down? | up? | e (mm) | fn_tot | v_rel | z_margin |
|-------|------|-------|-----|--------|--------|-------|----------|
{chr(10).join(rows)}

## Assertions

{json.dumps(jsafe(assertions), indent=2)}

## Axial preservation / HANDOFF

Δe_z vs post-slip: {ez_pres}
Ready: {ready}
Handoff: {json.dumps(jsafe(summary.get('handoff')), indent=2)}
Replay: {json.dumps(jsafe(replay), indent=2) if replay else 'not run'}

Failed phase: {failed_phase}

## Sequence rules enforced

1. No axial roll before fingertips-down (α≥{ALPHA_DOWN_GATE_DEG}°).
2. Forward tilt: ω_roll projected to 0.
3. Slip only at fingertips-up.
4. Reverse tilt: no roll (Phase-1 roll held).
5. Inverse roll only after reverse down gate.
6. Nominal residual tilt without roll, then HANDOFF.

## Videos / plots

{json.dumps(jsafe(videos), indent=2)}
{json.dumps(jsafe(plots), indent=2)}

## Invariants

Slip DualLoop unchanged. No physics/obs/action/nominal/SUCCESS_* changes.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "failed_phase": failed_phase, "ready": ready}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
