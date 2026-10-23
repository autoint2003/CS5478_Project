#!/usr/bin/env python
"""Focused: gravity-assisted hand-x COM correction after verified A_raw pivot.

Plan A ball_train_v2_0070 only. No pivot re-optimization, no Plan B/C, no training.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import W_MAX, grip_action  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import (  # noqa: E402
    _action_vw,
    _legal,
    _run,
    _wrist,
    audit_grasp,
    pivot_segment_metrics,
    primitive_stabilize_handoff,
)
from data_generation.oracles.precision.mechanics.grasp_stabilization import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.plan_c.contact_surface import classify_contacts  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_hand, hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR = ROOT / "results/diagnostics/position_preserving_pivot"
STATE_A = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
OUT = ROOT / "results/diagnostics/hand_x_gravity_reorientation"

DT = 0.02
FPS = 25
# Explicit modest budgets
TILT_TARGETS_DEG = (30.0, 45.0, 60.0, 75.0)
MAX_COMBINED = 48  # tilt×sign×unload×slip screens
MAX_HANDOFF = 3
MIN_USEFUL_DX_MM = 2.0  # meaningful hand-x correction before HANDOFF
MAX_WALL_S = 600.0


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
    env._obs_seq = []
    env._prev_seq = []
    return env


def parse_ppivot_tag(tag: str) -> dict:
    rest = tag[len("ppiv_v") :]
    pre_w, after_w = rest.split("_w", 1)
    vx_s, vz_s = pre_w.split("_", 1)
    pre_g, after_g = after_w.split("_g", 1)
    wx_s, wy_s = pre_g.split("_", 1)
    grip_s, dur_s = after_g.split("_d", 1)
    return {
        "vx": float(vx_s),
        "vz": float(vz_s),
        "wx": float(wx_s),
        "wy": float(wy_s),
        "grip": float(grip_s),
        "dur": float(dur_s),
    }


def pivot_actions_from_tag(tag: str) -> list:
    p = parse_ppivot_tag(tag)
    a_p = _action_vw([p["vx"], 0.0, p["vz"]], [p["wx"], p["wy"], 0.0], p["grip"])
    a_b = _wrist(0.0, 0.0, 1.0)
    n = max(1, int(round(p["dur"] / DT)))
    return [a_p] * n + [a_b] * 8


def sample_geom(sim) -> dict:
    Rh = hand_R(sim)
    pack = physical_pack(sim)
    m = measure(sim, U_TGT, P_TGT)
    surf = classify_contacts(sim)
    u = cylinder_axis_hand(sim)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    e = np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)
    g_h = np.asarray(pack["g_h"], float)
    # Lateral to cylinder in hand frame (unit vector in span of hand axes ⊥ u)
    # hand-x basis
    ex_h = np.array([1.0, 0.0, 0.0])
    # component of hand-x along cylinder
    hx_along_u = float(np.dot(ex_h, u))
    hx_lat = ex_h - hx_along_u * u
    hx_lat_n = float(np.linalg.norm(hx_lat))
    contacts = []
    for c in surf.get("contacts") or []:
        contacts.append(
            {
                "side": c.get("side"),
                "contact_s": c.get("contact_s"),
                "fn": c.get("fn"),
                "normal_hand": c.get("n_hand") or c.get("normal_hand"),
            }
        )
    # Tilt proxy: angle between hand +z and world -gravity (up)
    hz_w = Rh[:, 2]
    up = -np.array(sim.model.opt.gravity, float)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    # When hand z aligns with up, grasp is "horizontal cylinder" style depending on layout;
    # report angle between hand-x and world horizontal plane + |g_x|
    g_w = np.asarray(sim.model.opt.gravity, float)
    tilt_from_g = float(np.degrees(np.arctan2(abs(g_h[0]), max(abs(g_h[2]), 1e-9))))
    return {
        "Rh_world": Rh.tolist(),
        "hand_x_world": Rh[:, 0].tolist(),
        "hand_y_world": Rh[:, 1].tolist(),
        "hand_z_world": Rh[:, 2].tolist(),
        "u_hand": u.tolist(),
        "u_along_hand_x": hx_along_u,
        "hand_x_lateral_to_cyl_norm": hx_lat_n,
        "p_c_h": m["p_c_h"],
        "P_TGT": np.asarray(P_TGT, float).tolist(),
        "e_signed_mm": (e * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "g_h": g_h.tolist(),
        "g_x": float(g_h[0]),
        "g_y": float(g_h[1]),
        "g_z": float(g_h[2]),
        "tilt_proxy_deg_atan_gx_gz": tilt_from_g,
        "v_rel_h": np.asarray(pack["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(pack["w_rel_h"], float).tolist(),
        "v_rel_norm": float(np.linalg.norm(pack["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(pack["w_rel_h"])),
        "bilateral": bool(surf.get("bilateral")),
        "grasp_kind": surf.get("grasp_kind"),
        "contact_s_mean": surf.get("contact_s_mean"),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "contacts": contacts[:8],
        "opposing_sidewall": int(surf.get("opposing_sidewall") or 0),
    }


def desired_hand_x_sign(e_signed_mm) -> dict:
    """e_x > 0 ⇒ need object COM to move −hand-x relative to gripper."""
    ex = float(e_signed_mm[0])
    desire_dp_x = -1.0 if ex > 0 else (+1.0 if ex < 0 else 0.0)
    # Free-body slip under gravity moves object along g_h; want g_x same sign as desire_dp_x
    desire_g_x_sign = desire_dp_x
    return {
        "e_x_mm": ex,
        "desire_delta_p_rel_x_sign": desire_dp_x,
        "desire_g_x_sign": desire_g_x_sign,
        "note": "Reduce |e_x| by relative object motion opposite the signed error; g_x should match that slip direction when unloaded.",
    }


def tilt_schedule(target_deg: float, sign: float) -> list:
    """Open-loop wy at max legal rate for duration ≈ target_deg / ω."""
    omega = float(W_MAX)  # rad/s at action saturation
    dur = float(np.radians(target_deg)) / max(omega, 1e-6)
    dur = float(np.clip(dur, 0.10, 1.40))
    a = _wrist(0.0, sign * omega, 1.0)  # physical wy=±W_MAX after normalize clip
    # _wrist takes body rates before normalize — pass ±W_MAX
    a = _wrist(0.0, sign * float(W_MAX), 1.0)
    n = max(1, int(round(dur / DT)))
    return [a] * n, dur, n


def untilt_schedule(tilt_n: int, sign: float) -> list:
    a = _wrist(0.0, -sign * float(W_MAX), 1.0)
    return [a] * tilt_n + [_wrist(0.0, 0.0, 1.0)] * 5


def run_actions(env, blob, actions):
    restore_capture(env, blob["capture"])
    ok, reason = _run(env, blob, actions)
    return ok, reason, sample_geom(env.sim)


def classify_motion(p0, p1, theta0, theta1, bilateral) -> str:
    dp = np.asarray(p1, float) - np.asarray(p0, float)
    dpx = float(dp[0]) * 1e3
    dpn = float(np.linalg.norm(dp)) * 1e3
    dth = abs(float(theta1) - float(theta0))
    if not bilateral:
        return "uncontrolled_slip_or_loss_of_contact"
    if abs(dpx) >= MIN_USEFUL_DX_MM and dpn > 0 and abs(dpx) / max(dpn, 1e-6) > 0.5 and dth <= 8.0:
        return "useful_controlled_hand_x_sliding"
    if dpn >= 3.0 and abs(dpx) < 1.0:
        return "motion_not_along_hand_x"
    if dpn < 1.0 and dth < 2.0:
        return "rigid_hand_object_or_static"
    if abs(dpx) >= 1.0 and dth > 10.0:
        return "uncontrolled_slip"
    if abs(dpx) >= MIN_USEFUL_DX_MM:
        return "partial_hand_x_motion"
    return "contact_constrained_or_ineffective"


def screen_combined(env, book, post_blob, desire: dict) -> list:
    """Bounded tilt → unload slip → brake → untilt; measure surviving Δp_x."""
    want_dpx = float(desire["desire_delta_p_rel_x_sign"])
    want_gx = float(desire["desire_g_x_sign"])
    menu = []
    for tgt in TILT_TARGETS_DEG:
        for sign in (+1.0, -1.0):
            for unload in (float(grip_action(-5.0)), -0.48, 0.0):
                for slip_s in (0.35, 0.55, 0.75):
                    menu.append((tgt, sign, unload, slip_s))
    if len(menu) > MAX_COMBINED:
        idx = np.linspace(0, len(menu) - 1, num=MAX_COMBINED, dtype=int)
        menu = [menu[i] for i in idx]

    cands = []
    for tgt, sign, unload, slip_s in menu:
        tilt_acts, dur, n_tilt = tilt_schedule(tgt, sign)
        if not all(_legal(book, a) for a in tilt_acts[:1]):
            continue
        # Phase samples
        restore_capture(env, post_blob["capture"])
        g0 = sample_geom(env.sim)
        p0 = np.asarray(g0["p_c_h"], float)
        th0 = g0["e_theta_deg"]

        # tilt
        ok, reason = _run(env, post_blob, tilt_acts)
        if not ok:
            continue
        g_tilt = sample_geom(env.sim)
        if not g_tilt["bilateral"]:
            continue

        # slip (continue from tilted state — do not restore)
        a_slip = _wrist(0.0, 0.0, unload)
        if not _legal(book, a_slip):
            continue
        slip_acts = [a_slip] * max(1, int(round(slip_s / DT)))
        lost = 0
        from envs.outcomes.horizon_outcome import recovery_failed
        from envs.outcomes.support_drop import latched

        dropped = False
        for a in slip_acts:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
            if failed or latched(env.sim):
                dropped = True
                break
        if dropped:
            continue
        g_slip = sample_geom(env.sim)
        if not g_slip["bilateral"]:
            continue
        p_slip = np.asarray(g_slip["p_c_h"], float)
        dpx_slip = float((p_slip - p0)[0] * 1e3)

        # brake
        brake = [_wrist(0.0, 0.0, 1.0)] * 10
        for a in brake:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
            if failed or latched(env.sim):
                dropped = True
                break
        if dropped:
            continue
        g_brake = sample_geom(env.sim)

        # untilt / restore
        ret = untilt_schedule(n_tilt, sign)
        for a in ret:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post_blob["ref"], lost)
            if failed or latched(env.sim):
                dropped = True
                break
        if dropped or not sample_geom(env.sim)["bilateral"]:
            continue
        g_end = sample_geom(env.sim)
        p_end = np.asarray(g_end["p_c_h"], float)
        dp = (p_end - p0) * 1e3
        dpx = float(dp[0])
        useful = (dpx * want_dpx) >= MIN_USEFUL_DX_MM  # moved in desired direction by ≥2mm
        gx_ok = (g_tilt["g_x"] * want_gx) > 0
        cls = classify_motion(p0, p_end, th0, g_end["e_theta_deg"], g_end["bilateral"])
        score = (
            20.0 * max(0.0, dpx * want_dpx)
            - 3.0 * abs(g_end["e_theta_deg"] - th0)
            - 0.5 * g_end["e_pos_mm"]
            + (25.0 if useful and g_end["e_theta_deg"] <= 10.0 else 0.0)
            + (10.0 if gx_ok else 0.0)
        )
        full_actions = tilt_acts + slip_acts + brake + ret
        cands.append(
            {
                "tag": f"tilt{tgt}_s{sign:+.0f}_g{unload}_slip{slip_s}",
                "target_tilt_deg": tgt,
                "sign": sign,
                "unload": unload,
                "slip_s": slip_s,
                "tilt_dur_s": dur,
                "n_tilt": n_tilt,
                "g_x_after_tilt": g_tilt["g_x"],
                "tilt_proxy_after_tilt": g_tilt["tilt_proxy_deg_atan_gx_gz"],
                "gx_matches_desire": gx_ok,
                "dpx_during_slip_mm": dpx_slip,
                "delta_p_rel_end_mm": dp.tolist(),
                "dpx_surviving_mm": dpx,
                "dpy_surviving_mm": float(dp[1]),
                "dpz_surviving_mm": float(dp[2]),
                "e_pos0": g0["e_pos_mm"],
                "e_pos_slip": g_slip["e_pos_mm"],
                "e_pos_end": g_end["e_pos_mm"],
                "e_signed0": g0["e_signed_mm"],
                "e_signed_end": g_end["e_signed_mm"],
                "theta0": th0,
                "theta_slip": g_slip["e_theta_deg"],
                "theta_end": g_end["e_theta_deg"],
                "survives_restore": True,
                "useful_hand_x": useful,
                "classification": cls,
                "score": score,
                "actions": full_actions,
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def control_reorient_only(env, book, post_blob, tgt=60.0, sign=1.0) -> dict:
    tilt_acts, dur, n = tilt_schedule(tgt, sign)
    restore_capture(env, post_blob["capture"])
    g0 = sample_geom(env.sim)
    p0 = np.asarray(g0["p_c_h"], float)
    ok, _, g1 = True, None, None
    for a in tilt_acts:
        step_legal(env.sim, a)
    g_tilt = sample_geom(env.sim)
    for a in untilt_schedule(n, sign):
        step_legal(env.sim, a)
    g_end = sample_geom(env.sim)
    dp = (np.asarray(g_end["p_c_h"], float) - p0) * 1e3
    return {
        "control": "reorient_no_unload",
        "target_tilt_deg": tgt,
        "sign": sign,
        "g_x_peak": g_tilt["g_x"],
        "delta_p_rel_end_mm": dp.tolist(),
        "dpx_mm": float(dp[0]),
        "e_pos_end": g_end["e_pos_mm"],
        "theta_end": g_end["e_theta_deg"],
        "bilateral_end": g_end["bilateral"],
        "classification": classify_motion(p0, g_end["p_c_h"], g0["e_theta_deg"], g_end["e_theta_deg"], g_end["bilateral"]),
    }


def control_unload_only(env, book, post_blob, unload=None, slip_s=0.55) -> dict:
    if unload is None:
        unload = float(grip_action(-5.0))
    restore_capture(env, post_blob["capture"])
    g0 = sample_geom(env.sim)
    p0 = np.asarray(g0["p_c_h"], float)
    acts = [_wrist(0.0, 0.0, unload)] * max(1, int(round(slip_s / DT))) + [_wrist(0.0, 0.0, 1.0)] * 10
    ok, reason = _run(env, post_blob, acts)
    g_end = sample_geom(env.sim)
    dp = (np.asarray(g_end["p_c_h"], float) - p0) * 1e3
    return {
        "control": "unload_no_reorient",
        "unload": unload,
        "slip_s": slip_s,
        "ok": ok,
        "reason": reason,
        "g_x": g0["g_x"],
        "delta_p_rel_end_mm": dp.tolist(),
        "dpx_mm": float(dp[0]),
        "e_pos_end": g_end["e_pos_mm"],
        "theta_end": g_end["e_theta_deg"],
        "bilateral_end": g_end["bilateral"],
        "classification": classify_motion(p0, g_end["p_c_h"], g0["e_theta_deg"], g_end["e_theta_deg"], g_end["bilateral"]),
    }


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    bar = min(out.shape[0], 18 + 16 * len(lines))
    out[:bar] = (12, 12, 12)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 4
        for line in lines:
            dr.text((6, y), line, fill=(255, 255, 255))
            y += 16
        return np.asarray(im)
    except Exception:
        return out


def render_video(env, blob, actions, dest: Path, title: str, phase_cuts=None):
    """phase_cuts: list of (end_index_exclusive, phase_name)."""
    restore_capture(env, blob["capture"])
    g0 = sample_geom(env.sim)
    p0 = np.asarray(g0["p_c_h"], float)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 400, 560)
    rh = mujoco.Renderer(env.sim.model, 400, 560)
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    frames = []
    cuts = phase_cuts or [(len(actions), "RUN")]

    def phase_at(i):
        for end, name in cuts:
            if i < end:
                return name
        return "RUN"

    def cams():
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam_w.lookat[:] = 0.55 * po + 0.45 * ph
        cam_w.distance = 0.52
        cam_w.azimuth = 138.0
        cam_w.elevation = -18.0
        look = ph + Rh @ np.array([0.0, 0.0, 0.09])
        pos = ph + Rh @ np.array([0.28, -0.32, 0.18])
        vec = pos - look
        dist = float(np.linalg.norm(vec))
        cam_h.lookat[:] = look
        cam_h.distance = dist
        cam_h.azimuth = float(np.degrees(np.arctan2(vec[1], vec[0])))
        cam_h.elevation = float(np.degrees(np.arcsin(np.clip(vec[2] / max(dist, 1e-9), -1, 1))))

    def draw(scn):
        from evaluation.grasp_geometry import cylinder_axis_world

        scn.ngeom = 0
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ p0, [1.0, 0.2, 0.85, 1.0], 0.006)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        # hand-x axis
        add_capsule(scn, ph, ph + 0.08 * Rh[:, 0], [1.0, 0.3, 0.3, 1.0], 0.003)

    def frame(phase):
        cams()
        g = sample_geom(env.sim)
        dp = (np.asarray(g["p_c_h"], float) - p0) * 1e3
        line = (
            f"{phase} t={float(env.sim.data.time)-t0:.2f} θ={g['e_theta_deg']:.1f} "
            f"e={g['e_pos_mm']:.1f} e_x={g['e_signed_mm'][0]:+.1f} "
            f"g_x={g['g_x']:+.1f} tilt~{g['tilt_proxy_deg_atan_gx_gz']:.0f} "
            f"Δp_x={dp[0]:+.1f} bil={int(g['bilateral'])}"
        )
        rw.update_scene(env.sim.data, cam_w)
        draw(rw.scene)
        iw = overlay(np.asarray(rw.render(), np.uint8), ["WORLD " + line, title])
        rh.update_scene(env.sim.data, cam_h)
        draw(rh.scene)
        ih = overlay(np.asarray(rh.render(), np.uint8), ["HAND-REL " + line, title])
        return np.concatenate([iw, ih], axis=1)

    img = frame("BEFORE")
    for _ in range(int(4 * FPS)):
        frames.append(img)
    img = frame("BEFORE")
    for _ in range(int(2.5 * FPS)):
        frames.append(img)

    # Active phases at 0.25×; return/brake also 0.25×
    dup = max(1, int(round((DT / 0.25) * FPS)))
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        img = frame(phase_at(i))
        for _ in range(dup):
            frames.append(img)
    img = frame("AFTER")
    for _ in range(int(4 * FPS)):
        frames.append(img)
    rw.close()
    rh.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "duration_s": len(frames) / FPS, "n_frames": len(frames)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)

    search = json.loads((PRIOR / "search.json").read_text(encoding="utf-8"))
    tag = search["A_raw"]["best"]["tag"]
    state = pickle.loads(STATE_A.read_bytes())
    env = make_env()
    book = load_codebook()
    blob0 = {"capture": state["capture"], "ref": state["ref"]}
    piv = pivot_actions_from_tag(tag)

    print("=== Replay verified pivot ===", flush=True)
    met = pivot_segment_metrics(env, blob0, piv, record_series=False)
    print(
        f"  θ {met['theta0']:.3f}->{met['theta1']:.3f} Δp={met['delta_p_rel_mm']:.3f} "
        f"class={met['classification']}",
        flush=True,
    )
    restore_capture(env, blob0["capture"])
    for a in piv:
        step_legal(env.sim, np.asarray(a, np.float32))
    post_cap = take_capture(env)
    post_blob = {"capture": post_cap, "ref": state["ref"]}
    geom0 = sample_geom(env.sim)
    desire = desired_hand_x_sign(geom0["e_signed_mm"])
    print(f"  post e_signed={geom0['e_signed_mm']} desire={desire}", flush=True)
    (OUT / "snapshots" / "A_post_pivot.pkl").write_bytes(
        pickle.dumps({"capture": post_cap, "ref": state["ref"], "pivot_tag": tag, "geom": geom0, "desire": desire})
    )

    # Coordinate report
    coord = {
        "pivot_verify": {
            "theta0": met["theta0"],
            "theta1": met["theta1"],
            "delta_p_rel_mm": met["delta_p_rel_mm"],
            "delta_p_components_mm": met["delta_p_rel_components_mm"],
            "classification": met["classification"],
        },
        "post_pivot_geometry": geom0,
        "desire": desire,
        "W_MAX_rad_s": float(W_MAX),
        "distinctions": {
            "hand_x": "Rh column 0 (finger/pinch axis in recovery convention)",
            "cylinder_axis": "u_hand (body +Z of cylinder in hand frame)",
            "cylinder_lateral": "directions in hand ⊥ u_hand; hand-x may have axial+lateral parts",
            "u_along_hand_x": geom0["u_along_hand_x"],
            "hand_x_lateral_fraction": geom0["hand_x_lateral_to_cyl_norm"],
        },
    }
    (OUT / "coordinate_geometry.json").write_text(json.dumps(jsafe(coord), indent=2), encoding="utf-8")

    # Controls
    print("=== Diagnostic controls ===", flush=True)
    print("  pivot→HANDOFF", flush=True)
    b_piv = {k: v for k, v in primitive_stabilize_handoff(env, blob0, actions_prefix=piv).items() if k != "actions"}
    print(f"    NEW={b_piv.get('SUCCESS_NEW')} reason={b_piv.get('reason')} e={b_piv.get('pre_pos_mm')}", flush=True)

    # Choose tilt sign that produces g_x in desired direction (probe both at 45°)
    probe = []
    for sign in (+1.0, -1.0):
        tilt_acts, _, _ = tilt_schedule(45.0, sign)
        restore_capture(env, post_cap)
        for a in tilt_acts:
            step_legal(env.sim, a)
        g = sample_geom(env.sim)
        probe.append({"sign": sign, "g_x": g["g_x"], "bilateral": g["bilateral"], "tilt_proxy": g["tilt_proxy_deg_atan_gx_gz"]})
    prefer_sign = max(probe, key=lambda r: r["g_x"] * desire["desire_g_x_sign"] if r["bilateral"] else -1e9)["sign"]
    print(f"  tilt probe: {probe} prefer_sign={prefer_sign}", flush=True)

    c_reori = control_reorient_only(env, book, post_blob, tgt=60.0, sign=prefer_sign)
    print(f"  reorient-only: dpx={c_reori['dpx_mm']:.2f} class={c_reori['classification']}", flush=True)
    c_unload = control_unload_only(env, book, post_blob)
    print(f"  unload-only: dpx={c_unload['dpx_mm']:.2f} class={c_unload['classification']}", flush=True)

    # Tilt vs gravity table (reorient only, no unload)
    tilt_vs_g = []
    for tgt in TILT_TARGETS_DEG:
        for sign in (+1.0, -1.0):
            tilt_acts, dur, n = tilt_schedule(tgt, sign)
            restore_capture(env, post_cap)
            g0 = sample_geom(env.sim)
            for a in tilt_acts:
                step_legal(env.sim, a)
            g1 = sample_geom(env.sim)
            tilt_vs_g.append(
                {
                    "target_deg": tgt,
                    "sign": sign,
                    "dur_s": dur,
                    "g_x0": g0["g_x"],
                    "g_x1": g1["g_x"],
                    "tilt_proxy1": g1["tilt_proxy_deg_atan_gx_gz"],
                    "bilateral": g1["bilateral"],
                    "theta1": g1["e_theta_deg"],
                    "e_pos1": g1["e_pos_mm"],
                }
            )
    (OUT / "tilt_vs_gravity.json").write_text(json.dumps(jsafe(tilt_vs_g), indent=2), encoding="utf-8")

    # Combined screen
    print(f"=== Combined tilt+slip screen (≤{MAX_COMBINED}) ===", flush=True)
    cands = screen_combined(env, book, post_blob, desire)
    print(f"  retained={len(cands)}", flush=True)
    if cands:
        b = cands[0]
        print(
            f"  best {b['tag']} dpx_surv={b['dpx_surviving_mm']:.2f} e_end={b['e_pos_end']:.2f} "
            f"θ={b['theta_end']:.2f} useful={b['useful_hand_x']} class={b['classification']}",
            flush=True,
        )
    top = [{k: v for k, v in c.items() if k != "actions"} for c in cands[:12]]
    (OUT / "combined_screen.json").write_text(json.dumps(jsafe(top), indent=2), encoding="utf-8")

    # HANDOFF only for useful survivors
    handoffs = []
    useful = [c for c in cands if c["useful_hand_x"] and c["theta_end"] <= 12.0]
    useful = useful[:MAX_HANDOFF]
    for c in useful:
        print(f"  HANDOFF {c['tag']} ...", flush=True)
        h = {k: v for k, v in primitive_stabilize_handoff(env, blob0, actions_prefix=piv + c["actions"]).items() if k != "actions"}
        print(f"    NEW={h.get('SUCCESS_NEW')} reason={h.get('reason')} e={h.get('pre_pos_mm')} θ={h.get('pre_theta')}", flush=True)
        handoffs.append({"tag": c["tag"], "handoff": h, "dpx": c["dpx_surviving_mm"], "e_pos_end": c["e_pos_end"]})

    ic_replay = None
    if any(h["handoff"].get("SUCCESS_NEW") for h in handoffs):
        from evaluation.handoff_from_state import evaluate_handoff_from_state

        best = next(h for h in handoffs if h["handoff"].get("SUCCESS_NEW"))
        acts = next(c["actions"] for c in cands if c["tag"] == best["tag"])
        trials = []
        for t in range(3):
            root = pickle.loads(IC_A.read_bytes())
            restore_capture(env, root["capture"])
            for a in state["stabilization_prefix_actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            for a in piv + acts:
                step_legal(env.sim, np.asarray(a, np.float32))
            for _ in range(8):
                step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
            meta = evaluate_handoff_from_state(env.sim, root["ref"], stabilize_n=0)
            trials.append(
                {
                    "trial": t,
                    "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
                    "reason": meta.get("reason"),
                    "pre_theta": meta.get("pre_theta"),
                    "pre_pos_mm": meta.get("pre_pos_mm"),
                }
            )
        ic_replay = {"trials": trials, "SUCCESS_NEW_3of3": all(x["SUCCESS_NEW"] for x in trials)}

    # Videos: best combined + unload-only + reorient-only if informative
    videos = {}
    if cands:
        best = cands[0]
        n_tilt = best["n_tilt"]
        n_slip = max(1, int(round(best["slip_s"] / DT)))
        cuts = [
            (n_tilt, "TILT"),
            (n_tilt + n_slip, "SLIP"),
            (n_tilt + n_slip + 10, "BRAKE"),
            (len(best["actions"]), "UNTILT"),
        ]
        print("  Rendering best combined video...", flush=True)
        videos["best_combined"] = render_video(
            env, post_blob, best["actions"], OUT / "videos" / "A_hand_x_best_combined.mp4", "A hand-x combined", cuts
        )
    # control videos from post-pivot
    tilt_acts, _, n = tilt_schedule(60.0, prefer_sign)
    ret = untilt_schedule(n, prefer_sign)
    videos["reorient_only"] = render_video(
        env,
        post_blob,
        tilt_acts + ret,
        OUT / "videos" / "A_hand_x_reorient_only.mp4",
        "A reorient-only",
        [(len(tilt_acts), "TILT"), (len(tilt_acts) + len(ret), "UNTILT")],
    )
    unload = float(grip_action(-5.0))
    uacts = [_wrist(0.0, 0.0, unload)] * int(round(0.55 / DT)) + [_wrist(0.0, 0.0, 1.0)] * 10
    videos["unload_only"] = render_video(
        env,
        post_blob,
        uacts,
        OUT / "videos" / "A_hand_x_unload_only.mp4",
        "A unload-only",
        [(int(round(0.55 / DT)), "SLIP"), (len(uacts), "BRAKE")],
    )

    # Conclusion classification
    best_dpx = None if not cands else cands[0]["dpx_surviving_mm"]
    best_useful = bool(cands and cands[0]["useful_hand_x"])
    if best_useful:
        conclusion = "useful_controlled_hand_x_sliding"
    elif cands and abs(cands[0]["dpx_surviving_mm"]) >= 1.0:
        conclusion = cands[0]["classification"]
    elif abs(c_unload.get("dpx_mm") or 0) < 1.0 and abs(c_reori.get("dpx_mm") or 0) < 1.0:
        conclusion = "contact_constrained_inability_to_move_laterally"
    else:
        conclusion = "no_stable_surviving_hand_x_correction"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "ic": "ball_train_v2_0070",
        "pivot_tag": tag,
        "pivot_verify": coord["pivot_verify"],
        "desire": desire,
        "post_pivot_e_signed_mm": geom0["e_signed_mm"],
        "tilt_probe": probe,
        "prefer_tilt_sign": prefer_sign,
        "controls": {
            "pivot_handoff": b_piv,
            "reorient_only": c_reori,
            "unload_only": c_unload,
        },
        "n_combined_retained": len(cands),
        "best_combined": None if not cands else {k: v for k, v in cands[0].items() if k != "actions"},
        "handoffs": handoffs,
        "ic_replay": ic_replay,
        "videos": videos,
        "conclusion": conclusion,
        "budgets": {"MAX_COMBINED": MAX_COMBINED, "MAX_HANDOFF": MAX_HANDOFF, "MIN_USEFUL_DX_MM": MIN_USEFUL_DX_MM},
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # REPORT
    lines = [
        "# Hand-X Gravity Reorientation After Pivot — Plan A",
        "",
        f"Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s",
        "",
        "## Pivot verification",
        "",
        f"- θ {met['theta0']:.3f}° → {met['theta1']:.3f}°, Δp_rel={met['delta_p_rel_mm']:.3f} mm",
        f"- Post-pivot e_signed ≈ {geom0['e_signed_mm']} mm (e_pos={geom0['e_pos_mm']:.2f})",
        "",
        "## Coordinate geometry",
        "",
        f"- Desire: {desire}",
        f"- u_hand · hand_x = {geom0['u_along_hand_x']:.3f} (axial share of hand-x)",
        f"- ‖hand_x ⊥ u‖ = {geom0['hand_x_lateral_to_cyl_norm']:.3f}",
        f"- g_h post-pivot = {geom0['g_h']}",
        f"- See `coordinate_geometry.json` for Rh, contacts, normals.",
        "",
        "## Tilt → g_x",
        "",
    ]
    for row in tilt_vs_g:
        if row["sign"] == prefer_sign:
            lines.append(
                f"- target {row['target_deg']:.0f}° sign={row['sign']:+.0f}: "
                f"g_x {row['g_x0']:.2f}→{row['g_x1']:.2f}, tilt_proxy≈{row['tilt_proxy1']:.1f}°, "
                f"bilateral={row['bilateral']}"
            )
    lines += [
        "",
        "## Diagnostic controls",
        "",
        f"- pivot→HANDOFF: NEW={b_piv.get('SUCCESS_NEW')} reason={b_piv.get('reason')} "
        f"pre_e={b_piv.get('pre_pos_mm')} preθ={b_piv.get('pre_theta')}",
        f"- reorient-only (60°): dpx={c_reori['dpx_mm']:.2f} mm class={c_reori['classification']}",
        f"- unload-only: dpx={c_unload['dpx_mm']:.2f} mm class={c_unload['classification']}",
        "",
        "## Combined tilt + slip + brake + untilt",
        "",
        f"Retained {len(cands)} / budget {MAX_COMBINED}. Useful = surviving Δp_x in desire direction ≥ {MIN_USEFUL_DX_MM} mm.",
        "",
    ]
    if cands:
        b = cands[0]
        lines += [
            f"**Best:** `{b['tag']}`",
            f"- Surviving Δp_rel = [{b['dpx_surviving_mm']:.2f}, {b['dpy_surviving_mm']:.2f}, {b['dpz_surviving_mm']:.2f}] mm",
            f"- e_pos {b['e_pos0']:.2f} → {b['e_pos_end']:.2f} mm; θ {b['theta0']:.2f} → {b['theta_end']:.2f}°",
            f"- g_x after tilt = {b['g_x_after_tilt']:.2f} (matches desire: {b['gx_matches_desire']})",
            f"- useful_hand_x={b['useful_hand_x']} classification={b['classification']}",
            "",
        ]
        lines.append("Top survivors:")
        for c in cands[:5]:
            lines.append(
                f"- {c['tag']}: dpx={c['dpx_surviving_mm']:+.2f} e={c['e_pos_end']:.1f} "
                f"θ={c['theta_end']:.1f} useful={c['useful_hand_x']} ({c['classification']})"
            )
    else:
        lines.append("No candidates retained bilateral through tilt→slip→brake→untilt.")
    lines += ["", "## HANDOFF / SUCCESS_NEW", ""]
    if not handoffs:
        lines.append("No HANDOFF launched (no useful surviving hand-x correction ≥ 2 mm with θ≤12°).")
    else:
        for h in handoffs:
            lines.append(f"- {h['tag']}: {h['handoff']}")
        lines.append(f"- IC replay: {ic_replay}")
    lines += [
        "",
        "## Videos",
        "",
    ]
    for k, v in videos.items():
        lines.append(f"- `{k}`: {v.get('path')} ({v.get('duration_s')} s)")
    lines += [
        "",
        "## Conclusion",
        "",
        f"**{conclusion}**",
        "",
        "Distinguish:",
        f"- Useful controlled hand-x sliding: {best_useful}",
        f"- Best surviving dpx (mm): {best_dpx}",
        f"- Reorient-without-unload dpx: {c_reori['dpx_mm']:.2f} ({c_reori['classification']})",
        f"- Unload-without-reorient dpx: {c_unload['dpx_mm']:.2f} ({c_unload['classification']})",
        "",
        "Stopped after Plan A focused experiment.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe({"conclusion": conclusion, "best_dpx": best_dpx, "handoffs": len(handoffs), "wall_s": summary["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
