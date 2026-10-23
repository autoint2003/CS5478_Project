#!/usr/bin/env python
"""Pivot-first precision: verified pivot → controlled slide → HANDOFF (A_raw, B_raw only).

Reuses exact pivot tags from position_preserving_pivot/. No new pivot optimization.
No bulk A/B/C datagen / SFT / GRPO.
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

from controllers.recovery.action_map import grip_action  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import decode_tok, load_vocab  # noqa: E402
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
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_hand, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR = ROOT / "results/diagnostics/position_preserving_pivot"
OUT = ROOT / "results/diagnostics/pivot_first_precision"
STATE_A = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
STATE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"

DT = 0.02
FPS = 25
# Explicit budgets (task §6 / §11)
MAX_SLIDE_SCREEN = 36
MAX_STAGE_C = 3
MAX_WALL_S = 900.0


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


def diagnose_post_pivot(sim) -> dict:
    a = audit_grasp(sim)
    m = measure(sim, U_TGT, P_TGT)
    e = np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)
    u = cylinder_axis_hand(sim)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    e_axial = float(np.dot(e, u))
    e_lat_vec = e - e_axial * u
    e_lat = float(np.linalg.norm(e_lat_vec))
    # Gravity along ±u (slip-useful projection)
    g_h = np.asarray(a["g_h"], float)
    g_along_u = float(np.dot(g_h, u))
    # Desired relative object slip in hand frame to reduce e: object should move −e
    desire_obj_h = -e
    desire_axial = float(np.dot(desire_obj_h, u))
    return {
        "e_signed_mm": (e * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "u_hand": u.tolist(),
        "e_axial_mm": e_axial * 1e3,
        "e_lateral_mm": e_lat * 1e3,
        "e_lateral_vec_mm": (e_lat_vec * 1e3).tolist(),
        "g_h": g_h.tolist(),
        "g_along_u": g_along_u,
        "desire_obj_slip_axial_mm": desire_axial * 1e3,
        "axial_fraction": abs(e_axial) / max(float(np.linalg.norm(e)), 1e-12),
        "contact_s_mean": a.get("contact_s_mean"),
        "grasp_kind": a.get("grasp_kind"),
        "Fn_L": a.get("Fn_L"),
        "Fn_R": a.get("Fn_R"),
        "bilateral": a.get("bilateral"),
        "v_rel_norm": a.get("v_rel_norm"),
        "w_rel_norm": a.get("w_rel_norm"),
        "note": (
            "Axial = projection of e onto cylinder axis in hand frame. "
            "Large lateral fraction cannot be fixed by pure axial sliding alone."
        ),
    }


def handoff_eval(env, blob, actions_prefix=None) -> dict:
    r = primitive_stabilize_handoff(env, blob, actions_prefix=actions_prefix)
    return {k: v for k, v in r.items() if k != "actions"}


def screen_post_pivot_slides(env, blob, book, diag: dict, *, budget: int = MAX_SLIDE_SCREEN) -> list:
    """Stage A: short physical screening of legal slides that preserve θ."""
    restore_capture(env, blob["capture"])
    a0 = audit_grasp(env.sim)
    e0 = np.asarray(a0["e_signed_mm"], float) * 1e-3
    u = np.asarray(diag["u_hand"], float)
    # Compensating hand velocity roughly opposite the error (hand-frame)
    # Also try gravity-tilt unload slips along ±hand-x / ±u-related wrist
    menu = []
    unload_opts = (float(grip_action(-5.0)), -0.48, 0.2)
    # Pure gravity-slip family
    for wy, tilt, slip, unload in (
        (3.0, 0.35, 0.45, unload_opts[0]),
        (3.0, 0.45, 0.60, unload_opts[1]),
        (-1.0, 0.40, 0.50, unload_opts[0]),
        (-1.0, 0.55, 0.70, unload_opts[1]),
        (1.0, 0.40, 0.55, unload_opts[0]),
        (2.0, 0.40, 0.50, unload_opts[1]),
        (-3.0, 0.40, 0.55, unload_opts[0]),
    ):
        menu.append(("grav", 0.0, 0.0, 0.0, 0.0, wy, tilt, slip, unload))
    # Compensating Cartesian during unload (hand moves to reduce e)
    for scale in (0.08, 0.15, 0.22):
        v = -scale * e0 / max(float(np.linalg.norm(e0)), 1e-6)
        v = np.clip(v, -0.35, 0.35)
        for wy in (0.0, 1.0, 3.0, -1.0):
            for unload in unload_opts[:2]:
                for slip in (0.40, 0.60):
                    menu.append(("comp", float(v[0]), float(v[1]), float(v[2]), 0.0, wy, 0.20, slip, unload))
    # Axial-aligned: move hand along ±u so relative slip is along cylinder
    for sign in (+1.0, -1.0):
        for speed in (0.10, 0.18):
            v = sign * speed * u
            for unload in unload_opts[:2]:
                for slip in (0.45, 0.65):
                    menu.append(
                        (
                            "axial",
                            float(v[0]),
                            float(v[1]),
                            float(v[2]),
                            0.0,
                            0.0,
                            0.15,
                            slip,
                            unload,
                        )
                    )

    if budget < len(menu):
        idx = np.linspace(0, len(menu) - 1, num=budget, dtype=int)
        menu = [menu[i] for i in idx]

    cands = []
    for kind, vx, vy, vz, wx, wy, tilt_s, slip_s, unload in menu:
        a_tilt = _action_vw([0.0, 0.0, 0.0], [wx, wy, 0.0], 1.0)
        a_slip = _action_vw([vx, vy, vz], [0.0, 0.0, 0.0], unload)
        a_brake = _wrist(0.0, 0.0, 1.0)
        if not all(_legal(book, a) for a in (a_tilt, a_slip, a_brake)):
            continue
        acts = (
            [a_tilt] * max(1, int(round(tilt_s / DT)))
            + [a_slip] * max(1, int(round(slip_s / DT)))
            + [a_brake] * 10
        )
        restore_capture(env, blob["capture"])
        ok, reason = _run(env, blob, acts)
        a1 = audit_grasp(env.sim)
        if not ok or not a1["bilateral"] or a1["ground_drop"]:
            continue
        d_pos = float(a0["e_pos_mm"] - a1["e_pos_mm"])
        th1 = float(a1["e_theta_deg"])
        # Reject uncontrolled / orientation blowup
        if th1 > 15.0:
            continue
        if a1["v_rel_norm"] > 0.25:
            continue
        if d_pos < -1.0:  # clearly worse
            continue
        score = 10.0 * d_pos - 2.0 * max(0.0, th1 - 10.0) - 0.3 * th1
        if th1 <= 10.0:
            score += 15.0
        if a1["e_pos_mm"] <= 12.0 and th1 <= 10.0:
            score += 40.0
        cands.append(
            {
                "tag": f"{kind}_v{vx:.2f}_{vz:.2f}_wy{wy}_t{tilt_s}_s{slip_s}_g{unload}",
                "kind": kind,
                "actions": acts,
                "e_pos0": a0["e_pos_mm"],
                "e_pos1": a1["e_pos_mm"],
                "e_signed0": a0["e_signed_mm"],
                "e_signed1": a1["e_signed_mm"],
                "theta0": a0["e_theta_deg"],
                "theta1": th1,
                "d_pos_mm": d_pos,
                "d_theta": th1 - float(a0["e_theta_deg"]),
                "in_band": bool(th1 <= 10.0 and a1["e_pos_mm"] <= 12.0),
                "theta_ok": bool(th1 <= 10.0),
                "a1": a1,
                "score": score,
            }
        )
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def stage_b_stabilize(env, blob, actions) -> dict:
    restore_capture(env, blob["capture"])
    ok, reason = _run(env, blob, actions)
    if not ok:
        return {"ok": False, "reason": reason}
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    # restore pre-hold for snapshot
    restore_capture(env, blob["capture"])
    ok2, _ = _run(env, blob, actions)
    geom = audit_grasp(env.sim)
    return {
        "ok": bool(ok2 and stab.get("stable") and geom["bilateral"]),
        "stable": stab.get("stable"),
        "reasons": stab.get("reasons"),
        "geom": geom,
        "theta": geom["e_theta_deg"],
        "e_pos_mm": geom["e_pos_mm"],
        "theta_ok": geom["e_theta_deg"] <= 10.0,
        "in_band": geom["e_theta_deg"] <= 10.0 and geom["e_pos_mm"] <= 12.0,
    }


def replay_ic_full(env, case_id: str, stab_state: dict, precision_actions: list, table=None, book48=None) -> dict:
    """3× exact replay from original airborne IC through stab+precision+HANDOFF."""
    from evaluation.handoff_from_state import evaluate_handoff_from_state

    results = []
    for trial in range(3):
        if case_id == "A":
            root = pickle.loads(IC_A.read_bytes())
            restore_capture(env, root["capture"])
            for a in stab_state["stabilization_prefix_actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
        else:
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for tok in stab_state["stabilization_prefix_tokens"]:
                step_legal(env.sim, decode_tok(table, book48, int(tok)))
            ref = root["ref"]
        for a in precision_actions:
            step_legal(env.sim, np.asarray(a, np.float32))
        brake = _wrist(0.0, 0.0, 1.0)
        for _ in range(8):
            step_legal(env.sim, brake)
        meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
        win = meta.get("window") or {}
        results.append(
            {
                "trial": trial,
                "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
                "reason": meta.get("reason"),
                "pre_theta": meta.get("pre_theta"),
                "pre_pos_mm": meta.get("pre_pos_mm"),
                "longest_both_ok_s": win.get("longest_both_ok_s"),
            }
        )
    return {"trials": results, "SUCCESS_NEW_3of3": all(r["SUCCESS_NEW"] for r in results)}


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    bar = 18 + 16 * len(lines)
    out[: min(bar, out.shape[0])] = (12, 12, 12)
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


def render_extended(env, blob, actions, dest: Path, title: str):
    restore_capture(env, blob["capture"])
    ph0 = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    refs = {
        "p0": np.asarray(measure(env.sim, U_TGT, P_TGT)["p_c_h"], float),
        "u0": cylinder_axis_hand(env.sim).copy(),
    }
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 400, 560)
    rh = mujoco.Renderer(env.sim.model, 400, 560)
    cam_w = mjv_camera_from_preset({"lookat": ph0, "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": ph0, "distance": 0.5, "azimuth": 140, "elevation": -18})
    frames = []

    def cams():
        from evaluation.grasp_geometry import hand_R

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
        from evaluation.grasp_geometry import cylinder_axis_world, hand_R

        scn.ngeom = 0
        Rh = hand_R(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_sphere(scn, ph + Rh @ refs["p0"], [1.0, 0.2, 0.85, 1.0], 0.006)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        u0w = Rh @ refs["u0"]
        u0w = u0w / max(float(np.linalg.norm(u0w)), 1e-12)
        add_capsule(scn, po - 0.05 * u0w, po + 0.05 * u0w, [1.0, 0.55, 0.1, 1.0], 0.0025)

    def frame(phase):
        cams()
        a = audit_grasp(env.sim)
        dp = 1e3 * (np.asarray(measure(env.sim, U_TGT, P_TGT)["p_c_h"], float) - refs["p0"])
        line = (
            f"{phase} t={float(env.sim.data.time)-t0:.2f} θ={a['e_theta_deg']:.1f} "
            f"e={a['e_pos_mm']:.1f} Δp={np.linalg.norm(dp):.1f} bil={int(a['bilateral'])}"
        )
        rw.update_scene(env.sim.data, cam_w)
        draw(rw.scene)
        iw = overlay(np.asarray(rw.render(), np.uint8), ["WORLD " + line, title])
        rh.update_scene(env.sim.data, cam_h)
        draw(rh.scene)
        ih = overlay(np.asarray(rh.render(), np.uint8), ["HAND-REL " + line, title])
        return np.concatenate([iw, ih], axis=1)

    # presentation holds (frozen — not extra control)
    img = frame("BEFORE")
    for _ in range(int(4.0 * FPS)):
        frames.append(img)
    img = frame("BEFORE")
    for _ in range(int(3.0 * FPS)):
        frames.append(img)
    # actions at 0.30×
    speed = 0.30
    dup = max(1, int(round((DT / speed) * FPS)))
    n_piv = min(len(actions), 21)  # first ~pivot portion annotated
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        phase = "PIVOT" if i < n_piv else ("SLIDE" if i < len(actions) - 10 else "BRAKE")
        img = frame(phase)
        for _ in range(dup):
            frames.append(img)
    img = frame("AFTER")
    for _ in range(int(4.0 * FPS)):
        frames.append(img)
    rw.close()
    rh.close()
    # trim/pad to ~15–25s
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": path, "duration_s": len(frames) / FPS, "n_frames": len(frames)}


def run_case(env, book, case_id: str, state: dict, pivot_tag: str, table=None, book48=None) -> dict:
    print(f"\n======== CASE {case_id} ========", flush=True)
    blob0 = {"capture": state["capture"], "ref": state["ref"]}
    piv_acts = pivot_actions_from_tag(pivot_tag)

    # Verify pivot
    met = pivot_segment_metrics(env, blob0, piv_acts, record_series=False)
    print(
        f"  pivot verify θ {met['theta0']:.3f}->{met['theta1']:.3f} Δp={met['delta_p_rel_mm']:.3f} "
        f"class={met['classification']}",
        flush=True,
    )
    restore_capture(env, blob0["capture"])
    for a in piv_acts:
        step_legal(env.sim, np.asarray(a, np.float32))
    post_cap = take_capture(env)
    post_blob = {"capture": post_cap, "ref": state["ref"]}
    diag = diagnose_post_pivot(env.sim)
    stab_post = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, post_cap)
    post_path = OUT / "snapshots" / f"{case_id}_post_pivot.pkl"
    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_bytes(
        pickle.dumps(
            {
                "case": case_id,
                "capture": post_cap,
                "ref": state["ref"],
                "pivot_tag": pivot_tag,
                "pivot_metrics": {k: met[k] for k in met if k not in ("series", "a0", "a1", "p0", "p1")},
                "diagnosis": diag,
                "stability": {k: stab_post[k] for k in ("stable", "reasons", "e_theta_deg", "e_pos_mm")},
            }
        )
    )

    # Baselines
    print("  Baseline 0: stable → HANDOFF", flush=True)
    b0 = handoff_eval(env, blob0, actions_prefix=[])
    print(f"    NEW={b0.get('SUCCESS_NEW')} S12={b0.get('SUCCESS_12S')} reason={b0.get('reason')} "
          f"preθ={b0.get('pre_theta')} pre_e={b0.get('pre_pos_mm')}", flush=True)

    print("  Baseline 1: stable → pivot → HANDOFF", flush=True)
    b1 = handoff_eval(env, blob0, actions_prefix=piv_acts)
    print(f"    NEW={b1.get('SUCCESS_NEW')} S12={b1.get('SUCCESS_12S')} reason={b1.get('reason')} "
          f"preθ={b1.get('pre_theta')} pre_e={b1.get('pre_pos_mm')}", flush=True)

    out = {
        "case": case_id,
        "ic_id": state["ic_id"],
        "pivot_tag": pivot_tag,
        "pivot_verify": {
            "theta0": met["theta0"],
            "theta1": met["theta1"],
            "delta_p_rel_mm": met["delta_p_rel_mm"],
            "delta_p_rel_components_mm": met["delta_p_rel_components_mm"],
            "classification": met["classification"],
            "bilateral_end": met["p1"]["bilateral"],
        },
        "post_pivot_diagnosis": diag,
        "post_pivot_stable": stab_post.get("stable"),
        "baseline0_handoff": b0,
        "baseline1_pivot_handoff": b1,
        "post_pivot_snapshot": str(post_path),
    }

    if b1.get("SUCCESS_NEW"):
        out["slide_search"] = {"skipped": True, "reason": "pivot_alone_SUCCESS_NEW"}
        out["best_complete"] = {"order": "stable→pivot→HANDOFF", "SUCCESS_NEW": True, "handoff": b1}
        out["ic_replay"] = replay_ic_full(env, case_id, state, piv_acts, table, book48)
        return out

    # Stage A slide screen
    print(f"  Stage A: screen ≤{MAX_SLIDE_SCREEN} slides...", flush=True)
    cands = screen_post_pivot_slides(env, post_blob, book, diag, budget=MAX_SLIDE_SCREEN)
    print(f"    retained={len(cands)} best_score={None if not cands else cands[0]['score']:.1f}", flush=True)
    out["slide_search"] = {
        "n_screened_ok": len(cands),
        "budget": MAX_SLIDE_SCREEN,
        "top": [{k: v for k, v in c.items() if k not in ("actions", "a1")} for c in cands[:8]],
    }

    # Stage B on top candidates
    stage_b = []
    for c in cands[:8]:
        sb = stage_b_stabilize(env, post_blob, c["actions"])
        stage_b.append(
            {
                "tag": c["tag"],
                "score": c["score"],
                "d_pos_mm": c["d_pos_mm"],
                "theta1": c["theta1"],
                "e_pos1": c["e_pos1"],
                **{k: sb[k] for k in sb if k != "geom"},
                "geom_theta": None if not sb.get("geom") else sb["geom"]["e_theta_deg"],
                "geom_e_pos": None if not sb.get("geom") else sb["geom"]["e_pos_mm"],
                "actions_n": len(c["actions"]),
            }
        )
    out["stage_b"] = stage_b

    # Select Stage C: prefer in_band / theta_ok / best d_pos
    pool = []
    for c, sb in zip(cands[:8], stage_b):
        if not sb.get("ok"):
            continue
        if sb.get("theta_ok") or c["theta1"] <= 12.0:
            pool.append((c, sb))
    pool.sort(key=lambda x: (-int(x[1].get("in_band") or x[0]["in_band"]), -x[0]["score"]))
    stage_c = []
    for c, sb in pool[:MAX_STAGE_C]:
        print(f"  Stage C: {c['tag']} ...", flush=True)
        full_prefix = piv_acts + c["actions"]
        h = handoff_eval(env, blob0, actions_prefix=full_prefix)
        print(
            f"    NEW={h.get('SUCCESS_NEW')} S12={h.get('SUCCESS_12S')} reason={h.get('reason')} "
            f"preθ={h.get('pre_theta')} pre_e={h.get('pre_pos_mm')} hold={h.get('longest_both_ok_s')}",
            flush=True,
        )
        stage_c.append(
            {
                "tag": c["tag"],
                "handoff": h,
                "pre_slide": {k: c[k] for k in ("e_pos1", "theta1", "e_signed1", "d_pos_mm", "in_band")},
                "stage_b": {k: sb[k] for k in sb if k != "actions"},
                "n_actions_slide": len(c["actions"]),
            }
        )
    out["stage_c"] = stage_c

    best = None
    for row in stage_c:
        if row["handoff"].get("SUCCESS_NEW"):
            best = row
            break
    if best is None and stage_c:
        # best by longest_both_ok / lowest pre_pos among theta_ok
        def key(r):
            h = r["handoff"]
            return (
                int(h.get("SUCCESS_12S") or False),
                float(h.get("longest_both_ok_s") or 0),
                -float(h.get("pre_pos_mm") or 99),
            )

        best = max(stage_c, key=key)
        best = {**best, "SUCCESS_NEW": False}

    out["best_complete"] = None
    if best is not None:
        slide_acts = next(c["actions"] for c in cands if c["tag"] == best["tag"])
        out["best_complete"] = {
            "order": "stable→pivot→slide→HANDOFF",
            "tag": best["tag"],
            "SUCCESS_NEW": bool(best["handoff"].get("SUCCESS_NEW")),
            "handoff": best["handoff"],
            "n_pivot": len(piv_acts),
            "n_slide": len(slide_acts),
        }
        if best["handoff"].get("SUCCESS_NEW"):
            out["ic_replay"] = replay_ic_full(env, case_id, state, piv_acts + slide_acts, table, book48)
        else:
            out["ic_replay"] = {"skipped": True, "reason": "no_SUCCESS_NEW_from_branch"}

    return out, piv_acts, post_blob, cands, diag


def diagnose_from_blob(env, blob):
    restore_capture(env, blob["capture"])
    return diagnose_post_pivot(env.sim)


def order4_slide_then_pivot(env, book, blob0, piv_acts) -> dict:
    diag0 = diagnose_from_blob(env, blob0)
    cands = screen_post_pivot_slides(env, blob0, book, diag0, budget=12)
    if not cands:
        return {"ok": False, "reason": "no_slide_candidates"}
    best_slide = cands[0]
    prefix = best_slide["actions"] + piv_acts
    h = handoff_eval(env, blob0, actions_prefix=prefix)
    return {
        "ok": True,
        "slide_tag": best_slide["tag"],
        "slide_e_pos1": best_slide["e_pos1"],
        "slide_theta1": best_slide["theta1"],
        "handoff": h,
    }


def failure_analysis(out: dict) -> dict:
    diag = out.get("post_pivot_diagnosis") or {}
    top = (out.get("slide_search") or {}).get("top") or []
    stage_c = out.get("stage_c") or []
    reasons = []
    if diag.get("axial_fraction", 1) < 0.35:
        reasons.append("remaining_error_mostly_non_axial")
    if not top:
        reasons.append("no_controllable_slide_passed_screen")
    elif all(t.get("d_pos_mm", 0) < 0.5 for t in top[:3]):
        reasons.append("sliding_does_not_improve_e_pos")
    if top and all(t.get("theta1", 99) > 10 for t in top[:3]):
        reasons.append("sliding_causes_orientation_regression")
    if stage_c and not any(r["handoff"].get("SUCCESS_NEW") for r in stage_c):
        if any(r["handoff"].get("SUCCESS_12S") for r in stage_c):
            reasons.append("nominal_continuation_or_joint_band_fails_SUCCESS_NEW")
        rs = [r["handoff"].get("reason") for r in stage_c]
        if any(x and "DROP" in str(x) for x in rs):
            reasons.append("drop_during_or_after_handoff")
    best = out.get("best_complete") or {}
    h = best.get("handoff") or (stage_c[0]["handoff"] if stage_c else {})
    return {
        "mechanisms": reasons or ["bounded_search_exhausted_without_SUCCESS_NEW"],
        "best_pre_theta": h.get("pre_theta"),
        "best_pre_pos_mm": h.get("pre_pos_mm"),
        "best_longest_both_ok_s": h.get("longest_both_ok_s"),
        "best_reason": h.get("reason"),
        "post_pivot_e_signed_mm": diag.get("e_signed_mm"),
        "post_pivot_axial_mm": diag.get("e_axial_mm"),
        "post_pivot_lateral_mm": diag.get("e_lateral_mm"),
    }


def main():
    t_wall0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)

    search = json.loads((PRIOR / "search.json").read_text(encoding="utf-8"))
    tag_a = search["A_raw"]["best"]["tag"]
    tag_b = search["B_raw"]["best"]["tag"]
    state_a = pickle.loads(STATE_A.read_bytes())
    state_b = pickle.loads(STATE_B.read_bytes())
    env = make_env()
    book = load_codebook()
    table, book48 = load_vocab()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budgets": {"MAX_SLIDE_SCREEN": MAX_SLIDE_SCREEN, "MAX_STAGE_C": MAX_STAGE_C, "MAX_WALL_S": MAX_WALL_S},
        "cases": {},
    }

    # --- Case A ---
    out_a, piv_a, post_a, cands_a, diag_a = run_case(env, book, "A", state_a, tag_a, table, book48)
    o4_a = order4_slide_then_pivot(env, book, {"capture": state_a["capture"], "ref": state_a["ref"]}, piv_a)
    out_a["order4_slide_then_pivot"] = o4_a
    out_a["failure_analysis"] = failure_analysis(out_a)
    # video for best prefix
    blob0a = {"capture": state_a["capture"], "ref": state_a["ref"]}
    if out_a.get("best_complete") and out_a["best_complete"].get("tag"):
        slide = next(c["actions"] for c in cands_a if c["tag"] == out_a["best_complete"]["tag"])
        acts_vid = piv_a + slide
    else:
        acts_vid = piv_a
    print("  Rendering A video...", flush=True)
    out_a["video"] = render_extended(env, blob0a, acts_vid, OUT / "videos" / "A_pivot_first_extended.mp4", "A pivot-first")

    # --- Case B ---
    if time.time() - t_wall0 > MAX_WALL_S:
        print("WALL BUDGET exceeded before B", flush=True)
    out_b, piv_b, post_b, cands_b, diag_b = run_case(env, book, "B", state_b, tag_b, table, book48)
    o4_b = order4_slide_then_pivot(env, book, {"capture": state_b["capture"], "ref": state_b["ref"]}, piv_b)
    out_b["order4_slide_then_pivot"] = o4_b
    out_b["failure_analysis"] = failure_analysis(out_b)
    blob0b = {"capture": state_b["capture"], "ref": state_b["ref"]}
    if out_b.get("best_complete") and out_b["best_complete"].get("tag"):
        slide = next(c["actions"] for c in cands_b if c["tag"] == out_b["best_complete"]["tag"])
        acts_vid = piv_b + slide
    else:
        acts_vid = piv_b
    print("  Rendering B video...", flush=True)
    out_b["video"] = render_extended(env, blob0b, acts_vid, OUT / "videos" / "B_pivot_first_extended.mp4", "B pivot-first")

    summary["cases"]["A"] = out_a
    summary["cases"]["B"] = out_b
    summary["wall_s"] = round(time.time() - t_wall0, 1)

    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "case_A.json").write_text(json.dumps(jsafe(out_a), indent=2), encoding="utf-8")
    (OUT / "case_B.json").write_text(json.dumps(jsafe(out_b), indent=2), encoding="utf-8")

    # Report
    lines = [
        "# Pivot-First Precision Manipulation — A_raw & B_raw",
        "",
        f"Created UTC: {summary['created_utc']}",
        f"Wall: {summary['wall_s']} s",
        f"Budgets: slide_screen≤{MAX_SLIDE_SCREEN}, stage_C≤{MAX_STAGE_C}",
        "",
        "Pivot prefixes reused exactly from `position_preserving_pivot/` (no re-optimization).",
        "",
        "## Pivot verification",
        "",
    ]
    for cid, o in (("A", out_a), ("B", out_b)):
        v = o["pivot_verify"]
        lines.append(
            f"- **{cid}**: θ {v['theta0']:.3f}→{v['theta1']:.3f}°, Δp_rel={v['delta_p_rel_mm']:.3f} mm, "
            f"class={v['classification']}, bilateral={v['bilateral_end']}"
        )
    lines += ["", "## Post-pivot diagnosis", ""]
    for cid, o in (("A", out_a), ("B", out_b)):
        d = o["post_pivot_diagnosis"]
        lines.append(
            f"- **{cid}**: e_pos={d['e_pos_mm']:.2f} mm, e_signed={d['e_signed_mm']}, "
            f"axial={d['e_axial_mm']:.2f} mm, lateral={d['e_lateral_mm']:.2f} mm "
            f"(axial_frac={d['axial_fraction']:.2f}), g·u={d['g_along_u']:.2f}"
        )
    lines += ["", "## Baselines (12 s nominal)", ""]
    for cid, o in (("A", out_a), ("B", out_b)):
        for name, key in (("0 stable→HANDOFF", "baseline0_handoff"), ("1 stable→pivot→HANDOFF", "baseline1_pivot_handoff")):
            h = o[key]
            lines.append(
                f"- **{cid} {name}**: NEW={h.get('SUCCESS_NEW')} S12={h.get('SUCCESS_12S')} "
                f"reason={h.get('reason')} preθ={h.get('pre_theta')} pre_e={h.get('pre_pos_mm')} "
                f"hold={h.get('longest_both_ok_s')}"
            )
    lines += ["", "## Pivot → slide → HANDOFF", ""]
    for cid, o in (("A", out_a), ("B", out_b)):
        bc = o.get("best_complete")
        lines.append(f"- **{cid} best**: {bc}")
        if o.get("ic_replay"):
            lines.append(f"  - IC replay: {o['ic_replay']}")
        lines.append(f"  - Failure analysis: {o.get('failure_analysis')}")
    lines += ["", "## Order comparison (incl. slide→pivot)", ""]
    for cid, o in (("A", out_a), ("B", out_b)):
        lines.append(f"- **{cid} order4 slide→pivot**: {o.get('order4_slide_then_pivot')}")
    lines += [
        "",
        "## Videos",
        "",
        f"- `{out_a.get('video', {}).get('path')}` ({out_a.get('video', {}).get('duration_s')} s)",
        f"- `{out_b.get('video', {}).get('path')}` ({out_b.get('video', {}).get('duration_s')} s)",
        "",
        "## Conclusion",
        "",
    ]
    a_new = bool((out_a.get("best_complete") or {}).get("SUCCESS_NEW") or out_a["baseline1_pivot_handoff"].get("SUCCESS_NEW"))
    b_new = bool((out_b.get("best_complete") or {}).get("SUCCESS_NEW") or out_b["baseline1_pivot_handoff"].get("SUCCESS_NEW"))
    lines.append(
        f"Pivot-first SUCCESS_NEW: A={a_new}, B={b_new}. "
        "See failure_analysis for physical bottlenecks when unsuccessful. "
        "Stopped after these two cases."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe({"wall_s": summary["wall_s"], "A": out_a.get("best_complete"), "B": out_b.get("best_complete")}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
