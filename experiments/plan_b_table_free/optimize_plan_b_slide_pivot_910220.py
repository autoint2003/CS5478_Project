#!/usr/bin/env python
"""Bounded slide-parameter sweep + slide↔pivot search on hard_910220 post-B1.

Legal continuous 7D only. Staged short screens → capped SUCCESS_NEW → 3/3 replay.
Does not modify physics / SUCCESS_NEW / K49. No SFT/GRPO / Plan B expansion.
"""

from __future__ import annotations

import argparse
import csv
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

from controllers.recovery.action_map import (  # noqa: E402
    AIRBORNE_V3,
    grip_action,
    grip_tau,
    normalize_action,
)
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import (  # noqa: E402
    CAP_STEPS,
    legalize,
)
from data_generation.oracles.plan_b.branch_compare import (  # noqa: E402
    exact_replay_3x,
    load_vocab,
    run_tokens,
)
from data_generation.oracles.representability import representability  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.overlays import caption  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_b_slide_pivot_opt_910220"
POST_PKL = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
ROOT_PKL = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
B1_TOKENS = [32, 12, 12, 0, 0, 0, 0, 0]
IC = "hard_910220"
DT = 0.02
# Legal codebook-friendly rates (same family as Plan A)
WY_FAST = 3.0
WY_SLOW = 1.0
WX_INWARD = 1.12


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


def wrist(wx, wy, grip) -> np.ndarray:
    return np.asarray(
        normalize_action(np.zeros(3), np.array([wx, wy, 0.0], float), grip_tau(float(grip)), AIRBORNE_V3),
        np.float32,
    )


def legal_ok(book, a) -> bool:
    packed = representability(np.asarray(a, np.float32).reshape(1, 7), book)
    return float(packed["max_linf"]) <= 0.25 + 1e-8


def snap_geom(sim, R0=None) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    cs = contact_summary(sim)
    Rh = hand_R(sim)
    p_c = np.asarray(m["p_c_h"], float)
    e_h = p_c - np.asarray(P_TGT, float)
    g = np.asarray(pack["g_h"], float)
    # World COM / hand for frame-rotation analysis
    p_hand_w = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    p_obj_w = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    rel_w = p_obj_w - p_hand_w
    out = {
        "e_signed_mm": (e_h * 1e3).tolist(),
        "ex_mm": float(e_h[0] * 1e3),
        "ey_mm": float(e_h[1] * 1e3),
        "ez_mm": float(e_h[2] * 1e3),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "gx": float(g[0]),
        "gy": float(g[1]),
        "gz": float(g[2]),
        "g_h": g.tolist(),
        "rh_mm": (np.asarray(pack["rh"], float) * 1e3).tolist(),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "v_rel_h": np.asarray(pack["v_rel_h"], float).tolist(),
        "v_rel_norm": float(np.linalg.norm(pack["v_rel_h"])),
        "ground_drop": bool(latched(sim)),
        "obj_z": float(m["obj_z"]),
        "p_hand_w": p_hand_w.tolist(),
        "p_obj_w": p_obj_w.tolist(),
    }
    if R0 is not None:
        # Relative COM in initial hand frame; target expressed in R0
        rel_R0 = R0.T @ rel_w
        tgt_R0 = R0.T @ (Rh @ np.asarray(P_TGT, float))
        e_R0 = rel_R0 - tgt_R0
        out["e_signed_R0_mm"] = (e_R0 * 1e3).tolist()
        out["ez_R0_mm"] = float(e_R0[2] * 1e3)
        out["ex_R0_mm"] = float(e_R0[0] * 1e3)
        # Rotation of hand vs start
        err = R0.T @ Rh
        c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
        out["hand_rot_from_start_deg"] = float(np.degrees(np.arccos(c)))
    return out


def ensure_post(env, table, book48):
    if POST_PKL.exists():
        return pickle.loads(POST_PKL.read_bytes())
    root = pickle.loads(ROOT_PKL.read_bytes())
    restore_capture(env, root["capture"])
    ok, reason = run_tokens(env, root, table, book48, B1_TOKENS)
    if not ok:
        raise RuntimeError(reason)
    blob = {"capture": take_capture(env), "ref": root["ref"], "b1_tokens": B1_TOKENS}
    POST_PKL.parent.mkdir(parents=True, exist_ok=True)
    POST_PKL.write_bytes(pickle.dumps(blob))
    return blob


def run_actions(env, post, actions, *, R0=None, phase_marks=None):
    """Execute actions; optionally sample at phase mark step indices."""
    restore_capture(env, post["capture"])
    if R0 is None:
        R0 = hand_R(env.sim).copy()
    marks = {"before": snap_geom(env.sim, R0)}
    lost = 0
    drop_step = None
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, post["ref"], lost)
        if failed or latched(env.sim):
            drop_step = i
            break
        if phase_marks and i + 1 in phase_marks:
            marks[phase_marks[i + 1]] = snap_geom(env.sim, R0)
    marks["end"] = snap_geom(env.sim, R0)
    marks["drop_step"] = drop_step
    marks["R0"] = R0
    return marks


def build_slide_schedule(
    book,
    *,
    tilt_sign: float,
    wy_rate: float,
    gx_target_abs: float,
    max_tilt_deg: float,
    slip_s: float,
    unload_grip: float,
    brake_s: float,
    return_frac: float,
    secure_grip: float = 1.0,
) -> tuple[list, dict] | tuple[None, str]:
    """Closed-loop tilt to |gx| then open-loop slip/brake/return. Returns actions + meta template."""
    # Pre-build legal action templates
    a_tilt = wrist(0.0, float(tilt_sign) * float(wy_rate), secure_grip)
    a_hold = wrist(0.0, 0.0, secure_grip)
    a_slip = wrist(0.0, 0.0, unload_grip)
    a_brake = wrist(0.0, 0.0, secure_grip)
    a_return = wrist(0.0, -float(tilt_sign) * WY_SLOW, secure_grip)
    for name, a in (("tilt", a_tilt), ("slip", a_slip), ("brake", a_brake), ("return", a_return)):
        if not legal_ok(book, a):
            return None, f"illegal_{name}"

    meta = {
        "req_tilt_sign": float(tilt_sign),
        "req_wy_rate": float(wy_rate),
        "req_gx_target_abs": float(gx_target_abs),
        "req_max_tilt_deg": float(max_tilt_deg),
        "req_slip_s": float(slip_s),
        "req_unload_grip": float(unload_grip),
        "req_brake_s": float(brake_s),
        "req_return_frac": float(return_frac),
    }
    return {
        "a_tilt": a_tilt,
        "a_hold": a_hold,
        "a_slip": a_slip,
        "a_brake": a_brake,
        "a_return": a_return,
        "meta": meta,
    }, None


def execute_slide(env, post, book, spec: dict) -> dict:
    """Tilt until gx ready or angle cap, then slip/brake/return with phase snapshots."""
    restore_capture(env, post["capture"])
    R0 = hand_R(env.sim).copy()
    phases = {"before_tilt": snap_geom(env.sim, R0)}
    actions: list[np.ndarray] = []
    lost = 0
    turned = 0.0
    gx_tgt = -float(spec["meta"]["req_tilt_sign"]) * float(spec["meta"]["req_gx_target_abs"])
    max_tilt = float(np.deg2rad(spec["meta"]["req_max_tilt_deg"]))
    wy = float(spec["meta"]["req_tilt_sign"]) * float(spec["meta"]["req_wy_rate"])
    t0 = float(env.sim.data.time)
    # --- tilt ---
    tilt_steps = 0
    while tilt_steps < int(2.5 / DT):
        pack = physical_pack(env.sim)
        gx = float(pack["g_h"][0])
        ready = (gx <= gx_tgt) if gx_tgt < 0 else (gx >= gx_tgt)
        if ready or abs(turned) >= max_tilt - 1e-4:
            break
        a = spec["a_tilt"]
        actions.append(a.copy())
        step_legal(env.sim, a)
        turned += wy * DT
        tilt_steps += 1
        failed, lost = recovery_failed(env.sim, post["ref"], lost)
        if failed or latched(env.sim):
            phases["after_tilt"] = snap_geom(env.sim, R0)
            return _pack_slide_result(spec, actions, phases, R0, turned, drop=True, stage="tilt")
    phases["after_tilt"] = snap_geom(env.sim, R0)
    ach_gx = phases["after_tilt"]["gx"]
    ach_tilt = float(np.degrees(abs(turned)))
    # --- slip ---
    n_slip = max(1, int(round(spec["meta"]["req_slip_s"] / DT)))
    for _ in range(n_slip):
        actions.append(spec["a_slip"].copy())
        step_legal(env.sim, spec["a_slip"])
        failed, lost = recovery_failed(env.sim, post["ref"], lost)
        if failed or latched(env.sim):
            phases["after_slip"] = snap_geom(env.sim, R0)
            return _pack_slide_result(
                spec, actions, phases, R0, turned, drop=True, stage="slip", ach_gx=ach_gx, ach_tilt=ach_tilt
            )
    phases["after_slip"] = snap_geom(env.sim, R0)
    # --- brake ---
    n_brake = max(1, int(round(spec["meta"]["req_brake_s"] / DT)))
    for _ in range(n_brake):
        actions.append(spec["a_brake"].copy())
        step_legal(env.sim, spec["a_brake"])
        failed, lost = recovery_failed(env.sim, post["ref"], lost)
        if failed or latched(env.sim):
            phases["after_brake"] = snap_geom(env.sim, R0)
            return _pack_slide_result(
                spec, actions, phases, R0, turned, drop=True, stage="brake", ach_gx=ach_gx, ach_tilt=ach_tilt
            )
    phases["after_brake"] = snap_geom(env.sim, R0)
    # --- return / untilt ---
    ret_frac = float(spec["meta"]["req_return_frac"])
    if ret_frac > 0:
        n_ret = max(1, int(round(abs(turned) * ret_frac / (WY_SLOW * DT))))
        for _ in range(n_ret):
            actions.append(spec["a_return"].copy())
            step_legal(env.sim, spec["a_return"])
            failed, lost = recovery_failed(env.sim, post["ref"], lost)
            if failed or latched(env.sim):
                phases["after_untilt"] = snap_geom(env.sim, R0)
                return _pack_slide_result(
                    spec, actions, phases, R0, turned, drop=True, stage="return", ach_gx=ach_gx, ach_tilt=ach_tilt
                )
            # stop if nearly back
            cur = hand_R(env.sim)
            err = R0.T @ cur
            c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
            if float(np.degrees(np.arccos(c))) <= 3.0:
                break
    phases["after_untilt"] = snap_geom(env.sim, R0)
    # short stabilize
    for _ in range(5):
        actions.append(spec["a_brake"].copy())
        step_legal(env.sim, spec["a_brake"])
    phases["after_stabilize"] = snap_geom(env.sim, R0)
    return _pack_slide_result(
        spec, actions, phases, R0, turned, drop=False, stage="done", ach_gx=ach_gx, ach_tilt=ach_tilt
    )


def _pack_slide_result(spec, actions, phases, R0, turned, *, drop, stage, ach_gx=None, ach_tilt=None):
    end = phases.get("after_stabilize") or phases.get("after_untilt") or phases.get("after_brake") or phases.get("after_slip") or phases.get("after_tilt")
    before = phases["before_tilt"]
    after_slip = phases.get("after_slip") or end
    d_rhx = float(after_slip["rh_mm"][0] - before["rh_mm"][0]) if after_slip else None
    # z growth analysis
    z_analysis = None
    if "after_untilt" in phases and "after_tilt" in phases:
        z_analysis = {
            "ez_before": before["ez_mm"],
            "ez_after_tilt": phases["after_tilt"]["ez_mm"],
            "ez_after_slip": phases.get("after_slip", {}).get("ez_mm"),
            "ez_after_untilt": phases["after_untilt"]["ez_mm"],
            "ez_R0_before": before.get("ez_R0_mm"),
            "ez_R0_after_tilt": phases["after_tilt"].get("ez_R0_mm"),
            "ez_R0_after_slip": (phases.get("after_slip") or {}).get("ez_R0_mm"),
            "ez_R0_after_untilt": phases["after_untilt"].get("ez_R0_mm"),
            "hand_rot_after_tilt": phases["after_tilt"].get("hand_rot_from_start_deg"),
            "hand_rot_after_untilt": phases["after_untilt"].get("hand_rot_from_start_deg"),
            "interpretation": (
                "If ez_hand rises during tilt but ez_R0 stays flat → mostly frame rotation. "
                "If ez_R0 also rises during slip → real relative COM motion along initial z."
            ),
        }
    score = -1e9
    if not drop and end and end["bilateral"] and not end["ground_drop"]:
        # Joint promise: prefer low e_pos and low theta together (soft)
        score = (
            50.0
            - float(end["e_pos_mm"])
            - 0.5 * float(end["e_theta_deg"])
            - 2.0 * max(0.0, float(end["e_pos_mm"]) - 12.0)
            - 2.0 * max(0.0, float(end["e_theta_deg"]) - 10.0)
            + max(0.0, -d_rhx if d_rhx is not None else 0.0)
        )
    return {
        **spec["meta"],
        "ach_tilt_deg": ach_tilt if ach_tilt is not None else float(np.degrees(abs(turned))),
        "ach_gx_after_tilt": ach_gx if ach_gx is not None else (phases.get("after_tilt") or {}).get("gx"),
        "ach_gx_during_slip": (phases.get("after_slip") or {}).get("gx"),
        "d_rhx_mm": d_rhx,
        "d_ex_mm": None if not end else float(end["ex_mm"] - before["ex_mm"]),
        "d_ez_mm": None if not end else float(end["ez_mm"] - before["ez_mm"]),
        "drop": drop,
        "fail_stage": stage if drop else None,
        "stable": bool(not drop and end and end["bilateral"] and not end["ground_drop"] and end["v_rel_norm"] < 0.15),
        "phases": {k: v for k, v in phases.items()},
        "end": end,
        "z_analysis": z_analysis,
        "score": score,
        "n_actions": len(actions),
        "actions": actions,
        "tag": (
            f"gx{spec['meta']['req_gx_target_abs']}_wy{spec['meta']['req_wy_rate']}_"
            f"s{spec['meta']['req_tilt_sign']:+.0f}_slip{spec['meta']['req_slip_s']}_"
            f"g{spec['meta']['req_unload_grip']}_br{spec['meta']['req_brake_s']}_"
            f"ret{spec['meta']['req_return_frac']}"
        ),
    }


def build_pivot_menu(book) -> list[dict]:
    """Legal open-loop pivot / orient corrections (reduce θ without assuming slide)."""
    menu = []
    grips = (1.0, -0.48)
    for grip in grips:
        for wx, wy, s in (
            (0.0, 1.0, 0.24),
            (0.0, -1.0, 0.24),
            (0.0, 1.0, 0.48),
            (0.0, -1.0, 0.48),
            (0.0, 3.0, 0.24),
            (0.0, -1.0, 0.40),
            (WX_INWARD, 2.0, 0.30),
            (WX_INWARD, -1.0, 0.30),
            (-WX_INWARD, 2.0, 0.30),
            (0.0, 1.0, 0.16),
            (0.0, -1.0, 0.16),
            (1.12, 0.0, 0.30),
            (-1.12, 0.0, 0.30),
        ):
            a = wrist(wx, wy, grip)
            if not legal_ok(book, a):
                continue
            n = max(1, int(round(s / DT)))
            # retighten tail
            a_b = wrist(0.0, 0.0, 1.0)
            acts = [a.copy()] * n + [a_b.copy()] * 8
            menu.append({"name": f"piv_wx{wx}_wy{wy}_g{grip}_s{s}", "actions": acts})
    # unique by name
    seen, out = set(), []
    for m in menu:
        if m["name"] in seen:
            continue
        seen.add(m["name"])
        out.append(m)
    return out


def screen_ok(geom: dict) -> bool:
    if not geom:
        return False
    if geom.get("ground_drop") or not geom.get("bilateral"):
        return False
    if float(geom.get("v_rel_norm") or 0) > 0.25:
        return False
    if float(geom.get("obj_z") or 0) < 0.15:
        return False
    return True


def joint_score(geom: dict) -> float:
    """Higher is better; does not reject intermediate θ>10."""
    return (
        -float(geom["e_pos_mm"])
        - 0.35 * float(geom["e_theta_deg"])
        - 3.0 * max(0.0, float(geom["e_pos_mm"]) - 12.0)
        - 1.5 * max(0.0, float(geom["e_theta_deg"]) - 12.0)
        + (5.0 if geom.get("bilateral") else -30.0)
    )


def full_eval(env, post, actions) -> dict:
    restore_capture(env, post["capture"])
    lost = 0
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        failed, lost = recovery_failed(env.sim, post["ref"], lost)
        if failed or latched(env.sim):
            g = snap_geom(env.sim)
            return {
                "SUCCESS_12S": False,
                "SUCCESS_NEW": False,
                "reason": "DROP_DURING_CORRECTION",
                "pre": g,
                "longest_both_ok_s": 0.0,
            }
    pre = snap_geom(env.sim)
    meta = evaluate_handoff_from_state(env.sim, post["ref"], stabilize_n=0)
    win = meta.get("window") or {}
    return {
        "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "reason": meta.get("reason"),
        "pre": pre,
        "pre_theta": meta.get("pre_theta"),
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "longest_both_ok_s": win.get("longest_both_ok_s"),
        "mean_theta_final2": win.get("mean_theta"),
        "mean_pos_final2": win.get("mean_pos"),
        "max_theta_final2": win.get("max_theta"),
        "max_pos_final2": win.get("max_pos"),
        "working_SUCCESS_NEW_hold": win.get("working_SUCCESS_NEW_hold"),
    }


def render_video(env, post, actions, dest: Path, label: str):
    restore_capture(env, post["capture"])
    look = 0.5 * (
        np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -20.0})
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames = []
    t0 = float(env.sim.data.time)

    def grab(tag):
        cam.lookat[:] = 0.5 * (
            np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        renderer.update_scene(env.sim.data, cam)
        m = measure(env.sim, U_TGT, P_TGT)
        img = np.asarray(renderer.render(), np.uint8).copy()
        frames.append(
            caption(
                img,
                f"{label} {tag} t={float(env.sim.data.time)-t0:.2f}s "
                f"e_pos={m['e_pos_mm']:.1f} θ={m['e_theta_deg']:.1f}",
            )
        )

    grab("START")
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == len(actions) - 1:
            grab("CORR")
    # short nominal peek (not full 12s) for video
    from envs.outcomes.recovery_success import apply_handoff, nominal_step

    apply_handoff(env.sim, post["ref"])
    for i in range(80):
        nominal_step(env.sim)
        if i % 4 == 0:
            grab("NOM")
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_frames_mp4(frames, dest, fps=25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-full", type=int, default=14)
    ap.add_argument("--max-slide-keep", type=int, default=10)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    table, book48 = load_vocab()
    book = load_codebook()
    env = make_env()
    post = ensure_post(env, table, book48)
    root = pickle.loads(ROOT_PKL.read_bytes())
    t_wall0 = time.time()

    print(f"Slide–pivot optimization {IC} max_full={args.max_full}", flush=True)

    # --- Parameter grid (bounded, around demonstrated region) ---
    unload_opts = (-0.48, float(grip_action(-5.0)), -1.0)
    grid = []
    for gx_abs in (3.0, 4.5, 6.0, 7.5):
        for wy_rate in (WY_SLOW, WY_FAST):
            for tilt_sign in (+1.0, -1.0):
                for slip_s in (0.30, 0.50, 0.70, 0.90):
                    for unload in unload_opts:
                        for brake_s in (0.20, 0.40):
                            for ret in (0.0, 0.5, 1.0):
                                # denser near demonstrated: gx=6, wy=3, sign=+ (pinch>0), slip~0.7, ret=1
                                near = (
                                    abs(gx_abs - 6.0) < 0.1
                                    and abs(wy_rate - WY_FAST) < 0.1
                                    and tilt_sign > 0
                                )
                                if not near and (slip_s not in (0.50, 0.70) or unload != float(grip_action(-5.0))):
                                    continue
                                if not near and ret == 0.5:
                                    continue
                                grid.append(
                                    dict(
                                        tilt_sign=tilt_sign,
                                        wy_rate=wy_rate,
                                        gx_target_abs=gx_abs,
                                        max_tilt_deg=100.0 if wy_rate >= 2 else 70.0,
                                        slip_s=slip_s,
                                        unload_grip=unload,
                                        brake_s=brake_s,
                                        return_frac=ret,
                                    )
                                )
    # Dedup
    uniq, seen = [], set()
    for g in grid:
        k = tuple(sorted(g.items()))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(g)
    print(f"Slide grid size={len(uniq)}", flush=True)

    slide_rows = []
    for i, g in enumerate(uniq):
        built, err = build_slide_schedule(book, **g)
        if err:
            continue
        t0 = time.time()
        out = execute_slide(env, post, book, built)
        out["wall_s"] = round(time.time() - t0, 3)
        # strip actions for CSV later
        row = {k: v for k, v in out.items() if k not in ("actions", "phases")}
        row["end_ex"] = (out.get("end") or {}).get("ex_mm")
        row["end_ey"] = (out.get("end") or {}).get("ey_mm")
        row["end_ez"] = (out.get("end") or {}).get("ez_mm")
        row["end_e_pos"] = (out.get("end") or {}).get("e_pos_mm")
        row["end_theta"] = (out.get("end") or {}).get("e_theta_deg")
        row["end_bilateral"] = (out.get("end") or {}).get("bilateral")
        slide_rows.append(out)
        if (i + 1) % 10 == 0 or i == 0:
            e = out.get("end") or {}
            print(
                f"  [{i+1}/{len(uniq)}] {out['tag']} drop={out['drop']} "
                f"tilt={out['ach_tilt_deg']:.1f} gx={out['ach_gx_after_tilt']} "
                f"d_rhx={out['d_rhx_mm']} e_pos={e.get('e_pos_mm')} θ={e.get('e_theta_deg')} "
                f"score={out['score']:.1f}",
                flush=True,
            )

    stable = [r for r in slide_rows if r.get("stable")]
    stable.sort(key=lambda r: r["score"], reverse=True)
    keep = stable[: args.max_slide_keep]
    print(f"Stable slides={len(stable)} keeping={len(keep)}", flush=True)

    pivot_menu = build_pivot_menu(book)
    print(f"Pivot menu={len(pivot_menu)}", flush=True)

    # --- slide → pivot ---
    seq_sp = []
    for s in keep:
        mid_cap = None
        restore_capture(env, post["capture"])
        for a in s["actions"]:
            step_legal(env.sim, a)
        if not screen_ok(snap_geom(env.sim)):
            continue
        mid_cap = take_capture(env)
        mid_blob = {"capture": mid_cap, "ref": post["ref"]}
        for piv in pivot_menu:
            restore_capture(env, mid_cap)
            lost = 0
            drop = False
            for a in piv["actions"]:
                step_legal(env.sim, a)
                failed, lost = recovery_failed(env.sim, post["ref"], lost)
                if failed or latched(env.sim):
                    drop = True
                    break
            g = snap_geom(env.sim)
            if drop or not screen_ok(g):
                continue
            acts = list(s["actions"]) + list(piv["actions"])
            seq_sp.append(
                {
                    "order": "slide_then_pivot",
                    "slide_tag": s["tag"],
                    "pivot": piv["name"],
                    "actions": acts,
                    "geom": g,
                    "score": joint_score(g),
                    "slide_end": s.get("end"),
                }
            )
    seq_sp.sort(key=lambda r: r["score"], reverse=True)
    print(f"slide→pivot survivors={len(seq_sp)} best_score={seq_sp[0]['score'] if seq_sp else None}", flush=True)

    # --- pivot → slide (control) ---
    seq_ps = []
    for piv in pivot_menu[:16]:  # bound
        restore_capture(env, post["capture"])
        lost = 0
        drop = False
        for a in piv["actions"]:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post["ref"], lost)
            if failed or latched(env.sim):
                drop = True
                break
        if drop or not screen_ok(snap_geom(env.sim)):
            continue
        piv_cap = take_capture(env)
        # re-run a compact slide grid from this state
        for g in (
            dict(tilt_sign=1.0, wy_rate=WY_FAST, gx_target_abs=6.0, max_tilt_deg=100.0, slip_s=0.5, unload_grip=float(grip_action(-5.0)), brake_s=0.4, return_frac=1.0),
            dict(tilt_sign=1.0, wy_rate=WY_FAST, gx_target_abs=4.5, max_tilt_deg=100.0, slip_s=0.7, unload_grip=-0.48, brake_s=0.4, return_frac=1.0),
            dict(tilt_sign=1.0, wy_rate=WY_SLOW, gx_target_abs=6.0, max_tilt_deg=70.0, slip_s=0.7, unload_grip=float(grip_action(-5.0)), brake_s=0.4, return_frac=1.0),
            dict(tilt_sign=-1.0, wy_rate=WY_FAST, gx_target_abs=6.0, max_tilt_deg=100.0, slip_s=0.5, unload_grip=float(grip_action(-5.0)), brake_s=0.4, return_frac=1.0),
        ):
            built, err = build_slide_schedule(book, **g)
            if err:
                continue
            # execute slide from piv_cap
            piv_post = {"capture": piv_cap, "ref": post["ref"]}
            sres = execute_slide(env, piv_post, book, built)
            if not sres.get("stable"):
                continue
            acts = list(piv["actions"]) + list(sres["actions"])
            seq_ps.append(
                {
                    "order": "pivot_then_slide",
                    "pivot": piv["name"],
                    "slide_tag": sres["tag"],
                    "actions": acts,
                    "geom": sres["end"],
                    "score": joint_score(sres["end"]),
                    "slide_detail": {k: sres[k] for k in ("ach_tilt_deg", "ach_gx_after_tilt", "d_rhx_mm", "end")},
                }
            )
    seq_ps.sort(key=lambda r: r["score"], reverse=True)
    print(f"pivot→slide survivors={len(seq_ps)} best_score={seq_ps[0]['score'] if seq_ps else None}", flush=True)

    # --- Full SUCCESS_NEW on diverse top candidates ---
    candidates = []
    if keep:
        candidates.append({"family": "slide_only", "name": keep[0]["tag"], "actions": keep[0]["actions"], "geom": keep[0]["end"]})
    for r in seq_sp[:6]:
        candidates.append({"family": "slide_then_pivot", "name": f"{r['slide_tag']}+{r['pivot']}", "actions": r["actions"], "geom": r["geom"]})
    for r in seq_ps[:4]:
        candidates.append({"family": "pivot_then_slide", "name": f"{r['pivot']}+{r['slide_tag']}", "actions": r["actions"], "geom": r["geom"]})
    # also top-3 slides by score
    for s in keep[1:4]:
        candidates.append({"family": "slide_only", "name": s["tag"], "actions": s["actions"], "geom": s["end"]})

    # dedup by name
    seen_n, uniq_c = set(), []
    for c in candidates:
        if c["name"] in seen_n:
            continue
        seen_n.add(c["name"])
        uniq_c.append(c)
    uniq_c = uniq_c[: args.max_full]

    full_rows = []
    best_new = None
    print(f"Full SUCCESS_NEW evals={len(uniq_c)}", flush=True)
    for c in uniq_c:
        t1 = time.time()
        ev = full_eval(env, post, c["actions"])
        ev.update({"family": c["family"], "name": c["name"], "n_actions": len(c["actions"]), "wall_s": round(time.time() - t1, 2)})
        ev["pre_ex"] = (ev.get("pre") or {}).get("ex_mm")
        ev["pre_ez"] = (ev.get("pre") or {}).get("ez_mm")
        full_rows.append(ev)
        print(
            f"  FULL {c['family']}|{c['name'][:60]} NEW={ev['SUCCESS_NEW']} S12={ev['SUCCESS_12S']} "
            f"reason={ev['reason']} θ={ev.get('pre_theta')} pos={ev.get('pre_pos_mm')} "
            f"hold={ev.get('longest_both_ok_s')} wall={ev['wall_s']}s",
            flush=True,
        )
        if ev["SUCCESS_NEW"] and best_new is None:
            best_new = {**c, "full": ev}

    replay = None
    if best_new is not None:
        arr = np.stack([np.asarray(a, np.float32) for a in best_new["actions"]], axis=0)
        tokens, _, _, _ = legalize(arr, book)
        full_toks = list(B1_TOKENS) + [int(t) for t in tokens]
        print(f"exact_replay 3x n={len(full_toks)}", flush=True)
        replay = exact_replay_3x(env, root, table, book48, full_toks, branch="B2", prefix_len=1)
        print(f"replay {replay['n_pass']}/3 ok={replay['ok']}", flush=True)
        (OUT / "SUCCESS_NEW_expert.json").write_text(
            json.dumps(
                jsafe(
                    {
                        "ic_id": IC,
                        "family": best_new["family"],
                        "name": best_new["name"],
                        "tokens": full_toks,
                        "exact_replay": replay,
                        "full": best_new["full"],
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    # Videos for most informative
    videos = {}
    if not args.skip_video:
        picks = []
        if keep:
            picks.append(("best_slide_only", keep[0]["actions"]))
        if seq_sp:
            picks.append(("best_slide_then_pivot", seq_sp[0]["actions"]))
        if seq_ps:
            picks.append(("best_pivot_then_slide", seq_ps[0]["actions"]))
        if best_new:
            picks.append(("SUCCESS_NEW", best_new["actions"]))
        for label, acts in picks:
            dest = OUT / "videos" / f"{IC}__{label}.mp4"
            print(f"  render {dest.name} ...", flush=True)
            try:
                path = render_video(env, post, acts, dest, label)
                videos[label] = str(path) if path else None
            except Exception as exc:  # noqa: BLE001
                videos[label] = f"error:{exc}"

    # Sweep CSV
    csv_path = OUT / "SLIDE_SWEEP.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "tag",
            "req_gx_target_abs",
            "req_wy_rate",
            "req_tilt_sign",
            "req_slip_s",
            "req_unload_grip",
            "req_brake_s",
            "req_return_frac",
            "ach_tilt_deg",
            "ach_gx_after_tilt",
            "ach_gx_during_slip",
            "d_rhx_mm",
            "d_ex_mm",
            "d_ez_mm",
            "end_ex",
            "end_ey",
            "end_ez",
            "end_e_pos",
            "end_theta",
            "end_bilateral",
            "drop",
            "fail_stage",
            "stable",
            "score",
        ]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in slide_rows:
            w.writerow(
                {
                    **{k: r.get(k) for k in fields},
                    "end_ex": (r.get("end") or {}).get("ex_mm"),
                    "end_ey": (r.get("end") or {}).get("ey_mm"),
                    "end_ez": (r.get("end") or {}).get("ez_mm"),
                    "end_e_pos": (r.get("end") or {}).get("e_pos_mm"),
                    "end_theta": (r.get("end") or {}).get("e_theta_deg"),
                    "end_bilateral": (r.get("end") or {}).get("bilateral"),
                }
            )

    # Limitation analysis
    limitation = None
    if best_new is None:
        best_s = keep[0] if keep else None
        best_sp = seq_sp[0] if seq_sp else None
        best_ps = seq_ps[0] if seq_ps else None
        limitation = {
            "reachable_position": None
            if not best_s
            else {
                "best_e_pos": (best_s.get("end") or {}).get("e_pos_mm"),
                "best_ex_ez": [(best_s.get("end") or {}).get("ex_mm"), (best_s.get("end") or {}).get("ez_mm")],
                "note": "If best e_pos still >12 after diverse gx/slip, position band may be hard on this grasp",
            },
            "pivot_induced_position_loss": None
            if not best_sp
            else {
                "slide_e_pos": (best_sp.get("slide_end") or {}).get("e_pos_mm"),
                "after_pivot_e_pos": (best_sp.get("geom") or {}).get("e_pos_mm"),
                "after_pivot_theta": (best_sp.get("geom") or {}).get("e_theta_deg"),
            },
            "best_full_rows": [
                {k: r.get(k) for k in ("family", "name", "SUCCESS_NEW", "SUCCESS_12S", "reason", "pre_theta", "pre_pos_mm", "longest_both_ok_s", "mean_pos_final2", "mean_theta_final2")}
                for r in full_rows
            ],
            "z_growth_examples": [s.get("z_analysis") for s in keep[:3] if s.get("z_analysis")],
        }

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ic_id": IC,
        "n_slide_grid": len(uniq),
        "n_slide_stable": len(stable),
        "n_slide_pivot": len(seq_sp),
        "n_pivot_slide": len(seq_ps),
        "n_full": len(full_rows),
        "wall_s": round(time.time() - t_wall0, 1),
        "best_slide_only": None
        if not keep
        else {
            "tag": keep[0]["tag"],
            "end": keep[0]["end"],
            "ach_tilt_deg": keep[0]["ach_tilt_deg"],
            "ach_gx_after_tilt": keep[0]["ach_gx_after_tilt"],
            "d_rhx_mm": keep[0]["d_rhx_mm"],
            "z_analysis": keep[0].get("z_analysis"),
            "score": keep[0]["score"],
        },
        "best_slide_then_pivot": None
        if not seq_sp
        else {"name": f"{seq_sp[0]['slide_tag']}+{seq_sp[0]['pivot']}", "geom": seq_sp[0]["geom"], "score": seq_sp[0]["score"]},
        "best_pivot_then_slide": None
        if not seq_ps
        else {"name": f"{seq_ps[0]['pivot']}+{seq_ps[0]['slide_tag']}", "geom": seq_ps[0]["geom"], "score": seq_ps[0]["score"]},
        "full_rows": full_rows,
        "SUCCESS_NEW": bool(best_new and replay and replay.get("ok")),
        "exact_replay": replay,
        "videos": videos,
        "limitation": limitation,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "full_evals.json").write_text(json.dumps(jsafe(full_rows), indent=2), encoding="utf-8")
    # Save phase detail for best slide
    if keep:
        (OUT / "best_slide_phases.json").write_text(
            json.dumps(jsafe({"tag": keep[0]["tag"], "phases": keep[0]["phases"], "z_analysis": keep[0].get("z_analysis")}), indent=2),
            encoding="utf-8",
        )

    lines = [
        "# Plan B Slide–Pivot Optimization — hard_910220",
        "",
        f"Created UTC: {summary['created_utc']}",
        f"Wall: {summary['wall_s']}s | slide grid={summary['n_slide_grid']} stable={summary['n_slide_stable']} "
        f"S→P={summary['n_slide_pivot']} P→S={summary['n_pivot_slide']} full={summary['n_full']}",
        "",
        "## Best slide-only",
        "",
        f"```json\n{json.dumps(jsafe(summary['best_slide_only']), indent=2)[:2000]}\n```",
        "",
        "## Best slide→pivot",
        "",
        f"```json\n{json.dumps(jsafe(summary['best_slide_then_pivot']), indent=2)[:1500]}\n```",
        "",
        "## Best pivot→slide",
        "",
        f"```json\n{json.dumps(jsafe(summary['best_pivot_then_slide']), indent=2)[:1500]}\n```",
        "",
        f"## SUCCESS_NEW: {summary['SUCCESS_NEW']}",
        "",
        f"Replay: {summary['exact_replay']}",
        "",
        f"Videos: {summary['videos']}",
        "",
        "## Limitation",
        "",
        f"```json\n{json.dumps(jsafe(limitation), indent=2)[:3000]}\n```",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe({k: summary[k] for k in ("wall_s", "n_slide_stable", "SUCCESS_NEW", "best_slide_only", "best_slide_then_pivot", "best_pivot_then_slide", "videos")}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
