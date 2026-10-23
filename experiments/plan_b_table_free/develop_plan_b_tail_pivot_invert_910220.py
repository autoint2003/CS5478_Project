#!/usr/bin/env python
"""Staged Plan B hypothesis: tail-slide → pivot → invert → final-slide on hard_910220.

Demonstrates each phase with MuJoCo contact geometry before chaining.
Legal continuous 7D only. No physics / SUCCESS_NEW / K49 changes. No SFT/GRPO.
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

from controllers.recovery.action_map import (  # noqa: E402
    AIRBORNE_V3,
    grip_action,
    grip_tau,
    normalize_action,
)
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import legalize  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import (  # noqa: E402
    exact_replay_3x,
    load_vocab,
    run_tokens,
)
from data_generation.oracles.plan_c.catch_stabilization import grasp_geometry  # noqa: E402
from data_generation.oracles.plan_c.contact_surface import classify_contacts  # noqa: E402
from data_generation.oracles.representability import representability  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_scene import CYL_HALF, CYL_R  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_hand, cylinder_axis_world, hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.overlays import caption  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_b_tail_pivot_invert_910220"
POST_PKL = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
ROOT_PKL = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
B1_TOKENS = [32, 12, 12, 0, 0, 0, 0, 0]
IC = "hard_910220"
DT = 0.02
# Tail sidewall band: near end but not face pinch
TAIL_S_MIN = 0.018  # |contact_s| must reach at least this (mm scale: 18mm of 30mm half)
TAIL_S_MAX = CYL_HALF - 0.004  # stay off end-face band
DS_TAIL_MIN = 0.003  # ≥3 mm axial migration toward intended tail
DS_REL_PIVOT_MIN_DEG = 8.0  # meaningful relative orientation change
G_AXIAL_TARGET = 5.0


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


def wrist(wx, wy, grip, v_hand=None) -> np.ndarray:
    v = np.zeros(3, float) if v_hand is None else np.asarray(v_hand, float).reshape(3)
    return np.asarray(
        normalize_action(v, np.array([wx, wy, 0.0], float), grip_tau(float(grip)), AIRBORNE_V3),
        np.float32,
    )


def legal_ok(book, a) -> bool:
    packed = representability(np.asarray(a, np.float32).reshape(1, 7), book)
    return float(packed["max_linf"]) <= 0.25 + 1e-8


def audit(sim) -> dict:
    surf = classify_contacts(sim)
    geo = grasp_geometry(sim)
    m = measure(sim, U_TGT, P_TGT)
    pack = physical_pack(sim)
    u_w = cylinder_axis_world(sim)
    u_h = cylinder_axis_hand(sim)
    g_h = np.asarray(pack["g_h"], float)
    e_h = np.asarray(m["p_c_h"], float) - np.asarray(P_TGT, float)
    s = surf.get("contact_s_mean")
    if s is None or (isinstance(s, float) and not np.isfinite(s)):
        s = geo.get("contact_s_mean")
    # Per-finger mean axial
    sL = [c["contact_s"] for c in surf.get("contacts") or [] if c["side"] == "L"]
    sR = [c["contact_s"] for c in surf.get("contacts") or [] if c["side"] == "R"]
    return {
        "contact_s_mean": None if s is None else float(s),
        "contact_s_L": None if not sL else float(np.mean(sL)),
        "contact_s_R": None if not sR else float(np.mean(sR)),
        "dist_end_plus_m": geo.get("dist_end_plus_m"),
        "dist_end_minus_m": geo.get("dist_end_minus_m"),
        "dist_nearest_end_m": geo.get("dist_nearest_end_m"),
        "near_end": geo.get("near_end"),
        "grasp_kind": surf.get("grasp_kind"),
        "opposing_sidewall": int(surf.get("opposing_sidewall") or 0),
        "n_sidewall": surf.get("n_sidewall"),
        "n_end_face": surf.get("n_end_face"),
        "maj_L": surf.get("maj_L"),
        "maj_R": surf.get("maj_R"),
        "u_world": u_w.tolist(),
        "u_hand": u_h.tolist(),
        "g_h": g_h.tolist(),
        "g_axial": float(np.dot([0, 0, -9.81], u_w)),  # world gravity · cylinder axis
        "g_dot_u_hand": float(np.dot(g_h, u_h)),
        "e_signed_mm": (e_h * 1e3).tolist(),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "bilateral": bool(surf.get("bilateral")),
        "Fn_L": float(pack["Fn_L"]),
        "Fn_R": float(pack["Fn_R"]),
        "v_rel_h": np.asarray(pack["v_rel_h"], float).tolist(),
        "v_rel_norm": float(np.linalg.norm(pack["v_rel_h"])),
        "w_rel_h": np.asarray(pack["w_rel_h"], float).tolist(),
        "obj_z": float(m["obj_z"]),
        "ground_drop": bool(latched(sim)),
        "cyl_half": float(CYL_HALF),
        "cyl_r": float(CYL_R),
    }


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


def save_stage_snap(env, path: Path, meta: dict):
    blob = {"capture": take_capture(env), "ref": meta.get("ref"), "meta": meta, "audit": audit(env.sim)}
    path.write_bytes(pickle.dumps(blob))
    (path.with_suffix(".json")).write_text(json.dumps(jsafe(blob["audit"] | {"meta": meta}), indent=2), encoding="utf-8")
    return blob


def render_vid(env, start_cap, actions, dest: Path, label: str, ref):
    restore_capture(env, start_cap)
    look = 0.5 * (
        np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.5, "azimuth": 140.0, "elevation": -18.0})
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames = []
    t0 = float(env.sim.data.time)

    def grab(tag):
        cam.lookat[:] = 0.5 * (
            np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        renderer.update_scene(env.sim.data, cam)
        a = audit(env.sim)
        img = np.asarray(renderer.render(), np.uint8).copy()
        frames.append(
            caption(
                img,
                f"{label} {tag} t={float(env.sim.data.time)-t0:.2f} "
                f"s={a['contact_s_mean']} θ={a['e_theta_deg']:.1f} e={a['e_pos_mm']:.1f} {a['grasp_kind']}",
            )
        )

    grab("START")
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == len(actions) - 1:
            grab("RUN")
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_frames_mp4(frames, dest, fps=25)


# ---------- Phase I: axial slide to tail ----------
def phase1_tail_slide(env, post, book) -> dict:
    """Search wrist+unload schedules that migrate contact_s toward nearer cylinder end."""
    restore_capture(env, post["capture"])
    a0 = audit(env.sim)
    near = a0["near_end"] or "plus"
    # Intended tail = nearer geometric end
    tail_sign = +1.0 if near == "plus" else -1.0
    s0 = a0["contact_s_mean"]
    print(
        f"  Phase I geometry: s={s0:.4f} near_end={near} dist_nearest={a0['dist_nearest_end_m']:.4f} "
        f"kind={a0['grasp_kind']} opp_sw={a0['opposing_sidewall']} g_axial={a0['g_axial']:.2f}",
        flush=True,
    )
    # Object motion that INCREASES s toward +end: object slides along -u (see geometry notes).
    # Empirically try both gravity signs / wrist axes.
    candidates = []
    unload_opts = (-0.48, float(grip_action(-5.0)))
    # Bounded menu: cover gravity/wrist signs without repeating Plan A e_pos sweeps
    for wx, wy in (
        (0.0, 3.0),
        (0.0, -1.0),
        (1.12, 2.0),
        (1.12, -1.0),
        (-1.12, 2.0),
        (1.12, 0.0),
        (-1.12, 0.0),
        (0.0, 1.0),
    ):
        for tilt_s in (0.35, 0.55):
            for slip_s in (0.45, 0.70):
                for unload in unload_opts:
                    for retighten_s in (0.20,):
                        a_tilt = wrist(wx, wy, 1.0)
                        a_slip = wrist(0.0, 0.0, unload)
                        a_brake = wrist(0.0, 0.0, 1.0)
                        if not all(legal_ok(book, a) for a in (a_tilt, a_slip, a_brake)):
                            continue
                        acts = (
                            [a_tilt] * max(1, int(round(tilt_s / DT)))
                            + [a_slip] * max(1, int(round(slip_s / DT)))
                            + [a_brake] * max(1, int(round(retighten_s / DT)))
                        )
                        restore_capture(env, post["capture"])
                        lost = 0
                        drop = False
                        for a in acts:
                            step_legal(env.sim, a)
                            failed, lost = recovery_failed(env.sim, post["ref"], lost)
                            if failed or latched(env.sim):
                                drop = True
                                break
                        a1 = audit(env.sim)
                        if drop or not a1["bilateral"]:
                            continue
                        s1 = a1["contact_s_mean"]
                        if s0 is None or s1 is None:
                            continue
                        ds = float(s1 - s0)
                        # Toward intended tail: for plus, ds>0; for minus, ds<0
                        toward_tail = (ds * tail_sign) > 0
                        d_near = float(a0["dist_nearest_end_m"] - a1["dist_nearest_end_m"])
                        # True sliding: |ds| large and not just rigid (check |rh| change or v)
                        abs_s1 = abs(s1)
                        tail_band = TAIL_S_MIN <= abs_s1 <= TAIL_S_MAX
                        # Prefer sidewall near tail, reject pure end_face pinch
                        not_face = a1["grasp_kind"] != "end_face_pinch" and int(a1.get("n_end_face") or 0) <= 2
                        ok = (
                            toward_tail
                            and abs(ds) >= DS_TAIL_MIN
                            and d_near >= 0.002
                            and a1["bilateral"]
                            and not_face
                        )
                        score = (
                            (20 if ok else 0)
                            + 1000.0 * abs(ds) * (1 if toward_tail else -1)
                            + 500.0 * d_near
                            + (10 if tail_band else 0)
                            + (25 if a1["opposing_sidewall"] else 0)
                            - (15 if a1["grasp_kind"] in ("edge_only", "end_face_pinch") else 0)
                            - 0.2 * float(a1["e_theta_deg"])
                        )
                        # Prefer opposing sidewall in-band over edge-at-face even if ds larger
                        if a1["opposing_sidewall"] and tail_band:
                            score += 40
                        candidates.append(
                            {
                                "tag": f"tilt{tilt_s}_slip{slip_s}_wx{wx}_wy{wy}_g{unload}",
                                "actions": acts,
                                "s0": s0,
                                "s1": s1,
                                "ds": ds,
                                "d_dist_nearest": d_near,
                                "toward_tail": toward_tail,
                                "tail_band": tail_band,
                                "a0": a0,
                                "a1": a1,
                                "ok": ok,
                                "score": score,
                                "drop": drop,
                            }
                        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    ok_cands = [c for c in candidates if c["ok"]]
    print(f"  Phase I screened={len(candidates)} ok_tail={len(ok_cands)}", flush=True)
    if ok_cands:
        for c in ok_cands[:5]:
            print(
                f"    OK {c['tag']}: s {c['s0']:.4f}→{c['s1']:.4f} ds={c['ds']*1e3:.1f}mm "
                f"d_near={c['d_dist_nearest']*1e3:.1f}mm kind={c['a1']['grasp_kind']} "
                f"opp={c['a1']['opposing_sidewall']}",
                flush=True,
            )
    else:
        for c in candidates[:5]:
            print(
                f"    best-fail {c['tag']}: s {c['s0']:.4f}→{c['s1']:.4f} ds={c['ds']*1e3:.1f}mm "
                f"toward={c['toward_tail']} kind={c['a1']['grasp_kind']}",
                flush=True,
            )
    best = ok_cands[0] if ok_cands else None
    return {
        "phase": "I_tail_slide",
        "intended_tail": near,
        "tail_sign": tail_sign,
        "geometry0": a0,
        "n_screened": len(candidates),
        "n_ok": len(ok_cands),
        "success": best is not None,
        "best": None
        if best is None
        else {k: v for k, v in best.items() if k != "actions"},
        "best_actions": None if best is None else best["actions"],
        "top_fail": None
        if ok_cands
        else [{k: v for k, v in c.items() if k not in ("actions", "a0", "a1")} | {"a1_kind": c["a1"]["grasp_kind"], "a1_s": c["a1"]["contact_s_mean"]} for c in candidates[:3]],
    }


# ---------- Phase II: relative pivot at tail ----------
def phase2_pivot(env, stage_blob, book, ref) -> dict:
    """From tail grasp: unload → wrist pulse → brake → return hand → secure.

    Genuineness is judged AFTER return: cylinder axis in the hand must change while
    residual hand rotation is small (rules out rigid co-rotation).
    """
    restore_capture(env, stage_blob["capture"])
    a0 = audit(env.sim)
    u0 = np.asarray(a0["u_hand"], float)
    print(
        f"  Phase II start: s={a0['contact_s_mean']:.4f} kind={a0['grasp_kind']} "
        f"opp={a0['opposing_sidewall']} θ={a0['e_theta_deg']:.2f}",
        flush=True,
    )
    candidates = []
    n_tried = 0
    n_drop = 0
    # (wx, wy, dur, unload, return_frac, v_hand_xyz or None)
    menu = []
    wrist_dirs = (
        (0.0, 3.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (1.12, 2.0),
        (-1.12, 2.0),
        (1.12, 0.0),
        (-1.12, 0.0),
    )
    # Small hand-frame translations that can load the tail as a fulcrum
    v_opts = (None, (0.15, 0.0, 0.0), (-0.15, 0.0, 0.0), (0.0, 0.15, 0.0))
    for wx, wy in wrist_dirs:
        for dur in (0.30, 0.50, 0.80):
            for unload in (-0.48, float(grip_action(-5.0))):
                for ret_frac in (0.8, 1.1):
                    for vh in v_opts:
                        menu.append((wx, wy, dur, unload, ret_frac, vh))
    for wx, wy, dur, unload, ret_frac, vh in menu:
        a_piv = wrist(wx, wy, unload, v_hand=vh)
        a_hold = wrist(0.0, 0.0, unload)  # brief settle under unload
        a_ret = wrist(-wx, -wy, unload, v_hand=None if vh is None else (-vh[0], -vh[1], -vh[2]))
        a_brake = wrist(0.0, 0.0, 1.0)
        if not all(legal_ok(book, a) for a in (a_piv, a_hold, a_ret, a_brake)):
            continue
        n_tried += 1
        n_piv = max(1, int(round(dur / DT)))
        n_ret = max(1, int(round(dur * float(ret_frac) / DT)))
        acts = (
            [a_piv] * n_piv
            + [a_hold] * 5
            + [a_ret] * n_ret
            + [a_brake] * 12
        )
        restore_capture(env, stage_blob["capture"])
        R_h0 = hand_R(env.sim).copy()
        peak_w_rel = 0.0
        peak_v_rel = 0.0
        lost = 0
        drop = False
        for a in acts:
            step_legal(env.sim, a)
            pack = physical_pack(env.sim)
            peak_w_rel = max(peak_w_rel, float(np.linalg.norm(pack["w_rel_h"])))
            peak_v_rel = max(peak_v_rel, float(np.linalg.norm(pack["v_rel_h"])))
            failed, lost = recovery_failed(env.sim, ref, lost)
            if failed or latched(env.sim):
                drop = True
                break
        a1 = audit(env.sim)
        if drop:
            n_drop += 1
            continue
        if not a1["bilateral"]:
            n_drop += 1
            continue
        u1 = np.asarray(a1["u_hand"], float)
        d_u = float(np.degrees(np.arccos(float(np.clip(abs(np.dot(u0, u1)), 0.0, 1.0)))))
        Rh1 = hand_R(env.sim)
        err = R_h0.T @ Rh1
        c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
        hand_rot_residual = float(np.degrees(np.arccos(c)))
        s1 = a1["contact_s_mean"]
        near_tail = s1 is not None and abs(s1) >= TAIL_S_MIN - 0.006
        d_theta = float(a0["e_theta_deg"] - a1["e_theta_deg"])
        # After return: relative axis change must dominate residual hand rotation
        genuine = (
            d_u >= DS_REL_PIVOT_MIN_DEG
            and near_tail
            and peak_w_rel >= 0.12
            and hand_rot_residual <= 25.0
            and d_u >= hand_rot_residual + 3.0
            and d_theta >= 2.0  # meaningful orientation-error reduction after return
        )
        vtag = "v0" if vh is None else f"v{vh[0]:.2f}_{vh[1]:.2f}"
        score = (
            (40 if genuine else 0)
            + d_u
            + 1.5 * d_theta
            + 4.0 * peak_w_rel
            - 0.4 * hand_rot_residual
            - 0.15 * abs(float(a1["e_pos_mm"]) - float(a0["e_pos_mm"]))
            + (8 if a1.get("opposing_sidewall") else 0)
        )
        candidates.append(
            {
                "tag": f"piv_wx{wx}_wy{wy}_d{dur}_g{unload}_r{ret_frac}_{vtag}",
                "actions": acts,
                "d_u_hand_deg": d_u,
                "hand_rot_deg": hand_rot_residual,
                "d_theta": d_theta,
                "peak_w_rel": peak_w_rel,
                "peak_v_rel": peak_v_rel,
                "s0": a0["contact_s_mean"],
                "s1": s1,
                "genuine_relative_pivot": genuine,
                "rigid_dominated": hand_rot_residual > d_u + 2.0 or d_u < DS_REL_PIVOT_MIN_DEG,
                "a1_kind": a1["grasp_kind"],
                "a1_opp": int(a1.get("opposing_sidewall") or 0),
                "a1_theta": a1["e_theta_deg"],
                "a1_e_pos": a1["e_pos_mm"],
                "ok": genuine and a1["bilateral"],
                "score": score,
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    ok_cands = [c for c in candidates if c["ok"]]
    print(
        f"  Phase II tried={n_tried} drop/lost={n_drop} survived={len(candidates)} "
        f"genuine_pivot={len(ok_cands)}",
        flush=True,
    )
    if ok_cands:
        c = ok_cands[0]
        print(
            f"    best d_u={c['d_u_hand_deg']:.1f}° hand_res={c['hand_rot_deg']:.1f}° "
            f"peak_w={c['peak_w_rel']:.2f} dθ={c['d_theta']:.1f} s={c['s1']:.4f} "
            f"rigid_dom={c['rigid_dominated']}",
            flush=True,
        )
    else:
        for c in candidates[:5]:
            print(
                f"    near-miss {c['tag']}: d_u={c['d_u_hand_deg']:.1f} "
                f"hand_res={c['hand_rot_deg']:.1f} w={c['peak_w_rel']:.2f} "
                f"dθ={c['d_theta']:.1f} s1={c['s1']} genuine={c['genuine_relative_pivot']}",
                flush=True,
            )
    best = ok_cands[0] if ok_cands else None
    return {
        "phase": "II_pivot",
        "geometry0": a0,
        "n_tried": n_tried,
        "n_drop": n_drop,
        "n_screened": len(candidates),
        "n_ok": len(ok_cands),
        "success": best is not None,
        "best": None if best is None else {k: v for k, v in best.items() if k != "actions"},
        "best_actions": None if best is None else best["actions"],
        "near_misses": [{k: v for k, v in c.items() if k != "actions"} for c in candidates[:5]],
        "near_miss_actions": None if not candidates else candidates[0]["actions"],
    }


# ---------- Phase III: invert / reorient gravity ----------
def phase3_invert(env, stage_blob, book, ref) -> dict:
    restore_capture(env, stage_blob["capture"])
    a0 = audit(env.sim)
    g0 = float(a0["g_dot_u_hand"])
    candidates = []
    for wx, wy, dur in (
        (0.0, 3.0, 0.60),
        (0.0, 3.0, 1.00),
        (0.0, -1.0, 0.80),
        (0.0, -1.0, 1.20),
        (1.12, 2.0, 0.80),
        (-1.12, 2.0, 0.80),
        (0.0, 3.0, 1.40),
        (1.12, 0.0, 1.00),
        (-1.12, 0.0, 1.00),
        (0.0, 1.0, 1.60),
    ):
        a_rot = wrist(wx, wy, 1.0)
        a_brake = wrist(0.0, 0.0, 1.0)
        if not legal_ok(book, a_rot):
            continue
        acts = [a_rot] * max(1, int(round(dur / DT))) + [a_brake] * 8
        restore_capture(env, stage_blob["capture"])
        R0 = hand_R(env.sim).copy()
        lost = 0
        drop = False
        lost_contact = False
        for a in acts:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, ref, lost)
            if not audit(env.sim)["bilateral"]:
                lost_contact = True
            if failed or latched(env.sim):
                drop = True
                break
        a1 = audit(env.sim)
        if drop:
            continue
        Rh1 = hand_R(env.sim)
        err = R0.T @ Rh1
        c = float(np.clip(0.5 * (np.trace(err) - 1.0), -1.0, 1.0))
        hand_rot = float(np.degrees(np.arccos(c)))
        g1 = float(a1["g_dot_u_hand"])
        # Effective reorientation: gravity projection along cylinder (hand) flips sign or |Δg| large
        flipped = (g0 * g1) < 0.0 and abs(g1) > 2.0
        boosted = abs(g1) - abs(g0) >= 3.0
        ok = a1["bilateral"] and not lost_contact and hand_rot >= 25.0 and (flipped or boosted)
        score = (40 if ok else 0) + hand_rot + 2.0 * abs(g1) + (20 if flipped else 0)
        candidates.append(
            {
                "tag": f"inv_wx{wx}_wy{wy}_d{dur}",
                "actions": acts,
                "hand_rot_deg": hand_rot,
                "g_dot_u_0": g0,
                "g_dot_u_1": g1,
                "flipped": flipped,
                "boosted": boosted,
                "lost_contact_mid": lost_contact,
                "a1": a1,
                "ok": ok,
                "score": score,
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    ok_cands = [c for c in candidates if c["ok"]]
    print(f"  Phase III screened={len(candidates)} stable_reorient={len(ok_cands)}", flush=True)
    if ok_cands:
        c = ok_cands[0]
        print(
            f"    best rot={c['hand_rot_deg']:.1f}° g·u {c['g_dot_u_0']:.2f}→{c['g_dot_u_1']:.2f} "
            f"flip={c['flipped']} boost={c['boosted']}",
            flush=True,
        )
    best = ok_cands[0] if ok_cands else None
    return {
        "phase": "III_invert",
        "geometry0": a0,
        "n_screened": len(candidates),
        "n_ok": len(ok_cands),
        "success": best is not None,
        "best": None if best is None else {k: v for k, v in best.items() if k not in ("actions", "a1")},
        "best_actions": None if best is None else best["actions"],
        "best_a1": None if best is None else best["a1"],
    }


# ---------- Phase IV: final slide + V HANDOFF ----------
def phase4_final_slide(env, stage_blob, book, ref) -> dict:
    restore_capture(env, stage_blob["capture"])
    a0 = audit(env.sim)
    candidates = []
    for wx, wy, tilt_s, slip_s, unload in (
        (0.0, 3.0, 0.40, 0.50, float(grip_action(-5.0))),
        (0.0, -1.0, 0.40, 0.50, float(grip_action(-5.0))),
        (0.0, 3.0, 0.50, 0.70, -0.48),
        (0.0, 1.0, 0.30, 0.60, -0.48),
        (1.12, 2.0, 0.40, 0.50, -0.48),
        (0.0, 3.0, 0.60, 0.40, 1.0),
        (0.0, -1.0, 0.60, 0.70, float(grip_action(-5.0))),
    ):
        a_tilt = wrist(wx, wy, 1.0)
        a_slip = wrist(0.0, 0.0, unload)
        a_brake = wrist(0.0, 0.0, 1.0)
        if not all(legal_ok(book, a) for a in (a_tilt, a_slip, a_brake)):
            continue
        acts = (
            [a_tilt] * max(1, int(round(tilt_s / DT)))
            + [a_slip] * max(1, int(round(slip_s / DT)))
            + [a_brake] * 12
        )
        restore_capture(env, stage_blob["capture"])
        lost = 0
        drop = False
        for a in acts:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, ref, lost)
            if failed or latched(env.sim):
                drop = True
                break
        a1 = audit(env.sim)
        if drop or not a1["bilateral"]:
            continue
        d_pos = float(a0["e_pos_mm"] - a1["e_pos_mm"])
        score = d_pos - 0.3 * float(a1["e_theta_deg"]) + (5 if a1["bilateral"] else -20)
        candidates.append(
            {
                "tag": f"fin_wx{wx}_wy{wy}_t{tilt_s}_s{slip_s}",
                "actions": acts,
                "a1": a1,
                "d_e_pos": d_pos,
                "ok": a1["bilateral"] and a1["e_pos_mm"] < a0["e_pos_mm"] + 1.0,
                "score": score,
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # Full SUCCESS_NEW on top few
    full_rows = []
    best_new = None
    for c in candidates[:8]:
        restore_capture(env, stage_blob["capture"])
        lost = 0
        drop = False
        for a in c["actions"]:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, ref, lost)
            if failed:
                drop = True
                break
        if drop:
            continue
        meta = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
        win = meta.get("window") or {}
        row = {
            "tag": c["tag"],
            "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
            "reason": meta.get("reason"),
            "pre_theta": meta.get("pre_theta"),
            "pre_pos_mm": meta.get("pre_pos_mm"),
            "longest_both_ok_s": win.get("longest_both_ok_s"),
            "actions": c["actions"],
            "pre_audit": audit(env.sim) if False else c["a1"],
        }
        # re-audit pre-handoff from stored a1
        full_rows.append(row)
        print(
            f"    FULL {c['tag']}: NEW={row['SUCCESS_NEW']} S12={row['SUCCESS_12S']} "
            f"θ={row['pre_theta']} pos={row['pre_pos_mm']} hold={row['longest_both_ok_s']}",
            flush=True,
        )
        if row["SUCCESS_NEW"] and best_new is None:
            best_new = row
    return {
        "phase": "IV_final_slide",
        "geometry0": a0,
        "n_screened": len(candidates),
        "full_rows": [{k: v for k, v in r.items() if k != "actions"} for r in full_rows],
        "success_new": best_new is not None,
        "best_new_actions": None if best_new is None else best_new["actions"],
        "best_screen": None
        if not candidates
        else {k: v for k, v in candidates[0].items() if k not in ("actions", "a1")},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "stages").mkdir(parents=True, exist_ok=True)
    table, book48 = load_vocab()
    book = load_codebook()
    env = make_env()
    post = ensure_post(env, table, book48)
    root = pickle.loads(ROOT_PKL.read_bytes())
    t0 = time.time()

    print("=== Tail→Pivot→Invert→Slide staged feasibility hard_910220 ===", flush=True)

    # Geometry report
    restore_capture(env, post["capture"])
    geom0 = audit(env.sim)
    (OUT / "geometry_post_b1.json").write_text(json.dumps(jsafe(geom0), indent=2), encoding="utf-8")
    print(
        f"Post-B1: s={geom0['contact_s_mean']} near={geom0['near_end']} "
        f"kind={geom0['grasp_kind']} L/R s=({geom0['contact_s_L']},{geom0['contact_s_R']}) "
        f"θ={geom0['e_theta_deg']:.2f} e_pos={geom0['e_pos_mm']:.2f}",
        flush=True,
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ic_id": IC,
        "geometry_post_b1": geom0,
        "phases": {},
        "chained_SUCCESS_NEW": False,
        "videos": {},
    }

    # ----- Phase I -----
    print("\n--- Phase I: slide to tail ---", flush=True)
    p1 = phase1_tail_slide(env, post, book)
    report["phases"]["I"] = {k: v for k, v in p1.items() if k != "best_actions"}
    if not p1["success"]:
        report["bottleneck"] = {
            "failed_phase": "I_tail_slide",
            "reason": "Could not establish genuine axial migration to tail sidewall grasp",
            "top_fail": p1.get("top_fail"),
        }
        # still render best-effort fail video if any actions from top fail candidate — skip
        (OUT / "summary.json").write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")
        _write_report(report)
        print(json.dumps(jsafe(report["bottleneck"]), indent=2), flush=True)
        print(f"Wrote {OUT} (stopped at Phase I)", flush=True)
        return 0

    # Save Phase I snapshot via uninterrupted execution
    restore_capture(env, post["capture"])
    for a in p1["best_actions"]:
        step_legal(env.sim, a)
    stage1 = save_stage_snap(
        env,
        OUT / "stages" / "phase1_tail.pkl",
        {"ref": post["ref"], "phase": "I", "tag": p1["best"]["tag"]},
    )
    report["videos"]["phase1"] = render_vid(
        env, post["capture"], p1["best_actions"], OUT / "videos" / "phase1_tail_slide.mp4", "P1_TAIL", post["ref"]
    )
    print(f"  Phase I SUCCESS s→{p1['best']['s1']:.4f}", flush=True)

    # ----- Phase II -----
    print("\n--- Phase II: pivot at tail ---", flush=True)
    p2 = phase2_pivot(env, stage1, book, post["ref"])
    report["phases"]["II"] = {k: v for k, v in p2.items() if k not in ("best_actions",)}
    if not p2["success"]:
        report["bottleneck"] = {
            "failed_phase": "II_pivot",
            "reason": "No genuine relative pivot retained near tail after hand return (co-rotation or contact loss)",
            "n_tried": p2.get("n_tried"),
            "n_drop": p2.get("n_drop"),
            "n_screened": p2["n_screened"],
            "near_misses": p2.get("near_misses"),
        }
        if p2.get("near_miss_actions"):
            report["videos"]["phase2_near_miss"] = render_vid(
                env,
                stage1["capture"],
                p2["near_miss_actions"],
                OUT / "videos" / "phase2_pivot_near_miss.mp4",
                "P2_NEAR",
                post["ref"],
            )
        (OUT / "summary.json").write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")
        _write_report(report)
        print(
            json.dumps(
                jsafe({k: report["bottleneck"][k] for k in ("failed_phase", "reason", "n_tried", "n_drop", "n_screened")}),
                indent=2,
            ),
            flush=True,
        )
        print(f"Wrote {OUT} (stopped at Phase II)", flush=True)
        return 0

    restore_capture(env, stage1["capture"])
    for a in p2["best_actions"]:
        step_legal(env.sim, a)
    stage2 = save_stage_snap(
        env,
        OUT / "stages" / "phase2_pivot.pkl",
        {"ref": post["ref"], "phase": "II", "tag": p2["best"]["tag"]},
    )
    report["videos"]["phase2"] = render_vid(
        env, stage1["capture"], p2["best_actions"], OUT / "videos" / "phase2_pivot.mp4", "P2_PIVOT", post["ref"]
    )
    print(f"  Phase II SUCCESS d_u={p2['best']['d_u_hand_deg']:.1f}°", flush=True)

    # ----- Phase III -----
    print("\n--- Phase III: invert / reorient gravity ---", flush=True)
    p3 = phase3_invert(env, stage2, book, post["ref"])
    report["phases"]["III"] = {k: v for k, v in p3.items() if k not in ("best_actions", "best_a1")}
    if not p3["success"]:
        report["bottleneck"] = {
            "failed_phase": "III_invert",
            "reason": "No stable hand reorientation that flips/boosts gravity along cylinder with contact retained",
            "n_screened": p3["n_screened"],
        }
        (OUT / "summary.json").write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")
        _write_report(report)
        print(json.dumps(jsafe(report["bottleneck"]), indent=2), flush=True)
        print(f"Wrote {OUT} (stopped at Phase III)", flush=True)
        return 0

    restore_capture(env, stage2["capture"])
    for a in p3["best_actions"]:
        step_legal(env.sim, a)
    stage3 = save_stage_snap(
        env,
        OUT / "stages" / "phase3_invert.pkl",
        {"ref": post["ref"], "phase": "III", "tag": p3["best"]["tag"]},
    )
    report["videos"]["phase3"] = render_vid(
        env, stage2["capture"], p3["best_actions"], OUT / "videos" / "phase3_invert.mp4", "P3_INVERT", post["ref"]
    )
    print(f"  Phase III SUCCESS rot={p3['best']['hand_rot_deg']:.1f}°", flush=True)

    # ----- Phase IV + V -----
    print("\n--- Phase IV/V: final slide + HANDOFF ---", flush=True)
    p4 = phase4_final_slide(env, stage3, book, post["ref"])
    report["phases"]["IV"] = {k: v for k, v in p4.items() if k != "best_new_actions"}

    # Chain full uninterrupted trajectory from post-B1
    if p4.get("success_new") and p4.get("best_new_actions"):
        chain = list(p1["best_actions"]) + list(p2["best_actions"]) + list(p3["best_actions"]) + list(p4["best_new_actions"])
        # Verify chain == stage execution by full SUCCESS_NEW from post
        restore_capture(env, post["capture"])
        lost = 0
        for a in chain:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, post["ref"], lost)
            if failed:
                report["bottleneck"] = {"failed_phase": "chain", "reason": "DROP during uninterrupted chain"}
                break
        else:
            meta = evaluate_handoff_from_state(env.sim, post["ref"], stabilize_n=0)
            report["chained_handoff"] = {
                "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(meta.get("reason") == "SUCCESS_12S"),
                "reason": meta.get("reason"),
                "pre_theta": meta.get("pre_theta"),
                "pre_pos_mm": meta.get("pre_pos_mm"),
                "longest_both_ok_s": (meta.get("window") or {}).get("longest_both_ok_s"),
            }
            if meta.get("SUCCESS_NEW"):
                arr = np.stack(chain, axis=0)
                tokens, _, _, _ = legalize(arr, book)
                full_toks = list(B1_TOKENS) + [int(t) for t in tokens]
                replay = exact_replay_3x(env, root, table, book48, full_toks, branch="B2", prefix_len=1)
                report["chained_SUCCESS_NEW"] = bool(replay.get("ok"))
                report["exact_replay"] = replay
                report["n_chain_actions"] = len(chain)
                report["videos"]["full_chain"] = render_vid(
                    env, post["capture"], chain, OUT / "videos" / "full_chain.mp4", "CHAIN", post["ref"]
                )
                (OUT / "SUCCESS_NEW_expert.json").write_text(
                    json.dumps(
                        jsafe(
                            {
                                "ic_id": IC,
                                "family": "B2_tail_pivot_invert",
                                "tokens": full_toks,
                                "exact_replay": replay,
                                "phases": ["I", "II", "III", "IV"],
                            }
                        ),
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                report["bottleneck"] = {
                    "failed_phase": "V_HANDOFF",
                    "reason": "Chained trajectory did not achieve SUCCESS_NEW",
                    "handoff": report.get("chained_handoff"),
                }
    else:
        report["bottleneck"] = {
            "failed_phase": "IV_final_slide",
            "reason": "Final slide from inverted grasp did not yield SUCCESS_NEW",
            "full_rows": p4.get("full_rows"),
        }
        if p4.get("best_screen"):
            report["videos"]["phase4_best_screen"] = render_vid(
                env,
                stage3["capture"],
                # re-get actions from a re-run of best screen tag — skip if missing
                p4.get("best_new_actions") or [],
                OUT / "videos" / "phase4_screen.mp4",
                "P4",
                post["ref"],
            ) if p4.get("best_new_actions") else None

    report["wall_s"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")
    _write_report(report)
    print(json.dumps(jsafe({k: report.get(k) for k in ("chained_SUCCESS_NEW", "bottleneck", "wall_s", "videos")}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


def _write_report(report: dict):
    lines = [
        "# Plan B Tail → Pivot → Invert → Slide — hard_910220",
        "",
        f"Created UTC: {report.get('created_utc')}",
        f"Wall: {report.get('wall_s')}s",
        "",
        "## Post-B1 geometry",
        "",
        f"```json\n{json.dumps(jsafe(report.get('geometry_post_b1')), indent=2)[:2500]}\n```",
        "",
        "## Phase results",
        "",
    ]
    for key in ("I", "II", "III", "IV"):
        ph = (report.get("phases") or {}).get(key)
        if not ph:
            continue
        lines += [f"### Phase {key}", "", f"- success: {ph.get('success', ph.get('success_new'))}", ""]
        if ph.get("best"):
            lines += [f"```json\n{json.dumps(jsafe(ph['best']), indent=2)[:1500]}\n```", ""]
        if ph.get("full_rows"):
            lines += [f"Full rows: {json.dumps(jsafe(ph['full_rows'])[:3], indent=2)[:1500]}", ""]
    if report.get("bottleneck"):
        lines += ["## Bottleneck", "", f"```json\n{json.dumps(jsafe(report['bottleneck']), indent=2)[:2500]}\n```", ""]
    lines += [
        f"## Chained SUCCESS_NEW: {report.get('chained_SUCCESS_NEW')}",
        "",
        f"Videos: {report.get('videos')}",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
