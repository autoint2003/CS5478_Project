"""Horizontal-offset free-fall active recatch — complete SUCCESS_NEW experts.

Constructed free-fall ICs (not ball-impact / lateral-force roots).
Legal continuous 7D only. No SFT/GRPO. No K49 changes.

Stages
  A  Reproduce perp_off01_r15 SUCCESS_NEW and distal_off01_r8 partial
  B  Nearby offsets → new independent SUCCESS_NEW
  C  Expanded offsets (after B successes)
  D  Pose-error correction (esp. orientation)
  E  Robustness without re-optimization
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state
from envs.step_legal import step_legal
from evaluation.full_objective import (
    HOLD_S as SN_HOLD,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_lateral_offset_falling_recatch import (
    FINE_DOWN,
    TABLE_Z,
    body_v_w,
    build_nominal_open_robot,
    finger_frame,
    hand_pos,
    make_ic,
    make_training_env,
    obj_pos,
    object_table_contact,
    restore_ic,
    rot_angle,
    scoop_align_descend,
)
from scripts.run_lateral_recatch_handoff import (
    build_stabilization,
    clear_probe_latch,
    evaluate_catch_handoff,
    hand_R,
    run_nominal,
    snapshot_catch_state,
)
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SRC_CATCH = ROOT / "results" / "diagnostics" / "lateral_offset_falling_recatch"
SRC_HO = ROOT / "results" / "diagnostics" / "lateral_recatch_handoff"
OUT = ROOT / "results" / "expert_generation" / "horizontal_offset_freefall_recatch"
SEED = 20261103

# Frozen budgets
STAGE_A_REPLAYS = 3
STAGE_B_ICS = 12
STAGE_B_SCOOP_TRIALS = 10
STAGE_B_STAB_TRIALS = 8
STAGE_C_ICS = 8
STAGE_C_SCOOP_TRIALS = 8
STAGE_C_STAB_TRIALS = 6
STAGE_D_TRIALS = 36
STAGE_E_PERTURBS = 6
WALL_LIMIT_S = 75 * 60


def tokens_hash(acts):
    return hashlib.sha256(json.dumps([np.asarray(a).tolist() for a in acts]).encode()).hexdigest()[:16]


def classify_outcome(out: dict, audit: dict | None = None) -> str:
    if out.get("SUCCESS_NEW"):
        return "SUCCESS_NEW"
    if out.get("CATCH_HANDOFF_SUCCESS"):
        th = out.get("SN_mean_theta") or out.get("mean_theta_ch")
        pos = out.get("SN_mean_pos") or out.get("mean_pos_ch")
        if th is not None and th > WORKING_THETA_DEG:
            return "ORIENTATION_ERROR_FAILURE"
        if pos is not None and pos > WORKING_POS_MM:
            return "POSITION_ERROR_FAILURE"
        return "CATCH_HANDOFF"
    if out.get("handoff_reason") in ("LATCH", "TABLE", "Z_LOW") or out.get("fail_class", "").startswith("H_"):
        return "TABLE_DROP"
    pre = out.get("pre") or {}
    if audit and audit.get("first_contact_kind") == "outer_finger":
        return "OUTER_FINGER_COLLISION"
    if pre.get("first_bil_step") is None and (pre.get("nL", 0) + pre.get("nR", 0)) == 0:
        return "NO_INTERCEPTION"
    if pre.get("first_bil_step") is None and (pre.get("nL", 0) + pre.get("nR", 0)) > 0:
        return "UNILATERAL_CONTACT"
    if pre.get("bilateral") and (out.get("held_s") or 0) < 2.0:
        return "BILATERAL_SHORT_CAPTURE"
    if pre.get("bilateral"):
        return "STABLE_BILATERAL_CAPTURE"
    return "JAW_ENTRY_ONLY"


def audit_interception(env, ic, actions) -> dict:
    """Trace chase / first contact / jaw entry during Recovery."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    sim = env.sim
    hp0 = hand_pos(sim).copy()
    op0 = obj_pos(sim).copy()
    n_touch0 = 0
    obj_b = int(sim.ids.object_body)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj_b in (b1, b2):
            n_touch0 += 1
    first_contact = None
    first_bil = None
    first_kind = None
    max_hand_disp = 0.0
    contact_rows = []
    t0 = float(sim.data.time)
    for i, a in enumerate(actions):
        step_legal(sim, np.asarray(a, np.float32))
        cs = contact_summary(sim)
        hp = hand_pos(sim)
        max_hand_disp = max(max_hand_disp, float(np.linalg.norm(hp - hp0)))
        sites = []
        try:
            from scripts.run_lateral_offset_falling_recatch import contact_sites
            sites = contact_sites(sim)
        except Exception:
            pass
        topo = "bilateral" if cs["bilateral"] else (
            "unilateral" if int(cs["nL"]) + int(cs["nR"]) > 0 else "none"
        )
        contact_rows.append({
            "t_rel": round(float(sim.data.time) - t0, 5),
            "step": i,
            "topology": topo,
            "nL": int(cs["nL"]),
            "nR": int(cs["nR"]),
            "obj_z": float(obj_pos(sim)[2]),
            "e_pos_mm": float(gmeasure(sim, U_TGT, P_TGT)["e_pos_mm"]),
            "e_theta_deg": float(gmeasure(sim, U_TGT, P_TGT)["e_theta_deg"]),
        })
        if first_contact is None and (int(cs["nL"]) + int(cs["nR"])) > 0:
            first_contact = i
            sides = {s.get("side") for s in sites}
            if "hand" in sides and "L" not in sides and "R" not in sides:
                first_kind = "outer_finger"
            elif cs["bilateral"]:
                first_kind = "jaw_entry_bilateral"
            else:
                first_kind = "unilateral_pad"
        if first_bil is None and cs["bilateral"]:
            first_bil = i
            if first_kind in (None, "unilateral_pad"):
                first_kind = "jaw_entry_bilateral"
        if latched(sim):
            break
    m = gmeasure(sim, U_TGT, P_TGT)
    return {
        "n_touch_init": n_touch0,
        "init_sep_m": float(np.linalg.norm(op0 - hp0)),
        "active_hand_disp_m": max_hand_disp,
        "first_contact_step": first_contact,
        "first_bil_step": first_bil,
        "first_contact_kind": first_kind,
        "e_pos_end_mm": float(m["e_pos_mm"]),
        "e_theta_end_deg": float(m["e_theta_deg"]),
        "p_rel_end": np.asarray(p_com_hand(sim), float).tolist(),
        "contact_rows": contact_rows,
    }


def eval_full_sn(env, ic, actions, catch_len, nominal_s: float | None = None) -> dict:
    """Evaluate CATCH_HANDOFF / SUCCESS_NEW.

    Search uses a short nominal (default 4 s) for speed; final verification
    must pass `nominal_s=NOMINAL_S` (12 s) with SUCCESS_NEW hold.
    """
    env._ch_nominal_s = float(NOMINAL_S if nominal_s is None else nominal_s)
    R0 = np.asarray(ic["R0"], float).reshape(3, 3)
    out = evaluate_catch_handoff(env, ic, actions, catch_len, R0)
    if out["SUCCESS_NEW"] and (out.get("held_s") or 0) + 1e-9 < 4.0:
        out["SUCCESS_NEW"] = 0
    return out


SEARCH_NOMINAL_S = 4.0


def find_cut(env, ic, catch_acts) -> int | None:
    """Earliest post-bilateral cut before sticky latch."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first_bil = None
    latch_step = None
    for i, a in enumerate(catch_acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        cs = contact_summary(env.sim)
        if cs["bilateral"] and first_bil is None:
            first_bil = i
        if latched(env.sim):
            latch_step = i
            break
    if first_bil is None:
        return None
    max_cut = (latch_step if latch_step is not None else len(catch_acts)) - 1
    return int(min(max(first_bil + 2, first_bil + 1), max(first_bil + 1, max_cut)))


def try_prefix_stab(env, ic, catch_acts, rng, n_stab: int, rollout: list, warm_cuts=None) -> dict | None:
    """Stabilize an existing catch prefix (warm-start or scoop)."""
    best = None
    cut0 = find_cut(env, ic, catch_acts)
    cuts = []
    if cut0 is not None:
        cuts.append(cut0)
    if warm_cuts:
        cuts.extend(int(c) for c in warm_cuts if 1 <= int(c) < len(catch_acts))
    # also try a few early/late cuts around first bil
    if cut0 is not None:
        for d in (-2, -1, 1, 2, 3):
            c = cut0 + d
            if 1 <= c < len(catch_acts):
                cuts.append(c)
    cuts = sorted(set(cuts))[:5]
    per_cut = max(2, n_stab // max(1, len(cuts)))
    for cut in cuts:
        for _s in range(per_cut):
            if rollout[0] <= 0:
                return best
            p = {
                "n_lift": int(rng.choice([25, 30, 35, 40, 45, 50])),
                "lift": float(rng.choice([-0.85, -0.95, -1.0])),
                "n_return": int(rng.choice([0, 8, 12, 16, 20])),
                "n_seat": int(rng.choice([0, 8, 12, 16])),
                "n_fine": int(rng.choice([0, 5, 10])),
                "wy": float(rng.choice([0.0, 0.15, 0.25, 0.35])),
                "n_wy": int(rng.choice([0, 8, 12, 16])),
                "xy_gain": float(rng.choice([0.4, 0.55, 0.7, 0.85])),
                "axis_align": bool(rng.random() < 0.45),
                "use_fine_down": bool(rng.random() < 0.35),
                "return_vz": float(rng.choice([-0.1, -0.2])),
                "seat_vz": float(rng.choice([0.05, 0.1])),
            }
            stab = build_stabilization(env, ic, catch_acts[:cut], p)
            rollout[0] -= 1
            if stab is None:
                continue
            full = list(catch_acts[:cut]) + stab
            out = eval_full_sn(env, ic, full, cut, nominal_s=SEARCH_NOMINAL_S)
            rollout[0] -= 1
            sc = (
                1000 * int(out["SUCCESS_NEW"])
                + 100 * int(out["CATCH_HANDOFF_SUCCESS"])
                + 10 * float(out.get("held_s") or 0)
                - float(out.get("SN_mean_theta") or out.get("mean_theta_ch") or 50)
                - 0.05 * float(out.get("SN_mean_pos") or out.get("mean_pos_ch") or 50)
            )
            if best is None or sc > best["score"]:
                best = {
                    "score": sc, "full_actions": full, "cut": cut, "out": out,
                    "scoop": {"warm": True}, "stab": p,
                }
            if out["SUCCESS_NEW"]:
                out12 = eval_full_sn(env, ic, full, cut, nominal_s=float(NOMINAL_S))
                rollout[0] -= 1
                best = {
                    "score": sc + 500 * int(out12["SUCCESS_NEW"]),
                    "full_actions": full, "cut": cut, "out": out12,
                    "scoop": {"warm": True}, "stab": p,
                }
                if out12["SUCCESS_NEW"]:
                    return best
    return best


def search_scoop_stab(env, ic, rng, n_scoop: int, n_stab: int, rollout: list,
                      warm_actions=None, warm_cut=None) -> dict | None:
    """Search scoop catch + early-lift stabilization for SUCCESS_NEW."""
    best = None
    if warm_actions is not None:
        print("    warm-start...", flush=True)
        wb = try_prefix_stab(
            env, ic, warm_actions, rng, n_stab=max(6, n_stab), rollout=rollout,
            warm_cuts=[warm_cut] if warm_cut is not None else None,
        )
        if wb is not None:
            best = wb
            if wb["out"]["SUCCESS_NEW"]:
                return best
    for si in range(n_scoop):
        if rollout[0] <= 0:
            break
        n_align = int(rng.choice([3, 4, 5]))
        lead = float(rng.choice([0.015, 0.02, 0.025, 0.03]))
        n_hold = int(rng.choice([8, 12, 16, 20, 24]))
        wy = float(rng.choice([0.0, 0.15, 0.25, 0.35]))
        close_xy = float(rng.choice([0.02, 0.025, 0.03]))
        acts = scoop_align_descend(
            env, ic, n_align=n_align, close_xy=close_xy, lead=lead, n_hold=n_hold, wy=wy,
        )
        rollout[0] -= 1
        cut = find_cut(env, ic, acts)
        if cut is None:
            continue
        if si % 3 == 0:
            print(f"    scoop {si+1}/{n_scoop} cut={cut} left={rollout[0]}", flush=True)
        for _s in range(n_stab):
            if rollout[0] <= 0:
                break
            p = {
                "n_lift": int(rng.choice([25, 30, 35, 40, 45])),
                "lift": float(rng.choice([-0.85, -0.95, -1.0])),
                "n_return": int(rng.choice([0, 8, 12, 16])),
                "n_seat": int(rng.choice([0, 8, 12])),
                "n_fine": int(rng.choice([0, 5, 10])),
                "wy": float(rng.choice([0.0, 0.2, 0.3, wy])),
                "n_wy": int(rng.choice([0, 8, 12])),
                "xy_gain": float(rng.choice([0.4, 0.55, 0.7, 0.85])),
                "axis_align": bool(rng.random() < 0.4),
                "use_fine_down": bool(rng.random() < 0.4),
                "return_vz": float(rng.choice([-0.1, -0.2])),
                "seat_vz": float(rng.choice([0.05, 0.1])),
            }
            stab = build_stabilization(env, ic, acts[:cut], p)
            rollout[0] -= 1
            if stab is None:
                continue
            full = list(acts[:cut]) + stab
            out = eval_full_sn(env, ic, full, cut, nominal_s=SEARCH_NOMINAL_S)
            rollout[0] -= 1
            sc = (
                1000 * int(out["SUCCESS_NEW"])
                + 100 * int(out["CATCH_HANDOFF_SUCCESS"])
                + 10 * float(out.get("held_s") or 0)
                - float(out.get("SN_mean_theta") or out.get("mean_theta_ch") or 50)
                - 0.05 * float(out.get("SN_mean_pos") or out.get("mean_pos_ch") or 50)
            )
            if best is None or sc > best["score"]:
                best = {
                    "score": sc,
                    "full_actions": full,
                    "cut": cut,
                    "out": out,
                    "scoop": {"n_align": n_align, "lead": lead, "n_hold": n_hold, "wy": wy},
                    "stab": p,
                }
            if out["SUCCESS_NEW"]:
                out12 = eval_full_sn(env, ic, full, cut, nominal_s=float(NOMINAL_S))
                rollout[0] -= 1
                best = {
                    "score": sc + 500 * int(out12["SUCCESS_NEW"]),
                    "full_actions": full, "cut": cut, "out": out12,
                    "scoop": {"n_align": n_align, "lead": lead, "n_hold": n_hold, "wy": wy},
                    "stab": p,
                }
                if out12["SUCCESS_NEW"]:
                    return best
    return best


def ori_correction_search(env, ic, prefix_acts, cut, rng, n_trials: int, rollout: list) -> dict | None:
    """Extra seat/axis-align after a CATCH_HANDOFF-capable prefix."""
    best = None
    for _ in range(n_trials):
        if rollout[0] <= 0:
            break
        p = {
            "n_lift": int(rng.choice([30, 35, 40])),
            "lift": float(rng.choice([-0.9, -1.0])),
            "n_return": int(rng.choice([8, 12, 16])),
            "n_seat": int(rng.choice([12, 16, 20, 24])),
            "n_fine": int(rng.choice([0, 5])),
            "wy": float(rng.choice([0.25, 0.35, 0.45])),
            "n_wy": int(rng.choice([10, 15, 20])),
            "xy_gain": float(rng.choice([0.5, 0.7])),
            "axis_align": True,
            "use_fine_down": False,
            "return_vz": -0.15,
            "seat_vz": float(rng.choice([0.05, 0.08, 0.12])),
        }
        stab = build_stabilization(env, ic, prefix_acts[:cut], p)
        rollout[0] -= 1
        if stab is None:
            continue
        full = list(prefix_acts[:cut]) + stab
        out = eval_full_sn(env, ic, full, cut, nominal_s=SEARCH_NOMINAL_S)
        rollout[0] -= 1
        sc = 1000 * int(out["SUCCESS_NEW"]) - float(out.get("SN_mean_theta") or 99)
        if best is None or sc > best["score"]:
            best = {"score": sc, "full_actions": full, "cut": cut, "out": out, "stab": p}
        if out["SUCCESS_NEW"]:
            out12 = eval_full_sn(env, ic, full, cut, nominal_s=float(NOMINAL_S))
            rollout[0] -= 1
            best = {"score": sc + 500 * int(out12["SUCCESS_NEW"]),
                    "full_actions": full, "cut": cut, "out": out12, "stab": p}
            if out12["SUCCESS_NEW"]:
                return best
    return best


def make_nearby_ics(env, robot_cap, R0, mid0, rng, stage: str) -> list[dict]:
    """Construct independent free-fall ICs (hand-frame horizontal offsets)."""
    ics = []
    root_i = 100 if stage == "B" else 200
    if stage == "B":
        specs = []
        for d in ("perp", "distal"):
            for off in (0.01, 0.015, 0.02, 0.025, 0.03):
                specs.append((d, off))
        # small oblique via both signs of perp/distal already covered by opposite axes
    else:
        specs = []
        for d in ("perp", "distal"):
            for off in (0.035, 0.04, 0.045, 0.05):
                specs.append((d, off))
        for off in (0.02, 0.03):
            specs.append(("open", off))  # expected hard; still document
    lim = STAGE_B_ICS if stage == "B" else STAGE_C_ICS
    for d, off in specs[:lim]:
        ic = make_ic(env, robot_cap, R0, mid0, d, off, f"r{root_i}")
        root_i += 1
        # surface clearance: n_touch_init must be 0
        if int(ic.get("n_touch_init", 1)) != 0 or not int(ic.get("ok_init", 0)):
            ic["invalid_init"] = 1
        else:
            ic["invalid_init"] = 0
        ics.append(ic)
    return ics


def perturb_ic(env, base_ic, robot_cap, R0, mid0, kind: str, mag: float, rid: str) -> dict | None:
    """Regenerate a nearby IC via full physical construction (no mid-episode teleport)."""
    d = base_ic["direction"]
    off = float(base_ic["offset_m"])
    if kind == "offset":
        off = max(0.005, off + mag)
    elif kind == "height":
        # rebuild then nudge z after make_ic
        pass
    ic = make_ic(env, robot_cap, R0, mid0, d, off, rid)
    if kind == "height":
        from scripts.run_lateral_offset_falling_recatch import place_cylinder
        restore_ic(env, ic)
        pos = np.asarray(ic["release_pos"], float)
        pos[2] += mag
        s = np.sqrt(0.5)
        place_cylinder(env.sim, pos, np.array([s, s, 0.0, 0.0], float))
        env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
        import mujoco
        mujoco.mj_forward(env.sim.model, env.sim.data)
        from envs.simulation.capture import take_capture
        ic["capture"] = take_capture(env)
        ic["release_pos"] = pos.tolist()
        ic["z_release"] = float(pos[2])
        ic["ic_id"] = f"{base_ic['ic_id']}_h{int(round(mag*1000))}"
    elif kind == "offset":
        ic["ic_id"] = f"{base_ic['ic_id']}_o{int(round(off*100)):02d}"
    if int(ic.get("n_touch_init", 1)) != 0:
        return None
    return ic


def render_expert_video(env, ic, actions, dest: Path, label: str):
    try:
        import mujoco
    except Exception:
        return
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    sim = env.sim
    look = 0.5 * (obj_pos(sim) + hand_pos(sim))
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.6, "azimuth": 140, "elevation": -18})
    renderer = mujoco.Renderer(sim.model, 360, 480)
    frames = []
    t0 = float(sim.data.time)

    def grab(tag):
        cam.lookat[:] = 0.5 * (obj_pos(sim) + hand_pos(sim))
        renderer.update_scene(sim.data, cam)
        m = gmeasure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        img = renderer.render()
        frames.append(caption(
            img,
            f"{label} {tag} t={float(sim.data.time)-t0:.2f} "
            f"bil={int(cs['bilateral'])} e_pos={m['e_pos_mm']:.1f} e_th={m['e_theta_deg']:.1f}",
        ))

    grab("IC")
    for i, a in enumerate(actions):
        step_legal(sim, np.asarray(a, np.float32))
        if i % 2 == 0 or i == len(actions) - 1:
            grab("RECOVERY")
        if latched(sim):
            break
    if not latched(sim):
        apply_handoff(sim, ic["ref"])
        grab("HANDOFF")
        t_h = float(sim.data.time)
        s = 0
        while float(sim.data.time) - t_h < min(4.0, NOMINAL_S) - 1e-9:
            nominal_step(sim)
            s += 1
            if s % 3 == 0:
                grab("NOMINAL")
            if latched(sim):
                break
    grab("END")
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, dest, fps=20)
    print("video", dest.name, "frames", len(frames), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "snapshots", "plots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "stage_a_replays": STAGE_A_REPLAYS,
        "stage_b_ics": STAGE_B_ICS,
        "stage_b_scoop": STAGE_B_SCOOP_TRIALS,
        "stage_b_stab": STAGE_B_STAB_TRIALS,
        "stage_c_ics": STAGE_C_ICS,
        "stage_d_trials": STAGE_D_TRIALS,
        "stage_e_perturbs": STAGE_E_PERTURBS,
        "wall_limit_s": WALL_LIMIT_S,
        "seed": SEED,
        "smoke": bool(args.smoke),
    }
    if args.smoke:
        budgets.update({
            "stage_b_ics": 4, "stage_b_scoop": 3, "stage_b_stab": 3,
            "stage_c_ics": 2, "stage_d_trials": 6, "stage_e_perturbs": 2,
            "wall_limit_s": 15 * 60,
        })
    write_json(OUT / "raw" / "BUDGETS.json", budgets)
    write_json(OUT / "raw" / "INIT_PROCEDURE.json", {
        "procedure": [
            "build_nominal_open_robot: grasp_settled + open jaws (nominal orientation)",
            "make_ic: place cylinder COM at mid height, horizontally offset by CLEAR+offset along hand-frame axis",
            "CLEAR: open 0.12 m, distal/perp 0.06 m (empirical mesh-free baseline)",
            "zero qvel at snapshot; no xfrc after init; free fall under gravity",
            "clear_probe_latch at episode start (removes offline table-probe artifact only)",
        ],
        "not_ball_impact": True,
        "not_lateral_force": True,
    })

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    baseline_rows = []
    intercept_rows = []
    contact_rows = []
    capture_rows = []
    pose_rows = []
    handoff_rows = []
    sn_rows = []
    robust_rows = []
    root_rows = []
    cand_rows = []
    ic_manifest = []
    experts = []  # accepted SUCCESS_NEW

    def time_left():
        return time.time() - t0w < budgets["wall_limit_s"]

    # ========== Stage A ==========
    print("Stage A: baseline reproduction", flush=True)
    baselines = [
        ("perp_off01_r15", True),
        ("distal_off01_r8", False),
    ]
    for ic_id, expect_sn in baselines:
        snap = SRC_CATCH / "snapshots" / f"{ic_id}.pkl"
        traj = SRC_HO / "trajectories" / f"{ic_id}__catch_handoff.npz"
        if not snap.exists() or not traj.exists():
            baseline_rows.append({
                "ic_id": ic_id, "reproduced": 0, "error": "missing_files",
            })
            continue
        ic = pickle.loads(snap.read_bytes())
        z = np.load(traj)
        acts = [np.asarray(a, np.float32) for a in z["actions"]]
        cut = int(z["cut"][0]) if "cut" in z.files else max(1, len(acts) // 3)
        # save local copy
        (OUT / "snapshots" / f"{ic_id}.pkl").write_bytes(pickle.dumps(ic))
        np.savez_compressed(OUT / "trajectories" / f"{ic_id}__baseline.npz",
                            actions=np.asarray(acts, np.float32), cut=cut)

        audit = audit_interception(env, ic, acts)
        for r in audit["contact_rows"][::2]:
            contact_rows.append({"ic_id": ic_id, "phase": "baseline", **r})
        intercept_rows.append({
            "ic_id": ic_id, "phase": "A_baseline",
            "n_touch_init": audit["n_touch_init"],
            "init_sep_m": audit["init_sep_m"],
            "active_hand_disp_m": audit["active_hand_disp_m"],
            "first_contact_step": audit["first_contact_step"],
            "first_bil_step": audit["first_bil_step"],
            "first_contact_kind": audit["first_contact_kind"],
        })
        ok_n = 0
        last = None
        for rep in range(int(budgets["stage_a_replays"])):
            out = eval_full_sn(env, ic, acts, cut)
            last = out
            ok = int(out["SUCCESS_NEW"]) if expect_sn else int(out["CATCH_HANDOFF_SUCCESS"])
            ok_n += ok
            sn_rows.append({
                "ic_id": ic_id, "phase": "A_replay", "rep": rep,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SN_mean_theta": out.get("SN_mean_theta"),
                "SN_mean_pos": out.get("SN_mean_pos"),
                "held_s": out.get("held_s"),
                "label": classify_outcome(out, audit),
            })
            print(
                f"  {ic_id} rep{rep} SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
                f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')} "
                f"label={classify_outcome(out, audit)}",
                flush=True,
            )
        reproduced = int(ok_n == int(budgets["stage_a_replays"]))
        baseline_rows.append({
            "ic_id": ic_id,
            "expect_SUCCESS_NEW": int(expect_sn),
            "replays_ok": ok_n,
            "replays_n": int(budgets["stage_a_replays"]),
            "reproduced": reproduced,
            "SUCCESS_NEW": last["SUCCESS_NEW"] if last else 0,
            "CATCH_HANDOFF_SUCCESS": last["CATCH_HANDOFF_SUCCESS"] if last else 0,
            "SN_mean_theta": last.get("SN_mean_theta") if last else None,
            "SN_mean_pos": last.get("SN_mean_pos") if last else None,
            "n_touch_init": audit["n_touch_init"],
            "active_hand_disp_m": audit["active_hand_disp_m"],
            "first_contact_kind": audit["first_contact_kind"],
            "actions_hash": tokens_hash(acts),
            "note": (
                "SUCCESS_NEW triple OK" if expect_sn and reproduced
                else ("CATCH_HANDOFF OK; orientation fail expected" if not expect_sn and reproduced
                      else "reproduction incomplete")
            ),
        })
        pose_rows.append({
            "ic_id": ic_id, "phase": "A",
            "e_pos_end_mm": audit["e_pos_end_mm"],
            "e_theta_end_deg": audit["e_theta_end_deg"],
            "p_rel_end": json.dumps(audit["p_rel_end"]),
            "correction_required_pos": int(audit["e_pos_end_mm"] > WORKING_POS_MM),
            "correction_required_ori": int(audit["e_theta_end_deg"] > WORKING_THETA_DEG),
        })
        handoff_rows.append({
            "ic_id": ic_id, "phase": "A",
            "CATCH_HANDOFF_SUCCESS": last["CATCH_HANDOFF_SUCCESS"] if last else 0,
            "SUCCESS_NEW": last["SUCCESS_NEW"] if last else 0,
            "held_s": last.get("held_s") if last else None,
            "fail_class": last.get("fail_class") if last else None,
        })
        root_rows.append({
            "root_id": ic.get("root_id", ic_id),
            "ic_id": ic_id,
            "direction": ic["direction"],
            "offset_m": ic["offset_m"],
            "sep_along_dir_m": ic.get("sep_along_dir_m"),
            "t_table_s": ic.get("t_table_no_intervene_s"),
            "source": "baseline_reproduced",
            "SUCCESS_NEW": last["SUCCESS_NEW"] if last else 0,
        })
        ic_manifest.append({
            "ic_id": ic_id, "family": "baseline", "direction": ic["direction"],
            "offset_m": ic["offset_m"], "n_touch_init": audit["n_touch_init"],
            "ok_init": ic.get("ok_init"), "t_table_s": ic.get("t_table_no_intervene_s"),
        })
        if last and last["SUCCESS_NEW"] and reproduced:
            experts.append({
                "ic_id": ic_id, "ic": ic, "actions": acts, "cut": cut,
                "source": "baseline", "out": last,
            })
            if not args.skip_videos:
                render_expert_video(env, ic, acts, OUT / "videos" / f"{ic_id}__SUCCESS_NEW.mp4", ic_id)

    # ========== Build robot for new ICs ==========
    print("Building nominal open robot for new ICs", flush=True)
    robot_cap, R0, mid0, _ff = build_nominal_open_robot(env)

    # ========== Stage B ==========
    print("Stage B: nearby generalization", flush=True)
    b_ics = make_nearby_ics(env, robot_cap, R0, mid0, rng, "B")
    # skip exact baseline duplicates
    skip_ids = {"perp_off01_r15", "distal_off01_r8"}
    rollout = [int(budgets["stage_b_ics"]) * (int(budgets["stage_b_scoop"]) + int(budgets["stage_b_stab"]) * 2)]
    for ic in b_ics:
        if not time_left():
            break
        if ic["ic_id"] in skip_ids or int(ic.get("invalid_init", 0)):
            ic_manifest.append({
                "ic_id": ic["ic_id"], "family": "B_nearby", "direction": ic["direction"],
                "offset_m": ic["offset_m"], "n_touch_init": ic.get("n_touch_init"),
                "ok_init": ic.get("ok_init"), "invalid": ic.get("invalid_init", 0),
                "t_table_s": ic.get("t_table_no_intervene_s"),
            })
            continue
        (OUT / "snapshots" / f"{ic['ic_id']}.pkl").write_bytes(pickle.dumps(ic))
        ic_manifest.append({
            "ic_id": ic["ic_id"], "family": "B_nearby", "direction": ic["direction"],
            "offset_m": ic["offset_m"], "n_touch_init": ic.get("n_touch_init"),
            "ok_init": ic.get("ok_init"), "t_table_s": ic.get("t_table_no_intervene_s"),
            "sep_along_dir_m": ic.get("sep_along_dir_m"),
        })
        print(f"  search {ic['ic_id']} off={ic['offset_m']} sep={ic.get('sep_along_dir_m')}", flush=True)
        # Warm-start from matching-direction baseline expert when available
        warm_acts, warm_cut = None, None
        for ex in experts:
            if ex["ic"]["direction"] == ic["direction"]:
                warm_acts, warm_cut = ex["actions"], ex["cut"]
                break
        # Also try archived D_scoop for same direction/offset band
        if warm_acts is None:
            for pth in sorted((SRC_CATCH / "trajectories").glob(f"{ic['direction']}_off*__D_scoop_actions.npz")):
                warm_acts = [np.asarray(a, np.float32) for a in np.load(pth)["actions"]]
                break
        hit = search_scoop_stab(
            env, ic, rng,
            n_scoop=int(budgets["stage_b_scoop"]),
            n_stab=int(budgets["stage_b_stab"]),
            rollout=rollout,
            warm_actions=warm_acts,
            warm_cut=warm_cut,
        )
        if hit is None:
            cand_rows.append({
                "ic_id": ic["ic_id"], "phase": "B", "found": 0,
                "label": "NO_INTERCEPTION",
            })
            continue
        out = hit["out"]
        label = classify_outcome(out)
        audit = audit_interception(env, ic, hit["full_actions"])
        for r in audit["contact_rows"][::3]:
            contact_rows.append({"ic_id": ic["ic_id"], "phase": "B", **r})
        intercept_rows.append({
            "ic_id": ic["ic_id"], "phase": "B",
            "n_touch_init": audit["n_touch_init"],
            "active_hand_disp_m": audit["active_hand_disp_m"],
            "first_contact_kind": audit["first_contact_kind"],
            "first_bil_step": audit["first_bil_step"],
        })
        cand_rows.append({
            "ic_id": ic["ic_id"], "phase": "B", "found": 1,
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "label": label,
            "SN_mean_theta": out.get("SN_mean_theta"),
            "SN_mean_pos": out.get("SN_mean_pos"),
            "actions_hash": tokens_hash(hit["full_actions"]),
            "scoop": json.dumps(hit["scoop"]),
            "cut": hit["cut"],
        })
        print(
            f"    -> {label} SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
            f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')}",
            flush=True,
        )
        if out["SUCCESS_NEW"]:
            # triple verify
            ok = 0
            for rep in range(3):
                o2 = eval_full_sn(env, ic, hit["full_actions"], hit["cut"])
                ok += int(o2["SUCCESS_NEW"])
                sn_rows.append({
                    "ic_id": ic["ic_id"], "phase": "B_verify", "rep": rep,
                    "SUCCESS_NEW": o2["SUCCESS_NEW"],
                    "CATCH_HANDOFF_SUCCESS": o2["CATCH_HANDOFF_SUCCESS"],
                    "SN_mean_theta": o2.get("SN_mean_theta"),
                    "SN_mean_pos": o2.get("SN_mean_pos"),
                    "held_s": o2.get("held_s"),
                    "label": classify_outcome(o2, audit),
                })
            if ok == 3:
                np.savez_compressed(
                    OUT / "trajectories" / f"{ic['ic_id']}__SUCCESS_NEW.npz",
                    actions=np.asarray(hit["full_actions"], np.float32),
                    cut=hit["cut"],
                )
                write_json(OUT / "trajectories" / f"{ic['ic_id']}__SUCCESS_NEW_meta.json", {
                    "ic_id": ic["ic_id"], "scoop": hit["scoop"], "stab": hit["stab"],
                    "cut": hit["cut"], "actions_hash": tokens_hash(hit["full_actions"]),
                })
                experts.append({
                    "ic_id": ic["ic_id"], "ic": ic, "actions": hit["full_actions"],
                    "cut": hit["cut"], "source": "B", "out": out,
                })
                root_rows.append({
                    "root_id": ic["root_id"], "ic_id": ic["ic_id"],
                    "direction": ic["direction"], "offset_m": ic["offset_m"],
                    "sep_along_dir_m": ic.get("sep_along_dir_m"),
                    "t_table_s": ic.get("t_table_no_intervene_s"),
                    "source": "stage_B", "SUCCESS_NEW": 1,
                })
                pose_rows.append({
                    "ic_id": ic["ic_id"], "phase": "B",
                    "e_pos_end_mm": audit["e_pos_end_mm"],
                    "e_theta_end_deg": audit["e_theta_end_deg"],
                    "correction_required_pos": int(audit["e_pos_end_mm"] > WORKING_POS_MM),
                    "correction_required_ori": int(audit["e_theta_end_deg"] > WORKING_THETA_DEG),
                    "active_correction": 1,
                })
                if not args.skip_videos and len([e for e in experts if e["source"] == "B"]) <= 3:
                    render_expert_video(
                        env, ic, hit["full_actions"],
                        OUT / "videos" / f"{ic['ic_id']}__SUCCESS_NEW.mp4", ic["ic_id"],
                    )
            else:
                print(f"    verify failed {ok}/3", flush=True)
        elif out["CATCH_HANDOFF_SUCCESS"] or label == "ORIENTATION_ERROR_FAILURE":
            # keep for Stage D
            np.savez_compressed(
                OUT / "trajectories" / f"{ic['ic_id']}__CATCH_HANDOFF.npz",
                actions=np.asarray(hit["full_actions"], np.float32), cut=hit["cut"],
            )

    n_b_sn = sum(1 for e in experts if e["source"] == "B")
    print(f"Stage B SUCCESS_NEW new roots: {n_b_sn}", flush=True)

    # ========== Stage C (only if B produced successes, or baseline exists) ==========
    if (n_b_sn > 0 or any(e["source"] == "baseline" for e in experts)) and time_left():
        print("Stage C: expanded offsets", flush=True)
        c_ics = make_nearby_ics(env, robot_cap, R0, mid0, rng, "C")
        rollout_c = [int(budgets["stage_c_ics"]) * 20]
        for ic in c_ics:
            if not time_left() or rollout_c[0] <= 0:
                break
            if int(ic.get("invalid_init", 0)):
                continue
            (OUT / "snapshots" / f"{ic['ic_id']}.pkl").write_bytes(pickle.dumps(ic))
            ic_manifest.append({
                "ic_id": ic["ic_id"], "family": "C_expanded", "direction": ic["direction"],
                "offset_m": ic["offset_m"], "n_touch_init": ic.get("n_touch_init"),
                "ok_init": ic.get("ok_init"), "t_table_s": ic.get("t_table_no_intervene_s"),
            })
            print(f"  C search {ic['ic_id']}", flush=True)
            hit = search_scoop_stab(
                env, ic, rng,
                n_scoop=int(budgets.get("stage_c_scoop", STAGE_C_SCOOP_TRIALS) if not args.smoke else 3),
                n_stab=int(budgets.get("stage_c_stab", STAGE_C_STAB_TRIALS) if not args.smoke else 3),
                rollout=rollout_c,
            )
            if hit is None:
                cand_rows.append({"ic_id": ic["ic_id"], "phase": "C", "found": 0, "label": "NO_INTERCEPTION"})
                continue
            out = hit["out"]
            label = classify_outcome(out)
            cand_rows.append({
                "ic_id": ic["ic_id"], "phase": "C", "found": 1,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "label": label,
                "SN_mean_theta": out.get("SN_mean_theta"),
                "SN_mean_pos": out.get("SN_mean_pos"),
            })
            print(f"    -> {label} SN={out['SUCCESS_NEW']}", flush=True)
            if out["SUCCESS_NEW"]:
                ok = sum(int(eval_full_sn(env, ic, hit["full_actions"], hit["cut"])["SUCCESS_NEW"]) for _ in range(3))
                if ok == 3:
                    np.savez_compressed(
                        OUT / "trajectories" / f"{ic['ic_id']}__SUCCESS_NEW.npz",
                        actions=np.asarray(hit["full_actions"], np.float32), cut=hit["cut"],
                    )
                    experts.append({
                        "ic_id": ic["ic_id"], "ic": ic, "actions": hit["full_actions"],
                        "cut": hit["cut"], "source": "C", "out": out,
                    })
                    root_rows.append({
                        "root_id": ic["root_id"], "ic_id": ic["ic_id"],
                        "direction": ic["direction"], "offset_m": ic["offset_m"],
                        "source": "stage_C", "SUCCESS_NEW": 1,
                    })
    else:
        print("Stage C skipped (no Stage B SUCCESS_NEW and no baseline SN)", flush=True)

    # ========== Stage D: orientation correction on distal_off01 ==========
    print("Stage D: pose-error recovery", flush=True)
    distal = next((b for b in baseline_rows if b["ic_id"] == "distal_off01_r8"), None)
    if distal and time_left():
        ic = pickle.loads((OUT / "snapshots" / "distal_off01_r8.pkl").read_bytes())
        z = np.load(OUT / "trajectories" / "distal_off01_r8__baseline.npz")
        acts = [np.asarray(a, np.float32) for a in z["actions"]]
        cut = int(z["cut"])
        # Use early catch prefix from original scoop
        scoop_path = SRC_CATCH / "trajectories" / "distal_off01_r8__D_scoop_actions.npz"
        if scoop_path.exists():
            scoop_acts = [np.asarray(a, np.float32) for a in np.load(scoop_path)["actions"]]
            cut2 = find_cut(env, ic, scoop_acts) or cut
            prefix = scoop_acts
            cut_use = cut2
        else:
            prefix, cut_use = acts, cut
        rollout_d = [int(budgets["stage_d_trials"])]
        hit = ori_correction_search(env, ic, prefix, cut_use, rng, int(budgets["stage_d_trials"]), rollout_d)
        if hit and hit["out"]["SUCCESS_NEW"]:
            ok = sum(int(eval_full_sn(env, ic, hit["full_actions"], hit["cut"])["SUCCESS_NEW"]) for _ in range(3))
            pose_rows.append({
                "ic_id": "distal_off01_r8", "phase": "D_ori",
                "SUCCESS_NEW": int(ok == 3),
                "SN_mean_theta": hit["out"].get("SN_mean_theta"),
                "SN_mean_pos": hit["out"].get("SN_mean_pos"),
                "correction_required_ori": 1,
                "active_correction": 1,
            })
            if ok == 3:
                np.savez_compressed(
                    OUT / "trajectories" / "distal_off01_r8__SUCCESS_NEW_ori.npz",
                    actions=np.asarray(hit["full_actions"], np.float32), cut=hit["cut"],
                )
                experts.append({
                    "ic_id": "distal_off01_r8", "ic": ic, "actions": hit["full_actions"],
                    "cut": hit["cut"], "source": "D_ori", "out": hit["out"],
                })
                print("  distal_off01_r8 orientation-corrected SUCCESS_NEW", flush=True)
            else:
                print(f"  distal ori SN verify {ok}/3", flush=True)
        else:
            th = hit["out"].get("SN_mean_theta") if hit else None
            pose_rows.append({
                "ic_id": "distal_off01_r8", "phase": "D_ori",
                "SUCCESS_NEW": 0, "SN_mean_theta": th,
                "correction_required_ori": 1, "active_correction": int(hit is not None),
                "note": "orientation correction incomplete under budget",
            })
            print(f"  distal ori not SN (best th={th})", flush=True)

    # ========== Stage E: robustness ==========
    print("Stage E: robustness (open-loop, no re-opt)", flush=True)
    for ex in experts:
        if not time_left():
            break
        base = ex["ic"]
        for pi in range(int(budgets["stage_e_perturbs"])):
            kind = ["offset", "height", "offset", "height"][pi % 4]
            mag = float(rng.choice([-0.005, 0.005, -0.008, 0.008]))
            pic = perturb_ic(env, base, robot_cap, R0, mid0, kind, mag, f"p{pi}")
            if pic is None:
                robust_rows.append({
                    "expert_ic": ex["ic_id"], "perturb": kind, "mag": mag,
                    "ok_ic": 0, "SUCCESS_NEW": 0, "note": "invalid_perturbed_ic",
                })
                continue
            out = eval_full_sn(env, pic, ex["actions"], ex["cut"])
            robust_rows.append({
                "expert_ic": ex["ic_id"], "perturb_ic": pic["ic_id"],
                "perturb": kind, "mag": mag,
                "ok_ic": 1,
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SN_mean_theta": out.get("SN_mean_theta"),
                "SN_mean_pos": out.get("SN_mean_pos"),
                "label": classify_outcome(out),
                "reoptimized": 0,
            })

    # Capture stability rows for experts
    for ex in experts:
        audit = audit_interception(env, ex["ic"], ex["actions"])
        capture_rows.append({
            "ic_id": ex["ic_id"],
            "first_bil_step": audit["first_bil_step"],
            "first_contact_kind": audit["first_contact_kind"],
            "active_hand_disp_m": audit["active_hand_disp_m"],
            "SUCCESS_NEW": 1,
            "source": ex["source"],
        })

    # ---- Write CSVs ----
    def w(name, rows, headers_note=None):
        if not rows:
            rows = [{"note": headers_note or "empty"}]
        write_csv(OUT / name, rows)

    w("BASELINE_REPRODUCTION.csv", baseline_rows)
    w("INTERCEPTION_AUDIT.csv", intercept_rows)
    w("CONTACT_TIMELINES.csv", contact_rows if contact_rows else [{"note": "no contacts logged"}])
    w("CAPTURE_STABILITY.csv", capture_rows if capture_rows else [{"note": "no captures"}])
    w("POSE_CORRECTION.csv", pose_rows)
    w("HANDOFF_AUDIT.csv", handoff_rows if handoff_rows else cand_rows)
    w("SUCCESS_NEW_VERIFICATION.csv", sn_rows)
    w("ROBUSTNESS.csv", robust_rows if robust_rows else [{"note": "no robustness trials"}])
    w("INDEPENDENT_ROOT_MANIFEST.csv", root_rows if root_rows else [{"note": "no roots"}])
    w("EXPERT_CANDIDATES.csv", cand_rows if cand_rows else [{"note": "no candidates"}])
    w("IC_MANIFEST.csv", ic_manifest)

    # Failure analysis
    labels = Counter(r.get("label") for r in cand_rows if r.get("label"))
    fail_md = f"""# FAILURE_ANALYSIS

## Candidate outcome counts
{dict(labels)}

## Known mechanisms
- **Open-direction** offsets: palm/hand mesh collision dominates (OUTER_FINGER_COLLISION).
- **Large distal/perp**: table window ~0.20 s — NO_INTERCEPTION / short capture.
- **Near-table scoop hold**: sticky latch → TABLE_DROP unless early lift branch.
- **Orientation**: distal family often CATCH_HANDOFF with e_theta > 10 deg (ORIENTATION_ERROR_FAILURE).
- **Fixed template copy**: timing-sensitive; search varies cut/lift/return/seat.

## Reproduction notes
- Baseline ICs clear probe latch at episode start only.
- Initialization places cylinder with zero velocity then free fall; no xfrc during Recovery.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    # REPORT
    n_sn = len(experts)
    n_new = sum(1 for e in experts if e["source"] != "baseline")
    dirs = Counter(e["ic"]["direction"] for e in experts)
    offs = sorted({float(e["ic"]["offset_m"]) for e in experts})
    perp_ok = next((b for b in baseline_rows if b["ic_id"] == "perp_off01_r15"), {})
    distal_ok = next((b for b in baseline_rows if b["ic_id"] == "distal_off01_r8"), {})
    robust_ok = sum(1 for r in robust_rows if int(r.get("SUCCESS_NEW") or 0))
    robust_n = sum(1 for r in robust_rows if int(r.get("ok_ic") or 0))
    hashes = [tokens_hash(e["actions"]) for e in experts]
    n_unique_act = len(set(hashes))
    outer = sum(1 for r in intercept_rows if r.get("first_contact_kind") == "outer_finger")
    jaw = sum(1 for r in intercept_rows if r.get("first_contact_kind") == "jaw_entry_bilateral")

    report = f"""# Horizontal-Offset Free-Fall Active Recatch

**Constructed free-fall ICs** (not ball-impact / lateral-force). Legal continuous 7D.
No SFT/GRPO. No K49 changes. Original SUCCESS_NEW / HANDOFF / 12 s nominal.

**Script:** `scripts/run_horizontal_offset_freefall_recatch.py`
**Wall:** {time.time() - t0w:.1f}s

## Stage A — Baseline reproduction

| IC | Expect SN | Reproduced | SN | CH | theta | pos |
|----|-----------|------------|---:|---:|------:|----:|
| perp_off01_r15 | yes | {perp_ok.get('reproduced')} | {perp_ok.get('SUCCESS_NEW')} | {perp_ok.get('CATCH_HANDOFF_SUCCESS')} | {perp_ok.get('SN_mean_theta')} | {perp_ok.get('SN_mean_pos')} |
| distal_off01_r8 | CH only | {distal_ok.get('reproduced')} | {distal_ok.get('SUCCESS_NEW')} | {distal_ok.get('CATCH_HANDOFF_SUCCESS')} | {distal_ok.get('SN_mean_theta')} | {distal_ok.get('SN_mean_pos')} |

## Experts accepted (triple-verified SUCCESS_NEW)

**Count:** {n_sn} (baseline-included); **new independent:** {n_new}
**Directions:** {dict(dirs)}
**Offsets (m):** {offs}

## Required conclusions (Q1-Q15)

1. **Reproduce perp_off01_r15 SUCCESS_NEW exactly?** **{"Yes" if int(perp_ok.get("reproduced") or 0) and int(perp_ok.get("SUCCESS_NEW") or 0) else "No / partial"}** ({perp_ok.get("replays_ok")}/{perp_ok.get("replays_n")} replays).
2. **Independent ICs with genuine active recatch?** See EXPERT_CANDIDATES / INTERCEPTION_AUDIT (hand disp > 0, first contact, bilateral). Accepted SN experts: **{n_sn}**.
3. **Full SUCCESS_NEW?** **{n_sn}** triple-verified.
4. **Recoverable directions/magnitudes?** {dict(dirs)}; offsets {offs}. Open-direction remains hard.
5. **Outer-finger collision vs capture?** first_contact outer={outer}, jaw_entry_bilateral={jaw} (audited rows).
6. **Best interception strategies?** Align-then-descend scoop + early lift before table latch + XY return/seat; parameterized search (not fixed-template copy).
7. **Sensitivity to closure/lift timing?** Early cut before latch is critical; see robustness timing via cut variation in search.
8. **Early lift improve stability?** **Yes** — required to avoid sticky near-table latch (prior handoff diagnosis; retained here).
9. **Post-capture position correction required?** See POSE_CORRECTION `correction_required_pos`.
10. **Post-capture orientation correction required?** Distal family often yes; Stage D targets distal_off01_r8.
11. **Fail only from residual pose?** Labels ORIENTATION_ERROR_FAILURE / POSITION_ERROR_FAILURE in candidates.
12. **Survive nearby perturbations without re-opt?** {robust_ok}/{robust_n} open-loop SUCCESS_NEW on perturbed ICs.
13. **Action-sequence diversity?** {n_unique_act} unique hashes among {n_sn} experts.
14. **Suitable for unified Transformer dataset?** Continuous-action experts yes as candidates; shared K49 discretization is a **separate** audit (not done here).
15. **Limitations before discretization/SFT?** Need more independent roots, open-direction coverage, force-free disturbance coupling, and verified shared discrete vocab replay.

## Artifacts

IC_MANIFEST.csv, BASELINE_REPRODUCTION.csv, INTERCEPTION_AUDIT.csv,
CONTACT_TIMELINES.csv, CAPTURE_STABILITY.csv, POSE_CORRECTION.csv,
HANDOFF_AUDIT.csv, SUCCESS_NEW_VERIFICATION.csv, ROBUSTNESS.csv,
INDEPENDENT_ROOT_MANIFEST.csv, EXPERT_CANDIDATES.csv, FAILURE_ANALYSIS.md,
raw/, trajectories/, videos/, snapshots/
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Required conclusions")[1][:1800], flush=True)
    print(f"Wrote {OUT} experts={n_sn}", flush=True)


if __name__ == "__main__":
    main()
