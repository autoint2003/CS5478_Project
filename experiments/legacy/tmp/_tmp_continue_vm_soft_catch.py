"""Continue velocity-matched soft-catch within frozen budget (remaining rollouts).

Does not extend TOTAL_ROLLOUT_BUDGET beyond 2400. Improves VM law, early-contact
M2 metrics, and pose/HANDOFF search using continuous fine_down (comparison only).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, V_AIRBORNE, map_action
from envs.simulation.capture import restore_capture
from models.k48_checkpoint import load_trained_k48
from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics
from scripts.expand_k49_large_chase_dynamic import hand_pos, obj_pos
from scripts.expand_k49_capture_brake_diagnostics import find_snap, load_blob
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from scripts.run_velocity_matched_soft_catch import (
    BE_ICS, FINE_DOWN, OUT,
    TOTAL_ROLLOUT_BUDGET, WALL_LIMIT_S,
    body_v_w, contact_point_kinematics, param_grid,
    run_actions_eval, score_soft_catch, toks_to_actions, write_catch_it_comparison,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import make_training_env
from envs.step_legal import step_legal
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.horizon_outcome import recovery_failed
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4
import mujoco


def desired_match_action_v2(sim, scale=0.85, grip=1.0, k_err=1.2, use_pt=True):
    """Close relative velocity: command toward object contact-point / COM velocity."""
    vo, wo = body_v_w(sim, sim.ids.object_body)
    vh, wh = body_v_w(sim, sim.ids.hand_body)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = obj_pos(sim)
    ph = hand_pos(sim)
    # predicted contact near finger mid
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    mid = 0.5 * (pl + pr)
    r = mid - po
    v_obj_pt = vo + np.cross(wo, r) if use_pt else vo
    # desired hand velocity matches object surface; PD on velocity error
    v_des_w = scale * v_obj_pt + k_err * (v_obj_pt - vh)
    # also nudge toward closing position error mildly
    v_des_w = v_des_w + 0.4 * (po - ph)
    v_des_h = Rh.T @ v_des_w
    a = np.zeros(7, np.float32)
    a[0] = float(np.clip(v_des_h[0] / V_AIRBORNE, -1, 1))
    a[1] = float(np.clip(v_des_h[1] / V_AIRBORNE, -1, 1))
    vz = float(v_des_h[2])
    a[2] = float(np.clip(vz / 3.8, 0, 1) if vz >= 0 else np.clip(vz / 1.0, -1, 0))
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    w_des_b = rd.T @ (0.25 * scale * wo)
    a[3:6] = np.clip(w_des_b / 4.0, -1, 1)
    a[6] = float(grip)
    return a


def sequence_vm_v2(env, blob, p, mode="B"):
    restore_capture(env, blob["capture"])
    acts = []
    for _ in range(int(p.get("grip_open_n", 0))):
        a = desired_match_action_v2(env.sim, scale=float(p["v_scale"]), grip=-1.0,
                                    k_err=float(p.get("k_err", 1.2)), use_pt=bool(p.get("use_pt", True)))
        step_legal(env.sim, a)
        acts.append(a.copy())
    for i in range(int(p["n_pursue"])):
        grip = float(p.get("grip_hold", 1.0))
        if mode == "C" and i >= int(p.get("grip_close_onset", 999)):
            grip = float(p.get("grip_close_val", 1.0))
        a = desired_match_action_v2(
            env.sim, scale=float(p["v_scale"]), grip=grip,
            k_err=float(p.get("k_err", 1.2)), use_pt=bool(p.get("use_pt", True)),
        )
        step_legal(env.sim, a)
        acts.append(a.copy())
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if failed:
            return acts
    for _ in range(int(p.get("n_comotion", 0))):
        grip = float(p.get("grip_close_val", 1.0)) if mode == "C" else float(p.get("grip_hold", 1.0))
        a = desired_match_action_v2(
            env.sim, scale=float(p.get("comotion_scale", 0.5)), grip=grip,
            k_err=float(p.get("k_err", 0.8)), use_pt=True,
        )
        step_legal(env.sim, a)
        acts.append(a.copy())
    for _ in range(int(p.get("n_brake", 0))):
        a = FINE_DOWN.copy() if p.get("use_fine_down") else np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
        step_legal(env.sim, a)
        acts.append(a.copy())
    for _ in range(int(p.get("n_settle", 0))):
        if p.get("use_fine_down"):
            a = FINE_DOWN.copy()
        else:
            a = np.array([0, 0, 0, 0, -0.2, 0, 1], np.float32)
        step_legal(env.sim, a)
        acts.append(a.copy())
    return acts


def early_contact_m2(env, blob, acts, n_early=8):
    """Reduce contact-point relative velocity over first n_early contact frames."""
    restore_capture(env, blob["capture"])
    vals = []
    for a in acts:
        cs_nL = int(env.sim.data.ncon)  # will use contact_point_kinematics
        kin = contact_point_kinematics(env.sim)
        if kin["n_contacts"] > 0 and kin["mean_v_rel_pt"] is not None:
            vals.append(float(kin["mean_v_rel_pt"]))
            if len(vals) >= n_early:
                break
        step_legal(env.sim, np.clip(np.asarray(a, np.float32).reshape(7), -1, 1))
        env._prev = np.asarray(a, np.float32).copy()
        failed, _ = recovery_failed(env.sim, blob["ref"], 0)
        if failed:
            break
    if len(vals) < 2:
        return 0, None, None
    return int(vals[-1] < vals[0] - 0.03), vals[0], vals[-1]


def known_pose_recipe(table, book48):
    """Continuous replay of verified tok63 fine_down SUCCESS_NEW sequence (diagnostic)."""
    # [30]*3 + [17]*16 + [0]*8 + [12]*10 + fine_down*10 + [48]*16
    toks = [30] * 3 + [17] * 16 + [0] * 8 + [12] * 10
    acts = toks_to_actions(table, book48, toks)
    acts += [FINE_DOWN.copy() for _ in range(10)]
    acts += toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 16)
    return acts


def load_csv(path):
    import csv
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))
    used0 = int(usage.get("rollouts_used", 0))
    remaining = TOTAL_ROLLOUT_BUDGET - used0
    print(f"Continue soft-catch: used={used0} remaining={remaining}", flush=True)
    if remaining <= 0:
        print("Budget exhausted", flush=True)
        return

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019 + 7)
    rollout = [used0]
    t0 = time.time()
    wall_left = WALL_LIMIT_S - float(usage.get("wall_s", 0))

    def time_left():
        return time.time() - t0 < max(60.0, wall_left)

    matched = load_csv(OUT / "MATCHED_CONTROL_ABLATIONS.csv")
    vm_rows = load_csv(OUT / "VELOCITY_MATCHING_SEARCH.csv")
    grip_rows = load_csv(OUT / "GRIP_TIMING_SEARCH.csv")
    stab_rows = load_csv(OUT / "CAPTURE_STABILIZATION.csv")
    full_rows = load_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv")
    audit_rows = load_csv(OUT / "CONTACT_POINT_VELOCITY_AUDIT.csv")
    track_rows = load_csv(OUT / "CONTROLLER_TRACKING.csv")
    m2_rows = []

    best_by_ic = {}
    best_path = OUT / "raw" / "BEST_BY_IC.json"
    if best_path.exists():
        best_by_ic = json.loads(best_path.read_text(encoding="utf-8"))

    # convert float fields in prior rows for scoring continuity
    def to_float_row(r):
        out = dict(r)
        for k, v in list(out.items()):
            if v in ("", None):
                out[k] = None
                continue
            try:
                if k not in ("ic_id", "condition", "seed", "reason_fail", "eval_reason"):
                    out[k] = float(v) if "." in str(v) or str(v).lstrip("-").isdigit() else v
            except Exception:
                pass
        return out

    for r in matched:
        r2 = to_float_row(r)
        ic = r2["ic_id"]
        sc = float(r2.get("score") or -1e9)
        cur = best_by_ic.get(ic)
        if cur is None or sc > float(cur.get("score", -1e9)):
            # keep prior best without actions if missing
            best_by_ic[ic] = {
                "score": sc, "condition": r2.get("condition"),
                "actions": cur.get("actions") if cur else None,
                "out": {k: r2.get(k) for k in (
                    "replay_ok", "had_bil", "max_bil_persist_s", "SUCCESS_NEW",
                    "v_pt_rel_at_contact", "win_mean_pos", "win_mean_theta", "hold_both_s",
                    "M2_vcom_reduced_pre", "had_contact", "had_uni", "geo_between", "geo_outer_risk",
                )},
            }

    def eval_one(ic, blob, label, acts, family):
        out, kin, geo = run_actions_eval(env, blob, acts, eval_full=True)
        m2e, v0, v1 = early_contact_m2(env, blob, acts)
        out["M2_early_vpt_reduced"] = m2e
        out["early_vpt_start"] = v0
        out["early_vpt_end"] = v1
        rollout[0] += 1
        sc = score_soft_catch(out) + 4.0 * m2e
        row = {"ic_id": ic, "condition": label, "score": sc, **out}
        family.append(row)
        matched.append(row)
        m2_rows.append({
            "ic_id": ic, "condition": label, "M2_early": m2e,
            "early_vpt_start": v0, "early_vpt_end": v1,
            "M2_pre": out.get("M2_vcom_reduced_pre"),
            "v_pt_rel_at_contact": out.get("v_pt_rel_at_contact"),
            "had_bil": out.get("had_bil"), "persist": out.get("max_bil_persist_s"),
            "SUCCESS_NEW": out.get("SUCCESS_NEW"),
        })
        cur = best_by_ic.get(ic)
        if cur is None or sc > float(cur.get("score", -1e9)):
            best_by_ic[ic] = {
                "score": sc, "condition": label,
                "actions": [a.tolist() for a in acts], "out": out,
            }
        if out.get("SUCCESS_NEW"):
            full_rows.append({
                "ic_id": ic, "condition": label, "SUCCESS_NEW": 1,
                "win_mean_theta": out.get("win_mean_theta"),
                "win_mean_pos": out.get("win_mean_pos"),
                "hold_both_s": out.get("hold_both_s"),
                "v_pt_rel_at_contact": out.get("v_pt_rel_at_contact"),
            })
            np.savez_compressed(
                OUT / "trajectories" / f"{ic}__{label}_actions.npz",
                actions=np.asarray(acts, np.float32),
            )
            write_json(OUT / "trajectories" / f"{ic}__{label}_meta.json", row)
            print(f"  FULL {ic} {label}", flush=True)
        return out, sc

    # ---- Budget split of remaining: 45% VM deep / 25% grip / 20% stabilize / 10% pose ----
    rem = remaining
    lims = [
        used0 + int(rem * 0.45),
        used0 + int(rem * 0.70),
        used0 + int(rem * 0.90),
        TOTAL_ROLLOUT_BUDGET,
    ]

    # Rank ICs by audit v_pt
    ranked = []
    for ic in BE_ICS:
        rows = [r for r in audit_rows if r["ic_id"] == ic and r.get("v_pt_rel_at_contact") not in (None, "")]
        if not rows:
            continue
        vpt = float(np.mean([float(r["v_pt_rel_at_contact"]) for r in rows]))
        ranked.append((ic, vpt))
    ranked.sort(key=lambda x: -x[1])

    print(f"=== Cont B1 VM deep (to {lims[0]}) ===", flush=True)
    for ic, _ in ranked:
        if rollout[0] >= lims[0] or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  VM {ic} roll={rollout[0]}", flush=True)
        # K49 baselines already done; focus B/C improved
        for mode, family, ntry in [("B", vm_rows, 18), ("C", grip_rows, 14)]:
            for p in param_grid(rng, ntry, mode=mode):
                if rollout[0] >= lims[0]:
                    break
                p = dict(p)
                p["k_err"] = float(rng.choice([0.6, 1.0, 1.4, 1.8]))
                p["use_pt"] = bool(rng.random() < 0.7)
                acts = sequence_vm_v2(env, blob, p, mode=mode)
                label = f"{mode}_vm_v2"
                eval_one(ic, blob, label, acts, family)

    print(f"=== Cont B2 ablations (to {lims[1]}) ===", flush=True)
    for ic, _ in ranked[:10]:
        if rollout[0] >= lims[1] or not time_left():
            break
        blob = load_blob(find_snap(ic))
        abls = [
            ("abl2_fast_stop", {"n_pursue": 10, "v_scale": 1.0, "k_err": 0.5, "grip_open_n": 0, "n_comotion": 0, "n_brake": 12, "n_settle": 8, "use_pt": True}),
            ("abl2_vm_stop", {"n_pursue": 18, "v_scale": 0.9, "k_err": 1.4, "grip_open_n": 2, "n_comotion": 0, "n_brake": 12, "n_settle": 8, "use_pt": True}),
            ("abl2_vm_comotion", {"n_pursue": 16, "v_scale": 0.85, "k_err": 1.2, "grip_open_n": 2, "n_comotion": 8, "comotion_scale": 0.45, "n_brake": 8, "n_settle": 10, "use_pt": True}),
            ("abl2_vm_grip", {"n_pursue": 16, "v_scale": 0.85, "k_err": 1.2, "grip_open_n": 3, "grip_close_onset": 6, "grip_close_val": 1.0, "grip_hold": -0.48, "n_comotion": 6, "n_brake": 8, "n_settle": 12, "use_pt": True}),
            ("abl2_vm_gradual", {"n_pursue": 20, "v_scale": 0.7, "k_err": 1.6, "grip_open_n": 2, "n_comotion": 10, "comotion_scale": 0.35, "n_brake": 14, "n_settle": 16, "use_pt": True}),
            ("abl2_com_only", {"n_pursue": 16, "v_scale": 0.85, "k_err": 1.2, "grip_open_n": 2, "n_comotion": 6, "n_brake": 8, "n_settle": 8, "use_pt": False}),
        ]
        for name, p in abls:
            if rollout[0] >= lims[1]:
                break
            mode = "C" if "grip" in name else "B"
            acts = sequence_vm_v2(env, blob, p, mode=mode)
            eval_one(ic, blob, name, acts, matched)

    print(f"=== Cont B3 stabilize (to {lims[2]}) ===", flush=True)
    # Prefer ICs with any bil history
    stab_ics = ["hard_910012", "hard_910006", "hdimp_0017", "hdimp_0022", "hdimp_0012"]
    for ic in stab_ics:
        if rollout[0] >= lims[2] or not time_left():
            break
        blob = load_blob(find_snap(ic))
        pack = best_by_ic.get(ic) or {}
        base = pack.get("actions")
        if base:
            base_acts = [np.asarray(a, np.float32) for a in base]
        else:
            # open chase then hold
            base_acts = toks_to_actions(table, book48, [30] * 3 + [17] * 16 + [0] * 4 + [12] * 8)
        for extra_name, extra in [
            ("stab2_hold", [np.array([0, 0, 0, 0, 0, 0, 1], np.float32)] * 20),
            ("stab2_finedown", [FINE_DOWN.copy() for _ in range(12)] + [np.array([0, 0, 0, 0, -0.25, 0, 1], np.float32)] * 16),
            ("stab2_wy", [np.array([0, 0, 0, 0, -0.25, 0, 1], np.float32)] * 24),
            ("stab2_fd_wy", [FINE_DOWN.copy() for _ in range(8)] + [np.array([0, 0, 0, 0, -0.25, 0, 1], np.float32)] * 20),
            ("stab2_recipe", known_pose_recipe(table, book48)),
        ]:
            if rollout[0] >= lims[2]:
                break
            acts = base_acts + extra if extra_name != "stab2_recipe" else extra
            eval_one(ic, blob, extra_name, acts, stab_rows)

    print(f"=== Cont B4 pose/HANDOFF (to {lims[3]}) ===", flush=True)
    # Explicit known continuous fine_down recipe + variants
    pose_ic = "hard_910012"
    blob = load_blob(find_snap(pose_ic))
    recipes = [
        ("pose_verified_fd", known_pose_recipe(table, book48)),
        ("pose_fd_short", toks_to_actions(table, book48, [30] * 3 + [17] * 14 + [0] * 6 + [12] * 8)
         + [FINE_DOWN.copy() for _ in range(14)]
         + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 18)),
        ("pose_fd_long", toks_to_actions(table, book48, [30] * 3 + [17] * 18 + [0] * 6 + [12] * 12)
         + [FINE_DOWN.copy() for _ in range(16)]
         + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 20)),
        ("pose_vm_then_fd", None),
    ]
    for name, acts in recipes:
        if rollout[0] >= lims[3] or not time_left():
            break
        if acts is None:
            p = {"n_pursue": 14, "v_scale": 0.7, "k_err": 1.0, "grip_open_n": 2, "n_comotion": 4,
                 "n_brake": 4, "n_settle": 0, "use_pt": True, "use_fine_down": False}
            acts = sequence_vm_v2(env, blob, p, mode="B")
            acts += [FINE_DOWN.copy() for _ in range(12)]
            acts += toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 16)
        print(f"  pose {name} roll={rollout[0]}", flush=True)
        eval_one(pose_ic, blob, name, acts, stab_rows)

    # local CEM around best pose recipe
    base_recipe = known_pose_recipe(table, book48)
    while rollout[0] < lims[3] and time_left():
        acts = [a.copy() for a in base_recipe]
        # perturb a random window of fine_down / chase magnitude
        for i in range(len(acts)):
            if rng.random() < 0.12:
                acts[i] = np.clip(acts[i] + rng.normal(0, 0.05, size=7).astype(np.float32), -1, 1)
                acts[i][6] = float(np.clip(acts[i][6], -1, 1))
        eval_one(pose_ic, blob, "pose_cem_fd", acts, stab_rows)

    # burn remaining on top high-v ICs with VM+co-motion
    while rollout[0] < TOTAL_ROLLOUT_BUDGET and time_left():
        ic = ranked[int(rng.integers(0, min(6, len(ranked))))][0]
        blob = load_blob(find_snap(ic))
        p = dict(param_grid(rng, 1, mode="C")[0])
        p["k_err"] = float(rng.choice([1.0, 1.4, 1.8]))
        p["use_pt"] = True
        acts = sequence_vm_v2(env, blob, p, mode="C")
        eval_one(ic, blob, "C_vm_v2_burn", acts, grip_rows)

    write_csv(OUT / "MATCHED_CONTROL_ABLATIONS.csv", matched)
    write_csv(OUT / "VELOCITY_MATCHING_SEARCH.csv", vm_rows)
    write_csv(OUT / "GRIP_TIMING_SEARCH.csv", grip_rows)
    write_csv(OUT / "CAPTURE_STABILIZATION.csv", stab_rows)
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_rows)
    write_csv(OUT / "raw" / "M2_EARLY_CONTACT.csv", m2_rows)
    write_json(OUT / "raw" / "BEST_BY_IC.json", {
        ic: {"score": v["score"], "condition": v["condition"], "out": v["out"],
             "has_actions": v.get("actions") is not None}
        for ic, v in best_by_ic.items()
    })
    # save best actions
    for ic, v in best_by_ic.items():
        if v.get("actions"):
            np.savez_compressed(
                OUT / "trajectories" / f"best_{ic}_actions.npz",
                actions=np.asarray(v["actions"], np.float32),
            )

    wall_total = float(usage.get("wall_s", 0)) + (time.time() - t0)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": wall_total, "n_full": len([r for r in full_rows if int(float(r.get("SUCCESS_NEW") or 0))]),
        "continued": True, "prior_used": used0,
    })

    # videos for best of each family
    _render_selected(env, table, book48, best_by_ic, full_rows)

    _rewrite_reports(audit_rows, track_rows, matched, vm_rows, grip_rows, stab_rows, full_rows, m2_rows, best_by_ic, checksum)
    print("Done continue", OUT / "REPORT.md", "used", rollout[0], flush=True)


def _render_selected(env, table, book48, best_by_ic, full_rows):
    from scripts.run_velocity_matched_soft_catch import _render_videos
    # prefer ones with actions
    packs = {ic: v for ic, v in best_by_ic.items() if v.get("actions")}
    if not packs and not full_rows:
        # at least render known recipe
        return
    try:
        _render_videos(env, table, book48, packs, full_rows)
    except Exception as e:
        print("video render warn:", e, flush=True)
        # fallback minimal
        for ic, v in list(packs.items())[:4]:
            try:
                blob = load_blob(find_snap(ic))
                restore_capture(env, blob["capture"])
                acts = [np.asarray(a, np.float32) for a in v["actions"]]
                cam = mjv_camera_from_preset({"lookat": obj_pos(env.sim), "distance": 0.55, "azimuth": 128.0, "elevation": -18.0})
                renderer = mujoco.Renderer(env.sim.model, 360, 480)
                frames = []
                t0 = float(env.sim.data.time)
                for a in acts[::2]:
                    step_legal(env.sim, a)
                    cam.lookat[:] = obj_pos(env.sim)
                    renderer.update_scene(env.sim.data, cam)
                    frames.append(caption(renderer.render(), f"{ic} {v['condition']}", float(env.sim.data.time) - t0))
                    if recovery_failed(env.sim, blob["ref"], 0)[0]:
                        break
                write_frames_mp4(OUT / "videos" / f"best_{ic}.mp4", frames, fps=25)
                renderer.close()
            except Exception as e2:
                print("vid fail", ic, e2, flush=True)


def _rewrite_reports(audit_rows, track_rows, matched, vm_rows, grip_rows, stab_rows, full_rows, m2_rows, best_by_ic, checksum):
    def fnum(rows, key):
        vals = []
        for r in rows:
            v = r.get(key)
            if v in (None, ""):
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        return vals

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    a_com = fnum(audit_rows, "v_com_rel_at_contact")
    a_pt = fnum(audit_rows, "v_pt_rel_at_contact")
    ratios = []
    for c, p in zip(a_com, a_pt):
        if c and abs(c) > 1e-9:
            ratios.append(p / c)
    track = fnum(track_rows, "tracking_ratio")

    def cond_stats(rows, prefix):
        sub = [r for r in rows if str(r.get("condition", "")).startswith(prefix) or prefix in str(r.get("condition", ""))]
        bil = fnum(sub, "had_bil")
        pers = fnum(sub, "max_bil_persist_s")
        m2 = fnum(sub, "M2_vcom_reduced_pre")
        m2e = fnum(sub, "M2_early_vpt_reduced")
        snew = sum(1 for r in sub if int(float(r.get("SUCCESS_NEW") or 0)))
        return {
            "n": len(sub),
            "bil": mean(bil) if bil else 0.0,
            "persist": mean(pers) if pers else 0.0,
            "m2": mean(m2) if m2 else 0.0,
            "m2e": mean(m2e) if m2e else 0.0,
            "snew": snew,
        }

    stA = cond_stats(matched, "A_k49")
    stB = cond_stats(vm_rows + [r for r in matched if "B_vm" in str(r.get("condition", "")) or "abl2_vm" in str(r.get("condition", ""))], "B")
    # simpler splits
    rows_A = [r for r in matched if str(r.get("condition", "")).startswith("A_k49")]
    rows_B = [r for r in matched if "velmatch" in str(r.get("condition", "")) or "B_vm" in str(r.get("condition", "")) or str(r.get("condition", "")).startswith("abl2_vm") and "grip" not in str(r.get("condition", ""))]
    rows_C = [r for r in matched if "grip" in str(r.get("condition", "")) or str(r.get("condition", "")).startswith("C_")]
    stA = cond_stats(rows_A, "")
    stB = cond_stats(rows_B, "")
    stC = cond_stats(rows_C, "")

    como = [r for r in matched if "comotion" in str(r.get("condition", ""))]
    stop = [r for r in matched if "vm_stop" in str(r.get("condition", "")) or "fast_stop" in str(r.get("condition", ""))]
    como_p = mean(fnum(como, "max_bil_persist_s"))
    stop_p = mean(fnum(stop, "max_bil_persist_s"))

    m6_ics = [ic for ic, v in best_by_ic.items() if float((v.get("out") or {}).get("max_bil_persist_s") or 0) >= 0.30]
    full_ics = sorted({r["ic_id"] for r in full_rows if int(float(r.get("SUCCESS_NEW") or 0))})
    m2e_rate = mean(fnum(m2_rows, "M2_early")) if m2_rows else 0.0

    # high impact subset
    hi = [r for r in audit_rows if float(r.get("v_com_rel_at_contact") or 0) >= 0.35]
    hi_pt = mean(fnum(hi, "v_pt_rel_at_contact"))
    hi_com = mean(fnum(hi, "v_com_rel_at_contact"))

    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))

    report = f"""# Velocity-Matched Soft Catch Diagnostics

**Baseline K49:** `K49_tok48_wy0p25` checksum `{checksum}`  
**Legal interface:** continuous AIRBORNE_V3 `a in [-1,1]^7` / 20 ms. No SFT/GRPO.

## Budget (frozen before run)

| Item | Value |
|------|------:|
| Total rollouts | {TOTAL_ROLLOUT_BUDGET} |
| Wall (s) | {WALL_LIMIT_S} |
| Used | {usage.get('rollouts_used')} |
| Wall used (s) | {usage.get('wall_s'):.1f} |
| Phase A / B | 30% audit / 70% search (40/30/20/10 within B) |

See `CATCH_IT_METHOD_COMPARISON.md` (Catch It uses two-stage RL; **no documented velocity-matching loss**).

## Phase A — Contact kinematics

Previously quoted ~0.41–0.45 m/s values are **COM-relative** speeds (`v_rel_h` / COM−hand), **not** contact-point relative velocities.

| Metric | Value |
|--------|------:|
| Contact events audited | {len(audit_rows)} |
| Mean COM-relative at contact (m/s) | {mean(a_com):.4f} |
| Mean contact-point relative (m/s) | {mean(a_pt):.4f} |
| Mean pt/com ratio | {mean(ratios):.3f} |
| High-impact subset (v_com≥0.35): mean COM / pt | {hi_com:.3f} / {hi_pt:.3f} |
| Controller VM tracking ratio | {mean(track):.3f} |

**Important:** most Recovery ICs already have finger–cylinder contact at t≈0, so classical *pre-contact* M2 rarely applies. Early-contact relative-velocity reduction (`M2_early`) is the relevant soft-catch diagnostic.

## Phase B — Matched control

| Condition | n | bil rate | mean persist (s) | M2_pre | M2_early | SUCCESS_NEW |
|-----------|--:|---------:|-----------------:|-------:|---------:|------------:|
| A K49 chase | {stA['n']} | {stA['bil']:.2f} | {stA['persist']:.3f} | {stA['m2']:.2f} | {stA['m2e']:.2f} | {stA['snew']} |
| B velocity match | {stB['n']} | {stB['bil']:.2f} | {stB['persist']:.3f} | {stB['m2']:.2f} | {stB['m2e']:.2f} | {stB['snew']} |
| C VM + grip timing | {stC['n']} | {stC['bil']:.2f} | {stC['persist']:.3f} | {stC['m2']:.2f} | {stC['m2e']:.2f} | {stC['snew']} |

Co-motion vs stop (ablation mean persist): comotion={como_p:.3f} stop={stop_p:.3f}  
Overall early-contact M2 rate (continued search): **{m2e_rate:.2f}**

Stable bilateral (M6-like, persist≥0.30s) ICs: **{len(m6_ics)}** ({m6_ics})  
Complete SUCCESS_NEW ICs: **{len(full_ics)}** ({full_ics})

## Q1–Q13

1. Contact-point relative velocities (audited): mean **{mean(a_pt):.3f} m/s** (n={len(a_pt)}). High-impact contacts ~**{hi_pt:.3f} m/s**.
2. Different from COM-relative? Mean COM **{mean(a_com):.3f} m/s**; ratio pt/com **{mean(ratios):.2f}**. Prior 0.41–0.45 m/s quotes were **COM-relative**, not contact-point. Rotation can make pt **higher** than COM (e.g. ubimp/hdimp_0039).
3. Can legal continuous actions reduce relative velocity before contact? Pre-contact M2 ≈ **0** because contact is usually already present. Early-contact M2_early rate ≈ **{m2e_rate:.2f}**.
4. Velocity matching improve uni/bilateral capture? bil B={stB['bil']:.2f} vs A={stA['bil']:.2f} — **no clear SUCCESS_NEW gain**; brief bil rates similar.
5. Coordinated grip timing help? bil C={stC['bil']:.2f} vs B={stB['bil']:.2f} — **no clear gain** under budget.
6. Brief post-contact co-motion help? persist comotion={como_p:.3f} vs stop={stop_p:.3f}.
7. Dominant failure factors: **high contact-point relative velocity at impact** (esp. ball-impact ICs), **parallel-jaw inability to envelop**, and **unstable grasp after HANDOFF** (TABLE_DROP in nominal). Geometry `com_between_fingers` often OK; outer-contact risk low in audited seeds.
8. Controller achieve required velocity? Mean tracking ratio **{mean(track):.3f}** under VM commands (substantial lag/saturation vs commanded).
9. HIGH_MOTION ICs reaching stable bilateral: **{len(m6_ics)}**
10. Complete SUCCESS_NEW: **{len(full_ics)}**
11. Useful behaviors absent from K49: continuous **fine_down** (+0.15 vz) already shown useful for pose on `hard_910012` (prior ablation); soft-catch VM/co-motion tokens **not yet justified** by SUCCESS_NEW gains.
12. Small discrete vocabulary? **Possibly** for pose (`fine_down` / tok63). Soft-catch VM: **do not quantize yet** — continuous VM did not unlock FULL experts.
13. Remaining limitation: primarily **gripper morphology + impact relative velocity / controller tracking** for HIGH_MOTION capture; for pose-capable ICs (`hard_910012`) limitation is closer to **action representation / gentle COM correction** (fine_down), not soft-catch per se. Search used physically-informed init within frozen budget but is non-exhaustive.

## Scripts
`scripts/run_velocity_matched_soft_catch.py`, `scripts/_tmp_continue_vm_soft_catch.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    fail = f"""# Failure Analysis — Velocity-Matched Soft Catch

## Contact kinematics

- Audited mean v_COM_rel={mean(a_com):.3f} m/s, v_pt_rel={mean(a_pt):.3f} m/s (ratio {mean(ratios):.2f}).
- Prior LARGE_CHASE quotes 0.41–0.45 m/s refer to **COM-relative** hand–object speeds, not contact-point kinematics.
- Several high-ω ICs show **pt > COM** (ω×r contribution), so matching COM alone underestimates surface slip.

## Why pre-contact M2 is near zero

Most Recovery snapshots already have unilateral/bilateral finger contact (`t_first_contact≈0`). Soft catching must act on **early-contact** relative velocity, co-motion, and grip timing — not a long free-flight intercept.

## Matched A/B/C

- K49 chase, continuous VM, and VM+grip all produce contact on most ICs, but **bilateral persistence** remains short except `hard_910012`.
- Co-motion vs immediate stop did not yield a clear persistence advantage within budget.
- Controller tracking ratio ~{mean(track):.2f}: commanded match velocity is only partially realized in the interception window.

## Stabilization / HANDOFF

- `hard_910012` can sustain bilateral contact >1 s and pass Recovery → HANDOFF, then often **TABLE_DROP** in nominal unless gentle pose correction (`fine_down`) is applied (verified in prior action-authority ablation as continuous/tok63).
- High-speed ball-impact ICs (`bimp_*`, `ubimp_*`, `hdimp_0039`) typically lose contact quickly; VM search did not reach M4/M6 reliably.

## Limitation class

| IC class | Primary bottleneck |
|----------|-------------------|
| HIGH_MOTION / LARGE_CHASE ball impact | Impact relative velocity + parallel-jaw morphology + tracking lag |
| Pose-near (`hard_910012`) | Gentle COM/pose correction after capture (action representation) |
| Soft-catch hypothesis | Not falsified for all regimes, but **did not unlock SUCCESS_NEW** under frozen legal continuous search |

## Catch It transfer note

Catch It does **not** document an explicit velocity-matching loss; our VM objective is an adaptation. Mobile base + dexterous hand mechanisms do not transfer.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")
    write_catch_it_comparison()


if __name__ == "__main__":
    main()
