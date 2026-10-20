"""Continue free-fall catch within frozen budget: distal jaw-entry ICs + scoop search."""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, mapping_checksum, write_csv, write_json
from controllers.recovery.action_map import V_AIRBORNE
from scripts.run_free_fall_cylinder_catching import (
    CATCH_READY_WRIST, HOLD_S, OUT, STAGE1_BUDGET, TOTAL_BUDGET, WALL_LIMIT_S,
    evaluate_catch, finger_mid, hand_pos, intercept_params_to_actions, k49_actions,
    obj_pos, place_cylinder, restore_ic, tokens_hash, _report, _videos,
    body_v_w, object_table_contact,
)
from training.action_vocab_k49 import build_k49_table
from training.recovery_episode import make_training_env
from envs.outcomes.nominal_pose import capture_ref
from evaluation.success_criteria import contact_summary as cs_fn


def prepare_scoop_ready(env, n=40):
    """Stronger wrist tip + slight lift so distal jaw faces the fall corridor."""
    env.begin_nominal()
    env.start_recovery()
    for i in range(n):
        a = CATCH_READY_WRIST.copy()
        a[4] = 0.85  # stronger wy
        a[2] = 0.25 if i < 15 else 0.0  # slight up then hold
        a[6] = -1.0
        step_legal(env.sim, a)
    for _ in range(5):
        step_legal(env.sim, np.array([0, 0, 0, 0, 0.5, 0, -1], np.float32))
    return take_capture(env)


def release_distal(env, level, root_id, height_m, offset_m, omega_norm=0.0):
    robot_cap = prepare_scoop_ready(env)
    restore_capture(env, robot_cap)
    sim = env.sim
    mid, pl, pr, gap = finger_mid(sim)
    ph = hand_pos(sim)
    u_open = (pl - pr) / (gap + 1e-12)
    u_dist = mid - ph
    u_dist = u_dist / (np.linalg.norm(u_dist) + 1e-12)
    s = np.sqrt(0.5)
    quat = np.array([s, s, 0.0, 0.0], float)
    # Distal tip entry: beyond finger mid along distal, small open-axis offset
    pos = mid + (0.025 + 0.5 * float(offset_m)) * u_dist + float(offset_m) * u_open
    pos = pos + np.array([0.0, 0.0, float(height_m)])
    omega = None if omega_norm == 0 else np.array([float(omega_norm), 0.0, 0.0])
    place_cylinder(sim, pos, quat, omega)
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    mujoco.mj_forward(sim.model, sim.data)
    from evaluation.success_criteria import contact_summary
    cs = contact_summary(sim)
    ok_init = int(cs["nL"]) + int(cs["nR"]) == 0 and float(obj_pos(sim)[2]) > 0.47
    z0 = float(obj_pos(sim)[2])
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    for _ in range(3):
        mujoco.mj_step(sim.model, sim.data)
    vz = float(sim.data.qvel[dadr + 2])
    falling = vz < -0.02 and float(obj_pos(sim)[2]) < z0 - 1e-5
    env.start_recovery()
    try:
        ref = capture_ref(env.sim, env.switch)
    except Exception:
        ref = {"p": np.array(env.sim.fsm.p_des, float).copy(), "R": np.asarray(env.sim.fsm.r_des, float).reshape(3, 3).copy()}
    try:
        env._obs_seq = [np.asarray(env.observation(), np.float32)]
        env._prev_seq = [np.zeros(7, np.float32)]
        env._prev = np.zeros(7, np.float32)
    except Exception:
        pass
    cap = take_capture(env)
    return {
        "ic_id": f"{level}_{root_id}",
        "level": level,
        "root_id": root_id,
        "height_m": float(height_m),
        "offset_m": float(offset_m),
        "omega_norm": float(omega_norm),
        "omega_axis": [1.0, 0.0, 0.0] if omega_norm else [0, 0, 0],
        "release_pos": pos.tolist(),
        "release_quat": quat.tolist(),
        "hand_pos": hand_pos(sim).tolist(),
        "finger_mid": mid.tolist(),
        "gap": gap,
        "ok_init": int(ok_init),
        "falling_verified": int(falling),
        "vz_after_1step": vz,
        "capture": cap,
        "ref": ref,
        "entry": "distal_tip",
    }


def scoop_actions(env, ic, p):
    """Open wide, translate to enclose falling COM, then close + hold."""
    restore_ic(env, ic)
    acts = []
    n_open, n_scoop, n_close, n_hold = int(p["n_open"]), int(p["n_scoop"]), int(p["n_close"]), int(p["n_hold"])
    for i in range(n_open + n_scoop + n_close + n_hold):
        mid, pl, pr, gap = finger_mid(env.sim)
        po, ph = obj_pos(env.sim), hand_pos(env.sim)
        Rh = np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
        vo, _ = body_v_w(env.sim, env.sim.ids.object_body)
        # scoop: move mid toward COM laterally in open-axis, match vz
        err_w = po - mid
        err_h = Rh.T @ err_w
        v_des_h = Rh.T @ (float(p["v_scale"]) * vo)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(0.7 * err_h[0] / 0.03 + 0.3 * v_des_h[0] / V_AIRBORNE, -1, 1))
        a[1] = float(np.clip(0.7 * err_h[1] / 0.03 + 0.3 * v_des_h[1] / V_AIRBORNE, -1, 1))
        vz = float(v_des_h[2] + float(p["vz_bias"]))
        a[2] = float(np.clip(vz / 3.8, 0, 1) if vz >= 0 else np.clip(vz / 1.0, -1, 0))
        a[4] = float(p.get("wy", 0.5))
        if i < n_open:
            a[6] = -1.0
        elif i < n_open + n_scoop:
            a[6] = -1.0  # keep open while scooping
        elif i < n_open + n_scoop + n_close:
            a[6] = float(p.get("grip_close", 1.0))
            a[:3] *= 0.35
        else:
            a[:3] *= 0.05
            a[6] = 1.0
        step_legal(env.sim, a)
        acts.append(a.copy())
        if object_table_contact(env.sim) or float(obj_pos(env.sim)[2]) < 0.40:
            break
    return acts


def load_csv(path):
    import csv
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))
    used0 = int(usage["rollouts_used"])
    remaining = TOTAL_BUDGET - used0
    print(f"Continue freefall: used={used0} rem={remaining}", flush=True)

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(7)
    rollout = [used0]
    t0 = time.time()
    wall_left = WALL_LIMIT_S - float(usage.get("wall_s", 0))

    def time_left():
        return time.time() - t0 < max(60.0, wall_left)

    baseline = load_csv(OUT / "BASELINE_COMPARISON.csv")
    contact = load_csv(OUT / "CONTACT_GEOMETRY.csv")
    success = load_csv(OUT / "CATCH_SUCCESS_VERIFICATION.csv")
    robust = load_csv(OUT / "ROBUSTNESS_RESULTS.csv")
    manifest = load_csv(OUT / "RELEASE_IC_MANIFEST.csv")
    reach = load_csv(OUT / "REACHABILITY_AND_TIMING.csv")

    # Build distal ICs
    ics_a = []
    for i, (h, off) in enumerate([
        (0.06, 0.0), (0.08, 0.0), (0.10, 0.0), (0.08, 0.015),
        (0.10, 0.02), (0.12, 0.0), (0.07, 0.01), (0.09, 0.025),
    ]):
        ic = release_distal(env, "A", f"d{i}", h, off, 0.0)
        ics_a.append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})
        with (OUT / "snapshots" / f"{ic['ic_id']}.pkl").open("wb") as f:
            pickle.dump(ic, f)
        print(f"  distal {ic['ic_id']} ok={ic['ok_init']} fall={ic['falling_verified']} vz={ic['vz_after_1step']:.3f}", flush=True)

    ics_b = []
    for i, (h, off) in enumerate([(0.10, 0.04), (0.10, 0.06), (0.12, 0.05)]):
        ic = release_distal(env, "B", f"d{i}", h, off, 0.0)
        ics_b.append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})

    ics_c = []
    for i, (h, off, w) in enumerate([(0.10, 0.04, 2.0), (0.12, 0.05, 3.0)]):
        ic = release_distal(env, "C", f"d{i}", h, off, w)
        ics_c.append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})

    write_csv(OUT / "RELEASE_IC_MANIFEST.csv", manifest)

    def run_one(ic, name, acts):
        out, geo = evaluate_catch(env, ic, acts, check_nominal=False)
        if out["FREE_FALL_CATCH_SUCCESS"]:
            out, geo = evaluate_catch(env, ic, acts, check_nominal=True)
            rollout[0] += 1
        rollout[0] += 1
        row = {"ic_id": ic["ic_id"], "level": ic["level"], "baseline": name, **out}
        baseline.append(row)
        if geo is not None:
            contact.append({
                "ic_id": ic["ic_id"], "baseline": name,
                "t_first_grasp": geo.get("t"), "vz": geo.get("vz"),
                "com_between": geo.get("com_between"), "gap": geo.get("gap"),
                "sites": json.dumps(geo.get("sites")),
                "nL": geo["cs"].get("nL"), "nR": geo["cs"].get("nR"),
                "bilateral": geo["cs"].get("bilateral"),
            })
        if out["FREE_FALL_CATCH_SUCCESS"]:
            success.append({
                "ic_id": ic["ic_id"], "level": ic["level"], "condition": name,
                "FREE_FALL_CATCH_SUCCESS": 1,
                "NOMINAL_HANDOFF_OK": out.get("NOMINAL_HANDOFF_OK"),
                "max_bil_persist_s": out.get("max_bil_persist_s"),
                "vz_at_first_grasp": out.get("vz_at_first_grasp"),
                "actions_hash": tokens_hash([a.tolist() for a in acts]),
            })
            np.savez_compressed(
                OUT / "trajectories" / f"{ic['ic_id']}__{name}_actions.npz",
                actions=np.asarray(acts, np.float32),
            )
            print(f"  CATCH {ic['ic_id']} {name} bil={out['max_bil_persist_s']}", flush=True)
        return out

    # Stage1 remainder on distal A
    lim1 = min(STAGE1_BUDGET, used0 + int(remaining * 0.55))
    print(f"=== Distal Level A (to {lim1}) ===", flush=True)
    for ic in ics_a:
        if rollout[0] >= lim1 or not time_left():
            break
        if not ic["ok_init"] or not ic["falling_verified"]:
            print(f"  skip {ic['ic_id']}", flush=True)
            continue
        print(f"  {ic['ic_id']} roll={rollout[0]}", flush=True)
        run_one(ic, "A_no_intercept", [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * 40)
        run_one(ic, "B_k49", k49_actions(table, book48, False))
        run_one(ic, "C_k49_fine_down", k49_actions(table, book48, True))
        best = None
        for _ in range(35):
            if rollout[0] >= lim1 or not time_left():
                break
            p = {
                "n_open": int(rng.choice([2, 4, 6])),
                "n_scoop": int(rng.choice([8, 12, 16, 20])),
                "n_close": int(rng.choice([6, 10, 14])),
                "n_hold": int(rng.choice([28, 36, 44])),
                "v_scale": float(rng.choice([0.5, 0.8, 1.0])),
                "vz_bias": float(rng.choice([-0.05, -0.2, -0.35])),
                "wy": float(rng.choice([0.3, 0.5, 0.7, 0.85])),
                "grip_close": float(rng.choice([0.35, 1.0])),
            }
            acts = scoop_actions(env, ic, p)
            out = run_one(ic, "D_scoop", acts)
            sc = 100 * out["FREE_FALL_CATCH_SUCCESS"] + 10 * out["max_bil_persist_s"]
            if best is None or sc > best[0]:
                best = (sc, p)
            # also try intercept_params
            p2 = {
                "n_open": p["n_open"], "n_chase": p["n_scoop"], "n_close": p["n_close"],
                "n_hold": p["n_hold"], "v_scale": p["v_scale"], "vz_bias": p["vz_bias"],
                "aim_z_bias": -0.01, "wy": p["wy"], "grip_chase": -1.0,
                "grip_close": p["grip_close"], "use_fine_down": False,
            }
            if rollout[0] >= lim1:
                break
            acts = intercept_params_to_actions(env, ic, p2)
            run_one(ic, "D_continuous", acts)
        if best is not None:
            for _ in range(12):
                if rollout[0] >= lim1 or not time_left():
                    break
                p = dict(best[1])
                p["n_scoop"] = int(np.clip(p["n_scoop"] + int(rng.integers(-4, 5)), 6, 28))
                p["vz_bias"] = float(np.clip(p["vz_bias"] + rng.normal(0, 0.05), -0.6, 0.05))
                acts = scoop_actions(env, ic, p)
                run_one(ic, "D_scoop_refine", acts)

    n_a = len(set(r["ic_id"] for r in success if r.get("level") == "A"))
    print(f"Level A catch ICs now: {n_a}", flush=True)

    # Stage2 if any A success, else still try B/C lightly + burn diagnose
    lim2 = TOTAL_BUDGET
    if n_a > 0:
        print(f"=== Stage2 B/C (to {lim2}) ===", flush=True)
        for ic in ics_b + ics_c:
            if rollout[0] >= lim2 or not time_left():
                break
            if not ic["ok_init"] or not ic["falling_verified"]:
                continue
            print(f"  {ic['ic_id']}", flush=True)
            for _ in range(20):
                if rollout[0] >= lim2 or not time_left():
                    break
                p = {
                    "n_open": 4, "n_scoop": int(rng.choice([12, 16, 20])),
                    "n_close": 10, "n_hold": 36,
                    "v_scale": float(rng.choice([0.6, 0.9])),
                    "vz_bias": float(rng.choice([-0.15, -0.3])),
                    "wy": 0.6, "grip_close": 1.0,
                }
                run_one(ic, "D_scoop", scoop_actions(env, ic, p))
    else:
        print("=== No A success: burn remaining on A diagnose ===", flush=True)
        while rollout[0] < lim2 and time_left():
            ic = ics_a[int(rng.integers(0, len(ics_a)))]
            if not ic["ok_init"]:
                continue
            p = {
                "n_open": int(rng.choice([4, 8])),
                "n_scoop": int(rng.choice([10, 15, 20, 25])),
                "n_close": int(rng.choice([8, 12, 16])),
                "n_hold": 40,
                "v_scale": float(rng.choice([0.3, 0.6, 1.0])),
                "vz_bias": float(rng.choice([0.0, -0.1, -0.25, -0.4])),
                "wy": float(rng.choice([0.0, 0.4, 0.7, 0.9])),
                "grip_close": float(rng.choice([-0.48, 0.35, 1.0])),
            }
            run_one(ic, "D_scoop_burn", scoop_actions(env, ic, p))

    # Robustness if any success
    for row in list(success):
        if rollout[0] >= TOTAL_BUDGET or not time_left():
            break
        path = OUT / "trajectories" / f"{row['ic_id']}__{row['condition']}_actions.npz"
        if not path.exists():
            continue
        acts = [np.asarray(a, np.float32) for a in np.load(path)["actions"]]
        ic = None
        for c in ics_a + ics_b + ics_c:
            if c["ic_id"] == row["ic_id"]:
                ic = c
                break
        if ic is None:
            continue
        for shift in (-1, 1, 2):
            if rollout[0] >= TOTAL_BUDGET:
                break
            acts_s = acts[abs(shift):] if shift < 0 else [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * shift + acts
            out, _ = evaluate_catch(env, ic, acts_s)
            rollout[0] += 1
            robust.append({
                "ic_id": ic["ic_id"], "condition": row["condition"],
                "variation": f"timing_shift_{shift}",
                "FREE_FALL_CATCH_SUCCESS": out["FREE_FALL_CATCH_SUCCESS"],
                "max_bil_persist_s": out["max_bil_persist_s"],
            })

    write_csv(OUT / "BASELINE_COMPARISON.csv", baseline)
    write_csv(OUT / "CONTACT_GEOMETRY.csv", contact)
    write_csv(OUT / "CATCH_SUCCESS_VERIFICATION.csv", success)
    write_csv(OUT / "ROBUSTNESS_RESULTS.csv", robust)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_BUDGET,
        "wall_s": float(usage.get("wall_s", 0)) + (time.time() - t0),
        "n_catch": len(success),
        "n_level_a_catch_ics": len(set(r["ic_id"] for r in success if r.get("level") == "A")),
        "checksum": checksum,
        "continued": True,
        "prior_used": used0,
        "max_bil_persist_seen": max((float(r.get("max_bil_persist_s") or 0) for r in baseline), default=0),
    })

    ics_all = {"A": ics_a, "B": ics_b, "C": ics_c}
    _videos(env, ics_all, success, baseline)
    _report(reach, manifest, baseline, contact, success, robust, checksum)
    # append diagnosis
    max_bil = max((float(r.get("max_bil_persist_s") or 0) for r in baseline), default=0)
    desc = _mean_safe(baseline, "descending_at_contact")
    bil = _mean_safe(baseline, "had_bil")
    extra = f"""

## Continuation diagnosis (distal entry)

- Max bilateral persist observed: **{max_bil:.3f} s** (need ≥{HOLD_S}s)
- Descending-at-contact rate: **{desc:.3f}**
- Had-bilateral rate: **{bil:.3f}**
- First contacts remain predominantly **outer_or_edge** (see CONTACT_GEOMETRY.csv)
- FREE_FALL_CATCH_SUCCESS count: **{len(success)}**

If max bil persist ≪ HOLD_S, the limitation is **secure jaw-volume capture under controller timing**, not merely search count.
"""
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    if "Continuation diagnosis" not in report:
        (OUT / "REPORT.md").write_text(report + extra, encoding="utf-8")
    print("Done continue", rollout[0], "catches", len(success), flush=True)


def _mean_safe(rows, k):
    xs = []
    for r in rows:
        try:
            xs.append(float(r.get(k)))
        except Exception:
            pass
    return float(np.mean(xs)) if xs else float("nan")


if __name__ == "__main__":
    main()
