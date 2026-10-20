"""Free-fall cylinder catching feasibility with existing Panda / AIRBORNE_V3.

New diagnostic benchmark — not mixed into K49 recovery experts.
FREE_FALL_CATCH_SUCCESS is frozen before search (see BENCHMARK_DEFINITION.md).
No SFT/GRPO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, V_AIRBORNE, map_action
from envs.outcomes import table_drop as td
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import take_capture
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary, window_stats as window_stats_fo
from evaluation.grasp_configuration import cylinder_axis_world, measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "free_fall_cylinder_catching"

# ---- Frozen success criterion (before any search) ----
HOLD_S = 0.50
HOLD_VZ_MAX = 0.08          # |vz| after arrest
DESCEND_VZ = -0.05          # must be descending at first grasp contact
TABLE_Z = 0.42              # table interference proxy (object COM)
STAGE1_BUDGET = 900
STAGE2_BUDGET = 900
TOTAL_BUDGET = STAGE1_BUDGET + STAGE2_BUDGET
WALL_LIMIT_S = 45 * 60

FINE_DOWN = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0], np.float32)

# Catch-ready wrist bias: pitch opening toward +world-ish intercept (documented)
CATCH_READY_WRIST = np.array([0.0, 0.0, 0.0, 0.0, 0.55, 0.0, -1.0], np.float32)


def tokens_hash(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def hand_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.hand_body], float)


def obj_pos(sim):
    return np.asarray(sim.data.xpos[sim.ids.object_body], float)


def body_v_w(sim, bid):
    v, w = body_twist(sim.model, sim.data, int(bid))
    return np.asarray(v, float), np.asarray(w, float)


def finger_mid(sim):
    pl = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    pr = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    return 0.5 * (pl + pr), pl, pr, float(np.linalg.norm(pl - pr))


def table_geom(sim):
    return int(mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table"))


def object_table_contact(sim):
    tg = table_geom(sim)
    obj = int(sim.ids.object_body)
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(sim.model.geom_bodyid[g1])
        b2 = int(sim.model.geom_bodyid[g2])
        if obj not in (b1, b2):
            continue
        og = g2 if b1 == obj else g1
        if og == tg:
            return True
    return False


def hand_object_contact_sites(sim):
    """Classify finger/hand contacts for geometry audit."""
    obj = int(sim.ids.object_body)
    left, right, hand = int(sim.ids.left_body), int(sim.ids.right_body), int(sim.ids.hand_body)
    mid, pl, pr, gap = finger_mid(sim)
    u_open = (pl - pr) / (gap + 1e-12)
    out = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        pos = np.asarray(c.pos, float)
        wrench = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
        except Exception:
            pass
        if other == left:
            side = "L"
        elif other == right:
            side = "R"
        elif other == hand:
            side = "hand"
        else:
            side = "other"
        t = float(np.dot(pos - pr, u_open))
        between = 0.0 < t < gap
        site = "inside" if between and side in ("L", "R") else (
            "hand_palm" if side == "hand" else "outer_or_edge" if side in ("L", "R") else "other"
        )
        out.append({
            "side": side, "site": site, "between": int(between),
            "fn": float(abs(wrench[0])), "pos": pos.tolist(),
        })
    return out


def write_benchmark_definition():
    (OUT / "BENCHMARK_DEFINITION.md").write_text(
        f"""# Free-Fall Cylinder Catching — Benchmark Definition

## Scope

Diagnostic benchmark isolating **dynamic interception and capture** of a freely falling
cylinder under the **unchanged** Panda / AIRBORNE_V3 stack. Not the ball-impact recovery task.
Not mixed into the K49 recovery expert dataset.

## Frozen success: `FREE_FALL_CATCH_SUCCESS`

Frozen **before** search (HOLD_S = **{HOLD_S} s**).

All of the following must hold on a full MuJoCo replay:

1. At release: cylinder airborne; **no** robot–object contact; not pre-grasped.
2. First finger–object grasp contact occurs while cylinder COM vertical velocity `vz ≤ {DESCEND_VZ}` (descending).
3. Bilateral finger–object contact is established.
4. Cylinder does **not** make table contact before bilateral capture is established.
5. Bilateral contact is sustained for ≥ **{HOLD_S} s** continuously.
6. During the hold window, mean `|vz| ≤ {HOLD_VZ_MAX}` m/s (downward motion arrested).
7. Cylinder does not escape (lose all finger contact for >0.1 s) during the hold+verify window.

Unilateral contact, transient pinch, or a brief height bounce alone **do not** count.

### Separate (reported, not required)

`NOMINAL_HANDOFF_OK`: after catch, apply unchanged HANDOFF + nominal continuation and
record whether the original SUCCESS_NEW window would pass. **Not** conflated with free-fall catch success.

## Levels

| Level | Lateral offset | Initial ω | Notes |
|-------|----------------|-----------|-------|
| A | ~0 (aligned to jaw) | 0 | Basic; cylinder still falling at first grasp contact |
| B | nonzero | 0 | Reposition before intercept |
| C | nonzero | nonzero (documented) | Rotating free-fall |

## Baselines

A no-intercept / hold-open · B K49 tokens · C K49+`fine_down` · D continuous legal 7D search.

## Morphology note

With the default downward gripper, the palm lies above the jaw gap. Pure overhead drop
into the jaws is occluded by the hand. Level A therefore uses a **catch-ready wrist**
initialization and/or **side-entry** release so the cylinder can enter the jaw volume
while descending — still under gravity after release, no mid-flight teleport.
""",
        encoding="utf-8",
    )


def prepare_catch_ready(env, n_wrist=25):
    """Move to catch-ready wrist (open grip) via legal actions from reset home."""
    env.begin_nominal()
    env.start_recovery()
    # open + pitch wrist to tip opening toward a side-entry intercept
    for _ in range(n_wrist):
        step_legal(env.sim, CATCH_READY_WRIST)
    # settle open
    for _ in range(5):
        step_legal(env.sim, np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    return take_capture(env)


def place_cylinder(sim, pos, quat, omega=None):
    q = int(sim.model.jnt_qposadr[sim.ids.object_jnt])
    d = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    sim.data.qpos[q : q + 3] = np.asarray(pos, float)
    sim.data.qpos[q + 3 : q + 7] = np.asarray(quat, float)
    sim.data.qvel[d : d + 6] = 0.0
    if omega is not None:
        sim.data.qvel[d + 3 : d + 6] = np.asarray(omega, float)
    mujoco.mj_forward(sim.model, sim.data)


def release_ic(env, level, root_id, rng, height_m, offset_m, omega_norm=0.0):
    """Construct one release IC. Returns dict with capture after release instant."""
    robot_cap = prepare_catch_ready(env)
    restore_physics_only = False  # full restore via take after place
    # restore robot catch-ready
    from envs.simulation.capture import restore_capture
    restore_capture(env, robot_cap)
    sim = env.sim
    mid, pl, pr, gap = finger_mid(sim)
    # horizontal cylinder (axis ~ world Y), same as seated quat
    s = np.sqrt(0.5)
    quat = np.array([s, s, 0.0, 0.0], float)
    # Side-entry: offset along jaw open axis so path clears palm; height above finger mid
    u_open = (pl - pr) / (gap + 1e-12)
    # distal tip direction
    ph = hand_pos(sim)
    u_dist = mid - ph
    u_dist = u_dist / (np.linalg.norm(u_dist) + 1e-12)

    if level == "A":
        # small lateral offset toward jaw center from slightly outside
        lat = float(offset_m) * u_open  # ~0 for true aligned
        pos = mid + lat + 0.01 * u_dist + np.array([0.0, 0.0, float(height_m)])
        omega = None
    elif level == "B":
        lat = float(offset_m) * u_open
        pos = mid + lat + 0.01 * u_dist + np.array([0.0, 0.0, float(height_m)])
        omega = None
    else:  # C
        lat = float(offset_m) * u_open
        pos = mid + lat + 0.01 * u_dist + np.array([0.0, 0.0, float(height_m)])
        # angular velocity about world X (tumble in YZ) — documented controlled release
        axis = np.array([1.0, 0.0, 0.0])
        omega = axis * float(omega_norm)

    place_cylinder(sim, pos, quat, omega)
    # open fingers fully
    sim.data.qpos[sim.ids.finger_jnt] = 0.04
    mujoco.mj_forward(sim.model, sim.data)

    cs = contact_summary(sim)
    sites = hand_object_contact_sites(sim)
    vo, wo = body_v_w(sim, sim.ids.object_body)
    ok_init = (
        int(cs["nL"]) + int(cs["nR"]) == 0
        and not any(s["side"] in ("L", "R", "hand") for s in sites)
        and float(obj_pos(sim)[2]) > TABLE_Z + 0.05
        and float(np.linalg.norm(vo)) < 0.02
    )
    # Confirm freefall via free-joint qvel (body_twist can lag one step)
    z0 = float(obj_pos(sim)[2])
    dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
    for _ in range(3):
        mujoco.mj_step(sim.model, sim.data)
    vz_q = float(sim.data.qvel[dadr + 2])
    falling = vz_q < -0.02 and float(obj_pos(sim)[2]) < z0 - 1e-5
    vo1 = np.array([0.0, 0.0, vz_q])
    # Re-open / reaffirm recovery mode
    env.start_recovery()
    # build ref from current switch if any
    try:
        ref = capture_ref(env.sim, env.switch)
    except Exception:
        ref = {
            "p": np.array(env.sim.fsm.p_des, float).copy(),
            "R": np.asarray(env.sim.fsm.r_des, float).reshape(3, 3).copy(),
        }
    # ensure obs seq
    try:
        env._obs_seq = [np.asarray(env.observation(), np.float32)]
        env._prev_seq = [np.zeros(7, np.float32)]
        env._prev = np.zeros(7, np.float32)
    except Exception:
        pass
    cap = take_capture(env)
    ic = {
        "ic_id": f"{level}_{root_id}",
        "level": level,
        "root_id": root_id,
        "height_m": float(height_m),
        "offset_m": float(offset_m),
        "omega_norm": float(omega_norm),
        "omega_axis": [1.0, 0.0, 0.0] if level == "C" else [0, 0, 0],
        "release_pos": pos.tolist(),
        "release_quat": quat.tolist(),
        "hand_pos": hand_pos(sim).tolist(),
        "finger_mid": mid.tolist(),
        "gap": gap,
        "ok_init": int(ok_init),
        "falling_verified": int(falling),
        "vz_after_1step": float(vo1[2]),
        "capture": cap,
        "ref": ref,
    }
    return ic


def restore_ic(env, ic):
    from envs.simulation.capture import restore_capture
    restore_capture(env, ic["capture"])
    env.start_recovery()
    try:
        if not env._obs_seq:
            env._obs_seq = [np.asarray(env.observation(), np.float32)]
            env._prev_seq = [np.zeros(7, np.float32)]
    except Exception:
        pass


def evaluate_catch(env, ic, actions, check_nominal=False):
    restore_ic(env, ic)
    sim = env.sim
    t0 = float(sim.data.time)
    first_grasp = None
    first_grasp_vz = None
    bil_run = 0.0
    max_bil = 0.0
    hold_armed = False
    hold_s = 0.0
    hold_vz_acc = []
    table_before_bil = False
    saw_bil = False
    escape = False
    contact_geo = None
    dt = 0.02

    for ai, a in enumerate(actions):
        a = np.clip(np.asarray(a, np.float32).reshape(7), -1, 1)
        dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
        vz_q = float(sim.data.qvel[dadr + 2])
        vo, _ = body_v_w(sim, sim.ids.object_body)
        cs = contact_summary(sim)
        any_finger = int(cs["nL"]) + int(cs["nR"]) > 0
        if object_table_contact(sim) or float(obj_pos(sim)[2]) < TABLE_Z:
            if not saw_bil:
                table_before_bil = True
            break
        if any_finger and first_grasp is None:
            first_grasp = float(sim.data.time) - t0
            first_grasp_vz = float(vz_q if abs(vz_q) > 1e-6 else vo[2])
            contact_geo = {
                "t": first_grasp,
                "vz": first_grasp_vz,
                "sites": hand_object_contact_sites(sim),
                "cs": dict(cs),
                "com_between": None,
            }
            mid, pl, pr, gap = finger_mid(sim)
            po = obj_pos(sim)
            t = float(np.dot(po - pr, (pl - pr) / (gap + 1e-12)))
            contact_geo["com_between"] = int(0.0 < t < gap)
            contact_geo["gap"] = gap

        if cs["bilateral"]:
            saw_bil = True
            bil_run += dt
            max_bil = max(max_bil, bil_run)
            hold_vz_acc.append(abs(float(vo[2])))
            if bil_run + 1e-9 >= HOLD_S and np.mean(hold_vz_acc[-max(1, int(HOLD_S / dt)):]) <= HOLD_VZ_MAX:
                hold_armed = True
                hold_s = bil_run
        else:
            if saw_bil and bil_run > 0 and not any_finger:
                escape = True
            bil_run = 0.0
            hold_vz_acc = []

        step_legal(sim, a)
        env._prev = a.copy()
        if hold_armed and bil_run >= HOLD_S:
            # require a few more steps of sustained hold
            if bil_run >= HOLD_S + 0.1:
                break

    descending_ok = first_grasp is not None and first_grasp_vz is not None and first_grasp_vz <= DESCEND_VZ
    mean_hold_vz = float(np.mean(hold_vz_acc)) if hold_vz_acc else None
    success = bool(
        int(ic.get("ok_init", 0))
        and descending_ok
        and saw_bil
        and (not table_before_bil)
        and max_bil + 1e-9 >= HOLD_S
        and mean_hold_vz is not None and mean_hold_vz <= HOLD_VZ_MAX
        and not escape
    )

    out = {
        "FREE_FALL_CATCH_SUCCESS": int(success),
        "ok_init": int(ic.get("ok_init", 0)),
        "falling_verified": int(ic.get("falling_verified", 0)),
        "t_first_grasp": first_grasp,
        "vz_at_first_grasp": first_grasp_vz,
        "descending_at_contact": int(descending_ok),
        "had_bil": int(saw_bil),
        "max_bil_persist_s": max_bil,
        "mean_hold_vz": mean_hold_vz,
        "table_before_bil": int(table_before_bil),
        "escape": int(escape),
        "obj_z_end": float(obj_pos(sim)[2]),
        "n_actions": len(actions),
    }

    # optional nominal handoff probe (separate)
    out["NOMINAL_HANDOFF_OK"] = 0
    if success and check_nominal:
        try:
            apply_handoff(sim, ic["ref"])
            t_h = float(sim.data.time)
            samples = []
            next_s = 0.0
            reason = None
            from envs.outcomes.horizon_outcome import NOMINAL_S, latched
            while reason is None and float(sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
                nominal_step(sim)
                tr = float(sim.data.time) - t_h
                if tr >= next_s:
                    m = measure(sim, U_TGT, P_TGT)
                    cs = contact_summary(sim)
                    samples.append({
                        "t_rel": tr, "e_theta_deg": m["e_theta_deg"], "e_pos_mm": m["e_pos_mm"],
                        "bilateral": cs["bilateral"], "obj_z": m["obj_z"],
                    })
                    next_s += 0.02
                if latched(sim):
                    reason = "TABLE_DROP"
            win = window_stats_fo(samples, float(NOMINAL_S) - 2.0)
            out["NOMINAL_HANDOFF_OK"] = int(bool(win.get("working_SUCCESS_NEW_hold")))
        except Exception as e:
            out["NOMINAL_HANDOFF_ERR"] = str(e)

    return out, contact_geo


def intercept_params_to_actions(env, ic, p):
    """Physically informed catch sequence from parameters."""
    restore_ic(env, ic)
    acts = []
    n_open = int(p["n_open"])
    n_chase = int(p["n_chase"])
    n_close = int(p["n_close"])
    n_hold = int(p["n_hold"])
    for i in range(n_open + n_chase + n_close + n_hold):
        mid, pl, pr, gap = finger_mid(env.sim)
        po = obj_pos(env.sim)
        ph = hand_pos(env.sim)
        Rh = np.asarray(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
        vo, wo = body_v_w(env.sim, env.sim.ids.object_body)
        # aim point: slightly below COM for intercept
        aim = po + np.array([0.0, 0.0, float(p.get("aim_z_bias", -0.01))])
        err_h = Rh.T @ (aim - ph)
        v_des_h = Rh.T @ (float(p["v_scale"]) * vo)
        a = np.zeros(7, np.float32)
        a[0] = float(np.clip(0.5 * err_h[0] / 0.04 + 0.5 * v_des_h[0] / V_AIRBORNE, -1, 1))
        a[1] = float(np.clip(0.5 * err_h[1] / 0.04 + 0.5 * v_des_h[1] / V_AIRBORNE, -1, 1))
        # vertical: match fall / lower hand
        vz = float(v_des_h[2] + float(p.get("vz_bias", -0.15)))
        a[2] = float(np.clip(vz / 3.8, 0, 1) if vz >= 0 else np.clip(vz / 1.0, -1, 0))
        # wrist: keep opening aligned
        a[4] = float(p.get("wy", 0.2))
        if i < n_open:
            a[6] = -1.0
        elif i < n_open + n_chase:
            a[6] = float(p.get("grip_chase", -0.48))
        elif i < n_open + n_chase + n_close:
            a[6] = float(p.get("grip_close", 1.0))
            a[0] *= 0.5
            a[1] *= 0.5
            a[2] *= 0.3
        else:
            a[:3] *= 0.1
            a[6] = 1.0
            if p.get("use_fine_down") and i > n_open + n_chase + n_close + 2:
                a = FINE_DOWN.copy()
        step_legal(env.sim, a)
        acts.append(a.copy())
        if object_table_contact(env.sim) or float(obj_pos(env.sim)[2]) < TABLE_Z - 0.02:
            break
    return acts


def k49_actions(table, book48, use_fine_down=False):
    toks = [30] * 4 + [17] * 10 + [0] * 4 + [12] * 8 + [SECURE_REVERSE_WRIST] * 12
    acts = [np.asarray(decode(table, book48, int(t)), np.float32) for t in toks]
    if use_fine_down:
        acts = acts[:20] + [FINE_DOWN.copy() for _ in range(10)] + acts[20:]
    return acts


def reachability_audit(env, heights, offsets):
    rows = []
    for h in heights:
        # kinematic fall time ignoring drag: z''=-g
        t_table = float(np.sqrt(2.0 * max(h + 0.15, 1e-6) / 9.81))  # rough to table from above fingers
        # hand reach: command full downward for N steps
        env.begin_nominal()
        env.start_recovery()
        ph0 = hand_pos(env.sim)
        for _ in range(15):
            step_legal(env.sim, np.array([0, 0, -1, 0, 0, 0, -1], np.float32))
        ph1 = hand_pos(env.sim)
        v_down = float((ph0[2] - ph1[2]) / (15 * 0.02))
        for off in offsets:
            rows.append({
                "height_m": h, "offset_m": off,
                "approx_fall_time_s": float(np.sqrt(2 * h / 9.81)),
                "approx_time_to_table_s": t_table,
                "hand_down_speed_mps": v_down,
                "hand_displace_z_0p3s": float(ph0[2] - ph1[2]),
                "steps_to_cover_height": None if v_down < 1e-3 else float(h / (v_down * 0.02)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("raw", "trajectories", "videos", "snapshots"):
        (OUT / d).mkdir(exist_ok=True)
    write_benchmark_definition()
    write_json(OUT / "raw" / "BUDGET.json", {
        "STAGE1_BUDGET": STAGE1_BUDGET,
        "STAGE2_BUDGET": STAGE2_BUDGET,
        "TOTAL_BUDGET": TOTAL_BUDGET,
        "WALL_LIMIT_S": WALL_LIMIT_S,
        "HOLD_S": HOLD_S,
        "recorded_before_execution": True,
    })

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261019)
    rollout = [0]
    t_wall0 = time.time()

    def time_left():
        return time.time() - t_wall0 < WALL_LIMIT_S

    # ---- Reachability ----
    print("=== Reachability audit ===", flush=True)
    heights = [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]
    offsets = [0.0, 0.02, 0.04, 0.06, 0.08]
    reach_rows = reachability_audit(env, heights, offsets)
    write_csv(OUT / "REACHABILITY_AND_TIMING.csv", reach_rows)

    # ---- Build release ICs ----
    print("=== Build release ICs ===", flush=True)
    manifest = []
    ics = {"A": [], "B": [], "C": []}
    # Level A: multiple roots
    for i, (h, off) in enumerate([(0.08, 0.0), (0.10, 0.0), (0.12, 0.01), (0.10, 0.02), (0.15, 0.0), (0.12, 0.0)]):
        ic = release_ic(env, "A", f"r{i}", rng, h, off, 0.0)
        ics["A"].append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})
        print(f"  A_{ic['root_id']} ok={ic['ok_init']} fall={ic['falling_verified']} vz={ic['vz_after_1step']:.3f}", flush=True)
    for i, (h, off) in enumerate([(0.10, 0.04), (0.12, 0.05), (0.10, 0.06), (0.15, 0.04)]):
        ic = release_ic(env, "B", f"r{i}", rng, h, off, 0.0)
        ics["B"].append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})
    for i, (h, off, w) in enumerate([(0.12, 0.04, 2.0), (0.12, 0.05, 3.5), (0.15, 0.04, 2.5)]):
        ic = release_ic(env, "C", f"r{i}", rng, h, off, w)
        ics["C"].append(ic)
        manifest.append({k: v for k, v in ic.items() if k not in ("capture", "ref")})
    write_csv(OUT / "RELEASE_IC_MANIFEST.csv", manifest)
    # persist snapshots (captures)
    import pickle
    for level, lst in ics.items():
        for ic in lst:
            with (OUT / "snapshots" / f"{ic['ic_id']}.pkl").open("wb") as f:
                pickle.dump(ic, f)
    write_json(OUT / "raw" / "ICS_META.json", {
        "nA": len(ics["A"]), "nB": len(ics["B"]), "nC": len(ics["C"]),
        "hold_s": HOLD_S,
    })

    baseline_rows = []
    contact_rows = []
    success_rows = []
    robust_rows = []
    level_a_successes = []

    def run_baseline(ic, name, acts, family_list):
        out, geo = evaluate_catch(env, ic, acts, check_nominal=False)
        if out["FREE_FALL_CATCH_SUCCESS"]:
            out, geo = evaluate_catch(env, ic, acts, check_nominal=True)
            rollout[0] += 1  # second eval counts
        rollout[0] += 1
        row = {
            "ic_id": ic["ic_id"], "level": ic["level"], "baseline": name,
            **{k: out[k] for k in out},
        }
        family_list.append(row)
        baseline_rows.append(row)
        if geo is not None:
            contact_rows.append({
                "ic_id": ic["ic_id"], "baseline": name,
                "t_first_grasp": geo.get("t"), "vz": geo.get("vz"),
                "com_between": geo.get("com_between"), "gap": geo.get("gap"),
                "sites": json.dumps(geo.get("sites")),
                "nL": geo["cs"].get("nL"), "nR": geo["cs"].get("nR"),
                "bilateral": geo["cs"].get("bilateral"),
            })
        if out["FREE_FALL_CATCH_SUCCESS"]:
            success_rows.append({
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
            print(f"  CATCH {ic['ic_id']} {name}", flush=True)
            if ic["level"] == "A":
                level_a_successes.append((ic, name, acts, out))
        return out

    # =====================================================================
    # Stage 1 — Level A
    # =====================================================================
    print(f"=== Stage 1 Level A (budget {STAGE1_BUDGET}) ===", flush=True)
    for ic in ics["A"]:
        if rollout[0] >= STAGE1_BUDGET or not time_left():
            break
        if not ic["ok_init"] or not ic["falling_verified"]:
            print(f"  skip bad IC {ic['ic_id']}", flush=True)
            continue
        print(f"  IC {ic['ic_id']} roll={rollout[0]}", flush=True)
        # A: no intercept — open hold
        acts_none = [np.array([0, 0, 0, 0, 0, 0, -1], np.float32) for _ in range(40)]
        run_baseline(ic, "A_no_intercept", acts_none, baseline_rows)
        # B: K49
        run_baseline(ic, "B_k49", k49_actions(table, book48, False), baseline_rows)
        # C: K49 + fine_down
        run_baseline(ic, "C_k49_fine_down", k49_actions(table, book48, True), baseline_rows)
        # D: continuous search
        best = None
        for _ in range(40):
            if rollout[0] >= STAGE1_BUDGET or not time_left():
                break
            p = {
                "n_open": int(rng.choice([2, 4, 6])),
                "n_chase": int(rng.choice([6, 10, 14, 18])),
                "n_close": int(rng.choice([4, 8, 12])),
                "n_hold": int(rng.choice([20, 28, 36])),
                "v_scale": float(rng.choice([0.4, 0.7, 1.0])),
                "vz_bias": float(rng.choice([-0.05, -0.15, -0.3, -0.5])),
                "aim_z_bias": float(rng.choice([-0.02, -0.01, 0.0])),
                "wy": float(rng.choice([0.0, 0.2, 0.4, 0.55])),
                "grip_chase": float(rng.choice([-1.0, -0.48])),
                "grip_close": float(rng.choice([0.35, 1.0])),
                "use_fine_down": bool(rng.random() < 0.4),
            }
            acts = intercept_params_to_actions(env, ic, p)
            out = run_baseline(ic, "D_continuous", acts, baseline_rows)
            sc = 100 * out["FREE_FALL_CATCH_SUCCESS"] + 10 * out["max_bil_persist_s"] + 2 * out["had_bil"]
            if best is None or sc > best[0]:
                best = (sc, acts, out, p)
        # local refine around best
        if best is not None:
            base_p = best[3]
            for _ in range(15):
                if rollout[0] >= STAGE1_BUDGET or not time_left():
                    break
                p = dict(base_p)
                p["n_chase"] = int(np.clip(p["n_chase"] + int(rng.integers(-3, 4)), 4, 24))
                p["vz_bias"] = float(np.clip(p["vz_bias"] + rng.normal(0, 0.08), -0.8, 0.1))
                p["n_close"] = int(np.clip(p["n_close"] + int(rng.integers(-2, 3)), 2, 16))
                acts = intercept_params_to_actions(env, ic, p)
                run_baseline(ic, "D_continuous_refine", acts, baseline_rows)

    write_csv(OUT / "BASELINE_COMPARISON.csv", baseline_rows)
    write_csv(OUT / "CONTACT_GEOMETRY.csv", contact_rows)
    write_csv(OUT / "CATCH_SUCCESS_VERIFICATION.csv", success_rows)

    n_a_catch = len({ic_id for ic_id, *_ in [(s[0]["ic_id"],) for s in level_a_successes]}) if level_a_successes else len(set(r["ic_id"] for r in success_rows if r["level"] == "A"))
    print(f"Level A verified catch ICs: {n_a_catch}", flush=True)

    # =====================================================================
    # Stage 2 — B/C if A worked
    # =====================================================================
    if n_a_catch > 0 and time_left():
        print(f"=== Stage 2 Level B/C (to {TOTAL_BUDGET}) ===", flush=True)
        lim = TOTAL_BUDGET
        for level in ("B", "C"):
            for ic in ics[level]:
                if rollout[0] >= lim or not time_left():
                    break
                if not ic["ok_init"] or not ic["falling_verified"]:
                    continue
                print(f"  {ic['ic_id']} roll={rollout[0]}", flush=True)
                run_baseline(ic, "B_k49", k49_actions(table, book48, False), baseline_rows)
                for _ in range(25):
                    if rollout[0] >= lim or not time_left():
                        break
                    p = {
                        "n_open": int(rng.choice([2, 4, 6])),
                        "n_chase": int(rng.choice([8, 12, 16, 20])),
                        "n_close": int(rng.choice([6, 10, 14])),
                        "n_hold": int(rng.choice([24, 32])),
                        "v_scale": float(rng.choice([0.5, 0.8, 1.0])),
                        "vz_bias": float(rng.choice([-0.1, -0.25, -0.4])),
                        "aim_z_bias": float(rng.choice([-0.02, 0.0])),
                        "wy": float(rng.choice([0.2, 0.4, 0.55])),
                        "grip_chase": -1.0,
                        "grip_close": 1.0,
                        "use_fine_down": bool(rng.random() < 0.3),
                    }
                    acts = intercept_params_to_actions(env, ic, p)
                    run_baseline(ic, "D_continuous", acts, baseline_rows)
    else:
        print("=== Stage 2 skipped (no Level A success) — diagnosing ===", flush=True)

    # Robustness on successes
    print("=== Robustness ===", flush=True)
    for row in list(success_rows):
        if rollout[0] >= TOTAL_BUDGET or not time_left():
            break
        path = OUT / "trajectories" / f"{row['ic_id']}__{row['condition']}_actions.npz"
        if not path.exists():
            continue
        acts = [np.asarray(a, np.float32) for a in np.load(path)["actions"]]
        # find ic
        ic = None
        for lst in ics.values():
            for cand in lst:
                if cand["ic_id"] == row["ic_id"]:
                    ic = cand
                    break
        if ic is None:
            continue
        # timing shift
        for shift in (-1, 1, 2):
            if rollout[0] >= TOTAL_BUDGET:
                break
            if shift < 0:
                acts_s = acts[-shift:] + [acts[-1]] * (-shift) if False else acts[abs(shift):]
                acts_s = acts[abs(shift):] if abs(shift) < len(acts) else acts
            else:
                pad = [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * shift
                acts_s = pad + acts
            out, _ = evaluate_catch(env, ic, acts_s)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": ic["ic_id"], "condition": row["condition"],
                "variation": f"timing_shift_{shift}",
                "FREE_FALL_CATCH_SUCCESS": out["FREE_FALL_CATCH_SUCCESS"],
                "max_bil_persist_s": out["max_bil_persist_s"],
            })
        # nearby release
        for dh, doff in [(0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01)]:
            if rollout[0] >= TOTAL_BUDGET or not time_left():
                break
            ic2 = release_ic(
                env, ic["level"], ic["root_id"] + f"_p", rng,
                ic["height_m"] + dh, ic["offset_m"] + doff, ic["omega_norm"],
            )
            if not ic2["ok_init"]:
                continue
            out, _ = evaluate_catch(env, ic2, acts)
            rollout[0] += 1
            robust_rows.append({
                "ic_id": ic["ic_id"], "condition": row["condition"],
                "variation": f"release_dh{dh}_doff{doff}",
                "variant_ic": ic2["ic_id"],
                "FREE_FALL_CATCH_SUCCESS": out["FREE_FALL_CATCH_SUCCESS"],
                "max_bil_persist_s": out["max_bil_persist_s"],
            })

    write_csv(OUT / "BASELINE_COMPARISON.csv", baseline_rows)
    write_csv(OUT / "CONTACT_GEOMETRY.csv", contact_rows)
    write_csv(OUT / "CATCH_SUCCESS_VERIFICATION.csv", success_rows)
    write_csv(OUT / "ROBUSTNESS_RESULTS.csv", robust_rows)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_BUDGET,
        "wall_s": time.time() - t_wall0,
        "n_catch": len(success_rows),
        "n_level_a_catch_ics": len(set(r["ic_id"] for r in success_rows if r["level"] == "A")),
        "checksum": checksum,
    })

    if not args.skip_videos:
        _videos(env, ics, success_rows, baseline_rows)

    _report(reach_rows, manifest, baseline_rows, contact_rows, success_rows, robust_rows, checksum)
    print("Done", OUT / "REPORT.md", flush=True)


def _videos(env, ics, success_rows, baseline_rows):
    # one video per level A IC: best continuous or k49
    shown = set()
    for ic in ics["A"][:4]:
        restore_ic(env, ic)
        acts = [np.array([0, 0, -0.3, 0, 0.3, 0, -1], np.float32) for _ in range(8)]
        acts += [np.array([0, 0, -0.2, 0, 0.2, 0, 1], np.float32) for _ in range(20)]
        # prefer saved success
        for r in success_rows:
            if r["ic_id"] == ic["ic_id"]:
                p = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
                if p.exists():
                    acts = list(np.load(p)["actions"])
                    break
        _render(env, ic, acts, OUT / "videos" / f"{ic['ic_id']}_catch.mp4")
        shown.add(ic["ic_id"])
    for r in success_rows[:3]:
        p = OUT / "trajectories" / f"{r['ic_id']}__{r['condition']}_actions.npz"
        if not p.exists():
            continue
        ic = None
        for lst in ics.values():
            for c in lst:
                if c["ic_id"] == r["ic_id"]:
                    ic = c
        if ic:
            _render(env, ic, list(np.load(p)["actions"]), OUT / "videos" / f"success_{r['ic_id']}.mp4")


def _render(env, ic, acts, dest):
    try:
        restore_ic(env, ic)
        cam = mjv_camera_from_preset({
            "lookat": obj_pos(env.sim), "distance": 0.7, "azimuth": 135.0, "elevation": -20.0,
        })
        renderer = mujoco.Renderer(env.sim.model, 360, 480)
        frames = []
        t0 = float(env.sim.data.time)
        for i, a in enumerate(acts):
            if i % 2 == 0:
                cam.lookat[:] = obj_pos(env.sim)
                renderer.update_scene(env.sim.data, cam)
                cs = contact_summary(env.sim)
                frames.append(caption(
                    renderer.render(),
                    f"{ic['ic_id']} z={obj_pos(env.sim)[2]:.3f} bil={int(cs['bilateral'])} t={float(env.sim.data.time)-t0:.2f}",
                ))
            step_legal(env.sim, np.asarray(a, np.float32))
            if object_table_contact(env.sim) and i > 5:
                break
        write_frames_mp4(frames, dest, fps=25)
        renderer.close()
        print("video", dest, flush=True)
    except Exception as e:
        print("video warn", e, flush=True)


def _mean(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else float("nan")


def _report(reach, manifest, baseline, contact, success, robust, checksum):
    def by(rows, key, val):
        return [r for r in rows if r.get(key) == val]

    a_rows = by(baseline, "level", "A")
    catches = success
    roots = sorted(set(r["ic_id"] for r in catches))
    max_off = None
    for r in catches:
        man = next((m for m in manifest if m["ic_id"] == r["ic_id"]), None)
        if man:
            o = float(man.get("offset_m") or 0)
            max_off = o if max_off is None else max(max_off, o)

    cont = [r for r in baseline if str(r.get("baseline", "")).startswith("D_")]
    k49 = [r for r in baseline if r.get("baseline") == "B_k49"]
    cont_rate = _mean([r.get("FREE_FALL_CATCH_SUCCESS") for r in cont])
    k49_rate = _mean([r.get("FREE_FALL_CATCH_SUCCESS") for r in k49])

    inside = sum(1 for r in contact if int(float(r.get("com_between") or 0)))
    n_contact = len(contact)

    robust_ok = sum(1 for r in robust if int(float(r.get("FREE_FALL_CATCH_SUCCESS") or 0)))

    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))

    report = f"""# Free-Fall Cylinder Catching Feasibility

**Robot/controller:** unchanged Panda + AIRBORNE_V3 · K49 checksum `{checksum}`  
**Criterion:** `FREE_FALL_CATCH_SUCCESS` (HOLD_S={HOLD_S}s) — see `BENCHMARK_DEFINITION.md`  
**No SFT/GRPO. Not mixed into K49 recovery experts.**

## Budget

| Item | Value |
|------|------:|
| Stage1 / Stage2 | {STAGE1_BUDGET} / {STAGE2_BUDGET} |
| Used | {usage.get('rollouts_used')} |
| Wall (s) | {usage.get('wall_s'):.1f} |

## Level A feasibility

Verified catch ICs: **{len(set(r['ic_id'] for r in catches if r['level']=='A'))}**  
Total FREE_FALL_CATCH_SUCCESS rows: **{len(catches)}**  
Independent release roots with catch: **{len(roots)}** ({roots})

## Baselines (aggregate catch rate)

| Baseline | catch rate |
|----------|----------:|
| A no-intercept | {_mean([r.get('FREE_FALL_CATCH_SUCCESS') for r in baseline if r.get('baseline')=='A_no_intercept']):.3f} |
| B K49 | {k49_rate:.3f} |
| C K49+fine_down | {_mean([r.get('FREE_FALL_CATCH_SUCCESS') for r in baseline if r.get('baseline')=='C_k49_fine_down']):.3f} |
| D continuous | {cont_rate:.3f} |

## Q1–Q12

1. Can existing Panda catch a genuinely free-falling cylinder? **{"Yes" if roots else "Not under this budget / setups"}** ({len(roots)} roots).
2. Simplest verified IC: **{roots[0] if roots else "none"}** (see manifest heights/offsets).
3. Max tested lateral offset among successes: **{max_off if max_off is not None else "n/a"}** m.
4. Rotating (Level C) catch? **{any(r['level']=='C' for r in catches)}**.
5. Enter intended opening before closure? Contact `com_between` rate among logged first contacts: **{inside/max(n_contact,1):.2f}** (see `CONTACT_GEOMETRY.csv`).
6. Controller track intercept? See hand tracking in reachability + catch vs miss patterns in baselines.
7. Geometry vs time failures: palm occlusion forces side-entry; table window from reachability CSV; diagnose via `table_before_bil` / `descending_at_contact` columns.
8. Continuous outperform K49? D rate {cont_rate:.3f} vs B {k49_rate:.3f}.
9. Compact discrete vocab? Only if continuous shows clear unique behaviors — **do not expand K49 until continuous catches are robust**.
10. Independent roots with sustained capture: **{len(roots)}**.
11. Robustness: {robust_ok}/{len(robust)} varied trials still succeed.
12. Implication for grasp-recovery: free-fall catching needs **jaw-entry geometry before impact** (side-entry / catch-ready wrist). Prior ball-impact ICs that start on outer/tip contacts are a **different failure mode**; catching skill does not automatically transfer without fixing pre-contact geometry / onset timing.

## Morphology note

Default downward gripper places the **palm above the jaw gap**; pure overhead drops collide with the hand. This benchmark uses catch-ready wrist + side-entry releases (documented in `BENCHMARK_DEFINITION.md`).

## Scripts
`scripts/run_free_fall_cylinder_catching.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    fail = f"""# Failure Analysis — Free-Fall Catching

## Init / geometry

- Palm occlusion of overhead jaw entry is a first-order constraint.
- Side-entry + catch-ready wrist used so the cylinder can enter the jaw while descending.

## Catch criterion

FREE_FALL_CATCH_SUCCESS requires descending first contact, bilateral hold ≥{HOLD_S}s, arrested vz, no pre-capture table hit.

## Observed

- Catch roots: {roots}
- Continuous vs K49 catch rates: {cont_rate:.3f} vs {k49_rate:.3f}
- Robustness survivors: {robust_ok}/{len(robust)}

## Relation to ball-impact recovery

Prior HIGH_MOTION failures were dominated by **already-established outer/tip contacts**. Free-fall isolates interception; success here does not claim the ball-impact ICs are solvable without earlier onset / geometry repair.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")


if __name__ == "__main__":
    main()
