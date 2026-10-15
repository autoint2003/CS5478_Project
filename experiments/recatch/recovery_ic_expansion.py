"""Local ball-impact IC expansion around the verified legal recatch.

Does not train. Does not change step_legal, K48, TABLE_DROP, or the nominal controller.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_generation.strategies.recatch_oracle import _flight_step, _ref_now

from controllers.geometry_switch import hand_pose
from controllers.recovery_actions import AIRBORNE_V3, normalize_action
from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from data_generation.ball_impact_dataset import aim_ball, cylinder_contacts, physics_record
from data_generation.strategies.capture_geometry import capture_geometry
from envs.horizon_outcome import NOMINAL_S, nominal_verify
from training.impact_ball import apply_ball_free_state
from experiments.recatch.legal_recatch_speed_match import run_speed
from experiments.recatch.legal_recatch_transfer import (
    _ball_touch,
    _bottom,
    _enable_supports,
    _prepare,
    _sample,
)
from training.nominal_pose_audit import hand_R
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from experiments.recatch.recatch_braking_optimization import BASE_SPEC, _advance, _law
from experiments.recatch.recatch_interface_comparison import _capture_window, _render
from training.recovery_runtime import apply_dyn, make_snap, restore_ballistic
from envs.recovery_success import nominal_step
from training.replay_core import TABLE_TOP
from envs.table_drop import latched, restore_state
from training.unified_k47 import load_codebook

OUT = ROOT / "results" / "diagnostics" / "recovery_ic_expansion"
KNOWN = {
    "dive": 1.2,
    "dive_s": 0.04,
    "lead": 0.0,
    "tol": 0.12,
    "settle": 0.04,
    "accel": 3.0,
    "handoff_stable_s": 0.10,
}


def _cells():
    rows = []

    def add(split, family, azimuth, speed, z_off=0.0, phi=0.0, delay=0.0, **extra):
        rows.append({
            "split": split,
            "family": family,
            "azimuth_deg": float(azimuth),
            "speed_mps": float(speed),
            "z_off_m": float(z_off),
            "phi_deg": float(phi),
            "delay_s": float(delay),
            "mass_kg": extra.get("mass_kg"),
            "friction": extra.get("friction"),
            "grasp_offset": extra.get("grasp_offset"),
            "seat_xy": extra.get("seat_xy"),
            "kind": extra.get("kind", "ball_impact"),
        })

    for speed in (12.0, 14.0, 16.0, 18.0):
        for z_off in (0.0, 0.020):
            for phi in (0.0, 15.0):
                add("train", "oblique_+X_+Y", 45.0, speed, z_off=z_off, phi=phi)
    for speed in (12.0, 14.0, 16.0, 18.0):
        for z_off in (0.0, 0.020):
            add("valid", "oblique_-X_+Y", 135.0, speed, z_off=z_off)
    add("train", "+X", 0.0, 8.0)
    add("train", "+X", 0.0, 14.0)
    add("valid", "-Y", 270.0, 10.0)
    add("physics_side", "oblique_+X_+Y", 45.0, 14.0, z_off=0.020, mass_kg=0.25, kind="mass_friction")
    add("physics_side", "oblique_+X_+Y", 45.0, 14.0, z_off=0.020, friction=0.6, kind="mass_friction")
    return rows




def _features(sim):
    geo = capture_geometry(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    return {
        "bottom_m": round(_bottom(sim), 4),
        "zo_m": round(float(rel["p_obj_w"][2]), 4),
        "xy_m": [round(float(v), 4) for v in rel["p_obj_w"][:2]],
        "p_rel_h": [round(float(v), 4) for v in rel["p_rel_h"]],
        "v_rel_h": [round(float(v), 4) for v in rel["v_rel_h"]],
        "v_obj": [round(float(v), 4) for v in rel["v_obj_w"]],
        "w_obj": [round(float(v), 3) for v in body_twist(sim.model, sim.data, sim.ids.object_body)[1]],
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "gap_mm": round(float(geo["inner_gap_m"]) * 1e3, 2),
        "between": int(bool(geo["between"])),
        "table_drop": bool(latched(sim)),
        "over_table": bool(0.38 <= rel["p_obj_w"][0] <= 0.62 and -0.12 <= rel["p_obj_w"][1] <= 0.12),
    }


def _table_touch(sim):
    import mujoco

    table = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    obj = int(sim.ids.object_geom)
    n = 0
    for index in range(int(sim.data.ncon)):
        geoms = {int(sim.data.contact[index].geom1), int(sim.data.contact[index].geom2)}
        if obj in geoms and table in geoms:
            n += 1
    return n


def _apply_variant(sim, nominal, cell):
    if cell.get("mass_kg") is None and cell.get("friction") is None:
        return
    apply_dyn(sim, nominal, cell.get("mass_kg"), cell.get("friction"))
    if cell.get("friction") is not None:
        from envs.dynamics import set_finger_object_sliding_mu

        set_finger_object_sliding_mu(sim.model, sim.ids, float(cell["friction"]), sim.data)




def _impact(sim, base, nominal, cell):
    """Nominal hold, then one real ball flight. The cylinder is not moved by script."""
    restore_ballistic(sim, base)
    _apply_variant(sim, nominal, cell)
    restore_state(sim, None)
    if cell.get("mass_kg") is not None or cell.get("friction") is not None:
        for _ in range(int(0.20 / 0.002)):
            nominal_step(sim)
            if latched(sim) or _bottom(sim) < TABLE_TOP:
                return None, "dynamics_change_dropped_before_launch"
    sim.model.opt.timestep = 0.002
    for _ in range(int(float(cell["delay_s"]) / 0.002)):
        nominal_step(sim)
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            return None, "nominal_failed_before_launch"
    sim.set_guide_weld(False)
    launch = aim_ball(sim, cell["azimuth_deg"], cell["speed_mps"], cell["z_off_m"], cell["phi_deg"])
    apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
    hits, _other = cylinder_contacts(sim)
    if hits:
        return None, "ball_overlapped_at_launch"
    contact_t = None
    lost = False
    separated = False
    off_since = None
    uni_since = None
    t0 = float(sim.data.time)
    peak_fn = 0.0
    for _ in range(int(0.70 / 0.002)):
        _flight_step(sim)
        now = float(sim.data.time)
        step_hits, _other = cylinder_contacts(sim)
        if step_hits and contact_t is None:
            contact_t = now
        if step_hits:
            peak_fn = max(peak_fn, max(float(item["fn"]) for item in step_hits))
        elif contact_t is not None and now >= contact_t + 0.004:
            separated = True
        geo = capture_geometry(sim)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        none = int(geo["nL"]) == 0 and int(geo["nR"]) == 0
        if latched(sim) or _bottom(sim) < TABLE_TOP:
            return None, "table_or_floor_before_recovery"
        if contact_t is None:
            if now > t0 + 0.45:
                break
            continue
        if none:
            off_since = now if off_since is None else off_since
            if now - off_since >= 0.010 and _bottom(sim) > 0.50:
                lost = True
                break
        else:
            off_since = None
        if not both:
            uni_since = now if uni_since is None else uni_since
            if now - uni_since >= 0.030 and _bottom(sim) > 0.50:
                lost = False
                break
        else:
            uni_since = None
        if now >= contact_t + 0.18:
            break
    if contact_t is None:
        return None, "no_ball_cylinder_contact"
    if _table_touch(sim):
        return None, "cylinder_already_touching_table"
    from training.replay_core import park_impact_ball

    park_impact_ball(sim)
    snap = make_snap(sim)
    info = {
        "contact_t": round(float(contact_t), 4),
        "launch_position": [round(float(x), 5) for x in launch["p_launch"]],
        "launch_velocity": [round(float(x), 5) for x in launch["v0"]],
        "peak_normal_n": round(float(peak_fn), 3),
        "ball_separated": bool(separated),
        "bilateral_lost_before_ic": bool(lost),
        "flight_s": round(float(contact_t - t0), 4),
    }
    return snap, info


def _baseline(sim, snap, nominal, ref, cell):
    restore_ballistic(sim, snap)
    _apply_variant(sim, nominal, cell)
    restore_state(sim, None)

    class Env:
        pass

    env = Env()
    env.sim = sim
    try:
        out = nominal_verify(env, ref, seconds=NOMINAL_S)
    except Exception as exc:
        return {"success": False, "reason": f"baseline_error:{exc}"}
    return {
        "success": bool(out.get("success")),
        "reason": out.get("reason"),
        "nominal_elapsed": None if out.get("nominal_elapsed") is None else round(float(out["nominal_elapsed"]), 3),
        "end_z": None if out.get("end") is None else out["end"].get("z"),
        "end_nL": None if out.get("end") is None else out["end"].get("nL"),
        "end_nR": None if out.get("end") is None else out["end"].get("nR"),
        "table_drop": bool(latched(sim)),
    }


def _slow_xy(velocity, accel=2.5):
    xy = np.array(velocity[:2], float)
    speed = float(np.linalg.norm(xy))
    if speed < 1e-6:
        return np.zeros(2)
    return xy / speed * max(0.0, speed - float(accel) * 0.02)


def _settled(sim, vo, vh):
    geo = capture_geometry(sim)
    both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
    return bool(
        both
        and float(np.linalg.norm(vo)) < 0.12
        and float(np.linalg.norm(vh)) < 0.12
        and _bottom(sim) >= TABLE_TOP + 0.02
    )


def _rollout(sim, spec, mode):
    """Closed-loop oracle search. Commands leave only through step_legal."""
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(rel0["p_rel_h"], float)
    phase = "chase"
    free_s = both_s = 0.0
    armed = False
    v_cmd_z = None
    settle_until = None
    zo_brake = None
    brake_t = None
    quiet = 0
    rows = []
    actions = []
    saw_off = False
    saw_re = False
    for _ in range(int(2.4 / 0.02)):
        geo = capture_geometry(sim)
        phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, 0.02)
        if mode == "recatch" and phase in ("close", "hold"):
            phase = "chase"
        now = float(sim.data.time) - t0
        vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
        vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
        both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
        none = int(geo["nL"]) == 0 and int(geo["nR"]) == 0
        if none:
            saw_off = True
        elif saw_off and both:
            saw_re = True
        xy = _slow_xy(vo) if both else np.array(vo[:2], float)
        quiet_need = max(int(float(spec["handoff_stable_s"]) / 0.02), 13)
        if mode == "hold":
            v_star = np.array([float(xy[0]), float(xy[1]), float(np.clip(vo[2], -1.0, 1.0))], float)
            tau = -18.0
            phase = "hold"
            quiet = quiet + 1 if _settled(sim, vo, vh) else 0
            if quiet >= quiet_need:
                phase = "handoff"
        else:
            _p, v_law, _e = _law(
                sim, rh, BASE_SPEC["aim_down"], BASE_SPEC["k"], BASE_SPEC["v_clip"], False, BASE_SPEC["corr_clip"],
            )
            hand_z = float(sim.data.xpos[sim.ids.hand_body][2])
            obj_z = float(sim.data.xpos[sim.ids.object_body][2])
            lateral = obj_z > hand_z - 0.05
            if v_cmd_z is None:
                if (not lateral) and now < float(spec["dive_s"]):
                    vz = -float(spec["dive"])
                else:
                    vz = min(float(v_law[2]), float(vo[2]) - float(spec["lead"]))
                vz = float(np.clip(vz, -3.8, 1.0))
                v_star = np.array([float(v_law[0]), float(v_law[1]), vz], float)
                gap = float(geo["inner_gap_m"])
                if gap < 0.040:
                    tau = 2.0
                elif gap > 0.060:
                    tau = -4.0
                else:
                    tau = 0.0
                ready = _capture_window(geo, free_s, armed) and float(vh[2]) <= float(vo[2]) + float(spec["tol"])
                if ready:
                    phase = "settle"
                    settle_until = now + float(spec["settle"])
                    v_cmd_z = float(min(vo[2], 0.0))
                    tau = -18.0
                    v_star[2] = v_cmd_z
            elif phase == "settle" or (v_cmd_z is not None and phase == "chase"):
                phase = "settle"
                v_cmd_z = float(min(vo[2], 0.0))
                v_star = np.array([float(xy[0]), float(xy[1]), v_cmd_z], float)
                tau = -18.0
                if now >= settle_until and both:
                    phase = "brake"
                    brake_t = now
                    zo_brake = float(sim.data.xpos[sim.ids.object_body][2])
                    v_cmd_z = float(min(vh[2], vo[2], 0.0))
            elif phase == "brake":
                if both and float(vh[2]) > float(vo[2]) + 0.08:
                    v_cmd_z = float(min(vo[2], 0.0))
                else:
                    v_cmd_z = min(0.0, float(v_cmd_z) + float(spec["accel"]) * 0.02)
                if both and v_cmd_z >= -0.02 and float(vo[2]) > -0.15:
                    phase = "lift"
                v_star = np.array([float(xy[0]), float(xy[1]), float(v_cmd_z)], float)
                tau = -18.0
            elif phase == "lift":
                v_cmd_z = min(0.20, float(v_cmd_z) + float(spec["accel"]) * 0.02)
                if both and float(vo[2]) > -0.05 and float(sim.data.xpos[sim.ids.object_body][2]) >= float(zo_brake or 0.0) + 0.02:
                    phase = "settle_hold"
                v_star = np.array([float(xy[0]), float(xy[1]), float(v_cmd_z)], float)
                tau = -18.0
            else:
                v_cmd_z = 0.0
                v_star = np.array([float(xy[0]), float(xy[1]), 0.0], float)
                tau = -18.0
                quiet = quiet + 1 if _settled(sim, vo, vh) else 0
                if quiet >= quiet_need:
                    phase = "handoff"
        if phase == "handoff":
            rows.append(_sample(sim, t0, phase, v_star, tau))
            break
        hand = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
        action = normalize_action(hand.T @ np.asarray(v_star, float), np.zeros(3), float(tau), AIRBORNE_V3)
        step_legal(sim, action)
        actions.append(np.asarray(action, np.float32))
        rows.append(_sample(sim, t0, phase, v_star, tau))
        if latched(sim) or (_bottom(sim) < TABLE_TOP and _table_touch(sim)):
            break
    return {
        "actions": actions,
        "rows": rows,
        "saw_off": saw_off,
        "saw_re": saw_re,
        "brake_t": brake_t,
        "phase": rows[-1]["phase"] if rows else "empty",
        "table_drop": bool(latched(sim)),
        "table_touch": int(_table_touch(sim)),
        "ball_hits": int(_ball_touch(sim)),
    }


def _replay(sim, snap, nominal, cell, actions):
    restore_ballistic(sim, snap)
    _apply_variant(sim, nominal, cell)
    restore_state(sim, None)
    t0 = float(sim.data.time)
    rows = []
    for action in actions:
        step_legal(sim, action)
        rows.append(_sample(sim, t0, "replay", np.zeros(3), 0.0))
        if latched(sim) or _bottom(sim) < TABLE_TOP - 0.01:
            break
    return rows


def _finish(sim, ref):
    class Env:
        pass

    env = Env()
    env.sim = sim
    import envs.horizon_outcome as horizon

    saved = {}
    real = horizon.apply_handoff

    def _remember(sim_now, ref_now):
        record = real(sim_now, ref_now)
        saved.update(record)
        return record

    horizon.apply_handoff = _remember
    try:
        out = nominal_verify(env, ref, seconds=NOMINAL_S)
    finally:
        horizon.apply_handoff = real
    return out, saved


def _variants(lost):
    if not lost:
        return [("hold", KNOWN), ("recatch", KNOWN)]
    specs = [
        KNOWN,
        {**KNOWN, "dive": 2.0, "dive_s": 0.06, "accel": 3.0},
        {**KNOWN, "dive": 0.6, "dive_s": 0.0, "accel": 2.0, "tol": 0.25, "lead": 0.0},
        {**KNOWN, "dive": 1.2, "accel": 6.0, "settle": 0.02, "handoff_stable_s": 0.20},
    ]
    return [("recatch", spec) for spec in specs]


def _save_trace(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(path, t=np.zeros(0))
        return
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows], np.float32),
        bottom=np.array([r["bottom"] for r in rows], np.float32),
        zo=np.array([r["zo"] for r in rows], np.float32),
        xy=np.array([r["xy"] for r in rows], np.float32),
        nL=np.array([r["nL"] for r in rows], np.int16),
        nR=np.array([r["nR"] for r in rows], np.int16),
        vh=np.stack([r["vh"] for r in rows]).astype(np.float32),
        vo=np.stack([r["vo"] for r in rows]).astype(np.float32),
        qpos=np.stack([r["qpos"] for r in rows]).astype(np.float32),
    )


def _regression(sim, snap, nominal, pads, pad_mu, book):
    first = run_speed(sim, snap, nominal, pads, pad_mu, KNOWN)
    second = run_speed(sim, snap, nominal, pads, pad_mu, KNOWN)
    a = np.stack(first["actions"])
    b = np.stack(second["actions"])
    q1 = first["rows"][-1]["qpos"]
    q2 = second["rows"][-1]["qpos"]
    saved = np.load(ROOT / "results" / "diagnostics" / "legal_recatch_transfer" / "traces" / "legal_A_actions.npz")["actions"]
    packed = representability(a, book)
    tokens = np.asarray(packed["tokens"], np.int64)
    residual = a - book[tokens]
    decoded = np.asarray(packed["quantized"], np.float32)
    token_only = book[tokens]
    folder = OUT / "SUCCESS_DEMOS" / "regression_seated_freefall"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "actions.npz", actions=a, tokens=tokens, residual=residual.astype(np.float32), decoded=decoded)
    with (folder / "seated_snapshot.pkl").open("wb") as handle:
        pickle.dump(snap, handle)
    # Independent open-loop replay plus nominal handoff.
    from experiments.recatch.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    ref = _ref_now(sim)
    t0 = float(sim.data.time)
    rows = []
    for action in decoded:
        step_legal(sim, action)
        rows.append(_sample(sim, t0, "replay", np.zeros(3), 0.0))
    out, handoff = _finish(sim, ref)
    record = {
        "source": "seated_horizontal_grasp_freefall",
        "ball": "parked_and_welded_before_release",
        "not_a_ball_impact": True,
        "scene": physics_record(sim),
        "table_contype": 1,
        "deterministic_action_linf": float(np.max(np.abs(a - b))),
        "deterministic_qpos_linf": float(np.max(np.abs(q1 - q2))),
        "matches_saved_actions": bool(a.shape == saved.shape and np.max(np.abs(a - saved)) < 1e-5),
        "k48_max_linf_to_codeword": packed["max_linf"],
        "k48_fraction_within_0.25": packed["fraction_within_0.25"],
        "residual_required": bool(packed["max_linf"] > 1e-6),
        "token_only_max_abs_error": float(np.max(np.abs(token_only - a))),
        "decoded_matches_actions": bool(np.max(np.abs(decoded - a)) < 1e-6),
        "n_actions": int(len(a)),
        "replay_min_bottom_m": round(float(min(r["bottom"] for r in rows)), 4),
        "replay_end_n": [int(rows[-1]["nL"]), int(rows[-1]["nR"])],
        "replay_table_drop": bool(latched(sim)),
        "nominal_reason": out.get("reason"),
        "nominal_success": bool(out.get("success")),
        "nominal_elapsed": None if out.get("nominal_elapsed") is None else round(float(out["nominal_elapsed"]), 3),
        "end_z": None if out.get("end") is None else out["end"].get("z"),
        "handoff_t": handoff.get("t"),
        "handoff_p_des": handoff.get("p_des"),
        "handoff_v_cmd_norm": handoff.get("v_cmd_norm"),
        "handoff_r_des_to_saved_rad": handoff.get("r_des_to_saved_rad"),
        "handoff_physics_edited": handoff.get("physics_edited"),
        "grip_ctrl": handoff.get("grip_ctrl"),
    }
    (folder / "regression.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("REGRESSION", record["nominal_reason"], "det", record["deterministic_action_linf"], "residual", record["token_only_max_abs_error"], flush=True)
    return record


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _diversity(manifest):
    official = [row for row in manifest if row["kind"] == "ball_impact" and row["split"] in ("train", "valid")]
    successes = [row for row in official if row.get("demo_success")]
    families = sorted({row["family"] for row in successes})
    train_f = sorted({row["family"] for row in successes if row["split"] == "train"})
    valid_f = sorted({row["family"] for row in successes if row["split"] == "valid"})
    speeds = [float(row["speed_mps"]) for row in successes]
    shared = sorted(set(train_f) & set(valid_f))
    return {
        "official_ic": len(official),
        "official_success": len(successes),
        "success_families": families,
        "train_families": train_f,
        "valid_families": valid_f,
        "shared_families": shared,
        "speed_min": None if not speeds else min(speeds),
        "speed_max": None if not speeds else max(speeds),
        "airborne_success": sum(1 for row in successes if row.get("airborne_recatch")),
    }


def _write_reports(regression, manifest, searches, div):
    counts = {}
    for row in manifest:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    demos = [row for row in manifest if row.get("demo_success")]
    airborne = [row for row in demos if row.get("airborne_recatch")]
    k48_kept = [row for row in demos if row.get("k48_replay_kept")]
    lines = [
        "# Recovery IC expansion",
        "",
        "本轮没有训练。正式 `step_legal`、K48 codebook、`TABLE_DROP` 和名义控制器都没有修改。",
        "",
        "## 回归用例",
        "",
        "已验证轨迹是坐姿水平抓取后的自由落体，球在释放前被停住并焊接，不是球冲击。",
        f"两次独立重放的动作差 L∞ = {regression['deterministic_action_linf']:.3e}，末端 qpos 差 L∞ = {regression['deterministic_qpos_linf']:.3e}。",
        f"与已保存动作带一致：{regression['matches_saved_actions']}。",
        f"K48 最近码字的最大 L∞ 是 {regression['k48_max_linf_to_codeword']:.4f}。残差必须保留：只用离散 token 的最大绝对误差是 {regression['token_only_max_abs_error']:.4f}。残差在 ±0.25 内时，token+residual 与原动作一致：{regression['decoded_matches_actions']}。",
        f"量化动作重放后的名义结果是 {regression['nominal_reason']}，handoff 不改物理状态：{regression['handoff_physics_edited']}。",
        "",
        "## 冲击初值",
        "",
        f"共生成并完成判定的 IC：{len(manifest)}。",
        f"TRIVIAL {counts.get('TRIVIAL', 0)}，RECOVERY_NEEDED {counts.get('RECOVERY_NEEDED', 0)}，UNSOLVED {counts.get('UNSOLVED', 0)}，INVALID {counts.get('INVALID', 0)}。",
        f"合法 `SUCCESS_12S` demonstration：{len(demos)}。其中空中再夹持：{len(airborne)}。K48 重放仍成功：{len(k48_kept)}。",
        "",
        "球的位置和速度只在接触前设置一次。圆柱没有被传送。质量或摩擦改动放在 `physics_side`，不进入 train/valid。",
        "",
        "## 多样性",
        "",
        f"正式 ball-impact IC {div['official_ic']} 个，其中成功 {div['official_success']} 个。",
        f"成功所在方向族：{div['success_families'] or '无'}。训练族 {div['train_families'] or '无'}，验证族 {div['valid_families'] or '无'}，两集共享族 {div['shared_families'] or '无'}。",
        f"成功球速范围：{div['speed_min']} 到 {div['speed_max']} m/s。",
        "",
        "同一方向族里的速度和高度微扰不是相互独立的泛化样本。验证集只使用 -X 和 -Y，训练集使用 +X、+Y 和斜向，方向族没有交叉。",
        "",
    ]
    fail = {}
    for row in manifest:
        if row["label"] in ("UNSOLVED", "INVALID"):
            key = row.get("fail_reason") or row["label"]
            fail[key] = fail.get(key, 0) + 1
    lines.append("失败原因计数：")
    lines.append("")
    for key, value in sorted(fail.items(), key=lambda item: -item[1]):
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "## 是否可以开始 SFT / GRPO",
        "",
    ])
    ready = (
        len(demos) >= 4
        and len(div["success_families"]) >= 2
        and not div["shared_families"]
        and len(k48_kept) == len(demos)
    )
    if ready:
        lines.append("正式冲击成功分布跨了至少两个不共享的方向族，并且每条成功都能由 K48 token+residual 重放后完成 `SUCCESS_12S`。可以开始组建数据集，但不要把同一族里的微扰当成独立样本。")
    else:
        lines.append("还不够。成功要么数量不足，要么集中在单一方向族，要么验证族与训练族没有分开的成功。本轮不开始训练。")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    div_lines = [
        "# Diversity",
        "",
        "划分单位是冲击方向族，不是单条轨迹的邻域微扰。",
        "",
        f"- 正式 IC：{div['official_ic']}",
        f"- 正式成功：{div['official_success']}",
        f"- 成功方向族：{div['success_families']}",
        f"- train 族：{div['train_families']}",
        f"- valid 族：{div['valid_families']}",
        f"- 共享族：{div['shared_families']}",
        f"- 空中再夹持成功：{div['airborne_success']}",
        "",
        "高度相关的同族速度微扰只说明局部连续性，不能相加成为泛化率。",
        "",
    ]
    (OUT / "DIVERSITY_ANALYSIS.md").write_text("\n".join(div_lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SUCCESS_DEMOS").mkdir(exist_ok=True)
    (OUT / "FAILURE_TRACES").mkdir(exist_ok=True)
    (OUT / "VIDEOS").mkdir(exist_ok=True)
    sim, snap, nominal, pads, pad_mu = _prepare()
    _enable_supports(sim)
    book = load_codebook()
    regression = _regression(sim, snap, nominal, pads, pad_mu, book)
    base = make_snap(sim)
    # The regression replay leaves the sim elsewhere. Rebuild the seated base.
    from experiments.recatch.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
        sim.set_guide_weld(True)
    restore_state(sim, None)
    ref = _ref_now(sim)
    base = make_snap(sim)
    manifest = []
    searches = []
    videos = 0
    for index, cell in enumerate(_cells()):
        ic_id = f"ic_{index:03d}_{cell['family']}_{cell['speed_mps']}"
        print("IC", ic_id, cell["kind"], flush=True)
        impact_snap, info = _impact(sim, base, nominal, cell)
        row = {
            "id": ic_id,
            "split": cell["split"],
            "family": cell["family"],
            "kind": cell["kind"],
            "azimuth_deg": cell["azimuth_deg"],
            "speed_mps": cell["speed_mps"],
            "z_off_m": cell["z_off_m"],
            "phi_deg": cell["phi_deg"],
            "delay_s": cell["delay_s"],
            "label": "INVALID",
            "airborne_required": False,
            "demo_success": False,
            "airborne_recatch": False,
            "k48_replay_kept": False,
            "fail_reason": info if isinstance(info, str) else "",
        }
        if impact_snap is None:
            manifest.append(row)
            print(" ", row["label"], row["fail_reason"], flush=True)
            continue
        feat = _features(sim)
        row.update({f"ic_{key}": value for key, value in feat.items() if not isinstance(value, (list, dict))})
        row["airborne_required"] = bool(info["bilateral_lost_before_ic"])
        row["contact_t"] = info["contact_t"]
        row["peak_normal_n"] = info["peak_normal_n"]
        base_out = _baseline(sim, impact_snap, nominal, ref, cell)
        if base_out.get("success"):
            row["label"] = "TRIVIAL"
            row["fail_reason"] = "nominal_without_recovery"
            manifest.append(row)
            print(" ", "TRIVIAL", flush=True)
            continue
        tried = []
        winner = None
        last_reason = "no_attempt"
        for mode, spec in _variants(bool(info["bilateral_lost_before_ic"])):
            if not info["bilateral_lost_before_ic"] and mode == "hold":
                pass
            restore_ballistic(sim, impact_snap)
            _apply_variant(sim, nominal, cell)
            restore_state(sim, None)
            rolled = _rollout(sim, spec, "recatch" if info["bilateral_lost_before_ic"] else mode)
            reason = rolled["phase"]
            if rolled["table_drop"] or rolled["table_touch"]:
                reason = "table_contact"
            last_reason = reason
            tried.append({"mode": mode, "phase": rolled["phase"], "table": rolled["table_touch"], "off": rolled["saw_off"], "re": rolled["saw_re"], "detail": reason})
            _save_trace(OUT / "FAILURE_TRACES" / f"{ic_id}_{mode}.npz", rolled["rows"])
            if rolled["phase"] != "handoff" or rolled["table_drop"] or rolled["table_touch"]:
                continue
            actions = np.stack(rolled["actions"]) if rolled["actions"] else np.zeros((0, 7), np.float32)
            replay_rows = _replay(sim, impact_snap, nominal, cell, actions)
            if not replay_rows or latched(sim) or _table_touch(sim):
                reason = "open_loop_diverged"
                last_reason = reason
                continue
            packed = representability(actions, book)
            if packed["max_linf"] is None or packed["max_linf"] > 0.25 + 1e-8:
                reason = "k48_not_representable"
                last_reason = reason
                continue
            decoded = np.asarray(packed["quantized"], np.float32)
            k_rows = _replay(sim, impact_snap, nominal, cell, decoded)
            if not k_rows or latched(sim) or _table_touch(sim) or k_rows[-1]["nL"] == 0 or k_rows[-1]["nR"] == 0:
                reason = "k48_replay_lost"
                last_reason = reason
                continue
            out, handoff = _finish(sim, ref)
            if not out.get("success"):
                reason = str(out.get("reason"))
                last_reason = reason
                tried[-1]["detail"] = reason
                continue
            last_reason = reason
            winner = {
                "mode": mode,
                "actions": actions,
                "tokens": np.asarray(packed["tokens"], np.int64),
                "residual": actions - book[np.asarray(packed["tokens"], np.int64)],
                "decoded": decoded,
                "rows": rolled["rows"],
                "k_rows": k_rows,
                "out": out,
                "handoff": handoff,
                "max_linf": packed["max_linf"],
                "airborne": bool(rolled["saw_off"] and rolled["saw_re"]),
            }
            break
        searches.append({"id": ic_id, "tries": tried, "won": winner is not None})
        if winner is None:
            row["label"] = "UNSOLVED"
            row["fail_reason"] = last_reason
            manifest.append(row)
            print(" ", "UNSOLVED", row["fail_reason"], flush=True)
            continue
        folder = OUT / "SUCCESS_DEMOS" / ic_id
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "actions.npz",
            actions=winner["actions"],
            tokens=winner["tokens"],
            residual=winner["residual"].astype(np.float32),
            decoded=winner["decoded"],
        )
        with (folder / "ic.pkl").open("wb") as handle:
            pickle.dump({"snap": impact_snap, "cell": cell, "impact": info, "ref": ref}, handle)
        (folder / "result.json").write_text(json.dumps({
            "nominal_reason": winner["out"].get("reason"),
            "end_z": None if winner["out"].get("end") is None else winner["out"]["end"].get("z"),
            "handoff_t": winner["handoff"].get("t"),
            "handoff_p_des": winner["handoff"].get("p_des"),
            "handoff_physics_edited": winner["handoff"].get("physics_edited"),
            "k48_max_linf": winner["max_linf"],
            "airborne_recatch": winner["airborne"],
            "oracle_during_search": True,
            "execution": "open_loop_step_legal",
        }, indent=2, default=str), encoding="utf-8")
        if videos < 6 and winner["rows"] and "qpos" in winner["rows"][0]:
            _render(sim, {"rows": winner["rows"]}, OUT / "VIDEOS" / f"{ic_id}.mp4", ic_id)
            videos += 1
        row["label"] = "RECOVERY_NEEDED"
        row["demo_success"] = True
        row["airborne_recatch"] = bool(winner["airborne"])
        row["k48_replay_kept"] = True
        row["fail_reason"] = ""
        row["handoff_t"] = winner["handoff"].get("t")
        row["end_z"] = None if winner["out"].get("end") is None else winner["out"]["end"].get("z")
        manifest.append(row)
        print(" ", "SUCCESS", row["airborne_recatch"], flush=True)
    fields = [
        "id", "split", "family", "kind", "label", "airborne_required", "demo_success", "airborne_recatch",
        "k48_replay_kept", "azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s", "contact_t",
        "peak_normal_n", "fail_reason", "ic_bottom_m", "ic_zo_m", "ic_nL", "ic_nR", "ic_gap_mm", "end_z", "handoff_t",
    ]
    _csv(OUT / "IC_MANIFEST.csv", manifest, fields)
    _csv(OUT / "SEARCH_RESULTS.csv", [
        {"id": item["id"], "won": item["won"], "tries": json.dumps(item["tries"])} for item in searches
    ], ["id", "won", "tries"])
    div = _diversity(manifest)
    _write_reports(regression, manifest, searches, div)
    print("DONE", json.dumps(div), flush=True)


if __name__ == "__main__":
    main()
