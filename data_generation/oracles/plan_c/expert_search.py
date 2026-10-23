"""Active Plan C expert search: surface-aware catch → lift → SUCCESS_NEW.

Privileged contact-surface features are offline-only. Runtime stays 39D / legal 7D.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

import numpy as np

from controllers.recovery.geometry_switch import GeometrySwitch, hand_pose
from data_generation.oracles.plan_c.catch_stabilization import fast_world_lift_sequence
from data_generation.oracles.plan_c.contact_surface import classify_contacts, score_catch_out
from data_generation.oracles.plan_c.dynamic_catch import (
    CatchParams,
    catch_param_grid,
    dynamic_catch_sequence,
    propose_catch_interceptions,
)
from data_generation.oracles.plan_c.future_trajectory import (
    hold_action,
    pad_bilateral,
    pose_correct_and_handoff,
    record_no_intervention_future,
)
from data_generation.oracles.plan_c.interception import (
    FULL_DETACH_ACTIVE_RECATCH,
    rebuild_settled_root,
    reconstruct_impact_timeline,
)
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.outcomes.nominal_pose import capture_ref
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import make_snap, restore_ballistic

DT_POL = 0.02


def make_ref(sim):
    sw = GeometrySwitch()
    p0, r0 = hand_pose(sim)
    sw.p0 = np.asarray(p0, float).copy()
    sw.r0 = np.asarray(r0, float).copy()
    return capture_ref(sim, sw, live=True)


def physical_root_fingerprint(cfg: dict, snap: dict) -> str:
    """Stable id for one physical impact root (not trajectory variants)."""
    qpos = np.asarray(snap.get("qpos"), float).reshape(-1)
    # Object freejoint is typically the last 7 qpos of the object; hash full qpos + cfg knobs.
    cfg_keys = (
        "axial_offset",
        "z",
        "tilt_deg",
        "approach",
        "speed",
        "offset",
        "z_off",
    )
    payload = {
        "cfg": {k: cfg.get(k) for k in cfg_keys},
        "qpos": qpos.round(8).tolist(),
        "time": float(snap.get("time", 0.0)),
    }
    raw = repr(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def build_physical_root(sim, cfg: dict, *, max_s: float = 1.2) -> dict | None:
    """Settled grasp → real ball impact → branch-A post-impact snap with genuine loss."""
    pre = rebuild_settled_root(sim, cfg)
    if not pre.get("bilateral"):
        return None
    ref = make_ref(sim)
    tl = reconstruct_impact_timeline(sim, cfg=cfg, ball=None, max_s=max_s)
    ev = tl["events"]
    snaps = tl["snaps"]
    t_imp = ev.get("first_ball_contact_t")
    t_sus = ev.get("first_sustained_loss_t")
    if t_imp is None or t_sus is None:
        return None
    # Prefer branch A (first ball contact / policy boundary); fall back to sustained-loss snap.
    snap = snaps.get("A") or snaps.get("E")
    if snap is None:
        return None
    # Require measurable detach window before ground.
    t_g = ev.get("first_ground_drop_t")
    if t_g is not None and float(t_g) - float(t_imp) < 0.08:
        return None
    fp = physical_root_fingerprint(cfg, snap)
    return {
        "cfg": dict(cfg),
        "snap": snap,
        "ref": ref,
        "events": ev,
        "pre": pre,
        "fingerprint": fp,
        "slip_s": float(t_sus) - float(t_imp),
        "airborne_s": None if t_g is None else float(t_g) - float(t_imp),
    }


def search_catch(
    sim,
    snap,
    ref,
    *,
    rng: np.random.Generator,
    max_rollouts: int = 40,
) -> dict:
    """Surface-aware catch search; returns best out + sidewall shortlist."""
    future = record_no_intervention_future(sim, snap, max_s=0.90)
    cands = propose_catch_interceptions(future, n_times=8, n_axial=5)
    pairs = catch_param_grid(rng, cands)
    best = None
    sidewall = []
    tried = 0
    funnel = {
        "rollouts": 0,
        "jaw_entered": 0,
        "end_bilateral": 0,
        "full_detach": 0,
        "opposing_sidewall": 0,
        "sidewall_partial": 0,
        "end_face": 0,
        "continuous_150": 0,
    }
    for params, cand in pairs:
        if tried >= max_rollouts:
            break
        out = dynamic_catch_sequence(sim, snap, ref, params, cand, horizon_s=0.70)
        tried += 1
        funnel["rollouts"] += 1
        funnel["jaw_entered"] += int(out.get("jaw_entered") or 0)
        funnel["end_bilateral"] += int(out.get("end_bilateral") or 0)
        funnel["full_detach"] += int(out.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH)
        funnel["continuous_150"] += int(out.get("continuous_150") or 0)
        kind = out.get("grasp_kind")
        if kind == "opposing_sidewall":
            funnel["opposing_sidewall"] += 1
        elif kind == "sidewall_partial":
            funnel["sidewall_partial"] += 1
        elif kind == "end_face_pinch":
            funnel["end_face"] += 1
        sc = score_catch_out(out)
        packed = {**out, "score": sc, "params": params, "cand": cand}
        if best is None or sc > best["score"]:
            best = packed
        if (
            out.get("mechanism") == FULL_DETACH_ACTIVE_RECATCH
            and out.get("end_bilateral")
            and kind in ("opposing_sidewall", "sidewall_partial")
            and float(out.get("e_theta_end") or 99) < 35.0
        ):
            sidewall.append(packed)
    sidewall.sort(key=lambda o: o["score"], reverse=True)
    return {"best": best, "sidewall": sidewall, "funnel": funnel, "future": future}


def lift_and_validate(
    sim,
    snap,
    ref,
    catch_out: dict,
    *,
    vz_up: float = 0.12,
    target_z: float = 0.55,
) -> dict:
    """Replay catch → fast world lift → conditional pose_correct → SUCCESS_NEW."""
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    prefix: list[np.ndarray] = []
    for a in catch_out["capture_actions"]:
        a = np.asarray(a, np.float32)
        prefix.append(a.copy())
        step_legal(sim, a)
        if latched(sim):
            return {"SUCCESS_NEW": False, "stage": "catch", "reason": "GROUND_DROP"}
    if not pad_bilateral(sim)["bilateral"]:
        return {"SUCCESS_NEW": False, "stage": "catch", "reason": "NO_BILATERAL"}
    surf_brake = classify_contacts(sim)
    m_brake = measure(sim, U_TGT, P_TGT)
    brake_snap = make_snap(sim)

    lift = fast_world_lift_sequence(sim, brake_snap, vz_up=vz_up, target_z=target_z, max_s=8.0)
    for a in lift.get("actions") or []:
        prefix.append(np.asarray(a, np.float32).copy())
    if latched(sim) or not pad_bilateral(sim)["bilateral"]:
        return {
            "SUCCESS_NEW": False,
            "stage": "lift",
            "reason": "LOST_LIFT",
            "surf_brake": surf_brake.get("grasp_kind"),
            "obj_z": lift.get("obj_z_end"),
        }
    surf_elev = classify_contacts(sim)
    m_elev = measure(sim, U_TGT, P_TGT)
    elev_snap = make_snap(sim)
    if surf_elev.get("grasp_kind") == "end_face_pinch":
        return {
            "SUCCESS_NEW": False,
            "stage": "lift",
            "reason": "END_FACE_AFTER_LIFT",
            "surf_elev": surf_elev.get("grasp_kind"),
            "e_theta": m_elev["e_theta_deg"],
        }

    # Conditional pose correction + handoff; if DROP, try light in-hand reseat then retry once.
    pose_acts, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
    for a in pose_acts:
        prefix.append(np.asarray(a, np.float32).copy())
    if not meta.get("SUCCESS_NEW") and meta.get("reason") in (
        "DROP_AFTER_HANDOFF",
        "LOST_GRASP",
        "NOMINAL_LIFT_NOT_COMPLETED",
    ):
        restore_ballistic(sim, elev_snap)
        restore_drop(sim, None)
        arm(sim)
        from controllers.recovery.action_map import TAU_SECURE
        from data_generation.oracles.plan_b.release_recatch import _action_from
        from data_generation.oracles.plan_c.future_trajectory import object_axis

        for _ in range(25):
            Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
            po = np.array(sim.data.xpos[sim.ids.object_body], float)
            ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
            prel = Rh.T @ (po - ph)
            err_p = np.asarray(P_TGT, float) - prel
            v_h = np.clip(-err_p * 2.5, -0.22, 0.22)
            u = object_axis(sim)
            uh = Rh.T @ u
            uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
            axis = np.cross(uh, np.asarray(U_TGT, float))
            w_body = np.clip(axis * 0.5, -0.5, 0.5)
            a = _action_from(sim, Rh @ v_h, w_body, TAU_SECURE)
            a[6] = 0.85
            prefix.append(a.copy())
            step_legal(sim, a)
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                break
        for _ in range(10):
            prefix.append(hold_action(1.0))
            step_legal(sim, hold_action(1.0))
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                break
        if pad_bilateral(sim)["bilateral"] and not latched(sim):
            elev_snap = make_snap(sim)
            surf_elev = classify_contacts(sim)
            m_elev = measure(sim, U_TGT, P_TGT)
            pose_acts2, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
            for a in pose_acts2:
                prefix.append(np.asarray(a, np.float32).copy())
    return {
        "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
        "stage": "handoff",
        "reason": meta.get("reason"),
        "prefix_actions": prefix,
        "elev_snap": elev_snap,
        "surf_brake": {
            "grasp_kind": surf_brake.get("grasp_kind"),
            "opposing_sidewall": int(surf_brake.get("opposing_sidewall") or 0),
            "n_sidewall": int(surf_brake.get("n_sidewall") or 0),
            "n_end_face": int(surf_brake.get("n_end_face") or 0),
            "contact_s_mean": surf_brake.get("contact_s_mean"),
        },
        "surf_elev": {
            "grasp_kind": surf_elev.get("grasp_kind"),
            "opposing_sidewall": int(surf_elev.get("opposing_sidewall") or 0),
            "n_sidewall": int(surf_elev.get("n_sidewall") or 0),
            "n_end_face": int(surf_elev.get("n_end_face") or 0),
            "contact_s_mean": surf_elev.get("contact_s_mean"),
        },
        "m_brake": {k: m_brake[k] for k in ("e_theta_deg", "e_pos_mm", "obj_z")},
        "m_elev": {k: m_elev[k] for k in ("e_theta_deg", "e_pos_mm", "obj_z")},
        "lift": {
            "reached": lift.get("reached_target_z"),
            "obj_z_end": lift.get("obj_z_end"),
            "vz_up": vz_up,
        },
        "handoff_meta": {
            "SUCCESS_NEW": bool(meta.get("SUCCESS_NEW")),
            "reason": meta.get("reason"),
            "success_12s": bool(meta.get("success_12s")),
            "m_handoff": meta.get("m_handoff"),
            "n_pose_actions": meta.get("n_pose_actions"),
        },
        "catch_score": catch_out.get("score"),
        "catch_kind": catch_out.get("grasp_kind"),
        "mechanism": catch_out.get("mechanism"),
        "params": catch_out.get("params"),
        "cand": catch_out.get("cand"),
        "capture_actions": [np.asarray(a, np.float32) for a in catch_out["capture_actions"]],
        "vrel_at_first_contact": catch_out.get("vrel_at_first_contact"),
        "max_contact_free_s": catch_out.get("max_contact_free_s"),
    }


def exact_replay_3of3(snap, ref, prefix_actions: list, elev_snap) -> dict:
    """Require 3/3 SUCCESS_NEW from elevated (stabilize + handoff path)."""
    wins = 0
    details = []
    for i in range(3):
        # Independent elevated handoff
        from envs.simulation.ground_sim import make_ground_sim

        sim, _ = make_ground_sim()
        restore_ballistic(sim, elev_snap)
        restore_drop(sim, None)
        arm(sim)
        for _ in range(8):
            step_legal(sim, hold_action(1.0))
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                break
        _a, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
        ok = bool(meta.get("SUCCESS_NEW"))
        details.append({"i": i, "SUCCESS_NEW": ok, "reason": meta.get("reason")})
        if ok:
            wins += 1
    # Full-tape spot check (1×) — catch snap → prefix → handoff
    from envs.simulation.ground_sim import make_ground_sim

    sim, _ = make_ground_sim()
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    lost = False
    for a in prefix_actions:
        step_legal(sim, np.asarray(a, np.float32))
        if latched(sim):
            lost = True
            break
    full_ok = False
    full_reason = "LOST_PREFIX"
    if not lost and pad_bilateral(sim)["bilateral"]:
        _a, meta = pose_correct_and_handoff(sim, ref, max_s=4.0)
        full_ok = bool(meta.get("SUCCESS_NEW"))
        full_reason = meta.get("reason")
    return {
        "elev_wins": wins,
        "elev_n": 3,
        "elev_ok_3of3": wins == 3,
        "elev_details": details,
        "full_ok": full_ok,
        "full_reason": full_reason,
        "ok": wins == 3 and full_ok,
    }


def pack_expert(
    *,
    root_id: str,
    fingerprint: str,
    cfg: dict,
    snap,
    ref,
    validated: dict,
    events: dict | None = None,
) -> dict:
    pref = validated["prefix_actions"]
    cap = validated["capture_actions"]
    params = validated.get("params")
    cand = validated.get("cand") or {}
    return {
        "root_id": root_id,
        "fingerprint": fingerprint,
        "cfg": cfg,
        "snap": snap,
        "catch_snap": snap,
        "elev_snap": validated["elev_snap"],
        "ref": ref,
        "actions": np.stack(pref, axis=0),
        "prefix_actions": np.stack(pref, axis=0),
        "capture_actions": np.stack(cap, axis=0),
        "SUCCESS_NEW": True,
        "mechanism": validated.get("mechanism") or FULL_DETACH_ACTIVE_RECATCH,
        "grasp_kind": (validated.get("surf_elev") or {}).get("grasp_kind"),
        "opposing_sidewall": int((validated.get("surf_elev") or {}).get("opposing_sidewall") or 0),
        "surface_brake": validated.get("surf_brake"),
        "surface_elev": validated.get("surf_elev"),
        "error_at_elevated": validated.get("m_elev"),
        "handoff_meta": validated.get("handoff_meta"),
        "params": params if not hasattr(params, "__dataclass_fields__") else asdict(params),
        "cand": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cand.items()
        },
        "events": events,
        "vrel_at_first_contact": validated.get("vrel_at_first_contact"),
        "max_contact_free_s": validated.get("max_contact_free_s"),
        "lift": validated.get("lift"),
    }


def diverse_impact_configs(n: int, seed: int = 20261021) -> list[dict]:
    """Expanded stratified impact configs for Plan C scale-up."""
    from data_generation.oracles.plan_c.future_trajectory import PilotBudget, impact_config_grid

    base = impact_config_grid(PilotBudget(max_impact_configs=max(n, 30)), seed=seed)
    rng = np.random.default_rng(seed + 7)
    # Extra gradual-slip / lateral family (proven recoverable)
    extra = []
    for sp in (6.0, 6.5, 7.0, 7.5, 8.0):
        for ap in ("lateral", "oblique", "top", "side"):
            for ax in (0.018, 0.022, 0.026, 0.028):
                for off in (-0.012, -0.006, 0.0, 0.006, 0.012):
                    for tilt in (-20.0, -10.0, 0.0, 10.0, 20.0):
                        for z in (0.55, 0.60, 0.65):
                            extra.append({
                                "config_id": f"pc_s{sp}_{ap}_a{ax}_t{tilt}_o{off}_z{z}",
                                "axial_offset": float(ax),
                                "z": float(z),
                                "tilt_deg": float(tilt),
                                "approach": ap,
                                "speed": float(sp),
                                "offset": float(off),
                                "z_off": float(rng.uniform(-0.005, 0.008)),
                            })
    rng.shuffle(extra)
    # Merge unique-ish by rounded knobs
    seen = set()
    out = []
    for cfg in list(base) + extra:
        key = (
            round(float(cfg["axial_offset"]), 4),
            round(float(cfg["z"]), 3),
            round(float(cfg["tilt_deg"]), 1),
            str(cfg["approach"]),
            round(float(cfg["speed"]), 2),
            round(float(cfg["offset"]), 4),
            round(float(cfg.get("z_off", 0.0)), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        row = dict(cfg)
        row["config_id"] = f"gbi_{len(out):03d}"
        out.append(row)
        if len(out) >= n:
            break
    return out
