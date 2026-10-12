"""Independent autonomous episodes and the frozen train/validation/test split.

The 24-row benchmark is not this dataset. Suffixes of one impact are not
separate cases. This module does not train and does not change the geometry
switch. A demonstration is kept only when a verified skill, started at the
live geometry entry, still holds the object through absolute t = 12.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.autonomous_episode import (
    Z_ABORT,
    ball_substep,
    empty_ball,
    make_training_env,
)
from training.balanced_unified_recovery import EVAL_NATURAL, TRAIN_NATURAL
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import T_TASK, continue_long, physical_of
from training.full_3d_airborne_recatch_dataset import TRACK

RAW = ROOT / "results" / "diagnostics" / "raw" / "autonomous_training"

# One physical episode is one fresh nominal episode and one disturbance.
# Wrist reseating, gravity-inward sliding, and the open-finger recatch are
# recovery mechanisms played after entry. They are not extra disturbances.
IMPACT = (
    {"id": "impact_zm6", "v_mps": 6.5, "z_off_m": -0.006, "phi_deg": 0.0},
    {"id": "impact_z3", "v_mps": 6.5, "z_off_m": -0.003, "phi_deg": 0.0},
    {"id": "impact_zp6", "v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0},
    {"id": "impact_phi90", "v_mps": 6.5, "z_off_m": 0.0, "phi_deg": 90.0},
)
AIRBORNE = (
    {"id": "air_lat_pos", "speed_mps": 0.30, "sign": 1.0, "duration_s": 0.08},
    {"id": "air_lat_neg", "speed_mps": 0.30, "sign": -1.0, "duration_s": 0.08},
    {"id": "air_fast_pos", "speed_mps": 0.45, "sign": 1.0, "duration_s": 0.08},
    {"id": "air_slow_neg", "speed_mps": 0.20, "sign": -1.0, "duration_s": 0.08},
)

# Mechanisms with a verified snapshot recovery on the zm6 ball. The open-finger
# schedule and the inward slide are recovery skills, not disturbance episodes.
SKILLS = (
    {"name": "wrist_reseat", "pool": "wrist", "schedule": "wrist_reseat"},
    {"name": "gravity_inward", "pool": "sliding", "schedule": "gravity_inward"},
    {"name": "recatch_open20", "pool": "recatch", "schedule": "recatch_open20"},
)

# Natural offsets already used to train the balanced policy stay on the
# training side, except two held out for validation. The four published
# held-out windows stay on the test side.
_NAT_TRAIN = ("dz13.2", "dz13.5", "dz14.2", "dvz_m")
_NAT_VAL = ("dz14.6", "dvz_p")


def _natural(tag):
    for spec in TRAIN_NATURAL + EVAL_NATURAL:
        if spec["tag"] == tag:
            return {
                "id": "nl_" + tag,
                "dz_mm": spec["dz"],
                "dx_mm": spec.get("dx", 0.0),
                "dy_mm": spec.get("dy", 0.0),
                "dvz_mps": spec.get("dvz", 0.0),
            }
    raise KeyError(tag)


SPLIT = {
    "train": {
        "impact": ["impact_zm6"],
        "airborne": [],
        "natural": ["nl_" + t for t in _NAT_TRAIN],
        "mechanisms_on_impact_zm6": [s["name"] for s in SKILLS],
    },
    "validation": {
        "impact": ["impact_z3"],
        "airborne": ["air_lat_pos", "air_fast_pos"],
        "natural": ["nl_" + t for t in _NAT_VAL],
    },
    "test": {
        "impact": ["impact_zp6", "impact_phi90"],
        "airborne": ["air_lat_neg", "air_slow_neg"],
        "natural": ["nl_" + spec["tag"] for spec in EVAL_NATURAL],
    },
}


def episode_index():
    """Every independent episode, with the split it belongs to."""
    where = {}
    for part in ("train", "validation", "test"):
        for fam in ("impact", "airborne", "natural"):
            for i in SPLIT[part][fam]:
                where[i] = part
    rows = []
    for spec in IMPACT:
        rows.append({"id": spec["id"], "family": "impact", "split": where[spec["id"]], **spec})
    for spec in AIRBORNE:
        rows.append({"id": spec["id"], "family": "airborne", "split": where[spec["id"]], **spec})
    for tag in _NAT_TRAIN + _NAT_VAL + tuple(s["tag"] for s in EVAL_NATURAL):
        spec = _natural(tag)
        rows.append({"id": spec["id"], "family": "natural", "split": where[spec["id"]], **spec})
    return rows


def write_split():
    RAW.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": SPLIT,
        "episodes": episode_index(),
        "skills": [dict(s) for s in SKILLS],
        "note": "Thresholds are config/geometry_switch.yaml. This file does not retune them.",
    }
    (RAW / "split.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return RAW / "split.json"


def _until_impact_entry(env, spec):
    env.begin_nominal()
    ball = empty_ball()
    switch = env.switch
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            ball_substep(
                env.sim, env.gains, env.nominal, ball,
                spec["v_mps"], spec["z_off_m"], spec["phi_deg"], None,
            )
            if float(env.sim.data.time) >= T_TASK - 1e-9:
                break
        row = switch.measure(env.sim)
        if ball["released"]:
            switch.update_entry(env.sim)
        if ball["released"] and row["z"] < Z_ABORT:
            break
        if (not ball["released"]) and float(env.sim.data.time) > 8.0:
            break
    return ball


def demonstrate_skill(env, skill_name):
    """Play one verified schedule from the live zm6 geometry entry. No snapshot restore."""
    spec = next(s for s in IMPACT if s["id"] == "impact_zm6")
    skill = next(s for s in SKILLS if s["name"] == skill_name)
    ball = _until_impact_entry(env, spec)
    switch = env.switch
    if not switch.entered:
        return {"name": skill_name, "kept": False, "why": "no_entry", "t_entry": None}
    t_entry = float(env.sim.data.time)
    actions = schedules()[skill["schedule"]]
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_l, act_l, prev_l = [], [], []
    for act in actions:
        if float(env.sim.data.time) >= T_TASK - 1e-9:
            break
        obs = observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist)
        a3 = v3.v1_normalized_to_v3(np.asarray(act, np.float32))
        obs_l.append(np.asarray(obs, np.float32))
        act_l.append(a3)
        prev_l.append(prev.copy())
        v3.step_v3(env.sim, a3)
        prev = a3
        if float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
            return {
                "name": skill_name, "pool": skill["pool"], "kept": False,
                "why": "fell_during_skill", "t_entry": t_entry, "steps": len(obs_l),
            }
    if abs(float(env.sim.data.time) - t_entry) < 1e-9:
        return {"name": skill_name, "kept": False, "why": "clock_did_not_advance", "t_entry": t_entry}
    rec = continue_long(env.sim, TRACK, T_TASK, skill_name)
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    kept = y == 1
    out = {
        "name": skill_name,
        "pool": skill["pool"],
        "episode": "impact_zm6",
        "kept": bool(kept),
        "why": "RETAINED_TO_12" if kept else kind,
        "t_entry": t_entry,
        "t_impact": ball["t_impact"],
        "t_end": float(env.sim.data.time),
        "steps": len(obs_l),
        "clock_reset": False,
        "snapshot_restore": False,
    }
    if kept:
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RAW / f"demo_{skill_name}.npz",
            obs=np.stack(obs_l),
            act=np.stack(act_l),
            prev=np.stack(prev_l),
            pool=np.array(skill["pool"]),
            episode=np.array("impact_zm6"),
        )
    return out


def demonstrate_natural(env, tag, base=None):
    """Privileged intercept from the live geometry entry of one seated offset.

    The seated grasp is the initial nominal state. The offset is the disturbance.
    The intercept is not started by restoring a loss snapshot.
    """
    import training.natural_loss_recatch_data_scaling as scale
    from envs.airborne_obs import relative_state
    from training.autonomous_episode import fresh_nominal_state, nominal_substep, seat_base
    from training.full_3d_airborne_recatch_dataset import restore_to

    spec = next(s for s in TRAIN_NATURAL if s["tag"] == tag)
    sim = env.sim
    if base is None:
        base, _q = seat_base(sim, env.nominal, env.pads, env.pad_mu)
    info = restore_to(sim, base, env.nominal, env.pads, env.pad_mu)
    if not info.get("pose_ok", False):
        return {"name": tag, "kept": False, "why": "seat_failed"}
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    switch = env.switch
    env.switch = switch.__class__()
    switch = env.switch
    switch.capture_if_ready(sim)
    if switch.p0 is None:
        from controllers.geometry_switch import hand_pose
        switch.p0, switch.r0 = hand_pose(sim)
    scale.apply_spec(sim, spec)
    st = fresh_nominal_state()
    while float(sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(sim, TRACK, st)
        row = switch.measure(sim)
        switch.update_entry(sim)
        if row["z"] < Z_ABORT:
            break
    if not switch.entered:
        return {"name": tag, "pool": "natural", "kept": False, "why": "no_entry", "t_entry": None}
    t_entry = float(sim.data.time)
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_l, act_l, prev_l = [], [], []
    phase = "chase"
    both_hold = 0.0
    for _ in range(int(2.4 / v3.DT_POLICY)):
        if float(sim.data.time) >= T_TASK - 1e-9:
            break
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        excess = float(rel["p_rel_h"][2] - pinch[2])
        bilateral = int(o["nL"]) > 0 and int(o["nR"]) > 0
        near = (
            abs(excess) < 0.012
            and abs(float(rel["p_rel_h"][0] - pinch[0])) < 0.012
            and abs(float(rel["p_rel_h"][1] - pinch[1])) < 0.012
        )
        vrel = rel["v_obj_w"] - rel["v_hand_w"]
        slow = abs(float(vrel[2])) < 0.15 and float(np.linalg.norm(vrel[:2])) < 0.20
        if phase == "chase" and float(o["aperture"]) > 0.036 and near and slow:
            phase = "close"
        if phase == "close":
            if bilateral:
                both_hold += v3.DT_POLICY
                if both_hold >= 0.04:
                    phase = "hold"
            else:
                both_hold = 0.0
        action, _v = v3.intercept_action(sim, pinch, phase)
        obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
        act = np.asarray(action, np.float32)
        obs_l.append(np.asarray(obs, np.float32))
        act_l.append(act)
        prev_l.append(prev.copy())
        v3.step_v3(sim, act)
        prev = act
        if float(o["obj_z"]) < Z_ABORT:
            return {
                "name": tag, "pool": "natural", "episode": "nl_" + tag,
                "kept": False, "why": "fell_during_intercept", "t_entry": t_entry,
                "steps": len(obs_l),
            }
    rec = continue_long(sim, TRACK, T_TASK, tag)
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    kept = y == 1
    out = {
        "name": tag, "pool": "natural", "episode": "nl_" + tag,
        "kept": bool(kept), "why": "RETAINED_TO_12" if kept else kind,
        "t_entry": t_entry, "t_end": float(sim.data.time), "steps": len(obs_l),
        "snapshot_restore": False,
    }
    if kept:
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RAW / f"demo_nl_{tag}.npz",
            obs=np.stack(obs_l), act=np.stack(act_l), prev=np.stack(prev_l),
            pool=np.array("natural"), episode=np.array("nl_" + tag),
        )
    return out


def generate_train_demos(env=None):
    """Training-side impact skills only. Does not train. Does not touch the test split."""
    env = env or make_training_env()
    rows = [demonstrate_skill(env, s["name"]) for s in SKILLS]
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "demo_status.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
