"""Replay frozen Plan C golden experts. Does not generate new tapes.

These fixtures predate the Precision Oracle refactor. SUCCESS_NEW is the
historical Plan C handoff criterion and is not relabeled as experts_v2.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from controllers.recovery.action_map import AIRBORNE_V3
from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.handoff_from_state import evaluate_handoff_from_state
from training.recovery_execution import restore_ballistic

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "unified_experts_v1"
DT = 0.02
PLAN_C_SRC = ROOT / "results" / "diagnostics" / "ground_plan_c_final_success" / "raw"
PLAN_C_SCALE = ROOT / "results" / "diagnostics" / "ground_plan_c_scale50" / "raw" / "accepted"
PLAN_C_SIDEWALL = ROOT / "results" / "diagnostics" / "ground_plan_c_sidewall_research" / "raw"
PLAN_C_ROOTS = ["gbi_010", "gbi_022"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def discover_plan_c_roots() -> list[tuple[str, Path]]:
    """Scale50 accepted experts first; golden fixtures only if not already present."""
    out: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()
    seen_fps: set[str] = set()

    def _fp_of(path: Path) -> str | None:
        try:
            blob = pickle.loads(path.read_bytes())
            return blob.get("fingerprint")
        except Exception:
            return None

    def _add(rid: str, path: Path) -> None:
        if not path.exists():
            return
        resolved = str(path.resolve())
        if resolved in seen_paths:
            return
        fp = _fp_of(path)
        if fp and fp in seen_fps:
            return
        seen_paths.add(resolved)
        if fp:
            seen_fps.add(fp)
        out.append((rid, path))

    scale_files = sorted(PLAN_C_SCALE.glob("*__SUCCESS_NEW_expert.pkl")) if PLAN_C_SCALE.is_dir() else []
    if scale_files:
        for path in scale_files:
            stem = path.name[: -len("__SUCCESS_NEW_expert.pkl")]
            _add(stem, path)
        return out
    for rid in PLAN_C_ROOTS:
        _add(rid, PLAN_C_SRC / f"{rid}__SUCCESS_NEW_expert.pkl")
    _add("gbi_014", PLAN_C_SIDEWALL / "gbi_014__SUCCESS_NEW_sidewall_expert.pkl")
    return out


def load_plan_c_expert(root_id: str, src: Path | None = None) -> tuple[dict, Path]:
    if src is None:
        golden = PLAN_C_SRC / f"{root_id}__SUCCESS_NEW_expert.pkl"
        if root_id in PLAN_C_ROOTS and golden.is_file():
            src = golden
        else:
            pairs = {rid: path for rid, path in discover_plan_c_roots()}
            if root_id not in pairs:
                raise FileNotFoundError(f"missing Plan C expert for {root_id}")
            src = pairs[root_id]
    blob = pickle.loads(src.read_bytes())
    if "catch_snap" not in blob and "snap" in blob:
        blob["catch_snap"] = blob["snap"]
    if "actions" not in blob and "prefix_actions" in blob:
        blob["actions"] = blob["prefix_actions"]
    return blob, src


def _obs(sim, hist) -> np.ndarray:
    obs = np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32)
    if obs.shape != (OBS_DIM_AIR,):
        raise RuntimeError(f"obs dim {obs.shape}, expected {(OBS_DIM_AIR,)}")
    return obs


def replay_plan_c(root_id: str, n_replay: int = 3, src: Path | None = None) -> tuple[dict, bytes, list]:
    expert, src_path = load_plan_c_expert(root_id, src=src)
    actions = np.asarray(expert["actions"], np.float32)
    snap = expert["catch_snap"]
    ref = expert["ref"]

    sim, _ = make_ground_sim()
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    hist = ObservableObsState(DT)
    obs_seq = []
    act_seq = []
    for action in actions:
        obs_seq.append(_obs(sim, hist))
        action = np.asarray(action, np.float32)
        act_seq.append(action.copy())
        step_legal(sim, action)
        if latched(sim):
            break
    obs_arr = np.stack(obs_seq, axis=0)
    act_arr = np.stack(act_seq, axis=0)
    timestamps = (np.arange(len(act_arr), dtype=np.float64) * DT).tolist()

    replay_rows = []
    for i in range(n_replay):
        sim_r, _ = make_ground_sim()
        restore_ballistic(sim_r, snap)
        restore_drop(sim_r, None)
        arm(sim_r)
        lost = False
        for action in actions:
            step_legal(sim_r, np.asarray(action, np.float32))
            if latched(sim_r):
                lost = True
                break
        if lost:
            replay_rows.append({"replay_i": i, "SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF"})
            continue
        meta = evaluate_handoff_from_state(sim_r, ref, stabilize_n=0)
        replay_rows.append({
            "replay_i": i,
            "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
            "reason": meta["reason"],
            "pre_pos_mm": meta.get("pre_pos_mm"),
            "pre_theta": meta.get("pre_theta"),
            "success_12s": int(meta.get("success_12s") or 0),
        })

    n_ok = sum(int(row["SUCCESS_NEW"]) for row in replay_rows)
    handoff_meta = expert.get("handoff_meta") or {}
    phys_id = expert.get("fingerprint") or root_id
    cfg_id = (expert.get("cfg") or {}).get("config_id") or expert.get("root_id") or root_id.split("__")[0]
    expert_key = f"{cfg_id}_{str(phys_id)[:10]}"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "expert_id": f"plan_c__{expert_key}",
        "plan_family": "C",
        "physical_root_id": expert_key,
        "scene": "GROUND_SCENE",
        "environment_version": "ground_ballistic_v1",
        "initial_physical_state_ref": {
            "source_pkl": str(src_path.relative_to(ROOT)).replace("\\", "/"),
            "snap_key": "catch_snap",
        },
        "actions_7d": act_arr,
        "observations_39d": obs_arr,
        "action_timestamps_s": timestamps,
        "control_period_s": DT,
        "handoff": {
            "token_id": 47,
            "timing": "immediate_after_final_action",
            "pre_pos_mm": handoff_meta.get("pre_pos_mm"),
            "pre_theta_deg": handoff_meta.get("pre_theta"),
        },
        "nominal_continuation_outcome": replay_rows[0].get("reason") if replay_rows else None,
        "SUCCESS_12S": bool(n_ok == n_replay),
        "SUCCESS_NEW": bool(n_ok == n_replay),
        "exact_replay": {"n": n_replay, "n_pass": n_ok, "rows": replay_rows},
        "contact_grasp_mechanism": expert.get("mechanism") or "FULL_DETACH_ACTIVE_RECATCH",
        "contact_surface": expert.get("surface_elev") or expert.get("grasp_kind"),
        "physical_root_fingerprint": expert.get("fingerprint"),
        "source_experiment": str(src_path.parent.relative_to(ROOT)).replace("\\", "/"),
        "provenance": {
            "expert_pkl": str(src_path.relative_to(ROOT)).replace("\\", "/"),
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "action_interface": "legal_7d",
            "obs_version": "obs_gt_airborne_39",
            "privileged_runtime": False,
        },
        "action_map": AIRBORNE_V3,
    }
    raw = pickle.dumps({
        **{k: v for k, v in payload.items() if k not in ("actions_7d", "observations_39d")},
        "actions_7d": act_arr,
        "observations_39d": obs_arr,
        "catch_snap": snap,
        "ref": ref,
    })
    payload["data_integrity_checksum"] = sha256_bytes(raw)
    return payload, raw, replay_rows
