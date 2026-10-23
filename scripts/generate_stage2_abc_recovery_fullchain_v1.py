#!/usr/bin/env python
"""A/B/C Recovery-entry → Stage-1 → Precision → HANDOFF → 12s expert dataset.

Uses the frozen speed-optimized Precision Oracle (defaults in PrecisionConfig).
Writes results/data_generation/stage2_abc_recovery_fullchain_v1/
Does not overwrite prior datasets. Does not start SFT/RL.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.precision_goals import assess_need  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    PrecisionConfig,
    entry_gate,
    measure_errors,
    probe_direct_handoff,
    run_precision,
)
from envs.observations.airborne_obs import OBS_VERSION, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

OUT = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
MANIFEST_UE = ROOT / "data/unified_experts/manifest.jsonl"
HARD_IC_DIR = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
FREEZE_MANIFEST = (
    ROOT / "results/diagnostics/precision_oracle_speed_optimization/freeze/ORACLE_MANIFEST.json"
)
K49_CONFIG = "K49_tok48_wy0p25"
DEDUP = 1e-3
DATASET_VERSION = "stage2_abc_recovery_fullchain_v1"
ORACLE_TAG = "precision_speed_opt_v1"


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


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


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


def _truthy(x) -> bool:
    """Safe bool for scalars / 0-d arrays (avoid numpy ambiguity errors)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return bool(np.asarray(x).reshape(-1)[0]) if x.size else False
    return bool(x)


def observe(env) -> np.ndarray:
    return observe_airborne(
        env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist
    )


def capture_fp(cap: dict) -> str:
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / DEDUP).astype(np.int64).tobytes()).hexdigest()[:16]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def nearest_motion_token(table: np.ndarray, action: np.ndarray) -> tuple[int, float, np.ndarray]:
    a = np.asarray(action, np.float32).reshape(7)
    best_i, best_d = 0, float("inf")
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < best_d:
            best_i, best_d = i, d
    return int(best_i), float(best_d), table[best_i].copy()


def quantize_actions(actions: list, table: np.ndarray) -> dict:
    acts = np.asarray(actions, np.float32)
    if acts.size == 0:
        return {
            "tokens": np.zeros((0,), np.int64),
            "proto_actions": np.zeros((0, 7), np.float32),
            "residuals": np.zeros((0, 7), np.float32),
            "recon_actions": np.zeros((0, 7), np.float32),
            "max_l2": 0.0,
            "mean_l2": 0.0,
            "max_recon_l2": 0.0,
            "residual_clipped_frac": 0.0,
        }
    tokens, protos, residuals, recons, dists, rl2s = [], [], [], [], [], []
    clipped = 0
    for a in acts:
        ti, d, ap = nearest_motion_token(table, a)
        raw = a - ap
        res = np.clip(raw, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw - res))) > 1e-8:
            clipped += 1
        recon = np.clip(ap + res, -1.0, 1.0).astype(np.float32)
        tokens.append(ti)
        protos.append(ap)
        residuals.append(res)
        recons.append(recon)
        dists.append(d)
        rl2s.append(float(np.linalg.norm(recon - a)))
    dist = np.asarray(dists, np.float32)
    rl2 = np.asarray(rl2s, np.float32)
    return {
        "tokens": np.asarray(tokens, np.int64),
        "proto_actions": np.asarray(protos, np.float32),
        "residuals": np.asarray(residuals, np.float32),
        "recon_actions": np.asarray(recons, np.float32),
        "max_l2": float(dist.max()),
        "mean_l2": float(dist.mean()),
        "max_recon_l2": float(rl2.max()),
        "residual_clipped_frac": float(clipped / len(acts)),
    }


def replay_actions_record_obs(env, start_cap, actions: list, ref) -> dict:
    restore_capture(env, start_cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_list = []
    for a in actions:
        obs_list.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "observations": np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32),
    }


def replay_tokens(env, start_cap, tokens, table, ref, residuals=None) -> dict:
    restore_capture(env, start_cap)
    res = None if residuals is None else np.asarray(residuals, np.float32)
    for i, t in enumerate(tokens):
        a = table[int(t)].copy()
        if res is not None and i < len(res):
            a = np.clip(a + np.clip(res[i], -CAP, CAP), -1.0, 1.0).astype(np.float32)
        step_legal(env.sim, a)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
    }


def load_ue() -> list[dict]:
    return [json.loads(l) for l in MANIFEST_UE.read_text(encoding="utf-8").splitlines() if l.strip()]


def resolve_recovery_ab(row: dict) -> dict | None:
    ref = row["initial_physical_state_ref"]
    snap = ROOT / ref["snapshot"]
    cont = ROOT / ref["continuous_npz"]
    if not snap.exists() or not cont.exists():
        return None
    fam = row["plan_family"]
    return {
        "family": fam,
        "source_label": fam,
        "ic_id": row["physical_root_id"],
        "expert_id": row["expert_id"],
        "lineage_id": f"{fam}__{row['physical_root_id']}",
        "snapshot_path": snap,
        "stage1_npz": cont,
        "ue_row": row,
        "kind": "UE_AB",
    }


def resolve_recovery_c(row: dict) -> dict | None:
    ref = row["initial_physical_state_ref"]
    pkl = ROOT / ref["source_pkl"]
    if not pkl.exists():
        return None
    return {
        "family": "C",
        "source_label": "C",
        "ic_id": row["physical_root_id"],
        "expert_id": row["expert_id"],
        "lineage_id": f"C__{row['physical_root_id']}",
        "expert_pkl": pkl,
        "ue_row": row,
        "kind": "UE_C",
    }


def restore_recovery_ab(env, meta: dict) -> tuple[dict, dict]:
    root = pickle.loads(meta["snapshot_path"].read_bytes())
    restore_capture(env, root["capture"])
    return root["capture"], root["ref"]


def restore_recovery_c(env, meta: dict) -> tuple[dict, dict]:
    pack = pickle.loads(meta["expert_pkl"].read_bytes())
    snap = pack.get("catch_snap") or pack.get("snap") or pack.get("elev_snap")
    if snap is None:
        raise KeyError(f"no catch/elev snap in {meta.get('expert_pkl')}")
    # Ballistic packs use flat qpos keys; capture dicts use physics.
    if isinstance(snap, dict) and "physics" in snap and "qpos" not in snap:
        restore_capture(env, snap)
        return snap, pack.get("ref") or snap.get("ref")
    restore_ballistic(env.sim, snap)
    restore_drop(env.sim, None)
    arm(env.sim)
    cap = take_capture(env)
    return cap, pack["ref"]


def run_stage1_ab(env, recovery_cap, ref, cont_path: Path) -> dict:
    acts = np.asarray(np.load(cont_path)["actions"], np.float32)
    restore_capture(env, recovery_cap)
    used = []
    stable_i = None
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        used.append(np.asarray(a, np.float32).copy())
        if (i + 1) % 5 != 0 and i + 1 < len(acts):
            continue
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        if _truthy(stab.get("stable")):
            gate = entry_gate(env)
            restore_capture(env, cap)
            if _truthy(gate.get("ok")):
                stable_i = len(used)
                break
    if stable_i is None:
        # consume full tape then check
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        gate = entry_gate(env)
        restore_capture(env, cap)
        if not (_truthy(stab.get("stable")) and _truthy(gate.get("ok"))):
            return {
                "ok": False,
                "reason": "stage1_not_stable",
                "actions": used,
                "capture": cap,
                "n_actions": len(used),
            }
        stable_i = len(used)
    # rebuild exact prefix
    restore_capture(env, recovery_cap)
    for a in used[:stable_i]:
        step_legal(env.sim, a)
    cap = take_capture(env)
    return {
        "ok": True,
        "actions": used[:stable_i],
        "capture": cap,
        "n_actions": stable_i,
        "reason": "stable_after_stage1",
    }


def run_stage1_c(env, recovery_cap, ref, expert_pkl: Path) -> dict:
    pack = pickle.loads(expert_pkl.read_bytes())
    raw_acts = pack.get("actions")
    if raw_acts is None:
        raw_acts = pack.get("prefix_actions")
    acts = np.asarray(raw_acts, np.float32)
    restore_capture(env, recovery_cap)
    used = []
    stable_i = None
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        used.append(np.asarray(a, np.float32).copy())
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        if z < 0.48:
            continue
        if (i + 1) % 5 != 0 and i + 1 < len(acts):
            continue
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        if _truthy(stab.get("stable")):
            gate = entry_gate(env)
            restore_capture(env, cap)
            if _truthy(gate.get("ok")):
                stable_i = len(used)
                break
    if stable_i is None and pack.get("elev_snap") is not None:
        # Fall back: restore elevated snap; Stage-1 actions = full prefix to elev by replay count
        restore_ballistic(env.sim, pack["catch_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        used = []
        for a in acts:
            step_legal(env.sim, np.asarray(a, np.float32))
            used.append(np.asarray(a, np.float32).copy())
            z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            if z >= 0.50:
                break
        restore_ballistic(env.sim, pack["elev_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        gate = entry_gate(env)
        restore_capture(env, cap)
        if _truthy(stab.get("stable")) and _truthy(gate.get("ok")):
            return {
                "ok": True,
                "actions": used,
                "capture": cap,
                "n_actions": len(used),
                "reason": "elev_snap_fallback",
            }
        return {
            "ok": False,
            "reason": "stage1_c_not_stable",
            "actions": used,
            "capture": cap,
            "n_actions": len(used),
        }
    if stable_i is None:
        cap = take_capture(env)
        return {
            "ok": False,
            "reason": "stage1_c_not_stable",
            "actions": used,
            "capture": cap,
            "n_actions": len(used),
        }
    restore_capture(env, recovery_cap)
    for a in used[:stable_i]:
        step_legal(env.sim, a)
    cap = take_capture(env)
    return {
        "ok": True,
        "actions": used[:stable_i],
        "capture": cap,
        "n_actions": stable_i,
        "reason": "stable_after_stage1_c",
    }


def primitives_needed(need: dict) -> list[str]:
    out = []
    if need.get("dtheta_needed"):
        out.append("dtheta")
    if need.get("dx_needed"):
        out.append("dx")
    if need.get("dz_needed"):
        out.append("dz")
    return out


def process_one(env, meta: dict, table: np.ndarray, cfg: PrecisionConfig, dirs: dict) -> dict:
    sample_id = meta["sample_id"]
    out_path = dirs["outcomes"] / f"{sample_id}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))

    t0 = time.time()
    # --- Recovery entry ---
    if meta["kind"] in ("UE_AB", "HARD_STABLE"):
        if meta["kind"] == "UE_AB":
            recovery_cap, ref = restore_recovery_ab(env, meta)
        else:
            blob = pickle.loads(meta["snapshot_path"].read_bytes())
            recovery_cap, ref = blob["capture"], blob["ref"]
            restore_capture(env, recovery_cap)
    else:
        recovery_cap, ref = restore_recovery_c(env, meta)

    restore_capture(env, recovery_cap)
    err_rec = measure_errors(env.sim)
    gate_rec = entry_gate(env)
    restore_capture(env, recovery_cap)
    direct_rec = probe_direct_handoff(env, recovery_cap, ref, stabilize_n=8)
    restore_capture(env, recovery_cap)

    ic_path = dirs["ics"] / f"{sample_id}__recovery.pkl"
    pickle.dump(
        {
            "capture": recovery_cap,
            "ref": ref,
            "sample_id": sample_id,
            "lineage_id": meta["lineage_id"],
            "source_label": meta["source_label"],
            "errors_recovery": err_rec,
            "gate_recovery": {k: gate_rec[k] for k in gate_rec if k != "stability"},
            "provenance": meta.get("provenance"),
        },
        ic_path.open("wb"),
    )

    stage1_executed = False
    stage1_actions: list = []
    stable_cap = recovery_cap
    stage1_info: dict = {"ok": True, "n_actions": 0, "reason": "skipped"}

    if meta["source_label"] in ("STABLE_PRECISION", "STABLE_DIRECT"):
        # Already at Recovery; skip Stage-1 expert
        restore_capture(env, recovery_cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, recovery_cap)
        gate = entry_gate(env)
        restore_capture(env, recovery_cap)
        if not (_truthy(stab.get("stable")) and _truthy(gate.get("ok"))):
            row = {
                "sample_id": sample_id,
                "source_label": meta["source_label"],
                "lineage_id": meta["lineage_id"],
                "category": "REJECT_NOT_STABLE",
                "partition": "diagnostic",
                "stage1_executed": False,
                "reason": "recovery_not_stable_for_stable_label",
            }
            atomic_json(out_path, row)
            return row
        stable_cap = recovery_cap
        stage1_info = {"ok": True, "n_actions": 0, "reason": "skipped_already_stable"}
        if meta["source_label"] == "STABLE_DIRECT":
            # Screen already found direct SUCCESS_NEW; do not re-enter Precision.
            direct_stable = probe_direct_handoff(env, stable_cap, ref, stabilize_n=8)
            restore_capture(env, stable_cap)
            full_actions: list = []
            handoff = direct_stable if _truthy(direct_stable.get("SUCCESS_NEW")) else {
                "SUCCESS_NEW": True, "SUCCESS_12S": True, "reason": "screen_stable_direct"
            }
            row = {
                "sample_id": sample_id,
                "dataset_version": DATASET_VERSION,
                "oracle_tag": ORACLE_TAG,
                "source_label": "STABLE_DIRECT",
                "family": meta["family"],
                "lineage_id": meta["lineage_id"],
                "expert_id": meta.get("expert_id"),
                "ic_id": meta["ic_id"],
                "category": "STABLE_DIRECT",
                "partition": "expert",
                "stage1_executed": False,
                "precision_executed": False,
                "precision_skipped": True,
                "primitives_required": [],
                "combined_correction": False,
                "errors_recovery": jsafe(err_rec),
                "errors_stable": jsafe(measure_errors(env.sim)),
                "handoff": jsafe({
                    "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
                    "SUCCESS_12S": handoff.get("SUCCESS_12S"),
                    "reason": handoff.get("std_reason") or handoff.get("reason"),
                }),
                "stage1": stage1_info,
                "segments": {
                    "recovery_start": 0, "stage1_end": 0, "precision_end": 0,
                    "stage1_executed": False, "precision_executed": False, "precision_skipped": True,
                },
                "n_actions_full": 0,
                "n_actions_stage1": 0,
                "n_actions_precision": 0,
                "sim_s_full": 0.0,
                "ic_pkl": str(ic_path.relative_to(OUT)),
                "obs_version": OBS_VERSION,
                "dt": DT_POLICY,
                "continuous_demo": None,
                "k49_demo": None,
                "wall_s": round(time.time() - t0, 2),
            }
            # Empty-tape demo: recovery already in SUCCESS basin; record obs0 only
            cont_path = dirs["continuous"] / f"{sample_id}__fullchain_continuous.npz"
            restore_capture(env, recovery_cap)
            obs0 = observe(env)
            np.savez_compressed(
                cont_path,
                actions=np.zeros((0, 7), np.float32),
                observations=np.asarray([obs0], np.float32),
                timestamps=np.zeros((0,), np.float32),
                stage1_end_idx=np.asarray([0], np.int64),
                precision_end_idx=np.asarray([0], np.int64),
            )
            # Verify via direct handoff settle only
            rep = probe_direct_handoff(env, recovery_cap, ref, stabilize_n=8)
            row["continuous_demo"] = {
                "path": str(cont_path.relative_to(OUT)),
                "n_actions": 0,
                "replay_SUCCESS_NEW": bool(rep.get("SUCCESS_NEW")),
                "replay_SUCCESS_12S": bool(rep.get("SUCCESS_12S")),
                "verified": bool(rep.get("SUCCESS_NEW")),
            }
            if not row["continuous_demo"]["verified"]:
                row["partition"] = "diagnostic"
                row["category"] = "STABLE_DIRECT_REGRESS"
            atomic_json(out_path, row)
            return row
    else:
        stage1_executed = True
        if meta["kind"] == "UE_AB":
            stage1_info = run_stage1_ab(env, recovery_cap, ref, meta["stage1_npz"])
        else:
            stage1_info = run_stage1_c(env, recovery_cap, ref, meta["expert_pkl"])
        stage1_actions = list(stage1_info.get("actions") or [])
        if not stage1_info.get("ok"):
            row = {
                "sample_id": sample_id,
                "source_label": meta["source_label"],
                "lineage_id": meta["lineage_id"],
                "category": "STAGE1_FAILURE",
                "partition": "diagnostic",
                "stage1_executed": True,
                "stage1": {k: stage1_info[k] for k in stage1_info if k not in ("actions", "capture")},
                "errors_recovery": jsafe(err_rec),
            }
            atomic_json(dirs["failures"] / f"{sample_id}__stage1.json", row)
            atomic_json(out_path, row)
            return row
        stable_cap = stage1_info["capture"]

    restore_capture(env, stable_cap)
    err_stable = measure_errors(env.sim)
    need = assess_need(err_stable, cfg)
    prims = primitives_needed(need)
    direct_stable = probe_direct_handoff(env, stable_cap, ref, stabilize_n=8)
    restore_capture(env, stable_cap)

    precision_actions: list = []
    prec_log = {}
    if _truthy(direct_stable.get("SUCCESS_NEW")):
        # Direct HANDOFF from stable — optional STABLE_DIRECT / A/B/C direct
        category = "STABLE_DIRECT" if meta["source_label"] in ("STABLE_PRECISION", "STABLE_DIRECT") else "DIRECT_HANDOFF"
        source_label = meta["source_label"] if meta["source_label"] != "STABLE_PRECISION" else "STABLE_DIRECT"
        res_status = "PRECISION_SKIPPED_DIRECT_SUCCESS"
        handoff = direct_stable
        precision_executed = False
        precision_skipped = True
    else:
        t_p = time.time()
        res = run_precision(env, {"capture": stable_cap, "ref": ref}, meta["family"], cfg)
        precision_actions = list(res.actions or [])
        precision_executed = bool(res.precision_executed)
        precision_skipped = bool(res.precision_skipped)
        handoff = res.handoff or {}
        res_status = res.status
        prec_log = {
            "status": res.status,
            "phases": list(res.phases or []),
            "failures": list(res.failures or []),
            "n_actions": len(precision_actions),
            "wall_s": round(time.time() - t_p, 2),
            "task_outcomes": (res.primitive_logs or {}).get("task_outcomes"),
            "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
            "dz_s1_reused": (res.primitive_logs or {}).get("dz_s1_reused"),
        }
        if _truthy(handoff.get("SUCCESS_NEW")):
            category = "PRECISION_SUCCESS"
            source_label = meta["source_label"]
        else:
            category = "PRECISION_FAILURE"
            source_label = meta["source_label"]

    full_actions = list(stage1_actions) + list(precision_actions)
    n_s1 = len(stage1_actions)
    n_prec = len(precision_actions)

    # Segment index markers
    segments = {
        "recovery_start": 0,
        "stage1_end": n_s1,
        "precision_end": n_s1 + n_prec,
        "stage1_executed": stage1_executed,
        "precision_executed": precision_executed,
        "precision_skipped": precision_skipped,
    }

    row = {
        "sample_id": sample_id,
        "dataset_version": DATASET_VERSION,
        "oracle_tag": ORACLE_TAG,
        "source_label": source_label,
        "family": meta["family"],
        "lineage_id": meta["lineage_id"],
        "expert_id": meta.get("expert_id"),
        "ic_id": meta["ic_id"],
        "category": category,
        "partition": "expert" if category in (
            "DIRECT_HANDOFF", "PRECISION_SUCCESS", "STABLE_DIRECT", "STABLE_PRECISION"
        ) and bool(handoff.get("SUCCESS_NEW")) else "diagnostic",
        "stage1_executed": stage1_executed,
        "precision_executed": precision_executed,
        "precision_skipped": precision_skipped,
        "primitives_required": prims,
        "combined_correction": len(prims) >= 2,
        "errors_recovery": jsafe(err_rec),
        "errors_stable": jsafe(err_stable),
        "need_at_stable": jsafe({k: need[k] for k in need if k != "axis_decision"}),
        "direct_recovery": jsafe({
            "SUCCESS_NEW": direct_rec.get("SUCCESS_NEW"),
            "SUCCESS_12S": direct_rec.get("SUCCESS_12S"),
        }),
        "direct_stable": jsafe({
            "SUCCESS_NEW": direct_stable.get("SUCCESS_NEW"),
            "SUCCESS_12S": direct_stable.get("SUCCESS_12S"),
        }),
        "handoff": jsafe({
            "SUCCESS_NEW": handoff.get("SUCCESS_NEW"),
            "SUCCESS_12S": handoff.get("SUCCESS_12S"),
            "reason": handoff.get("std_reason") or handoff.get("reason"),
        }),
        "precision": prec_log,
        "stage1": {k: stage1_info[k] for k in stage1_info if k not in ("actions", "capture")},
        "segments": segments,
        "n_actions_full": len(full_actions),
        "n_actions_stage1": n_s1,
        "n_actions_precision": n_prec,
        "sim_s_full": len(full_actions) * DT_POLICY,
        "ic_pkl": str(ic_path.relative_to(OUT)),
        "obs_version": OBS_VERSION,
        "dt": DT_POLICY,
        "action_dim": 7,
        "obs_dim": 39,
        "wall_s": round(time.time() - t0, 2),
        "continuous_demo": None,
        "k49_demo": None,
        "repr_gap": False,
    }

    # Fix STABLE_PRECISION label when precision was needed and succeeded
    if meta["source_label"] == "STABLE_PRECISION" and category == "PRECISION_SUCCESS":
        row["source_label"] = "STABLE_PRECISION"
        row["partition"] = "expert"
    if category == "PRECISION_SUCCESS" and meta["source_label"] == "STABLE_PRECISION":
        row["partition"] = "expert"
    if category == "PRECISION_FAILURE" or category == "STAGE1_FAILURE":
        row["partition"] = "diagnostic"
        atomic_json(dirs["failures"] / f"{sample_id}__fail.json", row)

    success_for_expert = bool(handoff.get("SUCCESS_NEW")) and category in (
        "DIRECT_HANDOFF", "PRECISION_SUCCESS", "STABLE_DIRECT", "STABLE_PRECISION"
    )
    # STABLE_PRECISION category name when precision ran successfully from stable recovery
    if meta["source_label"] == "STABLE_PRECISION" and success_for_expert and precision_executed:
        row["category"] = "STABLE_PRECISION"
        row["partition"] = "expert"
        success_for_expert = True
    elif meta["source_label"] in ("STABLE_PRECISION", "STABLE_DIRECT") and success_for_expert and not precision_executed:
        row["category"] = "STABLE_DIRECT"
        row["source_label"] = "STABLE_DIRECT"
        row["partition"] = "expert"

    if success_for_expert or (
        bool(handoff.get("SUCCESS_NEW")) and category in ("DIRECT_HANDOFF", "PRECISION_SUCCESS", "STABLE_DIRECT", "STABLE_PRECISION")
    ):
        cont_rep = replay_actions_record_obs(env, recovery_cap, full_actions, ref)
        cont_path = dirs["continuous"] / f"{sample_id}__fullchain_continuous.npz"
        np.savez_compressed(
            cont_path,
            actions=np.asarray(full_actions, np.float32),
            observations=cont_rep["observations"],
            timestamps=np.arange(len(full_actions), dtype=np.float32) * DT_POLICY,
            stage1_end_idx=np.asarray([n_s1], np.int64),
            precision_end_idx=np.asarray([n_s1 + n_prec], np.int64),
            source_label=np.asarray([row["source_label"]]),
        )
        row["continuous_demo"] = {
            "path": str(cont_path.relative_to(OUT)),
            "n_actions": len(full_actions),
            "replay_SUCCESS_NEW": cont_rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": cont_rep["SUCCESS_12S"],
            "verified": bool(cont_rep["SUCCESS_NEW"]),
        }
        q = quantize_actions(full_actions, table)
        tok_path = dirs["tokens"] / f"{sample_id}__fullchain_k49.npz"
        np.savez_compressed(
            tok_path,
            tokens=q["tokens"],
            proto_actions=q["proto_actions"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(full_actions, np.float32),
            stage1_end_idx=np.asarray([n_s1], np.int64),
            cap=np.asarray([CAP], np.float32),
        )
        tok_rep = replay_tokens(env, recovery_cap, q["tokens"], table, ref, residuals=q["residuals"])
        verified_k49 = bool(tok_rep["SUCCESS_NEW"])
        row["k49_demo"] = {
            "path": str(tok_path.relative_to(OUT)),
            "n_tokens": int(len(q["tokens"])),
            "max_l2": q["max_l2"],
            "mean_l2": q["mean_l2"],
            "max_recon_l2": q["max_recon_l2"],
            "residual_clipped_frac": q["residual_clipped_frac"],
            "replay_SUCCESS_NEW": tok_rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": tok_rep["SUCCESS_12S"],
            "verified": verified_k49,
            "config_id": K49_CONFIG,
        }
        row["repr_gap"] = bool(cont_rep["SUCCESS_NEW"]) and not verified_k49
        if not cont_rep["SUCCESS_NEW"]:
            row["partition"] = "diagnostic"
            row["category"] = "CONTINUOUS_REPLAY_FAIL"
        elif verified_k49:
            row["partition"] = "expert"
        else:
            # continuous OK, K49 fail — keep continuous expert, flag K49
            row["partition"] = "expert_continuous_only"

    # Persist stage1/precision action tapes separately
    if stage1_actions:
        np.savez_compressed(
            dirs["stage1"] / f"{sample_id}__stage1.npz",
            actions=np.asarray(stage1_actions, np.float32),
        )
    if precision_actions:
        np.savez_compressed(
            dirs["precision"] / f"{sample_id}__precision.npz",
            actions=np.asarray(precision_actions, np.float32),
        )

    atomic_json(out_path, row)
    return row


def discover_sources() -> list[dict]:
    sources = []
    seen_lineage = set()
    for row in load_ue():
        fam = row["plan_family"]
        if fam in ("A", "B"):
            m = resolve_recovery_ab(row)
        else:
            m = resolve_recovery_c(row)
        if m is None:
            continue
        if m["lineage_id"] in seen_lineage:
            continue
        seen_lineage.add(m["lineage_id"])
        m["sample_id"] = f"{fam}__{m['ic_id']}__{capture_fp_from_meta(m)}"
        m["provenance"] = {
            "ue_expert_id": row["expert_id"],
            "split_ue": row.get("split"),
            "source_experiment": row.get("source_experiment"),
        }
        sources.append(m)

    # STABLE_PRECISION / STABLE_DIRECT from hard ICs (independent Recovery entries)
    if HARD_IC_DIR.exists():
        for p in sorted(HARD_IC_DIR.glob("*.pkl")):
            lid = f"hard_{p.stem}"
            if lid in seen_lineage or p.stem in seen_lineage:
                continue
            sources.append({
                "family": "B",
                "source_label": "STABLE_PRECISION",  # reclassified after screen
                "ic_id": p.stem,
                "expert_id": f"hard_ic__{p.stem}",
                "lineage_id": lid,
                "snapshot_path": p,
                "kind": "HARD_STABLE",
                "sample_id": f"STABLE__{p.stem}",
                "provenance": {"source": "hard_ic_mining", "path": str(p.relative_to(ROOT))},
            })
            seen_lineage.add(lid)
    return sources


def capture_fp_from_meta(m: dict) -> str:
    try:
        if m["kind"] in ("UE_AB", "HARD_STABLE"):
            blob = pickle.loads(m["snapshot_path"].read_bytes())
            return capture_fp(blob["capture"])
        pack = pickle.loads(m["expert_pkl"].read_bytes())
        # catch_snap may be ballistic pack; hash qpos
        q = np.asarray(pack["catch_snap"].get("qpos") or pack["catch_snap"]["physics"]["qpos"], float)
        return hashlib.sha256(np.round(q / DEDUP).astype(np.int64).tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(m["lineage_id"].encode()).hexdigest()[:16]


def screen_hard_stable(env, meta: dict, cfg: PrecisionConfig) -> str | None:
    """Return STABLE_PRECISION, STABLE_DIRECT, or None (skip)."""
    blob = pickle.loads(meta["snapshot_path"].read_bytes())
    if "capture" not in blob or "ref" not in blob:
        return None
    restore_capture(env, blob["capture"])
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, blob["capture"])
    gate = entry_gate(env)
    restore_capture(env, blob["capture"])
    if not (_truthy(stab.get("stable")) and _truthy(gate.get("ok"))):
        return None
    direct = probe_direct_handoff(env, blob["capture"], blob["ref"], stabilize_n=8)
    if _truthy(direct.get("SUCCESS_NEW")):
        return "STABLE_DIRECT"
    need = assess_need(measure_errors(env.sim), cfg)
    if need.get("dtheta_needed") or need.get("dx_needed") or need.get("dz_needed"):
        return "STABLE_PRECISION"
    return "STABLE_DIRECT"


def split_by_lineage(rows: list[dict], seed: int = 5478) -> dict:
    """IC-lineage split: same lineage never crosses train/val/test."""
    lineages = sorted({r["lineage_id"] for r in rows})
    rng = np.random.RandomState(seed)
    rng.shuffle(lineages)
    n = len(lineages)
    n_test = max(1, int(round(0.15 * n))) if n >= 7 else max(1, n // 5) if n else 0
    n_val = max(1, int(round(0.15 * n))) if n >= 7 else max(1, n // 5) if n else 0
    test_set = set(lineages[:n_test])
    val_set = set(lineages[n_test:n_test + n_val])
    train_set = set(lineages[n_test + n_val:])
    if not train_set and lineages:
        train_set = {lineages[-1]}
        val_set.discard(lineages[-1])
        test_set.discard(lineages[-1])
    split = {"train": [], "validation": [], "test": []}
    for r in rows:
        lid = r["lineage_id"]
        if lid in test_set:
            split["test"].append(r["sample_id"])
        elif lid in val_set:
            split["validation"].append(r["sample_id"])
        else:
            split["train"].append(r["sample_id"])
    return {
        "seed": seed,
        "n_lineages": n,
        "n_train_lineages": len(train_set),
        "n_val_lineages": len(val_set),
        "n_test_lineages": len(test_set),
        "splits": split,
        "rule": "split by original IC lineage_id; no lineage crosses splits",
    }


def summarize(rows: list[dict]) -> dict:
    by_fam = Counter(r.get("source_label") or r.get("family") for r in rows)
    expert = [r for r in rows if r.get("partition") in ("expert", "expert_continuous_only")]
    diag = [r for r in rows if r.get("partition") == "diagnostic"]
    per = defaultdict(lambda: Counter())
    for r in rows:
        lab = r.get("source_label") or r.get("family")
        per[lab]["total"] += 1
        per[lab][r.get("category", "?")] += 1
        if r.get("stage1_executed"):
            per[lab]["stage1_executed"] += 1
        else:
            per[lab]["stage1_skipped"] += 1
        for p in r.get("primitives_required") or []:
            per[lab][f"need_{p}"] += 1
        if r.get("combined_correction"):
            per[lab]["combined"] += 1
        cd = r.get("continuous_demo") or {}
        kd = r.get("k49_demo") or {}
        if cd.get("verified"):
            per[lab]["SUCCESS_NEW_continuous"] += 1
        if kd.get("verified"):
            per[lab]["SUCCESS_NEW_k49"] += 1
        if (r.get("handoff") or {}).get("SUCCESS_NEW"):
            per[lab]["SUCCESS_NEW"] += 1
    lengths = [r.get("n_actions_full") or 0 for r in expert]
    return {
        "n_total": len(rows),
        "n_expert": len(expert),
        "n_diagnostic": len(diag),
        "by_source_label": dict(by_fam),
        "per_family": {k: dict(v) for k, v in per.items()},
        "trajectory_length": {
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "p10": float(np.percentile(lengths, 10)) if lengths else 0.0,
            "p90": float(np.percentile(lengths, 90)) if lengths else 0.0,
            "max": int(max(lengths) if lengths else 0),
        },
        "failure_reasons": dict(Counter(
            (r.get("category") or "unk") for r in diag
        )),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = {
        "ics": OUT / "ics",
        "stage1": OUT / "stage1",
        "precision": OUT / "precision",
        "continuous": OUT / "continuous",
        "tokens": OUT / "tokens",
        "outcomes": OUT / "outcomes",
        "failures": OUT / "failures",
        "diagnostics": OUT / "diagnostics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Copy freeze manifest for provenance
    if FREEZE_MANIFEST.exists():
        shutil.copy2(FREEZE_MANIFEST, OUT / "ORACLE_MANIFEST.json")

    cfg = PrecisionConfig(evaluate_handoff=True)
    table = build_k49_table(config_id=K49_CONFIG)
    env = make_env()

    print("Discovering Recovery-entry sources...", flush=True)
    sources = discover_sources()
    print(f"  discovered {len(sources)} lineages", flush=True)

    # Screen HARD_STABLE labels
    filtered = []
    hard_skip = 0
    for m in sources:
        if m["kind"] != "HARD_STABLE":
            filtered.append(m)
            continue
        label = screen_hard_stable(env, m, cfg)
        if label is None:
            hard_skip += 1
            continue
        m["source_label"] = label
        # refresh sample_id with label
        m["sample_id"] = f"{label}__{m['ic_id']}"
        filtered.append(m)
    print(f"  after hard-IC screen: {len(filtered)} (skipped unstable hard={hard_skip})", flush=True)
    atomic_json(OUT / "SOURCE_INVENTORY.json", {
        "n": len(filtered),
        "by_label": dict(Counter(m["source_label"] for m in filtered)),
        "hard_skipped_unstable": hard_skip,
    })

    rows = []
    for i, m in enumerate(filtered):
        print(f"[{i+1}/{len(filtered)}] {m['sample_id']} ...", flush=True)
        try:
            row = process_one(env, m, table, cfg, dirs)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(tb, flush=True)
            row = {
                "sample_id": m["sample_id"],
                "source_label": m["source_label"],
                "lineage_id": m["lineage_id"],
                "category": "EXCEPTION",
                "partition": "diagnostic",
                "error": str(e),
                "traceback": tb[-2000:],
            }
            atomic_json(dirs["failures"] / f"{m['sample_id']}__exception.json", row)
            atomic_json(dirs["outcomes"] / f"{m['sample_id']}.json", row)
        rows.append(row)
        # incremental manifest
        if (i + 1) % 5 == 0:
            atomic_json(OUT / "SUMMARY_PARTIAL.json", summarize(rows))

    # Prefer expert partitions for split
    expert_rows = [r for r in rows if r.get("partition") in ("expert", "expert_continuous_only")]
    splits = split_by_lineage(expert_rows)
    summary = summarize(rows)
    summary["splits"] = {
        "n_train": len(splits["splits"]["train"]),
        "n_validation": len(splits["splits"]["validation"]),
        "n_test": len(splits["splits"]["test"]),
    }

    # Write manifest.jsonl
    man_path = OUT / "manifest.jsonl"
    if man_path.exists():
        man_path.unlink()
    with man_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(jsafe(r), ensure_ascii=False) + "\n")

    atomic_json(OUT / "splits.json", splits)
    atomic_json(OUT / "SUMMARY.json", summary)
    schema = {
        "dataset_version": DATASET_VERSION,
        "oracle_tag": ORACLE_TAG,
        "obs_dim": 39,
        "obs_version": OBS_VERSION,
        "action_dim": 7,
        "dt": DT_POLICY,
        "k49_config": K49_CONFIG,
        "source_labels": ["A", "B", "C", "STABLE_PRECISION", "STABLE_DIRECT"],
        "partitions": ["expert", "expert_continuous_only", "diagnostic"],
        "pipeline": (
            "Recovery entry IC → optional Stage-1 A/B/C expert → stable grasp → "
            "Precision Oracle if needed → HANDOFF → 12s SUCCESS_NEW/12S"
        ),
        "git_HEAD": git_rev(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fields": [
            "sample_id", "source_label", "lineage_id", "stage1_executed",
            "primitives_required", "continuous_demo", "k49_demo",
            "handoff.SUCCESS_NEW", "handoff.SUCCESS_12S", "partition",
        ],
    }
    atomic_json(OUT / "DATASET_SCHEMA.json", schema)

    # Replay validation sample per family
    replay_val = []
    for lab in ("A", "B", "C", "STABLE_PRECISION", "STABLE_DIRECT"):
        cands = [
            r for r in expert_rows
            if r.get("source_label") == lab and (r.get("continuous_demo") or {}).get("verified")
        ]
        if not cands:
            continue
        r = cands[0]
        ic = pickle.loads((OUT / r["ic_pkl"]).read_bytes())
        acts = np.load(OUT / r["continuous_demo"]["path"])["actions"]
        rep = replay_actions_record_obs(env, ic["capture"], list(acts), ic["ref"])
        replay_val.append({
            "sample_id": r["sample_id"],
            "source_label": lab,
            "replay_SUCCESS_NEW": rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": rep["SUCCESS_12S"],
            "matches_stored": rep["SUCCESS_NEW"] == r["continuous_demo"]["replay_SUCCESS_NEW"],
        })
    atomic_json(OUT / "REPLAY_VALIDATION.json", {"samples": replay_val})

    report = f"""# Dataset REPORT — {DATASET_VERSION}

Created: {datetime.now(timezone.utc).isoformat()}
Oracle: `{ORACLE_TAG}` (see `ORACLE_MANIFEST.json`)
Git HEAD: `{git_rev()}`

## Pipeline

```
Recovery entry IC
  → Stage-1 A/B/C expert (skipped for STABLE_*)
  → Stable grasp
  → Precision Oracle if needed (frozen speed-optimized)
  → HANDOFF
  → 12s SUCCESS_NEW / SUCCESS_12S
```

## Counts

- Total lineages processed: **{summary['n_total']}**
- Expert partition: **{summary['n_expert']}**
- Diagnostic partition: **{summary['n_diagnostic']}**

### Per source label

```
{json.dumps(summary['per_family'], indent=2)}
```

### Splits (by IC lineage)

- train: {summary['splits']['n_train']}
- validation: {summary['splits']['n_validation']}
- test: {summary['splits']['n_test']}

### Trajectory lengths (expert)

{json.dumps(summary['trajectory_length'], indent=2)}

## Notes

- Demos begin at genuine Recovery-entry captures (not post-Stage-1 relabeling).
- K49 success is independently replay-verified; continuous success ≠ automatic K49 success.
- Failed/aborted runs live under `failures/` and `partition=diagnostic`.
- Prior datasets under `stage2_abc_*` were not overwritten.
- No SFT/RL launched.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
