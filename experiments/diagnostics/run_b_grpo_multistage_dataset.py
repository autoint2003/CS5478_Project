#!/usr/bin/env python
"""Validate/expand Tree80 E3 B multi-stage experts + M111 representability.

Supplement handcrafted B; do not replace. No GRPO retrain, no M111 edits,
no SFT/RL packing. Original datasets unchanged.

Outputs:
  results/diagnostics/b_grpo_multistage_validation/
  results/data_generation/stage2_b_grpo_multistage_v1/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.precision_goals import (  # noqa: E402
    assess_need,
    entry_gate,
    measure_errors,
)
from data_generation.oracles.precision.precision_handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import run_precision  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched, terminal_reason  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.k48_checkpoint import HANDOFF as K48_HANDOFF  # noqa: E402
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP as K49_CAP  # noqa: E402
from training.action_vocab_k49 import HANDOFF as K49_HANDOFF  # noqa: E402
from training.action_vocab_k49 import build_k49_table  # noqa: E402
from training.k47_codebook import decode_action, load_codebook  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402
from training.skill_aware_action_vocab import (  # noqa: E402
    DEFAULT_WEIGHTS,
    ResidualBudget,
    load_motion_codebook,
    quantize_trajectory,
)

DS_FULL = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
HARD = ROOT / "results/diagnostics/hard_ic_mining"
PRIOR = ROOT / "results/diagnostics/grpo_vs_handcrafted_b"
VAL = ROOT / "results/diagnostics/b_grpo_multistage_validation"
DS_OUT = ROOT / "results/data_generation/stage2_b_grpo_multistage_v1"
TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
K49_CONFIG = "K49_tok48_wy0p25"
DT = float(DT_POLICY)
LIMIT_S = float(SELECTED_LIMIT_S)
CAP_STEPS = 200
DATASET_VERSION = "stage2_b_grpo_multistage_v1"
ORACLE_TAG = "tree80_e3_plus_precision"


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


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def snap(sim) -> dict:
    err = measure_errors(sim)
    st = sample_state(sim)
    return {
        "bilateral": bool(err["bilateral"]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "dx_mm": float(err["dx_mm"]),
        "dy_mm": float(err["dy_mm"]),
        "dz_mm": float(err["dz_mm"]),
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "w_rel": float(st.get("w_rel_norm") or 0.0),
        "obj_z": float(err["obj_z"]),
        "fn_L": float(err["fn_L"]),
        "fn_R": float(err["fn_R"]),
        "latched": bool(latched(sim)),
    }


def make_plain_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def observe(env) -> np.ndarray:
    return observe_airborne(
        env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist
    )


def lineage_of(ic_id: str, pool: str) -> str:
    return f"B__{pool}__{ic_id}"


def capture_fp(cap: dict) -> str:
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / 1e-6).astype(np.int64).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# IC loading
# ---------------------------------------------------------------------------


def load_handcrafted_b_direct() -> list[dict]:
    rows = [
        json.loads(l)
        for l in (DS_FULL / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out = []
    for r in rows:
        if r.get("source_label") != "B":
            continue
        if not (r.get("continuous_demo") or {}).get("verified"):
            continue
        ic = pickle.loads((DS_FULL / r["ic_pkl"]).read_bytes())
        cont = np.load(DS_FULL / r["continuous_demo"]["path"])
        acts = np.asarray(cont["actions"], np.float32)
        n_s1 = int(r.get("n_actions_stage1") or len(acts))
        out.append({
            "ic_id": r["ic_id"],
            "pool": "easy_handcrafted",
            "source": "handcrafted_fullchain_B",
            "capture": ic["capture"],
            "ref": ic["ref"],
            "handcrafted_stage1": acts[:n_s1],
            "prior_sample_id": r["sample_id"],
            "meta": {
                "azimuth_deg": None,
                "speed_mps": None,
            },
        })
    return out


def load_hard_snapshots(exclude_ids: set[str] | None = None) -> list[dict]:
    exclude_ids = exclude_ids or set()
    # Prefer candidate_manifest for diversity metadata; fall back to glob
    by_id = {}
    cm = HARD / "candidate_manifest.jsonl"
    if cm.exists():
        for line in cm.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            by_id[r["id"]] = r
    out = []
    for p in sorted((HARD / "snapshots").glob("hard_*.pkl")):
        ic_id = p.stem
        if ic_id in exclude_ids:
            continue
        blob = pickle.loads(p.read_bytes())
        if "capture" not in blob or "ref" not in blob:
            continue
        meta = by_id.get(ic_id, {})
        out.append({
            "ic_id": ic_id,
            "pool": "hard_ball_impact",
            "source": "hard_ic_mining_snapshot",
            "capture": blob["capture"],
            "ref": blob["ref"],
            "handcrafted_stage1": None,
            "meta": {
                "azimuth_deg": meta.get("azimuth_deg"),
                "speed_mps": meta.get("speed_mps"),
                "z_off_m": meta.get("z_off_m"),
                "phi_deg": meta.get("phi_deg"),
                "delay_s": meta.get("delay_s"),
                "states": meta.get("states"),
                "nominal_reason": meta.get("nominal_reason"),
                "assigned_split": meta.get("assigned_split"),
            },
        })
    return out


def prior_multistage_ic_ids() -> list[str]:
    """Unique ICs from prior MULTI_STAGE_CANDIDATES (any controller)."""
    path = PRIOR / "MULTI_STAGE_CANDIDATES.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # Prefer Tree80 true-recatch; still verify all unique ICs with Tree80
    return sorted({r["ic_id"] for r in rows})


# ---------------------------------------------------------------------------
# Tree80 Stage-1
# ---------------------------------------------------------------------------


def run_tree80_stage1(env, model, book, device, cap, ref) -> dict:
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._obs_seq = []
    env._prev_seq = []
    env._prev = np.zeros(7, np.float32)
    t0 = float(env.sim.data.time)
    lost = 0
    tokens, actions, obs_list = [], [], []
    both_off = False
    bil_after = False
    how = None
    reason = None
    contact_trace = []
    for step_i in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        obs_list.append(obs.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        tokens.append(token)
        if token == K48_HANDOFF:
            how = "handoff"
            break
        if token < 0 or token >= K48_HANDOFF:
            how = "illegal_token"
            reason = f"token_{token}"
            break
        _res, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        actions.append(executed)
        st = snap(env.sim)
        if step_i % 5 == 0 or step_i < 3:
            contact_trace.append({"i": step_i, "bilateral": st["bilateral"], "e_pos_mm": st["e_pos_mm"]})
        if not st["bilateral"]:
            both_off = True
        elif both_off:
            bil_after = True
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            how = "fail"
            reason = terminal_reason(env.sim) or "DROP"
            break
    exit_cap = take_capture(env)
    exit_st = snap(env.sim)
    restore_capture(env, exit_cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, exit_cap)
    gate = entry_gate(env)
    restore_capture(env, exit_cap)
    genuine_recatch = bool(both_off and bil_after and exit_st["bilateral"])
    ok = how == "handoff" and bool(exit_st["bilateral"]) and not bool(exit_st["latched"])
    return {
        "ok": ok,
        "how": how,
        "fail_reason": reason,
        "n_steps": len(actions),
        "tokens": tokens,
        "actions": np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32),
        "observations": np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32),
        "exit_capture": exit_cap,
        "exit_state": exit_st,
        "hold_stable": bool(stab.get("stable")),
        "gate_ok": bool(gate.get("ok")),
        "saw_both_off": both_off,
        "recatch_bilateral": bil_after,
        "genuine_release_recatch": genuine_recatch,
        "contact_trace": contact_trace,
        "duration_s": len(actions) * DT,
    }


def run_action_tape(env, cap, ref, actions: np.ndarray) -> dict:
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    lost = 0
    both_off = False
    bil_after = False
    obs_list = []
    for a in actions:
        obs_list.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        st = snap(env.sim)
        if not st["bilateral"]:
            both_off = True
        elif both_off:
            bil_after = True
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            break
    exit_cap = take_capture(env)
    exit_st = snap(env.sim)
    restore_capture(env, exit_cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, exit_cap)
    gate = entry_gate(env)
    restore_capture(env, exit_cap)
    return {
        "ok": bool(exit_st["bilateral"]) and not exit_st["latched"],
        "how": "stored_prefix",
        "fail_reason": None,
        "n_steps": len(actions),
        "tokens": [],
        "actions": np.asarray(actions, np.float32),
        "observations": np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32),
        "exit_capture": exit_cap,
        "exit_state": exit_st,
        "hold_stable": bool(stab.get("stable")),
        "gate_ok": bool(gate.get("ok")),
        "saw_both_off": both_off,
        "recatch_bilateral": bil_after,
        "genuine_release_recatch": bool(both_off and bil_after),
        "contact_trace": [],
        "duration_s": len(actions) * DT,
    }


def probe_direct(env, exit_cap, ref) -> dict:
    d = probe_direct_handoff(env, exit_cap, ref, stabilize_n=8)
    return {
        "SUCCESS_NEW": bool(d.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(d.get("SUCCESS_12S")),
        "reason": d.get("reason"),
    }


def run_prec(env, exit_cap, ref) -> dict:
    cfg = PrecisionConfig()
    restore_capture(env, exit_cap)
    entry = snap(env.sim)
    need = assess_need(measure_errors(env.sim), cfg)
    res = run_precision(env, {"capture": exit_cap, "ref": ref}, "B", cfg)
    h = res.handoff or {}
    exit_err = res.errors_exit or {}
    exit_e = float(exit_err.get("e_pos_mm") if exit_err.get("e_pos_mm") is not None else entry["e_pos_mm"])
    exit_th = float(exit_err.get("dtheta_deg") if exit_err.get("dtheta_deg") is not None else entry["dtheta_deg"])
    phases = list(res.phases or [])
    return {
        "executed": bool(res.precision_executed),
        "skipped": bool(res.precision_skipped),
        "status": res.status,
        "actions": np.asarray(res.actions or [], np.float32),
        "n_actions": len(res.actions or []),
        "phases": phases,
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "failures": list(res.failures or [])[:8],
        "entry_e_pos": float(entry["e_pos_mm"]),
        "entry_theta": float(entry["dtheta_deg"]),
        "exit_e_pos": exit_e,
        "exit_theta": exit_th,
        "need_dx": bool(need.get("dx_needed")),
        "need_dz": bool(need.get("dz_needed")),
        "need_dtheta": bool(need.get("dtheta_needed")),
        "physical_correction": bool(
            abs(exit_e - entry["e_pos_mm"]) > 0.5
            or abs(exit_th - entry["dtheta_deg"]) > 0.5
            or any(str(p).startswith("correct_") for p in phases)
        ),
    }


def replay_fullchain(env, start_cap, ref, actions: np.ndarray) -> dict:
    restore_capture(env, start_cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
    }


def quantize_and_replay(env, start_cap, ref, actions, book, rho, w, label: str) -> dict:
    q = quantize_trajectory(book, actions, rho, w=w)
    phys = replay_fullchain(env, start_cap, ref, q["recon_actions"])
    return {
        "vocab": label,
        "SUCCESS_NEW": phys["SUCCESS_NEW"],
        "SUCCESS_12S": phys["SUCCESS_12S"],
        "reason": phys["reason"],
        "clipped_frac": q["clipped_frac"],
        "mean_recon_wlinf": q["mean_recon_wlinf"],
        "max_recon_wlinf": q["max_recon_wlinf"],
        "n_unique_tokens": len(set(int(t) for t in q["tokens"])),
        "tokens": q["tokens"],
        "residuals": q["residuals"],
        "recon_actions": q["recon_actions"],
        "sat_frac_per_dim": q["sat_frac_per_dim"].tolist(),
        "mean_abs_residual": q["mean_abs_residual"].tolist(),
    }


def quantize_k49(env, start_cap, ref, actions, table) -> dict:
    motion_idx = [i for i in range(len(table)) if i != K49_HANDOFF]
    book = table[np.asarray(motion_idx)]
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    if len(acts) == 0:
        return {"vocab": "K49", "SUCCESS_NEW": False, "SUCCESS_12S": False, "clipped_frac": 0.0}
    d = np.linalg.norm(acts[:, None, :] - book[None, :, :], axis=2)
    tok = np.argmin(d, axis=1)
    proto = book[tok]
    raw = acts - proto
    res = np.clip(raw, -K49_CAP, K49_CAP).astype(np.float32)
    recon = np.clip(proto + res, -1.0, 1.0).astype(np.float32)
    phys = replay_fullchain(env, start_cap, ref, recon)
    return {
        "vocab": "K49",
        "SUCCESS_NEW": phys["SUCCESS_NEW"],
        "SUCCESS_12S": phys["SUCCESS_12S"],
        "reason": phys["reason"],
        "clipped_frac": float(np.mean(np.any(np.abs(raw) > K49_CAP + 1e-8, axis=1))),
        "mean_recon_l2": float(np.mean(np.linalg.norm(recon - acts, axis=1))),
        "n_unique_tokens": len(set(int(t) for t in tok)),
        "tokens": tok.astype(np.int64),
        "residuals": res,
        "recon_actions": recon,
    }


# ---------------------------------------------------------------------------
# Process one IC through Tree80 pipeline
# ---------------------------------------------------------------------------


def process_tree80_ic(
    env_pol,
    env_plain,
    model,
    book,
    device,
    ic: dict,
    *,
    force_precision_if_direct_ok: bool = False,
) -> dict:
    t0 = time.time()
    s1 = run_tree80_stage1(env_pol, model, book, device, ic["capture"], ic["ref"])
    row = {
        "ic_id": ic["ic_id"],
        "pool": ic["pool"],
        "source": ic["source"],
        "lineage_id": lineage_of(ic["ic_id"], ic["pool"]),
        "stage1_ok": s1["ok"],
        "stage1_how": s1["how"],
        "n_stage1": s1["n_steps"],
        "genuine_release_recatch": s1["genuine_release_recatch"],
        "saw_both_off": s1["saw_both_off"],
        "recatch_bilateral": s1["recatch_bilateral"],
        "hold_stable": s1["hold_stable"],
        "gate_ok": s1["gate_ok"],
        "exit_bilateral": s1["exit_state"]["bilateral"],
        "exit_e_pos": s1["exit_state"]["e_pos_mm"],
        "exit_theta": s1["exit_state"]["dtheta_deg"],
        "exit_v_rel": s1["exit_state"]["v_rel"],
        "exit_w_rel": s1["exit_state"]["w_rel"],
        "category": None,
        "partition": None,
        "direct_NEW": None,
        "direct_12S": None,
        "prec_NEW": None,
        "prec_12S": None,
        "fullchain_NEW": None,
        "fullchain_12S": None,
        "n_precision": 0,
        "wall_s": None,
    }
    if not s1["ok"] or s1["exit_capture"] is None:
        row["category"] = "STAGE1_FAILURE"
        row["partition"] = "diagnostic"
        row["fail_reason"] = s1.get("fail_reason") or s1.get("how")
        row["wall_s"] = time.time() - t0
        row["_s1"] = s1
        row["_prec"] = None
        row["_full_actions"] = s1["actions"]
        return row

    direct = probe_direct(env_plain, s1["exit_capture"], ic["ref"])
    row["direct_NEW"] = direct["SUCCESS_NEW"]
    row["direct_12S"] = direct["SUCCESS_12S"]
    row["direct_reason"] = direct["reason"]

    prec = None
    full_actions = s1["actions"]
    if direct["SUCCESS_NEW"] and not force_precision_if_direct_ok:
        row["category"] = "TREE80_DIRECT"
        row["partition"] = "expert"
        # fullchain = stage1 only; handoff is control switch after tape
        fc = replay_fullchain(env_plain, ic["capture"], ic["ref"], full_actions)
        row["fullchain_NEW"] = fc["SUCCESS_NEW"]
        row["fullchain_12S"] = fc["SUCCESS_12S"]
        row["fullchain_reason"] = fc["reason"]
        if not fc["SUCCESS_NEW"]:
            row["partition"] = "diagnostic"
            row["category"] = "TREE80_DIRECT_REPLAY_FAIL"
    else:
        # Direct failed (or forced) → Precision
        prec = run_prec(env_plain, s1["exit_capture"], ic["ref"])
        row["prec_NEW"] = prec["SUCCESS_NEW"]
        row["prec_12S"] = prec["SUCCESS_12S"]
        row["prec_reason"] = prec["reason"]
        row["n_precision"] = prec["n_actions"]
        row["precision_status"] = prec["status"]
        row["precision_phases"] = prec["phases"]
        row["physical_correction"] = prec["physical_correction"]
        row["need_dx"] = prec["need_dx"]
        row["need_dz"] = prec["need_dz"]
        row["need_dtheta"] = prec["need_dtheta"]
        full_actions = (
            np.concatenate([s1["actions"], prec["actions"]], axis=0)
            if len(prec["actions"])
            else s1["actions"]
        )
        fc = replay_fullchain(env_plain, ic["capture"], ic["ref"], full_actions)
        row["fullchain_NEW"] = fc["SUCCESS_NEW"]
        row["fullchain_12S"] = fc["SUCCESS_12S"]
        row["fullchain_reason"] = fc["reason"]
        # Require: Direct fail, genuine release–recatch, Precision executed with NEW, fullchain NEW
        if (
            not direct["SUCCESS_NEW"]
            and prec["SUCCESS_NEW"]
            and fc["SUCCESS_NEW"]
            and s1["genuine_release_recatch"]
            and prec["executed"]
            and not prec["skipped"]
            and (prec["physical_correction"] or prec["n_actions"] > 25)
        ):
            row["category"] = "TREE80_PRECISION"
            row["partition"] = "expert"
            row["multi_stage_verified"] = True
        elif not direct["SUCCESS_NEW"] and prec["SUCCESS_NEW"] and fc["SUCCESS_NEW"]:
            row["category"] = "TREE80_PRECISION_NO_RECATCH"
            row["partition"] = "diagnostic"
            row["multi_stage_verified"] = False
        else:
            row["category"] = "PRECISION_FAILURE" if not direct["SUCCESS_NEW"] else "TREE80_DIRECT"
            row["partition"] = "diagnostic"
            row["multi_stage_verified"] = False

    row["wall_s"] = time.time() - t0
    row["_s1"] = s1
    row["_prec"] = prec
    row["_full_actions"] = np.asarray(full_actions, np.float32)
    row["_direct"] = direct
    return row


def save_expert_row(dirs: dict, ic: dict, row: dict, *, expert_kind: str) -> dict:
    """Persist tapes + outcome for a successful or diagnostic trajectory."""
    sid = f"B__{expert_kind}__{ic['ic_id']}__{capture_fp(ic['capture'])}"
    s1 = row["_s1"]
    prec = row.get("_prec")
    full = row["_full_actions"]
    n_s1 = int(s1["n_steps"])
    n_prec = int(prec["n_actions"]) if prec else 0

    ic_path = dirs["ics"] / f"{sid}__recovery.pkl"
    pickle.dump(
        {
            "capture": ic["capture"],
            "ref": ic["ref"],
            "sample_id": sid,
            "ic_id": ic["ic_id"],
            "lineage_id": row["lineage_id"],
            "source": ic["source"],
            "expert_kind": expert_kind,
            "meta": ic.get("meta"),
        },
        ic_path.open("wb"),
    )

    if s1["actions"] is not None and len(s1["actions"]):
        np.savez_compressed(dirs["stage1"] / f"{sid}__stage1.npz", actions=s1["actions"], tokens=np.asarray(s1.get("tokens") or [], np.int64))
    if prec and prec["n_actions"]:
        np.savez_compressed(dirs["precision"] / f"{sid}__precision.npz", actions=prec["actions"])

    # Build observations for fullchain by replay
    env = make_plain_env()
    restore_capture(env, ic["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_list = []
    for a in full:
        obs_list.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    cont_path = dirs["continuous"] / f"{sid}__fullchain_continuous.npz"
    np.savez_compressed(
        cont_path,
        actions=np.asarray(full, np.float32),
        observations=np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32),
        timestamps=np.arange(len(full), dtype=np.float32) * DT,
        stage1_end_idx=np.asarray([n_s1], np.int64),
        precision_end_idx=np.asarray([n_s1 + n_prec], np.int64),
        source_label=np.asarray(["B"]),
        expert_kind=np.asarray([expert_kind]),
    )

    out = {
        "sample_id": sid,
        "dataset_version": DATASET_VERSION,
        "oracle_tag": ORACLE_TAG,
        "source_label": "B",
        "family": "B",
        "expert_kind": expert_kind,
        "ic_id": ic["ic_id"],
        "lineage_id": row["lineage_id"],
        "category": row["category"],
        "partition": row["partition"],
        "stage1_executed": True,
        "precision_executed": bool(prec and prec["executed"]),
        "precision_skipped": bool(prec and prec["skipped"]) if prec else True,
        "genuine_release_recatch": row.get("genuine_release_recatch"),
        "n_actions_full": int(len(full)),
        "n_actions_stage1": n_s1,
        "n_actions_precision": n_prec,
        "sim_s_full": float(len(full) * DT),
        "direct_stable": {
            "SUCCESS_NEW": row.get("direct_NEW"),
            "SUCCESS_12S": row.get("direct_12S"),
            "reason": row.get("direct_reason"),
        },
        "handoff": {
            "SUCCESS_NEW": row.get("fullchain_NEW"),
            "SUCCESS_12S": row.get("fullchain_12S"),
            "reason": row.get("fullchain_reason"),
        },
        "precision": None
        if prec is None
        else {
            "status": prec["status"],
            "phases": prec["phases"],
            "n_actions": prec["n_actions"],
            "physical_correction": prec["physical_correction"],
            "entry_e_pos": prec["entry_e_pos"],
            "exit_e_pos": prec["exit_e_pos"],
            "entry_theta": prec["entry_theta"],
            "exit_theta": prec["exit_theta"],
        },
        "stage1": {
            "ok": row["stage1_ok"],
            "how": row["stage1_how"],
            "n_actions": n_s1,
            "genuine_release_recatch": row.get("genuine_release_recatch"),
            "exit_e_pos": row.get("exit_e_pos"),
            "exit_theta": row.get("exit_theta"),
        },
        "ic_meta": ic.get("meta"),
        "ic_pkl": str(ic_path.relative_to(DS_OUT)).replace("\\", "/"),
        "continuous_demo": {
            "path": str(cont_path.relative_to(DS_OUT)).replace("\\", "/"),
            "n_actions": int(len(full)),
            "replay_SUCCESS_NEW": row.get("fullchain_NEW"),
            "replay_SUCCESS_12S": row.get("fullchain_12S"),
            "verified": bool(row.get("fullchain_NEW") and row.get("partition") == "expert"),
        },
        "dt": DT,
        "action_dim": 7,
        "obs_dim": 39,
        "controller_stage1": "tree80_e3" if expert_kind != "handcrafted_direct" else "handcrafted_prefix",
        "wall_s": row.get("wall_s"),
    }
    atomic_json(dirs["outcomes"] / f"{sid}.json", out)
    if row["partition"] != "expert":
        atomic_json(dirs["failures"] / f"{sid}__fail.json", out)
    return out


def copy_handcrafted_expert(dirs: dict, ic: dict, env_plain) -> dict | None:
    """Re-validate handcrafted B Direct and store in new dataset."""
    s1 = run_action_tape(env_plain, ic["capture"], ic["ref"], ic["handcrafted_stage1"])
    direct = probe_direct(env_plain, s1["exit_capture"], ic["ref"])
    fc = replay_fullchain(env_plain, ic["capture"], ic["ref"], s1["actions"])
    row = {
        "ic_id": ic["ic_id"],
        "pool": ic["pool"],
        "source": ic["source"],
        "lineage_id": lineage_of(ic["ic_id"], "handcrafted"),
        "stage1_ok": s1["ok"],
        "stage1_how": "stored_prefix",
        "n_stage1": s1["n_steps"],
        "genuine_release_recatch": s1["genuine_release_recatch"],
        "saw_both_off": s1["saw_both_off"],
        "recatch_bilateral": s1["recatch_bilateral"],
        "hold_stable": s1["hold_stable"],
        "gate_ok": s1["gate_ok"],
        "exit_bilateral": s1["exit_state"]["bilateral"],
        "exit_e_pos": s1["exit_state"]["e_pos_mm"],
        "exit_theta": s1["exit_state"]["dtheta_deg"],
        "exit_v_rel": s1["exit_state"]["v_rel"],
        "exit_w_rel": s1["exit_state"]["w_rel"],
        "direct_NEW": direct["SUCCESS_NEW"],
        "direct_12S": direct["SUCCESS_12S"],
        "direct_reason": direct["reason"],
        "fullchain_NEW": fc["SUCCESS_NEW"],
        "fullchain_12S": fc["SUCCESS_12S"],
        "fullchain_reason": fc["reason"],
        "category": "HANDCRAFTED_DIRECT" if fc["SUCCESS_NEW"] else "HANDCRAFTED_REGRESS",
        "partition": "expert" if fc["SUCCESS_NEW"] else "diagnostic",
        "n_precision": 0,
        "wall_s": 0.0,
        "_s1": s1,
        "_prec": None,
        "_full_actions": s1["actions"],
    }
    return save_expert_row(dirs, ic, row, expert_kind="handcrafted_direct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-expand", type=int, default=0, help="0 = all hard snapshots")
    ap.add_argument("--skip-m111", action="store_true")
    ap.add_argument("--phase1-only", action="store_true")
    args = ap.parse_args()

    VAL.mkdir(parents=True, exist_ok=True)
    for sub in ("ics", "continuous", "stage1", "precision", "outcomes", "failures", "tokens_m111", "tokens_k49"):
        (DS_OUT / sub).mkdir(parents=True, exist_ok=True)
    dirs = {k: DS_OUT / k for k in ("ics", "continuous", "stage1", "precision", "outcomes", "failures", "tokens_m111", "tokens_k49")}

    device = torch.device(args.device)
    print("Loading Tree80 E3…", flush=True)
    model, book = load_trained_k48(TREE80, device)
    env_pol = make_training_env(device, model=model)
    env_plain = make_plain_env()

    # ---- Phase 1: verify prior multi-stage unique ICs with Tree80 ----
    prior_ids = prior_multistage_ic_ids()
    print(f"Phase1 prior multi-stage unique ICs: {len(prior_ids)} → {prior_ids}", flush=True)
    hard_all = {ic["ic_id"]: ic for ic in load_hard_snapshots()}
    phase1_rows = []
    verified_multi = []
    for ic_id in prior_ids:
        if ic_id not in hard_all:
            phase1_rows.append({"ic_id": ic_id, "status": "missing_snapshot"})
            continue
        ic = hard_all[ic_id]
        print(f"  [P1] verify {ic_id}", flush=True)
        row = process_tree80_ic(env_pol, env_plain, model, book, device, ic)
        phase1_rows.append({k: v for k, v in row.items() if not k.startswith("_")})
        if row.get("category") == "TREE80_PRECISION" and row.get("partition") == "expert":
            verified_multi.append(row)
            print(f"    VERIFIED multi-stage NEW={row['fullchain_NEW']}", flush=True)
        else:
            print(
                f"    result cat={row.get('category')} recatch={row.get('genuine_release_recatch')} "
                f"direct={row.get('direct_NEW')} prec={row.get('prec_NEW')}",
                flush=True,
            )
    write_csv(VAL / "PHASE1_VERIFICATION.csv", phase1_rows)
    unique_verified = sorted({r["ic_id"] for r in verified_multi})
    atomic_json(
        VAL / "PHASE1_SUMMARY.json",
        {
            "prior_unique_ics": prior_ids,
            "independently_verified_multistage_ics": unique_verified,
            "n_verified": len(unique_verified),
            "note": "Counted by unique IC lineage; Tree80 E3 only; requires genuine release-recatch + Precision SUCCESS_NEW",
        },
    )
    print(f"Phase1 independently verified unique multi-stage ICs: {len(unique_verified)}", flush=True)

    if args.phase1_only:
        print("phase1-only done", flush=True)
        return

    # ---- Phase 2–3: expand hard ICs + generate ----
    expand_ics = load_hard_snapshots()
    if args.max_expand > 0:
        expand_ics = expand_ics[: args.max_expand]
    print(f"Phase2/3 expand hard ICs: {len(expand_ics)}", flush=True)

    funnel = Counter()
    gen_rows = []
    manifest = []
    # Also keep handcrafted B Direct
    print("Copying/revalidating handcrafted B Direct…", flush=True)
    for ic in load_handcrafted_b_direct():
        out = copy_handcrafted_expert(dirs, ic, env_plain)
        if out:
            manifest.append(out)
            funnel["handcrafted_direct_expert"] += int(out["partition"] == "expert")
            funnel["handcrafted_direct_fail"] += int(out["partition"] != "expert")

    # Process all hard with Tree80
    for i, ic in enumerate(expand_ics):
        print(f"=== [{i+1}/{len(expand_ics)}] {ic['ic_id']} ===", flush=True)
        row = process_tree80_ic(env_pol, env_plain, model, book, device, ic)
        funnel["tested"] += 1
        funnel["stage1_ok"] += int(bool(row["stage1_ok"]))
        funnel["genuine_recatch"] += int(bool(row.get("genuine_release_recatch")))
        funnel["direct_NEW"] += int(bool(row.get("direct_NEW")))
        if row.get("direct_NEW") is False and row.get("stage1_ok"):
            funnel["direct_fail_given_s1ok"] += 1
            funnel["prec_NEW_given_direct_fail"] += int(bool(row.get("prec_NEW")))
        funnel[str(row.get("category"))] += 1

        gen_rows.append({k: v for k, v in row.items() if not k.startswith("_")})

        if row["partition"] == "expert" and row["category"] in ("TREE80_DIRECT", "TREE80_PRECISION"):
            kind = "tree80_direct" if row["category"] == "TREE80_DIRECT" else "tree80_precision"
            out = save_expert_row(dirs, ic, row, expert_kind=kind)
            manifest.append(out)
            print(f"  SAVED expert {kind} NEW={out['handoff']['SUCCESS_NEW']}", flush=True)
        elif row["partition"] == "diagnostic":
            # still save diagnostic tapes for analysis
            kind = "diagnostic"
            out = save_expert_row(dirs, ic, row, expert_kind=kind)
            manifest.append(out)
            print(f"  diagnostic {row.get('category')}", flush=True)

        if (i + 1) % 10 == 0:
            write_csv(VAL / "GENERATION_LOG.csv", gen_rows)

    write_csv(VAL / "GENERATION_LOG.csv", gen_rows)

    # Dedup expert manifest by lineage: prefer TREE80_PRECISION > TREE80_DIRECT > HANDCRAFTED
    rank = {"TREE80_PRECISION": 3, "TREE80_DIRECT": 2, "HANDCRAFTED_DIRECT": 1}
    by_lin = {}
    for m in manifest:
        if m.get("partition") != "expert":
            continue
        lin = m["lineage_id"]
        # handcrafted and tree80 on same physical easy IC are different lineages (pool tag)
        prev = by_lin.get(lin)
        if prev is None or rank.get(m["category"], 0) > rank.get(prev["category"], 0):
            by_lin[lin] = m
    # Also: for same ic_id across handcrafted vs tree80 on easy — keep both as separate strategies
    expert_manifest = list(by_lin.values())
    # Re-add if we lost handcrafted due to same lineage — use sample_id uniqueness
    # Actually handcrafted lineage is B__handcrafted__id vs B__hard_ball_impact__id — distinct.

    diag_manifest = [m for m in manifest if m.get("partition") != "expert"]

    # ---- Phase 4 analysis ----
    experts = expert_manifest
    multi = [m for m in experts if m["category"] == "TREE80_PRECISION"]
    direct_t = [m for m in experts if m["category"] == "TREE80_DIRECT"]
    hand = [m for m in experts if m["category"] == "HANDCRAFTED_DIRECT"]
    phase4 = {
        "n_tested_hard": int(funnel["tested"]),
        "n_independent_lineages_expert": len({m["lineage_id"] for m in experts}),
        "n_expert_trajectories": len(experts),
        "handcrafted_direct": len(hand),
        "tree80_direct": len(direct_t),
        "tree80_precision_multistage": len(multi),
        "true_recatch_rate_hard": (funnel["genuine_recatch"] / funnel["tested"]) if funnel["tested"] else None,
        "direct_success_rate_hard": (funnel["direct_NEW"] / funnel["tested"]) if funnel["tested"] else None,
        "precision_recovery_rate_given_direct_fail": (
            funnel["prec_NEW_given_direct_fail"] / funnel["direct_fail_given_s1ok"]
            if funnel["direct_fail_given_s1ok"]
            else None
        ),
        "fullchain_expert_counts": {
            "HANDCRAFTED_DIRECT": len(hand),
            "TREE80_DIRECT": len(direct_t),
            "TREE80_PRECISION": len(multi),
        },
        "stage1_duration_s": {
            "multi_mean": float(np.mean([m["n_actions_stage1"] * DT for m in multi])) if multi else None,
            "direct_mean": float(np.mean([m["n_actions_stage1"] * DT for m in direct_t])) if direct_t else None,
            "hand_mean": float(np.mean([m["n_actions_stage1"] * DT for m in hand])) if hand else None,
        },
        "precision_duration_s": {
            "multi_mean": float(np.mean([m["n_actions_precision"] * DT for m in multi])) if multi else None,
        },
        "funnel": dict(funnel),
        "phase1_verified_unique": unique_verified,
        "limitation": None
        if len(multi) >= 5
        else (
            f"Only {len(multi)} verified Tree80→Precision multi-stage experts from physical distribution; "
            "criteria not weakened."
        ),
    }
    # diversity of impact meta among multi
    if multi:
        az = [m.get("ic_meta", {}) or {} for m in multi]
        phase4["multi_impact_diversity"] = {
            "n": len(multi),
            "speed_mps": [a.get("speed_mps") for a in az],
            "azimuth_deg": [a.get("azimuth_deg") for a in az],
            "unique_ic_ids": sorted({m["ic_id"] for m in multi}),
        }
    atomic_json(VAL / "PHASE4_ANALYSIS.json", phase4)

    # ---- Phase 5 M111 ----
    m111_rows = []
    if not args.skip_m111 and M111_META.exists():
        print("Phase5 M111 representability…", flush=True)
        cb = load_motion_codebook(M111_META)
        k49_table = build_k49_table(config_id=K49_CONFIG)
        w = DEFAULT_WEIGHTS
        for m in experts:
            acts = np.asarray(np.load(DS_OUT / m["continuous_demo"]["path"])["actions"], np.float32)
            ic_blob = pickle.loads((DS_OUT / m["ic_pkl"]).read_bytes())
            q_m = quantize_and_replay(
                env_plain, ic_blob["capture"], ic_blob["ref"], acts, cb.prototypes, cb.residual, w, "M111"
            )
            q_k = quantize_k49(env_plain, ic_blob["capture"], ic_blob["ref"], acts, k49_table)
            # save tokens
            np.savez_compressed(
                dirs["tokens_m111"] / f"{m['sample_id']}__m111.npz",
                tokens=q_m["tokens"],
                residuals=q_m["residuals"],
                recon_actions=q_m["recon_actions"],
                handoff_id=np.asarray([cb.handoff_id], np.int64),
            )
            np.savez_compressed(
                dirs["tokens_k49"] / f"{m['sample_id']}__k49.npz",
                tokens=q_k.get("tokens", np.zeros(0, np.int64)),
                residuals=q_k.get("residuals", np.zeros((0, 7), np.float32)),
                recon_actions=q_k.get("recon_actions", np.zeros((0, 7), np.float32)),
            )
            m111_rows.append({
                "sample_id": m["sample_id"],
                "ic_id": m["ic_id"],
                "category": m["category"],
                "expert_kind": m["expert_kind"],
                "continuous_NEW": True,
                "M111_NEW": q_m["SUCCESS_NEW"],
                "M111_12S": q_m["SUCCESS_12S"],
                "M111_clip": q_m["clipped_frac"],
                "M111_mean_wlinf": q_m["mean_recon_wlinf"],
                "M111_n_tok": q_m["n_unique_tokens"],
                "K49_NEW": q_k["SUCCESS_NEW"],
                "K49_12S": q_k["SUCCESS_12S"],
                "K49_clip": q_k["clipped_frac"],
            })
            print(
                f"  {m['sample_id'][:50]} M111={q_m['SUCCESS_NEW']} K49={q_k['SUCCESS_NEW']}",
                flush=True,
            )
        write_csv(VAL / "M111_REPRESENTABILITY.csv", m111_rows)
        by_cat = defaultdict(list)
        for r in m111_rows:
            by_cat[r["category"]].append(r)
        m111_sum = {
            "M": cb.M,
            "rho": cb.residual.to_list(),
            "handoff_id": cb.handoff_id,
            "n": len(m111_rows),
            "M111_NEW": sum(1 for r in m111_rows if r["M111_NEW"]),
            "K49_NEW": sum(1 for r in m111_rows if r["K49_NEW"]),
            "by_category": {
                c: {
                    "n": len(xs),
                    "M111_NEW": sum(1 for r in xs if r["M111_NEW"]),
                    "K49_NEW": sum(1 for r in xs if r["K49_NEW"]),
                }
                for c, xs in by_cat.items()
            },
        }
        atomic_json(VAL / "M111_SUMMARY.json", m111_sum)
    else:
        m111_sum = {"skipped": True}

    # ---- Phase 6 dataset release ----
    # Write manifest.jsonl
    man_path = DS_OUT / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for m in sorted(experts, key=lambda x: x["sample_id"]):
            f.write(json.dumps(jsafe(m), ensure_ascii=False) + "\n")
    # diagnostic index
    with (DS_OUT / "diagnostic_manifest.jsonl").open("w", encoding="utf-8") as f:
        for m in diag_manifest:
            f.write(json.dumps(jsafe(m), ensure_ascii=False) + "\n")

    summary = {
        "dataset_version": DATASET_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "n_expert": len(experts),
        "n_diagnostic": len(diag_manifest),
        "by_category": dict(Counter(m["category"] for m in experts)),
        "by_kind": dict(Counter(m["expert_kind"] for m in experts)),
        "phase1_verified_multistage_ics": unique_verified,
        "tree80_checkpoint": str(TREE80.relative_to(ROOT)).replace("\\", "/"),
        "m111": m111_sum,
        "phase4": phase4,
        "original_datasets_unchanged": [
            str(DS_FULL.relative_to(ROOT)).replace("\\", "/"),
            "results/diagnostics/grpo_vs_handcrafted_b",
        ],
    }
    atomic_json(DS_OUT / "SUMMARY.json", summary)
    atomic_json(VAL / "DATASET_SUMMARY.json", summary)

    # Readiness recommendation
    n_multi = len(multi)
    n_direct = len(direct_t)
    n_hand = len(hand)
    m111_ok = None
    if m111_rows:
        m111_ok = sum(1 for r in m111_rows if r["M111_NEW"]) / len(m111_rows)
    ready = n_hand >= 10 and (n_multi + n_direct) >= 5 and (m111_ok is None or m111_ok >= 0.7)
    rec = {
        "ready_for_policy_data_packing": bool(ready),
        "recommendation": (
            "READY for B-only packing pilot"
            if ready
            else "NOT YET — expand multi-stage coverage or investigate M111 gaps before packing"
        ),
        "counts": {"handcrafted_direct": n_hand, "tree80_direct": n_direct, "tree80_precision": n_multi},
        "m111_retention": m111_ok,
        "notes": [
            "Handcrafted easy B Direct preserved.",
            "Tree80 supplements hard B Direct + multi-stage.",
            "Do not replace handcrafted with Adaptive-G E5.",
            "No SFT/RL started in this task.",
        ],
    }
    atomic_json(VAL / "RECOMMENDATION.json", rec)
    atomic_json(DS_OUT / "RECOMMENDATION.json", rec)

    # Markdown reports
    md = []
    md.append("# B GRPO Multi-Stage Validation & Dataset")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Phase 1 — Independent verification")
    md.append(f"- Prior unique multi-stage ICs: {len(prior_ids)}")
    md.append(f"- Independently verified Tree80→Precision (unique ICs): **{len(unique_verified)}** → {unique_verified}")
    md.append("")
    md.append("## Phase 2–3 — Expansion")
    md.append(f"- Hard ICs tested: {funnel['tested']}")
    md.append(f"- Genuine release–recatch: {funnel['genuine_recatch']}")
    md.append(f"- Direct NEW: {funnel['direct_NEW']}")
    md.append(f"- Experts saved: handcrafted={n_hand}, tree80_direct={n_direct}, tree80_precision={n_multi}")
    md.append("")
    md.append("## Phase 4 — Mechanisms")
    md.append(f"```json\n{json.dumps(jsafe(phase4), indent=2)}\n```")
    md.append("")
    md.append("## Phase 5 — M111")
    md.append(f"```json\n{json.dumps(jsafe(m111_sum), indent=2)}\n```")
    md.append("")
    md.append("## Phase 6 — Dataset")
    md.append(f"- Path: `{DS_OUT.relative_to(ROOT).as_posix()}`")
    md.append(f"- Expert n={len(experts)} diagnostic n={len(diag_manifest)}")
    md.append("")
    md.append("## Recommendation")
    md.append(f"**{rec['recommendation']}** (ready={ready})")
    (VAL / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (DS_OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("DONE", json.dumps(jsafe(rec)), flush=True)


if __name__ == "__main__":
    main()
