#!/usr/bin/env python
"""Stage-2 A/B/C nontrivial sample mining (v2.3 Precision Oracle frozen).

Writes results/data_generation/stage2_abc_mining/ with a NEW versioned manifest.
Does not overwrite stage2_abc_pilot/ic_manifest_frozen.jsonl.
Does not start SFT/RL training. Does not modify the Oracle.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import load_vocab, run_tokens  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist, audit_grasp  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    PrecisionConfig,
    entry_gate,
    measure_errors,
    probe_direct_handoff,
    run_precision,
)
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402

PILOT = ROOT / "results/data_generation/stage2_abc_pilot"
OUT = ROOT / "results/data_generation/stage2_abc_mining"
VERSION = "v1"
K49_CONFIG = "K49_tok48_wy0p25"
DEDUP_ATOL = 1e-3

STABLE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
HARD_DIR = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
B1_PER_IC = ROOT / "results/diagnostics/plan_b_b0_b1_b2_ground/per_ic"
SCREEN_JSON = PILOT / "plan_b_candidate_screen.json"
VR = ROOT / "data/vertical_recatch/train"
SNAP_TRAIN = ROOT / "data/snapshots/train"
B_POST_LAT = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"

# Minimum genuine Plan-B NONTRIVIAL demos before declaring Phase-2 shortfall addressed
TARGET_PLAN_B_NONTRIVIAL = 8


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


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def capture_fp(cap) -> str:
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / DEDUP_ATOL).astype(np.int64).tobytes()).hexdigest()[:16]


def observe(env) -> np.ndarray:
    return observe_airborne(env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist)


def dynamics_params(sim) -> dict:
    mid = sim.ids.object_body
    try:
        mass = float(sim.model.body_mass[mid])
    except Exception:
        mass = None
    frictions = []
    try:
        for gi in range(sim.model.ngeom):
            if int(sim.model.geom_bodyid[gi]) == int(mid):
                frictions.append(np.asarray(sim.model.geom_friction[gi], float).tolist())
    except Exception:
        pass
    return {"object_mass": mass, "object_geom_friction": frictions[:4]}


def state_fp(sim) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(sim.data.qpos, np.float64).tobytes())
    h.update(np.asarray(sim.data.qvel, np.float64).tobytes())
    return h.hexdigest()[:16]


def nearest_motion_token(table, action):
    a = np.asarray(action, np.float32).reshape(7)
    best_i, best_d = 0, float("inf")
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < best_d:
            best_i, best_d = i, d
    return int(best_i), float(best_d), table[best_i].copy()


def quantize_actions(actions, table):
    acts = np.asarray(actions, np.float32)
    if acts.size == 0:
        return {
            "tokens": np.zeros((0,), np.int64),
            "residuals": np.zeros((0, 7), np.float32),
            "recon_actions": np.zeros((0, 7), np.float32),
            "max_recon_l2": 0.0,
            "residual_clipped_frac": 0.0,
            "max_l2": 0.0,
            "mean_l2": 0.0,
        }
    tokens, residuals, recons, dists, rl2s = [], [], [], [], []
    clipped = 0
    for a in acts:
        ti, d, ap = nearest_motion_token(table, a)
        raw = a - ap
        res = np.clip(raw, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw - res))) > 1e-8:
            clipped += 1
        recon = np.clip(ap + res, -1.0, 1.0).astype(np.float32)
        tokens.append(ti)
        residuals.append(res)
        recons.append(recon)
        dists.append(d)
        rl2s.append(float(np.linalg.norm(recon - a)))
    rl2 = np.asarray(rl2s, np.float32)
    dist = np.asarray(dists, np.float32)
    return {
        "tokens": np.asarray(tokens, np.int64),
        "residuals": np.asarray(residuals, np.float32),
        "recon_actions": np.asarray(recons, np.float32),
        "max_recon_l2": float(rl2.max()),
        "mean_recon_l2": float(rl2.mean()),
        "residual_clipped_frac": float(clipped / len(acts)),
        "max_l2": float(dist.max()),
        "mean_l2": float(dist.mean()),
    }


def replay_cont(env, cap, actions, ref):
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs = []
    for a in actions:
        obs.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "observations": np.asarray(obs, np.float32) if obs else np.zeros((0, 39), np.float32),
    }


def replay_k49(env, cap, tokens, residuals, table, ref):
    restore_capture(env, cap)
    res = np.asarray(residuals, np.float32)
    for i, t in enumerate(tokens):
        a = table[int(t)].copy()
        if i < len(res):
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


def dirs() -> dict:
    base = OUT / VERSION
    d = {
        "root": base,
        "ics": base / "ics",
        "outcomes": base / "outcomes",
        "continuous": base / "continuous",
        "tokens": base / "tokens",
        "failures": base / "failures",
        "diagnostics": base / "diagnostics",
        "sft": base / "sft",
        "rl": base / "rl",
        "manifests": base / "manifests",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------


def load_phase1_candidates() -> list[dict]:
    """Deduped PRECISION_CANDIDATE list with provenance (not auto-labeled as Plan B)."""
    screen = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    rows = [r for r in screen["rows"] if r.get("bin") == "PRECISION_CANDIDATE"]
    out = []
    seen_ids = set()

    # Prefer explicit Plan-B diagnostic stable
    if STABLE_B.exists():
        out.append(
            {
                "ic_id": "hard_910220",
                "source_kind": "plan_b_diagnostic_stable",
                "plan_provenance": "B",
                "oracle_route": "B",
                "path": str(STABLE_B.relative_to(ROOT)),
                "genuine_stage1_plan_b": True,
            }
        )
        seen_ids.add("hard_910220")

    for r in rows:
        sid = r["ic_id"]
        sk = r.get("source_kind")
        if sk == "diagnostic_stable":
            continue  # already added via STABLE_B
        if sk == "hard_ic_mining":
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            hp = HARD_DIR / f"{sid}.pkl"
            if not hp.exists():
                continue
            out.append(
                {
                    "ic_id": sid,
                    "source_kind": "hard_ic_mining",
                    "plan_provenance": "HARD_IC",
                    "oracle_route": "B",  # teacher routing only; not Plan-B Stage-1 claim
                    "path": str(hp.relative_to(ROOT)),
                    "genuine_stage1_plan_b": False,
                    "scenario_family": "ballistic_impact",
                }
            )
        elif sk == "follow_early_extra" and sid == "ball_train_v2_0268":
            # handled as unassigned below
            pass

    # Unassigned Stage-1 tape
    rid = "ball_train_v2_0268"
    npz = VR / f"{rid}__follow_early.npz"
    snap = SNAP_TRAIN / f"{rid}.pkl"
    if npz.exists() and snap.exists() and rid not in seen_ids:
        out.append(
            {
                "ic_id": rid,
                "source_kind": "stage1_follow_early_unassigned",
                "plan_provenance": "UNASSIGNED_STAGE1",
                "oracle_route": "A",  # geometry closer to Plan-A tapes; not claimed as B
                "snapshot": str(snap.relative_to(ROOT)),
                "continuous_npz": str(npz.relative_to(ROOT)),
                "genuine_stage1_plan_b": False,
            }
        )
    return out


def extract_from_follow_early(env, snap_path: Path, cont_path: Path) -> dict | None:
    root = pickle.loads(snap_path.read_bytes())
    acts = np.asarray(np.load(cont_path)["actions"], np.float32)
    restore_capture(env, root["capture"])
    best_i = None
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if (i + 1) % 5 != 0 and i + 1 < len(acts):
            continue
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        if stab["stable"]:
            best_i = i + 1
            break
    if best_i is None:
        return None
    restore_capture(env, root["capture"])
    for a in acts[:best_i]:
        step_legal(env.sim, np.asarray(a, np.float32))
    cap = take_capture(env)
    restore_capture(env, cap)
    gate = entry_gate(env)
    restore_capture(env, cap)
    if not gate.get("ok"):
        return None
    return {
        "capture": cap,
        "ref": root["ref"],
        "stabilization_prefix_actions": acts[:best_i].tolist(),
        "n_prefix": int(best_i),
    }


def load_candidate_capture(env, cand: dict) -> dict | None:
    if cand["source_kind"] == "stage1_follow_early_unassigned":
        return extract_from_follow_early(env, ROOT / cand["snapshot"], ROOT / cand["continuous_npz"])
    path = ROOT / cand["path"]
    pack = pickle.loads(path.read_bytes())
    if "capture" not in pack or "ref" not in pack:
        return None
    return {
        "capture": pack["capture"],
        "ref": pack["ref"],
        "stabilization_prefix_actions": list(
            pack.get("stabilization_prefix_actions")
            or pack.get("stabilization_prefix_tokens")
            or pack.get("actions_prefix")
            or []
        ),
        "n_prefix": int(
            len(
                pack.get("stabilization_prefix_actions")
                or pack.get("stabilization_prefix_tokens")
                or pack.get("actions_prefix")
                or []
            )
        ),
        "pack_meta": {k: pack[k] for k in pack if k not in ("capture", "ref")},
    }


# ---------------------------------------------------------------------------
# Evaluate one IC
# ---------------------------------------------------------------------------


def sft_category(direct_new: bool, prec_new: bool | None, n_actions: int, *, intermediate: bool = False) -> str:
    if intermediate:
        return "INTERMEDIATE_CURRICULUM"
    if direct_new and n_actions == 0:
        return "DIRECT_HANDOFF"
    if direct_new:
        return "DIRECT_HANDOFF"  # direct ok even if oracle also ran
    if prec_new:
        return "NONTRIVIAL_PRECISION"
    return "FAILED_ORACLE"


def evaluate_candidate(env, cand: dict, table, cfg, d: dict, *, intermediate: bool = False) -> dict | None:
    sample_id = f"{cand['plan_provenance']}__{cand['ic_id']}__{VERSION}"
    # refine after capture_fp known
    outcome_guess = d["outcomes"] / f"{sample_id}.json"
    # We'll rewrite sample_id with fp below

    blob = load_candidate_capture(env, cand)
    if blob is None:
        return {
            "ic_id": cand["ic_id"],
            "status": "LOAD_FAIL",
            "plan_provenance": cand["plan_provenance"],
            "source_kind": cand["source_kind"],
        }

    cap0, ref = blob["capture"], blob["ref"]
    fp = capture_fp(cap0)
    sample_id = f"{cand['plan_provenance']}__{cand['ic_id']}__{fp}"
    outcome_path = d["outcomes"] / f"{sample_id}.json"
    if outcome_path.exists():
        return json.loads(outcome_path.read_text(encoding="utf-8"))

    restore_capture(env, cap0)
    fp0 = state_fp(env.sim)
    gate = entry_gate(env)
    restore_capture(env, cap0)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap0)
    err = measure_errors(env.sim)
    geom = audit_grasp(env.sim)
    dyn = dynamics_params(env.sim)
    obs0 = observe(env)
    restore_capture(env, cap0)
    fp1 = state_fp(env.sim)

    if not (gate.get("ok") and stab.get("stable")):
        row = {
            "sample_id": sample_id,
            "ic_id": cand["ic_id"],
            "plan_provenance": cand["plan_provenance"],
            "oracle_route": cand["oracle_route"],
            "source_kind": cand["source_kind"],
            "genuine_stage1_plan_b": cand.get("genuine_stage1_plan_b", False),
            "sft_category": "FAILED_ORACLE",
            "status": "GATE_FAIL",
            "entry_gate_ok": bool(gate.get("ok")),
            "stable": bool(stab.get("stable")),
            "errors_entry": err,
            "source": cand,
        }
        atomic_write_json(outcome_path, row)
        return row

    # Direct HANDOFF
    direct = probe_direct_handoff(env, cap0, ref, stabilize_n=8)
    restore_capture(env, cap0)
    fp2 = state_fp(env.sim)

    # Precision
    t0 = time.time()
    res = run_precision(env, {"capture": cap0, "ref": ref}, cand["oracle_route"], cfg)
    wall = time.time() - t0
    restore_capture(env, cap0)
    fp3 = state_fp(env.sim)
    prec_new = bool((res.handoff or {}).get("SUCCESS_NEW")) if res.handoff else False
    actions = list(res.actions or [])
    cat = sft_category(bool(direct.get("SUCCESS_NEW")), prec_new, len(actions), intermediate=intermediate)

    # Persist IC
    ic_pkl = d["ics"] / f"{sample_id}.pkl"
    ic_pkl.write_bytes(
        pickle.dumps(
            {
                "capture": cap0,
                "ref": ref,
                "stabilization_prefix_actions": blob.get("stabilization_prefix_actions") or [],
            }
        )
    )

    row = {
        "sample_id": sample_id,
        "ic_id": cand["ic_id"],
        "plan_provenance": cand["plan_provenance"],
        "oracle_route": cand["oracle_route"],
        "source_kind": cand["source_kind"],
        "genuine_stage1_plan_b": bool(cand.get("genuine_stage1_plan_b", False)),
        "scenario_family": cand.get("scenario_family"),
        "sft_category": cat,
        "status": "OK",
        "capture_fp": fp,
        "ic_pkl": str(ic_pkl.relative_to(d["root"])),
        "entry_gate_ok": True,
        "stable": True,
        "errors_entry": err,
        "geom": {k: geom[k] for k in ("grasp_kind", "e_pos_mm", "e_theta_deg", "e_signed_mm", "bilateral") if k in geom},
        "dynamics": dyn,
        "obs0_39d": obs0.tolist(),
        "fingerprints": {
            "pre": fp0,
            "after_measure": fp1,
            "after_direct": fp2,
            "after_precision": fp3,
            "restore_ok": fp0 == fp1 == fp2 == fp3,
        },
        "direct": {
            "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
            "reason": direct.get("reason"),
        },
        "precision": {
            "status": res.status,
            "SUCCESS_NEW": prec_new,
            "SUCCESS_12S": bool((res.handoff or {}).get("SUCCESS_12S")),
            "n_actions": len(actions),
            "phases": list(res.phases or []),
            "failures": list(res.failures or []),
            "precision_executed": res.precision_executed,
            "precision_skipped": res.precision_skipped,
            "wall_s": round(wall, 2),
        },
        "continuous_demo": None,
        "k49_demo": None,
        "repr_gap": False,
        "source": {k: v for k, v in cand.items() if k != "pack_meta"},
        "intermediate_curriculum": bool(intermediate),
    }

    if cat in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION") and (
        bool(direct.get("SUCCESS_NEW")) or prec_new
    ):
        cont = replay_cont(env, cap0, actions, ref)
        cont_path = d["continuous"] / f"{sample_id}__continuous.npz"
        np.savez_compressed(
            cont_path,
            actions=np.asarray(actions, np.float32),
            observations=cont["observations"],
            timestamps=np.arange(len(actions), dtype=np.float32) * DT_POLICY,
        )
        row["continuous_demo"] = {
            "path": str(cont_path.relative_to(d["root"])),
            "n_actions": len(actions),
            "replay_SUCCESS_NEW": cont["SUCCESS_NEW"],
            "verified": bool(cont["SUCCESS_NEW"]),
            "reason": cont["reason"],
        }
        q = quantize_actions(actions, table)
        tok_path = d["tokens"] / f"{sample_id}__k49.npz"
        np.savez_compressed(
            tok_path,
            tokens=q["tokens"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(actions, np.float32),
            cap=np.asarray([CAP], np.float32),
        )
        # Empty tape: K49 nontrivial representability is N/A
        if len(actions) == 0:
            row["k49_demo"] = {
                "path": str(tok_path.relative_to(d["root"])),
                "n_tokens": 0,
                "verified": bool(cont["SUCCESS_NEW"]),
                "nontrivial_representability": False,
                "note": "empty_tape_direct_handoff_not_k49_precision_evidence",
                "config_id": K49_CONFIG,
            }
            row["repr_gap"] = False
        else:
            tok_rep = replay_k49(env, cap0, q["tokens"], q["residuals"], table, ref)
            verified_k49 = bool(tok_rep["SUCCESS_NEW"])
            repr_gap = bool(cont["SUCCESS_NEW"]) and (
                (not verified_k49) or q["residual_clipped_frac"] > 0.25 or q["max_recon_l2"] > 0.15
            )
            row["k49_demo"] = {
                "path": str(tok_path.relative_to(d["root"])),
                "n_tokens": int(len(q["tokens"])),
                "max_recon_l2": q["max_recon_l2"],
                "residual_clipped_frac": q["residual_clipped_frac"],
                "encoding": "nearest_motion_token_plus_residual_cap_0p25",
                "replay_SUCCESS_NEW": tok_rep["SUCCESS_NEW"],
                "verified": verified_k49,
                "nontrivial_representability": verified_k49,
                "config_id": K49_CONFIG,
            }
            row["repr_gap"] = repr_gap

    if cat == "FAILED_ORACLE":
        fail_path = d["failures"] / f"{sample_id}__failure.json"
        atomic_write_json(
            fail_path,
            {
                "sample_id": sample_id,
                "plan_provenance": cand["plan_provenance"],
                "status": res.status,
                "failures": res.failures,
                "phases": res.phases,
                "errors_entry": err,
                "note": "Do not treat as physically impossible without investigation.",
            },
        )
        row["failure_diag"] = str(fail_path.relative_to(d["root"]))

    atomic_write_json(
        d["diagnostics"] / f"{sample_id}__teacher_diag.json",
        {
            "sample_id": sample_id,
            "privileged_errors_entry": err,
            "privileged_geom": geom,
            "phases": res.phases,
            "failures": res.failures,
            "note": "Privileged teacher diagnostics — not policy inputs.",
        },
    )
    atomic_write_json(outcome_path, row)
    print(
        f"  [{cand['plan_provenance']}] {cand['ic_id']} cat={cat} "
        f"direct={direct.get('SUCCESS_NEW')} prec={prec_new} n_act={len(actions)} "
        f"e_pos={err['e_pos_mm']:.1f}",
        flush=True,
    )
    return row


# ---------------------------------------------------------------------------
# Phase 2 — genuine Plan B from B1 prefixes
# ---------------------------------------------------------------------------


def extract_post_b1_stable(env, hard_id: str, tokens: list[int], table, book48) -> dict | None:
    snap = HARD_DIR / f"{hard_id}.pkl"
    if not snap.exists():
        return None
    root = pickle.loads(snap.read_bytes())
    restore_capture(env, root["capture"])
    ok, reason = run_tokens(env, root, table, book48, tokens)
    if not ok:
        return {"error": reason, "ic_id": hard_id}
    # Capture before hold; confirm stable
    cap = take_capture(env)
    restore_capture(env, cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap)
    gate = entry_gate(env)
    restore_capture(env, cap)
    if not (stab.get("stable") and gate.get("ok")):
        return {
            "error": "unstable_or_gate",
            "ic_id": hard_id,
            "stable": stab.get("stable"),
            "gate": gate.get("ok"),
        }
    err = measure_errors(env.sim)
    return {
        "ic_id": hard_id,
        "capture": cap,
        "ref": root["ref"],
        "stabilization_prefix_tokens": list(tokens),
        "errors_entry": err,
        "source_kind": "plan_b_stage1_post_b1",
        "plan_provenance": "B",
        "oracle_route": "B",
        "genuine_stage1_plan_b": True,
        "path": str(snap.relative_to(ROOT)),
        "b1_tokens": list(tokens),
    }


# Verified Plan-B B1 token patterns from plan_b_b0_b1_b2_ground (Recovery-only; no HANDOFF).
PLAN_B_B1_TOKEN_PATTERNS = (
    [32, 12, 12, 0, 0, 0, 0, 0],
    [30, 0, 0, 0, 0, 0],
    [30, 12, 12, 0, 0, 0, 0, 0],
    [44, 0, 0, 0, 0, 0],
    [36],
    [36, 36],
    [39],
)


def phase2_plan_b_candidates(
    env,
    table,
    book48,
    *,
    extra_hard_ids: list[str] | None = None,
    max_extra: int = 12,
) -> list[dict]:
    """Build genuine Plan-B Stage-1 ICs via B1 Recovery tokens → stable pre-HANDOFF capture."""
    cands = []
    seen = set()

    def try_extract(hard_id: str, toks: list[int], tag: str) -> None:
        if hard_id in seen:
            return
        print(f"  Phase2 extract post-B1 {hard_id} tokens={toks} ({tag})", flush=True)
        got = extract_post_b1_stable(env, hard_id, toks, table, book48)
        if got is None or got.get("error"):
            print(f"    skip: {got}", flush=True)
            return
        # Keep only if direct HANDOFF fails SUCCESS_NEW (nontrivial Stage-2 need)
        restore_capture(env, got["capture"])
        direct = probe_direct_handoff(env, got["capture"], got["ref"], stabilize_n=8)
        restore_capture(env, got["capture"])
        if direct.get("SUCCESS_NEW"):
            print(f"    skip: post-B1 already DIRECT_SUCCESS e_pos={got['errors_entry']['e_pos_mm']:.1f}", flush=True)
            return
        seen.add(hard_id)
        got["phase2_tag"] = tag
        cands.append(got)
        print(
            f"    keep: PRECISION_CANDIDATE e_pos={got['errors_entry']['e_pos_mm']:.1f} "
            f"th={got['errors_entry']['dtheta_deg']:.1f}",
            flush=True,
        )

    # 1) Known B1 rolls with SUCCESS_12S and not SUCCESS_NEW
    if B1_PER_IC.exists():
        for jf in sorted(B1_PER_IC.glob("hard_*.json")):
            meta = json.loads(jf.read_text(encoding="utf-8"))
            b1 = meta.get("B1") or {}
            if not b1.get("ok_recovery") or not b1.get("SUCCESS_12S") or b1.get("SUCCESS_NEW"):
                continue
            toks = list(b1.get("tokens") or [])
            if toks:
                try_extract(jf.stem, toks, "per_ic_b1_12s_not_new")

    # 2) Expand on Phase-1 FAILED_ORACLE hard ICs using verified B1 patterns
    extras = list(extra_hard_ids or [])
    n_extra = 0
    for hid in extras:
        if n_extra >= max_extra:
            break
        if hid in seen:
            continue
        before = len(cands)
        for toks in PLAN_B_B1_TOKEN_PATTERNS:
            try_extract(hid, list(toks), "pattern_search_on_failed_hard_ic")
            if hid in seen:
                n_extra += 1
                break
        if len(cands) == before:
            print(f"  Phase2 no stable nontrivial post-B1 for {hid}", flush=True)
    return cands


def materialize_phase2_cand(got: dict, d: dict) -> dict:
    """Convert extract_post_b1_stable result into evaluate_candidate cand + save capture pack."""
    sid_tmp = f"B_postb1_{got['ic_id']}"
    pack_path = d["ics"] / f"_src_{sid_tmp}.pkl"
    pack_path.write_bytes(
        pickle.dumps(
            {
                "capture": got["capture"],
                "ref": got["ref"],
                "stabilization_prefix_tokens": got["stabilization_prefix_tokens"],
                "b1_tokens": got["b1_tokens"],
                "provenance": {
                    "hard_snapshot": got["path"],
                    "extraction": "plan_b_B1_tokens_then_earliest_stable_pre_handoff",
                },
            }
        )
    )
    return {
        "ic_id": f"{got['ic_id']}__post_b1",
        "source_kind": "plan_b_stage1_post_b1",
        "plan_provenance": "B",
        "oracle_route": "B",
        "path": str(pack_path.relative_to(ROOT)),
        "genuine_stage1_plan_b": True,
    }


# ---------------------------------------------------------------------------
# Import pilot A/B/C
# ---------------------------------------------------------------------------


def import_pilot_outcomes(d: dict) -> list[dict]:
    """Copy pilot outcomes into mining v1 with remapped categories; preserve pilot freeze."""
    imported = []
    man = PILOT / "ic_manifest_frozen.jsonl"
    if not man.exists():
        return imported
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        plan = meta["plan"]
        sid_old = meta["sample_id"]
        op = PILOT / "outcomes" / f"{sid_old}.json"
        if not op.exists():
            continue
        old = json.loads(op.read_text(encoding="utf-8"))
        # Map pilot category → SFT category
        if old["category"] == "DIRECT_SUCCESS":
            sft_cat = "DIRECT_HANDOFF"
        elif old["category"] == "PRECISION_NEEDED":
            sft_cat = "NONTRIVIAL_PRECISION"
        else:
            sft_cat = "FAILED_ORACLE"
        sample_id = f"PILOT_{plan}__{meta['ic_id']}__{meta.get('capture_fp') or sid_old.split('__')[-1]}"
        # Copy IC pickle
        src_ic = PILOT / meta["ic_pkl"]
        dst_ic = d["ics"] / f"{sample_id}.pkl"
        if src_ic.exists() and not dst_ic.exists():
            shutil.copy2(src_ic, dst_ic)
        # Copy demos if present
        cont = old.get("continuous_demo") or {}
        k49 = old.get("k49_demo") or {}
        cont_new = None
        k49_new = None
        if cont.get("path"):
            sp = PILOT / cont["path"]
            dp = d["continuous"] / f"{sample_id}__continuous.npz"
            if sp.exists():
                shutil.copy2(sp, dp)
                cont_new = {**cont, "path": str(dp.relative_to(d["root"]))}
        if k49.get("path"):
            sp = PILOT / k49["path"]
            dp = d["tokens"] / f"{sample_id}__k49.npz"
            if sp.exists():
                shutil.copy2(sp, dp)
                k49_new = {
                    **k49,
                    "path": str(dp.relative_to(d["root"])),
                    "nontrivial_representability": bool(k49.get("verified")) and int(k49.get("n_tokens") or 0) > 0,
                }
        row = {
            "sample_id": sample_id,
            "ic_id": meta["ic_id"],
            "plan_provenance": plan,
            "oracle_route": plan,
            "source_kind": "pilot_import",
            "genuine_stage1_plan_b": plan == "B",
            "sft_category": sft_cat,
            "status": "OK",
            "pilot_category": old["category"],
            "capture_fp": meta.get("capture_fp"),
            "ic_pkl": str(dst_ic.relative_to(d["root"])) if dst_ic.exists() else None,
            "errors_entry": meta.get("errors_entry") or old.get("errors_entry"),
            "dynamics": meta.get("dynamics"),
            "direct": old.get("direct"),
            "precision": old.get("precision"),
            "continuous_demo": cont_new,
            "k49_demo": k49_new,
            "repr_gap": old.get("repr_gap"),
            "source": {
                "pilot_sample_id": sid_old,
                "pilot_manifest": "results/data_generation/stage2_abc_pilot/ic_manifest_frozen.jsonl",
                "note": "Imported from pilot; freeze preserved",
            },
            "intermediate_curriculum": False,
        }
        atomic_write_json(d["outcomes"] / f"{sample_id}.json", row)
        imported.append(row)
    print(f"Imported {len(imported)} pilot samples", flush=True)
    return imported


# ---------------------------------------------------------------------------
# Intermediate curriculum (B_post_lateral)
# ---------------------------------------------------------------------------


def evaluate_intermediate(env, table, cfg, d: dict) -> dict | None:
    if not B_POST_LAT.exists():
        return None
    cand = {
        "ic_id": "B_post_lateral",
        "source_kind": "intermediate_post_lateral",
        "plan_provenance": "B",
        "oracle_route": "B",
        "path": str(B_POST_LAT.relative_to(ROOT)),
        "genuine_stage1_plan_b": False,  # post-Precision intermediate
    }
    return evaluate_candidate(env, cand, table, cfg, d, intermediate=True)


# ---------------------------------------------------------------------------
# SFT / RL export + report
# ---------------------------------------------------------------------------


def build_exports(d: dict, outcomes: list[dict]) -> dict:
    # SFT indexes
    sft_rows = []
    for r in outcomes:
        cat = r.get("sft_category")
        if cat not in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION", "INTERMEDIATE_CURRICULUM"):
            continue
        cont = r.get("continuous_demo") or {}
        k49 = r.get("k49_demo") or {}
        if cat == "FAILED_ORACLE":
            continue
        # Require continuous verified for SFT teachable rows (except we keep direct even if empty)
        if cat != "DIRECT_HANDOFF" and not cont.get("verified"):
            continue
        sft_rows.append(
            {
                "sample_id": r["sample_id"],
                "sft_category": cat,
                "plan_provenance": r["plan_provenance"],
                "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                "ic_pkl": r.get("ic_pkl"),
                "continuous_path": cont.get("path"),
                "continuous_verified": cont.get("verified"),
                "continuous_n_actions": cont.get("n_actions"),
                "k49_path": k49.get("path"),
                "k49_verified": k49.get("verified"),
                "k49_nontrivial_representability": k49.get("nontrivial_representability"),
                "obs_note": "observations in continuous npz are frozen 39D policy inputs only",
            }
        )
    sft_man = d["sft"] / "sft_manifest.jsonl"
    if sft_man.exists():
        sft_man.unlink()
    for row in sft_rows:
        append_jsonl(sft_man, row)

    # RL ICs — all physically complete captures with gate OK
    rl_rows = []
    for r in outcomes:
        if r.get("status") != "OK" or not r.get("ic_pkl"):
            continue
        if not r.get("entry_gate_ok", True):
            continue
        rl_rows.append(
            {
                "sample_id": r["sample_id"],
                "plan_provenance": r["plan_provenance"],
                "source_kind": r.get("source_kind"),
                "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                "ic_pkl": r.get("ic_pkl"),
                "direct_SUCCESS_NEW": (r.get("direct") or {}).get("SUCCESS_NEW"),
                "oracle_SUCCESS_NEW": (r.get("precision") or {}).get("SUCCESS_NEW"),
                "sft_category": r.get("sft_category"),
                "difficulty": (
                    "easy_direct"
                    if (r.get("direct") or {}).get("SUCCESS_NEW")
                    else (
                        "nontrivial_oracle_ok"
                        if (r.get("precision") or {}).get("SUCCESS_NEW")
                        else "hard_oracle_fail"
                    )
                ),
                "e_pos_mm": (r.get("errors_entry") or {}).get("e_pos_mm"),
                "dynamics": r.get("dynamics"),
                "source_episode": r.get("ic_id"),
                "intermediate_curriculum": r.get("intermediate_curriculum", False),
            }
        )

    # Split by source episode hash (before derived samples)
    episodes = sorted({r["source_episode"] for r in rl_rows})
    rng = np.random.default_rng(5478)
    order = list(episodes)
    rng.shuffle(order)
    n = len(order)
    n_hold = max(1, int(0.15 * n)) if n >= 7 else max(0, n // 5)
    n_val = max(1, int(0.15 * n)) if n >= 7 else max(0, n // 5)
    hold = set(order[:n_hold])
    val = set(order[n_hold : n_hold + n_val])
    train = set(order[n_hold + n_val :])
    splits = {"train": [], "val": [], "holdout": []}
    for r in rl_rows:
        ep = r["source_episode"]
        if ep in hold:
            split = "holdout"
        elif ep in val:
            split = "val"
        else:
            split = "train"
        r["split"] = split
        splits[split].append(r["sample_id"])

    rl_man = d["rl"] / "rl_ic_manifest.jsonl"
    if rl_man.exists():
        rl_man.unlink()
    for row in rl_rows:
        append_jsonl(rl_man, row)
    atomic_write_json(d["rl"] / "splits.json", splits)

    return {"n_sft": len(sft_rows), "n_rl": len(rl_rows), "splits": {k: len(v) for k, v in splits.items()}}


def summarize(outcomes: list[dict]) -> dict:
    def filt(**kw):
        rows = outcomes
        for k, v in kw.items():
            rows = [r for r in rows if r.get(k) == v]
        return rows

    plans = sorted({r.get("plan_provenance") for r in outcomes if r.get("plan_provenance")})
    per = {}
    for p in plans:
        rows = filt(plan_provenance=p)
        per[p] = {
            "n": len(rows),
            "genuine_stage1_plan_b": sum(1 for r in rows if r.get("genuine_stage1_plan_b")),
            "DIRECT_HANDOFF": sum(1 for r in rows if r.get("sft_category") == "DIRECT_HANDOFF"),
            "NONTRIVIAL_PRECISION": sum(1 for r in rows if r.get("sft_category") == "NONTRIVIAL_PRECISION"),
            "FAILED_ORACLE": sum(1 for r in rows if r.get("sft_category") == "FAILED_ORACLE"),
            "INTERMEDIATE_CURRICULUM": sum(1 for r in rows if r.get("sft_category") == "INTERMEDIATE_CURRICULUM"),
            "oracle_SUCCESS_NEW": sum(1 for r in rows if (r.get("precision") or {}).get("SUCCESS_NEW")),
            "continuous_verified": sum(1 for r in rows if (r.get("continuous_demo") or {}).get("verified")),
            "k49_verified": sum(1 for r in rows if (r.get("k49_demo") or {}).get("verified")),
            "k49_nontrivial_ok": sum(
                1 for r in rows if (r.get("k49_demo") or {}).get("nontrivial_representability")
            ),
            "distinct_episodes": len({r.get("ic_id") for r in rows}),
            "distinct_mass": len(
                {
                    (r.get("dynamics") or {}).get("object_mass")
                    for r in rows
                    if (r.get("dynamics") or {}).get("object_mass") is not None
                }
            ),
        }
    return {
        "n_total": len(outcomes),
        "sft_categories": dict(Counter(r.get("sft_category") for r in outcomes)),
        "source_kinds": dict(Counter(r.get("source_kind") for r in outcomes)),
        "per_plan_provenance": per,
        "genuine_plan_b_nontrivial": sum(
            1
            for r in outcomes
            if r.get("genuine_stage1_plan_b") and r.get("sft_category") == "NONTRIVIAL_PRECISION"
        ),
        "intermediate_curriculum": sum(1 for r in outcomes if r.get("sft_category") == "INTERMEDIATE_CURRICULUM"),
    }


def write_report(d: dict, summary: dict, export_stats: dict, phase1_n: int, phase2_n: int) -> None:
    per = summary["per_plan_provenance"]
    lines = [
        f"# Stage-2 A/B/C Nontrivial Mining — {VERSION}",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        f"Git: `{git_rev()}`",
        f"K49: `{K49_CONFIG}` + residual ±{CAP}",
        "Precision Oracle: **v2.3 frozen** (not modified).",
        "Pilot freeze: **preserved** at `results/data_generation/stage2_abc_pilot/ic_manifest_frozen.jsonl`.",
        "",
        "## Pool composition",
        "",
        f"- Phase-1 screened candidates: {phase1_n}",
        f"- Phase-2 genuine Plan-B post-B1 extracts evaluated: {phase2_n}",
        f"- Pilot imports: included under `source_kind=pilot_import`",
        f"- Intermediate curriculum: {summary['intermediate_curriculum']}",
        "",
        "## Per plan_provenance",
        "",
        "| Prov | n | Genuine B Stage-1 | DIRECT_HANDOFF | NONTRIVIAL | FAILED | Cont OK | K49 OK | K49 nontrivial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p, s in per.items():
        lines.append(
            f"| {p} | {s['n']} | {s['genuine_stage1_plan_b']} | {s['DIRECT_HANDOFF']} | "
            f"{s['NONTRIVIAL_PRECISION']} | {s['FAILED_ORACLE']} | {s['continuous_verified']} | "
            f"{s['k49_verified']} | {s['k49_nontrivial_ok']} |"
        )
    lines += [
        "",
        f"**Genuine Plan-B NONTRIVIAL_PRECISION demos:** {summary['genuine_plan_b_nontrivial']}",
        "",
        "## SFT / RL exports",
        "",
        f"- SFT manifest rows: {export_stats['n_sft']} → `{VERSION}/sft/sft_manifest.jsonl`",
        f"- RL IC rows: {export_stats['n_rl']} → `{VERSION}/rl/rl_ic_manifest.jsonl`",
        f"- Splits (by source episode): {export_stats['splits']}",
        "",
        "## Provenance rules",
        "",
        "- `HARD_IC` ≠ Plan B. Teacher may use `oracle_route=B` without claiming Stage-1 Plan B.",
        "- UE-B pilot DIRECT_HANDOFF rows kept as HANDOFF examples.",
        "- `B_post_lateral` marked `INTERMEDIATE_CURRICULUM` only.",
        "- Empty action tapes are not counted as nontrivial K49 representability.",
        "",
        "## Layout",
        "",
        "```",
        f"{VERSION}/ics/ outcomes/ continuous/ tokens/ failures/ diagnostics/",
        f"{VERSION}/sft/sft_manifest.jsonl",
        f"{VERSION}/rl/rl_ic_manifest.jsonl splits.json",
        f"{VERSION}/manifests/ic_manifest_{VERSION}.jsonl",
        f"{VERSION}/summary.json REPORT.md",
        "```",
        "",
        "Stop: no bulk SFT/RL training launched.",
    ]
    (d["root"] / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    d = dirs()
    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)  # frozen K49 for SFT labels / residual replay
    table_b, book48 = load_vocab()  # Plan-B Stage-1 token decode table
    cfg = PrecisionConfig()

    # --- Import pilot (A/B/C backbone) ---
    print("=== Import pilot freeze (read-only source) ===", flush=True)
    outcomes = import_pilot_outcomes(d)

    # --- Phase 1 ---
    print("=== Phase 1: screen PRECISION_CANDIDATE pool ===", flush=True)
    phase1 = load_phase1_candidates()
    print(f"Candidates: {len(phase1)}", flush=True)
    phase1_outcomes = []
    for cand in phase1:
        row = evaluate_candidate(env, cand, table, cfg, d)
        if row:
            phase1_outcomes.append(row)
            outcomes.append(row)

    # --- Intermediate ---
    print("=== Intermediate curriculum: B_post_lateral ===", flush=True)
    inter = evaluate_intermediate(env, table, cfg, d)
    if inter:
        outcomes.append(inter)

    # --- Phase 2 if Plan-B nontrivial shortfall ---
    genuine_b_nt = sum(
        1
        for r in outcomes
        if r.get("genuine_stage1_plan_b") and r.get("sft_category") == "NONTRIVIAL_PRECISION"
    )
    phase2_outcomes = []
    print(f"Genuine Plan-B NONTRIVIAL so far: {genuine_b_nt} (target {TARGET_PLAN_B_NONTRIVIAL})", flush=True)
    if genuine_b_nt < TARGET_PLAN_B_NONTRIVIAL:
        print("=== Phase 2: extract genuine Plan-B post-B1 stables ===", flush=True)
        failed_hard = [
            r["ic_id"]
            for r in phase1_outcomes
            if r.get("plan_provenance") == "HARD_IC" and r.get("sft_category") == "FAILED_ORACLE"
        ]
        got_list = phase2_plan_b_candidates(
            env,
            table_b,
            book48,
            extra_hard_ids=failed_hard,
            max_extra=max(0, TARGET_PLAN_B_NONTRIVIAL - genuine_b_nt + 4),
        )
        for got in got_list:
            cand = materialize_phase2_cand(got, d)
            row = evaluate_candidate(env, cand, table, cfg, d)
            if row:
                phase2_outcomes.append(row)
                outcomes.append(row)
        genuine_b_nt = sum(
            1
            for r in outcomes
            if r.get("genuine_stage1_plan_b") and r.get("sft_category") == "NONTRIVIAL_PRECISION"
        )
        print(f"Genuine Plan-B NONTRIVIAL after Phase 2: {genuine_b_nt}", flush=True)
        if genuine_b_nt < TARGET_PLAN_B_NONTRIVIAL:
            print(
                f"WARNING: Plan-B nontrivial shortfall remains "
                f"({genuine_b_nt}/{TARGET_PLAN_B_NONTRIVIAL}). "
                "HARD_IC demos kept separately; do not relabel as Stage-1 Plan B.",
                flush=True,
            )

    # Dedup outcomes by sample_id
    by_id = {}
    for r in outcomes:
        by_id[r["sample_id"]] = r
    outcomes = list(by_id.values())

    # Versioned IC manifest
    man_path = d["manifests"] / f"ic_manifest_{VERSION}.jsonl"
    if man_path.exists():
        man_path.unlink()
    for r in outcomes:
        append_jsonl(
            man_path,
            {
                "sample_id": r["sample_id"],
                "ic_id": r.get("ic_id"),
                "plan_provenance": r.get("plan_provenance"),
                "source_kind": r.get("source_kind"),
                "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                "sft_category": r.get("sft_category"),
                "ic_pkl": r.get("ic_pkl"),
                "outcome_json": f"outcomes/{r['sample_id']}.json",
            },
        )

    export_stats = build_exports(d, outcomes)
    summary = summarize(outcomes)
    summary["export"] = export_stats
    summary["version"] = VERSION
    summary["git_rev"] = git_rev()
    summary["created_utc"] = datetime.now(timezone.utc).isoformat()
    summary["phase1_n"] = len(phase1)
    summary["phase2_n"] = len(phase2_outcomes)
    summary["target_plan_b_nontrivial"] = TARGET_PLAN_B_NONTRIVIAL
    atomic_write_json(d["root"] / "summary.json", summary)
    write_report(d, summary, export_stats, len(phase1), len(phase2_outcomes))
    # Convenience copy at mining root
    shutil.copy2(d["root"] / "REPORT.md", OUT / "REPORT.md")
    atomic_write_json(OUT / "LATEST.json", {"version": VERSION, "path": str(d["root"].relative_to(ROOT))})
    print(json.dumps(jsafe(summary), indent=2), flush=True)
    print("Done →", d["root"], flush=True)


if __name__ == "__main__":
    main()
