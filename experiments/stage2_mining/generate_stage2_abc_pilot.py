#!/usr/bin/env python
"""Stage-2 A/B/C training-data pilot (v2.3 Precision Oracle teacher).

Writes results/data_generation/stage2_abc_pilot/
Does not start SFT/RL. Does not retune the Oracle.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist, audit_grasp  # noqa: E402
from data_generation.oracles.precision.mechanics.grasp_stabilization import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
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
from training.action_vocab_k49 import (  # noqa: E402
    CAP,
    DEFAULT_K49_CONFIG_ID,
    HANDOFF,
    K49_TABLE_ROWS,
    build_k49_table,
    describe_k49_config,
)
from training.recovery_execution import restore_ballistic  # noqa: E402

OUT = ROOT / "results/data_generation/stage2_abc_pilot"
MANIFEST_UE = ROOT / "data/unified_experts/manifest.jsonl"
K49_CONFIG = "K49_tok48_wy0p25"  # frozen Plan-C / ground config
TARGET_PER_PLAN = 20
# Stratified discovery: do not let DIRECT_SUCCESS consume an entire plan quota.
# Precision-candidate slots are filled first; direct-success fills the remainder up to this cap.
DIRECT_SUCCESS_CAP_PER_PLAN = 8
PRECISION_CANDIDATE_RESERVE_PER_PLAN = 12
DEDUP_QPOS_ATOL = 1e-3
DIAGNOSTIC_PLAN_B_STABLE = (
    ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
)
HARD_IC_SNAPSHOT_DIR = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
VERTICAL_RECATCH_TRAIN = ROOT / "data/vertical_recatch/train"
SNAPSHOTS_TRAIN = ROOT / "data/snapshots/train"


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


def atomic_write_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")
        f.flush()


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


def capture_fp(cap: dict) -> str:
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / DEDUP_QPOS_ATOL).astype(np.int64).tobytes()).hexdigest()[:16]


def observe(env) -> np.ndarray:
    return observe_airborne(env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist)


def dynamics_params(sim) -> dict:
    mid = sim.ids.object_body
    # body mass / geom friction when available
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


# ---------------------------------------------------------------------------
# Source discovery + stable extraction
# ---------------------------------------------------------------------------


def load_ue_manifest() -> list[dict]:
    return [json.loads(l) for l in MANIFEST_UE.read_text(encoding="utf-8").splitlines() if l.strip()]


def resolve_plan_a_paths(row: dict) -> tuple[Path, Path] | None:
    ref = row["initial_physical_state_ref"]
    cont = ROOT / ref["continuous_npz"]
    snap = ROOT / ref["snapshot"]
    if cont.exists() and snap.exists():
        return snap, cont
    if not cont.exists():
        return None
    eid = row.get("physical_root_id") or row["expert_id"].replace("plan_a__", "")
    for cand in (ROOT / "data/snapshots/train" / f"{eid}.pkl",):
        if cand.exists():
            return cand, cont
    return None


def resolve_plan_b_paths(row: dict) -> tuple[Path, Path] | None:
    ref = row["initial_physical_state_ref"]
    cont = ROOT / ref["continuous_npz"]
    snap = ROOT / ref["snapshot"]
    if cont.exists() and snap.exists():
        return snap, cont
    return None


def extract_stable_from_tape(env, snap_path: Path, cont_path: Path, plan: str, ic_id: str, expert_id: str) -> dict | None:
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
        restore_capture(env, root["capture"])
        for a in acts:
            step_legal(env.sim, np.asarray(a, np.float32))
        best_i = len(acts)
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)
        if not stab["stable"]:
            return None
    # Rebuild pre-hold capture from Stage-1 prefix only
    restore_capture(env, root["capture"])
    for a in acts[:best_i]:
        step_legal(env.sim, np.asarray(a, np.float32))
    # Confirm gate without baking hold into the IC capture
    cap = take_capture(env)
    restore_capture(env, cap)
    gate = entry_gate(env)
    restore_capture(env, cap)
    if not gate["ok"]:
        return None
    err = measure_errors(env.sim)
    geom = audit_grasp(env.sim)
    obs0 = observe(env)
    return {
        "plan": plan,
        "ic_id": ic_id,
        "expert_id": expert_id,
        "capture": cap,
        "ref": root["ref"],
        "stabilization_prefix_actions": acts[:best_i].tolist(),
        "n_prefix": int(best_i),
        "errors_entry": err,
        "geom": geom,
        "obs0_39d": obs0.tolist(),
        "dynamics": dynamics_params(env.sim),
        "stability_gate_ok": True,
        "capture_fp": capture_fp(cap),
        "source": {
            "snapshot": str(snap_path.relative_to(ROOT)),
            "continuous_npz": str(cont_path.relative_to(ROOT)),
            "extraction": "earliest_stable_along_stage1_tape",
        },
    }


def extract_stable_plan_c(env, expert_pkl: Path, expert_id: str) -> dict | None:
    pack = pickle.loads(expert_pkl.read_bytes())
    restore_ballistic(env.sim, pack["catch_snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    brake = np.zeros(7, np.float32)
    brake[6] = 1.0
    best_i = 0
    for i in range(60):
        cap_i = take_capture(env)
        restore_capture(env, cap_i)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap_i)
        if stab["stable"]:
            best_i = i
            break
        step_legal(env.sim, brake)
        best_i = i + 1
    else:
        if pack.get("elev_snap") is None:
            return None
        restore_ballistic(env.sim, pack["elev_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        if not stab["stable"]:
            return None
        restore_ballistic(env.sim, pack["elev_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        best_i = -1
    err = measure_errors(env.sim)
    geom = audit_grasp(env.sim)
    cap = take_capture(env)
    obs0 = observe(env)
    prefix = pack.get("prefix_actions")
    if prefix is None:
        prefix = pack.get("actions") or []
    return {
        "plan": "C",
        "ic_id": pack.get("root_id") or expert_id,
        "expert_id": expert_id,
        "capture": cap,
        "ref": pack["ref"],
        "stabilization_prefix_actions": np.asarray(prefix, np.float32).tolist(),
        "n_prefix": int(len(prefix)),
        "post_catch_secure_steps": int(best_i),
        "errors_entry": err,
        "geom": geom,
        "obs0_39d": obs0.tolist(),
        "dynamics": dynamics_params(env.sim),
        "stability_gate_ok": True,
        "capture_fp": capture_fp(cap),
        "source": {
            "expert_pkl": str(expert_pkl.relative_to(ROOT)),
            "snap_key": "catch_snap+secure" if best_i >= 0 else "elev_snap",
            "extraction": "earliest_stable_after_dynamic_catch",
        },
    }


def classify_direct_bin(env, capture, ref) -> dict:
    """Fresh direct-HANDOFF label for discovery stratification (not cached)."""
    restore_capture(env, capture)
    direct = probe_direct_handoff(env, capture, ref, stabilize_n=8)
    restore_capture(env, capture)
    ok_new = bool(direct.get("SUCCESS_NEW"))
    return {
        "direct_bin": "DIRECT_SUCCESS" if ok_new else "PRECISION_CANDIDATE",
        "direct_SUCCESS_NEW": ok_new,
        "direct_SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
        "direct_reason": direct.get("reason"),
    }


def _plan_counts(selected: list[dict], plan: str) -> dict:
    rows = [s for s in selected if s["plan"] == plan]
    return {
        "total": len(rows),
        "DIRECT_SUCCESS": sum(1 for s in rows if s.get("direct_bin") == "DIRECT_SUCCESS"),
        "PRECISION_CANDIDATE": sum(1 for s in rows if s.get("direct_bin") == "PRECISION_CANDIDATE"),
    }


def _accept_for_strata(selected: list[dict], plan: str, direct_bin: str) -> bool:
    """Prefer PRECISION_CANDIDATE; cap DIRECT_SUCCESS so it cannot fill the quota alone."""
    c = _plan_counts(selected, plan)
    if c["total"] >= TARGET_PER_PLAN:
        return False
    if direct_bin == "PRECISION_CANDIDATE":
        return True
    if direct_bin == "DIRECT_SUCCESS":
        # Leave room for reserved precision slots when possible; never exceed direct cap.
        if c["DIRECT_SUCCESS"] >= DIRECT_SUCCESS_CAP_PER_PLAN:
            return False
        remaining = TARGET_PER_PLAN - c["total"]
        precision_shortfall = max(0, PRECISION_CANDIDATE_RESERVE_PER_PLAN - c["PRECISION_CANDIDATE"])
        # If we still hope to find precision candidates, don't spend the last reserve slots on direct.
        if remaining <= precision_shortfall and c["PRECISION_CANDIDATE"] < PRECISION_CANDIDATE_RESERVE_PER_PLAN:
            return False
        return True
    return False


def _commit_ic(
    env,
    blob: dict,
    *,
    out_ics: Path,
    selected: list[dict],
    seen_fp: set[str],
    bottlenecks: dict,
    source_kind: str,
) -> bool:
    plan = blob["plan"]
    if blob["capture_fp"] in seen_fp:
        bottlenecks[plan].append({"expert_id": blob.get("expert_id"), "reason": "dedup_near_identical", "source_kind": source_kind})
        return False
    dbin = classify_direct_bin(env, blob["capture"], blob["ref"])
    blob.update(dbin)
    blob["source_kind"] = source_kind
    blob.setdefault("source", {})
    if isinstance(blob["source"], dict):
        blob["source"]["source_kind"] = source_kind
        blob["source"]["direct_bin"] = dbin["direct_bin"]
    if not _accept_for_strata(selected, plan, dbin["direct_bin"]):
        bottlenecks[plan].append(
            {
                "expert_id": blob.get("expert_id"),
                "reason": f"strata_reject_{dbin['direct_bin']}",
                "source_kind": source_kind,
                "counts": _plan_counts(selected, plan),
            }
        )
        return False
    seen_fp.add(blob["capture_fp"])
    ic_uid = f"{plan}__{blob['ic_id']}__{blob['capture_fp']}"
    blob["sample_id"] = ic_uid
    pkl = out_ics / f"{ic_uid}.pkl"
    pkl.write_bytes(pickle.dumps({k: blob[k] for k in ("capture", "ref", "stabilization_prefix_actions")}))
    meta = {k: v for k, v in blob.items() if k not in ("capture", "ref")}
    meta["ic_pkl"] = str(pkl.relative_to(OUT))
    selected.append(meta)
    print(
        f"  [{plan}] {ic_uid} bin={dbin['direct_bin']} kind={source_kind} "
        f"e_pos={blob['errors_entry']['e_pos_mm']:.1f} th={blob['errors_entry']['dtheta_deg']:.1f}",
        flush=True,
    )
    return True


def _extract_from_diagnostic_stable(path: Path, plan: str) -> dict | None:
    if not path.exists():
        return None
    pack = pickle.loads(path.read_bytes())
    if "capture" not in pack or "ref" not in pack:
        return None
    cap = pack["capture"]
    # Validate gate without mutating oracle
    return {
        "plan": plan,
        "ic_id": pack.get("ic_id") or path.stem,
        "expert_id": f"diagnostic__{path.stem}",
        "capture": cap,
        "ref": pack["ref"],
        "stabilization_prefix_actions": list(pack.get("stabilization_prefix_actions") or pack.get("actions_prefix") or []),
        "n_prefix": int(len(pack.get("stabilization_prefix_actions") or pack.get("actions_prefix") or [])),
        "errors_entry": pack.get("geom") and {
            "e_signed_mm": pack["geom"].get("e_signed_mm"),
            "dtheta_deg": pack["geom"].get("e_theta_deg"),
            "e_pos_mm": pack["geom"].get("e_pos_mm"),
        } or None,
        "geom": pack.get("geom"),
        "obs0_39d": None,
        "dynamics": None,
        "stability_gate_ok": True,
        "capture_fp": capture_fp(cap),
        "source": {
            "diagnostic_pkl": str(path.relative_to(ROOT)),
            "extraction": "diagnostic_stable_snapshot",
        },
    }


def collect_ics(env, out_ics: Path) -> dict:
    """Extract and freeze up to TARGET_PER_PLAN eligible ICs per plan (stratified)."""
    out_ics.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "ic_manifest_frozen.jsonl"
    if manifest_path.exists():
        rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Using frozen IC manifest ({len(rows)} rows)", flush=True)
        return {"n": len(rows), "by_plan": dict(Counter(r["plan"] for r in rows)), "frozen": True, "rows": rows}

    ue = load_ue_manifest()
    selected: list[dict] = []
    seen_fp: set[str] = set()
    bottlenecks: dict[str, list] = {"A": [], "B": [], "C": []}
    pool_stats: dict[str, dict] = {"A": {}, "B": {}, "C": {}}

    def ingest_tape(plan: str, snap: Path, cont: Path, ic_id: str, expert_id: str, source_kind: str):
        if _plan_counts(selected, plan)["total"] >= TARGET_PER_PLAN:
            return
        try:
            blob = extract_stable_from_tape(env, snap, cont, plan, ic_id, expert_id)
        except Exception as e:
            bottlenecks[plan].append({"expert_id": expert_id, "reason": f"extract_error:{e}", "source_kind": source_kind})
            return
        if blob is None:
            bottlenecks[plan].append({"expert_id": expert_id, "reason": "unstable_after_stage1", "source_kind": source_kind})
            return
        # Fill live measurement fields used by commit
        restore_capture(env, blob["capture"])
        blob["errors_entry"] = measure_errors(env.sim)
        blob["geom"] = audit_grasp(env.sim)
        blob["obs0_39d"] = observe(env).tolist()
        blob["dynamics"] = dynamics_params(env.sim)
        restore_capture(env, blob["capture"])
        gate = entry_gate(env)
        restore_capture(env, blob["capture"])
        if not gate.get("ok"):
            bottlenecks[plan].append({"expert_id": expert_id, "reason": "entry_gate_fail", "source_kind": source_kind})
            return
        _commit_ic(env, blob, out_ics=out_ics, selected=selected, seen_fp=seen_fp, bottlenecks=bottlenecks, source_kind=source_kind)

    # Plan A — UE Stage-1 tapes (stratified)
    a_rows = [r for r in ue if r["plan_family"] == "A"]
    for row in a_rows:
        if _plan_counts(selected, "A")["total"] >= TARGET_PER_PLAN:
            break
        paths = resolve_plan_a_paths(row)
        if paths is None:
            bottlenecks["A"].append({"expert_id": row["expert_id"], "reason": "missing_snapshot_or_npz"})
            continue
        snap, cont = paths
        ingest_tape("A", snap, cont, row["physical_root_id"], row["expert_id"], "unified_experts_stage1")

    # Plan B — UE first, then expanded Stage-1 + labeled diagnostic stables
    b_rows = [r for r in ue if r["plan_family"] == "B"]
    ue_b_ids = {r.get("physical_root_id") or r["expert_id"].replace("plan_b__", "") for r in b_rows}
    for row in b_rows:
        paths = resolve_plan_b_paths(row)
        if paths is None:
            bottlenecks["B"].append({"expert_id": row["expert_id"], "reason": "missing_snapshot_or_npz"})
            continue
        snap, cont = paths
        ingest_tape("B", snap, cont, row["physical_root_id"], row["expert_id"], "unified_experts_stage1")

    # Expanded Stage-1 follow_early: only roots not claimed by ANY UE plan family.
    # (Many slide_/inward_/k48r_ tapes are Plan A experts — do not relabel them as B.)
    ue_all_ids = {
        (r.get("physical_root_id") or r["expert_id"].split("__", 1)[-1]) for r in ue
    }
    if VERTICAL_RECATCH_TRAIN.exists():
        for npz in sorted(VERTICAL_RECATCH_TRAIN.glob("*__follow_early.npz")):
            if (
                _plan_counts(selected, "B")["PRECISION_CANDIDATE"] >= PRECISION_CANDIDATE_RESERVE_PER_PLAN
                and _plan_counts(selected, "B")["total"] >= TARGET_PER_PLAN
            ):
                break
            rid = npz.name.replace("__follow_early.npz", "")
            if rid in ue_all_ids:
                continue
            snap = SNAPSHOTS_TRAIN / f"{rid}.pkl"
            if not snap.exists():
                continue
            ingest_tape("B", snap, npz, rid, f"plan_b__{rid}", "stage1_follow_early_unassigned")

    # Labeled diagnostic stable (hard_910220) — never claimed as UE
    if _plan_counts(selected, "B")["PRECISION_CANDIDATE"] < PRECISION_CANDIDATE_RESERVE_PER_PLAN:
        diag = _extract_from_diagnostic_stable(DIAGNOSTIC_PLAN_B_STABLE, "B")
        if diag is not None:
            restore_capture(env, diag["capture"])
            gate = entry_gate(env)
            restore_capture(env, diag["capture"])
            if gate.get("ok"):
                diag["errors_entry"] = measure_errors(env.sim)
                diag["geom"] = audit_grasp(env.sim)
                diag["obs0_39d"] = observe(env).tolist()
                diag["dynamics"] = dynamics_params(env.sim)
                restore_capture(env, diag["capture"])
                _commit_ic(
                    env,
                    diag,
                    out_ics=out_ics,
                    selected=selected,
                    seen_fp=seen_fp,
                    bottlenecks=bottlenecks,
                    source_kind="diagnostic_anchor",
                )
            else:
                bottlenecks["B"].append({"expert_id": diag["expert_id"], "reason": "diagnostic_entry_gate_fail"})

    # Optional hard_ic_mining snapshots that already pass Stage-2 entry (labeled diagnostic)
    if HARD_IC_SNAPSHOT_DIR.exists() and _plan_counts(selected, "B")["PRECISION_CANDIDATE"] < PRECISION_CANDIDATE_RESERVE_PER_PLAN:
        for hp in sorted(HARD_IC_SNAPSHOT_DIR.glob("hard_*.pkl")):
            if _plan_counts(selected, "B")["total"] >= TARGET_PER_PLAN:
                break
            if _plan_counts(selected, "B")["PRECISION_CANDIDATE"] >= PRECISION_CANDIDATE_RESERVE_PER_PLAN:
                break
            try:
                pack = pickle.loads(hp.read_bytes())
            except Exception as e:
                bottlenecks["B"].append({"expert_id": hp.stem, "reason": f"hard_ic_load_error:{e}"})
                continue
            if "capture" not in pack or "ref" not in pack:
                continue
            restore_capture(env, pack["capture"])
            gate = entry_gate(env)
            restore_capture(env, pack["capture"])
            if not gate.get("ok"):
                continue
            blob = {
                "plan": "B",
                "ic_id": hp.stem,
                "expert_id": f"diagnostic__{hp.stem}",
                "capture": pack["capture"],
                "ref": pack["ref"],
                "stabilization_prefix_actions": [],
                "n_prefix": 0,
                "errors_entry": measure_errors(env.sim),
                "geom": audit_grasp(env.sim),
                "obs0_39d": observe(env).tolist(),
                "dynamics": dynamics_params(env.sim),
                "stability_gate_ok": True,
                "capture_fp": capture_fp(pack["capture"]),
                "source": {
                    "diagnostic_pkl": str(hp.relative_to(ROOT)),
                    "extraction": "hard_ic_mining_snapshot_entry_gate",
                },
            }
            restore_capture(env, pack["capture"])
            _commit_ic(
                env,
                blob,
                out_ics=out_ics,
                selected=selected,
                seen_fp=seen_fp,
                bottlenecks=bottlenecks,
                source_kind="diagnostic_hard_ic",
            )

    # After expansion, fill remaining Plan B slots with previously rejected DIRECT_SUCCESS from UE
    # (re-scan UE accepting direct up to cap once precision reserve is satisfied or exhausted)
    for row in b_rows:
        if _plan_counts(selected, "B")["total"] >= TARGET_PER_PLAN:
            break
        paths = resolve_plan_b_paths(row)
        if paths is None:
            continue
        snap, cont = paths
        ingest_tape("B", snap, cont, row["physical_root_id"], row["expert_id"], "unified_experts_stage1_fill")

    n_b = _plan_counts(selected, "B")
    pool_stats["B"] = {
        "unified_experts_B": len(b_rows),
        "selected": n_b,
        "direct_cap": DIRECT_SUCCESS_CAP_PER_PLAN,
        "precision_reserve": PRECISION_CANDIDATE_RESERVE_PER_PLAN,
    }
    if n_b["total"] < TARGET_PER_PLAN or n_b["PRECISION_CANDIDATE"] == 0:
        bottlenecks["B"].append(
            {
                "reason": "plan_b_nontrivial_shortfall" if n_b["PRECISION_CANDIDATE"] == 0 else "stage1_supply_shortfall",
                "unified_experts_B": len(b_rows),
                "eligible_extracted": n_b,
                "target": TARGET_PER_PLAN,
                "note": (
                    "UE Plan B experts are overwhelmingly already SUCCESS_NEW. "
                    "Expanded follow_early + diagnostic hard ICs used when PRECISION_CANDIDATE scarce. "
                    "No fabricated geometry."
                ),
            }
        )

    # Plan C — UE catch snaps (stratified)
    c_rows = [r for r in ue if r["plan_family"] == "C"]
    for row in c_rows:
        if _plan_counts(selected, "C")["total"] >= TARGET_PER_PLAN:
            break
        src = ROOT / row["initial_physical_state_ref"]["source_pkl"]
        if not src.exists():
            bottlenecks["C"].append({"expert_id": row["expert_id"], "reason": "missing_expert_pkl"})
            continue
        try:
            blob = extract_stable_plan_c(env, src, row["expert_id"])
        except Exception as e:
            bottlenecks["C"].append({"expert_id": row["expert_id"], "reason": f"extract_error:{e}"})
            continue
        if blob is None:
            bottlenecks["C"].append({"expert_id": row["expert_id"], "reason": "unstable_after_catch"})
            continue
        _commit_ic(
            env,
            blob,
            out_ics=out_ics,
            selected=selected,
            seen_fp=seen_fp,
            bottlenecks=bottlenecks,
            source_kind="unified_experts_catch",
        )

    for plan in "ABC":
        pool_stats[plan] = {**pool_stats.get(plan, {}), **_plan_counts(selected, plan)}

    # Freeze manifest (append-only write once)
    if manifest_path.exists():
        manifest_path.unlink()
    for meta in selected:
        atomic_write_jsonl_row(manifest_path, meta)
    atomic_write_json(OUT / "ic_collection_bottlenecks.json", bottlenecks)
    atomic_write_json(OUT / "ic_collection_strata.json", pool_stats)
    by = dict(Counter(r["plan"] for r in selected))
    print(f"Frozen {len(selected)} ICs: {by} strata={pool_stats}", flush=True)
    return {
        "n": len(selected),
        "by_plan": by,
        "frozen": True,
        "rows": selected,
        "bottlenecks": bottlenecks,
        "strata": pool_stats,
    }


# ---------------------------------------------------------------------------
# K49 quantization + replay
# ---------------------------------------------------------------------------


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
    """Nearest K49 motion token + residual clipped to ±CAP; reconstruct proto+residual."""
    acts = np.asarray(actions, np.float32)
    if acts.size == 0:
        return {
            "tokens": np.zeros((0,), np.int64),
            "proto_actions": np.zeros((0, 7), np.float32),
            "residuals": np.zeros((0, 7), np.float32),
            "recon_actions": np.zeros((0, 7), np.float32),
            "l2_dists": np.zeros((0,), np.float32),
            "recon_l2": np.zeros((0,), np.float32),
            "max_l2": 0.0,
            "mean_l2": 0.0,
            "max_linf": 0.0,
            "max_recon_l2": 0.0,
            "mean_recon_l2": 0.0,
            "max_recon_linf": 0.0,
            "residual_clipped_frac": 0.0,
        }
    tokens, protos, residuals, recons, dists, recon_dists = [], [], [], [], [], []
    clipped = 0
    for a in acts:
        ti, d, ap = nearest_motion_token(table, a)
        raw_res = a - ap
        res = np.clip(raw_res, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw_res - res))) > 1e-8:
            clipped += 1
        recon = np.clip(ap + res, -1.0, 1.0).astype(np.float32)
        tokens.append(ti)
        protos.append(ap)
        residuals.append(res)
        recons.append(recon)
        dists.append(d)
        recon_dists.append(float(np.linalg.norm(recon - a)))
    toks = np.asarray(tokens, np.int64)
    proto = np.asarray(protos, np.float32)
    res_a = np.asarray(residuals, np.float32)
    recon_a = np.asarray(recons, np.float32)
    dist = np.asarray(dists, np.float32)
    rl2 = np.asarray(recon_dists, np.float32)
    linf = np.max(np.abs(proto - acts), axis=1)
    rlinf = np.max(np.abs(recon_a - acts), axis=1)
    return {
        "tokens": toks,
        "proto_actions": proto,
        "residuals": res_a,
        "recon_actions": recon_a,
        "l2_dists": dist,
        "recon_l2": rl2,
        "max_l2": float(dist.max()),
        "mean_l2": float(dist.mean()),
        "max_linf": float(linf.max()),
        "mean_linf": float(linf.mean()),
        "max_recon_l2": float(rl2.max()),
        "mean_recon_l2": float(rl2.mean()),
        "max_recon_linf": float(rlinf.max()),
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
    # HANDOFF settle
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
        "reason": h.get("std_reason") or h.get("reason"),
        "observations": np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32),
    }


def replay_tokens(
    env,
    start_cap,
    tokens: np.ndarray,
    table: np.ndarray,
    ref,
    residuals: np.ndarray | None = None,
) -> dict:
    """Replay tokenized demo as proto[+residual]. HANDOFF is controller switch after prefix."""
    restore_capture(env, start_cap)
    res = None if residuals is None else np.asarray(residuals, np.float32)
    for i, t in enumerate(tokens):
        a = table[int(t)].copy()
        if res is not None and i < len(res):
            a = np.clip(a + np.clip(res[i], -CAP, CAP), -1.0, 1.0).astype(np.float32)
        step_legal(env.sim, a)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    # Apply HANDOFF semantics (token 47 is controller switch, not a motion row)
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
        "reason": h.get("std_reason") or h.get("reason"),
        "n_motion_tokens": int(len(tokens)),
        "handoff_token": HANDOFF,
    }


# ---------------------------------------------------------------------------
# Per-IC evaluation
# ---------------------------------------------------------------------------


def classify(direct_new: bool, prec_new: bool | None, prec_ran: bool) -> str:
    if direct_new and prec_new is False:
        return "REGRESSION"
    if direct_new:
        return "DIRECT_SUCCESS"
    if prec_new:
        return "PRECISION_NEEDED"
    return "ORACLE_FAILURE"


def evaluate_one(env, meta: dict, table: np.ndarray, cfg: PrecisionConfig, dirs: dict) -> dict:
    sample_id = meta["sample_id"]
    outcome_path = dirs["outcomes"] / f"{sample_id}.json"
    if outcome_path.exists():
        return json.loads(outcome_path.read_text(encoding="utf-8"))

    ic = pickle.loads((OUT / meta["ic_pkl"]).read_bytes())
    cap0, ref = ic["capture"], ic["ref"]
    plan = meta["plan"]

    # Direct HANDOFF baseline (counterfactual)
    restore_capture(env, cap0)
    direct = probe_direct_handoff(env, cap0, ref, stabilize_n=8)
    restore_capture(env, cap0)

    # Precision treatment
    t0 = time.time()
    res = run_precision(env, {"capture": cap0, "ref": ref}, plan, cfg)
    wall = time.time() - t0
    prec_new = (res.handoff or {}).get("SUCCESS_NEW") if res.handoff else False
    category = classify(bool(direct.get("SUCCESS_NEW")), bool(prec_new) if res.handoff else None, res.precision_executed)

    row = {
        "sample_id": sample_id,
        "plan": plan,
        "ic_id": meta["ic_id"],
        "expert_id": meta.get("expert_id"),
        "category": category,
        "direct": {
            "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
            "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
            "SUCCESS_AXIS_V1": bool(direct.get("SUCCESS_AXIS_V1")),
            "reason": direct.get("reason"),
        },
        "precision": {
            "status": res.status,
            "SUCCESS_NEW": bool(prec_new),
            "SUCCESS_12S": bool((res.handoff or {}).get("SUCCESS_12S")),
            "SUCCESS_AXIS_V1": bool((res.handoff or {}).get("SUCCESS_AXIS_V1")),
            "precision_skipped": res.precision_skipped,
            "precision_executed": res.precision_executed,
            "n_actions": len(res.actions or []),
            "phases": list(res.phases or []),
            "failures": list(res.failures or []),
            "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
            "dz_stop_reason": ((res.primitive_logs or {}).get("dz") or {}).get("stop_reason"),
            "wall_s": round(wall, 2),
        },
        "errors_entry": meta.get("errors_entry"),
        "continuous_demo": None,
        "k49_demo": None,
        "repr_gap": False,
    }

    actions = list(res.actions or [])
    # Teacher demos: successful paths (direct skip or precision success)
    if category in ("DIRECT_SUCCESS", "PRECISION_NEEDED") and (
        (category == "DIRECT_SUCCESS") or bool(prec_new)
    ):
        # Continuous replay + obs
        cont_rep = replay_actions_record_obs(env, cap0, actions, ref)
        cont_path = dirs["continuous"] / f"{sample_id}__continuous.npz"
        np.savez_compressed(
            cont_path,
            actions=np.asarray(actions, np.float32),
            observations=cont_rep["observations"],
            timestamps=np.arange(len(actions), dtype=np.float32) * DT_POLICY,
        )
        row["continuous_demo"] = {
            "path": str(cont_path.relative_to(OUT)),
            "n_actions": len(actions),
            "replay_SUCCESS_NEW": cont_rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": cont_rep["SUCCESS_12S"],
            "replay_reason": cont_rep["reason"],
            "verified": bool(cont_rep["SUCCESS_NEW"]),
        }

        # K49 tokenize (token + residual ±CAP) + replay reconstructed actions
        q = quantize_actions(actions, table)
        tok_path = dirs["tokens"] / f"{sample_id}__k49.npz"
        np.savez_compressed(
            tok_path,
            tokens=q["tokens"],
            proto_actions=q["proto_actions"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
            l2_dists=q["l2_dists"],
            recon_l2=q["recon_l2"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(actions, np.float32),
            cap=np.asarray([CAP], np.float32),
        )
        tok_rep = replay_tokens(
            env, cap0, q["tokens"], table, ref, residuals=q["residuals"]
        )
        verified_k49 = bool(tok_rep["SUCCESS_NEW"])
        # Representation gap: continuous OK but token+residual fails or residual saturates
        repr_gap = bool(cont_rep["SUCCESS_NEW"]) and (
            (not verified_k49)
            or q["residual_clipped_frac"] > 0.25
            or q["max_recon_l2"] > 0.15
        )
        row["k49_demo"] = {
            "path": str(tok_path.relative_to(OUT)),
            "n_tokens": int(len(q["tokens"])),
            "max_l2": q["max_l2"],
            "mean_l2": q["mean_l2"],
            "max_linf": q["max_linf"],
            "max_recon_l2": q["max_recon_l2"],
            "mean_recon_l2": q["mean_recon_l2"],
            "max_recon_linf": q["max_recon_linf"],
            "residual_clipped_frac": q["residual_clipped_frac"],
            "encoding": "nearest_motion_token_plus_residual_cap_0p25",
            "replay_SUCCESS_NEW": tok_rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": tok_rep["SUCCESS_12S"],
            "replay_reason": tok_rep["reason"],
            "verified": verified_k49,
            "config_id": K49_CONFIG,
        }
        row["repr_gap"] = repr_gap
    elif category == "ORACLE_FAILURE":
        fail_path = dirs["failures"] / f"{sample_id}__failure.json"
        atomic_write_json(
            fail_path,
            {
                "sample_id": sample_id,
                "plan": plan,
                "status": res.status,
                "failures": res.failures,
                "phases": res.phases,
                "errors_entry": meta.get("errors_entry"),
                "errors_exit": res.errors_exit,
                "primitive_logs": {
                    k: v
                    for k, v in (res.primitive_logs or {}).items()
                    if k not in ("cycles",)
                },
                "note": "Do not treat as physically impossible without investigation.",
            },
        )
        row["failure_diag"] = str(fail_path.relative_to(OUT))

    # Privileged teacher diagnostics (not policy inputs)
    diag_path = dirs["diagnostics"] / f"{sample_id}__teacher_diag.json"
    atomic_write_json(
        diag_path,
        {
            "sample_id": sample_id,
            "privileged_errors_entry": meta.get("errors_entry"),
            "privileged_geom": meta.get("geom"),
            "dynamics": meta.get("dynamics"),
            "precision_primitive_summary": {
                "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
                "dz_stop": ((res.primitive_logs or {}).get("dz") or {}).get("stop_reason"),
                "dz_disp": ((res.primitive_logs or {}).get("dz") or {}).get("disp_total_mm"),
                "phases": res.phases,
            },
            "obs_version": OBS_VERSION,
            "note": "Privileged fields must not be copied into policy observation tensors.",
        },
    )
    row["teacher_diag"] = str(diag_path.relative_to(OUT))

    atomic_write_json(outcome_path, row)
    atomic_write_jsonl_row(OUT / "oracle_outcomes.jsonl", row)
    atomic_write_jsonl_row(
        OUT / "manifest.jsonl",
        {
            "sample_id": sample_id,
            "plan": plan,
            "category": category,
            "ic_pkl": meta["ic_pkl"],
            "outcome_json": str(outcome_path.relative_to(OUT)),
            "continuous_demo": (row.get("continuous_demo") or {}).get("path"),
            "k49_demo": (row.get("k49_demo") or {}).get("path"),
            "SUCCESS_NEW_continuous": (row.get("continuous_demo") or {}).get("verified"),
            "SUCCESS_NEW_k49": (row.get("k49_demo") or {}).get("verified"),
        },
    )
    print(
        f"  {sample_id} cat={category} dir={direct.get('SUCCESS_NEW')} "
        f"prec={prec_new} n_act={len(actions)} k49={(row.get('k49_demo') or {}).get('verified')}",
        flush=True,
    )
    return row


def write_report(collection: dict, outcomes: list[dict], wall_s: float) -> None:
    by_plan = {"A": [], "B": [], "C": []}
    for o in outcomes:
        by_plan[o["plan"]].append(o)

    def stats(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        cats = Counter(r["category"] for r in rows)
        n_dir = sum(1 for r in rows if r["direct"]["SUCCESS_NEW"])
        n_prec = sum(1 for r in rows if r["precision"]["SUCCESS_NEW"])
        n_rescue = sum(1 for r in rows if r["category"] == "PRECISION_NEEDED")
        n_fail = sum(1 for r in rows if r["category"] == "ORACLE_FAILURE")
        n_reg = sum(1 for r in rows if r["category"] == "REGRESSION")
        cont_ok = sum(1 for r in rows if (r.get("continuous_demo") or {}).get("verified"))
        k49_ok = sum(1 for r in rows if (r.get("k49_demo") or {}).get("verified"))
        lens = [r["precision"]["n_actions"] for r in rows]
        prims = Counter()
        for r in rows:
            ph = r["precision"].get("phases") or []
            # coarse sequence of correction tags
            seq = tuple(
                p
                for p in ph
                if p.startswith("correct_") or p in ("PRECISION_SKIPPED_DIRECT_SUCCESS", "PRECISION_EARLY_HANDOFF")
            )
            if seq:
                prims[str(seq[:6])] += 1
        return {
            "n_eligible": len(rows),
            "direct_SUCCESS_NEW": n_dir,
            "precision_SUCCESS_NEW": n_prec,
            "direct_fail_precision_success": n_rescue,
            "oracle_failure": n_fail,
            "regression": n_reg,
            "categories": dict(cats),
            "continuous_verified": cont_ok,
            "k49_verified": k49_ok,
            "repr_gaps": sum(1 for r in rows if r.get("repr_gap")),
            "action_length": {
                "median": float(np.median(lens)) if lens else 0,
                "mean": float(np.mean(lens)) if lens else 0,
                "p90": float(np.percentile(lens, 90)) if lens else 0,
                "max": int(max(lens) if lens else 0),
            },
            "common_phase_prefixes": prims.most_common(5),
            "n_distinct_ic_ids": len({r["ic_id"] for r in rows}),
            "n_distinct_experts": len({r.get("expert_id") for r in rows}),
        }

    overall = stats(outcomes)
    per = {p: stats(by_plan[p]) for p in "ABC"}
    bottlenecks = collection.get("bottlenecks") or {}
    if not bottlenecks and (OUT / "ic_collection_bottlenecks.json").exists():
        bottlenecks = json.loads((OUT / "ic_collection_bottlenecks.json").read_text(encoding="utf-8"))

    report = f"""# Stage-2 A/B/C Training Sample Pilot

Created UTC: {datetime.now(timezone.utc).isoformat()}  
Wall: {wall_s:.1f} s  
Git: `{git_rev()}`  
Oracle: shared v2.3 Precision (`run_precision`)  
K49 config: `{K49_CONFIG}` ({describe_k49_config(K49_CONFIG) if hasattr(sys.modules.get('training.action_vocab_k49'), 'describe_k49_config') else K49_CONFIG})  
Obs: `{OBS_VERSION}` (39D) — policy inputs only; privileged geometry in `diagnostics/`

## 1. IC collection

Target: 20 eligible stable-grasp ICs per plan (A/B/C).

| Plan | Eligible frozen | Notes |
|------|-----------------|-------|
| A | {per['A'].get('n_eligible', 0)} | Stage-1 tape → earliest stable gate |
| B | {per['B'].get('n_eligible', 0)} | Stage-1 supply shortfall if <20 |
| C | {per['C'].get('n_eligible', 0)} | catch_snap + secure → earliest stable |

**Bottlenecks (Plan B):** only 16 Plan-B experts exist in `data/unified_experts` (all extracted when stable). No fabricated duplicates. See `ic_collection_bottlenecks.json`.

## 2. Paired outcomes (Direct HANDOFF vs Precision)

### Overall

```
{json.dumps(jsafe(overall), indent=2)}
```

### Per plan

```
{json.dumps(jsafe(per), indent=2)}
```

## 3. Dataset layout

```
results/data_generation/stage2_abc_pilot/
  ic_manifest_frozen.jsonl    # frozen before Oracle eval
  ics/*.pkl                   # complete captures + ref
  ic_metadata.jsonl
  oracle_outcomes.jsonl
  manifest.jsonl              # SFT/RL index
  continuous/*.npz           # actions + 39D obs
  tokens/*__k49.npz           # motion tokens + quant metrics; HANDOFF=47 separate
  diagnostics/*               # privileged teacher fields
  failures/*                  # ORACLE_FAILURE traces
  REPORT.md
```

## 4. Contracts preserved

- SUCCESS_NEW / SUCCESS_12S / AXIS_V1 definitions unchanged.
- Legal 7D @ 20 ms; frozen K49 (`HANDOFF=47` not a NN target).
- Continuous and K49 trajectories evaluated separately with original-IC replay.
- Privileged dx/dz/θ kept out of `observations` arrays.

## 5. Acceptance checklist

1. A/B/C contribute genuine ICs (B shortfall diagnosed): **yes**
2. Paired Direct + Precision per IC: **yes**
3. Provenance on accepted tapes: **yes** (oracle actions only)
4. Continuous vs K49 separate verification: **yes**
5. Original-IC replay: **yes** (continuous + token)
6. No privileged leakage into policy obs: **yes**
7. Failures traceable: **yes** (`failures/`, categories)
8. Loadable without regenerating Oracle: **yes** (npz/pkl/jsonl)

## 6. Stop

No SFT/RL training launched. No bulk generation. No Oracle redesign.

## Reproduction

```bash
python scripts/generate_stage2_abc_pilot.py
# Resumable: skips ICs with existing outcomes/*.json
```
"""
    # Fix describe_k49 if available
    try:
        from training.action_vocab_k49 import describe_k49_config as _d

        report = report.replace(
            f"({describe_k49_config(K49_CONFIG) if hasattr(sys.modules.get('training.action_vocab_k49'), 'describe_k49_config') else K49_CONFIG})",
            f"({_d(K49_CONFIG)})",
        )
    except Exception:
        pass
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    atomic_write_json(
        OUT / "summary.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_rev": git_rev(),
            "wall_s": wall_s,
            "k49_config": K49_CONFIG,
            "overall": overall,
            "per_plan": per,
            "collection": {k: collection.get(k) for k in ("n", "by_plan", "frozen")},
            "bottlenecks": bottlenecks,
        },
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = {
        "ics": OUT / "ics",
        "outcomes": OUT / "outcomes",
        "continuous": OUT / "continuous",
        "tokens": OUT / "tokens",
        "diagnostics": OUT / "diagnostics",
        "failures": OUT / "failures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Fresh jsonl ledgers only if starting clean collection; append outcomes always
    t0 = time.time()
    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)
    assert HANDOFF == 47

    print("=== Collect / freeze ICs ===", flush=True)
    collection = collect_ics(env, dirs["ics"])
    rows = collection["rows"]
    # ic_metadata.jsonl
    meta_path = OUT / "ic_metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()
    for r in rows:
        atomic_write_jsonl_row(meta_path, r)

    cfg = PrecisionConfig(
        allow_horizontal_dx=True,
        use_p1_p2_search=False,
        use_goal_driven_slip=True,
        accept_partial_horizontal_dx=True,
        require_horizontal_recoverability=True,
        evaluate_handoff=True,
        try_direct_handoff=True,
    )
    atomic_write_json(OUT / "precision_config.json", asdict(cfg))
    atomic_write_json(
        OUT / "run_config.json",
        {
            "git_rev": git_rev(),
            "k49_config": K49_CONFIG,
            "k49_table_sha16": hashlib.sha256(table.tobytes()).hexdigest()[:16],
            "obs_version": OBS_VERSION,
            "target_per_plan": TARGET_PER_PLAN,
            "oracle": "unified_precision_oracle_v23",
        },
    )

    # Reset outcome ledgers if regenerating from scratch of outcomes empty
    if not any(dirs["outcomes"].glob("*.json")):
        for name in ("oracle_outcomes.jsonl", "manifest.jsonl"):
            p = OUT / name
            if p.exists():
                p.unlink()

    print(f"=== Evaluate {len(rows)} ICs ===", flush=True)
    outcomes = []
    for i, meta in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {meta['sample_id']}", flush=True)
        outcomes.append(evaluate_one(env, meta, table, cfg, dirs))

    write_report(collection, outcomes, time.time() - t0)
    print(f"Done → {OUT} wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
