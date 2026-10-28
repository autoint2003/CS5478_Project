"""P21 — Recover full Stage-1 expert trajectories; M111 full-horizon replay audit.

Does not train SFT. Does not mutate Oracle / M111 / physics / STRICT.
Does not fabricate demos. Uses stored expert actions only (or refuses).
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import OBS_DIM_AIR, OBS_VERSION, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from models.m111_checkpoint import HANDOFF, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/diagnostics/p21_full_stage1_experts"
DATASET = OUT / "dataset"
NPZ_DIR = DATASET / "npz"
UNIFIED = ROOT / "data/unified_experts/manifest.jsonl"
FULLCHAIN = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OFFICIAL = ROOT / "data/experts_v2_official"
P17_SPLIT = ROOT / "results/diagnostics/p17_sft_readiness/lineage_split.json"
P19 = ROOT / "results/diagnostics/p19_sft_packaging"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def audit_sources() -> dict[str, Any]:
    """Phase 1: locate what is stored where (no physics)."""
    official = []
    man = OFFICIAL / "manifest.jsonl"
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            raw = pickle.loads((ROOT / m["record_path"]).read_bytes())
            acts = np.asarray(raw.get("actions_7d"), np.float32).reshape(-1, 7)
            official.append(
                {
                    "traj_id": m.get("expert_id") or m.get("attempt_id"),
                    "branch_id": m.get("branch_id"),
                    "source_ic_id": m.get("source_ic_id") or raw.get("source_ic_id"),
                    "source_ic_path": raw.get("source_ic_path") or m.get("source_ic_path"),
                    "n_actions_stored": int(acts.shape[0]),
                    "has_stage1_actions_field": any(
                        k in raw for k in ("stage1_actions", "prefix_actions", "recovery_actions")
                    ),
                    "stage1_label": raw.get("stage1"),
                    "continuation": raw.get("continuation"),
                    "start_is_post_stage1_ic": True,  # experts_v2 generate.py loads stage2 IC
                }
            )

    unified = []
    for line in UNIFIED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fam = r["plan_family"]
        n = None
        start_kind = None
        action_source = None
        if fam in ("A", "B"):
            a = r.get("actions_7d")
            if a is not None:
                n = len(a)
                action_source = "unified_manifest.actions_7d"
            init = r.get("initial_physical_state_ref") or {}
            start_kind = "snapshot_capture"
            if init.get("snapshot"):
                action_source = action_source or f"continuous:{init.get('continuous_npz')}"
        else:
            pkl = ROOT / ((r.get("provenance") or {}).get("expert_pkl") or "")
            if pkl.is_file():
                raw = pickle.loads(pkl.read_bytes())
                acts = raw.get("actions")
                if acts is None:
                    acts = raw.get("prefix_actions")
                n = len(acts) if acts is not None else 0
                action_source = str(pkl.relative_to(ROOT)).replace("\\", "/")
                start_kind = "catch_snap"
        unified.append(
            {
                "expert_id": r["expert_id"],
                "plan_family": fam,
                "physical_root_id": r.get("physical_root_id"),
                "n_actions": n,
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "mechanism": r.get("contact_grasp_mechanism"),
                "start_kind": start_kind,
                "action_source": action_source,
                "stage1_actions_stored": bool(n and n > 0),
            }
        )

    fullchain = []
    fc_man = FULLCHAIN / "manifest.jsonl"
    if fc_man.is_file():
        for line in fc_man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cd = r.get("continuous_demo") or {}
            fullchain.append(
                {
                    "sample_id": r.get("sample_id"),
                    "source_label": r.get("source_label") or r.get("family"),
                    "partition": r.get("partition"),
                    "category": r.get("category"),
                    "n_actions_stage1": r.get("n_actions_stage1"),
                    "n_actions_precision": r.get("n_actions_precision"),
                    "n_actions_full": r.get("n_actions_full"),
                    "stage1_executed": r.get("stage1_executed"),
                    "precision_executed": r.get("precision_executed"),
                    "continuous_verified": bool(isinstance(cd, dict) and cd.get("replay_SUCCESS_NEW")),
                    "starts_at_recovery_ic": True,
                    "ic_pkl": r.get("ic_pkl"),
                    "lineage_id": r.get("lineage_id"),
                }
            )

    return {
        "official_n": len(official),
        "official_with_stage1_action_field": sum(1 for r in official if r["has_stage1_actions_field"]),
        "official_mean_n_actions": float(np.mean([r["n_actions_stored"] for r in official])) if official else 0,
        "unified_n": len(unified),
        "unified_by_family": dict(Counter(r["plan_family"] for r in unified)),
        "unified_stage1_stored": sum(1 for r in unified if r["stage1_actions_stored"]),
        "fullchain_n": len(fullchain),
        "fullchain_expert_verified": sum(1 for r in fullchain if r["continuous_verified"]),
        "fullchain_s1_gt20": sum(1 for r in fullchain if int(r.get("n_actions_stage1") or 0) > 20),
        "fullchain_recovery_plus_precision": sum(
            1
            for r in fullchain
            if r["continuous_verified"]
            and int(r.get("n_actions_stage1") or 0) > 0
            and int(r.get("n_actions_precision") or 0) > 0
        ),
        "official_rows": official,
        "unified_rows": unified,
        "fullchain_rows": fullchain,
    }


def load_unified_episode(row: dict) -> dict[str, Any] | None:
    """Load recovery start + Stage-1 actions from unified_experts (no fabrication)."""
    fam = row["plan_family"]
    if fam in ("A", "B"):
        acts = row.get("actions_7d")
        init = row.get("initial_physical_state_ref") or {}
        if acts is None:
            cont = ROOT / (init.get("continuous_npz") or "")
            if not cont.is_file():
                return None
            z = np.load(cont)
            key = "actions" if "actions" in z.files else None
            if key is None:
                return None
            acts = z[key]
        acts = np.asarray(acts, np.float32).reshape(-1, 7)
        snap_path = ROOT / (init.get("snapshot") or "")
        if not snap_path.is_file():
            return None
        snap = pickle.loads(snap_path.read_bytes())
        cap = snap.get("capture")
        ref = snap.get("ref")
        if cap is None or ref is None:
            return None
        return {
            "traj_id": row["expert_id"],
            "family": fam,
            "lineage_id": row.get("physical_root_id") or row["expert_id"],
            "skill": fam,
            "source_corpus": "unified_experts",
            "start_kind": "snapshot_capture",
            "capture": cap,
            "ref": ref,
            "actions": acts,
            "n_stage1": len(acts),
            "n_precision": 0,
            "includes_precision": False,
            "mechanism": row.get("contact_grasp_mechanism"),
            "horizon_kind": "recovery_stage1_only",
        }
    # C
    pkl = ROOT / ((row.get("provenance") or {}).get("expert_pkl") or "")
    if not pkl.is_file():
        return None
    raw = pickle.loads(pkl.read_bytes())
    acts = raw.get("actions")
    if acts is None:
        acts = raw.get("prefix_actions")
    if acts is None:
        return None
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    catch = raw.get("catch_snap")
    ref = raw.get("ref")
    if catch is None or ref is None:
        return None
    return {
        "traj_id": row["expert_id"],
        "family": "C",
        "lineage_id": row.get("physical_root_id") or row["expert_id"],
        "skill": "C",
        "source_corpus": "unified_experts",
        "start_kind": "catch_snap",
        "catch_snap": catch,
        "capture": None,  # restored via ballistic
        "ref": ref,
        "actions": acts,
        "n_stage1": len(acts),
        "n_precision": 0,
        "includes_precision": False,
        "mechanism": row.get("contact_grasp_mechanism"),
        "horizon_kind": "recovery_stage1_only",
        "raw_pkl": str(pkl.relative_to(ROOT)).replace("\\", "/"),
    }


def load_fullchain_episode(row: dict) -> dict[str, Any] | None:
    """Load Recovery IC → Stage-1 → optional Precision from fullchain corpus."""
    cd = row.get("continuous_demo") or {}
    if not isinstance(cd, dict) or not cd.get("path"):
        return None
    if not cd.get("replay_SUCCESS_NEW"):
        return None
    cont_path = FULLCHAIN / str(cd["path"]).replace("\\", "/")
    ic_path = FULLCHAIN / str(row.get("ic_pkl") or "").replace("\\", "/")
    if not cont_path.is_file() or not ic_path.is_file():
        return None
    z = np.load(cont_path)
    acts = np.asarray(z["actions"], np.float32).reshape(-1, 7)
    ic = pickle.loads(ic_path.read_bytes())
    cap = ic.get("capture")
    ref = ic.get("ref")
    if cap is None or ref is None:
        return None
    if row.get("n_actions_stage1") is not None:
        n_s1 = int(row["n_actions_stage1"])
    elif "stage1_end_idx" in z.files:
        n_s1 = int(z["stage1_end_idx"][0])
    else:
        n_s1 = 0
    n_prec = int(row.get("n_actions_precision") or max(0, len(acts) - n_s1))
    lab = str(row.get("source_label") or row.get("family") or "?")
    return {
        "traj_id": row["sample_id"],
        "family": lab if lab in ("A", "B", "C") else lab,
        "lineage_id": row.get("lineage_id") or row["sample_id"],
        "skill": lab,
        "source_corpus": "stage2_abc_recovery_fullchain_v1",
        "start_kind": "recovery_ic_capture",
        "capture": cap,
        "ref": ref,
        "actions": acts,
        "n_stage1": n_s1,
        "n_precision": n_prec,
        "includes_precision": n_prec > 0,
        "mechanism": row.get("category"),
        "horizon_kind": "recovery_plus_precision" if n_prec > 0 else "recovery_stage1_only",
        "obs_precomputed": np.asarray(z["observations"], np.float32) if "observations" in z.files else None,
    }


def restore_start(env, ep: dict) -> None:
    if ep.get("start_kind") == "catch_snap":
        restore_ballistic(env.sim, ep["catch_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
    else:
        restore_capture(env, ep["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)


def replay_from_start(
    env,
    ep: dict,
    actions: np.ndarray,
    *,
    record_obs: bool = False,
    stabilize_n: int = 8,
) -> dict:
    restore_start(env, ep)
    hist = env._hist
    obs_list = []
    prev_list = []
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    for a in acts:
        if record_obs:
            obs = np.asarray(
                observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
                np.float32,
            )
            obs_list.append(obs)
            prev_list.append(env._prev.copy())
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=int(stabilize_n))
    out = {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "reason": hand.get("std_reason") or hand.get("reason"),
        "n_actions": int(len(acts)),
    }
    if record_obs:
        # append HANDOFF obs
        hob = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_list.append(hob)
        prev_list.append(env._prev.copy())
        out["observations"] = np.stack(obs_list, axis=0).astype(np.float32)
        out["prev_actions"] = np.stack(prev_list, axis=0).astype(np.float32)
    return out


def package_episode(ep: dict, cont: dict, q: dict, book, handoff_id: int, split: str) -> dict:
    obs = cont["observations"]
    prev = cont["prev_actions"]
    n_motion = int(ep["actions"].shape[0])
    tokens = np.concatenate([np.asarray(q["tokens"], np.int64), np.array([handoff_id], np.int64)])
    residuals = np.concatenate(
        [np.asarray(q["residuals"], np.float32), np.zeros((1, 7), np.float32)], axis=0
    )
    assert obs.shape[0] == len(tokens) == n_motion + 1
    stem = hashlib.sha1(ep["traj_id"].encode()).hexdigest()[:16]
    rel = f"dataset/npz/{stem}.npz"
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=obs.astype(np.float32),
        prev_actions=prev.astype(np.float32),
        tokens=tokens,
        residuals=residuals,
        executed_actions=np.concatenate(
            [np.asarray(q["recon_actions"], np.float32), np.zeros((1, 7), np.float32)], axis=0
        ),
        continuous_actions=np.asarray(ep["actions"], np.float32),
        loss_mask=np.ones(len(tokens), dtype=np.bool_),
        episode_boundary=np.array([False] * (len(tokens) - 1) + [True]),
        handoff_id=np.int64(handoff_id),
        n_stage1=np.int32(ep["n_stage1"]),
        n_precision=np.int32(ep["n_precision"]),
    )
    return {
        "traj_id": ep["traj_id"],
        "split": split,
        "family": ep["family"],
        "lineage_id": ep["lineage_id"],
        "source_corpus": ep["source_corpus"],
        "horizon_kind": ep["horizon_kind"],
        "includes_precision": ep["includes_precision"],
        "n_stage1": ep["n_stage1"],
        "n_precision": ep["n_precision"],
        "n_motion": n_motion,
        "n_steps_with_handoff": int(len(tokens)),
        "npz_relpath": rel.replace("\\", "/"),
        "obs_dim": OBS_DIM_AIR,
        "obs_version": OBS_VERSION,
        "start_distribution": "recovery_ic_pre_stage1",
        "mechanism": ep.get("mechanism"),
    }


def assign_split(lineage_id: str, p17_map: dict[str, str], rng: np.random.Generator) -> str:
    # try exact / suffix match against P17
    if lineage_id in p17_map:
        return p17_map[lineage_id]
    for k, v in p17_map.items():
        if lineage_id in k or k in lineage_id:
            return v
    # deterministic hash split 80/20
    h = int(hashlib.sha1(lineage_id.encode()).hexdigest()[:8], 16)
    return "train" if (h % 100) < 80 else "val"


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    print("=== P21 audit ===", flush=True)
    audit = audit_sources()

    audit_md = f"""# STAGE1_SOURCE_AUDIT

Generated: {_utc()}

## Question: are complete Stage-1 action sequences already stored?

### 1. Official `experts_v2` (P18/P19/P20 source)

- N records: **{audit['official_n']}**
- Records with a `stage1_actions` (or equivalent) field: **{audit['official_with_stage1_action_field']}**
- Mean stored `actions_7d` length: **{audit['official_mean_n_actions']:.1f}**
- Start state: **post–Stage-1 IC** from `stage2_abc_final_v2/ics` (see `generate.load_ic_state`)
- Branch label (`A_STABLE_*`, `plan_a_stable_grasp`, …) does **not** imply Stage-1 controls are on the tape

**Verdict:** Official experts are **Stage-2 continuation only**. Full Stage-1 A/B/C actions are **not** stored here.

### 2. `data/unified_experts` (Plan A/B/C frozen experts)

- N: **{audit['unified_n']}** ({audit['unified_by_family']})
- With stored Stage-1 actions: **{audit['unified_stage1_stored']}**
- Start: A/B `snapshot.capture` (recovery entry); C `catch_snap`
- Actions: A/B `actions_7d` / continuous NPZ; C expert pkl `actions`/`prefix_actions`

**Verdict:** This is the primary recoverable corpus for **genuine Stage-1 A/B/C** recovery tapes.

### 3. `stage2_abc_recovery_fullchain_v1`

- N manifest: **{audit['fullchain_n']}**; continuous-verified: **{audit['fullchain_expert_verified']}**
- Stage-1 length >20: **{audit['fullchain_s1_gt20']}** (mostly Plan C; A/B often truncate to ~5 settle steps)
- Recovery→Precision verified: **{audit['fullchain_recovery_plus_precision']}**
- Starts at Recovery-entry IC; stores fullchain continuous NPZ + stage1/precision splits

**Verdict:** Usable for full-horizon **Recovery IC → (short/long Stage-1) → Precision** where verified.
A/B “Stage-1” prefixes are often short settle segments of older tapes — still pre–Stage-2, but not always a long Plan A/B oracle rollouts.

### 4. Reproduction policy (this run)

- Prefer **stored actions** from unified_experts and fullchain continuous NPZs
- Restore the **recovery IC / catch_snap**, not the Stage-1 exit capture
- Do **not** interpolate or invent actions
- Refuse candidates missing start capture or action arrays
"""
    _write_md(OUT / "STAGE1_SOURCE_AUDIT.md", audit_md)
    (OUT / "source_audit_raw.json").write_text(
        json.dumps({k: v for k, v in audit.items() if not k.endswith("_rows")}, indent=2),
        encoding="utf-8",
    )

    # Load candidates
    print("=== load candidates ===", flush=True)
    unified_raw = [json.loads(L) for L in UNIFIED.read_text(encoding="utf-8").splitlines() if L.strip()]
    episodes: list[dict] = []
    coverage_rows = []
    for r in unified_raw:
        if not r.get("SUCCESS_NEW"):
            continue
        ep = load_unified_episode(r)
        if ep is None:
            coverage_rows.append(
                {
                    "traj_id": r.get("expert_id"),
                    "family": r.get("plan_family"),
                    "status": "LOAD_FAIL",
                    "n_stage1": None,
                }
            )
            continue
        episodes.append(ep)
        coverage_rows.append(
            {
                "traj_id": ep["traj_id"],
                "family": ep["family"],
                "source_corpus": ep["source_corpus"],
                "status": "LOADED",
                "n_stage1": ep["n_stage1"],
                "n_precision": ep["n_precision"],
                "horizon_kind": ep["horizon_kind"],
                "start_kind": ep["start_kind"],
                "mechanism": ep.get("mechanism"),
            }
        )

    # Fullchain recovery+precision (and long-C stage1-only already partly in unified)
    fc_rows = [json.loads(L) for L in (FULLCHAIN / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if L.strip()]
    seen_lin = {e["lineage_id"] for e in episodes}
    for r in fc_rows:
        if r.get("partition") not in ("expert", "expert_continuous_only"):
            continue
        cd = r.get("continuous_demo") or {}
        if not (isinstance(cd, dict) and cd.get("replay_SUCCESS_NEW")):
            continue
        # Prefer precision-containing or C with long stage1 not already covered
        n_prec = int(r.get("n_actions_precision") or 0)
        n_s1 = int(r.get("n_actions_stage1") or 0)
        lab = str(r.get("source_label") or "")
        if lab.startswith("STABLE"):
            continue  # not A/B/C Stage-1 recovery
        if n_prec <= 0 and n_s1 <= 20:
            continue  # skip tiny settle-only fullchain A/B that add little vs unified
        ep = load_fullchain_episode(r)
        if ep is None:
            continue
        # Avoid exact traj_id dup; allow same lineage if precision adds horizon
        if ep["traj_id"] in {e["traj_id"] for e in episodes}:
            continue
        episodes.append(ep)
        coverage_rows.append(
            {
                "traj_id": ep["traj_id"],
                "family": ep["family"],
                "source_corpus": ep["source_corpus"],
                "status": "LOADED",
                "n_stage1": ep["n_stage1"],
                "n_precision": ep["n_precision"],
                "horizon_kind": ep["horizon_kind"],
                "start_kind": ep["start_kind"],
                "mechanism": ep.get("mechanism"),
            }
        )

    _write_csv(OUT / "STAGE1_ACTION_COVERAGE.csv", coverage_rows)
    print(f"loaded episodes: {len(episodes)}", Counter(e["family"] for e in episodes), flush=True)

    # Splits
    p17_map = {}
    if P17_SPLIT.is_file():
        split = json.loads(P17_SPLIT.read_text(encoding="utf-8"))
        p17_map = {L["lineage_id"]: L["split"] for L in split.get("lineages", [])}
    rng = np.random.default_rng(21)

    book, rho, cb = load_m111_book()
    handoff_id = int(cb.handoff_id)
    env = make_env()

    manifest_rows = []
    replay_rows = []
    print("=== full-horizon continuous + M111 replay ===", flush=True)
    for i, ep in enumerate(episodes):
        t_i = time.time()
        sp = assign_split(str(ep["lineage_id"]), p17_map, rng)
        row = {
            "traj_id": ep["traj_id"],
            "family": ep["family"],
            "lineage_id": ep["lineage_id"],
            "source_corpus": ep["source_corpus"],
            "horizon_kind": ep["horizon_kind"],
            "includes_precision": ep["includes_precision"],
            "n_stage1": ep["n_stage1"],
            "n_precision": ep["n_precision"],
            "n_actions": int(len(ep["actions"])),
            "split": sp,
            "start_kind": ep["start_kind"],
        }
        try:
            cont = replay_from_start(env, ep, ep["actions"], record_obs=True, stabilize_n=8)
        except Exception as exc:
            row.update({"cont_SUCCESS_NEW": False, "cont_error": str(exc), "eligible": False})
            replay_rows.append(row)
            print(f"  FAIL cont {ep['traj_id'][:40]} {exc}", flush=True)
            continue
        row["cont_SUCCESS_NEW"] = cont["SUCCESS_NEW"]
        row["cont_SUCCESS_12S"] = cont["SUCCESS_12S"]
        row["cont_reason"] = cont["reason"]
        if not cont["SUCCESS_NEW"]:
            row["eligible"] = False
            row["m111_SUCCESS_NEW"] = None
            replay_rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(episodes)}] cont fail {ep['traj_id'][:40]}", flush=True)
            continue

        q = quantize_trajectory(book, ep["actions"], rho, w=DEFAULT_WEIGHTS)
        try:
            m111 = replay_from_start(
                env, ep, q["recon_actions"], record_obs=False, stabilize_n=8
            )
        except Exception as exc:
            row.update({"m111_SUCCESS_NEW": False, "m111_error": str(exc), "eligible": False})
            replay_rows.append(row)
            continue
        row["m111_SUCCESS_NEW"] = m111["SUCCESS_NEW"]
        row["m111_SUCCESS_12S"] = m111["SUCCESS_12S"]
        row["m111_reason"] = m111["reason"]
        row["m111_clipped_frac"] = float(q["clipped_frac"])
        row["m111_mean_recon_wlinf"] = float(q["mean_recon_wlinf"])
        row["m111_n_unique_tokens"] = int(len(set(int(t) for t in q["tokens"]))) if len(q["tokens"]) else 0
        row["eligible"] = bool(m111["SUCCESS_NEW"])
        row["elapsed_s"] = round(time.time() - t_i, 3)

        if row["eligible"]:
            pack = package_episode(ep, cont, q, book, handoff_id, sp)
            manifest_rows.append(pack)

        replay_rows.append(row)
        if (i + 1) % 5 == 0 or ep["family"] == "C":
            print(
                f"  [{i+1}/{len(episodes)}] {ep['family']} {ep['traj_id'][:36]} "
                f"cont={row['cont_SUCCESS_NEW']} m111={row['m111_SUCCESS_NEW']} "
                f"n={row['n_actions']} {row['elapsed_s']}s",
                flush=True,
            )

    _write_csv(OUT / "FULL_HORIZON_EXPERT_MANIFEST.csv", manifest_rows)
    _write_csv(OUT / "FULL_HORIZON_M111_REPLAY.csv", replay_rows)

    # Aggregates
    by_fam = defaultdict(lambda: {"cont": 0, "m111": 0, "n": 0})
    for r in replay_rows:
        by_fam[r["family"]]["n"] += 1
        if r.get("cont_SUCCESS_NEW"):
            by_fam[r["family"]]["cont"] += 1
        if r.get("m111_SUCCESS_NEW"):
            by_fam[r["family"]]["m111"] += 1
    rec_only = [r for r in replay_rows if not r.get("includes_precision")]
    rec_prec = [r for r in replay_rows if r.get("includes_precision")]
    elig = [r for r in replay_rows if r.get("eligible")]
    lineages = {r["lineage_id"] for r in manifest_rows}

    # Divergence notes
    cont_ok_m111_fail = [
        r for r in replay_rows if r.get("cont_SUCCESS_NEW") and r.get("m111_SUCCESS_NEW") is False
    ]
    both_fail = [r for r in replay_rows if r.get("cont_SUCCESS_NEW") is False]

    proposal = f"""# TRAINING_DATASET_PROPOSAL

Generated: {_utc()}

## Primary dataset (task-aligned full-horizon)

Path: `results/diagnostics/p21_full_stage1_experts/dataset/`

- **Start distribution:** recovery IC / catch_snap **before** Stage-1 (not Stage-1 exit)
- **Horizon:** `Recovery IC → Stage-1 [→ Precision] → HANDOFF`
- **N packaged (M111 + continuous STRICT):** **{len(manifest_rows)}**
- **Distinct lineages:** **{len(lineages)}**
- Train/val: lineage-aware (`split` column); P17 map when available else hash 80/20
- Policy inputs: 39-D airborne obs + prev action; targets: M111 token/residual; terminal HANDOFF
- Skill / corpus labels are analysis-only

### Composition (packaged)

| Family | N |
|---|---:|
"""
    for fam, c in Counter(r["family"] for r in manifest_rows).most_common():
        proposal += f"| {fam} | {c} |\n"
    proposal += f"""
| horizon recovery_stage1_only | {sum(1 for r in manifest_rows if r['horizon_kind']=='recovery_stage1_only')} |
| horizon recovery_plus_precision | {sum(1 for r in manifest_rows if r['horizon_kind']=='recovery_plus_precision')} |

## Optional secondary resource (do not mix unlabeled)

P19 Stage-2 continuation-only package (`results/diagnostics/p19_sft_packaging/`) remains available for
HANDOFF/Precision-from-stable training. Its start distribution is **post–Stage-1 capture**.

**Mixing rule:** never pool P19 and P21 samples in one dataloader without an explicit
`start_distribution` field and separate loss groups / curricula.

## Recommended SFT curriculum (future; not run here)

1. Warm-start on P21 full-horizon A/B/C (motion-heavy).
2. Optionally add P19 direct-HANDOFF with down-weighting.
3. Evaluate closed-loop from recovery ICs (not from Stage-1 exits).

## Not ready until

- Full-horizon M111 STRICT rates reviewed in `FINAL_REPORT.md`
- No silent inclusion of continuation-only tapes as “recovery”
"""
    _write_md(OUT / "TRAINING_DATASET_PROPOSAL.md", proposal)

    # Also write JSON manifest
    (OUT / "SFT_FULL_HORIZON_MANIFEST.json").write_text(
        json.dumps(
            {
                "created_utc": _utc(),
                "dataset_id": "p21_full_horizon_m111_v1",
                "n_trajectories": len(manifest_rows),
                "n_train": sum(1 for r in manifest_rows if r["split"] == "train"),
                "n_val": sum(1 for r in manifest_rows if r["split"] == "val"),
                "start_distribution": "recovery_ic_pre_stage1",
                "vocab": "M111",
                "trajectories": manifest_rows,
                "optional_continuation_only": str(P19.relative_to(ROOT)).replace("\\", "/"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_abc = len(manifest_rows)
    n_rec_prec = sum(1 for r in manifest_rows if r["includes_precision"])
    m111_rate = sum(1 for r in elig) / max(1, sum(1 for r in replay_rows if r.get("cont_SUCCESS_NEW")))

    suitable = (
        n_abc >= 50
        and sum(1 for r in manifest_rows if r["family"] in ("A", "B", "C")) >= 40
        and m111_rate >= 0.90
    )

    final = f"""# FINAL_REPORT — P21 Full Stage-1 Experts

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **How many complete A/B/C expert trajectories can be recovered?**  
   Packaged full-horizon (continuous STRICT + M111 STRICT from recovery IC): **{n_abc}**  
   Attempted: {len(episodes)}; continuous-ok: {sum(1 for r in replay_rows if r.get('cont_SUCCESS_NEW'))};  
   continuous-fail (refused, not fabricated): {len(both_fail)}.

2. **How many distinct recovery IC lineages?**  
   **{len(lineages)}** among packaged trajectories.

3. **Does M111 preserve complete Stage-1 recovery behavior?**  
   Among continuous-ok full-horizon replays: M111 success **{sum(1 for r in elig)}/{sum(1 for r in replay_rows if r.get('cont_SUCCESS_NEW'))}** ({m111_rate:.1%}).  
   Cont-ok but M111-fail: **{len(cont_ok_m111_fail)}**.  
   Per family (cont / m111 / attempted): { {k: dict(v) for k,v in by_fam.items()} }.  
   **Do not cite P18 221/221** — that was continuation-from-capture only.

4. **How many trajectories include Recovery → Precision?**  
   Packaged with `includes_precision`: **{n_rec_prec}**.  
   Replay rows tagged recovery+precision: {sum(1 for r in replay_rows if r.get('includes_precision'))}.

5. **Suitable for genuine multi-skill recovery SFT?**  
   **{"YES — proceed to packaging-gated SFT (not started here)" if suitable else "CONDITIONAL / NOT YET — see rates and coverage gaps"}**.  
   Keep P19 continuation-only as a **separate** optional resource with explicit start_distribution tags.

## Integrity

- No Oracle / M111 / physics / STRICT changes
- No interpolated or fabricated actions
- No SFT training launched
- Official experts_v2 left untouched

## Key files

- `STAGE1_SOURCE_AUDIT.md`
- `FULL_HORIZON_EXPERT_MANIFEST.csv` / `SFT_FULL_HORIZON_MANIFEST.json`
- `STAGE1_ACTION_COVERAGE.csv`
- `FULL_HORIZON_M111_REPLAY.csv`
- `TRAINING_DATASET_PROPOSAL.md`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p21_summary.json").write_text(
        json.dumps(
            {
                "n_packaged": n_abc,
                "n_lineages": len(lineages),
                "n_recovery_plus_precision": n_rec_prec,
                "m111_rate_on_cont_ok": m111_rate,
                "by_family": {k: dict(v) for k, v in by_fam.items()},
                "suitable": suitable,
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P21 done ===", f"packaged={n_abc}", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
