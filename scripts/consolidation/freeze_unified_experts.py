#!/usr/bin/env python
"""Freeze Plan A/B/C experts into a versioned unified dataset schema.

Plan C: reconstruct obs/action tapes from golden continuous-action experts.
Plan A/B: import verified SUCCESS_NEW tokenized trajectories from existing
manifests where provenance exists; do not invent continuous actions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3  # noqa: E402
from envs.observations.airborne_obs import OBS_DIM_AIR, observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
# evaluate_handoff_from_state lives in the Plan C runner; import after path setup
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.full_objective import P_TGT, U_TGT  # noqa: E402
from evaluation.grasp_configuration import measure  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

SCHEMA_VERSION = "unified_experts_v1"
OUT = ROOT / "data" / "unified_experts"
TRACKED = ROOT / "datasets" / "unified_experts"
RECORDS = OUT / "records"
REPORT = ROOT / "results" / "consolidation" / "GOLDEN_REGRESSION.md"
DT = 0.02

PLAN_C_SRC = ROOT / "results" / "diagnostics" / "ground_plan_c_final_success" / "raw"
PLAN_C_FALLBACK = ROOT / "archive" / "golden_experts" / "ground_plan_c_final_success" / "raw"
PLAN_C_ROOTS = ["gbi_010", "gbi_022"]


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


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_plan_c_expert(root_id: str) -> dict:
    for base in (PLAN_C_SRC, PLAN_C_FALLBACK):
        p = base / f"{root_id}__SUCCESS_NEW_expert.pkl"
        if p.exists():
            return pickle.loads(p.read_bytes()), p
    raise FileNotFoundError(root_id)


def _obs(sim, hist) -> np.ndarray:
    obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
    obs = np.asarray(obs, np.float32)
    if obs.shape != (OBS_DIM_AIR,):
        raise RuntimeError(f"obs dim {obs.shape}, expected {(OBS_DIM_AIR,)}")
    return obs


def evaluate_handoff(sim, ref) -> dict:
    """Match Plan C SUCCESS_NEW evaluation path (immediate handoff)."""
    import importlib.util

    path = ROOT / "scripts" / "run_ground_plan_c_final_success.py"
    spec = importlib.util.spec_from_file_location("run_ground_plan_c_final_success", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.evaluate_handoff_from_state(sim, ref, stabilize_n=0)


def replay_plan_c(root_id: str, n_replay: int = 3) -> dict:
    expert, src_path = load_plan_c_expert(root_id)
    actions = np.asarray(expert["actions"], np.float32)
    snap = expert["catch_snap"]
    ref = expert["ref"]

    # Record obs/action alignment from one recording pass
    sim, _ = make_ground_sim()
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    hist = ObservableObsState(DT)
    obs_seq = []
    act_seq = []
    t0 = float(sim.data.time)
    for a in actions:
        obs_seq.append(_obs(sim, hist))
        a = np.asarray(a, np.float32)
        act_seq.append(a.copy())
        step_legal(sim, a)
        if latched(sim):
            break
    obs_arr = np.stack(obs_seq, axis=0)
    act_arr = np.stack(act_seq, axis=0)
    timestamps = (np.arange(len(act_arr), dtype=np.float64) * DT).tolist()

    # Independent replays
    replay_rows = []
    for i in range(n_replay):
        sim_r, _ = make_ground_sim()
        restore_ballistic(sim_r, snap)
        restore_drop(sim_r, None)
        arm(sim_r)
        lost = False
        for a in actions:
            step_legal(sim_r, np.asarray(a, np.float32))
            if latched(sim_r):
                lost = True
                break
        if lost:
            replay_rows.append({"replay_i": i, "SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF"})
            continue
        meta = evaluate_handoff(sim_r, ref)
        replay_rows.append({
            "replay_i": i,
            "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
            "reason": meta["reason"],
            "pre_pos_mm": meta.get("pre_pos_mm"),
            "pre_theta": meta.get("pre_theta"),
            "success_12s": int(meta.get("success_12s") or 0),
        })

    n_ok = sum(int(r["SUCCESS_NEW"]) for r in replay_rows)
    handoff_meta = expert.get("handoff_meta") or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "expert_id": f"plan_c__{root_id}",
        "plan_family": "C",
        "physical_root_id": root_id,
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
        "contact_grasp_mechanism": "FULL_DETACH_ACTIVE_RECATCH",
        "source_experiment": "results/diagnostics/ground_plan_c_final_success",
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


def _plan_family_from_row(obj: dict) -> str:
    fam = str(obj.get("family") or "").lower()
    strat = str(obj.get("strategy") or obj.get("skill") or "").lower()
    if fam in {"sliding"} or any(k in strat for k in ("slide", "inward", "controlled")):
        return "A"
    if fam in {"recatch", "vertical_recatch", "regrasp", "chase", "release"} or any(
        k in strat for k in ("recatch", "release", "chase", "horizontal", "follow", "later_close")
    ):
        return "B"
    if strat in {"immediate_handoff", "reference"} or fam in {"control"}:
        return "A"  # nominal-solvable / handoff controls grouped with Plan A metadata
    return "B" if "recatch" in fam else "A"


def import_tokenized_plan_ab() -> list[dict]:
    """Import verified Plan A/B rows from existing jsonl without inventing actions."""
    rows = []
    primary = ROOT / "data" / "supervised_targets_with_horizontal.jsonl"
    files = [primary] if primary.exists() else []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not obj.get("verified_success", False):
                    continue
                ic = obj.get("id") or obj.get("ic_id") or obj.get("episode_id")
                if ic is None:
                    continue
                token_npz = obj.get("token_residual_npz")
                continuous_npz = obj.get("continuous_npz")
                if not token_npz and not continuous_npz and "tokens" not in obj:
                    continue
                plan = _plan_family_from_row(obj)
                tags = str(obj.get("strategy") or obj.get("family") or "")
                # Physical root: prefer snapshot stem to avoid tape-level leakage in splits
                snap = obj.get("source_snapshot") or obj.get("snapshot") or ""
                root_id = Path(str(snap)).stem if snap else str(ic).split("__")[0]
                expert_id = f"plan_{plan.lower()}__{ic}"
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "expert_id": expert_id,
                    "plan_family": plan,
                    "physical_root_id": str(root_id),
                    "scene": "legacy_elevated",
                    "environment_version": "legacy_elevated_table",
                    "initial_physical_state_ref": {
                        "snapshot": snap,
                        "source_manifest": str(f.relative_to(ROOT)).replace("\\", "/"),
                        "token_residual_npz": token_npz,
                        "continuous_npz": continuous_npz,
                    },
                    "tokens_k49_ref": token_npz,
                    "actions_7d": None,
                    "observations_39d": None,
                    "control_period_s": DT,
                    "handoff": {"token_id": 47, "label": obj.get("handoff_label")},
                    "SUCCESS_12S": True,
                    "SUCCESS_NEW": True,
                    "exact_replay": {
                        "n": None,
                        "n_pass": None,
                        "note": (
                            "Provenance from verified_success supervised manifest; "
                            "npz tapes are the authoritative token/obs sequences."
                        ),
                    },
                    "contact_grasp_mechanism": tags or "legacy_verified",
                    "source_experiment": str(f.relative_to(ROOT)).replace("\\", "/"),
                    "provenance": {
                        "frozen_utc": datetime.now(timezone.utc).isoformat(),
                        "manifest_row": {
                            k: obj.get(k)
                            for k in (
                                "id", "strategy", "family", "mechanism",
                                "sample_weight", "nominal_class", "split",
                            )
                            if k in obj
                        },
                        "privileged_runtime": False,
                    },
                    "regression_fixture": False,
                    "data_integrity_checksum": sha256_bytes(line.encode("utf-8")),
                })
    seen = {}
    for r in rows:
        seen[r["expert_id"]] = r
    return list(seen.values())


def write_splits(records: list[dict]) -> dict:
    """Split by physical root; Plan C too small for meaningful holdout."""
    by_plan = {"A": [], "B": [], "C": []}
    for r in records:
        by_plan.setdefault(r["plan_family"], []).append(r)

    # Plan C: all as regression fixtures; no fake generalization split
    plan_c = by_plan.get("C", [])
    # Plan A/B: root-level 80/10/10 if enough roots
    def split_roots(items, seed=0):
        roots = sorted({r["physical_root_id"] for r in items})
        rng = np.random.default_rng(seed)
        rng.shuffle(roots)
        n = len(roots)
        if n < 5:
            return {"train": roots, "val": [], "test": [], "note": "too_few_roots_for_holdout"}
        n_test = max(1, int(round(0.1 * n)))
        n_val = max(1, int(round(0.1 * n)))
        test = roots[:n_test]
        val = roots[n_test:n_test + n_val]
        train = roots[n_test + n_val:]
        return {"train": train, "val": val, "test": test}

    splits = {
        "schema_version": SCHEMA_VERSION,
        "split_rule": "by_physical_root_id",
        "plan_C_limitation": (
            "Only two independent Plan C SUCCESS_NEW roots (gbi_010, gbi_022). "
            "Both are regression fixtures; they do not provide a statistically "
            "meaningful held-out generalization estimate."
        ),
        "plan_C": {
            "regression_fixtures": [r["physical_root_id"] for r in plan_c],
            "train": [],
            "val": [],
            "test": [],
        },
        "plan_A": split_roots(by_plan.get("A", []), seed=1),
        "plan_B": split_roots(by_plan.get("B", []), seed=2),
    }
    # Assign split labels
    def label_for(r):
        if r["plan_family"] == "C":
            return "regression_fixture"
        sp = splits[f"plan_{r['plan_family']}"]
        rid = r["physical_root_id"]
        for name in ("test", "val", "train"):
            if rid in sp.get(name, []):
                return name
        return "train"

    for r in records:
        r["split"] = label_for(r)
        r["regression_fixture"] = r["split"] == "regression_fixture"
    return splits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "consolidation").mkdir(parents=True, exist_ok=True)

    records = []
    regression_lines = [
        "# Golden Expert Regression (post-consolidation)",
        "",
        f"UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Plan C continuous-action replay (3/3 required)",
        "",
        "| Root | Passes | Details |",
        "|---|---:|---|",
    ]

    for rid in PLAN_C_ROOTS:
        print(f"Freezing Plan C {rid}...", flush=True)
        payload, raw, replay_rows = replay_plan_c(rid, n_replay=3)
        out_pkl = RECORDS / f"{payload['expert_id']}.pkl"
        out_pkl.write_bytes(raw)
        meta = {k: v for k, v in payload.items() if k not in ("actions_7d", "observations_39d")}
        meta["record_path"] = str(out_pkl.relative_to(ROOT)).replace("\\", "/")
        meta["n_steps"] = int(payload["actions_7d"].shape[0])
        meta["obs_dim"] = int(payload["observations_39d"].shape[1])
        meta["action_dim"] = int(payload["actions_7d"].shape[1])
        (RECORDS / f"{payload['expert_id']}.meta.json").write_text(
            json.dumps(jsafe(meta), indent=2), encoding="utf-8"
        )
        records.append(meta)
        n_ok = payload["exact_replay"]["n_pass"]
        regression_lines.append(
            f"| {rid} | {n_ok}/3 | {json.dumps(jsafe(replay_rows))} |"
        )
        print(f"  {rid}: {n_ok}/3 SUCCESS_NEW", flush=True)

    print("Importing Plan A/B verified manifests...", flush=True)
    ab = import_tokenized_plan_ab()
    for r in ab:
        records.append(r)
    print(f"  imported {len(ab)} Plan A/B rows", flush=True)

    splits = write_splits(records)
    index = {
        "schema_version": SCHEMA_VERSION,
        "n_experts": len(records),
        "by_plan": {
            p: sum(1 for r in records if r["plan_family"] == p) for p in ("A", "B", "C")
        },
        "experts": [
            {
                "expert_id": r["expert_id"],
                "plan_family": r["plan_family"],
                "physical_root_id": r["physical_root_id"],
                "split": r.get("split"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "checksum": r.get("data_integrity_checksum"),
            }
            for r in records
        ],
    }
    for dest in (OUT, TRACKED):
        (dest / "index.json").write_text(json.dumps(jsafe(index), indent=2), encoding="utf-8")
        (dest / "splits.json").write_text(json.dumps(jsafe(splits), indent=2), encoding="utf-8")
        with (dest / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(jsafe(r)) + "\n")

    regression_lines += [
        "",
        "## Plan A/B",
        "",
        f"Imported `{len(ab)}` verified SUCCESS_NEW / verified_success rows from existing manifests.",
        "These retain legacy elevated-table scene provenance. Continuous 7D tapes are used only when present;",
        "tokenized experts keep K49 token sequences without inventing continuous actions.",
        "",
        f"Tracked index: `datasets/unified_experts/index.json`",
        f"Heavy records (gitignored): `data/unified_experts/records/`",
        "",
    ]
    REPORT.write_text("\n".join(regression_lines), encoding="utf-8")
    print(f"Wrote {TRACKED / 'index.json'} and {REPORT}")


if __name__ == "__main__":
    main()
