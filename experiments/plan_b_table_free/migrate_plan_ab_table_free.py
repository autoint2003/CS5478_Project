#!/usr/bin/env python
"""Migrate Plan A/B recovery roots from elevated-table captures onto GROUND_SCENE.

Does not change recovery episode boundaries, regenerate Plan C, or invent pickup
trajectories. Validates each unified A/B expert by identity restore + legal tape
replay + SUCCESS_NEW (3/3 for accepted).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import resolve_stored_path  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched, terminal_reason  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.cross_scene_restore import airborne_root_ok  # noqa: E402
from envs.simulation.ground_scene import SCENE_ID as GROUND_SCENE_ID  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration"
MANIFEST = ROOT / "data" / "unified_experts" / "manifest.jsonl"


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


def load_ab_rows():
    rows = []
    with MANIFEST.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("plan_family") in ("A", "B"):
                rows.append(r)
    return rows


def ref_from_flat(sim, snap: dict) -> dict:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    return {
        "R_hand": Rh,
        "r_des": np.asarray(snap["r_des"], float).reshape(3, 3).copy(),
        "p_nom": np.asarray(snap.get("prel_ref", ph), float).reshape(-1)[:3].copy(),
        "R_rel": np.asarray(snap["r_rel_ref"], float).reshape(3, 3).copy(),
        "t": float(snap.get("time", sim.data.time)),
    }


def load_actions(ref: dict) -> np.ndarray:
    path = ref.get("continuous_npz")
    if not path:
        return np.zeros((0, 7), np.float32)
    z = np.load(resolve_stored_path(path))
    if "actions" not in z.files:
        return np.zeros((0, 7), np.float32)
    return np.asarray(z["actions"], np.float32)


def restore_root(env, row: dict) -> tuple[dict, dict]:
    """Restore physical IC into GROUND_SCENE. Returns (blob_meta, airborne)."""
    snap_path = resolve_stored_path(row["initial_physical_state_ref"]["snapshot"])
    blob = pickle.loads(snap_path.read_bytes())
    if "capture" in blob:
        restore_capture(env, blob["capture"])
        ref = blob["ref"]
        kind = "capture"
    else:
        restore_ballistic(env.sim, blob)
        from envs.outcomes.support_drop import restore_state as restore_drop
        from envs.outcomes.support_drop import arm_recovery_drop

        restore_drop(env.sim, None)
        arm_recovery_drop(env.sim)
        mujoco.mj_forward(env.sim.model, env.sim.data)
        ref = ref_from_flat(env.sim, blob)
        kind = "flat_snap"
        env.switch = GeometrySwitch()
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        env._obs_seq = []
        env._prev_seq = []
    air = airborne_root_ok(env.sim)
    table = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    ground = mujoco.mj_name2id(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    meta = {
        "restore_kind": kind,
        "scene_id": str(getattr(env.sim, "scene_id", "") or ""),
        "table_geom": int(table),
        "ground_geom": int(ground),
        "ref": ref,
    }
    return meta, air


def replay_once(env, row: dict, ref: dict, actions: np.ndarray) -> dict:
    lost = 0
    t0 = float(env.sim.data.time)
    fail_reason = None
    for i, action in enumerate(actions):
        step_legal(env.sim, action)
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            fail_reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            return {
                "SUCCESS_NEW": 0,
                "success_12s": 0,
                "reason": fail_reason,
                "fail_step": i,
                "recovery_elapsed": float(env.sim.data.time) - t0,
            }
    hand = evaluate_handoff_from_state(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": int(hand["SUCCESS_NEW"]),
        "success_12s": int(hand["reason"] == "SUCCESS_12S"),
        "reason": hand["reason"],
        "fail_step": None,
        "recovery_elapsed": float(env.sim.data.time) - t0,
        "pre_pos_mm": hand.get("pre_pos_mm"),
        "pre_theta": hand.get("pre_theta"),
        "window_hold": bool(hand.get("SUCCESS_NEW")),
    }


def validate_row(row: dict, n_replay: int = 3) -> dict:
    env = make_env()
    actions = load_actions(row["initial_physical_state_ref"])
    meta, air = restore_root(env, row)
    out = {
        "expert_id": row["expert_id"],
        "plan_family": row["plan_family"],
        "physical_root_id": row["physical_root_id"],
        "mechanism": row.get("contact_grasp_mechanism"),
        "n_actions": int(actions.shape[0]),
        "restore_kind": meta["restore_kind"],
        "scene_id": meta["scene_id"],
        "table_absent": int(meta["table_geom"] < 0),
        "ground_present": int(meta["ground_geom"] >= 0),
        "airborne": air,
        "identical_layout": bool(env.sim.model.nq == 23),
    }
    if not air["airborne"]:
        out["status"] = "FAIL_NOT_AIRBORNE"
        out["exact_replay"] = {"n": 0, "n_pass": 0, "rows": []}
        return out
    if meta["scene_id"] != GROUND_SCENE_ID or meta["table_geom"] >= 0:
        out["status"] = "FAIL_SCENE"
        out["exact_replay"] = {"n": 0, "n_pass": 0, "rows": []}
        return out

    rows = []
    for i in range(int(n_replay)):
        env = make_env()
        meta_i, _ = restore_root(env, row)
        one = replay_once(env, row, meta_i["ref"], actions)
        one["replay_i"] = i
        rows.append(one)
    n_ok = sum(int(r["SUCCESS_NEW"]) for r in rows)
    n_s12 = sum(int(r.get("success_12s") or 0) for r in rows)
    out["exact_replay"] = {"n": n_replay, "n_pass": n_ok, "n_success_12s": n_s12, "rows": rows}
    if n_ok == n_replay:
        out["status"] = "PASS"
    elif n_s12 == n_replay and n_ok == 0:
        out["status"] = "FAIL_SUCCESS_12S_NOT_NEW"
        out["root_cause"] = "SUCCESS_12S_WITHOUT_SUCCESS_NEW_HOLD"
        out["needs_renewed_oracle"] = False
    elif n_ok == 0:
        out["status"] = "FAIL_REPLAY"
        out["root_cause"] = rows[0].get("reason")
        out["needs_renewed_oracle"] = True
    else:
        out["status"] = "FAIL_PARTIAL"
        out["root_cause"] = Counter(r.get("reason") for r in rows).most_common(1)[0][0]
        out["needs_renewed_oracle"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="optional cap for smoke tests")
    ap.add_argument("--plans", default="A,B")
    ap.add_argument("--n-replay", type=int, default=3)
    ap.add_argument("--expert-id", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plans = {p.strip() for p in args.plans.split(",") if p.strip()}
    rows = [r for r in load_ab_rows() if r["plan_family"] in plans]
    if args.expert_id:
        rows = [r for r in rows if r["expert_id"] == args.expert_id]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Migrating/validating {len(rows)} Plan A/B roots on GROUND_SCENE", flush=True)
    results = []
    t_wall0 = time.time()
    for i, row in enumerate(rows):
        t0 = time.time()
        try:
            out = validate_row(row, n_replay=args.n_replay)
        except Exception as exc:
            out = {
                "expert_id": row["expert_id"],
                "plan_family": row["plan_family"],
                "physical_root_id": row["physical_root_id"],
                "status": "ERROR",
                "root_cause": f"{type(exc).__name__}: {exc}",
                "exact_replay": {"n": 0, "n_pass": 0, "rows": []},
            }
        out["wall_s"] = round(time.time() - t0, 3)
        results.append(out)
        print(
            f"[{i+1}/{len(rows)}] {out['expert_id']} {out['status']} "
            f"{out.get('exact_replay', {}).get('n_pass')}/{out.get('exact_replay', {}).get('n')} "
            f"{out.get('root_cause') or out.get('exact_replay', {}).get('rows', [{}])[0].get('reason', '')} "
            f"{out['wall_s']}s",
            flush=True,
        )

    by_status = Counter(r["status"] for r in results)
    by_plan = {
        p: Counter(r["status"] for r in results if r["plan_family"] == p) for p in sorted(plans)
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(results),
        "by_status": dict(by_status),
        "by_plan": {k: dict(v) for k, v in by_plan.items()},
        "wall_s": round(time.time() - t_wall0, 2),
        "n_replay": args.n_replay,
        "scene": GROUND_SCENE_ID,
        "pass_ids": [r["expert_id"] for r in results if r["status"] == "PASS"],
        "fail_ids": [r["expert_id"] for r in results if r["status"] != "PASS"],
    }
    (OUT / "migration_results.jsonl").write_text(
        "\n".join(json.dumps(jsafe(r), ensure_ascii=False) for r in results) + ("\n" if results else ""),
        encoding="utf-8",
    )
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    lines = [
        "# Plan A/B table-free migration",
        "",
        f"Scene: `{GROUND_SCENE_ID}` (no table). Recovery episode boundaries unchanged.",
        f"Validated `{len(results)}` roots with `{args.n_replay}/` exact SUCCESS_NEW replay.",
        "",
        "## Status counts",
        "",
        "```",
        json.dumps(summary["by_status"], indent=2),
        "```",
        "",
        "## By plan",
        "",
        "```",
        json.dumps(summary["by_plan"], indent=2),
        "```",
        "",
        f"Wall time: {summary['wall_s']} s",
        "",
        "## Failures",
        "",
    ]
    fails = [r for r in results if r["status"] != "PASS"]
    if not fails:
        lines.append("None.")
    else:
        for r in fails:
            cause = r.get("root_cause") or (
                (r.get("exact_replay") or {}).get("rows") or [{}]
            )[0].get("reason")
            air = r.get("airborne") or {}
            lines.append(
                f"- `{r['expert_id']}` `{r['status']}` cause=`{cause}` "
                f"airborne={air.get('airborne')} z={air.get('obj_z')}"
            )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["by_status"], indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0 if by_status.get("PASS", 0) == len(results) and len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
