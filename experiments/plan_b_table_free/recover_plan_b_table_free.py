#!/usr/bin/env python
"""Recover failed Plan B roots on GROUND_SCENE with legal 7D search.

Does not modify accepted A/B/C experts. Preserves physical ICs. Re-searches
continuous actions where old tapes are invalid without the elevated table.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import AIRBORNE_V3, TAU_OPEN, TAU_SECURE, normalize_action  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.datasets.recovery_rebuild import resolve_stored_path  # noqa: E402
from data_generation.oracles.plan_a.contact_geometry import capture_geometry  # noqa: E402
from data_generation.oracles.plan_b.release_recatch import _class_force, contacts_now  # noqa: E402
from data_generation.oracles.plan_b.vertical_recatch import (  # noqa: E402
    _bottom,
    _command,
    _sample,
    project,
    specs as vertical_specs,
)
from data_generation.oracles.representability import representability  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import arm_recovery_drop, latched, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.success_criteria import WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from evaluation.handoff_from_state import evaluate_handoff_from_state  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

MIG = ROOT / "results" / "diagnostics" / "plan_ab_table_free_migration"
BAK = MIG / "pre_update_unified_manifest.jsonl"
OUT = ROOT / "results" / "diagnostics" / "plan_b_table_free_recovery"
RAW = OUT / "raw" / "accepted"
DT = 0.02
LIMIT = 4.0


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
    env._hist = ObservableObsState(DT)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def load_failed_b(status_filter=None):
    uref = {}
    with BAK.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("plan_family") == "B":
                uref[r["expert_id"]] = r
    mig = []
    with (MIG / "migration_results.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("plan_family") != "B":
                continue
            if r.get("status") == "PASS":
                continue
            if status_filter and r.get("status") not in status_filter:
                continue
            if r["expert_id"] in uref:
                r = {**r, "uref": uref[r["expert_id"]]}
                mig.append(r)
    return mig


def restore_ic(env, uref_row):
    ref = uref_row["initial_physical_state_ref"]
    blob = pickle.loads(resolve_stored_path(ref["snapshot"]).read_bytes())
    if "capture" in blob:
        restore_capture(env, blob["capture"])
        return blob["ref"], "capture", blob
    restore_ballistic(env.sim, blob)
    restore_drop(env.sim, None)
    arm_recovery_drop(env.sim)
    Rh = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    grasp_ref = {
        "R_hand": Rh,
        "r_des": np.asarray(blob["r_des"], float).reshape(3, 3).copy(),
        "p_nom": np.asarray(blob.get("prel_ref", ph), float).reshape(-1)[:3].copy(),
        "R_rel": np.asarray(blob["r_rel_ref"], float).reshape(3, 3).copy(),
        "t": float(blob.get("time", env.sim.data.time)),
    }
    return grasp_ref, "flat_snap", blob


def rollout_ground_recatch(env, grasp_ref, book, spec):
    """Vertical open-chase-close-brake without elevated TABLE_TOP abort."""
    sim = env.sim
    t0 = float(sim.data.time)
    hist = ObservableObsState(DT)
    state = {"mode": "open", "released": False, "pad_run": 0, "sim": sim, "v": np.zeros(3)}
    actions = []
    observations = []
    rows = []
    while float(sim.data.time) - t0 < LIMIT - 1e-9 and state["mode"] not in ("handoff", "fail"):
        sample = _sample(sim, t0)
        # GROUND_SCENE: irreversible support latch only (no virtual table height kill).
        if latched(sim):
            state["mode"] = "ground_drop"
            rows.append(sample)
            break
        if sample["bottom"] < 0.06:
            state["mode"] = "too_low"
            rows.append(sample)
            break
        velocity, tau = _command(sample, state, spec)
        sample["mode"] = state["mode"]
        raw = normalize_action(velocity, np.zeros(3), tau, AIRBORNE_V3)
        action, linf = project(raw, book)
        observations.append(
            np.asarray(observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist), np.float32)
        )
        step_legal(sim, action)
        actions.append(action)
        rows.append(sample)
    return {
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 7), np.float32),
        "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 39), np.float32),
        "mode": state["mode"],
        "rows": rows,
        "n_steps": len(actions),
    }


def short_hold_corrections():
    """Minimal legal hold / micro-adjust sequences after a successful tape."""
    hold = np.array([0, 0, 0, 0, 0, 0, 1.0], np.float32)
    open_slight = np.array([0, 0, 0, 0, 0, 0, 0.5], np.float32)
    down = np.array([0, 0, 0.15, 0, 0, 0, 1.0], np.float32)
    up = np.array([0, 0, -0.15, 0, 0, 0, 1.0], np.float32)
    seqs = []
    for n in (5, 10, 20, 40):
        seqs.append(("hold_%d" % n, np.tile(hold, (n, 1))))
    for n in (5, 10):
        seqs.append(("down_hold_%d" % n, np.vstack([np.tile(down, (3, 1)), np.tile(hold, (n, 1))])))
        seqs.append(("up_hold_%d" % n, np.vstack([np.tile(up, (3, 1)), np.tile(hold, (n, 1))])))
    seqs.append(("pulse_open", np.vstack([np.tile(open_slight, (2, 1)), np.tile(hold, (15, 1))])))
    return seqs


def replay_actions(env, grasp_ref, actions, uref_row):
    restore_ic(env, uref_row)
    lost = 0
    from envs.outcomes.horizon_outcome import recovery_failed

    for a in actions:
        step_legal(env.sim, a)
        failed, lost = recovery_failed(env.sim, grasp_ref, lost)
        if failed:
            return {"SUCCESS_NEW": 0, "reason": "DROP_DURING_RECOVERY", "lost": True}
    meta = evaluate_handoff_from_state(env.sim, grasp_ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
        "reason": meta["reason"],
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "pre_theta": meta.get("pre_theta"),
        "lost": False,
    }


def confirm_3x(uref_row, actions):
    from envs.outcomes.horizon_outcome import recovery_failed

    rows = []
    for i in range(3):
        env = make_env()
        grasp_ref, _, _ = restore_ic(env, uref_row)
        lost = 0
        failed_mid = False
        for a in actions:
            step_legal(env.sim, a)
            failed, lost = recovery_failed(env.sim, grasp_ref, lost)
            if failed:
                rows.append({"replay_i": i, "SUCCESS_NEW": 0, "reason": "DROP_DURING_RECOVERY"})
                failed_mid = True
                break
        if failed_mid:
            continue
        meta = evaluate_handoff_from_state(env.sim, grasp_ref, stabilize_n=0)
        rows.append(
            {
                "replay_i": i,
                "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
                "reason": meta["reason"],
                "pre_pos_mm": meta.get("pre_pos_mm"),
                "pre_theta": meta.get("pre_theta"),
            }
        )
    n_ok = sum(int(r["SUCCESS_NEW"]) for r in rows)
    return n_ok, rows


def try_s12_hold_fix(uref_row, book):
    """Replay original tape then append minimal legal hold corrections."""
    ref_path = uref_row["initial_physical_state_ref"]
    zpath = resolve_stored_path(ref_path["continuous_npz"])
    if not zpath.exists() or "actions" not in np.load(zpath).files:
        return None
    base = np.asarray(np.load(zpath)["actions"], np.float32)
    env = make_env()
    grasp_ref, kind, _ = restore_ic(env, uref_row)
    # baseline
    env = make_env()
    restore_ic(env, uref_row)
    base_out = replay_actions(env, grasp_ref, base, uref_row)
    if base_out.get("SUCCESS_NEW"):
        n_ok, rows = confirm_3x(uref_row, base)
        if n_ok == 3:
            return {"actions": base, "tag": "original_tape", "exact_replay": rows, "kind": kind}
    if base_out.get("reason") != "SUCCESS_12S" and not base_out.get("SUCCESS_NEW"):
        return None
    for tag, corr in short_hold_corrections():
        acts = np.vstack([base, corr])
        env = make_env()
        restore_ic(env, uref_row)
        out = replay_actions(env, grasp_ref, acts, uref_row)
        if out.get("SUCCESS_NEW"):
            n_ok, rows = confirm_3x(uref_row, acts)
            if n_ok == 3:
                return {"actions": acts, "tag": tag, "exact_replay": rows, "kind": kind}
    return None


def _spec_grid():
    grid = list(vertical_specs())
    for kp, kz, down, open_min, brake, name in (
        (12.0, 5.0, 1.0, 0.03, 9.0, "mid"),
        (14.0, 5.0, 0.9, 0.025, 10.0, "balanced"),
        (20.0, 7.0, 1.4, 0.05, 7.0, "aggressive"),
        (6.0, 2.5, 0.5, 0.015, 14.0, "soft"),
        (10.0, 4.0, 1.5, 0.02, 10.0, "fast_down"),
        (16.0, 6.0, 0.7, 0.04, 12.0, "slow_down"),
    ):
        spec = dict(vertical_specs()[0])
        spec.update(
            {"kp": kp, "kz": kz, "down": down, "open_min": open_min, "brake": brake, "name": name}
        )
        grid.append(spec)
    return grid


def try_oracle_search(uref_row, book):
    env = make_env()
    grasp_ref, kind, _ = restore_ic(env, uref_row)
    for spec in _spec_grid():
        env = make_env()
        restore_ic(env, uref_row)
        roll = rollout_ground_recatch(env, grasp_ref, book, spec)
        if roll["mode"] in ("ground_drop", "too_low", "fail") and roll["n_steps"] < 8:
            continue
        if roll["mode"] not in ("handoff", "settle"):
            # still try handoff if bilateral pads present at end
            from evaluation.success_criteria import contact_summary

            cs = contact_summary(env.sim)
            if not cs["bilateral"]:
                continue
        meta = evaluate_handoff_from_state(env.sim, grasp_ref, stabilize_n=0)
        if not meta["SUCCESS_NEW"]:
            continue
        n_ok, rows = confirm_3x(uref_row, roll["actions"])
        if n_ok == 3:
            return {
                "actions": roll["actions"],
                "tag": f"vertical_{spec['name']}",
                "mode": roll["mode"],
                "meta": meta,
                "exact_replay": rows,
                "kind": kind,
            }
    return None


def save_expert(eid, uref_row, found):
    RAW.mkdir(parents=True, exist_ok=True)
    env = make_env()
    grasp_ref, _, blob = restore_ic(env, uref_row)
    # rebuild obs tape
    restore_ic(env, uref_row)
    hist = ObservableObsState(DT)
    obs = []
    for a in found["actions"]:
        obs.append(np.asarray(observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist), np.float32))
        step_legal(env.sim, a)
    payload = {
        "expert_id": eid,
        "physical_root_id": uref_row["physical_root_id"],
        "plan_family": "B",
        "scene": "GROUND_SCENE",
        "actions": np.asarray(found["actions"], np.float32),
        "observations": np.stack(obs, axis=0) if obs else np.zeros((0, 39), np.float32),
        "ref": grasp_ref,
        "catch_snap": blob["capture"] if isinstance(blob, dict) and "capture" in blob else blob,
        "tag": found["tag"],
        "exact_replay": {"n": 3, "n_pass": 3, "rows": found["exact_replay"]},
        "SUCCESS_NEW": True,
        "source_snapshot": uref_row["initial_physical_state_ref"].get("snapshot"),
        "mechanism": uref_row.get("contact_grasp_mechanism"),
        "recovery_method": "table_free_research",
    }
    path = RAW / f"{eid}__SUCCESS_NEW_expert.pkl"
    path.write_bytes(pickle.dumps(payload))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--status", default="FAIL_REPLAY,FAIL_SUCCESS_12S_NOT_NEW")
    ap.add_argument("--expert-id", default="")
    ap.add_argument("--skip-oracle", action="store_true")
    ap.add_argument("--shard", default="", help="optional output suffix for parallel shards")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    statuses = [s.strip() for s in args.status.split(",") if s.strip()]
    rows = load_failed_b(statuses)
    if args.expert_id:
        rows = [r for r in rows if r["expert_id"] == args.expert_id]
    if args.offset > 0:
        rows = rows[args.offset :]
    if args.limit > 0:
        rows = rows[: args.limit]
    book = load_codebook()
    shard = args.shard.strip()
    tag = f"shard{shard}" if shard else "main"
    print(f"Recovering {len(rows)} Plan B failed roots ({tag})", flush=True)
    results = []
    t0 = time.time()
    for i, row in enumerate(rows):
        eid = row["expert_id"]
        uref = row["uref"]
        t = time.time()
        found = None
        path = None
        existing = RAW / f"{eid}__SUCCESS_NEW_expert.pkl"
        try:
            if existing.is_file():
                blob = pickle.loads(existing.read_bytes())
                if blob.get("SUCCESS_NEW") and (blob.get("exact_replay") or {}).get("n_pass") == 3:
                    status = "PASS"
                    path = existing
                    found = {"tag": blob.get("tag") or "cached", "exact_replay": blob.get("exact_replay")}
                else:
                    status = "FAIL"
            else:
                if row["status"] == "FAIL_SUCCESS_12S_NOT_NEW":
                    found = try_s12_hold_fix(uref, book)
                if found is None and not args.skip_oracle:
                    found = try_oracle_search(uref, book)
                if found is not None and found.get("exact_replay"):
                    path = save_expert(eid, uref, found)
                    status = "PASS"
                else:
                    status = "FAIL"
        except Exception as exc:
            status = "ERROR"
            found = {"error": f"{type(exc).__name__}: {exc}"}
        out = {
            "expert_id": eid,
            "mig_status": row["status"],
            "mechanism": row.get("mechanism"),
            "status": status,
            "tag": None if found is None else found.get("tag"),
            "path": None if path is None else str(Path(path).relative_to(ROOT)).replace("\\", "/"),
            "wall_s": round(time.time() - t, 3),
            "error": None if found is None else found.get("error"),
        }
        results.append(out)
        print(
            f"[{i+1}/{len(rows)}] {eid} {status} tag={out['tag']} {out['wall_s']}s",
            flush=True,
        )

    by = Counter(r["status"] for r in results)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(results),
        "by_status": dict(by),
        "by_mig_status": {
            s: dict(Counter(r["status"] for r in results if r["mig_status"] == s))
            for s in statuses
        },
        "wall_s": round(time.time() - t0, 2),
        "pass_ids": [r["expert_id"] for r in results if r["status"] == "PASS"],
        "shard": shard or None,
    }
    suffix = f"_{shard}" if shard else ""
    (OUT / f"recovery_results{suffix}.jsonl").write_text(
        "\n".join(json.dumps(jsafe(r), ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8",
    )
    (OUT / f"recovery_summary{suffix}.json").write_text(
        json.dumps(jsafe(summary), indent=2), encoding="utf-8"
    )
    if not shard:
        (OUT / "recovery_summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
        (OUT / "recovery_results.jsonl").write_text(
            "\n".join(json.dumps(jsafe(r), ensure_ascii=False) for r in results) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary["by_status"], indent=2), flush=True)
    print(f"PASS {len(summary['pass_ids'])} / {len(results)}", flush=True)
    return 0 if summary["pass_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
