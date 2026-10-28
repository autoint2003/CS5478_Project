#!/usr/bin/env python
"""Bounded P2 revised pilot: A2/B2 volume + optional C2 + synthetic D2 coverage.

Does not merge into experts_v3. Does not change Stage-1/Precision laws.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.expert_pipeline.live_stage1 import (  # noqa: E402
    generate_a2_from_raw,
    generate_b2_from_raw,
    generate_c2_from_stage1_exit_sim,
    generate_d2_from_synthetic_ic,
)
from data_generation.oracles.plan_b.branch_compare import load_vocab, regrasp_candidates  # noqa: E402
from data_generation.oracles.plan_c.expert_search import run_stage1_catch_and_lift  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.precision_required_pilot_v2.d2_synthetic import (  # noqa: E402
    D2_RECIPES,
    construct_d2_ic,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

OUT = ROOT / "results/data_generation/precision_required_pilot_v2"
TRAJ = OUT / "trajectories"
SEED = 47

# Attempt budgets (pilot; not success quotas). Wall-clock bounded; report actuals.
BUDGET_A = 20
BUDGET_B = 15
BUDGET_C = 6
BUDGET_D = 16


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _rng():
    return np.random.default_rng(SEED)


def discover_plan_a_raw(rng: np.random.Generator) -> list[dict]:
    """Unified Plan A roots with raw snapshot + continuous Stage-1 actions."""
    man = ROOT / "data/unified_experts/manifest.jsonl"
    rows = []
    if not man.is_file():
        return rows
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("plan_family", "")).upper() != "A":
            continue
        ref = row.get("initial_physical_state_ref") or {}
        snap_rel = ref.get("snapshot")
        npz_rel = ref.get("continuous_npz")
        root_id = str(row.get("physical_root_id") or row.get("expert_id"))
        if not snap_rel:
            continue
        snap_path = ROOT / str(snap_rel)
        if not snap_path.is_file():
            alts = [
                ROOT / "data/snapshots/train" / f"{root_id}.pkl",
                ROOT / "data/snapshots" / f"{root_id}.pkl",
            ]
            snap_path = next((p for p in alts if p.is_file()), None)
        if snap_path is None or not npz_rel:
            continue
        npz_path = ROOT / str(npz_rel)
        if not npz_path.is_file():
            continue
        rows.append(
            {
                "raw_id": f"A_RAW__{root_id}",
                "root_id": root_id,
                "snap_path": snap_path,
                "npz_path": npz_path,
                "expert_id": row.get("expert_id"),
                "mechanism": row.get("contact_grasp_mechanism"),
            }
        )
    rng.shuffle(rows)
    return rows


def discover_plan_b_raw(rng: np.random.Generator) -> list[dict]:
    hard = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
    rows = []
    if hard.is_dir():
        for p in sorted(hard.glob("hard_*.pkl")):
            rows.append({"raw_id": f"B_RAW__{p.stem}", "path": p, "kind": "hard_ic"})
    # also stage2 hard roots if present
    v2 = ROOT / "results/data_generation/stage2_abc_final_v2/ics"
    if v2.is_dir():
        for p in sorted(v2.glob("HARD_IC__*.pkl"))[:40]:
            rows.append({"raw_id": f"B_RAW__{p.stem}", "path": p, "kind": "stage2_hard"})
    rng.shuffle(rows)
    return rows


def discover_plan_c_packs(rng: np.random.Generator) -> list[dict]:
    acc = ROOT / "results/diagnostics/ground_plan_c_scale50/raw/accepted"
    rows = []
    if not acc.is_dir():
        return rows
    for p in sorted(acc.glob("*__SUCCESS_NEW_expert.pkl")):
        rows.append({"raw_id": f"C_RAW__{p.stem}", "path": p})
    rng.shuffle(rows)
    return rows


def discover_d2_bases(rng: np.random.Generator) -> list[dict]:
    """Stable bilateral bases for synthetic perturbation (prefer A1 terminals)."""
    bases = []
    for folder in (
        ROOT / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF",
        ROOT / "results/data_generation/plan_a_stage1_terminals_v1/ics",
    ):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.pkl")):
            bases.append({"path": p, "sample_id": p.stem})
    rng.shuffle(bases)
    return bases


def b1_token_candidates(rng: np.random.Generator, n: int = 6) -> list[tuple[list[int], int]]:
    """Short open prefix + regrasp bodies (Plan B Stage-1)."""
    prefixes = [
        [36, 36, 36],
        [44, 44],
        [30, 30, 30, 30],
        [36, 36, 0, 0],
    ]
    bodies = regrasp_candidates(rng, n=max(8, n * 2))
    out = []
    for pref in prefixes:
        for body in bodies[: max(2, n // len(prefixes) + 1)]:
            toks = list(pref) + list(body)
            out.append((toks, len(pref)))
    rng.shuffle(out)
    return out[:n]


def append_csv(path: Path, row: dict, fieldnames: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fieldnames or sorted(row.keys())
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in keys})


def save_traj(row, exit_blob: dict | None, tag: str):
    TRAJ.mkdir(parents=True, exist_ok=True)
    sid = row.raw_scenario_id or tag
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)[:120]
    path = TRAJ / f"{row.branch_target}__{safe}.pkl"
    payload = {
        "row": row.to_row(),
        "stage1_actions": row.stage1_actions,
        "precision_actions": row.precision_actions,
        "provenance": row.provenance,
        "exit_blob": exit_blob,
        "synthetic_initial_condition": row.synthetic_initial_condition,
        "source_type": row.source_type,
    }
    path.write_bytes(pickle.dumps(payload))
    return path


def run_a_pilot(env, rng, cfg, funnel: list, accepted: list, rejected: list):
    raws = discover_plan_a_raw(rng)
    print(f"[A] raw scenarios available={len(raws)} budget={BUDGET_A}", flush=True)
    n = 0
    for src in raws:
        if n >= BUDGET_A:
            break
        n += 1
        t0 = time.perf_counter()
        try:
            blob = pickle.loads(src["snap_path"].read_bytes())
            if "capture" not in blob or "ref" not in blob:
                raise KeyError("missing capture/ref")
            arch = np.load(src["npz_path"])
            if "actions" not in arch.files:
                funnel.append(
                    {
                        "branch": "A",
                        "raw_id": src["raw_id"],
                        "attempt": n,
                        "stage": "load",
                        "ok": 0,
                        "reason": "NO_ACTIONS_IN_NPZ",
                        "wall_s": time.perf_counter() - t0,
                    }
                )
                continue
            actions = arch["actions"]
            print(f"[A {n}/{BUDGET_A}] {src['raw_id']} actions={len(actions)}", flush=True)
            row = generate_a2_from_raw(
                env,
                blob,
                actions,
                raw_scenario_id=src["raw_id"],
                precision_config=cfg,
                provenance={
                    "snap": str(src["snap_path"].relative_to(ROOT)).replace("\\", "/"),
                    "npz": str(src["npz_path"].relative_to(ROOT)).replace("\\", "/"),
                    "mechanism": src.get("mechanism"),
                },
            )
            funnel.append(
                {
                    "branch": "A",
                    "raw_id": src["raw_id"],
                    "attempt": n,
                    "stage": "full",
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": "|".join(row.precision_skills or []),
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                }
            )
            if row.accepted:
                p = save_traj(row, None, src["raw_id"])
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())
        except Exception as exc:
            funnel.append(
                {
                    "branch": "A",
                    "raw_id": src["raw_id"],
                    "attempt": n,
                    "stage": "exception",
                    "ok": 0,
                    "reason": f"{type(exc).__name__}:{exc}",
                    "wall_s": time.perf_counter() - t0,
                }
            )
            print(f"  EXC {exc}", flush=True)


def run_b_pilot(env, rng, cfg, funnel: list, accepted: list, rejected: list):
    raws = discover_plan_b_raw(rng)
    table, book48 = load_vocab()
    token_sets = b1_token_candidates(rng, n=5)
    print(f"[B] raw={len(raws)} token_sets={len(token_sets)} budget={BUDGET_B}", flush=True)
    n = 0
    for src in raws:
        if n >= BUDGET_B:
            break
        try:
            blob = pickle.loads(src["path"].read_bytes())
            if "capture" not in blob or "ref" not in blob:
                continue
        except Exception:
            continue
        for toks, pref_len in token_sets:
            if n >= BUDGET_B:
                break
            n += 1
            rid = f"{src['raw_id']}__tok{n}"
            print(f"[B {n}/{BUDGET_B}] {rid} n_tok={len(toks)}", flush=True)
            row = generate_b2_from_raw(
                env,
                blob,
                toks,
                raw_scenario_id=rid,
                prefix_len=pref_len,
                precision_config=cfg,
                provenance={"raw_path": str(src["path"]).replace("\\", "/"), "kind": src.get("kind")},
                table=table,
                book48=book48,
            )
            funnel.append(
                {
                    "branch": "B",
                    "raw_id": rid,
                    "attempt": n,
                    "stage": "full",
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": "|".join(row.precision_skills or []),
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                }
            )
            if row.accepted:
                p = save_traj(row, None, rid)
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
                break  # next raw IC
            rejected.append(row.to_row())
            # if Stage-1 failed chronology, try next token set; if B1, skip remaining tokens for this raw
            if row.reject_reason and "DIRECT_ELIGIBLE" in str(row.reject_reason):
                break


def run_c_pilot(env, rng, cfg, funnel: list, accepted: list, rejected: list):
    packs = discover_plan_c_packs(rng)
    print(f"[C] packs={len(packs)} budget={BUDGET_C} (optional)", flush=True)
    n = 0
    for src in packs:
        if n >= BUDGET_C:
            break
        n += 1
        t0 = time.perf_counter()
        try:
            pack = pickle.loads(src["path"].read_bytes())
            snap = pack.get("catch_snap") or pack.get("snap") or pack.get("root_snap")
            ref = pack.get("ref")
            raw_acts = pack.get("capture_actions")
            if raw_acts is None:
                raw_acts = pack.get("actions")
            if raw_acts is None:
                raw_acts = []
            catch_out = {
                "capture_actions": [np.asarray(a, np.float32) for a in list(raw_acts)],
                "mechanism": pack.get("mechanism"),
                "params": pack.get("params") or {},
                "cand": pack.get("cand") or {},
            }
            if snap is None or ref is None or len(catch_out["capture_actions"]) == 0:
                funnel.append(
                    {
                        "branch": "C",
                        "raw_id": src["raw_id"],
                        "attempt": n,
                        "ok": 0,
                        "reason": "MISSING_CATCH_FIELDS",
                        "wall_s": time.perf_counter() - t0,
                    }
                )
                continue
            print(f"[C {n}/{BUDGET_C}] {src['raw_id']}", flush=True)
            s1 = run_stage1_catch_and_lift(env.sim, snap, ref, catch_out)
            if not s1.get("ok") or s1.get("pose_correct_called"):
                funnel.append(
                    {
                        "branch": "C",
                        "raw_id": src["raw_id"],
                        "attempt": n,
                        "ok": 0,
                        "reason": s1.get("reason") or "STAGE1_FAIL",
                        "pose_correct_called": s1.get("pose_correct_called"),
                        "wall_s": time.perf_counter() - t0,
                    }
                )
                rejected.append(
                    {
                        "branch_target": "REJECT",
                        "plan_id": "C",
                        "raw_scenario_id": src["raw_id"],
                        "accepted": False,
                        "reject_reason": s1.get("reason"),
                    }
                )
                continue
            # Convert elev ballistic snap → ground capture via restore + take_capture
            restore_ballistic(env.sim, s1["elev_snap"])
            restore_drop(env.sim, None)
            arm(env.sim)
            exit_cap = take_capture(env)
            row = generate_c2_from_stage1_exit_sim(
                env,
                stage1_actions=list(s1["stage1_actions"]),
                exit_capture=exit_cap,
                ref=ref,
                raw_scenario_id=src["raw_id"],
                precision_config=cfg,
                provenance={"pack": str(src["path"]).replace("\\", "/"), "pose_correct_called": False},
                pose_correct_called=False,
            )
            funnel.append(
                {
                    "branch": "C",
                    "raw_id": src["raw_id"],
                    "attempt": n,
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": "|".join(row.precision_skills or []),
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                }
            )
            if row.accepted:
                p = save_traj(row, {"capture": exit_cap, "ref": ref}, src["raw_id"])
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())
        except Exception as exc:
            funnel.append(
                {
                    "branch": "C",
                    "raw_id": src["raw_id"],
                    "attempt": n,
                    "ok": 0,
                    "reason": f"{type(exc).__name__}:{exc}",
                    "wall_s": time.perf_counter() - t0,
                }
            )
            print(f"  EXC {exc}", flush=True)


def run_d_pilot(env, rng, cfg, funnel: list, accepted: list, rejected: list, coverage: list):
    bases = discover_d2_bases(rng)
    print(f"[D] bases={len(bases)} recipes={len(D2_RECIPES)} budget={BUDGET_D}", flush=True)
    if not bases:
        return
    n = 0
    for recipe in D2_RECIPES:
        if n >= BUDGET_D:
            break
        for base in bases[:3]:
            if n >= BUDGET_D:
                break
            n += 1
            base_blob = pickle.loads(base["path"].read_bytes())
            base_blob["sample_id"] = base["sample_id"]
            rid = f"D2__{recipe['target_skill']}__{base['sample_id']}__{n}"
            print(f"[D {n}/{BUDGET_D}] target={recipe['target_skill']} base={base['sample_id']}", flush=True)
            built = construct_d2_ic(env, base_blob, **{k: recipe[k] for k in recipe})
            if not built.get("ok"):
                funnel.append(
                    {
                        "branch": "D",
                        "raw_id": rid,
                        "attempt": n,
                        "ok": 0,
                        "reason": built.get("reason"),
                        "target_skill": recipe["target_skill"],
                    }
                )
                coverage.append(
                    {
                        "target_skill": recipe["target_skill"],
                        "status": "construction_failed",
                        "reason": built.get("reason"),
                        "raw_id": rid,
                    }
                )
                rejected.append(
                    {
                        "branch_target": "REJECT",
                        "plan_id": "D",
                        "raw_scenario_id": rid,
                        "accepted": False,
                        "reject_reason": built.get("reason"),
                        "synthetic_initial_condition": True,
                    }
                )
                continue
            row = generate_d2_from_synthetic_ic(
                env,
                {"capture": built["capture"], "ref": built["ref"], "sample_id": rid},
                raw_scenario_id=rid,
                construction_meta=built["construction"],
                precision_config=cfg,
            )
            executed = "|".join(row.precision_skills or [])
            targeted = recipe["target_skill"]
            if not row.stage1_success_12s:
                status = "targeted_but_unreachable_12s"
            elif not row.precision_required:
                status = "targeted_but_not_precision_required"
            elif not row.accepted:
                status = "executed_but_failed"
            else:
                # Did we actually hit the targeted family?
                hit = False
                t = targeted.replace("+", " ")
                for part in targeted.split("+"):
                    part = part.strip()
                    if part == "support_transfer" and "support_transfer" in executed:
                        hit = True
                    elif part.startswith("dz") and ("dz" in executed):
                        hit = True
                    elif part == "dx" and ("dx" in executed or "support_transfer" in executed):
                        hit = True
                    elif part == "dtheta" and "dtheta" in executed:
                        hit = True
                status = "executed_and_successful" if hit else "executed_success_other_skills"
            coverage.append(
                {
                    "target_skill": targeted,
                    "executed_skills": executed,
                    "status": status,
                    "SUCCESS_NEW": row.SUCCESS_NEW,
                    "precision_status": row.precision_status,
                    "raw_id": rid,
                    "capture_fp": row.capture_fp,
                    "entry_dtheta": (row.entry_errors or {}).get("dtheta_deg"),
                    "entry_dx": (row.entry_errors or {}).get("dx_mm"),
                    "entry_dz": (row.entry_errors or {}).get("dz_mm"),
                }
            )
            funnel.append(
                {
                    "branch": "D",
                    "raw_id": rid,
                    "attempt": n,
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": executed,
                    "target_skill": targeted,
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                }
            )
            if row.accepted:
                p = save_traj(row, {"capture": built["capture"], "ref": built["ref"]}, rid)
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())


def _write_table(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    env = make_env()
    rng = _rng()
    cfg = PrecisionConfig(use_verified_lateral_fallback=False)
    funnel: list[dict] = []
    accepted: list[dict] = []
    rejected: list[dict] = []
    coverage: list[dict] = []

    t_all = time.perf_counter()
    run_a_pilot(env, rng, cfg, funnel, accepted, rejected)
    run_b_pilot(env, rng, cfg, funnel, accepted, rejected)
    run_c_pilot(env, rng, cfg, funnel, accepted, rejected)
    run_d_pilot(env, rng, cfg, funnel, accepted, rejected, coverage)

    _write_table(OUT / "A2_B2_GENERATION_FUNNEL.csv", [r for r in funnel if r.get("branch") in ("A", "B")])
    _write_table(OUT / "ACCEPTED_TRAJECTORIES.csv", accepted)
    _write_table(OUT / "REJECTED_CANDIDATES.csv", rejected)
    _write_table(OUT / "D2_SKILL_COVERAGE_MATRIX.csv", coverage)
    _write_table(OUT / "FULL_FUNNEL.csv", funnel)

    # C2 optional results
    c_rows = [r for r in funnel if r.get("branch") == "C"]
    c_acc = [r for r in accepted if str(r.get("branch_target")) == "C2"]
    (OUT / "C2_OPTIONAL_RESULTS.md").write_text(
        "\n".join(
            [
                "# C2_OPTIONAL_RESULTS",
                "",
                f"Attempts: {len(c_rows)}",
                f"Accepted C2: {len(c_acc)}",
                f"Stage-1 12S among attempts: {sum(1 for r in c_rows if r.get('stage1_12s'))}",
                f"Precision-required: {sum(1 for r in c_rows if r.get('precision_required'))}",
                "",
                "C2 is optional; low yield does not block the dataset.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "budgets": {"A": BUDGET_A, "B": BUDGET_B, "C": BUDGET_C, "D": BUDGET_D},
        "wall_s_total": round(time.perf_counter() - t_all, 2),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "accepted_by_branch": {},
        "funnel_attempts": {},
    }
    for r in accepted:
        b = r.get("branch_target") or "?"
        summary["accepted_by_branch"][b] = summary["accepted_by_branch"].get(b, 0) + 1
    for r in funnel:
        b = r.get("branch") or "?"
        summary["funnel_attempts"][b] = summary["funnel_attempts"].get(b, 0) + 1
    (OUT / "PILOT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
