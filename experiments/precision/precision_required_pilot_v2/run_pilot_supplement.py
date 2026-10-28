#!/usr/bin/env python
"""Supplemental pilot pass targeting A2/B2 yield and milder D2 recipes.

Appends to precision_required_pilot_v2 artifacts (does not overwrite prior funnel).
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
from data_generation.oracles.plan_b.branch_compare import load_vocab  # noqa: E402
from data_generation.oracles.plan_c.expert_search import run_stage1_catch_and_lift  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.stage1_exit import finalize_stage1_exit  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.precision_required_pilot_v2.d2_synthetic import construct_d2_ic  # noqa: E402
from experiments.precision.precision_required_pilot_v2.run_pilot import (  # noqa: E402
    OUT,
    TRAJ,
    discover_d2_bases,
    discover_plan_a_raw,
    discover_plan_c_packs,
    save_traj,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

SEED = 91
BUDGET_A = 18
BUDGET_B = 18
BUDGET_C = 6
BUDGET_D = 12

MILD_D2 = [
    {"target_skill": "dtheta", "dtheta_deg": 12.0, "dx_mm": 1.0, "dz_mm": 0.0},
    {"target_skill": "dtheta", "dtheta_deg": -14.0, "dx_mm": 0.0, "dz_mm": 1.0},
    {"target_skill": "dx", "dtheta_deg": 2.0, "dx_mm": 11.0, "dz_mm": 1.0},
    {"target_skill": "dx", "dtheta_deg": 0.0, "dx_mm": -11.0, "dz_mm": 0.0},
    {"target_skill": "dz_outward", "dtheta_deg": 0.0, "dx_mm": 1.0, "dz_mm": 14.0},
    {"target_skill": "dz_inward", "dtheta_deg": 0.0, "dx_mm": 1.0, "dz_mm": -14.0},
    {"target_skill": "support_transfer", "dtheta_deg": 4.0, "dx_mm": 13.0, "dz_mm": 2.0},
    {"target_skill": "dtheta+dx", "dtheta_deg": 12.0, "dx_mm": 10.0, "dz_mm": 0.0},
]


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _append(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.is_file() and path.stat().st_size:
        existing = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = sorted({k for r in existing + rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in existing + rows:
            w.writerow({k: r.get(k, "") for k in keys})


def try_a_prefix_cuts(env, blob, actions, raw_id, cfg):
    """Try Stage-1 prefix fractions; Precision only on first Precision-required cut."""
    from data_generation.oracles.plan_a.stage1_runner import replay_plan_a_actions_to_exit
    from data_generation.expert_pipeline.live_stage1 import _accept_precision

    acts = np.asarray(actions, np.float32)
    if len(acts) < 8:
        return None
    fracs = [0.4, 0.55, 0.7, 0.85]
    for fr in fracs:
        end = max(4, int(round(len(acts) * fr)))
        end = min(end, len(acts))
        exit_obj = replay_plan_a_actions_to_exit(
            env,
            {"capture": blob["capture"], "ref": blob["ref"], "sample_id": raw_id},
            acts[:end],
            plan_id="A",
            source_ic_id=f"{raw_id}__cut{int(fr*100)}",
            precision_config=cfg,
        )
        if not exit_obj.stage1_success_12s:
            continue
        if not exit_obj.precision_required:
            continue
        _, row = _accept_precision(env, exit_obj, plan_id="A", cfg=cfg)
        row.provenance = {**(row.provenance or {}), "cut_frac": fr, "n_actions": end, "source": "prefix_cut"}
        return row
    return None


def load_b1_pairs(rng, limit=30):
    base = ROOT / "results/data_generation/stage2_abc_mining/v2/outcomes"
    pairs = []
    if not base.is_dir():
        return pairs
    for jf in sorted(base.glob("B__hard_*__post_b1_v2__*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        toks = list(d.get("b1_tokens") or [])
        hid = None
        # sample_id like B__hard_910007__post_b1_v2__hash
        parts = jf.stem.split("__")
        for p in parts:
            if p.startswith("hard_"):
                hid = p
                break
        if not hid or not toks:
            continue
        snap = ROOT / "results/diagnostics/hard_ic_mining/snapshots" / f"{hid}.pkl"
        if not snap.is_file():
            continue
        pairs.append({"hard_id": hid, "tokens": toks, "path": snap, "prefix_len": min(3, len(toks))})
    rng.shuffle(pairs)
    return pairs[:limit]


def main() -> int:
    env = make_env()
    rng = np.random.default_rng(SEED)
    cfg = PrecisionConfig(use_verified_lateral_fallback=False)
    funnel, accepted, rejected, coverage = [], [], [], []

    # --- A2 via prefix cuts ---
    raws = discover_plan_a_raw(rng)
    print(f"[A-sup] raw={len(raws)} budget={BUDGET_A}", flush=True)
    n = 0
    for src in raws:
        if n >= BUDGET_A:
            break
        try:
            blob = pickle.loads(src["snap_path"].read_bytes())
            arch = np.load(src["npz_path"])
            if "actions" not in arch.files:
                continue
            n += 1
            print(f"[A-sup {n}/{BUDGET_A}] {src['raw_id']}", flush=True)
            row = try_a_prefix_cuts(env, blob, arch["actions"], src["raw_id"], cfg)
            if row is None:
                funnel.append({"branch": "A", "raw_id": src["raw_id"], "ok": 0, "reason": "NO_CUT_WORKED", "pass": "supplement"})
                continue
            funnel.append(
                {
                    "branch": "A",
                    "raw_id": row.raw_scenario_id,
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": "|".join(row.precision_skills or []),
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                    "pass": "supplement",
                }
            )
            if row.accepted:
                p = save_traj(row, None, row.raw_scenario_id)
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())
        except Exception as exc:
            funnel.append({"branch": "A", "raw_id": src["raw_id"], "ok": 0, "reason": str(exc), "pass": "supplement"})
            print(f"  EXC {exc}", flush=True)

    # --- B2 via known post_b1 token pairs ---
    table, book48 = load_vocab()
    pairs = load_b1_pairs(rng, limit=BUDGET_B)
    print(f"[B-sup] pairs={len(pairs)}", flush=True)
    for i, pair in enumerate(pairs, 1):
        blob = pickle.loads(pair["path"].read_bytes())
        rid = f"B_RAW__{pair['hard_id']}__known_b1"
        print(f"[B-sup {i}/{len(pairs)}] {rid} n_tok={len(pair['tokens'])}", flush=True)
        row = generate_b2_from_raw(
            env,
            blob,
            pair["tokens"],
            raw_scenario_id=rid,
            prefix_len=pair["prefix_len"],
            precision_config=cfg,
            provenance={"source": "stage2_known_b1_tokens", "hard_id": pair["hard_id"]},
            table=table,
            book48=book48,
        )
        funnel.append(
            {
                "branch": "B",
                "raw_id": rid,
                "ok": int(row.accepted),
                "classification": row.classification,
                "stage1_12s": row.stage1_success_12s,
                "precision_required": row.precision_required,
                "precision_status": row.precision_status,
                "skills": "|".join(row.precision_skills or []),
                "reason": row.reject_reason,
                "wall_s": round(row.wall_s, 3),
                "pass": "supplement",
            }
        )
        if row.accepted:
            p = save_traj(row, None, rid)
            accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
        else:
            rejected.append(row.to_row())

    # --- C2 optional retry ---
    packs = discover_plan_c_packs(rng)[:BUDGET_C]
    print(f"[C-sup] packs={len(packs)}", flush=True)
    for i, src in enumerate(packs, 1):
        try:
            pack = pickle.loads(src["path"].read_bytes())
            snap = pack.get("catch_snap") or pack.get("snap") or pack.get("root_snap")
            ref = pack.get("ref")
            raw_acts = pack.get("capture_actions")
            if raw_acts is None:
                raw_acts = pack.get("actions") or []
            catch_out = {
                "capture_actions": [np.asarray(a, np.float32) for a in list(raw_acts)],
                "mechanism": pack.get("mechanism"),
                "params": pack.get("params") or {},
                "cand": pack.get("cand") or {},
            }
            if snap is None or ref is None or len(catch_out["capture_actions"]) == 0:
                continue
            print(f"[C-sup {i}] {src['raw_id']}", flush=True)
            s1 = run_stage1_catch_and_lift(env.sim, snap, ref, catch_out)
            if not s1.get("ok"):
                funnel.append({"branch": "C", "raw_id": src["raw_id"], "ok": 0, "reason": s1.get("reason"), "pass": "supplement"})
                continue
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
                provenance={"pass": "supplement"},
                pose_correct_called=False,
            )
            funnel.append(
                {
                    "branch": "C",
                    "raw_id": src["raw_id"],
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": "|".join(row.precision_skills or []),
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                    "pass": "supplement",
                }
            )
            if row.accepted:
                p = save_traj(row, {"capture": exit_cap, "ref": ref}, src["raw_id"])
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())
        except Exception as exc:
            funnel.append({"branch": "C", "raw_id": src["raw_id"], "ok": 0, "reason": str(exc), "pass": "supplement"})
            print(f"  EXC {exc}", flush=True)

    # --- Milder D2 ---
    bases = discover_d2_bases(rng)
    print(f"[D-sup] bases={len(bases)} recipes={len(MILD_D2)}", flush=True)
    n = 0
    for recipe in MILD_D2:
        if n >= BUDGET_D:
            break
        for base in bases[:2]:
            if n >= BUDGET_D:
                break
            n += 1
            base_blob = pickle.loads(base["path"].read_bytes())
            base_blob["sample_id"] = base["sample_id"]
            rid = f"D2m__{recipe['target_skill']}__{n}"
            print(f"[D-sup {n}] {recipe['target_skill']}", flush=True)
            built = construct_d2_ic(env, base_blob, settle_n=16, **recipe)
            if not built.get("ok"):
                coverage.append({"target_skill": recipe["target_skill"], "status": "construction_failed", "reason": built.get("reason"), "raw_id": rid, "pass": "supplement"})
                continue
            row = generate_d2_from_synthetic_ic(
                env,
                {"capture": built["capture"], "ref": built["ref"], "sample_id": rid},
                raw_scenario_id=rid,
                construction_meta=built["construction"],
                precision_config=cfg,
            )
            executed = "|".join(row.precision_skills or [])
            if row.accepted:
                status = "executed_and_successful"
            elif row.precision_skills:
                status = "executed_but_failed"
            elif not row.stage1_success_12s:
                status = "targeted_but_unreachable_12s"
            elif not row.precision_required:
                status = "targeted_but_not_precision_required"
            else:
                status = "other"
            coverage.append(
                {
                    "target_skill": recipe["target_skill"],
                    "executed_skills": executed,
                    "status": status,
                    "SUCCESS_NEW": row.SUCCESS_NEW,
                    "precision_status": row.precision_status,
                    "raw_id": rid,
                    "pass": "supplement",
                }
            )
            funnel.append(
                {
                    "branch": "D",
                    "raw_id": rid,
                    "ok": int(row.accepted),
                    "classification": row.classification,
                    "stage1_12s": row.stage1_success_12s,
                    "precision_required": row.precision_required,
                    "precision_status": row.precision_status,
                    "skills": executed,
                    "target_skill": recipe["target_skill"],
                    "reason": row.reject_reason,
                    "wall_s": round(row.wall_s, 3),
                    "pass": "supplement",
                }
            )
            if row.accepted:
                p = save_traj(row, {"capture": built["capture"], "ref": built["ref"]}, rid)
                accepted.append({**row.to_row(), "traj_path": str(p.relative_to(ROOT)).replace("\\", "/")})
            else:
                rejected.append(row.to_row())

    _append(OUT / "FULL_FUNNEL.csv", funnel)
    _append(OUT / "A2_B2_GENERATION_FUNNEL.csv", [r for r in funnel if r.get("branch") in ("A", "B")])
    _append(OUT / "ACCEPTED_TRAJECTORIES.csv", accepted)
    _append(OUT / "REJECTED_CANDIDATES.csv", rejected)
    _append(OUT / "D2_SKILL_COVERAGE_MATRIX.csv", coverage)

    # rewrite C2 optional
    all_funnel = list(csv.DictReader((OUT / "FULL_FUNNEL.csv").open(encoding="utf-8")))
    c_rows = [r for r in all_funnel if r.get("branch") == "C"]
    all_acc = list(csv.DictReader((OUT / "ACCEPTED_TRAJECTORIES.csv").open(encoding="utf-8"))) if (OUT / "ACCEPTED_TRAJECTORIES.csv").stat().st_size else []
    c_acc = [r for r in all_acc if r.get("branch_target") == "C2"]
    (OUT / "C2_OPTIONAL_RESULTS.md").write_text(
        f"# C2_OPTIONAL_RESULTS\n\nAttempts: {len(c_rows)}\nAccepted C2: {len(c_acc)}\n",
        encoding="utf-8",
    )

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pass": "supplement",
        "seed": SEED,
        "n_accepted_this_pass": len(accepted),
        "accepted_by_branch_this_pass": {},
        "total_accepted_file": len(all_acc) + len(accepted),  # approximate if append duplicated - fixed below
    }
    # re-read accepted after append
    all_acc = list(csv.DictReader((OUT / "ACCEPTED_TRAJECTORIES.csv").open(encoding="utf-8"))) if (OUT / "ACCEPTED_TRAJECTORIES.csv").is_file() else []
    summary["total_accepted_file"] = len(all_acc)
    for r in accepted:
        b = r.get("branch_target") or "?"
        summary["accepted_by_branch_this_pass"][b] = summary["accepted_by_branch_this_pass"].get(b, 0) + 1
    (OUT / "PILOT_SUMMARY_SUPPLEMENT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
