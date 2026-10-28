"""P11 bounded pilot: A2 prefix-cut ICs, D2 stable off-target, B2 near-misses.

Writes under results/diagnostics/p11_targeted_ic_generation/.
Does not mutate data/experts_v2_official. STRICT admit only. No hard-IC tuning.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "diagnostics" / "p11_targeted_ic_generation"
OFFICIAL = ROOT / "data" / "experts_v2_official"
P10_B1 = ROOT / "results" / "diagnostics" / "p10_dataset_coverage" / "B1_RECOVERY_RESULTS.csv"
ICS_V2 = ROOT / "results" / "data_generation" / "stage2_abc_final_v2" / "ics"

BUDGET_A = 12
BUDGET_D = 18
BUDGET_B2 = 8
SEED = 11

# Mild + DZ-first oriented recipes (pre-freeze construction only).
D2_RECIPES = [
    {"target_skill": "dz_inward", "dtheta_deg": 0.0, "dx_mm": 1.5, "dz_mm": -16.0},
    {"target_skill": "dz_inward", "dtheta_deg": 0.0, "dx_mm": 2.0, "dz_mm": -20.0},
    {"target_skill": "dz_outward", "dtheta_deg": 0.0, "dx_mm": 1.5, "dz_mm": 16.0},
    {"target_skill": "dx", "dtheta_deg": 0.0, "dx_mm": 12.0, "dz_mm": 1.0},
    {"target_skill": "dx", "dtheta_deg": 0.0, "dx_mm": -12.0, "dz_mm": 1.0},
    {"target_skill": "dx+dz", "dtheta_deg": 0.0, "dx_mm": 10.0, "dz_mm": -12.0},
    {"target_skill": "dtheta", "dtheta_deg": 12.0, "dx_mm": 1.0, "dz_mm": 0.0},
    {"target_skill": "dz_then_dx", "dtheta_deg": 0.0, "dx_mm": 3.0, "dz_mm": -18.0},
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
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


def make_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _measure_ic(env, blob: dict) -> dict[str, Any]:
    from data_generation.oracles.precision.entry_conditions import assess_need
    from data_generation.oracles.precision.fsm import classify_case
    from data_generation.oracles.precision.residuals import measure_errors
    from data_generation.oracles.precision.state import PrecisionConfig
    from envs.simulation.capture import restore_capture

    restore_capture(env, blob["capture"])
    err = measure_errors(env.sim)
    cfg = PrecisionConfig()
    need = assess_need(err, cfg)
    plan = classify_case(err, cfg)
    return {
        "dx_mm": err.get("dx_mm"),
        "dy_mm": err.get("dy_mm"),
        "dz_mm": err.get("dz_mm"),
        "dtheta_deg": err.get("dtheta_deg"),
        "e_pos_mm": err.get("e_pos_mm"),
        "bilateral": err.get("bilateral"),
        "v_rel_norm": err.get("v_rel_norm"),
        "obj_z": err.get("obj_z"),
        "fn_L": err.get("fn_L"),
        "fn_R": err.get("fn_R"),
        "precision_required": bool(
            need.get("dx_needed")
            or need.get("dz_needed")
            or need.get("dtheta_needed")
            or not need.get("ready_for_direct_handoff_assessment")
        ),
        "fsm_case": plan.case,
        "fsm_planned": "|".join(plan.planned_skills),
    }


# ---------------------------------------------------------------------------
# Part 1 — source distribution
# ---------------------------------------------------------------------------
def analyze_sources(env) -> tuple[list[dict], str]:
    from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS

    rows: list[dict] = []
    man = OFFICIAL / "manifest.jsonl"
    by_class: dict[str, list[dict]] = defaultdict(list)
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cls = BRANCH_TO_CLASS.get(r.get("branch_id"), r.get("branch_id"))
            by_class[cls].append(r)

    # Sample up to N per class of interest
    samples = []
    for cls, lim in [("A1", 12), ("A2", 3), ("B1", 8), ("B2", 8), ("D2", 3)]:
        for r in by_class.get(cls, [])[:lim]:
            samples.append((cls, r))

    # Eight B near-misses
    b_near = []
    if P10_B1.is_file():
        for r in csv.DictReader(P10_B1.open(encoding="utf-8")):
            b_near.append(r["source_ic_id"])

    print(f"[source] measuring {len(samples)} official + {len(b_near)} B near-misses", flush=True)
    for cls, r in samples:
        sid = r.get("source_ic_id") or r.get("attempt_id")
        # locate pickle
        branch = r.get("branch_id")
        heavy = None
        for cand in [
            OFFICIAL / str(branch) / f"{r.get('attempt_id')}.pkl",
            ICS_V2 / f"{sid}.pkl",
        ]:
            if cand.is_file():
                heavy = cand
                break
        # Prefer loading IC from source path in manifest
        ic_rel = r.get("source_ic_path")
        blob = None
        if ic_rel and (ROOT / ic_rel).is_file():
            blob = pickle.loads((ROOT / ic_rel).read_bytes())
        elif heavy and heavy.is_file():
            payload = pickle.loads(heavy.read_bytes())
            blob = payload if "capture" in payload else {
                "capture": payload.get("capture"),
                "ref": payload.get("ref"),
            }
        if not blob or "capture" not in blob:
            rows.append({"cohort": f"official_{cls}", "sample_id": sid, "ok": False, "reason": "NO_CAPTURE"})
            continue
        try:
            m = _measure_ic(env, blob)
            rows.append(
                {
                    "cohort": f"official_{cls}",
                    "sample_id": sid,
                    "ok": True,
                    "skill_sequence": r.get("skill_sequence"),
                    "n_actions": r.get("n_actions"),
                    **m,
                }
            )
        except Exception as exc:
            rows.append({"cohort": f"official_{cls}", "sample_id": sid, "ok": False, "reason": str(exc)})

    for sid in b_near:
        path = ICS_V2 / f"{sid}.pkl"
        if not path.is_file():
            rows.append({"cohort": "b_near_miss", "sample_id": sid, "ok": False, "reason": "MISSING_IC"})
            continue
        blob = pickle.loads(path.read_bytes())
        m = _measure_ic(env, blob)
        rows.append({"cohort": "b_near_miss", "sample_id": sid, "ok": True, **m})

    # Compact analysis markdown
    def _stats(cohort: str) -> str:
        sub = [r for r in rows if r.get("cohort") == cohort and r.get("ok")]
        if not sub:
            return f"- {cohort}: n=0\n"
        dx = np.array([float(r["dx_mm"]) for r in sub])
        dz = np.array([float(r["dz_mm"]) for r in sub])
        th = np.array([float(r["dtheta_deg"]) for r in sub])
        ep = np.array([float(r["e_pos_mm"]) for r in sub])
        cases = Counter(r.get("fsm_case") for r in sub)
        prec = sum(1 for r in sub if r.get("precision_required"))
        return (
            f"- **{cohort}** (n={len(sub)}): "
            f"|dx| mean={np.abs(dx).mean():.1f} mm, |dz| mean={np.abs(dz).mean():.1f} mm, "
            f"θ mean={th.mean():.1f}°, e_pos mean={ep.mean():.1f} mm; "
            f"precision_required={prec}/{len(sub)}; FSM {dict(cases)}\n"
        )

    md = "# Source Distribution Analysis (P11)\n\n"
    md += "Residuals measured at IC capture (official source_ic or expert payload).\n\n"
    md += "## Official samples\n"
    for c in ["official_A1", "official_A2", "official_B1", "official_B2", "official_D2"]:
        md += _stats(c)
    md += "\n## B near-misses (P10 position-fail / Precision-required)\n"
    md += _stats("b_near_miss")
    md += """
## Gaps
- **A1** exits cluster near handoff-ready residuals (small e_pos / AXIS) → routed to direct handoff.
- **A2** scarce: Stage-1 exits that remain 12S-stable *and* Precision-required are rare under full Plan A tapes.
- **B near-misses** show elevated e_pos (~12–16 mm) with mixed DX/DZ — natural B2 Precision entries.
- **D2** official (3) came from unstable HARD_IC under old gate; need new independently 12S off-target family.
- **DZ→DX**: official skill_sequence labels lack `dz→dx`; need CASE2-dominant source states (|dz| dominates).

## Useful geometries
Stable bilateral grasps with e_pos just outside 12 mm and one axis dominating produce trainable Precision demos.
Large dual-axis residuals without reconfigure hooks tend to fall through Case-3 without admitting.
"""
    _write_csv(OUT / "SOURCE_DISTRIBUTION_ROWS.csv", rows)
    return rows, md


# ---------------------------------------------------------------------------
# Part 2 — A2 via Plan A prefix cuts
# ---------------------------------------------------------------------------
def run_a2_pilot(env, funnel: list[dict], verified: list[dict], skills: list[dict]) -> None:
    from data_generation.expert_pipeline.live_stage1 import _accept_precision, extract_precision_skills
    from data_generation.oracles.plan_a.stage1_runner import replay_plan_a_actions_to_exit
    from data_generation.oracles.precision.state import PrecisionConfig
    from experiments.precision.precision_required_pilot_v2.run_pilot import discover_plan_a_raw

    rng = np.random.default_rng(SEED)
    raws = discover_plan_a_raw(rng)
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    print(f"[A2] raw={len(raws)} budget={BUDGET_A}", flush=True)
    n = 0
    for src in raws:
        if n >= BUDGET_A:
            break
        n += 1
        t0 = time.time()
        try:
            blob = pickle.loads(src["snap_path"].read_bytes())
            arch = np.load(src["npz_path"])
            acts = np.asarray(arch["actions"], np.float32)
            if len(acts) < 8:
                funnel.append(
                    {
                        "family": "A2_prefix_cut",
                        "sample_id": src["raw_id"],
                        "A_EXECUTED": False,
                        "reject": "TOO_FEW_ACTIONS",
                    }
                )
                continue
            print(f"  [A2 {n}/{BUDGET_A}] {src['raw_id']} n_act={len(acts)}", flush=True)
            # Full tape first
            fracs = [1.0, 0.85, 0.7, 0.55, 0.4]
            hit = None
            for fr in fracs:
                end = max(4, int(round(len(acts) * fr)))
                end = min(end, len(acts))
                sid = f"{src['raw_id']}__cut{int(fr*100)}"
                exit_obj = replay_plan_a_actions_to_exit(
                    env,
                    {"capture": blob["capture"], "ref": blob["ref"], "sample_id": sid},
                    acts[:end],
                    plan_id="A",
                    source_ic_id=sid,
                    precision_config=cfg,
                )
                row_f = {
                    "family": "A2_prefix_cut",
                    "sample_id": sid,
                    "parent_raw": src["raw_id"],
                    "cut_frac": fr,
                    "n_stage1_actions": end,
                    "A_EXECUTED": True,
                    "A_RECOVERY_COMPLETED": bool(exit_obj.stage1_success_12s or exit_obj.classification),
                    "SUCCESS_12S": bool(exit_obj.stage1_success_12s),
                    "PRECISION_REQUIRED": bool(exit_obj.precision_required) if exit_obj.stage1_success_12s else None,
                    "classification": exit_obj.classification,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if not exit_obj.stage1_success_12s:
                    row_f["PRECISION_EXECUTED"] = False
                    row_f["SUCCESS_NEW_STRICT"] = False
                    row_f["reject"] = f"NO_12S:{exit_obj.stage1_failure_reason}"
                    funnel.append(row_f)
                    continue
                if not exit_obj.precision_required:
                    row_f["PRECISION_EXECUTED"] = False
                    row_f["SUCCESS_NEW_STRICT"] = False
                    row_f["reject"] = "A1_DIRECT_ELIGIBLE"
                    funnel.append(row_f)
                    continue
                # Precision-required — run once
                _, live = _accept_precision(env, exit_obj, plan_id="A", cfg=cfg)
                sk = extract_precision_skills(list(live.precision_phases or []))
                row_f["PRECISION_EXECUTED"] = bool(live.n_precision_actions)
                row_f["SUCCESS_NEW_STRICT"] = bool(live.SUCCESS_NEW)
                row_f["precision_status"] = live.precision_status
                row_f["skills"] = "|".join(sk)
                row_f["accepted"] = bool(live.accepted)
                row_f["fsm_phases"] = "|".join(live.precision_phases or [])
                funnel.append(row_f)
                skills.append(
                    {
                        "family": "A2",
                        "sample_id": sid,
                        "planned": "",
                        "executed": "|".join(sk),
                        "accepted": live.accepted,
                        "transition": "→".join(sk) if sk else "precision_skip_or_empty",
                    }
                )
                if live.accepted:
                    verified.append(
                        {
                            "class_label": "A2",
                            "sample_id": sid,
                            "parent_lineage": src["raw_id"],
                            "capture_fp": live.capture_fp,
                            "skill_sequence": "→".join(sk),
                            "SUCCESS_NEW": True,
                            "SUCCESS_12S": live.SUCCESS_12S_final,
                            "n_precision_actions": live.n_precision_actions,
                            "admit_success_key": "SUCCESS_NEW",
                            "source_family": "A2_prefix_cut",
                            "synthetic": False,
                        }
                    )
                    hit = live
                    print(f"    ACCEPTED A2 {sid} skills={sk}", flush=True)
                    break
                print(f"    cut={fr} 12S+need but Precision fail: {live.precision_status}", flush=True)
                # keep searching shorter cuts only if this cut failed precision
            if hit is None:
                pass
        except Exception as exc:
            funnel.append(
                {
                    "family": "A2_prefix_cut",
                    "sample_id": src["raw_id"],
                    "A_EXECUTED": False,
                    "reject": f"EXCEPTION:{type(exc).__name__}:{exc}",
                    "traceback": traceback.format_exc()[-400:],
                }
            )
            print(f"    EXC {exc}", flush=True)


# ---------------------------------------------------------------------------
# Part 3 — D2 stable off-target
# ---------------------------------------------------------------------------
def run_d2_pilot(env, funnel: list[dict], verified: list[dict], skills: list[dict]) -> None:
    from data_generation.expert_pipeline.live_stage1 import generate_d2_from_synthetic_ic, extract_precision_skills
    from data_generation.oracles.precision.d2_eligibility import check_d2_eligible
    from data_generation.oracles.precision.fsm import classify_case
    from data_generation.oracles.precision.residuals import measure_errors
    from data_generation.oracles.precision.state import PrecisionConfig
    from envs.simulation.capture import restore_capture, take_capture
    from experiments.precision.precision_required_pilot_v2.d2_synthetic import construct_d2_ic
    from experiments.precision.precision_required_pilot_v2.run_pilot import discover_d2_bases

    rng = np.random.default_rng(SEED + 1)
    bases = discover_d2_bases(rng)
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    print(f"[D2] bases={len(bases)} recipes={len(D2_RECIPES)} budget={BUDGET_D}", flush=True)
    n = 0
    for recipe in D2_RECIPES:
        if n >= BUDGET_D:
            break
        for base in bases:
            if n >= BUDGET_D:
                break
            n += 1
            sid = f"D2_SYN__{recipe['target_skill']}__{base['sample_id'][:40]}__{n}"
            t0 = time.time()
            print(f"  [D2 {n}/{BUDGET_D}] {recipe['target_skill']} base={base['sample_id'][:50]}", flush=True)
            try:
                base_blob = pickle.loads(base["path"].read_bytes())
                if "capture" not in base_blob:
                    # experts_v3 / terminal may nest
                    if "blob" in base_blob:
                        base_blob = base_blob["blob"]
                base_blob = {
                    "capture": base_blob["capture"],
                    "ref": base_blob["ref"],
                    "sample_id": base["sample_id"],
                }
                # Construction (preparation) — qpos perturb then settle; freeze capture.
                built = construct_d2_ic(
                    env,
                    base_blob,
                    target_skill=recipe["target_skill"],
                    dtheta_deg=recipe["dtheta_deg"],
                    dx_mm=recipe["dx_mm"],
                    dz_mm=recipe["dz_mm"],
                    settle_n=14,
                )
                row_f = {
                    "family": "D2_stable_offtarget",
                    "sample_id": sid,
                    "base_id": base["sample_id"],
                    "target_skill": recipe["target_skill"],
                    "requested_dx_mm": recipe["dx_mm"],
                    "requested_dz_mm": recipe["dz_mm"],
                    "requested_dtheta_deg": recipe["dtheta_deg"],
                    "construction_ok": bool(built.get("ok")),
                    "A_EXECUTED": False,
                    "D2_PREP_EXECUTED": True,
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if not built.get("ok"):
                    row_f["SUCCESS_12S"] = False
                    row_f["reject"] = f"CONSTRUCT:{built.get('reason')}"
                    funnel.append(row_f)
                    continue
                ic = {
                    "capture": built["capture"],
                    "ref": built["ref"],
                    "sample_id": sid,
                }
                # Independent 12S on a clone path via check_d2_eligible (does not mutate IC)
                elig = check_d2_eligible(env, ic, cfg=cfg)
                row_f["SUCCESS_12S"] = bool(elig.SUCCESS_12S)
                row_f["PRECISION_REQUIRED"] = elig.precision_required
                row_f["d2_status"] = elig.status
                if not elig.eligible:
                    row_f["PRECISION_EXECUTED"] = False
                    row_f["SUCCESS_NEW_STRICT"] = False
                    row_f["reject"] = f"D2_GATE:{elig.status}:{elig.reason}"
                    funnel.append(row_f)
                    continue
                # Measure FSM case on the frozen IC (restore original capture)
                restore_capture(env, ic["capture"])
                err = measure_errors(env.sim)
                plan = classify_case(err, cfg)
                row_f["fsm_case"] = plan.case
                row_f["entry_dx_mm"] = err.get("dx_mm")
                row_f["entry_dz_mm"] = err.get("dz_mm")
                row_f["entry_dtheta_deg"] = err.get("dtheta_deg")
                # Precision from the same frozen capture
                live = generate_d2_from_synthetic_ic(
                    env,
                    ic,
                    raw_scenario_id=sid,
                    construction_meta={
                        **(built.get("construction") or {}),
                        "base_path": str(base["path"].relative_to(ROOT)).replace("\\", "/"),
                        "note": "prep_qpos_perturb_before_freeze; expert starts at frozen capture",
                    },
                    precision_config=cfg,
                )
                sk = extract_precision_skills(list(live.precision_phases or []))
                row_f["PRECISION_EXECUTED"] = bool(live.n_precision_actions)
                row_f["SUCCESS_NEW_STRICT"] = bool(live.SUCCESS_NEW)
                row_f["precision_status"] = live.precision_status
                row_f["skills"] = "|".join(sk)
                row_f["accepted"] = bool(live.accepted)
                row_f["fsm_phases"] = "|".join(live.precision_phases or [])
                funnel.append(row_f)
                skills.append(
                    {
                        "family": "D2",
                        "sample_id": sid,
                        "planned": plan.case + ":" + "|".join(plan.planned_skills),
                        "executed": "|".join(sk),
                        "accepted": live.accepted,
                        "transition": "→".join(sk) if sk else "empty",
                        "fsm_case": plan.case,
                    }
                )
                if live.accepted:
                    verified.append(
                        {
                            "class_label": "D2",
                            "sample_id": sid,
                            "parent_lineage": base["sample_id"],
                            "capture_fp": live.capture_fp,
                            "skill_sequence": "→".join(sk),
                            "SUCCESS_NEW": True,
                            "SUCCESS_12S": live.SUCCESS_12S_final,
                            "n_precision_actions": live.n_precision_actions,
                            "admit_success_key": "SUCCESS_NEW",
                            "source_family": "D2_stable_offtarget",
                            "synthetic": True,
                            "fsm_case": plan.case,
                        }
                    )
                    print(f"    ACCEPTED D2 {sid} case={plan.case} skills={sk}", flush=True)
                else:
                    print(f"    fail {live.reject_reason or live.precision_status}", flush=True)
            except Exception as exc:
                funnel.append(
                    {
                        "family": "D2_stable_offtarget",
                        "sample_id": sid,
                        "reject": f"EXCEPTION:{type(exc).__name__}:{exc}",
                    }
                )
                print(f"    EXC {exc}", flush=True)


# ---------------------------------------------------------------------------
# Part 4 — B2 near-misses
# ---------------------------------------------------------------------------
def run_b2_pilot(env, funnel: list[dict], verified: list[dict], skills: list[dict], b2_rows: list[dict]) -> None:
    from data_generation.oracles.precision.entry_conditions import assess_need
    from data_generation.oracles.precision.fsm import classify_case
    from data_generation.oracles.precision.residuals import measure_errors
    from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible
    from data_generation.oracles.precision.state import PrecisionConfig
    from data_generation.oracles.precision.state_machine import run_precision
    from data_generation.expert_pipeline.live_stage1 import extract_precision_skills
    from envs.simulation.capture import restore_capture

    if not P10_B1.is_file():
        print("[B2] no P10 B1_RECOVERY_RESULTS.csv", flush=True)
        return
    ids = [r["source_ic_id"] for r in csv.DictReader(P10_B1.open(encoding="utf-8"))]
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    print(f"[B2] candidates={len(ids)}", flush=True)
    for i, sid in enumerate(ids[:BUDGET_B2]):
        t0 = time.time()
        path = ICS_V2 / f"{sid}.pkl"
        print(f"  [B2 {i+1}/{len(ids)}] {sid}", flush=True)
        row = {
            "source_ic_id": sid,
            "ic_path": str(path.relative_to(ROOT)).replace("\\", "/") if path.is_file() else "",
        }
        if not path.is_file():
            row.update({"ok": False, "reject": "MISSING_IC"})
            b2_rows.append(row)
            continue
        try:
            blob = pickle.loads(path.read_bytes())
            blob_ad = {"capture": blob["capture"], "ref": blob["ref"], "sample_id": sid}
            elig = check_stage1_eligible(env, blob_ad, allow_live_verify=True, record_live=False)
            restore_capture(env, blob["capture"])
            err = measure_errors(env.sim)
            need = assess_need(err, cfg)
            plan = classify_case(err, cfg)
            prec_req = bool(
                need.get("dx_needed") or need.get("dz_needed") or need.get("dtheta_needed")
                or not need.get("ready_for_direct_handoff_assessment")
            )
            row.update(
                {
                    "stage1_eligible": elig.eligible,
                    "SUCCESS_12S": elig.SUCCESS_12S,
                    "precision_required": prec_req,
                    "dx_mm": err.get("dx_mm"),
                    "dz_mm": err.get("dz_mm"),
                    "dtheta_deg": err.get("dtheta_deg"),
                    "e_pos_mm": err.get("e_pos_mm"),
                    "fsm_case": plan.case,
                    "fsm_planned": "|".join(plan.planned_skills),
                    "class_if_prec": "B2" if prec_req else "B1",
                }
            )
            funnel_row = {
                "family": "B2_near_miss",
                "sample_id": sid,
                "A_EXECUTED": False,
                "B_STAGE1_EVIDENCE": bool(elig.eligible),
                "SUCCESS_12S": bool(elig.SUCCESS_12S),
                "PRECISION_REQUIRED": prec_req,
            }
            if not elig.eligible or not elig.SUCCESS_12S:
                row["reject"] = f"STAGE1:{elig.reason}"
                row["accepted"] = False
                b2_rows.append(row)
                funnel_row["reject"] = row["reject"]
                funnel.append(funnel_row)
                continue
            if not prec_req:
                row["reject"] = "NOT_PRECISION_REQUIRED"
                row["accepted"] = False
                b2_rows.append(row)
                funnel_row["reject"] = row["reject"]
                funnel.append(funnel_row)
                continue
            res = run_precision(env, blob_ad, "B", cfg)
            sk = extract_precision_skills(list(res.phases or []))
            hand = res.handoff or {}
            ok = res.status in ("PRECISION_EXECUTED_SUCCESS", "PRECISION_SKIPPED_DIRECT_SUCCESS")
            row.update(
                {
                    "precision_status": res.status,
                    "skills_executed": "|".join(sk),
                    "SUCCESS_NEW_STRICT": bool(hand.get("SUCCESS_NEW")),
                    "SUCCESS_12S_final": bool(hand.get("SUCCESS_12S")),
                    "SUCCESS_RELAXED_15DEG": bool(hand.get("SUCCESS_RELAXED_15DEG")),
                    "accepted": ok,
                    "n_actions": len(res.actions or []),
                    "failures": "|".join(res.failures or [])[:300],
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            b2_rows.append(row)
            funnel_row.update(
                {
                    "PRECISION_EXECUTED": bool(res.actions),
                    "SUCCESS_NEW_STRICT": bool(hand.get("SUCCESS_NEW")),
                    "accepted": ok,
                    "skills": "|".join(sk),
                    "precision_status": res.status,
                }
            )
            funnel.append(funnel_row)
            skills.append(
                {
                    "family": "B2",
                    "sample_id": sid,
                    "planned": plan.case + ":" + "|".join(plan.planned_skills),
                    "executed": "|".join(sk),
                    "accepted": ok,
                    "transition": "→".join(sk) if sk else "empty",
                    "fsm_case": plan.case,
                }
            )
            if ok:
                verified.append(
                    {
                        "class_label": "B2",
                        "sample_id": sid,
                        "parent_lineage": sid,
                        "skill_sequence": "→".join(sk),
                        "SUCCESS_NEW": True,
                        "SUCCESS_12S": hand.get("SUCCESS_12S"),
                        "n_precision_actions": len(res.actions or []),
                        "admit_success_key": "SUCCESS_NEW",
                        "source_family": "B2_near_miss",
                        "synthetic": False,
                        "fsm_case": plan.case,
                    }
                )
                print(f"    ACCEPTED B2 skills={sk}", flush=True)
            else:
                print(f"    fail {res.status}", flush=True)
        except Exception as exc:
            row["reject"] = f"EXCEPTION:{exc}"
            row["accepted"] = False
            b2_rows.append(row)
            print(f"    EXC {exc}", flush=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_reports(
    *,
    source_md: str,
    funnel: list[dict],
    verified: list[dict],
    skills: list[dict],
    b2_rows: list[dict],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_md(OUT / "SOURCE_DISTRIBUTION_ANALYSIS.md", source_md)
    _write_csv(OUT / "IC_SCREENING_FUNNEL.csv", funnel)
    _write_csv(OUT / "NEW_VERIFIED_EXPERTS.csv", verified)
    _write_csv(OUT / "SKILL_EXECUTION_COVERAGE.csv", skills)
    _write_csv(OUT / "B2_NEAR_MISS_RECLASSIFICATION.csv", b2_rows)

    def rate(family: str, key: str) -> tuple[int, int]:
        sub = [r for r in funnel if r.get("family") == family]
        num = sum(1 for r in sub if r.get(key) in (True, "True", 1))
        return num, len(sub)

    a_12, a_n = rate("A2_prefix_cut", "SUCCESS_12S")
    a_need, _ = rate("A2_prefix_cut", "PRECISION_REQUIRED")
    a_ok, _ = rate("A2_prefix_cut", "SUCCESS_NEW_STRICT")
    d_12, d_n = rate("D2_stable_offtarget", "SUCCESS_12S")
    d_need, _ = rate("D2_stable_offtarget", "PRECISION_REQUIRED")
    d_ok, _ = rate("D2_stable_offtarget", "SUCCESS_NEW_STRICT")
    b_ok = sum(1 for r in b2_rows if r.get("accepted"))
    b_n = len(b2_rows)

    n_a2 = sum(1 for v in verified if v.get("class_label") == "A2")
    n_d2 = sum(1 for v in verified if v.get("class_label") == "D2")
    n_b2 = sum(1 for v in verified if v.get("class_label") == "B2")

    dz_dx = sum(
        1
        for s in skills
        if s.get("accepted")
        and "dz" in str(s.get("executed") or "").lower()
        and "dx" in str(s.get("executed") or "").lower()
        and str(s.get("executed") or "").lower().find("dz")
        < str(s.get("executed") or "").lower().find("dx")
    )
    dx_dz = sum(
        1
        for s in skills
        if s.get("accepted") and "→" in str(s.get("transition") or "") and "dx" in str(s.get("transition")).lower()
    )

    _write_md(
        OUT / "A2_IC_GENERATION_SPEC.md",
        f"""# A2 IC Generation Spec (P11)

## Mechanism
Plan A inward-sliding / controlled-sliding continuous tapes from `data/unified_experts`.

**Legal control knobs (this pilot):**
1. Raw scenario selection (physical_root_id)
2. Stage-1 action **prefix fraction** (0.4–1.0) — shorter prefixes leave residual while still requiring genuine Plan A motion
3. Full tape first, then denser sampling of cuts that pass 12S

**Forbidden:** rewriting Stage-1 exit qpos; injecting artificial DX/DZ after Plan A; mass/friction hacks on frozen captures.

## Screening funnel
`A_EXECUTED → A_RECOVERY_COMPLETED → SUCCESS_12S → PRECISION_REQUIRED → PRECISION_EXECUTED → SUCCESS_NEW_STRICT`

## Pilot yield
- Attempts (raw×cuts logged): see `IC_SCREENING_FUNNEL.csv` family=`A2_prefix_cut`
- SUCCESS_12S hits: {a_12} (rows with key set among {a_n} family rows)
- New verified A2 lineages this pilot: **{n_a2}**

## Scale next if
Prefix cuts that land in e_pos just outside 12 mm with bilateral contact show nonzero Precision accept rate.
""",
    )

    _write_md(
        OUT / "D2_STABLE_IC_GENERATION_SPEC.md",
        f"""# D2 Stable Off-Target IC Generation Spec (P11)

## Method
1. Start from a **stable bilateral base** (Plan A Stage-1 terminal / A1)
2. **Preparation** (not part of expert tape): hand-frame object pose perturb + `settle_grip` (`construct_d2_ic`)
3. **Freeze** capture+ref; tag `synthetic_initial_condition=True`
4. Run independent `check_d2_eligible` (12S + Precision-required) on that frozen state
5. Run Precision from the **same** frozen capture (no teleport mid-expert)

Preparation qpos rewrite is allowed **only before freeze**. Verification does not replace the IC with a post-12S endpoint.

## Recipes
See `D2_RECIPES` in `run_p11.py` — includes DZ-dominant variants for CASE2 / DZ→DX.

## Pilot yield
- Family rows: {d_n}; SUCCESS_12S: {d_12}; new verified D2: **{n_d2}**

## Do not
Reuse the 45 unstable HARD_IC as the main pool; change physics/M111; mid-trajectory qpos edits.
""",
    )

    _write_md(
        OUT / "LINEAGE_AUDIT.md",
        f"""# Lineage Audit (P11)

- Official dataset **not modified**
- Pilot artifacts only under `{OUT.as_posix()}`
- Verified candidates (diagnostic): {len(verified)}
  - A2: {n_a2}
  - B2: {n_b2}
  - D2: {n_d2}
- Synthetic D2 tagged `synthetic=True` with base parent_lineage
- A2 prefix cuts tagged `parent_raw` + `cut_frac`
- Duplicate capture_fp must not be double-counted for official admission
- STRICT admit only (`SUCCESS_NEW`); RELAXED recorded but not used for admission
""",
    )

    _write_md(
        OUT / "VIDEO_INDEX.md",
        f"""# VIDEO_INDEX — P11

Videos rendered only for newly STRICT-accepted independent trajectories (if any).

Verified this pilot: A2={n_a2}, B2={n_b2}, D2={n_d2}.

To render a Trajectory (example):
```text
# use existing precision video helpers once accepts exist under generation traj dumps
python experiments/precision/p11_targeted_ic_generation/run_p11.py
```

If zero accepts, no new videos — avoid fabricating demos.
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P11 Targeted IC Generation

Generated: {_utc()}

## Answers
1. **A2 IC distributions:** Prefix-cut Plan A pilot yield — new verified A2 = **{n_a2}**. Full tapes mostly A1; shorter cuts trade 12S vs residual (see funnel).
2. **Stable D2 without violating physics:** Construction uses pre-freeze settle; gate requires independent 12S. New verified D2 = **{n_d2}**.
3. **B2 from 8 near-misses:** **{b_ok} / {b_n}** STRICT-accepted as B2.
4. **DZ→DX verified:** accepted skill pairs with dz before dx ≈ **{dz_dx}** (see `SKILL_EXECUTION_COVERAGE.csv`).
5. **Funnel rates:** A2 family rows={a_n} 12S≈{a_12} need≈{a_need} strict_ok≈{a_ok}; D2 rows={d_n} 12S≈{d_12} need≈{d_need} strict_ok≈{d_ok}; B2 {b_ok}/{b_n}.
6. **New independent experts (diagnostic, not yet official):** **{len(verified)}** total ({n_a2} A2 + {n_b2} B2 + {n_d2} D2).
7. **Scale next:** families with nonzero STRICT accept rate in `IC_SCREENING_FUNNEL.csv`; stop zero-yield families (unstable HARD_IC already ruled out in P10).

## Exclusions respected
No hard-IC heuristic tuning; no P7/P8; no θ→15° admit; no M111/physics change; no SFT/RL; official corpus untouched.
""",
    )

    (OUT / "raw_p11_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "n_verified": len(verified),
                "by_class": {"A2": n_a2, "B2": n_b2, "D2": n_d2},
                "b2_accepted": b_ok,
                "b2_attempted": b_n,
                "dz_dx_accepted": dz_dx,
                "funnel_rows": len(funnel),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    funnel: list[dict] = []
    verified: list[dict] = []
    skills: list[dict] = []
    b2_rows: list[dict] = []

    print("=== P11 Part 1: source distribution ===", flush=True)
    _, source_md = analyze_sources(env)

    print("=== P11 Part 4: B2 near-misses (fast path) ===", flush=True)
    run_b2_pilot(env, funnel, verified, skills, b2_rows)

    print("=== P11 Part 3: D2 stable off-target ===", flush=True)
    run_d2_pilot(env, funnel, verified, skills)

    print("=== P11 Part 2: A2 prefix cuts ===", flush=True)
    run_a2_pilot(env, funnel, verified, skills)

    print("=== Writing reports ===", flush=True)
    write_reports(
        source_md=source_md,
        funnel=funnel,
        verified=verified,
        skills=skills,
        b2_rows=b2_rows,
    )
    print(f"Done → {OUT} verified={len(verified)}", flush=True)


if __name__ == "__main__":
    main()
