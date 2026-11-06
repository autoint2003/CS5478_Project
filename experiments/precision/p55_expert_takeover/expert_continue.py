"""Phase 3 — Expert continuation from policy-visited intermediate states."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.oracles.plan_c.expert_search import run_stage1_catch_and_lift, search_catch
from data_generation.oracles.precision.state_machine import run_precision
from envs.outcomes.support_drop import latched
from envs.simulation.capture import take_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import restore_branch_state
from experiments.precision.p53_independent_grpo64.success_relaxed import evaluate_success_relaxed_12s
from experiments.precision.p55_expert_takeover.ic_resolve import resolve_starts
from experiments.precision.p55_expert_takeover.paths import OUT, SEED
from training.recovery_execution import make_snap


def _ref_of(start) -> Any:
    if start.kind == "capture":
        return start.ref
    return (start.ep or {}).get("ref")


def _eval_relaxed(env, ref) -> dict:
    ev = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    rel = evaluate_success_relaxed_12s(ev)
    return {
        "SUCCESS_RELAXED_12S": bool(rel["SUCCESS_RELAXED_12S"]),
        "SUCCESS_12S": bool(rel["SUCCESS_12S"]),
        "SUCCESS_NEW": bool(rel["SUCCESS_NEW"]),
        "reason": ev.get("reason"),
        "pre_pos_mm": ev.get("pre_pos_mm"),
        "pre_theta": ev.get("pre_theta"),
    }


def _try_precision(env, ref, plan_id: str) -> dict:
    """Attempt precision expert; may reject unstable states (do not force)."""
    cap = take_capture(env)
    blob = {"capture": cap, "ref": ref, "ic_id": plan_id}
    t0 = time.perf_counter()
    try:
        res = run_precision(env, blob, plan_id=plan_id)
    except Exception as e:
        return {
            "accepted": False,
            "status": f"PRECISION_EXCEPTION:{type(e).__name__}",
            "actions": [],
            "wall_s": time.perf_counter() - t0,
            "error": str(e)[:200],
        }
    actions = list(res.actions or []) if hasattr(res, "actions") else []
    status = str(getattr(res, "status", None) or "UNKNOWN")
    accepted = status not in (
        "STABLE_GATE_FAILED",
        "STAGE1_INELIGIBLE",
        "PRECISION_SKIPPED",
    ) and "INELIGIBLE" not in status.upper() and "GATE_FAILED" not in status.upper()
    # Soft accept: executed something or skipped because already good
    if getattr(res, "precision_skipped", False) and getattr(res, "gate_ok", False):
        accepted = True
    return {
        "accepted": bool(accepted or getattr(res, "precision_executed", False)),
        "status": status,
        "actions": [np.asarray(a, np.float32) for a in actions],
        "wall_s": time.perf_counter() - t0,
        "gate_ok": bool(getattr(res, "gate_ok", False)),
        "precision_executed": bool(getattr(res, "precision_executed", False)),
        "precision_skipped": bool(getattr(res, "precision_skipped", False)),
        "note": str(getattr(res, "note", "") or "")[:300],
    }


def _try_plan_a_schedules(env, ref, *, max_schedules: int = 6) -> dict:
    """Existing Plan-A inward-reseat seed schedules from current capture (no expert edits)."""
    from data_generation.oracles.plan_a.inward_reseat import SEEDS, build_schedule, rollout
    from training.k47_codebook import load_codebook

    book = load_codebook()
    cap = take_capture(env)
    blob = {"capture": cap, "ref": ref}
    t0 = time.perf_counter()
    best = None
    tried = 0
    for seed in SEEDS:
        for wy_sign in (+1.0, -1.0):
            if tried >= max_schedules:
                break
            actions, _cuts, illegal = build_schedule(seed, wy_sign, book)
            tried += 1
            if illegal or actions is None or len(actions) == 0:
                continue
            executed, _rows, drop_step = rollout(env, blob, actions)
            if drop_step is not None:
                continue
            # Evaluate without forcing — legal handoff path
            if latched(env.sim):
                continue
            outcome = _eval_relaxed(env, ref)
            packed = {
                "accepted": True,
                "status": f"PLAN_A_SEED:{seed['name']}:wy{int(wy_sign)}",
                "actions": [np.asarray(a, np.float32) for a in executed],
                "outcome": outcome,
            }
            if outcome.get("SUCCESS_RELAXED_12S"):
                packed["wall_s"] = time.perf_counter() - t0
                return packed
            if best is None or (
                int(outcome.get("SUCCESS_12S") or 0) > int((best.get("outcome") or {}).get("SUCCESS_12S") or 0)
            ):
                best = packed
        if tried >= max_schedules:
            break
    if best is None:
        return {
            "accepted": False,
            "status": "PLAN_A_NO_SURVIVING_SCHEDULE",
            "actions": [],
            "wall_s": time.perf_counter() - t0,
            "tried": tried,
        }
    best["wall_s"] = time.perf_counter() - t0
    best["tried"] = tried
    return best


def _try_plan_c_catch(env, ref, *, max_rollouts: int = 20) -> dict:
    snap = make_snap(env.sim)
    rng = np.random.default_rng(SEED)
    t0 = time.perf_counter()
    try:
        search = search_catch(env.sim, snap, ref, rng=rng, max_rollouts=max_rollouts)
    except Exception as e:
        return {
            "accepted": False,
            "status": f"CATCH_EXCEPTION:{type(e).__name__}",
            "actions": [],
            "wall_s": time.perf_counter() - t0,
            "error": str(e)[:200],
        }
    best = search.get("best")
    if not best or not best.get("end_bilateral"):
        return {
            "accepted": False,
            "status": "CATCH_NO_BILATERAL",
            "actions": [],
            "wall_s": time.perf_counter() - t0,
            "funnel": search.get("funnel"),
        }
    stage = run_stage1_catch_and_lift(env.sim, snap, ref, best)
    acts = list(stage.get("stage1_actions") or [])
    # append catch capture actions already in stage1
    ok = bool(stage.get("ok"))
    return {
        "accepted": ok,
        "status": "CATCH_STAGE1_OK" if ok else str(stage.get("reason") or "CATCH_STAGE1_FAIL"),
        "actions": [np.asarray(a, np.float32) for a in acts],
        "wall_s": time.perf_counter() - t0,
        "funnel": search.get("funnel"),
        "stage1": {k: stage.get(k) for k in ("ok", "reason", "SUCCESS_NEW") if k in stage},
    }


def continue_expert_from_candidate(
    env,
    start,
    cand: dict,
    *,
    c_catch_rollouts: int = 20,
) -> dict:
    """Restore branch capture and invoke skill-appropriate expert (no forcing)."""
    branch = pickle.loads(Path(cand["capture_path"]).read_bytes())
    restore_branch_state(env, branch)
    ref = _ref_of(start)
    skill = start.skill
    m0 = state_metrics(env.sim)
    if latched(env.sim) or m0.get("latched"):
        return {
            "candidate_id": cand["candidate_id"],
            "ic_id": cand["ic_id"],
            "skill": skill,
            "expert": None,
            "accepted": False,
            "status": "ALREADY_LATCHED",
            "metrics_pre": m0,
            "SUCCESS_RELAXED_12S": False,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "n_expert_actions": 0,
            "wall_s": 0.0,
        }

    expert_actions: list[np.ndarray] = []
    expert_meta: dict[str, Any] = {}
    t0 = time.perf_counter()

    if skill == "C":
        # Prefer catch if not bilateral; else precision
        if int(m0.get("bilateral") or 0) == 0:
            catch = _try_plan_c_catch(env, ref, max_rollouts=c_catch_rollouts)
            expert_meta["catch"] = {k: v for k, v in catch.items() if k != "actions"}
            expert_actions.extend(catch.get("actions") or [])
            if catch.get("accepted"):
                # After catch+lift, try precision then handoff eval
                prec = _try_precision(env, ref, plan_id=f"p55_c_{cand['ic_id']}")
                expert_meta["precision"] = {k: v for k, v in prec.items() if k != "actions"}
                expert_actions.extend(prec.get("actions") or [])
            else:
                # Still try precision in case catch left partial contact after search side effects
                restore_branch_state(env, branch)
                prec = _try_precision(env, ref, plan_id=f"p55_c_{cand['ic_id']}")
                expert_meta["precision"] = {k: v for k, v in prec.items() if k != "actions"}
                expert_actions.extend(prec.get("actions") or [])
        else:
            prec = _try_precision(env, ref, plan_id=f"p55_c_{cand['ic_id']}")
            expert_meta["precision"] = {k: v for k, v in prec.items() if k != "actions"}
            expert_actions.extend(prec.get("actions") or [])
            if not prec.get("accepted"):
                restore_branch_state(env, branch)
                catch = _try_plan_c_catch(env, ref, max_rollouts=c_catch_rollouts)
                expert_meta["catch"] = {k: v for k, v in catch.items() if k != "actions"}
                expert_actions.extend(catch.get("actions") or [])
    else:
        # A/B/D: precision when eligible; A also tries Plan-A seed schedules from capture
        prec = _try_precision(env, ref, plan_id=f"p55_{skill}_{cand['ic_id']}")
        expert_meta["precision"] = {k: v for k, v in prec.items() if k != "actions"}
        expert_actions.extend(prec.get("actions") or [])
        if skill == "A1" and not prec.get("accepted"):
            restore_branch_state(env, branch)
            pa = _try_plan_a_schedules(env, ref, max_schedules=6)
            expert_meta["plan_a"] = {k: v for k, v in pa.items() if k not in ("actions", "outcome")}
            expert_actions[:] = list(pa.get("actions") or [])  # BC targets = expert suffix only
            if pa.get("outcome"):
                expert_meta["plan_a_outcome"] = dict(pa["outcome"])

    accepted = bool(
        (expert_meta.get("precision") or {}).get("accepted")
        or (expert_meta.get("catch") or {}).get("accepted")
        or (expert_meta.get("plan_a") or {}).get("accepted")
    )

    # Prefer Plan-A's already-evaluated outcome; do not re-eval leftover schedule state
    plan_a_out = expert_meta.get("plan_a_outcome")
    if plan_a_out is not None and (expert_meta.get("plan_a") or {}).get("accepted"):
        outcome = {
            "SUCCESS_RELAXED_12S": bool(plan_a_out.get("SUCCESS_RELAXED_12S")),
            "SUCCESS_12S": bool(plan_a_out.get("SUCCESS_12S")),
            "SUCCESS_NEW": bool(plan_a_out.get("SUCCESS_NEW")),
            "reason": plan_a_out.get("reason"),
            "pre_pos_mm": plan_a_out.get("pre_pos_mm"),
            "pre_theta": plan_a_out.get("pre_theta"),
        }
    elif "plan_a" in expert_meta and not (expert_meta.get("plan_a") or {}).get("accepted"):
        outcome = {
            "SUCCESS_RELAXED_12S": False,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "reason": (expert_meta.get("plan_a") or {}).get("status") or "PLAN_A_FAILED",
        }
    elif latched(env.sim):
        outcome = {
            "SUCCESS_RELAXED_12S": False,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "reason": "DROP_BEFORE",
        }
    else:
        # Brief grip settle then handoff eval (existing SUCCESS path)
        for _ in range(4):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            step_legal(env.sim, a)
            expert_actions.append(a.copy())
            if latched(env.sim):
                break
        if latched(env.sim):
            outcome = {
                "SUCCESS_RELAXED_12S": False,
                "SUCCESS_12S": False,
                "SUCCESS_NEW": False,
                "reason": "DROP_BEFORE",
            }
        else:
            outcome = _eval_relaxed(env, ref)

    status = (
        (expert_meta.get("plan_a") or {}).get("status")
        or (expert_meta.get("catch") or {}).get("status")
        or (expert_meta.get("precision") or {}).get("status")
        or "NO_EXPERT"
    )

    return {
        "candidate_id": cand["candidate_id"],
        "ic_id": cand["ic_id"],
        "skill": skill,
        "class": cand.get("class"),
        "step": cand.get("step"),
        "phase": cand.get("phase"),
        "metrics_pre": {
            k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
            for k, v in m0.items()
            if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_))
        },
        "expert_meta": expert_meta,
        "accepted": accepted,
        "status": status,
        "n_expert_actions": len(expert_actions),
        "expert_actions_path": None,  # filled by caller if saving
        "SUCCESS_RELAXED_12S": bool(outcome["SUCCESS_RELAXED_12S"]),
        "SUCCESS_12S": bool(outcome["SUCCESS_12S"]),
        "SUCCESS_NEW": bool(outcome["SUCCESS_NEW"]),
        "eval_reason": outcome.get("reason"),
        "pre_pos_mm": outcome.get("pre_pos_mm"),
        "pre_theta": outcome.get("pre_theta"),
        "wall_s": time.perf_counter() - t0,
        "_actions": expert_actions,  # transient
    }


def run_expert_continuation(*, c_catch_rollouts: int = 16) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cand_path = OUT / "TAKEOVER_CANDIDATES.json"
    if not cand_path.is_file():
        raise FileNotFoundError(cand_path)
    blob = json.loads(cand_path.read_text(encoding="utf-8"))
    candidates = blob["candidates"]
    starts = resolve_starts(sorted({c["ic_id"] for c in candidates}))
    env = make_env()

    results = []
    rescued_ics: set[str] = set()
    for i, cand in enumerate(candidates):
        start = starts.get(cand["ic_id"])
        if start is None:
            results.append({**cand, "accepted": False, "status": "MISSING_START", "SUCCESS_RELAXED_12S": False})
            continue
        print(
            f"[{i+1}/{len(candidates)}] expert {cand['skill']} {cand['candidate_id']} "
            f"phase={cand['phase']} bil={cand['metrics'].get('bilateral')}",
            flush=True,
        )
        row = continue_expert_from_candidate(
            env, start, cand, c_catch_rollouts=c_catch_rollouts
        )
        acts = row.pop("_actions", [])
        if row.get("SUCCESS_RELAXED_12S"):
            rescued_ics.add(cand["ic_id"])
            # save expert actions for replay
            ap = OUT / "replay" / f"expert_actions_{cand['candidate_id'].replace(':','_').replace('/','_')}.npy"
            ap.parent.mkdir(parents=True, exist_ok=True)
            if acts:
                np.save(ap, np.stack([np.asarray(a, np.float32) for a in acts]))
                row["expert_actions_path"] = str(ap)
        results.append(row)
        print(
            f"  -> accepted={row['accepted']} relaxed={row['SUCCESS_RELAXED_12S']} "
            f"status={row['status']} reason={row.get('eval_reason')}",
            flush=True,
        )

    by_skill: dict[str, dict] = {}
    for sk in ("A1", "B1", "C", "D"):
        sub = [r for r in results if r.get("skill") == sk]
        ics = {r["ic_id"] for r in sub}
        rescued = {r["ic_id"] for r in sub if r.get("SUCCESS_RELAXED_12S")}
        by_skill[sk] = {
            "n_candidates": len(sub),
            "n_ics": len(ics),
            "n_accepted": sum(1 for r in sub if r.get("accepted")),
            "n_success_relaxed": sum(1 for r in sub if r.get("SUCCESS_RELAXED_12S")),
            "n_ics_rescued": len(rescued),
            "rescue_rate_ics": len(rescued) / max(len(ics), 1),
        }

    out = {
        "n_candidates": len(results),
        "n_ics": len({r["ic_id"] for r in results}),
        "n_ics_rescued": len(rescued_ics),
        "rescued_ics": sorted(rescued_ics),
        "by_skill": by_skill,
        "results": results,
    }
    (OUT / "EXPERT_CONTINUATION_RESULTS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--c-catch-rollouts", type=int, default=16)
    args = ap.parse_args()
    rep = run_expert_continuation(c_catch_rollouts=args.c_catch_rollouts)
    print("rescued_ics", rep["n_ics_rescued"], "/", rep["n_ics"])
    print(json.dumps(rep["by_skill"], indent=2))
