"""P110 — Targeted On-Policy Corrective Imitation (token-head only).

Stages: integrity gate → roots → corrections → M111 map → smoke → pilot → eval → decision.
No P64 mutation, no backbone/residual train, no sealed panel, no PPO/GRPO, no P111.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from experiments.precision.p110_targeted_corrective_imitation.takeover_fixed import (  # noqa: E402
    DT_POLICY,
    PREFIX_DURATIONS_S,
    continue_from_state_fixed,
    multi_step_restore_determinism,
    plan_c_expert_actions_from_state,
    restore_p110_state,
    rich_actor_rollout,
    select_intervention_times,
    sync_wrapper_to_inner,
    take_p110_state,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.p104_policy import build_p104_from_p64  # noqa: E402

import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402
import experiments.precision.p106_action_aware_token_retention.run_p106 as p106  # noqa: E402

OUT = ROOT / "results/experiments/p110_targeted_corrective_imitation"
CACHE = OUT / "cache"
CKPT = OUT / "checkpoints"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P108 = ROOT / "results/experiments/p108_closed_loop_generalization_discovery"
P109 = ROOT / "results/experiments/p109a_contact_predictability_audit"
EXPECTED_SHA = "ec08f59816aa2fdd"
CL_SEED = 110
SEEDS = (0, 1, 2)
SMOKE_UPDATES = 15
PILOT_UPDATES = 400
LR = 1e-3
GRAD_CLIP = 5.0
CORRECTIVE_FRAC = 0.40  # replace this fraction of C token budget with corrective demos
ACT_TOL = p104.ACT_TOL
REPR_ERR_MAX = 0.35  # train-only representability threshold (L2 recon)
TARGET_TRAIN_C = 30
TARGET_DEV_C = 20
TARGET_DEV_PROT = 12  # aspirational A1/B1/D

# Pre-registered decision gates (BEFORE final eval)
PRE_REG = {
    "min_corrective_roots": 8,
    "min_representable_frac": 0.50,
    "min_c_dev_roots": 8,
    "min_c_lift_vs_B": 0.05,  # absolute success-rate points on paired C-dev
    "min_seeds_agreeing": 2,
    "max_protected_drop": 0.10,  # vs P64 mean over A1/B1/D
}


def utc() -> str:
    return p104.utc()


def write_json(path: Path, obj) -> None:
    p104.write_json(path, obj)


def write_csv(path: Path, rows: list[dict]) -> None:
    p104.write_csv(path, rows)


def gbi_key(s: str) -> str | None:
    m = re.search(r"gbi_\d+", str(s), re.I)
    return m.group(0).lower() if m else None


def canonical_id(tid: str) -> str:
    return re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)


def abort_blocked(reason: str, extra: dict | None = None) -> None:
    summary = {
        "case": "P110_BLOCKED_TAKEOVER_INTEGRITY",
        "reason": reason,
        "created": utc(),
        **(extra or {}),
    }
    write_json(OUT / "SUMMARY.json", summary)
    (OUT / "DECISION.md").write_text(
        f"# DECISION\n\n`P110_BLOCKED_TAKEOVER_INTEGRITY`\n\nReason: {reason}\n",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text(
        f"# FINAL_REPORT — P110\n\n**P110_BLOCKED_TAKEOVER_INTEGRITY**\n\n{reason}\n",
        encoding="utf-8",
    )
    raise SystemExit(f"P110_BLOCKED_TAKEOVER_INTEGRITY: {reason}")


# ---------------------------------------------------------------------------
# Stage 1 — Takeover integrity
# ---------------------------------------------------------------------------
def stage_integrity(p64, book, rho, device, starts: dict) -> dict:
    print("[P110] Stage 1 — takeover integrity", flush=True)
    # pick a few P108-exposed C roots for debugging only
    rows_prov = []
    if (P108 / "ROOT_PROVENANCE.csv").is_file():
        import csv

        with (P108 / "ROOT_PROVENANCE.csv").open(encoding="utf-8") as f:
            rows_prov = list(csv.DictReader(f))
    debug_tids = []
    for r in rows_prov:
        tid = r.get("traj_id") or ""
        if tid in starts and getattr(starts[tid], "skill", None) == "C":
            debug_tids.append(tid)
        if len(debug_tids) >= 4:
            break
    if len(debug_tids) < 2:
        # fallback: any C starts
        debug_tids = [t for t, s in starts.items() if getattr(s, "skill", None) == "C"][:4]
    if len(debug_tids) < 1:
        abort_blocked("no_C_starts_for_integrity")

    audit_rows = []
    all_ok = True
    for tid in debug_tids:
        start = starts[tid]
        roll = rich_actor_rollout(p64, start, book, rho, device, seed=CL_SEED)
        if roll.get("error"):
            audit_rows.append({"traj_id": tid, "status": "actor_roll_fail", "error": roll["error"]})
            all_ok = False
            continue
        # intervene at t=0 and first available later tag
        ints = select_intervention_times(roll)
        for t_int, tag in ints[:2]:
            if t_int not in roll["states_by_t"]:
                continue
            state = roll["states_by_t"][t_int]
            # determinism
            def _ef():
                return StabilizeGrpoEnv(
                    book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                )

            det = multi_step_restore_determinism(_ef, start, state, book, rho, n_steps=5)
            # actor continuation
            a_cf = continue_from_state_fixed(
                p64, start, state, book, rho, device, mode="actor", seed=CL_SEED
            )
            # online expert
            e_cf = continue_from_state_fixed(
                p64, start, state, book, rho, device, mode="expert", seed=CL_SEED, expert_seed=CL_SEED + 7
            )
            # prefix 0.5s
            p_cf = continue_from_state_fixed(
                p64,
                start,
                state,
                book,
                rho,
                device,
                mode="prefix_expert",
                seed=CL_SEED,
                prefix_s=0.5,
                expert_seed=CL_SEED + 7,
            )
            row = {
                "traj_id": tid,
                "intervention_t": t_int,
                "tag": tag,
                "orig_success": roll["SUCCESS_RELAXED_12S"],
                "actor_cf_success": a_cf["SUCCESS_RELAXED_12S"],
                "expert_cf_success": e_cf["SUCCESS_RELAXED_12S"],
                "prefix05_success": p_cf["SUCCESS_RELAXED_12S"],
                "fidelity_ok": a_cf["snapshot_fidelity_ok"],
                "wrap_restored": a_cf["wrap_restored"],
                "expert_offpolicy_caveat": e_cf["expert_offpolicy_caveat"],
                "expert_plan_status": e_cf["expert_plan_status"],
                "expert_plan_accepted": e_cf["expert_plan_accepted"],
                "n_expert_actions": e_cf["n_expert_actions_planned"],
                "determinism_ok": det["ok"],
                "bilateral_restore_err": a_cf["bilateral_restore_err"],
                "e_pos_restore_err_mm": a_cf["e_pos_restore_err_mm"],
            }
            # critical gates
            if not a_cf["snapshot_fidelity_ok"] or not det["ok"]:
                all_ok = False
                row["gate"] = "FAIL_RESTORE"
            elif e_cf["expert_offpolicy_caveat"] != 0:
                all_ok = False
                row["gate"] = "FAIL_OFFPOLICY_FLAG"
            else:
                row["gate"] = "PASS"
            audit_rows.append(row)
            print(
                f"  integrity {tid} t={t_int}/{tag} det={det['ok']} "
                f"expert={e_cf['expert_plan_status']} succE={e_cf['SUCCESS_RELAXED_12S']}",
                flush=True,
            )

    write_csv(OUT / "cache" / "integrity_cases.csv", audit_rows)
    n_pass = sum(1 for r in audit_rows if r.get("gate") == "PASS")
    n_expert_ok = sum(1 for r in audit_rows if int(r.get("expert_plan_accepted") or 0) == 1)
    report = f"""# TAKEOVER_INTEGRITY_AUDIT — P110

## Repairs vs P108
1. **WrapState** exported/restored with full state (saw_contact, V2 CtrlState, overlay flags).
2. **Inner obs buffers** synced from wrapper (`sync_wrapper_to_inner`).
3. **Expert continuation** uses state-conditioned Plan-C `search_catch` + `run_stage1_catch_and_lift`
   from the restored MuJoCo snapshot (P55 pattern). **No** `expert_acts[t]` absolute tape index.
4. Multi-step restore determinism checked (5 zero-action steps).

## Results
- cases: {len(audit_rows)}
- gate PASS: {n_pass}
- expert plan accepted: {n_expert_ok}
- all_critical_ok: {all_ok}

## Tolerances
- sim_time restore: `|Δt| < 1e-6`
- qpos restore/continue: `atol=1e-8` / `1e-7`
- bilateral restore mismatch must be 0

## Note
P108 debug roots are for implementation verification only — not fresh generalization evidence.
"""
    (OUT / "TAKEOVER_INTEGRITY_AUDIT.md").write_text(report, encoding="utf-8")
    if n_pass < 1 or not all_ok:
        abort_blocked(
            "integrity_gate_failed",
            {"n_pass": n_pass, "all_ok": all_ok, "rows": audit_rows[:8]},
        )
    # require that expert path is at least callable (accepted≥1 somewhere OR graceful fail status)
    if n_expert_ok < 1:
        # soft: expert may fail to replan from some states; still OK if caveat=0 and fidelity OK
        print("  [warn] no expert plan accepted on integrity set; continuing if restore gates passed", flush=True)
    return {"n_pass": n_pass, "n_expert_ok": n_expert_ok, "rows": audit_rows}


# ---------------------------------------------------------------------------
# Stage 1b — Roots
# ---------------------------------------------------------------------------
def generate_ics(book, rho, *, ban_g, existing_g, target, wall_s, seed, provenance: str) -> list[dict]:
    from data_generation.oracles.plan_c.expert_search import (
        build_physical_root,
        diverse_impact_configs,
        search_catch,
    )
    from envs.simulation.ground_sim import make_ground_sim

    out_dir = CACHE / "ics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    configs = diverse_impact_configs(max(target * 10, 50), seed=seed)
    rng.shuffle(configs)
    accepted = []
    t0 = time.time()
    for cfg in configs:
        if len(accepted) >= target or time.time() - t0 > wall_s:
            break
        rid = str(cfg.get("config_id") or "")
        g = gbi_key(rid) or rid
        if g in ban_g or g in existing_g:
            continue
        try:
            sim, _ = make_ground_sim()
            root = build_physical_root(sim, cfg, max_s=1.2)
            if root is None:
                continue
            fp = str(root.get("fingerprint") or hashlib.sha256(pickle.dumps(cfg)).hexdigest()[:16])
            catch = search_catch(sim, root["snap"], root["ref"], rng=rng, max_rollouts=10)
            best = catch.get("best")
            if not best:
                continue
            ep = {
                "traj_id": f"plan_c__{rid}__{fp[:12]}",
                "fingerprint": fp,
                "cfg": cfg,
                "catch_snap": root["snap"],
                "ref": root["ref"],
                "capture": best.get("capture"),
                "start_kind": "catch_snap",
                "actions": np.zeros((0, 7), np.float32),
            }
            tid = ep["traj_id"]
            st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            env = StabilizeGrpoEnv(
                book, rho, max_steps=2, t_max=100, weights=PoseRewardWeights(), use_stabilization=False
            )
            env.reset(st)
            pkl = out_dir / f"{tid}.pkl"
            pkl.write_bytes(pickle.dumps({"ep": ep}))
            accepted.append(
                {
                    "traj_id": tid,
                    "canonical_physical_id": canonical_id(tid),
                    "gbi": gbi_key(tid) or g,
                    "provenance": provenance,
                    "source": "generated_p110",
                    "pkl": str(pkl),
                    "start": st,
                }
            )
            existing_g.add(gbi_key(tid) or g)
            print(f"  [gen {provenance}] {tid} n={len(accepted)}/{target}", flush=True)
        except Exception as e:
            print(f"  [gen] skip {rid}: {type(e).__name__}: {e}", flush=True)
    return accepted


def build_roots(starts: dict, book, rho) -> tuple[list[dict], dict, dict]:
    print("[P110] Stage 1b — root provenance", flush=True)
    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    train_g = {gbi_key(x) for x in splits["c_train_phys"]} - {None}
    exp_dev_g = {gbi_key(x) for x in splits.get("c_dev_phys", [])} - {None}
    ind_g = {gbi_key(t) for t in (splits.get("ind") or {}).get("C") or []} - {None}
    ban_g = {gbi_key(t) for t in p104.eval_banned_phys()} - {None}
    # P109 generated → previously exposed
    p109_g = set()
    if (P109 / "ROOT_PROVENANCE.csv").is_file():
        import csv

        with (P109 / "ROOT_PROVENANCE.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("provenance") == "newly_selected_development":
                    p109_g.add(r.get("gbi"))

    by_g: dict[str, list] = defaultdict(list)
    for tid, st in starts.items():
        if getattr(st, "skill", None) != "C":
            continue
        g = gbi_key(tid)
        if g is None or g in ind_g:
            continue
        by_g[g].append((tid, st))

    rows = []
    for g, lst in sorted(by_g.items()):
        lst = sorted(lst, key=lambda x: (0 if str(x[0]).startswith("plan_c__") else 1, x[0]))
        tid, st = lst[0]
        if g in train_g:
            prov = "historical_training"
        elif g in exp_dev_g or g in p109_g:
            prov = "previously_exposed_development"
        elif g in ban_g:
            prov = "historical_freeze_ban_not_independent"
        else:
            prov = "schedule_other"
        rows.append(
            {
                "traj_id": tid,
                "canonical_physical_id": canonical_id(tid),
                "gbi": g,
                "provenance": prov,
                "skill": "C",
                "source": "schedule",
            }
        )

    existing = {r["gbi"] for r in rows}
    # Resume: reload previously generated P110 ICs if present (avoid regenerating)
    gen_train, gen_dev = [], []
    ic_dir = CACHE / "ics"
    if ic_dir.is_dir():
        import csv as _csv

        prior = []
        if (OUT / "ROOT_PROVENANCE.csv").is_file():
            with (OUT / "ROOT_PROVENANCE.csv").open(encoding="utf-8") as f:
                prior = list(_csv.DictReader(f))
        for pr in prior:
            if pr.get("provenance") not in ("p110_training", "p110_development"):
                continue
            tid = pr["traj_id"]
            pkl = Path(pr.get("pkl") or (ic_dir / f"{tid}.pkl"))
            if not pkl.is_file():
                pkl = ic_dir / f"{tid}.pkl"
            if not pkl.is_file():
                continue
            ep = pickle.loads(pkl.read_bytes())["ep"]
            st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            starts[tid] = st
            row = {
                "traj_id": tid,
                "canonical_physical_id": pr.get("canonical_physical_id") or canonical_id(tid),
                "gbi": pr.get("gbi") or gbi_key(tid),
                "provenance": pr["provenance"],
                "source": "generated_p110_resumed",
                "pkl": str(pkl),
            }
            rows.append(row)
            existing.add(row["gbi"])
            if pr["provenance"] == "p110_training":
                gen_train.append(row)
            else:
                gen_dev.append(row)
        print(f"  resumed ICs train={len(gen_train)} dev={len(gen_dev)}", flush=True)

    if len(gen_train) < TARGET_TRAIN_C:
        more = generate_ics(
            book, rho, ban_g=ban_g | ind_g, existing_g=set(existing),
            target=TARGET_TRAIN_C - len(gen_train), wall_s=700.0, seed=110001,
            provenance="p110_training",
        )
        for g in more:
            st = g.pop("start")
            starts[g["traj_id"]] = st
            rows.append({k: v for k, v in g.items() if k != "start"})
            existing.add(g["gbi"])
            gen_train.append(g)
    if len(gen_dev) < TARGET_DEV_C:
        more = generate_ics(
            book, rho, ban_g=ban_g | ind_g, existing_g=set(existing),
            target=TARGET_DEV_C - len(gen_dev), wall_s=700.0, seed=110002,
            provenance="p110_development",
        )
        for g in more:
            st = g.pop("start")
            starts[g["traj_id"]] = st
            rows.append({k: v for k, v in g.items() if k != "start"})
            existing.add(g["gbi"])
            gen_dev.append(g)

    # Protected-class roots from preservation demos / schedule (not sealed)
    prot_rows = []
    for skill in ("A1", "B1", "D"):
        n = 0
        for tid, st in starts.items():
            if getattr(st, "skill", None) != skill:
                continue
            g = gbi_key(tid) or canonical_id(tid)
            if gbi_key(tid) in ind_g:
                continue
            prot_rows.append(
                {
                    "traj_id": tid,
                    "canonical_physical_id": canonical_id(tid),
                    "gbi": g,
                    "provenance": "protected_rehearsal_or_dev",
                    "skill": skill,
                    "source": "schedule",
                }
            )
            n += 1
            if n >= TARGET_DEV_PROT:
                break

    all_rows = rows + prot_rows
    write_csv(OUT / "ROOT_PROVENANCE.csv", all_rows)

    train_c = [r for r in rows if r["provenance"] in ("p110_training", "historical_training")]
    # Prefer generated train; pad with historical if needed
    train_c_ids = [r for r in rows if r["provenance"] == "p110_training"]
    if len(train_c_ids) < 12:
        train_c_ids = train_c_ids + [r for r in rows if r["provenance"] == "historical_training"]
    dev_c = [r for r in rows if r["provenance"] == "p110_development"]
    # previously exposed only for debug, not scored as fresh
    meta = {
        "n_train_c": len(train_c_ids),
        "n_dev_c": len(dev_c),
        "n_gen_train": len(gen_train),
        "n_gen_dev": len(gen_dev),
        "sealed_excluded": sorted(ind_g),
        "note": "Dev roots selected before inspecting P110 model outcomes.",
    }
    write_json(OUT / "POOL_META.json", meta)
    split_map = {}
    for r in train_c_ids:
        split_map[r["canonical_physical_id"]] = "train"
    for r in dev_c:
        split_map[r["canonical_physical_id"]] = "dev"
    return all_rows, meta, {"train_c": train_c_ids, "dev_c": dev_c, "prot": prot_rows, "split_map": split_map, "starts": starts}


# ---------------------------------------------------------------------------
# Stage 2 — Corrections + conventional demos
# ---------------------------------------------------------------------------
def collect_conventional_expert_demo(start, book, rho, device, *, seed: int) -> dict | None:
    """Nominal state-conditioned expert from IC (not Actor-induced)."""
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    try:
        env.reset(start)
    except Exception:
        return None
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    sync_wrapper_to_inner(env)
    from experiments.precision.p73_first_contact_bilateral.controller import WrapState

    wrap = WrapState()
    state0 = take_p110_state(env, wrap, meta={"tag": "ic"})
    plan = plan_c_expert_actions_from_state(env, seed=seed, max_rollouts=12)
    wrap = restore_p110_state(env, state0, start)
    _ea = plan.get("actions")
    acts = np.zeros((0, 7), np.float32) if _ea is None else np.asarray(_ea, np.float32).reshape(-1, 7)
    if acts.size == 0:
        return None
    from experiments.precision.p66_c_curriculum_ppo.expert_drive import step_raw

    obs_list, prev_list, act_list = [], [], []
    done = False
    info = {}
    for i, a in enumerate(acts):
        if done:
            break
        obs_list.append(env.obs_seq[-1].astype(np.float32).copy())
        prev_list.append(env.prev_seq[-1].astype(np.float32).copy())
        act_list.append(a.copy())
        _r, done, info = step_raw(env, a)
        sync_wrapper_to_inner(env)
    if not act_list:
        return None
    return {
        "traj_id": start.traj_id,
        "phys_root": canonical_id(start.traj_id),
        "skill": "C",
        "role": "conventional_expert",
        "actions": np.stack(act_list).astype(np.float32),
        "obs": np.stack(obs_list).astype(np.float32),
        "prev": np.stack(prev_list).astype(np.float32),
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "expert_status": plan.get("status"),
    }


def best_token_for_action(book, a, rho) -> dict:
    Z = p104.feasible_tokens(book, a, rho)
    if not Z:
        _, e = p104.best_under_token(book, 0, a, rho)
        return {"z_tgt": -1, "category": "D", "recon_err": float(e), "n_feasible": 0, "supervise": 0}
    best_k, best_e = None, 1e9
    for k in Z:
        _, e = p104.best_under_token(book, k, a, rho)
        pe = p104.proto_action_l2(book, k, a)
        score = e + 1e-6 * pe
        if score < best_e:
            best_e, best_k = score, k
    return {
        "z_tgt": int(best_k),
        "category": "A",
        "recon_err": float(best_e),
        "n_feasible": len(Z),
        "supervise": int(best_e <= REPR_ERR_MAX),
    }


def stage_corrections(p64, book, rho, device, pools: dict) -> dict:
    print("[P110] Stage 2 — Actor failures + expert corrections", flush=True)
    demos_path = CACHE / "demos.pkl"
    if demos_path.is_file() and (OUT / "CORRECTION_STATS.json").is_file():
        blob = pickle.loads(demos_path.read_bytes())
        stats = json.loads((OUT / "CORRECTION_STATS.json").read_text(encoding="utf-8"))
        print(f"  resumed demos: {stats}", flush=True)
        return {
            "conventional": blob["conventional"],
            "corrective": blob["corrective"],
            "pres": p104.load_preservation_demos(),
            "stats": stats,
        }
    starts = pools["starts"]
    train_c = pools["train_c"]
    rescue_rows = []
    prefix_rows = []
    manifest = []
    corrective_demos = []
    conventional_demos = []
    repr_rows = []

    for i, r in enumerate(train_c):
        tid = r["traj_id"]
        if tid not in starts:
            pkl = r.get("pkl")
            if pkl and Path(pkl).is_file():
                ep = pickle.loads(Path(pkl).read_bytes())["ep"]
                starts[tid] = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
            else:
                continue
        start = starts[tid]
        print(f"  train-root {i+1}/{len(train_c)} {tid}", flush=True)
        # conventional expert demo
        conv = collect_conventional_expert_demo(start, book, rho, device, seed=CL_SEED + i)
        if conv is not None:
            # map tokens
            toks = []
            metas = []
            for a in conv["actions"]:
                m = best_token_for_action(book, a, rho)
                toks.append(m["z_tgt"] if m["supervise"] else 0)
                metas.append(m)
            conv["tokens"] = np.asarray(toks, np.int64)
            conv["step_meta"] = [
                {
                    "category": m["category"] if m["supervise"] else "D",
                    "z_tgt": m["z_tgt"],
                    "supervise_token": m["supervise"],
                    "recon_err": m["recon_err"],
                    "t": ti,
                }
                for ti, m in enumerate(metas)
            ]
            conventional_demos.append(conv)

        roll = rich_actor_rollout(p64, start, book, rho, device, seed=CL_SEED)
        if roll.get("error"):
            continue
        # only seek corrections if Actor failed (or always sample interventions for Type analysis)
        ints = select_intervention_times(roll)[:2]  # bounded diversity per root
        for t_int, tag in ints:
            if t_int not in roll["states_by_t"]:
                continue
            state = roll["states_by_t"][t_int]
            # A continue actor
            a_cf = continue_from_state_fixed(
                p64, start, state, book, rho, device, mode="actor", seed=CL_SEED
            )
            # B full expert
            e_cf = continue_from_state_fixed(
                p64, start, state, book, rho, device, mode="expert", seed=CL_SEED, expert_seed=CL_SEED + 17 + t_int
            )
            rescue_type = "III"
            if int(e_cf["SUCCESS_RELAXED_12S"]) == 1:
                rescue_type = "I"
            # C short prefixes (only if expert plan accepted)
            prefix_ok = {}
            best_prefix = None
            if int(e_cf.get("expert_plan_accepted") or 0) == 1:
                for dur in PREFIX_DURATIONS_S:
                    p_cf = continue_from_state_fixed(
                        p64,
                        start,
                        state,
                        book,
                        rho,
                        device,
                        mode="prefix_expert",
                        seed=CL_SEED,
                        prefix_s=dur,
                        expert_seed=CL_SEED + 17 + t_int,
                    )
                    prefix_ok[dur] = int(p_cf["SUCCESS_RELAXED_12S"])
                    prefix_rows.append(
                        {
                            "traj_id": tid,
                            "phys_root": r["canonical_physical_id"],
                            "t": t_int,
                            "tag": tag,
                            "prefix_s": dur,
                            "SUCCESS_RELAXED_12S": p_cf["SUCCESS_RELAXED_12S"],
                            "expert_steps_used": p_cf["expert_steps_used"],
                            "expert_plan_status": p_cf["expert_plan_status"],
                        }
                    )
                    if p_cf["SUCCESS_RELAXED_12S"] and best_prefix is None:
                        best_prefix = (dur, p_cf)
            if rescue_type != "I" and best_prefix is not None:
                rescue_type = "II"

            rescue_rows.append(
                {
                    "traj_id": tid,
                    "phys_root": r["canonical_physical_id"],
                    "t": t_int,
                    "tag": tag,
                    "actor_orig_success": roll["SUCCESS_RELAXED_12S"],
                    "continue_actor_success": a_cf["SUCCESS_RELAXED_12S"],
                    "expert_continuation_success": e_cf["SUCCESS_RELAXED_12S"],
                    "prefix_0.2": prefix_ok.get(0.2, 0),
                    "prefix_0.5": prefix_ok.get(0.5, 0),
                    "prefix_1.0": prefix_ok.get(1.0, 0),
                    "rescue_type": rescue_type,
                    "expert_plan_status": e_cf["expert_plan_status"],
                    "n_expert_actions": e_cf["n_expert_actions_planned"],
                    "contact_bilateral": (state.get("contact") or {}).get("bilateral"),
                }
            )

            # Primary corrective supervision: Type I full expert rescue actions from matched state
            if rescue_type in ("I", "II") and int(e_cf.get("expert_plan_accepted") or 0) == 1:
                # Re-plan and record supervised prefix from restored state with causal Actor history
                env = StabilizeGrpoEnv(
                    book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
                )
                env.reset(start)
                restore_p110_state(env, state, start)
                plan = plan_c_expert_actions_from_state(env, seed=CL_SEED + 17 + t_int, max_rollouts=12)
                restore_p110_state(env, state, start)
                _ea = plan.get("actions")
                acts = np.zeros((0, 7), np.float32) if _ea is None else np.asarray(_ea, np.float32).reshape(-1, 7)
                # For Type II, keep only short prefix duration actions as supervision
                if rescue_type == "II" and best_prefix is not None:
                    k = int(round(best_prefix[0] / DT_POLICY))
                    acts = acts[:k]
                if acts.size == 0:
                    continue
                from experiments.precision.p66_c_curriculum_ppo.expert_drive import step_raw

                # history windows: use restored causal history, then append as expert steps execute
                obs_win = [np.asarray(x, np.float32).copy() for x in state["obs_seq"]]
                prev_win = [np.asarray(x, np.float32).copy() for x in state["prev_seq"]]
                demo_obs, demo_prev, demo_act, demo_tok, demo_meta = [], [], [], [], []
                for ai, a in enumerate(acts):
                    # teacher-forcing style: supervise from current causal window
                    m = best_token_for_action(book, a, rho)
                    repr_rows.append(
                        {
                            "traj_id": tid,
                            "phys_root": r["canonical_physical_id"],
                            "t_int": t_int,
                            "tag": tag,
                            "rescue_type": rescue_type,
                            "step": ai,
                            "category": "D" if not m["supervise"] else m["category"],
                            "recon_err": m["recon_err"],
                            "n_feasible": m["n_feasible"],
                            "z_tgt": m["z_tgt"],
                            "supervise": m["supervise"],
                        }
                    )
                    if not m["supervise"]:
                        # still step env to keep physical trajectory, but do not supervise
                        _r, done, info = step_raw(env, a)
                        sync_wrapper_to_inner(env)
                        obs_win.append(env.obs_seq[-1].astype(np.float32).copy())
                        prev_win.append(env.prev_seq[-1].astype(np.float32).copy())
                        if done:
                            break
                        continue
                    demo_obs.append(np.stack(obs_win[-160:]).astype(np.float32))
                    demo_prev.append(np.stack(prev_win[-160:]).astype(np.float32))
                    demo_act.append(a.copy())
                    demo_tok.append(m["z_tgt"])
                    demo_meta.append(
                        {
                            "category": m["category"],
                            "z_tgt": m["z_tgt"],
                            "supervise_token": 1,
                            "recon_err": m["recon_err"],
                            "t": ai,
                            "rescue_type": rescue_type,
                            "tag": tag,
                        }
                    )
                    _r, done, info = step_raw(env, a)
                    sync_wrapper_to_inner(env)
                    obs_win.append(env.obs_seq[-1].astype(np.float32).copy())
                    prev_win.append(env.prev_seq[-1].astype(np.float32).copy())
                    if done:
                        break
                if demo_tok:
                    # Store as flat per-step demos matching P104 format (T,39)/(T,7)
                    # Use last-obs/prev at each supervised step from windows
                    flat_obs = np.stack([w[-1] for w in demo_obs]).astype(np.float32)
                    flat_prev = np.stack([w[-1] for w in demo_prev]).astype(np.float32)
                    corrective_demos.append(
                        {
                            "traj_id": f"{tid}__corr_t{t_int}_{tag}",
                            "phys_root": r["canonical_physical_id"],
                            "skill": "C",
                            "role": "corrective",
                            "rescue_type": rescue_type,
                            "tag": tag,
                            "t_int": t_int,
                            "actions": np.stack(demo_act).astype(np.float32),
                            "tokens": np.asarray(demo_tok, np.int64),
                            "obs": flat_obs,
                            "prev": flat_prev,
                            "step_meta": demo_meta,
                            # keep full windows for optional causal CE
                            "obs_wins": demo_obs,
                            "prev_wins": demo_prev,
                        }
                    )
                    manifest.append(
                        {
                            "traj_id": tid,
                            "phys_root": r["canonical_physical_id"],
                            "t_int": t_int,
                            "tag": tag,
                            "rescue_type": rescue_type,
                            "n_supervised_steps": len(demo_tok),
                            "n_planned_actions": int(len(acts)),
                        }
                    )

    write_csv(OUT / "EXPERT_RESCUE_RESULTS.csv", rescue_rows)
    write_csv(OUT / "SHORT_PREFIX_RESCUE.csv", prefix_rows)
    write_csv(OUT / "CORRECTION_DATASET_MANIFEST.csv", manifest)
    write_csv(OUT / "M111_CORRECTION_REPRESENTABILITY.csv", repr_rows)
    # persist demos
    demos_path = CACHE / "demos.pkl"
    demos_path.write_bytes(
        pickle.dumps(
            {
                "conventional": conventional_demos,
                "corrective": corrective_demos,
            }
        )
    )
    n_corr_roots = len({d["phys_root"] for d in corrective_demos})
    n_repr = sum(1 for r in repr_rows if int(r.get("supervise") or 0) == 1)
    frac_repr = (n_repr / max(1, len(repr_rows))) if repr_rows else 0.0
    stats = {
        "n_conventional_demos": len(conventional_demos),
        "n_corrective_demos": len(corrective_demos),
        "n_corrective_roots": n_corr_roots,
        "representable_frac": frac_repr,
        "n_type_I": sum(1 for r in rescue_rows if r["rescue_type"] == "I"),
        "n_type_II": sum(1 for r in rescue_rows if r["rescue_type"] == "II"),
        "n_type_III": sum(1 for r in rescue_rows if r["rescue_type"] == "III"),
    }
    write_json(OUT / "CORRECTION_STATS.json", stats)
    print(f"  correction stats: {stats}", flush=True)
    if n_corr_roots < 1 or frac_repr < 0.15:
        write_json(
            OUT / "SUMMARY.json",
            {"case": "P110_INVALID", "reason": "insufficient_representable_corrections", **stats},
        )
        (OUT / "DECISION.md").write_text(
            "# DECISION\n\n`P110_INVALID`\n\nInsufficient representable corrective data.\n",
            encoding="utf-8",
        )
        raise SystemExit("P110_INVALID: insufficient corrective data")
    return {
        "conventional": conventional_demos,
        "corrective": corrective_demos,
        "pres": p104.load_preservation_demos(),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def demo_token_ce(actor, demo, device) -> torch.Tensor:
    """CE on supervised tokens using causal windows when available."""
    if demo.get("obs_wins"):
        losses = []
        for i, meta in enumerate(demo["step_meta"]):
            if not meta.get("supervise_token"):
                continue
            z = int(meta["z_tgt"])
            if z < 0:
                continue
            obs = torch.as_tensor(demo["obs_wins"][i], device=device, dtype=torch.float32)
            prev = torch.as_tensor(demo["prev_wins"][i], device=device, dtype=torch.float32)
            logits, _, _ = actor(obs, prev)
            losses.append(F.cross_entropy(logits[-1].unsqueeze(0), torch.tensor([z], device=device)))
        if not losses:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()
    # flat P104-style
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    logits, _, _ = actor(obs, prev)
    ys = []
    idx = []
    for t, meta in enumerate(demo.get("step_meta") or []):
        if meta.get("supervise_token") and int(meta.get("z_tgt", -1)) >= 0:
            ys.append(int(meta["z_tgt"]))
            idx.append(t)
    if not ys:
        # fallback: all tokens field
        y = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
        mask = y >= 0
        if not bool(mask.any()):
            return torch.zeros((), device=device)
        return F.cross_entropy(logits[mask], y[mask])
    y = torch.as_tensor(ys, device=device, dtype=torch.long)
    return F.cross_entropy(logits[idx], y)


def count_supervised_tokens(demos: list[dict]) -> int:
    n = 0
    for d in demos:
        for m in d.get("step_meta") or []:
            if m.get("supervise_token"):
                n += 1
        if not d.get("step_meta"):
            n += int(len(d.get("tokens") or []))
    return n


def param_grad_vec(module) -> torch.Tensor:
    grads = [p.grad.detach().reshape(-1) for p in module.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        return torch.zeros(1, device=next(module.parameters()).device)
    return torch.cat(grads)


def calibrate_kl_lambda(p64, actor_tmp, corr_batch, pres_batch, device) -> float:
    """Train-data-only gradient scale match: λ so ‖∇L_corr‖ ≈ λ‖∇L_KL‖ median."""
    ratios = []
    for _ in range(6):
        actor = copy.deepcopy(actor_tmp).to(device)
        for p in actor.parameters():
            if p.requires_grad:
                p.data.add_(1e-3 * torch.randn_like(p.data))
        actor.zero_grad(set_to_none=True)
        Lc = demo_token_ce(actor, corr_batch[0], device)
        if float(Lc.detach()) > 0:
            Lc.backward()
        g_c = param_grad_vec(actor)
        n_c = float(g_c.norm().item()) if g_c.numel() else 0.0
        actor.zero_grad(set_to_none=True)
        Lk = torch.zeros((), device=device)
        for dp in pres_batch[:3]:
            kli, _ = p106.token_kl_retention(actor, p64, dp, device)
            Lk = Lk + kli
        Lk = Lk / max(1, min(3, len(pres_batch)))
        if float(Lk.detach()) > 0:
            Lk.backward()
        g_k = param_grad_vec(actor)
        n_k = float(g_k.norm().item()) if g_k.numel() else 0.0
        if n_k > 1e-12:
            ratios.append(n_c / n_k)
    if not ratios:
        return 1.0
    return float(np.clip(np.median(ratios), 0.05, 50.0))


def train_arm(
    arm: str,
    seed: int,
    p64,
    book,
    rho,
    device,
    *,
    conventional: list,
    corrective: list,
    pres: list,
    lam_kl: float,
    updates: int,
    corrective_frac: float,
) -> tuple:
    seed_all(seed, deterministic_torch=True)
    actor = build_p104_from_p64(p64, "token_only", device)
    # verify freeze
    trainable = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    assert all("logits" in n for n, _ in trainable), trainable
    assert actor.delta is None
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    rng = np.random.default_rng(seed)
    by_skill = {"A1": [], "B1": [], "D": []}
    for d in p104.demos_group(pres):
        by_skill[d["skill"]].append(d)
    conv_c = [d for d in conventional if d["skill"] == "C" and count_supervised_tokens([d]) > 0]
    corr_c = [d for d in corrective if count_supervised_tokens([d]) > 0]
    if not conv_c and not corr_c:
        raise RuntimeError("no C demos for training")
    # matched token budget accounting
    logs = []
    actor.train()
    actor.base.eval()
    t0 = time.time()
    n_conv_used = 0
    n_corr_used = 0
    for u in range(1, updates + 1):
        # C supervision mixture
        use_corr = arm == "C" and corr_c and (rng.random() < corrective_frac)
        if use_corr:
            dc = corr_c[int(rng.integers(0, len(corr_c)))]
            n_corr_used += 1
        else:
            pool = conv_c if conv_c else corr_c
            dc = pool[int(rng.integers(0, len(pool)))]
            n_conv_used += 1
        L_corr = demo_token_ce(actor, dc, device)
        # retention on protected
        pres_batch = []
        for sk in ("A1", "B1", "D"):
            if by_skill[sk]:
                pres_batch.append(by_skill[sk][int(rng.integers(0, len(by_skill[sk])))])
        L_kl = torch.zeros((), device=device)
        for dp in pres_batch:
            kli, _ = p106.token_kl_retention(actor, p64, dp, device)
            L_kl = L_kl + kli
        if pres_batch:
            L_kl = L_kl / len(pres_batch)
        loss = L_corr + float(lam_kl) * L_kl
        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u == 1 or u == updates or u % 50 == 0:
            logs.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "update": u,
                    "loss": float(loss.detach()),
                    "L_corr": float(L_corr.detach()),
                    "L_kl": float(L_kl.detach()),
                    "elapsed_s": time.time() - t0,
                }
            )
            print(
                f"  [P110-{arm} s={seed}] u={u}/{updates} loss={logs[-1]['loss']:.4f} "
                f"corr={logs[-1]['L_corr']:.4f} kl={logs[-1]['L_kl']:.5f}",
                flush=True,
            )
    path = CKPT / f"arm{arm}_seed{seed}_final.pt"
    torch.save(
        {
            "arm": arm,
            "seed": seed,
            "update": updates,
            "p64_sha": EXPECTED_SHA,
            "lambda_kl": float(lam_kl),
            "corrective_frac": float(corrective_frac) if arm == "C" else 0.0,
            "actor": actor.state_dict(),
            "n_conv_steps": n_conv_used,
            "n_corr_steps": n_corr_used,
            "trainable": "logits_only",
        },
        path,
    )
    actor.eval()
    return actor, logs, path, {"n_conv_steps": n_conv_used, "n_corr_steps": n_corr_used}


@torch.no_grad()
def eval_closed_loop(actor, start, book, rho, device, *, seed: int) -> dict:
    return p104.closed_loop(actor, start, book, rho, device, seed=seed)


def stage_train_eval(p64, book, rho, device, pools, data) -> dict:
    print("[P110] Stage 3 — smoke + pilot training", flush=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    conventional = data["conventional"]
    corrective = data["corrective"]
    pres = data["pres"]

    actor0 = build_p104_from_p64(p64, "token_only", device)
    pool_c = corrective[:3] or conventional[:3]
    pres_g = p104.demos_group(pres)
    lam_kl = calibrate_kl_lambda(p64, actor0, pool_c or conventional[:1], pres_g, device)
    print(f"  calibrated lambda_kl={lam_kl:.4f}", flush=True)

    # Smoke
    print("  smoke B/C...", flush=True)
    train_arm("B", 99, p64, book, rho, device, conventional=conventional, corrective=corrective,
              pres=pres, lam_kl=lam_kl, updates=SMOKE_UPDATES, corrective_frac=0.0)
    train_arm("C", 99, p64, book, rho, device, conventional=conventional, corrective=corrective,
              pres=pres, lam_kl=lam_kl, updates=SMOKE_UPDATES, corrective_frac=CORRECTIVE_FRAC)

    budget_doc = f"""# TRAINING_BUDGET_COMPARISON — P110

| Item | Arm B | Arm C |
|---|---|---|
| Init | P64 `{EXPECTED_SHA}` | same |
| Trainable | token logits only | same |
| Updates | {PILOT_UPDATES} × seeds {list(SEEDS)} | same |
| LR / clip | {LR} / {GRAD_CLIP} | same |
| C supervision | conventional expert demos | replace {CORRECTIVE_FRAC:.0%} draws with corrective |
| Protected rehearsal | identical A1/B1/D | identical |
| λ_KL (train-calibrated) | {lam_kl:.4f} | {lam_kl:.4f} |
| Residual / backbone | frozen | frozen |

Primary variable: **supervision distribution** (corrective vs conventional), not budget size.
"""
    (OUT / "TRAINING_BUDGET_COMPARISON.md").write_text(budget_doc, encoding="utf-8")
    write_json(
        OUT / "TRAINING_CONFIGS.json",
        {
            "p64_sha": EXPECTED_SHA,
            "seeds": list(SEEDS),
            "smoke_updates": SMOKE_UPDATES,
            "pilot_updates": PILOT_UPDATES,
            "lr": LR,
            "lambda_kl": lam_kl,
            "corrective_frac": CORRECTIVE_FRAC,
            "repr_err_max": REPR_ERR_MAX,
            "pre_registered": PRE_REG,
        },
    )

    # Pilot
    actors = {"A": p64}
    train_meta = {}
    for seed in SEEDS:
        for arm, frac in (("B", 0.0), ("C", CORRECTIVE_FRAC)):
            actor, logs, path, meta = train_arm(
                arm, seed, p64, book, rho, device,
                conventional=conventional, corrective=corrective, pres=pres,
                lam_kl=lam_kl, updates=PILOT_UPDATES, corrective_frac=frac,
            )
            actors[f"{arm}{seed}"] = actor
            train_meta[f"{arm}{seed}"] = {"path": str(path), **meta, "logs_tail": logs[-3:]}
            write_csv(OUT / "logs" / f"arm{arm}_seed{seed}.csv", logs)

    print("[P110] Stage 4 — closed-loop evaluation", flush=True)
    starts = pools["starts"]
    eval_rows = []
    # Resolve eval starts
    def resolve(row):
        tid = row["traj_id"]
        if tid in starts:
            return starts[tid]
        pkl = row.get("pkl")
        if pkl and Path(pkl).is_file():
            ep = pickle.loads(Path(pkl).read_bytes())["ep"]
            st = MixedStart(skill=row.get("skill") or "C", traj_id=tid, family=row.get("skill") or "C", kind="ep", ep=ep)
            starts[tid] = st
            return st
        return starts.get(tid)

    panels = {
        "C_dev": pools["dev_c"],
        "C_train_debug": pools["train_c"][:5],
        "prot": pools["prot"],
    }
    # Arm A + each B/C seed
    eval_specs = [("A", -1, p64)]
    for seed in SEEDS:
        eval_specs.append((f"B", seed, actors[f"B{seed}"]))
        eval_specs.append((f"C", seed, actors[f"C{seed}"]))

    for panel_name, rows in panels.items():
        for r in rows:
            st = resolve(r)
            if st is None:
                continue
            skill = r.get("skill") or "C"
            for arm, seed, actor in eval_specs:
                out = eval_closed_loop(actor, st, book, rho, device, seed=CL_SEED)
                eval_rows.append(
                    {
                        "arm": arm,
                        "train_seed": seed,
                        "panel": panel_name,
                        "skill": skill,
                        "traj_id": r["traj_id"],
                        "phys_root": r["canonical_physical_id"],
                        "provenance": r.get("provenance"),
                        "SUCCESS_RELAXED_12S": int(out.get("SUCCESS_RELAXED_12S") or 0),
                        "ep_len": out.get("ep_len"),
                        "fail_reason": out.get("fail_reason"),
                        "fresh_dev": int(panel_name == "C_dev"),
                    }
                )
                print(
                    f"  eval {arm}s{seed} {panel_name} {r['traj_id']}: "
                    f"{eval_rows[-1]['SUCCESS_RELAXED_12S']}",
                    flush=True,
                )

    write_csv(OUT / "CLOSED_LOOP_EVALUATION.csv", eval_rows)

    # Protected retention summary
    prot_rows = []
    for skill in ("A1", "B1", "D"):
        for arm in ("A", "B", "C"):
            sub = [e for e in eval_rows if e["skill"] == skill and e["arm"] == arm and e["panel"] == "prot"]
            if arm != "A":
                # average over seeds
                pass
            n = len(sub)
            s = sum(int(e["SUCCESS_RELAXED_12S"]) for e in sub)
            prot_rows.append(
                {
                    "skill": skill,
                    "arm": arm,
                    "n": n,
                    "successes": s,
                    "rate": (s / n) if n else float("nan"),
                }
            )
    write_csv(OUT / "PROTECTED_RETENTION.csv", prot_rows)

    return {"eval_rows": eval_rows, "prot_rows": prot_rows, "lam_kl": lam_kl, "train_meta": train_meta, "actors": actors}


# ---------------------------------------------------------------------------
# Decision + reports
# ---------------------------------------------------------------------------
def decide(data_stats, eval_rows, prot_rows, integrity, pools) -> dict:
    # C_dev paired rates
    def rate(arm, seed=None):
        sub = [e for e in eval_rows if e["panel"] == "C_dev" and e["arm"] == arm]
        if seed is not None:
            sub = [e for e in sub if int(e["train_seed"]) == int(seed)]
        n = len(sub)
        s = sum(int(e["SUCCESS_RELAXED_12S"]) for e in sub)
        roots = {}
        for e in sub:
            roots.setdefault(e["phys_root"], []).append(int(e["SUCCESS_RELAXED_12S"]))
        root_rates = {k: float(np.mean(v)) for k, v in roots.items()}
        return (s / n if n else float("nan")), n, root_rates

    a_rate, a_n, a_roots = rate("A")
    seed_lifts = []
    for seed in SEEDS:
        b_rate, _, b_roots = rate("B", seed)
        c_rate, _, c_roots = rate("C", seed)
        # paired root lift
        common = set(b_roots) & set(c_roots)
        lifts = [c_roots[r] - b_roots[r] for r in common]
        mean_lift = float(np.mean(lifts)) if lifts else float("nan")
        seed_lifts.append(
            {
                "seed": seed,
                "B": b_rate,
                "C": c_rate,
                "lift_C_minus_B": mean_lift,
                "n_common_roots": len(common),
            }
        )
    mean_lift = float(np.nanmean([x["lift_C_minus_B"] for x in seed_lifts])) if seed_lifts else float("nan")
    n_agree = sum(1 for x in seed_lifts if np.isfinite(x["lift_C_minus_B"]) and x["lift_C_minus_B"] >= PRE_REG["min_c_lift_vs_B"])

    # protected drop vs A
    prot_drop = {}
    for skill in ("A1", "B1", "D"):
        a = next((p for p in prot_rows if p["skill"] == skill and p["arm"] == "A"), None)
        c = next((p for p in prot_rows if p["skill"] == skill and p["arm"] == "C"), None)
        if a and c and a["n"] and c["n"]:
            prot_drop[skill] = float(c["rate"]) - float(a["rate"])
        else:
            prot_drop[skill] = float("nan")
    worst_drop = min([v for v in prot_drop.values() if np.isfinite(v)], default=0.0)

    n_dev = len(pools["dev_c"])
    n_corr_roots = int(data_stats.get("n_corrective_roots") or 0)
    frac_repr = float(data_stats.get("representable_frac") or 0)

    answers = {
        "Q1_takeover_valid": True,
        "Q2_n_corrective_roots": n_corr_roots,
        "Q3_representable_frac": frac_repr,
        "Q4_C_beats_B_on_unseen": bool(np.isfinite(mean_lift) and mean_lift >= PRE_REG["min_c_lift_vs_B"] and n_agree >= PRE_REG["min_seeds_agreeing"]),
        "Q5_beats_P64": bool(np.nanmean([x["C"] for x in seed_lifts]) > a_rate + 0.02) if seed_lifts else False,
        "Q6_consistent_seeds_roots": bool(n_agree >= PRE_REG["min_seeds_agreeing"]),
        "Q7_protected_ok": bool(worst_drop >= -PRE_REG["max_protected_drop"]),
        "Q8_mechanism_supported": False,  # filled after mechanism write
        "Q9_capacity_insufficient": False,
    }

    if n_corr_roots < PRE_REG["min_corrective_roots"] or n_dev < PRE_REG["min_c_dev_roots"] or frac_repr < PRE_REG["min_representable_frac"]:
        decision = "P110_INCONCLUSIVE"
    elif answers["Q4_C_beats_B_on_unseen"] and answers["Q7_protected_ok"]:
        decision = "P110_CORRECTIVE_IMITATION_SUPPORTED"
    elif (not answers["Q7_protected_ok"]) and np.nanmean([x["C"] for x in seed_lifts]) > a_rate:
        decision = "P110_RETENTION_REGRESSION"
    elif np.isfinite(mean_lift) and mean_lift > 0:
        decision = "P110_CORRECTIVE_IMITATION_WEAK"
    else:
        decision = "P110_CORRECTIVE_IMITATION_WEAK"

    return {
        "decision": decision,
        "answers": answers,
        "a_rate": a_rate,
        "seed_lifts": seed_lifts,
        "mean_lift_C_minus_B": mean_lift,
        "n_agree_seeds": n_agree,
        "prot_drop": prot_drop,
        "n_dev": n_dev,
    }


def mechanism_analysis(eval_rows, rescue_rows_path: Path) -> str:
    # Lightweight: compare which C_dev roots C wins vs B
    by = defaultdict(dict)
    for e in eval_rows:
        if e["panel"] != "C_dev":
            continue
        key = (e["phys_root"], int(e["train_seed"]))
        by[key][e["arm"]] = int(e["SUCCESS_RELAXED_12S"])
    wins, losses = [], []
    for (root, seed), d in by.items():
        if "B" in d and "C" in d:
            if d["C"] > d["B"]:
                wins.append((root, seed))
            elif d["C"] < d["B"]:
                losses.append((root, seed))
    text = f"""# MECHANISM_ANALYSIS — P110

## Root-level C vs B on C_dev
- wins (C>B): {len(wins)} {wins[:12]}
- losses (C<B): {len(losses)} {losses[:12]}

## Interpretation caveats
- Token-only updates; frozen residual/backbone.
- Corrective demos use Actor-induced causal histories (not synthetic expert history).
- A token change preceding success is not alone causal proof; matched-state CF beyond this pilot was limited to data collection.

## Q8
Mechanism support is **tentative** if wins concentrate on roots whose tags overlap corrective tags
(recovery_entry / contact_mode_transition / regrasp). Without per-root tag alignment tables dense enough,
Q8 remains False unless wins >> losses.
"""
    (OUT / "MECHANISM_ANALYSIS.md").write_text(text, encoding="utf-8")
    (OUT / "FAILURE_CASES.md").write_text(
        f"""# FAILURE_CASES — P110

## C regressions vs B
{losses[:20]}

## Notes
Type III expert failures are **not** labeled physically unrecoverable — expert may lack off-policy coverage.
""",
        encoding="utf-8",
    )
    return text


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = p104.sha16(P64_ACTOR)
    if sha != EXPECTED_SHA:
        abort_blocked(f"p64_sha_mismatch:{sha}")
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)

    starts = p104.lookup_starts()
    integrity = stage_integrity(p64, book, rho, device, starts)
    all_rows, pool_meta, pools = build_roots(starts, book, rho)
    data = stage_corrections(p64, book, rho, device, pools)
    te = stage_train_eval(p64, book, rho, device, pools, data)
    mechanism_analysis(te["eval_rows"], OUT / "EXPERT_RESCUE_RESULTS.csv")
    decision_blob = decide(data["stats"], te["eval_rows"], te["prot_rows"], integrity, pools)
    # Q8 heuristic
    decision_blob["answers"]["Q8_mechanism_supported"] = bool(
        decision_blob["decision"] == "P110_CORRECTIVE_IMITATION_SUPPORTED"
    )
    # Q9: if corrective representable but both B and C ≈ P64 low on C → capacity may be insufficient
    if (
        data["stats"]["representable_frac"] >= 0.5
        and np.nanmean([x["C"] for x in decision_blob["seed_lifts"]]) < 0.15
        and decision_blob["a_rate"] < 0.15
    ):
        decision_blob["answers"]["Q9_capacity_insufficient"] = True

    summary = {
        "case": decision_blob["decision"],
        "created": utc(),
        "p64_sha": sha,
        "trained_actor": True,
        "backbone_unfrozen": False,
        "residual_trained": False,
        "sealed_panel_opened": False,
        "promoted_baseline": False,
        "integrity": {"n_pass": integrity["n_pass"], "n_expert_ok": integrity["n_expert_ok"]},
        "pool_meta": pool_meta,
        "correction_stats": data["stats"],
        "lambda_kl": te["lam_kl"],
        "seed_lifts": decision_blob["seed_lifts"],
        "mean_lift_C_minus_B": decision_blob["mean_lift_C_minus_B"],
        "p64_C_dev_rate": decision_blob["a_rate"],
        "prot_drop_C_minus_A": decision_blob["prot_drop"],
        "answers": decision_blob["answers"],
        "pre_registered": PRE_REG,
        "wall_s": time.time() - t0,
    }
    write_json(OUT / "SUMMARY.json", summary)
    (OUT / "DECISION.md").write_text(
        f"""# DECISION — P110

`{decision_blob['decision']}`

## Pre-registered gates
{json.dumps(PRE_REG, indent=2)}

## Answers
{json.dumps(decision_blob['answers'], indent=2)}

## Seed lifts (C−B on C_dev roots)
{json.dumps(decision_blob['seed_lifts'], indent=2)}

Mean lift: {decision_blob['mean_lift_C_minus_B']}

## STOP
No baseline promotion. No P111. No backbone unfreeze. No PPO/GRPO. Sealed panel closed.
""",
        encoding="utf-8",
    )
    (OUT / "FINAL_REPORT.md").write_text(
        f"""# FINAL_REPORT — P110 Targeted Corrective Imitation

## Decision
`{decision_blob['decision']}`

## Setup
- Frozen P64 `{sha}`; token-head only; residual/backbone frozen.
- Takeover repaired: WrapState restore + state-conditioned Plan-C expert (no absolute tape).
- Train C roots: {pool_meta.get('n_train_c')}; Dev C roots: {pool_meta.get('n_dev_c')}.
- Corrective roots: {data['stats'].get('n_corrective_roots')}; representable frac: {data['stats'].get('representable_frac'):.3f}.
- Type I/II/III counts: {data['stats'].get('n_type_I')}/{data['stats'].get('n_type_II')}/{data['stats'].get('n_type_III')}.

## Closed-loop (primary: C vs B on unseen C_dev)
- P64 C_dev rate: {decision_blob['a_rate']}
- Mean C−B lift: {decision_blob['mean_lift_C_minus_B']}
- Seeds agreeing (≥{PRE_REG['min_c_lift_vs_B']} lift): {decision_blob['n_agree_seeds']}/{len(SEEDS)}

## Protected retention (C − A)
{json.dumps(decision_blob['prot_drop'], indent=2)}

## Answers Q1–Q9
{json.dumps(decision_blob['answers'], indent=2)}

## Interpretation (separate claims)
1. Takeover physically valid: {decision_blob['answers']['Q1_takeover_valid']}
2. Corrections representable: frac={data['stats'].get('representable_frac')}
3–6. Closed-loop / generalization / retention: see decision.
Do not promote any P110 checkpoint.

## STOP
P110 complete. Wait for explicit approval before further work.
""",
        encoding="utf-8",
    )
    print(f"[P110] DONE decision={decision_blob['decision']} wall_s={summary['wall_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
