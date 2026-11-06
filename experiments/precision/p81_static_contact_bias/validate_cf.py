"""Phases 3–4 — matched Actor vs tok_56 validation + timing sensitivity."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p73_first_contact_bilateral.controller import (
    EventRule,
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p81_static_contact_bias.paths import (
    CACHE,
    MAX_EP_STEPS,
    N_CF_SEEDS,
    N_CF_VALIDATE_ICS,
    N_TIMING_SNAPS,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    P79_BANK,
    P79_CACHE,
    SEED,
    STATE_DIR,
    T_MAX,
    TOKEN_56,
)


def _rule_56() -> EventRule:
    return EventRule(
        name="tok56_once",
        sequence=[TOKEN_56],
        max_steps=1,
        stop_on_obs_bilateral=False,
        abort_on_contact_loss=False,
        require_unilateral_start=False,
    )


def _continue(env, actor, critic, start, state, device, *, rule, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    st = WrapState()
    st.saw_contact = True
    st.v2.phase = "done"
    st.v2.activated = True
    v2 = v2_spec()
    done = False
    info = {}
    g0 = relative_geometry(env.env.sim)
    ever = {k: False for k in ("opp", "bil", "loss", "e9", "r3", "sw")}
    had = bool(g0["has_pad_contact"])
    t0 = int(env.t)
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=rule
        )
        _r, done, info = env.step(int(tok), resid)
        g = relative_geometry(env.env.sim)
        if g["correct_sidewall"]:
            ever["sw"] = True
        if g["bilateral"] and g["correct_sidewall"]:
            ever["opp"] = True
        if g["bilateral"]:
            ever["bil"] = True
        if g["e_pos_mm"] <= 9:
            ever["e9"] = True
        if had and not g["has_pad_contact"]:
            ever["loss"] = True
        had = bool(g["has_pad_contact"])
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever["r3"] = True
    term = _r3_terminal(env, info)
    return {
        "sw0": int(g0["correct_sidewall"]),
        "e0": float(g0["e_pos_mm"]),
        "r3_ready": int(ever["r3"] or term["r3_ready"]),
        "SUCCESS_RELAXED": int(term["SUCCESS_RELAXED"]),
        "opposing": int(ever["opp"]),
        "bilateral": int(ever["bil"]),
        "contact_loss": int(ever["loss"]),
        "e9": int(ever["e9"]),
        "sw_ever": int(ever["sw"]),
        "horizon": int(env.t) - t0,
    }


def _ic_boot(deltas_by_ic: dict, seed: int = SEED) -> dict:
    ics = list(deltas_by_ic)
    if len(ics) < 3:
        vals = list(deltas_by_ic.values())
        return {"n_ics": len(ics), "mean": float(np.mean(vals)) if vals else None, "ci95": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray([deltas_by_ic[i] for i in ics], float)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(800)]
    return {
        "n_ics": len(ics),
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def run_matched_validation(*, device: str = "cuda", force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "validate_rows.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return _summarize_validation(rows)

    # Reuse P79 primary FC snaps + mine extras up to N_CF_VALIDATE_ICS
    bank = json.loads(P79_BANK.read_text(encoding="utf-8"))
    primary = list(bank["populations"]["A_natural_fc_primary"])
    # Prefer unique ICs
    seen = []
    metas = []
    for m in primary:
        if m["traj_id"] in seen:
            continue
        seen.append(m["traj_id"])
        metas.append(m)
        if len(metas) >= N_CF_VALIDATE_ICS:
            break

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}
    rule56 = _rule_56()
    rows = []

    for si, sm in enumerate(metas):
        tid = sm["traj_id"]
        if tid not in lookup:
            continue
        start = lookup[tid]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        for k in range(N_CF_SEEDS):
            seed = int(sm["seed"]) + 1009 * k + 3
            base = _continue(env, actor, critic, start, state, device_t, rule=None, seed=seed)
            alt = _continue(env, actor, critic, start, state, device_t, rule=rule56, seed=seed)
            rows.append(
                {
                    "snap_id": sm["id"],
                    "traj_id": tid,
                    "seed_k": k,
                    "seed": seed,
                    "stratum": sm.get("stratum"),
                    "population": "natural_p79",
                    "stage": "first_actor_owned_fc",
                    "actor": base,
                    "tok56": alt,
                    "d_succ": alt["SUCCESS_RELAXED"] - base["SUCCESS_RELAXED"],
                    "d_r3": alt["r3_ready"] - base["r3_ready"],
                    "d_opp": alt["opposing"] - base["opposing"],
                    "d_loss": alt["contact_loss"] - base["contact_loss"],
                }
            )
        print(f"[val] {si+1}/{len(metas)} {tid[-18:]} d_succ_mean={np.mean([r['d_succ'] for r in rows if r['snap_id']==sm['id']]):+.2f}", flush=True)

    # Correct-sidewall scarcity note: try to mine a few more if missing
    n_sw = sum(1 for r in rows if r["actor"]["sw0"] == 1 and r["seed_k"] == 0)
    coverage_note = {
        "n_correct_sw_snaps_in_validation": n_sw,
        "limitation": "Correct-sidewall FC remains scarce in natural C rollouts; not mixed with synthetic stress tests.",
    }
    (CACHE / "validate_coverage.json").write_text(json.dumps(coverage_note, indent=2), encoding="utf-8")
    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return _summarize_validation(rows)


def _summarize_validation(rows: list[dict]) -> dict:
    # Aggregate per IC (mean over seeds), then bootstrap
    by_ic = defaultdict(list)
    for r in rows:
        by_ic[r["traj_id"]].append(r)
    d_succ = {tid: float(np.mean([x["d_succ"] for x in rs])) for tid, rs in by_ic.items()}
    d_r3 = {tid: float(np.mean([x["d_r3"] for x in rs])) for tid, rs in by_ic.items()}

    # Seed agreement within snap
    by_snap = defaultdict(list)
    for r in rows:
        by_snap[r["snap_id"]].append(r)
    seed_stable = 0
    seed_n = 0
    for sid, rs in by_snap.items():
        if len({r["seed_k"] for r in rs}) < 2:
            continue
        seed_n += 1
        signs = [np.sign(r["d_succ"]) for r in rs]
        if len(set(signs)) == 1:
            seed_stable += 1

    # Exclude ties for "benefit" claims
    nontie_ics = {tid: d for tid, d in d_succ.items() if abs(d) > 1e-9}
    pos_ics = {tid: d for tid, d in nontie_ics.items() if d > 0}
    neg_ics = {tid: d for tid, d in nontie_ics.items() if d < 0}

    actor_succ = float(np.mean([r["actor"]["SUCCESS_RELAXED"] for r in rows if r["seed_k"] == 0]))
    t56_succ = float(np.mean([r["tok56"]["SUCCESS_RELAXED"] for r in rows if r["seed_k"] == 0]))

    report = {
        "n_rows": len(rows),
        "n_independent_ics": len(by_ic),
        "n_seeds": N_CF_SEEDS,
        "actor_success_mean": actor_succ,
        "tok56_success_mean": t56_succ,
        "paired_delta_succ_ic": _ic_boot(d_succ),
        "paired_delta_r3_ic": _ic_boot(d_r3, seed=SEED + 1),
        "n_ics_tok56_better_succ": len(pos_ics),
        "n_ics_actor_better_succ": len(neg_ics),
        "n_ics_tie_succ": len(by_ic) - len(nontie_ics),
        "seed_sign_stability": {
            "n": seed_n,
            "frac_stable": float(seed_stable / seed_n) if seed_n else None,
        },
        "correct_sw_rows": sum(1 for r in rows if r["actor"]["sw0"] == 1),
        "strata": dict(__import__("collections").Counter(r.get("stratum") for r in rows if r["seed_k"] == 0)),
        "reproduces_p80_direction": bool(t56_succ >= actor_succ - 1e-9),
        "ci_excludes_0_for_succ": bool(
            (_ic_boot(d_succ).get("ci95") or [0, 0])[0] > 0
        ),
    }
    (OUT / "MATCHED_INTERVENTION_VALIDATION.md").write_text(
        "# Matched Intervention Validation (P81 Phase 3)\n\n"
        "Single-decision tok_56 vs Actor, then resume frozen P64+V2.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("n_independent_ics", "actor_success_mean", "tok56_success_mean", "paired_delta_succ_ic", "ci_excludes_0_for_succ")}, indent=2), flush=True)
    return report


def run_timing_analysis(*, device: str = "cuda", force: bool = False) -> dict:
    """Phase 4 — single-token tok_56 at different stages."""
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "timing_rows.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return _summarize_timing(rows)

    bank = json.loads(P79_BANK.read_text(encoding="utf-8"))
    # Use diverse kinds from bank
    by_kind = defaultdict(list)
    for m in bank["populations"]["A_natural_all"]:
        by_kind[m["kind"]].append(m)
    # Map kinds to timing labels
    kind_map = {
        "first_contact": "first_actor_owned_fc",
        "shortly_after_fc": "early_post_contact",
        "persist_incomplete": "before_opposing",
        "approach_loss": "before_contact_loss_causal",
    }
    metas = []
    for kind, stage in kind_map.items():
        for m in by_kind.get(kind, [])[: max(1, N_TIMING_SNAPS // 4)]:
            metas.append({**m, "timing_stage": stage})

    # Also need post-opposing adverse check from first_bilateral / after if present
    for m in (bank["populations"].get("A_natural_all") or []):
        if m.get("kind") in ("first_contact",) and m.get("opposing"):
            pass

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}
    rule56 = _rule_56()
    rows = []

    for si, sm in enumerate(metas):
        tid = sm["traj_id"]
        if tid not in lookup or not Path(sm["path"]).exists():
            continue
        start = lookup[tid]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        # Check if already opposing — flag adverse risk
        seed_all(int(sm["seed"]), deterministic_torch=True)
        restore_full_state(env, state, start)
        g0 = relative_geometry(env.env.sim)
        already_opp = bool(g0["bilateral"] and g0["correct_sidewall"])
        seed = int(sm["seed"]) + 17
        base = _continue(env, actor, critic, start, state, device_t, rule=None, seed=seed)
        alt = _continue(env, actor, critic, start, state, device_t, rule=rule56, seed=seed)
        rows.append(
            {
                "snap_id": sm["id"],
                "traj_id": tid,
                "timing_stage": sm["timing_stage"],
                "kind": sm.get("kind"),
                "already_opposing": int(already_opp),
                "sw0": int(g0["correct_sidewall"]),
                "d_succ": alt["SUCCESS_RELAXED"] - base["SUCCESS_RELAXED"],
                "d_r3": alt["r3_ready"] - base["r3_ready"],
                "d_loss": alt["contact_loss"] - base["contact_loss"],
                "d_opp": alt["opposing"] - base["opposing"],
                "actor_succ": base["SUCCESS_RELAXED"],
                "tok56_succ": alt["SUCCESS_RELAXED"],
            }
        )
        if (si + 1) % 4 == 0:
            print(f"[timing] {si+1}/{len(metas)}", flush=True)

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return _summarize_timing(rows)


def _summarize_timing(rows: list[dict]) -> dict:
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["timing_stage"]].append(r)
    stage = {}
    for st, rs in by_stage.items():
        stage[st] = {
            "n": len(rs),
            "n_ics": len({r["traj_id"] for r in rs}),
            "mean_d_succ": float(np.mean([r["d_succ"] for r in rs])),
            "mean_d_r3": float(np.mean([r["d_r3"] for r in rs])),
            "mean_d_loss": float(np.mean([r["d_loss"] for r in rs])),
            "frac_help": float(np.mean([r["d_succ"] > 0 for r in rs])),
            "frac_harm": float(np.mean([r["d_succ"] < 0 for r in rs])),
        }
    adverse = [r for r in rows if r["already_opposing"]]
    report = {
        "by_stage": stage,
        "narrow_vs_broad": {
            "helps_multiple_stages": sum(1 for s in stage.values() if s["mean_d_succ"] > 0.02) >= 2,
            "helps_only_fc": (
                (stage.get("first_actor_owned_fc") or {}).get("mean_d_succ", 0) > 0.02
                and sum(1 for k, s in stage.items() if k != "first_actor_owned_fc" and s["mean_d_succ"] > 0.02) == 0
            ),
        },
        "already_opposing_adverse": {
            "n": len(adverse),
            "mean_d_succ": float(np.mean([r["d_succ"] for r in adverse])) if adverse else None,
            "note": "tok_56 after correct opposing — check for harm",
        },
        "causal_trigger_note": (
            "before_contact_loss_causal uses approach_loss snaps mined with current-frame "
            "min_pad/e_pos indicators only — not future loss labels."
        ),
    }
    (OUT / "TIMING_SENSITIVITY.md").write_text(
        "# Timing Sensitivity (P81 Phase 4)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(stage, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_matched_validation()
    run_timing_analysis()
