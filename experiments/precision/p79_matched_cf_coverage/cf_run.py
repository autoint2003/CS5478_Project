"""Phases 3–5 — matched single-token counterfactuals."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p73_first_contact_bilateral.controller import (
    EventRule,
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p78_action_relevant_representation.features import (
    obs27,
    rep_A_current,
    rep_B_current_plus_eng,
    rep_C_history,
    rep_D_history_plus_eng,
)
from experiments.precision.p79_matched_cf_coverage.paths import (
    CACHE,
    CANDIDATE_TOKENS,
    ENG_HIST,
    MAX_EP_STEPS,
    N_PILOT_IC,
    N_SEEDS_EXPAND,
    N_SEEDS_PILOT,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    PRIMARY_CANDIDATES,
    SEED,
    T_MAX,
)


def candidate_rules() -> dict[str, EventRule | None]:
    """Fixed single-decision interventions (equal duration = 1 token)."""
    rules: dict[str, EventRule | None] = {}
    for name in PRIMARY_CANDIDATES:
        tok = CANDIDATE_TOKENS[name]
        if tok is None:
            rules[name] = None
        else:
            rules[name] = EventRule(
                name=name,
                sequence=[int(tok)],
                max_steps=1,
                stop_on_obs_bilateral=False,
                abort_on_contact_loss=False,
                require_unilateral_start=False,
            )
    return rules


def write_action_protocol() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    proto = {
        "intervention_type": "single_decision_token",
        "candidates": {
            name: {"token": CANDIDATE_TOKENS[name], "meaning": name}
            for name in PRIMARY_CANDIDATES
        },
        "sources": [
            "baseline = frozen Actor sampled token (no force)",
            "tok_56_soft / tok_50_approach / tok_82_grip = existing M111 skill-family tokens (P73)",
        ],
        "protocol_per_snapshot": [
            "1. Restore identical pre-action full_state",
            "2. Extract observation features BEFORE intervention",
            "3. Execute one candidate M111 token (or Actor sample for baseline)",
            "4. Resume unchanged frozen Actor + P70 V2",
            "5. Continue to episode end; record intermediate + terminal outcomes",
        ],
        "matched_seeds": "Same seed schedule for all candidates from a snapshot",
        "no_m111_expansion": True,
        "no_privileged_action_selection": True,
        "multi_token_excluded_from_primary_ranking": True,
    }
    (OUT / "ACTION_CANDIDATE_PROTOCOL.md").write_text(
        "# Action Candidate Protocol (P79 Phase 3)\n\n```json\n"
        + json.dumps(proto, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return proto


def _extract_rep(env) -> dict:
    frames = [obs27(np.asarray(o, float)) for o in env.obs_seq]
    if not frames:
        frames = [np.zeros(27)]
    hh = frames[-ENG_HIST:]
    hist = frames[-max(8, ENG_HIST) :]
    return {
        "A": rep_A_current(frames[-1]),
        "B": rep_B_current_plus_eng(hh),
        "C": rep_C_history(hist),
        "D": rep_D_history_plus_eng(hist),
        "n_hist": len(frames),
    }


def _continue_single(
    env,
    actor,
    critic,
    start,
    state,
    device,
    *,
    rule: EventRule | None,
    seed: int,
    perturb_qpos: float = 0.0,
) -> dict:
    seed_all(seed, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    if perturb_qpos and perturb_qpos > 0:
        rng = np.random.default_rng(int(seed) ^ 0x5479)
        dq = rng.normal(0.0, float(perturb_qpos), size=np.asarray(env.env.sim.data.qpos).shape)
        env.env.sim.data.qpos[:] = np.asarray(env.env.sim.data.qpos, float) + dq
        mujoco.mj_forward(env.env.sim.model, env.env.sim.data)

    st = WrapState()
    st.saw_contact = True
    st.v2.phase = "done"
    st.v2.activated = True
    v2 = v2_spec()
    done = False
    info: dict = {}
    g0 = relative_geometry(env.env.sim)
    ever = {k: False for k in ("contact", "sw", "opp", "bil", "pad4", "e9", "frame", "r3", "loss")}
    t_opp = t_loss = t_r3 = None
    t0 = int(env.t)
    had_contact = bool(g0["has_pad_contact"])
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=rule
        )
        _r, done, info = env.step(int(tok), resid)
        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"]:
            ever["contact"] = True
        elif had_contact and not g["has_pad_contact"]:
            ever["loss"] = True
            if t_loss is None:
                t_loss = int(env.t) - t0
        had_contact = bool(g["has_pad_contact"])
        if g["correct_sidewall"]:
            ever["sw"] = True
        if g["bilateral"] and g["correct_sidewall"]:
            ever["opp"] = True
            if t_opp is None:
                t_opp = int(env.t) - t0
        if g["bilateral"]:
            ever["bil"] = True
        if g["min_pad"] >= 4:
            ever["pad4"] = True
        if g["e_pos_mm"] <= 9:
            ever["e9"] = True
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"]:
            ever["frame"] = True
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever["r3"] = True
            if t_r3 is None:
                t_r3 = int(env.t) - t0
    term = _r3_terminal(env, info)
    gf = relative_geometry(env.env.sim)
    return {
        "contact_preserved": int(ever["contact"] and not ever["loss"]),
        "correct_sidewall_ever": int(ever["sw"]),
        "opposing_bilateral": int(ever["opp"]),
        "reached_bilateral": int(ever["bil"]),
        "min_pad_ge4": int(ever["pad4"]),
        "e_pos_le9": int(ever["e9"]),
        "frame_ok": int(ever["frame"]),
        "r3_ready": int(ever["r3"] or term["r3_ready"]),
        "SUCCESS_RELAXED": int(term["SUCCESS_RELAXED"]),
        "contact_loss": int(ever["loss"]),
        "t_opp": t_opp,
        "t_loss": t_loss,
        "t_r3": t_r3,
        "e_final": float(gf["e_pos_mm"]),
        "forced_first": None if rule is None else (rule.sequence[0] if rule.sequence else None),
    }


def run_matched_cf(
    *,
    device: str = "cuda",
    pilot: bool = True,
    force: bool = False,
    perturb_qpos: float = 0.0,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    write_action_protocol()
    tag = "pilot" if pilot else "expand"
    cache_path = CACHE / f"cf_rows_{tag}.pkl"
    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return {"rows": rows, "cached": True, "tag": tag}

    bank = json.loads((OUT / "SNAP_BANK.json").read_text(encoding="utf-8"))
    primary = list(bank["populations"]["A_natural_fc_primary"])
    seen: list[str] = []
    snaps = []
    for m in primary:
        if m["traj_id"] in seen:
            continue
        seen.append(m["traj_id"])
        snaps.append(m)
        if pilot and len(seen) >= N_PILOT_IC:
            break
    if not pilot:
        snaps = []
        seen = []
        for m in primary:
            if m["traj_id"] in seen:
                continue
            seen.append(m["traj_id"])
            snaps.append(m)

    n_seeds = N_SEEDS_PILOT if pilot else N_SEEDS_EXPAND
    rules = candidate_rules()
    schedule = build_p54_dataset(seed=5478)["schedule"]
    lookup = {s.traj_id: s for s in schedule}

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    rows = []
    for si, sm in enumerate(snaps):
        tid = sm["traj_id"]
        if tid not in lookup:
            continue
        start = lookup[tid]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        seed_all(int(sm["seed"]), deterministic_torch=True)
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        g0 = relative_geometry(env.env.sim)
        rep = _extract_rep(env)
        for cname in PRIMARY_CANDIDATES:
            rule = rules[cname]
            for k in range(n_seeds):
                seed = int(sm["seed"]) + 10007 * k + 17
                out = _continue_single(
                    env,
                    actor,
                    critic,
                    start,
                    state,
                    device_t,
                    rule=rule,
                    seed=seed,
                    perturb_qpos=perturb_qpos,
                )
                rows.append(
                    {
                        "snap_id": sm["id"],
                        "traj_id": tid,
                        "split": sm.get("split", "fit"),
                        "stratum": sm.get("stratum"),
                        "kind": sm.get("kind"),
                        "population": "A_natural",
                        "candidate": cname,
                        "seed_k": k,
                        "seed": seed,
                        "sw0": int(g0["correct_sidewall"]),
                        "bil0": int(g0["bilateral"]),
                        "e0": float(g0["e_pos_mm"]),
                        "rep_A": rep["A"],
                        "rep_B": rep["B"],
                        "rep_C": rep["C"],
                        "rep_D": rep["D"],
                        **out,
                    }
                )
        print(f"[cf:{tag}] {si+1}/{len(snaps)} {tid[-20:]} rows={len(rows)}", flush=True)

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return {"rows": rows, "cached": False, "tag": tag, "n_snaps": len(snaps), "n_ics": len(seen)}


def check_determinism(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["snap_id"], r["candidate"])].append(r)
    identical = 0
    compared = 0
    keys = ("r3_ready", "SUCCESS_RELAXED", "opposing_bilateral", "contact_loss", "e_pos_le9")
    for _key, group in by.items():
        if len(group) < 2:
            continue
        compared += 1
        same = all(all(g[k] == group[0][k] for k in keys) for g in group[1:])
        identical += int(same)
    return {
        "n_compared_snap_actions": compared,
        "n_identical_across_seeds": identical,
        "deterministic_fraction": float(identical / compared) if compared else None,
        "implication": (
            "If deterministic_fraction≈1, do not claim independent evidence from repeated seeds."
            if compared
            else "Only one seed used; no stochasticity check from repeats."
        ),
    }


def analyze_outcomes(rows: list[dict]) -> dict:
    snap_cand: dict = defaultdict(dict)
    for r in rows:
        snap_cand[r["snap_id"]].setdefault(r["candidate"], []).append(r)

    comparisons = []
    n_disagree = n_tie = n_valid = 0
    ics = set()
    for sid, cands in snap_cand.items():
        if len(cands) < 2:
            continue
        scores = {}
        for c, rs in cands.items():
            scores[c] = {
                "r3": float(np.mean([x["r3_ready"] for x in rs])),
                "succ": float(np.mean([x["SUCCESS_RELAXED"] for x in rs])),
                "opp": float(np.mean([x["opposing_bilateral"] for x in rs])),
                "loss": float(np.mean([x["contact_loss"] for x in rs])),
                "tid": rs[0]["traj_id"],
                "stratum": rs[0].get("stratum"),
                "sw0": rs[0].get("sw0"),
            }
            ics.add(rs[0]["traj_id"])
        names = list(scores.keys())

        def pref_key(c):
            s = scores[c]
            return (s["succ"], s["r3"], s["opp"], -s["loss"])

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                n_valid += 1
                ka, kb = pref_key(a), pref_key(b)
                if ka == kb:
                    n_tie += 1
                    pref = "tie"
                else:
                    n_disagree += 1
                    pref = a if ka > kb else b
                comparisons.append(
                    {
                        "snap_id": sid,
                        "traj_id": scores[a]["tid"],
                        "a": a,
                        "b": b,
                        "pref": pref,
                        "stratum": scores[a]["stratum"],
                        "sw0": scores[a]["sw0"],
                        "scores": {a: scores[a], b: scores[b]},
                    }
                )

    nontrivial = [
        c
        for c in comparisons
        if c["pref"] != "tie"
        and (
            c["scores"][c["a"]]["r3"] != c["scores"][c["b"]]["r3"]
            or c["scores"][c["a"]]["succ"] != c["scores"][c["b"]]["succ"]
            or c["scores"][c["a"]]["opp"] != c["scores"][c["b"]]["opp"]
        )
    ]

    by_stratum: dict = defaultdict(lambda: {"disagree": 0, "tie": 0, "n": 0})
    for c in comparisons:
        st = c.get("stratum") or "unknown"
        by_stratum[st]["n"] += 1
        if c["pref"] == "tie":
            by_stratum[st]["tie"] += 1
        else:
            by_stratum[st]["disagree"] += 1

    vs_base: dict = defaultdict(lambda: {"better": 0, "worse": 0, "tie": 0})
    for _sid, cands in snap_cand.items():
        if "baseline" not in cands:
            continue
        base = cands["baseline"]
        bscore = (
            float(np.mean([x["SUCCESS_RELAXED"] for x in base])),
            float(np.mean([x["r3_ready"] for x in base])),
            float(np.mean([x["opposing_bilateral"] for x in base])),
        )
        for c, rs in cands.items():
            if c == "baseline":
                continue
            sc = (
                float(np.mean([x["SUCCESS_RELAXED"] for x in rs])),
                float(np.mean([x["r3_ready"] for x in rs])),
                float(np.mean([x["opposing_bilateral"] for x in rs])),
            )
            if sc > bscore:
                vs_base[c]["better"] += 1
            elif sc < bscore:
                vs_base[c]["worse"] += 1
            else:
                vs_base[c]["tie"] += 1

    report = {
        "n_rows": len(rows),
        "n_independent_ics": len(ics),
        "n_unique_snapshots": len(snap_cand),
        "n_valid_action_comparisons": n_valid,
        "n_ties": n_tie,
        "n_disagreements": n_disagree,
        "n_nontrivial_outcome_disagreements": len(nontrivial),
        "disagree_rate": float(n_disagree / n_valid) if n_valid else None,
        "by_stratum": dict(by_stratum),
        "vs_baseline": dict(vs_base),
        "determinism": check_determinism(rows),
        "outcome_means_by_candidate": {
            c: {
                k: float(np.mean([r[k] for r in rows if r["candidate"] == c]))
                for k in (
                    "r3_ready",
                    "SUCCESS_RELAXED",
                    "opposing_bilateral",
                    "contact_loss",
                    "e_pos_le9",
                    "min_pad_ge4",
                    "frame_ok",
                    "contact_preserved",
                )
            }
            for c in PRIMARY_CANDIDATES
        },
    }
    (OUT / "MATCHED_COUNTERFACTUAL_OUTCOMES.md").write_text(
        "# Matched Counterfactual Outcomes (P79 Phase 5)\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "ACTION_DISAGREEMENT_ANALYSIS.md").write_text(
        "# Action Disagreement Analysis (P79)\n\n```json\n"
        + json.dumps(
            {
                "n_nontrivial": len(nontrivial),
                "disagree_rate": report["disagree_rate"],
                "vs_baseline": dict(vs_base),
                "example_disagreements": nontrivial[:20],
            },
            indent=2,
            default=str,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    (CACHE / "comparisons.pkl").write_bytes(pickle.dumps(comparisons, protocol=pickle.HIGHEST_PROTOCOL))
    return report


if __name__ == "__main__":
    write_action_protocol()
    out = run_matched_cf(pilot=True)
    analyze_outcomes(out["rows"])
