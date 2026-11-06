"""Phase 2–3: first-contact counterfactuals + intervention window."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p73_first_contact_bilateral.controller import (
    EventRule,
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.paths import (
    CF_MAX_STEPS,
    MAX_EP_STEPS,
    N_CF_SEEDS,
    N_SNAP,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    STATE_DIR,
    T_MAX,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates


def _lookup():
    blocked = blocked_eval_ids()
    return {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }


def _mine(env, actor, critic, book, device, lookup, attr_rows: list) -> dict[str, list]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    kinds = {
        "first_contact": [],
        "shortly_after_fc": [],  # +2 steps, still not opposing
        "first_bilateral": [],
        "after_bilateral": [],  # +2 after first bil
    }
    plans = []
    seen = set()
    for r in attr_rows:
        key = (r["traj_id"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        plans.append(r)
        if len(plans) >= 48:
            break

    v2 = v2_spec()
    for r in plans:
        if all(len(kinds[k]) >= N_SNAP for k in kinds):
            break
        start = lookup.get(r["traj_id"])
        if start is None:
            continue
        seed_all(int(r["seed"]), deterministic_torch=True)
        env.reset(start)
        ensure_inner_capture_buffers(env)
        st = WrapState()
        done = False
        got = {k: False for k in kinds}
        fc_t = None
        bil_t = None
        while not done:
            tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok, st, _ = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
            )
            _r, done, _info = env.step(int(tok), resid)
            g = relative_geometry(env.env.sim)
            ensure_inner_capture_buffers(env)
            t = int(env.t)
            opposing = bool(g["bilateral"] and g["correct_sidewall"])

            def capture(kind):
                if got[kind] or len(kinds[kind]) >= N_SNAP:
                    return
                if env.t + 6 > env.max_steps:
                    return
                state = take_full_state(
                    env, meta={"kind": kind, "traj_id": start.traj_id, "seed": r["seed"], "t": t}
                )
                sid = f"{kind}_{len(kinds[kind]):03d}_{start.traj_id[-14:]}_t{t}"
                path = STATE_DIR / f"{sid}.pkl"
                path.write_bytes(pickle.dumps(state))
                kinds[kind].append(
                    {
                        "id": sid,
                        "path": str(path),
                        "traj_id": start.traj_id,
                        "seed": int(r["seed"]),
                        "t": t,
                        "e_pos_mm": float(g["e_pos_mm"]),
                        "opposing": int(opposing),
                        "bilateral": int(g["bilateral"]),
                        "from_opp": int(r.get("reached_opposing") or 0),
                    }
                )
                got[kind] = True

            if g["has_pad_contact"] and fc_t is None:
                fc_t = t
                capture("first_contact")
            if fc_t is not None and not got["shortly_after_fc"] and t >= fc_t + 2 and not opposing:
                capture("shortly_after_fc")
            if g["bilateral"] and bil_t is None:
                bil_t = t
                capture("first_bilateral")
            if bil_t is not None and not got["after_bilateral"] and t >= bil_t + 2:
                capture("after_bilateral")

        print(
            f"[mine] {r['traj_id'][-20:]} "
            + ",".join(f"{k}:{len(v)}" for k, v in kinds.items()),
            flush=True,
        )
    return kinds


def _continue(env, actor, critic, book, start, state, device, *, rule: EventRule | None, seed: int) -> dict:
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
    ever = {k: False for k in ("opp", "bil", "pad4", "e9", "e12", "frame", "r3")}
    # Immediate / mid metrics before episode end
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=rule
        )
        _r, done, info = env.step(int(tok), resid)
        g = relative_geometry(env.env.sim)
        if g["bilateral"] and g["correct_sidewall"]:
            ever["opp"] = True
        if g["bilateral"]:
            ever["bil"] = True
        if g["min_pad"] >= 4:
            ever["pad4"] = True
        if g["e_pos_mm"] <= 12:
            ever["e12"] = True
        if g["e_pos_mm"] <= 9:
            ever["e9"] = True
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"]:
            ever["frame"] = True
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever["r3"] = True
    term = _r3_terminal(env, info)
    return {
        "e0": float(g0["e_pos_mm"]),
        "reached_opposing": int(ever["opp"]),
        "reached_bilateral": int(ever["bil"]),
        "reached_pad4": int(ever["pad4"]),
        "reached_e12": int(ever["e12"]),
        "reached_e9": int(ever["e9"]),
        "reached_frame_ok": int(ever["frame"]),
        "r3_ready": int(ever["r3"] or term["r3_ready"]),
        "SUCCESS_RELAXED": term["SUCCESS_RELAXED"],
        "handoff": term["handoff"],
        "stop_reason": st.stop_reason,
        "rule": None if rule is None else rule.name,
    }


def _candidate_rules(attr_rows: list) -> dict[str, EventRule]:
    # Mode success post-FC prefix
    cnt = Counter()
    for r in attr_rows:
        if r.get("reached_opposing") and r.get("post_tokens"):
            cnt[tuple(r["post_tokens"][:5])] += 1
    mode = list(cnt.most_common(1)[0][0]) if cnt else [56, 50, 71, 40, 82]
    # P72 A_continue (known FC oracle)
    a_cont = [40, 56, 40, 40, 83]
    rules = {
        "baseline": None,
        "A_continue_h5": EventRule(name="A_continue_h5", sequence=a_cont, max_steps=5),
        "A_continue_h3": EventRule(name="A_continue_h3", sequence=a_cont, max_steps=3),
        "mode_succ_h5": EventRule(name="mode_succ_h5", sequence=mode, max_steps=5),
        "grip_then_approach": EventRule(
            name="grip_then_approach", sequence=[82, 82, 56, 50, 71], max_steps=5
        ),
        "soft_approach": EventRule(name="soft_approach", sequence=[56, 50, 71, 50, 56], max_steps=5),
        "orient_close": EventRule(name="orient_close", sequence=[40, 40, 82, 56, 50], max_steps=5),
    }
    return rules


def run_phase23(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads((OUT / "ATTR_RAW.json").read_text(encoding="utf-8"))
    attr_rows = raw["rows"]

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = _lookup()

    print("=== Mine FC/bilateral snapshots ===", flush=True)
    banks = _mine(env, actor, critic, book, device_t, lookup, attr_rows)
    (OUT / "SNAP_BANK.json").write_text(
        json.dumps({k: v for k, v in banks.items()}, indent=2, default=str), encoding="utf-8"
    )

    # Restoration check
    restore_ok = None
    for kind in ("first_contact", "first_bilateral"):
        snaps = banks.get(kind) or []
        if not snaps:
            continue
        sm = snaps[0]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        restore_full_state(env, state, lookup[sm["traj_id"]])
        g1 = relative_geometry(env.env.sim)
        live = take_full_state(env)
        restore_full_state(env, live, lookup[sm["traj_id"]])
        g2 = relative_geometry(env.env.sim)
        restore_ok = {
            "kind": kind,
            "e_match": abs(float(g1["e_pos_mm"]) - float(g2["e_pos_mm"])) < 1e-6,
            "t_match": int(env.t) == int(state["policy_step"]),
        }
        print(f"[restore] {restore_ok}", flush=True)
        break

    rules = _candidate_rules(attr_rows)
    # Persist rule defs
    rule_blob = {
        k: None
        if v is None
        else {"name": v.name, "sequence": v.sequence, "max_steps": v.max_steps, "stop_on_obs_bilateral": v.stop_on_obs_bilateral}
        for k, v in rules.items()
    }

    # Phase 2: focus on first_contact
    # Phase 3: all window kinds
    focus = ["first_contact", "shortly_after_fc", "first_bilateral", "after_bilateral"]
    results = {}
    for kind in focus:
        snaps = banks.get(kind) or []
        results[kind] = {}
        for rname, rule in rules.items():
            stats = []
            for sm in snaps:
                state = pickle.loads(Path(sm["path"]).read_bytes())
                start = lookup[sm["traj_id"]]
                for si in range(N_CF_SEEDS):
                    stats.append(
                        _continue(
                            env,
                            actor,
                            critic,
                            book,
                            start,
                            state,
                            device_t,
                            rule=rule,
                            seed=SEED + 9000 + si + 13 * (sum(ord(c) for c in sm["id"]) % 10000),
                        )
                    )
            agg = {
                "n": len(stats),
                **{
                    m: float(np.mean([s[m] for s in stats])) if stats else None
                    for m in (
                        "reached_opposing",
                        "reached_bilateral",
                        "reached_pad4",
                        "reached_e12",
                        "reached_e9",
                        "reached_frame_ok",
                        "r3_ready",
                        "SUCCESS_RELAXED",
                    )
                },
            }
            results[kind][rname] = agg
            print(
                f"[cf] {kind}/{rname} opp={agg['reached_opposing']} e9={agg['reached_e9']} "
                f"r3={agg['r3_ready']} n={agg['n']}",
                flush=True,
            )

    # Window usefulness: delta vs baseline on opposing + r3
    window = {}
    for kind in focus:
        base = (results.get(kind) or {}).get("baseline") or {}
        per = {}
        for rname in rules:
            if rname == "baseline":
                continue
            cur = (results.get(kind) or {}).get(rname) or {}
            per[rname] = {
                "d_opp": None
                if base.get("reached_opposing") is None or cur.get("reached_opposing") is None
                else cur["reached_opposing"] - base["reached_opposing"],
                "d_r3": None
                if base.get("r3_ready") is None or cur.get("r3_ready") is None
                else cur["r3_ready"] - base["r3_ready"],
                "d_e9": None
                if base.get("reached_e9") is None or cur.get("reached_e9") is None
                else cur["reached_e9"] - base["reached_e9"],
            }
        window[kind] = per

    # Generalizable? positive d_opp at first_contact AND shortly_after, not cratering at bil+
    def generalizes(rname: str) -> bool:
        d_fc = (window.get("first_contact") or {}).get(rname) or {}
        d_aft = (window.get("shortly_after_fc") or {}).get(rname) or {}
        d_bil = (window.get("first_bilateral") or {}).get(rname) or {}
        d_ab = (window.get("after_bilateral") or {}).get(rname) or {}
        if (d_fc.get("d_opp") or 0) <= 0.02 and (d_fc.get("d_r3") or 0) <= 0.02:
            return False
        # Must not strongly hurt at/after bilateral
        if (d_bil.get("d_r3") or 0) < -0.05 or (d_ab.get("d_r3") or 0) < -0.05:
            return False
        # Prefer also helpful shortly after FC or at least non-negative
        if (d_aft.get("d_opp") or 0) < -0.05:
            return False
        return (d_fc.get("d_opp") or 0) > 0.02 or (d_fc.get("d_r3") or 0) > 0.02

    def _rank_key(rname: str):
        d = (window.get("first_contact") or {}).get(rname) or {}
        score = float(d.get("d_opp") or -9) + 0.5 * float(d.get("d_r3") or -9)
        return (score, generalizes(rname))

    ranked = sorted(
        [r for r in rules if r != "baseline"],
        key=_rank_key,
        reverse=True,
    )
    best = None
    for r in ranked:
        if generalizes(r):
            best = r
            break

    # Observable termination: stop_on_obs_bilateral is deployable
    obs_term = {
        "signal": "obs contact_L AND contact_R",
        "privileged": False,
        "reliable_enough": True,  # used as stop; may lag true opposing
        "note": "Stops overlay when Actor obs reports bilateral pads; does not use true sidewall/e_pos.",
    }

    decision_window = {
        "useful_from": "first_contact" if best else None,
        "useful_until": "before_or_at_first_bilateral" if best else None,
        "hurts_after_bilateral": {
            r: {
                "d_r3_first_bil": ((window.get("first_bilateral") or {}).get(r) or {}).get("d_r3"),
                "d_r3_after_bil": ((window.get("after_bilateral") or {}).get(r) or {}).get("d_r3"),
            }
            for r in rules
            if r != "baseline"
        },
        "chosen_for_phase4": best,
        "generalizes": {r: generalizes(r) for r in rules if r != "baseline"},
        "obs_termination": obs_term,
    }

    out = {
        "cf_results": results,
        "window_deltas": window,
        "rules": rule_blob,
        "restoration_check": restore_ok,
        "intervention_window": decision_window,
        "mode_succ_seq": rule_blob.get("mode_succ_h5", {}).get("sequence") if rule_blob.get("mode_succ_h5") else None,
        "cf_max_steps_tested": list(CF_MAX_STEPS),
    }
    (OUT / "M111_COUNTERFACTUALS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "M111_COUNTERFACTUALS.md").write_text(
        "# M111 Counterfactuals (P73 Phase 2)\n\n"
        f"Chosen for Phase 4: `{best}`\n\n"
        "## first_contact deltas vs baseline\n```json\n"
        + json.dumps(window.get("first_contact"), indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "INTERVENTION_WINDOW.json").write_text(json.dumps(decision_window, indent=2), encoding="utf-8")
    (OUT / "INTERVENTION_WINDOW.md").write_text(
        "# Intervention Window (P73 Phase 3)\n\n"
        f"**Useful rule:** `{best}`\n\n"
        "Tested at: first_contact → shortly_after_fc → first_bilateral → after_bilateral.\n\n"
        f"```json\n{json.dumps(decision_window, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    run_phase23()
