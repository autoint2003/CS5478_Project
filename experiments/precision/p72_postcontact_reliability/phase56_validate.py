"""Phase 5–6: freeze refinement from mechanism; full-episode + panel validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p62_grpo_optimization.paths import PER_CLASS_DEV
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p65_ppo_continuation.dev_panel import load_panel_starts
from experiments.precision.p72_postcontact_reliability.instrumented import rollout_instrumented
from experiments.precision.p72_postcontact_reliability.paths import (
    MAX_EP_STEPS,
    N_VAL_ICS,
    N_VAL_ROLLOUTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    PANEL_C_ROLLOUTS,
    SEED,
    T_MAX,
)
from experiments.precision.p72_postcontact_reliability.v2_wrap import PostRefinement


def _starts(ids):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def _agg(rows):
    def m(k):
        xs = [float(r[k]) for r in rows if r.get(k) is not None]
        return float(np.mean(xs)) if xs else None

    return {
        "n": len(rows),
        "r3_ready": m("r3_ready"),
        "handoff": m("handoff"),
        "SUCCESS_RELAXED": m("SUCCESS_RELAXED"),
        "reached_e9": m("reached_e9"),
        "ever_pad4": float(np.mean([r["ever"]["min_pad_ge4"] for r in rows])) if rows else None,
        "ever_sw": float(np.mean([r["ever"]["opposing_bilateral"] for r in rows])) if rows else None,
        "near_ready": m("near_ready_seen"),
        "contact_loss": float(np.mean([r["lost_after_counts"]["first_contact"] > 0 for r in rows])) if rows else None,
        "mean_t_to_9": float(np.mean([r["t_to_9"] for r in rows if r.get("t_to_9") is not None])) if rows else None,
        "mean_max_ok_s": m("max_ok_s"),
    }


def _load_post() -> PostRefinement | None:
    mech = json.loads((OUT / "DOMINANT_MECHANISM.json").read_text(encoding="utf-8"))
    posts = json.loads((OUT / "MATCHED_COUNTERFACTUALS.json").read_text(encoding="utf-8")).get("posts") or {}
    chosen = mech.get("chosen_intervention") or "baseline"
    if chosen in (None, "baseline", "none"):
        return None
    blob = posts.get(chosen) or {}
    if not blob:
        seq_key = {"A_continue": "corr_seq", "B_preserve": "preserve_seq", "C_nearhold": "hold_seq"}.get(
            chosen, "corr_seq"
        )
        seq = mech.get(seq_key) or mech.get("corr_seq") or []
        kind = {"A_continue": "continue_seq", "B_preserve": "contact_preserve", "C_nearhold": "near_ready_hold"}.get(
            chosen, "continue_seq"
        )
        return PostRefinement(
            name=chosen,
            kind=kind,
            sequence=list(seq),
            max_steps=5,
            require_bilateral_obs=(chosen == "C_nearhold"),
            abort_on_contact_loss=True,
        )
    return PostRefinement(
        name=blob["name"],
        kind=blob["kind"],
        sequence=list(blob.get("sequence") or []),
        max_steps=int(blob.get("max_steps") or 5),
        require_bilateral_obs=(blob.get("kind") == "near_ready_hold"),
        abort_on_contact_loss=True,
    )


def _eval_arms(env, actor, critic, book, starts, device, *, n_roll, split, post):
    arms = {"V0": "baseline_actor", "V2": "v2", "P72": "p72"}
    out = {k: [] for k in arms}
    for st in starts:
        for i in range(n_roll):
            seed = rollout_seed(split=split, traj_id=st.traj_id, rollout_i=99500 + i)
            # V0: no V2 — pass post=None but need actor-only. Use instrumented with a hack:
            # simplest: temporarily use post=None and a flag... instrumented always uses V2 wrap.
            # For V0, call select with v2 but we need actor only — add path via post name
            from experiments.precision.p72_postcontact_reliability.instrumented import rollout_instrumented as ri
            from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
            from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
            from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
            from experiments.precision.p68_precontact_geometry.geometry import relative_geometry

            # V0 pure actor
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            done = False
            info = {}
            ever_e9 = ever_pad4 = ever_sw = near = False
            lost = False
            had_c = False
            t_to_9 = None
            t_fc = None
            max_ok = 0.0
            while not done:
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                _r, done, info = env.step(tok, resid)
                g = relative_geometry(env.env.sim)
                tr = env.mon.tracker
                ok_s = float(tr.ok_s) if tr else 0.0
                max_ok = max(max_ok, ok_s)
                if g["has_pad_contact"]:
                    if t_fc is None:
                        t_fc = int(env.t)
                    had_c = True
                if had_c and not g["has_pad_contact"]:
                    lost = True
                if g["e_pos_mm"] <= 9 and t_to_9 is None and t_fc is not None:
                    t_to_9 = int(env.t - t_fc)
                    ever_e9 = True
                if g["min_pad"] >= 4:
                    ever_pad4 = True
                if g["correct_sidewall"] and g["bilateral"]:
                    ever_sw = True
                preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
                if preds["frame_ok"] and ok_s > 0:
                    near = True
            preds = evaluate_predicates(env.env.sim, ok_s=float(env.mon.tracker.ok_s) if env.mon.tracker else 0.0)
            handoff = int(info.get("handoff") or 0)
            r3 = int(bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff))
            out["V0"].append(
                {
                    "r3_ready": r3,
                    "handoff": handoff,
                    "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
                    "reached_e9": int(ever_e9),
                    "ever": {"min_pad_ge4": int(ever_pad4), "opposing_bilateral": int(ever_sw)},
                    "near_ready_seen": int(near),
                    "lost_after_counts": {"first_contact": int(lost)},
                    "t_to_9": t_to_9,
                    "max_ok_s": max_ok,
                    "traj_id": st.traj_id,
                }
            )

            r2 = ri(env, actor, critic, book, st, device, seed=seed, post=None, tag="V2", record_trace=False)
            out["V2"].append(r2)
            r3p = ri(env, actor, critic, book, st, device, seed=seed, post=post, tag="P72", record_trace=False)
            out["P72"].append(r3p)
            print(
                f"[val:{split}] {st.traj_id[-22:]} i={i} "
                f"V0={out['V0'][-1]['r3_ready']} V2={r2['r3_ready']} P72={r3p['r3_ready']}",
                flush=True,
            )
    return {k: {"rows": v, "agg": _agg(v)} for k, v in out.items()}


def run_phase56(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    mech = json.loads((OUT / "DOMINANT_MECHANISM.json").read_text(encoding="utf-8"))
    post = _load_post()
    frozen = {
        "case": mech.get("case"),
        "chosen": mech.get("chosen_intervention"),
        "post": None
        if post is None
        else {
            "name": post.name,
            "kind": post.kind,
            "sequence": post.sequence,
            "max_steps": post.max_steps,
            "require_bilateral_obs": post.require_bilateral_obs,
            "abort_on_contact_loss": post.abort_on_contact_loss,
            "uses_privileged_e_pos": False,
            "precontact_v2_unchanged": True,
        },
        "fallback": "P70/P71 V2 then Actor",
    }
    (OUT / "FIXED_REFINEMENT.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (OUT / "FIXED_REFINEMENT.md").write_text(
        "# Fixed Refinement (P72 Phase 5)\n\n"
        f"Case `{frozen['case']}` chosen=`{frozen['chosen']}`\n\n"
        "Pre-contact V2 unchanged. Post-contact overlay is observation-only "
        "(pad contact / bilateral contact flags from Actor obs).\n\n"
        f"```json\n{json.dumps(frozen, indent=2)}\n```\n",
        encoding="utf-8",
    )

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    # Independent: held-out train ICs (not used in Phase 1 discovery funnel)
    val_ids = list(split["heldout_train_ics"])[:N_VAL_ICS]
    print(f"=== Held-out train C validation best={frozen['chosen']} ===", flush=True)
    held = _eval_arms(
        env, actor, critic, book, _starts(val_ids), device_t, n_roll=N_VAL_ROLLOUTS, split="train", post=post
    )

    # Panels after freeze
    held_p61 = load_heldout_starts(per_class=PER_CLASS_DEV)["C"]
    print(f"=== P61/P62 dev C n={len(held_p61)} ===", flush=True)
    p61 = _eval_arms(
        env, actor, critic, book, list(held_p61), device_t, n_roll=PANEL_C_ROLLOUTS, split="heldout", post=post
    )
    try:
        p65_c = list(load_panel_starts().get("C") or [])
    except Exception as e:  # noqa: BLE001
        print(f"P65 load failed: {e}", flush=True)
        p65_c = []
    p65 = {}
    if p65_c:
        print(f"=== P65 extra C n={len(p65_c)} ===", flush=True)
        p65 = _eval_arms(
            env, actor, critic, book, p65_c, device_t, n_roll=PANEL_C_ROLLOUTS, split="p65_extra_dev", post=post
        )

    def deltas(blob):
        a = blob["V2"]["agg"]
        b = blob["P72"]["agg"]
        z = blob["V0"]["agg"]
        return {
            "P72_minus_V2": {k: (None if a.get(k) is None or b.get(k) is None else b[k] - a[k]) for k in a},
            "V2_minus_V0": {k: (None if z.get(k) is None or a.get(k) is None else a[k] - z[k]) for k in a},
            "P72_minus_V0": {k: (None if z.get(k) is None or b.get(k) is None else b[k] - z[k]) for k in a},
        }

    # Promotion
    dh = deltas(held)
    d72 = dh["P72_minus_V2"]
    promote = bool(
        post is not None
        and (d72.get("r3_ready") or 0) > 0.02
        and (d72.get("SUCCESS_RELAXED") or 0) >= -0.01
        and (d72.get("contact_loss") or 0) <= 0.05
    )
    # panel check
    if promote and p61:
        dp = deltas(p61)["P72_minus_V2"]
        if (dp.get("r3_ready") or 0) < -0.02:
            promote = False

    decision = {
        "promote_p72_over_v2": promote,
        "retain_v2": not promote,
        "chosen": frozen["chosen"],
        "held_deltas": dh,
        "held_agg": {k: v["agg"] for k, v in held.items()},
        "p61_agg": {k: v["agg"] for k, v in p61.items()},
        "p65_agg": {k: v["agg"] for k, v in p65.items()} if p65 else None,
    }
    (OUT / "FULL_EPISODE_VALIDATION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    md = [
        "# Full-Episode Validation (P72 Phase 6)\n",
        f"Promote P72 over V2: **{promote}** (chosen=`{frozen['chosen']}`)\n",
        "## Held-out train C\n```json\n" + json.dumps(decision["held_agg"], indent=2) + "\n```\n",
        "## Δ P72−V2 / V2−V0\n```json\n" + json.dumps(dh, indent=2) + "\n```\n",
        "## P61 C\n```json\n" + json.dumps(decision["p61_agg"], indent=2) + "\n```\n",
    ]
    if p65:
        md.append("## P65 C\n```json\n" + json.dumps(decision["p65_agg"], indent=2) + "\n```\n")
    (OUT / "FULL_EPISODE_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return decision


if __name__ == "__main__":
    run_phase56()
