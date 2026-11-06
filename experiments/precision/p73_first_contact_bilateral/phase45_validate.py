"""Phase 4–5: freeze event rule (if any); full-episode + panel validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p62_grpo_optimization.paths import PER_CLASS_DEV
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p65_ppo_continuation.dev_panel import load_panel_starts
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p73_first_contact_bilateral.controller import EventRule
from experiments.precision.p73_first_contact_bilateral.paths import (
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
from experiments.precision.p73_first_contact_bilateral.rollout import rollout


def _starts(ids):
    blocked = blocked_eval_ids()
    by = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by[t] for t in ids if t in by]


def _wilson(k, n, z=1.96):
    if n <= 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(p), float(centre - half), float(centre + half)


def _agg(rows):
    n = len(rows)

    def m(key):
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(xs)) if xs else None

    def rate(key):
        xs = [int(r[key]) for r in rows if r.get(key) is not None]
        if not xs:
            return {"n": 0, "rate": None, "lo": None, "hi": None}
        k = sum(xs)
        r, lo, hi = _wilson(k, len(xs))
        return {"n": len(xs), "k": k, "rate": r, "wilson95": [lo, hi]}

    return {
        "n": n,
        "r3_ready": m("r3_ready"),
        "SUCCESS_RELAXED": m("SUCCESS_RELAXED"),
        "handoff": m("handoff"),
        "reached_opposing": m("reached_opposing"),
        "reached_bilateral": m("reached_bilateral"),
        "reached_pad4": m("reached_pad4"),
        "reached_e9": m("reached_e9"),
        "reached_frame_ok": m("reached_frame_ok"),
        "contact_loss": m("contact_loss"),
        "r3_wilson": rate("r3_ready"),
        "opp_wilson": rate("reached_opposing"),
        "succ_wilson": rate("SUCCESS_RELAXED"),
    }


def _load_rule() -> EventRule | None:
    win = json.loads((OUT / "INTERVENTION_WINDOW.json").read_text(encoding="utf-8"))
    cf = json.loads((OUT / "M111_COUNTERFACTUALS.json").read_text(encoding="utf-8"))
    chosen = win.get("chosen_for_phase4")
    if not chosen:
        return None
    blob = (cf.get("rules") or {}).get(chosen)
    if not blob:
        return None
    return EventRule(
        name=blob["name"],
        sequence=list(blob["sequence"]),
        max_steps=int(blob["max_steps"]),
        stop_on_obs_bilateral=bool(blob.get("stop_on_obs_bilateral", True)),
        abort_on_contact_loss=True,
        require_unilateral_start=True,
    )


def _v0_rollout(env, actor, critic, start, device, seed):
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    done = False
    info = {}
    ever_opp = ever_bil = ever_pad4 = ever_e9 = ever_frame = False
    lost = False
    had_c = False
    while not done:
        tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        _r, done, info = env.step(tok, resid)
        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"]:
            had_c = True
        if had_c and not g["has_pad_contact"]:
            lost = True
        if g["bilateral"] and g["correct_sidewall"]:
            ever_opp = True
        if g["bilateral"]:
            ever_bil = True
        if g["min_pad"] >= 4:
            ever_pad4 = True
        if g["e_pos_mm"] <= 9:
            ever_e9 = True
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"]:
            ever_frame = True
    preds = evaluate_predicates(env.env.sim, ok_s=float(env.mon.tracker.ok_s) if env.mon.tracker else 0.0)
    handoff = int(info.get("handoff") or 0)
    r3 = int(bool((preds["dwell_complete"] and preds["frame_ok"]) or handoff))
    return {
        "r3_ready": r3,
        "handoff": handoff,
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "reached_opposing": int(ever_opp),
        "reached_bilateral": int(ever_bil),
        "reached_pad4": int(ever_pad4),
        "reached_e9": int(ever_e9),
        "reached_frame_ok": int(ever_frame),
        "contact_loss": int(lost),
        "traj_id": start.traj_id,
    }


def _eval_arms(env, actor, critic, book, starts, device, *, n_roll, split, rule):
    out = {"V0": [], "V2": [], "P73": []}
    for st in starts:
        for i in range(n_roll):
            seed = rollout_seed(split=split, traj_id=st.traj_id, rollout_i=99700 + i)
            out["V0"].append(_v0_rollout(env, actor, critic, st, device, seed))
            r2 = rollout(
                env, actor, critic, book, st, device, seed=seed, rule=None, tag="V2", record_fc_window=False
            )
            out["V2"].append(r2)
            if rule is None:
                # No overlay ≡ V2 (avoid duplicate rollouts)
                r3 = dict(r2)
                r3["tag"] = "P73"
                r3["rule"] = None
            else:
                r3 = rollout(
                    env, actor, critic, book, st, device, seed=seed, rule=rule, tag="P73", record_fc_window=False
                )
            out["P73"].append(r3)
            print(
                f"[val:{split}] {st.traj_id[-20:]} i={i} "
                f"V0={out['V0'][-1]['r3_ready']} V2={r2['r3_ready']} P73={r3['r3_ready']} "
                f"opp V2/P73={r2['reached_opposing']}/{r3['reached_opposing']}",
                flush=True,
            )
    # per-IC paired deltas P73-V2
    by_ic = {}
    for a, b in zip(out["V2"], out["P73"]):
        tid = a["traj_id"]
        by_ic.setdefault(tid, {"d_r3": [], "d_opp": [], "d_succ": []})
        by_ic[tid]["d_r3"].append(int(b["r3_ready"]) - int(a["r3_ready"]))
        by_ic[tid]["d_opp"].append(int(b["reached_opposing"]) - int(a["reached_opposing"]))
        by_ic[tid]["d_succ"].append(int(b["SUCCESS_RELAXED"]) - int(a["SUCCESS_RELAXED"]))
    per_ic = {
        tid: {k: float(np.mean(v)) for k, v in d.items()}
        for tid, d in by_ic.items()
    }
    return {k: {"rows": v, "agg": _agg(v)} for k, v in out.items()}, per_ic


def run_phase45(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    win = json.loads((OUT / "INTERVENTION_WINDOW.json").read_text(encoding="utf-8"))
    rule = _load_rule()

    frozen = {
        "chosen": None if rule is None else rule.name,
        "deploy": rule is not None,
        "rule": None
        if rule is None
        else {
            "name": rule.name,
            "sequence": rule.sequence,
            "max_steps": rule.max_steps,
            "stop_on_obs_bilateral": rule.stop_on_obs_bilateral,
            "abort_on_contact_loss": rule.abort_on_contact_loss,
            "require_unilateral_start": rule.require_unilateral_start,
            "uses_privileged_e_pos": False,
            "uses_true_sidewall": False,
            "precontact_v2_unchanged": True,
        },
        "window_meta": win,
        "fallback": "P70/P71 V2",
    }
    (OUT / "FIXED_EVENT_RULE.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (OUT / "FIXED_EVENT_RULE.md").write_text(
        "# Fixed Event Rule (P73 Phase 4)\n\n"
        + (
            f"**No deployable rule** — Case C/D path. Keep V2.\n\n```json\n{json.dumps(frozen, indent=2)}\n```\n"
            if rule is None
            else f"**Candidate:** `{rule.name}`\n\nActivate after obs first contact; stop on obs bilateral / max_steps / contact loss.\n\n"
            f"```json\n{json.dumps(frozen, indent=2)}\n```\n"
        ),
        encoding="utf-8",
    )

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    val_ids = list(split["heldout_train_ics"])[:N_VAL_ICS]
    print(f"=== Held-out validation rule={frozen['chosen']} ===", flush=True)
    held, held_per_ic = _eval_arms(
        env, actor, critic, book, _starts(val_ids), device_t, n_roll=N_VAL_ROLLOUTS, split="train", rule=rule
    )

    p61_starts = list(load_heldout_starts(per_class=PER_CLASS_DEV)["C"])
    print(f"=== P61 C n={len(p61_starts)} ===", flush=True)
    p61, p61_per_ic = _eval_arms(
        env, actor, critic, book, p61_starts, device_t, n_roll=PANEL_C_ROLLOUTS, split="heldout", rule=rule
    )

    try:
        p65_c = list(load_panel_starts().get("C") or [])
    except Exception as e:  # noqa: BLE001
        print(f"P65 load failed: {e}", flush=True)
        p65_c = []
    p65, p65_per_ic = {}, {}
    if p65_c:
        print(f"=== P65 C n={len(p65_c)} ===", flush=True)
        p65, p65_per_ic = _eval_arms(
            env, actor, critic, book, p65_c, device_t, n_roll=PANEL_C_ROLLOUTS, split="p65_extra_dev", rule=rule
        )

    def deltas(blob):
        a, b, z = blob["V2"]["agg"], blob["P73"]["agg"], blob["V0"]["agg"]
        keys = [
            "r3_ready",
            "SUCCESS_RELAXED",
            "reached_opposing",
            "reached_pad4",
            "reached_e9",
            "reached_frame_ok",
            "contact_loss",
        ]
        return {
            "P73_minus_V2": {k: (None if a.get(k) is None or b.get(k) is None else b[k] - a[k]) for k in keys},
            "V2_minus_V0": {k: (None if z.get(k) is None or a.get(k) is None else a[k] - z[k]) for k in keys},
        }

    dh = deltas(held)
    d72 = dh["P73_minus_V2"]

    # Decision cases A–D
    if rule is None:
        case = "D"
        promote_candidate = False
        reason = "No reproducible beneficial window / generalizable rule from Phase 2–3."
    else:
        opp_up = (d72.get("reached_opposing") or 0) > 0.03
        r3_up = (d72.get("r3_ready") or 0) > 0.02
        r3_flat = (d72.get("r3_ready") or 0) >= -0.01
        succ_ok = (d72.get("SUCCESS_RELAXED") or 0) >= -0.01
        # panel check
        panel_ok = True
        if p61:
            dp = deltas(p61)["P73_minus_V2"]
            if (dp.get("r3_ready") or 0) < -0.03:
                panel_ok = False
        if opp_up and r3_up and succ_ok and panel_ok:
            case = "A"
            promote_candidate = True
            reason = "Bilateral and R3 improve on held-out; candidate for further independent validation (not auto-promoted to replace V2 without extra gate)."
        elif opp_up and not r3_up:
            case = "B"
            promote_candidate = False
            reason = "Bilateral improves but R3 does not — do not promote."
        elif not opp_up and not r3_up:
            # Check if CF had gains (oracle) — Case C
            gen = (win.get("generalizes") or {}).get(rule.name)
            if gen:
                case = "C"  # thought generalizable but full-ep failed
                reason = "CF looked useful but full-episode gains did not hold — insufficiently generalizable."
            else:
                case = "C"
                reason = "Only snapshot/oracle-style gains; full-episode no benefit."
            promote_candidate = False
        else:
            case = "C"
            promote_candidate = False
            reason = "Full-episode results do not support promotion."

    # Never auto-replace V2 in this study without Case A + extra validation flag
    replace_v2 = False  # P73 spec: Case A = candidate for additional validation, not immediate replace

    decision = {
        "case": case,
        "promote_candidate": promote_candidate,
        "replace_v2": replace_v2,
        "retain_v2": True,
        "chosen_rule": frozen["chosen"],
        "reason": reason,
        "held_deltas": dh,
        "held_agg": {k: v["agg"] for k, v in held.items()},
        "held_per_ic": held_per_ic,
        "p61_agg": {k: v["agg"] for k, v in p61.items()},
        "p61_per_ic": p61_per_ic,
        "p65_agg": {k: v["agg"] for k, v in p65.items()} if p65 else None,
        "p65_per_ic": p65_per_ic if p65 else None,
    }
    (OUT / "FULL_EPISODE_VALIDATION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "FULL_EPISODE_VALIDATION.md").write_text(
        f"# Full-Episode Validation (P73 Phase 5)\n\n**Case {case}** — retain_v2={True}\n\n{reason}\n\n"
        f"## Held-out\n```json\n{json.dumps(decision['held_agg'], indent=2)}\n```\n\n"
        f"## Δ P73−V2\n```json\n{json.dumps(dh, indent=2)}\n```\n\n"
        f"## P61\n```json\n{json.dumps(decision['p61_agg'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    return decision


if __name__ == "__main__":
    run_phase45()
