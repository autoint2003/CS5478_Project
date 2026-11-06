"""Phase 3 — Fixed sequence intervention, no Actor/Critic updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.paths import (
    MAX_EP_STEPS,
    N_HELD_OUT_ICS,
    N_ROLLOUTS_PER_IC,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def _lookup_starts(traj_ids: list[str]) -> list:
    blocked = blocked_eval_ids()
    by_id = {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }
    return [by_id[t] for t in traj_ids if t in by_id]


def _no_pad_contact_obs_available(env) -> bool:
    """Deployable contact gate: use runtime pad counts (also present in tactile obs stream)."""
    g = relative_geometry(env.env.sim)
    return not g["has_pad_contact"]


def rollout_episode(
    env,
    actor,
    critic,
    start,
    device,
    *,
    seed: int,
    rule: dict | None,
) -> dict[str, Any]:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    done = False
    info: dict = {}
    intervening = False
    seq_i = 0
    seq = list((rule or {}).get("sequence") or [])
    max_step = int((rule or {}).get("activation", {}).get("max_policy_step", 7))
    forced_steps = 0
    last_no_contact = None
    first_contact_t = None
    ever_sw = False
    max_pad = 0
    near = r3 = False
    extra_fwd = 0  # forced tokens still call Actor for resid → same cost; count forced overrides

    while not done:
        geo = relative_geometry(env.env.sim)
        if not geo["has_pad_contact"]:
            last_no_contact = {
                "t": int(env.t),
                "e_pos_mm": float(geo["e_pos_mm"]),
                "theta_deg": float(geo["theta_deg"]),
            }
        use_forced = False
        if rule is not None and seq and seq_i < len(seq):
            act = rule.get("activation") or {}
            if (
                int(env.t) <= max_step
                and (not act.get("require_no_pad_contact") or _no_pad_contact_obs_available(env))
            ):
                use_forced = True
                intervening = True
        _tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        if use_forced:
            tok = int(seq[seq_i])
            seq_i += 1
            forced_steps += 1
            extra_fwd += 0  # Actor forward already done for residual; override is free of extra GPU
        else:
            tok = int(_tok)
            # stop intervention once sequence exhausted or contact
            if intervening and (seq_i >= len(seq) or geo["has_pad_contact"]):
                intervening = False
        _r, done, info = env.step(tok, resid)
        ga = relative_geometry(env.env.sim)
        if first_contact_t is None and ga["has_pad_contact"]:
            first_contact_t = int(env.t)
        ever_sw = ever_sw or ga["correct_sidewall"]
        max_pad = max(max_pad, ga["min_pad"])
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"] and ok_s > 0:
            near = True
        if preds["dwell_complete"] and preds["frame_ok"]:
            r3 = True

    return {
        "traj_id": start.traj_id,
        "seed": seed,
        "pre_contact_e_pos_mm": None if not last_no_contact else last_no_contact["e_pos_mm"],
        "pre_contact_theta_deg": None if not last_no_contact else last_no_contact["theta_deg"],
        "first_contact_t": first_contact_t,
        "correct_sidewall": int(ever_sw),
        "min_pad_ge4": int(max_pad >= 4),
        "near_ready": int(near),
        "r3_ready": int(r3 or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "forced_steps": forced_steps,
        "used_intervention": int(forced_steps > 0),
    }


def run_phase3(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    split_path = OUT / "DISCOVERY_VALIDATION_SPLIT.json"
    hyp_path = OUT / "FIXED_SEQUENCE_HYPOTHESIS.json"
    if not split_path.exists() or not hyp_path.exists():
        raise FileNotFoundError("Need Phase 1 outputs: DISCOVERY_VALIDATION_SPLIT + FIXED_SEQUENCE_HYPOTHESIS")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    rule = json.loads(hyp_path.read_text(encoding="utf-8"))
    held_ids = list(split.get("heldout_train_ics") or [])[:N_HELD_OUT_ICS]
    starts = _lookup_starts(held_ids)

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    rows_base = []
    rows_intv = []
    for st in starts:
        for i in range(N_ROLLOUTS_PER_IC):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=93000 + i)
            b = rollout_episode(env, actor, critic, st, device_t, seed=seed, rule=None)
            v = rollout_episode(env, actor, critic, st, device_t, seed=seed, rule=rule)
            rows_base.append(b)
            rows_intv.append(v)
            print(
                f"[p3] {st.traj_id[-28:]} i={i} "
                f"base_r3={b['r3_ready']} intv_r3={v['r3_ready']} "
                f"base_e={b['pre_contact_e_pos_mm']} intv_e={v['pre_contact_e_pos_mm']} "
                f"forced={v['forced_steps']}",
                flush=True,
            )

    def agg(rows: list[dict]) -> dict:
        def mean(k):
            xs = [float(r[k]) for r in rows if r.get(k) is not None]
            return float(np.mean(xs)) if xs else None

        return {
            "n": len(rows),
            "SUCCESS_RELAXED": mean("SUCCESS_RELAXED"),
            "r3_ready": mean("r3_ready"),
            "handoff": mean("handoff"),
            "correct_sidewall": mean("correct_sidewall"),
            "min_pad_ge4": mean("min_pad_ge4"),
            "near_ready": mean("near_ready"),
            "pre_contact_e_pos_mm": mean("pre_contact_e_pos_mm"),
            "frac_intervention_used": mean("used_intervention"),
            "mean_forced_steps": mean("forced_steps"),
        }

    base = agg(rows_base)
    intv = agg(rows_intv)
    delta = {
        k: (None if base[k] is None or intv[k] is None else float(intv[k]) - float(base[k]))
        for k in ("SUCCESS_RELAXED", "r3_ready", "handoff", "correct_sidewall", "min_pad_ge4", "pre_contact_e_pos_mm")
    }

    # Gate for Phase 4 (spec):
    # - Proceed if R3F *progression* improves (r3/handoff OR contact-geometry gates).
    # - Stop if ONLY e_pos improves without contact geometry or R3F.
    r3_up = (delta["r3_ready"] or 0) > 0.01 or (delta["handoff"] or 0) > 0.01
    geom_up = (delta["correct_sidewall"] or 0) > 0.05 or (delta["min_pad_ge4"] or 0) > 0.03
    e_improved = (delta["pre_contact_e_pos_mm"] or 0) < -5.0  # lower is better
    e_only = e_improved and not r3_up and not geom_up
    proceed_ppo = bool((r3_up or geom_up) and not e_only)

    # Per-IC paired
    by_ic = {}
    for b, v in zip(rows_base, rows_intv):
        tid = b["traj_id"]
        by_ic.setdefault(tid, {"base_r3": [], "intv_r3": [], "base_e": [], "intv_e": []})
        by_ic[tid]["base_r3"].append(b["r3_ready"])
        by_ic[tid]["intv_r3"].append(v["r3_ready"])
        if b["pre_contact_e_pos_mm"] is not None:
            by_ic[tid]["base_e"].append(b["pre_contact_e_pos_mm"])
        if v["pre_contact_e_pos_mm"] is not None:
            by_ic[tid]["intv_e"].append(v["pre_contact_e_pos_mm"])
    per_ic = []
    for tid, d in by_ic.items():
        per_ic.append(
            {
                "traj_id": tid,
                "base_r3": float(np.mean(d["base_r3"])),
                "intv_r3": float(np.mean(d["intv_r3"])),
                "delta_r3": float(np.mean(d["intv_r3"])) - float(np.mean(d["base_r3"])),
                "base_e": float(np.mean(d["base_e"])) if d["base_e"] else None,
                "intv_e": float(np.mean(d["intv_e"])) if d["intv_e"] else None,
            }
        )

    report = {
        "rule": rule,
        "n_ics": len(starts),
        "n_rollouts_per_ic": N_ROLLOUTS_PER_IC,
        "baseline": base,
        "intervention": intv,
        "delta": delta,
        "per_ic": per_ic,
        "proceed_to_ppo": proceed_ppo,
        "inference_cost": {
            "extra_actor_forwards_per_forced_step": 0,
            "note": "Forced token overrides Actor sample after one forward for residual; no extra GPU forward vs baseline.",
            "mean_forced_steps": intv.get("mean_forced_steps"),
        },
        "obs_only": True,
    }
    (OUT / "NO_TRAIN_INTERVENTION_RESULTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# No-Train Fixed Sequence Intervention (P69 Phase 3)\n",
        f"Held-out train ICs: **{len(starts)}** × {N_ROLLOUTS_PER_IC} rollouts (matched seeds).\n",
        f"Rule sequence: `{rule.get('sequence')}` families=`{rule.get('families')}`\n",
        "## Aggregate\n",
        f"| Metric | Baseline | Intervention | Δ |",
        f"|---|---:|---:|---:|",
    ]
    for k in ("SUCCESS_RELAXED", "r3_ready", "handoff", "correct_sidewall", "min_pad_ge4", "pre_contact_e_pos_mm"):
        md.append(f"| {k} | {base.get(k)} | {intv.get(k)} | {delta.get(k)} |")
    md.append(f"\n**Proceed to PPO**: `{proceed_ppo}`\n")
    md.append("## Per-IC\n")
    for row in per_ic:
        md.append(
            f"- `{row['traj_id'][-40:]}`: Δr3={row['delta_r3']:.3f} "
            f"base_e={row['base_e']} intv_e={row['intv_e']}"
        )
    (OUT / "NO_TRAIN_INTERVENTION_RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:25]), flush=True)
    return report


if __name__ == "__main__":
    run_phase3()
