"""Phase 2 — C stagnation diagnosis (failure stages + critic signals)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from experiments.precision.p61_reproducible_joint_eval.eval_core import load_actor
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_critic
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor as load_actor_train
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p65_ppo_continuation.paths import OUT, P64_STEP4_ACTOR, P64_STEP4_CRITIC


def _bucket(info: dict, c_diag: dict) -> str:
    if int(info.get("SUCCESS_RELAXED_12S") or 0):
        return "SUCCESS_RELAXED"
    if int(info.get("handoff") or 0):
        return "HANDOFF_NO_SUCCESS"
    rec = int(c_diag.get("recatch") or 0)
    stab = int(c_diag.get("stable_hold") or 0)
    lost = int(c_diag.get("lost_after_recatch") or 0)
    if int(info.get("drop") or 0):
        if not rec:
            return "FAILED_RECATCH"
        if lost or (rec and not stab):
            return "UNSTABLE_AFTER_RECATCH"
        return "UNSTABLE_AFTER_RECATCH"
    if int(info.get("truncated") or 0) or str(info.get("handoff_reason") or "") == "TIMEOUT":
        if stab:
            return "TIMEOUT_AFTER_STABLE"
        if rec:
            return "TIMEOUT_AFTER_RECATCH"
        return "TIMEOUT"
    return "OTHER"


def diagnostic_c_rollouts(*, device: str = "cuda", n_per_ic: int = 16) -> dict:
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor_train(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = make_env(book, rho, enable_c_grasp_reward=False)
    flat, held = _flat_starts(4)
    starts = held["C"]

    buckets = Counter()
    rows = []
    vals, rets, advs, succ_flags, returns = [], [], [], [], []
    sidewall_hits = 0
    bil_hits = 0
    n_ep = 0
    stable_then_r3f = []
    hand_then_succ = []

    for st in starts:
        ic_succ = 0
        ic_buckets = Counter()
        for i in range(n_per_ic):
            seed = rollout_seed(split="heldout", traj_id=st.traj_id, rollout_i=i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ep_v, ep_r, ep_done = [], [], []
            done = False
            info = {}
            ever_sidewall = False
            ever_bil = False
            max_stable_run = 0
            cur_stable = 0
            while not done:
                tok, resid, logp, val = act_with_value(
                    actor, critic, env.obs_seq, env.prev_seq, device_t
                )
                ep_v.append(val)
                r, done, info = env.step(tok, resid)
                ep_r.append(float(r))
                ep_done.append(1.0 if done else 0.0)
                audit = classify_contacts(env.env.sim)
                if is_correct_sidewall(audit):
                    ever_sidewall = True
                if audit.get("bilateral"):
                    ever_bil = True
                    cur_stable += 1
                    max_stable_run = max(max_stable_run, cur_stable)
                else:
                    cur_stable = 0
            cd = info.get("c_diag") or {}
            b = _bucket(info, cd)
            buckets[b] += 1
            ic_buckets[b] += 1
            n_ep += 1
            sidewall_hits += int(ever_sidewall)
            bil_hits += int(ever_bil)
            succ = int(info.get("SUCCESS_RELAXED_12S") or 0)
            ic_succ += succ
            hand = int(info.get("handoff") or 0)
            stab = int(cd.get("stable_hold") or 0)
            if stab:
                stable_then_r3f.append(int(hand))
            if hand:
                hand_then_succ.append(succ)

            adv, ret = compute_gae(
                np.asarray(ep_r, np.float32),
                np.asarray(ep_v + [0.0], np.float32),
                np.asarray(ep_done, np.float32),
                1.0,
                0.95,
            )
            vals.extend(ep_v)
            rets.extend(ret.tolist())
            advs.extend(adv.tolist())
            succ_flags.append(succ)
            returns.append(float(np.sum(ep_r)))

        rows.append(
            {
                "traj_id": st.traj_id,
                "n": n_per_ic,
                "n_success": ic_succ,
                "p_success": ic_succ / n_per_ic,
                "buckets": dict(ic_buckets),
            }
        )
        print(f"[C-diag] {st.traj_id}: succ={ic_succ}/{n_per_ic} buckets={dict(ic_buckets)}", flush=True)

    y_pred = np.asarray(vals, np.float32)
    y_true = np.asarray(rets, np.float32)
    ret_succ = [r for r, s in zip(returns, succ_flags) if s]
    ret_fail = [r for r, s in zip(returns, succ_flags) if not s]
    overlap = None
    if ret_succ and ret_fail:
        # crude overlap: fail_p75 vs succ_p25
        overlap = float(np.percentile(ret_fail, 75) - np.percentile(ret_succ, 25))

    report = {
        "policy": str(P64_STEP4_ACTOR),
        "n_episodes": n_ep,
        "n_ics": len(starts),
        "rollouts_per_ic": n_per_ic,
        "bucket_counts": dict(buckets),
        "bucket_rates": {k: v / max(1, n_ep) for k, v in buckets.items()},
        "sidewall_recatch_rate": sidewall_hits / max(1, n_ep),
        "bilateral_rate": bil_hits / max(1, n_ep),
        "p_r3f_given_stable": float(np.mean(stable_then_r3f)) if stable_then_r3f else None,
        "p_success_given_handoff": float(np.mean(hand_then_succ)) if hand_then_succ else None,
        "per_ic": rows,
        "critic": {
            "explained_var": explained_variance(y_pred, y_true),
            "value_bias": float(np.mean(y_pred - y_true)),
            "mae": float(np.mean(np.abs(y_pred - y_true))),
            "adv_mean": float(np.mean(advs)),
            "adv_std": float(np.std(advs)),
            "return_succ_mean": float(np.mean(ret_succ)) if ret_succ else None,
            "return_fail_mean": float(np.mean(ret_fail)) if ret_fail else None,
            "return_overlap_fail_p75_minus_succ_p25": overlap,
            "ep_success_rate": float(np.mean(succ_flags)),
        },
    }
    # hypothesis ranking
    rates = report["bucket_rates"]
    hyp = []
    if rates.get("FAILED_RECATCH", 0) > 0.4:
        hyp.append("contact_geometry_or_intercept_failure_dominant")
    if rates.get("UNSTABLE_AFTER_RECATCH", 0) > 0.25:
        hyp.append("unstable_after_recatch_dominant")
    if report["critic"]["ep_success_rate"] < 0.05 and report["critic"]["explained_var"] < 0.2:
        hyp.append("poor_value_estimation_plus_rare_success")
    if report["critic"]["ep_success_rate"] < 0.05:
        hyp.append("lack_of_successful_exploration")
    report["hypotheses"] = hyp
    return report


def run_c_diagnosis(*, device: str = "cuda", n_per_ic: int = 16) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Also summarize existing P64 C per_ic MSP
    from experiments.precision.p64_ppo_multiround.paths import OUT as P64

    existing = {}
    for st in (1, 2, 3, 4):
        d = json.loads((P64 / f"P64_step{st}.json").read_text(encoding="utf-8"))
        per = [r for r in (d.get("eval_full_per_ic") or []) if r.get("skill") == "C"]
        existing[f"P64_step{st}"] = [
            {"traj_id": r["traj_id"], "p_relaxed": r["p_relaxed"], "p_recatch": r.get("p_recatch"), "p_stable": r.get("p_stable"), "p_handoff": r.get("p_handoff"), "zero": r.get("zero_success")}
            for r in per
        ]

    print("=== C diagnostic rollouts (P64_step4) ===", flush=True)
    diag = diagnostic_c_rollouts(device=device, n_per_ic=n_per_ic)
    report = {"existing_dev_per_ic": existing, "diagnostic": diag}
    (OUT / "C_BOTTLENECK_DIAGNOSIS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# C_BOTTLENECK_DIAGNOSIS",
        "",
        "Policy: P64_step4. Privilege contact geometry used for **audit only**.",
        "",
        "## Existing P61 C MSP by checkpoint",
        "",
    ]
    for tag, rows in existing.items():
        lines.append(f"### {tag}")
        for r in rows:
            lines.append(
                f"- `{r['traj_id']}`: p_rel={r['p_relaxed']:.3f} recatch={r.get('p_recatch')} "
                f"stable={r.get('p_stable')} hand={r.get('p_handoff')} zero={r.get('zero')}"
            )
        lines.append("")
    d = diag
    lines += [
        "## Diagnostic rollout buckets",
        "",
        f"- n_episodes={d['n_episodes']} ({d['rollouts_per_ic']}/IC × {d['n_ics']} ICs)",
        f"- bucket_rates={json.dumps(d['bucket_rates'], indent=2)}",
        f"- sidewall_recatch_rate={d['sidewall_recatch_rate']:.3f}",
        f"- bilateral_rate={d['bilateral_rate']:.3f}",
        f"- P(R3F|stable)={d['p_r3f_given_stable']}",
        f"- P(SUCCESS|HANDOFF)={d['p_success_given_handoff']}",
        "",
        "## Critic / advantage on C",
        "",
        "```json",
        json.dumps(d["critic"], indent=2),
        "```",
        "",
        f"**Hypotheses:** {d['hypotheses']}",
        "",
    ]
    (OUT / "C_BOTTLENECK_DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")
    # also write critic analysis shared file
    (OUT / "CRITIC_ADVANTAGE_ANALYSIS.json").write_text(
        json.dumps({"C": d["critic"], "source": "P64_step4_diagnostic"}, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-per-ic", type=int, default=16)
    args = ap.parse_args()
    print(json.dumps(run_c_diagnosis(device=args.device, n_per_ic=args.n_per_ic), indent=2)[:3000])
