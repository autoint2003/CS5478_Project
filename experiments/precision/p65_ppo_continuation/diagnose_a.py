"""Phase 1 — A regression diagnosis across P63-A .. P64_step4."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p61_reproducible_joint_eval.eval_core import (
    load_actor,
    make_eval_env,
    rollout_seeded,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts
from experiments.precision.p62_grpo_optimization.paths import DEV_ROLLOUTS
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
from experiments.precision.p65_ppo_continuation.paths import (
    OUT,
    P63_A_ACTOR,
    P64_OUT,
    P64_STEP1_ACTOR,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
)


def _stage(row_or_info: dict) -> str:
    """Earliest recovery stage reached."""
    if int(row_or_info.get("SUCCESS_RELAXED_12S") or row_or_info.get("n_success_relaxed") or 0) > 0:
        # for aggregate rows use p_relaxed elsewhere
        pass
    succ = int(row_or_info.get("SUCCESS_RELAXED_12S") or 0)
    hand = int(row_or_info.get("handoff") or 0)
    stab = int(row_or_info.get("stable_hold") or row_or_info.get("p_stable") or 0)
    rec = int(row_or_info.get("recatch") or 0)
    if succ:
        return "SUCCESS"
    if hand:
        return "HANDOFF"
    if stab:
        return "STABLE_RECATCH"
    if rec:
        return "RECATCH"
    if int(row_or_info.get("drop") or 0):
        return "DROP"
    return "TIMEOUT_OR_OTHER"


def _load_step_per_ic(step: int) -> list[dict]:
    p = P64_OUT / f"P64_step{step}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return list(d.get("eval_full_per_ic") or [])


def _eval_actor_a_ics(tag: str, ckpt: Path, starts, device: str) -> list[dict]:
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(ckpt, device_t)
    env = make_eval_env(book, rho)
    rows = []
    for st in starts:
        succ = drop = hand = rec = stab = 0
        stages = []
        fails = Counter()
        for i in range(DEV_ROLLOUTS):
            seed = rollout_seed(split="heldout", traj_id=st.traj_id, rollout_i=i)
            m = rollout_seeded(env, actor, st, device_t, seed=seed)
            succ += int(m["SUCCESS_RELAXED_12S"])
            drop += int(m["drop"])
            hand += int(m["handoff"])
            rec += int(m["recatch"])
            stab += int(m["stable_hold"])
            stages.append(_stage(m))
            if not m["SUCCESS_RELAXED_12S"]:
                if m["drop"] and not m["recatch"]:
                    fails["DROP_NO_RECATCH"] += 1
                elif m["drop"] and m["recatch"] and not m["stable_hold"]:
                    fails["DROP_AFTER_RECATCH"] += 1
                elif m["drop"] and m["stable_hold"]:
                    fails["DROP_AFTER_STABLE"] += 1
                elif m["handoff"]:
                    fails["HANDOFF_THEN_FAIL"] += 1
                else:
                    fails["TIMEOUT"] += 1
        n = DEV_ROLLOUTS
        rows.append(
            {
                "tag": tag,
                "skill": st.skill,
                "traj_id": st.traj_id,
                "n_success_relaxed": succ,
                "p_relaxed": succ / n,
                "p_drop": drop / n,
                "p_handoff": hand / n,
                "p_recatch": rec / n,
                "p_stable": stab / n,
                "zero_success": int(succ == 0),
                "stage_mode": Counter(stages).most_common(1)[0][0],
                "fail_dist": dict(fails),
            }
        )
        print(
            f"[{tag}/A] {st.traj_id}: {succ}/{n} ({succ/n:.3f}) stage_mode={rows[-1]['stage_mode']}",
            flush=True,
        )
    return rows


def _per_ic_table_from_agg(per: list[dict], tag: str) -> list[dict]:
    out = []
    for r in per:
        if r.get("skill") != "A1":
            continue
        out.append(
            {
                "tag": tag,
                "traj_id": r["traj_id"],
                "n_success_relaxed": r.get("n_success_relaxed"),
                "p_relaxed": r.get("p_relaxed"),
                "p_recatch": r.get("p_recatch"),
                "p_stable": r.get("p_stable"),
                "p_handoff": r.get("p_handoff"),
                "zero_success": r.get("zero_success"),
            }
        )
    return out


def policy_drift_a_vs_step1(*, device: str = "cuda") -> dict:
    """KL of later actors vs P64_step1 on fixed A histories (reset states)."""
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    flat, held = _flat_starts(4)
    a_starts = held["A1"]
    ref, book, rho, _ = load_actor(P64_STEP1_ACTOR, device_t)
    env = make_eval_env(book, rho)

    def logp(actor, st):
        env.reset(st)
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device_t, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device_t, dtype=torch.float32)
        with torch.no_grad():
            logits, _, _ = actor(obs, prev)
            return torch.log_softmax(mask_handoff_logits(logits[-1].float()), dim=-1)

    def kl(p_log, q_log):
        p = p_log.exp()
        return float((p * (p_log - q_log)).sum().cpu())

    ref.eval()
    rows = []
    ckpts = {
        "P63_PPO_A": P63_A_ACTOR,
        "P64_step2": P64_OUT / "checkpoints/P64_step2_actor.pt",
        "P64_step3": P64_OUT / "checkpoints/P64_step3_actor.pt",
        "P64_step4": P64_STEP4_ACTOR,
    }
    for tag, path in ckpts.items():
        cur, _, _, _ = load_actor(path, device_t)
        cur.eval()
        kls, ents_ref, ents_cur, top_delta = [], [], [], Counter()
        for st in a_starts:
            lr = logp(ref, st)
            lc = logp(cur, st)
            kls.append(kl(lc, lr))
            ents_ref.append(float(-(lr.exp() * lr).sum().cpu()))
            ents_cur.append(float(-(lc.exp() * lc).sum().cpu()))
            # token shift
            d = (lc.exp() - lr.exp()).abs()
            top = int(torch.argmax(d).item())
            top_delta[top] += 1
        rows.append(
            {
                "tag": tag,
                "mean_kl_vs_step1": float(np.mean(kls)),
                "max_kl_vs_step1": float(np.max(kls)),
                "entropy_step1_mean": float(np.mean(ents_ref)),
                "entropy_cur_mean": float(np.mean(ents_cur)),
                "top_changed_tokens": top_delta.most_common(8),
            }
        )
    return {"ref": "P64_step1", "n_a_ics": len(a_starts), "rows": rows}


def critic_advantage_a(*, device: str = "cuda", n_traj: int = 4) -> dict:
    """On-policy A rollouts under P64_step4 for value/advantage diagnostics."""
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    # use matching critic
    from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor as la

    act2, _, _, _ = la(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, act2, device_t)
    env = make_env(book, rho, enable_c_grasp_reward=False)
    flat, held = _flat_starts(4)
    vals, rets, advs, succ_flags = [], [], [], []
    for st in held["A1"]:
        for _ in range(n_traj):
            env.reset(st)
            ep_v, ep_r, ep_done = [], [], []
            done = False
            info = {}
            while not done:
                tok, resid, logp, val = act_with_value(
                    act2, critic, env.obs_seq, env.prev_seq, device_t
                )
                ep_v.append(val)
                r, done, info = env.step(tok, resid)
                ep_r.append(float(r))
                ep_done.append(1.0 if done else 0.0)
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
            succ_flags.append(int(info.get("SUCCESS_RELAXED_12S") or 0))
    y_pred = np.asarray(vals, np.float32)
    y_true = np.asarray(rets, np.float32)
    succ_ret = [np.sum(rets)]  # placeholder
    return {
        "n_value_steps": len(vals),
        "explained_var": explained_variance(y_pred, y_true),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "adv_mean": float(np.mean(advs)),
        "adv_std": float(np.std(advs)),
        "value_bias": float(np.mean(y_pred - y_true)),
        "n_episodes": len(succ_flags),
        "ep_success_rate": float(np.mean(succ_flags)),
    }


def run_a_diagnosis(*, device: str = "cuda", skip_p63_eval: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    flat, held = _flat_starts(4)
    a_starts = held["A1"]

    by_tag = {}
    if not skip_p63_eval:
        print("=== Eval P63-A on A ICs ===", flush=True)
        by_tag["P63_PPO_A"] = _eval_actor_a_ics("P63_PPO_A", P63_A_ACTOR, a_starts, device)
    for st in (1, 2, 3, 4):
        by_tag[f"P64_step{st}"] = _per_ic_table_from_agg(_load_step_per_ic(st), f"P64_step{st}")

    # Align traj_ids
    trajs = [s.traj_id for s in a_starts]
    matrix = []
    for tid in trajs:
        row = {"traj_id": tid}
        for tag, rows in by_tag.items():
            m = next((r for r in rows if r["traj_id"] == tid), None)
            row[tag] = {
                "n_success": None if m is None else m.get("n_success_relaxed"),
                "p_relaxed": None if m is None else m.get("p_relaxed"),
            }
        # regression vs step1
        p1 = row.get("P64_step1", {}).get("p_relaxed")
        p4 = row.get("P64_step4", {}).get("p_relaxed")
        if p1 is not None and p4 is not None:
            row["delta_step4_minus_step1"] = float(p4) - float(p1)
            row["lost_recoverable"] = bool(p1 > 0.05 and p4 + 1e-12 < p1 - 0.05)
        matrix.append(row)

    deltas = [r.get("delta_step4_minus_step1") for r in matrix if r.get("delta_step4_minus_step1") is not None]
    concentrated = False
    if deltas:
        abs_d = [abs(x) for x in deltas]
        concentrated = max(abs_d) > 0.6 * sum(abs_d)

    print("=== A policy drift vs step1 ===", flush=True)
    drift = policy_drift_a_vs_step1(device=device)
    print("=== A critic/advantage (P64_step4) ===", flush=True)
    crit = critic_advantage_a(device=device)

    report = {
        "n_a_ics": len(trajs),
        "per_ic": matrix,
        "by_tag_rows": {k: v for k, v in by_tag.items()},
        "regression_summary": {
            "mean_delta_step4_vs_step1": float(np.mean(deltas)) if deltas else None,
            "concentrated_in_one_ic": concentrated,
            "n_lost_recoverable": int(sum(1 for r in matrix if r.get("lost_recoverable"))),
        },
        "drift_vs_step1": drift,
        "critic_advantage_p64_step4": crit,
    }
    (OUT / "A_REGRESSION_DIAGNOSIS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_a_md(report)
    return report


def _write_a_md(report: dict):
    lines = [
        "# A_REGRESSION_DIAGNOSIS",
        "",
        f"A ICs (P61 heldout): **{report['n_a_ics']}**",
        "",
        "## Per-IC success /64 (p_relaxed)",
        "",
        "| traj_id | P63-A | step1 | step2 | step3 | step4 | Δ4−1 | lost? |",
        "|---------|------:|------:|------:|------:|------:|------:|:-----:|",
    ]
    for r in report["per_ic"]:

        def p(tag):
            v = (r.get(tag) or {}).get("p_relaxed")
            return "—" if v is None else f"{v:.3f}"

        lines.append(
            f"| `{r['traj_id']}` | {p('P63_PPO_A')} | {p('P64_step1')} | {p('P64_step2')} | "
            f"{p('P64_step3')} | {p('P64_step4')} | {r.get('delta_step4_minus_step1')} | "
            f"{r.get('lost_recoverable')} |"
        )
    rs = report["regression_summary"]
    lines += [
        "",
        f"- mean Δ(step4−step1) = **{rs.get('mean_delta_step4_vs_step1')}**",
        f"- concentrated in one IC = **{rs.get('concentrated_in_one_ic')}**",
        f"- lost previously recoverable ICs = **{rs.get('n_lost_recoverable')}**",
        "",
        "## Policy drift vs P64_step1 (fixed A resets)",
        "",
        "```json",
        json.dumps(report["drift_vs_step1"], indent=2),
        "```",
        "",
        "## Critic / advantage (P64_step4 on A)",
        "",
        "```json",
        json.dumps(report["critic_advantage_p64_step4"], indent=2),
        "```",
        "",
        "KL evidence is correlative only — not causal proof of A regression.",
        "",
    ]
    (OUT / "A_REGRESSION_DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-p63-eval", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run_a_diagnosis(device=args.device, skip_p63_eval=args.skip_p63_eval), indent=2)[:2500])
