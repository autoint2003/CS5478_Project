"""Probabilistic MSP evaluation: N independent rollouts per IC (default 64)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p26_c_curriculum_rl.ppo_m111 import load_actor_critic_full
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import holdout_ids, root_id
from experiments.precision.p51_four_class_mixed_ppo.paths import P44_EVAL
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.grpo_independent import rollout_one
from experiments.precision.p53_independent_grpo64.paths import (
    CLASS_LABEL,
    MAX_EP_STEPS,
    OUT,
    P47T_CKPT,
    P50_CKPT,
    P51_CKPT,
    SEED,
    SKILLS,
    T_MAX,
    TEMPERATURE,
)
from experiments.precision.p53_independent_grpo64.policy_sample import as_actor, load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights


def load_heldout_starts(per_class: int = 4) -> dict[str, list[MixedStart]]:
    """Held-out ICs from P44 eval CSV + expert jobs (never used for training)."""
    held = holdout_ids()
    jobs = collect_jobs()
    by: dict[str, list[MixedStart]] = {s: [] for s in SKILLS}
    # Prefer holdout-tagged rows from P44
    holdout_tids = {s: set() for s in SKILLS}
    if P44_EVAL.is_file():
        for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")):
            if r.get("tag") != "holdout":
                continue
            sk = r["skill"]
            if sk in holdout_tids:
                holdout_tids[sk].add(r["traj_id"])
    for j in jobs:
        sk = j["skill"]
        if sk not in SKILLS:
            continue
        tid = j["traj_id"]
        hset = held[sk]
        if tid not in holdout_tids[sk] and root_id(tid) not in hset and tid not in hset:
            ok = False
            for h in holdout_tids[sk] | hset:
                if h and (h in tid or tid in h or root_id(tid) in h):
                    ok = True
                    break
            if not ok:
                continue
        if j.get("capture") is not None:
            by[sk].append(
                MixedStart(
                    skill=sk,
                    traj_id=tid,
                    family=j.get("family") or sk,
                    kind="capture",
                    capture=j["capture"],
                    ref=j.get("ref"),
                )
            )
        elif j.get("ep") is not None:
            by[sk].append(
                MixedStart(
                    skill=sk,
                    traj_id=tid,
                    family=j.get("family") or sk,
                    kind="ep",
                    ep=j["ep"],
                )
            )
    rng = np.random.default_rng(SEED)
    out: dict[str, list[MixedStart]] = {}
    for sk in SKILLS:
        pool = by[sk]
        rng.shuffle(pool)
        # de-dup traj_id
        seen = set()
        uniq = []
        for s in pool:
            if s.traj_id in seen:
                continue
            seen.add(s.traj_id)
            uniq.append(s)
        out[sk] = uniq[:per_class]
    return out


def load_eval_actor(ckpt: Path, device):
    actor, book, rho, saved = load_policy_actor(ckpt, device)
    # PPO checkpoints may have companion value head; actor weights already in .pt
    vpath = Path(ckpt).with_suffix(".value.pt")
    if vpath.is_file():
        try:
            ac = load_actor_critic_full(ckpt, book, device, rho)
            actor = ac.actor
        except Exception:
            pass
    return as_actor(actor), book, rho


def pass_at_k(successes: list[int], ks=(1, 4, 8, 16, 32, 64)) -> dict[str, float]:
    """Empirical Pass@k from binary success list of length N (order = rollout order)."""
    n = len(successes)
    out = {}
    for k in ks:
        if k > n:
            continue
        # unbiased: 1 - C(n-c,k)/C(n,k) if n-c>=k else 1
        c = int(sum(successes))
        if n - c < k:
            out[f"pass@{k}"] = 1.0
        elif c == 0:
            out[f"pass@{k}"] = 0.0
        else:
            # product form
            p = 1.0
            for i in range(k):
                p *= (n - c - i) / (n - i)
            out[f"pass@{k}"] = float(1.0 - p)
    return out


def eval_checkpoint(
    tag: str,
    ckpt: Path,
    starts: dict[str, list[MixedStart]],
    *,
    n_rollouts: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho = load_eval_actor(ckpt, device_t)
    actor.eval()
    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    per_ic = []
    for sk in SKILLS:
        for st in starts.get(sk) or []:
            succ_r, succ12, succ_new, drops, hands = [], [], [], [], []
            fails = defaultdict(int)
            e_pos, e_th = [], []
            for i in range(int(n_rollouts)):
                m = rollout_one(
                    env,
                    actor,
                    st,
                    device_t,
                    temperature=TEMPERATURE,
                    gamma_return=1.0,
                )
                info = m["info"]
                succ_r.append(int(m["SUCCESS_RELAXED_12S"]))
                succ12.append(int(m["SUCCESS_12S"]))
                succ_new.append(int(m["SUCCESS_NEW"]))
                drops.append(int(m["drop"]))
                hands.append(int(m["handoff"]))
                fr = info.get("fail_reason") or ("OK" if m["SUCCESS_RELAXED_12S"] else "UNKNOWN")
                fails[str(fr)] += 1
                met = info.get("metrics") or {}
                if met:
                    e_pos.append(float(met.get("e_pos_mm") or 0.0))
                    e_th.append(float(met.get("theta") or 0.0))
            n = len(succ_r)
            row = {
                "tag": tag,
                "skill": sk,
                "class": CLASS_LABEL[sk],
                "traj_id": st.traj_id,
                "n_rollouts": n,
                "p_relaxed": float(sum(succ_r) / n),
                "p_12s": float(sum(succ12) / n),
                "p_new": float(sum(succ_new) / n),
                "n_success_relaxed": int(sum(succ_r)),
                "n_drop": int(sum(drops)),
                "n_handoff": int(sum(hands)),
                "zero_success": int(sum(succ_r) == 0),
                "fail_causes": dict(fails),
                "e_pos_mean": float(np.mean(e_pos)) if e_pos else None,
                "e_theta_mean": float(np.mean(e_th)) if e_th else None,
                **pass_at_k(succ_r),
            }
            per_ic.append(row)
            print(
                f"[{tag}] {sk} {st.traj_id}: p_rel={row['p_relaxed']:.3f} "
                f"p12={row['p_12s']:.3f} pnew={row['p_new']:.3f}",
                flush=True,
            )

    def msp(key: str) -> float:
        return float(np.mean([r[key] for r in per_ic])) if per_ic else 0.0

    by_class = {}
    for sk in SKILLS:
        rows = [r for r in per_ic if r["skill"] == sk]
        if not rows:
            continue
        by_class[CLASS_LABEL[sk]] = {
            "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in rows])),
            "MSP_12S": float(np.mean([r["p_12s"] for r in rows])),
            "MSP_NEW": float(np.mean([r["p_new"] for r in rows])),
            "n_ic": len(rows),
            "zero_success_ics": int(sum(r["zero_success"] for r in rows)),
        }
    summary = {
        "tag": tag,
        "ckpt": str(ckpt),
        "n_rollouts_per_ic": n_rollouts,
        "temperature": TEMPERATURE,
        "MSP_RELAXED_12S": msp("p_relaxed"),
        "MSP_12S": msp("p_12s"),
        "MSP_NEW": msp("p_new"),
        "n_ic": len(per_ic),
        "zero_success_ics": int(sum(r["zero_success"] for r in per_ic)),
        "by_class": by_class,
        "per_ic": per_ic,
    }
    return summary


def compare_baselines(
    starts: dict[str, list[MixedStart]] | None = None,
    *,
    n_rollouts: int = 64,
    p53_ckpt: Path | None = None,
    device: str = "cpu",
    per_class: int = 2,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if starts is None:
        starts = load_heldout_starts(per_class=per_class)
    ckpts = [("P47T", P47T_CKPT), ("P50", P50_CKPT), ("P51", P51_CKPT)]
    if p53_ckpt and Path(p53_ckpt).is_file():
        ckpts.append(("P53", Path(p53_ckpt)))
    results = {}
    for tag, ck in ckpts:
        if not Path(ck).is_file():
            results[tag] = {"error": f"missing {ck}"}
            continue
        results[tag] = eval_checkpoint(tag, Path(ck), starts, n_rollouts=n_rollouts, device=device)

    # paired ΔMSP vs P47T
    paired = {}
    if "P47T" in results and "per_ic" in results["P47T"]:
        base = {(r["skill"], r["traj_id"]): r for r in results["P47T"]["per_ic"]}
        for tag, summ in results.items():
            if tag == "P47T" or "per_ic" not in summ:
                continue
            deltas = []
            for r in summ["per_ic"]:
                b = base.get((r["skill"], r["traj_id"]))
                if b:
                    deltas.append(r["p_relaxed"] - b["p_relaxed"])
            paired[tag] = {
                "mean_delta_p_relaxed": float(np.mean(deltas)) if deltas else None,
                "n_improved": int(sum(d > 1e-9 for d in deltas)),
                "n_regressed": int(sum(d < -1e-9 for d in deltas)),
                "n_tied": int(sum(abs(d) <= 1e-9 for d in deltas)),
            }
    out = {"summaries": results, "paired_vs_P47T": paired, "n_rollouts": n_rollouts}
    (OUT / "probabilistic_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
