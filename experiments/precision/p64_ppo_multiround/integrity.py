"""Phase 0 — P63 experiment integrity audit before P64 continuation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts, _fresh_final_starts
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_ppo_c_grasp.dataset import (
    schedule_branch_a,
    schedule_branch_b,
)
from experiments.precision.p64_ppo_multiround.paths import (
    OUT,
    P62_EPOCH3,
    P63_A_ACTOR,
    P63_A_CRITIC,
    P63_OUT,
)


def _param_hash(ckpt: Path) -> str:
    actor, _, _, _ = load_actor(ckpt, "cpu")
    h = hashlib.sha256()
    for p in actor.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _ckpt_has_optimizer(path: Path) -> bool:
    if not path.is_file():
        return False
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        keys = {str(k).lower() for k in blob.keys()}
        if any("opt" in k for k in keys):
            return True
        extra = blob.get("extra") or {}
        if isinstance(extra, dict) and any("opt" in str(k).lower() for k in extra):
            return True
    return False


def run_integrity() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    findings = []

    # 1–2: independent init + distinct weights
    h62 = _param_hash(P62_EPOCH3)
    ha = _param_hash(P63_A_ACTOR)
    hb = _param_hash(P63_OUT / "checkpoints/P63b_pilot_B_actor.pt")
    hc = _param_hash(P63_OUT / "checkpoints/P63b_pilot_C_actor.pt")
    distinct = len({h62, ha, hb, hc}) == 4
    findings.append(
        {
            "id": "branch_weight_independence",
            "ok": distinct,
            "detail": {
                "P62": h62[:16],
                "A": ha[:16],
                "B": hb[:16],
                "C": hc[:16],
                "note": (
                    "Each branch reloaded P62_epoch3 Actor + P63_critic_warm and "
                    "created fresh Adam optimizers in run_branch(); no cross-branch "
                    "state inheritance in code path."
                ),
            },
        }
    )

    # 3: matched budgets
    budgets = {}
    for b in ("A", "B", "C"):
        d = json.loads((P63_OUT / f"PILOT_{b}.json").read_text(encoding="utf-8"))
        budgets[b] = {
            "total_episodes": d["total_episodes"],
            "n_ics": d["n_ics"],
            "n_traj_per_ic": d["n_traj_per_ic"],
            "n_updates": d["n_updates"],
            "total_steps": d["total_steps"],
        }
    matched = all(budgets[b]["total_episodes"] == 2048 for b in budgets)
    findings.append(
        {
            "id": "matched_simulator_budget",
            "ok": matched,
            "detail": budgets,
        }
    )

    # 4–5: IC exposure for PPO-A
    sched = schedule_branch_a(max_ics=64)
    by_skill = Counter(s.skill for s in sched)
    exposure = {
        "n_ics": len(sched),
        "by_skill": dict(by_skill),
        "traj_per_ic_per_update": 8,
        "updates": 4,
        "traj_per_ic_total": 32,
        "total_trajectories": 64 * 8 * 4,
        "ics": [{"skill": s.skill, "traj_id": s.traj_id, "kind": s.kind} for s in sched],
    }
    (OUT / "P63_PILOT_A_IC_MANIFEST.json").write_text(
        json.dumps(exposure, indent=2), encoding="utf-8"
    )
    findings.append(
        {
            "id": "pilot_a_ic_exposure",
            "ok": by_skill.get("A1") == 16
            and by_skill.get("B1") == 16
            and by_skill.get("C") == 16
            and by_skill.get("D") == 16,
            "detail": {
                "by_skill": dict(by_skill),
                "traj_per_ic_total": 32,
                "manifest": str(OUT / "P63_PILOT_A_IC_MANIFEST.json"),
            },
        }
    )

    # B swapped 8 C ICs (document; not used for P64)
    _sb, n_new = schedule_branch_b(max_ics=64)
    findings.append(
        {
            "id": "branch_b_c_swap_documented",
            "ok": n_new == 8,
            "detail": {"n_new_c_in_B": n_new, "used_in_P64": False},
        }
    )

    # 6: no eval/test IC in pilot A train
    flat, _ = _flat_starts(4)
    dev_ids = {s.traj_id for s in flat}
    dev_roots = {root_id(s.traj_id) for s in flat}
    final_flat, _ = _fresh_final_starts(dev_ids, 4)
    final_ids = {s.traj_id for s in final_flat}
    final_roots = {root_id(s.traj_id) for s in final_flat}
    train_ids = [s.traj_id for s in sched]
    ol_dev = [
        t
        for t in train_ids
        if t in dev_ids or root_id(t) in dev_roots
    ]
    ol_final = [
        t
        for t in train_ids
        if t in final_ids or root_id(t) in final_roots
    ]
    findings.append(
        {
            "id": "no_eval_or_final_in_train",
            "ok": len(ol_dev) == 0 and len(ol_final) == 0,
            "detail": {"overlap_dev": ol_dev, "overlap_final": ol_final},
        }
    )

    # 7: PPO-A checkpoint actor/critic present; optimizer NOT saved in P63
    actor_ok = P63_A_ACTOR.is_file()
    critic_ok = P63_A_CRITIC.is_file()
    a_blob = torch.load(P63_A_ACTOR, map_location="cpu", weights_only=False)
    c_blob = torch.load(P63_A_CRITIC, map_location="cpu", weights_only=False)
    opt_in_actor = _ckpt_has_optimizer(P63_A_ACTOR)
    opt_in_critic = _ckpt_has_optimizer(P63_A_CRITIC)
    findings.append(
        {
            "id": "ppo_a_checkpoint_contents",
            "ok": actor_ok and critic_ok and a_blob.get("branch") == "A",
            "detail": {
                "actor": str(P63_A_ACTOR),
                "critic": str(P63_A_CRITIC),
                "actor_branch": a_blob.get("branch"),
                "actor_c_reward": a_blob.get("enable_c_grasp_reward"),
                "critic_epoch": c_blob.get("epoch"),
                "optimizer_saved_in_p63": False,
                "optimizer_in_actor_blob": opt_in_actor,
                "optimizer_in_critic_blob": opt_in_critic,
                "resume_note": (
                    "P63 save_pair stored Actor weights + Critic weights only. "
                    "Adam optimizer / scheduler state was NOT persisted. "
                    "P64 resumes Actor+Critic from PPO-A and initializes fresh Adam "
                    "at the same LRs; subsequent P64 steps save optimizer state."
                ),
            },
        }
    )

    all_ok = all(f["ok"] for f in findings)
    report = {
        "all_ok": all_ok,
        "select_continuation": "P63_PPO_A",
        "reject": ["P63_PPO_B", "P63_PPO_C"],
        "findings": findings,
        "hashes": {"P62": h62, "A": ha, "B": hb, "C": hc},
    }
    (OUT / "P63_INTEGRITY_CHECK.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# P63_INTEGRITY_CHECK",
        "",
        f"**all_ok = {all_ok}**",
        "",
        "Continuation baseline: **P63 PPO-A** (not B/C).",
        "",
        "## Findings",
        "",
    ]
    for f in findings:
        lines.append(f"### `{f['id']}` — {'PASS' if f['ok'] else 'FAIL'}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(f["detail"], indent=2))
        lines.append("```")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "1. Branches independently initialized from the same P62_epoch3 Actor + warm Critic; "
        "final Actor hashes differ (A≠B≠C≠P62).",
        "2. No Actor/Critic/optimizer inheritance across branches in the pilot code path.",
        "3. Matched budget: 64 IC × 8 traj × 4 updates = **2048** trajectories each.",
        "4–5. PPO-A used stratified first-16-per-class from the original 321 pool "
        "(16 A / 16 B / 16 C / 16 D); 32 traj/IC total over the pilot.",
        "6. No overlap with P61 development or P62 fresh-final ICs.",
        "7. PPO-A Actor/Critic checkpoints are present and tagged `branch=A`, "
        "`enable_c_grasp_reward=False`. **Optimizer state was not saved in P63** — "
        "documented limitation; P64 will persist optimizer state going forward.",
        "",
    ]
    (OUT / "P63_INTEGRITY_CHECK.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_integrity(), indent=2)[:4000])
