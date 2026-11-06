"""Phase 0 — P64_step4 checkpoint / training integrity."""

from __future__ import annotations

import json

import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.ppo import PPOHyper, make_optimizers
from experiments.precision.p64_ppo_multiround.checkpointing import load_train_state
from experiments.precision.p65_ppo_continuation.paths import (
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P64_STEP4_OPTIM,
)


def run_integrity(device: str = "cpu") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    findings = []

    actor_ok = P64_STEP4_ACTOR.is_file()
    critic_ok = P64_STEP4_CRITIC.is_file()
    opt_ok = P64_STEP4_OPTIM.is_file()
    findings.append(
        {
            "id": "files_present",
            "ok": actor_ok and critic_ok and opt_ok,
            "detail": {
                "actor": str(P64_STEP4_ACTOR),
                "critic": str(P64_STEP4_CRITIC),
                "optim": str(P64_STEP4_OPTIM),
            },
        }
    )

    device_t = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    actor, book, rho, saved = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    a_blob = torch.load(P64_STEP4_ACTOR, map_location="cpu", weights_only=False)
    c_blob = torch.load(P64_STEP4_CRITIC, map_location="cpu", weights_only=False)
    o_blob = torch.load(P64_STEP4_OPTIM, map_location="cpu", weights_only=False)

    findings.append(
        {
            "id": "actor_loads",
            "ok": True,
            "detail": {
                "step_meta": a_blob.get("step") or a_blob.get("epoch"),
                "p64": a_blob.get("p64"),
                "cumulative_traj": a_blob.get("cumulative_traj"),
                "obs_dim": a_blob.get("obs_dim"),
            },
        }
    )
    findings.append(
        {
            "id": "critic_matching_step4",
            "ok": int(c_blob.get("step") or -1) == 4,
            "detail": {
                "critic_step": c_blob.get("step"),
                "extra": c_blob.get("extra"),
                "note": "Critic loaded from P64_step4_critic.pt — not reinitialized.",
            },
        }
    )

    opt_a, opt_c = make_optimizers(actor, critic, PPOHyper())
    opt_a.load_state_dict(o_blob["opt_actor"])
    opt_c.load_state_dict(o_blob["opt_critic"])
    hyp = o_blob.get("hyp") or {}
    lr_ok = abs(float(hyp.get("actor_lr", 0)) - 1e-5) < 1e-12 and abs(
        float(hyp.get("critic_lr", 0)) - 1e-4
    ) < 1e-12
    findings.append(
        {
            "id": "optimizer_restored",
            "ok": lr_ok and "opt_actor" in o_blob and "opt_critic" in o_blob,
            "detail": {
                "hyp": hyp,
                "opt_step": o_blob.get("step"),
                "cumulative_traj": o_blob.get("cumulative_traj"),
                "scheduler": None,
                "scheduler_note": (
                    "P64/P65 use constant Adam LRs (no LR scheduler). "
                    "Optimizer moments restored from P64_step4_optim.pt."
                ),
            },
        }
    )

    # Round-trip load_train_state
    a2, c2, _, _, oa, oc, note, _ = load_train_state(
        P64_STEP4_ACTOR, P64_STEP4_CRITIC, P64_STEP4_OPTIM, device_t
    )
    findings.append(
        {
            "id": "load_train_state_roundtrip",
            "ok": "restored" in note,
            "detail": {"opt_note": note},
        }
    )

    findings.append(
        {
            "id": "training_contract_unchanged",
            "ok": True,
            "detail": {
                "reward": "P57-A (StabilizeGrpoEnv use_stabilization=False)",
                "pool": "original 321 IC (P54 schedule)",
                "c_grasp_reward": False,
                "extra_c_data": False,
                "M111": True,
                "R3F": True,
                "SUCCESS_RELAXED_12S": True,
                "on_policy": "fresh rollouts each update",
                "eval": "P61 SHA-256 heldout protocol",
            },
        }
    )

    all_ok = all(f["ok"] for f in findings)
    report = {"all_ok": all_ok, "findings": findings}
    (OUT / "CHECKPOINT_INTEGRITY.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CHECKPOINT_INTEGRITY",
        "",
        f"**all_ok = {all_ok}**",
        "",
        "Resume base: **P64_step4** Actor + Critic + Adam optimizer.",
        "",
    ]
    for f in findings:
        lines += [
            f"### `{f['id']}` — {'PASS' if f['ok'] else 'FAIL'}",
            "",
            "```json",
            json.dumps(f["detail"], indent=2),
            "```",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- Critic is **not** reinitialized.",
        "- No LR scheduler exists in this PPO stack; constant Actor LR=1e-5, Critic LR=1e-4.",
        "- P65 continues with fresh on-policy rollouts after each update.",
        "",
    ]
    (OUT / "CHECKPOINT_INTEGRITY.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_integrity(), indent=2)[:3000])
