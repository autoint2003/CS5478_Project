"""Multi-epoch Independent GRPO-64 continuation from P57_A_epoch1 (unchanged objective)."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
    independent_group,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p62_grpo_optimization.eval_dev import (
    eval_checkpoint_dev,
    paired_delta,
)
from experiments.precision.p62_grpo_optimization.paths import (
    CKPT_DIR,
    CLIP,
    GAMMA_RETURN,
    GROUP,
    KL_COEF,
    LR,
    MAX_EP_STEPS,
    MAX_EXTRA_EPOCHS,
    OUT,
    P57A_CKPT,
    P57A_OPT,
    SEED,
    T_MAX,
    TEMPERATURE,
)
from models.m111_checkpoint import save_m111

PROGRESS_FIELDS = [
    "epoch",
    "group_index",
    "ic_id",
    "class",
    "skill",
    "success_relaxed_count",
    "reward_mean",
    "reward_std",
    "skipped_zero_var",
    "policy_kl",
    "policy_entropy",
    "loss",
    "grad_norm",
    "handoff_count",
    "drop_before_count",
    "elapsed_s",
]


def _class_label(skill: str) -> str:
    return {"A1": "A", "B1": "B", "C": "C", "D": "D"}[skill]


def run_continuation(*, device: str = "cuda", max_extra_epochs: int = MAX_EXTRA_EPOCHS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if not P57A_CKPT.is_file():
        raise FileNotFoundError(P57A_CKPT)

    ds = build_p54_dataset(seed=SEED)
    schedule = list(ds["schedule"])
    n_train = len(schedule)
    assert n_train == 321, f"expected 321 ICs, got {n_train}"

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(P57A_CKPT, device_t)
    # Frozen ref = start of this continuation stage (P57_A_epoch1)
    ref_actor, _, _, _ = load_policy_actor(P57A_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    opt_restored = False
    opt_note = "fresh Adam(lr=1e-5); P57 opt state not loaded"
    if P57A_OPT.is_file():
        try:
            opt.load_state_dict(torch.load(P57A_OPT, map_location=device_t, weights_only=False))
            # Ensure LR continuity
            for pg in opt.param_groups:
                pg["lr"] = LR
            opt_restored = True
            opt_note = f"restored {P57A_OPT.name}; lr forced to {LR}"
        except Exception as e:
            opt_note = f"failed to restore optimizer ({type(e).__name__}: {e}); fresh Adam lr={LR}"

    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    cfg = {
        "experiment": "P62",
        "init": str(P57A_CKPT),
        "frozen_ref": str(P57A_CKPT),
        "optimizer_restored": opt_restored,
        "optimizer_note": opt_note,
        "schedule": "P54 A60/B100/C68/D93 = 321",
        "group": GROUP,
        "rollouts_per_epoch": n_train * GROUP,
        "max_extra_epochs": max_extra_epochs,
        "epoch_labels": [f"P62_epoch{e}" for e in range(2, 2 + max_extra_epochs)],
        "lr": LR,
        "kl_coef": KL_COEF,
        "clip": CLIP,
        "temperature": TEMPERATURE,
        "gamma_return": GAMMA_RETURN,
        "reward": "P57-A / PoseRewardWeights (no stabilization)",
        "expert_bc": False,
        "seed": SEED,
        "device": str(device_t),
        "note": (
            "Logical continuation of P57-A stage: epoch1 already completed; "
            "P62 trains epoch2.. up to epoch4. Fresh rollout randomness between epochs "
            "(torch/numpy not reseeded to a fixed value each epoch)."
        ),
    }
    (OUT / "TRAINING_CONFIG.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Baseline dev eval (P57_A)
    print("=== dev eval baseline P57_A ===", flush=True)
    base_dev = eval_checkpoint_dev("P57_A", P57A_CKPT, device=device)
    val_blob = {"P57_A": {k: v for k, v in base_dev.items() if k != "per_ic"}, "epochs": {}}
    (OUT / "VALIDATION_RESULTS.json").write_text(json.dumps(val_blob, indent=2), encoding="utf-8")

    progress = OUT / "P62_TRAIN_PROGRESS.csv"
    if progress.is_file():
        progress.unlink()

    state_path = OUT / "P62_TRAIN_STATE.json"
    start_epoch = 2
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        start_epoch = int(st.get("next_epoch", 2))
        latest = st.get("latest_ckpt")
        if latest and Path(latest).is_file() and start_epoch > 2:
            actor, book, rho, _ = load_policy_actor(latest, device_t)
            opt_p = st.get("opt")
            if opt_p and Path(opt_p).is_file():
                opt.load_state_dict(torch.load(opt_p, map_location=device_t, weights_only=False))
                for pg in opt.param_groups:
                    pg["lr"] = LR
            print(f"RESUME from epoch {start_epoch} ckpt={latest}", flush=True)

    epoch_summaries = []
    stop_reason = None
    last_dev = base_dev

    for epoch in range(start_epoch, 2 + max_extra_epochs):
        print(f"=== P62 training epoch {epoch} ===", flush=True)
        t_ep = time.perf_counter()
        updates = 0
        zero_var = 0
        by_succ = {"A": 0, "B": 0, "C": 0, "D": 0}
        by_n = {"A": 0, "B": 0, "C": 0, "D": 0}
        # Fresh randomness: advance torch seed with epoch salt once at epoch start
        torch.manual_seed(SEED + 62000 + epoch * 9973)
        np.random.seed(SEED + 62000 + epoch * 9973)

        for gi, st in enumerate(schedule):
            t1 = time.perf_counter()
            members = independent_group(
                env, actor, st, device_t, group=GROUP, temperature=TEMPERATURE, gamma_return=GAMMA_RETURN
            )
            returns = [float(m["return"]) for m in members]
            adv, std, skipped, n_nz = _group_advantage(returns)
            n_rel = int(sum(m["SUCCESS_RELAXED_12S"] for m in members))
            n_hand = int(sum(m["handoff"] for m in members))
            n_drop = int(sum(m["drop"] for m in members))
            cl = _class_label(st.skill)
            by_n[cl] += 1
            by_succ[cl] += n_rel

            loss_v = kl_v = ent_v = gn = None
            opt.zero_grad(set_to_none=True)
            if skipped:
                zero_var += 1
            else:
                loss, meta = grpo_loss_from_group(
                    actor, ref_actor, members, adv, device_t, clip=CLIP, kl_coef=KL_COEF
                )
                if loss is not None:
                    loss.backward()
                    gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                    if np.isfinite(gn):
                        opt.step()
                        updates += 1
                    loss_v = float(loss.detach())
                    kl_v = float(meta["kl"])
                    ent_v = float(meta["entropy_mean"])

            row = {
                "epoch": epoch,
                "group_index": gi + 1,
                "ic_id": st.traj_id,
                "class": cl,
                "skill": st.skill,
                "success_relaxed_count": n_rel,
                "reward_mean": float(np.mean(returns)),
                "reward_std": float(std),
                "skipped_zero_var": int(skipped),
                "policy_kl": "" if kl_v is None else kl_v,
                "policy_entropy": "" if ent_v is None else ent_v,
                "loss": "" if loss_v is None else loss_v,
                "grad_norm": "" if gn is None else gn,
                "handoff_count": n_hand,
                "drop_before_count": n_drop,
                "elapsed_s": time.perf_counter() - t1,
            }
            new = not progress.is_file()
            with progress.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            if (gi + 1) % 10 == 0 or gi == 0:
                print(
                    f"[ep{epoch} {gi+1}/{n_train}] {cl} succ={n_rel}/64 skip={int(skipped)} "
                    f"upd={updates} ({row['elapsed_s']:.1f}s)",
                    flush=True,
                )

        ckpt_path = CKPT_DIR / f"P62_epoch{epoch}.pt"
        save_m111(
            ckpt_path,
            actor,
            book,
            epoch=epoch,
            rho=rho,
            extra={
                "p62": True,
                "logical_stage": "P57_A_continuation",
                "epoch": epoch,
                "init": str(P57A_CKPT),
                "frozen_ref": str(P57A_CKPT),
                "n_train_ics": n_train,
                "group": GROUP,
            },
        )
        opt_path = CKPT_DIR / f"P62_epoch{epoch}_opt.pt"
        torch.save(opt.state_dict(), opt_path)
        latest = CKPT_DIR / "P62_latest.pt"
        save_m111(
            latest, actor, book, epoch=epoch, rho=rho, extra={"p62": True, "epoch": epoch}
        )

        print(f"=== dev eval P62_epoch{epoch} ===", flush=True)
        dev = eval_checkpoint_dev(f"P62_epoch{epoch}", ckpt_path, device=device)
        paired_base = paired_delta(base_dev, dev)
        paired_prev = paired_delta(last_dev, dev)

        ep_sum = {
            "epoch": epoch,
            "ckpt": str(ckpt_path),
            "wall_s": time.perf_counter() - t_ep,
            "updates": updates,
            "zero_var_groups": zero_var,
            "train_p_relaxed_by_class": {
                c: by_succ[c] / max(by_n[c] * GROUP, 1) for c in ("A", "B", "C", "D")
            },
            "dev": {k: v for k, v in dev.items() if k != "per_ic"},
            "paired_vs_P57_A": paired_base,
            "paired_vs_prev": paired_prev,
        }
        epoch_summaries.append(ep_sum)
        val_blob["epochs"][f"epoch{epoch}"] = ep_sum
        (OUT / "VALIDATION_RESULTS.json").write_text(
            json.dumps(
                {
                    **val_blob,
                    "P57_A": {k: v for k, v in base_dev.items() if k != "per_ic"},
                    "epoch_per_ic": {
                        **(val_blob.get("epoch_per_ic") or {}),
                        f"epoch{epoch}": dev["per_ic"],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(
                {
                    "next_epoch": epoch + 1,
                    "latest_ckpt": str(ckpt_path),
                    "opt": str(opt_path),
                    "updates_last_epoch": updates,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"epoch_done": epoch, "dev_msp": dev["MSP_RELAXED"], "vs_base": paired_base}, indent=2), flush=True)

        # Early stopping rules
        msp = float(dev["MSP_RELAXED"])
        msp_base = float(base_dev["MSP_RELAXED"])
        msp_prev = float(last_dev["MSP_RELAXED"])
        # Severe broad regression: >2pp overall drop vs baseline
        if msp + 0.02 < msp_base:
            stop_reason = f"severe regression vs P57_A ({msp:.4f} << {msp_base:.4f})"
            last_dev = dev
            break
        if epoch == 2:
            # Continue to epoch3 unless clear instability (already checked)
            last_dev = dev
            continue
        if epoch == 3:
            # Continue to epoch4 only if credible improvement or unresolved convergence
            improved = msp > msp_base + 0.003 or msp > msp_prev + 0.003
            # also continue if still clearly below a rising trend possibility: train updates healthy
            unresolved = updates >= int(0.9 * n_train) and zero_var < 5 and abs(msp - msp_base) < 0.01
            if not (improved or unresolved):
                stop_reason = (
                    f"stop after epoch3: no credible improvement "
                    f"(msp={msp:.4f}, base={msp_base:.4f}, prev={msp_prev:.4f})"
                )
                last_dev = dev
                break
            last_dev = dev
            continue
        last_dev = dev

    result = {
        "config": cfg,
        "epochs": epoch_summaries,
        "stop_reason": stop_reason or "completed_max_epochs",
        "baseline_dev_msp": base_dev["MSP_RELAXED"],
    }
    (OUT / "P62_TRAIN_SUMMARY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-extra-epochs", type=int, default=MAX_EXTRA_EPOCHS)
    args = ap.parse_args()
    print(json.dumps(run_continuation(device=args.device, max_extra_epochs=args.max_extra_epochs), indent=2)[:2000])
