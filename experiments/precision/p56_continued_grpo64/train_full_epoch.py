"""P56 full Independent GRPO-64 epoch on P54 schedule; init=P55-A; no expert BC."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
    independent_group,
)
from experiments.precision.p53_independent_grpo64.paths import (
    CLIP,
    GAMMA_RETURN,
    KL_COEF,
    LR,
    TEMPERATURE,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p56_continued_grpo64.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    OUT,
    P55A_CKPT,
    SEED,
    T_MAX,
)
from models.m111_checkpoint import save_m111

STATE_PATH = OUT / "P56_GRPO64_STATE.json"
PROGRESS_CSV = OUT / "P56_FULL_EPOCH_PROGRESS.csv"
CONFIG_PATH = OUT / "P56_TRAIN_CONFIG.json"
EPOCH_CKPT = CKPT_DIR / "P56_epoch1.pt"
LATEST_CKPT = CKPT_DIR / "P56_grpo64_latest.pt"
PID_LOCK = OUT / "P56_GRPO64_TRAIN.pid"
OPT_PATH = CKPT_DIR / "P56_grpo64_opt.pt"

PROGRESS_FIELDS = [
    "group_index",
    "ic_id",
    "class",
    "skill",
    "group_size",
    "success_relaxed_count",
    "success_12s_count",
    "success_new_count",
    "reward_mean",
    "reward_std",
    "advantage_std",
    "nonzero_adv",
    "skipped_zero_var",
    "policy_kl",
    "policy_entropy",
    "drop_before_count",
    "handoff_count",
    "e_pos_mean",
    "e_theta_mean",
    "n_zero_success_group",
    "n_any_success_group",
    "n_mixed_group",
    "n_all_success_group",
    "failed_rollouts",
    "retried_rollouts",
    "loss",
    "elapsed_s",
    "wall_time",
]


def _class_label(skill: str) -> str:
    return {"A1": "A", "B1": "B", "C": "C", "D": "D"}[skill]


def _acquire_pid_lock() -> None:
    import os

    if PID_LOCK.is_file():
        try:
            old = int(PID_LOCK.read_text(encoding="utf-8").strip())
            os.kill(old, 0)
            raise SystemExit(f"Another P56 trainer holds {PID_LOCK} pid={old}")
        except SystemExit:
            raise
        except Exception:
            pass
    PID_LOCK.write_text(str(os.getpid()), encoding="utf-8")


def _release_pid_lock() -> None:
    import os

    if PID_LOCK.is_file():
        try:
            if int(PID_LOCK.read_text(encoding="utf-8").strip()) == os.getpid():
                PID_LOCK.unlink()
        except Exception:
            pass


def _append_progress(row: dict) -> None:
    new = not PROGRESS_CSV.is_file()
    with PROGRESS_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in PROGRESS_FIELDS})


def run_full_epoch(*, device: str = "cuda", resume: bool = True, seed: int = SEED) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _acquire_pid_lock()
    try:
        return _run(device=device, resume=resume, seed=seed)
    finally:
        _release_pid_lock()


def _run(*, device: str, resume: bool, seed: int) -> dict:
    if not P55A_CKPT.is_file():
        raise FileNotFoundError(f"P55-A checkpoint missing: {P55A_CKPT}")
    built = build_p54_dataset(seed=seed)
    schedule = built["schedule"]
    n_train = len(schedule)
    assert GROUP == 64

    config = {
        "experiment": "P56",
        "init": str(P55A_CKPT),
        "init_note": "P55-A control (GRPO-only continuation from P54); no expert BC",
        "expert_bc": False,
        "group_size": GROUP,
        "n_train_ics": n_train,
        "n_expected_rollouts": n_train * GROUP,
        "seed": seed,
        "kl_coef": KL_COEF,
        "clip": CLIP,
        "lr": LR,
        "temperature": TEMPERATURE,
        "gamma_return": GAMMA_RETURN,
        "max_ep_steps": MAX_EP_STEPS,
        "t_max": T_MAX,
        "sampling": "independent",
        "dataset": "P54 rebalanced (fixed proportions)",
        "reward": "P53 PoseRewardWeights unchanged",
        "success": "SUCCESS_RELAXED_12S unchanged",
        "device": device,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    device_t = torch.device(
        device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    )

    start_idx = 0
    state = None
    if resume and STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if not resume:
        if STATE_PATH.is_file():
            STATE_PATH.unlink()
        if PROGRESS_CSV.is_file():
            PROGRESS_CSV.unlink()
        state = None

    if state and int(state.get("group_size", 0)) == GROUP:
        if state.get("epoch_complete") and int(state.get("next_index", 0)) >= n_train:
            print("EPOCH already complete", flush=True)
            return {**state, "already_complete": True}
        start_idx = int(state.get("next_index", 0))
        ckpt_load = Path(state.get("latest_ckpt") or LATEST_CKPT)
        if ckpt_load.is_file() and start_idx > 0:
            actor, book, rho, _ = load_policy_actor(ckpt_load, device_t)
            print(f"RESUME from {start_idx}/{n_train}", flush=True)
        else:
            actor, book, rho, _ = load_policy_actor(P55A_CKPT, device_t)
            start_idx = 0
    else:
        actor, book, rho, _ = load_policy_actor(P55A_CKPT, device_t)

    # KL reference = frozen P55-A (start of this epoch)
    ref_actor, _, _, _ = load_policy_actor(P55A_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    if state and state.get("opt") and Path(state["opt"]).is_file() and start_idx > 0:
        opt.load_state_dict(torch.load(state["opt"], map_location="cpu", weights_only=False))

    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )
    updates_done = int(state.get("updates_done", 0)) if state and start_idx > 0 else 0
    nonzero_signal = int(state.get("nonzero_signal_groups", 0)) if state and start_idx > 0 else 0
    failed_total = int(state.get("failed_rollouts_total", 0)) if state and start_idx > 0 else 0
    by_class_zero = {c: 0 for c in ("A", "B", "C", "D")}
    by_class_groups = {c: 0 for c in ("A", "B", "C", "D")}
    if state and start_idx > 0:
        by_class_zero = dict(state.get("by_class_zero") or by_class_zero)
        by_class_groups = dict(state.get("by_class_groups") or by_class_groups)
    t0_epoch = time.perf_counter()

    for idx in range(start_idx, n_train):
        st = schedule[idx]
        t0 = time.perf_counter()
        failed_rollouts = 0
        retried = 0
        members = None
        for attempt in range(2):
            try:
                members = independent_group(
                    env,
                    actor,
                    st,
                    device_t,
                    group=GROUP,
                    temperature=TEMPERATURE,
                    gamma_return=GAMMA_RETURN,
                )
                if len(members) != GROUP:
                    raise RuntimeError(f"group_size={len(members)}")
                break
            except Exception as e:
                failed_rollouts += 1
                retried += 1
                print(f"RETRY {st.traj_id} {e}", flush=True)
                if attempt == 1:
                    raise
        assert members is not None
        failed_total += failed_rollouts

        returns = [float(m["return"]) for m in members]
        adv, std, skipped, n_nz = _group_advantage(returns)
        n_rel = int(sum(m["SUCCESS_RELAXED_12S"] for m in members))
        n_12 = int(sum(m["SUCCESS_12S"] for m in members))
        n_new = int(sum(m["SUCCESS_NEW"] for m in members))
        n_drop = int(sum(m["drop"] for m in members))
        n_hand = int(sum(m["handoff"] for m in members))
        e_pos, e_th = [], []
        for m in members:
            met = (m.get("info") or {}).get("metrics") or {}
            if met:
                e_pos.append(float(met.get("e_pos_mm") or 0.0))
                e_th.append(float(met.get("theta") or 0.0))

        cl = _class_label(st.skill)
        by_class_groups[cl] = by_class_groups.get(cl, 0) + 1
        if n_rel == 0:
            by_class_zero[cl] = by_class_zero.get(cl, 0) + 1

        opt.zero_grad(set_to_none=True)
        loss_v = kl_v = ent_v = None
        did_update = False
        if not skipped:
            loss, meta = grpo_loss_from_group(
                actor, ref_actor, members, adv, device_t, clip=CLIP, kl_coef=KL_COEF
            )
            if loss is not None:
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0))
                if np.isfinite(gn):
                    opt.step()
                    did_update = True
                    updates_done += 1
                loss_v = float(loss.detach())
                kl_v = float(meta["kl"])
                ent_v = float(meta["entropy_mean"])
                if n_nz > 0:
                    nonzero_signal += 1

        row = {
            "group_index": idx,
            "ic_id": st.traj_id,
            "class": cl,
            "skill": st.skill,
            "group_size": GROUP,
            "success_relaxed_count": n_rel,
            "success_12s_count": n_12,
            "success_new_count": n_new,
            "reward_mean": float(np.mean(returns)),
            "reward_std": float(std),
            "advantage_std": float(np.std(adv)),
            "nonzero_adv": n_nz,
            "skipped_zero_var": int(skipped),
            "policy_kl": kl_v,
            "policy_entropy": ent_v,
            "drop_before_count": n_drop,
            "handoff_count": n_hand,
            "e_pos_mean": float(np.mean(e_pos)) if e_pos else None,
            "e_theta_mean": float(np.mean(e_th)) if e_th else None,
            "n_zero_success_group": int(n_rel == 0),
            "n_any_success_group": int(n_rel > 0),
            "n_mixed_group": int(0 < n_rel < GROUP),
            "n_all_success_group": int(n_rel == GROUP),
            "failed_rollouts": failed_rollouts,
            "retried_rollouts": retried,
            "loss": loss_v,
            "elapsed_s": time.perf_counter() - t0,
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _append_progress(row)
        print(
            json.dumps(
                {
                    "idx": idx,
                    "n": n_train,
                    "skill": st.skill,
                    "succ": n_rel,
                    "drop": n_drop,
                    "upd": int(did_update),
                    "kl": kl_v,
                    "s": round(row["elapsed_s"], 1),
                }
            ),
            flush=True,
        )

        save_m111(
            LATEST_CKPT,
            actor,
            book,
            epoch=idx + 1,
            rho=rho,
            extra={"p56": True, "group": GROUP, "ic_index": idx, "n_train": n_train},
        )
        torch.save(opt.state_dict(), OPT_PATH)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "next_index": idx + 1,
                    "n_train": n_train,
                    "group_size": GROUP,
                    "updates_done": updates_done,
                    "nonzero_signal_groups": nonzero_signal,
                    "failed_rollouts_total": failed_total,
                    "by_class_zero": by_class_zero,
                    "by_class_groups": by_class_groups,
                    "latest_ckpt": str(LATEST_CKPT),
                    "opt": str(OPT_PATH),
                    "last_ic": st.traj_id,
                    "last_skill": st.skill,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if (idx + 1) % 25 == 0:
            save_m111(
                CKPT_DIR / f"P56_grpo64_ic{idx+1:04d}.pt",
                actor,
                book,
                epoch=idx + 1,
                rho=rho,
                extra={"p56": True, "snap": True},
            )

    save_m111(
        EPOCH_CKPT,
        actor,
        book,
        epoch=1,
        rho=rho,
        extra={
            "p56": True,
            "tag": "grpo64_epoch1",
            "complete_epoch": True,
            "init": str(P55A_CKPT),
            "n_train_ics": n_train,
            "n_rollouts": n_train * GROUP,
            "updates_done": updates_done,
            "expert_bc": False,
        },
    )
    summary = {
        "complete": True,
        "epoch_complete": True,
        "n_train_ics": n_train,
        "n_groups": n_train,
        "n_rollouts": n_train * GROUP,
        "updates_done": updates_done,
        "nonzero_signal_groups": nonzero_signal,
        "failed_rollouts_total": failed_total,
        "by_class_zero_success_groups": by_class_zero,
        "by_class_groups": by_class_groups,
        "epoch_ckpt": str(EPOCH_CKPT),
        "elapsed_s": time.perf_counter() - t0_epoch,
        "counts": built["counts"],
    }
    STATE_PATH.write_text(
        json.dumps(
            {
                **(json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file() else {}),
                **summary,
                "next_index": n_train,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "P56_EPOCH_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EPOCH_COMPLETE", json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run_full_epoch(device=args.device, resume=not args.no_resume, seed=args.seed)


if __name__ == "__main__":
    main()
