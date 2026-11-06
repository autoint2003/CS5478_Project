"""Full Independent GRPO-64 epoch over all eligible train ICs (resumable)."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.full_dataset import load_manifest_schedule
from experiments.precision.p53_independent_grpo64.grpo_independent import (
    _group_advantage,
    grpo_loss_from_group,
    independent_group,
)
from experiments.precision.p53_independent_grpo64.paths import (
    CKPT_DIR,
    CLIP,
    GAMMA_RETURN,
    KL_COEF,
    LR,
    MAX_EP_STEPS,
    OUT,
    P47T_CKPT,
    SEED,
    SKILLS,
    T_MAX,
    TEMPERATURE,
)
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from models.m111_checkpoint import save_m111

GROUP = 64
STATE_PATH = OUT / "FULL_GRPO64_STATE.json"
PROGRESS_CSV = OUT / "FULL_EPOCH_PROGRESS.csv"
CONFIG_PATH = OUT / "FULL_GRPO64_CONFIG.json"
EPOCH_CKPT = CKPT_DIR / "P53_grpo64_epoch1.pt"
LATEST_CKPT = CKPT_DIR / "P53_grpo64_latest.pt"
PID_LOCK = OUT / "FULL_GRPO64_TRAIN.pid"


def _acquire_pid_lock() -> None:
    import os

    if PID_LOCK.is_file():
        try:
            old = int(PID_LOCK.read_text(encoding="utf-8").strip())
        except Exception:
            old = -1
        # If process still alive, refuse
        try:
            os.kill(old, 0)
            alive = True
        except Exception:
            alive = False
        if alive and old != os.getpid():
            raise SystemExit(f"Another full-epoch trainer holds {PID_LOCK} pid={old}")
    PID_LOCK.write_text(str(os.getpid()), encoding="utf-8")


def _release_pid_lock() -> None:
    import os

    if PID_LOCK.is_file():
        try:
            if int(PID_LOCK.read_text(encoding="utf-8").strip()) == os.getpid():
                PID_LOCK.unlink()
        except Exception:
            pass


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


def _append_progress(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = not PROGRESS_CSV.is_file()
    with PROGRESS_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in PROGRESS_FIELDS})


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state() -> dict | None:
    if not STATE_PATH.is_file():
        return None
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def run_full_epoch(
    *,
    device: str = "cuda",
    resume: bool = True,
    seed: int = SEED,
    max_ics: int | None = None,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _acquire_pid_lock()
    try:
        return _run_full_epoch_inner(
            device=device, resume=resume, seed=seed, max_ics=max_ics
        )
    finally:
        _release_pid_lock()


def _run_full_epoch_inner(
    *,
    device: str = "cuda",
    resume: bool = True,
    seed: int = SEED,
    max_ics: int | None = None,
) -> dict:
    built = load_manifest_schedule(seed=seed)
    schedule = built["schedule"]
    if max_ics is not None:
        schedule = schedule[: int(max_ics)]
    n_train = len(schedule)
    assert GROUP == 64

    config = {
        "group_size": GROUP,
        "n_train_ics": n_train,
        "n_expected_groups": n_train,
        "n_expected_rollouts": n_train * GROUP,
        "seed": seed,
        "init": str(P47T_CKPT),
        "kl_coef": KL_COEF,
        "clip": CLIP,
        "lr": LR,
        "temperature": TEMPERATURE,
        "gamma_return": GAMMA_RETURN,
        "max_ep_steps": MAX_EP_STEPS,
        "t_max": T_MAX,
        "sampling": "independent",
        "tree_branching": False,
        "device": device,
        "manifest": str(OUT / "FULL_TRAIN_IC_MANIFEST.json"),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    device_t = torch.device(
        device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    )

    start_idx = 0
    state = _load_state() if resume else None
    if not resume:
        if STATE_PATH.is_file():
            STATE_PATH.unlink()
        if PROGRESS_CSV.is_file():
            PROGRESS_CSV.unlink()
        state = None

    if state and int(state.get("group_size", 0)) == GROUP:
        start_idx = int(state.get("next_index", 0))
        if state.get("epoch_complete") and start_idx >= n_train:
            print("EPOCH already complete; refusing to re-run without --no-resume", flush=True)
            return {**state, "already_complete": True}
        ckpt_load = Path(state.get("latest_ckpt") or LATEST_CKPT)
        if ckpt_load.is_file() and start_idx > 0:
            actor, book, rho, _ = load_policy_actor(ckpt_load, device_t)
            print(f"RESUME from index {start_idx}/{n_train} ckpt={ckpt_load}", flush=True)
        else:
            actor, book, rho, _ = load_policy_actor(P47T_CKPT, device_t)
            start_idx = 0
    else:
        actor, book, rho, _ = load_policy_actor(P47T_CKPT, device_t)
        start_idx = 0

    ref_actor, _, _, _ = load_policy_actor(P47T_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    if state and state.get("opt") and Path(state["opt"]).is_file() and start_idx > 0:
        opt.load_state_dict(torch.load(state["opt"], map_location="cpu", weights_only=False))

    env = GrpoRecoveryEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights()
    )

    opt_path = CKPT_DIR / "P53_grpo64_opt.pt"
    failed_total = 0
    retried_total = 0
    updates_done = int(state.get("updates_done", 0)) if state and start_idx > 0 else 0
    nonzero_signal_groups = int(state.get("nonzero_signal_groups", 0)) if state and start_idx > 0 else 0

    t_epoch0 = time.perf_counter()
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
                    raise RuntimeError(f"group_size={len(members)} != {GROUP}")
                break
            except Exception as e:
                failed_rollouts += 1
                retried += 1
                print(f"RETRY ic={st.traj_id} attempt={attempt} err={e}", flush=True)
                if attempt == 1:
                    raise
        assert members is not None and len(members) == GROUP
        failed_total += failed_rollouts
        retried_total += retried

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

        zero_succ = int(n_rel == 0)
        any_succ = int(n_rel > 0)
        all_succ = int(n_rel == GROUP)
        mixed = int(0 < n_rel < GROUP)

        opt.zero_grad(set_to_none=True)
        loss_v = None
        kl_v = None
        ent_v = None
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
                    nonzero_signal_groups += 1

        row = {
            "group_index": idx,
            "ic_id": st.traj_id,
            "class": _class_label(st.skill),
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
            "n_zero_success_group": zero_succ,
            "n_any_success_group": any_succ,
            "n_mixed_group": mixed,
            "n_all_success_group": all_succ,
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
                    "std": round(float(std), 3),
                    "upd": int(did_update),
                    "kl": kl_v,
                    "s": round(row["elapsed_s"], 1),
                }
            ),
            flush=True,
        )

        # Resumable checkpoint after every IC
        save_m111(
            LATEST_CKPT,
            actor,
            book,
            epoch=idx + 1,
            rho=rho,
            extra={
                "p53": True,
                "tag": "grpo64_full",
                "group": GROUP,
                "ic_index": idx,
                "n_train": n_train,
            },
        )
        torch.save(opt.state_dict(), opt_path)
        _save_state(
            {
                "next_index": idx + 1,
                "n_train": n_train,
                "group_size": GROUP,
                "updates_done": updates_done,
                "nonzero_signal_groups": nonzero_signal_groups,
                "failed_rollouts_total": failed_total,
                "retried_rollouts_total": retried_total,
                "latest_ckpt": str(LATEST_CKPT),
                "opt": str(opt_path),
                "last_ic": st.traj_id,
                "last_skill": st.skill,
            }
        )

        # Periodic named snapshot every 25 ICs
        if (idx + 1) % 25 == 0:
            snap = CKPT_DIR / f"P53_grpo64_ic{idx+1:04d}.pt"
            save_m111(
                snap,
                actor,
                book,
                epoch=idx + 1,
                rho=rho,
                extra={"p53": True, "tag": "grpo64_snap", "ic_index": idx},
            )

    # Epoch complete
    save_m111(
        EPOCH_CKPT,
        actor,
        book,
        epoch=1,
        rho=rho,
        extra={
            "p53": True,
            "tag": "grpo64_epoch1",
            "group": GROUP,
            "n_train_ics": n_train,
            "n_rollouts": n_train * GROUP,
            "updates_done": updates_done,
            "complete_epoch": True,
        },
    )
    summary = {
        "complete": True,
        "n_train_ics": n_train,
        "n_groups": n_train,
        "n_rollouts": n_train * GROUP,
        "updates_done": updates_done,
        "nonzero_signal_groups": nonzero_signal_groups,
        "failed_rollouts_total": failed_total,
        "retried_rollouts_total": retried_total,
        "epoch_ckpt": str(EPOCH_CKPT),
        "elapsed_s": time.perf_counter() - t_epoch0,
        "verify": {
            "N_groups_eq_N_train": True,
            "N_rollouts_eq_64_N_train": True,
        },
    }
    _save_state({**(_load_state() or {}), "next_index": n_train, "epoch_complete": True, **summary})
    (OUT / "FULL_GRPO64_EPOCH_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EPOCH_COMPLETE", json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-ics", type=int, default=None, help="debug only; omit for full epoch")
    args = ap.parse_args()
    if args.max_ics is not None:
        print("WARNING: max-ics set — not a full epoch", flush=True)
    run_full_epoch(
        device=args.device,
        resume=not args.no_resume,
        seed=args.seed,
        max_ics=args.max_ics,
    )


if __name__ == "__main__":
    main()
