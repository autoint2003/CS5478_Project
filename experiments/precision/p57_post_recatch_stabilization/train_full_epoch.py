"""P57 full Independent GRPO-64 epoch: branch A (control) or B (stabilization)."""

from __future__ import annotations

import argparse
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
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p57_post_recatch_stabilization.paths import (
    CKPT_DIR,
    GROUP,
    MAX_EP_STEPS,
    OUT,
    P56_CKPT,
    SEED,
    T_MAX,
)
from experiments.precision.p57_post_recatch_stabilization.reward_stabilize import (
    StabilizationWeights,
    stab_weights_dict,
)
from models.m111_checkpoint import save_m111

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
    "recatch_count",
    "stable_hold_count",
    "lost_after_recatch_count",
    "failed_rollouts",
    "retried_rollouts",
    "loss",
    "elapsed_s",
    "wall_time",
]


def _class_label(skill: str) -> str:
    return {"A1": "A", "B1": "B", "C": "C", "D": "D"}[skill]


def _paths(branch: str):
    tag = "A" if branch == "A" else "B"
    return {
        "state": OUT / f"P57_{tag}_GRPO64_STATE.json",
        "progress": OUT / f"P57_{tag}_FULL_EPOCH_PROGRESS.csv",
        "config": OUT / f"P57_{tag}_TRAIN_CONFIG.json",
        "epoch_ckpt": CKPT_DIR / f"P57_{tag}_epoch1.pt",
        "latest": CKPT_DIR / f"P57_{tag}_grpo64_latest.pt",
        "opt": CKPT_DIR / f"P57_{tag}_grpo64_opt.pt",
        "pid": OUT / f"P57_{tag}_GRPO64_TRAIN.pid",
        "summary": OUT / f"P57_{tag}_EPOCH_SUMMARY.json",
        "report": OUT / f"P57_{tag}_TRAINING_REPORT.md",
    }


def _acquire(pid_path: Path) -> None:
    import os

    if pid_path.is_file():
        try:
            old = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(old, 0)
            raise SystemExit(f"Another trainer holds {pid_path} pid={old}")
        except SystemExit:
            raise
        except Exception:
            pass
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def _release(pid_path: Path) -> None:
    import os

    if pid_path.is_file():
        try:
            if int(pid_path.read_text(encoding="utf-8").strip()) == os.getpid():
                pid_path.unlink()
        except Exception:
            pass


def run_branch(*, branch: str, device: str = "cuda", resume: bool = True, seed: int = SEED) -> dict:
    assert branch in ("A", "B")
    use_stab = branch == "B"
    P = _paths(branch)
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    _acquire(P["pid"])
    try:
        return _run(branch=branch, use_stab=use_stab, device=device, resume=resume, seed=seed, P=P)
    finally:
        _release(P["pid"])


def _run(*, branch, use_stab, device, resume, seed, P) -> dict:
    if not P56_CKPT.is_file():
        raise FileNotFoundError(P56_CKPT)
    built = build_p54_dataset(seed=seed)
    schedule = built["schedule"]
    n_train = len(schedule)
    assert GROUP == 64

    config = {
        "experiment": f"P57-{branch}",
        "init": str(P56_CKPT),
        "use_stabilization": use_stab,
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
        "dataset": "P54 rebalanced fixed order",
        "stab_weights": stab_weights_dict() if use_stab else None,
        "device": device,
    }
    P["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Shared TRAINING_CONFIG
    (OUT / "TRAINING_CONFIG.json").write_text(
        json.dumps(
            {
                "shared": {
                    "init": str(P56_CKPT),
                    "n_train_ics": n_train,
                    "group": GROUP,
                    "seed": seed,
                    "epochs": 1,
                    "identical_except_reward": True,
                },
                "A": {"use_stabilization": False},
                "B": {"use_stabilization": True, "stab_weights": stab_weights_dict()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device_t = torch.device(
        device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
    )
    start_idx = 0
    state = None
    if resume and P["state"].is_file():
        state = json.loads(P["state"].read_text(encoding="utf-8"))
    if not resume:
        if P["state"].is_file():
            P["state"].unlink()
        if P["progress"].is_file():
            P["progress"].unlink()
        state = None

    if state and int(state.get("group_size", 0)) == GROUP:
        if state.get("epoch_complete") and int(state.get("next_index", 0)) >= n_train:
            print(f"P57-{branch} EPOCH already complete", flush=True)
            return {**state, "already_complete": True}
        start_idx = int(state.get("next_index", 0))
        ckpt_load = Path(state.get("latest_ckpt") or P["latest"])
        if ckpt_load.is_file() and start_idx > 0:
            actor, book, rho, _ = load_policy_actor(ckpt_load, device_t)
            print(f"RESUME P57-{branch} from {start_idx}/{n_train}", flush=True)
        else:
            actor, book, rho, _ = load_policy_actor(P56_CKPT, device_t)
            start_idx = 0
    else:
        actor, book, rho, _ = load_policy_actor(P56_CKPT, device_t)

    ref_actor, _, _, _ = load_policy_actor(P56_CKPT, device_t)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    if state and state.get("opt") and Path(state["opt"]).is_file() and start_idx > 0:
        opt.load_state_dict(torch.load(state["opt"], map_location="cpu", weights_only=False))

    env = StabilizeGrpoEnv(
        book,
        rho,
        max_steps=MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        stab_weights=StabilizationWeights(),
        use_stabilization=use_stab,
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
        n_recatch = n_stable = n_lost = 0
        e_pos, e_th = [], []
        for m in members:
            info = m.get("info") or {}
            cd = info.get("c_diag") or {}
            n_recatch += int(cd.get("recatch") or 0)
            n_stable += int(cd.get("stable_hold") or 0)
            n_lost += int(cd.get("lost_after_recatch") or 0)
            met = info.get("metrics") or {}
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
            "recatch_count": n_recatch,
            "stable_hold_count": n_stable,
            "lost_after_recatch_count": n_lost,
            "failed_rollouts": failed_rollouts,
            "retried_rollouts": retried,
            "loss": loss_v,
            "elapsed_s": time.perf_counter() - t0,
            "wall_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        new = not P["progress"].is_file()
        with P["progress"].open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in PROGRESS_FIELDS})
        print(
            json.dumps(
                {
                    "branch": branch,
                    "idx": idx,
                    "n": n_train,
                    "skill": st.skill,
                    "succ": n_rel,
                    "recatch": n_recatch,
                    "stable": n_stable,
                    "upd": int(did_update),
                    "s": round(row["elapsed_s"], 1),
                }
            ),
            flush=True,
        )

        save_m111(
            P["latest"],
            actor,
            book,
            epoch=idx + 1,
            rho=rho,
            extra={"p57": True, "branch": branch, "use_stabilization": use_stab, "ic_index": idx},
        )
        torch.save(opt.state_dict(), P["opt"])
        P["state"].write_text(
            json.dumps(
                {
                    "next_index": idx + 1,
                    "n_train": n_train,
                    "group_size": GROUP,
                    "branch": branch,
                    "updates_done": updates_done,
                    "nonzero_signal_groups": nonzero_signal,
                    "failed_rollouts_total": failed_total,
                    "by_class_zero": by_class_zero,
                    "by_class_groups": by_class_groups,
                    "latest_ckpt": str(P["latest"]),
                    "opt": str(P["opt"]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    save_m111(
        P["epoch_ckpt"],
        actor,
        book,
        epoch=1,
        rho=rho,
        extra={
            "p57": True,
            "branch": branch,
            "use_stabilization": use_stab,
            "complete_epoch": True,
            "init": str(P56_CKPT),
            "n_train_ics": n_train,
            "n_rollouts": n_train * GROUP,
        },
    )
    summary = {
        "complete": True,
        "epoch_complete": True,
        "branch": branch,
        "use_stabilization": use_stab,
        "n_train_ics": n_train,
        "n_rollouts": n_train * GROUP,
        "updates_done": updates_done,
        "nonzero_signal_groups": nonzero_signal,
        "failed_rollouts_total": failed_total,
        "by_class_zero_success_groups": by_class_zero,
        "by_class_groups": by_class_groups,
        "epoch_ckpt": str(P["epoch_ckpt"]),
        "elapsed_s": time.perf_counter() - t0_epoch,
    }
    P["state"].write_text(
        json.dumps({**summary, "next_index": n_train, "group_size": GROUP}, indent=2),
        encoding="utf-8",
    )
    P["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_train_report(branch, use_stab, summary, P["progress"], P["report"])
    print("EPOCH_COMPLETE", json.dumps(summary), flush=True)
    return summary


def _write_train_report(branch, use_stab, summary, progress_csv: Path, report_path: Path):
    rows = list(csv.DictReader(progress_csv.open(encoding="utf-8"))) if progress_csv.is_file() else []
    by = {}
    for r in rows:
        by.setdefault(r["class"], []).append(r)
    lines = [
        f"# P57_{branch}_TRAINING_REPORT",
        "",
        f"- Init: P56_epoch1.pt",
        f"- Stabilization reward: **{use_stab}**",
        f"- Groups: {summary['n_train_ics']} × 64 = {summary['n_rollouts']}",
        f"- Updates: {summary['updates_done']}; wall {summary['elapsed_s']:.0f}s",
        f"- Zero-success groups: {summary['by_class_zero_success_groups']}",
        "",
        "| Class | groups | p_relaxed | zero_groups | mean_recatch/64 | mean_stable/64 |",
        "|-------|-------:|----------:|------------:|----------------:|---------------:|",
    ]
    for c in ("A", "B", "C", "D"):
        sub = by.get(c) or []
        if not sub:
            continue
        n = len(sub)
        p = sum(int(r["success_relaxed_count"]) for r in sub) / (n * 64)
        z = sum(int(r["n_zero_success_group"]) for r in sub)
        mr = sum(int(r["recatch_count"]) for r in sub) / (n * 64)
        ms = sum(int(r["stable_hold_count"]) for r in sub) / (n * 64)
        lines.append(f"| {c} | {n} | {p:.4f} | {z} | {mr:.3f} | {ms:.3f} |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["A", "B", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.branch in ("A", "both"):
        run_branch(branch="A", device=args.device, resume=not args.no_resume, seed=args.seed)
    if args.branch in ("B", "both"):
        run_branch(branch="B", device=args.device, resume=not args.no_resume, seed=args.seed)


if __name__ == "__main__":
    main()
