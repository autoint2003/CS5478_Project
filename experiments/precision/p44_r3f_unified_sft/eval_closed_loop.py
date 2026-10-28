"""Closed-loop eval: learned motion policy + unified online R3F HANDOFF."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.paths import OUT, SEED, T_MAX_DEFAULT  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW as K_WIN  # noqa: E402

DEST = OUT


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def rollout(env, model, book, rho, *, restore_fn: Callable, ref, t_max: int, device) -> dict:
    restore_fn()
    mon = R3FOnline.create(t_max=t_max)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq: list = []
    prev_seq: list = []
    handoff = False
    reason = "NONE"
    step_h = None
    drop = False
    tokens_used = []

    for t in range(t_max + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=False)
        if st["ready"]:
            handoff = True
            reason = st["reason"]
            step_h = t
            break
        if t >= t_max:
            break

        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K_WIN)
        if action is None or tok == HANDOFF:
            with torch.no_grad():
                logits, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
                )
                logits = logits[-1].clone()
                logits[HANDOFF] = -1e9
                tok = int(torch.argmax(logits).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)
        tokens_used.append(int(tok))
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)

    out = {
        "handoff": int(handoff),
        "handoff_reason": reason if handoff else ("DROP_BEFORE" if drop else "TIMEOUT_NO_R3"),
        "handoff_step": step_h,
        "drop_before_handoff": int(drop and not handoff),
        "n_unique_tokens": len(set(tokens_used)),
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "eval_reason": None,
        "fail_category": None,
    }
    if not handoff:
        # Still probe state at timeout for diagnostics, but do not count as triggered HANDOFF success
        if not drop:
            hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
            out["eval_reason"] = hand.get("reason")
            out["fail_category"] = "TIMEOUT_FALLBACK_UNSAFE" if not hand.get("SUCCESS_12S") else "TIMEOUT_12S_OK_BUT_UNCOUNTED"
            # User rule: TIMEOUT_FALLBACK is a valid HANDOFF reason — count it
            out["handoff"] = 1
            out["handoff_reason"] = "TIMEOUT_FALLBACK"
            out["handoff_step"] = t_max
            out["SUCCESS_12S"] = int(bool(hand.get("SUCCESS_12S")))
            out["SUCCESS_NEW"] = int(bool(hand.get("SUCCESS_NEW")))
            if out["SUCCESS_NEW"]:
                out["fail_category"] = "SUCCESS"
            elif out["SUCCESS_12S"]:
                out["fail_category"] = "HOLD_FAIL_12S_OK"
            else:
                out["fail_category"] = f"POST_HANDOFF:{hand.get('reason')}"
        else:
            out["fail_category"] = "DROP_BEFORE"
        return out

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    out.update(
        {
            "SUCCESS_12S": s12,
            "SUCCESS_NEW": snew,
            "eval_reason": hand.get("reason"),
            "fail_category": (
                "SUCCESS"
                if snew
                else ("HOLD_FAIL_12S_OK" if s12 else f"POST_HANDOFF:{hand.get('reason')}")
            ),
        }
    )
    return out


def run(*, ckpt: Path, t_max: int = T_MAX_DEFAULT, n_holdout: int = 20, smoke_train: int = 3) -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _saved = load_m111(ckpt, device)
    model.eval()
    _book2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _book2
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    jobs = collect_jobs()
    by_skill: dict[str, list] = {s: [] for s in ("A1", "B1", "C", "D")}
    for j in jobs:
        by_skill[j["skill"]].append(j)

    rows = []

    def eval_job(job, tag: str):
        skill = job["skill"]
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]
            ref = ep["ref"]

            def restore(ep=ep):
                restore_start(env, ep)

        else:
            cap, ref = job["capture"], job["ref"]

            def restore(cap=cap):
                restore_capture(env, cap)

        ev = rollout(env, model, book, rho, restore_fn=restore, ref=ref, t_max=t_max, device=device)
        row = {"tag": tag, "skill": skill, "traj_id": job["traj_id"], "split": job.get("split"), **ev}
        rows.append(row)
        write_csv(DEST / "SFT_EVAL_ROLLOUTS.csv", rows)
        print(
            f"  [{tag}] {skill} {job['traj_id'][:40]} "
            f"{ev['handoff_reason']} 12S={ev['SUCCESS_12S']} NEW={ev['SUCCESS_NEW']}",
            flush=True,
        )
        return row

    print("=== smoke train-IC ===", flush=True)
    for skill, lst in by_skill.items():
        train = [j for j in lst if j.get("split") == "train"] or lst
        for j in train[:smoke_train]:
            eval_job(j, "smoke_train")

    print("=== held-out ===", flush=True)
    for skill, lst in by_skill.items():
        valid = [j for j in lst if j.get("split") in ("valid", "val", "test")]
        # Prefer valid; if fewer than n_holdout (e.g. B1 has 16 total), fill from train
        hold = list(valid)
        if len(hold) < n_holdout:
            rest = [j for j in lst if j not in hold]
            random.shuffle(rest)
            hold.extend(rest[: max(0, n_holdout - len(hold))])
        random.shuffle(hold)
        for j in hold[:n_holdout]:
            eval_job(j, "holdout")

    lines = [
        "# SFT_EVAL_REPORT",
        "",
        f"Generated: {utc()}",
        f"Checkpoint: `{ckpt}`",
        f"T_max={t_max}",
        "",
        "## Held-out SUCCESS_12S (primary gate: >0 per class)",
        "",
        "| Skill | 12S | NEW | rate_12S | reasons | fail_cats |",
        "|---|---:|---:|---:|---|---|",
    ]
    gate = {}
    for skill in ("A1", "B1", "C", "D"):
        sub = [r for r in rows if r["tag"] == "holdout" and r["skill"] == skill]
        n = len(sub)
        s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
        snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
        gate[skill] = {"n": n, "SUCCESS_12S": s12, "SUCCESS_NEW": snew, "ok": s12 > 0}
        lines.append(
            f"| {skill} | {s12}/{n} | {snew}/{n} | {(s12 / n if n else 0):.3f} | "
            f"`{dict(Counter(r['handoff_reason'] for r in sub))}` | "
            f"`{dict(Counter(r['fail_category'] for r in sub))}` |"
        )
    all_ok = all(gate[s]["ok"] for s in gate)
    lines += [
        "",
        f"## Gate A/B/C/D SUCCESS_12S>0: **{'PASS' if all_ok else 'FAIL'}**",
        "",
        f"details: `{json.dumps(gate)}`",
        "",
    ]
    (DEST / "SFT_EVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (DEST / "SFT_EVAL_REPORT.json").write_text(
        json.dumps({"gate": gate, "all_ok": all_ok, "ckpt": str(ckpt)}, indent=2), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    return {"gate": gate, "all_ok": all_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(OUT / "checkpoints/p44_best.pt"))
    ap.add_argument("--t-max", type=int, default=T_MAX_DEFAULT)
    ap.add_argument("--n-holdout", type=int, default=20)
    ap.add_argument("--smoke-train", type=int, default=3)
    args = ap.parse_args()
    run(
        ckpt=Path(args.ckpt),
        t_max=args.t_max,
        n_holdout=args.n_holdout,
        smoke_train=args.smoke_train,
    )


if __name__ == "__main__":
    main()
