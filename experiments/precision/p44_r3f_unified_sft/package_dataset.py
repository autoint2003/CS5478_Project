"""Package R3F-truncated expert episodes for balanced SFT.

Reuses P21/P35 NPZ + P43 R3F handoff cuts; re-verifies SUCCESS_12S under
current nominal. Only SUCCESS_12S=1 episodes enter the SFT pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_npz_flex  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p39_working_hold_handoff.paths import P21, P21_MAN, P35, P35_MAN  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.paths import OUT, SEED, T_MAX_DEFAULT  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from models.m111_checkpoint import HANDOFF, MOTION_K, decode_m111_action, load_m111_book  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402

DEST = OUT / "dataset"
CONTROLLER_TAG = "recovery_success.apply_handoff_v_live_hand"
CONFIG_TAG = f"R3F_t_max={T_MAX_DEFAULT}"


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


def cut_at_r3f(env, job, book, rho, t_max: int) -> dict:
    skill = job["skill"]
    if skill in ("A1", "B1", "C"):
        ep = job["ep"]
        ref = ep["ref"]

        def restore():
            restore_start(env, ep)

    else:
        cap, ref = job["capture"], job["ref"]

        def restore():
            restore_capture(env, cap)

    restore()
    mon = R3FOnline.create(t_max=t_max)
    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    obs = job.get("observations")
    prev = job.get("prev_actions")
    # load full npz arrays if not on job
    if obs is None:
        # filled by caller via npz
        pass

    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    reason = "NONE"
    cut = n
    for t in range(n + 1):
        st = mon.observe(env.sim, t, tape_ended=(t >= n))
        if st["ready"]:
            reason = st["reason"]
            cut = t
            break
        if t >= n:
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    return {
        "cut": int(cut),
        "n_motion": int(n),
        "handoff_reason": reason,
        "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
        "eval_reason": hand.get("reason"),
        "ref": ref,
    }


def motion_ep(obs, prev, tokens, residuals, cut: int) -> dict | None:
    n = int(cut)
    if n <= 0:
        return None
    tokens = np.asarray(tokens[:n], np.int64)
    mask = (tokens != HANDOFF) & (tokens >= 0) & (tokens < MOTION_K)
    if not bool(mask.any()):
        return None
    return {
        "observations": np.asarray(obs[:n], np.float32),
        "prev_actions": np.asarray(prev[:n], np.float32),
        "tokens": tokens,
        "residuals": np.asarray(residuals[:n], np.float32),
        "loss_mask": mask,
    }


def run(*, t_max: int = T_MAX_DEFAULT, limit: int = 0) -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "npz").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = collect_jobs()
    if limit > 0:
        kept, counts = [], {}
        for j in jobs:
            c = counts.get(j["skill"], 0)
            if c < limit:
                kept.append(j)
                counts[j["skill"]] = c + 1
        jobs = kept

    # Attach observations from source npz
    man21 = json.loads(P21_MAN.read_text(encoding="utf-8"))
    by21 = {r["traj_id"]: r for r in man21["trajectories"]}
    p35_rows = {r["traj_id"]: r for r in csv.DictReader(P35_MAN.open(encoding="utf-8"))}

    rows = []
    chunks_by_skill: dict[str, list] = {s: [] for s in ("A1", "B1", "C", "D")}
    print(f"P44 package jobs={len(jobs)} t_max={t_max}", flush=True)

    for i, job in enumerate(jobs):
        skill = job["skill"]
        tid = job["traj_id"]
        if skill in ("A1", "B1", "C"):
            meta = by21[tid]
            z = load_npz_flex(P21, meta["npz_relpath"])
        else:
            meta = p35_rows[tid]
            raw = np.load(P35 / meta["npz_relpath"], allow_pickle=False)
            z = {k: raw[k] for k in raw.files}

        obs = np.asarray(z["observations"], np.float32)
        prev = np.asarray(z["prev_actions"], np.float32)
        tokens = np.asarray(z["tokens"], np.int64)
        residuals = np.asarray(z["residuals"], np.float32)
        job = {**job, "tokens": tokens, "residuals": residuals}

        info = cut_at_r3f(env, job, book, rho, t_max=t_max)
        ok = int(info["SUCCESS_12S"]) == 1
        row = {
            "traj_id": tid,
            "skill": skill,
            "split": job.get("split"),
            "lineage_id": job.get("lineage_id"),
            "handoff_reason": info["handoff_reason"],
            "cut_step": info["cut"],
            "n_motion_orig": info["n_motion"],
            "traj_len_cut": info["cut"],
            "SUCCESS_12S": info["SUCCESS_12S"],
            "SUCCESS_NEW": info["SUCCESS_NEW"],
            "eval_reason": info["eval_reason"],
            "admitted_sft": int(ok),
            "controller": CONTROLLER_TAG,
            "config": CONFIG_TAG,
            "ic_id": tid,
        }
        if not ok:
            row["fail_reason"] = info["eval_reason"]
            rows.append(row)
            write_csv(DEST / "EPISODE_MANIFEST.csv", rows)
            print(
                f"  [{i+1}/{len(jobs)}] REJECT {skill} {tid[:40]} "
                f"{info['handoff_reason']} 12S=0",
                flush=True,
            )
            continue

        ep = motion_ep(obs, prev, tokens, residuals, info["cut"])
        if ep is None:
            row["admitted_sft"] = 0
            row["fail_reason"] = "EMPTY_MOTION"
            rows.append(row)
            continue

        rel = f"npz/{skill}_{tid[:48]}.npz"
        np.savez_compressed(
            DEST / rel,
            observations=ep["observations"],
            prev_actions=ep["prev_actions"],
            tokens=ep["tokens"],
            residuals=ep["residuals"],
            loss_mask=ep["loss_mask"],
            handoff_reason=np.asarray([info["handoff_reason"]]),
            cut_step=np.asarray([info["cut"]]),
            SUCCESS_12S=np.asarray([1]),
            SUCCESS_NEW=np.asarray([info["SUCCESS_NEW"]]),
        )
        row["npz_relpath"] = rel
        rows.append(row)

        meta_c = {
            "traj_id": tid,
            "skill": skill,
            "split": job.get("split") or "train",
            "handoff_reason": info["handoff_reason"],
            "SUCCESS_NEW": info["SUCCESS_NEW"],
        }
        for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
            mask = np.asarray(ch["loss_mask"], dtype=np.bool_)
            mask &= (np.asarray(ch["tokens"]) != HANDOFF)
            mask &= (np.asarray(ch["tokens"]) >= 0) & (np.asarray(ch["tokens"]) < MOTION_K)
            if not bool(mask.any()):
                continue
            chunks_by_skill[skill].append(
                {
                    **ch,
                    "loss_mask": mask,
                    **meta_c,
                    "chunk_index": ci,
                    "n_tokens": int(mask.sum()),
                    "sample_weight": 1.0,
                }
            )

        write_csv(DEST / "EPISODE_MANIFEST.csv", rows)
        print(
            f"  [{i+1}/{len(jobs)}] OK {skill} {tid[:40]} "
            f"{info['handoff_reason']} cut={info['cut']}/{info['n_motion']} "
            f"NEW={info['SUCCESS_NEW']}",
            flush=True,
        )

    # Save chunk pools
    for skill, chs in chunks_by_skill.items():
        path = DEST / f"chunks_{skill}.pkl"
        path.write_bytes(pickle.dumps(chs))
        print(f"chunks {skill}: {len(chs)}", flush=True)

    # Summary
    admitted = [r for r in rows if int(r["admitted_sft"]) == 1]
    summary = {
        "utc": utc(),
        "t_max": t_max,
        "controller": CONTROLLER_TAG,
        "config": CONFIG_TAG,
        "n_jobs": len(jobs),
        "n_admitted": len(admitted),
        "by_skill": {
            s: {
                "admitted": sum(1 for r in admitted if r["skill"] == s),
                "rejected": sum(1 for r in rows if r["skill"] == s and int(r["admitted_sft"]) == 0),
                "mean_cut": float(
                    np.mean([int(r["cut_step"]) for r in admitted if r["skill"] == s] or [0])
                ),
                "mean_orig": float(
                    np.mean([int(r["n_motion_orig"]) for r in admitted if r["skill"] == s] or [0])
                ),
                "reasons": dict(
                    Counter(r["handoff_reason"] for r in admitted if r["skill"] == s)
                ),
                "NEW_rate": (
                    sum(int(r["SUCCESS_NEW"]) for r in admitted if r["skill"] == s)
                    / max(1, sum(1 for r in admitted if r["skill"] == s))
                ),
                "n_chunks": len(chunks_by_skill[s]),
            }
            for s in ("A1", "B1", "C", "D")
        },
    }
    (DEST / "DATASET_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# DATASET_SUMMARY",
        "",
        f"Generated: {utc()}",
        f"Controller: `{CONTROLLER_TAG}`",
        f"Rule: `{CONFIG_TAG}`",
        f"Admitted (SUCCESS_12S): {len(admitted)}/{len(jobs)}",
        "",
        "| Skill | admitted | mean_cut | mean_orig | shrink | NEW_rate | chunks | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s, a in summary["by_skill"].items():
        shrink = 1.0 - (a["mean_cut"] / a["mean_orig"]) if a["mean_orig"] else 0.0
        lines.append(
            f"| {s} | {a['admitted']} | {a['mean_cut']:.1f} | {a['mean_orig']:.1f} | "
            f"{shrink:.1%} | {a['NEW_rate']:.2f} | {a['n_chunks']} | `{a['reasons']}` |"
        )
    lines += [
        "",
        "SFT pool excludes rejected episodes. SUCCESS_NEW recorded but not required.",
        "",
    ]
    (OUT / "DATASET_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (DEST / "DATASET_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-max", type=int, default=T_MAX_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(t_max=args.t_max, limit=args.limit)


if __name__ == "__main__":
    main()
