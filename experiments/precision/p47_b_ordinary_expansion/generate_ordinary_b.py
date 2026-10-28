"""P47 — Generate ordinary (non-hard, non-holdout) B experts via table-free + Tree80.

No K48 GRPO retrain. Frozen Tree80 E3 only as stage-1 generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.diagnostics.run_b_grpo_multistage_dataset import (  # noqa: E402
    DEFAULT_WEIGHTS,
    DT,
    capture_fp,
    lineage_of,
    load_trained_k48,
    make_plain_env,
    make_training_env,
    observe,
    process_tree80_ic,
    quantize_and_replay,
    write_csv,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from experiments.precision.p47_b_ordinary_expansion.paths import (  # noqa: E402
    BALL_V2,
    M111_META,
    OUT,
    P44_EVAL,
    SEED,
    TABLE_FREE,
    TREE80,
)
from models.m111_checkpoint import HANDOFF, MOTION_K  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402
from training.skill_aware_action_vocab import load_motion_codebook as load_cb  # noqa: E402

DATASET_VERSION = "p47_b_ordinary_v1"
B_WEIGHT = 3.0
EARLY_WEIGHT = 2.0


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_id(tid: str) -> str:
    t = re.sub(r"^(A|B|C)__", "", tid)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    for p in ("plan_a__", "plan_b__", "plan_c__"):
        if t.startswith(p):
            t = t[len(p) :]
            break
    return t


def holdout_b_roots() -> set[str]:
    return {
        root_id(r["traj_id"])
        for r in csv.DictReader(P44_EVAL.open(encoding="utf-8"))
        if r.get("skill") == "B1" and r.get("tag") == "holdout"
    }


def load_ordinary_ball_ics(held: set[str], *, max_n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cands = []
    for p in sorted(BALL_V2.glob("*.pkl")):
        ic_id = p.stem  # ball_train_v2_XXXX
        if ic_id in held or ic_id.startswith("hard_"):
            continue
        blob = pickle.loads(p.read_bytes())
        if "capture" not in blob or "ref" not in blob:
            continue
        cands.append(
            {
                "ic_id": ic_id,
                "pool": "ordinary_ball_impact_v2",
                "source": "ball_impact_dataset_v2_train",
                "capture": blob["capture"],
                "ref": blob["ref"],
                "meta": {"spec": blob.get("spec")},
            }
        )
    rng.shuffle(cands)
    return cands[:max_n]


def pack_table_free(env, cb, held: set[str], dirs: dict) -> tuple[list[dict], list[dict]]:
    """M111-quantize existing ordinary plan_b table-free experts."""
    chunks, rows = [], []
    if not TABLE_FREE.exists():
        return chunks, rows
    w = DEFAULT_WEIGHTS
    for p in sorted(TABLE_FREE.glob("*__SUCCESS_NEW_expert.pkl")):
        obj = pickle.loads(p.read_bytes())
        ic_id = root_id(obj.get("physical_root_id") or obj.get("expert_id") or p.stem)
        row = {"source": "table_free", "ic_id": ic_id, "admitted": 0}
        if ic_id in held:
            row["reject"] = "holdout"
            rows.append(row)
            continue
        if str(ic_id).startswith("hard_"):
            row["reject"] = "hard"
            rows.append(row)
            continue
        acts = np.asarray(obj["actions"], np.float32)
        cap = obj["catch_snap"]
        ref = obj["ref"]
        q = quantize_and_replay(env, cap, ref, acts, cb.prototypes, cb.residual, w, "M111")
        row.update(
            {
                "M111_12S": int(bool(q["SUCCESS_12S"])),
                "M111_NEW": int(bool(q["SUCCESS_NEW"])),
                "n_actions": int(len(acts)),
                "reason": q.get("reason"),
            }
        )
        if not q["SUCCESS_12S"]:
            row["reject"] = "m111_12S_fail"
            rows.append(row)
            print(f"  reject table_free {ic_id} 12S=0", flush=True)
            continue
        # persist minimal IC + continuous for provenance
        sid = f"B__table_free__{ic_id}__{capture_fp(cap)}"
        ic_path = dirs["ics"] / f"{sid}__recovery.pkl"
        pickle.dump(
            {
                "capture": cap,
                "ref": ref,
                "sample_id": sid,
                "ic_id": ic_id,
                "lineage_id": lineage_of(ic_id, "ordinary_table_free"),
                "source": "table_free",
                "expert_kind": "table_free_ordinary",
            },
            ic_path.open("wb"),
        )
        cont_path = dirs["continuous"] / f"{sid}__fullchain_continuous.npz"
        np.savez_compressed(
            cont_path,
            actions=acts,
            observations=np.asarray(obj["observations"], np.float32),
            tokens=q["tokens"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
        )
        np.savez_compressed(
            dirs["tokens_m111"] / f"{sid}__m111.npz",
            tokens=q["tokens"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
            handoff_id=np.asarray([cb.handoff_id], np.int64),
        )
        chs = episode_to_chunks(
            obs=np.asarray(obj["observations"], np.float32),
            tokens=q["tokens"],
            residuals=q["residuals"],
            recon=q["recon_actions"],
            traj_id=f"plan_b__ordinary__{ic_id}",
            ic_id=ic_id,
            expert_kind="table_free_ordinary",
        )
        chunks.extend(chs)
        row.update({"admitted": 1, "sample_id": sid, "n_chunks": len(chs)})
        rows.append(row)
        print(f"  ADMIT table_free {ic_id} steps={len(acts)} chunks={len(chs)}", flush=True)
    return chunks, rows


def episode_to_chunks(*, obs, tokens, residuals, recon, traj_id, ic_id, expert_kind):
    tokens = np.asarray(tokens, np.int64)
    residuals = np.asarray(residuals, np.float32)
    recon = np.asarray(recon, np.float32)
    obs = np.asarray(obs, np.float32)
    n = len(tokens)
    if n and int(tokens[-1]) == HANDOFF:
        n -= 1
        tokens, residuals, recon, obs = tokens[:n], residuals[:n], recon[:n], obs[:n]
    if n == 0 or len(obs) != n:
        # rebuild obs length from actions if needed
        if len(obs) != n and n > 0:
            return []
        if n == 0:
            return []
    prev = np.zeros((n, 7), np.float32)
    if n > 1:
        prev[1:] = recon[: n - 1]
    mask = (tokens != HANDOFF) & (tokens >= 0) & (tokens < MOTION_K)
    if not bool(mask.any()):
        return []
    ep = {
        "observations": obs,
        "prev_actions": prev,
        "tokens": tokens,
        "residuals": residuals,
        "loss_mask": mask,
    }
    out = []
    for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
        m = np.asarray(ch["loss_mask"], dtype=np.bool_)
        m &= np.asarray(ch["tokens"]) != HANDOFF
        m &= (np.asarray(ch["tokens"]) >= 0) & (np.asarray(ch["tokens"]) < MOTION_K)
        if not bool(m.any()):
            continue
        w = B_WEIGHT * (EARLY_WEIGHT if ci == 0 else 1.0)
        out.append(
            {
                **ch,
                "loss_mask": m,
                "traj_id": traj_id,
                "skill": "B1",
                "split": "train",
                "chunk_index": ci,
                "n_tokens": int(m.sum()),
                "sample_weight": float(w),
                "source": "p47_ordinary_b",
                "expert_kind": expert_kind,
                "ic_id": ic_id,
                "SUCCESS_12S": 1,
            }
        )
    return out


def _replay_obs(env, capture, actions: np.ndarray) -> np.ndarray:
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_list = []
    for a in actions:
        obs_list.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    return np.asarray(obs_list, np.float32) if obs_list else np.zeros((0, 39), np.float32)


def run_tree80_ordinary(
    *,
    max_ics: int,
    device: torch.device,
    held: set[str],
    dirs: dict,
    cb,
) -> tuple[list[dict], list[dict]]:
    model, book = load_trained_k48(TREE80, device)
    env_pol = make_training_env(device, model=model)
    env_plain = make_plain_env()
    ics = load_ordinary_ball_ics(held, max_n=max_ics, seed=SEED)
    print(f"Tree80 ordinary ICs to try: {len(ics)}", flush=True)
    chunks, rows = [], []
    w = DEFAULT_WEIGHTS
    for i, ic in enumerate(ics):
        print(f"=== [{i+1}/{len(ics)}] Tree80 {ic['ic_id']} ===", flush=True)
        row = process_tree80_ic(env_pol, env_plain, model, book, device, ic)
        meta = {k: v for k, v in row.items() if not k.startswith("_")}
        meta["source"] = "tree80_ordinary"
        ok_expert = row.get("partition") == "expert" and bool(row.get("fullchain_12S"))
        if not ok_expert:
            meta["admitted"] = 0
            meta["reject"] = row.get("category") or "not_expert"
            rows.append(meta)
            print(f"  reject {ic['ic_id']} cat={row.get('category')}", flush=True)
            continue

        kind = (
            "tree80_ordinary_direct"
            if row.get("category") == "TREE80_DIRECT"
            else "tree80_ordinary_precision"
        )
        acts = np.asarray(row["_full_actions"], np.float32)
        q = quantize_and_replay(
            env_plain, ic["capture"], ic["ref"], acts, cb.prototypes, cb.residual, w, "M111"
        )
        sid = f"B__{kind}__{ic['ic_id']}__{capture_fp(ic['capture'])}"
        meta.update(
            {
                "M111_12S": int(bool(q["SUCCESS_12S"])),
                "M111_NEW": int(bool(q["SUCCESS_NEW"])),
                "sample_id": sid,
                "expert_kind": kind,
            }
        )
        if not q["SUCCESS_12S"]:
            meta["admitted"] = 0
            meta["reject"] = "m111_12S_fail"
            rows.append(meta)
            print(f"  reject {ic['ic_id']} m111_12S=0", flush=True)
            continue

        obs = _replay_obs(env_plain, ic["capture"], q["recon_actions"])
        n = min(len(obs), len(q["tokens"]))
        pickle.dump(
            {
                "capture": ic["capture"],
                "ref": ic["ref"],
                "sample_id": sid,
                "ic_id": ic["ic_id"],
                "lineage_id": lineage_of(ic["ic_id"], ic["pool"]),
                "source": ic["source"],
                "expert_kind": kind,
            },
            (dirs["ics"] / f"{sid}__recovery.pkl").open("wb"),
        )
        np.savez_compressed(
            dirs["continuous"] / f"{sid}__fullchain_continuous.npz",
            actions=acts[:n],
            observations=obs[:n],
            timestamps=np.arange(n, dtype=np.float32) * DT,
        )
        np.savez_compressed(
            dirs["tokens_m111"] / f"{sid}__m111.npz",
            tokens=q["tokens"][:n],
            residuals=q["residuals"][:n],
            recon_actions=q["recon_actions"][:n],
            handoff_id=np.asarray([cb.handoff_id], np.int64),
        )
        chs = episode_to_chunks(
            obs=obs[:n],
            tokens=q["tokens"][:n],
            residuals=q["residuals"][:n],
            recon=q["recon_actions"][:n],
            traj_id=f"plan_b__ordinary__{ic['ic_id']}",
            ic_id=ic["ic_id"],
            expert_kind=kind,
        )
        chunks.extend(chs)
        meta.update({"admitted": 1, "n_chunks": len(chs), "n_steps": n})
        rows.append(meta)
        print(
            f"  ADMIT {kind} {ic['ic_id']} steps={n} chunks={len(chs)} "
            f"cat={row.get('category')}",
            flush=True,
        )
    return chunks, rows


def merge_and_save(b_chunks: list[dict]) -> dict:
    from experiments.precision.p47_b_ordinary_expansion.paths import P45

    pools = {}
    for sk in ("A1", "B1", "C", "D"):
        base = pickle.loads((P45 / "dataset" / f"chunks_{sk}_P45B.pkl").read_bytes())
        pools[sk] = list(base)
    # replace hard tree80 from p46 if any leaked — P45B B has no hard; only add ordinary
    pools["B1"] = list(pools["B1"]) + list(b_chunks)
    ds = OUT / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    for sk, chs in pools.items():
        (ds / f"chunks_{sk}_P47.pkl").write_bytes(pickle.dumps(chs))
    (ds / "chunks_B1_ordinary_only.pkl").write_bytes(pickle.dumps(b_chunks))
    return {sk: len(pools[sk]) for sk in pools}


def run(*, max_tree80: int = 48) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw"
    dirs = {}
    for sub in ("ics", "continuous", "stage1", "precision", "outcomes", "failures", "tokens_m111"):
        d = raw / sub
        d.mkdir(parents=True, exist_ok=True)
        dirs[sub] = d

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    held = holdout_b_roots()
    print(f"Holdout B roots excluded: {len(held)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb = load_cb(M111_META)
    env = make_plain_env()

    print("=== Phase1: table_free ordinary → M111 ===", flush=True)
    ch_tf, rows_tf = pack_table_free(env, cb, held, dirs)

    print("=== Phase2: Tree80 on ordinary ball_impact ICs ===", flush=True)
    # save_expert_row writes paths relative to dirs values; fix continuous path lookup
    # Patch: save_expert_row uses dirs["continuous"] absolute — good.
    # But man['continuous_demo']['path'] is relative like continuous/...
    # We'll load via raw /
    ch_t80, rows_t80 = run_tree80_ordinary(
        max_ics=max_tree80, device=device, held=held, dirs=dirs, cb=cb
    )

    # Fix continuous load path inside run — already used OUT/raw / path

    all_chunks = ch_tf + ch_t80
    all_rows = rows_tf + rows_t80
    write_csv(OUT / "GENERATION_ROWS.csv", all_rows)
    pool_sizes = merge_and_save(all_chunks)

    admitted = [r for r in all_rows if int(r.get("admitted") or 0) == 1]
    summary = {
        "generated": utc(),
        "dataset_version": DATASET_VERSION,
        "n_admitted": len(admitted),
        "by_kind": dict(Counter(r.get("expert_kind") for r in admitted)),
        "by_source": dict(Counter(r.get("source") for r in admitted)),
        "n_chunks": len(all_chunks),
        "n_steps": int(sum(int(r.get("n_steps") or r.get("n_actions") or 0) for r in admitted)),
        "pool_sizes": pool_sizes,
        "holdout_excluded": len(held),
        "max_tree80": max_tree80,
        "note": "No hard_* ICs. No holdout B. Frozen Tree80 only.",
    }
    (OUT / "DATA_GENERATION_REPORT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# DATA_GENERATION_REPORT — P47 ordinary B",
        "",
        f"Generated: {summary['generated']}",
        "",
        f"- Admitted: **{summary['n_admitted']}** `{summary['by_kind']}`",
        f"- Chunks: {summary['n_chunks']} steps≈{summary['n_steps']}",
        f"- Pools: `{pool_sizes}`",
        f"- Tree80 attempts cap: {max_tree80}",
        "",
        "Excluded: hard_* and P44 B holdout. No K48 GRPO retrain.",
        "",
    ]
    (OUT / "DATA_GENERATION_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tree80", type=int, default=48)
    args = ap.parse_args()
    run(max_tree80=args.max_tree80)


if __name__ == "__main__":
    main()
