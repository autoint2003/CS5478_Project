"""Execute frozen small-scale independent evaluation (1 rollout / IC)."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p82_global_failure_budget.collect import rollout_episode
from experiments.precision.final_independent_success_rate.manifest import build_and_freeze_manifest
from experiments.precision.final_independent_success_rate.paths import (
    CACHE,
    MAX_EP_STEPS,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    SKILLS,
    T_MAX,
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_evaluation(*, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    results_pkl = CACHE / "episode_results.pkl"
    csv_path = OUT / "EPISODE_RESULTS.csv"

    manifest = build_and_freeze_manifest(force=False)
    if results_pkl.is_file() and csv_path.is_file() and not force:
        rows = pickle.loads(results_pkl.read_bytes())
        return {"rows": rows, "manifest": manifest, "cached": True}

    starts_blob = pickle.loads((CACHE / "selected_starts.pkl").read_bytes())
    lookup = {}
    for sk in SKILLS:
        for s in starts_blob[sk]:
            lookup[(sk, s.traj_id)] = s

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)

    env = StabilizeGrpoEnv(
        book,
        rho,
        max_steps=MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        use_stabilization=False,
    )

    actor_md5 = _md5(P64_ACTOR)
    critic_md5 = _md5(P64_CRITIC)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i, ep in enumerate(manifest["episodes"]):
        start = lookup[(ep["skill"], ep["traj_id"])]
        seed = int(ep["seed"])
        print(
            f"[final-eval] {i+1}/{len(manifest['episodes'])} {ep['skill']} "
            f"{ep['traj_id'][-24:]} seed={seed}",
            flush=True,
        )
        try:
            rec = rollout_episode(env, actor, critic, book, start, device_t, seed=seed)
            info = getattr(env, "last_info", None) or {}
            # Prefer live env terminal flags when present
            succ_new = int(info.get("SUCCESS_NEW") or 0)
            succ_12 = int(info.get("SUCCESS_12S") or 0)
            forced = int(rec.get("forced_steps") or 0)
            actor_steps = int(rec.get("actor_steps") or 0)
            total_ctrl = forced + actor_steps
            row = {
                "skill": ep["skill"],
                "traj_id": ep["traj_id"],
                "root_id": ep["root_id"],
                "seed": seed,
                "actor_md5": actor_md5,
                "critic_md5": critic_md5,
                "SUCCESS_RELAXED_12S": int(rec.get("SUCCESS_RELAXED") or 0),
                "SUCCESS_NEW": succ_new,
                "SUCCESS_12S": succ_12,
                "r3_ready": int(rec.get("r3_ready") or 0),
                "handoff": int(rec.get("handoff") or 0),
                "earliest_decisive_failure": rec.get("earliest_decisive_failure"),
                "ep_len": int(rec.get("ep_len") or 0),
                "forced_steps": forced,
                "actor_steps": actor_steps,
                "v2_activated": int(forced > 0),
                "predominantly_actor_controlled": int(forced == 0 or (total_ctrl > 0 and forced / max(total_ctrl, 1) < 0.1)),
                "first_contact": int(rec.get("first_contact") or 0),
                "opposing_ever": int(rec.get("opposing_ever") or 0),
                "bilateral_ever": int(rec.get("bilateral_ever") or 0),
                "min_pad_ge4": int(rec.get("min_pad_ge4") or 0),
                "e_pos_le9": int(rec.get("e_pos_le9") or 0),
                "frame_ok": int(rec.get("frame_ok") or 0),
                "timeout": int(rec.get("timeout") or 0),
                "wall_s": None,
                "infra_error": None,
            }
        except Exception as e:  # noqa: BLE001
            row = {
                "skill": ep["skill"],
                "traj_id": ep["traj_id"],
                "root_id": ep["root_id"],
                "seed": seed,
                "actor_md5": actor_md5,
                "critic_md5": critic_md5,
                "SUCCESS_RELAXED_12S": 0,
                "SUCCESS_NEW": 0,
                "SUCCESS_12S": 0,
                "r3_ready": 0,
                "handoff": 0,
                "earliest_decisive_failure": "infra_error",
                "ep_len": 0,
                "forced_steps": 0,
                "actor_steps": 0,
                "v2_activated": 0,
                "predominantly_actor_controlled": 1,
                "first_contact": 0,
                "opposing_ever": 0,
                "bilateral_ever": 0,
                "min_pad_ge4": 0,
                "e_pos_le9": 0,
                "frame_ok": 0,
                "timeout": 0,
                "wall_s": None,
                "infra_error": f"{type(e).__name__}: {e}",
            }
            print(f"  INFRA ERROR: {row['infra_error']}", flush=True)
        rows.append(row)
        print(
            f"  -> relaxed={row['SUCCESS_RELAXED_12S']} fail={row['earliest_decisive_failure']} "
            f"v2_forced={row['forced_steps']}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    results_pkl.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = {
        "n_rows": len(rows),
        "elapsed_s": elapsed,
        "n_infra_errors": sum(1 for r in rows if r.get("infra_error")),
        "cached": False,
    }
    (CACHE / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"rows": rows, "manifest": manifest, "cached": False, "meta": meta}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = run_evaluation(device=args.device, force=args.force)
    print(json.dumps({"n": len(out["rows"]), "cached": out.get("cached")}, indent=2))
