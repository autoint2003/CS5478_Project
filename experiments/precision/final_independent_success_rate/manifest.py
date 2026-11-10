"""Freeze evaluation manifest BEFORE any rollouts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.final_independent_success_rate.paths import (
    EVAL_SPLIT,
    MANIFEST_SEED,
    MAX_EP_STEPS,
    N_ROLLOUTS_PER_IC,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    P82_FREEZE,
    PROTOCOL_VERSION,
    SKILLS,
    T_MAX,
    TARGET_PER_CLASS,
)
from experiments.precision.final_independent_success_rate.pool import eligible_heldout_pools, sample_manifest_starts
from experiments.precision.p73_first_contact_bilateral.controller import v2_spec


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_and_freeze_manifest(*, force: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "FROZEN_TEST_MANIFEST.json"
    if man_path.is_file() and not force:
        return json.loads(man_path.read_text(encoding="utf-8"))

    pools = eligible_heldout_pools()
    selected, gaps = sample_manifest_starts(
        pools["eligible"], n_per_class=TARGET_PER_CLASS, seed=MANIFEST_SEED
    )

    actor_md5 = _md5(P64_ACTOR)
    critic_md5 = _md5(P64_CRITIC)
    v2 = v2_spec()
    v2_cfg = {k: getattr(v2, k) for k in v2.__dataclass_fields__}  # type: ignore[attr-defined]
    for k, v in list(v2_cfg.items()):
        if isinstance(v, tuple):
            v2_cfg[k] = list(v)

    freeze_ref = {}
    if P82_FREEZE.is_file():
        freeze_ref = json.loads(P82_FREEZE.read_text(encoding="utf-8"))

    episodes = []
    for sk in SKILLS:
        for i, s in enumerate(selected[sk]):
            seed = rollout_seed(
                split=EVAL_SPLIT, traj_id=s.traj_id, rollout_i=i
            )
            episodes.append(
                {
                    "skill": sk,
                    "traj_id": s.traj_id,
                    "root_id": root_id(s.traj_id),
                    "family": getattr(s, "family", sk),
                    "kind": s.kind,
                    "seed": int(seed),
                    "rollout_i": 0,
                    "index_in_class": i,
                }
            )

    shortfalls = {sk: gaps[sk]["shortfall"] for sk in SKILLS}
    can_meet_target = all(v == 0 for v in shortfalls.values())

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_name": "small-scale independent evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_per_class": TARGET_PER_CLASS,
        "n_rollouts_per_ic": N_ROLLOUTS_PER_IC,
        "manifest_seed": MANIFEST_SEED,
        "can_meet_10_per_class": can_meet_target,
        "shortfalls": shortfalls,
        "gaps": gaps,
        "pool_summary": pools["summary"],
        "exclusion_set_sizes": pools["exclusion_set_sizes"],
        "n_episodes_planned": len(episodes),
        "episodes": episodes,
        "frozen_system": {
            "actor_path": str(P64_ACTOR),
            "actor_md5": actor_md5,
            "critic_path": str(P64_CRITIC),
            "critic_md5": critic_md5,
            "v2_controller": v2_cfg,
            "action_vocab": "original M111 from Actor checkpoint",
            "success_criterion": "SUCCESS_RELAXED_12S",
            "max_ep_steps": MAX_EP_STEPS,
            "t_max": T_MAX,
            "inference": "canonical p82.collect.rollout_episode (Actor sample + V2 gate, rule=None)",
            "p82_freeze_actor_md5": ((freeze_ref.get("baseline") or {}).get("actor") or {}).get("md5"),
            "hashes_match_p82_freeze": actor_md5
            == ((freeze_ref.get("baseline") or {}).get("actor") or {}).get("md5"),
        },
        "constraints": [
            "No training / tuning / overlays / reward / obs / M111 changes",
            "No checkpoint selection on these outcomes",
            "One rollout per IC; no repeated-IC as independent trials",
            "No padding shortfalls with train or diagnostic ICs",
        ],
        "excluded_examples_by_class": {
            sk: pools["excluded_rows"][sk][:20] for sk in SKILLS
        },
    }

    # Serialize starts lookup separately (pickle) for runner
    starts_blob = {sk: selected[sk] for sk in SKILLS}
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    import pickle

    (OUT / "cache" / "selected_starts.pkl").write_bytes(
        pickle.dumps(starts_blob, protocol=pickle.HIGHEST_PROTOCOL)
    )
    man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
