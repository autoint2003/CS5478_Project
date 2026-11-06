"""Additional independent development panel (pre-registered)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id
from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p61_reproducible_joint_eval.eval_core import eval_starts
from experiments.precision.p61_reproducible_joint_eval.seeding import GLOBAL_SEED
from experiments.precision.p62_grpo_optimization.eval_dev import _flat_starts, _fresh_final_starts
from experiments.precision.p65_ppo_continuation.paths import (
    OUT,
    PANEL_PER_CLASS,
    PANEL_ROLLOUTS,
    PANEL_SEED,
    SEED,
)


def _blocked() -> set[str]:
    """Exclude train + primary P61/P62 dev + prior final-test slice.

    Remaining holdout ICs are eligible for the extra development panel.
    Do **not** blanket-block all P44 holdouts — that empties the panel.
    """
    blocked: set[str] = set()
    for s in build_p54_dataset(seed=SEED)["schedule"]:
        blocked.add(s.traj_id)
        blocked.add(root_id(s.traj_id))
    flat, _ = _flat_starts(4)
    for s in flat:
        blocked.add(s.traj_id)
        blocked.add(root_id(s.traj_id))
    final, _ = _fresh_final_starts({s.traj_id for s in flat}, 4)
    for s in final:
        blocked.add(s.traj_id)
        blocked.add(root_id(s.traj_id))
    return blocked


def build_panel_manifest(*, per_class: int = PANEL_PER_CLASS) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    blocked = _blocked()
    # request larger holdout pool then filter
    pool = load_heldout_starts(per_class=per_class * 6)
    rng = np.random.default_rng(PANEL_SEED)
    selected = {}
    for sk in ("A1", "B1", "C", "D"):
        cand = [
            s
            for s in pool[sk]
            if s.traj_id not in blocked and root_id(s.traj_id) not in blocked
        ]
        # deterministic shuffle
        order = list(range(len(cand)))
        rng.shuffle(order)
        picks = [cand[i] for i in order[:per_class]]
        selected[sk] = picks

    rows = []
    for sk, starts in selected.items():
        for i, s in enumerate(starts):
            rows.append(
                {
                    "skill": sk,
                    "class": {"A1": "A", "B1": "B", "C": "C", "D": "D"}[sk],
                    "index": i,
                    "traj_id": s.traj_id,
                    "kind": s.kind,
                }
            )
    manifest = {
        "panel_name": "P65_extra_dev",
        "panel_seed": PANEL_SEED,
        "global_seed": GLOBAL_SEED,
        "protocol_split": "p65_extra_dev",
        "per_class": per_class,
        "rollouts_per_ic_default": PANEL_ROLLOUTS,
        "excluded": {
            "p54_train": True,
            "p61_p62_primary_dev": True,
            "p62_fresh_final_slice": True,
        },
        "ics": rows,
        "n_ics": len(rows),
        "pre_registered": True,
    }
    path = OUT / "DEV_PANEL_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "starts": selected, "path": str(path)}


def load_panel_starts():
    path = OUT / "DEV_PANEL_MANIFEST.json"
    if not path.is_file():
        return build_panel_manifest()["starts"]
    man = json.loads(path.read_text(encoding="utf-8"))
    wanted = {(r["skill"], r["traj_id"]) for r in man["ics"]}
    pool = load_heldout_starts(per_class=PANEL_PER_CLASS * 6)
    selected = {sk: [] for sk in ("A1", "B1", "C", "D")}
    for sk in selected:
        for s in pool[sk]:
            if (sk, s.traj_id) in wanted:
                selected[sk].append(s)
    return selected


def eval_on_panel(tag: str, ckpt: Path, *, device: str = "cuda", n_rollouts: int = PANEL_ROLLOUTS) -> dict:
    starts_map = load_panel_starts()
    flat = []
    for sk in ("A1", "B1", "C", "D"):
        flat.extend(starts_map[sk])
    rep = eval_starts(
        tag,
        ckpt,
        flat,
        split="p65_extra_dev",
        n_rollouts=n_rollouts,
        device=device,
    )
    # summarize like eval_dev
    by_class = {}
    for sk, lab in (("A1", "A"), ("B1", "B"), ("C", "C"), ("D", "D")):
        rows = [r for r in rep["per_ic"] if r["skill"] == sk]
        by_class[lab] = {
            "MSP_RELAXED": float(np.mean([r["p_relaxed"] for r in rows])) if rows else None,
            "zero_success_ics": int(sum(r["zero_success"] for r in rows)),
            "n_ic": len(rows),
        }
    out = {
        "tag": tag,
        "ckpt": str(ckpt),
        "panel": "P65_extra_dev",
        "n_rollouts": n_rollouts,
        "MSP_RELAXED": rep["MSP_RELAXED"],
        "zero_success_ics": rep["zero_success_ics"],
        "by_class": by_class,
        "per_ic": rep["per_ic"],
    }
    (OUT / f"PANEL_EVAL_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(json.dumps(build_panel_manifest()["manifest"], indent=2)[:2000])
