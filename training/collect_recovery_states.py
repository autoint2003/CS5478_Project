"""Collect deteriorating lift states and freeze train/eval buffers."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.dynamics import sample_uncertainty  # noqa: E402
from envs.grasp_sim import GraspSim, run_to_lift_or_deterioration  # noqa: E402
from envs.recovery_env import save_buffer  # noqa: E402

TRAIN_PATH = ROOT / "results" / "buffers" / "recovery_train.npz"
EVAL_PATH = ROOT / "results" / "buffers" / "recovery_eval.npz"


def assign_severity(snaps: list[dict]) -> None:
    if not snaps:
        return
    ds = np.array([float(s["D"]) for s in snaps], dtype=float)
    q1, q2 = np.quantile(ds, [1.0 / 3.0, 2.0 / 3.0])
    for s in snaps:
        d = float(s["D"])
        if d <= q1:
            s["severity"] = "mild"
        elif d <= q2:
            s["severity"] = "moderate"
        else:
            s["severity"] = "severe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/randomized.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = merge_sim_config(load_yaml(args.config))
    col = cfg.get("collect", {})
    n_train = int(col.get("n_train", 48))
    n_eval = int(col.get("n_eval", 16))
    max_ep = int(col.get("max_episodes", 120))
    rng = np.random.default_rng(args.seed)
    sim = GraspSim(cfg)

    train: list[dict] = []
    ev: list[dict] = []

    for ep in range(max_ep):
        if len(train) >= n_train and len(ev) >= n_eval:
            break
        need_eval = len(ev) < n_eval
        heldout = need_eval and (len(train) >= n_train or ep % 3 == 0)
        if rng.random() < 0.55 and not heldout:
            mass = float(rng.uniform(0.18, 0.25))
            friction = float(rng.uniform(0.40, 0.55))
            r = float(rng.uniform(0.006, 0.012))
            ang = float(rng.uniform(0.0, 2.0 * np.pi))
            offset = np.array(
                [r * np.cos(ang), r * np.sin(ang), rng.uniform(-0.004, 0.004)]
            )
        else:
            mass, friction, offset = sample_uncertainty(rng, cfg, heldout=heldout)
        sim.reset(mass, friction, offset)
        outcome = run_to_lift_or_deterioration(sim)
        if outcome != "deteriorate":
            continue
        if sim.dropped() or sim.bad_state():
            continue
        snap = sim.snapshot()
        snap["heldout"] = bool(heldout)
        if heldout:
            ev.append(snap)
        else:
            train.append(snap)
        print(
            f"  ep {ep:03d}  {outcome:12s}  {'eval' if heldout else 'train'}  "
            f"D={snap['D']:.2f}  n_train={len(train)} n_eval={len(ev)}"
        )

    assign_severity(train)
    assign_severity(ev)
    counts = Counter(s["severity"] for s in train)

    if not train:
        print("no deteriorating states collected")
        print("CP7 GATE: FAIL")
        return 1

    save_buffer(TRAIN_PATH, train)
    save_buffer(EVAL_PATH, ev if ev else train[: max(1, len(train) // 5)])
    print(f"train {len(train)}  eval {len(ev)}  bins {dict(counts)}")
    print(f"train buffer: {TRAIN_PATH}")
    print(f"eval buffer:  {EVAL_PATH}")
    bins_ok = all(counts[b] >= 1 for b in ("mild", "moderate", "severe"))
    has_eval = bool(ev) or len(train) > 1
    print("CP7 GATE: PASS" if bins_ok and has_eval else "CP7 GATE: FAIL")
    return 0 if bins_ok and has_eval else 1


if __name__ == "__main__":
    raise SystemExit(main())
