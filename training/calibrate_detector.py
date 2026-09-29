"""Calibrate D_t scales/thresholds on stable vs deteriorating lifts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.grasp_sim import GraspSim  # noqa: E402

FIG = ROOT / "results" / "figures" / "detector_calibration.png"


def rollout_D(sim: GraspSim, mass: float, friction: float, offset: np.ndarray) -> dict:
    sim.reset(mass, friction, offset)
    timeout = float(sim.cfg["fsm"]["episode_timeout"])
    n_steps = int(round(timeout / sim.model.opt.timestep))
    hist = []
    outcome = "timeout"
    for i in range(n_steps):
        sim.physics_step(None, in_recovery=False)
        sim.maybe_capture_reference()
        if sim.fsm.phase == "lift" and sim.captured and (i + 1) % sim.n_sub == 0:
            snap = sim.meter.compute(sim.model, sim.data, sim.ids, sim.dt_policy)
            hist.append(snap.D)
            if snap.D > sim.meter.D_enter and outcome == "timeout":
                outcome = "deteriorate"
        if sim.dropped():
            outcome = "drop"
            break
        if sim.fsm.success:
            outcome = "success"
            break
    arr = np.asarray(hist, dtype=float)
    return {
        "D": arr,
        "max_D": float(arr.max()) if arr.size else 0.0,
        "mean_D": float(arr.mean()) if arr.size else 0.0,
        "outcome": outcome,
        "success": outcome == "success",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nominal.yaml")
    args = parser.parse_args()
    cfg = merge_sim_config(load_yaml(args.config))
    sim = GraspSim(cfg)
    D_enter = sim.meter.D_enter

    stable = []
    fail = []
    for k in range(4):
        stable.append(rollout_D(sim, 0.08, 1.0, np.zeros(3)))
    harsh = [
        (0.22, 0.42, np.array([0.010, 0.0, 0.0])),
        (0.24, 0.40, np.array([0.0, 0.010, 0.0])),
        (0.20, 0.45, np.array([0.008, 0.008, 0.0])),
        (0.25, 0.40, np.array([0.012, 0.0, 0.003])),
    ]
    for m, mu, off in harsh:
        fail.append(rollout_D(sim, m, mu, off))

    stable_max = [r["max_D"] for r in stable if r["D"].size]
    fail_enter = []
    for r in fail:
        if r["D"].size:
            fail_enter.append(bool(np.any(r["D"] > D_enter)))

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in stable:
        if r["D"].size:
            ax.plot(r["D"], color="0.35", alpha=0.8)
    for r in fail:
        if r["D"].size:
            ax.plot(r["D"], color="#8b2e1a", alpha=0.8)
    ax.axhline(D_enter, color="#8b2e1a", ls="--", label="D_enter")
    ax.axhline(sim.meter.D_exit, color="#2d5a3d", ls="--", label="D_exit")
    ax.set_xlabel("policy steps during lift")
    ax.set_ylabel("D_t")
    ax.set_title("Deterioration score: stable (gray) vs harsh (red)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    plt.close(fig)

    stable_below = all(m < D_enter for m in stable_max) if stable_max else False
    fail_hits = sum(fail_enter)
    chatter = False
    for r in stable:
        if r["D"].size < 4:
            continue
        above = r["D"] > D_enter
        if np.any(np.diff(above.astype(int)) != 0) and np.sum(above) < 3:
            chatter = True

    print(f"stable max D: {stable_max}")
    print(f"stable outcomes: {[r['outcome'] for r in stable]}")
    print(f"harsh outcomes: {[r['outcome'] for r in fail]}")
    print(f"harsh crossed D_enter: {fail_hits}/{len(fail)}")
    print(f"fig: {FIG}")
    passed = stable_below and fail_hits >= 2 and not chatter
    print("CP6 GATE: PASS" if passed else "CP6 GATE: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
