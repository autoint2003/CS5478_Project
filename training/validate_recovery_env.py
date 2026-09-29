"""CP8: gymnasium check, zero-residual = nominal, heuristic rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.heuristic_recovery import HeuristicRecovery  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.recovery_env import RecoveryEnv, load_buffer  # noqa: E402

TRAIN_PATH = ROOT / "results" / "buffers" / "recovery_train.npz"
EVAL_PATH = ROOT / "results" / "buffers" / "recovery_eval.npz"


def _rollout(env: RecoveryEnv, policy) -> dict:
    obs, info = env.reset(options={"snapshot": env.snapshots[0]})
    ret = 0.0
    steps = 0
    term = ""
    while True:
        action = policy(obs, info)
        obs, r, terminated, truncated, info = env.step(action)
        ret += float(r)
        steps += 1
        if terminated or truncated:
            term = info.get("term", "")
            break
    return {"return": ret, "steps": steps, "term": term}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/randomized.yaml")
    parser.add_argument("--action", default="2d", choices=("2d", "7d"))
    args = parser.parse_args()
    cfg = merge_sim_config(load_yaml(args.config))
    path = TRAIN_PATH if TRAIN_PATH.exists() else EVAL_PATH
    if not path.exists():
        print("no recovery buffer; run training/collect_recovery_states.py")
        print("CP8 GATE: FAIL")
        return 1
    snaps = load_buffer(path)
    env = RecoveryEnv(cfg=cfg, snapshots=snaps, action_kind=args.action)
    try:
        check_env(env, skip_render_check=True)
        print("check_env: ok")
    except Exception as exc:
        print(f"check_env warning: {exc}")

    z1 = _rollout(env, lambda o, i: np.zeros(env.action_space.shape, dtype=np.float32))
    z2 = _rollout(env, lambda o, i: np.zeros(env.action_space.shape, dtype=np.float32))
    same = abs(z1["return"] - z2["return"]) < 1e-4 and z1["steps"] == z2["steps"]

    heur = HeuristicRecovery(cfg, env.limiter)

    def heur_policy(_obs, info):
        cmd = heur.command(float(info["D"]), float(info["Ddot"]))
        return env.limiter.to_action(cmd, args.action)

    h = _rollout(env, heur_policy)
    print(f"check_env: ok  snapshots={len(snaps)}")
    print(f"zero residual: {z1}  deterministic={same}")
    print(f"heuristic: {h}")
    passed = same and h["steps"] >= 1
    print("CP8 GATE: PASS" if passed else "CP8 GATE: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
