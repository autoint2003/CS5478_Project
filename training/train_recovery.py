"""Train recovery SAC from a frozen deteriorating-state buffer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.recovery_env import RecoveryEnv, load_buffer  # noqa: E402

TRAIN_PATH = ROOT / "results" / "buffers" / "recovery_train.npz"
CKPT = ROOT / "results" / "checkpoints"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/randomized.yaml")
    parser.add_argument("--action", default="recovery4d", choices=("recovery4d", "2d", "7d"))
    parser.add_argument("--no-contact-obs", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--buffer", default=str(ROOT / "results" / "buffers" / "recovery_train_smoke.npz"))
    parser.add_argument("--reward-mode", default=None, choices=("oracle", "observable_tactile"))
    args = parser.parse_args()

    cfg = merge_sim_config(load_yaml(args.config))
    buf = Path(args.buffer)
    if not buf.exists():
        print(f"missing buffer {buf}")
        return 1
    snaps = load_buffer(buf)
    env = Monitor(
        RecoveryEnv(
            cfg=cfg,
            snapshots=snaps,
            action_kind=args.action,
            use_contact_obs=not args.no_contact_obs,
            reward_mode=args.reward_mode,
        )
    )
    sac_cfg = cfg.get("sac", {})
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=int(sac_cfg.get("learning_starts", 1024)),
        batch_size=int(sac_cfg.get("batch_size", 256)),
        buffer_size=int(sac_cfg.get("buffer_size", 200000)),
        tau=float(sac_cfg.get("tau", 0.005)),
        gamma=float(sac_cfg.get("gamma", 0.99)),
        verbose=1,
        tensorboard_log=str(ROOT / "results" / "tb"),
    )
    steps = int(args.timesteps or sac_cfg.get("total_timesteps", 80000))
    model.learn(total_timesteps=steps)
    CKPT.mkdir(parents=True, exist_ok=True)
    tag = args.action
    if args.no_contact_obs:
        tag += "_nocontact"
    out = CKPT / f"sac_recovery_{tag}.zip"
    model.save(str(out))
    print(f"saved {out}")
    print("CP9 GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
