"""CP11: 7-D Cartesian residual and contact-observation ablation wiring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.recovery_env import RecoveryEnv, load_buffer  # noqa: E402

TRAIN_PATH = ROOT / "results" / "buffers" / "recovery_train.npz"


def _probe(env: RecoveryEnv, label: str) -> None:
    obs, _ = env.reset(seed=0)
    a = env.action_space.sample()
    nobs, _r, _t, _tr, info = env.step(a)
    print(
        f"{label}: act={env.action_space.shape} obs={obs.shape}->{nobs.shape} "
        f"contact_obs={env.use_contact_obs} D={info['D']:.3f}"
    )
    assert obs.shape == nobs.shape == env.observation_space.shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/randomized.yaml")
    args = parser.parse_args()
    cfg = merge_sim_config(load_yaml(args.config))
    snaps = load_buffer(TRAIN_PATH) if TRAIN_PATH.exists() else None
    e2 = RecoveryEnv(cfg=cfg, snapshots=snaps, action_kind="2d", use_contact_obs=True)
    e7 = RecoveryEnv(cfg=cfg, snapshots=snaps, action_kind="7d", use_contact_obs=True)
    e_nc = RecoveryEnv(
        cfg=cfg, snapshots=snaps, action_kind="2d", use_contact_obs=False
    )
    _probe(e2, "2d+contact")
    _probe(e7, "7d+contact")
    _probe(e_nc, "2d+no-contact-obs")
    assert e2.action_space.shape == (2,)
    assert e7.action_space.shape == (7,)
    print("Use training/train_recovery.py --action 7d and --no-contact-obs to retrain.")
    print("CP11 GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
