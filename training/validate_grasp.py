"""CP3: deterministic no-RL grasp FSM (approach → descend → close → lift)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.gripper_controller import finger_opening, read_touch  # noqa: E402
from controllers.nominal import GraspFSM, object_pos  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.ids import resolve_ids  # noqa: E402
from envs.xml_build import write_panda_torque  # noqa: E402

SCENE = ROOT / "assets" / "scene.xml"


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids,
    cfg: dict,
) -> dict:
    _reset_home(model, data)
    fsm = GraspFSM(cfg)
    fsm.reset()
    timeout = float(cfg["fsm"]["episode_timeout"])
    n_steps = int(round(timeout / model.opt.timestep))
    max_z = float(object_pos(data, ids)[2])
    peak_touch = 0.0
    for _ in range(n_steps):
        mujoco.mj_forward(model, data)
        fsm.step(model, data, ids)
        mujoco.mj_step(model, data)
        z = float(object_pos(data, ids)[2])
        max_z = max(max_z, z)
        peak_touch = max(peak_touch, float(np.max(read_touch(data, ids))))
        if fsm.success:
            break
    obj = object_pos(data, ids)
    return {
        "success": bool(fsm.success),
        "phase": fsm.phase,
        "cube_z": float(obj[2]),
        "max_cube_z": max_z,
        "held": float(fsm.t_held),
        "opening": finger_opening(data, ids),
        "peak_touch": peak_touch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nominal.yaml")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="run one episode in a passive viewer",
    )
    args = parser.parse_args()

    write_panda_torque()
    cfg = merge_sim_config(load_yaml(args.config))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = float(cfg["physics_dt"])
    data = mujoco.MjData(model)
    ids = resolve_ids(model)

    n_ok = 0
    for k in range(args.n_trials):
        out = run_episode(model, data, ids, cfg)
        n_ok += int(out["success"])
        mark = "OK" if out["success"] else "FAIL"
        print(
            f"  trial {k:02d}  {mark}  phase={out['phase']:8s}  "
            f"cube_z={out['cube_z']:.3f}  max_z={out['max_cube_z']:.3f}  "
            f"held={out['held']:.2f}s  open={out['opening']:.3f}  "
            f"touch={out['peak_touch']:.3f}"
        )

    rate = n_ok / args.n_trials
    print(
        f"grasp success {n_ok}/{args.n_trials} = {100 * rate:.0f}%  "
        f"(object lifted and held {cfg['fsm']['hold_success']:.1f}s)"
    )
    passed = n_ok == args.n_trials
    print("CP3 GATE: PASS" if passed else "CP3 GATE: FAIL")

    if args.viewer:
        _reset_home(model, data)
        fsm = GraspFSM(cfg)
        fsm.reset()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                fsm.step(model, data, ids)
                mujoco.mj_step(model, data)
                viewer.sync()
                if fsm.success:
                    fsm.reset()
                    _reset_home(model, data)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
