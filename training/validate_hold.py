"""CP1: torque actuators + bias-force (gravity) hold at the Panda home pose."""

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

from controllers.bias import hold_ctrl  # noqa: E402
from envs.config_util import load_yaml  # noqa: E402
from envs.ids import resolve_ids  # noqa: E402
from envs.xml_build import write_panda_torque  # noqa: E402

SCENE = ROOT / "assets" / "scene.xml"
MAX_DRIFT_RAD = 0.05
HOLD_SECONDS = 5.0


def _assert_torque_actuators(model: mujoco.MjModel) -> None:
    if model.nu != 8:
        raise RuntimeError(f"expected 8 actuators, got {model.nu}")
    # mjGAIN_FIXED == 0, mjBIAS_NONE == 0. Position-PD Menagerie actuators
    # use affine bias and would fail this check.
    if np.any(model.actuator_biastype != 0):
        raise RuntimeError("actuators still have bias (not torque motors)")
    if np.any(model.actuator_gaintype != 0):
        raise RuntimeError("actuators are not fixed-gain motors")


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids,
    duration_s: float,
    compensate: bool,
    gripper: str,
) -> dict:
    _reset_home(model, data)
    q0 = data.qpos[ids.arm_jnt].copy()
    qf0 = data.qpos[ids.finger_jnt].copy()
    n_steps = int(round(duration_s / model.opt.timestep))
    max_arm = 0.0
    max_finger = 0.0
    for _ in range(n_steps):
        mujoco.mj_forward(model, data)
        if compensate:
            data.ctrl[:] = hold_ctrl(model, data, ids, gripper=gripper)
        else:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        arm_err = np.abs(data.qpos[ids.arm_jnt] - q0)
        max_arm = max(max_arm, float(arm_err.max()))
        max_finger = max(
            max_finger, float(np.abs(data.qpos[ids.finger_jnt] - qf0).max())
        )
    return {
        "max_arm_drift_rad": max_arm,
        "max_finger_drift_m": max_finger,
        "final_arm_qpos": data.qpos[ids.arm_jnt].copy(),
        "home_arm_qpos": q0,
        "n_steps": n_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=HOLD_SECONDS)
    parser.add_argument("--max-drift", type=float, default=MAX_DRIFT_RAD)
    parser.add_argument(
        "--gripper",
        choices=("zero", "bias", "open"),
        default="zero",
        help="gripper hold: zero torque, mapped finger bias, or small open force",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open a passive viewer for the compensated hold",
    )
    args = parser.parse_args()

    xml_path = write_panda_torque()
    sim = load_yaml(ROOT / "config" / "sim.yaml")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = float(sim["physics_dt"])
    data = mujoco.MjData(model)
    ids = resolve_ids(model)
    _assert_torque_actuators(model)

    uncomp = rollout(
        model, data, ids, min(1.0, args.duration), compensate=False, gripper="zero"
    )
    comp = rollout(
        model, data, ids, args.duration, compensate=True, gripper=args.gripper
    )

    print(f"generated: {xml_path}")
    print(f"scene:     {SCENE}")
    print(f"timestep:  {model.opt.timestep:.4f} s  ({1.0 / model.opt.timestep:.0f} Hz)")
    print(f"actuators: {model.nu} torque motors (biastype=none)")
    print(
        "uncompensated (1s, ctrl=0): "
        f"max arm drift {uncomp['max_arm_drift_rad']:.4f} rad"
    )
    print(
        f"compensated ({args.duration:.1f}s, tau=b(q,qdot)): "
        f"max arm drift {comp['max_arm_drift_rad']:.4f} rad  "
        f"(gate < {args.max_drift} rad)"
    )
    print(f"finger drift: {comp['max_finger_drift_m']:.5f} m")
    print("home arm qpos:", np.array2string(comp["home_arm_qpos"], precision=4))
    print("final arm qpos:", np.array2string(comp["final_arm_qpos"], precision=4))

    passed = comp["max_arm_drift_rad"] < args.max_drift
    if uncomp["max_arm_drift_rad"] <= args.max_drift:
        print(
            "WARN: uncompensated arm also held still; "
            "check that actuators are really torque motors."
        )
    if passed:
        print("CP1 GATE: PASS")
    else:
        print("CP1 GATE: FAIL")

    if args.viewer:
        _reset_home(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                data.ctrl[:] = hold_ctrl(model, data, ids, gripper=args.gripper)
                mujoco.mj_step(model, data)
                viewer.sync()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
