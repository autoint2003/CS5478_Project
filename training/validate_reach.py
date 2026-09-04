"""CP2: 6D Cartesian PD reaching in torque space (position + orientation)."""

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

from controllers.jacobian_controller import (  # noqa: E402
    apply_cartesian_ctrl,
    ori_error_deg,
)
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


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def sample_target(
    rng: np.random.Generator, r_home: np.ndarray, reach: dict
) -> tuple[np.ndarray, np.ndarray]:
    p = np.array(
        [
            rng.uniform(*reach["x_range"]),
            rng.uniform(*reach["y_range"]),
            rng.uniform(*reach["z_range"]),
        ]
    )
    yaw = np.deg2rad(rng.uniform(*reach["yaw_range_deg"]))
    pitch = np.deg2rad(rng.uniform(*reach["pitch_range_deg"]))
    r_des = r_home @ _rotz(yaw) @ _roty(pitch)
    return p, r_des


def cartesian_gains(cfg: dict) -> dict:
    block = cfg.get("cartesian", {})
    keys = ("kp_pos", "kd_pos", "kp_ori", "kd_ori", "kp_null", "kd_null")
    return {k: float(block[k]) for k in keys if k in block}


def run_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids,
    p_des: np.ndarray,
    r_des: np.ndarray,
    duration: float,
    gains: dict,
) -> dict:
    _reset_home(model, data)
    n_steps = int(round(duration / model.opt.timestep))
    settle = max(1, int(round(0.5 / model.opt.timestep)))
    pos_hist = np.zeros(settle)
    ori_hist = np.zeros(settle)
    for i in range(n_steps):
        mujoco.mj_forward(model, data)
        apply_cartesian_ctrl(model, data, ids, p_des, r_des, **gains)
        mujoco.mj_step(model, data)
        p = data.xpos[ids.hand_body]
        r = data.xmat[ids.hand_body].reshape(3, 3)
        pos_err = float(np.linalg.norm(p_des - p))
        ori_err = ori_error_deg(r_des, r)
        if i >= n_steps - settle:
            pos_hist[i - (n_steps - settle)] = pos_err
            ori_hist[i - (n_steps - settle)] = ori_err
    return {
        "pos_err": float(pos_hist.mean()),
        "ori_err_deg": float(ori_hist.mean()),
        "pos_err_final": float(pos_hist[-1]),
        "ori_err_final": float(ori_hist[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nominal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="replay the first sampled target in a viewer",
    )
    args = parser.parse_args()

    write_panda_torque()
    cfg = merge_sim_config(load_yaml(args.config))
    reach = cfg["reach"]
    n_trials = int(args.n_trials or reach["n_trials"])
    duration = float(args.duration or reach["duration"])
    pos_tol = float(reach["pos_tol"])
    ori_tol = float(reach["ori_tol_deg"])
    min_rate = float(reach.get("min_success_rate", 0.8))
    gains = cartesian_gains(cfg)

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = float(cfg["physics_dt"])
    data = mujoco.MjData(model)
    ids = resolve_ids(model)

    _reset_home(model, data)
    p_home = data.xpos[ids.hand_body].copy()
    r_home = data.xmat[ids.hand_body].reshape(3, 3).copy()
    rng = np.random.default_rng(args.seed)

    rows = []
    n_ok = 0
    for k in range(n_trials):
        p_des, r_des = sample_target(rng, r_home, reach)
        out = run_trial(model, data, ids, p_des, r_des, duration, gains)
        ok = out["pos_err"] < pos_tol and out["ori_err_deg"] < ori_tol
        n_ok += int(ok)
        rows.append((ok, out["pos_err"], out["ori_err_deg"]))
        mark = "OK" if ok else "FAIL"
        print(
            f"  trial {k:02d}  {mark}  "
            f"pos {out['pos_err']*100:.2f} cm  ori {out['ori_err_deg']:.2f} deg"
        )

    rate = n_ok / n_trials
    mean_pos = float(np.mean([r[1] for r in rows]))
    mean_ori = float(np.mean([r[2] for r in rows]))
    print(f"home EE pos: {np.array2string(p_home, precision=3)}")
    print(f"timestep: {model.opt.timestep:.4f} s  duration: {duration:.1f} s")
    print(
        f"success {n_ok}/{n_trials} = {100*rate:.0f}%  "
        f"(gate: pos < {pos_tol*100:.0f} cm, ori < {ori_tol:.0f} deg, "
        f"rate >= {100*min_rate:.0f}%)"
    )
    print(f"mean pos error {mean_pos*100:.2f} cm  mean ori error {mean_ori:.2f} deg")
    passed = rate >= min_rate
    print("CP2 GATE: PASS" if passed else "CP2 GATE: FAIL")

    if args.viewer:
        rng = np.random.default_rng(args.seed)
        p_des, r_des = sample_target(rng, r_home, reach)
        _reset_home(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                apply_cartesian_ctrl(model, data, ids, p_des, r_des, **gains)
                mujoco.mj_step(model, data)
                viewer.sync()

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
