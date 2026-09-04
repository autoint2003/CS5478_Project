"""CP4: log fingertip touch and cube-contact forces through a nominal grasp."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.gripper_controller import finger_opening  # noqa: E402
from controllers.nominal import PHASES, GraspFSM, cube_pos  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.contact import finger_cube_normals, touch_values  # noqa: E402
from envs.ids import resolve_ids  # noqa: E402
from envs.xml_build import write_panda_torque  # noqa: E402

SCENE = ROOT / "assets" / "scene.xml"
CSV_PATH = ROOT / "results" / "logs" / "contact_grasp.csv"
FIG_PATH = ROOT / "results" / "figures" / "contact_grasp.png"

PHASE_ID = {name: i for i, name in enumerate(PHASES)}


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def record_grasp(model, data, ids, cfg: dict) -> dict:
    _reset_home(model, data)
    fsm = GraspFSM(cfg)
    fsm.reset()
    timeout = float(cfg["fsm"]["episode_timeout"])
    n_steps = int(round(timeout / model.opt.timestep))
    rows: list[dict] = []
    for k in range(n_steps):
        mujoco.mj_forward(model, data)
        fsm.step(model, data, ids)
        mujoco.mj_step(model, data)
        touch = touch_values(data, ids)
        con = finger_cube_normals(model, data, ids)
        rows.append(
            {
                "t": (k + 1) * float(model.opt.timestep),
                "phase": fsm.phase,
                "phase_id": PHASE_ID[fsm.phase],
                "touch_left": touch[0],
                "touch_right": touch[1],
                "con_left": con[0],
                "con_right": con[1],
                "cube_z": float(cube_pos(data, ids)[2]),
                "opening": finger_opening(data, ids),
            }
        )
        if fsm.success:
            break
    return {"rows": rows, "success": bool(fsm.success)}


def _phase_mask(rows: list[dict], phase: str) -> np.ndarray:
    return np.array([r["phase"] == phase for r in rows], dtype=bool)


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def evaluate_gate(rows: list[dict]) -> dict:
    touch = np.maximum(_col(rows, "touch_left"), _col(rows, "touch_right"))
    con = np.maximum(_col(rows, "con_left"), _col(rows, "con_right"))
    ap = _phase_mask(rows, "approach")
    cl = _phase_mask(rows, "close")
    lf = _phase_mask(rows, "lift")

    free_touch = float(touch[ap].mean()) if ap.any() else 0.0
    free_con = float(con[ap].mean()) if ap.any() else 0.0
    pinch_touch = float(touch[cl].max()) if cl.any() else 0.0
    pinch_con = float(con[cl].max()) if cl.any() else 0.0
    lift_touch = float(touch[lf].mean()) if lf.any() else 0.0
    lift_con = float(con[lf].mean()) if lf.any() else 0.0

    # Prefer cube-contact normals; touch must agree in shape (rise then persist).
    free_ok = free_touch < 1.0 and free_con < 1.0
    pinch_ok = pinch_touch > 1.0 and pinch_con > 1.0
    lift_ok = lift_touch > 1.0 and lift_con > 1.0
    return {
        "free_touch": free_touch,
        "free_con": free_con,
        "pinch_touch": pinch_touch,
        "pinch_con": pinch_con,
        "lift_touch": lift_touch,
        "lift_con": lift_con,
        "passed": bool(free_ok and pinch_ok and lift_ok),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t",
        "phase",
        "phase_id",
        "touch_left",
        "touch_right",
        "con_left",
        "con_right",
        "cube_z",
        "opening",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _col(rows, "t")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)

    axes[0].plot(t, _col(rows, "touch_left"), label="touch left")
    axes[0].plot(t, _col(rows, "touch_right"), label="touch right")
    axes[0].set_ylabel("touch")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Fingertip contact through a nominal grasp")

    axes[1].plot(t, _col(rows, "con_left"), label="cube contact left")
    axes[1].plot(t, _col(rows, "con_right"), label="cube contact right")
    axes[1].set_ylabel("normal force")
    axes[1].legend(loc="upper left")

    axes[2].plot(t, _col(rows, "phase_id"), drawstyle="steps-post", color="0.3")
    axes[2].set_yticks(range(len(PHASES)), PHASES)
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("FSM")

    colors = {"approach": "#dddddd", "descend": "#cfe8ff", "close": "#ffe0b3", "lift": "#c8e6c9"}
    for ax in axes:
        t0 = t[0]
        prev = rows[0]["phase"]
        for r in rows:
            if r["phase"] != prev:
                ax.axvspan(t0, r["t"], color=colors[prev], alpha=0.35, lw=0)
                t0 = r["t"]
                prev = r["phase"]
        ax.axvspan(t0, t[-1], color=colors[prev], alpha=0.35, lw=0)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nominal.yaml")
    args = parser.parse_args()

    write_panda_torque()
    cfg = merge_sim_config(load_yaml(args.config))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = float(cfg["physics_dt"])
    data = mujoco.MjData(model)
    ids = resolve_ids(model)

    out = record_grasp(model, data, ids, cfg)
    rows = out["rows"]
    gate = evaluate_gate(rows)
    write_csv(rows, CSV_PATH)
    write_plot(rows, FIG_PATH)

    print(f"samples: {len(rows)}  grasp success: {out['success']}")
    print(
        f"free space (approach)  mean touch {gate['free_touch']:.4f}  "
        f"mean cube-contact {gate['free_con']:.4f}"
    )
    print(
        f"pinch (close)          max touch  {gate['pinch_touch']:.3f}  "
        f"max cube-contact  {gate['pinch_con']:.3f}"
    )
    print(
        f"lift                   mean touch {gate['lift_touch']:.3f}  "
        f"mean cube-contact {gate['lift_con']:.3f}"
    )
    print(f"csv: {CSV_PATH}")
    print(f"fig: {FIG_PATH}")
    print("CP4 GATE: PASS" if gate["passed"] else "CP4 GATE: FAIL")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
