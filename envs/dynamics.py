"""Set cylinder mass/friction and sample train vs held-out uncertainty."""

from __future__ import annotations

import numpy as np
import mujoco

from envs.ids import PandaIds


def set_object_dynamics(
    model: mujoco.MjModel,
    ids: PandaIds,
    mass: float,
    friction: float,
    data: mujoco.MjData | None = None,
) -> None:
    bid = ids.object_body
    gid = ids.object_geom
    old = float(model.body_mass[bid])
    mass = float(max(mass, 1e-6))
    if old > 0:
        model.body_inertia[bid] = model.body_inertia[bid] * (mass / old)
    model.body_mass[bid] = mass
    model.geom_friction[gid] = np.array([float(friction), 0.005, 0.0001])
    if data is not None:
        try:
            mujoco.mj_setConst(model, data)
        except Exception:
            mujoco.mj_forward(model, data)


def in_heldout(mass: float, friction: float, offset: np.ndarray, cfg: dict) -> bool:
    h = cfg.get("heldout")
    if not h:
        return False
    m0, m1 = h["mass_range"]
    f0, f1 = h["friction_range"]
    oxy = h.get("grasp_offset_xy", [0.006, 0.012])
    if isinstance(oxy, (int, float)):
        o0, o1 = 0.0, float(oxy)
    else:
        o0, o1 = float(oxy[0]), float(oxy[1])
    rxy = float(np.linalg.norm(offset[:2]))
    return (m0 <= mass <= m1) and (f0 <= friction <= f1) and (o0 <= rxy <= o1)


def sample_uncertainty(rng: np.random.Generator, cfg: dict, heldout: bool) -> tuple[float, float, np.ndarray]:
    if heldout:
        h = cfg["heldout"]
        mass = float(rng.uniform(*h["mass_range"]))
        friction = float(rng.uniform(*h["friction_range"]))
        oxy = h.get("grasp_offset_xy", [0.006, 0.012])
        if isinstance(oxy, (int, float)):
            r = float(rng.uniform(0.0, float(oxy)))
        else:
            r = float(rng.uniform(float(oxy[0]), float(oxy[1])))
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        zlim = float(cfg.get("train", {}).get("grasp_offset_z", 0.005))
        offset = np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(-zlim, zlim)])
        return mass, friction, offset

    t = cfg.get("train", {})
    for _ in range(64):
        mass = float(rng.uniform(*t.get("mass_range", [0.05, 0.25])))
        friction = float(rng.uniform(*t.get("friction_range", [0.4, 1.2])))
        rmax = float(t.get("grasp_offset_xy", 0.012))
        zlim = float(t.get("grasp_offset_z", 0.005))
        r = float(rng.uniform(0.0, rmax))
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        offset = np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(-zlim, zlim)])
        if not in_heldout(mass, friction, offset, cfg):
            return mass, friction, offset
    return mass, friction, offset
