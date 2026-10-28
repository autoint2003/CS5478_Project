"""Physically consistent Precision IC construction for P34.

Extends P14 absolute residual targeting with optional orientation, mild
relative velocity, and pre-freeze mass/friction scales. No post-freeze teleport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from data_generation.expert_pipeline.live_stage1 import settle_grip
from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.fsm import classify_case
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
from data_generation.oracles.precision.state import PrecisionConfig
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure, p_com_hand
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from experiments.precision.p34_precision_data_expansion.regimes import RegimeSpec


def _object_qpos_adr(sim) -> int | None:
    jnt = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if jnt >= 0:
        return int(sim.model.jnt_qposadr[jnt])
    bid = int(sim.ids.object_body)
    for j in range(sim.model.njnt):
        if int(sim.model.jnt_bodyid[j]) == bid and int(sim.model.jnt_type[j]) == int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            return int(sim.model.jnt_qposadr[j])
    return None


def _penetration_m(sim) -> float:
    d = 0.0
    for i in range(int(sim.data.ncon)):
        dist = float(sim.data.contact[i].dist)
        if dist < 0:
            d = max(d, -dist)
    return d


def _object_geom_ids(sim) -> list[int]:
    bid = int(sim.ids.object_body)
    out = []
    for g in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[g]) == bid:
            out.append(int(g))
    return out


@dataclass
class DynamicsOverride:
    body_mass: float | None = None
    geom_friction: dict[int, list[float]] | None = None  # geom_id -> friction[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_mass": self.body_mass,
            "geom_friction": {str(k): v for k, v in (self.geom_friction or {}).items()},
        }

    @staticmethod
    def from_dict(d: dict | None) -> "DynamicsOverride | None":
        if not d:
            return None
        gf = d.get("geom_friction") or {}
        return DynamicsOverride(
            body_mass=d.get("body_mass"),
            geom_friction={int(k): list(v) for k, v in gf.items()} if gf else None,
        )


def apply_dynamics_override(sim, ov: DynamicsOverride | None) -> None:
    if ov is None:
        return
    bid = int(sim.ids.object_body)
    if ov.body_mass is not None:
        sim.model.body_mass[bid] = float(ov.body_mass)
    if ov.geom_friction:
        for gid, fr in ov.geom_friction.items():
            sim.model.geom_friction[int(gid)][:] = np.asarray(fr, float)


def snapshot_default_dynamics(sim) -> DynamicsOverride:
    bid = int(sim.ids.object_body)
    gf = {g: np.asarray(sim.model.geom_friction[g], float).tolist() for g in _object_geom_ids(sim)}
    return DynamicsOverride(body_mass=float(sim.model.body_mass[bid]), geom_friction=gf)


def restore_default_dynamics(sim, defaults: DynamicsOverride) -> None:
    apply_dynamics_override(sim, defaults)


def measure_pack(sim) -> dict[str, Any]:
    err = measure_errors(sim)
    cs = contact_summary(sim)
    m = measure(sim, U_TGT, P_TGT)
    return {
        "dx_mm": err.get("dx_mm"),
        "dy_mm": err.get("dy_mm"),
        "dz_mm": err.get("dz_mm"),
        "dtheta_deg": err.get("dtheta_deg"),
        "e_pos_mm": err.get("e_pos_mm"),
        "v_rel_norm": err.get("v_rel_norm"),
        "bilateral": bool(cs.get("bilateral")),
        "nL": int(cs.get("nL") or 0),
        "nR": int(cs.get("nR") or 0),
        "penetration_m": _penetration_m(sim),
        "p_c_h": m.get("p_c_h"),
    }


@dataclass
class BuildResult:
    ok: bool
    stage: str
    reason: str | None = None
    capture: Any = None
    ref: Any = None
    capture_fp: str | None = None
    dynamics_override: DynamicsOverride | None = None
    log: dict = field(default_factory=dict)


def build_regime_ic(
    env,
    base_blob: dict,
    spec: RegimeSpec,
    *,
    cfg: PrecisionConfig | None = None,
    defaults: DynamicsOverride | None = None,
) -> BuildResult:
    """Construct IC from donor grasp + regime; freeze complete snapshot."""
    pcfg = cfg or PrecisionConfig()
    sim = env.sim
    if defaults is None:
        defaults = snapshot_default_dynamics(sim)

    # Always start from defaults then apply this candidate's scales.
    restore_default_dynamics(sim, defaults)
    restore_capture(env, base_blob["capture"])
    mujoco.mj_forward(sim.model, sim.data)

    base_pack = measure_pack(sim)
    if not base_pack["bilateral"]:
        return BuildResult(ok=False, stage="BASE", reason="BASE_NOT_BILATERAL", log={"base": base_pack})

    bid = int(sim.ids.object_body)
    dyn: DynamicsOverride | None = None
    if abs(spec.mass_scale - 1.0) > 1e-9 or abs(spec.friction_scale - 1.0) > 1e-9:
        mass0 = float(defaults.body_mass) if defaults.body_mass is not None else float(sim.model.body_mass[bid])
        gf = {}
        for g in _object_geom_ids(sim):
            fr0 = np.asarray(
                (defaults.geom_friction or {}).get(g, sim.model.geom_friction[g]), float
            ).copy()
            fr0 *= float(spec.friction_scale)
            gf[g] = fr0.tolist()
        dyn = DynamicsOverride(body_mass=mass0 * float(spec.mass_scale), geom_friction=gf)
        apply_dynamics_override(sim, dyn)
        mujoco.mj_forward(sim.model, sim.data)

    adr = _object_qpos_adr(sim)
    if adr is None:
        restore_default_dynamics(sim, defaults)
        return BuildResult(ok=False, stage="CONSTRUCT", reason="NO_OBJECT_FREE_JOINT")

    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    p_c_h = np.asarray(p_com_hand(sim), float)
    e_cur = p_c_h - np.asarray(P_TGT, float)
    # Absolute residual target; keep current ey unless dy specified.
    ey = float(spec.target_dy_mm) * 1e-3 if abs(spec.target_dy_mm) > 1e-12 else float(e_cur[1])
    e_tgt = np.array([spec.target_dx_mm * 1e-3, ey, spec.target_dz_mm * 1e-3], float)
    delta_h = e_tgt - e_cur
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    po_new = po + Rh @ delta_h

    qpos = np.array(sim.data.qpos, float).copy()
    qpos[adr : adr + 3] = po_new
    if abs(spec.dtheta_deg) > 1e-9:
        axis_w = Rh[:, 1]
        axis_w = axis_w / max(float(np.linalg.norm(axis_w)), 1e-12)
        ang = np.deg2rad(float(spec.dtheta_deg))
        qw, qx, qy, qz = qpos[adr + 3 : adr + 7]
        half = 0.5 * ang
        dw = np.cos(half)
        dx_, dy_, dz_ = axis_w * np.sin(half)
        nw = dw * qw - dx_ * qx - dy_ * qy - dz_ * qz
        nx = dw * qx + dx_ * qw + dy_ * qz - dz_ * qy
        ny = dw * qy - dx_ * qz + dy_ * qw + dz_ * qx
        nz = dw * qz + dx_ * qy - dy_ * qx + dz_ * qw
        qn = np.array([nw, nx, ny, nz], float)
        qn /= max(float(np.linalg.norm(qn)), 1e-12)
        qpos[adr + 3 : adr + 7] = qn

    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    pre_settle = measure_pack(sim)
    pen = float(pre_settle["penetration_m"])
    max_pen = 0.003
    if pen > max_pen:
        restore_default_dynamics(sim, defaults)
        return BuildResult(
            ok=False,
            stage="PRE_SETTLE",
            reason=f"PENETRATION:{pen:.4f}",
            log={"pre_settle": pre_settle, "spec": spec.to_dict()},
        )

    settle_grip(env, n=14)
    for _ in range(4):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(sim, a)
    mujoco.mj_forward(sim.model, sim.data)

    # Mild relative velocity after settle (stored in freeze via qvel).
    v_lin = np.asarray(spec.v_lin_hand_mms, float) * 1e-3  # m/s
    w_deg = np.asarray(spec.w_hand_dps, float)
    if np.any(np.abs(v_lin) > 1e-9) or np.any(np.abs(w_deg) > 1e-9):
        # free joint qvel: [vx,vy,vz, wx,wy,wz] in world; map hand-frame → world
        v_w = Rh @ v_lin
        w_w = Rh @ np.deg2rad(w_deg)
        qvel = np.array(sim.data.qvel, float).copy()
        # free joint dofadr
        jnt = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        if jnt >= 0:
            da = int(sim.model.jnt_dofadr[jnt])
            qvel[da : da + 3] = v_w
            qvel[da + 3 : da + 6] = w_w
            sim.data.qvel[:] = qvel
            mujoco.mj_forward(sim.model, sim.data)

    post = measure_pack(sim)
    if not post["bilateral"]:
        restore_default_dynamics(sim, defaults)
        return BuildResult(
            ok=False,
            stage="POST_SETTLE",
            reason="LOST_BILATERAL",
            log={"post_settle": post, "spec": spec.to_dict()},
        )
    if float(post["penetration_m"]) > max_pen:
        restore_default_dynamics(sim, defaults)
        return BuildResult(
            ok=False,
            stage="POST_SETTLE",
            reason=f"PENETRATION_POST:{post['penetration_m']:.4f}",
            log={"post_settle": post, "spec": spec.to_dict()},
        )

    err = measure_errors(sim)
    need = assess_need(err, pcfg)
    plan = classify_case(err, pcfg)
    prec_req = bool(
        need.get("dx_needed")
        or need.get("dz_needed")
        or need.get("dtheta_needed")
        or not need.get("ready_for_direct_handoff_assessment")
    )

    cap = take_capture(env)
    # Stash dynamics override inside capture for restore fidelity
    if dyn is not None:
        cap = dict(cap)
        cap["dynamics_override"] = dyn.to_dict()
    fp = capture_fingerprint(cap)
    log = {
        "regime_id": spec.regime_id,
        "family": spec.family,
        "bucket": spec.bucket,
        "base_sample_id": base_blob.get("sample_id"),
        "requested_dx_mm": spec.target_dx_mm,
        "requested_dy_mm": spec.target_dy_mm,
        "requested_dz_mm": spec.target_dz_mm,
        "requested_dtheta_deg": spec.dtheta_deg,
        "mass_scale": spec.mass_scale,
        "friction_scale": spec.friction_scale,
        "v_lin_hand_mms": list(spec.v_lin_hand_mms),
        "w_hand_dps": list(spec.w_hand_dps),
        "base_dx_mm": base_pack["dx_mm"],
        "base_dz_mm": base_pack["dz_mm"],
        "post_settle_dx_mm": post["dx_mm"],
        "post_settle_dy_mm": post["dy_mm"],
        "post_settle_dz_mm": post["dz_mm"],
        "post_settle_dtheta_deg": post["dtheta_deg"],
        "post_settle_v_rel": post["v_rel_norm"],
        "post_settle_bilateral": post["bilateral"],
        "post_settle_nL": post["nL"],
        "post_settle_nR": post["nR"],
        "precision_required": prec_req,
        "dx_needed": need.get("dx_needed"),
        "dz_needed": need.get("dz_needed"),
        "dtheta_needed": need.get("dtheta_needed"),
        "fsm_case": plan.case,
        "fsm_planned": "|".join(plan.planned_skills),
        "capture_fp": fp,
        "construction": "p34_regime_absolute_residual_pre_freeze",
        "object_mass_kg": float(sim.model.body_mass[bid]),
        "object_friction0": float(sim.model.geom_friction[_object_geom_ids(sim)[0]][0])
        if _object_geom_ids(sim)
        else None,
    }
    return BuildResult(
        ok=True,
        stage="FREEZE",
        capture=cap,
        ref=base_blob["ref"],
        capture_fp=fp,
        dynamics_override=dyn,
        log=log,
    )


def restore_ic(env, capture, *, defaults: DynamicsOverride | None = None) -> None:
    """Restore capture including optional dynamics_override; reset defaults first."""
    sim = env.sim
    if defaults is None:
        defaults = snapshot_default_dynamics(sim)
    restore_default_dynamics(sim, defaults)
    ov = DynamicsOverride.from_dict(capture.get("dynamics_override") if isinstance(capture, dict) else None)
    apply_dynamics_override(sim, ov)
    restore_capture(env, capture)
