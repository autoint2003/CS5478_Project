"""Small-error D2 IC construction from known-good grasps.

Targets absolute hand-frame residuals ``e = p_c_h - P_TGT`` (metres→mm),
matching ``measure_errors`` / Precision ``dx_mm`` / ``dz_mm``.

Construction only (pre-FREEZE). After freeze: no teleport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass
class SmallErrorSpec:
    family: str  # DX_ONLY | DZ_ONLY | COMBINED
    target_dx_mm: float = 0.0
    target_dz_mm: float = 0.0
    settle_n: int = 14
    max_penetration_m: float = 0.003

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildResult:
    ok: bool
    stage: str
    reason: str | None = None
    capture: Any = None
    ref: Any = None
    capture_fp: str | None = None
    log: dict = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        return {"ok": self.ok, "stage": self.stage, "reason": self.reason, **self.log}


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
        "P_TGT": np.asarray(P_TGT, float).tolist(),
    }


def build_small_error_ic(
    env,
    base_blob: dict,
    spec: SmallErrorSpec,
    *,
    cfg: PrecisionConfig | None = None,
) -> BuildResult:
    """Move object so hand-frame residual targets (dx,0,dz) mm, then settle+freeze."""
    pcfg = cfg or PrecisionConfig()
    restore_capture(env, base_blob["capture"])
    sim = env.sim
    mujoco.mj_forward(sim.model, sim.data)

    base_pack = measure_pack(sim)
    if not base_pack["bilateral"]:
        return BuildResult(
            ok=False,
            stage="BASE",
            reason="BASE_NOT_BILATERAL",
            log={"base": base_pack, "spec": spec.to_dict()},
        )

    adr = _object_qpos_adr(sim)
    if adr is None:
        return BuildResult(ok=False, stage="CONSTRUCT", reason="NO_OBJECT_FREE_JOINT")

    # Absolute residual target in hand frame (matches e_signed = p_c_h - P_TGT).
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    p_c_h = np.asarray(p_com_hand(sim), float)
    e_cur = p_c_h - np.asarray(P_TGT, float)  # metres
    e_tgt = np.array(
        [spec.target_dx_mm * 1e-3, e_cur[1], spec.target_dz_mm * 1e-3],
        float,
    )
    # Keep current ey; only set ex, ez.
    delta_h = e_tgt - e_cur  # metres, hand frame
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    po_new = po + Rh @ delta_h

    qpos = np.array(sim.data.qpos, float).copy()
    qpos[adr : adr + 3] = po_new
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    pre_settle = measure_pack(sim)
    pen = float(pre_settle["penetration_m"])
    if pen > float(spec.max_penetration_m):
        return BuildResult(
            ok=False,
            stage="PRE_SETTLE",
            reason=f"PENETRATION:{pen:.4f}",
            log={
                "base": base_pack,
                "pre_settle": pre_settle,
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "delta_h_mm": (delta_h * 1e3).tolist(),
                "family": spec.family,
                "base_sample_id": base_blob.get("sample_id"),
            },
        )

    settle_grip(env, n=int(spec.settle_n))
    for _ in range(4):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(sim, a)
    mujoco.mj_forward(sim.model, sim.data)

    post = measure_pack(sim)
    if not post["bilateral"]:
        return BuildResult(
            ok=False,
            stage="POST_SETTLE",
            reason="LOST_BILATERAL",
            log={
                "base": base_pack,
                "pre_settle": pre_settle,
                "post_settle": post,
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "family": spec.family,
                "base_sample_id": base_blob.get("sample_id"),
            },
        )
    if float(post["penetration_m"]) > float(spec.max_penetration_m):
        return BuildResult(
            ok=False,
            stage="POST_SETTLE",
            reason=f"PENETRATION_POST:{post['penetration_m']:.4f}",
            log={
                "base": base_pack,
                "pre_settle": pre_settle,
                "post_settle": post,
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "family": spec.family,
                "base_sample_id": base_blob.get("sample_id"),
            },
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
    fp = capture_fingerprint(cap)
    log = {
        "family": spec.family,
        "base_sample_id": base_blob.get("sample_id"),
        "requested_dx_mm": spec.target_dx_mm,
        "requested_dz_mm": spec.target_dz_mm,
        "delta_h_mm": (delta_h * 1e3).tolist(),
        "base_dx_mm": base_pack["dx_mm"],
        "base_dz_mm": base_pack["dz_mm"],
        "base_e_pos_mm": base_pack["e_pos_mm"],
        "pre_settle_dx_mm": pre_settle["dx_mm"],
        "pre_settle_dz_mm": pre_settle["dz_mm"],
        "pre_settle_e_pos_mm": pre_settle["e_pos_mm"],
        "pre_settle_bilateral": pre_settle["bilateral"],
        "pre_settle_penetration_m": pre_settle["penetration_m"],
        "post_settle_dx_mm": post["dx_mm"],
        "post_settle_dz_mm": post["dz_mm"],
        "post_settle_e_pos_mm": post["e_pos_mm"],
        "post_settle_dtheta_deg": post["dtheta_deg"],
        "post_settle_bilateral": post["bilateral"],
        "post_settle_penetration_m": post["penetration_m"],
        "post_settle_v_rel": post["v_rel_norm"],
        "precision_required": prec_req,
        "dx_needed": need.get("dx_needed"),
        "dz_needed": need.get("dz_needed"),
        "fsm_case": plan.case,
        "fsm_planned": "|".join(plan.planned_skills),
        "ex_tol_mm": pcfg.ex_tol_mm,
        "ez_tol_mm": pcfg.ez_tol_mm,
        "working_pos_mm": pcfg.working_pos_mm,
        "capture_fp": fp,
        "construction": "absolute_hand_frame_residual_target",
    }
    return BuildResult(
        ok=True,
        stage="FREEZE",
        capture=cap,
        ref=base_blob["ref"],
        capture_fp=fp,
        log=log,
    )
