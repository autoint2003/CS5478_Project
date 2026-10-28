"""D2 teleport IC builder.

Pipeline (construction only):
  TELEPORT_CONSTRUCTION → SETTLE → VALIDATE → FREEZE_IC

After FREEZE_IC the capture is immutable for the expert: no qpos rewrite,
no privileged forces. Teleport metadata is not a learned skill.
"""

from __future__ import annotations

import hashlib
import pickle
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
from evaluation.grasp_geometry import measure
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


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        float,
    )


def _penetration_stats(sim) -> dict[str, float]:
    """Negative contact distance = penetration depth (m)."""
    depths = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        depths.append(float(c.dist))
    if not depths:
        return {"ncon": 0, "min_dist": 0.0, "max_penetration_m": 0.0}
    arr = np.array(depths, float)
    pen = arr[arr < 0.0]
    return {
        "ncon": int(len(arr)),
        "min_dist": float(arr.min()),
        "max_penetration_m": float((-pen).max()) if len(pen) else 0.0,
    }


@dataclass
class TeleportSpec:
    """Commanded construction offsets.

    ``dx_mm`` / ``dz_mm`` / ``dy_mm`` are **hand-frame** translations (mm),
    matching Precision residual axes (not world-frame).
    ``s_axial_mm`` is additional offset along the object long axis (object frame).
    ``dtheta_deg`` rotates about the hand open axis (hand +Y).
    ``finger_open_delta`` adds to both finger joint positions (m), if set.
    """

    family: str  # D2-DX | D2-DZ | D2-DX-DZ | D2-DZ-DX
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    s_axial_mm: float = 0.0
    dtheta_deg: float = 0.0
    finger_open_delta: float | None = None
    settle_n: int = 16
    max_penetration_m: float = 0.004  # reject deeper than 4 mm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeleportBuildResult:
    ok: bool
    stage: str
    reason: str | None = None
    capture: Any = None
    ref: Any = None
    capture_fp: str | None = None
    lineage_key: str | None = None
    construction: dict = field(default_factory=dict)
    settled_errors: dict = field(default_factory=dict)
    contact: dict = field(default_factory=dict)
    penetration: dict = field(default_factory=dict)
    need: dict = field(default_factory=dict)
    fsm_case: str | None = None
    fsm_planned: list[str] = field(default_factory=list)
    precision_required: bool | None = None

    def to_log_row(self) -> dict[str, Any]:
        se = self.settled_errors or {}
        return {
            "ok": self.ok,
            "stage": self.stage,
            "reason": self.reason,
            "capture_fp": self.capture_fp,
            "lineage_key": self.lineage_key,
            "family": (self.construction or {}).get("family"),
            "cmd_dx_mm": (self.construction or {}).get("cmd_dx_mm"),
            "cmd_dz_mm": (self.construction or {}).get("cmd_dz_mm"),
            "cmd_dy_mm": (self.construction or {}).get("cmd_dy_mm"),
            "cmd_s_axial_mm": (self.construction or {}).get("cmd_s_axial_mm"),
            "cmd_dtheta_deg": (self.construction or {}).get("cmd_dtheta_deg"),
            "settled_dx_mm": se.get("dx_mm"),
            "settled_dz_mm": se.get("dz_mm"),
            "settled_dy_mm": se.get("dy_mm"),
            "settled_dtheta_deg": se.get("dtheta_deg"),
            "settled_e_pos_mm": se.get("e_pos_mm"),
            "bilateral": (self.contact or {}).get("bilateral"),
            "nL": (self.contact or {}).get("nL"),
            "nR": (self.contact or {}).get("nR"),
            "max_penetration_m": (self.penetration or {}).get("max_penetration_m"),
            "precision_required": self.precision_required,
            "fsm_case": self.fsm_case,
            "fsm_planned": "|".join(self.fsm_planned or []),
            "base_sample_id": (self.construction or {}).get("base_sample_id"),
        }


def lineage_fingerprint(
    *,
    capture_fp: str,
    settled_dx_mm: float,
    settled_dz_mm: float,
    settled_dtheta_deg: float,
    quant_mm: float = 1.0,
    quant_deg: float = 1.0,
) -> str:
    """Collapse near-duplicate settled ICs (same physical niche)."""
    qx = round(float(settled_dx_mm) / quant_mm) * quant_mm
    qz = round(float(settled_dz_mm) / quant_mm) * quant_mm
    qt = round(float(settled_dtheta_deg) / quant_deg) * quant_deg
    raw = f"{capture_fp}|dx={qx:.1f}|dz={qz:.1f}|th={qt:.1f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_teleport_ic(
    env,
    base_blob: dict,
    spec: TeleportSpec,
    *,
    cfg: PrecisionConfig | None = None,
) -> TeleportBuildResult:
    """Construct → settle → validate → freeze. Teleport only before FREEZE."""
    pcfg = cfg or PrecisionConfig()
    if "capture" not in base_blob or "ref" not in base_blob:
        return TeleportBuildResult(ok=False, stage="TELEPORT_CONSTRUCTION", reason="MISSING_BASE_CAPTURE")

    restore_capture(env, base_blob["capture"])
    sim = env.sim
    mujoco.mj_forward(sim.model, sim.data)
    cs0 = contact_summary(sim)
    if not cs0.get("bilateral"):
        return TeleportBuildResult(ok=False, stage="TELEPORT_CONSTRUCTION", reason="BASE_NOT_BILATERAL")

    adr = _object_qpos_adr(sim)
    if adr is None:
        return TeleportBuildResult(ok=False, stage="TELEPORT_CONSTRUCTION", reason="NO_OBJECT_FREE_JOINT")

    # --- TELEPORT_CONSTRUCTION ---
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float).copy()

    # Hand-frame residual offsets (explicitly not world-frame).
    offset_hand = np.array([spec.dx_mm, spec.dy_mm, spec.dz_mm], float) * 1e-3
    # Object long-axis (cylinder axis ≈ object +Z in this model).
    offset_axial = Ro[:, 2] * (float(spec.s_axial_mm) * 1e-3)
    po_new = po + Rh @ offset_hand + offset_axial

    qpos = np.array(sim.data.qpos, float).copy()
    qpos[adr : adr + 3] = po_new

    if abs(spec.dtheta_deg) > 1e-9:
        axis_w = Rh[:, 1]
        axis_w = axis_w / max(float(np.linalg.norm(axis_w)), 1e-12)
        half = 0.5 * np.deg2rad(float(spec.dtheta_deg))
        q_delta = np.array(
            [np.cos(half), *(axis_w * np.sin(half))],
            float,
        )
        q_old = qpos[adr + 3 : adr + 7].copy()
        q_new = _quat_mul(q_delta, q_old)
        q_new /= max(float(np.linalg.norm(q_new)), 1e-12)
        qpos[adr + 3 : adr + 7] = q_new

    if spec.finger_open_delta is not None:
        for j in sim.ids.finger_jnt:
            qa = int(sim.model.jnt_qposadr[int(j)])
            qpos[qa] = float(qpos[qa]) + float(spec.finger_open_delta)

    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)

    pen0 = _penetration_stats(sim)
    if float(pen0["max_penetration_m"]) > float(spec.max_penetration_m):
        return TeleportBuildResult(
            ok=False,
            stage="TELEPORT_CONSTRUCTION",
            reason=f"SEVERE_PENETRATION:{pen0['max_penetration_m']:.4f}",
            penetration=pen0,
            construction=_meta(spec, base_blob),
        )

    # --- SETTLE (legal control only) ---
    settle_grip(env, n=int(spec.settle_n))
    # brief zero-wrist hold for contact topology
    for _ in range(4):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(sim, a)
    mujoco.mj_forward(sim.model, sim.data)

    # --- VALIDATE ---
    cs1 = contact_summary(sim)
    pen1 = _penetration_stats(sim)
    if not cs1.get("bilateral"):
        return TeleportBuildResult(
            ok=False,
            stage="VALIDATE",
            reason="LOST_BILATERAL_AFTER_SETTLE",
            contact=cs1,
            penetration=pen1,
            construction=_meta(spec, base_blob),
        )
    if float(pen1["max_penetration_m"]) > float(spec.max_penetration_m):
        return TeleportBuildResult(
            ok=False,
            stage="VALIDATE",
            reason=f"PENETRATION_AFTER_SETTLE:{pen1['max_penetration_m']:.4f}",
            contact=cs1,
            penetration=pen1,
            construction=_meta(spec, base_blob),
        )
    m = measure(sim, U_TGT, P_TGT)
    if float(m["obj_z"]) < 0.20:
        return TeleportBuildResult(
            ok=False,
            stage="VALIDATE",
            reason="OBJECT_TOO_LOW",
            contact=cs1,
            penetration=pen1,
            construction=_meta(spec, base_blob),
        )
    err = measure_errors(sim)
    if float(err.get("v_rel_norm") or 0) > 0.25:
        return TeleportBuildResult(
            ok=False,
            stage="VALIDATE",
            reason=f"HIGH_SLIP:{err.get('v_rel_norm')}",
            settled_errors=err,
            contact=cs1,
            penetration=pen1,
            construction=_meta(spec, base_blob),
        )

    need = assess_need(err, pcfg)
    plan = classify_case(err, pcfg)
    prec_req = bool(
        need.get("dx_needed")
        or need.get("dz_needed")
        or need.get("dtheta_needed")
        or not need.get("ready_for_direct_handoff_assessment")
    )

    # --- FREEZE_IC ---
    cap = take_capture(env)
    fp = capture_fingerprint(cap)
    lin = lineage_fingerprint(
        capture_fp=fp,
        settled_dx_mm=float(err.get("dx_mm") or 0),
        settled_dz_mm=float(err.get("dz_mm") or 0),
        settled_dtheta_deg=float(err.get("dtheta_deg") or 0),
    )
    meta = _meta(spec, base_blob)
    meta.update(
        {
            "frame": "hand_frame_dx_dy_dz_plus_object_axial",
            "pipeline": "TELEPORT→SETTLE→VALIDATE→FREEZE_IC",
            "teleport_allowed_after_freeze": False,
        }
    )
    return TeleportBuildResult(
        ok=True,
        stage="FREEZE_IC",
        reason=None,
        capture=cap,
        ref=base_blob["ref"],
        capture_fp=fp,
        lineage_key=lin,
        construction=meta,
        settled_errors=dict(err),
        contact={
            "nL": int(cs1.get("nL") or 0),
            "nR": int(cs1.get("nR") or 0),
            "bilateral": bool(cs1.get("bilateral")),
        },
        penetration=pen1,
        need={k: need[k] for k in need if k != "axis_decision"},
        fsm_case=plan.case,
        fsm_planned=list(plan.planned_skills),
        precision_required=prec_req,
    )


def _meta(spec: TeleportSpec, base_blob: dict) -> dict[str, Any]:
    return {
        "family": spec.family,
        "cmd_dx_mm": float(spec.dx_mm),
        "cmd_dy_mm": float(spec.dy_mm),
        "cmd_dz_mm": float(spec.dz_mm),
        "cmd_s_axial_mm": float(spec.s_axial_mm),
        "cmd_dtheta_deg": float(spec.dtheta_deg),
        "finger_open_delta": spec.finger_open_delta,
        "settle_n": int(spec.settle_n),
        "base_sample_id": base_blob.get("sample_id"),
        "synthetic_initial_condition": True,
        "construction_method": "teleport_qpos_pre_freeze",
    }


def frozen_ic_blob(result: TeleportBuildResult) -> dict[str, Any]:
    """Expert-start payload. Construction history is metadata only."""
    if not result.ok:
        raise ValueError("cannot freeze failed build")
    return {
        "capture": result.capture,
        "ref": result.ref,
        "sample_id": f"TELEPORT__{result.lineage_key}",
        "construction": result.construction,
        "capture_fp": result.capture_fp,
        "lineage_key": result.lineage_key,
        "settled_errors": result.settled_errors,
        "synthetic_initial_condition": True,
    }


def clone_capture(capture: Any) -> Any:
    """Deep copy for independent 12S verification without advancing the IC."""
    return pickle.loads(pickle.dumps(capture, protocol=4))
