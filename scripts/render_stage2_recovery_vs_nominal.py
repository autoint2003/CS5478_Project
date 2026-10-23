#!/usr/bin/env python
"""Render Stage-2 Recovery vs Direct-Nominal side-by-side videos.

Uses identical IC captures from stage2_abc_final (or mining v1/v2).
Does not alter physics / success criteria / K49.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.primitives import _wrist
from data_generation.oracles.precision.unified_precision_oracle import measure_errors
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.ground_drop import arm, restore_state as restore_drop
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from models.transformer_policy import DT_POLICY
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

FINAL = ROOT / "results/data_generation/stage2_abc_final"
V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
V2 = ROOT / "results/data_generation/stage2_abc_mining/v2"
OUT = FINAL / "videos"
SAMPLE_S = 0.04
DT = 0.02


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def resolve_ic(sample_id: str) -> Path:
    for base in (FINAL, V2, V1):
        p = base / "ics" / f"{sample_id}.pkl"
        if p.exists():
            return p
    raise FileNotFoundError(sample_id)


def resolve_actions(sample_id: str) -> np.ndarray:
    for base in (FINAL, V2, V1):
        for p in (base / "continuous").glob(f"{sample_id}*"):
            z = np.load(p)
            if "actions" in z:
                return np.asarray(z["actions"], np.float32)
    # outcome may point to continuous
    for base in (FINAL, V2, V1):
        op = base / "outcomes" / f"{sample_id}.json"
        if not op.exists():
            continue
        o = json.loads(op.read_text(encoding="utf-8"))
        rel = (o.get("continuous_demo") or {}).get("path") or o.get("continuous_npz")
        if rel:
            p = base / rel
            if p.exists():
                return np.asarray(np.load(p)["actions"], np.float32)
    return np.zeros((0, 7), np.float32)


def resolve_outcome(sample_id: str) -> dict:
    for base in (FINAL, V2, V1):
        p = base / "outcomes" / f"{sample_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def shoot(sim, cam, renderer):
    look = 0.5 * (
        np.array(sim.data.xpos[sim.ids.object_body], float)
        + np.array(sim.data.xpos[sim.ids.hand_body], float)
    )
    cam.lookat[:] = look
    renderer.update_scene(sim.data, cam)
    return np.asarray(renderer.render(), np.uint8).copy()


def overlay_text(plan, sid, branch, phase, t_sim, err, handoff, succ12, succ_new, contact):
    e_pos = err.get("e_pos_mm")
    th = err.get("dtheta_deg")
    return (
        f"{branch}\n"
        f"Plan {plan} | {sid[:28]}\n"
        f"phase={phase}  t={t_sim:.2f}s\n"
        f"e_pos={e_pos:.1f}mm  dth={th:.1f}deg\n"
        f"contact={'BIL' if contact else 'NO'}  HANDOFF={'Y' if handoff else 'N'}\n"
        f"12S={'Y' if succ12 else 'N'}  NEW={'Y' if succ_new else 'N'}"
    )


def run_branch(env, cap, ref, actions, *, label_plan, sid, branch_name, sample_every=SAMPLE_S):
    """Return frames + numeric outcome. actions empty => direct handoff."""
    restore_capture(env, cap)
    t0 = float(env.sim.data.time)
    frames = []
    phase = "recovery" if len(actions) else "direct_handoff"
    next_t = t0
    handoff_done = False
    succ12 = False
    succ_new = False

    def maybe_frame(ph):
        nonlocal next_t
        now = float(env.sim.data.time)
        if now + 1e-9 < next_t:
            return
        err = measure_errors(env.sim)
        bil = bool(err.get("bilateral"))
        img = shoot(env.sim, cam, renderer)
        frames.append(
            caption(
                img,
                overlay_text(
                    label_plan,
                    sid,
                    branch_name,
                    ph,
                    now - t0,
                    err,
                    handoff_done,
                    succ12,
                    succ_new,
                    bil,
                ),
            )
        )
        next_t = now + sample_every

    # shared renderer via closure set by caller
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        maybe_frame("recovery")

    # stabilize then handoff
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
        maybe_frame("stabilize")
    apply_handoff(env.sim, ref)
    handoff_done = True
    maybe_frame("HANDOFF")

    # 12s nominal
    n_nom = int(round(NOMINAL_S / DT))
    for i in range(n_nom):
        nominal_step(env.sim)
        maybe_frame("nominal_12s")
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    succ12 = bool(h.get("SUCCESS_12S"))
    succ_new = bool(h.get("SUCCESS_NEW"))
    # stamp final frame with outcomes
    err = measure_errors(env.sim)
    img = shoot(env.sim, cam, renderer)
    frames.append(
        caption(
            img,
            overlay_text(
                label_plan,
                sid,
                branch_name,
                "FINAL",
                float(env.sim.data.time) - t0,
                err,
                True,
                succ12,
                succ_new,
                bool(err.get("bilateral")),
            ),
        )
    )
    return frames, {
        "SUCCESS_12S": succ12,
        "SUCCESS_NEW": succ_new,
        "reason": h.get("std_reason") or h.get("reason"),
        "n_recovery_actions": int(len(actions)),
        "sim_time_s": float(env.sim.data.time) - t0,
    }


def pad_pair(left, right):
    n = max(len(left), len(right))
    if not left:
        left = [np.zeros_like(right[0])]
    if not right:
        right = [np.zeros_like(left[0])]
    while len(left) < n:
        left.append(left[-1])
    while len(right) < n:
        right.append(right[-1])
    return [np.hstack([a, b]) for a, b in zip(left, right)]


def render_case(sample_id: str, *, plan: str, tag: str, actions_override: np.ndarray | None = None):
    from data_generation.oracles.precision.unified_precision_oracle import PrecisionConfig, run_precision

    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    global cam, renderer
    pack = pickle.loads(resolve_ic(sample_id).read_bytes())
    cap, ref = pack["capture"], pack["ref"]
    acts = actions_override if actions_override is not None else resolve_actions(sample_id)
    outcome = resolve_outcome(sample_id)
    # If no saved teacher tape (oracle failures), generate live Precision actions for visualization.
    if acts.size == 0 and "direct" not in tag.lower() and "skip" not in tag.lower():
        restore_capture(env, cap)
        res = run_precision(env, {"capture": cap, "ref": ref}, plan, PrecisionConfig())
        acts = np.asarray(res.actions or [], np.float32)
        restore_capture(env, cap)

    restore_capture(env, cap)
    look = 0.5 * (
        np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -20.0}
    )
    renderer = mujoco.Renderer(env.sim.model, 360, 480)

    # Left: direct nominal
    left, left_out = run_branch(env, cap, ref, np.zeros((0, 7), np.float32), label_plan=plan, sid=sample_id, branch_name="DIRECT NOMINAL")
    # Right: recovery
    right, right_out = run_branch(env, cap, ref, acts, label_plan=plan, sid=sample_id, branch_name="RECOVERY")

    frames = pad_pair(left, right)
    dest = OUT / f"{tag}__{sample_id}__vs_nominal.mp4"
    write_frames_mp4(frames, dest, fps=25)
    meta = {
        "sample_id": sample_id,
        "plan": plan,
        "tag": tag,
        "video": str(dest.relative_to(ROOT)),
        "n_frames": len(frames),
        "n_recovery_actions": int(len(acts)),
        "direct": left_out,
        "recovery": right_out,
        "dataset_outcome": {
            "final_category": outcome.get("final_category") or outcome.get("sft_category"),
            "precision": outcome.get("precision"),
            "k49_demo": outcome.get("k49_demo"),
        },
    }
    (OUT / f"{tag}__{sample_id}__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", dest, "direct_NEW", left_out["SUCCESS_NEW"], "rec_NEW", right_out["SUCCESS_NEW"], flush=True)
    return meta


def select_default_cases() -> list[dict]:
    """Pick representative cases from mining/final manifests."""
    cases = []
    # Plan A nontrivial K49
    for p in sorted((V1 / "outcomes").glob("PILOT_A__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        k = o.get("k49_demo") or {}
        if o.get("sft_category") == "NONTRIVIAL_PRECISION" and (k.get("nontrivial_representability") or k.get("verified")):
            cases.append({"sample_id": o["sample_id"], "plan": "A", "tag": "A_nt_k49"})
            if sum(1 for c in cases if c["tag"] == "A_nt_k49") >= 2:
                break
    # Plan B K49 verified
    for p in sorted((V2 / "outcomes").glob("B__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        k = o.get("k49_demo") or {}
        if o.get("sft_category") == "NONTRIVIAL_PRECISION" and k.get("nontrivial_representability"):
            cases.append({"sample_id": o["sample_id"], "plan": "B", "tag": "B_nt_k49"})
            break
    # Plan B continuous-only / quantization fail
    for sid in (
        "B__hard_910166__post_b1__7b45f78c0ce7796e",
        "B__hard_910220__0412621ee978627d",
    ):
        if (V1 / "outcomes" / f"{sid}.json").exists() or (V2 / "outcomes" / f"{sid}.json").exists():
            cases.append({"sample_id": sid, "plan": "B", "tag": "B_k49_quant_fail_diag"})
            break
    # Plan C unresolved failure
    for p in sorted((V2 / "outcomes").glob("C__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if o.get("sft_category") == "FAILED_ORACLE":
            cases.append({"sample_id": o["sample_id"], "plan": "C", "tag": "C_unresolved_fail"})
            break
    # Direct success C (pilot)
    for p in sorted((V1 / "outcomes").glob("PILOT_C__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if o.get("sft_category") == "DIRECT_HANDOFF":
            cases.append({"sample_id": o["sample_id"], "plan": "C", "tag": "C_direct_skip"})
            break
    # Direct B
    for p in sorted((V1 / "outcomes").glob("PILOT_B__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if o.get("sft_category") == "DIRECT_HANDOFF":
            cases.append({"sample_id": o["sample_id"], "plan": "B", "tag": "B_direct_skip"})
            break
    # Contact-loss / handoff-drop style B failure if present
    for p in sorted((V2 / "outcomes").glob("B__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        fails = str((o.get("precision") or {}).get("failures") or "")
        if o.get("sft_category") == "FAILED_ORACLE" and ("DROP" in fails or "contact_loss" in fails or "dtheta" in fails):
            cases.append({"sample_id": o["sample_id"], "plan": "B", "tag": "B_oracle_fail"})
            break
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select_default_cases()
    index = []
    for c in cases:
        try:
            meta = render_case(c["sample_id"], plan=c["plan"], tag=c["tag"])
            index.append(meta)
        except Exception as e:
            index.append({"sample_id": c["sample_id"], "tag": c["tag"], "error": str(e)})
            print("FAIL", c, e, flush=True)
    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("Done", len(index), "->", OUT)


if __name__ == "__main__":
    main()
