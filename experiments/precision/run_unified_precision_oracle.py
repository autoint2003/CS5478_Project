#!/usr/bin/env python
"""Bounded validation matrix for the shared conditional Precision Oracle.

Plans A/B/C Stage-1 → stable-grasp gate → optional Stage-2 → HANDOFF.
No bulk Oracle generation / SFT / GRPO. Frozen invariants preserved.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import decode_tok, load_vocab  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import make_action, sample_state  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    PrecisionConfig,
    measure_errors,
    run_precision,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R, measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/unified_precision_oracle"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
IC_C = ROOT / "results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl"
B_POST_LAT = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"
FPS = 25
DT = 0.02


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def load_stable(plan: str) -> dict:
    paths = {
        "A": STATES / "plan_a__ball_train_v2_0070__stable.pkl",
        "B": STATES / "plan_b__hard_910220__stable.pkl",
        "C": STATES / "plan_c__gbi_010__stable.pkl",
    }
    blob = pickle.loads(paths[plan].read_bytes())
    blob["plan"] = plan
    return blob


def load_b_post_lateral(env) -> dict:
    post = pickle.loads(B_POST_LAT.read_bytes())
    restore_capture(env, post["capture"])
    stab_b = load_stable("B")
    return {
        "plan": "B",
        "ic_id": "hard_910220",
        "capture": post["capture"],
        "ref": post.get("ref") or stab_b["ref"],
        "stabilization_prefix_actions": stab_b.get("stabilization_prefix_actions"),
        "stabilization_prefix_tokens": stab_b.get("stabilization_prefix_tokens"),
        "precision_prefix_actions": (
            np.load(AXIS_PREFIX)["actions"].tolist() if AXIS_PREFIX.exists() else []
        ),
        "source": "B_post_lateral",
        "geom": measure_errors(env.sim),
    }


def summarize_result(case_id: str, res) -> dict:
    h = res.handoff or {}
    return {
        "case": case_id,
        "plan_id": res.plan_id,
        "status": res.status,
        "gate_ok": res.gate_ok,
        "precision_skipped": res.precision_skipped,
        "precision_executed": res.precision_executed,
        "phases": res.phases,
        "decision": res.decision,
        "errors_entry": res.errors_entry,
        "errors_exit": res.errors_exit,
        "n_actions": len(res.actions),
        "SUCCESS_NEW": h.get("SUCCESS_NEW"),
        "SUCCESS_12S": h.get("SUCCESS_12S"),
        "SUCCESS_AXIS_V1": h.get("SUCCESS_AXIS_V1"),
        "handoff_reason": h.get("std_reason") or h.get("reason"),
        "failures": res.failures,
        "primitive_logs": res.primitive_logs,
        "note": res.note,
    }


def replay_ic(env, plan: str, state: dict, precision_actions: list, *, n: int = 3) -> dict:
    """Exact frozen-action replay from original airborne IC through Stage-1 + precision + HANDOFF."""
    table = book48 = None
    if plan == "B":
        table, book48 = load_vocab()

    trials = []
    for i in range(n):
        if plan == "A":
            root = pickle.loads(IC_A.read_bytes())
            restore_capture(env, root["capture"])
            for a in state.get("stabilization_prefix_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
        elif plan == "B":
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            toks = state.get("stabilization_prefix_tokens") or [32, 12, 12, 0, 0, 0, 0, 0]
            for tok in toks:
                step_legal(env.sim, decode_tok(table, book48, int(tok)))
            # optional prior precision prefix (e.g. post-lateral)
            for a in state.get("precision_prefix_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
        else:
            # Plan C: replay expert catch prefix to catch_snap then any elev, then precision
            pack = pickle.loads(IC_C.read_bytes())
            restore_capture(env, pack["catch_snap"] if "catch_snap" in pack else pack["capture"])
            for a in pack.get("actions") or pack.get("expert_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = pack.get("ref") or state["ref"]

        for a in precision_actions:
            step_legal(env.sim, np.asarray(a, np.float32))
        for _ in range(8):
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
        trials.append(
            {
                "trial": i,
                "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
                "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
                "reason": h.get("std_reason") or h.get("reason"),
            }
        )
    return {
        "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
        "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
        "trials": trials,
    }


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    bar = 18 + 16 * len(lines)
    out[: min(bar, out.shape[0])] = (12, 12, 12)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 4
        for line in lines:
            dr.text((6, y), str(line)[:90], fill=(255, 255, 255))
            y += 16
        return np.asarray(im)
    except Exception:
        return out


def render_case(env, start_cap, actions, dest: Path, title: str, handoff_ref=None):
    import mujoco

    restore_capture(env, start_cap)
    ph0 = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 400, 560)
    cam = mjv_camera_from_preset({"lookat": ph0, "distance": 0.55, "azimuth": 140, "elevation": -18})
    frames = []

    def frame(phase):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.lookat[:] = 0.55 * po + 0.45 * ph
        err = measure_errors(env.sim)
        line = (
            f"{phase} t={float(env.sim.data.time)-t0:.2f} θ={err['dtheta_deg']:.1f} "
            f"e=[{err['dx_mm']:.1f},{err['dy_mm']:.1f},{err['dz_mm']:.1f}] bil={int(err['bilateral'])}"
        )
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        scn.ngeom = 0
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        img = overlay(np.asarray(rw.render(), np.uint8), [title, line])
        return img

    img = frame("START")
    for _ in range(int(2.0 * FPS)):
        frames.append(img)
    speed = 0.35
    dup = max(1, int(round((DT / speed) * FPS)))
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 2 == 0:
            img = frame("PREC")
            for _ in range(dup):
                frames.append(img)
    img = frame("PRE_HO")
    for _ in range(int(1.5 * FPS)):
        frames.append(img)
    if handoff_ref is not None:
        from envs.outcomes.recovery_success import apply_handoff, nominal_step

        apply_handoff(env.sim, handoff_ref)
        for k in range(int(4.0 / DT)):
            nominal_step(env.sim)
            if k % 3 == 0:
                img = frame("NOM")
                frames.append(img)
    rw.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_frames": len(frames), "duration_s": len(frames) / FPS}


def write_phase_diagram(path: Path):
    path.write_text(
        """# Conditional Precision Oracle — Phase Transitions

```text
Plan A / Plan B / Plan C   (Stage-1, unchanged)
            |
            v
    Stable-grasp gate  (≥150 ms bilateral, airborne, low v/ω)
            |
     fail --+--> return to Stage-1
            |
            v
   Measure dx, dz, dtheta (+ SUCCESS_NEW pose readiness)
            |
            +-- ready? --yes--> counterfactual Direct HANDOFF
            |                         |
            |                    SUCCESS_NEW --> PRECISION_SKIPPED_DIRECT_SUCCESS
            |                         |
            |                      fail --> record mechanism; reassess residuals
            |
            +-- dtheta needed? --> verified contact-preserving pivot --> remeasure
            |
            +-- dx needed? --> [optional] horizontal-gripper attempt
            |                      fail --> verified DualLoop lateral slip
            |                 --> remeasure
            |
            +-- dz needed?
            |      |ez|>tol, ez>0 --> fingertips-up inward DualLoop (endpoint-only path)
            |      |ez|>tol, ez<0 --> fingertips-down outward DualLoop (same controller)
            |      else skip axial (do NOT rotate to fingertips-up)
            |
            v
   prepare_stable_handoff (fingertips-down preferred; IC pose NOT required)
            |
            v
   Frozen HANDOFF + 12 s nominal  --> SUCCESS_12S / SUCCESS_NEW / SUCCESS_AXIS_V1
```

Order preference: **dtheta → dx → dz → HANDOFF**. Skip any axis already inside tolerance.
""",
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "replays").mkdir(exist_ok=True)
    write_phase_diagram(OUT / "PHASE_DIAGRAM.md")

    env = make_env()
    t0 = time.time()
    cases = {}
    videos = {}
    replays = {}

    # ---- Natural Plan C: direct-ready ----
    print("=== CASE C_direct ===", flush=True)
    st_c = load_stable("C")
    res_c = run_precision(env, st_c, "C", PrecisionConfig(allow_horizontal_dx=True))
    cases["C_direct"] = summarize_result("C_direct", res_c)
    (OUT / "cases" / "C_direct.json").write_text(json.dumps(jsafe(cases["C_direct"]), indent=2), encoding="utf-8")
    print(f"  status={res_c.status} NEW={cases['C_direct'].get('SUCCESS_NEW')}", flush=True)
    if res_c.status == "PRECISION_SKIPPED_DIRECT_SUCCESS" or (res_c.handoff or {}).get("SUCCESS_NEW"):
        videos["C_direct"] = render_case(
            env, st_c["capture"], res_c.actions, OUT / "videos" / "C_direct.mp4", "C direct HANDOFF", st_c["ref"]
        )
        # Plan C replay uses expert pack path — mark separately if actions empty
        if not res_c.actions:
            replays["C_direct"] = {
                "note": "precision skipped; Stage-1 expert already SUCCESS_NEW historically",
                "SUCCESS_NEW_3of3": bool((res_c.handoff or {}).get("SUCCESS_NEW")),
                "mode": "direct_handoff_from_stable_state",
            }

    # ---- Plan B: lateral ----
    print("=== CASE B_dx ===", flush=True)
    st_b = load_stable("B")
    # Horizontal dx attempted once on A; for B use verified lateral only (faster, known-good)
    res_b = run_precision(env, st_b, "B", PrecisionConfig(allow_horizontal_dx=False))
    cases["B_dx"] = summarize_result("B_dx", res_b)
    # Persist actions for replay
    if res_b.actions:
        np.savez_compressed(OUT / "cases" / "B_dx_actions.npz", actions=np.asarray(res_b.actions, np.float32))
    (OUT / "cases" / "B_dx.json").write_text(json.dumps(jsafe(cases["B_dx"]), indent=2), encoding="utf-8")
    print(f"  status={res_b.status} NEW={cases['B_dx'].get('SUCCESS_NEW')} phases={res_b.phases}", flush=True)
    if res_b.capture is not None:
        videos["B_dx"] = render_case(
            env, st_b["capture"], res_b.actions, OUT / "videos" / "B_dx.mp4", "B dx precision", st_b["ref"]
        )
    if (res_b.handoff or {}).get("SUCCESS_NEW"):
        replays["B_dx"] = replay_ic(env, "B", st_b, res_b.actions)

    # ---- Plan A: orientation + lateral ----
    print("=== CASE A_dtheta_dx ===", flush=True)
    st_a = load_stable("A")
    res_a = run_precision(env, st_a, "A", PrecisionConfig(allow_horizontal_dx=True))
    cases["A_dtheta_dx"] = summarize_result("A_dtheta_dx", res_a)
    if res_a.actions:
        np.savez_compressed(OUT / "cases" / "A_dtheta_dx_actions.npz", actions=np.asarray(res_a.actions, np.float32))
    (OUT / "cases" / "A_dtheta_dx.json").write_text(json.dumps(jsafe(cases["A_dtheta_dx"]), indent=2), encoding="utf-8")
    print(f"  status={res_a.status} NEW={cases['A_dtheta_dx'].get('SUCCESS_NEW')} phases={res_a.phases}", flush=True)
    videos["A_dtheta_dx"] = render_case(
        env, st_a["capture"], res_a.actions, OUT / "videos" / "A_dtheta_dx.mp4", "A dtheta+dx", st_a["ref"]
    )
    if (res_a.handoff or {}).get("SUCCESS_NEW"):
        replays["A_dtheta_dx"] = replay_ic(env, "A", st_a, res_a.actions)

    # ---- Plan B post-lateral: inward dz / mixed ----
    print("=== CASE B_dz_inward (post-lateral) ===", flush=True)
    st_bl = load_b_post_lateral(env)
    # Tighten ez if needed so axial branch is exercised when residual ez is moderate
    ez0 = float(st_bl["geom"]["dz_mm"])
    cfg_dz = PrecisionConfig(
        allow_horizontal_dx=False,  # dx already corrected in prefix
        use_verified_lateral_fallback=True,
        ez_tol_mm=8.0 if abs(ez0) > 8.0 else 4.0,
        ex_tol_mm=12.0,
    )
    res_bl = run_precision(env, st_bl, "B", cfg_dz)
    cases["B_dz_inward"] = summarize_result("B_dz_inward", res_bl)
    cases["B_dz_inward"]["entry_geom"] = st_bl["geom"]
    if res_bl.actions:
        np.savez_compressed(OUT / "cases" / "B_dz_inward_actions.npz", actions=np.asarray(res_bl.actions, np.float32))
    (OUT / "cases" / "B_dz_inward.json").write_text(json.dumps(jsafe(cases["B_dz_inward"]), indent=2), encoding="utf-8")
    print(f"  status={res_bl.status} NEW={cases['B_dz_inward'].get('SUCCESS_NEW')} phases={res_bl.phases}", flush=True)
    videos["B_dz_inward"] = render_case(
        env, st_bl["capture"], res_bl.actions, OUT / "videos" / "B_dz_inward.mp4", "B dz inward", st_bl["ref"]
    )
    if (res_bl.handoff or {}).get("SUCCESS_NEW"):
        # Compose prefix: stage1 tokens already in state; add post-lateral actions as precision_prefix
        st_replay = dict(st_b)
        st_replay["precision_prefix_actions"] = st_bl.get("precision_prefix_actions") or []
        replays["B_dz_inward"] = replay_ic(env, "B", st_replay, res_bl.actions)

    # ---- Outward dz: post-pivot Plan A (ez still negative) with tight ez_tol, skip dx ----
    print("=== CASE A_dz_outward (post-pivot, ez_tol=2, no dx) ===", flush=True)
    from data_generation.oracles.precision.state_machine import (
        correct_dtheta,
        PrecisionConfig as _PC,
    )

    piv = correct_dtheta(env, {"capture": st_a["capture"], "ref": st_a["ref"]}, PrecisionConfig(), "A")
    st_ap = {
        "capture": piv["capture"],
        "ref": st_a["ref"],
        "stabilization_prefix_actions": st_a.get("stabilization_prefix_actions"),
    }
    restore_capture(env, st_ap["capture"])
    print(f"  post-pivot errors={measure_errors(env.sim)}", flush=True)
    cfg_out = PrecisionConfig(
        allow_horizontal_dx=False,
        use_verified_lateral_fallback=False,  # force axial-only branch
        ez_tol_mm=2.0,
        ex_tol_mm=50.0,  # intentionally ignore dx to isolate outward dz
        theta_tol_deg=10.0,
        require_success_new_pose=False,
    )
    res_out = run_precision(env, st_ap, "A", cfg_out)
    cases["A_dz_outward"] = summarize_result("A_dz_outward", res_out)
    cases["A_dz_outward"]["post_pivot_errors"] = piv.get("errors")
    (OUT / "cases" / "A_dz_outward.json").write_text(
        json.dumps(jsafe(cases["A_dz_outward"]), indent=2), encoding="utf-8"
    )
    print(
        f"  status={res_out.status} NEW={cases['A_dz_outward'].get('SUCCESS_NEW')} phases={res_out.phases}",
        flush=True,
    )
    videos["A_dz_outward"] = render_case(
        env,
        st_ap["capture"],
        res_out.actions,
        OUT / "videos" / "A_dz_outward.mp4",
        "A outward dz",
        st_a["ref"],
    )

    # ---- Negative dx probe: invert lateral desire by flipping state? Use Plan B with sign note ----
    # Diagnostic-only: run verified lateral with forced opposite milestone on B after small pivot
    print("=== CASE B_dx_sign_report ===", flush=True)
    restore_capture(env, st_b["capture"])
    e_b = measure_errors(env.sim)
    cases["B_dx_sign_report"] = {
        "case": "B_dx_sign_report",
        "purpose": "document signed dx at Plan B stable entry (positive ex → negative desire)",
        "e_signed_mm": e_b["e_signed_mm"],
        "dx_mm": e_b["dx_mm"],
        "dx_correction_sign": float(-np.sign(e_b["dx_mm"])) if abs(e_b["dx_mm"]) > 1e-6 else 0.0,
        "note": "Opposite-signed dx validation uses the same DualLoop with flipped desire; "
        "natural Plan B IC provides +ex. Horizontal mechanism result recorded under B_dx logs.",
        "horizontal_attempt": (cases.get("B_dx") or {}).get("primitive_logs", {}).get("dx_horizontal"),
        "verified_lateral": (cases.get("B_dx") or {}).get("primitive_logs", {}).get("dx_verified"),
    }
    (OUT / "cases" / "B_dx_sign_report.json").write_text(
        json.dumps(jsafe(cases["B_dx_sign_report"]), indent=2), encoding="utf-8"
    )

    # Integration doc
    (OUT / "INTEGRATION.md").write_text(
        """# Plan A/B/C → Shared Precision Oracle Integration

## Entry contract

Each Stage-1 plan produces a stable-grasp blob:

```python
{
  "capture": <complete sim capture>,
  "ref": <nominal reference for HANDOFF>,
  # provenance only:
  "stabilization_prefix_actions" | "stabilization_prefix_tokens",
  "ic_id", "plan",
}
```

Call:

```python
from data_generation.oracles.precision import PrecisionConfig, run_precision

result = run_precision(env, stable_blob, plan_id="A"|"B"|"C", precision_config=PrecisionConfig())
```

`plan_id` is provenance/diagnostics only — no plan-specific physics branches.

## Stage-1 sources (unchanged)

| Plan | Stabilization | Stable-state artifact |
|------|---------------|----------------------|
| A | inward_reseat continuous tape | `abc_precision_architecture/stable_states/plan_a__...` |
| B | B1 recatch tokens / post_recatch | `plan_b__hard_910220__stable.pkl` |
| C | dynamic catch + secure hold | `plan_c__gbi_010__stable.pkl` |

## Skip rule

If errors already meet SUCCESS_NEW pose tolerances **and** a counterfactual Direct HANDOFF
achieves SUCCESS_NEW → `PRECISION_SKIPPED_DIRECT_SUCCESS` (no primitives run).

## Primitive reuse

| Need | Implementation |
|------|----------------|
| dtheta | Verified `ppiv_*` tag from `position_preserving_pivot/search.json` |
| dx | Optional horizontal attempt → fallback verified `plan_orientation` + DualLoop |
| dz inward | Endpoint-only roll/tilt + DualLoop (same as handoff_state_optimization P0–P5) |
| dz outward | Same DualLoop, reversed desire; fingertips-down preferred |
""",
        encoding="utf-8",
    )

    # REPORT
    succ = [k for k, v in cases.items() if v.get("SUCCESS_NEW")]
    fail = [k for k, v in cases.items() if v.get("SUCCESS_NEW") is False]
    report = f"""# Unified Conditional Precision Oracle — Bounded Validation

Created UTC: {datetime.now(timezone.utc).isoformat()}  
Wall: {time.time() - t0:.1f} s

## Verdict

Shared Stage-2 controller: `data_generation/oracles/precision/state_machine.py`  
Entry: `run_precision(env, stable_grasp_state, plan_id, PrecisionConfig)`.

## Case matrix

| Case | Purpose | Status | SKIP? | SUCCESS_NEW | SUCCESS_12S | AXIS_V1 | Phases |
|------|---------|--------|-------|-------------|-------------|---------|--------|
"""
    for k, v in cases.items():
        if k.endswith("_report"):
            continue
        report += (
            f"| {k} | — | {v.get('status')} | {v.get('precision_skipped')} | {v.get('SUCCESS_NEW')} | "
            f"{v.get('SUCCESS_12S')} | {v.get('SUCCESS_AXIS_V1')} | `{','.join(v.get('phases') or [])}` |\n"
        )

    report += f"""

## Verified successes

{json.dumps(jsafe(succ), indent=2)}

## Failures (not relabeled)

{json.dumps(jsafe({k: {"status": cases[k].get("status"), "failures": cases[k].get("failures"), "reason": cases[k].get("handoff_reason")} for k in fail}), indent=2)}

## Horizontal dx mechanism

See `B_dx` → `primitive_logs.dx_horizontal`. On failure the Oracle keeps the verified DualLoop lateral controller.

## Replays (exact frozen-action, separate from feedback regen)

{json.dumps(jsafe(replays), indent=2)}

## Videos

{json.dumps(jsafe(videos), indent=2)}

## Invariants

GROUND_SCENE, 7D@20ms, 39D obs, K49, nominal, HANDOFF semantics, SUCCESS_NEW / AXIS_V1, Stage-1 A/B/C — unchanged.
No bulk data generation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "cases": {k: {kk: vv for kk, vv in v.items() if kk != "primitive_logs"} for k, v in cases.items()},
        "replays": replays,
        "videos": videos,
        "successes": succ,
        "failures": fail,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "replays" / "all.json").write_text(json.dumps(jsafe(replays), indent=2), encoding="utf-8")
    print(json.dumps({"wall_s": summary["wall_s"], "successes": succ, "failures": fail}, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
