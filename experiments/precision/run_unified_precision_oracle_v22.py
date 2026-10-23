#!/usr/bin/env python
"""v2.2: Goal-driven closed-loop Precision Oracle validation.

Results → results/diagnostics/unified_precision_oracle_v22/  (never overwrite v21)
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
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    PrecisionConfig,
    audit_action_capture_replay,
    correct_dx_horizontal_attempt,
    correct_dz_inward,
    correct_dz_outward,
    measure_errors,
    run_precision,
    search_p1_p2_horizontal,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/unified_precision_oracle_v22"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
POST_P5 = ROOT / "results/diagnostics/handoff_state_optimization/snapshots/post_P5.pkl"
B_POST_LAT = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"
FPS, DT = 25, 0.02


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
    p = {
        "A": STATES / "plan_a__ball_train_v2_0070__stable.pkl",
        "B": STATES / "plan_b__hard_910220__stable.pkl",
        "C": STATES / "plan_c__gbi_010__stable.pkl",
    }[plan]
    blob = pickle.loads(p.read_bytes())
    blob["plan"] = plan
    return blob


def slim(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("actions", "capture", "series")}


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    out[: min(18 + 16 * len(lines), out.shape[0])] = (12, 12, 12)
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


def render(env, start_cap, actions, dest: Path, title: str, ref=None):
    import mujoco
    from envs.outcomes.recovery_success import apply_handoff, nominal_step

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
            f"{phase} t={float(env.sim.data.time)-t0:.2f} "
            f"e=[{err['dx_mm']:.1f},{err['dz_mm']:.1f}] pos={err['e_pos_mm']:.1f} th={err['dtheta_deg']:.1f}"
        )
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        scn.ngeom = 0
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), [title, line])

    img = frame("START")
    for _ in range(int(1.0 * FPS)):
        frames.append(img)
    dup = max(1, int(round((DT / 0.4) * FPS)))
    for i, a in enumerate(actions or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 2 == 0:
            img = frame("RUN")
            for _ in range(dup):
                frames.append(img)
    if ref is not None:
        apply_handoff(env.sim, ref)
        for k in range(int(3.0 / DT)):
            nominal_step(env.sim)
            if k % 4 == 0:
                frames.append(frame("NOM"))
    rw.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_frames": len(frames)}


def replay_ic(env, plan: str, state: dict, precision_actions: list) -> dict:
    table = book48 = None
    if plan == "B":
        table, book48 = load_vocab()
    trials = []
    for i in range(3):
        if plan == "A":
            root = pickle.loads(IC_A.read_bytes())
            restore_capture(env, root["capture"])
            for a in state.get("stabilization_prefix_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
        elif plan == "B":
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for tok in state.get("stabilization_prefix_tokens") or [32, 12, 12, 0, 0, 0, 0, 0]:
                step_legal(env.sim, decode_tok(table, book48, int(tok)))
            for a in state.get("precision_prefix_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
        else:
            pack = pickle.loads(
                (ROOT / "results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl").read_bytes()
            )
            restore_capture(env, pack.get("catch_snap") or state["capture"])
            for a in pack.get("actions") or pack.get("expert_actions") or []:
                if isinstance(a, np.ndarray) or isinstance(a, (list, tuple)):
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
        "SUCCESS_12S_3of3": all(t["SUCCESS_12S"] for t in trials),
        "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
        "trials": trials,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "videos", "audits", "replays"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    t0 = time.time()
    cases, videos, audits, replays = {}, {}, {}, {}
    st_a, st_b, st_c = load_stable("A"), load_stable("B"), load_stable("C")

    # A. Direct success (Plan C)
    print("=== A direct / reg_C ===", flush=True)
    res_c = run_precision(env, st_c, "C", PrecisionConfig(allow_horizontal_dx=False))
    cases["A_direct_C"] = {
        "status": res_c.status,
        "SUCCESS_NEW": (res_c.handoff or {}).get("SUCCESS_NEW"),
        "precision_skipped": res_c.precision_skipped,
        "phases": res_c.phases,
        "n_actions": len(res_c.actions),
    }
    print(f"  {res_c.status} NEW={(res_c.handoff or {}).get('SUCCESS_NEW')} skip={res_c.precision_skipped}", flush=True)

    # B. Pivot-sufficient / reg_A
    print("=== B pivot+dx Plan A ===", flush=True)
    res_a = run_precision(env, st_a, "A", PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False))
    cases["B_reg_A"] = {
        "status": res_a.status,
        "SUCCESS_NEW": (res_a.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_a.phases,
        "dx_mechanism": (res_a.primitive_logs or {}).get("dx_mechanism"),
        "cycles": (res_a.primitive_logs or {}).get("cycles"),
        "n_actions": len(res_a.actions),
    }
    if (res_a.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_A"] = replay_ic(env, "A", st_a, res_a.actions)
        videos["reg_A"] = render(env, st_a["capture"], res_a.actions, OUT / "videos" / "reg_A.mp4", "reg A", st_a["ref"])
        audits["reg_A"] = audit_action_capture_replay(env, st_a["capture"], res_a.actions, res_a.capture)
    print(f"  {res_a.status} NEW={(res_a.handoff or {}).get('SUCCESS_NEW')}", flush=True)

    # C. Plan B +dx partial then dz (goal-driven + P1/P2 search)
    print("=== C Plan B +dx then dz ===", flush=True)
    blob_b = {"capture": st_b["capture"], "ref": st_b["ref"]}
    search = search_p1_p2_horizontal(env, blob_b)
    win = search.get("winner_params") if search.get("ok") else None
    cases["C_P1_P2_search"] = {
        "ok": search.get("ok"),
        "winner_params": win,
        "n_trials": search.get("n_trials"),
        "limiting": search.get("limiting"),
    }
    cfg_b = PrecisionConfig(
        allow_horizontal_dx=True,
        use_p1_p2_search=True,
        p1_p2_winner=win,
        accept_partial_horizontal_dx=True,
        use_goal_driven_slip=True,
    )
    res_b_goal = run_precision(env, st_b, "B", cfg_b)
    cases["C_B_partial_dx_dz"] = {
        "status": res_b_goal.status,
        "SUCCESS_NEW": (res_b_goal.handoff or {}).get("SUCCESS_NEW"),
        "SUCCESS_12S": (res_b_goal.handoff or {}).get("SUCCESS_12S"),
        "SUCCESS_AXIS_V1": (res_b_goal.handoff or {}).get("SUCCESS_AXIS_V1"),
        "phases": res_b_goal.phases,
        "dx_mechanism": (res_b_goal.primitive_logs or {}).get("dx_mechanism"),
        "dx_partial": (res_b_goal.primitive_logs or {}).get("dx_partial"),
        "dx_horizontal_recoverability": (res_b_goal.primitive_logs or {}).get("dx_horizontal_recoverability"),
        "dx_horizontal_rejected": (res_b_goal.primitive_logs or {}).get("dx_horizontal_rejected"),
        "cycles": (res_b_goal.primitive_logs or {}).get("cycles"),
        "failures": res_b_goal.failures,
        "n_actions": len(res_b_goal.actions),
        "errors_entry": res_b_goal.errors_entry,
        "errors_exit": res_b_goal.errors_exit,
        "dz_stop_reason": ((res_b_goal.primitive_logs or {}).get("dz") or {}).get("stop_reason"),
        "dz_disp_total_mm": ((res_b_goal.primitive_logs or {}).get("dz") or {}).get("disp_total_mm"),
    }
    (OUT / "cases" / "C_B_partial_dx_dz.json").write_text(
        json.dumps(jsafe(cases["C_B_partial_dx_dz"]), indent=2), encoding="utf-8"
    )
    print(
        f"  {res_b_goal.status} NEW={(res_b_goal.handoff or {}).get('SUCCESS_NEW')} "
        f"mech={(res_b_goal.primitive_logs or {}).get('dx_mechanism')} "
        f"partial={(res_b_goal.primitive_logs or {}).get('dx_partial')}",
        flush=True,
    )
    if res_b_goal.actions:
        videos["C_B_partial_dx_dz"] = render(
            env, st_b["capture"], res_b_goal.actions, OUT / "videos" / "C_B_partial_dx_dz.mp4", "B +dx->dz", st_b["ref"]
        )
        audits["C_B_partial_dx_dz"] = audit_action_capture_replay(
            env, st_b["capture"], res_b_goal.actions, res_b_goal.capture
        )
    if (res_b_goal.handoff or {}).get("SUCCESS_NEW"):
        replays["C_B_partial_dx_dz"] = replay_ic(env, "B", st_b, res_b_goal.actions)

    # D. Inward dz goal-driven
    print("=== D inward dz ===", flush=True)
    post = pickle.loads(B_POST_LAT.read_bytes())
    st_bl = {
        "capture": post["capture"],
        "ref": post.get("ref") or st_b["ref"],
        "stabilization_prefix_tokens": st_b.get("stabilization_prefix_tokens"),
        "precision_prefix_actions": np.load(AXIS_PREFIX)["actions"].tolist() if AXIS_PREFIX.exists() else [],
    }
    restore_capture(env, st_bl["capture"])
    e_in0 = measure_errors(env.sim)
    rin = correct_dz_inward(env, {"capture": st_bl["capture"], "ref": st_bl["ref"]}, PrecisionConfig())
    cases["D_inward_dz"] = {
        **slim(rin),
        "errors_entry": e_in0,
        "disp_total_mm": rin.get("disp_total_mm"),
        "stop_reason": rin.get("stop_reason"),
        "goal_reached": rin.get("goal_reached"),
    }
    print(
        f"  ok={rin.get('ok')} stop={rin.get('stop_reason')} disp={rin.get('disp_total_mm')} "
        f"dz0={e_in0.get('dz_mm')} dz1={(rin.get('errors') or {}).get('dz_mm')}",
        flush=True,
    )
    res_bl = run_precision(env, st_bl, "B", PrecisionConfig(allow_horizontal_dx=False, ez_tol_mm=8.0))
    cases["D_reg_B_dz_in"] = {
        "status": res_bl.status,
        "SUCCESS_NEW": (res_bl.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_bl.phases,
        "cycles": (res_bl.primitive_logs or {}).get("cycles"),
    }
    if (res_bl.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B_dz_in"] = replay_ic(env, "B", st_bl, res_bl.actions)
    print(f"  pipeline NEW={(res_bl.handoff or {}).get('SUCCESS_NEW')}", flush=True)

    # E. Outward dz goal-driven
    print("=== E outward dz ===", flush=True)
    restore_capture(env, st_c["capture"])
    e_out0 = measure_errors(env.sim)
    rout = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig())
    cases["E_outward_dz"] = {
        **slim(rout),
        "errors_entry": e_out0,
        "disp_total_mm": rout.get("disp_total_mm"),
        "stop_reason": rout.get("stop_reason"),
        "goal_reached": rout.get("goal_reached"),
    }
    if rout.get("ok"):
        restore_capture(env, rout["capture"])
        for _ in range(8):
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        h = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
        cases["E_outward_dz"]["handoff"] = {
            "SUCCESS_NEW": h.get("SUCCESS_NEW"),
            "SUCCESS_12S": h.get("SUCCESS_12S"),
            "SUCCESS_AXIS_V1": h.get("SUCCESS_AXIS_V1"),
        }
        videos["E_outward_dz"] = render(
            env, st_c["capture"], rout["actions"], OUT / "videos" / "E_outward_dz.mp4", "Out dz", st_c["ref"]
        )
        audits["E_outward_dz"] = audit_action_capture_replay(env, st_c["capture"], rout["actions"], rout.get("capture"))
        # 3x from stable C
        trials = []
        for i in range(3):
            restore_capture(env, st_c["capture"])
            for a in rout.get("actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            for _ in range(8):
                step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
            hh = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
            trials.append(
                {
                    "trial": i,
                    "SUCCESS_NEW": bool(hh.get("SUCCESS_NEW")),
                    "SUCCESS_12S": bool(hh.get("SUCCESS_12S")),
                    "SUCCESS_AXIS_V1": bool(hh.get("SUCCESS_AXIS_V1")),
                }
            )
        replays["E_outward_dz"] = {
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_12S_3of3": all(t["SUCCESS_12S"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
            "mode": "stable_C",
        }
    print(
        f"  ok={rout.get('ok')} stop={rout.get('stop_reason')} disp={rout.get('disp_total_mm')} "
        f"NEW={(cases['E_outward_dz'].get('handoff') or {}).get('SUCCESS_NEW')}",
        flush=True,
    )

    # F. Negative dx goal-driven from post_P5
    print("=== F negative dx ===", flush=True)
    p5 = pickle.loads(POST_P5.read_bytes())
    restore_capture(env, p5["capture"])
    e_n0 = measure_errors(env.sim)
    rhn = correct_dx_horizontal_attempt(
        env, {"capture": p5["capture"], "ref": p5["ref"]}, PrecisionConfig(use_goal_driven_slip=True)
    )
    # Closed-loop from post_P5: may skip all precision if already HANDOFF-ready.
    res_p5 = run_precision(
        env,
        {"capture": p5["capture"], "ref": p5["ref"]},
        "B",
        PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False, use_goal_driven_slip=True),
    )
    cases["F_neg_dx"] = {
        **slim(rhn),
        "errors_entry": e_n0,
        "disp_total_mm": rhn.get("disp_total_mm"),
        "stop_reason": rhn.get("stop_reason") or rhn.get("reason"),
        "goal_reached": rhn.get("goal_reached"),
        "partial_progress": rhn.get("partial_progress"),
        "skipped_unnecessary": bool(rhn.get("skipped")),
        "pipeline_status": res_p5.status,
        "pipeline_SUCCESS_NEW": (res_p5.handoff or {}).get("SUCCESS_NEW"),
        "pipeline_n_actions": len(res_p5.actions),
        "pipeline_phases": res_p5.phases,
    }
    (OUT / "cases" / "F_neg_dx.json").write_text(json.dumps(jsafe(cases["F_neg_dx"]), indent=2), encoding="utf-8")
    print(
        f"  ok={rhn.get('ok')} skip={rhn.get('skipped')} stop={rhn.get('stop_reason') or rhn.get('reason')} "
        f"disp={rhn.get('disp_total_mm')} dx0={e_n0.get('dx_mm')} "
        f"pipe={res_p5.status} NEW={(res_p5.handoff or {}).get('SUCCESS_NEW')}",
        flush=True,
    )
    if rhn.get("ok") and rhn.get("actions"):
        videos["F_neg_dx"] = render(env, p5["capture"], rhn["actions"], OUT / "videos" / "F_neg_dx.mp4", "H -dx")
        audits["F_neg_dx"] = audit_action_capture_replay(env, p5["capture"], rhn["actions"], rhn.get("capture"))
    elif res_p5.actions:
        videos["F_neg_dx"] = render(
            env, p5["capture"], res_p5.actions, OUT / "videos" / "F_neg_dx.mp4", "F post_P5", p5["ref"]
        )
        audits["F_neg_dx"] = audit_action_capture_replay(env, p5["capture"], res_p5.actions, res_p5.capture)

    # G/H. Plan B regression (lateral fallback, no forced +dx search)
    print("=== H reg_B lateral ===", flush=True)
    res_b = run_precision(env, st_b, "B", PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False))
    cases["H_reg_B"] = {
        "status": res_b.status,
        "SUCCESS_NEW": (res_b.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_b.phases,
        "dx_mechanism": (res_b.primitive_logs or {}).get("dx_mechanism"),
        "cycles": (res_b.primitive_logs or {}).get("cycles"),
        "n_actions": len(res_b.actions),
    }
    if (res_b.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B"] = replay_ic(env, "B", st_b, res_b.actions)
        videos["reg_B"] = render(env, st_b["capture"], res_b.actions, OUT / "videos" / "reg_B.mp4", "reg B", st_b["ref"])
        audits["reg_B"] = audit_action_capture_replay(env, st_b["capture"], res_b.actions, res_b.capture)
    print(f"  {res_b.status} NEW={(res_b.handoff or {}).get('SUCCESS_NEW')} mech={cases['H_reg_B'].get('dx_mechanism')}", flush=True)

    remaining = []
    notes = []
    c_b = cases.get("C_B_partial_dx_dz", {})
    if not c_b.get("SUCCESS_NEW"):
        remaining.append(
            "Plan B searched +dx then dz did not reach SUCCESS_NEW: "
            f"{c_b.get('status')} / {c_b.get('failures')}"
        )
    elif c_b.get("dx_mechanism") != "horizontal_dx":
        notes.append(
            "Plan B hard_910220: searched horizontal +dx post-state is not axially recoverable "
            f"(CF inward contact_loss); SUCCESS_NEW via {c_b.get('dx_mechanism')} + goal inward "
            f"(stop={c_b.get('dz_stop_reason')}, disp={c_b.get('dz_disp_total_mm')})."
        )
    fneg = cases.get("F_neg_dx", {})
    if fneg.get("skipped_unnecessary"):
        notes.append(
            "post_P5 negative-dx: already HANDOFF-geometry-ready (e_pos≤12, θ≤10); "
            "goal-aware controller skips unnecessary horizontal sliding."
        )
    elif not (fneg.get("ok") or fneg.get("pipeline_SUCCESS_NEW")):
        remaining.append(
            f"Negative dx goal-driven: {fneg.get('failure_code')} "
            f"phase={fneg.get('phase')} stop={fneg.get('stop_reason')}"
        )

    report = f"""# Unified Precision Oracle v2.2 — Goal-Driven Closed-Loop

Created UTC: {datetime.now(timezone.utc).isoformat()}  
Wall: {time.time() - t0:.1f} s

## Comparison vs v2.1

| Aspect | v2.1 | v2.2 |
|--------|------|------|
| Slip stop | Fixed `target_total_mm` / legacy 2+req | Geometric `SlipGoal` (axis band + e_pos) |
| `*_slip_mm` | Desired travel | Max **safe** travel only |
| Pipeline | One-pass dtheta→dx→dz | Bounded reassess cycles |
| Horizontal partial | Discarded if `|dx|>ex_tol` | Accepted if safe progress |
| Stop reasons | Phase failure codes | GOAL_REACHED / HANDOFF_READY / MAX_SAFE_TRAVEL / … |

## Answers

1. **Goal-driven stopping avoids unnecessary sliding?**  
   Outward/inward report `stop_reason` and `disp_total_mm` as diagnostics.  
   Outward: stop=`{cases.get('E_outward_dz', {}).get('stop_reason')}` disp=`{cases.get('E_outward_dz', {}).get('disp_total_mm')}`  
   Inward: stop=`{cases.get('D_inward_dz', {}).get('stop_reason')}` disp=`{cases.get('D_inward_dz', {}).get('disp_total_mm')}`  
   Neg-dx: stop=`{cases.get('F_neg_dx', {}).get('stop_reason')}` disp=`{cases.get('F_neg_dx', {}).get('disp_total_mm')}`

2. **Partial +dx then dz → SUCCESS_NEW?**  
   `{cases.get('C_B_partial_dx_dz', {}).get('SUCCESS_NEW')}`  
   mech=`{cases.get('C_B_partial_dx_dz', {}).get('dx_mechanism')}` partial=`{cases.get('C_B_partial_dx_dz', {}).get('dx_partial')}`  
   status=`{cases.get('C_B_partial_dx_dz', {}).get('status')}`  
   recoverability=`{json.dumps(jsafe(cases.get('C_B_partial_dx_dz', {}).get('dx_horizontal_recoverability')), indent=2)}`

3. **Inward/outward without fixed displacement targets?**  
   Inward: stop=`{cases.get('D_inward_dz', {}).get('stop_reason')}` disp=`{cases.get('D_inward_dz', {}).get('disp_total_mm')}` pipeline NEW=`{cases.get('D_reg_B_dz_in', {}).get('SUCCESS_NEW')}`  
   Outward: stop=`{cases.get('E_outward_dz', {}).get('stop_reason')}` disp=`{cases.get('E_outward_dz', {}).get('disp_total_mm')}` NEW=`{(cases.get('E_outward_dz') or {}).get('handoff', {}).get('SUCCESS_NEW')}`  
   Both use `contract=goal_driven` (disp is diagnostic only).

4. **Avoid unnecessary precision ops?**  
   Direct C: skip=`{cases.get('A_direct_C', {}).get('precision_skipped')}` n_actions=`{cases.get('A_direct_C', {}).get('n_actions')}`  
   Neg-dx / post_P5: skipped=`{cases.get('F_neg_dx', {}).get('skipped_unnecessary')}` stop=`{cases.get('F_neg_dx', {}).get('stop_reason')}`  
   Closed-loop early HANDOFF when geometry ready (`PRECISION_EARLY_HANDOFF`).

5. **Physically replayable?**  
   Audits: `{json.dumps(jsafe({k: v.get('ok') for k, v in audits.items()}), indent=2)}`  
   IC replays 3/3: `{json.dumps(jsafe({k: v.get('SUCCESS_NEW_3of3') for k, v in replays.items()}), indent=2)}`

6. **Unsupported / notes?**  
   Hard gaps: `{json.dumps(remaining, indent=2)}`  
   Notes: `{json.dumps(notes, indent=2)}`

## Case results

```
{json.dumps(jsafe(cases), indent=2)}
```

## Replays

```
{json.dumps(jsafe(replays), indent=2)}
```

## Videos

```
{json.dumps(jsafe(videos), indent=2)}
```

## Invariants

SUCCESS_NEW / nominal / physics / 7D@20ms / 39D / K49 unchanged.  
v2.1 evidence under `unified_precision_oracle_v21/` untouched.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "cases": cases,
        "replays": replays,
        "audits": {k: {"ok": v.get("ok"), "n_actions": v.get("n_actions")} for k, v in audits.items()},
        "videos": videos,
        "remaining_unsupported": remaining,
        "notes": notes,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "audits" / "provenance.json").write_text(json.dumps(jsafe(audits), indent=2), encoding="utf-8")
    (OUT / "replays" / "all.json").write_text(json.dumps(jsafe(replays), indent=2), encoding="utf-8")
    print(json.dumps({"wall_s": summary["wall_s"], "remaining": remaining}, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
