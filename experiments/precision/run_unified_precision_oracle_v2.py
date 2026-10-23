#!/usr/bin/env python
"""v2: Fix horizontal dx (P0–P9) + outward dz; regression on A/B/C successes.

Results → results/diagnostics/unified_precision_oracle_v2/
No bulk generation. Frozen invariants preserved.
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
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    PrecisionConfig,
    correct_dtheta,
    correct_dx_horizontal_attempt,
    correct_dx_verified,
    correct_dz_outward,
    measure_errors,
    run_precision,
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

OUT = ROOT / "results/diagnostics/unified_precision_oracle_v2"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
IC_A = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
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


def slim(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("actions", "capture", "series")}


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
            f"{phase} t={float(env.sim.data.time)-t0:.2f} α={err['alpha_up_deg']:.0f} "
            f"e=[{err['dx_mm']:.1f},{err['dz_mm']:.1f}] θ={err['dtheta_deg']:.1f}"
        )
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        scn.ngeom = 0
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.2, 0.95, 0.25, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), [title, line])

    img = frame("START")
    for _ in range(int(1.5 * FPS)):
        frames.append(img)
    dup = max(1, int(round((DT / 0.4) * FPS)))
    for i, a in enumerate(actions):
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
    return {"path": str(path), "n_frames": len(frames), "duration_s": len(frames) / FPS}


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
        else:
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for tok in state.get("stabilization_prefix_tokens") or [32, 12, 12, 0, 0, 0, 0, 0]:
                step_legal(env.sim, decode_tok(table, book48, int(tok)))
            for a in state.get("precision_prefix_actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            ref = root["ref"]
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "replays").mkdir(exist_ok=True)
    env = make_env()
    t0 = time.time()
    cases, videos, replays = {}, {}, {}

    st_a, st_b, st_c = load_stable("A"), load_stable("B"), load_stable("C")

    # ---------- Horizontal dx: +dx on Plan B (ex>0), 2 mm then 5 mm ----------
    print("=== H_dx_pos_2mm (Plan B) ===", flush=True)
    cfg2 = PrecisionConfig(horizontal_slip_mm=2.0, allow_horizontal_dx=True)
    rh = correct_dx_horizontal_attempt(env, {"capture": st_b["capture"], "ref": st_b["ref"]}, cfg2)
    cases["H_dx_pos_2mm"] = slim(rh)
    (OUT / "cases" / "H_dx_pos_2mm.json").write_text(json.dumps(jsafe(cases["H_dx_pos_2mm"]), indent=2), encoding="utf-8")
    print(f"  ok={rh.get('ok')} code={rh.get('failure_code')} phase={rh.get('phase')} disp={rh.get('disp_along_mm')}", flush=True)
    if rh.get("actions"):
        videos["H_dx_pos_2mm"] = render(env, st_b["capture"], rh["actions"], OUT / "videos" / "H_dx_pos_2mm.mp4", "H dx +2mm")

    if rh.get("ok"):
        print("=== H_dx_pos_5mm ===", flush=True)
        cfg5 = PrecisionConfig(horizontal_slip_mm=5.0)
        rh5 = correct_dx_horizontal_attempt(env, {"capture": st_b["capture"], "ref": st_b["ref"]}, cfg5)
        cases["H_dx_pos_5mm"] = slim(rh5)
        (OUT / "cases" / "H_dx_pos_5mm.json").write_text(json.dumps(jsafe(cases["H_dx_pos_5mm"]), indent=2), encoding="utf-8")
        print(f"  ok={rh5.get('ok')} disp={rh5.get('disp_along_mm')}", flush=True)
        if rh5.get("actions"):
            videos["H_dx_pos_5mm"] = render(env, st_b["capture"], rh5["actions"], OUT / "videos" / "H_dx_pos_5mm.mp4", "H dx +5mm")

    # Negative dx: after verified lateral overshoot or use Plan A post-pivot with flipped desire
    # Build state with negative ex by running verified lateral on B (ex→~-1) is too small.
    # Use A post-pivot (ex≈+15) is still positive. For −dx, start from B_post_lateral (ex≈-1.2)
    # and force a synthetic target via temporary error — better: run horizontal with desire
    # from a state that has ex < -12. Create by applying verified lateral beyond, or
    # use post-lateral and set horizontal with inverted test by flipping cfg via state clone.
    print("=== H_dx_neg_2mm (from B after verified overshoot prep) ===", flush=True)
    # Run verified lateral on B to get near-zero, then manually not good for -dx.
    # Instead: take B_post_lateral (ex≈-1.2) — too small. Use Plan A after pivot+verified
    # lateral leaves ex≈+5. For true negative, start from B_post_lateral and use
    # force: we construct desire by temporarily reporting — actually call horizontal
    # on a capture where we first apply a wrong-direction open-loop nudge is illegal.
    # Diagnostic: invert by running on state with e_x negative large — A stable has +15.
    # After A pivot+lateral success path exit was +5.7. Closest large negative:
    # Use endpoint post-P5 from handoff opt if available (ex≈-5.3).
    post_p5 = ROOT / "results/diagnostics/handoff_state_optimization/snapshots/post_P5.pkl"
    if post_p5.exists():
        p5 = pickle.loads(post_p5.read_bytes())
        restore_capture(env, p5["capture"])
        print(f"  post_P5 errors={measure_errors(env.sim)}", flush=True)
        rhn = correct_dx_horizontal_attempt(env, {"capture": p5["capture"], "ref": p5["ref"]}, cfg2)
        cases["H_dx_neg_2mm"] = slim(rhn)
        (OUT / "cases" / "H_dx_neg_2mm.json").write_text(json.dumps(jsafe(cases["H_dx_neg_2mm"]), indent=2), encoding="utf-8")
        print(f"  ok={rhn.get('ok')} code={rhn.get('failure_code')} phase={rhn.get('phase')} disp={rhn.get('disp_along_mm')}", flush=True)
        if rhn.get("actions"):
            videos["H_dx_neg_2mm"] = render(env, p5["capture"], rhn["actions"], OUT / "videos" / "H_dx_neg_2mm.mp4", "H dx -2mm")
        if rhn.get("ok"):
            rhn5 = correct_dx_horizontal_attempt(env, {"capture": p5["capture"], "ref": p5["ref"]}, PrecisionConfig(horizontal_slip_mm=5.0))
            cases["H_dx_neg_5mm"] = slim(rhn5)
            (OUT / "cases" / "H_dx_neg_5mm.json").write_text(json.dumps(jsafe(cases["H_dx_neg_5mm"]), indent=2), encoding="utf-8")
            if rhn5.get("actions"):
                videos["H_dx_neg_5mm"] = render(env, p5["capture"], rhn5["actions"], OUT / "videos" / "H_dx_neg_5mm.mp4", "H dx -5mm")
    else:
        cases["H_dx_neg_2mm"] = {"ok": False, "reason": "post_P5_missing"}

    # Compare verified lateral (unchanged) on B
    print("=== verified_lateral_B (control) ===", flush=True)
    rv = correct_dx_verified(env, {"capture": st_b["capture"], "ref": st_b["ref"]}, PrecisionConfig())
    cases["verified_lateral_B"] = slim(rv)
    (OUT / "cases" / "verified_lateral_B.json").write_text(json.dumps(jsafe(cases["verified_lateral_B"]), indent=2), encoding="utf-8")

    # ---------- Outward dz: Plan C stable (ex≈1.6, ez≈-5.2) — isolate axial ----------
    print("=== Out_dz_2mm (Plan C stable, ez<0, small ex) ===", flush=True)
    restore_capture(env, st_c["capture"])
    print(f"  C e={measure_errors(env.sim)} down={gates(env.sim)['fingertips_down']}", flush=True)
    cfg_out = PrecisionConfig(outward_slip_mm=2.0, evaluate_handoff=True)
    rout = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, cfg_out)
    cases["Out_dz_2mm"] = slim(rout)
    (OUT / "cases" / "Out_dz_2mm.json").write_text(json.dumps(jsafe(cases["Out_dz_2mm"]), indent=2), encoding="utf-8")
    print(f"  ok={rout.get('ok')} code={rout.get('failure_code')} phase={rout.get('phase')} disp={rout.get('disp_along_mm')}", flush=True)
    if rout.get("actions"):
        videos["Out_dz_2mm"] = render(
            env, st_c["capture"], rout["actions"], OUT / "videos" / "Out_dz_2mm.mp4", "Out dz 2mm", st_c["ref"]
        )
    if rout.get("ok"):
        restore_capture(env, rout["capture"])
        for _ in range(8):
            step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        h = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
        cases["Out_dz_2mm"]["handoff"] = {
            "SUCCESS_NEW": h.get("SUCCESS_NEW"),
            "SUCCESS_12S": h.get("SUCCESS_12S"),
            "SUCCESS_AXIS_V1": h.get("SUCCESS_AXIS_V1"),
            "reason": h.get("std_reason") or h.get("reason"),
        }
        rout5 = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig(outward_slip_mm=5.0))
        cases["Out_dz_5mm"] = slim(rout5)
        (OUT / "cases" / "Out_dz_5mm.json").write_text(json.dumps(jsafe(cases["Out_dz_5mm"]), indent=2), encoding="utf-8")
        print(f"  Out5 ok={rout5.get('ok')} disp={rout5.get('disp_along_mm')}", flush=True)
        if rout5.get("actions"):
            videos["Out_dz_5mm"] = render(
                env, st_c["capture"], rout5["actions"], OUT / "videos" / "Out_dz_5mm.mp4", "Out dz 5mm", st_c["ref"]
            )

    # Also record post-pivot A outward as prep-isolation diagnostic (large ex — may fail)
    print("=== Out_dz_A_postpivot_diag (large ex; expect harder) ===", flush=True)
    piv = correct_dtheta(env, {"capture": st_a["capture"], "ref": st_a["ref"]}, PrecisionConfig(), "A")
    rout_a = correct_dz_outward(env, {"capture": piv["capture"], "ref": st_a["ref"]}, PrecisionConfig(outward_slip_mm=2.0))
    cases["Out_dz_A_postpivot_diag"] = slim(rout_a)
    (OUT / "cases" / "Out_dz_A_postpivot_diag.json").write_text(
        json.dumps(jsafe(cases["Out_dz_A_postpivot_diag"]), indent=2), encoding="utf-8"
    )
    print(f"  ok={rout_a.get('ok')} code={rout_a.get('failure_code')} phase={rout_a.get('phase')}", flush=True)

    # ---------- Regressions ----------
    print("=== REGRESSION C_direct ===", flush=True)
    res_c = run_precision(env, st_c, "C", PrecisionConfig(allow_horizontal_dx=False))
    cases["reg_C_direct"] = {
        "status": res_c.status,
        "SUCCESS_NEW": (res_c.handoff or {}).get("SUCCESS_NEW"),
        "precision_skipped": res_c.precision_skipped,
        "phases": res_c.phases,
    }
    print(f"  {res_c.status} NEW={(res_c.handoff or {}).get('SUCCESS_NEW')}", flush=True)

    print("=== REGRESSION A_dtheta_dx ===", flush=True)
    res_a = run_precision(env, st_a, "A", PrecisionConfig(allow_horizontal_dx=True))
    cases["reg_A_dtheta_dx"] = {
        "status": res_a.status,
        "SUCCESS_NEW": (res_a.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_a.phases,
        "failures": res_a.failures,
        "dx_mechanism": (res_a.primitive_logs or {}).get("dx_mechanism"),
        "horizontal_rejected": (res_a.primitive_logs or {}).get("dx_horizontal_rejected"),
    }
    print(f"  {res_a.status} NEW={(res_a.handoff or {}).get('SUCCESS_NEW')} mech={cases['reg_A_dtheta_dx'].get('dx_mechanism')}", flush=True)
    if (res_a.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_A_dtheta_dx"] = replay_ic(env, "A", st_a, res_a.actions)
        videos["reg_A"] = render(env, st_a["capture"], res_a.actions, OUT / "videos" / "reg_A_dtheta_dx.mp4", "reg A", st_a["ref"])

    print("=== REGRESSION B_dx_inward ===", flush=True)
    res_b = run_precision(env, st_b, "B", PrecisionConfig(allow_horizontal_dx=True))
    cases["reg_B_dx"] = {
        "status": res_b.status,
        "SUCCESS_NEW": (res_b.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_b.phases,
        "failures": res_b.failures,
        "dx_mechanism": (res_b.primitive_logs or {}).get("dx_mechanism"),
    }
    print(f"  {res_b.status} NEW={(res_b.handoff or {}).get('SUCCESS_NEW')}", flush=True)
    if (res_b.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B_dx"] = replay_ic(env, "B", st_b, res_b.actions)
        videos["reg_B"] = render(env, st_b["capture"], res_b.actions, OUT / "videos" / "reg_B_dx.mp4", "reg B", st_b["ref"])

    print("=== REGRESSION B_dz_inward ===", flush=True)
    post = pickle.loads(B_POST_LAT.read_bytes())
    st_bl = {
        "capture": post["capture"],
        "ref": post.get("ref") or st_b["ref"],
        "stabilization_prefix_tokens": st_b.get("stabilization_prefix_tokens"),
        "precision_prefix_actions": np.load(AXIS_PREFIX)["actions"].tolist() if AXIS_PREFIX.exists() else [],
    }
    res_bl = run_precision(env, st_bl, "B", PrecisionConfig(allow_horizontal_dx=False, ez_tol_mm=8.0))
    cases["reg_B_dz_inward"] = {
        "status": res_bl.status,
        "SUCCESS_NEW": (res_bl.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_bl.phases,
        "failures": res_bl.failures,
    }
    print(f"  {res_bl.status} NEW={(res_bl.handoff or {}).get('SUCCESS_NEW')}", flush=True)
    if (res_bl.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B_dz_inward"] = replay_ic(env, "B", st_bl, res_bl.actions)

    if cases.get("Out_dz_2mm", {}).get("ok") and rout.get("actions"):
        print("=== Out_dz Plan C HANDOFF already recorded; no Stage-1 IC replay needed ===", flush=True)
        replays["Out_dz_C"] = {
            "mode": "from_stable_C",
            "SUCCESS_NEW": (cases["Out_dz_2mm"].get("handoff") or {}).get("SUCCESS_NEW"),
            "note": "Outward isolated on Plan C stable grasp; Stage-1 expert already verified separately",
        }

    succ = [k for k, v in cases.items() if v.get("ok") is True or v.get("SUCCESS_NEW") is True or v.get("status") in ("PRECISION_EXECUTED_SUCCESS", "PRECISION_SKIPPED_DIRECT_SUCCESS")]
    fail = [k for k, v in cases.items() if v.get("ok") is False or (v.get("SUCCESS_NEW") is False) or str(v.get("status", "")).endswith("FAIL") or str(v.get("status", "")).endswith("FAILED")]

    report = f"""# Unified Precision Oracle v2 — Horizontal dx & Outward dz

Created UTC: {datetime.now(timezone.utc).isoformat()}  
Wall: {time.time() - t0:.1f} s

## Changes

1. Horizontal dx: full P0–P9 with **P3/P6 sliding rolls**; P3 angle from contact/gravity search.
2. Shared `run_staged_axial_slip(direction=...)` for inward (−u_open) and outward (+u_open).
3. Always capture terminal state on slip failure; rejected horizontal branches omitted from live tape.
4. Phase-specific `failure_code` taxonomy.
5. `run_roll_phase(require_down=False)` final OK no longer requires fingertips-down.

## Case results

| Case | ok / NEW | failure_code | phase | notes |
|------|----------|--------------|-------|-------|
"""
    for k, v in cases.items():
        report += (
            f"| {k} | {v.get('ok', v.get('SUCCESS_NEW', v.get('status')))} | "
            f"{v.get('failure_code')} | {v.get('phase')} | "
            f"disp={v.get('disp_along_mm')} |\n"
        )

    report += f"""

## Replays

{json.dumps(jsafe(replays), indent=2)}

## Videos

{json.dumps(jsafe(videos), indent=2)}

## Regression targets

Preserve: A dθ+dx 3/3, B dx→inward dz 3/3, B inward dz 3/3, C direct skip.
Verified lateral fallback unchanged.

## Invariants

GROUND_SCENE, 7D@20ms, 39D, K49, nominal, HANDOFF, SUCCESS_NEW/AXIS_V1 — unchanged.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "cases": cases,
        "replays": replays,
        "videos": videos,
        "success_keys": succ,
        "failure_keys": fail,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "replays" / "all.json").write_text(json.dumps(jsafe(replays), indent=2), encoding="utf-8")
    print(json.dumps({"wall_s": summary["wall_s"], "success_keys": succ, "failure_keys": fail}, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
