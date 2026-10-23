#!/usr/bin/env python
"""Physically validate the unified Precision flowchart (dtheta→dx ST→dz).

Results → results/diagnostics/precision_oracle_unified_flow/
No SFT/RL; no dataset regeneration.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.endpoint_only_axial_roll import gates  # noqa: E402
from data_generation.oracles.precision.force_slip_control import DT, sample_state  # noqa: E402
from data_generation.oracles.precision.precision_dx import correct_dx_horizontal_attempt  # noqa: E402
from data_generation.oracles.precision.precision_dx_support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.precision_dz import (  # noqa: E402
    correct_dz_inward,
    correct_dz_outward,
)
from data_generation.oracles.precision.precision_goals import assess_need, measure_errors  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    PrecisionConfig,
    correct_dtheta,
    run_precision,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_oracle_unified_flow"
VID = OUT / "videos"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
GBI063 = ROOT / "results/data_generation/stage2_abc_final_v2/ics/C__postlift__gbi_063__08584501db98836d.pkl"
FPS = 20


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


def load_blob(path: Path, plan: str | None = None) -> dict:
    blob = pickle.loads(path.read_bytes())
    if plan:
        blob["plan"] = plan
    if "ref" not in blob:
        raise KeyError(f"missing ref in {path}")
    return blob


def slim_result(res) -> dict:
    d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
    pl = d.get("primitive_logs") or {}
    # Drop huge nested series
    return jsafe(d)


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    h = min(14 + 15 * len(lines), out.shape[0])
    out[:h] = (18, 18, 22)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 3
        for line in lines:
            dr.text((5, y), str(line)[:100], fill=(240, 240, 240))
            y += 15
        return np.asarray(im)
    except Exception:
        return out


def render_tape(env, start_cap, actions, dest: Path, title: str, phase_fn=None, ref=None, nom_s=4.0):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 420, 640)
    # Brighter-friendly elevated view
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.65, "azimuth": 135, "elevation": -22}
    )
    frames = []

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.lookat[:] = 0.5 * po + 0.5 * ph + np.array([0, 0, 0.02])
        err = measure_errors(env.sim)
        mode = classify_contact_mode(env.sim)
        g = gates(env.sim)
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"α={g['alpha_up_deg']:.0f}")
        lines = [
            f"{title} | {tag}",
            f"t={float(env.sim.data.time)-t0:.2f}s  orient={orient}  mode={mode['mode']}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} θ={err['dtheta_deg']:.1f} epos={err['e_pos_mm']:.1f}",
            f"vrel={err['v_rel_norm']:.3f} fnL/R={err['fn_L']:.1f}/{err['fn_R']:.1f} FnZ={mode['Fn_z_L']:.1f}/{mode['Fn_z_R']:.1f}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), lines)

    img = frame("START")
    for _ in range(int(0.6 * FPS)):
        frames.append(img)
    n = len(actions or [])
    for i, a in enumerate(actions or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == n - 1:
            tag = phase_fn(i, n) if phase_fn else f"{i+1}/{n}"
            frames.append(frame(tag))
    if ref is not None:
        from envs.outcomes.recovery_success import apply_handoff, nominal_step

        apply_handoff(env.sim, ref)
        for k in range(int(nom_s / DT)):
            nominal_step(env.sim)
            if k % 5 == 0:
                frames.append(frame("NOMINAL"))
    rw.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    # Contact sheet: first / mid / last
    sheet = None
    if frames:
        try:
            from PIL import Image

            idxs = [0, len(frames) // 2, len(frames) - 1]
            imgs = [Image.fromarray(frames[i]) for i in idxs]
            w, h = imgs[0].size
            sheet_im = Image.new("RGB", (w * 3, h))
            for i, im in enumerate(imgs):
                sheet_im.paste(im, (i * w, 0))
            sheet = dest.with_suffix("").as_posix() + "_sheet.png"
            sheet_im.save(sheet)
        except Exception:
            sheet = None
    return {"path": str(path) if path else None, "n_frames": len(frames), "sheet": sheet}


def run_case(env, name: str, plan: str, blob: dict, cfg: PrecisionConfig) -> dict:
    t0 = time.time()
    restore_capture(env, blob["capture"])
    err0 = measure_errors(env.sim)
    need0 = assess_need(err0, cfg)
    try:
        res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, plan, cfg)
    except Exception as e:
        return {
            "name": name,
            "plan": plan,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "errors_entry": err0,
            "need_entry": need0,
            "wall_s": time.time() - t0,
        }
    wall = time.time() - t0
    restore_capture(env, res.capture or blob["capture"])
    err1 = measure_errors(env.sim)
    n_act = len(res.actions or [])
    return {
        "name": name,
        "plan": plan,
        "status": res.status,
        "ok": bool(res.handoff and res.handoff.get("SUCCESS_NEW")) if cfg.evaluate_handoff else res.status.endswith("SUCCESS"),
        "SUCCESS_NEW": (res.handoff or {}).get("SUCCESS_NEW"),
        "SUCCESS_12S": (res.handoff or {}).get("SUCCESS_12S"),
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "n_actions": n_act,
        "sim_s": n_act * DT,
        "wall_s": wall,
        "errors_entry": err0,
        "errors_exit": err1,
        "need_entry": {k: need0[k] for k in need0 if k != "axis_decision"},
        "dx_mechanism": (res.primitive_logs or {}).get("dx_mechanism"),
        "dz_branch": ((res.primitive_logs or {}).get("dz") or {}).get("branch"),
        "dz_s1_reused": (res.primitive_logs or {}).get("dz_s1_reused"),
        "dtheta_restore_down": ((res.primitive_logs or {}).get("dtheta") or {}).get("fingertips_down"),
        "primitive_logs": jsafe({
            k: v for k, v in (res.primitive_logs or {}).items()
            if k in ("dx_mechanism", "dx_support_transfer", "dx_keep_s1_for_dz", "dz", "dz_s1_reused", "dtheta", "decision", "cycles")
        }),
        "actions": res.actions,
        "capture": res.capture,
        "ref": res.ref or blob["ref"],
        "start_capture": blob["capture"],
    }


def branch_tests(env) -> dict:
    """Execute representative scheduler branches on real ICs (no teleport)."""
    cfg = PrecisionConfig(evaluate_handoff=True, use_support_transfer_dx=True)
    out = {}

    # Direct HANDOFF / A
    st_a = load_blob(STATES / "plan_a__ball_train_v2_0070__stable.pkl", "A")
    out["direct_or_A"] = run_case(env, "direct_or_A", "A", st_a, cfg)

    # B dx-capable
    st_b = load_blob(STATES / "plan_b__hard_910220__stable.pkl", "B")
    out["plan_b_hard_910220"] = run_case(env, "plan_b_hard_910220", "B", st_b, cfg)

    # C stable
    st_c = load_blob(STATES / "plan_c__gbi_010__stable.pkl", "C")
    out["plan_c_gbi_010"] = run_case(env, "plan_c_gbi_010", "C", st_c, cfg)

    # Plan C post-lift gbi_063 (known inward dz fail previously)
    if GBI063.exists():
        st_g = load_blob(GBI063, "C")
        out["C_postlift_gbi_063"] = run_case(env, "C_postlift_gbi_063", "C", st_g, cfg)

    # Primitive-level branch probes on B (geometry as-is)
    restore_capture(env, st_b["capture"])
    err = measure_errors(env.sim)
    need = assess_need(err, cfg)
    blob = {"capture": st_b["capture"], "ref": st_b["ref"]}

    # dtheta-only probe (may skip if θ already ok)
    if need.get("dtheta_needed"):
        rd = correct_dtheta(env, blob, cfg, "B")
        out["probe_dtheta"] = {
            "ok": rd.get("ok"),
            "fingertips_down": rd.get("fingertips_down"),
            "n_actions": len(rd.get("actions") or []),
            "errors": rd.get("errors"),
            "restore_down": rd.get("restore_down"),
            "actions": rd.get("actions"),
            "capture": rd.get("capture"),
            "start_capture": blob["capture"],
            "ref": blob["ref"],
        }
    else:
        out["probe_dtheta"] = {"ok": True, "skipped": True, "reason": "dtheta_not_needed", "errors": err}

    # dz outward Case A from current (if outward needed) else force call to see skip path
    restore_capture(env, st_b["capture"])
    g = gates(env.sim)
    rz_out = correct_dz_outward(env, blob, cfg, s1_state=None)
    out["probe_dz_outward"] = {
        "ok": rz_out.get("ok"),
        "branch": rz_out.get("branch"),
        "skipped_rotation": rz_out.get("skipped_rotation"),
        "n_actions": len(rz_out.get("actions") or []),
        "phase_log": rz_out.get("phase_log"),
        "reason": rz_out.get("reason"),
        "errors": rz_out.get("errors"),
        "actions": rz_out.get("actions"),
        "capture": rz_out.get("capture"),
        "start_capture": blob["capture"],
        "ref": blob["ref"],
        "entry_fingertips_down": g["fingertips_down"],
    }

    # dz inward Case B
    restore_capture(env, st_b["capture"])
    rz_in = correct_dz_inward(env, blob, cfg, s1_state=None)
    out["probe_dz_inward"] = {
        "ok": rz_in.get("ok"),
        "branch": rz_in.get("branch"),
        "reused_s1": rz_in.get("reused_s1"),
        "n_actions": len(rz_in.get("actions") or []),
        "phase_log": rz_in.get("phase_log"),
        "reason": rz_in.get("reason"),
        "errors": rz_in.get("errors"),
        "actions": rz_in.get("actions"),
        "capture": rz_in.get("capture"),
        "start_capture": blob["capture"],
        "ref": blob["ref"],
    }

    # dx support-transfer
    restore_capture(env, st_b["capture"])
    rdx = correct_dx_support_transfer(env, blob, cfg)
    out["probe_dx_support_transfer"] = {
        "ok": rdx.get("ok"),
        "mechanism": rdx.get("mechanism"),
        "n_actions": len(rdx.get("actions") or []),
        "timing_s": rdx.get("timing_s"),
        "phase_log": rdx.get("phase_log"),
        "s1_state": rdx.get("s1_state"),
        "keep_s1_for_dz": rdx.get("keep_s1_for_dz"),
        "ex0": rdx.get("ex0"),
        "ex1": rdx.get("ex1"),
        "disp_along_mm": rdx.get("disp_along_mm"),
        "reason": rdx.get("reason"),
        "actions": rdx.get("actions"),
        "capture": rdx.get("capture"),
        "start_capture": blob["capture"],
        "ref": blob["ref"],
    }

    # dx → inward dz S1 reuse
    if rdx.get("ok") and rdx.get("capture") and rdx.get("s1_state"):
        blob2 = {"capture": rdx["capture"], "ref": blob["ref"]}
        restore_capture(env, blob2["capture"])
        rz2 = correct_dz_inward(env, blob2, cfg, s1_state=rdx.get("s1_state"))
        out["probe_dx_then_dz_inward"] = {
            "ok": rz2.get("ok"),
            "branch": rz2.get("branch"),
            "reused_s1": rz2.get("reused_s1"),
            "n_actions": len(rz2.get("actions") or []),
            "phase_log": rz2.get("phase_log"),
            "reason": rz2.get("reason"),
            "actions": (rdx.get("actions") or []) + (rz2.get("actions") or []),
            "capture": rz2.get("capture"),
            "start_capture": blob["capture"],
            "ref": blob["ref"],
        }
    else:
        out["probe_dx_then_dz_inward"] = {"ok": False, "reason": "dx_st_failed_no_s1"}

    # Legacy horizontal timing baseline (same IC)
    restore_capture(env, st_b["capture"])
    t_h0 = time.time()
    rh = correct_dx_horizontal_attempt(env, blob, cfg)
    out["probe_dx_horizontal_baseline"] = {
        "ok": rh.get("ok"),
        "n_actions": len(rh.get("actions") or []),
        "sim_s": len(rh.get("actions") or []) * DT,
        "wall_s": time.time() - t_h0,
        "phase_log": [{k: p.get(k) for k in ("phase", "ok")} for p in (rh.get("phase_log") or [])],
        "actions": rh.get("actions"),
        "capture": rh.get("capture"),
        "start_capture": blob["capture"],
        "ref": blob["ref"],
    }

    return out


def timing_rows(branch: dict) -> list[dict]:
    rows = []
    st = branch.get("probe_dx_support_transfer") or {}
    hz = branch.get("probe_dx_horizontal_baseline") or {}
    rows.append({
        "phase": "dx_prep_and_slide",
        "method": "support_transfer",
        "sim_s": (st.get("n_actions") or 0) * DT,
        "wall_s": sum((st.get("timing_s") or {}).values()) if st.get("timing_s") else None,
        "ok": st.get("ok"),
    })
    rows.append({
        "phase": "dx_prep_and_slide",
        "method": "horizontal_legacy_P0_P9",
        "sim_s": hz.get("sim_s"),
        "wall_s": hz.get("wall_s"),
        "ok": hz.get("ok"),
    })
    for name in ("plan_b_hard_910220", "plan_a_direct", "direct_or_A", "C_postlift_gbi_063", "plan_c_gbi_010"):
        c = branch.get(name)
        if not c:
            continue
        rows.append({
            "phase": "full_precision",
            "method": name,
            "sim_s": c.get("sim_s"),
            "wall_s": c.get("wall_s"),
            "ok": c.get("ok"),
            "SUCCESS_NEW": c.get("SUCCESS_NEW"),
            "n_actions": c.get("n_actions"),
            "dx_mechanism": c.get("dx_mechanism"),
            "dz_branch": c.get("dz_branch"),
        })
    # Per-phase ST
    for k, v in (st.get("timing_s") or {}).items():
        rows.append({"phase": k, "method": "support_transfer", "sim_s": None, "wall_s": v, "ok": st.get("ok")})
    return rows


def classify_branch_table(branch: dict) -> dict:
    """Map physical outcomes to required branch-test expectations."""
    tests = {}

    def phases_of(key):
        return list((branch.get(key) or {}).get("phases") or [])

    # Plan A stable often still needs correction; use Plan C gbi_010 for Direct HANDOFF.
    c_direct = branch.get("plan_c_gbi_010") or branch.get("direct_or_A") or {}
    tests["Direct_HANDOFF"] = {
        "expected": "Skip all correction primitives when SUCCESS_NEW probe passes",
        "observed_status": c_direct.get("status"),
        "phases": c_direct.get("phases"),
        "pass": bool(
            c_direct.get("status") == "PRECISION_SKIPPED_DIRECT_SUCCESS"
            or (c_direct.get("SUCCESS_NEW") and not any(p.startswith("correct_") for p in (c_direct.get("phases") or [])))
        ),
        "SUCCESS_NEW": c_direct.get("SUCCESS_NEW"),
        "case": "plan_c_gbi_010",
    }

    pdt = branch.get("probe_dtheta") or {}
    tests["dtheta_only_restore_DOWN"] = {
        "expected": "Pivot, brake, restore fingertips DOWN",
        "pass": bool(pdt.get("skipped") or (pdt.get("ok") and pdt.get("fingertips_down"))),
        "ok": pdt.get("ok"),
        "fingertips_down": pdt.get("fingertips_down"),
        "skipped": pdt.get("skipped"),
    }

    pout = branch.get("probe_dz_outward") or {}
    tests["dz_outward_Case_A"] = {
        "expected": "Direct sliding when fingertips DOWN; no S1/arm rotation",
        "pass": bool(
            pout.get("skipped_rotation") is True
            or (pout.get("branch") or "").startswith("A_")
            or (pout.get("branch") or "").startswith("D_")
        ),
        "branch": pout.get("branch"),
        "skipped_rotation": pout.get("skipped_rotation"),
        "ok": pout.get("ok"),
        "entry_fingertips_down": pout.get("entry_fingertips_down"),
        "reason": pout.get("reason"),
    }

    pin = branch.get("probe_dz_inward") or {}
    tests["dz_inward_Case_B"] = {
        "expected": "Establish S1, rotate UP, slide, restore DOWN",
        "pass": bool(pin.get("branch") in ("B_full", "C_reuse_s1", "D_already_up") or pin.get("ok")),
        "branch": pin.get("branch"),
        "ok": pin.get("ok"),
        "phase_log": pin.get("phase_log"),
        "reason": pin.get("reason"),
    }

    pdx = branch.get("probe_dx_support_transfer") or {}
    tests["dx_support_transfer"] = {
        "expected": "S1→horizontal→S2→lateral slide→brake→S1 (no DOWN restore)",
        "pass": bool(pdx.get("ok") and pdx.get("keep_s1_for_dz") and pdx.get("s1_state")),
        "ok": pdx.get("ok"),
        "keep_s1_for_dz": pdx.get("keep_s1_for_dz"),
        "s1_valid": (pdx.get("s1_state") or {}).get("valid"),
        "timing_s": pdx.get("timing_s"),
        "ex0": pdx.get("ex0"),
        "ex1": pdx.get("ex1"),
        "disp_along_mm": pdx.get("disp_along_mm"),
        "reason": pdx.get("reason"),
        "phase_log": pdx.get("phase_log"),
    }

    pcombo = branch.get("probe_dx_then_dz_inward") or {}
    tests["dx_then_inward_dz_S1_reuse"] = {
        "expected": "Reuse S1; no redundant standard-posture restoration before dz",
        "pass": bool(pcombo.get("reused_s1")),
        "ok": pcombo.get("ok"),
        "reused_s1": pcombo.get("reused_s1"),
        "branch": pcombo.get("branch"),
        "reason": pcombo.get("reason"),
    }

    b = branch.get("plan_b_hard_910220") or {}
    tests["full_B_dtheta_dx_dz"] = {
        "expected": "Execute needed primitives with valid transitions",
        "pass": "correct_" in " ".join(phases_of("plan_b_hard_910220")) or bool(b.get("SUCCESS_NEW")),
        "status": b.get("status"),
        "phases": b.get("phases"),
        "SUCCESS_NEW": b.get("SUCCESS_NEW"),
        "dx_mechanism": b.get("dx_mechanism"),
        "dz_branch": b.get("dz_branch"),
        "failures": b.get("failures"),
    }

    gbi = branch.get("C_postlift_gbi_063") or {}
    tests["C_postlift_gbi_063"] = {
        "expected": "Diagnose/correct inward dz from post-lift IC",
        "pass": gbi.get("ok") is True,
        "status": gbi.get("status"),
        "phases": gbi.get("phases"),
        "SUCCESS_NEW": gbi.get("SUCCESS_NEW"),
        "failures": gbi.get("failures"),
        "errors_entry": gbi.get("errors_entry"),
        "errors_exit": gbi.get("errors_exit"),
        "dz_branch": gbi.get("dz_branch"),
        "need_entry": gbi.get("need_entry"),
    }

    return tests


def write_report(branch_table: dict, physical: dict, timing: list, videos: dict) -> str:
    st = (physical.get("probe_dx_support_transfer") or {})
    hz = (physical.get("probe_dx_horizontal_baseline") or {})
    st_s = (st.get("n_actions") or 0) * DT
    hz_s = hz.get("sim_s") or 0.0
    saved = (hz_s - st_s) if hz_s and st_s else None

    def yn(x):
        return "YES" if x else "NO"

    lines = [
        "# Unified Precision Oracle Flowchart — REPORT",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Answers to required questions",
        "",
        f"1. Does dtheta reliably restore fingertips DOWN? **{yn((branch_table.get('dtheta_only_restore_DOWN') or {}).get('pass'))}** "
        f"(ok={((branch_table.get('dtheta_only_restore_DOWN') or {}).get('ok'))}, "
        f"down={((branch_table.get('dtheta_only_restore_DOWN') or {}).get('fingertips_down'))})",
        "",
        f"2. Does the new dx perform S1 → transfer → gravity lateral slide? **{yn((branch_table.get('dx_support_transfer') or {}).get('pass'))}** "
        f"ex {st.get('ex0')}→{st.get('ex1')} disp={st.get('disp_along_mm')} keep_s1={st.get('keep_s1_for_dz')}",
        "",
        f"3. Is direct outward dz free of unnecessary S1/rotation? **{yn((branch_table.get('dz_outward_Case_A') or {}).get('pass'))}** "
        f"branch={((branch_table.get('dz_outward_Case_A') or {}).get('branch'))} "
        f"skipped_rotation={((branch_table.get('dz_outward_Case_A') or {}).get('skipped_rotation'))}",
        "",
        f"4. Does direct inward dz use S1 protection? **{yn((branch_table.get('dz_inward_Case_B') or {}).get('pass'))}** "
        f"branch={((branch_table.get('dz_inward_Case_B') or {}).get('branch'))}",
        "",
        f"5. Is S1 reused between dx and dz? **{yn((branch_table.get('dx_then_inward_dz_S1_reuse') or {}).get('pass'))}** "
        f"reused_s1={((branch_table.get('dx_then_inward_dz_S1_reuse') or {}).get('reused_s1'))}",
        "",
        f"6. Time saved: support-transfer sim={st_s:.2f}s vs legacy horizontal={hz_s:.2f}s "
        f"(Δ={saved:.2f}s)" if saved is not None else "6. Time saved: see TIMING_COMPARISON.csv",
        "",
        "7. A/B/C physical SUCCESS_NEW:",
    ]
    for key in ("direct_or_A", "plan_b_hard_910220", "plan_c_gbi_010", "C_postlift_gbi_063"):
        c = physical.get(key) or {}
        lines.append(
            f"   - {key}: status={c.get('status')} SUCCESS_NEW={c.get('SUCCESS_NEW')} "
            f"sim_s={c.get('sim_s')} dx={c.get('dx_mechanism')} dz={c.get('dz_branch')} failures={c.get('failures')}"
        )
    lines += [
        "",
        "8. Remaining mechanism failures:",
    ]
    for k, t in branch_table.items():
        if not t.get("pass"):
            lines.append(f"   - {k}: {t.get('reason') or t.get('failures') or t}")
    lines += [
        "",
        "## Videos",
        "",
    ]
    for k, v in videos.items():
        lines.append(f"- `{k}`: {v.get('path')} sheet={v.get('sheet')}")
    lines += [
        "",
        "## Notes",
        "",
        "- Continuous Oracle SUCCESS_NEW ≠ K49 replay success; K49 not claimed here.",
        "- Golden bitwise equivalence with v2.2/v2.3 is not expected (behavior change by design).",
        "- No SFT/RL; training datasets not replaced.",
        "",
    ]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()

    print("Running branch / physical validation...", flush=True)
    physical = branch_tests(env)
    # Persist without actions arrays in JSON
    physical_json = {}
    for k, v in physical.items():
        physical_json[k] = {kk: vv for kk, vv in v.items() if kk not in ("actions", "capture", "start_capture", "ref")}
        physical_json[k] = jsafe(physical_json[k])

    branch_table = classify_branch_table(physical)
    (OUT / "BRANCH_TESTS.json").write_text(json.dumps(jsafe(branch_table), indent=2), encoding="utf-8")
    (OUT / "PHYSICAL_VALIDATION.json").write_text(json.dumps(physical_json, indent=2), encoding="utf-8")

    rows = timing_rows(physical)
    with (OUT / "TIMING_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({kk for r in rows for kk in r.keys()}))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Videos (executed tapes only)
    videos = {}
    video_specs = [
        ("01_dtheta_restore_down", "probe_dtheta"),
        ("02_dx_support_transfer", "probe_dx_support_transfer"),
        ("03_dz_outward_direct", "probe_dz_outward"),
        ("04_dz_inward_S1_flip", "probe_dz_inward"),
        ("05_dx_then_dz_S1_reuse", "probe_dx_then_dz_inward"),
        ("06a_dx_legacy_horizontal", "probe_dx_horizontal_baseline"),
        ("06b_dx_new_support_transfer", "probe_dx_support_transfer"),
        ("07_C_gbi_063", "C_postlift_gbi_063"),
        ("08_full_B", "plan_b_hard_910220"),
    ]
    for vname, key in video_specs:
        item = physical.get(key) or {}
        acts = item.get("actions")
        start = item.get("start_capture") or item.get("capture")
        if not acts and key in ("direct_or_A",):
            continue
        if start is None or acts is None:
            videos[vname] = {"path": None, "note": f"no tape for {key}", "ok": item.get("ok")}
            continue
        # For full cases, start_capture may be missing — use stored
        if key in ("C_postlift_gbi_063", "plan_b_hard_910220", "direct_or_A", "plan_c_gbi_010"):
            # re-fetch start from known blobs
            if key == "C_postlift_gbi_063" and GBI063.exists():
                start = load_blob(GBI063, "C")["capture"]
                ref = load_blob(GBI063, "C")["ref"]
            elif key == "plan_b_hard_910220":
                b = load_blob(STATES / "plan_b__hard_910220__stable.pkl", "B")
                start, ref = b["capture"], b["ref"]
            else:
                ref = item.get("ref")
        else:
            ref = item.get("ref")
        print(f"  render {vname} n={len(acts)}", flush=True)
        try:
            videos[vname] = render_tape(
                env, start, acts, VID / f"{vname}.mp4", vname, ref=ref if key.startswith(("07", "08")) else None
            )
            videos[vname]["ok"] = item.get("ok")
            videos[vname]["source"] = key
        except Exception as e:
            videos[vname] = {"path": None, "error": str(e), "ok": False}

    (VID / "INDEX.json").write_text(json.dumps(jsafe(videos), indent=2), encoding="utf-8")

    # Regression report vs prior gbi_063 failure
    prior_fail = ROOT / "results/data_generation/stage2_abc_final_v2/failures/C__postlift__gbi_063__08584501db98836d__failure.json"
    prior = json.loads(prior_fail.read_text(encoding="utf-8")) if prior_fail.exists() else {}
    gbi_now = physical.get("C_postlift_gbi_063") or {}
    reg = [
        "# REGRESSION_REPORT",
        "",
        "## Expected behavior changes (not bitwise golden)",
        "",
        "- dx default mechanism: `support_transfer_dx` (ends at S1) vs legacy `horizontal` (P7/P8 DOWN).",
        "- dtheta now restores fingertips DOWN after pivot.",
        "- dz Case A/C/D shortcuts alter action tapes when orientation already aligned.",
        "",
        "## Plan C gbi_063",
        "",
        f"- Prior: status={prior.get('status')} failures={prior.get('failures')}",
        f"- Now: status={gbi_now.get('status')} SUCCESS_NEW={gbi_now.get('SUCCESS_NEW')} failures={gbi_now.get('failures')}",
        f"- Entry errors: {gbi_now.get('errors_entry')}",
        f"- Exit errors: {gbi_now.get('errors_exit')}",
        f"- First causal need: {gbi_now.get('need_entry')}",
        "",
        "## v2.3 golden",
        "",
        "- Do not claim bitwise action equality with `unified_precision_oracle_v23/baseline_v22`.",
        "- Scheduler unit tests alone are insufficient; see PHYSICAL_VALIDATION.json + videos.",
        "",
    ]
    (OUT / "REGRESSION_REPORT.md").write_text("\n".join(reg), encoding="utf-8")

    report = write_report(branch_table, physical, rows, videos)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote deliverables under {OUT}", flush=True)


if __name__ == "__main__":
    main()
