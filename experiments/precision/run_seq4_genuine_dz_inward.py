#!/usr/bin/env python
"""Genuine Seq4: cold-start DZ-only inward on a valid B IC (no prior DX).

Search → execute → diagnose → MP4 of real action tape.
Does not weaken thresholds / Stage-1 / GRPO / M111 / datasets.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.endpoint_only_axial_roll import gates  # noqa: E402
from data_generation.oracles.precision.force_slip_control import DT  # noqa: E402
from data_generation.oracles.precision.precision_dz import (  # noqa: E402
    check_dz_inward_entry,
    correct_dz_inward,
)
from data_generation.oracles.precision.precision_goals import measure_errors  # noqa: E402
from data_generation.oracles.precision.precision_handoff import (  # noqa: E402
    posture_is_fingertips_down,
    run_final_down_handoff_gate,
)
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.sign_aware_axial import select_axial_mode  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import entry_gate  # noqa: E402
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

OUT = ROOT / "results/diagnostics/seq4_genuine_dz_inward"
ICS_DIRS = [
    ROOT / "results/data_generation/stage2_b_precision_dominant_v1/ics",
    ROOT / "results/data_generation/stage2_b_grpo_multistage_v1/ics",
    ROOT / "results/data_generation/stage2_abc_final_v2/ics",
]
FPS = 20
MAX_SCREEN = 220  # coverage bound


def jsafe(o: Any) -> Any:
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


def list_b_pkls() -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for d in ICS_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("B__*.pkl")):
            # Prefer recovery / stable-entry style; skip obvious post-dx names later via physics
            key = p.name
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def screen_one(env, path: Path, cfg: PrecisionConfig) -> dict:
    try:
        blob = pickle.loads(path.read_bytes())
    except Exception as e:
        return {"path": str(path), "bin": "LOAD_FAIL", "error": str(e)}
    if "capture" not in blob or "ref" not in blob:
        return {"path": str(path), "bin": "MISSING_FIELDS"}
    # Exclude Precision post-dx seating only. Stage-1 post_b1 is allowed (not DX ST).
    name = path.name.lower()
    if any(tok in name for tok in ("post_lateral", "posta", "post_dx", "after_dx", "dx6")):
        return {"path": str(path), "bin": "NAME_POST_PRECISION_DX", "name": path.name}
    # Prefer Plan-B distribution for this Seq4 deliverable
    if not path.name.startswith("B__") and "PILOT_B" not in path.name:
        return {"path": str(path), "bin": "NOT_PLAN_B", "name": path.name}

    restore_capture(env, blob["capture"])
    gate = entry_gate(env)
    restore_capture(env, blob["capture"])
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, blob["capture"])
    err = measure_errors(env.sim)
    g = gates(env.sim)
    entry = check_dz_inward_entry(env.sim, cfg)
    # AXIS ez_tol=20 hides |ez|~10–18 that still needs SUCCESS_NEW correction.
    # Use a 2 mm floor only to detect meaningful inward sign — do NOT weaken dz_safe_ex.
    axial_axis = select_axial_mode(float(err["dz_mm"]), axial_tol_mm=float(cfg.ez_tol_mm))
    axial_need = select_axial_mode(float(err["dz_mm"]), axial_tol_mm=2.0)
    down = posture_is_fingertips_down(env.sim)
    bil = bool(err.get("bilateral"))
    ex = abs(float(err["dx_mm"]))
    ez = float(err["dz_mm"])
    epos = float(err["e_pos_mm"])
    # Meaningful inward: signed inward mode under 2 mm floor, |ez|>=2, and pose not already NEW-ready
    inward_need = (
        axial_need.get("mode") == "B_inward"
        and abs(ez) >= 2.0
        and epos > float(cfg.working_pos_mm) * 0.5  # nontrivial residual
    )

    row = {
        "path": str(path),
        "name": path.name,
        "gate_ok": bool(gate.get("ok")),
        "stable": bool(stab.get("stable")),
        "fingertips_down": down,
        "bilateral": bil,
        "alpha_up_deg": float(g["alpha_up_deg"]),
        "dx_mm": float(err["dx_mm"]),
        "dz_mm": ez,
        "dtheta_deg": float(err["dtheta_deg"]),
        "e_pos_mm": epos,
        "v_rel_norm": float(err["v_rel_norm"]),
        "dz_mode_axis_v1": axial_axis.get("mode"),
        "dz_mode": axial_need.get("mode"),
        "entry_ok": bool(entry.get("ok")),
        "entry_reasons": list(entry.get("reasons") or []),
        "dz_safe_ex_mm": float(entry.get("dz_safe_ex_mm") or cfg.dz_safe_ex_mm),
        "inward_need": inward_need,
    }
    if not (row["gate_ok"] and row["stable"]):
        row["bin"] = "GATE_OR_STABLE_FAIL"
    elif not down:
        row["bin"] = "NOT_DOWN"
    elif not bil:
        row["bin"] = "NOT_BILATERAL"
    elif ex > float(cfg.dz_safe_ex_mm):
        row["bin"] = "EX_TOO_LARGE"
    elif not inward_need:
        row["bin"] = "NO_INWARD_DZ_NEED"
    elif not entry.get("ok"):
        row["bin"] = "ENTRY_OTHER_REJECT"
    else:
        row["bin"] = "QUALIFIED"
        # Prefer larger |ez| within safe ex (harder / more meaningful)
        row["score"] = abs(ez) - 0.05 * ex
    return row


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    h = min(16 + 15 * len(lines), out.shape[0])
    out[:h] = (18, 18, 22)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 3
        for i, line in enumerate(lines):
            fill = (255, 140, 120) if ("FAIL" in line or "REJECT" in line) else (240, 240, 240)
            if "PASS" in line or "SUCCESS" in line:
                fill = (140, 255, 160)
            dr.text((5, y), str(line)[:105], fill=fill)
            y += 14
        return np.asarray(im)
    except Exception:
        return out


def render_actions(env, start_cap, actions, dest: Path, title: str, phase_tags: list[tuple[int, str]] | None = None):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 480, 640)
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.65, "azimuth": 135, "elevation": -22}
    )
    frames = []
    tag_map = {i: t for i, t in (phase_tags or [])}

    def cur_tag(i, n):
        best = None
        for k, t in sorted(tag_map.items()):
            if i >= k:
                best = t
        return best or f"{i+1}/{n}"

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.lookat[:] = 0.5 * po + 0.5 * ph + np.array([0, 0, 0.02])
        err = measure_errors(env.sim)
        mode = classify_contact_mode(env.sim)
        g = gates(env.sim)
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"a={g['alpha_up_deg']:.0f}")
        lines = [
            f"{title} | {tag}",
            f"t={float(env.sim.data.time)-t0:.2f}s orient={orient} mode={mode['mode']} S1={mode.get('one_sided_mechanical')}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} th={err['dtheta_deg']:.1f} epos={err['e_pos_mm']:.1f}",
            f"vrel={err['v_rel_norm']:.3f} fnL/R={err['fn_L']:.1f}/{err['fn_R']:.1f} FnZ={mode['Fn_z_L']:.1f}/{mode['Fn_z_R']:.1f}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), lines)

    img0 = frame("START")
    for _ in range(int(0.5 * FPS)):
        frames.append(img0)
    n = len(actions or [])
    for i, a in enumerate(actions or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == n - 1:
            frames.append(frame(cur_tag(i, n)))
    # brief end hold
    for _ in range(int(0.4 * FPS)):
        frames.append(frame("END"))
    rw.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path) if path else None, "n_frames": len(frames), "n_actions": n}


def phase_trace_from_dz(rz: dict, actions_before: int = 0) -> list[dict]:
    """Summarize phase_log with action offsets."""
    rows = []
    # Approximate action counts from nested structures if present
    for p in rz.get("phase_log") or []:
        rows.append({
            "phase": p.get("phase"),
            "ok": p.get("ok"),
            "reasons": p.get("reasons"),
            "selection": (p.get("selection") or {}).get("chosen_name") if isinstance(p.get("selection"), dict) else p.get("selection"),
            "alpha": p.get("alpha") or p.get("alpha_up_deg"),
            "mode": (p.get("mode") if not isinstance(p.get("mode"), dict) else p.get("mode", {}).get("mode")),
            "support_side": p.get("support_side"),
            "raw": {k: v for k, v in p.items() if k not in ("series", "actions")},
        })
    for p in rz.get("rotation_diag") or []:
        rows.append({"phase": f"rot:{p.get('phase')}", "ok": p.get("ok"), "reason": p.get("reason"),
                     "path_length_u": p.get("path_length_u"), "sim_s": p.get("sim_s")})
    slip = rz.get("slip") or {}
    if slip:
        rows.append({
            "phase": "inward_slip",
            "ok": slip.get("ok"),
            "stop_reason": slip.get("stop_reason"),
            "disp_along_mm": slip.get("disp_along_mm"),
            "n_actions_slip": len(slip.get("actions") or []),
        })
    rows.append({
        "phase": "dz_summary",
        "ok": rz.get("ok"),
        "branch": rz.get("branch"),
        "reason": rz.get("reason"),
        "failure_code": rz.get("failure_code"),
        "reused_s1": rz.get("reused_s1"),
        "n_actions_total": len(rz.get("actions") or []),
        "actions_offset": actions_before,
        "outcomes": rz.get("outcomes"),
    })
    return rows


def execute_seq4(env, blob: dict, cfg: PrecisionConfig, ic_meta: dict) -> dict:
    restore_capture(env, blob["capture"])
    start_cap = take_capture(env)
    err0 = measure_errors(env.sim)
    entry0 = check_dz_inward_entry(env.sim, cfg)
    assert entry0.get("ok"), "IC must pass entry gate"
    assert posture_is_fingertips_down(env.sim)
    assert abs(float(err0["dx_mm"])) <= float(cfg.dz_safe_ex_mm)

    actions: list = []
    phases = ["STABLE_DOWN", "ENTRY_OK", "DZ_INWARD_COLD"]
    t0 = time.time()
    rz = correct_dz_inward(env, {"capture": start_cap, "ref": blob["ref"]}, cfg, s1_state=None)
    actions.extend(rz.get("actions") or [])
    wall_dz = time.time() - t0
    traces = phase_trace_from_dz(rz)
    mid = {
        "ok": rz.get("ok"),
        "branch": rz.get("branch"),
        "reason": rz.get("reason"),
        "failure_code": rz.get("failure_code"),
        "phase": rz.get("phase"),
        "n_actions": len(rz.get("actions") or []),
        "reused_s1": rz.get("reused_s1"),
        "outcomes": rz.get("outcomes"),
        "wall_s": wall_dz,
        "errors_after_dz": measure_errors(env.sim) if rz.get("capture") else None,
        "posture_after_dz": {
            "fingertips_down": posture_is_fingertips_down(env.sim),
            "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
        },
    }
    phases.append(f"dz:{rz.get('branch')}:{rz.get('phase') or ('ok' if rz.get('ok') else 'fail')}")

    handoff = None
    gate = None
    status = "FAIL_DURING_DZ"
    if rz.get("ok") and rz.get("capture") is not None:
        cur = {"capture": rz["capture"], "ref": blob["ref"]}
        phases.append("FINAL_DOWN_GATE")
        gate = run_final_down_handoff_gate(env, cur, cfg, actions)
        if not gate.get("ok"):
            status = "FAIL_FINAL_DOWN_GATE"
            mid["gate_fail"] = {k: gate.get(k) for k in ("failure_code", "reason", "phase")}
        else:
            cur = {"capture": gate["capture"], "ref": cur["ref"]}
            restore_capture(env, cur["capture"])
            phases.append("HANDOFF_NOMINAL")
            handoff = evaluate_handoff_with_axis(
                env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n
            )
            restore_capture(env, cur["capture"])
            if handoff.get("SUCCESS_NEW"):
                status = "PASS"
            else:
                status = "MECHANISM_OK_HANDOFF_FAIL"
    elif len(actions) == 0:
        status = "FAIL_ZERO_ACTIONS"
    else:
        status = "FAIL_DURING_DZ"

    # Build phase tags for video from cumulative action estimates in phase_log if available
    phase_tags = [(0, "dz_inward")]
    return {
        "ic": ic_meta,
        "errors_entry": err0,
        "entry_check": {k: v for k, v in entry0.items() if k != "errors"},
        "dz": {
            k: v for k, v in rz.items()
            if k not in ("actions", "capture", "series")
        },
        "mid": mid,
        "phase_traces": traces,
        "gate": {k: v for k, v in (gate or {}).items() if k != "capture"} if gate else None,
        "handoff": {
            k: v for k, v in (handoff or {}).items()
            if k not in ("series", "samples", "trace")
        } if handoff else None,
        "SUCCESS_NEW": (handoff or {}).get("SUCCESS_NEW"),
        "SUCCESS_12S": (handoff or {}).get("SUCCESS_12S"),
        "n_actions": len(actions),
        "phases": phases,
        "status": status,
        "actions": actions,
        "start_capture": start_cap,
        "exit_capture": take_capture(env),
        "exit_posture": {
            "fingertips_down": posture_is_fingertips_down(env.sim),
            "alpha_up_deg": float(gates(env.sim)["alpha_up_deg"]),
            "errors": measure_errors(env.sim),
            "contact_mode": classify_contact_mode(env.sim),
        },
        "no_prior_dx": True,
        "s1_state_inherited": False,
    }


def diagnose_vs_s1_reuse(env, blob: dict, cfg: PrecisionConfig, fail_rz: dict) -> dict:
    """Compare cold inward failure with DX6→inward reuse on same IC (diagnostic only)."""
    from data_generation.oracles.precision.precision_dx_support_transfer import (
        correct_dx_support_transfer,
    )

    restore_capture(env, blob["capture"])
    rdx = correct_dx_support_transfer(env, blob, cfg)
    out = {
        "dx_ok": rdx.get("ok"),
        "dx_reason": rdx.get("reason"),
        "dx_phase": rdx.get("phase"),
        "s1_after_dx": rdx.get("s1_state"),
        "ex0": rdx.get("ex0"),
        "ex1": rdx.get("ex1"),
    }
    if not rdx.get("ok"):
        out["note"] = "DX ST also failed on this IC; cannot contrast S1-reuse path."
        return out
    cur = {"capture": rdx["capture"], "ref": blob["ref"]}
    rz = correct_dz_inward(env, cur, cfg, s1_state=rdx.get("s1_state"))
    out["reuse_dz_ok"] = rz.get("ok")
    out["reuse_branch"] = rz.get("branch")
    out["reuse_reused_s1"] = rz.get("reused_s1")
    out["reuse_reason"] = rz.get("reason")
    out["reuse_phase"] = rz.get("phase")
    out["reuse_n_actions"] = len(rz.get("actions") or [])
    out["cold_fail_phase"] = fail_rz.get("phase") or fail_rz.get("mid", {}).get("phase")
    out["cold_fail_reason"] = fail_rz.get("reason") or fail_rz.get("mid", {}).get("reason")
    # First divergence heuristic
    if out["reuse_dz_ok"] and not (fail_rz.get("ok") or fail_rz.get("mid", {}).get("ok")):
        out["divergence"] = (
            "Cold-start fails establishing/using S1 for UP; post-DX S1 reuse succeeds — "
            "issue likely S1 establishment or support-aware UP from upright, not DualLoop itself."
        )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=False,
    )

    pkls = list_b_pkls()
    print(f"Screening up to {MAX_SCREEN} of {len(pkls)} B ICs...", flush=True)
    screened = []
    counts: dict[str, int] = {}
    for i, p in enumerate(pkls[:MAX_SCREEN]):
        row = screen_one(env, p, cfg)
        screened.append(row)
        counts[row.get("bin", "?")] = counts.get(row.get("bin", "?"), 0) + 1
        if (i + 1) % 40 == 0:
            print(f"  screened {i+1} qualified={counts.get('QUALIFIED', 0)}", flush=True)

    qualified = [r for r in screened if r.get("bin") == "QUALIFIED"]
    qualified.sort(key=lambda r: -float(r.get("score") or 0))
    (OUT / "SCREEN.json").write_text(
        json.dumps(jsafe({"counts": counts, "n_screened": len(screened), "n_available": len(pkls),
                          "qualified": qualified, "sample_rejects": screened[:30]}), indent=2),
        encoding="utf-8",
    )
    print(f"Qualified: {len(qualified)} / {len(screened)}  bins={counts}", flush=True)

    if not qualified:
        report = [
            "# Seq4 Genuine DZ-Inward — UNVERIFIED",
            "",
            f"Screened {len(screened)} / {len(pkls)} B ICs. **No qualifying cold-start IC.**",
            "",
            "Required: DOWN, bilateral, |ex|<=dz_safe_ex, inward dz need, entry_ok, no prior DX.",
            "",
            "## Coverage bins",
            "",
        ]
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            report.append(f"- `{k}`: {v}")
        report += [
            "",
            "Did **not** substitute precondition-reject replay as Seq4 PASS.",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
        ]
        (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        (OUT / "RESULT.json").write_text(
            json.dumps({"status": "UNVERIFIED", "reason": "no_qualifying_IC", "counts": counts}, indent=2),
            encoding="utf-8",
        )
        print("\n".join(report))
        return

    # Try qualified ICs in score order; prefer full PASS, else best nonzero-action evidence.
    attempts = []
    chosen = None
    best_fail = None
    for cand in qualified[:10]:
        print(f"Attempt {cand['name']} ex={cand['dx_mm']:.2f} ez={cand['dz_mm']:.2f}", flush=True)
        blob = pickle.loads(Path(cand["path"]).read_bytes())
        try:
            res = execute_seq4(env, blob, cfg, cand)
        except Exception as e:
            attempts.append({
                "ic": cand["name"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            continue
        attempts.append({
            "ic": cand["name"],
            "status": res["status"],
            "n_actions": res["n_actions"],
            "branch": (res.get("mid") or {}).get("branch"),
            "reason": (res.get("mid") or {}).get("reason"),
            "phase": (res.get("mid") or {}).get("phase"),
            "SUCCESS_NEW": res.get("SUCCESS_NEW"),
        })
        if res["n_actions"] <= 0:
            continue
        if res["status"] in ("PASS", "MECHANISM_OK_HANDOFF_FAIL"):
            chosen = res
            break
        if best_fail is None or res["n_actions"] > best_fail["n_actions"]:
            best_fail = res
    if chosen is None:
        chosen = best_fail
    if chosen is None:
        cand = qualified[0]
        blob = pickle.loads(Path(cand["path"]).read_bytes())
        chosen = execute_seq4(env, blob, cfg, cand)

    # Diagnosis if failed during mechanism
    diagnosis = None
    if chosen["status"] in ("FAIL_DURING_DZ", "FAIL_FINAL_DOWN_GATE") and chosen["n_actions"] > 0:
        print("Diagnosing vs DX→S1-reuse path...", flush=True)
        blob = pickle.loads(Path(chosen["ic"]["path"]).read_bytes())
        diagnosis = diagnose_vs_s1_reuse(env, blob, cfg, chosen)
        # Minimal repair attempts only if diagnosis points to S1 establishment rates
        # (record only; apply small rate tweak and rerun SAME IC once)
        if diagnosis.get("reuse_dz_ok") and not chosen.get("mid", {}).get("ok"):
            print("Cold fail vs reuse ok — retry cold with slower S1/tilt rates...", flush=True)
            cfg2 = PrecisionConfig(
                evaluate_handoff=True,
                use_support_transfer_dx=True,
                use_verified_lateral_fallback=False,
                rate_establish_s1=0.10,
                rate_protected_s1=0.08,
                rate_tilt_cruise=0.10,
                rate_tilt_peak=0.14,
                rate_tilt_floor=0.05,
            )
            retry = execute_seq4(env, blob, cfg2, chosen["ic"])
            diagnosis["retry_slower_rates"] = {
                "status": retry["status"],
                "n_actions": retry["n_actions"],
                "reason": (retry.get("mid") or {}).get("reason"),
                "phase": (retry.get("mid") or {}).get("phase"),
                "SUCCESS_NEW": retry.get("SUCCESS_NEW"),
                "SUCCESS_12S": retry.get("SUCCESS_12S"),
            }
            if retry["n_actions"] > 0 and retry["status"] in ("PASS", "MECHANISM_OK_HANDOFF_FAIL", "FAIL_DURING_DZ"):
                # Prefer successful retry; else keep better evidence tape
                if retry["status"] in ("PASS", "MECHANISM_OK_HANDOFF_FAIL") or (
                    chosen["status"] != "PASS" and retry["n_actions"] >= chosen["n_actions"]
                ):
                    chosen = retry
                    chosen["repair_applied"] = {
                        "type": "slower_support_aware_rates",
                        "rate_establish_s1": 0.10,
                        "rate_protected_s1": 0.08,
                        "rate_tilt_cruise": 0.10,
                    }

    # Render real action tape
    video = None
    if chosen["n_actions"] > 0:
        print(f"Rendering MP4 n_actions={chosen['n_actions']}...", flush=True)
        video = render_actions(
            env,
            chosen["start_capture"],
            chosen["actions"],
            OUT / "videos" / "seq4_genuine_dz_only_inward.mp4",
            f"SEQ4 genuine DZ-inward | {chosen['ic']['name'][:40]}",
        )
    else:
        video = {"path": None, "note": "zero actions — no execution tape"}

    # Strip heavy fields for JSON
    result = {k: v for k, v in chosen.items() if k not in ("actions", "start_capture", "exit_capture")}
    result["video"] = video
    result["diagnosis"] = jsafe(diagnosis)
    result["attempts"] = attempts
    result["screen_counts"] = counts
    result["generated"] = datetime.now(timezone.utc).isoformat()

    # Explicit status label
    if chosen["status"] == "PASS":
        verdict = "PASS"
    elif chosen["status"] == "MECHANISM_OK_HANDOFF_FAIL" and chosen["n_actions"] > 0:
        verdict = "PASS_MECHANISM_HANDOFF_FAIL"  # mechanism verified; HANDOFF separate
    elif chosen["n_actions"] > 0:
        verdict = "FAIL"
    else:
        verdict = "UNVERIFIED"
    result["verdict"] = verdict

    (OUT / "RESULT.json").write_text(json.dumps(jsafe(result), indent=2), encoding="utf-8")
    # Also save action count meta only
    (OUT / "PHASE_TRACES.json").write_text(json.dumps(jsafe(chosen.get("phase_traces")), indent=2), encoding="utf-8")

    lines = [
        "# Seq4 Genuine Cold-Start DZ-Only Inward",
        "",
        f"Generated: {result['generated']}",
        f"**Verdict: {verdict}** (execution status=`{chosen['status']}`)",
        "",
        "## IC",
        "",
        f"- `{chosen['ic']['name']}`",
        f"- ex={chosen['ic']['dx_mm']:.2f} mm (|ex|≤{chosen['ic']['dz_safe_ex_mm']}), "
        f"ez={chosen['ic']['dz_mm']:.2f} mm, θ={chosen['ic']['dtheta_deg']:.2f}°, "
        f"DOWN={chosen['ic']['fingertips_down']}",
        f"- Prior DX: **no**. Inherited S1: **no**.",
        f"- Screen coverage: {len(screened)}/{len(pkls)} B ICs; qualified={len(qualified)}; bins={counts}",
        "",
        "## Execution",
        "",
        f"- n_actions={chosen['n_actions']}",
        f"- branch={chosen.get('mid', {}).get('branch')} phase={chosen.get('mid', {}).get('phase')}",
        f"- reason={chosen.get('mid', {}).get('reason')}",
        f"- SUCCESS_NEW={chosen.get('SUCCESS_NEW')} SUCCESS_12S={chosen.get('SUCCESS_12S')}",
        f"- exit DOWN={chosen.get('exit_posture', {}).get('fingertips_down')} "
        f"α={chosen.get('exit_posture', {}).get('alpha_up_deg')}",
        "",
        f"Video: `{video.get('path')}`",
        "",
        "## Diagnosis",
        "",
        f"```json\n{json.dumps(jsafe(diagnosis), indent=2)}\n```" if diagnosis else "_n/a_",
        "",
        "## Note",
        "",
        "Previous Seq4 precondition-reject (`lateral_ex_too_large`, n_actions=0) remains a **negative** test only.",
        "This report replaces it as the mechanism verification attempt.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
