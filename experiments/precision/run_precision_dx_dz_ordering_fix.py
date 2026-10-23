#!/usr/bin/env python
"""Fix validation: dx/dz ordering + unsafe dz entry rejection.

Results → results/diagnostics/precision_oracle_dx_dz_ordering_fix/
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.endpoint_only_axial_roll import gates  # noqa: E402
from data_generation.oracles.precision.force_slip_control import DT, sample_state  # noqa: E402
from data_generation.oracles.precision.precision_dx_support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.precision_dz import (  # noqa: E402
    check_dz_inward_entry,
    correct_dz_inward,
    correct_dz_outward,
)
from data_generation.oracles.precision.precision_goals import assess_need, measure_errors  # noqa: E402
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_oracle_dx_dz_ordering_fix"
VID = OUT / "videos"
PRIOR = ROOT / "results/diagnostics/precision_oracle_rotation_fix/videos"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
PLAN_B = STATES / "plan_b__hard_910220__stable.pkl"
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
            dr.text((5, y), str(line)[:110], fill=(240, 240, 240))
            y += 15
        return np.asarray(im)
    except Exception:
        return out


def render_tape(env, start_cap, actions, dest: Path, title: str):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 420, 640)
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
            f"FnZ={mode['Fn_z_L']:.1f}/{mode['Fn_z_R']:.1f} bil={err['bilateral']}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        return overlay(np.asarray(rw.render(), np.uint8), lines)

    img = frame("START")
    for _ in range(int(0.5 * FPS)):
        frames.append(img)
    n = len(actions or [])
    for i, a in enumerate(actions or []):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == n - 1:
            frames.append(frame(f"{i+1}/{n}"))
    rw.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {"path": str(path), "n_actions": n, "n_frames": len(frames)}


def slim_dz(r):
    return jsafe({k: v for k, v in r.items() if k not in ("actions", "capture", "series")})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig(evaluate_handoff=True)
    blob_b = pickle.loads(PLAN_B.read_bytes())
    blob_gbi = pickle.loads(GBI063.read_bytes())

    results = {}
    force_rows = []

    # --- IC snapshot for Plan B ---
    restore_capture(env, blob_b["capture"])
    err0 = measure_errors(env.sim)
    need0 = assess_need(err0, cfg)
    m0 = classify_contact_mode(env.sim)
    entry0 = check_dz_inward_entry(env.sim, cfg)
    results["plan_b_ic"] = {
        "errors": jsafe(err0),
        "need": jsafe({k: need0[k] for k in need0 if k != "axis_decision"}),
        "contact": jsafe({k: m0[k] for k in (
            "mode", "bilateral", "Fn_z_L", "Fn_z_R", "fn_L", "fn_R", "support_side", "one_sided_mechanical"
        )}),
        "dz_entry": jsafe(entry0),
        "alpha_up_deg": gates(env.sim)["alpha_up_deg"],
    }

    # --- A1: standalone dz (must REJECT unsafe entry) ---
    print("A1 standalone dz (expect DZ_UNSAFE_ENTRY)...", flush=True)
    r_stand = correct_dz_inward(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["A1_standalone_dz"] = slim_dz(r_stand)

    # --- A2: scheduler-controlled full precision from identical IC ---
    print("A2 scheduler run_precision Plan B...", flush=True)
    res_sched = run_precision(
        env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, "B", cfg
    )
    results["A2_scheduler_plan_b"] = jsafe(res_sched.to_dict())
    results["A2_scheduler_plan_b"]["phases"] = list(res_sched.phases)
    results["A2_scheduler_plan_b"]["task_outcomes"] = jsafe(
        (res_sched.primitive_logs or {}).get("task_outcomes")
    )
    results["A2_scheduler_plan_b"]["n_actions"] = len(res_sched.actions or [])

    # --- B: gbi_063 scheduler + direct dz ---
    print("B gbi_063...", flush=True)
    restore_capture(env, blob_gbi["capture"])
    entry_g = check_dz_inward_entry(env.sim, cfg)
    r_gbi_dz = correct_dz_inward(env, {"capture": blob_gbi["capture"], "ref": blob_gbi["ref"]}, cfg)
    res_gbi = run_precision(
        env, {"capture": blob_gbi["capture"], "ref": blob_gbi["ref"]}, "C", cfg
    )
    results["B_gbi_entry"] = jsafe(entry_g)
    results["B_gbi_standalone_dz"] = slim_dz(r_gbi_dz)
    results["B_gbi_scheduler"] = jsafe(res_gbi.to_dict())
    results["B_gbi_scheduler"]["task_outcomes"] = jsafe(
        (res_gbi.primitive_logs or {}).get("task_outcomes")
    )
    results["B_gbi_scheduler"]["phases"] = list(res_gbi.phases)

    # --- C: dx then dz S1 reuse ---
    print("C dx→dz S1 reuse...", flush=True)
    st = correct_dx_support_transfer(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["C_dx"] = {
        "ok": st.get("ok"),
        "ex0": st.get("ex0"),
        "ex1": st.get("ex1"),
        "s1_valid": (st.get("s1_state") or {}).get("valid"),
        "s1_mode": (st.get("s1_state") or {}).get("mode"),
    }
    r_c = correct_dz_inward(
        env, {"capture": st["capture"], "ref": blob_b["ref"]}, cfg, s1_state=st.get("s1_state")
    )
    results["C_dz_after_dx"] = slim_dz(r_c)

    # --- D: ST dx preservation ---
    results["D_support_transfer_dx"] = results["C_dx"]

    # --- E: outward shortcut ---
    print("E outward shortcut...", flush=True)
    r_e = correct_dz_outward(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["E_outward"] = slim_dz(r_e)

    # Contact force rows
    for tag, r in (("stand", r_stand), ("gbi", r_gbi_dz), ("c_reuse", r_c)):
        for d in r.get("rotation_diag") or []:
            for t in ((d.get("selection") or {}).get("top_trials") or [])[:3]:
                force_rows.append({
                    "case": tag,
                    "phase": d.get("phase"),
                    "name": t.get("name"),
                    "sign": t.get("sign"),
                    "ok": t.get("ok"),
                    "min_fnz": t.get("min_fnz_support"),
                    "progress": t.get("progress_deg"),
                })

    # Videos
    videos = {}
    # BEFORE: prior bad standalone 04
    src04 = PRIOR / "04_dz_inward_S1_flip_AFTER.mp4"
    if src04.exists():
        dst = VID / "04_standalone_BEFORE_unsafe_entry.mp4"
        shutil.copy2(src04, dst)
        videos["04_BEFORE"] = {"path": str(dst), "note": "prior standalone dz with hidden seating"}

    print("  render 04_standalone_AFTER (rejected)...", flush=True)
    videos["04_standalone_AFTER"] = render_tape(
        env, blob_b["capture"], r_stand.get("actions") or [],
        VID / "04_standalone_AFTER_precondition_reject.mp4", "04_standalone_REJECT",
    )
    videos["04_standalone_AFTER"]["ok"] = r_stand.get("ok")
    videos["04_standalone_AFTER"]["failure_code"] = r_stand.get("failure_code")

    print("  render 04_scheduler_AFTER...", flush=True)
    videos["04_scheduler_AFTER"] = render_tape(
        env, blob_b["capture"], res_sched.actions or [],
        VID / "04_scheduler_dx_then_dz_AFTER.mp4", "04_scheduler",
    )
    videos["04_scheduler_AFTER"]["ok"] = res_sched.status
    videos["04_scheduler_AFTER"]["task_outcomes"] = results["A2_scheduler_plan_b"].get("task_outcomes")

    src07 = PRIOR / "07_C_gbi_063_AFTER.mp4"
    if src07.exists():
        shutil.copy2(src07, VID / "07_gbi_BEFORE_rotation_fix.mp4")
        videos["07_BEFORE"] = {"path": str(VID / "07_gbi_BEFORE_rotation_fix.mp4")}
    print("  render 07_gbi_AFTER...", flush=True)
    videos["07_gbi_AFTER"] = render_tape(
        env, blob_gbi["capture"], res_gbi.actions or [],
        VID / "07_C_gbi_063_AFTER.mp4", "07_gbi_sched",
    )
    videos["07_gbi_AFTER"]["ok"] = res_gbi.status

    print("  render 05_dx_dz...", flush=True)
    combo = list(st.get("actions") or []) + list(r_c.get("actions") or [])
    videos["05_dx_then_dz"] = render_tape(
        env, blob_b["capture"], combo, VID / "05_dx_then_dz_S1_reuse_AFTER.mp4", "05_dx_dz"
    )
    videos["05_dx_then_dz"]["ok"] = bool(st.get("ok") and r_c.get("ok"))
    videos["05_dx_then_dz"]["reused_s1"] = r_c.get("reused_s1")

    # Write artifacts
    (OUT / "PHYSICAL_RESULTS.json").write_text(json.dumps(jsafe(results), indent=2), encoding="utf-8")
    (VID / "INDEX.json").write_text(json.dumps(jsafe(videos), indent=2), encoding="utf-8")
    if force_rows:
        with (OUT / "CONTACT_FORCE_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(force_rows[0].keys()))
            w.writeheader()
            for row in force_rows:
                w.writerow(row)

    ic = results["plan_b_ic"]
    root = f"""# ROOT_CAUSE — `04_dz_inward_S1_flip_AFTER` failure

Generated: {datetime.now(timezone.utc).isoformat()}

## Verdict

**The standalone dz mechanism probe intentionally bypassed dx**, not a unified-scheduler
skip. Video generation in `run_precision_rotation_fix.py` rendered
`correct_dz_inward(...)` from the Plan-B IC with `|ex|≈13 mm`.

## IC (complete snapshot)

- signed e: ex={ic['errors']['dx_mm']:.3f} mm, ey={ic['errors']['dy_mm']:.3f}, ez={ic['errors']['dz_mm']:.3f}
- θ={ic['errors']['dtheta_deg']:.3f}°, e_pos={ic['errors']['e_pos_mm']:.3f} mm
- contact: {ic['contact']['mode']}, Fn_z L/R={ic['contact']['Fn_z_L']:.2f}/{ic['contact']['Fn_z_R']:.2f}
- assess_need: dx_needed={ic['need']['dx_needed']}, dz_needed={ic['need']['dz_needed']}
  (ez={ic['errors']['dz_mm']:.1f} < AXIS_V1 ez_tol=20 → dz not AXIS-flagged at entry)
- dz_entry_safe: ok={ic['dz_entry']['ok']} reasons={ic['dz_entry']['reasons']}

## Why the old `04` video looked wrong

1. Probe called `correct_dz_inward` directly → no scheduler dx stage.
2. Prior dz code **silently ran support-transfer seating** (`seat_for_UP`) inside the
   primitive, so primitive `ok=True` while the tape still began with large uncorrected ex.
3. Primitive flip success was conflated with grasp recovery / HANDOFF success.

## Fix

- `check_dz_inward_entry`: reject if `|ex| > dz_safe_ex_mm` (default 6 mm ≠ SUCCESS 12 mm).
- Remove silent ST seating from dz.
- Scheduler remains dtheta→dx→dz→HANDOFF; post-dx S1 reuse unchanged.
- Report FLIP_COMPLETED / DZ_SLIDE_COMPLETED / GRASP_RETAINED / … independently.
"""
    (OUT / "ROOT_CAUSE.md").write_text(root, encoding="utf-8")

    to_a2 = results["A2_scheduler_plan_b"].get("task_outcomes") or {}
    to_g = results["B_gbi_scheduler"].get("task_outcomes") or {}
    report = f"""# REPORT — dx/dz ordering fix

## 1. Cause of `04` failure
Standalone dz probe bypassed dx. IC had ex={ic['errors']['dx_mm']:.2f} mm
(dx_needed=True). Entry now rejected: `{r_stand.get('failure_code')}` / `{r_stand.get('reason')}`.

## 2. Scheduler path (identical IC)
- status: **{res_sched.status}**
- phases include dx before any dz: `{[p for p in res_sched.phases if 'dx' in p or 'dz' in p or 'reassess' in p][:20]}`
- task_outcomes: {json.dumps(to_a2)}
- SUCCESS_NEW={to_a2.get('SUCCESS_NEW')} SUCCESS_12S={to_a2.get('SUCCESS_12S')}
- GRASP_RETAINED={to_a2.get('GRASP_RETAINED')}

## 3. gbi_063 rotation preserved
- standalone entry ok={entry_g.get('ok')} (ex safe)
- standalone dz ok={r_gbi_dz.get('ok')} outcomes={r_gbi_dz.get('outcomes')}
- scheduler status={res_gbi.status} SUCCESS_NEW={to_g.get('SUCCESS_NEW')}

## 4. dx→dz S1 reuse
- ST ok={st.get('ok')} ex {st.get('ex0')}→{st.get('ex1')}
- dz reused_s1={r_c.get('reused_s1')} ok={r_c.get('ok')}

## 5. Outward shortcut
- skipped_rotation={r_e.get('skipped_rotation')} branch={r_e.get('branch')}

## 6. Claim
Standalone unsafe entry is blocked. Scheduler-controlled recovery from Plan B IC
must show retained grasp + applicable SUCCESS flags above before claiming solved.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "REGRESSION_REPORT.md").write_text(
        f"""# REGRESSION_REPORT

| Case | Result |
|------|--------|
| A1 standalone dz | ok={r_stand.get('ok')} code={r_stand.get('failure_code')} |
| A2 scheduler Plan B | status={res_sched.status} NEW={to_a2.get('SUCCESS_NEW')} 12S={to_a2.get('SUCCESS_12S')} grasp={to_a2.get('GRASP_RETAINED')} |
| B gbi scheduler | status={res_gbi.status} NEW={to_g.get('SUCCESS_NEW')} |
| C dx→dz reuse | st={st.get('ok')} dz={r_c.get('ok')} reused={r_c.get('reused_s1')} |
| E outward skip | skipped={r_e.get('skipped_rotation')} |
""",
        encoding="utf-8",
    )
    (OUT / "SCHEDULER_CHANGES.md").write_text(
        """# Scheduler and precondition changes

1. `check_dz_inward_entry` in `precision_dz.py` — structured `DZ_UNSAFE_ENTRY`.
2. Removed silent `seat_for_UP` / support-transfer from inside `correct_dz_inward`.
3. `PrecisionConfig.dz_safe_ex_mm=6.0` (not SUCCESS_NEW 12 mm).
4. `run_precision` logs `task_outcomes` separating FLIP vs SUCCESS_NEW.
5. Rotation planner (`support_aware_rotation.py`) unchanged.
""",
        encoding="utf-8",
    )
    print(report)
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
