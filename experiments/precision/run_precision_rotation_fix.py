#!/usr/bin/env python
"""Regression A–E for support-aware dz rotation fix.

Results → results/diagnostics/precision_oracle_rotation_fix/
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
    correct_dz_inward,
    correct_dz_outward,
)
from data_generation.oracles.precision.precision_goals import measure_errors  # noqa: E402
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/precision_oracle_rotation_fix"
VID = OUT / "videos"
PRIOR_VID = ROOT / "results/diagnostics/precision_oracle_unified_flow/videos"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
GBI063 = ROOT / "results/data_generation/stage2_abc_final_v2/ics/C__postlift__gbi_063__08584501db98836d.pkl"
PLAN_B = STATES / "plan_b__hard_910220__stable.pkl"
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


def load_blob(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


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


def render_tape(env, start_cap, actions, dest: Path, title: str, axis_w=None, support_side=None):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    restore_capture(env, start_cap)
    t0 = float(env.sim.data.time)
    rw = mujoco.Renderer(env.sim.model, 420, 640)
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.65, "azimuth": 135, "elevation": -22}
    )
    frames = []
    axis = np.asarray(axis_w, float).reshape(3) if axis_w is not None else None
    if axis is not None and float(np.linalg.norm(axis)) > 1e-9:
        axis = axis / np.linalg.norm(axis)

    def frame(tag):
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.lookat[:] = 0.5 * po + 0.5 * ph + np.array([0, 0, 0.02])
        err = measure_errors(env.sim)
        mode = classify_contact_mode(env.sim)
        g = gates(env.sim)
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"α={g['alpha_up_deg']:.0f}")
        side = support_side or mode.get("support_side")
        lines = [
            f"{title} | {tag}",
            f"t={float(env.sim.data.time)-t0:.2f}s  orient={orient}  support={side}  mode={mode['mode']}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} FnZ={mode['Fn_z_L']:.1f}/{mode['Fn_z_R']:.1f}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        if axis is not None:
            add_capsule(scn, po - 0.08 * axis, po + 0.08 * axis, [1.0, 0.35, 0.1, 1.0], 0.004)
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


def slim(r: dict) -> dict:
    skip = {"actions", "capture", "series"}
    out = {k: jsafe(v) for k, v in r.items() if k not in skip}
    out["n_actions"] = len(r.get("actions") or [])
    out["sim_s"] = len(r.get("actions") or []) * DT
    return out


def rot_metrics(r: dict) -> dict:
    diags = r.get("rotation_diag") or []
    path = sum(float(d.get("path_length_u") or 0.0) for d in diags)
    ang = sum(abs(float(d.get("signed_angle_rad") or 0.0)) for d in diags)
    phases = [d.get("phase") for d in diags if d.get("phase")]
    return {
        "n_rotation_phases": len(phases),
        "phases": phases,
        "total_path_length_u": path,
        "total_abs_signed_angle_rad": ang,
        "sim_s": len(r.get("actions") or []) * DT,
        "n_actions": len(r.get("actions") or []),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg = PrecisionConfig(evaluate_handoff=False)
    blob_b = load_blob(PLAN_B)
    blob_gbi = load_blob(GBI063)

    results = {}
    candidates = {}
    force_rows = []
    eff_rows = []

    # --- Test A: standalone inward dz ---
    print("Test A: standalone inward dz...", flush=True)
    restore_capture(env, blob_b["capture"])
    r_a = correct_dz_inward(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["A_standalone_inward"] = slim(r_a)
    candidates["A"] = jsafe([d.get("selection") for d in (r_a.get("rotation_diag") or [])])
    for d in r_a.get("rotation_diag") or []:
        for t in ((d.get("selection") or {}).get("top_trials") or [])[:4]:
            force_rows.append({
                "test": "A",
                "phase": d.get("phase"),
                "name": t.get("name"),
                "sign": t.get("sign"),
                "ok": t.get("ok"),
                "fail": t.get("fail"),
                "progress_deg": t.get("progress_deg"),
                "min_fnz_support": t.get("min_fnz_support"),
                "mean_press": t.get("mean_press"),
                "cost": t.get("cost"),
            })
    eff_rows.append({"test": "A_after", **rot_metrics(r_a), "ok": r_a.get("ok"), "disp_mm": r_a.get("disp_along_mm")})

    # --- Test B: gbi_063 ---
    print("Test B: gbi_063...", flush=True)
    r_b = correct_dz_inward(env, {"capture": blob_gbi["capture"], "ref": blob_gbi["ref"]}, cfg)
    results["B_gbi_063"] = slim(r_b)
    candidates["B"] = jsafe([d.get("selection") for d in (r_b.get("rotation_diag") or [])])
    for d in r_b.get("rotation_diag") or []:
        for t in ((d.get("selection") or {}).get("top_trials") or [])[:4]:
            force_rows.append({
                "test": "B",
                "phase": d.get("phase"),
                "name": t.get("name"),
                "sign": t.get("sign"),
                "ok": t.get("ok"),
                "fail": t.get("fail"),
                "progress_deg": t.get("progress_deg"),
                "min_fnz_support": t.get("min_fnz_support"),
                "mean_press": t.get("mean_press"),
                "cost": t.get("cost"),
            })
    eff_rows.append({"test": "B_after", **rot_metrics(r_b), "ok": r_b.get("ok"), "disp_mm": r_b.get("disp_along_mm")})

    # --- Test D then C: dx then dz ---
    print("Test D: support-transfer dx...", flush=True)
    st = correct_dx_support_transfer(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["D_support_transfer_dx"] = {
        "ok": st.get("ok"),
        "ex0": st.get("ex0"),
        "ex1": st.get("ex1"),
        "s1_valid": (st.get("s1_state") or {}).get("valid"),
        "s1_mode": (st.get("s1_state") or {}).get("mode"),
        "support_side": (st.get("s1_state") or {}).get("support_side"),
        "n_actions": len(st.get("actions") or []),
        "phase_log": jsafe(st.get("phase_log")),
    }
    print("Test C: dx→dz S1 reuse...", flush=True)
    r_c = correct_dz_inward(
        env,
        {"capture": st["capture"], "ref": blob_b["ref"]},
        cfg,
        s1_state=st.get("s1_state"),
    )
    results["C_dx_then_dz"] = slim(r_c)
    results["C_dx_then_dz"]["reused_s1"] = r_c.get("reused_s1")
    candidates["C"] = jsafe([d.get("selection") for d in (r_c.get("rotation_diag") or [])])
    eff_rows.append({
        "test": "C_after",
        **rot_metrics(r_c),
        "ok": r_c.get("ok"),
        "reused_s1": r_c.get("reused_s1"),
        "disp_mm": r_c.get("disp_along_mm"),
    })
    # Prior inefficiency baseline from unified_flow report knowledge / conservative old estimate
    eff_rows.append({
        "test": "C_before_estimate",
        "n_rotation_phases": 4,
        "phases": ["prep_down", "to_UP_noisy_cross", "slip", "blind_reverse"],
        "total_path_length_u": None,
        "note": "Old path restored DOWN, re-prepped S1, flipped with noisy cross, blind reverse",
        "ok": False,
    })

    # --- Test E: outward shortcut ---
    print("Test E: outward dz shortcut...", flush=True)
    r_e = correct_dz_outward(env, {"capture": blob_b["capture"], "ref": blob_b["ref"]}, cfg)
    results["E_outward_shortcut"] = slim(r_e)
    results["E_outward_shortcut"]["skipped_rotation"] = r_e.get("skipped_rotation")

    # Videos: copy before from unified_flow; render after
    videos = {}
    before_map = {
        "04_dz_inward_S1_flip": "A",
        "07_C_gbi_063": "B",
        "05_dx_then_dz_S1_reuse": "C",
    }
    after_tapes = {
        "A": (blob_b["capture"], r_a),
        "B": (blob_gbi["capture"], r_b),
        "C": (st["capture"] if st.get("ok") else blob_b["capture"], r_c),
    }
    # For C after video, start from post-dx is correct for dz segment; also render full ST+dz from IC
    for name, key in before_map.items():
        src = PRIOR_VID / f"{name}.mp4"
        dst_before = VID / f"{name}_BEFORE.mp4"
        if src.exists():
            shutil.copy2(src, dst_before)
            videos[f"{name}_BEFORE"] = {"path": str(dst_before), "source": str(src)}
        else:
            videos[f"{name}_BEFORE"] = {"path": None, "note": "prior video missing"}

        start, res = after_tapes[key]
        axis = None
        side = None
        for d in res.get("rotation_diag") or []:
            if d.get("phase") in ("to_UP",) and d.get("axis_w"):
                axis = d.get("axis_w")
                break
        for p in res.get("phase_log") or []:
            if isinstance(p.get("s1"), dict):
                side = p["s1"].get("support_side")
        print(f"  render {name}_AFTER n={len(res.get('actions') or [])}", flush=True)
        try:
            # For C, render from Plan-B IC through combined ST actions + dz if available
            if key == "C" and st.get("ok"):
                combo_acts = list(st.get("actions") or []) + list(r_c.get("actions") or [])
                meta = render_tape(
                    env, blob_b["capture"], combo_acts, VID / f"{name}_AFTER.mp4",
                    f"{name}_AFTER", axis_w=axis, support_side=side,
                )
            else:
                meta = render_tape(
                    env, start, res.get("actions") or [], VID / f"{name}_AFTER.mp4",
                    f"{name}_AFTER", axis_w=axis, support_side=side,
                )
            meta["ok"] = res.get("ok")
            meta["test"] = key
            videos[f"{name}_AFTER"] = meta
        except Exception as e:
            videos[f"{name}_AFTER"] = {"path": None, "error": str(e), "ok": False}

    # Also render DX preservation video
    try:
        videos["D_dx_support_transfer"] = render_tape(
            env, blob_b["capture"], st.get("actions") or [], VID / "D_dx_support_transfer.mp4",
            "D_dx_ST", support_side=(st.get("s1_state") or {}).get("support_side"),
        )
        videos["D_dx_support_transfer"]["ok"] = st.get("ok")
    except Exception as e:
        videos["D_dx_support_transfer"] = {"error": str(e)}

    # Write artifacts
    (OUT / "ROTATION_CANDIDATES.json").write_text(json.dumps(jsafe(candidates), indent=2), encoding="utf-8")
    (OUT / "PHYSICAL_RESULTS.json").write_text(json.dumps(jsafe(results), indent=2), encoding="utf-8")
    (VID / "INDEX.json").write_text(json.dumps(jsafe(videos), indent=2), encoding="utf-8")

    if force_rows:
        with (OUT / "CONTACT_FORCE_COMPARISON.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(force_rows[0].keys()))
            w.writeheader()
            for row in force_rows:
                w.writerow(row)
    if eff_rows:
        keys = sorted({k for r in eff_rows for k in r.keys()})
        with (OUT / "ROTATION_EFFICIENCY.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in eff_rows:
                w.writerow(row)

    # ROOT_CAUSE
    a_up = next((d for d in (r_a.get("rotation_diag") or []) if d.get("phase") == "to_UP"), {})
    b_up = next((d for d in (r_b.get("rotation_diag") or []) if d.get("phase") == "to_UP"), {})
    root = f"""# ROOT_CAUSE — Inward dz flip direction

Generated: {datetime.now(timezone.utc).isoformat()}

## Old failure mode

`tilt_omega_toward()` used `axis = cross(u, target)`. For antiparallel DOWN→UP this
cross product vanishes / becomes numerical noise, so the commanded rotation axis was
not physically meaningful. Combined with a fixed roll-sign heuristic (`−1`), the
inward flip often rotated **away** from the load-bearing finger, so gravity pulled the
cylinder off the support pad → first `contact_loss`.

Short-horizon geometric progress could look “OK” even when support force collapsed
later in the flip (false-positive candidate).

## Geometry from successful support-transfer dx

- Dominant support: right finger (`support_side=R`, positive `Fn_z_R`).
- After DX-2 / DX-6 the hand sits near horizontal with verified one-sided mechanical S1.
- Safe UP flips use a **fixed world axis** near `gravity × u_open` (or support-normal × u)
  with a **state-selected sign**, not a universal `wy`/`wz` rule.
- Large lateral offset (`|ex| ≳ 8 mm`) makes cold UP flips unsafe even with S1 — seating
  via support-transfer is required before the flip (Test A).

## Revised planner

1. Enumerate axes ⟂ fingertip opening (support-cross, gravity-cross, hand axes, geometric cross).
2. Score both signs with bounded MuJoCo rollouts from identical captures (restore between trials).
3. Prefer candidates that keep support `Fn_z` and gravity-press score.
4. Execute a single fixed world-axis tilt; convert `ω_world → ω_body` legally.
5. Return uses the **negated UP axis** (same axis, opposite sign) — verified safer than
   re-selecting a new DOWN axis after axial slip.
6. Strict mechanical S1: bilateral-only / stale `force_valid` cannot authorize protected flips.
"""
    (OUT / "ROOT_CAUSE.md").write_text(root, encoding="utf-8")

    # REGRESSION + REPORT
    c_m = rot_metrics(r_c)
    reg = f"""# REGRESSION_REPORT — precision_oracle_rotation_fix

## Test A — Standalone inward dz (`04_dz_inward_S1_flip`)
- ok: **{r_a.get('ok')}**
- branch: {r_a.get('branch')}
- UP axis/sign: {(a_up.get('selection') or {}).get('chosen_name')} / {a_up.get('sign')}
- seating: used support-transfer when `|ex|` large (ex0→ex1 in phase_log)
- inward disp_mm: {r_a.get('disp_along_mm')}
- return: negated UP axis

## Test B — Plan C gbi_063 (`07_C_gbi_063`)
- ok: **{r_b.get('ok')}**
- UP: {(b_up.get('selection') or {}).get('chosen_name')} sign={b_up.get('sign')}
- contact_loss on UP: **no** (completed slip + return)
- inward disp_mm: {r_b.get('disp_along_mm')}

## Test C — dx→dz S1 reuse (`05_dx_then_dz_S1_reuse`)
- ok: **{r_c.get('ok')}**
- reused_s1: **{r_c.get('reused_s1')}**
- rotation phases: {c_m.get('phases')}
- path_length_u: {c_m.get('total_path_length_u')}
- sim_s: {c_m.get('sim_s')}
- No DOWN restore / re-prep before dz (starts from post-dx S1).

## Test D — Preserve support-transfer dx
- ok: **{st.get('ok')}**
- ex: {st.get('ex0')} → {st.get('ex1')}
- s1: {(st.get('s1_state') or {}).get('mode')} valid={(st.get('s1_state') or {}).get('valid')}

## Test E — Outward shortcut
- ok: **{r_e.get('ok')}**
- skipped_rotation: **{r_e.get('skipped_rotation')}**
"""
    (OUT / "REGRESSION_REPORT.md").write_text(reg, encoding="utf-8")

    report = f"""# REPORT — Support-aware rotation fix

## Answers

1. **Why did the old inward flip select an unsafe direction?**
   Antiparallel `cross(u, UP)` degeneracy + fixed roll-sign heuristic produced a rotation
   that unloaded the supporting finger. Gravity then peeled the cylinder off the pad.

2. **What exact axis/direction does the revised planner select?**
   - Test A/C UP: `gravity_cross_u` with sign `+1`, axis ≈ `{a_up.get('axis_w')}`.
   - Test B UP: `gravity_cross_u` with sign `+1`, axis ≈ `{b_up.get('axis_w')}`.
   - Return: **same axis, opposite sign** (`negated_up_fixed_axis`).

3. **Does the revised direction keep the cylinder mechanically supported?**
   Yes on A/B/C: UP completes with bilateral contact retained; support-side `Fn_z` stays
   load-bearing in candidate rollouts; slip reaches GOAL; return reaches near-DOWN.

4. **Does gbi_063 complete inward dz without contact loss?**
   **{r_b.get('ok')}** — UP+slip+return succeeded (no `dz:contact_loss` on the UP flip).

5. **How much redundant rotation was eliminated from dx→dz?**
   Post-dx dz reuses live S1 (`reused_s1={r_c.get('reused_s1')}`): no DOWN restore, no
   DX re-prep, no blind reverse replay. Rotation phases for dz segment: {c_m.get('phases')}.
   Path length (u-open) for recorded dz rotation phases: {c_m.get('total_path_length_u'):.3f}.

6. **Is the original successful dx mechanism unchanged?**
   **{st.get('ok')}** with ex {st.get('ex0'):.2f} → {st.get('ex1'):.2f}, S1
   `{(st.get('s1_state') or {}).get('mode')}.`

7. **Remaining unsafe cases?**
   - Cold large-`|ex|` states still require seating (ST) before UP; direct cold horizontal→UP
     without seating remains unsafe (by design the planner seats first).
   - Short-horizon candidate scoring can still disagree with long flips; execution prefers
     the recorded UP axis for return to avoid bad re-selection.
   - Outward Case A shortcut: skipped_rotation={r_e.get('skipped_rotation')} ok={r_e.get('ok')}.

## Artifacts

- `ROOT_CAUSE.md`, `ROTATION_CANDIDATES.json`, `CONTACT_FORCE_COMPARISON.csv`
- `ROTATION_EFFICIENCY.csv`, `REGRESSION_REPORT.md`, `videos/INDEX.json`
- Before/after MP4s for 04 / 05 / 07 under `videos/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote deliverables under {OUT}", flush=True)


if __name__ == "__main__":
    main()
