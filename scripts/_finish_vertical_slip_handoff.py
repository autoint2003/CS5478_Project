"""Continue from saved fingertips-up pose: 2mm/5mm slip → restore → HANDOFF → update REPORT."""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.precision import run_finger_supported_vertical_axial as R  # noqa: E402

OUT = R.OUT
blob_post = R.load_blob(R.B_POST)
vert = pickle.loads((OUT / "snapshots/best_vertical.pkl").read_bytes())
vert["ref"] = blob_post["ref"]
best_axis = vert.get("axis") or "-wz"

env = R.make_env()
R.restore_capture(env, vert["capture"])
gate = R.vertical_gate_report(env.sim)
print(
    f"gate α={gate['alpha_open_up_deg']:.2f} g·d={gate['g_dot_d_inward']:.2f} "
    f"e={np.round(gate['e_signed_mm'],2)} bil={gate['bilateral']}",
    flush=True,
)
assert gate["fingertips_up_ok"] and gate["gravity_assists_inward"]

d_in_h = -R.U_OPEN_HAND
z_ref = float(gate["obj_z"])
e_x0 = float(gate["e_signed_mm"][0])
tilt = {"capture": vert["capture"], "ref": vert["ref"]}

t0 = time.time()
print("=== 2 mm slip ===", flush=True)
slip = R.run_axial_slip(env, tilt, d_in_h, milestone_mm=2.0, z_ref=z_ref, e_x0=e_x0)
print(
    f"  ok={slip.get('ok')} disp={slip.get('disp_along_mm')} "
    f"abort={slip.get('abort_reason')} e={slip.get('e_signed')}",
    flush=True,
)
(OUT / "axial_slip_2mm.json").write_text(
    json.dumps(R.jsafe({k: v for k, v in slip.items() if k not in ("actions", "series", "capture")}), indent=2),
    encoding="utf-8",
)

slip5 = None
handoff = None
restore_meta = None
replay = None
acts_extra = []

if slip.get("ok"):
    print("=== 5 mm slip ===", flush=True)
    tilt5 = {"capture": slip["capture"], "ref": vert["ref"]}
    slip5 = R.run_axial_slip(
        env, tilt5, d_in_h, milestone_mm=5.0, z_ref=z_ref, e_x0=float(slip["e_signed"][0])
    )
    print(f"  ok={slip5.get('ok')} disp={slip5.get('disp_along_mm')}", flush=True)
    (OUT / "axial_slip_5mm.json").write_text(
        json.dumps(
            R.jsafe({k: v for k, v in slip5.items() if k not in ("actions", "series", "capture")}),
            indent=2,
        ),
        encoding="utf-8",
    )
    post_cap = slip5["capture"] if slip5.get("ok") else slip["capture"]
    post = {"capture": post_cap, "ref": vert["ref"]}
    rplan = R.plan_restore_orientation(env, post)
    rexe = R.execute_restore_plan(env, post, rplan)
    restore_meta = {
        "plan_ok": rplan.get("ok"),
        "already_upright": rplan.get("already_upright"),
        "axis_name": rplan.get("axis_name"),
        "angle_deg": rplan.get("angle_deg"),
        "exe_ok": rexe.get("ok"),
    }
    handoff = R.handoff_rich(env, R.take_capture(env), vert["ref"])
    print(
        f"HANDOFF NEW={handoff.get('SUCCESS_NEW')} AXIS={handoff.get('SUCCESS_AXIS_V1')} "
        f"12S={handoff.get('SUCCESS_12S')} e={handoff.get('pre_e_signed_mm')}",
        flush=True,
    )
    (OUT / "handoff.json").write_text(
        json.dumps(
            R.jsafe(
                {
                    k: handoff.get(k)
                    for k in (
                        "SUCCESS_12S",
                        "SUCCESS_NEW",
                        "SUCCESS_AXIS_V1",
                        "pre_e_signed_mm",
                        "pre_theta",
                        "pre_pos_mm",
                        "reason",
                    )
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    acts_extra = list(slip.get("actions") or [])
    if slip5 and slip5.get("ok"):
        acts_extra += list(slip5.get("actions") or [])
    acts_extra += list(rexe.get("actions") or [])

    # Load prior rotation actions from path json isn't enough — re-run path briefly for npz
    # Use pickled path actions if present; else recompute Stage A+B for best_axis
    from data_generation.oracles.precision.finger_supported_vertical import run_support_then_vertical

    print("=== Rebuild complete action chain for replay ===", flush=True)
    path = run_support_then_vertical(
        env, blob_post, support_axis=best_axis, support_angle_deg=90.0, support_rate=0.18, tip_rate=0.15
    )
    all_acts = list(path.get("actions") or []) + acts_extra
    np.savez_compressed(OUT / "best_actions.npz", actions=np.asarray(all_acts, np.float32))

    if handoff.get("SUCCESS_NEW"):
        print("=== IC 3× frozen replay ===", flush=True)
        prefix = []
        if R.STABLE_JSON.exists():
            stj = json.loads(R.STABLE_JSON.read_text(encoding="utf-8"))
            prefix = [np.asarray(a, np.float32) for a in stj.get("stabilization_prefix_actions") or []]
        lat = np.load(R.AXIS_PREFIX)["actions"] if R.AXIS_PREFIX.exists() else []
        mb = np.load(OUT / "best_actions.npz")["actions"]
        trials = []
        for _ in range(3):
            root = pickle.loads(R.IC_B.read_bytes())
            R.restore_capture(env, root["capture"])
            for a in prefix:
                R.step_legal(env.sim, a)
            for a in lat:
                R.step_legal(env.sim, a)
            for a in mb:
                R.step_legal(env.sim, a)
            hr = R.handoff_rich(env, R.take_capture(env), vert["ref"])
            trials.append(
                {
                    "SUCCESS_12S": hr.get("SUCCESS_12S"),
                    "SUCCESS_NEW": hr.get("SUCCESS_NEW"),
                    "SUCCESS_AXIS_V1": hr.get("SUCCESS_AXIS_V1"),
                    "pre_e_signed_mm": hr.get("pre_e_signed_mm"),
                }
            )
        replay = {
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
        }
        (OUT / "replay_ic.json").write_text(json.dumps(R.jsafe(replay), indent=2), encoding="utf-8")
        print(replay, flush=True)

# Update summary / REPORT
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
if slip.get("ok") and handoff and handoff.get("SUCCESS_NEW"):
    rec = "fingertips_up_slip_handoff_SUCCESS_NEW"
elif slip.get("ok"):
    rec = "fingertips_up_2mm_slip_ok_handoff_limited"
else:
    rec = "fingertips_up_reached_slip_failed"

summary.update(
    {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finish_wall_s": round(time.time() - t0, 1),
        "slip_2mm": {
            k: slip.get(k)
            for k in ("ok", "disp_along_mm", "abort_reason", "reason", "lateral_drift_mm", "e_signed")
        },
        "slip_5mm": None
        if not slip5
        else {k: slip5.get(k) for k in ("ok", "disp_along_mm", "abort_reason", "reason", "e_signed")},
        "restore": restore_meta,
        "handoff": None
        if not handoff
        else {
            k: handoff.get(k)
            for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta")
        },
        "replay_ic": replay,
        "recommendation": rec,
        "slip_controller": {
            "FN_MIN": R.FN_MIN,
            "FN_MAX": R.FN_MAX,
            "FN_INIT": R.FN_INIT,
            "SLIP_KN": R.SLIP_KN,
            "note": "Switched from stiff FN_MIN=6.5 (stuck) to sign_aware DualLoop after vertical gate.",
        },
    }
)
(OUT / "summary.json").write_text(json.dumps(R.jsafe(summary), indent=2), encoding="utf-8")

# Append slip segment video of complete chain if we rebuilt path
if slip.get("ok") and (OUT / "best_actions.npz").exists():
    print("Rendering complete rotation+slip video...", flush=True)
    acts = np.load(OUT / "best_actions.npz")["actions"]
    vid = R.render_path_video(
        env,
        blob_post,
        [np.asarray(a, np.float32) for a in acts],
        OUT / "videos" / "complete_rotation_slip_handoff.mp4",
        f"{best_axis} → vertical → slip → restore",
    )
    summary["videos"] = {**(summary.get("videos") or {}), "complete": vid}
    (OUT / "summary.json").write_text(json.dumps(R.jsafe(summary), indent=2), encoding="utf-8")

report = f"""# Finger-Supported → Fingertips-Up → Axial Slip — Plan B hard_910220

Created UTC: {summary['created_utc']}

## Verdict

**{rec}**

Sequence: Stage A `±wz@90°` pad support → Stage B closed-loop tip to `α≤5°` → DualLoop inward slip.
Slip is **not** attempted at the intermediate pad-up pose.

| Path | A→B ok | α_final (°) | g·d_inward | Stage A Fn_z |
|------|--------|-------------|------------|--------------|
| +wz | {summary['paths']['+wz']['ok']} | {summary['paths']['+wz']['alpha_final']:.2f} | {summary['paths']['+wz']['g_dot_d_inward']:.2f} | {summary['paths']['+wz']['stage_A_Fn_z']:.2f} |
| -wz | {summary['paths']['-wz']['ok']} | {summary['paths']['-wz']['alpha_final']:.2f} | {summary['paths']['-wz']['g_dot_d_inward']:.2f} | {summary['paths']['-wz']['stage_A_Fn_z']:.2f} |

Best path: `{summary['best_axis']}`  Vertical gate: **PASSED**

## Vertical gate (pre-slip)

α_open_up = {gate['alpha_open_up_deg']:.2f}°  
g·d_inward = {gate['g_dot_d_inward']:.2f}  
e = {np.round(gate['e_signed_mm'], 2).tolist()}  θ = {gate['theta_deg']:.2f}°  
fn L/R = {gate['fn_L']:.2f}/{gate['fn_R']:.2f}  v_rel = {gate['v_rel_norm']:.4f}  
height margin = {gate['world_height_margin']:.3f} m

## Axial slip

Slip 2 mm: {json.dumps(R.jsafe(summary.get('slip_2mm')), indent=2)}

Slip 5 mm: {json.dumps(R.jsafe(summary.get('slip_5mm')), indent=2)}

Controller: FN_MIN={R.FN_MIN}, KN={R.SLIP_KN} (stiff FN_MIN=6.5 failed with disp≈0.17 mm at same pose).

## HANDOFF / replay

Handoff: {json.dumps(R.jsafe(summary.get('handoff')), indent=2)}

Replay IC: {json.dumps(R.jsafe(replay), indent=2) if replay else 'not run'}

## Videos

{json.dumps(R.jsafe(summary.get('videos')), indent=2)}

## Notes

- `wz` alone cannot tip `u_open`; Stage B uses `ω_body = R_des.T @ ((u_open×up)·Ω)`.
- Pad-up is intermediate support; at fingertips-up, pad normals are near-horizontal again and gravity drives inward.
- Preserved SUCCESS_NEW / AXIS_V1 / Plan A / Branch A. No physics/obs/action/nominal changes.
"""
(OUT / "REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps(R.jsafe({"recommendation": rec, "slip2": summary.get("slip_2mm"), "handoff": summary.get("handoff")}), indent=2), flush=True)
