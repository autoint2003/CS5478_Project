#!/usr/bin/env python
"""Audit prior A/B pivots and search for position-preserving relative pivots.

Does not alter historical results under abc_precision_architecture/.
Does not launch composition or dataset regeneration.
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
from data_generation.oracles.precision import (  # noqa: E402
    MAX_DELTA_P_REL_MM,
    audit_grasp,
    hold_and_confirm_stable,
    pivot_segment_metrics,
    primitive_axial_tail_slide,
    primitive_contact_pivot,
)
from data_generation.oracles.precision.corrections.contact_primitives import (  # noqa: E402
    MIN_THETA_IMPROVE_DEG,
    _action_vw,
    _wrist,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.overlays import caption  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "position_preserving_pivot"
VID = OUT / "videos"
PRIOR = ROOT / "results" / "diagnostics" / "abc_precision_architecture"
STATE_A = PRIOR / "stable_states" / "plan_a__ball_train_v2_0070__stable.pkl"
STATE_B = PRIOR / "stable_states" / "plan_b__hard_910220__stable.pkl"
PRIOR_PRIM = PRIOR / "primitive_results.json"

# Legacy wrist-only schedules from the previous milestone (for audit replay).
LEGACY_A = {"tag": "piv_wx0.0_wy1.0_g-0.48_s0.4", "wx": 0.0, "wy": 1.0, "grip": -0.48, "dur": 0.40}
LEGACY_B = {"tag": "piv_wx0.0_wy3.0_g1.0_s0.3", "wx": 0.0, "wy": 3.0, "grip": 1.0, "dur": 0.30}


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
    env._obs_seq = []
    env._prev_seq = []
    return env


def legacy_actions(spec) -> list:
    a_p = _wrist(spec["wx"], spec["wy"], spec["grip"])
    a_b = _wrist(0.0, 0.0, 1.0)
    return [a_p] * max(1, int(round(spec["dur"] / 0.02))) + [a_b] * 8


def advance_tail_slide(env, book, state) -> dict:
    """Reproduce prior milestone's post-tail-slide work state when available."""
    blob = {"capture": state["capture"], "ref": state["ref"]}
    full = primitive_axial_tail_slide(env, blob, book, budget=40)
    if not full.get("physically_demonstrated") or not full.get("actions"):
        return {
            "blob": blob,
            "tail_applied": False,
            "tail_meta": {k: v for k, v in full.items() if k != "actions"},
        }
    restore_capture(env, blob["capture"])
    for a in full["actions"]:
        step_legal(env.sim, np.asarray(a, np.float32))
    work = {"capture": take_capture(env), "ref": state["ref"]}
    return {
        "blob": work,
        "tail_applied": True,
        "tail_meta": {k: v for k, v in full.items() if k != "actions"},
        "geom_after_tail": audit_grasp(env.sim),
    }


def audit_legacy(env, blob, spec, label: str) -> dict:
    acts = legacy_actions(spec)
    met = pivot_segment_metrics(env, blob, acts, record_series=True)
    return {
        "label": label,
        "legacy_tag": spec["tag"],
        "n_actions": len(acts),
        "delta_p_rel_mm": met["delta_p_rel_mm"],
        "delta_p_rel_components_mm": met["delta_p_rel_components_mm"],
        "d_axis_deg": met["d_axis_deg"],
        "d_theta": met["d_theta"],
        "theta0": met["theta0"],
        "theta1": met["theta1"],
        "theta_improve_deg": float(met["theta0"] - met["theta1"]),
        "classification": met["classification"],
        "bilateral_end": met["p1"]["bilateral"],
        "contact0": {
            "grasp_kind": met["a0"].get("grasp_kind"),
            "contact_s_mean": met["a0"].get("contact_s_mean"),
            "Fn_L": met["a0"].get("Fn_L"),
            "Fn_R": met["a0"].get("Fn_R"),
            "opposing_sidewall": met["a0"].get("opposing_sidewall"),
        },
        "contact1": {
            "grasp_kind": met["a1"].get("grasp_kind"),
            "contact_s_mean": met["a1"].get("contact_s_mean"),
            "Fn_L": met["a1"].get("Fn_L"),
            "Fn_R": met["a1"].get("Fn_R"),
            "opposing_sidewall": met["a1"].get("opposing_sidewall"),
            "n_sidewall": met["a1"].get("n_sidewall"),
            "n_end_face": met["a1"].get("n_end_face"),
        },
        "series": met["series"],
        "position_preserving": met["classification"] == "position_preserving_relative_pivot",
        "note": (
            "Prior milestone accepted Δu_hand without a Δp_rel constraint; "
            "this audit reclassifies under MAX_DELTA_P_REL_MM="
            f"{MAX_DELTA_P_REL_MM}"
        ),
    }


def render_video(env, blob, actions, dest: Path, title: str):
    if not actions:
        return None
    restore_capture(env, blob["capture"])
    look = 0.5 * (
        np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.48, "azimuth": 135.0, "elevation": -20.0})
    renderer = __import__("mujoco").Renderer(env.sim.model, 480, 640)
    frames = []
    t0 = float(env.sim.data.time)

    def grab(tag):
        from data_generation.oracles.precision.corrections.contact_primitives import relative_pose

        cam.lookat[:] = 0.5 * (
            np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        renderer.update_scene(env.sim.data, cam)
        pr = relative_pose(env.sim)
        if not hasattr(grab, "p_start"):
            grab.p_start = np.asarray(pr["p_rel"], float)
        dp = 1e3 * (np.asarray(pr["p_rel"], float) - grab.p_start)
        img = np.asarray(renderer.render(), np.uint8).copy()
        frames.append(
            caption(
                img,
                f"{title} {tag} t={float(env.sim.data.time)-t0:.2f} "
                f"θ={pr['e_theta_deg']:.1f} Δp={np.linalg.norm(dp):.1f}mm "
                f"[{dp[0]:.1f},{dp[1]:.1f},{dp[2]:.1f}]",
            )
        )

    grab("START")
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 2 == 0 or i == len(actions) - 1:
            grab("RUN")
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_frames_mp4(frames, dest, fps=25)


def write_report(payload: dict):
    lines = [
        "# Position-Preserving Pivot — Focused Feasibility",
        "",
        f"Created UTC: {payload['created_utc']}",
        f"Wall: {payload['wall_s']} s",
        "",
        "## Constraint",
        "",
        f"- Primary: reduce θ (min improve {MIN_THETA_IMPROVE_DEG}°).",
        f"- Constraint: `delta_p_rel = ‖p_rel_end − p_rel_start‖ ≤ {MAX_DELTA_P_REL_MM} mm` (hand frame COM).",
        "- Hand world/Cartesian motion may compensate; object–hand relative COM must not slip.",
        "- Justification: SUCCESS_NEW e_pos band is 12 mm; pivot must not spend that budget (3 mm ≈ ¼ band).",
        "- Do **not** accept 10–15 mm relative translation as a pivot.",
        "",
        "## 1. Audit of prior A/B pivot trajectories",
        "",
        "Historical `abc_precision_architecture/` results are **not** modified.",
        "",
    ]
    for key in ("A_raw", "A_post_tail", "B_raw", "B_post_tail"):
        a = payload["audits"].get(key)
        if not a:
            continue
        lines += [
            f"### {key} (`{a['legacy_tag']}`)",
            "",
            f"- Classification: **{a['classification']}**",
            f"- Δp_rel = **{a['delta_p_rel_mm']:.2f} mm**  components = {a['delta_p_rel_components_mm']}",
            f"- Δu_hand = {a['d_axis_deg']:.2f}°, Δθ = {a['d_theta']:.2f}° (θ {a['theta0']:.1f}→{a['theta1']:.1f})",
            f"- Bilateral end: {a['bilateral_end']}; contact {a['contact0'].get('grasp_kind')} → {a['contact1'].get('grasp_kind')}",
            f"- Position-preserving under new criterion: {a['position_preserving']}",
            "",
        ]
    lines += [
        "## 2. New position-preserving search",
        "",
    ]
    for key, r in payload["searches"].items():
        lines += [
            f"### {key}",
            "",
            f"- Demonstrated: **{r.get('physically_demonstrated')}**",
            f"- Screened: {r.get('n_screened')}",
            f"- Best demonstrated: `{jsafe(r.get('best'))}`",
            f"- Best any: `{jsafe(r.get('best_any') or r.get('failure'))}`",
            "",
        ]
    feas = payload["feasibility_statement"]
    lines += [
        "## 3. Feasibility statement",
        "",
        feas,
        "",
        "## 4. Composition",
        "",
        "Not launched — pivot primitive must be demonstrated first.",
        "",
        "See `audit.json`, `search.json`, `videos/`.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = make_env()
    book = load_codebook()

    state_a = pickle.loads(STATE_A.read_bytes())
    state_b = pickle.loads(STATE_B.read_bytes())

    audits = {}
    # Plan A: audit on raw stable + post-tail (if available)
    blob_a = {"capture": state_a["capture"], "ref": state_a["ref"]}
    audits["A_raw"] = audit_legacy(env, blob_a, LEGACY_A, "A_raw")
    adv_a = advance_tail_slide(env, book, state_a)
    audits["A_post_tail"] = audit_legacy(env, adv_a["blob"], LEGACY_A, "A_post_tail")
    audits["A_post_tail"]["tail_applied"] = adv_a["tail_applied"]

    # Plan B: prior ~12.7° claim was on post-tail work state
    blob_b = {"capture": state_b["capture"], "ref": state_b["ref"]}
    audits["B_raw"] = audit_legacy(env, blob_b, LEGACY_B, "B_raw")
    adv_b = advance_tail_slide(env, book, state_b)
    audits["B_post_tail"] = audit_legacy(env, adv_b["blob"], LEGACY_B, "B_post_tail")
    audits["B_post_tail"]["tail_applied"] = adv_b["tail_applied"]
    if adv_b.get("geom_after_tail"):
        audits["B_post_tail"]["geom_after_tail"] = {
            k: adv_b["geom_after_tail"][k]
            for k in ("e_theta_deg", "e_pos_mm", "contact_s_mean", "grasp_kind", "e_signed_mm")
            if k in adv_b["geom_after_tail"]
        }

    # Videos of legacy B post-tail (the disputed 12.7° case)
    render_video(
        env,
        adv_b["blob"],
        legacy_actions(LEGACY_B),
        VID / "audit_B_post_tail_legacy_pivot.mp4",
        "AUDIT_B",
    )

    # New searches: prefer verified tail-ish states when available
    searches = {}
    for key, blob, note in (
        ("A_raw", blob_a, "stable Plan A mid-grasp"),
        ("A_post_tail", adv_a["blob"], "Plan A after tail-slide attempt"),
        ("B_raw", blob_b, "stable Plan B post-B1"),
        ("B_post_tail", adv_b["blob"], "Plan B after tail-slide (prior pivot start)"),
    ):
        print(f"Search position-preserving pivot on {key} ({note})...", flush=True)
        # Larger budget for the critical post-tail B case
        budget = 120 if key == "B_post_tail" else (72 if "post_tail" in key else 48)
        res = primitive_contact_pivot(env, blob, book, budget=budget)
        searches[key] = {k: v for k, v in res.items() if k != "actions"}
        searches[key]["note"] = note
        print(
            f"  demo={res['physically_demonstrated']} "
            f"best_any_class={(res.get('best_any') or {}).get('classification')} "
            f"Δp={(res.get('best_any') or {}).get('delta_p_rel_mm')} "
            f"dθ={(res.get('best_any') or {}).get('theta_improve_deg')}",
            flush=True,
        )
        if res.get("physically_demonstrated") and res.get("actions"):
            render_video(env, blob, res["actions"], VID / f"ppivot_{key}_best.mp4", f"PPIV_{key}")
            # Stability after pivot
            restore_capture(env, blob["capture"])
            for a in res["actions"]:
                step_legal(env.sim, np.asarray(a, np.float32))
            stab = hold_and_confirm_stable(env, hold_s=0.20)
            searches[key]["post_pivot_stable"] = {
                "stable": stab.get("stable"),
                "reasons": stab.get("reasons"),
                "e_theta_deg": stab.get("e_theta_deg"),
                "e_pos_mm": stab.get("e_pos_mm"),
            }

    any_demo = any(s.get("physically_demonstrated") for s in searches.values())
    if any_demo:
        feas = (
            "Physically feasible under current legal 7D / contact constraints for at least one "
            "stable-tail snapshot (see searches with physically_demonstrated=true). "
            "Still not claimed as generalized across geometries."
        )
    else:
        # Characterize dominant failure from B_post_tail
        ba = (searches.get("B_post_tail") or {}).get("best_any") or {}
        feas = (
            "Not demonstrated: no legal short-horizon schedule produced ≥"
            f"{MIN_THETA_IMPROVE_DEG}° θ improvement with Δp_rel≤{MAX_DELTA_P_REL_MM} mm "
            "while retaining bilateral contact. "
            f"Best B_post_tail attempt class={ba.get('classification')} "
            f"Δp_rel={ba.get('delta_p_rel_mm')} mm θ_improve={ba.get('theta_improve_deg')}°. "
            "Prior ~12.7° Plan B motion is reclassified as coupled rotation–translation "
            "(or similar), not a position-preserving precision pivot."
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "thresholds": {
            "MAX_DELTA_P_REL_MM": MAX_DELTA_P_REL_MM,
            "MIN_THETA_IMPROVE_DEG": MIN_THETA_IMPROVE_DEG,
        },
        "prior_results_preserved": str(PRIOR),
        "audits": audits,
        "searches": searches,
        "feasibility_statement": feas,
        "composition_launched": False,
    }
    (OUT / "audit.json").write_text(json.dumps(jsafe(audits), indent=2), encoding="utf-8")
    (OUT / "search.json").write_text(json.dumps(jsafe(searches), indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(jsafe(payload), indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps(jsafe({"feasibility": feas, "any_demo": any_demo, "wall_s": payload["wall_s"]}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
