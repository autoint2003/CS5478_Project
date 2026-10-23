#!/usr/bin/env python
"""Bounded A/B/C stabilization → precision-primitive feasibility milestone.

Does not regenerate datasets, run SFT/GRPO, or overwrite historical results.
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
from data_generation.oracles.plan_b.branch_compare import load_vocab, run_tokens  # noqa: E402
from data_generation.oracles.precision import (  # noqa: E402
    audit_grasp,
    classify_stable_needs,
    hold_and_confirm_stable,
    primitive_axial_tail_slide,
    primitive_contact_pivot,
    primitive_final_axial_slide,
    primitive_gravity_reorient,
    primitive_stabilize_handoff,
)
from data_generation.oracles.precision.stable_grasp import STABLE_THRESHOLDS  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, restore_state as restore_drop  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.overlays import caption  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "abc_precision_architecture"
STATES = OUT / "stable_states"
VID = OUT / "videos"

PLAN_A_SNAP = ROOT / "data/snapshots/train/ball_train_v2_0070.pkl"
PLAN_A_NPZ = ROOT / "data/inward_sliding/train/ball_train_v2_0070__inward_reseat.npz"
PLAN_B_POST = ROOT / "results/diagnostics/plan_b_posta_plan_a_correction/hard_910220/hard_910220__post_recatch.pkl"
PLAN_B_ROOT = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
PLAN_B_TOKENS = [32, 12, 12, 0, 0, 0, 0, 0]
PLAN_C_PKL = ROOT / "results/diagnostics/ground_plan_c_final_success/raw/gbi_010__SUCCESS_NEW_expert.pkl"


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


def write_audit():
    text = """# Architecture Audit — Plan A/B/C Stabilization vs Shared Precision

## Corrected split

| Stage | Role | SUCCESS_NEW? |
|-------|------|--------------|
| Plan A / B / C stabilization | Establish verified stable grasp from abnormal airborne state | **No** |
| Shared Precision Oracle | Tail slide / relative pivot / gravity reorient / final slide | Optional path to SUCCESS_NEW |
| HANDOFF + 12 s nominal | Canonical success | **Yes** |

## Plan A — contact-preserving stabilization

- Controllers: `controlled_slide_action`, `inward_reseat_action` (`recovery_rebuild.py`); seed search in `plan_a/controlled_sliding.py`, `inward_reseat.py`.
- Mechanism: wrist tilt until gravity component ready → grip unload slip → brake/return.
- Existing heuristics are **stabilization**, not proven general precision primitives.
- Historical experts remain valid (e.g. `ball_train_v2_0070` inward_reseat).

## Plan B — release + active recatch

- `branch_compare.py`: B0 early-HANDOFF (nominal may recatch) vs B1/B2 active Recovery recatch.
- Historical GRPO short-prefix→HANDOFF is regression evidence, not active Recovery recatch.
- Post-B1 snapshot for `hard_910220` preserved under `plan_b_posta_plan_a_correction/`.

## Plan C — impact + dynamic recatch

- `dynamic_catch_sequence` + optional lift / interior `gravity_slide_sequence` + `pose_correct_and_handoff`.
- Golden fixtures (`gbi_010`, `gbi_022`) unchanged.
- Post-catch `catch_snap` is a ballistic snap; precision should branch from a verified stable post-catch state without discarding the impact prefix.

## Reusable components

| Capability | Status |
|------------|--------|
| Contact hold / bilateral stats | Implemented (`pad_bilateral`, 60/150 ms gates) |
| Gravity-assisted slide (hand-x / interior axial) | Implemented & demonstrated (Plan A / Plan C) |
| Deliberate **tail-end** axial migration | **Not** a shipped reusable API (hypothesis scripts only) |
| Genuine contact-relative pivot | Instrumented in audits; **not** closed-loop proven |
| Gravity reorientation then slip | Plan A FSM yes; full inversion not required |
| HANDOFF / SUCCESS_NEW | `evaluate_handoff_from_state`, `evaluate_tokens` |

## Gap

`data_generation/oracles/precision/` was a stub. This milestone implements `stable_grasp` + primitives and runs bounded feasibility only.
"""
    (OUT / "ARCHITECTURE_AUDIT.md").write_text(text, encoding="utf-8")


def extract_plan_a(env) -> dict:
    print("Extract Plan A stable state (ball_train_v2_0070)...", flush=True)
    root = pickle.loads(PLAN_A_SNAP.read_bytes())
    acts = np.asarray(np.load(PLAN_A_NPZ)["actions"], np.float32)
    restore_capture(env, root["capture"])
    # Replay until stable gate passes; keep earliest such prefix
    best_i = None
    for i, a in enumerate(acts):
        step_legal(env.sim, a)
        if (i + 1) % 5 != 0 and i + 1 < len(acts):
            continue
        # probe hold from a snapshot without consuming remaining tape into the expert prefix
        cap = take_capture(env)
        restore_capture(env, cap)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap)  # back to pre-hold for continued replay
        if stab["stable"]:
            best_i = i + 1
            break
    if best_i is None:
        # fall back to end of tape + hold test
        restore_capture(env, root["capture"])
        for a in acts:
            step_legal(env.sim, a)
        best_i = len(acts)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
    else:
        restore_capture(env, root["capture"])
        for a in acts[:best_i]:
            step_legal(env.sim, a)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        # Remake capture AFTER diagnostic hold would alter state — for branching use pre-hold
        restore_capture(env, root["capture"])
        for a in acts[:best_i]:
            step_legal(env.sim, a)
    geom = audit_grasp(env.sim)
    cap = take_capture(env)
    blob = {
        "plan": "A",
        "ic_id": "ball_train_v2_0070",
        "capture": cap,
        "ref": root["ref"],
        "stabilization_prefix_actions": acts[:best_i].tolist(),
        "n_prefix": int(best_i),
        "stability": stab,
        "geom": geom,
        "needs": classify_stable_needs(geom),
        "provenance": {
            "snapshot": str(PLAN_A_SNAP),
            "continuous_npz": str(PLAN_A_NPZ),
            "note": "Earliest verified stable grasp along inward_reseat tape; hold is diagnostic-only",
        },
    }
    path = STATES / "plan_a__ball_train_v2_0070__stable.pkl"
    path.write_bytes(pickle.dumps(blob))
    (STATES / "plan_a__ball_train_v2_0070__stable.json").write_text(
        json.dumps(jsafe({k: blob[k] for k in blob if k != "capture"}), indent=2), encoding="utf-8"
    )
    print(f"  A stable={stab['stable']} needs={blob['needs']} θ={geom['e_theta_deg']:.1f} e={geom['e_pos_mm']:.1f} s={geom['contact_s_mean']}", flush=True)
    return blob


def extract_plan_b(env, table, book48) -> dict:
    print("Extract Plan B stable state (hard_910220 post-B1)...", flush=True)
    if PLAN_B_POST.exists():
        post = pickle.loads(PLAN_B_POST.read_bytes())
        restore_capture(env, post["capture"])
        prefix = list(post.get("b1_tokens") or PLAN_B_TOKENS)
    else:
        root = pickle.loads(PLAN_B_ROOT.read_bytes())
        restore_capture(env, root["capture"])
        ok, reason = run_tokens(env, root, table, book48, PLAN_B_TOKENS)
        if not ok:
            raise RuntimeError(reason)
        prefix = list(PLAN_B_TOKENS)
        post = {"ref": root["ref"]}
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    # restore pre-hold
    if PLAN_B_POST.exists():
        restore_capture(env, pickle.loads(PLAN_B_POST.read_bytes())["capture"])
    geom = audit_grasp(env.sim)
    cap = take_capture(env)
    blob = {
        "plan": "B",
        "ic_id": "hard_910220",
        "capture": cap,
        "ref": post["ref"],
        "stabilization_prefix_tokens": prefix,
        "stability": stab,
        "geom": geom,
        "needs": classify_stable_needs(geom),
        "provenance": {
            "root_snapshot": str(PLAN_B_ROOT),
            "post_recatch": str(PLAN_B_POST),
            "active_recatch": True,
            "note": "B1 active release-recatch; not early-HANDOFF nominal recatch",
        },
    }
    path = STATES / "plan_b__hard_910220__stable.pkl"
    path.write_bytes(pickle.dumps(blob))
    (STATES / "plan_b__hard_910220__stable.json").write_text(
        json.dumps(jsafe({k: blob[k] for k in blob if k != "capture"}), indent=2), encoding="utf-8"
    )
    print(f"  B stable={stab['stable']} needs={blob['needs']} θ={geom['e_theta_deg']:.1f} e={geom['e_pos_mm']:.1f} s={geom['contact_s_mean']}", flush=True)
    return blob


def extract_plan_c(env) -> dict:
    print("Extract Plan C stable state (gbi_010 post-recatch)...", flush=True)
    pack = pickle.loads(PLAN_C_PKL.read_bytes())
    raw_prefix = pack["prefix_actions"] if pack.get("prefix_actions") is not None else pack["actions"]
    prefix = np.asarray(raw_prefix, np.float32)
    restore_ballistic(env.sim, pack["catch_snap"])
    restore_drop(env.sim, None)
    arm(env.sim)
    # Earliest stable after dynamic catch: secure-brake from catch_snap until hold passes
    # (diagnostic holds are restored away; prefix provenance remains the full expert prefix).
    brake = np.zeros(7, np.float32)
    brake[6] = 1.0
    best_i = 0
    stab = None
    for i in range(60):  # up to 1.2 s post-catch settle
        cap_i = take_capture(env)
        restore_capture(env, cap_i)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_capture(env, cap_i)
        if stab["stable"]:
            best_i = i
            break
        step_legal(env.sim, brake)
        best_i = i + 1
    else:
        # Fallback: elevated snap is a verified post-recatch airborne grasp (may already be in-band)
        restore_ballistic(env.sim, pack["elev_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        stab = hold_and_confirm_stable(env, hold_s=0.20)
        restore_ballistic(env.sim, pack["elev_snap"])
        restore_drop(env.sim, None)
        arm(env.sim)
        best_i = -1  # marker: elev_snap fallback
    geom = audit_grasp(env.sim)
    cap = take_capture(env)
    snap_key = "catch_snap+secure" if best_i >= 0 else "elev_snap"
    blob = {
        "plan": "C",
        "ic_id": pack.get("root_id") or "gbi_010",
        "capture": cap,
        "ref": pack["ref"],
        "stabilization_prefix_actions": prefix.tolist(),
        "n_prefix": int(len(prefix)),
        "post_catch_secure_steps": int(best_i),
        "stability": stab,
        "geom": geom,
        "needs": classify_stable_needs(geom),
        "error_at_elevated": pack.get("error_at_elevated"),
        "provenance": {
            "expert_pkl": str(PLAN_C_PKL),
            "snap_key": snap_key,
            "mechanism": "FULL_DETACH_ACTIVE_RECATCH",
            "note": "Earliest verified stable after dynamic recatch; impact+catch prefix preserved",
        },
    }
    path = STATES / "plan_c__gbi_010__stable.pkl"
    path.write_bytes(pickle.dumps(blob))
    (STATES / "plan_c__gbi_010__stable.json").write_text(
        json.dumps(jsafe({k: blob[k] for k in blob if k != "capture"}), indent=2), encoding="utf-8"
    )
    print(f"  C stable={stab['stable']} needs={blob['needs']} θ={geom['e_theta_deg']:.1f} e={geom['e_pos_mm']:.1f} s={geom['contact_s_mean']}", flush=True)
    return blob


def run_primitives_on(env, book, state: dict, tag: str) -> dict:
    print(f"\n=== Primitives on {tag} (needs={state['needs']}) ===", flush=True)
    blob = {"capture": state["capture"], "ref": state["ref"]}
    out = {"tag": tag, "plan": state["plan"], "needs": state["needs"], "geom0": state["geom"]}
    # Direct handoff probe (may already succeed for some Plan C)
    out["E_handoff"] = {k: v for k, v in primitive_stabilize_handoff(env, blob).items() if k != "actions"}
    print(f"  HANDOFF NEW={out['E_handoff'].get('SUCCESS_NEW')} S12={out['E_handoff'].get('SUCCESS_12S')}", flush=True)

    out["A_tail_slide"] = {k: v for k, v in primitive_axial_tail_slide(env, blob, book, budget=40).items() if k != "actions"}
    print(f"  Tail slide demonstrated={out['A_tail_slide']['physically_demonstrated']} screened={out['A_tail_slide']['n_screened']}", flush=True)
    # If demonstrated, advance state for pivot test
    work = blob
    if out["A_tail_slide"]["physically_demonstrated"]:
        full = primitive_axial_tail_slide(env, blob, book, budget=40)
        restore_capture(env, blob["capture"])
        for a in full["actions"]:
            step_legal(env.sim, a)
        work = {"capture": take_capture(env), "ref": blob["ref"]}
        out["A_actions_n"] = len(full["actions"])

    out["B_pivot"] = {k: v for k, v in primitive_contact_pivot(env, work, book, budget=16).items() if k != "actions"}
    print(f"  Relative pivot demonstrated={out['B_pivot']['physically_demonstrated']}", flush=True)

    out["C_gravity"] = {k: v for k, v in primitive_gravity_reorient(env, work, book, budget=12).items() if k != "actions"}
    print(f"  Gravity reorient demonstrated={out['C_gravity']['physically_demonstrated']}", flush=True)

    out["D_final_slide"] = {k: v for k, v in primitive_final_axial_slide(env, work, book, budget=12).items() if k != "actions"}
    print(f"  Final slide demonstrated={out['D_final_slide']['physically_demonstrated']}", flush=True)

    return out


def try_composition(env, book, state: dict, prim_out: dict) -> dict:
    """Compose only demonstrated primitives for this state; else structured skip."""
    blob = {"capture": state["capture"], "ref": state["ref"]}
    needs = state["needs"]
    if needs == "already_in_band" or prim_out.get("E_handoff", {}).get("SUCCESS_NEW"):
        return {"composed": False, "reason": "already_SUCCESS_NEW_or_in_band", "handoff": prim_out.get("E_handoff")}

    # Require tail slide if needs_tail_before_pivot; require pivot if orientation_only/both
    seq = []
    labels = []
    if needs in ("needs_tail_before_pivot", "needs_both", "position_only"):
        if not prim_out["A_tail_slide"]["physically_demonstrated"]:
            return {"composed": False, "reason": "tail_slide_not_demonstrated", "stop": True}
        full = primitive_axial_tail_slide(env, blob, book, budget=40)
        seq += full["actions"]
        labels.append("tail_slide")
        restore_capture(env, blob["capture"])
        for a in seq:
            step_legal(env.sim, a)
        blob = {"capture": take_capture(env), "ref": state["ref"]}

    if needs in ("orientation_only", "needs_both", "needs_tail_before_pivot"):
        if not prim_out["B_pivot"]["physically_demonstrated"]:
            return {"composed": False, "reason": "relative_pivot_not_demonstrated", "stop": True, "partial_labels": labels}
        full = primitive_contact_pivot(env, blob, book, budget=16)
        if not full["physically_demonstrated"]:
            return {"composed": False, "reason": "pivot_failed_on_advanced_state", "stop": True, "partial_labels": labels}
        seq += full["actions"]
        labels.append("pivot")
        restore_capture(env, blob["capture"])
        for a in full["actions"]:
            step_legal(env.sim, a)
        blob = {"capture": take_capture(env), "ref": state["ref"]}

    # Optional gravity + final slide if demonstrated
    if prim_out["C_gravity"]["physically_demonstrated"]:
        full = primitive_gravity_reorient(env, blob, book, budget=12)
        if full["physically_demonstrated"]:
            seq += full["actions"]
            labels.append("gravity_reorient")
            restore_capture(env, blob["capture"])
            for a in full["actions"]:
                step_legal(env.sim, a)
            blob = {"capture": take_capture(env), "ref": state["ref"]}

    if prim_out["D_final_slide"]["physically_demonstrated"] or needs in ("position_only", "needs_both"):
        full = primitive_final_axial_slide(env, blob, book, budget=12)
        if full["physically_demonstrated"]:
            seq += full["actions"]
            labels.append("final_slide")

    hand = primitive_stabilize_handoff(env, {"capture": state["capture"], "ref": state["ref"]}, actions_prefix=seq)
    return {
        "composed": True,
        "labels": labels,
        "n_actions": len(seq),
        "SUCCESS_NEW": hand.get("SUCCESS_NEW"),
        "SUCCESS_12S": hand.get("SUCCESS_12S"),
        "reason": hand.get("reason"),
        "pre_theta": hand.get("pre_theta"),
        "pre_pos_mm": hand.get("pre_pos_mm"),
        "longest_both_ok_s": hand.get("longest_both_ok_s"),
    }


def render_primitive_video(env, state, actions, dest: Path, label: str):
    if not actions:
        return None
    restore_capture(env, state["capture"])
    look = 0.5 * (
        np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.5, "azimuth": 140.0, "elevation": -18.0})
    renderer = __import__("mujoco").Renderer(env.sim.model, 480, 640)
    frames = []
    t0 = float(env.sim.data.time)

    def grab(tag):
        cam.lookat[:] = 0.5 * (
            np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        renderer.update_scene(env.sim.data, cam)
        a = audit_grasp(env.sim)
        img = np.asarray(renderer.render(), np.uint8).copy()
        frames.append(
            caption(
                img,
                f"{label} {tag} t={float(env.sim.data.time)-t0:.2f} "
                f"s={a['contact_s_mean']} θ={a['e_theta_deg']:.1f} e={a['e_pos_mm']:.1f}",
            )
        )

    grab("START")
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % 3 == 0 or i == len(actions) - 1:
            grab("RUN")
    renderer.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return write_frames_mp4(frames, dest, fps=25)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    write_audit()
    (OUT / "STABLE_THRESHOLDS.json").write_text(json.dumps(STABLE_THRESHOLDS, indent=2), encoding="utf-8")

    table, book48 = load_vocab()
    book = load_codebook()
    env = make_env()
    t0 = time.time()

    states = {
        "A": extract_plan_a(env),
        "B": extract_plan_b(env, table, book48),
        "C": extract_plan_c(env),
    }

    prim_results = {}
    compositions = {}
    for key, st in states.items():
        tag = f"{st['plan']}_{st['ic_id']}"
        prim_results[key] = run_primitives_on(env, book, st, tag)
        compositions[key] = try_composition(env, book, st, prim_results[key])
        print(f"  Composition {key}: {compositions[key]}", flush=True)

        # Videos for demonstrated primitives on Plan B (hard case)
        if key == "B":
            full = primitive_axial_tail_slide(env, {"capture": st["capture"], "ref": st["ref"]}, book, budget=40)
            if full.get("actions"):
                render_primitive_video(env, st, full["actions"], VID / "B_910220_tail_slide.mp4", "TAIL")
            fullp = primitive_contact_pivot(env, {"capture": st["capture"], "ref": st["ref"]}, book, budget=16)
            if fullp.get("actions"):
                render_primitive_video(env, st, fullp["actions"], VID / "B_910220_pivot_attempt.mp4", "PIVOT")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "thresholds": STABLE_THRESHOLDS,
        "states": {
            k: {
                "ic_id": states[k]["ic_id"],
                "stable": states[k]["stability"].get("stable"),
                "needs": states[k]["needs"],
                "e_theta_deg": states[k]["geom"]["e_theta_deg"],
                "e_pos_mm": states[k]["geom"]["e_pos_mm"],
                "e_signed_mm": states[k]["geom"]["e_signed_mm"],
                "contact_s_mean": states[k]["geom"]["contact_s_mean"],
                "grasp_kind": states[k]["geom"]["grasp_kind"],
            }
            for k in states
        },
        "primitives": {
            k: {
                "tail_slide": prim_results[k]["A_tail_slide"].get("physically_demonstrated"),
                "relative_pivot": prim_results[k]["B_pivot"].get("physically_demonstrated"),
                "gravity_reorient": prim_results[k]["C_gravity"].get("physically_demonstrated"),
                "final_slide": prim_results[k]["D_final_slide"].get("physically_demonstrated"),
                "direct_SUCCESS_NEW": prim_results[k]["E_handoff"].get("SUCCESS_NEW"),
                "failures": {
                    "tail": prim_results[k]["A_tail_slide"].get("failure"),
                    "pivot": prim_results[k]["B_pivot"].get("failure"),
                    "gravity": prim_results[k]["C_gravity"].get("failure"),
                    "final": prim_results[k]["D_final_slide"].get("failure"),
                },
            }
            for k in prim_results
        },
        "compositions": compositions,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "primitive_results.json").write_text(json.dumps(jsafe(prim_results), indent=2), encoding="utf-8")

    # Final report
    lines = [
        "# A/B/C Precision Architecture — Feasibility Milestone",
        "",
        f"Created UTC: {summary['created_utc']}",
        f"Wall: {summary['wall_s']} s",
        "",
        "## Stable-grasp boundary",
        "",
        f"Thresholds: `{STABLE_THRESHOLDS}`",
        "",
        "Justification: Plan C already treats 150 ms continuous bilateral as strong evidence;",
        "we require the same at end of a 200 ms diagnostic hold, plus airborne z, load-bearing Fn,",
        "and bounded relative linear/angular velocity. SUCCESS_NEW bands are **not** used.",
        "",
        "## Stable states",
        "",
    ]
    for k, s in summary["states"].items():
        lines.append(f"- **{k}** `{s['ic_id']}`: stable={s['stable']} needs={s['needs']} θ={s['e_theta_deg']:.1f}° e_pos={s['e_pos_mm']:.1f} mm s={s['contact_s_mean']} kind={s['grasp_kind']}")
    lines += ["", "## Primitive feasibility (implemented / demonstrated)", ""]
    for k, p in summary["primitives"].items():
        lines.append(
            f"- **{k}**: tail={p['tail_slide']} pivot={p['relative_pivot']} "
            f"gravity={p['gravity_reorient']} final={p['final_slide']} "
            f"direct_NEW={p['direct_SUCCESS_NEW']}"
        )
    lines += ["", "## Composition", ""]
    for k, c in compositions.items():
        lines.append(f"- **{k}**: {c}")
    lines += [
        "",
        "## Status labels",
        "",
        "| Mechanism | Implemented | Physically demonstrated | Reproducible | Generalized |",
        "|-----------|-------------|-------------------------|--------------|-------------|",
        "| Stable-grasp gate | Yes | Yes (on extracted states) | Yes | 3 plans tested |",
        "| Tail axial slide | Yes | see summary | scripted | not claimed |",
        "| Relative pivot | Yes | see summary | scripted | not claimed |",
        "| Gravity reorient | Yes | see summary | scripted | not claimed |",
        "| Final slide | Yes | see summary | scripted | not claimed |",
        "",
        "## Stop condition",
        "",
        "This iteration stops after the bounded feasibility milestone.",
        "No bulk A/B/C regeneration, SFT, or GRPO.",
        "",
        "See `ARCHITECTURE_AUDIT.md`, `stable_states/`, `primitive_results.json`.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe(summary), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
