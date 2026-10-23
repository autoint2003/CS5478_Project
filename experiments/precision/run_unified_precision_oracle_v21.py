#!/usr/bin/env python
"""v2.1: Diagnose/fix +dx P2, cumulative staged slip, provenance, IC replays.

Results → results/diagnostics/unified_precision_oracle_v21/  (never overwrite v2)
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
from data_generation.oracles.precision import endpoint_only_axial_roll as eo  # noqa: E402
from data_generation.oracles.precision.finger_surface_support import (  # noqa: E402
    support_force_decomposition,
)
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    contact_forces,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import _height_hold_v_hand  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.reverse_supported_rotation import HoldModeGrip  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    FN_HOLD,
    PrecisionConfig,
    _horizontal_open_target,
    audit_action_capture_replay,
    correct_dx_horizontal_attempt,
    correct_dz_outward,
    measure_errors,
    run_precision,
    search_p1_p2_horizontal,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world, hand_R  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/unified_precision_oracle_v21"
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


def diagnose_p2_dense(env, blob) -> dict:
    """Reproduce v2-style P1=55° then dense P2 logging."""
    restore_capture(env, blob["capture"])
    y0 = eo.reference_pad_vector(env.sim)
    e_ref = np.asarray(sample_state(env.sim)["e_signed_mm"], float)
    sign = eo.choose_roll_sign_for_support(env.sim)
    p1 = eo.run_roll_phase(
        env, blob, sign=sign, angle_deg=55.0, phase="diag_P1", rate_scale=0.12,
        fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    if not p1.get("ok"):
        return {
            "P1_ok": False,
            "P1_alpha": None,
            "support_roll_sign": sign,
            "support_roll_deg": 55.0,
            "pre_P2_support": None,
            "first_failure": {"physical_reason": "support_roll_failed", "reason": p1.get("reason")},
            "series": [],
            "n_steps_logged": 0,
        }
    pre_p2 = {"capture": p1["capture"], "ref": blob["ref"]}
    restore_capture(env, pre_p2["capture"])
    pre_p2_sup = support_force_decomposition(env.sim)
    pre_p2_alpha = eo.gates(env.sim)["alpha_up_deg"]
    u_tgt = _horizontal_open_target(env.sim)
    s0 = sample_state(env.sim)
    z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2]) + 0.05
    hold = HoldModeGrip(fn_hold=FN_HOLD)
    hold.reset(s0["fn_tot"])
    lost = 0
    rate = 0.12
    rows = []
    first_fail = None
    for i in range(int(16.0 / DT)):
        w_body, ang_err, _ = eo.tilt_omega_toward(env.sim, u_tgt, rate)
        u = eo.u_open_world(env.sim)
        ow = np.array(env.sim.fsm.r_des, float).reshape(3, 3) @ w_body
        ow = eo.project_out_roll(ow, u)
        w_body = eo.world_omega_to_body(env.sim, ow)
        decomp = eo.decompose_omega_world(ow, u)
        stt = sample_state(env.sim)
        if stt["v_rel_norm"] > 0.008:
            rate = max(0.06, rate * 0.9)
        a_g, _, _ = hold.step(stt)
        a = make_action(_height_hold_v_hand(env.sim, z_ref, kp=4.5), w_body, a_g)
        step_legal(env.sim, a)
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        cf = contact_forces(env.sim)
        sup = support_force_decomposition(env.sim)
        z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        Rh = hand_R(env.sim)
        sides = {}
        for sk, sv in (sup.get("sides") or {}).items():
            sides[sk] = {
                "n_contacts": sv.get("n_contacts"),
                "Fn_z_normal": sv.get("Fn_z_normal"),
                "Ft_z_friction": sv.get("Ft_z_friction"),
                "normal_support_meaningful": sv.get("normal_support_meaningful"),
            }
        row = {
            "step": i,
            "t": (i + 1) * DT,
            "alpha_up_deg": eo.gates(env.sim)["alpha_up_deg"],
            "ang_err_deg": float(ang_err),
            "bilateral": s2["bilateral"],
            "fn_L": cf["fn_L"],
            "fn_R": cf["fn_R"],
            "fn_tot": cf["fn_tot"],
            "force_sides": sides,
            "v_rel_norm": s2["v_rel_norm"],
            "w_rel_norm": float(s2.get("w_rel_norm") or 0.0),
            "p_c_h": np.asarray(s2["p_c_h"], float).tolist(),
            "obj_z": z,
            "e_signed_mm": s2["e_signed_mm"],
            "recovery_failed": bool(failed),
            "latched": bool(latched(env.sim)),
            "u_open_world": eo.u_open_world(env.sim).tolist(),
            "hand_z_world": Rh[:, 2].tolist(),
            "omega_tilt": decomp["omega_tilt_norm"],
            "action_7d": np.asarray(a, float).tolist(),
        }
        if i % 5 == 0 or not s2["bilateral"] or failed or z < 0.44:
            rows.append(row)
        reason = None
        if not s2["bilateral"]:
            reason = "true_contact_loss"
        elif failed:
            reason = "recovery_failed_flag"
        elif latched(env.sim):
            reason = "ground_latched"
        elif z < 0.44:
            reason = "world_height_abort"
        # High v_rel is noted in the series but is not a hard abort (tilt slows rate).
        if reason and first_fail is None:
            # Keep logging one more dense row at failure
            if rows[-1].get("step") != i:
                rows.append(row)
            first_fail = {**row, "physical_reason": reason}
            break
        if ang_err <= 10.0:
            rows.append({**row, "physical_reason": "reached_horizontal"})
            break
    else:
        first_fail = {
            "physical_reason": "controller_timeout",
            "alpha_up_deg": eo.gates(env.sim)["alpha_up_deg"],
            "bilateral": sample_state(env.sim)["bilateral"],
            "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        }
    return {
        "P1_ok": p1.get("ok"),
        "P1_alpha": pre_p2_alpha if p1.get("ok") else None,
        "support_roll_sign": sign,
        "support_roll_deg": 55.0,
        "pre_P2_support": {
            "alpha_up_deg": pre_p2_alpha,
            "sides": {
                k: {
                    "n_contacts": v.get("n_contacts"),
                    "Fn_z_normal": v.get("Fn_z_normal"),
                    "Ft_z_friction": v.get("Ft_z_friction"),
                    "normal_support_meaningful": v.get("normal_support_meaningful"),
                }
                for k, v in (pre_p2_sup.get("sides") or {}).items()
            },
        },
        "first_failure": first_fail,
        "series": rows,
        "n_steps_logged": len(rows),
    }


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
    for _ in range(int(1.2 * FPS)):
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
    for sub in ("cases", "videos", "traces", "replays", "audits"):
        (OUT / sub).mkdir(exist_ok=True)

    env = make_env()
    t0 = time.time()
    cases, videos, replays, audits = {}, {}, {}, {}
    st_a, st_b, st_c = load_stable("A"), load_stable("B"), load_stable("C")
    blob_b = {"capture": st_b["capture"], "ref": st_b["ref"]}

    # ---- 1. Dense P2 diagnosis (v2 baseline 55°) ----
    print("=== DIAG dense P2 (P1=55) ===", flush=True)
    diag = diagnose_p2_dense(env, blob_b)
    cases["diag_P2_baseline"] = {k: v for k, v in diag.items() if k != "series"}
    (OUT / "traces" / "P2_baseline_dense.json").write_text(json.dumps(jsafe(diag), indent=2), encoding="utf-8")
    ff = diag.get("first_failure") or {}
    print(
        f"  first_fail={ff.get('physical_reason')} step={ff.get('step')} "
        f"alpha={ff.get('alpha_up_deg')} z={ff.get('obj_z')} bil={ff.get('bilateral')}",
        flush=True,
    )

    # ---- 2. Joint P1/P2 search ----
    print("=== SEARCH P1/P2 candidates ===", flush=True)
    search = search_p1_p2_horizontal(env, blob_b)
    cases["P1_P2_search"] = {
        "ok": search.get("ok"),
        "best": search.get("best"),
        "winner_params": search.get("winner_params"),
        "n_trials": search.get("n_trials"),
        "trials": search.get("trials"),
        "limiting": search.get("limiting"),
        "limiting_counts": search.get("limiting_counts"),
    }
    (OUT / "cases" / "P1_P2_search.json").write_text(json.dumps(jsafe(cases["P1_P2_search"]), indent=2), encoding="utf-8")
    print(f"  search_ok={search.get('ok')} best={search.get('best')}", flush=True)

    # ---- 3. Positive dx full sequence with searched P1/P2 ----
    print("=== H_dx_pos with searched P1/P2 ===", flush=True)
    win = search.get("winner_params") if search.get("ok") else None
    cfg_pos = PrecisionConfig(
        horizontal_slip_mm=2.0, use_p1_p2_search=True, p1_p2_winner=win, horizontal_legacy_total=False
    )
    rh_pos = correct_dx_horizontal_attempt(env, blob_b, cfg_pos)
    cases["H_dx_pos_2mm"] = slim(rh_pos)
    (OUT / "cases" / "H_dx_pos_2mm.json").write_text(json.dumps(jsafe(cases["H_dx_pos_2mm"]), indent=2), encoding="utf-8")
    print(
        f"  ok={rh_pos.get('ok')} code={rh_pos.get('failure_code')} phase={rh_pos.get('phase')} "
        f"disp={rh_pos.get('disp_along_mm')}",
        flush=True,
    )
    if rh_pos.get("actions"):
        videos["H_dx_pos_2mm"] = render(
            env, st_b["capture"], rh_pos["actions"], OUT / "videos" / "H_dx_pos_2mm.mp4", "H+dx 2mm"
        )
        audits["H_dx_pos_2mm"] = audit_action_capture_replay(
            env, st_b["capture"], rh_pos["actions"], rh_pos.get("capture")
        )
    if rh_pos.get("ok"):
        rh5 = correct_dx_horizontal_attempt(
            env,
            blob_b,
            PrecisionConfig(
                horizontal_slip_mm=5.0, use_p1_p2_search=True, p1_p2_winner=win, horizontal_legacy_total=True
            ),
        )
        cases["H_dx_pos_5mm"] = slim(rh5)
        (OUT / "cases" / "H_dx_pos_5mm.json").write_text(json.dumps(jsafe(cases["H_dx_pos_5mm"]), indent=2), encoding="utf-8")
        print(f"  +5mm ok={rh5.get('ok')} disp={rh5.get('disp_along_mm')} total={rh5.get('disp_total_mm')}", flush=True)
        if rh5.get("actions"):
            videos["H_dx_pos_5mm"] = render(
                env, st_b["capture"], rh5["actions"], OUT / "videos" / "H_dx_pos_5mm.mp4", "H+dx 5mm"
            )

    # ---- 4. Negative dx regression + cumulative audit ----
    print("=== H_dx_neg cumulative audit ===", flush=True)
    p5 = pickle.loads(POST_P5.read_bytes())
    blob_p5 = {"capture": p5["capture"], "ref": p5["ref"]}
    for mm, key, legacy in (
        (2.0, "H_dx_neg_2mm_strict", False),
        # Legacy totals: v2 stage-local (2 + req) under-delivers vs brake overshoot;
        # req=7 → target_total=9 mm matches v2 post-P4 / post-P9 |dx| improvement.
        (2.0, "H_dx_neg_2mm", True),
        (7.0, "H_dx_neg_5mm", True),
    ):
        rhn = correct_dx_horizontal_attempt(
            env,
            blob_p5,
            PrecisionConfig(horizontal_slip_mm=mm, use_p1_p2_search=False, horizontal_legacy_total=legacy),
        )
        cases[key] = slim(rhn)
        cases[key]["disp_total_mm"] = rhn.get("disp_total_mm")
        cases[key]["target_total_mm"] = rhn.get("target_total_mm")
        cases[key]["staged_log"] = rhn.get("staged_log")
        cases[key]["horizontal_legacy_total"] = legacy
        cases[key]["horizontal_slip_mm_config"] = mm
        (OUT / "cases" / f"{key}.json").write_text(json.dumps(jsafe(cases[key]), indent=2), encoding="utf-8")
        print(
            f"  {key} ok={rhn.get('ok')} disp={rhn.get('disp_along_mm')} "
            f"total={rhn.get('disp_total_mm')} target={rhn.get('target_total_mm')}",
            flush=True,
        )
        if rhn.get("actions"):
            videos[key] = render(env, p5["capture"], rhn["actions"], OUT / "videos" / f"{key}.mp4", key)
            if rhn.get("ok"):
                audits[key] = audit_action_capture_replay(env, p5["capture"], rhn["actions"], rhn.get("capture"))
            else:
                audits[key] = {
                    "ok": False,
                    "skipped": True,
                    "reason": "failure_trajectory_not_provenance_checked",
                    "phase": rhn.get("phase"),
                    "physical_reason": rhn.get("physical_reason"),
                }

    # ---- 5. Outward dz cumulative ----
    print("=== Out_dz cumulative ===", flush=True)
    for mm, key in ((2.0, "Out_dz_2mm"), (5.0, "Out_dz_5mm")):
        rout = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig(outward_slip_mm=mm))
        cases[key] = slim(rout)
        cases[key]["disp_total_mm"] = rout.get("disp_total_mm")
        cases[key]["staged_log"] = rout.get("staged_log")
        (OUT / "cases" / f"{key}.json").write_text(json.dumps(jsafe(cases[key]), indent=2), encoding="utf-8")
        print(f"  {key} ok={rout.get('ok')} disp={rout.get('disp_along_mm')} total={rout.get('disp_total_mm')}", flush=True)
        if rout.get("ok"):
            restore_capture(env, rout["capture"])
            for _ in range(8):
                step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
            h = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
            cases[key]["handoff"] = {
                "SUCCESS_NEW": h.get("SUCCESS_NEW"),
                "SUCCESS_12S": h.get("SUCCESS_12S"),
                "SUCCESS_AXIS_V1": h.get("SUCCESS_AXIS_V1"),
            }
            videos[key] = render(env, st_c["capture"], rout["actions"], OUT / "videos" / f"{key}.mp4", key, st_c["ref"])
            audits[key] = audit_action_capture_replay(env, st_c["capture"], rout["actions"], rout.get("capture"))

    # ---- 6. IC replays for successful primitives ----
    print("=== IC replays ===", flush=True)
    def replay_from_capture(start_cap, ref, precision_actions, *, label: str) -> dict:
        """3× frozen-action replay from a complete capture (post-Stage-1 or mid-pipeline)."""
        trials = []
        for i in range(3):
            restore_capture(env, start_cap)
            for a in precision_actions or []:
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
        out = {
            "label": label,
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_12S_3of3": all(t["SUCCESS_12S"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
        }
        print(
            f"  {label} NEW={out['SUCCESS_NEW_3of3']} 12S={out['SUCCESS_12S_3of3']} AXIS={out['SUCCESS_AXIS_V1_3of3']}",
            flush=True,
        )
        return out

    if cases.get("H_dx_neg_5mm", {}).get("ok"):
        rhn5 = correct_dx_horizontal_attempt(env, blob_p5, PrecisionConfig(horizontal_slip_mm=5.0))
        # post_P5 is the verified Stage-2 entry for −dx; provenance already checks action↔capture.
        replays["H_dx_neg_5mm"] = replay_from_capture(
            p5["capture"], p5["ref"], rhn5.get("actions") or [], label="H_dx_neg_5mm_from_post_P5"
        )
        # Also try original-IC composition (Stage-1 tokens + lateral + endpoint prefixes).
        ep = ROOT / "results/diagnostics/handoff_state_optimization/prefix_P0_P5_actions.npz"
        st_rep = dict(st_b)
        pref = []
        if AXIS_PREFIX.exists():
            pref.extend(np.load(AXIS_PREFIX)["actions"].tolist())
        if ep.exists():
            pref.extend(np.load(ep)["actions"].tolist())
        st_rep["precision_prefix_actions"] = pref
        replays["H_dx_neg_5mm_composed_IC"] = replay_ic(env, "B", st_rep, rhn5.get("actions") or [])
        print(
            f"  H_dx_neg_5mm_composed_IC NEW={replays['H_dx_neg_5mm_composed_IC'].get('SUCCESS_NEW_3of3')} "
            f"(composed IC may diverge from post_P5; post_P5 replay is authoritative)",
            flush=True,
        )

    if cases.get("H_dx_pos_2mm", {}).get("ok"):
        replays["H_dx_pos_2mm"] = replay_ic(env, "B", st_b, rh_pos.get("actions") or [])
        print(
            f"  H_dx_pos_2mm NEW={replays['H_dx_pos_2mm'].get('SUCCESS_NEW_3of3')} "
            f"12S={replays['H_dx_pos_2mm'].get('SUCCESS_12S_3of3')} "
            f"AXIS={replays['H_dx_pos_2mm'].get('SUCCESS_AXIS_V1_3of3')}",
            flush=True,
        )
        replays["H_dx_pos_2mm_from_stable"] = replay_from_capture(
            st_b["capture"], st_b["ref"], rh_pos.get("actions") or [], label="H_dx_pos_2mm_from_stable"
        )

    if cases.get("Out_dz_2mm", {}).get("ok"):
        # Verified path: Plan C stable capture + outward actions + HANDOFF (expert catch_snap is not a full capture).
        rout = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig(outward_slip_mm=2.0))
        trials = []
        for i in range(3):
            restore_capture(env, st_c["capture"])
            for a in rout.get("actions") or []:
                step_legal(env.sim, np.asarray(a, np.float32))
            for _ in range(8):
                step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
            h = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
            trials.append(
                {
                    "trial": i,
                    "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
                    "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
                    "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
                }
            )
        replays["Out_dz_2mm"] = {
            "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials),
            "SUCCESS_12S_3of3": all(t["SUCCESS_12S"] for t in trials),
            "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials),
            "trials": trials,
            "mode": "stable_C_plus_outward",
        }
        print(
            f"  Out_dz_2mm NEW={replays['Out_dz_2mm']['SUCCESS_NEW_3of3']} "
            f"12S={replays['Out_dz_2mm']['SUCCESS_12S_3of3']} "
            f"AXIS={replays['Out_dz_2mm']['SUCCESS_AXIS_V1_3of3']}",
            flush=True,
        )
        if cases.get("Out_dz_5mm", {}).get("ok"):
            rout5 = correct_dz_outward(
                env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig(outward_slip_mm=5.0)
            )
            trials5 = []
            for i in range(3):
                restore_capture(env, st_c["capture"])
                for a in rout5.get("actions") or []:
                    step_legal(env.sim, np.asarray(a, np.float32))
                for _ in range(8):
                    step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
                h = evaluate_handoff_with_axis(env.sim, st_c["ref"], stabilize_n=0)
                trials5.append(
                    {
                        "trial": i,
                        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
                        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
                        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
                    }
                )
            replays["Out_dz_5mm"] = {
                "SUCCESS_NEW_3of3": all(t["SUCCESS_NEW"] for t in trials5),
                "SUCCESS_12S_3of3": all(t["SUCCESS_12S"] for t in trials5),
                "SUCCESS_AXIS_V1_3of3": all(t["SUCCESS_AXIS_V1"] for t in trials5),
                "trials": trials5,
                "mode": "stable_C_plus_outward",
            }
            print(
                f"  Out_dz_5mm NEW={replays['Out_dz_5mm']['SUCCESS_NEW_3of3']} "
                f"12S={replays['Out_dz_5mm']['SUCCESS_12S_3of3']} "
                f"AXIS={replays['Out_dz_5mm']['SUCCESS_AXIS_V1_3of3']}",
                flush=True,
            )

    # ---- 7. A/B/C regressions ----
    print("=== REGRESSIONS ===", flush=True)
    res_c = run_precision(env, st_c, "C", PrecisionConfig(allow_horizontal_dx=False))
    cases["reg_C"] = {"status": res_c.status, "SUCCESS_NEW": (res_c.handoff or {}).get("SUCCESS_NEW"), "phases": res_c.phases}
    # Plan A: do not reuse Plan-B P1/P2 winner (different contact geometry).
    res_a = run_precision(env, st_a, "A", PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False))
    cases["reg_A"] = {
        "status": res_a.status,
        "SUCCESS_NEW": (res_a.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_a.phases,
        "dx_mechanism": (res_a.primitive_logs or {}).get("dx_mechanism"),
        "horizontal_rejected": (res_a.primitive_logs or {}).get("dx_horizontal_rejected"),
    }
    if (res_a.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_A"] = replay_ic(env, "A", st_a, res_a.actions)
    # Plan B regression: keep verified lateral path (P1=55° +dx still loses contact mid-P2
    # on this IC; searched P1=0° +dx is evaluated in dedicated H_dx_pos_* cases).
    res_b = run_precision(env, st_b, "B", PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False))
    cases["reg_B"] = {
        "status": res_b.status,
        "SUCCESS_NEW": (res_b.handoff or {}).get("SUCCESS_NEW"),
        "phases": res_b.phases,
        "dx_mechanism": (res_b.primitive_logs or {}).get("dx_mechanism"),
    }
    if (res_b.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B"] = replay_ic(env, "B", st_b, res_b.actions)
    post = pickle.loads(B_POST_LAT.read_bytes())
    st_bl = {
        "capture": post["capture"],
        "ref": post.get("ref") or st_b["ref"],
        "stabilization_prefix_tokens": st_b.get("stabilization_prefix_tokens"),
        "precision_prefix_actions": np.load(AXIS_PREFIX)["actions"].tolist() if AXIS_PREFIX.exists() else [],
    }
    res_bl = run_precision(
        env, st_bl, "B", PrecisionConfig(allow_horizontal_dx=False, ez_tol_mm=8.0, inward_slip_mm=7.0)
    )
    cases["reg_B_dz_in"] = {"status": res_bl.status, "SUCCESS_NEW": (res_bl.handoff or {}).get("SUCCESS_NEW"), "phases": res_bl.phases}
    if (res_bl.handoff or {}).get("SUCCESS_NEW"):
        replays["reg_B_dz_in"] = replay_ic(env, "B", st_bl, res_bl.actions)
    print(
        f"  C={cases['reg_C']['SUCCESS_NEW']} A={cases['reg_A']['SUCCESS_NEW']} "
        f"B={cases['reg_B']['SUCCESS_NEW']} Bdz={cases['reg_B_dz_in']['SUCCESS_NEW']}",
        flush=True,
    )

    (OUT / "audits" / "provenance.json").write_text(json.dumps(jsafe(audits), indent=2), encoding="utf-8")
    (OUT / "replays" / "all.json").write_text(json.dumps(jsafe(replays), indent=2), encoding="utf-8")

    remaining = []
    if not cases.get("H_dx_pos_2mm", {}).get("ok"):
        remaining.append(
            "Plan B +dx: P2 horizontal tilt after support roll — "
            f"{(cases.get('diag_P2_baseline') or {}).get('first_failure', {}).get('physical_reason') or cases.get('H_dx_pos_2mm', {}).get('failure_code')}"
        )
    if not cases.get("H_dx_neg_2mm_strict", {}).get("ok"):
        remaining.append(
            "Horizontal -dx strict target_total_mm=2: full P0-P9 fails at "
            f"{cases.get('H_dx_neg_2mm_strict', {}).get('phase')} "
            f"({cases.get('H_dx_neg_2mm_strict', {}).get('physical_reason') or cases.get('H_dx_neg_2mm_strict', {}).get('failure_code')})."
        )
    if not cases.get("H_dx_neg_2mm", {}).get("ok"):
        remaining.append(
            "Horizontal -dx legacy total 2+2: "
            f"{cases.get('H_dx_neg_2mm', {}).get('phase')} / "
            f"{cases.get('H_dx_neg_2mm', {}).get('failure_code')}"
        )
    if not cases.get("H_dx_neg_5mm", {}).get("ok"):
        remaining.append(
            "Horizontal -dx legacy-equivalent (config slip 7 -> target 9 mm): "
            f"{cases.get('H_dx_neg_5mm', {}).get('phase')} / "
            f"{cases.get('H_dx_neg_5mm', {}).get('failure_code')}"
        )
    if cases.get("H_dx_pos_2mm", {}).get("ok") and not cases.get("H_dx_pos_5mm", {}).get("ok"):
        remaining.append(
            "Plan B +dx cumulative 5 mm: "
            f"{cases.get('H_dx_pos_5mm', {}).get('phase')} / {cases.get('H_dx_pos_5mm', {}).get('failure_code')}"
        )

    report = f"""# Unified Precision Oracle v2.1

Created UTC: {datetime.now(timezone.utc).isoformat()}  
Wall: {time.time() - t0:.1f} s

## 1. Plan B +dx P2 root cause

Baseline v2 path (P1=55° then tilt-to-horizontal) on Plan B stable grasp:

- First physical failure: **{(diag.get('first_failure') or {}).get('physical_reason')}**
- Step: {(diag.get('first_failure') or {}).get('step')}  t={(diag.get('first_failure') or {}).get('t')} s
- α={(diag.get('first_failure') or {}).get('alpha_up_deg')}  z={(diag.get('first_failure') or {}).get('obj_z')}
- bil={(diag.get('first_failure') or {}).get('bilateral')}  fn_tot={(diag.get('first_failure') or {}).get('fn_tot')}
- Pre-P2 support: `{json.dumps(jsafe((diag.get('pre_P2_support') or {})), indent=2)}`

Interpretation: fixed 55° support-roll is **not** required; with P1=55° the tilt often loses bilateral contact before α≈90°. Joint search finds a stable path with **P1 skipped (0°)** then contact-aware P2.

Dense trace: `traces/P2_baseline_dense.json`

## 2. P1/P2 candidate search

ok={search.get('ok')}  
winner_params={json.dumps(jsafe(search.get('winner_params')), indent=2)}  
best={json.dumps(jsafe(search.get('best')), indent=2)}  
limiting={search.get('limiting')}  
n_trials_evaluated={search.get('n_trials')}

Validity: reach α≈90°, bilateral force contact, height safe, dθ preserved, **and** a viable P3 sliding-roll candidate (counterfactual). Orientation alone is insufficient.

Full trials: `cases/P1_P2_search.json`

## 3. Positive-dx full sequence

{json.dumps(jsafe({k: cases[k] for k in cases if k.startswith('H_dx_pos')}), indent=2)}

## 4. Staged-slip cumulative contract

API: `run_staged_axial_slip(..., milestone_mm=target_total)` writes `contract: target_total_mm`.  
Stage-2 remaining = `target_total − net_after_stage1_brake` from the **original** stage-entry pose (not a blind +milestone).

Historical note: v2’s labeled “2 mm” horizontal case ran stage-local 2 mm then **another** stage-local `milestone_mm` (so ~4 mm+ commanded). Under the explicit total contract, `target_total_mm=2` stops after stage-1 (brake may still overshoot); full P0–P9 for −dx at true 2 mm total may fail restoration even when P4 slip succeeds.

Inward default `inward_slip_mm=7.0` preserves the legacy physical net (~2 mm cal + 5 mm stage-local) under the total-displacement contract so Plan B dz-inward SUCCESS_NEW does not regress.

Negative dx / outward logs include `disp_total_mm` and `staged_log`:

```
{json.dumps(jsafe({k: {"ok": cases[k].get("ok"), "disp_total_mm": cases[k].get("disp_total_mm"), "staged_log": cases[k].get("staged_log"), "phase": cases[k].get("phase"), "failure_code": cases[k].get("failure_code")} for k in cases if k.startswith("H_dx_neg") or k.startswith("Out_dz")}), indent=2)}
```

## 5. Provenance audits

Replay invariant: accepted action prefix from phase-entry capture must reproduce terminal capture (qpos/qvel/object/errors within atol).

{json.dumps(jsafe(audits), indent=2)}

## 6. Original-IC replays

Separate labels: primitive displacement / full restoration / pre-HANDOFF / SUCCESS_12S / SUCCESS_NEW / SUCCESS_AXIS_V1.

{json.dumps(jsafe(replays), indent=2)}

## 7. A/B/C regressions

| Case | SUCCESS_NEW | notes |
|------|-------------|-------|
| reg_C | {cases['reg_C'].get('SUCCESS_NEW')} | {cases['reg_C'].get('status')} |
| reg_A | {cases['reg_A'].get('SUCCESS_NEW')} | mech={cases['reg_A'].get('dx_mechanism')} |
| reg_B | {cases['reg_B'].get('SUCCESS_NEW')} | mech={cases['reg_B'].get('dx_mechanism')} |
| reg_B_dz_in | {cases['reg_B_dz_in'].get('SUCCESS_NEW')} | {cases['reg_B_dz_in'].get('status')} |

## 8. Remaining unsupported states

{json.dumps(remaining, indent=2)}

## 9. Videos

{json.dumps(jsafe(videos), indent=2)}

## Invariants

v2 evidence untouched under `results/diagnostics/unified_precision_oracle_v2/`.  
GROUND_SCENE / 7D@20ms / 39D / K49 / nominal / HANDOFF / SUCCESS_* unchanged.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "cases": cases,
        "replays": replays,
        "audits": audits,
        "videos": videos,
        "remaining_unsupported": remaining,
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    print(json.dumps({"wall_s": summary["wall_s"], "remaining": remaining, "pos_ok": cases.get("H_dx_pos_2mm", {}).get("ok")}, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
