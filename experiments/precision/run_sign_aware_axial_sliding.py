#!/usr/bin/env python
"""Sign-aware axial sliding: Mode A outward vs Mode B inward (palm-supported).

Preserves SUCCESS_NEW / SUCCESS_AXIS_V1 / Plan A & Branch A artifacts.
Bounded mechanism validation — no bulk Oracle / SFT / GRPO.
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
from data_generation.oracles.precision.force_slip_control import (  # noqa: E402
    DT,
    DualLoopController,
    ForcePController,
    SlipOuterLoop,
    make_action,
    sample_state,
)
from data_generation.oracles.precision.orientation_planner import (  # noqa: E402
    _height_hold_v_hand,
    plan_restore_orientation,
    execute_restore_plan,
)
from data_generation.oracles.precision.sign_aware_axial import (  # noqa: E402
    G_AXIAL_USEFUL_MIN,
    REORI_DTHETA_DEG,
    REORI_DX_MM,
    desire_direction,
    fingertip_overlap_margin_mm,
    opening_axis_hand,
    search_stable_reorientation,
    select_axial_mode,
    signed_opening_error,
    support_force_analysis,
    waypoint_reorient,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import AXIS_V1_EZ_MM, evaluate_handoff_with_axis  # noqa: E402
from evaluation.success_criteria import WORKING_POS_MM  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

B_POST_LAT = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
B_STABLE = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_stable.pkl"
A_POST_PIV = ROOT / "results/diagnostics/geometry_inferred_precision/snapshots/A_post_pivot.pkl"
IC_B = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910220.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"
STABLE_JSON = ROOT / "results/diagnostics/axis_aware_precision/stable_state.json"
OUT = ROOT / "results/diagnostics/sign_aware_axial_sliding"

FORCE_KP, FORCE_DA = 0.10, 0.12
SLIP_KN = 900.0
V_SLIP = -0.001
FN_MIN, FN_MAX = 3.8, 11.0
FN_INIT, FN_BRAKE = 5.2, 14.0
Z_ABORT = 0.44
LATERAL_DRIFT_ABORT_MM = 4.0
FPS = 25


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
    d = pickle.loads(path.read_bytes())
    if "capture" not in d:
        raise ValueError(f"no capture in {path}")
    if "ref" not in d:
        # borrow from B stable
        st = pickle.loads(B_STABLE.read_bytes())
        d = {"capture": d["capture"], "ref": st.get("ref")}
    return d


def run_axial_slip(
    env,
    blob,
    d,
    *,
    milestone_mm: float,
    z_ref: float,
    e_x0: float,
    max_s: float = 12.0,
    firm: bool = False,
):
    restore_capture(env, blob["capture"])
    s0 = sample_state(env.sim)
    p0 = np.asarray(s0["p_c_h"], float)
    d = np.asarray(d, float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    axis = -d
    # Mode B (inward, opening-up) needs firmer grip — aggressive unload loses contact in <0.2 s
    fn_min = 6.5 if firm else FN_MIN
    fn_max = 14.0 if firm else FN_MAX
    fn_init = 9.0 if firm else FN_INIT
    kn = 400.0 if firm else SLIP_KN
    a0 = 0.55 if firm else 0.2
    ctl = DualLoopController(
        force=ForcePController(kp=FORCE_KP, da_max=FORCE_DA, a_min=-0.80, a_max=1.0),
        slip=SlipOuterLoop(vx_target=V_SLIP, fn_min=fn_min, fn_max=fn_max, kn=kn),
        use_slip=True,
    )
    ctl.reset(a0=a0, fn0=s0["fn_tot"], fn_target=fn_init, slip_axis=tuple(axis.tolist()))
    actions, series = [], []
    lost = 0
    reached = False
    abort = None
    u_open = np.asarray(opening_axis_hand(env.sim)["u_open"], float)
    for i in range(max(1, int(round(max_s / DT)))):
        s = sample_state(env.sim)
        _, meta = ctl.step(s)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], meta["a"])
        step_legal(env.sim, a)
        actions.append(a.copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        e2 = np.asarray(s2["e_signed_mm"], float)
        disp = float(np.dot(dp, d)) * 1e3
        margin = fingertip_overlap_margin_mm(env.sim, u_open)
        series.append(
            {
                "t": (i + 1) * DT,
                "disp_along_mm": disp,
                "e_signed": e2.tolist(),
                "theta": s2["e_theta_deg"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "fn_target": meta["fn_target"],
                "v_along": float(np.dot(s2["v_rel_h"], d)),
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "fingertip_margin_mm": margin,
                "bilateral": s2["bilateral"],
                "phase": "slip",
            }
        )
        if failed or latched(env.sim) or not s2["bilateral"]:
            return {"ok": False, "reason": "contact_loss", "actions": actions, "series": series}
        if abs(float(e2[0]) - e_x0) > LATERAL_DRIFT_ABORT_MM:
            abort = "lateral_drift"
            break
        # Outward only: abort if distal contacts vanish relative to COM
        if float(np.dot(d, u_open)) > 0.5 and margin < 0.5:
            abort = "fingertip_margin"
            break
        if disp >= milestone_mm:
            reached = True
            break
    # brake
    ctl_b = ForcePController(kp=0.15, da_max=0.2)
    ctl_b.reset(series[-1].get("fn_target", 5.0) / 20.0 if series else 0.5, series[-1]["fn_L"] + series[-1]["fn_R"] if series else 10.0)
    # simpler reset from last a
    ctl_b.reset(0.8, sample_state(env.sim)["fn_tot"])
    for j in range(int(0.40 / DT) + int(0.15 / DT)):
        s = sample_state(env.sim)
        a_g, _ = ctl_b.step(s["fn_tot"], FN_BRAKE)
        a = make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g)
        step_legal(env.sim, a)
        actions.append(a.copy())
        s2 = sample_state(env.sim)
        dp = np.asarray(s2["p_c_h"], float) - p0
        series.append(
            {
                "t": series[-1]["t"] + DT,
                "disp_along_mm": float(np.dot(dp, d) * 1e3),
                "e_signed": s2["e_signed_mm"],
                "theta": s2["e_theta_deg"],
                "fn_L": s2["fn_L"],
                "fn_R": s2["fn_R"],
                "fn_target": FN_BRAKE,
                "v_along": float(np.dot(s2["v_rel_h"], d)),
                "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
                "fingertip_margin_mm": fingertip_overlap_margin_mm(env.sim, u_open),
                "bilateral": s2["bilateral"],
                "phase": "brake" if j < int(0.40 / DT) else "hold",
            }
        )
        if not s2["bilateral"]:
            return {"ok": False, "reason": "loss_on_brake", "actions": actions, "series": series, "reached": reached}
    disp = float(series[-1]["disp_along_mm"])
    hold_ok = all(r["bilateral"] for r in series if r["phase"] == "hold")
    ok = bool(hold_ok and abort is None and (reached or disp >= 0.5 * milestone_mm))
    return {
        "ok": ok,
        "reached": reached,
        "abort_reason": abort,
        "disp_along_mm": disp,
        "e_signed": series[-1]["e_signed"],
        "theta": series[-1]["theta"],
        "lateral_drift_mm": abs(float(series[-1]["e_signed"][0]) - e_x0),
        "min_obj_z": float(min(r["obj_z"] for r in series)),
        "min_fingertip_margin_mm": float(min(r["fingertip_margin_mm"] for r in series)),
        "actions": actions,
        "series": series,
        "capture": take_capture(env),
    }


def handoff_rich(env, capture, ref):
    from data_generation.oracles.precision.primitives import _wrist

    restore_capture(env, capture)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    meta = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    meta.pop("samples", None)
    return meta


def make_plots(out_dir: Path, mode_results: dict) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    # Reori trial scatter
    for mode, res in mode_results.items():
        trials = (res.get("reori_search") or {}).get("trials") or []
        if not trials:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for t in trials:
            c = "C2" if t.get("ok") and t.get("clears_gravity") else ("C1" if t.get("ok") else "C3")
            ax.scatter(abs(t.get("delta_ex_mm") or 99), abs(t.get("delta_theta_deg") or 99), c=c, alpha=0.7)
        ax.axvline(REORI_DX_MM, ls="--", color="k", lw=0.8)
        ax.axhline(REORI_DTHETA_DEG, ls="--", color="k", lw=0.8)
        ax.set_xlabel("|Δe_x| mm")
        ax.set_ylabel("|Δθ| deg")
        ax.set_title(f"Reorientation search: {mode}")
        fig.tight_layout()
        p = out_dir / f"reori_search_{mode}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths[f"reori_{mode}"] = str(p)
        # slip traces
        slip = res.get("slip") or {}
        ser = slip.get("series") or []
        if ser:
            fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
            t = [r["t"] for r in ser]
            axs[0].plot(t, [r["e_signed"][0] for r in ser], label="e_x")
            axs[0].plot(t, [r["e_signed"][2] for r in ser], label="e_z/open")
            axs[0].legend()
            axs[1].plot(t, [r["fn_L"] for r in ser], label="FnL")
            axs[1].plot(t, [r["fn_R"] for r in ser], label="FnR")
            axs[1].legend()
            axs[2].plot(t, [r["obj_z"] for r in ser])
            axs[2].axhline(Z_ABORT, color="r", ls="--")
            axs[2].set_xlabel("t (s)")
            fig.suptitle(f"Axial slip {mode}")
            fig.tight_layout()
            p = out_dir / f"slip_{mode}.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths[f"slip_{mode}"] = str(p)
    return paths


def render_video(env, blob, actions, dest, title, d=None, plan=None):
    from experiments.precision.run_plan_b_geometry_precision_transfer import render_video as _rv

    return _rv(env, blob, actions, dest, title, d=d, plan=plan)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshots").mkdir(exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)

    env = make_env()

    # --- §2 Convention ---
    print("=== Opening-axis convention ===", flush=True)
    blob_b = load_blob(B_POST_LAT)
    restore_capture(env, blob_b["capture"])
    oa = opening_axis_hand(env.sim)
    err_b = signed_opening_error(env.sim, np.asarray(oa["u_open"], float))
    mode_b = select_axial_mode(err_b["e_open_mm"], axial_tol_mm=AXIS_V1_EZ_MM)
    # Diagnostic threshold for "needed" vs small opposite-sign test
    mode_b_diag = select_axial_mode(err_b["e_open_mm"], axial_tol_mm=5.0)
    convention = {
        "opening_axis": oa,
        "plan_b_post_lateral": err_b,
        "mode_at_AXIS_V1_tol_20": mode_b,
        "mode_at_diag_tol_5": mode_b_diag,
        "verified": bool(oa.get("hand_z_equals_u_open")),
    }
    print(
        f"  u_open={oa['u_open']} nearest={oa['nearest_hand_axis']} "
        f"e_open={err_b['e_open_mm']:.2f} mode20={mode_b['mode']} mode5={mode_b_diag['mode']}",
        flush=True,
    )

    blob_a = load_blob(A_POST_PIV)
    restore_capture(env, blob_a["capture"])
    oa_a = opening_axis_hand(env.sim)
    err_a = signed_opening_error(env.sim, np.asarray(oa_a["u_open"], float))
    mode_a = select_axial_mode(err_a["e_open_mm"], axial_tol_mm=2.0)  # low tol to exercise Mode A
    convention["plan_a_post_pivot"] = err_a
    convention["mode_a_at_tol_2"] = mode_a
    print(f"  A_post_piv e_open={err_a['e_open_mm']:.2f} mode@2mm={mode_a['mode']}", flush=True)
    (OUT / "opening_axis_convention.json").write_text(json.dumps(jsafe(convention), indent=2), encoding="utf-8")

    u_open = np.asarray(oa["u_open"], float)
    mode_results = {}

    # ========== Mode B (inward) on Plan B post-lateral ==========
    print("=== Mode B inward reorientation search (Plan B post-lateral) ===", flush=True)
    # Support analysis before
    restore_capture(env, blob_b["capture"])
    support0_b = support_force_analysis(env.sim)
    search_b = search_stable_reorientation(
        env,
        blob_b,
        mode="B_inward",
        u_open=u_open,
        angles=(35.0, 55.0, 70.0, 85.0, 100.0),
        rates=(0.12, 0.15, 0.20),
        axes=("+wy", "-wy"),
        g_useful_min=G_AXIAL_USEFUL_MIN,
    )
    best_b = search_b.get("best_full")
    mode_results["B_inward"] = {
        "state": "plan_b_post_lateral",
        "e_open_mm": err_b["e_open_mm"],
        "support_before": {
            k: support0_b[k]
            for k in support0_b
            if k != "contacts"
        },
        "reori_search": {k: v for k, v in search_b.items() if k != "best_full"},
        "reori_ok": bool(search_b.get("any_ok")),
        "reori_clears_g": bool(search_b.get("any_clears_gravity")),
    }

    slip_b = None
    handoff_b = None
    if best_b and best_b.get("ok"):
        print("=== Mode B short axial slip ===", flush=True)
        tilt = {"capture": best_b["capture"], "ref": blob_b["ref"]}
        z_ref = float(best_b.get("min_obj_z") or 0.5)
        d = desire_direction("B_inward", u_open)
        e_x0 = float(best_b["e0"][0])
        slip_b = run_axial_slip(env, tilt, d, milestone_mm=2.0, z_ref=z_ref, e_x0=e_x0, firm=True)
        print(
            f"  slip2: ok={slip_b.get('ok')} disp={slip_b.get('disp_along_mm')} "
            f"drift_x={slip_b.get('lateral_drift_mm')} abort={slip_b.get('abort_reason')} "
            f"reason={slip_b.get('reason')}",
            flush=True,
        )
        mode_results["B_inward"]["slip"] = {k: v for k, v in slip_b.items() if k not in ("actions", "series", "capture")}
        mode_results["B_inward"]["slip"]["series"] = slip_b.get("series")
        mode_results["B_inward"]["support_after_reori"] = (best_b.get("support_final") or {})
        if slip_b.get("ok"):
            # optional 5 mm from tilt again
            slip5 = run_axial_slip(env, tilt, d, milestone_mm=5.0, z_ref=z_ref, e_x0=e_x0, firm=True)
            mode_results["B_inward"]["slip_5mm"] = {k: v for k, v in slip5.items() if k not in ("actions", "series", "capture")}
            post = {"capture": slip_b["capture"], "ref": blob_b["ref"]}
            # soft restore
            rplan = plan_restore_orientation(env, post)
            rexe = execute_restore_plan(env, post, rplan)
            actions_full = list(best_b["actions"]) + list(slip_b["actions"]) + list(rexe.get("actions") or [])
            handoff_b = handoff_rich(env, take_capture(env), blob_b["ref"])
            mode_results["B_inward"]["handoff"] = {
                k: handoff_b[k]
                for k in (
                    "SUCCESS_12S",
                    "SUCCESS_NEW",
                    "SUCCESS_AXIS_V1",
                    "pre_e_signed_mm",
                    "pre_theta",
                    "pre_pos_mm",
                    "reason",
                    "window_SUCCESS_NEW",
                    "window_AXIS_V1",
                )
                if k in handoff_b
            }
            mode_results["B_inward"]["actions_n"] = len(actions_full)
            np.savez_compressed(OUT / "mode_b_actions.npz", actions=np.asarray(actions_full, np.float32))
            (OUT / "snapshots" / "B_mode_b_final.pkl").write_bytes(
                pickle.dumps({"capture": take_capture(env), "ref": blob_b["ref"]})
            )
    else:
        # Diagnose failure: pick least-bad trial for support inspection at abort
        print("  Mode B: no reorientation within Δex/Δθ budget — recording best abort case", flush=True)
        # Try one ultra-slow 35° for diagnostics
        diag = waypoint_reorient(
            env, blob_b, d=desire_direction("B_inward", u_open), axis_name="+wy", angle_deg=35.0, rate_scale=0.12, fn_target=13.0
        )
        mode_results["B_inward"]["ultra_slow_35"] = {
            k: v for k, v in diag.items() if k not in ("actions", "waypoints", "capture")
        }
        if diag.get("waypoints"):
            mode_results["B_inward"]["ultra_slow_35"]["waypoints"] = diag["waypoints"]

    # ========== Mode A (outward) on Plan A post-pivot (opposite sign) ==========
    print("=== Mode A outward reorientation search (Plan A post-pivot) ===", flush=True)
    restore_capture(env, blob_a["capture"])
    # Ensure A has a ref
    if blob_a.get("ref") is None:
        blob_a["ref"] = blob_b["ref"]
    support0_a = support_force_analysis(env.sim)
    # Mode A often already has g·d > 0 upright — check first
    d_a = desire_direction("A_outward", u_open)
    s_a0 = sample_state(env.sim)
    from data_generation.oracles.precision.orientation_planner import score_gravity_alignment

    g0_a = score_gravity_alignment(np.asarray(s_a0["g_h"], float), d_a)
    mode_results["A_outward"] = {
        "state": "plan_a_post_pivot",
        "e_open_mm": err_a["e_open_mm"],
        "g0_useful": g0_a["useful"],
        "already_sufficient_gravity": bool(g0_a["useful"] >= G_AXIAL_USEFUL_MIN),
        "support_before": {k: support0_a[k] for k in support0_a if k != "contacts"},
    }
    if g0_a["useful"] >= G_AXIAL_USEFUL_MIN:
        print(f"  Mode A: gravity already useful={g0_a['useful']:.2f} — skip deep tilt, try slip", flush=True)
        mode_results["A_outward"]["reori_ok"] = True
        mode_results["A_outward"]["reori_skipped"] = True
        z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
        slip_a = run_axial_slip(
            env, blob_a, d_a, milestone_mm=2.0, z_ref=z_ref, e_x0=float(err_a["e_signed_mm"][0])
        )
        print(
            f"  slip2 A: ok={slip_a.get('ok')} disp={slip_a.get('disp_along_mm')} "
            f"margin={slip_a.get('min_fingertip_margin_mm')} abort={slip_a.get('abort_reason')}",
            flush=True,
        )
        mode_results["A_outward"]["slip"] = {k: v for k, v in slip_a.items() if k not in ("actions", "capture")}
        if slip_a.get("ok"):
            handoff_a = handoff_rich(env, slip_a["capture"], blob_a["ref"])
            mode_results["A_outward"]["handoff"] = {
                k: handoff_a[k]
                for k in (
                    "SUCCESS_12S",
                    "SUCCESS_NEW",
                    "SUCCESS_AXIS_V1",
                    "pre_e_signed_mm",
                    "pre_theta",
                    "pre_pos_mm",
                    "reason",
                )
                if k in handoff_a
            }
            np.savez_compressed(OUT / "mode_a_actions.npz", actions=np.asarray(slip_a["actions"], np.float32))
    else:
        search_a = search_stable_reorientation(
            env,
            blob_a,
            mode="A_outward",
            u_open=u_open,
            angles=(25.0, 35.0, 45.0),
            rates=(0.15, 0.25),
            axes=("+wy", "-wy"),
        )
        mode_results["A_outward"]["reori_search"] = {k: v for k, v in search_a.items() if k != "best_full"}
        mode_results["A_outward"]["reori_ok"] = bool(search_a.get("any_ok"))
        best_a = search_a.get("best_full")
        if best_a and best_a.get("ok"):
            tilt = {"capture": best_a["capture"], "ref": blob_a["ref"]}
            slip_a = run_axial_slip(
                env,
                tilt,
                d_a,
                milestone_mm=2.0,
                z_ref=float(best_a.get("min_obj_z") or 0.5),
                e_x0=float(best_a["e0"][0]),
            )
            mode_results["A_outward"]["slip"] = {k: v for k, v in slip_a.items() if k not in ("actions", "capture")}

    # Conditional policy note for Plan B
    conditional = {
        "plan_b_e_open_mm": err_b["e_open_mm"],
        "within_AXIS_V1_ez_20": abs(err_b["e_open_mm"]) <= AXIS_V1_EZ_MM,
        "recommendation": (
            "prefer_direct_HANDOFF_AXIS_V1"
            if abs(err_b["e_open_mm"]) <= AXIS_V1_EZ_MM and not mode_results["B_inward"].get("reori_ok")
            else "attempt_mode_B_if_reori_stable"
        ),
        "branch_a_preserved": True,
        "note": "Do not force axial correction when AXIS_V1-stable and Mode B reori destroys lateral.",
    }

    # IC replay only if HANDOFF SUCCESS_NEW or improved AXIS with slip success
    replay = None
    if handoff_b and (handoff_b.get("SUCCESS_NEW") or (handoff_b.get("SUCCESS_AXIS_V1") and slip_b and slip_b.get("ok"))):
        print("=== IC 3× frozen replay Mode B ===", flush=True)
        prefix = []
        if STABLE_JSON.exists():
            stj = json.loads(STABLE_JSON.read_text(encoding="utf-8"))
            prefix = [np.asarray(a, np.float32) for a in stj.get("stabilization_prefix_actions") or []]
        lat = np.load(AXIS_PREFIX)["actions"] if AXIS_PREFIX.exists() else []
        mb = np.load(OUT / "mode_b_actions.npz")["actions"]
        results = []
        for trial in range(3):
            root = pickle.loads(IC_B.read_bytes())
            restore_capture(env, root["capture"])
            for a in prefix:
                step_legal(env.sim, a)
            for a in lat:
                step_legal(env.sim, a)
            for a in mb:
                step_legal(env.sim, a)
            hr = handoff_rich(env, take_capture(env), blob_b["ref"])
            results.append(
                {
                    "SUCCESS_12S": hr.get("SUCCESS_12S"),
                    "SUCCESS_NEW": hr.get("SUCCESS_NEW"),
                    "SUCCESS_AXIS_V1": hr.get("SUCCESS_AXIS_V1"),
                    "pre_e_signed_mm": hr.get("pre_e_signed_mm"),
                }
            )
        replay = {
            "SUCCESS_NEW_3of3": all(r["SUCCESS_NEW"] for r in results),
            "SUCCESS_AXIS_V1_3of3": all(r["SUCCESS_AXIS_V1"] for r in results),
            "trials": results,
        }

    # Plots / video
    plots = make_plots(OUT / "plots", mode_results)
    videos = {}
    if best_b and best_b.get("actions"):
        print("Rendering Mode B reori video...", flush=True)
        videos["mode_b_reori"] = render_video(
            env,
            blob_b,
            best_b["actions"],
            OUT / "videos" / "mode_b_reori.mp4",
            "Mode B inward reori",
            d=desire_direction("B_inward", u_open),
            plan={"axis_name": best_b.get("axis_name"), "angle_deg": best_b.get("angle_deg")},
        )
    elif mode_results["B_inward"].get("ultra_slow_35"):
        # render ultra-slow diagnostic if we have actions — re-run once to capture
        diag = waypoint_reorient(
            env, blob_b, d=desire_direction("B_inward", u_open), axis_name="+wy", angle_deg=35.0, rate_scale=0.12, fn_target=13.0
        )
        if diag.get("actions"):
            videos["mode_b_reori_fail_diag"] = render_video(
                env,
                blob_b,
                diag["actions"],
                OUT / "videos" / "mode_b_reori_fail_diag.mp4",
                "Mode B reori fail diag",
                d=desire_direction("B_inward", u_open),
                plan={"axis_name": "+wy", "angle_deg": 35.0},
            )

    # Write mode results (strip huge series optionally keep)
    for m in mode_results:
        if mode_results[m].get("slip") and mode_results[m]["slip"].get("series"):
            (OUT / f"slip_series_{m}.json").write_text(
                json.dumps(jsafe(mode_results[m]["slip"]["series"]), indent=2), encoding="utf-8"
            )
            mode_results[m]["slip"] = {k: v for k, v in mode_results[m]["slip"].items() if k != "series"}

    (OUT / "mode_results.json").write_text(json.dumps(jsafe(mode_results), indent=2), encoding="utf-8")
    (OUT / "conditional_policy.json").write_text(json.dumps(jsafe(conditional), indent=2), encoding="utf-8")
    if replay:
        (OUT / "replay_ic.json").write_text(json.dumps(jsafe(replay), indent=2), encoding="utf-8")

    # Verdict
    b_ok = mode_results["B_inward"].get("reori_ok")
    a_ok = mode_results["A_outward"].get("reori_ok") or mode_results["A_outward"].get("reori_skipped")
    a_slip = (mode_results["A_outward"].get("slip") or {}).get("ok")
    b_slip = (mode_results["B_inward"].get("slip") or {}).get("ok")
    if b_ok and b_slip and handoff_b and handoff_b.get("SUCCESS_NEW"):
        rec = "mode_b_inward_SUCCESS_NEW"
    elif a_ok and a_slip:
        rec = "mode_a_outward_slip_feasible_mode_b_reori_blocked"
    elif not b_ok and a_ok:
        rec = "mode_b_reorientation_fails_lateral_budget_prefer_AXIS_V1_handoff"
    else:
        rec = "sign_aware_axial_milestone_see_report"

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": round(time.time() - t0, 1),
        "convention": convention,
        "reori_targets_mm_deg": {"delta_ex_mm": REORI_DX_MM, "delta_theta_deg": REORI_DTHETA_DEG},
        "mode_results": mode_results,
        "conditional": conditional,
        "replay_ic": replay,
        "plots": plots,
        "videos": videos,
        "recommendation": rec,
        "preserved": {
            "SUCCESS_NEW": True,
            "SUCCESS_AXIS_V1": True,
            "plan_a_experts": True,
            "plan_b_branch_a": True,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    report = f"""# Sign-Aware Axial Sliding

Created UTC: {summary['created_utc']}  Wall: {summary['wall_s']} s

## 1. Opening-axis convention (verified)

- Finger bodies in hand frame: L/R at z≈+0.0584 m → **u_open = +hand_z** (palm → fingertips).
- `e_open = e · u_open` (mm).
- Plan B post-lateral: e_open = **{err_b['e_open_mm']:.2f} mm** → Mode B inward (toward palm) if correcting.
- Plan A post-pivot: e_open = **{err_a['e_open_mm']:.2f} mm** → Mode A outward (opposite sign).

## 2. Mode B inward (Plan B)

Reori search ok={mode_results['B_inward'].get('reori_ok')} clears_g={mode_results['B_inward'].get('reori_clears_g')}

Support before: proximal_frac={support0_b.get('proximal_fraction')} palm_detected={support0_b.get('palm_or_base_support_detected')}

Slip: {mode_results['B_inward'].get('slip')}

Handoff: {mode_results['B_inward'].get('handoff')}

## 3. Mode A outward (Plan A)

Already useful g: {mode_results['A_outward'].get('g0_useful')} skipped_reori={mode_results['A_outward'].get('reori_skipped')}

Slip: {mode_results['A_outward'].get('slip')}

Handoff: {mode_results['A_outward'].get('handoff')}

## 4. Conditional axial policy

{json.dumps(jsafe(conditional), indent=2)}

## 5. IC replay

{json.dumps(jsafe(replay), indent=2) if replay else 'Not run'}

## 6. Recommendation

**{rec}**

Engineering reori budgets: |Δe_x|≤{REORI_DX_MM} mm, |Δθ|≤{REORI_DTHETA_DEG}°.

Preserved: SUCCESS_NEW, SUCCESS_AXIS_V1, Plan A experts, Plan B Branch A.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(jsafe({"recommendation": rec, "wall_s": summary["wall_s"], "B_reori": b_ok, "A_slip": a_slip}), indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
