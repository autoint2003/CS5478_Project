#!/usr/bin/env python
"""Diagnose post-to_UP re-acceleration on Seq4 IC hard_910084.

Builds a phase-tagged action tape with MuJoCo orientation/ω traces across
to_UP stop → hold → slip_inward → return_DOWN, then identifies the first
frame where the gripper accelerates again after the UP stop.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import W_MAX
from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.contact_modes import classify_contact_mode
from data_generation.oracles.precision.endpoint_only_axial_roll import gates, u_open_world
from data_generation.oracles.precision.force_slip_control import DT
from data_generation.oracles.precision.precision_dz import correct_dz_inward
from data_generation.oracles.precision.precision_handoff import (
    posture_is_fingertips_down,
    run_final_down_handoff_gate,
)
from data_generation.oracles.precision.precision_types import PrecisionConfig
from data_generation.oracles.precision import precision_dz as pdz
from data_generation.oracles.precision import support_aware_rotation as sar
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from evaluation.grasp_geometry import cylinder_axis_world, hand_R
from models.transformer_policy import DT_POLICY

OUT = ROOT / "results/diagnostics/seq4_post_up_reaccel"
IC = ROOT / "results/data_generation/stage2_abc_final_v2/ics/B__hard_910084__post_b1_v2__71c154bb9a00bf73.pkl"
PRIOR_UP = ROOT / "results/diagnostics/seq4_to_up_brake_before_vertical/RESULT.json"


def _omega_hand(sim) -> np.ndarray:
    return np.asarray(sim.data.cvel[sim.ids.hand_body][0:3], float)


def _cmd_omega_world(sim, action) -> np.ndarray:
    """Map body-frame ω cmd (action[3:6] * W_MAX) to world."""
    a = np.asarray(action, float).reshape(-1)
    wb = a[3:6] * float(W_MAX)
    return hand_R(sim) @ wb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    cfg = PrecisionConfig()
    blob = pickle.loads(IC.read_bytes())

    # ---- Instrument run_fixed_axis_tilt to tag phases + collect per-step ----
    phase_segments: list[dict] = []
    tape: list[dict] = []  # global time series while we also rebuild via replay
    _orig = sar.run_fixed_axis_tilt

    def _wrap(env_, blob_, **kw):
        phase = kw.get("phase", "?")
        t0_seg = len(tape) * DT  # approximate; refined on full replay
        out = _orig(env_, blob_, **kw)
        n = len(out.get("actions") or [])
        phase_segments.append({
            "phase": phase,
            "n_actions": n,
            "ok": out.get("ok"),
            "reason": out.get("reason"),
            "stop_reason": out.get("stop_reason"),
            "ang0_deg": out.get("ang0_deg"),
            "ang1_deg": out.get("ang1_deg"),
            "alpha_final_deg": out.get("alpha_final_deg"),
            "target": "UP" if "to_UP" in phase else ("DOWN" if "DOWN" in phase or "down" in phase.lower() else "?"),
            "brake_begin": out.get("brake_begin"),
            "planner_hint": kw.get("phase"),
        })
        # stash actions with phase for later replay
        out["_phase_tag"] = phase
        out["_segment_meta"] = phase_segments[-1]
        return out

    sar.run_fixed_axis_tilt = _wrap
    pdz.run_fixed_axis_tilt = _wrap

    # Also wrap settle / slip by capturing correct_dz_inward phase_log + action count growth
    restore_capture(env, blob["capture"])
    start_cap = blob["capture"]
    rz = correct_dz_inward(env, {"capture": start_cap, "ref": blob["ref"]}, cfg, s1_state=None)
    actions = list(rz.get("actions") or [])
    phase_log = list(rz.get("phase_log") or [])
    rotation_diag = list(rz.get("rotation_diag") or [])

    gate = None
    handoff = None
    if rz.get("ok") and rz.get("capture") is not None:
        cur = {"capture": rz["capture"], "ref": blob["ref"]}
        gate = run_final_down_handoff_gate(env, cur, cfg, actions)
        if gate.get("ok"):
            cur = {"capture": gate["capture"], "ref": cur["ref"]}
            restore_capture(env, cur["capture"])
            handoff = evaluate_handoff_with_axis(
                env.sim, cur["ref"], stabilize_n=cfg.stabilize_handoff_n
            )

    # ---- Rebuild phase ranges from phase_log + segment n_actions ----
    # Reconstruct by replaying actions with phase attribution from segments in order.
    # correct_dz_inward also inserts unlabeled holds/slip between tilt segments.
    # Strategy: replay once from start, attribute using known segment action counts
    # and intervening unlabeled blocks inferred from total length.

    # Map tilt segments in order of appearance in phase_segments
    tilt_phases = [s for s in phase_segments]  # in call order

    # Infer unlabeled blocks by walking action tape while matching tilt chunks
    # Re-run a lightweight attribution: step through actions; assign phase by
    # consuming known tilt action counts when we hit them, else "hold_or_slip".
    # Better: re-execute DualLoop with step hook.

    # Full replay with phase cursor driven by segment action budgets + phase_log markers.
    restore_capture(env, start_cap)
    series = []
    # Build action→phase map by replaying instrumented path lengths from rz internals.
    # We know order: [S1 stuff] + to_UP(n) + hold(~0.2/DT) + slip(n_slip) + settle(0.25/DT)
    # + return_DOWN(n) + settle(0.35/DT) + gate extras
    n_up = next((s["n_actions"] for s in tilt_phases if "to_UP" in s["phase"]), 0)
    n_ret = next(
        (s["n_actions"] for s in tilt_phases if "return_DOWN" in s["phase"] or "DOWN" in s["phase"]),
        0,
    )
    # Prefer exact names
    n_ret = 0
    ret_phase_name = None
    for s in tilt_phases:
        if "return" in s["phase"].lower() or s["phase"].endswith("DOWN") or "DOWN" in s["phase"]:
            if "to_UP" in s["phase"]:
                continue
            n_ret = s["n_actions"]
            ret_phase_name = s["phase"]
            break
    # slip action count from phase_log / outcomes
    slip_info = next((p for p in phase_log if p.get("phase") == "slip_inward"), {})
    # Estimate hold/slip sizes from known constants in precision_dz
    n_hold_pre_slip = int(0.20 / DT)
    n_settle_pre_return = int(0.25 / DT)
    n_settle_post_return = int(0.35 / DT)

    # Find where to_UP starts: total before UP = len(actions) - (n_up + n_hold + n_slip + n_settle + n_ret + n_settle_post + gate)
    # Get slip n from re-deriving: after UP segment index
    # Walk: prefix = everything before first to_UP tilt actions
    # We stored tilt call order; first tilt may be S1 establish rolls via other APIs.

    # Count all tilt segment actions
    tilt_total = sum(s["n_actions"] for s in tilt_phases)
    unlabeled = len(actions) - tilt_total
    # Unlabeled = holds + slip + gate extras (if gate appended)
    n_gate_extra = max(0, len(actions) - (
        # rebuild expected from known structure for B_full path
        0
    ))

    # More reliable: patch step_legal during a second correct_dz_inward with phase state machine.
    # Do a dedicated second run with monkeypatched step_legal.

    phase_state = {"name": "pre", "sub": "init", "i": 0}
    series2: list[dict] = []
    action_phase: list[str] = []

    # Hook into modules used by DualLoop
    _orig_step = step_legal

    def _step_hook(sim, a):
        a = np.asarray(a, np.float32)
        g = gates(sim)
        u = u_open_world(sim)
        w_act = _omega_hand(sim)
        w_cmd_w = _cmd_omega_world(sim, a)
        # Before step: log
        row = {
            "i": len(series2),
            "t": len(series2) * DT,
            "phase": phase_state["name"],
            "subphase": phase_state["sub"],
            "alpha_up_deg": float(g["alpha_up_deg"]),
            "fingertips_up": bool(g["fingertips_up"]),
            "fingertips_down": bool(g["fingertips_down"]),
            "u_open": u.tolist(),
            "omega_act": w_act.tolist(),
            "omega_act_norm": float(np.linalg.norm(w_act)),
            "omega_cmd_world": w_cmd_w.tolist(),
            "omega_cmd_norm": float(np.linalg.norm(w_cmd_w)),
            "omega_cmd_body": (np.asarray(a, float)[3:6] * W_MAX).tolist(),
            "v_cmd_norm": float(np.linalg.norm(np.asarray(a, float)[:3])),
            "hand_pos": np.asarray(sim.data.xpos[sim.ids.hand_body], float).tolist(),
            "obj_pos": np.asarray(sim.data.xpos[sim.ids.object_body], float).tolist(),
        }
        series2.append(row)
        action_phase.append(f"{phase_state['name']}/{phase_state['sub']}")
        _orig_step(sim, a)
        # After step update omega for next is fine
        return None

    # Patch step_legal in the modules that import it
    import envs.step_legal as step_mod
    import data_generation.oracles.precision.support_aware_rotation as sar_mod
    import data_generation.oracles.precision.precision_dz as pdz_mod
    import data_generation.oracles.precision.force_slip_control as fsc_mod
    import data_generation.oracles.precision.precision_handoff as ph_mod

    # Also need slip module
    try:
        import data_generation.oracles.precision.precision_slip as pslip_mod
    except Exception:
        pslip_mod = None

    targets = [step_mod, sar_mod, pdz_mod, fsc_mod, ph_mod]
    if pslip_mod is not None:
        targets.append(pslip_mod)
    # endpoint / reverse modules that step
    try:
        import data_generation.oracles.precision.endpoint_only_axial_roll as ep_mod
        targets.append(ep_mod)
    except Exception:
        pass
    try:
        import data_generation.oracles.precision.reverse_supported_rotation as rev_mod
        targets.append(rev_mod)
    except Exception:
        pass
    try:
        import data_generation.oracles.precision.precision_dx_support_transfer as dxst
        targets.append(dxst)
    except Exception:
        pass

    # Phase markers via wrapping key functions
    _tilt_orig2 = sar_mod.run_fixed_axis_tilt

    def _tilt_phased(env_, blob_, **kw):
        phase = kw.get("phase", "tilt")
        prev = (phase_state["name"], phase_state["sub"])
        phase_state["name"] = phase
        phase_state["sub"] = "tilt_exec"
        if "to_UP" in phase:
            phase_state["name"] = "to_UP"
            phase_state["sub"] = "tilt_to_UP"
        elif "return" in phase.lower() or "DOWN" in phase:
            phase_state["name"] = "return_DOWN"
            phase_state["sub"] = phase
        out = _tilt_orig2(env_, blob_, **kw)
        phase_state["name"], phase_state["sub"] = prev
        return out

    # Wrap inward slip
    _slip_orig = getattr(pdz_mod, "_inward_slip_staged", None)

    def _slip_phased(*args, **kwargs):
        prev = (phase_state["name"], phase_state["sub"])
        phase_state["name"] = "slip_inward"
        phase_state["sub"] = "axial_inward"
        out = _slip_orig(*args, **kwargs)
        phase_state["name"], phase_state["sub"] = prev
        return out

    _settle_orig = getattr(pdz_mod, "settle_hold", None)

    def _settle_phased(*args, **kwargs):
        prev = (phase_state["name"], phase_state["sub"])
        hold_s = kwargs.get("hold_s", args[2] if len(args) > 2 else None)
        phase_state["name"] = "settle_hold"
        phase_state["sub"] = f"hold_{hold_s}s"
        out = _settle_orig(*args, **kwargs)
        phase_state["name"], phase_state["sub"] = prev
        return out

    # Install hooks
    for m in targets:
        if hasattr(m, "step_legal"):
            m.step_legal = _step_hook
    sar_mod.run_fixed_axis_tilt = _tilt_phased
    pdz_mod.run_fixed_axis_tilt = _tilt_phased
    if _slip_orig:
        pdz_mod._inward_slip_staged = _slip_phased
    if _settle_orig:
        pdz_mod.settle_hold = _settle_phased

    # Also mark the 0.20s pre-slip hold inside correct_dz_inward (inline loop, not settle_hold)
    # Monkeypatch make_action path won't help; detect via zero ω cmd after UP in analysis.

    series2.clear()
    action_phase.clear()
    phase_state.update({"name": "dz_inward", "sub": "entry", "i": 0})
    restore_capture(env, start_cap)
    rz2 = correct_dz_inward(env, {"capture": start_cap, "ref": blob["ref"]}, cfg, s1_state=None)
    actions2 = list(rz2.get("actions") or [])
    phase_log2 = list(rz2.get("phase_log") or [])

    # Continue with FINAL_DOWN gate under hook
    phase_state.update({"name": "final_down_gate", "sub": "handoff_prep"})
    gate2 = None
    if rz2.get("ok") and rz2.get("capture") is not None:
        cur = {"capture": rz2["capture"], "ref": blob["ref"]}
        gate2 = run_final_down_handoff_gate(env, cur, cfg, actions2)

    # Restore originals
    for m in targets:
        if hasattr(m, "step_legal"):
            m.step_legal = _orig_step
    sar_mod.run_fixed_axis_tilt = _orig
    pdz_mod.run_fixed_axis_tilt = _orig
    if _slip_orig:
        pdz_mod._inward_slip_staged = _slip_orig
    if _settle_orig:
        pdz_mod.settle_hold = _settle_orig

    if not series2:
        raise RuntimeError("No series captured — step_legal hook did not fire")

    # ---- Find UP stop and first re-acceleration ----
    # UP stop: last index in to_UP where omega_cmd_norm ~0 and omega_act decaying,
    # or last to_UP frame.
    idx_up = [i for i, r in enumerate(series2) if r["phase"] == "to_UP"]
    if not idx_up:
        # fallback: alpha near min then phase change
        alphas = np.array([r["alpha_up_deg"] for r in series2])
        idx_up = list(range(int(np.argmin(alphas))))

    i_up_start = idx_up[0]
    i_up_end = idx_up[-1]

    # Within UP: brake onset when cmd drops and stays low
    cmd = np.array([r["omega_cmd_norm"] for r in series2])
    act = np.array([r["omega_act_norm"] for r in series2])
    alpha = np.array([r["alpha_up_deg"] for r in series2])

    # UP stop frame: last to_UP with low cmd
    i_up_stop = i_up_end
    for i in range(i_up_end, i_up_start, -1):
        if cmd[i] < 0.02 and act[i] < 0.08:
            i_up_stop = i
            break
    # Prefer earliest of sustained low-ω window near end of UP
    for i in range(i_up_start, i_up_end + 1):
        if alpha[i] < 15.0 and cmd[i] < 0.02:
            # look ahead 5 steps still low cmd
            if np.all(cmd[i : min(i + 5, i_up_end + 1)] < 0.05):
                i_up_stop = i
                break

    # First re-accel after UP stop: omega_act or omega_cmd rises again
    i_reaccel = None
    reaccel_reason = None
    for i in range(i_up_stop + 1, len(series2)):
        # Commanded re-accel
        if cmd[i] > 0.05 and cmd[i] > cmd[i_up_stop] + 0.04:
            i_reaccel = i
            reaccel_reason = "commanded_omega_rise"
            break
        # Actual ω rise without necessarily large cmd (inertia) — skip small noise
        if act[i] > 0.12 and act[i] > act[max(i_up_stop, i - 1)] + 0.06 and cmd[i] > 0.03:
            i_reaccel = i
            reaccel_reason = "actual_omega_rise_with_cmd"
            break

    # Also detect by alpha leaving UP band toward DOWN (increasing alpha past 20)
    i_leave_up = None
    for i in range(i_up_stop + 1, len(series2)):
        if alpha[i] > max(alpha[i_up_stop] + 8.0, 20.0):
            i_leave_up = i
            break

    r_stop = series2[i_up_stop]
    r_rea = series2[i_reaccel] if i_reaccel is not None else None

    # Camera-relative check: hand/object world motion vs orientation change
    # Apparent motion if lookat/camera would move a lot while u_open stable
    def cam_lookat(r):
        ph = np.asarray(r["hand_pos"], float)
        po = np.asarray(r["obj_pos"], float)
        return 0.5 * po + 0.5 * ph

    cam_shift = None
    u_change = None
    if i_reaccel is not None:
        c0 = cam_lookat(r_stop)
        c1 = cam_lookat(r_rea)
        cam_shift = float(np.linalg.norm(c1 - c0))
        u0 = np.asarray(r_stop["u_open"], float)
        u1 = np.asarray(r_rea["u_open"], float)
        u_change = float(np.degrees(np.arccos(np.clip(np.dot(u0, u1), -1, 1))))

    # Classify cause
    phase_at_rea = r_rea["phase"] if r_rea else None
    sub_at_rea = r_rea["subphase"] if r_rea else None
    intentional_return = bool(
        phase_at_rea == "return_DOWN"
        or (phase_at_rea and "return" in str(phase_at_rea).lower())
        or (phase_at_rea and "DOWN" in str(phase_at_rea) and phase_at_rea != "to_UP")
    )
    intentional_slip = phase_at_rea == "slip_inward"
    # slip should be mostly translation; check ω vs v
    classification = "unknown"
    if intentional_return:
        classification = "intentional_RETURN_DOWN"
    elif intentional_slip:
        classification = "slip_inward_phase"
    elif phase_at_rea in ("settle_hold", "dz_inward") and r_rea and r_rea["omega_cmd_norm"] < 0.02:
        classification = "residual_inertia_or_camera"
    elif phase_at_rea == "to_UP":
        classification = "unintended_to_UP_reaccel"
    elif r_rea and r_rea["omega_cmd_norm"] > 0.05:
        classification = "unintended_commanded_rotation"
    else:
        classification = "investigate"

    # Necessity for DualLoop
    necessary = {
        "for_inward_DualLoop": classification in (
            "intentional_RETURN_DOWN", "slip_inward_phase"
        ),
        "for_DOWN_restoration": classification == "intentional_RETURN_DOWN",
        "note": (
            "DualLoop B_full = to_UP → slip_inward → return_DOWN. "
            "Post-UP rotation toward alpha≈180° is required for RETURN_DOWN_COMPLETED."
        ),
    }

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([r["t"] for r in series2])
        # Focus window: UP brake → +8s
        t0 = max(0.0, r_stop["t"] - 2.0)
        t1 = min(t[-1], (r_rea["t"] + 4.0) if r_rea else r_stop["t"] + 8.0)
        m = (t >= t0) & (t <= t1)

        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
        ax = axes[0]
        ax.plot(t[m], alpha[m], label="alpha_up (actual orient)", color="C0")
        ax.axhline(0.0, color="C0", ls=":", alpha=0.4, label="desired UP (0°)")
        ax.axhline(180.0, color="C3", ls=":", alpha=0.4, label="desired DOWN (180°)")
        ax.axvline(r_stop["t"], color="k", ls="--", label=f"UP stop t={r_stop['t']:.2f}")
        if r_rea:
            ax.axvline(r_rea["t"], color="C1", ls="--", label=f"reaccel t={r_rea['t']:.2f}")
        ax.set_ylabel("α_up [deg]")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Post-UP re-acceleration diagnosis (hard_910084)")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t[m], act[m], label="|ω_act| hand", color="C2")
        ax.plot(t[m], cmd[m], label="|ω_cmd| world", color="C3", alpha=0.85)
        ax.axvline(r_stop["t"], color="k", ls="--")
        if r_rea:
            ax.axvline(r_rea["t"], color="C1", ls="--")
        ax.set_ylabel("ω [rad/s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        # Phase as categorical
        phases = [r["phase"] for r in series2]
        uniq = []
        for p in phases:
            if p not in uniq:
                uniq.append(p)
        pmap = {p: i for i, p in enumerate(uniq)}
        ax.step(t[m], [pmap[p] for p in np.array(phases)[m]], where="post")
        ax.set_yticks(range(len(uniq)))
        ax.set_yticklabels(uniq, fontsize=7)
        ax.axvline(r_stop["t"], color="k", ls="--")
        if r_rea:
            ax.axvline(r_rea["t"], color="C1", ls="--")
        ax.set_ylabel("phase")
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        vcmd = np.array([r["v_cmd_norm"] for r in series2])
        ax.plot(t[m], vcmd[m], label="|v_cmd|", color="C4")
        ax.axvline(r_stop["t"], color="k", ls="--")
        if r_rea:
            ax.axvline(r_rea["t"], color="C1", ls="--")
        ax.set_ylabel("|v| cmd")
        ax.set_xlabel("t [s]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = OUT / "plots" / "post_up_reaccel_traces.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)

        # Zoom around reaccel ±1.5s
        if r_rea:
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            mz = (t >= r_rea["t"] - 1.5) & (t <= r_rea["t"] + 2.5)
            axes[0].plot(t[mz], alpha[mz], label="alpha")
            axes[0].axvline(r_rea["t"], color="C1", ls="--")
            axes[0].axvline(r_stop["t"], color="k", ls="--")
            axes[0].set_ylabel("α")
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[1].plot(t[mz], act[mz], label="|ω_act|")
            axes[1].plot(t[mz], cmd[mz], label="|ω_cmd|")
            axes[1].axvline(r_rea["t"], color="C1", ls="--")
            axes[1].set_ylabel("ω"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
            axes[2].step(t[mz], [pmap[p] for p in np.array(phases)[mz]], where="post")
            axes[2].set_yticks(range(len(uniq))); axes[2].set_yticklabels(uniq, fontsize=7)
            axes[2].axvline(r_rea["t"], color="C1", ls="--")
            axes[2].set_xlabel("t [s]"); axes[2].grid(True, alpha=0.3)
            fig.tight_layout()
            zoom_path = OUT / "plots" / "post_up_reaccel_zoom.png"
            fig.savefig(zoom_path, dpi=140)
            plt.close(fig)
        else:
            zoom_path = None
    except Exception as e:
        plot_path = None
        zoom_path = None
        plot_err = str(e)
    else:
        plot_err = None

    # Save CSV
    keys = [
        "i", "t", "phase", "subphase", "alpha_up_deg", "omega_act_norm", "omega_cmd_norm",
        "v_cmd_norm", "fingertips_up", "fingertips_down",
    ]
    lines = [",".join(keys)]
    for r in series2:
        lines.append(",".join(str(r.get(k, "")) for k in keys))
    (OUT / "series.csv").write_text("\n".join(lines), encoding="utf-8")

    # Phase spans
    spans = []
    cur_p = series2[0]["phase"]
    start_i = 0
    for i, r in enumerate(series2 + [{"phase": "__END__"}]):
        if r["phase"] != cur_p:
            spans.append({
                "phase": cur_p,
                "i0": start_i,
                "i1": i - 1,
                "t0": series2[start_i]["t"],
                "t1": series2[i - 1]["t"],
                "alpha0": series2[start_i]["alpha_up_deg"],
                "alpha1": series2[i - 1]["alpha_up_deg"],
                "n": i - start_i,
            })
            if r["phase"] == "__END__":
                break
            cur_p = r["phase"]
            start_i = i

    prior = json.loads(PRIOR_UP.read_text(encoding="utf-8")) if PRIOR_UP.exists() else {}

    result = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "ic": IC.name,
        "n_actions_rz": len(actions2),
        "n_series": len(series2),
        "phase_log": phase_log2,
        "phase_spans": spans,
        "up_stop": {
            "i": i_up_stop,
            "t": r_stop["t"],
            "phase": r_stop["phase"],
            "subphase": r_stop["subphase"],
            "alpha_up_deg": r_stop["alpha_up_deg"],
            "omega_act_norm": r_stop["omega_act_norm"],
            "omega_cmd_norm": r_stop["omega_cmd_norm"],
            "fingertips_up": r_stop["fingertips_up"],
        },
        "first_reaccel": None if r_rea is None else {
            "i": i_reaccel,
            "t": r_rea["t"],
            "dt_after_up_stop": r_rea["t"] - r_stop["t"],
            "phase": r_rea["phase"],
            "subphase": r_rea["subphase"],
            "alpha_up_deg": r_rea["alpha_up_deg"],
            "omega_act_norm": r_rea["omega_act_norm"],
            "omega_cmd_norm": r_rea["omega_cmd_norm"],
            "v_cmd_norm": r_rea["v_cmd_norm"],
            "detection": reaccel_reason,
            "delta_alpha_from_stop": r_rea["alpha_up_deg"] - r_stop["alpha_up_deg"],
        },
        "leave_up_band": None if i_leave_up is None else {
            "i": i_leave_up,
            "t": series2[i_leave_up]["t"],
            "phase": series2[i_leave_up]["phase"],
            "subphase": series2[i_leave_up]["subphase"],
            "alpha_up_deg": series2[i_leave_up]["alpha_up_deg"],
        },
        "camera_vs_actual": {
            "cam_lookat_shift_m": cam_shift,
            "u_open_change_deg": u_change,
            "interpretation": (
                "actual_rotation" if (u_change is not None and u_change > 5.0)
                else ("mostly_camera_or_translation" if (u_change is not None and u_change < 2.0) else "mixed_or_n/a")
            ),
        },
        "classification": classification,
        "necessary_for_DualLoop": necessary,
        "outcomes": rz2.get("outcomes"),
        "dz_ok": rz2.get("ok"),
        "branch": rz2.get("branch"),
        "SUCCESS_12S": (handoff or {}).get("SUCCESS_12S") if handoff else None,
        "prior_to_up_brake": (prior.get("to_up") or {}).get("brake_begin"),
        "plots": {"overview": str(plot_path) if plot_path else None, "zoom": str(zoom_path) if zoom_path else None, "error": plot_err},
        "code_change": "none — intentional RETURN_DOWN" if classification == "intentional_RETURN_DOWN" else "pending_if_unintended",
    }

    (OUT / "DIAGNOSIS.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    # Markdown report
    rea = result["first_reaccel"]
    lines_md = [
        "# Post-UP Re-acceleration Diagnosis",
        "",
        f"IC: `{IC.name}`",
        f"Generated: {result['generated']}",
        "",
        "## 1. UP stop frame",
        "",
        f"- frame i = **{result['up_stop']['i']}**, t = **{result['up_stop']['t']:.3f} s**",
        f"- phase/subphase: `{result['up_stop']['phase']}` / `{result['up_stop']['subphase']}`",
        f"- α = {result['up_stop']['alpha_up_deg']:.2f}°, |ω_act|={result['up_stop']['omega_act_norm']:.4f}, |ω_cmd|={result['up_stop']['omega_cmd_norm']:.4f}",
        "",
        "## 2. First re-acceleration after UP stop",
        "",
    ]
    if rea:
        lines_md += [
            f"- frame i = **{rea['i']}**, t = **{rea['t']:.3f} s** (Δt = {rea['dt_after_up_stop']:.3f} s after UP stop)",
            f"- **active phase/subphase: `{rea['phase']}` / `{rea['subphase']}`**",
            f"- α = {rea['alpha_up_deg']:.2f}° (Δα from stop = {rea['delta_alpha_from_stop']:.2f}°)",
            f"- |ω_act|={rea['omega_act_norm']:.4f}, |ω_cmd|={rea['omega_cmd_norm']:.4f}, |v_cmd|={rea['v_cmd_norm']:.4f}",
            f"- detection: `{rea['detection']}`",
            "",
        ]
    else:
        lines_md += ["- **No commanded re-acceleration detected after UP stop.**", ""]

    lines_md += [
        "## 3. Classification",
        "",
        f"- **{classification}**",
        f"- Camera vs actual: lookat shift={cam_shift}, u_open change={u_change}° → `{result['camera_vs_actual']['interpretation']}`",
        f"- Necessary for inward DualLoop: {necessary['for_inward_DualLoop']}",
        f"- Necessary for DOWN restoration: {necessary['for_DOWN_restoration']}",
        f"- Note: {necessary['note']}",
        "",
        "## 4. Phase spans",
        "",
    ]
    for s in spans:
        lines_md.append(
            f"- `{s['phase']}`: t=[{s['t0']:.2f},{s['t1']:.2f}] α={s['alpha0']:.1f}→{s['alpha1']:.1f} (n={s['n']})"
        )
    lines_md += [
        "",
        "## 5. DualLoop outcomes",
        "",
        f"- branch={result['branch']} ok={result['dz_ok']}",
        f"- outcomes={result['outcomes']}",
        "",
        f"Plots: `{plot_path}`",
        f"Zoom: `{zoom_path}`",
        f"Series: `{OUT / 'series.csv'}`",
        "",
    ]
    if classification == "intentional_RETURN_DOWN":
        lines_md += [
            "## 6. Action",
            "",
            "**No code change.** Post-UP rotation is intentional `return_DOWN` "
            f"(`{ret_phase_name or rea and rea.get('subphase')}`) after `slip_inward`, "
            "required for DualLoop DOWN restoration and subsequent FINAL_DOWN / HANDOFF.",
            "",
        ]
    else:
        lines_md += [
            "## 6. Action",
            "",
            "Motion not classified as intentional RETURN_DOWN — investigate fix at responsible phase.",
            "",
        ]

    (OUT / "REPORT.md").write_text("\n".join(lines_md), encoding="utf-8")
    print("\n".join(lines_md))


if __name__ == "__main__":
    main()
