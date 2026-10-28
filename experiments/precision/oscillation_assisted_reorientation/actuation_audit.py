#!/usr/bin/env python
"""Audit legal Panda action dimensions, grip modulation, and oscillation tracking.

Writes ACTUATION_AUDIT.md. Does not modify production.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import (  # noqa: E402
    AIRBORNE_V3,
    TAU_OPEN,
    TAU_SECURE,
    V_AIRBORNE,
    V_Z_DOWN,
    V_Z_UP,
    W_MAX,
    grip_tau,
    map_action,
    normalize_action,
)
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _action_vw  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import N_SUB, TRACK, step_legal  # noqa: E402
from experiments.precision.oscillation_assisted_reorientation.metrics import (  # noqa: E402
    hand_omega_world,
    omega_rel_hand,
)
from evaluation.grasp_geometry import hand_R  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/oscillation_assisted_reorientation"
DT = 0.02
AUDIT_IC = ROOT / "results/diagnostics/precision_candidate_audit/CLASSIFICATION.jsonl"


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def pick_ic():
    rows = [json.loads(l) for l in AUDIT_IC.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        if any(str(f).startswith("dtheta:") for f in ((r.get("precision") or {}).get("failures") or [])):
            return r
    return rows[0]


def measure_tracking(env, blob) -> dict:
    """Command sinusoidal body-y angular velocity; measure realized hand ω and grip response."""
    restore_capture(env, blob["capture"])
    amp_w = 0.6  # rad/s commanded (well below W_MAX=4)
    period = 1.0
    n = int(round(2.0 / DT))  # two periods
    series = []
    wh_prev = hand_omega_world(env.sim)
    t0 = float(env.sim.data.time)
    grip_hi, grip_lo = 0.9, -0.2
    for i in range(n):
        t = i * DT
        # sinusoid on body y
        w_cmd = amp_w * np.sin(2.0 * np.pi * t / period)
        # alternate grip every half period
        grip = grip_hi if (t % period) < 0.5 * period else grip_lo
        a = _action_vw([0.0, 0.0, 0.0], [0.0, float(w_cmd), 0.0], grip)
        # commanded physical via map
        rh = hand_R(env.sim)
        rd = np.asarray(env.sim.fsm.r_des, float).reshape(3, 3)
        cmd = map_action(a, rd, rh, AIRBORNE_V3)
        step_legal(env.sim, a)
        wh = hand_omega_world(env.sim)
        # approximate α from finite difference of hand ω
        alpha = (wh - wh_prev) / DT
        wh_prev = wh.copy()
        # projected onto commanded world axis ≈ rd @ [0,1,0]
        axis_w = rd @ np.array([0.0, 1.0, 0.0])
        axis_w = axis_w / max(float(np.linalg.norm(axis_w)), 1e-12)
        w_along = float(np.dot(wh, axis_w))
        series.append(
            {
                "t": t,
                "w_cmd_body_y": w_cmd,
                "w_cmd_world_norm": float(np.linalg.norm(cmd["w_world"])),
                "w_hand_along_cmd_axis": w_along,
                "w_hand_norm": float(np.linalg.norm(wh)),
                "alpha_norm": float(np.linalg.norm(alpha)),
                "grip_action": float(grip),
                "tau_cmd": float(cmd["tau"]),
                "fg_cmd": float(env.sim.fsm.fg_cmd),
                "tracking_ratio": (w_along / w_cmd) if abs(w_cmd) > 0.05 else None,
            }
        )
    ratios = [s["tracking_ratio"] for s in series if s["tracking_ratio"] is not None]
    # Grip step response: hold secure then open
    restore_capture(env, blob["capture"])
    grip_series = []
    for i in range(int(1.0 / DT)):
        g = 0.95 if i < 15 else -0.5
        a = _action_vw([0, 0, 0], [0, 0, 0], g)
        step_legal(env.sim, a)
        from data_generation.oracles.precision.mechanics.dual_loop import sample_state

        st = sample_state(env.sim)
        grip_series.append({"i": i, "grip": g, "tau": grip_tau(g), "fn_tot": st["fn_tot"]})

    return {
        "series": series,
        "mean_abs_tracking_ratio": float(np.mean(np.abs(ratios))) if ratios else None,
        "median_abs_tracking_ratio": float(np.median(np.abs(ratios))) if ratios else None,
        "peak_hand_w": max(s["w_hand_norm"] for s in series),
        "peak_cmd_w": max(s["w_cmd_world_norm"] for s in series),
        "peak_alpha": max(s["alpha_norm"] for s in series),
        "grip_series_tail": grip_series[::5],
        "fn_at_secure": grip_series[10]["fn_tot"] if len(grip_series) > 10 else None,
        "fn_after_open": grip_series[-1]["fn_tot"] if grip_series else None,
        "sim_dt": float(env.sim.model.opt.timestep),
        "policy_dt": DT,
        "n_sub": N_SUB,
        "track_gains": TRACK,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    row = pick_ic()
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    track = measure_tracking(env, blob)

    # interface summary
    iface = {
        "action_dim": 7,
        "version": AIRBORNE_V3,
        "channels": {
            "a[0:3]": f"hand-frame translation; vx,vy ±{V_AIRBORNE} m/s; +vz→{V_Z_DOWN}, -vz→{V_Z_UP} m/s",
            "a[3:6]": f"r_des-frame angular velocity; ±{W_MAX} rad/s each axis",
            "a[6]": f"grip; -1→τ={TAU_OPEN} N·m (open), +1→τ={TAU_SECURE} N·m (secure)",
        },
        "step_legal_dt_s": DT,
        "mujoco_substeps": N_SUB,
        "mujoco_timestep_s": track["sim_dt"],
        "note": "Commands are velocity-level Cartesian; joint limits/torque and TRACK PD determine realization.",
    }

    (OUT / "ACTUATION_AUDIT_RAW.json").write_text(
        json.dumps(
            {
                "sample_id": row["sample_id"],
                "interface": iface,
                "tracking": {k: v for k, v in track.items() if k != "series"},
                "series_len": len(track["series"]),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    # save series separately
    (OUT / "traces" / "actuation_tracking_series.json").write_text(
        json.dumps(track["series"], indent=2, default=float), encoding="utf-8"
    )

    ratio = track["mean_abs_tracking_ratio"]
    tracked = ratio is not None and ratio >= 0.25
    md = f"""# ACTUATION_AUDIT — OAR prerequisites

Generated: {datetime.now(timezone.utc).isoformat()}

## Legal action interface (unchanged)

- Version: `{AIRBORNE_V3}` via `step_legal` (official 20 ms recovery step).
- Dimension: **7** (vx, vy, vz, wx, wy, wz, grip), clipped to [-1, 1].
- Translation (hand frame): vx, vy ∈ ±{V_AIRBORNE} m/s; vz asymmetric (+{V_Z_DOWN} / −{V_Z_UP} m/s).
- Angular (r_des / body frame): each of wx, wy, wz ∈ ±**{W_MAX} rad/s**.
- Grip: action −1 → τ = {TAU_OPEN} N·m (open), +1 → τ = {TAU_SECURE} N·m (secure). Linear map via `grip_tau`.
- Policy step: **{DT} s** with **{N_SUB}** MuJoCo substeps (sim dt ≈ {track['sim_dt']} s).
- Cartesian TRACK gains: `{TRACK}`.

Oscillatory OAR commands use the **existing** `normalize_action` / `step_legal` path only — no control-interface change.

## Measured tracking (IC `{row['sample_id']}`)

Command: sinusoidal body-y ω with amplitude {0.6} rad/s, period 1.0 s, 2 s horizon; grip toggled mid-period.

| Metric | Value |
|---|---|
| Peak commanded ‖ω‖ | {track['peak_cmd_w']:.3f} rad/s |
| Peak realized hand ‖ω‖ | {track['peak_hand_w']:.3f} rad/s |
| Mean \|ω_along / ω_cmd\| | {ratio} |
| Median \|ratio\| | {track['median_abs_tracking_ratio']} |
| Peak ‖α_hand‖ (FD) | {track['peak_alpha']:.2f} rad/s² |
| fn_tot @ secure sample | {track['fn_at_secure']} |
| fn_tot after open cmd | {track['fn_after_open']} |

**Tracking usable for OAR?** {"YES (partial)" if tracked else "WEAK / ACTUATION_NOT_TRACKED risk"} —
commanded sinusoids are **not** perfectly realized; OAR must score net relative rotation from measured states, not from the command waveform.

## Grip modulation bandwidth

Grip action can change every 20 ms policy step. Contact force (`fn_tot`) responds within a few steps but is **not** an instantaneous map of τ — friction/contact dynamics filter the response. Fast grip toggling is interface-legal; effectiveness is an empirical OAR question (`GRIP_MODULATION_INEFFECTIVE` if force barely moves).

## Implications for OAR parameters

- Keep commanded ‖ω‖ well below {W_MAX} rad/s (start ≈ 0.4–1.0 rad/s).
- Angular amplitude per half-cycle: start small (≈ 8–15° ⇒ ~0.14–0.26 rad).
- Half-period ≥ 0.4 s so TRACK can accelerate/decelerate without saturating.
- Do not assume phase-perfect grip–ω synchronization; log both commanded and measured signals.

Raw series: `traces/actuation_tracking_series.json`.
"""
    (OUT / "ACTUATION_AUDIT.md").write_text(md, encoding="utf-8")
    print(md[:800], flush=True)
    print(f"wrote {OUT / 'ACTUATION_AUDIT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
