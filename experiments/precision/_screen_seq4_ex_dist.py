#!/usr/bin/env python
"""|ex| distribution among gate-ok DOWN bilateral B grasps."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.corrections.axial_inward import check_dz_inward_entry
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.handoff import posture_is_fingertips_down
from data_generation.oracles.precision.state import PrecisionConfig
from data_generation.oracles.precision.corrections.axial_sign import select_axial_mode
from data_generation.oracles.precision.mechanics.grasp_stabilization import hold_and_confirm_stable
from data_generation.oracles.precision.state_machine import entry_gate
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY


def main():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    cfg = PrecisionConfig()
    dirs = [
        ROOT / "results/data_generation/stage2_b_precision_dominant_v1/ics",
        ROOT / "results/data_generation/stage2_abc_final_v2/ics",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
        ROOT / "results/diagnostics/abc_precision_architecture/stable_states",
    ]
    rows = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.pkl")):
            if d.name == "snapshots" and not p.name.startswith("hard_"):
                continue
            try:
                blob = pickle.loads(p.read_bytes())
            except Exception:
                continue
            if "capture" not in blob or "ref" not in blob:
                continue
            restore_capture(env, blob["capture"])
            gate = entry_gate(env)
            restore_capture(env, blob["capture"])
            stab = hold_and_confirm_stable(env, hold_s=0.15)
            restore_capture(env, blob["capture"])
            if not (gate.get("ok") and stab.get("stable")):
                continue
            err = measure_errors(env.sim)
            if not posture_is_fingertips_down(env.sim) or not err["bilateral"]:
                continue
            axial = select_axial_mode(float(err["dz_mm"]), axial_tol_mm=float(cfg.ez_tol_mm))
            entry = check_dz_inward_entry(env.sim, cfg)
            rows.append(
                {
                    "name": p.name,
                    "dir": str(d.relative_to(ROOT)),
                    "ex": abs(float(err["dx_mm"])),
                    "ez": float(err["dz_mm"]),
                    "epos": float(err["e_pos_mm"]),
                    "th": float(err["dtheta_deg"]),
                    "mode": axial.get("mode"),
                    "entry_ok": bool(entry.get("ok")),
                    "reasons": list(entry.get("reasons") or []),
                }
            )

    exs = np.array([r["ex"] for r in rows]) if rows else np.array([])
    print("gate_ok DOWN bilateral count", len(rows))
    if not len(rows):
        return
    print("ex percentiles", np.percentile(exs, [0, 10, 25, 50, 75, 90, 100]))
    print("ex<=6", int(np.sum(exs <= 6)), "ex<=8", int(np.sum(exs <= 8)), "ex<=12", int(np.sum(exs <= 12)))
    near = [r for r in rows if r["ex"] <= 8 and r["mode"] == "B_inward" and abs(r["ez"]) >= 2]
    near.sort(key=lambda r: (r["ex"], -abs(r["ez"])))
    print("near ex<=8 inward", len(near))
    for r in near[:20]:
        print(
            f"  {r['name'][:58]} ex={r['ex']:.2f} ez={r['ez']:.2f} entry={r['entry_ok']} {r['reasons']}"
        )
    ok = [r for r in rows if r["entry_ok"] and r["mode"] == "B_inward" and abs(r["ez"]) >= 2]
    print("strict entry_ok inward", len(ok))
    for r in ok[:15]:
        print(f"  OK {r['name'][:58]} ex={r['ex']:.2f} ez={r['ez']:.2f} dir={r['dir']}")
    inv = [r for r in rows if r["mode"] == "B_inward" and abs(r["ez"]) >= 2]
    inv.sort(key=lambda r: r["ex"])
    print("smallest-ex inward top8:")
    for r in inv[:8]:
        print(f"  {r['name'][:60]} ex={r['ex']:.2f} ez={r['ez']:.2f} entry={r['entry_ok']}")


if __name__ == "__main__":
    main()
