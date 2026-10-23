#!/usr/bin/env python
"""What dz modes appear when |ex|<=dz_safe_ex among stable DOWN B grasps?"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
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

OUT = ROOT / "results/diagnostics/seq4_genuine_dz_inward"


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
    safe = []
    all_inward = []
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
            if not entry_gate(env).get("ok"):
                continue
            restore_capture(env, blob["capture"])
            if not hold_and_confirm_stable(env, hold_s=0.15).get("stable"):
                continue
            restore_capture(env, blob["capture"])
            err = measure_errors(env.sim)
            if not posture_is_fingertips_down(env.sim) or not err["bilateral"]:
                continue
            axial = select_axial_mode(float(err["dz_mm"]), axial_tol_mm=float(cfg.ez_tol_mm))
            # also with tighter tol to see any inward signal
            axial2 = select_axial_mode(float(err["dz_mm"]), axial_tol_mm=2.0)
            entry = check_dz_inward_entry(env.sim, cfg)
            row = {
                "name": p.name,
                "path": str(p),
                "ex": abs(float(err["dx_mm"])),
                "ez": float(err["dz_mm"]),
                "ey": abs(float(err["dy_mm"])),
                "epos": float(err["e_pos_mm"]),
                "th": float(err["dtheta_deg"]),
                "mode_ez20": axial.get("mode"),
                "mode_ez2": axial2.get("mode"),
                "entry_ok": bool(entry.get("ok")),
                "reasons": list(entry.get("reasons") or []),
            }
            if row["ex"] <= float(cfg.dz_safe_ex_mm):
                safe.append(row)
            if axial.get("mode") == "B_inward" or axial2.get("mode") == "B_inward":
                all_inward.append(row)

    modes = Counter(r["mode_ez20"] for r in safe)
    modes2 = Counter(r["mode_ez2"] for r in safe)
    print("safe |ex|<=6 count", len(safe), "modes@ez20", dict(modes), "modes@ez2", dict(modes2))
    # Any safe with |ez| meaningful even if mode none under 20mm?
    safe_sorted = sorted(safe, key=lambda r: -abs(r["ez"]))
    print("safe largest |ez|:")
    for r in safe_sorted[:15]:
        print(
            f"  ez={r['ez']:+.2f} ex={r['ex']:.2f} epos={r['epos']:.2f} "
            f"mode20={r['mode_ez20']} mode2={r['mode_ez2']} entry={r['entry_ok']} {r['name'][:50]}"
        )

    # inward need under ez_tol=2 among safe
    safe_in = [r for r in safe if r["mode_ez2"] == "B_inward" and abs(r["ez"]) >= 2.0]
    print("safe + inward@ez2", len(safe_in))
    for r in safe_in[:10]:
        print(f"  {r['name'][:55]} ex={r['ex']:.2f} ez={r['ez']:.2f} entry={r['entry_ok']} {r['reasons']}")

    # all inward (any ex) with entry_ok
    ok_in = [r for r in all_inward if r["entry_ok"]]
    print("any-ex entry_ok inward", len(ok_in))
    for r in sorted(ok_in, key=lambda x: x["ex"])[:10]:
        print(f"  {r['name'][:55]} ex={r['ex']:.2f} ez={r['ez']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SAFE_EX_MODES.json").write_text(
        json.dumps(
            {
                "n_safe": len(safe),
                "modes_ez20": dict(modes),
                "modes_ez2": dict(modes2),
                "safe_top_ez": safe_sorted[:30],
                "safe_inward_ez2": safe_in,
                "entry_ok_inward_any_ex": ok_in[:20],
                "dz_safe_ex_mm": float(cfg.dz_safe_ex_mm),
                "ez_tol_mm": float(cfg.ez_tol_mm),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
