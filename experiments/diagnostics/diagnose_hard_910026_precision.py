#!/usr/bin/env python
"""Single-IC deep trace: hard_910026 Precision Expert failure.

Diagnosis only — does not modify Oracle, M111, datasets, or success criteria.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.diagnostics import run_b_grpo_multistage_dataset as base  # noqa: E402

from data_generation.oracles.precision.corrections.residual_directions import lateral_desire_direction  # noqa: E402
from data_generation.oracles.precision.mechanics.dual_loop import sample_state  # noqa: E402
from data_generation.oracles.precision.corrections.lateral import correct_dx_verified  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.entry_conditions import assess_need
from data_generation.oracles.precision.residuals import (
    handoff_geometry_ready,
    measure_errors,
)
from data_generation.oracles.precision.handoff import (  # noqa: E402
    prepare_stable_handoff,
    probe_direct_handoff,
)
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import (  # noqa: E402
    AXIS_V1_EX_MM,
    AXIS_V1_EZ_MM,
    AXIS_V1_THETA_DEG,
    evaluate_handoff_with_axis,
)
from evaluation.success_criteria import WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

DS = ROOT / "results/data_generation/stage2_b_precision_dominant_v1"
OUT = ROOT / "results/diagnostics/precision_expert_failure_analysis"
IC_ID = "hard_910026"
SID = "B__diagnostic__hard_910026__5e9806de8bc40b97"


def pack(sim):
    e = measure_errors(sim)
    st = sample_state(sim)
    return {
        **{k: e[k] for k in (
            "dx_mm", "dy_mm", "dz_mm", "dtheta_deg", "e_pos_mm", "bilateral",
            "fn_L", "fn_R", "fn_tot", "obj_z", "fingertips_down", "fingertips_up",
        )},
        "v_rel": float(st.get("v_rel_norm") or 0),
        "e_signed_mm": e["e_signed_mm"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = base.make_plain_env()
    ic = pickle.loads((DS / "ics" / f"{SID}__recovery.pkl").read_bytes())
    s1_acts = np.asarray(np.load(DS / "stage1" / f"{SID}__stage1.npz")["actions"], np.float32)

    # --- Stage-1 replay → Precision entry ---
    restore_capture(env, ic["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in s1_acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    entry_cap = take_capture(env)
    entry = pack(env.sim)
    cfg = PrecisionConfig()
    need0 = assess_need(measure_errors(env.sim), cfg)
    d_lat = lateral_desire_direction(entry["e_signed_mm"])

    print("=== ENTRY (post Stage-1) ===", flush=True)
    print(json.dumps(base.jsafe({
        "entry": entry,
        "need": need0,
        "thresholds": {
            "AXIS_ex": AXIS_V1_EX_MM,
            "AXIS_ez": AXIS_V1_EZ_MM,
            "AXIS_theta": AXIS_V1_THETA_DEG,
            "WORKING_POS_MM": WORKING_POS_MM,
            "WORKING_THETA_DEG": WORKING_THETA_DEG,
            "lateral_axis_tol_mm": cfg.lateral_axis_tol_mm,
            "dz_safe_ex_mm": cfg.dz_safe_ex_mm,
        },
        "lateral_desire_direction": d_lat.tolist(),
        "euclid_check": float(np.linalg.norm([entry["dx_mm"], entry["dy_mm"], entry["dz_mm"]])),
    }), indent=2), flush=True)

    # --- Isolated support-transfer (preferred dx) ---
    restore_capture(env, entry_cap)
    st = correct_dx_support_transfer(env, {"capture": entry_cap, "ref": ic["ref"]}, cfg)
    print("\n=== DX support_transfer (isolated) ===", flush=True)
    print(json.dumps(base.jsafe({
        "ok": st.get("ok"),
        "skipped": st.get("skipped"),
        "reason": st.get("reason"),
        "failure_code": st.get("failure_code"),
        "phase": st.get("phase"),
        "n_actions": len(st.get("actions") or []),
        "errors_after": st.get("errors"),
        "timing_s": st.get("timing_s"),
    }), indent=2), flush=True)

    # --- Isolated verified lateral (fallback used in baseline) ---
    restore_capture(env, entry_cap)
    vl = correct_dx_verified(env, {"capture": entry_cap, "ref": ic["ref"]}, cfg)
    after_vl = pack(env.sim) if vl.get("ok") else None
    print("\n=== DX verified_lateral (isolated) ===", flush=True)
    print(json.dumps(base.jsafe({
        "ok": vl.get("ok"),
        "skipped": vl.get("skipped"),
        "reason": vl.get("reason"),
        "phase": vl.get("phase"),
        "stop_reason": vl.get("stop_reason"),
        "goal_reached": vl.get("goal_reached"),
        "disp_along_mm": vl.get("disp_along_mm"),
        "disp_total_mm": vl.get("disp_total_mm"),
        "n_actions": len(vl.get("actions") or []),
        "errors_logged": vl.get("errors"),
        "after_live": after_vl,
        "use_euclidean_stop": False,
        "lateral_axis_tol_mm": cfg.lateral_axis_tol_mm,
        "delta": None if after_vl is None else {
            "dx": after_vl["dx_mm"] - entry["dx_mm"],
            "dz": after_vl["dz_mm"] - entry["dz_mm"],
            "theta": after_vl["dtheta_deg"] - entry["dtheta_deg"],
            "e_pos": after_vl["e_pos_mm"] - entry["e_pos_mm"],
        },
    }), indent=2), flush=True)

    if after_vl is not None:
        need1 = assess_need(measure_errors(env.sim), cfg)
        geo = handoff_geometry_ready(measure_errors(env.sim), cfg)
        print("\n=== assess_need AFTER verified_lateral ===", flush=True)
        print(json.dumps(base.jsafe({
            "need": need1,
            "handoff_geometry_ready": geo,
            "margin_to_WORKING_POS": WORKING_POS_MM - after_vl["e_pos_mm"],
            "dominant_residual_axis": (
                "dz" if abs(after_vl["dz_mm"]) >= abs(after_vl["dx_mm"]) else "dx"
            ),
            "why_dz_not_flagged": (
                f"|dz|={abs(after_vl['dz_mm']):.3f} <= AXIS_ez={AXIS_V1_EZ_MM} "
                f"and e_pos={after_vl['e_pos_mm']:.3f} <= WORKING_POS={WORKING_POS_MM} "
                f"→ residual_note branch inactive"
            ),
        }), indent=2), flush=True)

        # Probe handoff from post-vl state (before prepare)
        probe = probe_direct_handoff(env, take_capture(env), ic["ref"], stabilize_n=cfg.stabilize_handoff_n)
        win = probe.get("handoff") or probe
        print("\n=== handoff probe AFTER verified_lateral (pre-prepare) ===", flush=True)
        # evaluate_handoff returns nested; probe_direct wraps it
        h = probe if "SUCCESS_NEW" in probe else (probe.get("handoff") or {})
        # probe_direct_handoff structure
        print(json.dumps(base.jsafe({
            "SUCCESS_NEW": probe.get("SUCCESS_NEW"),
            "SUCCESS_12S": probe.get("SUCCESS_12S"),
            "reason": probe.get("reason"),
            "pre_theta": probe.get("pre_theta") or (h.get("pre_theta") if isinstance(h, dict) else None),
            "pre_pos_mm": probe.get("pre_pos_mm") or (h.get("pre_pos_mm") if isinstance(h, dict) else None),
            "window": (h.get("window_SUCCESS_NEW") if isinstance(h, dict) else None) or probe.get("window_SUCCESS_NEW"),
            "classify_hint": None,
        }), indent=2), flush=True)

        # Full evaluate for window detail
        restore_capture(env, vl["capture"] if vl.get("capture") is not None else entry_cap)
        # re-run to get capture from live after_vl
        restore_capture(env, entry_cap)
        vl2 = correct_dx_verified(env, {"capture": entry_cap, "ref": ic["ref"]}, cfg)
        cap_vl = vl2.get("capture") or take_capture(env)
        restore_capture(env, cap_vl)
        prep = prepare_stable_handoff(env, {"capture": cap_vl, "ref": ic["ref"]}, [])
        print("\n=== prepare_stable_handoff after VL ===", flush=True)
        print(json.dumps(base.jsafe({
            "prep_ok": prep.get("ok"),
            "n_prep_actions": len(prep.get("actions") or []) if "actions" in prep else "in-place",
            "errors": prep.get("errors"),
            "live_after_prep": pack(env.sim),
        }), indent=2), flush=True)

        restore_capture(env, prep.get("capture") or take_capture(env))
        hand = evaluate_handoff_with_axis(env.sim, ic["ref"], stabilize_n=cfg.stabilize_handoff_n)
        win = hand.get("window_SUCCESS_NEW") or {}
        print("\n=== FINAL HANDOFF (after prepare) — SUCCESS_NEW gate ===", flush=True)
        print(json.dumps(base.jsafe({
            "SUCCESS_NEW": hand.get("SUCCESS_NEW"),
            "SUCCESS_12S": hand.get("SUCCESS_12S"),
            "reason": hand.get("reason"),
            "pre_theta": hand.get("pre_theta"),
            "pre_pos_mm": hand.get("pre_pos_mm"),
            "pre_e_signed_mm": hand.get("pre_e_signed_mm"),
            "pre_in_band_NEW": hand.get("pre_in_band_NEW"),
            "window_SUCCESS_NEW": win,
            "unsatisfied_NEW_condition": (
                "working_SUCCESS_NEW_hold failed: need ≥0.5s continuous "
                f"(θ≤{WORKING_THETA_DEG}° AND e_pos≤{WORKING_POS_MM}mm) in final 2s window; "
                f"got longest_both_ok_s={win.get('longest_both_ok_s')}, "
                f"max_theta={win.get('max_theta')}, max_pos={win.get('max_pos')}, "
                f"mean_theta={win.get('mean_theta')}, mean_pos={win.get('mean_pos')}"
            ),
            "samples_tail_stats": _tail_stats(hand.get("samples_tail") or []),
        }), indent=2), flush=True)

    # --- Full unified run (confirm phases) ---
    restore_capture(env, entry_cap)
    res = run_precision(env, {"capture": entry_cap, "ref": ic["ref"]}, "B", cfg)
    logs = res.primitive_logs or {}
    print("\n=== UNIFIED run_precision summary ===", flush=True)
    print(json.dumps(base.jsafe({
        "status": res.status,
        "phases": res.phases,
        "failures": res.failures,
        "dx_mechanism": logs.get("dx_mechanism"),
        "dx_support_transfer": {
            k: logs.get("dx_support_transfer", {}).get(k)
            for k in ("ok", "reason", "failure_code", "phase", "skipped")
        },
        "dx_verified": {
            k: (logs.get("dx_verified") or {}).get(k)
            for k in ("ok", "reason", "phase", "stop_reason", "goal_reached", "disp_along_mm")
        },
        "cycles": [
            {
                "cycle": c.get("cycle"),
                "primitive": c.get("primitive"),
                "ok": c.get("ok"),
                "need": c.get("need"),
                "handoff_probe": c.get("handoff_probe"),
                "errors_e_pos": (c.get("errors") or {}).get("e_pos_mm"),
                "errors_dx": (c.get("errors") or {}).get("dx_mm"),
                "errors_dz": (c.get("errors") or {}).get("dz_mm"),
                "errors_theta": (c.get("errors") or {}).get("dtheta_deg"),
            }
            for c in (logs.get("cycles") or [])
        ],
        "handoff": {k: (res.handoff or {}).get(k) for k in (
            "SUCCESS_NEW", "SUCCESS_12S", "reason", "pre_theta", "pre_pos_mm",
            "pre_in_band_NEW", "window_SUCCESS_NEW",
        )},
        "n_actions": len(res.actions or []),
    }), indent=2), flush=True)

    report = {
        "ic_id": IC_ID,
        "first_divergence": {
            "intended": "correct_dx via support-transfer (gravity-assisted dx under S1)",
            "actual": "support_transfer failed: support_lost_during_rotation @ DX2_horizontal",
            "fallback": "dx_verified_lateral executed and marked ok",
        },
        "verified_lateral_effect": {
            "intended_stop": f"|dx| <= lateral_axis_tol_mm={cfg.lateral_axis_tol_mm} (euclidean_stop=False)",
            "actual": "dx reduced strongly; dz and theta increased; e_pos only barely under WORKING_POS",
        },
        "scheduler_stop": {
            "reason": "NO_REMAINING_CORRECTION despite handoff_probe SUCCESS_NEW=False",
            "mechanism": (
                "assess_need / AXIS tolerances: |dx|<12, |dz|<20, θ<AXIS_theta, "
                "and e_pos<=12 so residual_note does not force dz; "
                "handoff_geometry_ready True; probe NEW fails but scheduler still exits"
            ),
        },
        "unsatisfied_SUCCESS_NEW": (
            "SUCCESS_12S holds, but working_SUCCESS_NEW_hold fails "
            "(need 0.5s with θ≤10° and e_pos≤12mm in final 2s window)"
        ),
    }
    (OUT / "HARD_910026_DIAGNOSIS.json").write_text(
        json.dumps(base.jsafe(report), indent=2), encoding="utf-8"
    )
    print("\nWrote", OUT / "HARD_910026_DIAGNOSIS.json", flush=True)


def _tail_stats(tail: list) -> dict:
    if not tail:
        return {}
    th = [float(s["e_theta_deg"]) for s in tail]
    pos = [float(s["e_pos_mm"]) for s in tail]
    ok = [(t <= WORKING_THETA_DEG and p <= WORKING_POS_MM) for t, p in zip(th, pos)]
    return {
        "n": len(tail),
        "theta_min_max_mean": [min(th), max(th), float(np.mean(th))],
        "pos_min_max_mean": [min(pos), max(pos), float(np.mean(pos))],
        "frac_in_WORKING_band": float(np.mean(ok)),
        "n_frames_pos_over_12": int(sum(1 for p in pos if p > WORKING_POS_MM)),
        "n_frames_theta_over_10": int(sum(1 for t in th if t > WORKING_THETA_DEG)),
    }


if __name__ == "__main__":
    main()
