#!/usr/bin/env python
"""Instrument dx_support_transfer DX1→DX2 on hard_910026 (diagnosis only)."""

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
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode, make_s1_state  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import (  # noqa: E402
    DOWN,
    choose_roll_sign_for_support,
    gates,
    reference_pad_vector,
    run_roll_phase,
    run_tilt_phase,
    u_open_world,
)
from data_generation.oracles.precision.mechanics.dual_loop import (  # noqa: E402
    contact_forces,
    sample_state,
)
from data_generation.oracles.precision.mechanics.finger_surface import support_force_decomposition  # noqa: E402
from data_generation.oracles.precision.corrections.lateral import (  # noqa: E402
    _horizontal_open_target,
    _physical_finger_support,
)
from data_generation.oracles.precision.residuals import measure_errors
from data_generation.oracles.precision.state import FN_HOLD, PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.mechanics.support_transfer import (  # noqa: E402
    correct_dx_support_transfer,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import hand_R  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

DS = ROOT / "results/data_generation/stage2_b_precision_dominant_v1"
OUT = ROOT / "results/diagnostics/precision_expert_failure_analysis"
SID = "B__diagnostic__hard_910026__5e9806de8bc40b97"
DT = 0.02


def snap(sim) -> dict:
    s = sample_state(sim)
    cf = contact_forces(sim)
    mode = classify_contact_mode(sim)
    g = gates(sim)
    sup = support_force_decomposition(sim)
    sides = {}
    for k, v in (sup.get("sides") or {}).items():
        sides[k] = {
            "n_contacts": v.get("n_contacts"),
            "Fn_z_normal": float(v.get("Fn_z_normal") or 0),
            "Ft_z_friction": float(v.get("Ft_z_friction") or 0),
            "normal_support_meaningful": v.get("normal_support_meaningful"),
        }
    return {
        "bilateral_geom": bool(s["bilateral"]),
        "fn_L": float(cf["fn_L"]),
        "fn_R": float(cf["fn_R"]),
        "fn_tot": float(cf["fn_tot"]),
        "bilateral_force": bool(s["bilateral"] and cf["fn_L"] > 1.0 and cf["fn_R"] > 1.0),
        "mode": mode["mode"],
        "support_side": mode["support_side"],
        "one_sided_mechanical": mode["one_sided_mechanical"],
        "Fn_z_L": mode["Fn_z_L"],
        "Fn_z_R": mode["Fn_z_R"],
        "alpha_up_deg": g["alpha_up_deg"],
        "fingertips_down": g["fingertips_down"],
        "e_signed_mm": s["e_signed_mm"],
        "e_pos_mm": s["e_pos_mm"],
        "e_theta_deg": s["e_theta_deg"],
        "v_rel": float(s["v_rel_norm"]),
        "w_rel": float(s.get("w_rel_norm") or 0),
        "obj_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "sides": sides,
        "u_open_w": u_open_world(sim).tolist(),
    }


def replay_stage1(env, ic, s1_acts):
    restore_capture(env, ic["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in s1_acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    return take_capture(env), snap(env.sim)


def series_first_bilateral_loss(series: list) -> dict | None:
    prev_bil = True
    for row in series:
        bil = bool(row.get("bilateral"))
        if prev_bil and not bil:
            return row
        prev_bil = bil
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = base.make_plain_env()
    ic = pickle.loads((DS / "ics" / f"{SID}__recovery.pkl").read_bytes())
    s1_acts = np.asarray(np.load(DS / "stage1" / f"{SID}__stage1.npz")["actions"], np.float32)
    cfg = PrecisionConfig()

    entry_cap, entry = replay_stage1(env, ic, s1_acts)
    d_lat = lateral_desire_direction(entry["e_signed_mm"])
    roll_sign = float(choose_roll_sign_for_support(env.sim))
    y0 = reference_pad_vector(env.sim)
    e_ref = np.asarray(entry["e_signed_mm"], float)
    cur = {"capture": entry_cap, "ref": ic["ref"]}

    rate_s1 = float(cfg.rate_establish_s1)
    rate_protected = float(cfg.rate_protected_s1)
    rate_horiz_cap = float(cfg.dx_horizontal_rate_cap)
    s1_roll_deg = float(cfg.dx_s1_roll_deg)

    report: dict = {
        "ic_id": "hard_910026",
        "entry": entry,
        "d_lat": d_lat.tolist(),
        "roll_sign": roll_sign,
        "rates": {
            "rate_s1": rate_s1,
            "rate_protected": rate_protected,
            "rate_horiz_cap": rate_horiz_cap,
            "s1_roll_deg": s1_roll_deg,
            "fn_hold": FN_HOLD,
        },
        "procedure_note": (
            "Intended: one-sided support → protected tilt to horizontal → slide → restore. "
            "run_tilt_phase aborts on NOT bilateral — potential logic conflict with one-sided S1."
        ),
    }

    # --- DX1 ---
    restore_capture(env, entry_cap)
    pre_dx1 = snap(env.sim)
    p1 = run_roll_phase(
        env, cur, sign=roll_sign, angle_deg=s1_roll_deg, phase="dxST_DX1_S1",
        rate_scale=rate_s1, fn_hold=FN_HOLD, require_down=True, y0_w=y0, e_ref=e_ref,
    )
    post_dx1 = snap(env.sim)
    bil1, sup1 = _physical_finger_support(env.sim)
    s1_state = make_s1_state(env.sim, roll_sign=roll_sign)
    report["DX1_S1"] = {
        "p1_ok": p1.get("ok"),
        "p1_reason": p1.get("reason"),
        "pre": pre_dx1,
        "post": post_dx1,
        "physical_bilateral": bil1,
        "support": {k: sup1.get(k) for k in ("fn_L", "fn_R", "fn_tot", "sides")},
        "s1_state_valid": s1_state.get("valid"),
        "s1_one_sided": s1_state.get("one_sided_mechanical"),
        "s1_support_side": s1_state.get("support_side"),
        "interpretation": (
            "DX1 only applies a modest DOWN roll for support *bias*. "
            f"After DX1, one_sided_mechanical={s1_state.get('one_sided_mechanical')}; "
            "full one-sided load is expected only under DX-2 gravity (code comment)."
        ),
    }
    cur = {"capture": p1["capture"], "ref": ic["ref"]}

    # --- DX2 attempt 1 ---
    restore_capture(env, cur["capture"])
    u_tgt = _horizontal_open_target(env.sim)
    pre_dx2 = snap(env.sim)
    p2 = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="dxST_DX2_horizontal",
        rate_scale=min(rate_protected, rate_horiz_cap),
        fn_hold=FN_HOLD, ang_tol_deg=12.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
        contact_aware=True, max_time_s=22.0, z_ref_boost=0.05, height_kp=5.0,
    )
    post_dx2_try1 = snap(env.sim)
    series1 = p2.get("series") or []
    loss1 = series_first_bilateral_loss(series1)
    # Find last good frame before loss
    last_bil = None
    for row in series1:
        if row.get("bilateral"):
            last_bil = row
        else:
            break

    report["DX2_try1"] = {
        "u_tgt": u_tgt.tolist(),
        "pre": pre_dx2,
        "p2_ok": p2.get("ok"),
        "p2_reason": p2.get("reason"),
        "alpha_final": p2.get("alpha_final"),
        "n_actions": len(p2.get("actions") or []),
        "n_series": len(series1),
        "first_bilateral_loss_frame": loss1,
        "last_bilateral_frame_before_loss": last_bil,
        "post": post_dx2_try1,
        "series_alpha_span": (
            [float(series1[0]["alpha_up_deg"]), float(series1[-1]["alpha_up_deg"])]
            if series1 else None
        ),
        "series_fn_span": (
            [float(series1[0]["fn_tot"]), float(series1[-1]["fn_tot"])] if series1 else None
        ),
    }

    # --- DX2 retry (same as production) ---
    restore_capture(env, cur["capture"])
    p2r = run_tilt_phase(
        env, cur, target_dir_w=u_tgt, phase="dxST_DX2_retry",
        rate_scale=min(0.05, rate_horiz_cap),
        fn_hold=FN_HOLD, ang_tol_deg=14.0, y0_w=y0, e_ref=e_ref, allow_roll=False,
        contact_aware=True, max_time_s=24.0, z_ref_boost=0.07, height_kp=5.5,
    )
    post_retry = snap(env.sim)
    series2 = p2r.get("series") or []
    loss2 = series_first_bilateral_loss(series2)
    bil2, sup2 = _physical_finger_support(env.sim)
    mode2 = classify_contact_mode(env.sim)
    alpha2 = float(gates(env.sim)["alpha_up_deg"])
    report["DX2_retry"] = {
        "p2_ok": p2r.get("ok"),
        "p2_reason": p2r.get("reason"),
        "alpha_final": p2r.get("alpha_final"),
        "n_actions": len(p2r.get("actions") or []),
        "first_bilateral_loss_frame": loss2,
        "post": post_retry,
        "physical_bilateral_end": bil2,
        "support_end": {k: sup2.get(k) for k in ("fn_L", "fn_R", "fn_tot", "sides")},
        "mode_end": mode2,
        "near_horizontal": 70.0 <= alpha2 <= 120.0,
        "production_fail_gate": (
            "support_lost_during_rotation" if not bil2
            else ("horizontal_unreachable" if not (p2r.get("ok") or (70 <= alpha2 <= 120)) else "would_pass")
        ),
    }

    # Full production call for confirmation
    restore_capture(env, entry_cap)
    full = correct_dx_support_transfer(env, {"capture": entry_cap, "ref": ic["ref"]}, cfg)
    report["production_correct_dx_support_transfer"] = {
        "ok": full.get("ok"),
        "phase": full.get("phase"),
        "reason": full.get("reason"),
        "failure_code": full.get("failure_code"),
        "phase_log": full.get("phase_log"),
        "timing_s": full.get("timing_s"),
        "n_actions": len(full.get("actions") or []),
    }

    # Causal summary
    loss = loss1 or loss2
    report["first_physical_event"] = {
        "event": "bilateral_contact_false_during_DX2_tilt",
        "evidence": loss,
        "mechanism": (
            "run_tilt_phase (contact_aware=True) aborts when sample_state.bilateral becomes False "
            "(or fn path later fails _physical_finger_support). "
            "DX1 does not establish verified one_sided_mechanical S1; it only rolls ~12° at DOWN. "
            "As the hand rotates toward horizontal, gravity unloading / slip drops one finger contact; "
            "the tilt controller treats that as fatal contact_loss even though the intended procedure "
            "is single-sided support during protected rotation."
        ),
        "logic_conflict": {
            "intended": "single-sided mechanical support during DX2 horizontal rotation",
            "implemented_abort": "run_tilt_phase requires continuous bilateral geom contact",
            "DX2_exit_check": "also requires bil2 (both pads fn>1) OR fails with support_lost_during_rotation",
            "code_refs": [
                "endpoint_only_axial_roll.run_tilt_phase L308-310: abort if not bilateral",
                "precision_dx_support_transfer.correct_dx_support_transfer L292-303: if not bil2 → support_lost_during_rotation",
                "precision_dx_support_transfer L241 note: full one-sided load expected under DX-2 gravity",
            ],
        },
    }

    # Fallback policy audit (static + facts from prior diagnosis)
    report["fallback_policy"] = {
        "invocation": (
            "unified_precision_oracle.run_precision: after support_transfer fails, "
            "if cfg.use_verified_lateral_fallback (default True), call correct_dx_verified"
        ),
        "code": "unified_precision_oracle.py L557-629",
        "why_permitted": (
            "Historical resilience: keep trying lateral reduction when ST path fails. "
            "allow_horizontal_dx legacy path is gated OFF when use_support_transfer_dx=True; "
            "verified_lateral is the remaining silent substitute."
        ),
        "bypasses_intended_procedure": True,
        "violates_contact_constraints": (
            "Uses orientation-plan tilt + gravity-assisted DualLoop slip with use_euclidean_stop=False; "
            "does not require one-sided S1; can raise θ and |dz| while reducing |dx|."
        ),
        "restores_fingertips_down": False,
        "stopping_vs_SUCCESS_NEW": (
            "Stops on |dx|≤lateral_axis_tol_mm (3mm), not Euclidean dwell; "
            "compatible with AXIS readiness but not sustained SUCCESS_NEW hold."
        ),
        "recommendation": (
            "Do not silently call correct_dx_verified after ST failure. "
            "Return PRECISION_PRIMITIVE_FAILED / explicit ST failure, or invoke a separately "
            "validated recovery that preserves the protected-support contract."
        ),
    }

    report["handoff_probe_gap"] = {
        "issue": (
            "After AXIS-ready snapshot, handoff_probe may report SUCCESS_NEW=False, "
            "but scheduler takes NO_REMAINING_CORRECTION because need_* flags are false."
        ),
        "code": "unified_precision_oracle.py correction loop: probe only early-exits on SUCCESS_NEW; "
                "failed NEW does not force residual correction",
        "hard_910026_fact": (
            "Pre-HANDOFF e_pos≈11.8 in band; final window mean_pos≈14.5 → working hold 0s. "
            "Instant e_pos≤12 ≠ sustained SUCCESS_NEW."
        ),
    }

    report["recommended_smallest_fix"] = {
        "priority": "Fix DX2 protected rotation under intended one-sided support — do not expand legacy fallback",
        "changes": [
            {
                "area": "Support establishment (DX1)",
                "change": (
                    "Before DX2, verify make_s1_state.valid / one_sided_mechanical (or equivalent force "
                    "asymmetry). If only bilateral friction bias exists, reinforce support roll/tilt "
                    "until mechanical S1 is true, or abort explicitly."
                ),
            },
            {
                "area": "Protected rotation (DX2)",
                "change": (
                    "Replace bilateral-abort in DX2 tilt with support-aware monitoring: allow "
                    "non-support finger unload if support_side Fn_z and contact remain; abort only on "
                    "support-side loss, latched drop, or unsafe v_rel/w_rel."
                ),
            },
            {
                "area": "Contact-loss / recovery",
                "change": (
                    "On support-side loss: restore last S1-valid capture, reduce rate, retry once; "
                    "if still failing, return structured failure — no silent verified_lateral."
                ),
            },
            {
                "area": "Posture restoration",
                "change": (
                    "After successful ST dx (or failed ST before exit), restore fingertips_down when "
                    "dz not needed; never leave mid-tilt from legacy path as default."
                ),
            },
            {
                "area": "Handoff readiness",
                "change": (
                    "If handoff_probe SUCCESS_NEW=False, do not NO_REMAINING_CORRECTION; "
                    "require margin (e.g. e_pos ≤ WORKING_POS−δ) and/or reopen dominant residual / DOWN restore."
                ),
            },
            {
                "area": "Fallback policy",
                "change": (
                    "Disable automatic use_verified_lateral_fallback after ST failure (default False for "
                    "production Expert), or gate it behind an explicit alternate procedure flag."
                ),
            },
        ],
    }

    # Persist compact series around loss
    if loss1:
        idx = next(i for i, r in enumerate(series1) if r is loss1 or r.get("t") == loss1.get("t"))
        report["DX2_try1"]["series_window"] = series1[max(0, idx - 3) : idx + 4]
    if loss2:
        idx = next(i for i, r in enumerate(series2) if r is loss2 or r.get("t") == loss2.get("t"))
        report["DX2_retry"]["series_window"] = series2[max(0, idx - 3) : idx + 4]

    path = OUT / "DX2_SUPPORT_LOSS_910026.json"
    path.write_text(json.dumps(base.jsafe(report), indent=2), encoding="utf-8")
    print(json.dumps(base.jsafe({
        "wrote": str(path),
        "DX1_one_sided": report["DX1_S1"]["s1_one_sided"],
        "DX2_try1_reason": report["DX2_try1"]["p2_reason"],
        "DX2_loss_t": (loss1 or {}).get("t"),
        "DX2_loss_alpha": (loss1 or {}).get("alpha_up_deg"),
        "production_reason": report["production_correct_dx_support_transfer"]["reason"],
    }), indent=2), flush=True)

    # Markdown conclusion
    md = OUT / "SUPPORT_TRANSFER_PROCEDURE_REPORT.md"
    md.write_text(_markdown(report), encoding="utf-8")
    print("Wrote", md, flush=True)


def _markdown(r: dict) -> str:
    dx1 = r["DX1_S1"]
    t1 = r["DX2_try1"]
    loss = t1.get("first_bilateral_loss_frame") or {}
    prod = r["production_correct_dx_support_transfer"]
    return f"""# Enforce Intended Precision Support-Transfer — Diagnosis (`hard_910026`)

## Required conclusion

**Why does `DX2_horizontal` lose supporting contact?**

After DX1, the grasp is still essentially a **bilateral DOWN** grasp with only a modest ~{r['rates']['s1_roll_deg']}° support-*bias* roll — **not** a verified `one_sided_mechanical` S1 (`one_sided={dx1['s1_one_sided']}`).

DX2 then calls `run_tilt_phase(..., contact_aware=True)` toward horizontal. That helper **aborts as soon as geometric bilateral contact is false** (`endpoint_only_axial_roll.py`). During the tilt, one finger unloads / separates under gravity and commanded rotation; the first logged loss is at **t≈{loss.get('t')} s**, **α≈{loss.get('alpha_up_deg')}°**, with `fn_tot≈{loss.get('fn_tot')}`, `v_rel≈{loss.get('v_rel_norm')}`. Production then labels this `support_lost_during_rotation` because end-state `_physical_finger_support` is non-bilateral (`phase={prod.get('phase')}`, `reason={prod.get('reason')}`).

There is a **procedure vs implementation conflict**: the flowchart intends **single-sided support during protected rotation**, but the tilt monitor and DX2 exit gate both require **continuous bilateral** contact. Unloading the non-support finger — which is expected when establishing S1 under gravity — is treated as fatal failure.

**What must change so the Oracle executes the intended protected sequence instead of legacy rotation?**

1. **Establish and verify mechanical S1 before DX2** (force-backed `one_sided_mechanical`), not only a small DOWN roll.
2. **Monitor support-side contact during DX2**, allowing non-support unload; abort only on support-side loss / drop / unsafe slip.
3. **On ST failure: do not silently call `correct_dx_verified`** — report failure or use an explicitly validated alternate.
4. **Restore fingertips-down** when dz is not next; do not leave mid-tilt.
5. **Handoff readiness**: failed `SUCCESS_NEW` probe must reopen residual/posture correction; instant `e_pos≤12` is not sustained NEW.

---

## Task 1 — DX2 instrumentation summary

| Phase | Result |
|---|---|
| Entry | e_pos={r['entry']['e_pos_mm']:.2f}, dx={r['entry']['e_signed_mm'][0]:.2f}, bilateral force OK |
| DX1 | ok={dx1['p1_ok']}, one_sided_mechanical={dx1['s1_one_sided']}, support_side={dx1['s1_support_side']} |
| DX2 try1 | reason=`{t1['p2_reason']}`, first bilateral loss t={loss.get('t')}, α={loss.get('alpha_up_deg')} |
| Production ST | `{prod.get('reason')}` @ `{prod.get('phase')}` |

Coordinate frames: hand-frame signed errors mm; α = angle of `u_open` to world UP; contact forces from MuJoCo contact solver via `contact_forces` / `support_force_decomposition`.

---

## Task 2 — Fallback policy

{json_block(r['fallback_policy'])}

---

## Task 3 — Handoff probe gap

{json_block(r['handoff_probe_gap'])}

---

## Task 4 — Smallest correct fix (recommended; not applied)

{json_block(r['recommended_smallest_fix'])}

---

Artifacts: `DX2_SUPPORT_LOSS_910026.json`
"""


def json_block(obj) -> str:
    return "```json\n" + json.dumps(obj, indent=2, default=str) + "\n```"


if __name__ == "__main__":
    main()
