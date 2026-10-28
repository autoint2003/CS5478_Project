#!/usr/bin/env python
"""P5 Phase I/II: A/B/C reconfiguration pilot + Precision continuation."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.abc_grasp_reconfiguration.strategies import (  # noqa: E402
    err_pack,
    reconfig_improved,
    run_strategy_A,
    run_strategy_B,
    run_strategy_C,
    verify_12s,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402

OUT = ROOT / "results/diagnostics/abc_grasp_reconfiguration"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
PRIORITY = ["hard_910132", "hard_910024", "hard_910084"]
REFERENCE = ["hard_910305"]
EXTRA = ["hard_910263", "hard_910277", "hard_910026"]


def jsafe(o: Any) -> Any:
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


def load_blob(hid: str) -> dict:
    blob = pickle.loads((EXIT / f"{hid}.pkl").read_bytes())
    meta = json.loads((EXIT / f"{hid}__meta.json").read_text(encoding="utf-8"))
    return {**blob, "sample_id": meta.get("raw_id"), "hard_id": hid, "meta": meta}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def run_R0(env, blob, pcfg) -> dict:
    restore_capture(env, blob["capture"])
    e0 = err_pack(env.sim)
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
        "B",
        pcfg,
    )
    hand = res.handoff or {}
    restore_capture(env, res.capture or blob["capture"]) if res.capture is not None else None
    e1 = err_pack(env.sim) if res.capture is not None else e0
    return {
        "strategy": "R0",
        "reconfig": None,
        "RECONFIG_STABLE": None,
        "RECONFIG_IMPROVED": None,
        "FINAL_SUCCESS": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S_precision": bool(hand.get("SUCCESS_12S")),
        "precision_status": res.status,
        "failures": list(res.failures or []),
        "errors0": e0,
        "errors_after_reconfig": None,
        "errors_final": {
            "dx": float((res.errors_exit or {}).get("dx_mm") or e1["dx"]),
            "dy": float((res.errors_exit or {}).get("dy_mm") or e1["dy"]),
            "dz": float((res.errors_exit or {}).get("dz_mm") or e1["dz"]),
            "dtheta": float((res.errors_exit or {}).get("dtheta_deg") or e1["dtheta"]),
            "e_pos": float((res.errors_exit or {}).get("e_pos_mm") or e1["e_pos"]),
        },
        "actions_reconfig": [],
        "actions_precision": list(res.actions or []),
        "n_actions_reconfig": 0,
        "n_actions_precision": len(res.actions or []),
    }


def run_Rx(env, blob, book, pcfg, letter: str, *, t_dx: float, t_dz: float) -> dict:
    fn = {"A": run_strategy_A, "B": run_strategy_B, "C": run_strategy_C}[letter]
    restore_capture(env, blob["capture"])
    try:
        rec = fn(env, blob, book)
    except Exception as exc:
        return {
            "strategy": f"R{ {'A':1,'B':2,'C':3}[letter] }",
            "reconfig_letter": letter,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "RECONFIG_STABLE": False,
            "RECONFIG_IMPROVED": False,
            "FINAL_SUCCESS": False,
            "failures": [f"reconfig_exception:{exc}"],
            "actions_reconfig": [],
            "actions_precision": [],
        }

    restore_capture(env, rec["capture"])
    e0, e1 = rec["errors0"], rec["errors1"]
    imp = reconfig_improved(e0, e1, t_dx=t_dx, t_dz=t_dz)
    # Stability only if bilateral after maneuver
    from envs.outcomes.ground_drop import latched as gl
    from envs.outcomes.table_drop import latched as tl

    stable_info = {"SUCCESS_12S": False}
    if e1.get("bilateral") and not gl(env.sim) and not tl(env.sim):
        stable_info = verify_12s(env, blob["ref"])

    reconfig_stable = bool(stable_info.get("SUCCESS_12S"))
    row = {
        "strategy": f"R{ {'A':1,'B':2,'C':3}[letter] }",
        "reconfig_letter": letter,
        "ok_maneuver": rec.get("ok_maneuver"),
        "maneuver_reason": rec.get("reason"),
        "released": rec.get("released"),
        "reacquired": rec.get("reacquired"),
        "genuine_slide": rec.get("genuine_slide"),
        "genuine_recatch": rec.get("genuine_recatch") or rec.get("genuine_controlled_release"),
        "t_release": rec.get("t_release"),
        "t_reacquire": rec.get("t_reacquire"),
        "contact_free_s": rec.get("contact_free_s"),
        "disp_mm": rec.get("disp_mm"),
        "mechanism": rec.get("mechanism") or rec.get("reason"),
        "errors0": e0,
        "errors_after_reconfig": e1,
        "RECONFIG_STABLE": reconfig_stable,
        # Official IMPROVED requires stability; also record band entry for diagnostics.
        "RECONFIG_IMPROVED": bool(imp["RECONFIG_IMPROVED"]) and reconfig_stable,
        "band_entry_without_stability": bool(imp["RECONFIG_IMPROVED"]) and not reconfig_stable,
        **{k: imp[k] for k in imp if k != "RECONFIG_IMPROVED"},
        "reconfig_12s": stable_info,
        "nominal_reason": stable_info.get("nominal_reason"),
        "actions_reconfig": list(rec.get("actions") or []),
        "n_actions_reconfig": int(rec.get("n_actions") or 0),
        "timeline": rec.get("timeline"),
        "c_meta": {
            k: rec.get(k)
            for k in ("plan_t_c", "plan_s_axial", "release_state", "free_motion_preview", "params")
            if k in rec
        },
        "capture_after_reconfig": rec.get("capture"),
        "FINAL_SUCCESS": False,
        "failures": [],
        "actions_precision": [],
        "n_actions_precision": 0,
    }

    if not reconfig_stable:
        row["failures"].append("reconfig_not_SUCCESS_12S")
        row["precision_status"] = "SKIPPED_UNSTABLE_RECONFIG"
        return row

    # Phase II — Precision from reconfigured grasp
    restore_capture(env, rec["capture"])
    res = run_precision(
        env,
        {
            "capture": rec["capture"],
            "ref": blob["ref"],
            "sample_id": blob.get("sample_id"),
        },
        "B",
        pcfg,
    )
    hand = res.handoff or {}
    row["FINAL_SUCCESS"] = bool(hand.get("SUCCESS_NEW"))
    row["SUCCESS_12S_precision"] = bool(hand.get("SUCCESS_12S"))
    row["precision_status"] = res.status
    row["failures"] += list(res.failures or [])
    row["actions_precision"] = list(res.actions or [])
    row["n_actions_precision"] = len(res.actions or [])
    row["errors_final"] = {
        "dx": float((res.errors_exit or {}).get("dx_mm") or 0),
        "dy": float((res.errors_exit or {}).get("dy_mm") or 0),
        "dz": float((res.errors_exit or {}).get("dz_mm") or 0),
        "dtheta": float((res.errors_exit or {}).get("dtheta_deg") or 0),
        "e_pos": float((res.errors_exit or {}).get("e_pos_mm") or 0),
    }
    return row


def flat_row(hid: str, packed: dict, t_dx: float, t_dz: float) -> dict:
    e0 = packed.get("errors0") or {}
    e1 = packed.get("errors_after_reconfig") or {}
    ef = packed.get("errors_final") or {}
    return {
        "hard_id": hid,
        "strategy": packed.get("strategy"),
        "init_dx": e0.get("dx"),
        "init_dy": e0.get("dy"),
        "init_dz": e0.get("dz"),
        "init_dtheta": e0.get("dtheta"),
        "init_e_pos": e0.get("e_pos"),
        "reconfig_dx": e1.get("dx"),
        "reconfig_dy": e1.get("dy"),
        "reconfig_dz": e1.get("dz"),
        "reconfig_dtheta": e1.get("dtheta"),
        "reconfig_e_pos": e1.get("e_pos"),
        "delta_abs_dx": packed.get("delta_abs_dx"),
        "delta_abs_dz": packed.get("delta_abs_dz"),
        "delta_e_pos": packed.get("delta_e_pos"),
        "released": packed.get("released"),
        "reacquired": packed.get("reacquired"),
        "genuine_maneuver": packed.get("genuine_recatch") or packed.get("genuine_slide") or packed.get("ok_maneuver"),
        "maneuver_reason": packed.get("maneuver_reason") or packed.get("mechanism"),
        "t_release": packed.get("t_release"),
        "t_reacquire": packed.get("t_reacquire"),
        "contact_free_s": packed.get("contact_free_s"),
        "RECONFIG_STABLE": packed.get("RECONFIG_STABLE"),
        "RECONFIG_IMPROVED": packed.get("RECONFIG_IMPROVED"),
        "band_entry_without_stability": packed.get("band_entry_without_stability"),
        "dx_in_band": packed.get("dx_in_band"),
        "dz_in_band": packed.get("dz_in_band"),
        "nominal_reason": packed.get("nominal_reason"),
        "T_dx": t_dx,
        "T_dz": t_dz,
        "FINAL_SUCCESS": packed.get("FINAL_SUCCESS"),
        "SUCCESS_12S_precision": packed.get("SUCCESS_12S_precision"),
        "precision_status": packed.get("precision_status"),
        "failures": "|".join(packed.get("failures") or []),
        "n_actions_reconfig": packed.get("n_actions_reconfig"),
        "n_actions_precision": packed.get("n_actions_precision"),
        "error": packed.get("error"),
    }


def write_static_docs(t_dx: float, t_dz: float) -> None:
    (OUT / "ABC_REUSE_AUDIT.md").write_text(
        f"""# ABC_REUSE_AUDIT

Production Plan A/B/C are **Stage-1 creators** from post-impact / offset ICs, not in-hand
reconfiguration APIs for a verified B2 `SUCCESS_12S` grasp. This pilot uses **experimental
wrappers** that reuse legal mechanisms without modifying production Oracle code.

## Plan A — Inward Sliding

| Item | Detail |
|---|---|
| Production entry | Recovery IC with bilateral grasp; useful slide if `|ex|≥3 mm` else immediate done |
| Actions | Wrist wx/wy + grip relax (−0.48) / retighten; K48-quantized |
| Contact | Requires bilateral; aborts on drop |
| Effect | Gravity-assisted slide along finger axis — primarily lateral offset |
| Termination | `|ex|` small, slip budget, or contact loss |
| Stable-B2 reuse | **Not as-is** (centered grasp → done). Experimental R1 still attempts reorient+relax |

## Plan B — Recatch

| Item | Detail |
|---|---|
| Production entry | Sliding / lost-contact snapshots; **requires full detach** for active recatch credit |
| Actions | OPEN → chase → CLOSE → brake → lift → settle |
| Contact | Full detach then bilateral reacquire in Recovery |
| Effect | New grasp geometry via reacquisition |
| Stable-B2 reuse | Must **deliberately open** (experimental R2). Mere regrip without detach ≠ recatch |

## Plan C — Dynamic Catch

| Item | Detail |
|---|---|
| Production entry | Post ball-impact airborne root on GROUND_SCENE |
| Actions | OPEN → intercept → velocity match → close → brake (+ optional pose_correct) |
| Contact | Designed to start from lost/airborne; OPEN always releases |
| Effect | Large recatch; mechanism label `FULL_DETACH_ACTIVE_RECATCH` if contact-free gap measured |
| Stable-B2 reuse | **Experimental R3**: controlled release from held grasp, plan intercept offline, execute legally. No ball impact. No teleport |

## Precision entry (continuation)

`check_stage1_eligible` → `assess_need` → dθ → dx → dz. Inward DZ requires `|ex|≤dz_safe_ex_mm={t_dx}`.

## Thresholds for RECONFIG_IMPROVED

- `T_dx = {t_dx} mm` (`dz_safe_ex_mm`) — enables inward DZ
- `T_dz = {t_dz} mm` — axial comfort band (WORKING_POS scale; not SUCCESS Euclidean alone)
""",
        encoding="utf-8",
    )


def write_final(rows: list, cases: list, t_dx: float, t_dz: float) -> None:
    def truthy(x):
        return x in (True, "True", "true", 1, "1")

    by = {}
    for r in rows:
        by.setdefault(r["hard_id"], {})[r["strategy"]] = r

    lines = [
        "# FINAL_REPORT — A/B/C Grasp Reconfiguration Pilot",
        "",
        "Feasibility only. Production Oracle unchanged.",
        "",
        f"Thresholds: T_dx={t_dx} mm, T_dz={t_dz} mm.",
        "",
        "## Outcome counts",
        "",
        "| Strategy | RECONFIG_STABLE | RECONFIG_IMPROVED | FINAL_SUCCESS (NEW) |",
        "|---|---:|---:|---:|",
    ]
    for s in ("R0", "R1", "R2", "R3"):
        rs = [r for r in rows if r.get("strategy") == s]
        if s == "R0":
            lines.append(
                f"| {s} | n/a | n/a | {sum(1 for r in rs if truthy(r.get('FINAL_SUCCESS')))}/{len(rs)} |"
            )
        else:
            lines.append(
                f"| {s} | {sum(1 for r in rs if truthy(r.get('RECONFIG_STABLE')))}/{len(rs)} | "
                f"{sum(1 for r in rs if truthy(r.get('RECONFIG_IMPROVED')))}/{len(rs)} | "
                f"{sum(1 for r in rs if truthy(r.get('FINAL_SUCCESS')))}/{len(rs)} |"
            )

    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. **Can A improve a stable poorly positioned grasp?** See R1 rows: genuine_slide + deltas.",
            "2. **Can B recatch into a better configuration?** Requires `released∧reacquired`; see R2.",
            "3. **Can C deliberately release and recatch?** Requires `genuine_controlled_release`; see R3 + CONTROLLED_RELEASE_C_ANALYSIS.",
            "4. **Most effective when DX and DZ both large?** Compare RECONFIG_IMPROVED / FINAL_SUCCESS among priority ICs.",
            "5. **Reduce one error without destroying stability?** RECONFIG_IMPROVED ∧ RECONFIG_STABLE.",
            "6. **Improve subsequent Precision?** FINAL_SUCCESS vs R0 on same IC.",
            "7. **Algorithmic vs physical limits?** Entry refusals / controller fails vs contact loss / ground drop.",
            "8. **Optional reconfiguration stage justified?** Only if FINAL_SUCCESS exceeds R0 with stable reconfig — not DX/DZ reduction alone.",
            "",
            "## Per-case",
            "",
            "| hard_id | S | STABLE | IMPROVED | NEW | reason |",
            "|---|---|---|---|---|---|",
        ]
    )
    for r in rows:
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('strategy')} | {r.get('RECONFIG_STABLE')} | "
            f"{r.get('RECONFIG_IMPROVED')} | {r.get('FINAL_SUCCESS')} | "
            f"{(r.get('maneuver_reason') or r.get('precision_status') or r.get('failures') or '—')[:60]} |"
        )

    # Recommendation
    r0_new = sum(1 for r in rows if r.get("strategy") == "R0" and truthy(r.get("FINAL_SUCCESS")))
    best = max(
        (
            (
                s,
                sum(1 for r in rows if r.get("strategy") == s and truthy(r.get("FINAL_SUCCESS"))),
                sum(1 for r in rows if r.get("strategy") == s and truthy(r.get("RECONFIG_STABLE"))),
            )
            for s in ("R1", "R2", "R3")
        ),
        key=lambda x: (x[1], x[2]),
    )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if best[1] > r0_new:
        lines.append(
            f"**CONDITIONAL interest** in {best[0]} ({best[1]} FINAL_SUCCESS vs R0 {r0_new}), "
            "pending more ICs. Do **not** integrate into production from this pilot alone."
        )
    else:
        lines.append(
            "**DO NOT adopt** an optional A/B/C reconfiguration stage into production from this pilot. "
            "Preserve failures; Precision-direct (R0) remains the baseline."
        )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_c_analysis(rows: list) -> None:
    lines = [
        "# CONTROLLED_RELEASE_C_ANALYSIS",
        "",
        "Experimental R3 only. Production Plan C ball-impact timing was **not** reused.",
        "",
        "| hard_id | released | reacquired | free_s | t_rel | t_re | STABLE | IMPROVED | NEW | mechanism |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        if r.get("strategy") != "R3":
            continue
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('released')} | {r.get('reacquired')} | "
            f"{r.get('contact_free_s')} | {r.get('t_release')} | {r.get('t_reacquire')} | "
            f"{r.get('RECONFIG_STABLE')} | {r.get('RECONFIG_IMPROVED')} | {r.get('FINAL_SUCCESS')} | "
            f"{r.get('maneuver_reason')} |"
        )
    lines.extend(
        [
            "",
            "A row counts as a **genuine controlled-release recatch** only if `released` and `reacquired` "
            "and contact-free duration ≥ one policy step (0.02 s).",
            "",
        ]
    )
    (OUT / "CONTROLLED_RELEASE_C_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def write_contact_analysis(rows: list) -> None:
    lines = [
        "# CONTACT_AND_STABILITY_ANALYSIS",
        "",
        "| hard_id | S | bilateral_init | released | reacq | STABLE_12S | IMPROVED | fail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e0 = {}
        # flat row may not have nested — use fields
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('strategy')} | — | {r.get('released')} | {r.get('reacquired')} | "
            f"{r.get('RECONFIG_STABLE')} | {r.get('RECONFIG_IMPROVED')} | {(r.get('failures') or '')[:50]} |"
        )
    (OUT / "CONTACT_AND_STABILITY_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--priority-only", action="store_true", help="Only hard_910132/024/084 (+305 ref)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    # IC analysis + thresholds
    from experiments.precision.abc_grasp_reconfiguration.ic_analysis import main as ic_main

    ic_main()
    ic = json.loads((OUT / "IC_ANALYSIS.json").read_text(encoding="utf-8"))
    t_dx = float(ic["T_dx_mm"])
    t_dz = float(ic["T_dz_mm"])
    # Prefer tighter T_dz for "improved" = WORKING-scale 12 mm single axis comfort
    from evaluation.success_criteria import WORKING_POS_MM

    t_dz = float(WORKING_POS_MM)  # 12 mm — not SUCCESS Euclidean alone, but Precision comfort
    write_static_docs(t_dx, t_dz)

    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]
    elif args.priority_only:
        ids = PRIORITY + REFERENCE
    else:
        ids = PRIORITY + REFERENCE  # Phase I pilot scope per user

    env = make_env()
    book = load_codebook()
    pcfg = PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)

    cases = []
    reconfig_rows = []
    precision_rows = []

    for hid in ids:
        print(f"\n=== {hid} ===", flush=True)
        blob = load_blob(hid)
        case = {"hard_id": hid, "variants": {}}
        for strat in ("R0", "R1", "R2", "R3"):
            print(f"  [{hid}] {strat}...", flush=True)
            try:
                if strat == "R0":
                    packed = run_R0(env, blob, pcfg)
                else:
                    letter = {"R1": "A", "R2": "B", "R3": "C"}[strat]
                    packed = run_Rx(env, blob, book, pcfg, letter, t_dx=t_dx, t_dz=t_dz)
                    if args.phase1_only:
                        packed["actions_precision"] = []
                        packed["FINAL_SUCCESS"] = None
                        packed["precision_status"] = "PHASE1_ONLY"

                acts_r = packed.pop("actions_reconfig", []) or []
                acts_p = packed.pop("actions_precision", []) or []
                cap = packed.pop("capture_after_reconfig", None)
                tl = packed.pop("timeline", None)
                np.savez_compressed(
                    OUT / "raw" / f"{hid}__{strat}__actions.npz",
                    reconfig=np.asarray(acts_r, np.float32) if len(acts_r) else np.zeros((0, 7), np.float32),
                    precision=np.asarray(acts_p, np.float32) if len(acts_p) else np.zeros((0, 7), np.float32),
                )
                if cap is not None:
                    (OUT / "raw" / f"{hid}__{strat}__capture.pkl").write_bytes(pickle.dumps(cap))
                if tl is not None:
                    (OUT / "raw" / f"{hid}__{strat}__timeline.json").write_text(
                        json.dumps(jsafe(tl), indent=2), encoding="utf-8"
                    )

                case["variants"][strat] = jsafe(packed)
                flat = flat_row(hid, packed, t_dx, t_dz)
                reconfig_rows.append(flat)
                precision_rows.append(
                    {
                        "hard_id": hid,
                        "strategy": strat,
                        "RECONFIG_STABLE": packed.get("RECONFIG_STABLE"),
                        "RECONFIG_IMPROVED": packed.get("RECONFIG_IMPROVED"),
                        "FINAL_SUCCESS": packed.get("FINAL_SUCCESS"),
                        "SUCCESS_12S_precision": packed.get("SUCCESS_12S_precision"),
                        "precision_status": packed.get("precision_status"),
                        "failures": "|".join(packed.get("failures") or []),
                        "final_dx": (packed.get("errors_final") or {}).get("dx"),
                        "final_dz": (packed.get("errors_final") or {}).get("dz"),
                        "final_e_pos": (packed.get("errors_final") or {}).get("e_pos"),
                        "final_dtheta": (packed.get("errors_final") or {}).get("dtheta"),
                    }
                )
                print(
                    f"    STABLE={packed.get('RECONFIG_STABLE')} IMPROVED={packed.get('RECONFIG_IMPROVED')} "
                    f"NEW={packed.get('FINAL_SUCCESS')} reason={packed.get('maneuver_reason') or packed.get('precision_status')}",
                    flush=True,
                )
            except Exception as exc:
                print(f"    ERROR {exc}", flush=True)
                case["variants"][strat] = {"error": str(exc), "traceback": traceback.format_exc()}
                reconfig_rows.append({"hard_id": hid, "strategy": strat, "error": str(exc), "FINAL_SUCCESS": False})
                precision_rows.append({"hard_id": hid, "strategy": strat, "error": str(exc), "FINAL_SUCCESS": False})
        cases.append(case)

    (OUT / "CASES.json").write_text(json.dumps(jsafe(cases), indent=2), encoding="utf-8")
    write_csv(OUT / "RECONFIGURATION_RESULTS.csv", reconfig_rows)
    write_csv(OUT / "PRECISION_CONTINUATION_RESULTS.csv", precision_rows)
    write_c_analysis(reconfig_rows)
    write_contact_analysis(reconfig_rows)
    write_final(reconfig_rows, cases, t_dx, t_dz)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ids": ids,
        "T_dx_mm": t_dx,
        "T_dz_mm": t_dz,
        "r0_new": sum(1 for r in reconfig_rows if r.get("strategy") == "R0" and r.get("FINAL_SUCCESS") is True),
        "r1_stable": sum(1 for r in reconfig_rows if r.get("strategy") == "R1" and r.get("RECONFIG_STABLE") is True),
        "r2_stable": sum(1 for r in reconfig_rows if r.get("strategy") == "R2" and r.get("RECONFIG_STABLE") is True),
        "r3_stable": sum(1 for r in reconfig_rows if r.get("strategy") == "R3" and r.get("RECONFIG_STABLE") is True),
        "r1_new": sum(1 for r in reconfig_rows if r.get("strategy") == "R1" and r.get("FINAL_SUCCESS") is True),
        "r2_new": sum(1 for r in reconfig_rows if r.get("strategy") == "R2" and r.get("FINAL_SUCCESS") is True),
        "r3_new": sum(1 for r in reconfig_rows if r.get("strategy") == "R3" and r.get("FINAL_SUCCESS") is True),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
