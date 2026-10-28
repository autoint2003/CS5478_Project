#!/usr/bin/env python
"""Complete C recatch+lift pilot on hard_910263 / hard_910277."""

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
from experiments.precision.c_reconfiguration_complete_lift.complete_c import (  # noqa: E402
    DEFAULT_TARGET_Z,
    err_pack,
    run_complete_c,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402

OUT = ROOT / "results/diagnostics/c_reconfiguration_complete_lift"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
PRIMARY = ["hard_910263", "hard_910277"]
SECONDARY = ["hard_910132", "hard_910024", "hard_910084", "hard_910305"]

# Small set of physically meaningful recatch offsets
VARIANTS = [
    {"label": "mid_body_s0", "s_axial": 0.0, "t_c": 0.14, "open_steps": 5, "rationale": "Mid-body sidewall catch; nominal timing"},
    {"label": "mid_body_early", "s_axial": 0.0, "t_c": 0.10, "open_steps": 4, "rationale": "Earlier intercept to reduce free-fall height loss"},
    {"label": "axial_neg", "s_axial": -0.015, "t_c": 0.14, "open_steps": 5, "rationale": "Negative axial offset to bias COM toward hand center (reduce |DX|)"},
    {"label": "axial_pos", "s_axial": 0.015, "t_c": 0.14, "open_steps": 5, "rationale": "Positive axial offset (opposite bias)"},
]


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


def write_audit() -> None:
    (OUT / "P5_C_MISSING_LIFT_AUDIT.md").write_text(
        f"""# P5_C_MISSING_LIFT_AUDIT

## What the prior experimental C did

In `experiments/precision/abc_grasp_reconfiguration/strategies.py` (`run_strategy_C`):

1. Controlled OPEN → intercept → CLOSE → brake — **yes**, with measured contact-free ≥0.12 s and bilateral reacquire on 132/024/084 (`FULL_DETACH_ACTIVE_RECATCH`).
2. Bilateral contact after reacquire — **yes** (Fn≈18).
3. **Lift stage — NO.** After reacquire the wrapper only ran a short secure hold:

```text
# Short secure hold (avoid aggressive world-lift — it dropped several recatches)
for _ in range(20): hold with TAU_SECURE
```

An earlier aggressive `vz=-0.35` “lift” attempt had destroyed recatches, so lift was **intentionally removed**, not accidentally skipped by a gate.

4. HANDOFF then ran via `nominal_verify` / `evaluate_handoff_with_axis` from the **post-recatch height** without restoring production elevation.
5. Failure mode: **`DROP_AFTER_HANDOFF`** — consistent with starting nominal lift/hold from an incomplete Stage-1 exit (missing production elevation + `fsm.grasp_z` update), not proof that completed C recovery is physically impossible.

## Production Plan C sequence

`run_stage1_catch_and_lift` (`plan_c/expert_search.py`):

1. Replay catch `capture_actions` until bilateral.
2. **`fast_world_lift_sequence`** (`catch_stabilization.py`) to **`target_z={DEFAULT_TARGET_Z}`** (world +z via legal `_action_from`).
3. Stop at elevated bilateral Stage-1 exit (production C2 does **not** call `pose_correct_and_handoff` here).

Historical search may call `lift_and_validate` → pose_correct → handoff; Stage-1 eligibility still requires full **SUCCESS_12S**.

## Fix in this package

Experimental sequence (isolated; production unchanged):

`Controlled Release → Large Recatch → Bilateral Stabilization → fast_world_lift_sequence → settle → 12S nominal_verify`

- Lift **reuses** `fast_world_lift_sequence` (no new lift controller).
- After lift, set `fsm.grasp_z` to the current hand height (as `raise_hand_airborne` does).
- Do **not** count bilateral reacquire alone as recovery success.
- Target height: `max(0.55, exit_obj_z)` when the Stage-1 exit is already elevated (B2 lifts).

## Classification

Prior P5 `DROP_AFTER_HANDOFF` on C must be labeled **`INCOMPLETE_C_MISSING_LIFT`**, not “completed C failed.”
""",
        encoding="utf-8",
    )


def write_primary_analysis(rows: list) -> None:
    lines = [
        "# PRIMARY_IC_ANALYSIS",
        "",
        "Primary budget: `hard_910263`, `hard_910277`. Secondary controls listed for context only.",
        "",
        "| hard_id | DX | DZ | dθ | e_pos | obj_z | hand_z | both>|6| | role |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        role = "PRIMARY" if r["hard_id"] in PRIMARY else "secondary"
        lines.append(
            f"| `{r['hard_id']}` | {r['dx_mm']:.2f} | {r['dz_mm']:.2f} | {r['dtheta_deg']:.2f} | "
            f"{r['e_pos_mm']:.2f} | {r['obj_z']:.3f} | {r['hand_z']:.3f} | {r['both_large']} | {role} |"
        )
    lines.extend(
        [
            "",
            "## Why 263 / 277",
            "",
            "These exits combine large |DX| and |DZ| (and large dθ on both), so direct Precision is "
            "blocked by ordering/entry gates (dθ failures / |ex|>dz_safe). Reconfiguration is the "
            "motivating use case. Prior P5 spent budget on 132/024/084/305 which are severe but not "
            "the user-identified primary pair.",
            "",
        ]
    )
    (OUT / "PRIMARY_IC_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")


def flat_result(hid: str, packed: dict) -> dict:
    e0 = packed.get("errors0") or {}
    ef = packed.get("errors_final") or packed.get("errors_after_lift") or {}
    return {
        "hard_id": hid,
        "variant": packed.get("variant"),
        "RELEASE_EXECUTED": packed.get("RELEASE_EXECUTED"),
        "REACQUIRE_EXECUTED": packed.get("REACQUIRE_EXECUTED"),
        "CONTACT_STABILIZED": packed.get("CONTACT_STABILIZED"),
        "LIFT_COMPLETED": packed.get("LIFT_COMPLETED"),
        "SUCCESS_12S": packed.get("SUCCESS_12S"),
        "RECONFIG_IMPROVED": packed.get("RECONFIG_IMPROVED"),
        "SUCCESS_NEW": packed.get("SUCCESS_NEW"),
        "stage_fail": packed.get("stage_fail"),
        "nominal_reason": packed.get("nominal_reason"),
        "init_dx": e0.get("dx"),
        "init_dz": e0.get("dz"),
        "init_dtheta": e0.get("dtheta"),
        "init_obj_z": e0.get("obj_z"),
        "final_dx": ef.get("dx"),
        "final_dz": ef.get("dz"),
        "final_dtheta": ef.get("dtheta"),
        "final_obj_z": ef.get("obj_z"),
        "delta_abs_dx": packed.get("delta_abs_dx"),
        "delta_abs_dz": packed.get("delta_abs_dz"),
        "dx_in_band": packed.get("dx_in_band"),
        "dz_in_band": packed.get("dz_in_band"),
        "obj_z_pre_lift": packed.get("obj_z_pre_lift"),
        "target_z": packed.get("target_z"),
        "xy_drift_mm": (packed.get("lift") or {}).get("xy_drift_mm"),
        "t_release": packed.get("t_release"),
        "t_reacquire": packed.get("t_reacquire"),
        "contact_free_s": packed.get("contact_free_s"),
        "n_actions": len(packed.get("actions") or []),
    }


def write_final(rows: list, prec_rows: list) -> None:
    lines = [
        "# FINAL_REPORT — Complete C Recatch + Lift",
        "",
        "Production Oracle unchanged. Prior `abc_grasp_reconfiguration/` artifacts preserved.",
        "",
        "## Missing-lift correction",
        "",
        "Prior P5 C stopped at bilateral reacquire (short hold only). This package inserts "
        "production `fast_world_lift_sequence` before 12S. See `P5_C_MISSING_LIFT_AUDIT.md`.",
        "",
        "## Primary IC results",
        "",
        "| hard_id | variant | REL | REACQ | STAB | LIFT | 12S | IMPROVED | NEW | fail |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("hard_id") not in PRIMARY:
            continue
        lines.append(
            f"| `{r.get('hard_id')}` | {r.get('variant')} | {r.get('RELEASE_EXECUTED')} | "
            f"{r.get('REACQUIRE_EXECUTED')} | {r.get('CONTACT_STABILIZED')} | {r.get('LIFT_COMPLETED')} | "
            f"{r.get('SUCCESS_12S')} | {r.get('RECONFIG_IMPROVED')} | {r.get('SUCCESS_NEW')} | "
            f"{r.get('stage_fail') or '—'} |"
        )
    primary_ok = [r for r in rows if r.get("hard_id") in PRIMARY and r.get("SUCCESS_12S") in (True, "True")]
    improved = [r for r in primary_ok if r.get("RECONFIG_IMPROVED") in (True, "True")]
    lines.extend(
        [
            "",
            f"Primary variants with SUCCESS_12S: **{len(primary_ok)}**",
            f"Of those, RECONFIG_IMPROVED: **{len(improved)}**",
            "",
            "## Precision continuation",
            "",
        ]
    )
    if not prec_rows:
        lines.append("No Precision continuation (no primary grasp with SUCCESS_12S ∧ RECONFIG_IMPROVED).")
    else:
        lines.append("| hard_id | variant | NEW | 12S | status |")
        lines.append("|---|---|---|---|---|")
        for p in prec_rows:
            lines.append(
                f"| `{p.get('hard_id')}` | {p.get('variant')} | {p.get('SUCCESS_NEW')} | "
                f"{p.get('SUCCESS_12S')} | {p.get('status')} |"
            )
    lines.extend(
        [
            "",
            "## Answer",
            "",
            "Can complete C recatch-and-lift transform 263/277 into a stable Precision-ready grasp? "
            "See counts above. Do **not** treat missing-lift failures from the prior pilot as evidence "
            "against completed C.",
            "",
        ]
    )
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--include-secondary", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    write_audit()

    # IC analysis
    from experiments.precision.c_reconfiguration_complete_lift.measure_ics import main as meas

    meas()
    ic_rows = json.loads((OUT / "IC_MEASUREMENTS.json").read_text(encoding="utf-8"))
    write_primary_analysis(ic_rows)

    if args.only:
        ids = [x.strip() for x in args.only.split(",") if x.strip()]
    else:
        ids = list(PRIMARY)
        if args.include_secondary:
            ids += SECONDARY

    env = make_env()
    book = load_codebook()
    pcfg = PrecisionConfig(evaluate_handoff=True, use_verified_lateral_fallback=False)

    result_rows = []
    trace_rows = []
    prec_rows = []
    cases = []

    for hid in ids:
        print(f"\n=== {hid} ===", flush=True)
        blob = load_blob(hid)
        case = {"hard_id": hid, "variants": {}}
        best = None
        for var in VARIANTS:
            print(f"  [{hid}] {var['label']}...", flush=True)
            try:
                packed = run_complete_c(env, blob, book, variant=var)
            except Exception as exc:
                print(f"    ERROR {exc}", flush=True)
                packed = {"variant": var["label"], "stage_fail": f"EXCEPTION:{exc}", "traceback": traceback.format_exc()}
            acts = packed.pop("actions", []) or []
            traces = packed.pop("traces", []) or []
            cap = packed.pop("capture", None)
            np.savez_compressed(
                OUT / "raw" / f"{hid}__{var['label']}__actions.npz",
                actions=np.asarray(acts, np.float32) if len(acts) else np.zeros((0, 7), np.float32),
            )
            if cap is not None:
                (OUT / "raw" / f"{hid}__{var['label']}__capture.pkl").write_bytes(pickle.dumps(cap))
            (OUT / "raw" / f"{hid}__{var['label']}__timeline.json").write_text(
                json.dumps(jsafe(traces), indent=2), encoding="utf-8"
            )
            for tr in traces:
                trace_rows.append({"hard_id": hid, "variant": var["label"], **tr})
            flat = flat_result(hid, packed)
            flat["rationale"] = var.get("rationale")
            result_rows.append(flat)
            case["variants"][var["label"]] = jsafe(packed)
            print(
                f"    REL={packed.get('RELEASE_EXECUTED')} REACQ={packed.get('REACQUIRE_EXECUTED')} "
                f"STAB={packed.get('CONTACT_STABILIZED')} LIFT={packed.get('LIFT_COMPLETED')} "
                f"12S={packed.get('SUCCESS_12S')} IMP={packed.get('RECONFIG_IMPROVED')} "
                f"fail={packed.get('stage_fail')}",
                flush=True,
            )
            if packed.get("SUCCESS_12S") and packed.get("RECONFIG_IMPROVED"):
                if best is None or abs(float((packed.get("errors_final") or {}).get("e_pos") or 1e9)) < abs(
                    float((best.get("errors_final") or {}).get("e_pos") or 1e9)
                ):
                    best = {**packed, "actions": acts, "capture": cap, "variant": var["label"]}

        # Precision continuation only for best improved+stable variant
        if best is not None and best.get("capture") is not None:
            print(f"  [{hid}] Precision continuation on {best['variant']}...", flush=True)
            restore_capture(env, best["capture"])
            res = run_precision(
                env,
                {"capture": best["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
                "B",
                pcfg,
            )
            hand = res.handoff or {}
            prec_rows.append(
                {
                    "hard_id": hid,
                    "variant": best["variant"],
                    "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
                    "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
                    "status": res.status,
                    "failures": "|".join(res.failures or []),
                }
            )
            # stamp SUCCESS_NEW onto matching result rows
            for r in result_rows:
                if r["hard_id"] == hid and r["variant"] == best["variant"]:
                    r["SUCCESS_NEW"] = bool(hand.get("SUCCESS_NEW"))
            np.savez_compressed(
                OUT / "raw" / f"{hid}__{best['variant']}__precision.npz",
                actions=np.asarray(res.actions or [], np.float32),
            )
            print(f"    NEW={hand.get('SUCCESS_NEW')} 12S={hand.get('SUCCESS_12S')}", flush=True)
        cases.append(case)

    (OUT / "CASES.json").write_text(json.dumps(jsafe(cases), indent=2), encoding="utf-8")
    write_csv(OUT / "C_RELEASE_RECATCH_LIFT_RESULTS.csv", result_rows)
    write_csv(OUT / "CONTACT_AND_HEIGHT_TRACES.csv", trace_rows)
    write_csv(OUT / "PRECISION_CONTINUATION_RESULTS.csv", prec_rows)
    write_final(result_rows, prec_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ids": ids,
        "n_primary_12s": sum(1 for r in result_rows if r.get("hard_id") in PRIMARY and r.get("SUCCESS_12S") is True),
        "n_primary_improved": sum(
            1 for r in result_rows if r.get("hard_id") in PRIMARY and r.get("RECONFIG_IMPROVED") is True
        ),
        "n_precision_new": sum(1 for r in prec_rows if r.get("SUCCESS_NEW") is True),
    }
    (OUT / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
