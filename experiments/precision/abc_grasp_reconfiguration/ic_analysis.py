#!/usr/bin/env python
"""Measure Stage-1 exit residuals and select reconfiguration candidates."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.axial_inward import check_dz_inward_entry  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import check_stage1_eligible  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from evaluation.axis_aware_success import AXIS_V1_EX_MM, AXIS_V1_EZ_MM  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
OUT = ROOT / "results/diagnostics/abc_grasp_reconfiguration"
HARD_IDS = [
    "hard_910026",
    "hard_910132",
    "hard_910024",
    "hard_910263",
    "hard_910277",
    "hard_910084",
    "hard_910305",
]
PRIORITY = ["hard_910132", "hard_910024", "hard_910084"]


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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    pcfg = PrecisionConfig()
    rows = []
    for hid in HARD_IDS:
        blob = load_blob(hid)
        restore_capture(env, blob["capture"])
        e = measure_errors(env.sim)
        elig = check_stage1_eligible(
            env,
            {"capture": blob["capture"], "ref": blob["ref"], "sample_id": blob.get("sample_id")},
            allow_live_verify=True,
        )
        entry = check_dz_inward_entry(env.sim, pcfg)
        both_large = abs(float(e["dx_mm"])) > float(pcfg.dz_safe_ex_mm) and abs(float(e["dz_mm"])) > float(
            pcfg.dz_safe_ex_mm
        )
        rows.append(
            {
                "hard_id": hid,
                "dx_mm": float(e["dx_mm"]),
                "dy_mm": float(e["dy_mm"]),
                "dz_mm": float(e["dz_mm"]),
                "dtheta_deg": float(e["dtheta_deg"]),
                "e_pos_mm": float(e["e_pos_mm"]),
                "bilateral": bool(e.get("bilateral")),
                "fn_tot": float(e.get("fn_tot") or 0),
                "alpha_up_deg": float(e.get("alpha_up_deg") or 0),
                "fingertips_down": bool(e.get("fingertips_down")),
                "stage1_eligible": bool(elig.eligible),
                "SUCCESS_12S_evidence": bool(getattr(elig, "SUCCESS_12S", False) or elig.eligible),
                "dz_entry_ok": bool(entry.get("ok")),
                "dz_entry_reasons": list(entry.get("reasons") or []),
                "both_dx_dz_above_dz_safe": both_large,
                "abs_dx": abs(float(e["dx_mm"])),
                "abs_dz": abs(float(e["dz_mm"])),
                "priority": hid in PRIORITY,
            }
        )

    # Controllable ranges from Precision primitive authority (not SUCCESS 12 mm):
    # - dz_safe_ex_mm=6: inward DZ refuses larger |ex|
    # - AXIS_V1 ex/ez: assess_need / local axis bands
    # Empirically: successful DZ-first on 910305 started with |dx|≈5.5 < 6 and finished
    # with |dz|≪ ez band. Use T_dx = dz_safe_ex, T_dz = AXIS ez as "Precision-ready one-axis".
    t_dx = float(pcfg.dz_safe_ex_mm)
    t_dz = float(AXIS_V1_EZ_MM)
    selected = [
        r
        for r in rows
        if r["both_dx_dz_above_dz_safe"]
        or (r["priority"] and (r["abs_dx"] > t_dx or r["abs_dz"] > t_dz))
        or (r["abs_dx"] > t_dx and r["abs_dz"] > 0.5 * t_dz)
    ]
    # Always keep priority three + 910305 reference
    sel_ids = {r["hard_id"] for r in selected}
    for hid in PRIORITY + ["hard_910305"]:
        sel_ids.add(hid)
    selected = [r for r in rows if r["hard_id"] in sel_ids]

    payload = {
        "T_dx_mm": t_dx,
        "T_dz_mm": t_dz,
        "T_rationale": {
            "T_dx": "dz_safe_ex_mm=6 — inward DZ entry gate; Precision can correct DZ once |dx|≤T_dx",
            "T_dz": f"AXIS_V1_EZ_MM={AXIS_V1_EZ_MM} — local axial band used by assess_need / EZ axis",
            "not_used": "SUCCESS_NEW Euclidean 12 mm (final goal, not reconfig readiness)",
            "AXIS_V1_EX_MM": float(AXIS_V1_EX_MM),
        },
        "all_ics": rows,
        "selected_for_pilot": [r["hard_id"] for r in selected],
        "phase1_priority": PRIORITY,
    }
    (OUT / "IC_ANALYSIS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# INITIAL_IC_ANALYSIS",
        "",
        "## Controllable thresholds (RECONFIG_IMPROVED)",
        "",
        f"- `T_dx = {t_dx:.1f} mm` (= `dz_safe_ex_mm`): after reconfig, |DX|≤T_dx enables inward DZ.",
        f"- `T_dz = {t_dz:.1f} mm` (= AXIS_V1 EZ): after reconfig, |DZ|≤T_dz is in the local axial need band.",
        "- These are **not** the final SUCCESS_NEW 12 mm Euclidean criterion.",
        "",
        "## All seven verified B2 exits",
        "",
        "| hard_id | DX | DZ | dθ | e_pos | bilateral | dz_entry | both>|safe| |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| `{r['hard_id']}` | {r['dx_mm']:.2f} | {r['dz_mm']:.2f} | {r['dtheta_deg']:.2f} | "
            f"{r['e_pos_mm']:.2f} | {r['bilateral']} | {r['dz_entry_ok']} | {r['both_dx_dz_above_dz_safe']} |"
        )
    md.extend(
        [
            "",
            f"## Selected for pilot: {', '.join(payload['selected_for_pilot'])}",
            "",
            f"Phase I priority: {', '.join(PRIORITY)}",
            "",
            "`hard_910305` kept as DZ-first reference (corrected S1 already SUCCESS_NEW).",
            "",
        ]
    )
    (OUT / "INITIAL_IC_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"selected": payload["selected_for_pilot"], "T_dx": t_dx, "T_dz": t_dz}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
