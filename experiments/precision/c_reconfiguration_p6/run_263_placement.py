#!/usr/bin/env python
"""P6 Part A: audit mid_body_early end-grasp on 910263; axial placement pilot."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.c_reconfiguration_complete_lift.complete_c import (  # noqa: E402
    run_complete_c,
)
from experiments.precision.c_reconfiguration_p6.geometry import snapshot_geometry  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402

OUT = ROOT / "results/diagnostics/c_reconfiguration_p6"
EXIT = ROOT / "results/diagnostics/b2_coupled_dx_dz/stage1_exits"
PRIOR = ROOT / "results/diagnostics/c_reconfiguration_complete_lift/raw"
HID = "hard_910263"

# Keep mid_body_early timing; only vary cylinder-axis recatch target s (meters along u).
# Rationale: baseline s=0 realized contact_s_mean≈+0.026 (near +end, CYL_HALF=0.03).
# Command negative s to bias interception toward mid-body / opposite end.
PLACEMENT = [
    {"label": "baseline_s0", "s_axial": 0.0, "t_c": 0.10, "open_steps": 4, "rationale": "Prior mid_body_early (end-biased)"},
    {"label": "s_m010", "s_axial": -0.010, "t_c": 0.10, "open_steps": 4, "rationale": "−10 mm along cylinder axis toward mid from +end bias"},
    {"label": "s_m015", "s_axial": -0.015, "t_c": 0.10, "open_steps": 4, "rationale": "−15 mm axial (stronger mid-body pull)"},
    {"label": "s_m020", "s_axial": -0.020, "t_c": 0.10, "open_steps": 4, "rationale": "−20 mm axial"},
    {"label": "s_p010", "s_axial": 0.010, "t_c": 0.10, "open_steps": 4, "rationale": "Control: +10 mm (expect worse end bias)"},
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


def load_blob():
    blob = pickle.loads((EXIT / f"{HID}.pkl").read_bytes())
    meta = json.loads((EXIT / f"{HID}__meta.json").read_text(encoding="utf-8"))
    return {**blob, "sample_id": meta.get("raw_id"), "hard_id": HID, "meta": meta}


def stage_geometry_along_actions(env, blob, actions: np.ndarray) -> dict:
    """Replay actions; sample geometry at release / reacquire / post-lift / end."""
    restore_capture(env, blob["capture"])
    stages = {"pre_release": snapshot_geometry(env.sim)}
    saw_off = False
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
        g = snapshot_geometry(env.sim)
        n_bil = g["bilateral"]
        # Heuristic phase markers via bilateral edges
        if not n_bil and not saw_off:
            saw_off = True
            stages["at_release"] = g
            stages["at_release_step"] = i
        if saw_off and n_bil and "at_reacquire" not in stages:
            stages["at_reacquire"] = g
            stages["at_reacquire_step"] = i
    stages["final"] = snapshot_geometry(env.sim)
    # Approximate after-lift: last time obj_z rose above 0.52 with bilateral
    restore_capture(env, blob["capture"])
    best_lift = None
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32).reshape(7))
        g = snapshot_geometry(env.sim)
        if g["bilateral"] and g["obj_z"] >= 0.52:
            best_lift = g
    if best_lift is not None:
        stages["after_lift"] = best_lift
    return stages


def audit_baseline(env, blob) -> dict:
    act_path = PRIOR / f"{HID}__mid_body_early__actions.npz"
    actions = np.load(act_path)["actions"]
    stages = stage_geometry_along_actions(env, blob, actions)
    # DZ deterioration source
    dz_path = []
    for key in ("pre_release", "at_release", "at_reacquire", "after_lift", "final"):
        if key in stages:
            dz_path.append({"stage": key, "dz": stages[key].get("dz"), "dx": stages[key].get("dx"), "s_mean": stages[key].get("contact_s_mean"), "kind": stages[key].get("grasp_kind")})
    # Find largest |dz| jump
    worst = None
    for a, b in zip(dz_path, dz_path[1:]):
        if a["dz"] is None or b["dz"] is None:
            continue
        jump = abs(float(b["dz"])) - abs(float(a["dz"]))
        if worst is None or jump > worst["jump"]:
            worst = {"from": a["stage"], "to": b["stage"], "jump": jump, "dz0": a["dz"], "dz1": b["dz"]}
    return {"stages": jsafe(stages), "dz_path": dz_path, "dz_worst_jump": worst, "n_actions": len(actions)}


def write_audit(audit: dict) -> None:
    st = audit["stages"]
    lines = [
        "# 910263_CONTACT_GEOMETRY_AUDIT",
        "",
        "Replay of prior successful `mid_body_early` (s_axial=0, t_c=0.10).",
        "",
        "## Cylinder / contact (MuJoCo)",
        "",
        f"- CYL_HALF = 0.03 m. End proximity when `|contact_s_mean| ≳ 0.022`.",
        f"- Final grasp_kind: `{((st.get('final') or {}).get('grasp_kind'))}`",
        f"- Final contact_s_mean: `{(st.get('final') or {}).get('contact_s_mean')}` m",
        f"- Final n_end_face: `{(st.get('final') or {}).get('n_end_face')}`, opposing_sidewall: `{(st.get('final') or {}).get('opposing_sidewall')}`",
        f"- Final dist_nearest_end: `{(st.get('final') or {}).get('dist_nearest_end_m')}` m",
        "",
        "## Residuals along trajectory",
        "",
        "| stage | DX | DZ | dθ | s_mean | grasp_kind |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in audit["dz_path"]:
        lines.append(
            f"| {row['stage']} | {row.get('dx')} | {row.get('dz')} | — | {row.get('s_mean')} | {row.get('kind')} |"
        )
    w = audit.get("dz_worst_jump") or {}
    lines.extend(
        [
            "",
            "## DZ deterioration",
            "",
            f"Largest |DZ| increase: **{w.get('from')} → {w.get('to')}** "
            f"(dz {w.get('dz0')} → {w.get('dz1')}, Δabs≈{w.get('jump')}).",
            "",
            "Interpretation: commanded `s_axial=0` (cylinder COM) realized contact near the **+end** "
            "(`contact_s_mean≈+0.026`). Placement bias is in cylinder-axis frame, not world-Z.",
            "",
            "## Placement pilot plan",
            "",
            "Keep mid_body_early timing; command negative `s_axial` to counteract +end bias.",
            "",
        ]
    )
    (OUT / "910263_CONTACT_GEOMETRY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    env = make_env()
    book = load_codebook()
    blob = load_blob()

    print("Auditing baseline mid_body_early geometry...", flush=True)
    audit = audit_baseline(env, blob)
    (OUT / "raw" / "910263_baseline_geometry.json").write_text(json.dumps(jsafe(audit), indent=2), encoding="utf-8")
    write_audit(audit)
    print("dz_worst", audit.get("dz_worst_jump"), flush=True)
    print("final s_mean", (audit["stages"].get("final") or {}).get("contact_s_mean"), flush=True)

    rows = []
    for var in PLACEMENT:
        print(f"  placement {var['label']} s={var['s_axial']}...", flush=True)
        packed = run_complete_c(
            env,
            blob,
            book,
            variant={
                "label": var["label"],
                "s_axial": var["s_axial"],
                "t_c": var["t_c"],
                "open_steps": var["open_steps"],
            },
        )
        acts = packed.pop("actions", []) or []
        traces = packed.pop("traces", []) or []
        cap = packed.pop("capture", None)
        np.savez_compressed(OUT / "raw" / f"910263__{var['label']}__actions.npz", actions=np.asarray(acts, np.float32))
        if cap is not None:
            (OUT / "raw" / f"910263__{var['label']}__capture.pkl").write_bytes(pickle.dumps(cap))
        # Geometry at end
        if cap is not None:
            restore_capture(env, cap)
            g = snapshot_geometry(env.sim)
        else:
            g = {}
        # Commanded vs realized
        row = {
            "hard_id": HID,
            "label": var["label"],
            "rationale": var["rationale"],
            "commanded_s_axial": var["s_axial"],
            "realized_contact_s_mean": g.get("contact_s_mean"),
            "grasp_kind": g.get("grasp_kind"),
            "n_end_face": g.get("n_end_face"),
            "opposing_sidewall": g.get("opposing_sidewall"),
            "dist_nearest_end_m": g.get("dist_nearest_end_m"),
            "CONTACT_PLACEMENT_VALID": g.get("CONTACT_PLACEMENT_VALID"),
            "REACQUIRE_EXECUTED": packed.get("REACQUIRE_EXECUTED"),
            "LIFT_COMPLETED": packed.get("LIFT_COMPLETED"),
            "SUCCESS_12S": packed.get("SUCCESS_12S"),
            "RECONFIG_IMPROVED": packed.get("RECONFIG_IMPROVED"),
            "dx_in_band": packed.get("dx_in_band"),
            "dz_in_band": packed.get("dz_in_band"),
            "init_dx": (packed.get("errors0") or {}).get("dx"),
            "init_dz": (packed.get("errors0") or {}).get("dz"),
            "final_dx": (packed.get("errors_final") or g).get("dx") if isinstance(packed.get("errors_final") or g, dict) else None,
            "final_dz": (packed.get("errors_final") or {}).get("dz") if packed.get("errors_final") else g.get("dz"),
            "final_dtheta": (packed.get("errors_final") or {}).get("dtheta") if packed.get("errors_final") else g.get("dtheta"),
            "stage_fail": packed.get("stage_fail"),
            "s_command_minus_realized": (
                float(var["s_axial"]) - float(g["contact_s_mean"])
                if g.get("contact_s_mean") is not None and np.isfinite(float(g.get("contact_s_mean")))
                else None
            ),
        }
        # Prefer errors_final when present
        ef = packed.get("errors_final") or {}
        if ef:
            row["final_dx"] = ef.get("dx")
            row["final_dz"] = ef.get("dz")
            row["final_dtheta"] = ef.get("dtheta")
        rows.append(row)
        (OUT / "raw" / f"910263__{var['label']}__meta.json").write_text(json.dumps(jsafe({**packed, "geometry": g}), indent=2), encoding="utf-8")
        print(
            f"    12S={row['SUCCESS_12S']} place={row['CONTACT_PLACEMENT_VALID']} "
            f"s_cmd={row['commanded_s_axial']} s_real={row['realized_contact_s_mean']} "
            f"kind={row['grasp_kind']} dx={row['final_dx']} dz={row['final_dz']} fail={row['stage_fail']}",
            flush=True,
        )

    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (OUT / "910263_RECATCH_PLACEMENT.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
