"""Compare contact-surface geometry for gbi_010 / gbi_014 / gbi_022 Plan C tapes.

Writes only under results/diagnostics/ground_plan_c_contact_surface/.
Does not modify existing results or runtime observations.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_generation.oracles.plan_c.dynamic_catch as dc
import data_generation.oracles.plan_c.interception as er

sys.modules.setdefault("data_generation.strategies.ground_dynamic_catch", dc)
sys.modules.setdefault("data_generation.strategies.ground_early_recovery", er)

from data_generation.oracles.plan_c.contact_surface import classify_contacts, surface_score
from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from training.recovery_execution import restore_ballistic

CATCH = ROOT / "results/diagnostics/ground_dynamic_catch_oracle/raw"
SUCCESS = ROOT / "results/diagnostics/ground_plan_c_final_success/raw"
OUT = ROOT / "results/diagnostics/ground_plan_c_contact_surface"
ROOTS = ["gbi_010", "gbi_022", "gbi_014"]


def jsafe(o):
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


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def snap_pack(sim, tag: str) -> dict:
    audit = classify_contacts(sim)
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    return {
        "phase": tag,
        "grasp_kind": audit["grasp_kind"],
        "opposing_sidewall": audit["opposing_sidewall"],
        "maj_L": audit["maj_L"],
        "maj_R": audit["maj_R"],
        "n_sidewall": audit["n_sidewall"],
        "n_end_face": audit["n_end_face"],
        "n_edge": audit["n_edge"],
        "nL": audit["nL"],
        "nR": audit["nR"],
        "bilateral": int(audit["bilateral"]),
        "normal_dot_LR": audit["normal_dot_LR"],
        "lateral_sep_dot": audit["lateral_sep_dot"],
        "contact_s_mean": audit["contact_s_mean"],
        "surface_score": surface_score(audit),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "u_c_h": m["u_c_h"],
        "fnL": float(pb["fnL"]),
        "fnR": float(pb["fnR"]),
        "obj_z": float(m["obj_z"]),
        "contacts": audit["contacts"],
        "contact_s_only_axial": True,
    }


def replay_catch_phases(sim, catch: dict) -> dict:
    """Replay capture tape; snapshot at first bil, end of capture (brake), mid-hold."""
    restore_ballistic(sim, catch["snap"])
    restore_drop(sim, None)
    arm(sim)
    acts = [np.asarray(a, np.float32) for a in catch["capture_actions"]]
    phases = {"pre": snap_pack(sim, "pre_recatch")}
    first_bil = None
    for i, a in enumerate(acts):
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        if first_bil is None and pb["bilateral"]:
            first_bil = snap_pack(sim, "first_bilateral")
            first_bil["step"] = i
        if latched(sim):
            break
    phases["first_bilateral"] = first_bil
    phases["after_brake"] = snap_pack(sim, "after_brake")
    # Continue remaining actions if catch pack has post-capture hold
    extra = catch.get("actions")
    if extra is not None and len(extra) > len(acts):
        for a in extra[len(acts) :]:
            step_legal(sim, np.asarray(a, np.float32))
            if latched(sim):
                break
        phases["after_full_actions"] = snap_pack(sim, "after_full_actions")
    return phases


def replay_success_phases(sim, expert: dict, catch: dict) -> dict:
    """Replay SUCCESS_NEW expert: catch → elevated → just before handoff if possible."""
    restore_ballistic(sim, catch["snap"])
    restore_drop(sim, None)
    arm(sim)
    prefix = expert.get("prefix_actions")
    if prefix is None:
        prefix = expert.get("actions")
    acts = [np.asarray(a, np.float32) for a in prefix]
    # Find first bil during prefix
    first_bil = None
    after_catch = None
    n_cap = len(catch["capture_actions"])
    for i, a in enumerate(acts):
        step_legal(sim, a)
        pb = pad_bilateral(sim)
        if first_bil is None and pb["bilateral"]:
            first_bil = snap_pack(sim, "first_bilateral")
            first_bil["step"] = i
        if i + 1 == n_cap:
            after_catch = snap_pack(sim, "after_brake")
        if latched(sim):
            break
    # Elevated / pre-handoff: end of prefix (SUCCESS tapes include lift; handoff is separate)
    pre_handoff = snap_pack(sim, "pre_handoff")
    return {
        "first_bilateral": first_bil,
        "after_brake": after_catch,
        "pre_handoff": pre_handoff,
        "n_prefix": len(acts),
    }


def compact(phase: dict | None) -> dict | None:
    if phase is None:
        return None
    keep = {
        k: phase[k]
        for k in (
            "phase",
            "step",
            "grasp_kind",
            "opposing_sidewall",
            "maj_L",
            "maj_R",
            "n_sidewall",
            "n_end_face",
            "n_edge",
            "nL",
            "nR",
            "bilateral",
            "normal_dot_LR",
            "lateral_sep_dot",
            "contact_s_mean",
            "surface_score",
            "e_theta_deg",
            "e_pos_mm",
            "u_c_h",
            "fnL",
            "fnR",
            "obj_z",
        )
        if k in phase
    }
    # Keep up to 6 raw contacts for evidence
    keep["contacts"] = (phase.get("contacts") or [])[:6]
    return keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim, _ = make_ground_sim()
    comparison = []
    detail = {}

    for rid in ROOTS:
        catch = pickle.loads((CATCH / f"{rid}__catch_expert.pkl").read_bytes())
        catch_phases = replay_catch_phases(sim, catch)
        succ_path = SUCCESS / f"{rid}__SUCCESS_NEW_expert.pkl"
        succ_phases = None
        if succ_path.exists():
            expert = pickle.loads(succ_path.read_bytes())
            succ_phases = replay_success_phases(sim, expert, catch)

        elev_path = SUCCESS / f"{rid}__elevated_snap.pkl"
        elev_pack = None
        if elev_path.exists():
            elev_pack = pickle.loads(elev_path.read_bytes())
            restore_ballistic(sim, elev_pack["elev_snap"])
            restore_drop(sim, None)
            arm(sim)
            elev_audit = snap_pack(sim, "elevated")
        else:
            elev_audit = None

        row = {
            "root_id": rid,
            "mechanism": catch.get("mechanism"),
            "cand_s": (catch.get("cand") or {}).get("s"),
            "catch_first_kind": (catch_phases.get("first_bilateral") or {}).get("grasp_kind"),
            "catch_first_oppose": (catch_phases.get("first_bilateral") or {}).get("opposing_sidewall"),
            "catch_first_theta": (catch_phases.get("first_bilateral") or {}).get("e_theta_deg"),
            "catch_first_s": (catch_phases.get("first_bilateral") or {}).get("contact_s_mean"),
            "catch_first_n_face": (catch_phases.get("first_bilateral") or {}).get("n_end_face"),
            "catch_first_n_wall": (catch_phases.get("first_bilateral") or {}).get("n_sidewall"),
            "catch_brake_kind": (catch_phases.get("after_brake") or {}).get("grasp_kind"),
            "catch_brake_oppose": (catch_phases.get("after_brake") or {}).get("opposing_sidewall"),
            "catch_brake_theta": (catch_phases.get("after_brake") or {}).get("e_theta_deg"),
            "catch_brake_s": (catch_phases.get("after_brake") or {}).get("contact_s_mean"),
            "elev_kind": (elev_audit or {}).get("grasp_kind"),
            "elev_oppose": (elev_audit or {}).get("opposing_sidewall"),
            "elev_theta": (elev_audit or {}).get("e_theta_deg"),
            "elev_pos_mm": (elev_audit or {}).get("e_pos_mm"),
            "elev_s": (elev_audit or {}).get("contact_s_mean"),
            "elev_n_face": (elev_audit or {}).get("n_end_face"),
            "elev_n_wall": (elev_audit or {}).get("n_sidewall"),
            "has_SUCCESS_NEW": int(succ_path.exists()),
        }
        if succ_phases:
            row["succ_pre_handoff_kind"] = (succ_phases.get("pre_handoff") or {}).get("grasp_kind")
            row["succ_pre_handoff_theta"] = (succ_phases.get("pre_handoff") or {}).get("e_theta_deg")
            row["succ_pre_handoff_oppose"] = (succ_phases.get("pre_handoff") or {}).get("opposing_sidewall")

        comparison.append(row)
        detail[rid] = {
            "catch": {k: compact(v) for k, v in catch_phases.items()},
            "elevated": compact(elev_audit),
            "success": {k: compact(v) for k, v in (succ_phases or {}).items() if isinstance(v, dict)},
        }
        print(
            f"{rid}: first={row['catch_first_kind']} theta={row['catch_first_theta']} "
            f"brake={row['catch_brake_kind']} elev={row['elev_kind']} elev_th={row['elev_theta']}",
            flush=True,
        )

    write_csv(OUT / "COMPARISON.csv", comparison)
    write_json(OUT / "DETAIL.json", detail)

    # Concise markdown report
    lines = [
        "# Plan C contact-surface comparison",
        "",
        "`contact_s_mean` = mean axial coordinate of contacts along the cylinder axis.",
        "It does **not** encode sidewall vs end-face identity.",
        "",
        "| root | first_bil kind | θ° | n_face/n_wall | brake kind | elev kind | elev θ° | SUCCESS_NEW |",
        "|---|---|---:|---|---|---|---:|---|",
    ]
    for r in comparison:
        lines.append(
            f"| {r['root_id']} | {r['catch_first_kind']} | "
            f"{r['catch_first_theta']:.1f} | "
            f"{r['catch_first_n_face']}/{r['catch_first_n_wall']} | "
            f"{r['catch_brake_kind']} | {r['elev_kind']} | "
            f"{(r['elev_theta'] if r['elev_theta'] is not None else float('nan')):.1f} | "
            f"{r['has_SUCCESS_NEW']} |"
        )
    lines.extend(
        [
            "",
            "## Distinguishing factor",
            "",
        ]
    )
    kinds = {r["root_id"]: r["catch_first_kind"] for r in comparison}
    elev_kinds = {r["root_id"]: r["elev_kind"] for r in comparison}
    if kinds.get("gbi_014") in ("end_face_pinch", "mixed_face_wall", "edge_only") and all(
        kinds.get(x) in ("opposing_sidewall", "sidewall_partial") for x in ("gbi_010", "gbi_022")
    ):
        lines.append(
            "Wrong-surface contact is the distinguishing factor at first recatch: "
            "gbi_014 differs from the two SUCCESS_NEW roots."
        )
    elif elev_kinds.get("gbi_014") in ("end_face_pinch", "mixed_face_wall") and all(
        elev_kinds.get(x) in ("opposing_sidewall", "sidewall_partial") for x in ("gbi_010", "gbi_022")
    ):
        lines.append(
            "At first bil the surfaces may be similar or mixed; by elevated lift "
            "gbi_014 is distinguished by end-face / mixed contact vs sidewall on SUCCESS roots."
        )
    else:
        lines.append(
            "See DETAIL.json for numerical contact points/normals. "
            "Wrong-surface may be present but not the sole separator — check θ and oppose flags."
        )
    lines.extend(
        [
            "",
            "Raw contact evidence (geom IDs, pos_w, n_w, rho, z, n_axial) is in `DETAIL.json`.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
