"""Stage A/B local search: delayed ZERO drop vs frozen heuristic rescue.

Does not overwrite config/impact_demo.yaml. Does not retune RULE or train SAC.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.physical_recovery import SUCCESS_HOLD, TABLE_DROP, Z_AIR  # noqa: E402
from training.impact_demo_core import LOG_DIR, frozen_hashes, load_demo_config, run_episode  # noqa: E402

OUT = LOG_DIR / "delayed_drop_search"
MASS = 0.35
VLATS = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4]
CAPTURED_AFTER_BALL = 0.20
PREFERRED_DROP_LATENCY = 0.30
PROBE_OFF = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]


def _cfg(vlat: float) -> dict:
    cfg = load_demo_config()
    cfg["physical"]["ball_mass"] = MASS
    cfg["physical"]["v_hit_lateral"] = float(vlat)
    return cfg


def _hold_after_ball(r: dict) -> float | None:
    t_ball = r.get("t_last_contact")
    vis = r["visualization"]
    t_loss = vis.get("t_both_contacts_lost")
    t_drop = vis.get("t_drop")
    end = t_loss if t_loss is not None else t_drop
    if t_ball is None or end is None:
        return None
    return float(end) - float(t_ball)


def classify_zero(r: dict) -> str:
    vis = r["visualization"]
    t_c = r.get("t_first_contact")
    t_drop = vis.get("t_drop")
    if t_c is None:
        return "NO_CONTACT"
    if r["final"].get("success_GT") and t_drop is None:
        return "PASSIVE_RECOVERY"
    if t_drop is not None:
        hold = _hold_after_ball(r)
        latency = float(t_drop) - float(t_c)
        if hold is not None and hold < CAPTURED_AFTER_BALL:
            return "IMMEDIATE_EJECTION"
        if latency < CAPTURED_AFTER_BALL:
            return "IMMEDIATE_EJECTION"
        return "DELAYED_DROP"
    return "PERSISTENT_DISTURBED"


def _stable_2s(h: dict) -> bool:
    vis = h["visualization"]
    t_eval = h.get("t_eval_end")
    if t_eval is None or not h["final"].get("success_GT"):
        return False
    if vis.get("t_drop") is not None and vis["t_drop"] < t_eval + 1.99:
        return False
    rows = [row for row in vis.get("post_impact_log") or [] if row["t"] >= t_eval - 1e-9]
    if not rows:
        tail = vis["tail_physical"]
        return (
            int(tail["nL"]) > 0
            and int(tail["nR"]) > 0
            and float(tail["obj_z"]) >= Z_AIR
            and (tail.get("fail_kind_live") or "") == ""
        )
    ok = True
    for row in rows:
        if int(row["nL"]) <= 0 or int(row["nR"]) <= 0:
            ok = False
            break
        if float(row["obj_z"]) < Z_AIR:
            ok = False
            break
        if row.get("fail_kind") == "drop":
            ok = False
            break
    dt_hold = float(rows[-1]["t"]) - float(t_eval)
    return ok and dt_hold >= 1.95


def _summary_zero(r: dict, vlat: float) -> dict:
    vis = r["visualization"]
    post = r.get("post_impact") or {}
    return {
        "mass": MASS,
        "v_lat": vlat,
        "v_hit_hx": r.get("v_hit_hx"),
        "v_hit_hand": r.get("v_hit_hand"),
        "v_hit_speed": r.get("pre_contact_speed"),
        "momentum": r.get("momentum"),
        "lateral_momentum_hx": r.get("lateral_momentum_hx"),
        "kinetic_energy": r.get("kinetic_energy"),
        "t_impact": r.get("t_first_contact"),
        "t_last_ball_contact": r.get("t_last_contact"),
        "t_first_unilateral": vis.get("t_first_unilateral_contact"),
        "t_both_lost": vis.get("t_both_contacts_lost"),
        "t_drop": vis.get("t_drop"),
        "hold_after_ball_s": _hold_after_ball(r),
        "drop_latency_s": (
            None
            if vis.get("t_drop") is None or r.get("t_first_contact") is None
            else float(vis["t_drop"]) - float(r["t_first_contact"])
        ),
        "post_e_x": post.get("e_x_GT"),
        "post_v_rel": post.get("v_rel_GT"),
        "post_omega": post.get("omega_rel_GT"),
        "post_nL": post.get("nL"),
        "post_nR": post.get("nR"),
        "post_obj_z": post.get("obj_z"),
        "eval_success_GT": r["final"]["success_GT"],
        "eval_fail_kind": r["final"]["fail_kind_GT"],
        "tail": vis.get("tail_physical"),
        "class": classify_zero(r),
        "preferred_latency": (
            vis.get("t_drop") is not None
            and r.get("t_first_contact") is not None
            and (float(vis["t_drop"]) - float(r["t_first_contact"])) >= PREFERRED_DROP_LATENCY
        ),
    }


def _qmax(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def audit_probes(z: dict, h: dict) -> dict:
    zp, hp = z.get("diag_probes") or [], h.get("diag_probes") or []
    rows = []
    n = min(len(zp), len(hp))
    first_div = None
    for i in range(n):
        a, b = zp[i], hp[i]
        rec = {
            "offset_s": a.get("offset_s"),
            "dqpos": _qmax(a["qpos"], b["qpos"]),
            "dqvel": _qmax(a["qvel"], b["qvel"]),
            "d_obj": _qmax(a["obj_pos"], b["obj_pos"]),
            "d_pdes": _qmax(a["p_des"], b["p_des"]),
            "d_ball": _qmax(a["ball_pos"], b["ball_pos"]),
            "zero": {
                "e_x": a["e_x"],
                "v_rel_h": a["v_rel_h"],
                "w_rel_h": a["w_rel_h"],
                "Fn_L": a["Fn_L"],
                "Fn_R": a["Fn_R"],
                "nL": a["nL"],
                "nR": a["nR"],
                "aperture": a["aperture"],
                "obj_z": a["obj_z"],
                "mode": a.get("mode"),
            },
            "heuristic": {
                "e_x": b["e_x"],
                "v_rel_h": b["v_rel_h"],
                "w_rel_h": b["w_rel_h"],
                "Fn_L": b["Fn_L"],
                "Fn_R": b["Fn_R"],
                "nL": b["nL"],
                "nR": b["nR"],
                "aperture": b["aperture"],
                "obj_z": b["obj_z"],
                "mode": b.get("mode"),
            },
        }
        material = rec["dqpos"] > 1e-5 or rec["d_pdes"] > 1e-5 or rec["d_obj"] > 1e-5
        rec["materially_diverged"] = bool(material)
        if material and first_div is None:
            first_div = rec["offset_s"]
        rows.append(rec)
    return {"first_divergence_offset_s": first_div, "rows": rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = frozen_hashes()
    stage_a = []
    print("STAGE A ZERO mass=0.35", flush=True)
    for vlat in VLATS:
        r = run_episode("zero", _cfg(vlat), screenshots=False, seed=0)
        row = _summary_zero(r, vlat)
        stage_a.append(row)
        (OUT / f"zero_vlat_{vlat:.1f}.json").write_text(
            json.dumps(
                {
                    "summary": row,
                    "t_eval_end": r.get("t_eval_end"),
                    "t_viewer_end": r.get("t_viewer_end"),
                    "final": r["final"],
                    "visualization": {
                        k: r["visualization"][k]
                        for k in (
                            "t_drop",
                            "t_first_unilateral_contact",
                            "t_both_contacts_lost",
                            "t_table_contact",
                            "tail_physical",
                            "no_drop_observed",
                        )
                    },
                    "post_impact": r.get("post_impact"),
                    "log": r["visualization"].get("post_impact_log"),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"A vlat={vlat:.1f} hx={row['v_hit_hx']:.3f} |v|={row['v_hit_speed']:.3f} "
            f"p={row['momentum']:.3f} KE={row['kinetic_energy']:.3f} class={row['class']} "
            f"t_both={row['t_both_lost']} t_drop={row['t_drop']} hold={row['hold_after_ball_s']}",
            flush=True,
        )

    delayed = [row for row in stage_a if row["class"] == "DELAYED_DROP"]
    stage_b = []
    winner = None
    print(f"STAGE B delayed-drop n={len(delayed)}", flush=True)
    for row in delayed:
        vlat = float(row["v_lat"])
        h = run_episode(
            "heuristic",
            _cfg(vlat),
            screenshots=False,
            seed=0,
            diag_probe_offsets=PROBE_OFF,
        )
        z = run_episode(
            "zero",
            _cfg(vlat),
            screenshots=False,
            seed=0,
            diag_probe_offsets=PROBE_OFF,
        )
        recov = bool(h["final"]["success_GT"]) and not h["final"]["failure_GT"]
        stable = _stable_2s(h)
        prevented = h["visualization"].get("t_drop") is None
        same_impact = (
            z["t_first_contact"] is not None
            and h["t_first_contact"] is not None
            and abs(float(z["t_first_contact"]) - float(h["t_first_contact"])) < 2e-3
            and abs(float(z["pre_contact_speed"]) - float(h["pre_contact_speed"])) < 1e-6
        )
        rec = {
            "v_lat": vlat,
            "heuristic_tested": True,
            "heuristic_recovered": recov,
            "heuristic_stable_2s": stable,
            "heuristic_prevented_drop": prevented,
            "same_impact": same_impact,
            "t_eval_end": h.get("t_eval_end"),
            "t_viewer_end": h.get("t_viewer_end"),
            "heur_final": h["final"],
            "heur_tail": h["visualization"].get("tail_physical"),
            "heur_modes": h["final"].get("heuristic_modes"),
            "zero_class": row["class"],
            "zero_t_drop": row["t_drop"],
            "audit": audit_probes(z, h),
        }
        rec["stage_b_pass"] = bool(
            recov and stable and prevented and same_impact and row.get("preferred_latency", True)
        )
        stage_b.append(rec)
        (OUT / f"stage_b_vlat_{vlat:.1f}.json").write_text(
            json.dumps(rec, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"B vlat={vlat:.1f} recov={recov} stable2s={stable} no_drop={prevented} "
            f"same={same_impact} pass={rec['stage_b_pass']} modes={rec['heur_modes']}",
            flush=True,
        )
        if rec["stage_b_pass"] and winner is None:
            winner = {"v_lat": vlat, "zero": row, "heur": rec, "z_run": z, "h_run": h}

    payload = {
        "hashes": hashes,
        "mass": MASS,
        "vlats": VLATS,
        "captured_after_ball_s": CAPTURED_AFTER_BALL,
        "preferred_drop_latency_s": PREFERRED_DROP_LATENCY,
        "table_disabled": True,
        "used_final_eval_data": False,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "winner": None
        if winner is None
        else {
            "label": "CANDIDATE_CANONICAL_DROP_CONTRAST",
            "v_lat": winner["v_lat"],
            "mass": MASS,
            "zero": winner["zero"],
            "heuristic": {k: winner["heur"][k] for k in winner["heur"] if k != "audit"},
            "audit": winner["heur"]["audit"],
        },
        "canonical_yaml_not_overwritten": True,
    }
    (OUT / "search.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("winner", None if winner is None else winner["v_lat"], flush=True)
    print("hashes", hashes["yaml"], hashes["controller_py"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
