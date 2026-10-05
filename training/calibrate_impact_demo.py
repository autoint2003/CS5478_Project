"""Stage A/B search for guided-approach + free lateral impact. Not the user CLI."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.impact_demo_core import (  # noqa: E402
    DEMO_YAML,
    LOG_DIR,
    frozen_hashes,
    load_demo_config,
    run_episode,
)


def _classify(post: dict | None, fin: dict) -> str:
    if post is None:
        if fin.get("failure_GT") or float(fin.get("obj_z", 0)) < 0.45:
            return "catastrophic"
        return "no_contact"
    z = float(post.get("obj_z", 0.0))
    nL, nR = int(post.get("nL", 0)), int(post.get("nR", 0))
    if (nL + nR) == 0 or z < 0.45 or fin.get("fail_kind_GT") in ("drop", "scene", "contact_loss"):
        return "catastrophic"
    if fin.get("success_GT"):
        return "too_weak"
    ez = abs(float(post.get("e_x_GT", 0.0)))
    vr = float(post.get("v_rel_GT", 0.0))
    if ez < 0.0015 and vr < 0.03 and post.get("captured_bilateral"):
        return "too_weak"
    if post.get("captured_bilateral") or (nL > 0 or nR > 0):
        return "disturbed_captured"
    return "catastrophic"


def stage_a() -> list[dict]:
    base = load_demo_config()
    masses = [0.20, 0.35, 0.50]
    vlats = [3.0, 5.0, 7.0]
    rows = []
    for m in masses:
        for vlat in vlats:
            cfg = deepcopy(base)
            cfg["physical"]["ball_mass"] = float(m)
            cfg["physical"]["v_hit_lateral"] = float(vlat)
            r = run_episode("zero", cfg, screenshots=False, seed=0)
            kind = _classify(r["post_impact"], r["final"])
            row = {
                "mass": m,
                "v_hit_lateral": vlat,
                "t_release": r.get("t_release"),
                "t_first_contact": r["t_first_contact"],
                "free_flight_s": r.get("free_flight_s"),
                "v_hit_hand": r.get("v_hit_hand"),
                "v_hit_hx": r.get("v_hit_hx"),
                "v_hit_hy": r.get("v_hit_hy"),
                "v_hit_hz": r.get("v_hit_hz"),
                "pre_speed": r["pre_contact_speed"],
                "lateral_momentum_hx": r.get("lateral_momentum_hx"),
                "kinetic_energy": r["kinetic_energy"],
                "release": r.get("release"),
                "post": r["post_impact"],
                "final": r["final"],
                "class": kind,
            }
            rows.append(row)
            print(
                f"A m={m:.2f} vlat={vlat:.2f} t_c={r['t_first_contact']} "
                f"hx={r.get('v_hit_hx')} hy={r.get('v_hit_hy')} hz={r.get('v_hit_hz')} "
                f"p_hx={r.get('lateral_momentum_hx')} KE={r['kinetic_energy']:.3f} class={kind} "
                f"Zsucc={r['final']['success_GT']} fail={r['final']['fail_kind_GT']}",
                flush=True,
            )
    return rows


def stage_b(candidates: list[dict]) -> list[dict]:
    base = load_demo_config()
    out = []
    for c in candidates:
        cfg = deepcopy(base)
        cfg["physical"]["ball_mass"] = float(c["mass"])
        cfg["physical"]["v_hit_lateral"] = float(c["v_hit_lateral"])
        z = run_episode("zero", cfg, screenshots=False, seed=0)
        h = run_episode("heuristic", cfg, screenshots=False, seed=0)
        rec = {
            "mass": c["mass"],
            "v_hit_lateral": c["v_hit_lateral"],
            "zero": z["final"],
            "heuristic": h["final"],
            "zero_t_contact": z["t_first_contact"],
            "heur_t_contact": h["t_first_contact"],
            "zero_v_hit_hand": z.get("v_hit_hand"),
            "heur_v_hit_hand": h.get("v_hit_hand"),
            "same_contact": (
                z["t_first_contact"] is not None
                and h["t_first_contact"] is not None
                and abs(z["t_first_contact"] - h["t_first_contact"]) < 2e-3
            ),
            "zero_fails_or_unrecovered": (not z["final"]["success_GT"]),
            "heuristic_recovers": bool(h["final"]["success_GT"]) and not h["final"]["failure_GT"],
        }
        rec["contrast"] = rec["zero_fails_or_unrecovered"] and rec["heuristic_recovers"]
        out.append(rec)
        print(
            f"B m={c['mass']:.2f} vlat={c['v_hit_lateral']:.2f} "
            f"ZERO success={z['final']['success_GT']} fail={z['final']['fail_kind_GT']} "
            f"HEUR success={h['final']['success_GT']} fail={h['final']['fail_kind_GT']} "
            f"contrast={rec['contrast']}",
            flush=True,
        )
    return out


def write_config(mass: float, vlat: float) -> None:
    cfg = load_demo_config()
    cfg["physical"]["ball_mass"] = float(mass)
    cfg["physical"]["v_hit_lateral"] = float(vlat)
    DEMO_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="AB", choices=("A", "B", "AB"))
    p.add_argument("--write-config", action="store_true")
    args = p.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"hashes_before_search": frozen_hashes(), "used_final_eval_data": False, "family": "guided_approach_free_flight"}
    a_rows = []
    if args.stage in ("A", "AB"):
        a_rows = stage_a()
        payload["stage_a"] = a_rows
        (LOG_DIR / "stage_a_guided.json").write_text(json.dumps(a_rows, indent=2, default=str), encoding="utf-8")
    if args.stage in ("B", "AB"):
        if not a_rows:
            a_rows = json.loads((LOG_DIR / "stage_a_guided.json").read_text(encoding="utf-8"))
        cand = [r for r in a_rows if r["class"] == "disturbed_captured"]
        payload["stage_b_candidates"] = cand
        b_rows = stage_b(cand) if cand else []
        payload["stage_b"] = b_rows
        (LOG_DIR / "stage_b_guided.json").write_text(json.dumps(b_rows, indent=2, default=str), encoding="utf-8")
        contrast = [r for r in b_rows if r["contrast"]]
        payload["NO_DEMO_CONTRAST_FOUND"] = not bool(contrast)
        if args.write_config:
            if contrast:
                write_config(contrast[0]["mass"], contrast[0]["v_hit_lateral"])
                payload["selected"] = contrast[0]
            elif cand:
                write_config(cand[0]["mass"], cand[0]["v_hit_lateral"])
                payload["selected"] = cand[0]
                payload["selected_note"] = "no ZERO-fail / HEURISTIC-success contrast; wrote first DISTURBED_CAPTURED"
            else:
                payload["selected_note"] = "no DISTURBED_CAPTURED band; config left as search defaults"
    (LOG_DIR / "calibration_guided.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("wrote", LOG_DIR / "calibration_guided.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
