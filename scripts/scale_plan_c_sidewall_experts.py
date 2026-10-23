#!/usr/bin/env python
"""Scale Plan C surface-aware sidewall recatch toward 50 independent SUCCESS_NEW roots.

Writes only under results/diagnostics/ground_plan_c_scale50/.
Does not modify existing results/ diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.expert_search import (
    build_physical_root,
    diverse_impact_configs,
    exact_replay_3of3,
    lift_and_validate,
    pack_expert,
    physical_root_fingerprint,
    search_catch,
)
from data_generation.oracles.plan_c.interception import FULL_DETACH_ACTIVE_RECATCH
from envs.simulation.ground_sim import make_ground_sim

OUT = ROOT / "results/diagnostics/ground_plan_c_scale50"
RAW = OUT / "raw"
ACCEPTED = RAW / "accepted"
GOLDEN_C = ROOT / "results/diagnostics/ground_plan_c_final_success/raw"
SIDEWALL_014 = ROOT / "results/diagnostics/ground_plan_c_sidewall_research/raw/gbi_014__SUCCESS_NEW_sidewall_expert.pkl"


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


def write_csv(path: Path, rows: list[dict]):
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


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    keys = list(row.keys())
    if exists:
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            keys = list(r.fieldnames or keys)
            for k in row:
                if k not in keys:
                    keys.append(k)
        # rewrite header if needed — simpler: append with extras
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: jsafe(row.get(k)) for k in keys})


def seed_golden(accepted_fps: set[str], success_rows: list) -> int:
    """Import verified gbi_010 / gbi_022 / new gbi_014 as regression seeds (count as successes)."""
    n = 0
    seeds = [
        ("gbi_010", GOLDEN_C / "gbi_010__SUCCESS_NEW_expert.pkl", "golden"),
        ("gbi_022", GOLDEN_C / "gbi_022__SUCCESS_NEW_expert.pkl", "golden"),
        ("gbi_014", SIDEWALL_014, "sidewall_research"),
    ]
    for rid, src, tag in seeds:
        if not src.exists():
            print(f"  seed missing {rid} {src}", flush=True)
            continue
        blob = pickle.loads(src.read_bytes())
        snap = blob.get("catch_snap") or blob.get("snap")
        if snap is None:
            print(f"  seed skip {rid}: no snap", flush=True)
            continue
        cfg = dict(blob.get("cfg") or {})
        if not cfg:
            prior = (
                ROOT
                / "results/diagnostics/ground_ball_impact_oracle_interception/raw"
                / f"{rid}__post_loss.pkl"
            )
            if prior.exists():
                cfg = dict(pickle.loads(prior.read_bytes()).get("config") or {})
        cfg.setdefault("config_id", rid)
        # Reject old end-face 014
        if rid == "gbi_014" and blob.get("grasp_kind") == "end_face_pinch":
            print("  skip old end-face gbi_014", flush=True)
            continue
        fp = blob.get("fingerprint") or physical_root_fingerprint(cfg, snap)
        if fp in accepted_fps:
            continue
        dest = ACCEPTED / f"{rid}__SUCCESS_NEW_expert.pkl"
        if "catch_snap" not in blob:
            blob["catch_snap"] = snap
        if "snap" not in blob:
            blob["snap"] = snap
        if "actions" not in blob and "prefix_actions" in blob:
            blob["actions"] = blob["prefix_actions"]
        blob["root_id"] = rid
        blob["cfg"] = cfg
        blob["fingerprint"] = fp
        blob["SUCCESS_NEW"] = True
        blob["seed_source"] = tag
        if "grasp_kind" not in blob:
            blob["grasp_kind"] = "opposing_sidewall"
            blob["opposing_sidewall"] = 1
        dest.write_bytes(pickle.dumps(blob))
        accepted_fps.add(fp)
        success_rows.append({
            "root_id": rid,
            "fingerprint": fp,
            "SUCCESS_NEW": 1,
            "seed": 1,
            "source": tag,
            "grasp_kind": blob.get("grasp_kind") or "opposing_sidewall",
            "approach": (cfg or {}).get("approach"),
            "speed": (cfg or {}).get("speed"),
        })
        n += 1
        print(f"  seeded {rid} fp={fp[:8]}…", flush=True)
    return n


def diversity_row(cfg: dict, validated: dict, events: dict | None) -> dict:
    m_e = validated.get("m_elev") or {}
    surf = validated.get("surf_elev") or {}
    return {
        "approach": cfg.get("approach"),
        "speed": cfg.get("speed"),
        "axial_offset": cfg.get("axial_offset"),
        "tilt_deg": cfg.get("tilt_deg"),
        "offset": cfg.get("offset"),
        "z": cfg.get("z"),
        "z_off": cfg.get("z_off"),
        "slip_s": None if not events else (
            None
            if events.get("first_sustained_loss_t") is None or events.get("first_ball_contact_t") is None
            else float(events["first_sustained_loss_t"]) - float(events["first_ball_contact_t"])
        ),
        "vrel_recatch": validated.get("vrel_at_first_contact"),
        "detach_s": validated.get("max_contact_free_s"),
        "grasp_kind": surf.get("grasp_kind"),
        "contact_s_mean": surf.get("contact_s_mean"),
        "e_theta_elev": m_e.get("e_theta_deg"),
        "e_pos_elev_mm": m_e.get("e_pos_mm"),
        "obj_z_elev": m_e.get("obj_z"),
        "n_prefix": len(validated.get("prefix_actions") or []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--max-configs", type=int, default=400)
    ap.add_argument("--rollouts-per-root", type=int, default=36)
    ap.add_argument("--max-sidewall-try", type=int, default=4)
    ap.add_argument("--wall-s", type=float, default=14000.0)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    ACCEPTED.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    accepted_fps: set[str] = set()
    success_rows: list[dict] = []
    funnel_rows: list[dict] = []
    stage_fail = {
        "roots_built": 0,
        "genuine_loss": 0,
        "catch_search": 0,
        "sidewall_hit": 0,
        "stable_recatch": 0,
        "lift_ok": 0,
        "handoff_ok": 0,
        "success_new": 0,
        "replay_ok": 0,
    }

    # Resume: load already-accepted experts
    for p in sorted(ACCEPTED.glob("*__SUCCESS_NEW_expert.pkl")):
        blob = pickle.loads(p.read_bytes())
        fp = blob.get("fingerprint") or p.stem
        if fp in accepted_fps:
            continue
        accepted_fps.add(fp)
        success_rows.append({
            "root_id": blob.get("root_id") or p.name.split("__")[0],
            "fingerprint": fp,
            "SUCCESS_NEW": 1,
            "seed": int(blob.get("seed_source") is not None),
            "grasp_kind": blob.get("grasp_kind"),
            "approach": (blob.get("cfg") or {}).get("approach"),
            "speed": (blob.get("cfg") or {}).get("speed"),
            "resumed": 1,
        })
    print(f"=== resume {len(success_rows)} accepted ===", flush=True)
    if len(success_rows) < 3:
        print("=== seed golden Plan C successes ===", flush=True)
        seed_golden(accepted_fps, success_rows)
    stage_fail["success_new"] = len(success_rows)
    stage_fail["replay_ok"] = len(success_rows)
    write_csv(OUT / "ACCEPTED.csv", success_rows)
    write_json(OUT / "PROGRESS.json", {
        "n_success": len(success_rows),
        "target": args.target,
        "wall_s": 0.0,
        "milestones": {"10": len(success_rows) >= 10, "25": len(success_rows) >= 25, "50": len(success_rows) >= 50},
    })

    configs = diverse_impact_configs(args.max_configs, seed=args.seed)
    print(f"=== search {len(configs)} impact configs; target={args.target} ===", flush=True)

    for ci, cfg in enumerate(configs):
        if len(success_rows) >= args.target:
            break
        if time.time() - t0 > args.wall_s:
            print("wall-clock budget exhausted", flush=True)
            break

        sim, _ = make_ground_sim()
        root = build_physical_root(sim, cfg, max_s=1.2)
        stage_fail["roots_built"] += 1
        if root is None:
            funnel_rows.append({"config_id": cfg["config_id"], "stage": "build", "ok": 0})
            continue
        stage_fail["genuine_loss"] += 1
        fp = root["fingerprint"]
        if fp in accepted_fps:
            funnel_rows.append({"config_id": cfg["config_id"], "stage": "dup_fp", "ok": 0, "fingerprint": fp})
            continue

        rid = cfg["config_id"]
        print(
            f"[{len(success_rows)}/{args.target}] {rid} slip={root['slip_s']:.3f} "
            f"ap={cfg['approach']} sp={cfg['speed']} ax={cfg['axial_offset']}",
            flush=True,
        )
        (RAW / f"{rid}__root.pkl").write_bytes(pickle.dumps(root))

        stage_fail["catch_search"] += 1
        catch = search_catch(
            sim, root["snap"], root["ref"], rng=rng, max_rollouts=args.rollouts_per_root
        )
        funnel = catch["funnel"]
        sidewall = catch["sidewall"]
        if sidewall:
            stage_fail["sidewall_hit"] += 1
        row = {
            "config_id": rid,
            "fingerprint": fp,
            "slip_s": root["slip_s"],
            "n_sidewall": len(sidewall),
            "best_kind": None if catch["best"] is None else catch["best"].get("grasp_kind"),
            "best_theta": None if catch["best"] is None else catch["best"].get("e_theta_end"),
            "best_score": None if catch["best"] is None else catch["best"].get("score"),
            **{f"f_{k}": v for k, v in funnel.items()},
            "SUCCESS_NEW": 0,
        }

        accepted_here = False
        for hi, hit in enumerate(sidewall[: args.max_sidewall_try]):
            if float(hit.get("max_consec_bil") or 0) >= 0.10:
                stage_fail["stable_recatch"] += 1
            val = lift_and_validate(sim, root["snap"], root["ref"], hit)
            if val.get("stage") == "lift" and val.get("reason") != "LOST_LIFT":
                pass
            if val.get("lift", {}).get("reached") or (val.get("m_elev") or {}).get("obj_z", 0) > 0.4:
                stage_fail["lift_ok"] += 1
            if val.get("SUCCESS_NEW"):
                stage_fail["handoff_ok"] += 1
                stage_fail["success_new"] += 1
                rep = exact_replay_3of3(
                    root["snap"], root["ref"], val["prefix_actions"], val["elev_snap"]
                )
                if rep["ok"]:
                    stage_fail["replay_ok"] += 1
                    expert = pack_expert(
                        root_id=rid,
                        fingerprint=fp,
                        cfg=cfg,
                        snap=root["snap"],
                        ref=root["ref"],
                        validated=val,
                        events=root["events"],
                    )
                    expert["exact_replay"] = rep
                    expert["diversity"] = diversity_row(cfg, val, root["events"])
                    # Filename includes fingerprint so independent roots never overwrite.
                    out_name = f"{rid}__{fp[:10]}__SUCCESS_NEW_expert.pkl"
                    (ACCEPTED / out_name).write_bytes(pickle.dumps(expert))
                    # Remove legacy same-root overwrite if present
                    legacy = ACCEPTED / f"{rid}__SUCCESS_NEW_expert.pkl"
                    if legacy.exists() and legacy.name != out_name:
                        try:
                            old = pickle.loads(legacy.read_bytes())
                            if old.get("fingerprint") and old.get("fingerprint") != fp:
                                renamed = ACCEPTED / f"{rid}__{str(old['fingerprint'])[:10]}__SUCCESS_NEW_expert.pkl"
                                if not renamed.exists():
                                    legacy.replace(renamed)
                                else:
                                    legacy.unlink()
                            elif old.get("fingerprint") == fp:
                                legacy.unlink()
                        except Exception:
                            pass
                    accepted_fps.add(fp)
                    success_rows.append({
                        "root_id": rid,
                        "fingerprint": fp,
                        "SUCCESS_NEW": 1,
                        "seed": 0,
                        "grasp_kind": expert.get("grasp_kind"),
                        "approach": cfg.get("approach"),
                        "speed": cfg.get("speed"),
                        "e_theta_elev": (val.get("m_elev") or {}).get("e_theta_deg"),
                        "e_pos_elev_mm": (val.get("m_elev") or {}).get("e_pos_mm"),
                        "hit_i": hi,
                        "score": hit.get("score"),
                    })
                    row["SUCCESS_NEW"] = 1
                    row["accepted_hit"] = hi
                    accepted_here = True
                    print(
                        f"  ACCEPTED {rid} kind={expert.get('grasp_kind')} "
                        f"th={(val.get('m_elev') or {}).get('e_theta_deg'):.1f} "
                        f"n={len(success_rows)}",
                        flush=True,
                    )
                    write_csv(OUT / "ACCEPTED.csv", success_rows)
                    break
                else:
                    row["replay_fail"] = rep
                    print(f"  SUCCESS flag but replay fail elev={rep['elev_wins']}/3 full={rep['full_ok']}", flush=True)
            else:
                row[f"try{hi}_reason"] = val.get("reason")
                row[f"try{hi}_stage"] = val.get("stage")
        if not accepted_here and catch["best"] is not None:
            # diagnose limiting stage
            b = catch["best"]
            if b.get("grasp_kind") == "end_face_pinch":
                row["limit"] = "end_face_geometry"
            elif not sidewall:
                row["limit"] = "no_sidewall_catch"
            else:
                row["limit"] = "lift_or_handoff"
        funnel_rows.append(row)
        append_csv(OUT / "FUNNEL.csv", row)

        write_json(OUT / "PROGRESS.json", {
            "n_success": len(success_rows),
            "target": args.target,
            "configs_tried": ci + 1,
            "wall_s": time.time() - t0,
            "stage_fail": stage_fail,
            "milestones": {
                "10": len(success_rows) >= 10,
                "25": len(success_rows) >= 25,
                "50": len(success_rows) >= 50,
            },
        })
        if len(success_rows) in (10, 25, 50):
            print(f"*** MILESTONE {len(success_rows)} ***", flush=True)

    # Diversity summary
    div_rows = []
    for p in sorted(ACCEPTED.glob("*__SUCCESS_NEW_expert.pkl")):
        blob = pickle.loads(p.read_bytes())
        d = blob.get("diversity") or {}
        cfg = blob.get("cfg") or {}
        div_rows.append({
            "root_id": blob.get("root_id"),
            "fingerprint": blob.get("fingerprint"),
            "grasp_kind": blob.get("grasp_kind"),
            "approach": d.get("approach", cfg.get("approach")),
            "speed": d.get("speed", cfg.get("speed")),
            "axial_offset": d.get("axial_offset", cfg.get("axial_offset")),
            "tilt_deg": d.get("tilt_deg", cfg.get("tilt_deg")),
            "offset": d.get("offset", cfg.get("offset")),
            "slip_s": d.get("slip_s"),
            "vrel_recatch": d.get("vrel_recatch"),
            "detach_s": d.get("detach_s"),
            "contact_s_mean": d.get("contact_s_mean"),
            "e_theta_elev": d.get("e_theta_elev"),
            "e_pos_elev_mm": d.get("e_pos_elev_mm"),
            "n_prefix": d.get("n_prefix"),
            "seed": int(blob.get("seed_source") is not None),
        })
    write_csv(OUT / "DIVERSITY.csv", div_rows)
    write_json(OUT / "STAGE_FUNNEL.json", stage_fail)
    write_csv(OUT / "ACCEPTED.csv", success_rows)

    n = len(success_rows)
    bottleneck = "none"
    if n < args.target:
        # Infer bottleneck from funnel
        if stage_fail["genuine_loss"] < max(1, stage_fail["roots_built"] // 5):
            bottleneck = "physical_root_generation_few_genuine_losses"
        elif stage_fail["sidewall_hit"] == 0:
            bottleneck = "sidewall_recatch_search"
        elif stage_fail["lift_ok"] < stage_fail["sidewall_hit"]:
            bottleneck = "lift_to_handoff_height"
        elif stage_fail["handoff_ok"] < stage_fail["lift_ok"]:
            bottleneck = "handoff_or_nominal_SUCCESS_NEW"
        elif stage_fail["replay_ok"] < stage_fail["handoff_ok"]:
            bottleneck = "exact_replay_3of3"
        else:
            bottleneck = "wall_clock_or_config_budget"

    report = [
        "# Plan C scale-50 report",
        "",
        f"- Independent SUCCESS_NEW (replay-verified seeds + new): **{n}** / target {args.target}",
        f"- Wall time: {time.time() - t0:.1f}s",
        f"- Bottleneck (if < target): `{bottleneck}`",
        "",
        "## Funnel",
        "",
        "```",
        json.dumps(stage_fail, indent=2),
        "```",
        "",
        "## Milestones",
        "",
        f"- 10: {'yes' if n >= 10 else 'no'}",
        f"- 25: {'yes' if n >= 25 else 'no'}",
        f"- 50: {'yes' if n >= 50 else 'no'}",
        "",
        "Old end-face `gbi_014` is not counted. Golden `gbi_010`/`gbi_022`/`gbi_014` sidewall are seeds.",
        "",
        "Experts: `raw/accepted/*__SUCCESS_NEW_expert.pkl`",
        "Diversity: `DIVERSITY.csv`",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(OUT / "SUMMARY.json", {
        "n_success": n,
        "target": args.target,
        "bottleneck": bottleneck,
        "stage_fail": stage_fail,
        "wall_s": time.time() - t0,
        "accepted_dir": str(ACCEPTED.relative_to(ROOT)).replace("\\", "/"),
    })
    print(f"DONE n_success={n} bottleneck={bottleneck}", flush=True)


if __name__ == "__main__":
    main()
