#!/usr/bin/env python
"""Regenerate Plan B on GROUND_SCENE with explicit B0 / B1 / B2 comparison.

B0 = short release (1–3 Recovery tokens) → HANDOFF → nominal regrasp
B1 = short release → active recatch in Recovery → HANDOFF
B2 = active recatch → optional slide/pivot → HANDOFF

Staged search: cheap contact filters before capped full 12 s SUCCESS_NEW evals.
Does not overwrite data/unified_experts. No SFT/GRPO.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import (  # noqa: E402
    exact_replay_3x,
    load_vocab,
    run_b0,
    search_b1,
    search_b2,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "plan_b_b0_b1_b2_ground"
AUDIT_CSV = ROOT / "results/diagnostics/nontrivial_recovery_ic_mining/PREFIX_HANDOFF_AUDIT.csv"

# Historically verified short open/release prefixes (Recovery tokens only; HANDOFF=47 separate).
HISTORICAL_SHORT_PREFIXES = (
    [36],
    [36, 36],
    [44],
    [30],
    [39],
)


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


def snapshot_for(ic_id: str) -> Path | None:
    cands = [
        ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{ic_id}.pkl",
        ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{ic_id}.pkl",
        ROOT / f"results/expert_generation/k49_in_recovery_recatch/snapshots/{ic_id}.pkl",
    ]
    if "leftover_" in ic_id:
        hard = ic_id.split("leftover_")[-1]
        cands += [
            ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{hard}.pkl",
            ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{hard}.pkl",
        ]
    for p in cands:
        if p.exists():
            return p
    return None


def load_short_handoff_ics(*, max_prefix_len: int = 3):
    rows = list(csv.DictReader(AUDIT_CSV.open(encoding="utf-8")))
    out = []
    for r in rows:
        if r["success"] != "1":
            continue
        ic = r["ic_id"]
        toks = [int(t) for t in ast.literal_eval(r["tokens"])]
        # Drop trailing HANDOFF if present; keep Recovery-only short prefix.
        if toks and toks[-1] == 47:
            toks = toks[:-1]
        if len(toks) > int(max_prefix_len):
            continue
        if len(toks) < 1:
            continue
        snap = snapshot_for(ic)
        if snap is None:
            continue
        out.append({"ic_id": ic, "prefix": toks, "snapshot": str(snap)})
    by = {}
    for e in out:
        ic = e["ic_id"]
        if ic not in by or len(e["prefix"]) < len(by[ic]["prefix"]):
            by[ic] = e
    return [by[k] for k in sorted(by)]


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def slim(out: dict) -> dict:
    keep = {k: v for k, v in out.items() if k not in ("trace_recovery",)}
    if "trace_recovery" in out:
        tr = out["trace_recovery"]
        keep["trace_n"] = len(tr)
        keep["trace_bilateral_frac"] = (
            float(np.mean([1.0 if r["bilateral"] else 0.0 for r in tr])) if tr else None
        )
    return keep


def b0_short_prefix_screen(env, blob, table, book48, seed_prefix: list[int]) -> dict:
    """Stage A: try historically short prefixes + IC seed; keep best SUCCESS_NEW / S12."""
    import time

    t0 = time.time()
    prefixes = []
    for p in HISTORICAL_SHORT_PREFIXES:
        if p not in prefixes:
            prefixes.append(list(p))
    if seed_prefix and seed_prefix not in prefixes and 1 <= len(seed_prefix) <= 3:
        prefixes.insert(0, list(seed_prefix))
    best = None
    rows = []
    for pref in prefixes:
        print(f"    B0 prefix={pref} ...", flush=True)
        out = run_b0(env, blob, table, book48, pref)
        rows.append(
            {
                "prefix": list(pref),
                "SUCCESS_NEW": bool(out.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(out.get("SUCCESS_12S")),
                "recatch_controller": out.get("recatch_controller"),
                "e_theta": (out.get("handoff_meta") or {}).get("pre_theta"),
                "e_pos_mm": (out.get("handoff_meta") or {}).get("pre_pos_mm"),
                "reason": out.get("terminal_reason"),
            }
        )
        print(
            f"      → NEW={out.get('SUCCESS_NEW')} S12={out.get('SUCCESS_12S')} "
            f"ctrl={out.get('recatch_controller')} "
            f"θ={(out.get('handoff_meta') or {}).get('pre_theta')} "
            f"pos={(out.get('handoff_meta') or {}).get('pre_pos_mm')}",
            flush=True,
        )
        if best is None:
            best = out
        else:
            score = (
                int(bool(out.get("SUCCESS_NEW"))) * 4
                + int(bool(out.get("SUCCESS_12S"))) * 2
                + int(out.get("recatch_controller") == "nominal")
            )
            best_score = (
                int(bool(best.get("SUCCESS_NEW"))) * 4
                + int(bool(best.get("SUCCESS_12S"))) * 2
                + int(best.get("recatch_controller") == "nominal")
            )
            if score > best_score or (
                score == best_score and len(pref) < len(best.get("tokens") or pref)
            ):
                best = out
    best = best or {
        "branch": "B0",
        "SUCCESS_NEW": False,
        "SUCCESS_12S": False,
        "terminal_reason": "NO_PREFIX",
        "tokens": list(seed_prefix),
    }
    best["b0_screen"] = {
        "prefixes_tried": prefixes,
        "rows": rows,
        "wall_s": round(time.time() - t0, 2),
        "n_full_nominal": len(prefixes),
    }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ics", type=int, default=22)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--budget-b1", type=int, default=28)
    ap.add_argument("--budget-b2-regrasp", type=int, default=20)
    ap.add_argument("--budget-b2-pose", type=int, default=8)
    ap.add_argument("--max-full-nominal-b1", type=int, default=6)
    ap.add_argument("--max-full-nominal-b2", type=int, default=8)
    ap.add_argument("--max-prefix-len", type=int, default=3)
    ap.add_argument("--skip-b2", action="store_true")
    ap.add_argument("--ic-id", default="")
    ap.add_argument(
        "--fast-diag",
        action="store_true",
        help="2-IC diagnostic defaults: tiny budgets, skip B2, max instrumentation",
    )
    args = ap.parse_args()

    if args.fast_diag:
        args.n_ics = min(args.n_ics, 2)
        args.budget_b1 = min(args.budget_b1, 8)
        args.budget_b2_regrasp = min(args.budget_b2_regrasp, 6)
        args.budget_b2_pose = min(args.budget_b2_pose, 3)
        args.max_full_nominal_b1 = min(args.max_full_nominal_b1, 3)
        args.max_full_nominal_b2 = min(args.max_full_nominal_b2, 3)
        print(
            "FAST_DIAG mode: "
            f"n_ics={args.n_ics} b1={args.budget_b1} "
            f"b2=({args.budget_b2_regrasp},{args.budget_b2_pose}) "
            f"full_nom=({args.max_full_nominal_b1},{args.max_full_nominal_b2})",
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_ic").mkdir(parents=True, exist_ok=True)
    (OUT / "experts").mkdir(parents=True, exist_ok=True)

    table, book48 = load_vocab()
    rng = np.random.default_rng(args.seed)
    ics = load_short_handoff_ics(max_prefix_len=args.max_prefix_len)
    if args.ic_id:
        ics = [e for e in ics if e["ic_id"] == args.ic_id]
    ics = ics[: args.n_ics]
    print(
        f"Plan B B0/B1/B2 on GROUND_SCENE — {len(ics)} ICs | "
        f"max_prefix_len={args.max_prefix_len} | "
        f"historical_short={HISTORICAL_SHORT_PREFIXES}",
        flush=True,
    )

    env = make_env()
    rows = []
    experts_b1 = []
    experts_b2 = []
    t_wall0 = time.time()

    for i, ic in enumerate(ics):
        t0 = time.time()
        ic_id = ic["ic_id"]
        prefix = list(ic["prefix"])
        t_init0 = time.time()
        blob = pickle.loads(Path(ic["snapshot"]).read_bytes())
        t_init = round(time.time() - t_init0, 3)
        if "capture" not in blob or "ref" not in blob:
            print(f"[{i+1}] {ic_id} SKIP no capture/ref", flush=True)
            continue
        print(f"\n[{i+1}/{len(ics)}] {ic_id} seed_prefix={prefix} init={t_init}s", flush=True)

        # --- Stage A: short-prefix B0 screen ---
        print("  Stage A: short-prefix B0 screen", flush=True)
        b0 = b0_short_prefix_screen(env, blob, table, book48, prefix)
        # Prefer IC seed prefix for B1/B2 seeding when length ok; else best historical short open.
        seed_for_search = prefix if 1 <= len(prefix) <= args.max_prefix_len else [36, 36]
        print(
            f"  B0 NEW={b0.get('SUCCESS_NEW')} S12={b0.get('SUCCESS_12S')} "
            f"ctrl={b0.get('recatch_controller')} tokens={b0.get('tokens')} "
            f"wall={b0.get('b0_screen', {}).get('wall_s')}s "
            f"full_nom={b0.get('b0_screen', {}).get('n_full_nominal')}",
            flush=True,
        )

        # --- Stage B/D: B1 active recatch ---
        print(
            f"  Stage B/D: B1 search seed={seed_for_search} "
            f"budget={args.budget_b1} max_full={args.max_full_nominal_b1}",
            flush=True,
        )
        b1 = search_b1(
            env,
            blob,
            table,
            book48,
            seed_for_search,
            rng,
            budget=args.budget_b1,
            max_full_nominal=args.max_full_nominal_b1,
        )
        st1 = b1.get("search_stats") or {}
        print(
            f"  B1 NEW={b1.get('SUCCESS_NEW')} S12={b1.get('SUCCESS_12S')} "
            f"active={b1.get('active_recatch_before_handoff')} ctrl={b1.get('recatch_controller')} "
            f"t_loss={b1.get('t_grasp_loss')} t_recatch={b1.get('t_active_recatch')} "
            f"t_handoff={b1.get('t_handoff')} "
            f"cands={st1.get('n_candidates')} release_ok={st1.get('n_release_ok')} "
            f"active_n={st1.get('n_active_recatch')} full_nom={st1.get('n_full_nominal')} "
            f"screen_rej={st1.get('n_screen_reject')} "
            f"best_θ={st1.get('best_theta')} best_pos={st1.get('best_pos_mm')} "
            f"t_search={st1.get('t_search_s')}s t_nominal={st1.get('t_nominal_s')}s "
            f"wall={st1.get('wall_s')}s",
            flush=True,
        )

        # --- Stage C/D: B2 optional correction ---
        b2 = {"branch": "B2", "SUCCESS_NEW": False, "skipped": True}
        if not args.skip_b2:
            if b1.get("SUCCESS_NEW"):
                b2 = {
                    "branch": "B2",
                    "SUCCESS_NEW": False,
                    "note": "B1_already_SUCCESS_NEW_no_forced_correction",
                    "skipped": True,
                }
                print("  Stage C: B2 skipped (B1 already SUCCESS_NEW)", flush=True)
            else:
                print(
                    f"  Stage C/D: B2 search budget_regrasp={args.budget_b2_regrasp} "
                    f"pose={args.budget_b2_pose} max_full={args.max_full_nominal_b2}",
                    flush=True,
                )
                b2 = search_b2(
                    env,
                    blob,
                    table,
                    book48,
                    seed_for_search,
                    rng,
                    budget_regrasp=args.budget_b2_regrasp,
                    budget_pose=args.budget_b2_pose,
                    max_full_nominal=args.max_full_nominal_b2,
                )
                if b2.get("branch") == "B1" and b2.get("SUCCESS_NEW"):
                    b1 = b2
                    b2 = {
                        "branch": "B2",
                        "SUCCESS_NEW": False,
                        "note": "search_found_B1_no_correction_needed",
                        "skipped": True,
                    }
                st2 = b2.get("search_stats") or {}
                print(
                    f"  B2 NEW={b2.get('SUCCESS_NEW')} S12={b2.get('SUCCESS_12S')} "
                    f"corr={b2.get('correction') or b2.get('note')} fam={b2.get('pose_family')} "
                    f"active={b2.get('active_recatch_before_handoff')} "
                    f"cands={st2.get('n_candidates')} pose_tried={st2.get('n_pose_tried')} "
                    f"full_nom={st2.get('n_full_nominal')} screen_rej={st2.get('n_screen_reject')} "
                    f"fams_tried={st2.get('pose_families_tried')} "
                    f"fams_full={st2.get('pose_families_full_nominal')} "
                    f"best_θ={st2.get('best_theta')} best_pos={st2.get('best_pos_mm')} "
                    f"t_search={st2.get('t_search_s')}s t_nominal={st2.get('t_nominal_s')}s "
                    f"wall={st2.get('wall_s')}s",
                    flush=True,
                )

        replay = None
        accepted = None
        t_replay = 0.0
        if b2.get("SUCCESS_NEW") and b2.get("active_recatch_before_handoff"):
            print("  Stage D: exact_replay 3x for B2", flush=True)
            t_r0 = time.time()
            replay = exact_replay_3x(
                env,
                blob,
                table,
                book48,
                b2["tokens"],
                branch="B2",
                prefix_len=len(seed_for_search),
            )
            t_replay = round(time.time() - t_r0, 2)
            print(
                f"  replay {replay['n_pass']}/3 active_ok="
                f"{sum(1 for r in replay['rows'] if r.get('active_recatch_before_handoff'))} "
                f"wall={t_replay}s",
                flush=True,
            )
            if replay["ok"]:
                accepted = {"family": "B2", **slim(b2), "exact_replay": replay}
                experts_b2.append(accepted)
        elif b1.get("SUCCESS_NEW") and b1.get("active_recatch_before_handoff"):
            print("  Stage D: exact_replay 3x for B1", flush=True)
            t_r0 = time.time()
            replay = exact_replay_3x(
                env,
                blob,
                table,
                book48,
                b1["tokens"],
                branch="B1",
                prefix_len=len(seed_for_search),
            )
            t_replay = round(time.time() - t_r0, 2)
            print(
                f"  replay {replay['n_pass']}/3 active_ok="
                f"{sum(1 for r in replay['rows'] if r.get('active_recatch_before_handoff'))} "
                f"wall={t_replay}s",
                flush=True,
            )
            if replay["ok"]:
                accepted = {"family": "B1", **slim(b1), "exact_replay": replay}
                experts_b1.append(accepted)

        if accepted:
            (OUT / "experts" / f"{ic_id}__{accepted['family']}.json").write_text(
                json.dumps(jsafe(accepted), indent=2), encoding="utf-8"
            )
            print(f"  ACCEPTED {accepted['family']} 3/3", flush=True)

        wall = round(time.time() - t0, 2)
        row = {
            "ic_id": ic_id,
            "prefix": prefix,
            "seed_for_search": seed_for_search,
            "snapshot": ic["snapshot"],
            "B0": slim(b0),
            "B1": slim(b1),
            "B2": slim(b2),
            "accepted": None if accepted is None else accepted["family"],
            "exact_replay": replay,
            "wall_s": wall,
            "stage_times": {
                "init": t_init,
                "B0": (b0.get("b0_screen") or {}).get("wall_s"),
                "B1": (b1.get("search_stats") or {}).get("wall_s"),
                "B1_search": (b1.get("search_stats") or {}).get("t_search_s"),
                "B1_nominal": (b1.get("search_stats") or {}).get("t_nominal_s"),
                "B2": (b2.get("search_stats") or {}).get("wall_s"),
                "B2_search": (b2.get("search_stats") or {}).get("t_search_s"),
                "B2_nominal": (b2.get("search_stats") or {}).get("t_nominal_s"),
                "exact_replay": t_replay,
                "total": wall,
            },
        }
        rows.append(row)
        (OUT / "per_ic" / f"{ic_id}.json").write_text(json.dumps(jsafe(row), indent=2), encoding="utf-8")
        print(
            f"  IC done wall={wall}s total_elapsed={round(time.time() - t_wall0, 1)}s "
            f"accepted={accepted['family'] if accepted else None} "
            f"times(init/B0/B1/B2/replay)="
            f"{t_init}/{(b0.get('b0_screen') or {}).get('wall_s')}/"
            f"{(b1.get('search_stats') or {}).get('wall_s')}/"
            f"{(b2.get('search_stats') or {}).get('wall_s')}/{t_replay}",
            flush=True,
        )

    csv_path = OUT / "MATCHED_COMPARISON.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "ic_id",
                "prefix",
                "B0_SUCCESS_NEW",
                "B0_SUCCESS_12S",
                "B0_recatch_controller",
                "B0_t_grasp_loss",
                "B0_t_handoff",
                "B1_SUCCESS_NEW",
                "B1_SUCCESS_12S",
                "B1_recatch_controller",
                "B1_t_active_recatch",
                "B1_t_handoff",
                "B1_full_nominal",
                "B2_SUCCESS_NEW",
                "B2_correction",
                "B2_full_nominal",
                "accepted",
                "replay_3x",
                "wall_s",
                "B0_e_theta",
                "B0_e_pos",
                "B1_e_theta",
                "B1_e_pos",
                "ground_contact_any",
            ],
        )
        w.writeheader()
        for r in rows:
            b0, b1, b2 = r["B0"], r["B1"], r["B2"]
            w.writerow(
                {
                    "ic_id": r["ic_id"],
                    "prefix": json.dumps(r["prefix"]),
                    "B0_SUCCESS_NEW": int(bool(b0.get("SUCCESS_NEW"))),
                    "B0_SUCCESS_12S": int(bool(b0.get("SUCCESS_12S"))),
                    "B0_recatch_controller": b0.get("recatch_controller"),
                    "B0_t_grasp_loss": b0.get("t_grasp_loss"),
                    "B0_t_handoff": b0.get("t_handoff"),
                    "B1_SUCCESS_NEW": int(bool(b1.get("SUCCESS_NEW"))),
                    "B1_SUCCESS_12S": int(bool(b1.get("SUCCESS_12S"))),
                    "B1_recatch_controller": b1.get("recatch_controller"),
                    "B1_t_active_recatch": b1.get("t_active_recatch"),
                    "B1_t_handoff": b1.get("t_handoff"),
                    "B1_full_nominal": (b1.get("search_stats") or {}).get("n_full_nominal"),
                    "B2_SUCCESS_NEW": int(bool(b2.get("SUCCESS_NEW"))),
                    "B2_correction": b2.get("correction") or b2.get("note"),
                    "B2_full_nominal": (b2.get("search_stats") or {}).get("n_full_nominal"),
                    "accepted": r.get("accepted"),
                    "replay_3x": None
                    if r.get("exact_replay") is None
                    else f"{r['exact_replay']['n_pass']}/3",
                    "wall_s": r.get("wall_s"),
                    "B0_e_theta": (b0.get("post_nominal") or {}).get("e_theta_deg")
                    or (b0.get("handoff_meta") or {}).get("pre_theta"),
                    "B0_e_pos": (b0.get("post_nominal") or {}).get("e_pos_mm")
                    or (b0.get("handoff_meta") or {}).get("pre_pos_mm"),
                    "B1_e_theta": (b1.get("handoff_meta") or {}).get("pre_theta"),
                    "B1_e_pos": (b1.get("handoff_meta") or {}).get("pre_pos_mm"),
                    "ground_contact_any": int(
                        bool(b0.get("ground_contact") or b1.get("ground_contact") or b2.get("ground_contact"))
                    ),
                }
            )

    n_b0 = sum(1 for r in rows if r["B0"].get("SUCCESS_NEW"))
    n_b1_new = sum(
        1
        for r in rows
        if r["B1"].get("SUCCESS_NEW") and r["B1"].get("active_recatch_before_handoff")
    )
    n_b2_new = sum(
        1
        for r in rows
        if r["B2"].get("SUCCESS_NEW") and r["B2"].get("active_recatch_before_handoff")
    )
    n_b1 = len(experts_b1)
    n_b2 = len(experts_b2)
    n_active_b1 = sum(1 for r in rows if r["B1"].get("active_recatch_before_handoff"))
    n_active_b2 = sum(1 for r in rows if r["B2"].get("active_recatch_before_handoff"))
    b2_slide = sum(1 for r in rows if str(r["B2"].get("correction") or "").startswith("slide"))
    b2_pivot = sum(1 for r in rows if str(r["B2"].get("correction") or "").startswith("pivot"))
    b2_bothish = sum(
        1
        for r in rows
        if r["B2"].get("pose_family") == "other"
        or str(r["B2"].get("correction") or "").startswith("slide_or_pivot")
    )
    total_full_nom = 0
    for r in rows:
        total_full_nom += int((r["B0"].get("b0_screen") or {}).get("n_full_nominal") or 0)
        total_full_nom += int((r["B1"].get("search_stats") or {}).get("n_full_nominal") or 0)
        total_full_nom += int((r["B2"].get("search_stats") or {}).get("n_full_nominal") or 0)
        if r.get("exact_replay"):
            total_full_nom += int((r["exact_replay"] or {}).get("n") or 0)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": "GROUND_SCENE",
        "n_ics": len(rows),
        "fast_diag": bool(args.fast_diag),
        "budgets": {
            "b1": args.budget_b1,
            "b2_regrasp": args.budget_b2_regrasp,
            "b2_pose": args.budget_b2_pose,
            "max_full_nominal_b1": args.max_full_nominal_b1,
            "max_full_nominal_b2": args.max_full_nominal_b2,
            "max_prefix_len": args.max_prefix_len,
        },
        "B0_SUCCESS_NEW": n_b0,
        "B1_SUCCESS_NEW_active": n_b1_new,
        "B2_SUCCESS_NEW_active": n_b2_new,
        "B1_accepted_3x": n_b1,
        "B2_accepted_3x": n_b2,
        "genuine_active_recatch_B1": n_active_b1,
        "genuine_active_recatch_B2": n_active_b2,
        "B2_correction_slide": b2_slide,
        "B2_correction_pivot": b2_pivot,
        "B2_correction_mixed": b2_bothish,
        "total_full_nominal_evals": total_full_nom,
        "wall_s": round(time.time() - t_wall0, 1),
        "mean_ic_s": None if not rows else round(sum(r["wall_s"] for r in rows) / len(rows), 1),
        "B1_ids": [r["ic_id"] for r in rows if r.get("accepted") == "B1"],
        "B2_ids": [r["ic_id"] for r in rows if r.get("accepted") == "B2"],
        "stage_times": [r.get("stage_times") for r in rows],
    }
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    (OUT / "all_rows.json").write_text(json.dumps(jsafe(rows), indent=2), encoding="utf-8")

    report = [
        "# Plan B B0 / B1 / B2 on GROUND_SCENE",
        "",
        f"Created UTC: {summary['created_utc']}",
        f"fast_diag={args.fast_diag} wall_s={summary['wall_s']} mean_ic_s={summary['mean_ic_s']}",
        "",
        "## Branch definitions",
        "",
        "- **B0:** short release (1–3 tokens) → HANDOFF → nominal",
        "- **B1:** short release → **active recatch before HANDOFF** → HANDOFF",
        "- **B2:** active recatch → slide/pivot when direct HANDOFF fails SUCCESS_NEW",
        "",
        f"- ICs: {summary['n_ics']}",
        f"- B0 SUCCESS_NEW: {n_b0}",
        f"- B1 SUCCESS_NEW (active): {n_b1_new}",
        f"- B2 SUCCESS_NEW (active): {n_b2_new}",
        f"- B1 accepted 3/3: {n_b1}",
        f"- B2 accepted 3/3: {n_b2}",
        f"- Genuine active B1/B2 recatches: {n_active_b1} / {n_active_b2}",
        f"- B2 slide/pivot/mixed corrections: {b2_slide}/{b2_pivot}/{b2_bothish}",
        f"- Total full nominal evals (incl. replay): {total_full_nom}",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
