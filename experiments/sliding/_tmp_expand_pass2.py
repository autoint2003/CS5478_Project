"""Pass-2: resume unified expansion from existing CANONICAL_MANIFEST.

Counts all SUCCESS_NEW roots. Searches remaining historical STABLE_POSE_ERROR
and additional ball impacts until stage target or budget exhausted.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Reuse expand module helpers
from scripts.expand_k49_unified_dataset import (
    CAPTURE_SEED,
    CONFIG_ID,
    HARD_SNAPS,
    HRV1_SNAPS,
    OUT,
    SFT,
    SEED,
    canonical_root,
    classify_ic_regime,
    classify_stability,
    difficulty_stratum,
    draw_lateral_spec,
    ic_state,
    load_seed_library,
    save_new_expert,
    search_ic,
    write_csv,
    write_json,
)
from data_generation.datasets.ball_impact import run_episode
from data_generation.datasets.recovery_rebuild import immediate_handoff
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    evaluate_full,
    mapping_checksum,
)
from training.action_vocab_k49 import assert_k48_prefix_identical, build_k49_table, describe_k49_config
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=75)
    ap.add_argument("--max-hist", type=int, default=100)
    ap.add_argument("--max-ball-gen", type=int, default=36)
    ap.add_argument("--max-ball-search", type=int, default=20)
    ap.add_argument("--budget-a", type=int, default=28)
    ap.add_argument("--budget-b", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0w = time.time()
    rng = np.random.default_rng(SEED + 99)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    canon_path = OUT / "CANONICAL_MANIFEST.csv"
    canonical = list(csv.DictReader(canon_path.open(encoding="utf-8")))
    solved = {r["root_episode"] for r in canonical}
    print(f"RESUME roots={len(solved)} traj={len(canonical)} target={args.target}", flush=True)

    # seed library from existing tokens
    seed_seqs = []
    for r in canonical:
        if r.get("tokens"):
            try:
                seed_seqs.append(json.loads(r["tokens"]))
            except Exception:
                pass
    seeds = load_seed_library(seed_seqs)

    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    new_experts = []
    strata_rows = []
    baseline_rows = []
    search_rows = []
    coverage = []
    new_roots_log = []
    n_rollouts = 0

    def n_roots():
        return len(solved)

    # historical candidates not solved
    hist = []
    for pool, folder, pat in (
        ("hard_recovery_v1", HRV1_SNAPS, "hrv1_*.pkl"),
        ("hard_ic_mining", HARD_SNAPS, "hard_*.pkl"),
    ):
        if not folder.exists():
            continue
        for sp in sorted(folder.glob(pat)):
            if sp.stem in solved:
                continue
            hist.append((pool, sp.stem, sp))
    rng.shuffle(hist)
    # prefer likely stable pose: filter after classify
    hist = hist[: args.max_hist]

    for ih, (pool, ic_id, sp) in enumerate(hist):
        if n_roots() >= args.target:
            break
        try:
            blob = pickle.loads(sp.read_bytes())
        except Exception:
            continue
        if "capture" not in blob or "ref" not in blob:
            continue
        st = ic_state(env, blob)
        klass = classify_stability(env, table, book48, blob, st)
        n_rollouts += 1
        strata = difficulty_stratum(st, klass)
        if klass in ("STABLE_CORRECT", "LOST_GRASP"):
            continue
        if klass == "UNSTABLE_CONTACT" and strata == "HARD" and n_roots() < 60:
            continue
        regimes = classify_ic_regime(blob, st) if isinstance(blob.get("evidence"), dict) else []
        strata_rows.append({
            "ic_id": ic_id, "root_episode": ic_id, "pool": pool,
            "stability_class": klass, "difficulty": strata,
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
            "pass": 2,
        })
        res_a = evaluate_full(env, table, book48, blob, [], record_obs=False); n_rollouts += 1
        res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False); n_rollouts += 1
        res_c = evaluate_full(env, table, book48, blob, [30] + [0] * 8, record_obs=False); n_rollouts += 1
        simple_ok = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        baseline_rows.append({
            "ic_id": ic_id, "pool": pool, "klass": klass,
            "A_NEW": int(res_a["SUCCESS_NEW"]), "B_NEW": int(res_b["SUCCESS_NEW"]),
            "C_NEW": int(res_c["SUCCESS_NEW"]), "simple_solves": int(simple_ok), "pass": 2,
        })
        print(f"[P2H {ih+1}/{len(hist)}] {ic_id} {klass}/{strata} "
              f"th={st['e_theta_deg']:.1f} pos={st['e_pos_mm']:.1f} roots={n_roots()}/{args.target}",
              flush=True)
        ba = args.budget_a + (8 if klass == "STABLE_POSE_ERROR" else 0)
        best, trials, n_try = search_ic(
            env, table, book48, blob, rng, seeds, klass, budget_a=ba, budget_b=args.budget_b,
        )
        n_rollouts += n_try
        for t in trials:
            search_rows.append({"ic_id": ic_id, "pass": 2, **t})
        if best is None:
            coverage.append({"ic_id": ic_id, "full": 0, "pass": 2})
            continue
        row = save_new_expert(
            env, table, book48, blob, ic_id, ic_id, best["tag"], best["tokens"],
            st, klass, strata, simple_ok, pool, mapping_meta, checksum, sp, regimes,
        )
        n_rollouts += 1
        if row is None:
            continue
        row["counts_as_new_root"] = 1
        row["pass"] = 2
        solved.add(ic_id)
        canonical.append(row)
        new_experts.append(row)
        # refresh seeds
        seeds = load_seed_library(seed_seqs + [json.loads(row["tokens"])])
        seed_seqs.append(json.loads(row["tokens"]))
        print(f"  FULL newly={row['newly_vs_simple']} roots={n_roots()}", flush=True)
        coverage.append({"ic_id": ic_id, "full": 1, "pass": 2})
        if (ih + 1) % 8 == 0:
            write_csv(OUT / "CANONICAL_MANIFEST.csv", canonical)

    # ball phase if still short
    if n_roots() < args.target:
        print("=== P2 ball ===", flush=True)
        ball = []
        for i in range(args.max_ball_gen):
            if n_roots() >= args.target:
                break
            spec = draw_lateral_spec(5000 + i)
            eid = f"ubimp2_{i:04d}"
            spec["attempt_id"] = eid
            evidence, blob = run_episode(env, spec)
            log = {"root_id": eid, "category": evidence.get("category"), "pass": 2}
            if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
                new_roots_log.append(log)
                continue
            nom = immediate_handoff(env, blob)
            if nom["nominal_class"] != "RECOVERY_CANDIDATE":
                new_roots_log.append(log)
                continue
            path = OUT / "snapshots" / f"{eid}.pkl"
            path.write_bytes(pickle.dumps(blob))
            st = ic_state(env, blob)
            regimes = classify_ic_regime(blob, st)
            log.update({"accepted": 1, "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"]})
            new_roots_log.append(log)
            ball.append({"ic_id": eid, "blob": blob, "st": st, "regimes": regimes, "snap": path})
        ball = sorted(ball, key=lambda r: (int(r["st"]["bilateral"]), -r["st"]["v_rel_norm"]), reverse=True)
        for br in ball[: args.max_ball_search]:
            if n_roots() >= args.target:
                break
            ic_id, blob, st = br["ic_id"], br["blob"], br["st"]
            klass = classify_stability(env, table, book48, blob, st)
            n_rollouts += 1
            strata = difficulty_stratum(st, klass)
            if klass == "LOST_GRASP" and n_roots() < 70:
                continue
            res_a = evaluate_full(env, table, book48, blob, [], record_obs=False); n_rollouts += 1
            res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False); n_rollouts += 1
            simple_ok = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"]
            print(f"[P2B] {ic_id} {klass} roots={n_roots()}/{args.target}", flush=True)
            best, trials, n_try = search_ic(
                env, table, book48, blob, rng, seeds, klass,
                budget_a=args.budget_a + 6, budget_b=args.budget_b,
            )
            n_rollouts += n_try
            if best is None:
                continue
            row = save_new_expert(
                env, table, book48, blob, ic_id, ic_id, best["tag"], best["tokens"],
                st, klass, strata, simple_ok, "generated_ball_p2", mapping_meta, checksum,
                br["snap"], br["regimes"],
            )
            n_rollouts += 1
            if row is None:
                continue
            row["counts_as_new_root"] = 1
            row["pass"] = 2
            solved.add(ic_id)
            canonical.append(row)
            new_experts.append(row)
            print(f"  FULL roots={n_roots()}", flush=True)

    write_csv(OUT / "CANONICAL_MANIFEST.csv", canonical)
    # merge pass2 logs
    prev_exp = list(csv.DictReader((OUT / "VERIFIED_FULL_EXPERTS.csv").open(encoding="utf-8")))
    all_exp = prev_exp + new_experts
    write_csv(OUT / "VERIFIED_FULL_EXPERTS.csv", all_exp)
    if strata_rows:
        prev_s = list(csv.DictReader((OUT / "IC_DIFFICULTY_STRATA.csv").open(encoding="utf-8"))) if (OUT / "IC_DIFFICULTY_STRATA.csv").exists() else []
        write_csv(OUT / "IC_DIFFICULTY_STRATA.csv", prev_s + strata_rows)
    if baseline_rows:
        prev_b = list(csv.DictReader((OUT / "MATCHED_BASELINES.csv").open(encoding="utf-8"))) if (OUT / "MATCHED_BASELINES.csv").exists() else []
        write_csv(OUT / "MATCHED_BASELINES.csv", prev_b + baseline_rows)
    if new_roots_log:
        prev_n = list(csv.DictReader((OUT / "NEW_PHYSICAL_ROOTS.csv").open(encoding="utf-8"))) if (OUT / "NEW_PHYSICAL_ROOTS.csv").exists() else []
        write_csv(OUT / "NEW_PHYSICAL_ROOTS.csv", prev_n + new_roots_log)
    if coverage:
        prev_c = list(csv.DictReader((OUT / "ROOT_COVERAGE.csv").open(encoding="utf-8"))) if (OUT / "ROOT_COVERAGE.csv").exists() else []
        write_csv(OUT / "ROOT_COVERAGE.csv", prev_c + coverage)

    # rebuild splits + answers
    roots = sorted(solved)
    rng2 = np.random.default_rng(SEED + 7)
    order = list(roots)
    rng2.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    test_set, val_set = set(order[:n_test]), set(order[n_test:n_test + n_val])
    split_rows = []
    for r in canonical:
        root = r["root_episode"]
        sp = "test" if root in test_set else ("validation" if root in val_set else "train")
        split_rows.append({
            "ic_id": r["ic_id"], "root_episode": root, "split": sp,
            "source_dataset": r.get("source_dataset") or r.get("source"),
            "provenance": r.get("provenance"),
        })
    write_csv(OUT / "TRAIN_VAL_TEST_SPLITS.csv", split_rows)

    audit = json.loads((OUT / "RAW" / "baseline_audit_summary.json").read_text(encoding="utf-8"))
    expansion = [r for r in canonical if r.get("provenance") == "expansion_search"]
    ball_roots = {r["root_episode"] for r in canonical
                  if str(r.get("source") or r.get("source_dataset") or "").find("ball") >= 0
                  or str(r["root_episode"]).startswith(("bimp_", "ubimp"))}
    newly = [r for r in expansion if int(float(r.get("newly_vs_simple") or 0)) == 1]
    stable = [r for r in expansion if r.get("stability_class") == "STABLE_POSE_ERROR"]
    lengths = [int(float(r["n_tokens"])) for r in canonical if r.get("n_tokens")]
    handoff_s = [0.02 * x for x in lengths]

    answers = {
        "Q1_n_verified_traj_before": audit["n_source_SUCCESS_NEW_rows"],
        "Q2_n_unique_ics_before": audit["n_unique_ics"],
        "Q2_n_unique_roots_before": audit["n_unique_roots"],
        "Q3_n_new_physical_roots_generated": sum(1 for r in csv.DictReader((OUT/"NEW_PHYSICAL_ROOTS.csv").open(encoding="utf-8")) if str(r.get("accepted")) == "1"),
        "Q4_n_new_independent_roots_SUCCESS_NEW": len(solved) - audit["n_unique_roots"],
        "Q5_n_unique_successful_roots_after": len(solved),
        "Q5_n_unique_ics_after": len({r["ic_id"] for r in canonical}),
        "Q5_n_trajectories_after": len(canonical),
        "Q6_n_newly_vs_simple": len(newly),
        "Q7_n_recatch_plus_slide_pivot_expansion": sum(1 for r in expansion if int(float(r.get("recatch_plus_manip") or 0)) == 1),
        "Q8_n_ball_impact_roots": len(ball_roots),
        "Q9_n_near_horizontal_expansion": sum(1 for r in expansion if "NEAR_HORIZONTAL" in str(r.get("regimes", ""))),
        "Q10_seq_len_mean": float(np.mean(lengths)) if lengths else None,
        "Q10_seq_len_median": float(np.median(lengths)) if lengths else None,
        "Q10_handoff_s_mean": float(np.mean(handoff_s)) if handoff_s else None,
        "Q10_handoff_s_median": float(np.median(handoff_s)) if handoff_s else None,
        "Q11_stable_pose_error_full": len(stable),
        "Q11_source_distribution_all": dict(Counter((r.get("source_dataset") or r.get("source") or r.get("provenance")) for r in canonical)),
        "Q12_underrepresented": [],
        "Q13_suitable_for_SFT": len(solved) >= 50,
        "pass2_new_experts": len(new_experts),
        "stage_target": args.target,
        "stage_reached": len(solved) >= args.target,
        "wall_s_pass2": time.time() - t0w,
        "n_rollouts_pass2": n_rollouts,
        "unique_roots_list": sorted(solved),
    }
    under = []
    if answers["Q8_n_ball_impact_roots"] < 12:
        under.append("ball_impact_roots")
    if answers["Q7_n_recatch_plus_slide_pivot_expansion"] < 8:
        under.append("recatch_plus_manipulation")
    if len(solved) < 100:
        under.append(f"below_100_roots_have_{len(solved)}")
    answers["Q12_underrepresented"] = under
    answers["Q13_note"] = (
        f"Dedup roots={len(solved)} (target {args.target}). "
        f"Prior={audit['n_unique_roots']}; net new={len(solved)-audit['n_unique_roots']}. "
        + ("Ready for initial unified SFT with root-grouped splits."
           if answers["Q13_suitable_for_SFT"] else
           "Continue expansion toward 75–100 roots.")
    )
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Unified K49 Dataset Expansion (incl. Pass-2)

**Config:** `{CONFIG_ID}`  
**Unique SUCCESS_NEW roots:** **{len(solved)}** / target {args.target}  
**Pass-2 added:** {len(new_experts)} experts, wall {answers['wall_s_pass2']:.1f}s

## Q1–Q13

1. Traj before: **{answers['Q1_n_verified_traj_before']}**
2. Unique ICs / roots before: **{answers['Q2_n_unique_ics_before']}** / **{answers['Q2_n_unique_roots_before']}**
3. New physical roots generated (accepted): **{answers['Q3_n_new_physical_roots_generated']}**
4. New independent SUCCESS_NEW roots: **{answers['Q4_n_new_independent_roots_SUCCESS_NEW']}**
5. Unique successful roots after: **{answers['Q5_n_unique_successful_roots_after']}**
6. Newly vs simple: **{answers['Q6_n_newly_vs_simple']}**
7. Recatch+manip expansion: **{answers['Q7_n_recatch_plus_slide_pivot_expansion']}**
8. Ball-impact roots: **{answers['Q8_n_ball_impact_roots']}**
9. Near-horizontal expansion: **{answers['Q9_n_near_horizontal_expansion']}**
10. Seq len mean/median: {answers['Q10_seq_len_mean']:.1f}/{answers['Q10_seq_len_median']:.1f}; HANDOFF mean/median: {answers['Q10_handoff_s_mean']:.2f}s/{answers['Q10_handoff_s_median']:.2f}s
11. STABLE_POSE_ERROR FULL: {answers['Q11_stable_pose_error_full']}; sources: {answers['Q11_source_distribution_all']}
12. Underrepresented: {answers['Q12_underrepresented']}
13. Suitable for SFT? **{answers['Q13_suitable_for_SFT']}** — {answers['Q13_note']}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    ready = f"""# Dataset Readiness — Unified K49 Expansion

## Verdict: {"YES" if answers["Q13_suitable_for_SFT"] else "NOT YET"}

- Unique SUCCESS_NEW roots: **{len(solved)}**
- Prior preserved: {audit['n_unique_roots']}
- Net new roots: {len(solved) - audit['n_unique_roots']}
- STABLE_POSE_ERROR expansion: {len(stable)}
- Ball-impact roots: {len(ball_roots)}

Use `CANONICAL_MANIFEST.csv` + `TRAIN_VAL_TEST_SPLITS.csv` (split by `root_episode`).
Do not start SFT/GRPO here.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    div = f"""# Diversity Report

Roots after pass-2: **{len(solved)}**
Sources: {answers['Q11_source_distribution_all']}
Ball-impact roots: {len(ball_roots)}
STABLE_POSE_ERROR expansion: {len(stable)}
Underrepresented: {under}
"""
    (OUT / "DIVERSITY_REPORT.md").write_text(div, encoding="utf-8")
    print("PASS2 DONE", json.dumps({k: v for k, v in answers.items() if k != "unique_roots_list"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
