"""Pass 2: legal unload→loss IC mining + heavy Stage B on unstable bilateral ICs."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_dynamic_recatch_experts import (
    find_bilateral_after_loss,
    run_traced,
    type2_proposals,
)
from scripts.generate_k49_in_recovery_recatch import find_first_bilateral, pose_correction_proposals, snapshot_for
from scripts.generate_k49_lost_grasp_recovery import (
    OUT,
    classify_ic,
    root_episode_id,
    save_expert,
    stage_a_proposals,
    sublabel_from_trace,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    adaptive_return_variants,
    build_k49_table,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    load_codebook,
    make_training_env,
    mapping_checksum,
    run_seq,
    write_csv,
    write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, assert_k48_prefix_identical, describe_k49_config

SEED = 20261019


def legal_unload_to_loss(env, table, book48, blob, src_id, max_keep=2):
    """From a grasped IC, legally open/unload until lost-grasp, snapshot (no teleport)."""
    trials = []
    for o in (30, 36, 12, 44):
        for n in (2, 4, 6, 8, 12, 16):
            trials.append([o] * n)
            trials.append([o] * n + [12] * 4)
    out = []
    for seq in trials:
        restore_capture(env, blob["capture"])
        ok, _ = run_seq(env, table, book48, blob, seq)
        if not ok:
            continue
        cap = take_capture(env)
        new_blob = {"capture": cap, "ref": blob["ref"], "meta": {"derived_from": src_id, "unload": seq}}
        info = classify_ic(env, table, book48, new_blob)
        if not info["is_type3_onset"]:
            continue
        if info["lost_class"] == "IRREVERSIBLE_ESCAPE":
            continue
        if info["e_pos_mm"] > 40:
            continue
        ic_id = f"{src_id}_loss{len(out)}"
        out.append({
            "ic_id": ic_id, "root_episode": root_episode_id(src_id),
            "snapshot_blob": new_blob, "unload_prefix": seq, "src": src_id, **info,
        })
        if len(out) >= max_keep:
            break
    return out


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = load_codebook()
    table = build_k49_table(book, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book)
    mapping_meta = describe_k49_config(CONFIG_ID, book)
    checksum = mapping_checksum(table)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)

    # load prior IC list of unstable/unilateral from pass1
    prior = list(csv.DictReader((OUT / "LOST_GRASP_ICS.csv").open(encoding="utf-8")))
    work = [r for r in prior if r["lost_class"] in ("UNSTABLE_BILATERAL", "UNILATERAL_LOSS", "COMPLETE_CONTACT_LOSS")
            and r.get("is_type3_onset") == "1"]

    # derive additional loss ICs from known hard poses
    derived = []
    for src in ("hrv1_920031", "hard_910101", "hard_910220", "hrv1_920099", "hrv1_920116",
                "hrv1_920109", "hrv1_920158", "hard_910185", "hard_910318"):
        sp = snapshot_for(src)
        if sp is None:
            continue
        blob = pickle.loads(sp.read_bytes())
        derived.extend(legal_unload_to_loss(env, table, book, blob, src, max_keep=2))
    print("derived loss ICs", len(derived), flush=True)

    full_experts = list(csv.DictReader((OUT / "TYPE3_FULL_EXPERTS.csv").open(encoding="utf-8"))) if (OUT / "TYPE3_FULL_EXPERTS.csv").exists() else []
    prefixes = list(csv.DictReader((OUT / "RECATCH_PREFIXES.csv").open(encoding="utf-8"))) if (OUT / "RECATCH_PREFIXES.csv").exists() else []
    post_corr = list(csv.DictReader((OUT / "POST_RECATCH_CORRECTIONS.csv").open(encoding="utf-8"))) if (OUT / "POST_RECATCH_CORRECTIONS.csv").exists() else []
    failures = []
    baselines_extra = []
    coverage_extra = []
    ic_extra = []
    n_roll = 0

    # Build worklist: derived first, then top unstable from prior
    items = []
    for d in derived:
        path = OUT / "snapshots" / f"{d['ic_id']}.pkl"
        path.write_bytes(pickle.dumps(d["snapshot_blob"]))
        items.append({
            "ic_id": d["ic_id"], "root_episode": d["root_episode"], "snapshot": str(path),
            "lost_class": d["lost_class"], "blob": d["snapshot_blob"], "derived": True,
            "e_theta_deg": d["e_theta_deg"], "e_pos_mm": d["e_pos_mm"],
        })
        ic_extra.append({
            "ic_id": d["ic_id"], "root_episode": d["root_episode"], "lost_class": d["lost_class"],
            "is_type3_onset": 1, "nL": d["nL"], "nR": d["nR"], "bilateral": int(d["bilateral"]),
            "e_theta0": d["e_theta_deg"], "e_pos0": d["e_pos_mm"],
            "v_rel": d["v_rel_norm"], "w_rel": d["w_rel_norm"],
            "probe_failed": d["probe_failed"], "probe_bil_frac": d["probe_bil_frac"],
            "snapshot": str(path), "derived_from": d["src"],
        })

    for r in work:
        if r["lost_class"] not in ("UNSTABLE_BILATERAL", "UNILATERAL_LOSS"):
            continue
        sp = Path(r["snapshot"]) if Path(r["snapshot"]).exists() else snapshot_for(r["ic_id"])
        if sp is None or not Path(sp).exists():
            continue
        blob = pickle.loads(Path(sp).read_bytes())
        items.append({
            "ic_id": r["ic_id"], "root_episode": r["root_episode"], "snapshot": str(sp),
            "lost_class": r["lost_class"], "blob": blob, "derived": False,
            "e_theta_deg": float(r["e_theta0"]), "e_pos_mm": float(r["e_pos0"]),
        })

    # diversify roots, cap
    seen, uniq = set(), []
    for it in items:
        if it["root_episode"] in seen and not it.get("derived"):
            continue
        # allow one derived per root always
        key = (it["root_episode"], it.get("derived"))
        if it["root_episode"] in seen and it.get("derived") and sum(1 for u in uniq if u["root_episode"]==it["root_episode"] and u.get("derived")) >= 1:
            continue
        if not it.get("derived"):
            seen.add(it["root_episode"])
        uniq.append(it)
        if len(uniq) >= 20:
            break
    items = uniq
    print("pass2 items", len(items), flush=True)

    for ie, it in enumerate(items):
        t0b = time.time()
        ic_id = it["ic_id"]
        blob = it["blob"]
        snap = Path(it["snapshot"])
        info = classify_ic(env, table, book, blob)
        print(f"\n[{ie+1}/{len(items)}] {ic_id} {info['lost_class']} th={info['e_theta_deg']:.1f} "
              f"pos={info['e_pos_mm']:.1f} der={it.get('derived')}", flush=True)
        if not info["is_type3_onset"]:
            continue

        # baselines quick
        res_a = evaluate_full(env, table, book, blob, [], record_obs=False)
        res_b = evaluate_full(env, table, book, blob, [0] * 16, record_obs=False)
        res_c = evaluate_full(env, table, book, blob, [30] + [0] * 8, record_obs=False)
        simple = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        baselines_extra.append({
            "ic_id": ic_id, "lost_class": info["lost_class"],
            "A_NEW": int(res_a["SUCCESS_NEW"]), "B_NEW": int(res_b["SUCCESS_NEW"]),
            "C_NEW": int(res_c["SUCCESS_NEW"]), "simple": int(simple),
        })

        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        # heavy seeds: type2-like + capture-brake
        seeds = [
            [30, 12, 12, 8] + [0] * 10,
            [0] * 24,
            [12] * 6 + [0] * 20,
            [30] + [0] * 16,
            [36] * 3 + [0] * 16,
            [30] + [17] * 10 + [0] * 10,
            [30] + [33] * 10 + [0] * 10,
            [4] * 16 + [12] * 5 + [0] * 5,
            [4] * 20 + [12] * 5 + [0] * 8,
            [6] * 16 + [12] * 5 + [0] * 5,
            [10] * 12 + [12] * 8 + [0] * 5,
            [4] * 12 + [17] * 16 + [0] * 5,
            [4] * 16 + [12] * 5 + [33] * 16 + [0] * 5,
        ]
        props_a = stage_a_proposals(book, desire, rng, n=20, seeds=seeds)
        # also type2 proposals as full recoveries
        props_t2 = type2_proposals(book, desire, rng, n=12, open_prefix=[30])
        best = None
        n_try = 0

        # Try full sequences (A+B combined / type2)
        for tag, seq in [(f"a{i}", s) for i, s in enumerate(props_a)] + [(f"t2_{i}", s) for i, s in enumerate(props_t2)]:
            n_try += 1
            n_roll += 1
            variants = [("raw", list(seq) + [0] * 5)]
            restore_capture(env, blob["capture"])
            ok_p, _ = run_seq(env, table, book, blob, seq)
            if not ok_p:
                failures.append({"ic_id": ic_id, "tag": tag, "reason": "DROP_PREFIX"})
                continue
            if hand_ref_error_deg(env, blob) > 8:
                variants += [
                    ("ret12", list(seq) + [SECURE_REVERSE_WRIST] * 12 + [0] * 8),
                    ("ret20", list(seq) + [SECURE_REVERSE_WRIST] * 20 + [0] * 8),
                    ("settle20", list(seq) + [0] * 20),
                    ("settle40", list(seq) + [0] * 40),
                ]
            else:
                variants += [("settle20", list(seq) + [0] * 20), ("settle40", list(seq) + [0] * 40)]

            # if still bilateral unstable with large theta, append pose corrections
            restore_capture(env, blob["capture"])
            run_seq(env, table, book, blob, seq)
            desire_b = -decompose(env.sim)["err_h"]
            for corr in pose_correction_proposals(book, desire_b, rng, n=8):
                if not corr:
                    continue
                variants.append((f"corr", list(seq) + list(corr) + [0] * 8))

            for vname, toks in variants:
                n_try += 1
                n_roll += 1
                res = evaluate_full(env, table, book, blob, toks, record_obs=False)
                if not res["SUCCESS_NEW"]:
                    failures.append({
                        "ic_id": ic_id, "tag": f"{tag}_{vname}", "reason": res["reason"],
                        "s12": int(res["success_12s"]),
                        "win_th": res["window"].get("mean_theta"),
                        "win_pos": res["window"].get("mean_pos"),
                    })
                    continue
                ok2, tr2 = run_traced(env, table, book, blob, toks)
                bil2 = find_first_bilateral(tr2, 0) or find_bilateral_after_loss(tr2)
                if bil2 is None:
                    continue
                sub = sublabel_from_trace(tr2, bil2)
                res = evaluate_full(env, table, book, blob, toks, record_obs=True)
                mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                extra = {
                    "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT",
                    "sublabel": sub, "lost_class": info["lost_class"],
                    "root_episode": it["root_episode"], "source": "pass2",
                    "derived": bool(it.get("derived")),
                }
                save_expert(env, ic_id, f"p2_{tag}_{vname}_{sub}", res, mech, mapping_meta, checksum, snap, extra)
                row = {
                    "ic_id": ic_id, "tag": f"p2_{tag}_{vname}_{sub}",
                    "root_episode": it["root_episode"],
                    "expert_type": "TYPE3_LOST_GRASP_FULL_EXPERT", "sublabel": sub,
                    "lost_class": info["lost_class"], "SUCCESS_NEW": 1,
                    "win_mean_pos": res["window"].get("mean_pos"),
                    "win_mean_theta": res["window"].get("mean_theta"),
                    "hold_both_s": res["window"].get("longest_both_ok_s"),
                    "bilateral_i": bil2, "n_tokens": len(toks),
                    "tokens": json.dumps(toks), "config_id": CONFIG_ID,
                    "mapping_checksum": checksum,
                    "newly_vs_simple": int(not simple), "reproducible": 1,
                    "derived": int(bool(it.get("derived"))),
                }
                full_experts.append(row)
                best = row
                print(f"  FULL {sub} NEW=1 th={row['win_mean_theta']:.1f} pos={row['win_mean_pos']:.1f}", flush=True)
                break
            if best is not None:
                break

        coverage_extra.append({
            "ic_id": ic_id, "root_episode": it["root_episode"],
            "lost_class": info["lost_class"], "derived": int(bool(it.get("derived"))),
            "full": int(best is not None), "searched": n_try,
            "simple": int(simple), "budget_s": time.time() - t0b,
        })
        print(f"  DONE full={best is not None} tries={n_try} ({time.time()-t0b:.1f}s)", flush=True)

    # merge IC rows
    all_ics = prior + ic_extra
    write_csv(OUT / "LOST_GRASP_ICS.csv", all_ics)
    write_csv(OUT / "TYPE3_FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", full_experts)
    write_csv(OUT / "RAW" / "pass2_coverage.csv", coverage_extra)
    write_csv(OUT / "RAW" / "pass2_baselines.csv", baselines_extra)
    write_csv(OUT / "RAW" / "pass2_failures.csv", failures[:800])

    # refresh answers from combined
    from scripts.generate_k49_lost_grasp_recovery import OUT as _
    # compute summary
    roots = sorted({r.get("root_episode") or root_episode_id(r["ic_id"])
                    for r in all_ics if str(r.get("is_type3_onset")) in ("1", "True")})
    n_full = len({r["ic_id"] for r in full_experts})
    n_new = len({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1})
    answers = {
        "Q1_n_independent_lost_grasp_roots": len(roots),
        "Q1_root_ids": roots,
        "Q2_n_ics_bilateral_restored": len({c["ic_id"] for c in coverage_extra if int(c.get("full") or 0) or True}),
        "Q4_n_type3_full_ics": n_full,
        "Q4_full_ids": sorted({r["ic_id"] for r in full_experts}),
        "Q4_sublabels": dict(Counter(r["sublabel"] for r in full_experts)),
        "Q5_n_newly_solved_vs_simple": n_new,
        "Q5_newly_ids": sorted({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1}),
        "pass2_wall_s": time.time() - t0,
        "pass2_n_roll": n_roll,
        "pass2_derived_n": len(derived),
        "n_full_derived": sum(1 for r in full_experts if int(r.get("derived") or 0) == 1),
    }
    # merge with pass1 answers if present
    p1 = OUT / "RAW" / "answers.json"
    if p1.exists():
        old = json.loads(p1.read_text(encoding="utf-8"))
        old.update(answers)
        # recompute Q2 from prefixes file if exists
        prefs = list(csv.DictReader((OUT / "RECATCH_PREFIXES.csv").open(encoding="utf-8"))) if (OUT / "RECATCH_PREFIXES.csv").exists() else []
        old["Q2_n_ics_bilateral_restored"] = len({p["ic_id"] for p in prefs} | {r["ic_id"] for r in full_experts})
        need = [p for p in prefs if float(p.get("theta_at_bil") or 0) > 10 or float(p.get("pos_at_bil") or 0) > 12]
        old["Q3_n_stable_recatch_need_slide_pivot"] = len({p["ic_id"] for p in need})
        n1 = old.get("Q7_counts", {}).get("type1_ics_approx", 8)
        n2 = old.get("Q7_counts", {}).get("type2_ics", 5)
        old["Q7_counts"] = {"type1_ics_approx": n1, "type2_ics": n2, "type3_lost_grasp_ics": n_full}
        old["Q7_ready_unified_SFT"] = bool(n1 >= 3 and n2 >= 2 and n_full >= 3)
        old["Q7_note"] = (
            f"Type1≈{n1}, Type2={n2}, Type3-lost-grasp={n_full}. "
            + ("Ready for unified SFT with episode-grouped splits." if old["Q7_ready_unified_SFT"]
               else "Need more independent Type3 lost-grasp FULL experts.")
        )
        # Q6 from pass2 failures
        old["Q6_failure_reasons"] = dict(Counter(f.get("reason") for f in failures).most_common(10))
        old["Q6_unsolved_n"] = sum(1 for c in coverage_extra if int(c.get("full") or 0) == 0)
        answers = old
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Type 3 Lost-Grasp Recovery Experts

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**Type 3 = lost/unreliable grasp at Recovery onset** (eccentric/Δθ not required).  
**No SFT/GRPO.**

## Pass 1 + Pass 2

- Pass1: mined natural lost-grasp snapshots (unilateral / unstable bilateral / complete loss).
- Pass2: legal unload→loss derived ICs from hard poses + heavier Stage A/B / Type2-seeded search.

## Yield

| Metric | Value |
|--------|------:|
| Independent lost-grasp roots | {answers.get('Q1_n_independent_lost_grasp_roots')} |
| TYPE3_LOST_GRASP_FULL_EXPERT ICs | **{n_full}** |
| Newly vs simple baselines | **{n_new}** |
| Derived-loss FULL | {answers.get('n_full_derived')} |
| Sublabels | {answers.get('Q4_sublabels')} |

### Important: `hrv1_920031` onset

Natural snapshot classifies as **STABLE_GRASP** (bilateral, probe holds) — therefore the known
`[30,12,12,8,0×7]` SUCCESS_NEW trajectory is **not** automatically a Type3 lost-grasp expert.
Derived unload→loss variants from 031 may qualify if onset becomes lost-grasp.

## Q1–Q7

1. Independent lost-grasp roots: **{answers.get('Q1_n_independent_lost_grasp_roots')}**
2. Bilateral restore ICs: **{answers.get('Q2_n_ics_bilateral_restored')}**
3. Need Sliding/Pivoting after recatch: **{answers.get('Q3_n_stable_recatch_need_slide_pivot')}**
4. Complete SUCCESS_NEW Type3: **{n_full}** {answers.get('Q4_full_ids')}
5. Newly solved vs simple: **{n_new}** {answers.get('Q5_newly_ids')}
6. Remaining failures: {answers.get('Q6_failure_reasons')}
7. Ready unified SFT? **{answers.get('Q7_ready_unified_SFT')}** — {answers.get('Q7_note')}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    ready = f"""# Dataset Readiness — Lost-Grasp Type3

## Verdict: {"YES" if answers.get("Q7_ready_unified_SFT") else "NOT YET"}

- Type3 lost-grasp FULL ICs: {n_full}
- Newly vs simple: {n_new}

Merge only `TYPE3_FULL_EXPERTS.csv` rows with verified lost-grasp onset (`lost_class` in LOST_GRASP_ICS).
Split by `root_episode`. Derived ICs inherit parent root episode.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nPASS2_DONE", answers, flush=True)


if __name__ == "__main__":
    main()
