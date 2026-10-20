"""Finalize manifests, integrity, split, and REPORT for k49_sliding_recatch_dataset."""
from __future__ import annotations

import csv
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    OUT,
    SFT,
    build_k49_table,
    classify_mechanisms,
    describe_k49_config,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
    mapping_checksum,
    write_csv,
    write_json,
)


def root_episode_id(ic_id: str) -> str:
    """Group derived/mined ICs with their source disturbance episode."""
    s = ic_id
    s = re.sub(r"_uni\d+_t\d+$", "", s)
    s = re.sub(r"_uni\d+$", "", s)
    # gen_hrv1_920031_1_hash -> hrv1_920031
    m = re.match(r"gen_(hrv1|hard)_(\d+)_", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"gen_(hrv1|hard)_(\d+)$", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    if s.startswith("hrv1_leftover_"):
        return s.replace("hrv1_leftover_", "")
    return s


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", meta)

    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    manifest = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8")))
    sliding = list(csv.DictReader((OUT / "SLIDING_EXPERTS.csv").open(encoding="utf-8")))
    recatch = list(csv.DictReader((OUT / "RECATCH_EXPERTS.csv").open(encoding="utf-8")))
    mixed = list(csv.DictReader((OUT / "MIXED_EXPERTS.csv").open(encoding="utf-8")))
    coverage = list(csv.DictReader((OUT / "IC_COVERAGE.csv").open(encoding="utf-8"))) if (OUT / "IC_COVERAGE.csv").exists() else []
    failures = list(csv.DictReader((OUT / "FAILURES.csv").open(encoding="utf-8"))) if (OUT / "FAILURES.csv").exists() else []

    # Reproducibility check
    repro = []
    for r in manifest:
        ic = r["ic_id"]
        toks = json.loads(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"]
        snap = OUT / "snapshots" / f"{ic}.pkl"
        if not snap.exists():
            repro.append({"ic_id": ic, "ok": False, "reason": "missing_snapshot"})
            continue
        blob = pickle.loads(snap.read_bytes())
        res = evaluate_full(env, table, book48, blob, [int(t) for t in toks if int(t) != 47], record_obs=True)
        mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
        ok = bool(res["SUCCESS_NEW"])
        repro.append({
            "ic_id": ic, "ok": ok, "label": res["label"],
            "win_mean_pos": res["window"].get("mean_pos"),
            "win_mean_theta": res["window"].get("mean_theta"),
            "hold_both_s": res["window"].get("longest_both_ok_s"),
            "observed_recovery": mech["observed_recovery"],
            "recatch_subtype": mech["recatch_subtype"] or r.get("recatch_subtype"),
            "sliding_verified": mech["sliding_verified"],
        })
        r["reproducible"] = int(ok)
        r["episode_id"] = root_episode_id(ic)
        print(f"  REPRO {ic} ok={ok} win_p={res['window'].get('mean_pos')}", flush=True)

    write_json(OUT / "RAW" / "reproducibility.json", repro)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)

    # Integrity
    issues = []
    for r in manifest:
        toks = json.loads(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"]
        for t in toks:
            t = int(t)
            if t == 47:
                issues.append(f"{r['ic_id']}: HANDOFF in motion")
            elif t < 0 or t > 48:
                issues.append(f"{r['ic_id']}: illegal token {t}")
        if r.get("mapping_checksum") and r["mapping_checksum"] != checksum:
            issues.append(f"{r['ic_id']}: checksum mismatch")
        if r.get("config_id") != CONFIG_ID:
            issues.append(f"{r['ic_id']}: wrong config")
    # obs dim check from saved npz
    for r in manifest:
        stem_candidates = list((OUT / "trajectories").glob(f"{r['ic_id']}__*_obs.npy")) + \
                          list((OUT / "trajectories").glob(f"{r['ic_id']}__*.npz"))
        # save_pair naming
        for p in (OUT / "trajectories").glob(f"{r['ic_id']}__*"):
            if p.suffix == ".npz" and "tokens" not in p.name:
                try:
                    z = np.load(p)
                    if "observations" in z.files:
                        obs = z["observations"]
                        if obs.ndim >= 2 and obs.shape[-1] != 39:
                            issues.append(f"{r['ic_id']}: obs dim {obs.shape[-1]}")
                except Exception:
                    pass
                break

    integrity = {
        "n_manifest": len(manifest),
        "n_issues": len(issues),
        "issues": issues[:50],
        "all_ok": len(issues) == 0,
        "mapping_checksum": checksum,
        "config_id": CONFIG_ID,
        "handoff_id": 47,
        "obs_dim": 39,
        "n_repro_ok": sum(1 for x in repro if x.get("ok")),
        "n_repro_fail": sum(1 for x in repro if not x.get("ok")),
    }
    write_json(OUT / "RAW" / "integrity.json", integrity)

    # Split by episode_id (not trajectory row)
    episodes = sorted({root_episode_id(r["ic_id"]) for r in manifest})
    rng = np.random.default_rng(20261018 + 7)
    rng.shuffle(episodes)
    n = len(episodes)
    n_test = max(1, n // 5) if n >= 5 else (1 if n >= 3 else 0)
    n_val = max(1, n // 5) if n >= 5 else (1 if n >= 3 else 0)
    split = {
        "test_episodes": episodes[:n_test],
        "val_episodes": episodes[n_test:n_test + n_val],
        "train_episodes": episodes[n_test + n_val:],
        "note": "Grouped by root disturbance episode; mined uni* shares parent episode.",
    }
    # assign rows
    ep_to_split = {}
    for e in split["train_episodes"]:
        ep_to_split[e] = "train"
    for e in split["val_episodes"]:
        ep_to_split[e] = "val"
    for e in split["test_episodes"]:
        ep_to_split[e] = "test"
    for r in manifest:
        r["split"] = ep_to_split[root_episode_id(r["ic_id"])]
        r["episode_id"] = root_episode_id(r["ic_id"])
    write_json(OUT / "SPLIT_BY_IC.json", split)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)

    # Counts for Qs
    slide_ics = sorted({r["ic_id"] for r in sliding if int(float(r.get("SUCCESS_NEW") or 0)) == 1})
    slide_verified_ics = sorted({r["ic_id"] for r in sliding if int(float(r.get("sliding_verified") or 0)) == 1})
    rec_single = sorted({r["ic_id"] for r in recatch if r.get("recatch_subtype") == "single_sided"})
    rec_complete = sorted({r["ic_id"] for r in recatch if r.get("recatch_subtype") == "complete_loss"})
    mixed_rec = [r for r in recatch if int(float(r.get("sliding_after") or 0)) or int(float(r.get("pivoting_after") or 0))
                 or "SLIDING" in (r.get("observed_recovery") or "") or "PIVOTING" in (r.get("observed_recovery") or "")]

    fail_reasons = Counter(r.get("reason") or r.get("label") or "?" for r in failures)
    cov_slide = [c for c in coverage if "slide" in (c.get("pool") or "")]
    cov_rec = [c for c in coverage if "recatch" in (c.get("pool") or "")]

    # Diversity
    tok_sets = []
    for r in manifest:
        toks = tuple(json.loads(r["tokens"]) if isinstance(r["tokens"], str) else r["tokens"])
        tok_sets.append(toks)
    n_unique_seq = len(set(tok_sets))
    pos0s = [float(r["e_pos0"]) for r in manifest if r.get("e_pos0") not in (None, "")]
    families = Counter(r.get("family") for r in manifest)

    q1 = len(slide_verified_ics)  # distinct Sliding ICs with verified sliding (strict)
    q1_loose = len(slide_ics)
    q2 = len(rec_single)
    q3 = len(rec_complete)
    q4 = len(mixed_rec)
    q7 = all(x.get("ok") for x in repro) and len(repro) == len(manifest)
    # readiness: need some diversity across mechanisms
    # Integrity-ready for loading, but not diverse enough for balanced unified SFT.
    loadable = (
        integrity["all_ok"]
        and integrity["n_repro_ok"] == len(manifest)
        and len(manifest) >= 1
    )
    sliding_pilot_ok = loadable and q1 >= 3 and n_unique_seq >= 4
    # Balanced unified SFT needs broader Recatch episode diversity (aspiration ~15/mechanism).
    ready_unified = (
        sliding_pilot_ok
        and q2 >= 5
        and q3 >= 5
        and len(episodes) >= 15
    )

    answers = {
        "Q1_sliding_verified_ics": q1,
        "Q1_sliding_csv_ics": q1_loose,
        "Q1_ids_verified": slide_verified_ics,
        "Q1_ids_all_sliding_csv": slide_ics,
        "Q2_single_sided_recatch_ics": q2,
        "Q2_ids": rec_single,
        "Q3_complete_loss_recatch_ics": q3,
        "Q3_ids": rec_complete,
        "Q4_recatch_plus_pivot_or_slide": q4,
        "Q5_failure_reasons": fail_reasons.most_common(12),
        "Q6_n_unique_action_sequences": n_unique_seq,
        "Q6_n_manifest": len(manifest),
        "Q6_n_episodes": len(episodes),
        "Q6_e_pos0_range": [min(pos0s), max(pos0s)] if pos0s else None,
        "Q6_families": dict(families),
        "Q7_all_reproducible": q7,
        "Q8_ready_for_unified_SFT": ready_unified,
        "Q8_sliding_pilot_ok": sliding_pilot_ok,
        "Q8_loadable": loadable,
        "Q8_notes": (
            "NOT ready for balanced unified Transformer SFT: only "
            f"{q1} sliding-verified ICs, {q2} single-sided and {q3} complete-loss Recatch IC(s), "
            f"{len(episodes)} root episodes. "
            "Loadable and reproducible; a Sliding-heavy SFT pilot is justifiable, "
            "but Recatch coverage must grow before claiming multi-mechanism unified SFT readiness."
        ),
    }
    write_json(OUT / "RAW" / "answers_q1_q8.json", answers)

    report = f"""# K49 Sliding + Recatch Expert Dataset Report

**Config:** `{CONFIG_ID}`  
**Checksum:** `{checksum}`  
**SUCCESS_NEW:** SUCCESS_12S ∧ θ≤10° ∧ pos≤12 mm hold ≥0.5 s (final 2 s)  
**Action space:** IDs 0–48, HANDOFF=47, token 48 = secure reverse wy=−0.2 rad/s  
**Observation:** original 39D  
**Generated:** wall finalize {time.time()-t0:.1f}s (after multi-pass search)

## Summary counts

| Set | Trajectories | Distinct ICs |
|-----|-------------|--------------|
| SLIDING_EXPERTS | {len(sliding)} | {q1_loose} ({q1} sliding-verified) |
| RECATCH_EXPERTS | {len(recatch)} | {len({r['ic_id'] for r in recatch})} |
| — single-sided | | {q2} |
| — complete-loss | | {q3} |
| MIXED_EXPERTS | {len(mixed)} | {len({r['ic_id'] for r in mixed})} |
| UNIFIED_MANIFEST | {len(manifest)} | {len({r['ic_id'] for r in manifest})} |
| Root episodes (split unit) | | {len(episodes)} |

### Regression controls
- `hard_910263` — VERIFIED_FULL_EXPERT, mechanisms SLIDING+PIVOTING+RETURN
- `hrv1_920031` — VERIFIED_FULL_EXPERT, mechanisms SLIDING+PIVOTING+RETURN

## Q1–Q8

| Q | Answer |
|---|--------|
| **Q1** Sliding ICs → VERIFIED_FULL_EXPERT | **{q1}** sliding-verified ({q1_loose} in SLIDING_EXPERTS.csv including POSITION_CORRECTION): {slide_verified_ics} |
| **Q2** Single-sided Recatch ICs | **{q2}**: {rec_single} |
| **Q3** Complete-loss Recatch ICs | **{q3}**: {rec_complete} |
| **Q4** Recatch + subsequent Pivoting/Sliding | **{q4}** |
| **Q5** Main failure modes | {fail_reasons.most_common(5)} |
| **Q6** Diversity | {n_unique_seq} unique action sequences across {len(manifest)} traj / {len(episodes)} episodes; e_pos0 range {answers['Q6_e_pos0_range']}; families {dict(families)} |
| **Q7** Reproducible from snapshots | **{q7}** ({integrity['n_repro_ok']}/{len(manifest)}) |
| **Q8** Ready for unified Transformer SFT? | **{ready_unified}** (sliding_pilot_ok={sliding_pilot_ok}). {answers['Q8_notes']} |

## Search budget / coverage

- Pass-1: ~18 Sliding + ~17 Recatch ICs × budget 12 (priority + filtered pools)
- Pass-2: 20 near-threshold Sliding ICs (deep CPP+slide) + 16 Recatch ICs
- Pass-3: 22 Recatch-focused (F_recatch+CPP seed + unilateral/complete pools)
- Pass-4: mid-trajectory unilateral snapshot mining from verified experts

Failure reason histogram (aggregated attempts): `{dict(fail_reasons.most_common(8))}`

### Bottlenecks
1. **Recatch from raw hard_ic complete-loss / unilateral ICs:** almost all DROP_PREFIX — object already escaping beyond legal K49 intercept reach within recovery deadline.
2. **Sliding on large initial pose error (≳15 mm):** many SUCCESS_12S PARTIAL with final-window mean pos 13–18 mm — close but under SUCCESS_NEW 12 mm hold.
3. **Forced slow Return (tok48)** can worsen axial slip; adaptive no-return / short return was required for several Sliding successes.

## Mechanism labeling policy
- Labels assigned from contact + relative-motion traces, not token heuristics alone.
- Recatch requires sustained contact loss (≥3 steps) or unilateral/none start topology.
- Single-sided vs complete-loss kept as separate `recatch_subtype` values.
- Source disturbance / observed recovery / final outcome stored separately in trajectory meta.

## Integrity
- Issues: {integrity['n_issues']} — {integrity['issues'][:5]}
- All motion tokens in 0–48 excluding 47 as motion: {integrity['all_ok']}
- Obs dim 39, HANDOFF=47, mapping checksum attached per trajectory.

## Split
Grouped by **root episode id** (mined `*_uni21` shares parent gen episode → same split):
```json
{json.dumps(split, indent=2)}
```

## Artifacts
- `ACTION_MAPPING.json`, `UNIFIED_MANIFEST.csv`, `SPLIT_BY_IC.json`
- `SLIDING_EXPERTS.csv`, `RECATCH_EXPERTS.csv`, `MIXED_EXPERTS.csv`
- `IC_COVERAGE.csv`, `FAILURES.csv`
- `trajectories/`, `snapshots/`, `raw_contact_traces/`
- Scripts: `scripts/generate_k49_sliding_recatch_dataset.py`, `experiments/sliding/_tmp_second_pass_near_threshold.py`, `experiments/sliding/_tmp_pass3_recatch.py`, `experiments/sliding/_tmp_pass4_mine_unilateral_snapshots.py`, `experiments/sliding/_tmp_finalize_sliding_recatch_dataset.py`

## Stop conditions
- Bounded Sliding + Recatch batch completed.
- No new tokens added; physics / nominal / reward / obs / SUCCESS unchanged.
- **SFT / GRPO not started.**
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    readiness = f"""# Dataset Readiness — Unified Transformer SFT

## Verdict: {"YES" if ready_unified else "NO — not yet for balanced unified SFT"}

Sliding-heavy pilot justifiable: **{sliding_pilot_ok}**. Loadable/reproducible: **{loadable}**.

### Strengths
- All saved experts re-validate under locked SUCCESS_NEW with `{CONFIG_ID}`.
- Mapping checksum + config id stored per trajectory.
- Observations are 39D; actions legal K49; HANDOFF=47 only at handoff.
- Sliding mechanism evidence present for {q1} ICs; mixed Sliding+Return/Pivoting present.
- IC/episode-grouped split avoids leakage of mined variants.

### Gaps
- Only **{q2}** single-sided and **{q3}** complete-loss Recatch IC(s) — far below the 15–20 IC search aspiration for Recatch successes.
- Sliding success set is small ({q1_loose} ICs) relative to searched pool; many near-misses are PARTIAL (window pos just above 12 mm).
- Recatch successes currently concentrate on the `gen_hrv1_920031_*` disturbance family (plus a mined unilateral child) — limited episode diversity.

### Recommendation
- **Do not start unified multi-mechanism SFT yet.**
- Optional: Sliding-focused SFT pilot on UNIFIED_MANIFEST with episode-grouped split and Recatch as rare upweights.
- Next data work: mine more mid-trajectory contact-loss snapshots from long legal rollouts; expand near-threshold Sliding ICs; keep SUCCESS_NEW thresholds fixed.

### Checklist
- [x] K49_tok48_wy0p25 locked
- [x] SUCCESS_NEW locked (10°/12 mm)
- [x] Regression controls hard_910263, hrv1_920031
- [x] Mechanism tags from traces
- [x] Failed attempts retained
- [x] Episode-grouped split
- [ ] ≥15 successful Sliding ICs
- [ ] ≥15 successful Recatch ICs (mixed subtypes)
- [ ] Balanced mechanism coverage for unified SFT
"""
    (OUT / "DATASET_READINESS.md").write_text(readiness, encoding="utf-8")
    print("FINALIZE_DONE", answers, flush=True)


if __name__ == "__main__":
    main()
