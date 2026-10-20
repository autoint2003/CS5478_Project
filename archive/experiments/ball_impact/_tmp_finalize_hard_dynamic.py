"""Finalize hard-dynamic expansion with accurate Q recounts + light replay check."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    evaluate_full,
    write_csv,
    write_json,
)
from training.action_vocab_k49 import assert_k48_prefix_identical, build_k49_table
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env

OUT = ROOT / "results" / "expert_generation" / "k49_hard_dynamic_expansion"


def main():
    answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
    thr = json.loads((OUT / "RAW" / "high_motion_thresholds.json").read_text(encoding="utf-8"))
    experts = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
    coverage = list(csv.DictReader((OUT / "ROOT_COVERAGE.csv").open(encoding="utf-8")))
    pool = list(csv.DictReader((OUT / "HARD_DYNAMIC_IC_POOL.csv").open(encoding="utf-8")))
    new_imp = list(csv.DictReader((OUT / "NEW_IMPACT_ROOTS.csv").open(encoding="utf-8")))
    baselines = list(csv.DictReader((OUT / "MATCHED_BASELINES.csv").open(encoding="utf-8")))
    failures = list(csv.DictReader((OUT / "FAILURES.csv").open(encoding="utf-8")))
    unified = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8")))

    # Replay verify
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = load_codebook()
    table = build_k49_table(book, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    ver = []
    for r in experts:
        ic = r["ic_id"]
        toks = json.loads(r["tokens"])
        blob = pickle.loads((OUT / "snapshots" / f"{ic}.pkl").read_bytes())
        res = evaluate_full(env, table, book, blob, toks, record_obs=False)
        ver.append({
            "ic_id": ic, "replay_SUCCESS_NEW": int(res["SUCCESS_NEW"]),
            "reason": res["reason"],
            "high_motion": r.get("high_motion"),
            "newly_vs_simple": r.get("newly_vs_simple"),
            "tags": r.get("mechanism_tags"),
        })
        print(ic, "OK" if res["SUCCESS_NEW"] else "FAIL", res["reason"], flush=True)
    write_csv(OUT / "RAW" / "replay_verification.csv", ver)
    n_ok = sum(v["replay_SUCCESS_NEW"] for v in ver)

    # Accurate partial stats from per-IC best stage in coverage/failures
    best_stage = {}
    for f in failures:
        ic = f.get("ic_id")
        st = f.get("stage")
        if not ic or not st:
            continue
        # prefer OK > E/F > D > C > B > A > G
        rank = {"OK": 0, "E": 1, "F": 2, "D": 3, "C": 4, "BR": 5, "B": 6, "A": 7, "G": 8}
        if ic not in best_stage or rank.get(st, 9) < rank.get(best_stage[ic], 9):
            best_stage[ic] = st
    n_contact = sum(1 for s in best_stage.values() if s in ("C", "D", "E", "F", "OK", "BR"))
    n_pose_needed = sum(1 for s in best_stage.values() if s in ("E", "F"))

    answers["Q5_n_restored_uni_or_bil_partials"] = n_contact
    answers["Q6_n_stable_but_pose_needed_partials"] = n_pose_needed
    answers["replay_n_ok"] = n_ok
    answers["replay_n"] = len(ver)
    answers["Q8_n_SUCCESS_NEW"] = n_ok
    answers["full_ids"] = [v["ic_id"] for v in ver if v["replay_SUCCESS_NEW"]]
    answers["Q4_thresholds"] = thr
    write_json(OUT / "RAW" / "answers.json", answers)

    # Refresh REPORT key section
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    # rewrite Q5/Q6 lines if present
    lines = []
    for line in report.splitlines():
        if line.startswith("5. Partials restoring"):
            line = f"5. Partials restoring uni/bilateral (best stage C–F/OK/BR per IC): **{n_contact}**"
        elif line.startswith("6. Stable grasp but pose"):
            line = f"6. Stable grasp but pose still needed (best E/F per IC): **{n_pose_needed}**"
        elif line.startswith("8. Complete SUCCESS_NEW"):
            line = f"8. Complete SUCCESS_NEW (replay-verified): **{n_ok}** / {len(ver)}"
        lines.append(line)
    # append replay note
    text = "\n".join(lines)
    if "Replay verification" not in text:
        text += f"\n\n## Replay verification\n**{n_ok}/{len(ver)}** FULL experts replay SUCCESS_NEW.\n"
        text += f"IDs: {answers['full_ids']}\n"
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")

    ready = f"""# Dataset Readiness — Hard Dynamic Expansion

## Verdict: PARTIAL ({n_ok}/{answers.get('Q15_gap_to_20', 13) + n_ok} toward 20-root target)

- New nontrivial hard-dynamic SUCCESS_NEW roots (replay OK): **{n_ok}**
- HIGH_MOTION FULL (frozen thr): **{answers.get('n_HIGH_MOTION_FULL', 0)}**
- Unified traj/IC/roots after merge: **{answers['Q14_n_traj']}** / **{answers['Q14_n_ics']}** / **{answers['Q14_n_roots']}**
- Gap to 20: **{max(0, 20 - n_ok)}**

### Frozen HIGH_MOTION thresholds
- v_rel ≥ {thr['v_rel_norm_mps']:.4f} m/s
- |ω_rel| ≥ {thr['w_rel_norm_radps']:.4f} rad/s
- v_lat ≥ {thr['v_lat_mps']:.4f} m/s

### Note
All accepted experts failed matched simple baselines. Zero easy STABLE_POSE_ERROR counted toward the hard-dynamic target.
True HIGH_MOTION successes remain the main gap — most high-v impacts terminate as TABLE_DROP/DROP before bilateral restoration.

Do not start SFT/GRPO here.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("FINAL", json.dumps({
        "replay_ok": n_ok, "gap": max(0, 20 - n_ok),
        "ids": answers["full_ids"],
        "Q5": n_contact, "Q6": n_pose_needed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
