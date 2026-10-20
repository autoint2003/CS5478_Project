"""Spot-check reproducibility and write SPLIT for in-recovery recatch dataset."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.generate_k49_in_recovery_recatch import OUT
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
    write_csv,
    write_json,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    man = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8")))
    repro = []
    for r in man:
        toks = [int(t) for t in json.loads(r["tokens"]) if int(t) != 47]
        snap = OUT / "snapshots" / f"{r['ic_id']}.pkl"
        blob = pickle.loads(snap.read_bytes())
        res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
        ok = bool(res["SUCCESS_NEW"])
        repro.append({"ic_id": r["ic_id"], "ok": ok, "win_pos": res["window"].get("mean_pos"),
                      "win_theta": res["window"].get("mean_theta")})
        r["reproducible"] = int(ok)
        print(f"REPRO {r['ic_id']} ok={ok}", flush=True)

    write_csv(OUT / "UNIFIED_MANIFEST.csv", man)
    write_json(OUT / "RAW" / "reproducibility.json", repro)

    ics = sorted({r["ic_id"] for r in man})
    rng = np.random.default_rng(20261018 + 11)
    rng.shuffle(ics)
    n = len(ics)
    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    split = {
        "test": ics[:n_test],
        "val": ics[n_test:n_test + n_val],
        "train": ics[n_test + n_val:],
        "note": "Split by IC / root episode; no trajectory-row leakage.",
    }
    for r in man:
        ep = r["ic_id"]
        if ep in split["train"]:
            r["split"] = "train"
        elif ep in split["val"]:
            r["split"] = "val"
        else:
            r["split"] = "test"
    write_json(OUT / "SPLIT_BY_IC.json", split)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", man)

    answers = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
    answers["n_repro_ok"] = sum(1 for x in repro if x["ok"])
    answers["n_repro_total"] = len(repro)
    answers["Q5_ready_to_merge"] = bool(
        answers["Q2_n_independent_ics_in_recovery"] >= 3
        and answers["n_repro_ok"] == answers["n_repro_total"]
    )
    write_json(OUT / "RAW" / "answers.json", answers)

    # refresh readiness
    ready = f"""# Dataset Readiness — In-Recovery Recatch

## Verdict: {"MERGE-CANDIDATE" if answers["Q5_ready_to_merge"] else "NOT READY"}

- IN_RECOVERY_RECATCH_FULL_EXPERT ICs: **{answers["Q2_n_independent_ics_in_recovery"]}**
- Reproducible: **{answers["n_repro_ok"]}/{answers["n_repro_total"]}**
- Historical HANDOFF-assisted S12 (reference only): {answers["historical_s12_count"]}
- Contact-only failures: {answers["contact_only_count"]}

### Merge rule
Merge only `expert_type=IN_RECOVERY_RECATCH_FULL_EXPERT` rows from `UNIFIED_MANIFEST.csv`.
Do **not** label historical HANDOFF-assisted traces as in-Recovery.

### Split
See `SPLIT_BY_IC.json` (IC-grouped).
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("FINAL", answers["Q5_ready_to_merge"], answers["n_repro_ok"], flush=True)


if __name__ == "__main__":
    main()
