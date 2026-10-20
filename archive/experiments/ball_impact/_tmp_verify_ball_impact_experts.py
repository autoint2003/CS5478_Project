"""Replay-verify BALL_IMPACT_FULL_EXPERT trajectories."""
from __future__ import annotations

import csv
import json
import pickle
import sys
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

OUT = ROOT / "results" / "expert_generation" / "k49_ball_impact_dynamic_recovery"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = load_codebook()
    table = build_k49_table(book, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    rows = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
    ver = []
    for r in rows:
        ic = r["ic_id"]
        toks = json.loads(r["tokens"])
        blob = pickle.loads((OUT / "snapshots" / f"{ic}.pkl").read_bytes())
        res = evaluate_full(env, table, book, blob, toks, record_obs=False)
        ok = bool(res["SUCCESS_NEW"])
        ver.append({
            "ic_id": ic,
            "replay_SUCCESS_NEW": int(ok),
            "reason": res["reason"],
            "win_th": res["window"].get("mean_theta"),
            "win_pos": res["window"].get("mean_pos"),
            "hold": res["window"].get("longest_both_ok_s"),
            "newly_vs_simple": r.get("newly_vs_simple"),
            "source": r.get("source"),
            "mechanism_tags": r.get("mechanism_tags"),
        })
        status = "OK" if ok else "FAIL"
        print(
            f"{ic} {status} {res['reason']} "
            f"th={res['window'].get('mean_theta'):.1f} "
            f"pos={res['window'].get('mean_pos'):.1f}",
            flush=True,
        )
    write_csv(OUT / "RAW" / "replay_verification.csv", ver)
    n_ok = sum(v["replay_SUCCESS_NEW"] for v in ver)
    summary = {"n": len(ver), "n_ok": n_ok, "all_pass": n_ok == len(ver)}
    write_json(OUT / "RAW" / "replay_verification_summary.json", summary)
    print("SUMMARY", summary, flush=True)


if __name__ == "__main__":
    main()
