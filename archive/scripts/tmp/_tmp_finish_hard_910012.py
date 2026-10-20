"""Focused post-capture completion attempt for hard_910012 (remaining budget)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table, describe_k49_config
from training.recovery_episode import make_training_env
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, decompose, evaluate_full, mapping_checksum, write_csv, write_json,
)
from scripts.generate_k49_in_recovery_recatch import pose_correction_proposals
from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics, is_high_motion
from scripts.expand_k49_large_chase_dynamic import is_large_chase
from scripts.expand_k49_capture_brake_diagnostics import (
    OUT, LC_THR, HM_THR, BRAKE_TOK, find_snap, load_blob, replay_rich, contact_events,
    milestones, classify_failure, tokens_hash,
)
from data_generation.recording.trajectory_recorder import save_pair

device = torch.device("cpu")
model, book48 = load_trained_k48(SFT, device)
table = build_k49_table(book48, config_id=CONFIG_ID)
checksum = mapping_checksum(table)
env = make_training_env(device, model=model)
rng = np.random.default_rng(20261019)

ic = "hard_910012"
blob = load_blob(find_snap(ic))
st0 = hard_ic_metrics(env, blob)

# Prefixes that previously produced long bil persist: open_t17 style + brake/hold
prefixes = [
    [30] * 3 + [17] * 16 + [0] * 8,
    [30] * 3 + [17] * 16 + [BRAKE_TOK] * 8 + [0] * 8,
    [30] * 3 + [17] * 16 + [0] * 4 + [BRAKE_TOK] * 8,
    [30] * 3 + [17] * 14 + [BRAKE_TOK] * 6 + [0] * 12,
    [17] * 12 + [0] * 16,
    [17] * 12 + [BRAKE_TOK] * 10 + [0] * 10,
    [5] * 16 + [0] * 12,
    [5] * 16 + [BRAKE_TOK] * 8 + [0] * 8,
]

restore_capture = __import__("envs.simulation.capture", fromlist=["restore_capture"]).restore_capture
restore_capture(env, blob["capture"])
desire = -decompose(env.sim)["err_h"]
pose_conts = []
for n_ret in (8, 12, 16, 20, 24, 28, 32, 40, 48):
    pose_conts.append([SECURE_REVERSE_WRIST] * n_ret + [0] * 8)
    pose_conts.append([0] * 8 + [SECURE_REVERSE_WRIST] * n_ret + [0] * 6)
    pose_conts.append([BRAKE_TOK] * 4 + [SECURE_REVERSE_WRIST] * n_ret + [0] * 6)
    pose_conts.append([0] * 16 + [SECURE_REVERSE_WRIST] * n_ret)
for piv in (4, 6, 10):
    pose_conts.append([0] * 6 + [piv] * 12 + [SECURE_REVERSE_WRIST] * 24 + [0] * 6)
    pose_conts.append([BRAKE_TOK] * 4 + [piv] * 16 + [0] * 4 + [SECURE_REVERSE_WRIST] * 24)
    pose_conts.append([0] * 8 + [piv] * 8 + [12] * 6 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4)
try:
    pose_conts.extend(pose_correction_proposals(book48, desire, rng, n=16))
except Exception:
    pass

rows = []
full = None
n = 0
for pi, pref in enumerate(prefixes):
    for ci, cont in enumerate(pose_conts):
        toks = list(pref) + list(cont)
        metrics, trace = replay_rich(env, table, book48, blob, toks)
        n += 1
        ev = contact_events(trace)
        ms = milestones(ev, metrics)
        primary, order = classify_failure(trace, metrics, ev)
        row = {
            "pref_i": pi, "cont_i": ci,
            "SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
            "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
            "max_bil_persist_s": ev["max_bil_persist_s"],
            "primary_fail": primary,
            "win_mean_theta": metrics.get("win_mean_theta"),
            "win_mean_pos": metrics.get("win_mean_pos"),
            **{k: ms[k] for k in ms},
        }
        rows.append(row)
        if metrics.get("SUCCESS_NEW"):
            res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
            stem = f"{ic}__focus_p{pi}_c{ci}"
            save_pair(
                OUT / "trajectories", stem,
                res["actions"], res["observations"], res["contacts"],
                res["tokens"], res["residuals"], res["handoff_obs"],
            )
            hm = int(is_high_motion(st0, HM_THR))
            lc = int(is_large_chase(metrics, LC_THR))
            erow = {
                "ic_id": ic, "root_episode": ic, "tag": stem.split("__", 1)[1],
                "expert_type": "CAPTURE_BRAKE_FULL_EXPERT", "SUCCESS_NEW": 1,
                "win_mean_theta": res["window"].get("mean_theta"),
                "win_mean_pos": res["window"].get("mean_pos"),
                "hold_both_s": res["window"].get("longest_both_ok_s"),
                "n_tokens": len(toks), "tokens": json.dumps(toks),
                "tokens_hash": tokens_hash(toks),
                "high_motion": hm, "large_chase": lc,
                "hand_max_disp_m": metrics.get("hand_max_disp_m"),
                "disp_to_recatch_m": metrics.get("disp_to_recatch_m"),
                "source": "hard_910012_focused", "round": "P3b",
                "config_id": CONFIG_ID, "mapping_checksum": checksum,
                "snapshot": str(OUT / "snapshots" / f"{ic}.pkl"),
                "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"], "v_lat0": st0["v_lat"],
            }
            write_csv(OUT / "FULL_EXPERTS.csv", [erow])
            write_json(OUT / "trajectories" / f"{stem}_meta.json", erow)
            full = erow
            print("FULL", erow["win_mean_theta"], erow["win_mean_pos"], "LC", lc, "HM", hm)
            break
    if full:
        break

write_csv(OUT / "RAW" / "HARD_910012_FOCUSED_SEARCH.csv", rows)
from collections import Counter
print("trials", n, "NEW", sum(int(r["SUCCESS_NEW"]) for r in rows))
print("reasons", Counter(r["reason"] for r in rows).most_common(6))
print("max_persist", max(float(r["max_bil_persist_s"]) for r in rows))
print("M5", sum(int(r["M5_stable_bil"]) for r in rows), "M7", sum(int(r["M7_SUCCESS_NEW"]) for r in rows))
print("best persist reasons", [
    (r["reason"], r["max_bil_persist_s"], r.get("win_mean_theta"), r.get("win_mean_pos"))
    for r in sorted(rows, key=lambda x: -float(x["max_bil_persist_s"]))[:5]
])
