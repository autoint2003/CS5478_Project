"""hard_910012: after stable capture, search position-correction suffixes for SUCCESS_NEW."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.recovery_episode import make_training_env
from data_generation.recording.trajectory_recorder import save_pair
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decompose, evaluate_full, mapping_checksum, slide_proposals, write_csv, write_json,
)
from scripts.expand_k49_hard_dynamic import ic_metrics as hard_ic_metrics, is_high_motion
from scripts.expand_k49_large_chase_dynamic import is_large_chase
from scripts.expand_k49_capture_brake_diagnostics import (
    OUT, LC_THR, HM_THR, BRAKE_TOK, find_snap, load_blob, replay_rich, contact_events, tokens_hash,
)

device = torch.device("cpu")
model, book48 = load_trained_k48(SFT, device)
table = build_k49_table(book48, config_id=CONFIG_ID)
checksum = mapping_checksum(table)
env = make_training_env(device, model=model)
rng = np.random.default_rng(42)

ic = "hard_910012"
blob = load_blob(find_snap(ic))
st0 = hard_ic_metrics(env, blob)

# Capture-stabilize prefixes known to reach SUCCESS_12S-ish
caps = [
    [30] * 3 + [17] * 16 + [0] * 8 + [BRAKE_TOK] * 6,
    [30] * 3 + [17] * 16 + [BRAKE_TOK] * 8 + [0] * 8,
    [5] * 16 + [BRAKE_TOK] * 8 + [0] * 8,
    [17] * 12 + [BRAKE_TOK] * 10 + [0] * 10,
]

rows = []
full = None
for ci, cap in enumerate(caps):
    restore_capture(env, blob["capture"])
    # run cap to get desire at post-capture
    from scripts.generate_k49_sliding_recatch_dataset import run_seq
    run_seq(env, table, book48, blob, cap)
    desire = -decompose(env.sim)["err_h"]
    slides = slide_proposals(book48, desire, rng, n=12)
    # also hand-crafted COM correction bursts
    extra = []
    for t in (17, 33, 1, 5, 13, 19):
        extra.append([t] * 12 + [BRAKE_TOK] * 4 + [0] * 4)
        extra.append([t] * 16 + [SECURE_REVERSE_WRIST] * 20 + [0] * 4)
        extra.append([t] * 8 + [12] * 4 + [t] * 8 + [SECURE_REVERSE_WRIST] * 16)
    conts = slides + extra
    for si, cont in enumerate(conts):
        for ret_n in (16, 28):
            toks = list(cap) + list(cont) + ([SECURE_REVERSE_WRIST] * ret_n if ret_n else []) + [0] * 6
            metrics, trace = replay_rich(env, table, book48, blob, toks)
            ev = contact_events(trace)
            row = {
                "cap_i": ci, "si": si, "ret_n": ret_n,
                "SUCCESS_NEW": int(metrics.get("SUCCESS_NEW") or 0),
                "reason": metrics.get("eval_reason") or metrics.get("reason_fail"),
                "win_mean_theta": metrics.get("win_mean_theta"),
                "win_mean_pos": metrics.get("win_mean_pos"),
                "hold_both_s": metrics.get("hold_both_s"),
                "max_bil_persist_s": ev["max_bil_persist_s"],
            }
            rows.append(row)
            if metrics.get("SUCCESS_NEW"):
                res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
                stem = f"{ic}__posfix_{ci}_{si}_{ret_n}"
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
                    "complete_contact_loss": int(metrics.get("complete_contact_loss") or 0),
                    "source": "hard_910012_pose_pos", "round": "P3c",
                    "config_id": CONFIG_ID, "mapping_checksum": checksum,
                    "snapshot": str(OUT / "snapshots" / f"{ic}.pkl"),
                    "v0": st0["v_rel_norm"], "w0": st0["w_rel_norm"], "v_lat0": st0["v_lat"],
                }
                write_csv(OUT / "FULL_EXPERTS.csv", [erow])
                write_json(OUT / "trajectories" / f"{stem}_meta.json", {"tokens": toks, **erow})
                full = erow
                print("FULL", erow)
                break
        if full:
            break
    if full:
        break
    # progress
    best_pos = min((float(r["win_mean_pos"]) for r in rows if r.get("win_mean_pos") is not None), default=None)
    print(f"cap {ci} done n={len(rows)} best_pos={best_pos} reasons={Counter(r['reason'] for r in rows).most_common(3)}", flush=True)

write_csv(OUT / "RAW" / "HARD_910012_POSE_POS_SEARCH.csv", rows)
s12 = [r for r in rows if r["reason"] == "SUCCESS_12S"]
print("total", len(rows), "NEW", sum(int(r["SUCCESS_NEW"]) for r in rows), "S12", len(s12))
if s12:
    print("best S12 pos", min(float(r["win_mean_pos"]) for r in s12 if r["win_mean_pos"]),
          "theta", min(float(r["win_mean_theta"]) for r in s12 if r["win_mean_theta"]))
    best = sorted([r for r in s12 if r["win_mean_pos"] is not None], key=lambda r: float(r["win_mean_pos"]))[:5]
    for r in best:
        print(" ", r["win_mean_theta"], r["win_mean_pos"], r["hold_both_s"])
