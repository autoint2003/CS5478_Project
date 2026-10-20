"""Search capture-brake continuations aiming for rot_window>=3 and SUCCESS_NEW."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from scripts.generate_k49_dynamic_recatch_experts import analyze_type3_mechanism, run_traced
from scripts.generate_k49_horizontal_endcatch_recatch import (
    NEAR_MISS_SEEDS,
    endcatch_proposals,
    snapshot_path,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    decode,
    decompose,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
)
from scripts.generate_k49_type3_capture_brake import (
    OUT,
    capture_brake_proposals,
    reconstruct_candidate_sequences,
    run_traced_with_snapshots,
    pick_branch_indices,
    restore_branch,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book = load_codebook()
table = build_k49_table(book, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

results = []
for root_id, tags in [
    ("hrv1_920031", [f"t3_{i}" for i in range(0, 12)]),
    ("hard_910318", [f"t3_{i}" for i in list(range(0, 12)) + list(range(29, 45))]),
]:
    sp = snapshot_path(root_id)
    blob = pickle.loads(sp.read_bytes())
    tag_to_seq, _ = reconstruct_candidate_sequences(
        env, table, book, blob, root_id, tags, budget=40, max_extra=24
    )
    print(root_id, "got", len(tag_to_seq), flush=True)
    rng = np.random.default_rng(20261019)
    for tag, seq in list(tag_to_seq.items())[:10]:
        ok, trace, snaps = run_traced_with_snapshots(env, table, book, blob, seq)
        picks = pick_branch_indices(trace)
        if not picks:
            continue
        named = picks["named"]
        for bname in ("first_eccentric_recontact", "max_useful_orientation", "peak_angular_speed"):
            if bname not in named:
                continue
            bidx = named[bname]
            prefix = seq[:bidx]
            snap = snaps[bidx]
            restore_branch(env, snap)
            desire = -decompose(env.sim)["err_h"]
            conts = capture_brake_proposals(book, desire, rng, n=16)
            # add slow-rotate-then-brake
            conts = [
                [12] * 6 + [0] * 16,
                [12] * 4 + [8] * 4 + [0] * 16,
                [12] * 4 + [4] * 8 + [0] * 14,
                [8] * 6 + [12] * 4 + [0] * 14,
                [0] * 24,
            ] + conts
            for cont in conts:
                for settle in ([0] * 3, [0] * 20, [0] * 12 + [SECURE_REVERSE_WRIST] * 12 + [0] * 8):
                    ftoks = list(prefix) + list(cont) + list(settle)
                    res = evaluate_full(env, table, book, blob, ftoks, record_obs=False)
                    if not (res["SUCCESS_NEW"] or (res["success_12s"] and (res["window"].get("mean_theta") or 99) < 12)):
                        continue
                    ok2, tr2 = run_traced(env, table, book, blob, ftoks)
                    m3 = analyze_type3_mechanism(tr2)
                    fc, bil = m3.get("first_contact_i"), m3.get("bilateral_i")
                    rot_w = 0.0
                    if fc is not None:
                        end_r = bil if bil is not None else min(len(tr2) - 1, fc + 20)
                        ths = [float(tr2[i]["e_theta_deg"]) for i in range(fc, end_r + 1)]
                        if ths:
                            rot_w = ths[0] - min(ths)
                    row = {
                        "root": root_id, "tag": tag, "branch": bname,
                        "NEW": int(res["SUCCESS_NEW"]), "s12": int(res["success_12s"]),
                        "th": res["window"].get("mean_theta"), "pos": res["window"].get("mean_pos"),
                        "hold": res["window"].get("longest_both_ok_s"),
                        "rot_window": rot_w,
                        "dth_m3": m3.get("delta_theta_deg"),
                        "ecc": int(bool(m3.get("eccentric"))),
                        "down": int(bool(m3.get("downward_motion"))),
                        "ben": int(bool(m3.get("beneficial_rotation"))),
                        "post": int(bool(m3.get("post_regrasp_correction"))),
                        "tokens": ftoks,
                    }
                    results.append(row)
                    if res["SUCCESS_NEW"]:
                        print("HIT", row["root"], "rot", round(rot_w, 2), "ben", row["ben"],
                              "th", round(row["th"] or -1, 2), "pos", round(row["pos"] or -1, 2), flush=True)

strong = [r for r in results if r["NEW"] and r["rot_window"] >= 3 and r["ecc"] and r["down"] and not r["post"]]
print("STRONG_FULL", len(strong))
(OUT / "RAW" / "strong_rot_search.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("n_results", len(results), "n_NEW", sum(1 for r in results if r["NEW"]))
