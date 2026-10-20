"""Focused brake search around known eccentric+beneficial DROP sequences."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_dynamic_recatch_experts import analyze_type3_mechanism, run_traced
from scripts.generate_k49_horizontal_endcatch_recatch import (
    OUT,
    analyze_capture_rotation_braking,
    brake_refine_around,
    snapshot_path,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    evaluate_full,
    load_codebook,
    make_training_env,
    write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, assert_k48_prefix_identical

# From smoke: Δθ≈22° no_bilateral
SEEDS = [
    [30] + [0] * 4 + [33] * 6 + [12] * 5 + [0] * 14,
    [30] + [12] * 2 + [14] * 8 + [12] * 4 + [0] * 14,
    [30, 0, 0, 0, 0] + [33] * 6 + [12] * 5 + [0] * 14,
    [30, 12, 12] + [14] * 8 + [12] * 4 + [0] * 14,
    # earlier close variants
    [30] + [0] * 4 + [33] * 4 + [0] * 20,
    [30] + [0] * 4 + [33] * 4 + [12] * 2 + [0] * 20,
    [30] + [0] * 3 + [33] * 3 + [12] * 12 + [0] * 16,
    [30] + [0] * 2 + [33] * 8 + [0] * 24,
    [30] + [0] * 5 + [33] * 5 + [12] * 15 + [0] * 10,
    [30] + [0] * 4 + [17] * 6 + [12] * 12 + [0] * 12,
    [30] + [0] * 4 + [5] * 6 + [12] * 12 + [0] * 12,
    [30] + [0] * 4 + [8] * 6 + [12] * 12 + [0] * 12,
    [30] + [0] * 4 + [15] * 6 + [12] * 12 + [0] * 12,
    [36] + [0] * 4 + [33] * 6 + [12] * 10 + [0] * 12,
    [30] + [0] * 6 + [33] * 4 + [12] * 16 + [0] * 10,
    # soft continuous support
    [30] + [12] * 3 + [33] * 4 + [12] * 10 + [0] * 14,
    [30] + [12] * 8 + [33] * 4 + [0] * 16,
    [30] + [12] * 4 + [0] * 4 + [33] * 4 + [12] * 8 + [0] * 12,
]

# 031 near-miss
SEEDS_031 = [
    [30, 12, 12] + [8] * 10 + [0] * 14,
    [30, 12, 12] + [15] * 10 + [0] * 14,
    [30, 12, 12] + [8] * 6 + [12] * 8 + [0] * 14,
    [30, 12, 12] + [8] * 4 + [0] * 20,
    [30, 12, 12] + [8] * 4 + [12] * 12 + [0] * 12,
    [30, 12, 12] + [15] * 4 + [12] * 12 + [0] * 12,
]


def expand(seeds):
    out, seen = [], set()
    for s in seeds:
        for v in brake_refine_around(s):
            k = tuple(v)
            if k in seen:
                continue
            seen.add(k)
            out.append(v)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book = load_codebook()
    table = build_k49_table(book, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    results = []
    for ic_id, seeds in [("hard_910318", expand(SEEDS)), ("hrv1_920031", expand(SEEDS_031))]:
        sp = snapshot_path(ic_id)
        blob = pickle.loads(sp.read_bytes())
        print(ic_id, "n_seq", len(seeds), flush=True)
        best = None
        for i, seq in enumerate(seeds):
            ok, tr = run_traced(env, table, book, blob, seq)
            ph = analyze_capture_rotation_braking(tr)
            m3 = ph["m3"]
            row = {
                "ic_id": ic_id, "i": i, "ok": int(ok),
                "ecc": int(bool(m3.get("eccentric"))),
                "down": int(bool(m3.get("downward_motion"))),
                "ben": int(bool(m3.get("beneficial_rotation"))),
                "sup": int(bool(ph.get("supported_while_rotating"))),
                "bil": int(m3.get("bilateral_i") is not None),
                "dth": m3.get("delta_theta_deg"),
                "ax": m3.get("axial_max_mm"),
                "th_bil": m3.get("theta_at_bil_deg"),
                "reason": m3.get("reason"),
                "limit": ph.get("limit_factor"),
                "tokens": seq,
            }
            if m3.get("eccentric") and m3.get("downward_motion") and m3.get("beneficial_rotation"):
                # evaluate full with settle
                for extra in ([0, 0, 0], [SECURE_REVERSE_WRIST] * 12 + [0, 0, 0]):
                    toks = list(seq) + list(extra)
                    res = evaluate_full(env, table, book, blob, toks, record_obs=False)
                    row2 = {
                        **{k: row[k] for k in row if k != "tokens"},
                        "extra": len(extra),
                        "NEW": int(res["SUCCESS_NEW"]),
                        "s12": int(res["success_12s"]),
                        "win_th": res["window"].get("mean_theta"),
                        "win_pos": res["window"].get("mean_pos"),
                        "fail_reason": res["reason"],
                        "tokens": toks,
                    }
                    results.append(row2)
                    if res["SUCCESS_NEW"]:
                        ok2, tr2 = run_traced(env, table, book, blob, toks)
                        ph2 = analyze_capture_rotation_braking(tr2)
                        m32 = ph2["m3"]
                        print("HIT NEW", ic_id, i, "mech", m32.get("is_type3"),
                              "post", m32.get("post_regrasp_correction"),
                              "sup", ph2.get("supported_while_rotating"), flush=True)
                        best = {"row": row2, "m3": m32, "phases": ph2}
                        break
                    if (
                        ok and m3.get("bilateral_i") is not None
                        and (best is None or (row2.get("win_th") or 99) < (best["row"].get("win_th") or 99))
                    ):
                        best = {"row": row2, "m3": m3, "phases": ph}
                if best and best["row"].get("NEW"):
                    break
            elif i < 5 or (ok and m3.get("bilateral_i") is not None):
                results.append(row)
            if i % 25 == 0:
                print("  prog", i, "ok", ok, "dth", m3.get("delta_theta_deg"),
                      "bil", m3.get("bilateral_i"), "why", m3.get("reason"), flush=True)
        print("BEST", ic_id, None if not best else best["row"], flush=True)

    write_json(OUT / "RAW" / "focused_brake_probe.json", results)
    # summarize
    good = [r for r in results if r.get("ecc") and r.get("down") and r.get("ben")]
    print("mech_cands", len(good),
          "bil", sum(1 for r in good if r.get("bil")),
          "NEW", sum(1 for r in good if r.get("NEW")),
          "max_dth", max((r.get("dth") or -1) for r in good) if good else None)


if __name__ == "__main__":
    main()
