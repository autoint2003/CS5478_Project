"""Partial-return sweep + slide×return grid for hard_910263."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture
from models.k48_checkpoint import load_trained_k48
from experiments.sliding.pilot_return_aware_sliding_k49 import (
    CPP,
    OUT,
    RETIGHTEN,
    SFT,
    SLIDE_BASE,
    SNAP,
    decompose,
    evaluate_tokens_vocab,
    pack_row,
    rich_snapshot,
    run_seq,
    save_expert,
    tokens_aligned,
    write_csv,
    write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env


def main():
    rng = np.random.default_rng(20261019)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())

    # --- Partial return after baseline slide ---
    print("PARTIAL_RETURN_SWEEP", flush=True)
    rows = []
    best = None
    for n_ret in list(range(0, 26, 2)) + [19, 21, 23]:
        tokens = CPP + SLIDE_BASE + RETIGHTEN + [SECURE_REVERSE_WRIST] * n_ret + RETIGHTEN
        res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False)
        pr = res.get("pre_handoff_rich") or {}
        row = {
            "n_ret": n_ret,
            "NEW": int(res["SUCCESS_NEW"]),
            "s12": int(res["success_12s"]),
            "handoff_pos": res.get("e_pos_at_handoff"),
            "handoff_th": res.get("e_theta_at_handoff"),
            "hand_ref": pr.get("hand_to_ref_deg"),
            "rdes_ref": pr.get("rdes_to_ref_deg"),
            "e_ax": pr.get("e_ax_mm"),
            "e_lat": pr.get("e_lat_mm"),
            "win_pos": res["window"].get("mean_pos"),
            "win_th": res["window"].get("mean_theta"),
            "hold": res["window"].get("longest_both_ok_s"),
            "reason": res["reason"],
            "tokens": json.dumps(tokens),
        }
        rows.append(row)
        print(
            f"  n={n_ret:2d} hand={row['hand_ref']} pos_h={row['handoff_pos']} "
            f"ax={row['e_ax']} NEW={row['NEW']} win_pos={row['win_pos']} win_th={row['win_th']} hold={row['hold']}",
            flush=True,
        )
        if res["SUCCESS_NEW"]:
            best = row
            res2 = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=True)
            save_expert(env, "hard_910263", f"partial_ret_{n_ret}", res2)
            break

    write_csv(OUT / "RAW" / "partial_return_sweep.csv", rows)
    write_json(OUT / "RAW" / "partial_return_best.json", best)

    # Pareto-ish: among S12, minimize max(win_pos-12,0)+max(win_th-10,0) with low hand_ref
    s12 = [r for r in rows if r["s12"] and r["win_pos"] is not None]
    if s12:
        def score(r):
            return (
                0 if r["NEW"] else 1,
                max(0.0, float(r["win_pos"]) - 12.0) + max(0.0, float(r["win_th"]) - 10.0),
                float(r["hand_ref"] or 99),
            )
        s12.sort(key=score)
        print("BEST_PARTIAL", s12[0], flush=True)

    # --- Grid: alternate slides × partial return counts ---
    print("SLIDE_X_PARTIAL_GRID", flush=True)
    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, CPP)
    d0 = decompose(env.sim)
    desire = -np.asarray(d0["err_h"], float)
    sec = [t for _, t, *_ in tokens_aligned(book48, desire, "secure", 8)]
    slides = [list(SLIDE_BASE)]
    for t in sec[:5]:
        for n in (12, 16, 20, 24):
            slides.append([12] * 5 + [t] * n + [0] * 5)
            slides.append([12] * 5 + [t, 12] * (n // 2) + [0] * 5)
    # also tok33 variants (often axial-ish in world)
    for n in (8, 12, 16, 20):
        slides.append([12] * 5 + [33] * n + [0] * 5)
        slides.append([12] * 3 + [33, 12] * (n // 2) + [0] * 5)

    seen, uniq = set(), []
    for s in slides:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    uniq = uniq[:30]
    ret_ns = [0, 4, 6, 8, 10, 12, 14, 16, 19]
    grid = []
    found = None
    for i, slide in enumerate(uniq):
        for n_ret in ret_ns:
            tokens = CPP + slide + RETIGHTEN + [SECURE_REVERSE_WRIST] * n_ret + RETIGHTEN
            res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False)
            pr = res.get("pre_handoff_rich") or {}
            row = {
                "slide_i": i,
                "n_ret": n_ret,
                "NEW": int(res["SUCCESS_NEW"]),
                "s12": int(res["success_12s"]),
                "handoff_pos": res.get("e_pos_at_handoff"),
                "hand_ref": pr.get("hand_to_ref_deg"),
                "e_ax": pr.get("e_ax_mm"),
                "e_lat": pr.get("e_lat_mm"),
                "win_pos": res["window"].get("mean_pos"),
                "win_th": res["window"].get("mean_theta"),
                "hold": res["window"].get("longest_both_ok_s"),
                "slide": json.dumps(slide),
                "tokens": json.dumps(tokens),
            }
            grid.append(row)
            if res["SUCCESS_NEW"] or (
                res["success_12s"]
                and res["window"].get("mean_pos") is not None
                and float(res["window"]["mean_pos"]) <= 12.5
                and float(res["window"].get("mean_theta") or 99) <= 10.5
            ):
                print(
                    f"  HIT slide={i} n_ret={n_ret} NEW={row['NEW']} "
                    f"win_pos={row['win_pos']} win_th={row['win_th']} hand={row['hand_ref']} "
                    f"handoff={row['handoff_pos']}",
                    flush=True,
                )
            if res["SUCCESS_NEW"]:
                found = row
                res2 = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=True)
                save_expert(env, "hard_910263", f"grid_s{i}_r{n_ret}", res2)
                break
        if found:
            break
        if i % 5 == 0:
            # print best so far for this slide across ret
            sub = [r for r in grid if r["slide_i"] == i and r["s12"] and r["win_pos"] is not None]
            if sub:
                b = min(sub, key=lambda r: (
                    max(0, float(r["win_pos"]) - 12) + max(0, float(r["win_th"] or 99) - 10),
                    float(r["handoff_pos"] or 99),
                ))
                print(
                    f"  slide[{i}] best n_ret={b['n_ret']} win_pos={b['win_pos']} "
                    f"win_th={b['win_th']} hand={b['hand_ref']} hold={b['hold']}",
                    flush=True,
                )

    write_csv(OUT / "RAW" / "slide_partial_grid.csv", grid)
    write_json(OUT / "RAW" / "slide_partial_grid_best.json", found)

    # Closest misses
    okish = [r for r in grid if r["s12"] and r["win_pos"] is not None]
    okish.sort(key=lambda r: (
        0 if r["NEW"] else 1,
        max(0.0, float(r["win_pos"]) - 12.0) + max(0.0, float(r["win_th"] or 99) - 10.0),
        float(r["hand_ref"] or 99),
    ))
    print("TOP_GRID", flush=True)
    for r in okish[:15]:
        print(
            f"  s={r['slide_i']} n={r['n_ret']} NEW={r['NEW']} "
            f"win_pos={r['win_pos']} win_th={r['win_th']} hold={r['hold']} "
            f"hand={r['hand_ref']} handoff={r['handoff_pos']}",
            flush=True,
        )
    print("PARTIAL_DONE found", found is not None, flush=True)


if __name__ == "__main__":
    main()
