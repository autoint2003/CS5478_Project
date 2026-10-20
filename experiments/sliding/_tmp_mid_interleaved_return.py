"""Mid-grip interleaved return: reduce axial drag during tok48 rotation."""

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


def build_patterns():
    patterns = []
    # pure baseline
    patterns.append(("pure48x19", [SECURE_REVERSE_WRIST] * 19))
    # interleave mid unload
    for a, b in ((2, 1), (3, 1), (2, 2), (4, 2), (3, 2), (1, 1), (5, 2), (4, 1)):
        seq = []
        while len(seq) < 28:
            seq.extend([SECURE_REVERSE_WRIST] * a)
            seq.extend([12] * b)
        patterns.append((f"alt_{a}x48_{b}x12", seq[:28]))
    # mid then burst return
    for m in (3, 5, 8):
        for n in (12, 16, 19, 22):
            patterns.append((f"mid{m}_then48x{n}", [12] * m + [SECURE_REVERSE_WRIST] * n))
    # return chunk + mid + axial-oppose translate + retighten + continue
    for cl in (4, 6, 8):
        for cn in (4, 6, 8):
            patterns.append((
                f"chunk{cl}_midcorr{cn}",
                None,  # dynamic
                cl,
                cn,
            ))
    return patterns


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())

    rows = []
    best = None

    # static patterns
    static = [p for p in build_patterns() if p[1] is not None]
    print("STATIC_INTERLEAVE", len(static), flush=True)
    for name, ret_seq in static:
        tokens = CPP + SLIDE_BASE + RETIGHTEN + list(ret_seq) + RETIGHTEN
        # truncate if over deadline roughly: keep <= 90 tokens
        if len(tokens) > 100:
            tokens = tokens[:100]
        res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False)
        pr = res.get("pre_handoff_rich") or {}
        row = {
            "name": name,
            "NEW": int(res["SUCCESS_NEW"]),
            "s12": int(res["success_12s"]),
            "handoff_pos": res.get("e_pos_at_handoff"),
            "e_ax": pr.get("e_ax_mm"),
            "e_lat": pr.get("e_lat_mm"),
            "hand_ref": pr.get("hand_to_ref_deg"),
            "win_pos": res["window"].get("mean_pos"),
            "win_th": res["window"].get("mean_theta"),
            "hold": res["window"].get("longest_both_ok_s"),
            "n_tok": len(tokens),
            "tokens": json.dumps(tokens),
        }
        rows.append(row)
        print(
            f"  {name}: hand={row['hand_ref']} pos={row['handoff_pos']} ax={row['e_ax']} "
            f"NEW={row['NEW']} win_pos={row['win_pos']} win_th={row['win_th']}",
            flush=True,
        )
        if res["SUCCESS_NEW"]:
            best = row
            res2 = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=True)
            save_expert(env, "hard_910263", f"inter_{name}", res2)
            break

    # dynamic chunked with mid+axial correction
    if best is None:
        print("DYNAMIC_CHUNK", flush=True)
        for cl in (4, 6, 8):
            for nch in (2, 3):
                for cn in (4, 6, 8):
                    restore_capture(env, blob["capture"])
                    t0 = float(env.sim.data.time)
                    used = list(CPP + SLIDE_BASE + RETIGHTEN)
                    ok, lost = run_seq(env, table, book48, blob, used)
                    if not ok:
                        continue
                    dropped = False
                    for _ in range(nch):
                        for __ in range(cl):
                            st = rich_snapshot(env.sim, blob, "")
                            if st["hand_to_ref_deg"] <= 8:
                                break
                            ok, lost = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost)
                            used.append(SECURE_REVERSE_WRIST)
                            if not ok:
                                dropped = True
                                break
                        if dropped:
                            break
                        st = rich_snapshot(env.sim, blob, "")
                        # mid unload + translate against axial error
                        u = np.asarray(st["u_c_h"], float)
                        # e_ax negative => want +u object motion => hand -u under mid
                        desire = -u if st["e_ax_mm"] < -3 else -np.asarray(st["err_h"], float)
                        ranked = tokens_aligned(book48, desire, "secure", 4)
                        tok = ranked[0][1] if ranked else 33
                        corr = [12] * 3 + [tok] * cn + [0] * 2
                        ok, lost = run_seq(env, table, book48, blob, corr, lost)
                        used.extend(corr)
                        if not ok:
                            dropped = True
                            break
                    if dropped:
                        continue
                    # finish return
                    for __ in range(16):
                        st = rich_snapshot(env.sim, blob, "")
                        if st["hand_to_ref_deg"] <= 8 and st["rdes_to_ref_deg"] <= 8:
                            break
                        if float(env.sim.data.time) - t0 > 3.9:
                            break
                        ok, lost = run_seq(env, table, book48, blob, [SECURE_REVERSE_WRIST], lost)
                        used.append(SECURE_REVERSE_WRIST)
                        if not ok:
                            dropped = True
                            break
                    if dropped:
                        continue
                    used.extend(RETIGHTEN)
                    res = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=False)
                    pr = res.get("pre_handoff_rich") or {}
                    name = f"dyn_cl{cl}_n{nch}_cn{cn}"
                    row = {
                        "name": name,
                        "NEW": int(res["SUCCESS_NEW"]),
                        "s12": int(res["success_12s"]),
                        "handoff_pos": res.get("e_pos_at_handoff"),
                        "e_ax": pr.get("e_ax_mm"),
                        "e_lat": pr.get("e_lat_mm"),
                        "hand_ref": pr.get("hand_to_ref_deg"),
                        "win_pos": res["window"].get("mean_pos"),
                        "win_th": res["window"].get("mean_theta"),
                        "hold": res["window"].get("longest_both_ok_s"),
                        "n_tok": len(used),
                        "tokens": json.dumps(used),
                    }
                    rows.append(row)
                    print(
                        f"  {name}: hand={row['hand_ref']} pos={row['handoff_pos']} ax={row['e_ax']} "
                        f"NEW={row['NEW']} win_pos={row['win_pos']}",
                        flush=True,
                    )
                    if res["SUCCESS_NEW"]:
                        best = row
                        res2 = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=True)
                        save_expert(env, "hard_910263", f"inter_{name}", res2)
                        break
                if best:
                    break
            if best:
                break

    write_csv(OUT / "RAW" / "mid_interleaved_return.csv", rows)
    write_json(OUT / "RAW" / "mid_interleaved_best.json", best)

    s12 = [r for r in rows if r["s12"] and r["win_pos"] is not None]
    s12.sort(key=lambda r: (
        0 if r["NEW"] else 1,
        max(0.0, float(r["win_pos"]) - 12) + max(0.0, float(r["win_th"] or 99) - 10),
        float(r["handoff_pos"] or 99),
    ))
    print("TOP", flush=True)
    for r in s12[:12]:
        print(
            f"  {r['name']}: NEW={r['NEW']} win_pos={r['win_pos']} win_th={r['win_th']} "
            f"hand={r['hand_ref']} handoff={r['handoff_pos']} ax={r['e_ax']}",
            flush=True,
        )
    print("INTERLEAVE_DONE found", best is not None, flush=True)


if __name__ == "__main__":
    main()
