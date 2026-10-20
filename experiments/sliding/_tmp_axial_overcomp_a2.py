"""Focused axial overcompensation search for hard_910263 (K49)."""

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
from experiments.sliding.pilot_return_aware_sliding_k49 import (  # noqa: E402
    CPP,
    OUT,
    RETIGHTEN,
    SFT,
    SLIDE_BASE,
    SNAP,
    decompose,
    evaluate_tokens_vocab,
    rich_snapshot,
    return_until,
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
    rng = np.random.default_rng(20261018)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    blob = pickle.loads(SNAP.read_bytes())

    restore_capture(env, blob["capture"])
    run_seq(env, table, book48, blob, CPP)
    d0 = decompose(env.sim)
    print("post_CPP", d0["e_pos_mm"], "ax", d0["e_ax_mm"], "lat", d0["e_lat_mm"], flush=True)
    u = np.asarray(d0["u_c_h"], float)
    # hand motion toward -u => object lags along +u under mid grip => e_ax increases
    desire_hand = -u
    sec_plus = [t for _, t, *_ in tokens_aligned(book48, desire_hand, "secure", 10)]
    sec_minus = [t for _, t, *_ in tokens_aligned(book48, u, "secure", 6)]
    print("sec_plus", sec_plus, "sec_minus", sec_minus, flush=True)

    cands = []
    for t in sec_plus[:6]:
        for n_mid in (3, 5, 8, 12):
            for n_tr in (8, 12, 16, 20, 24, 28, 32):
                cands.append([12] * n_mid + [t] * n_tr + [0] * 5)
                cands.append([12] * n_mid + [t, 12] * (n_tr // 2) + [0] * 5)
                cands.append([12] * 2 + [t] * n_tr + [12] * 5 + [0] * 5)
    for o in (36, 39, 30):
        for t in sec_plus[:4]:
            cands.append([o] * 2 + [12] * 3 + [t] * 16 + [0] * 5)
            cands.append([o] * 3 + [t] * 20 + [0] * 5)
    for t in sec_minus[:3]:
        cands.append([12] * 5 + [t] * 16 + [0] * 5)

    seen, uniq = set(), []
    for s in cands:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    rng.shuffle(uniq)
    uniq = uniq[:80]
    print("n_cands", len(uniq), flush=True)

    rows = []
    best = None
    for i, slide in enumerate(uniq):
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        pref = CPP + slide + RETIGHTEN
        ok, _ = run_seq(env, table, book48, blob, pref)
        if not ok:
            rows.append({"i": i, "ok": 0})
            continue
        ps = decompose(env.sim)
        ok_r, ret, _, why = return_until(env, table, book48, blob, t0, max_steps=28)
        pr = decompose(env.sim)
        tokens = pref + ret
        res = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=False)
        row = {
            "i": i,
            "ok": 1,
            "e_pos_slide": ps["e_pos_mm"],
            "e_ax_slide": ps["e_ax_mm"],
            "e_lat_slide": ps["e_lat_mm"],
            "e_pos_ret": pr["e_pos_mm"],
            "e_ax_ret": pr["e_ax_mm"],
            "e_lat_ret": pr["e_lat_mm"],
            "drift": pr["e_pos_mm"] - ps["e_pos_mm"],
            "NEW": int(res["SUCCESS_NEW"]),
            "s12": int(res["success_12s"]),
            "win_pos": res["window"].get("mean_pos"),
            "win_th": res["window"].get("mean_theta"),
            "hold": res["window"].get("longest_both_ok_s"),
            "handoff_pos": res.get("e_pos_at_handoff"),
            "hand_ref": (res.get("pre_handoff_rich") or {}).get("hand_to_ref_deg"),
            "why": why,
            "n_tok": len(tokens),
            "tokens": json.dumps(tokens),
            "slide": json.dumps(slide),
        }
        rows.append(row)
        interesting = (
            res["SUCCESS_NEW"]
            or (ps["e_ax_mm"] > -4.0 and pr["e_pos_mm"] < 13.5)
            or i % 8 == 0
        )
        if interesting:
            print(
                f"A2[{i}] ax_s={ps['e_ax_mm']:.2f} pos_s={ps['e_pos_mm']:.2f} "
                f"ax_r={pr['e_ax_mm']:.2f} pos_r={pr['e_pos_mm']:.2f} "
                f"NEW={res['SUCCESS_NEW']} win={res['window'].get('mean_pos')}",
                flush=True,
            )
        if res["SUCCESS_NEW"]:
            best = row
            res2 = evaluate_tokens_vocab(env, table, book48, blob, tokens, record_obs=True)
            save_expert(env, "hard_910263", f"A2_axial_{i}", res2)
            print("FOUND_FULL", i, res["window"], flush=True)
            break

    valid = [r for r in rows if r.get("ok") and r.get("e_ax_slide") is not None]
    valid.sort(key=lambda r: (r["e_pos_ret"], -r["e_ax_slide"]))
    print("TOP_BY_POS_RET", flush=True)
    for r in valid[:12]:
        print(
            r["i"],
            "ax_s", round(r["e_ax_slide"], 2),
            "pos_s", round(r["e_pos_slide"], 2),
            "ax_r", round(r["e_ax_ret"], 2),
            "pos_r", round(r["e_pos_ret"], 2),
            "NEW", r["NEW"],
            "win", r.get("win_pos"),
            flush=True,
        )
    # also sort by least-negative axial after slide
    valid2 = sorted(valid, key=lambda r: -r["e_ax_slide"])
    print("TOP_BY_AX_SLIDE", flush=True)
    for r in valid2[:8]:
        print(
            r["i"],
            "ax_s", round(r["e_ax_slide"], 2),
            "pos_s", round(r["e_pos_slide"], 2),
            "pos_r", round(r["e_pos_ret"], 2),
            "NEW", r["NEW"],
            flush=True,
        )

    write_csv(OUT / "RAW" / "axial_overcomp_A2.csv", rows)
    write_json(OUT / "RAW" / "axial_overcomp_A2_best.json", best)
    print("A2_DONE best_new", best is not None, "n", len(rows), flush=True)

    # Phase B2: segmented return with axial-aware mid corrections from baseline slide
    print("PHASE_B2_SEGMENTED_AXIAL", flush=True)
    b2_rows = []
    b2_best = None
    chunk_plans = []
    for cl in (3, 5, 7):
        for nch in (2, 3, 4):
            for cn in (4, 6, 8, 10, 12):
                for t in (sec_plus[:4] or [33]):
                    chunk_plans.append((cl, nch, cn, t))
    rng.shuffle(chunk_plans)
    chunk_plans = chunk_plans[:40]
    for i, (cl, nch, cn, ct) in enumerate(chunk_plans):
        restore_capture(env, blob["capture"])
        t0 = float(env.sim.data.time)
        used = list(CPP + SLIDE_BASE + RETIGHTEN)
        ok, lost = run_seq(env, table, book48, blob, used)
        if not ok:
            continue
        dropped = False
        for _ch in range(nch):
            for _ in range(cl):
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
                break
            st = rich_snapshot(env.sim, blob, "")
            # correct if axial still too negative
            if st["e_ax_mm"] < -5.0 or st["e_pos_mm"] > 11.0:
                err = np.asarray(st["err_h"], float)
                ranked = tokens_aligned(book48, -err, "secure", 4)
                tok = ranked[0][1] if ranked else ct
                corr = [12] * 3 + [tok] * cn + [0] * 3
                ok, lost = run_seq(env, table, book48, blob, corr, lost)
                used.extend(corr)
                if not ok:
                    dropped = True
                    break
        if dropped:
            continue
        ok_r, ret, _, why = return_until(env, table, book48, blob, t0, max_steps=18)
        used.extend(ret)
        used.extend(RETIGHTEN)
        run_seq(env, table, book48, blob, RETIGHTEN, lost)
        res = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=False)
        pr = res.get("pre_handoff_rich") or {}
        row = {
            "i": i,
            "plan": f"cl{cl}_n{nch}_cn{cn}_ct{ct}",
            "NEW": int(res["SUCCESS_NEW"]),
            "s12": int(res["success_12s"]),
            "handoff_pos": res.get("e_pos_at_handoff"),
            "e_ax": pr.get("e_ax_mm"),
            "e_lat": pr.get("e_lat_mm"),
            "hand_ref": pr.get("hand_to_ref_deg"),
            "win_pos": res["window"].get("mean_pos"),
            "win_th": res["window"].get("mean_theta"),
            "hold": res["window"].get("longest_both_ok_s"),
            "tokens": json.dumps(used),
        }
        b2_rows.append(row)
        if i % 5 == 0 or res["SUCCESS_NEW"] or (res.get("e_pos_at_handoff") or 99) < 12.5:
            print(
                f"B2[{i}] {row['plan']} handoff={row['handoff_pos']} ax={row['e_ax']} "
                f"NEW={row['NEW']} win={row['win_pos']}",
                flush=True,
            )
        if res["SUCCESS_NEW"]:
            b2_best = row
            res2 = evaluate_tokens_vocab(env, table, book48, blob, used, record_obs=True)
            save_expert(env, "hard_910263", f"B2_seg_{i}", res2)
            print("FOUND_FULL_B2", i, flush=True)
            break

    write_csv(OUT / "RAW" / "segmented_axial_B2.csv", b2_rows)
    write_json(OUT / "RAW" / "segmented_axial_B2_best.json", b2_best)
    print("B2_DONE best_new", b2_best is not None, flush=True)


if __name__ == "__main__":
    main()
