"""Second-pass expert search on near-threshold Sliding ICs + reachable Recatch ICs."""
from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    CPP0,
    CPP_HARD263,
    CPP_HRV1,
    OUT,
    SECURE_REVERSE_WRIST,
    SFT,
    adaptive_return_variants,
    build_k49_table,
    classify_mechanisms,
    decode,
    decompose,
    describe_k49_config,
    evaluate_full,
    ic_snapshot_state,
    load_codebook,
    load_trained_k48,
    make_training_env,
    mapping_checksum,
    restore_capture,
    return_slow,
    run_seq,
    save_expert,
    tokens_aligned,
    write_csv,
    write_json,
)

SEED = 20261018
BUDGET_SLIDE = 40
BUDGET_RECATCH = 24
N_SLIDE = 20
N_RECATCH = 16


def deep_slide_prefs(book, desire, rng, ic_cpp=None):
    sec = tokens_aligned(book, desire, "secure", 8) or [17, 33, 13, 1, 5, 8]
    opp = tokens_aligned(book, -np.asarray(desire, float), "secure", 4) or [19, 24]
    cpps = [list(ic_cpp)] if ic_cpp else []
    cpps += [[], list(CPP0), list(CPP_HARD263), list(CPP_HRV1),
             [4] * 16 + [12] * 5 + [0] * 3,
             [6] * 12 + [12] * 8 + [0] * 5,
             [10] * 10 + [12] * 5 + [0] * 3]
    sufs = []
    for t in sec[:6]:
        for n in (6, 10, 14, 18, 22, 28):
            for n12 in (2, 4, 6, 8):
                sufs.append([12] * n12 + [t] * n + [0] * 4)
                sufs.append([12] * n12 + [t, 12] * max(1, n // 2) + [0] * 4)
                sufs.append([t] * n + [12] * n12 + [0] * 4)
    for t in (17, 33, 13, 1, 5, 8, 15, 16, 24):
        for n in (10, 16, 22):
            sufs.append([12] * 5 + [t] * n + [0] * 5)
    for t in opp[:3]:
        sufs.append([12] * 5 + [t] * 12 + [0] * 5)
    # light mid-only refine (already near threshold)
    sufs.append([12] * 8 + [0] * 5)
    sufs.append([0] * 8)
    seqs = []
    for cpp in cpps:
        for suf in sufs:
            seqs.append(list(cpp) + list(suf))
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head = uniq[:12]
    tail = uniq[12:]
    rng.shuffle(tail)
    return head + tail


def append_csv(path, rows, existing_ids=None):
    path = Path(path)
    old = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    ids = {(r.get("ic_id"), r.get("tag")) for r in old}
    for r in rows:
        key = (r.get("ic_id"), r.get("tag"))
        if key in ids:
            continue
        old.append(r)
        ids.add(key)
    write_csv(path, old)
    return old


def main():
    t0w = time.time()
    rng = np.random.default_rng(SEED + 99)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    meta["checksum_sha256_16"] = checksum
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    pools = json.loads((OUT / "RAW" / "ic_pools_raw.json").read_text(encoding="utf-8"))
    known = {"hard_910263", "hrv1_920031", "hard_910101"}

    # --- near-threshold sliding ICs ---
    slide_cands = []
    for e in pools["sliding_candidates"]:
        sp = Path(e["snapshot"])
        if not sp.exists() or e["id"] in known:
            continue
        blob = pickle.loads(sp.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        if st["bilateral"] and 7.0 <= st["e_pos_mm"] <= 15.0:
            slide_cands.append({**e, "st": st, "cpp": None})
    for hid in sorted((ROOT / "results/benchmarks/hard_recovery_v1/snapshots").glob("hrv1_*.pkl")):
        if hid.stem in known:
            continue
        blob = pickle.loads(hid.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        if st["bilateral"] and 7.0 <= st["e_pos_mm"] <= 15.0:
            slide_cands.append({"id": hid.stem, "snapshot": str(hid), "st": st,
                                "cpp": list(CPP_HRV1), "states": ["hrv1"]})
    # prefer smaller pos error, then smaller theta
    slide_cands.sort(key=lambda e: (e["st"]["e_pos_mm"], e["st"]["e_theta_deg"]))
    # dedupe
    seen, slide_ids = set(), []
    for e in slide_cands:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        slide_ids.append(e)
    slide_ids = slide_ids[:N_SLIDE]
    print(f"PASS2_SLIDE n={len(slide_ids)}", flush=True)

    sliding_new, recatch_new, mixed_new, failures, coverage = [], [], [], [], []
    manifest_add = []

    def accept(ic_id, tag, family, tokens, snap, blob, pool):
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
        if not res["SUCCESS_NEW"]:
            return None, res
        mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
        if family == "sliding" and not mech["sliding_verified"]:
            if "SLIDING" not in mech["observed_recovery"]:
                mech["observed_recovery"] = list(mech["observed_recovery"]) + ["POSITION_CORRECTION"]
        if family == "recatch" and mech["recatch_subtype"] is None:
            if (res["cs0"]["nL"] + res["cs0"]["nR"]) == 0:
                mech["recatch_subtype"] = "complete_loss"
                mech["observed_recovery"] = ["RECATCH_COMPLETE_LOSS"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
            elif not res["cs0"]["bilateral"]:
                mech["recatch_subtype"] = "single_sided"
                mech["observed_recovery"] = ["RECATCH_SINGLE_SIDED"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
        save_expert(env, ic_id, tag, family, res, mech, meta, checksum, snap)
        row = {
            "ic_id": ic_id, "tag": tag, "family": family, "SUCCESS_NEW": 1,
            "label": res["label"],
            "win_mean_pos": res["window"].get("mean_pos"),
            "win_mean_theta": res["window"].get("mean_theta"),
            "hold_both_s": res["window"].get("longest_both_ok_s"),
            "hold_margin_mm": res.get("hold_margin_mm"),
            "sliding_verified": int(mech["sliding_verified"]),
            "observed_recovery": json.dumps(mech["observed_recovery"]),
            "recatch_subtype": mech["recatch_subtype"],
            "source_disturbance": mech["source_disturbance"],
            "sliding_after": int("SLIDING" in mech["observed_recovery"]),
            "pivoting_after": int("PIVOTING" in mech["observed_recovery"]),
            "n_tokens": len(tokens), "config_id": CONFIG_ID,
            "mapping_checksum": checksum, "reproducible": 1,
            "tokens": json.dumps(tokens),
            "e_pos0": res["d0"]["e_pos_mm"], "e_pos_handoff": res.get("e_pos_at_handoff"),
            "start_nL": res["cs0"]["nL"], "start_nR": res["cs0"]["nR"],
            "pass": "second_near_threshold",
        }
        pool.append(row)
        manifest_add.append(row)
        if len(set(mech["observed_recovery"])) >= 2:
            mixed_new.append(row)
        return row, res

    for ie, ic in enumerate(slide_ids):
        t0b = time.time()
        snap = Path(ic["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic['id']}.pkl")
        st0 = ic["st"]
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        props = deep_slide_prefs(book48, desire, rng, ic.get("cpp"))[:BUDGET_SLIDE]
        found = None
        n_try = 0
        best_partial = None
        for ip, pref in enumerate(props):
            if found:
                break
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            ok, _ = run_seq(env, table, book48, blob, pref)
            if not ok:
                n_try += 1
                failures.append({"ic_id": ic["id"], "tag": f"p2_slide_{ip}", "reason": "DROP_PREFIX"})
                continue
            for vname, tokens in adaptive_return_variants(env, table, book48, blob, t0, pref):
                n_try += 1
                row, res = accept(ic["id"], f"p2_{ip}_{vname}", "sliding", tokens, snap, blob, sliding_new)
                if row is not None:
                    found = row
                    print(f"  P2_SLIDE {ic['id']} NEW=1 win={row['win_mean_pos']:.2f} via {vname}", flush=True)
                    break
                wp = res["window"].get("mean_pos")
                if res.get("success_12s") and wp is not None:
                    if best_partial is None or wp < best_partial[0]:
                        best_partial = (wp, res["window"].get("mean_theta"), vname, res["reason"])
                failures.append({"ic_id": ic["id"], "tag": f"p2_{ip}_{vname}", "family": "sliding",
                                 "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                                 "win_mean_pos": wp})
                if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                    break
        coverage.append({
            "ic_id": ic["id"], "pool": "slide_pass2", "searched": n_try,
            "SUCCESS_NEW": int(found is not None),
            "e_pos0": st0["e_pos_mm"], "e_theta0": st0["e_theta_deg"],
            "bilateral0": 1, "budget_s": time.time() - t0b,
            "best_partial": None if best_partial is None else list(best_partial),
        })
        print(f"  [{ie+1}/{len(slide_ids)}] {ic['id']} pos0={st0['e_pos_mm']:.1f} "
              f"tries={n_try} found={found is not None} bestP={best_partial}", flush=True)

    # --- Recatch: prefer unilateral with object still near (small e_pos or low z drop) ---
    print("PASS2_RECATCH", flush=True)
    rec_cands = []
    for e in pools["unilateral"]:
        sp = Path(e["snapshot"])
        if not sp.exists():
            continue
        blob = pickle.loads(sp.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        any_c = st["nL"] + st["nR"] > 0
        if (not st["bilateral"]) and any_c and st["e_pos_mm"] < 40:
            rec_cands.append({**e, "st": st, "subtype": "single"})
    for e in pools["complete_loss"]:
        sp = Path(e["snapshot"])
        if not sp.exists():
            continue
        blob = pickle.loads(sp.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        if st["nL"] + st["nR"] == 0 and st["e_pos_mm"] < 50:
            rec_cands.append({**e, "st": st, "subtype": "complete"})
    rec_cands.sort(key=lambda e: (0 if e["subtype"] == "single" else 1, e["st"]["e_pos_mm"]))
    seen, rec_ids = set(), []
    for e in rec_cands:
        if e["id"] in seen:
            continue
        seen.add(e["id"])
        rec_ids.append(e)
    rec_ids = rec_ids[:N_RECATCH]
    print(f"PASS2_RECATCH selected={len(rec_ids)}", flush=True)

    for ie, ic in enumerate(rec_ids):
        t0b = time.time()
        snap = Path(ic["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic['id']}.pkl")
        st0 = ic["st"]
        subtype = ic["subtype"]
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        sec = tokens_aligned(book48, desire, "secure", 6) or [1, 5, 13, 17]
        props = []
        if subtype == "single":
            for t in sec[:5]:
                for n in (8, 12, 16, 20):
                    props.append([12] * 3 + [t] * n + [0] * 5)
                    props.append([t] * n + [12] * 5 + [0] * 5)
                    props.append([36] * 2 + [t] * n + [0] * 5)
                    props.append([30] * 2 + [12] * 2 + [t] * n + [0] * 5)
                    props.append([4] * 10 + [12] * 4 + [t] * 12 + [0] * 5)
        else:
            for o in (30, 36, 39, 32):
                for t in sec[:4]:
                    for n_o in (1, 2, 3):
                        for n_c in (10, 14, 18):
                            props.append([o] * n_o + [t] * n_c + [0] * 5)
                            props.append([o] * n_o + [12] * 2 + [t] * n_c + [0] * 5)
        rng.shuffle(props)
        props = props[:BUDGET_RECATCH]
        found = None
        n_try = 0
        for ip, pref in enumerate(props):
            if found:
                break
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            ok, _ = run_seq(env, table, book48, blob, pref)
            if not ok:
                n_try += 1
                failures.append({"ic_id": ic["id"], "tag": f"p2_rec_{ip}", "reason": "DROP_PREFIX",
                                 "subtype": subtype})
                continue
            dmid = decompose(env.sim)
            extra = []
            if dmid["bilateral"] and dmid["e_pos_mm"] > 12:
                sec2 = tokens_aligned(book48, -dmid["err_h"], "secure", 2)
                if sec2:
                    extra = [12] * 3 + [sec2[0]] * 12 + [0] * 3
            base = list(pref) + list(extra)
            for vname, tokens in adaptive_return_variants(env, table, book48, blob, t0, base):
                n_try += 1
                row, res = accept(ic["id"], f"p2_rec_{subtype}_{ip}_{vname}", "recatch",
                                  tokens, snap, blob, recatch_new)
                if row is not None:
                    found = row
                    print(f"  P2_RECATCH {ic['id']} {subtype} NEW=1 win={row['win_mean_pos']:.2f}", flush=True)
                    break
                failures.append({"ic_id": ic["id"], "tag": f"p2_rec_{ip}_{vname}", "family": "recatch",
                                 "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"],
                                 "subtype": subtype})
                if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                    break
        coverage.append({
            "ic_id": ic["id"], "pool": "recatch_pass2", "subtype": subtype,
            "searched": n_try, "SUCCESS_NEW": int(found is not None),
            "e_pos0": st0["e_pos_mm"], "bilateral0": int(st0["bilateral"]),
            "nL0": st0["nL"], "nR0": st0["nR"], "budget_s": time.time() - t0b,
        })
        print(f"  [{ie+1}/{len(rec_ids)}] {ic['id']} {subtype} tries={n_try} found={found is not None}", flush=True)

    # merge into existing CSVs
    sliding_all = append_csv(OUT / "SLIDING_EXPERTS.csv", sliding_new)
    recatch_all = append_csv(OUT / "RECATCH_EXPERTS.csv", recatch_new)
    mixed_all = append_csv(OUT / "MIXED_EXPERTS.csv", mixed_new)
    # coverage + failures append
    cov_old = list(csv.DictReader((OUT / "IC_COVERAGE.csv").open(encoding="utf-8"))) if (OUT / "IC_COVERAGE.csv").exists() else []
    write_csv(OUT / "IC_COVERAGE.csv", cov_old + coverage)
    fail_old = list(csv.DictReader((OUT / "FAILURES.csv").open(encoding="utf-8"))) if (OUT / "FAILURES.csv").exists() else []
    write_csv(OUT / "FAILURES.csv", (fail_old + failures)[:800])

    man_old = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8"))) if (OUT / "UNIFIED_MANIFEST.csv").exists() else []
    manifest = append_csv(OUT / "UNIFIED_MANIFEST.csv", man_old + manifest_add)

    write_json(OUT / "RAW" / "pass2_summary.json", {
        "slide_new": len(sliding_new),
        "recatch_new": len(recatch_new),
        "mixed_new": len(mixed_new),
        "slide_ics": sorted({r["ic_id"] for r in sliding_new}),
        "recatch_ics": sorted({r["ic_id"] for r in recatch_new}),
        "wall_s": time.time() - t0w,
        "n_slide_searched": len(slide_ids),
        "n_recatch_searched": len(rec_ids),
    })
    print("PASS2_DONE",
          f"slide_new={len(sliding_new)}",
          f"rec_new={len(recatch_new)}",
          f"manifest={len(manifest)}",
          flush=True)


if __name__ == "__main__":
    main()
