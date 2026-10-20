"""Mine single-sided Recatch experts from mid-traj unilateral snapshots of verified experts."""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    OUT,
    SFT,
    build_k49_table,
    classify_mechanisms,
    contact_summary,
    decode,
    describe_k49_config,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
    mapping_checksum,
    save_expert,
    step_legal,
    write_csv,
    write_json,
)


def append_csv(path, rows):
    path = Path(path)
    old = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
    keys = {(r.get("ic_id"), r.get("tag")) for r in old}
    for r in rows:
        k = (r.get("ic_id"), r.get("tag"))
        if k in keys:
            continue
        old.append(r)
        keys.add(k)
    write_csv(path, old)
    return old


def main():
    t0w = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    meta["checksum_sha256_16"] = checksum
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    # Use verified full experts as source rollouts
    man = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8")))
    new_rows, mixed, failures = [], [], []

    for row in man:
        ic = row["ic_id"]
        toks = json.loads(row["tokens"])
        snap = OUT / "snapshots" / f"{ic}.pkl"
        if not snap.exists():
            # try alternate
            alts = list((OUT / "snapshots").glob(f"{ic}*.pkl"))
            if not alts:
                print("no snap", ic, flush=True)
                continue
            snap = alts[0]
        blob = pickle.loads(snap.read_bytes())
        restore_capture(env, blob["capture"])
        # walk tokens, capture first sustained unilateral (>=3 steps)
        uni_run = 0
        uni_start = None
        captures = []
        for i, tok in enumerate(toks):
            cs = contact_summary(env.sim)
            any_c = int(cs["nL"]) + int(cs["nR"]) > 0
            uni = any_c and not cs["bilateral"]
            none = not any_c
            if uni:
                uni_run += 1
                if uni_run == 3 and uni_start is None:
                    uni_start = i - 2
                    # capture state NOW (at 3rd uni frame) before stepping further
                    cap = {
                        "capture": take_capture(env),
                        "ref": blob["ref"],
                        "src_ic": ic,
                        "src_tok_idx": i,
                        "kind": "unilateral",
                    }
                    captures.append(cap)
            elif none:
                uni_run = 0
            else:
                uni_run = 0
            action = np.asarray(step_legal(env.sim, decode(table, book48, tok)), np.float32)
            env._prev = action.copy()

        if not captures:
            print(f"  {ic}: no sustained unilateral mid-traj", flush=True)
            continue

        # For each captured unilateral state, remaining tokens as candidate
        for ci, cap in enumerate(captures[:2]):
            idx = int(cap["src_tok_idx"])
            remain = [int(t) for t in toks[idx + 1 :] if int(t) != 47]
            if len(remain) < 5:
                continue
            # also try remain + short pivot/return variants
            variants = [("remain", remain + [0, 0, 0])]
            for piv in ([4] * 12 + [12] * 4 + [0] * 3, [6] * 10 + [12] * 4 + [0] * 3, []):
                variants.append((f"remain_piv{len(piv)}", remain + list(piv) + [0, 0, 0]))

            new_id = f"{ic}_uni{ci}_t{idx}"
            # save snapshot
            snap_path = OUT / "snapshots" / f"{new_id}.pkl"
            with snap_path.open("wb") as f:
                pickle.dump({"capture": cap["capture"], "ref": cap["ref"],
                             "parent_ic": ic, "parent_tok_idx": idx, "kind": "unilateral_mined"}, f)

            mined_blob = {"capture": cap["capture"], "ref": cap["ref"]}
            found = False
            for vname, tokens in variants:
                # verify start is unilateral
                restore_capture(env, mined_blob["capture"])
                cs0 = contact_summary(env.sim)
                if cs0["bilateral"] or (cs0["nL"] + cs0["nR"]) == 0:
                    # skip if not single-sided
                    continue
                res = evaluate_full(env, table, book48, mined_blob, tokens, record_obs=True)
                if not res["SUCCESS_NEW"]:
                    failures.append({"ic_id": new_id, "tag": vname, "reason": res["reason"],
                                     "label": res["label"], "parent": ic})
                    continue
                mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                if mech["recatch_subtype"] is None:
                    mech["recatch_subtype"] = "single_sided"
                    mech["observed_recovery"] = ["RECATCH_SINGLE_SIDED"] + [
                        x for x in mech["observed_recovery"] if "RECATCH" not in x]
                tag = f"mined_uni_{vname}"
                save_expert(env, new_id, tag, "recatch", res, mech, meta, checksum, snap_path)
                out = {
                    "ic_id": new_id, "tag": tag, "family": "recatch", "SUCCESS_NEW": 1,
                    "label": res["label"],
                    "win_mean_pos": res["window"].get("mean_pos"),
                    "win_mean_theta": res["window"].get("mean_theta"),
                    "hold_both_s": res["window"].get("longest_both_ok_s"),
                    "hold_margin_mm": res.get("hold_margin_mm"),
                    "sliding_verified": int(mech["sliding_verified"]),
                    "observed_recovery": json.dumps(mech["observed_recovery"]),
                    "recatch_subtype": "single_sided",
                    "source_disturbance": mech["source_disturbance"],
                    "sliding_after": int("SLIDING" in mech["observed_recovery"]),
                    "pivoting_after": int("PIVOTING" in mech["observed_recovery"]),
                    "n_tokens": len(tokens), "config_id": CONFIG_ID,
                    "mapping_checksum": checksum, "reproducible": 1,
                    "tokens": json.dumps(tokens),
                    "e_pos0": res["d0"]["e_pos_mm"], "e_pos_handoff": res.get("e_pos_at_handoff"),
                    "start_nL": res["cs0"]["nL"], "start_nR": res["cs0"]["nR"],
                    "parent_ic": ic, "pass": "pass4_mined_unilateral",
                }
                new_rows.append(out)
                if out["sliding_after"] or out["pivoting_after"]:
                    mixed.append(out)
                found = True
                print(f"  MINED_UNI {new_id} NEW=1 parent={ic} win={out['win_mean_pos']:.2f}", flush=True)
                break
            if not found:
                print(f"  {ic} uni@{idx}: no SUCCESS_NEW from remain", flush=True)

    append_csv(OUT / "RECATCH_EXPERTS.csv", new_rows)
    append_csv(OUT / "MIXED_EXPERTS.csv", mixed)
    man_old = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8")))
    append_csv(OUT / "UNIFIED_MANIFEST.csv", man_old + new_rows)
    write_json(OUT / "RAW" / "pass4_summary.json", {
        "n_new": len(new_rows),
        "ics": [r["ic_id"] for r in new_rows],
        "parents": sorted({r["parent_ic"] for r in new_rows}),
        "wall_s": time.time() - t0w,
    })
    print("PASS4_DONE", len(new_rows), flush=True)


if __name__ == "__main__":
    main()
