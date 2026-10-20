"""Pass-3 Recatch expert search: seed from known F_recatch+CPP pattern; mine gen/unilateral ICs."""
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
    OUT,
    SFT,
    adaptive_return_variants,
    build_k49_table,
    classify_mechanisms,
    decompose,
    describe_k49_config,
    evaluate_full,
    ic_snapshot_state,
    load_codebook,
    load_trained_k48,
    make_training_env,
    mapping_checksum,
    restore_capture,
    run_seq,
    save_expert,
    tokens_aligned,
    write_csv,
    write_json,
    SECURE_REVERSE_WRIST,
)

SEED = 20261019


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
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    meta["checksum_sha256_16"] = checksum
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    # Known F_recatch seed
    f_tok = [int(t) for t in np.load(
        ROOT / "results/expert_generation/sliding_position_correction_v1/trajectories/"
        "gen_hrv1_920031_1_74e897d20146__F_recatch__tokens.npz"
    )["tokens"] if int(t) != 47]
    pivots = [
        [4] * 24 + [12] * 5 + [0] * 3,
        [4] * 20 + [12] * 5 + [0] * 3,
        [6] * 20 + [12] * 8 + [0] * 5,
        [10] * 16 + [12] * 5 + [0] * 3,
        [4] * 12 + [6] * 12 + [12] * 5 + [0] * 3,
    ]

    # Candidate ICs: gen snapshots + unilateral near-reachable
    cands = []
    for sp in [
        ROOT / "results/expert_generation/sliding_position_correction_v1/SNAPSHOTS/gen_hrv1_920031_1_74e897d20146.pkl",
        ROOT / "results/expert_generation/grasp_recovery_pose_v1/SNAPSHOTS/gen_hrv1_920031_1_74e897d20146.pkl",
    ]:
        if sp.exists():
            cands.append({"id": "gen_hrv1_920031_1_74e897d20146", "snapshot": str(sp), "subtype": "seed_complete"})

    # other gen large-tilt / pose gens
    for folder in [
        ROOT / "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots",
        ROOT / "results/expert_generation/grasp_recovery_pose_v1/SNAPSHOTS",
        ROOT / "results/expert_generation/sliding_position_correction_v1/SNAPSHOTS",
    ]:
        if not folder.exists():
            continue
        for sp in folder.glob("gen_*.pkl"):
            cands.append({"id": sp.stem, "snapshot": str(sp), "subtype": "gen"})

    pools = json.loads((OUT / "RAW" / "ic_pools_raw.json").read_text(encoding="utf-8"))
    for e in pools["unilateral"][:40]:
        cands.append({**e, "subtype": "single"})
    for e in pools["complete_loss"][:30]:
        cands.append({**e, "subtype": "complete"})

    seen, uniq = set(), []
    for e in cands:
        if e["id"] in seen:
            continue
        if not Path(e["snapshot"]).exists():
            continue
        seen.add(e["id"])
        uniq.append(e)

    # prioritize seed gen, then gens, then unilateral
    def rank(e):
        if e["id"].startswith("gen_hrv1_920031"):
            return 0
        if e["id"].startswith("gen_"):
            return 1
        if e.get("subtype") == "single":
            return 2
        return 3

    uniq.sort(key=rank)
    uniq = uniq[:22]
    print(f"PASS3_RECATCH n={len(uniq)}", flush=True)

    recatch_new, mixed_new, failures, coverage, manifest_add = [], [], [], [], []

    def accept(ic_id, tag, tokens, snap, blob, st0):
        res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
        if not res["SUCCESS_NEW"]:
            return None, res
        mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
        # require actual recatch evidence
        if mech["recatch_subtype"] is None:
            # allow if sustained loss during traj even if start bilateral (gen seed case)
            if not (mech["evidence"].get("had_loss_both") or mech["evidence"].get("had_unilateral")):
                return None, res
            if mech["evidence"].get("had_loss_both"):
                mech["recatch_subtype"] = "complete_loss"
                mech["observed_recovery"] = ["RECATCH_COMPLETE_LOSS"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
            else:
                mech["recatch_subtype"] = "single_sided"
                mech["observed_recovery"] = ["RECATCH_SINGLE_SIDED"] + [
                    x for x in mech["observed_recovery"] if "RECATCH" not in x]
        save_expert(env, ic_id, tag, "recatch", res, mech, meta, checksum, snap)
        row = {
            "ic_id": ic_id, "tag": tag, "family": "recatch", "SUCCESS_NEW": 1,
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
            "pass": "pass3_recatch",
        }
        recatch_new.append(row)
        manifest_add.append(row)
        if row["sliding_after"] or row["pivoting_after"] or len(json.loads(row["observed_recovery"])) >= 2:
            mixed_new.append(row)
        return row, res

    for ie, ic in enumerate(uniq):
        t0b = time.time()
        snap = Path(ic["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic['id']}.pkl")
        st0 = ic_snapshot_state(env, table, book48, blob)
        found = None
        n_try = 0
        props = []

        # Seed pattern for gen / known
        if ic["id"].startswith("gen_") or ic.get("subtype") == "seed_complete":
            for piv in pivots:
                props.append(list(f_tok) + list(piv) + [0, 0, 0])
                props.append(list(piv) + list(f_tok) + [0, 0, 0])
                for n48 in (0, 12, 24):
                    props.append(list(f_tok) + list(piv) + [SECURE_REVERSE_WRIST] * n48 + [0, 0, 0])

        # Local proposals from contact state
        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]
        sec = tokens_aligned(book48, desire, "secure", 5) or [1, 5, 13, 17]
        if (st0["nL"] + st0["nR"]) == 0 or not st0["bilateral"]:
            for o in (30, 36, 39, 32):
                for t in sec[:4]:
                    for n_o in (1, 2, 3):
                        for n_c in (10, 14, 18):
                            for piv in ([[], [4] * 16 + [12] * 4 + [0] * 3, [6] * 12 + [12] * 4 + [0] * 3]):
                                props.append([o] * n_o + [t] * n_c + list(piv) + [0] * 4)
        else:
            # bilateral start but may lose contact during open-heavy recovery
            for o in (36, 30, 39):
                for t in sec[:3]:
                    props.append([o] * 3 + [t] * 12 + [4] * 20 + [12] * 5 + [0] * 3)
                    props.append([o] * 2 + [t] * 10 + [6] * 16 + [12] * 5 + [0] * 3)

        # dedupe / bound
        seen_p, uniq_p = set(), []
        for p in props:
            k = tuple(p)
            if k in seen_p:
                continue
            seen_p.add(k)
            uniq_p.append(p)
        rng.shuffle(uniq_p)
        # keep seed patterns first for gen
        if ic["id"].startswith("gen_"):
            head = [p for p in props[:15]]
            uniq_p = head + [p for p in uniq_p if tuple(p) not in {tuple(h) for h in head}]
        uniq_p = uniq_p[:28]

        for ip, pref in enumerate(uniq_p):
            if found:
                break
            restore_capture(env, blob["capture"])
            t0 = float(env.sim.data.time)
            ok, _ = run_seq(env, table, book48, blob, pref)
            if not ok:
                n_try += 1
                failures.append({"ic_id": ic["id"], "tag": f"p3_{ip}", "reason": "DROP_PREFIX"})
                continue
            # try pref as-is and with adaptive return
            variants = [("raw", list(pref) + [0, 0, 0])]
            variants += adaptive_return_variants(env, table, book48, blob, t0, pref)
            for vname, tokens in variants:
                n_try += 1
                row, res = accept(ic["id"], f"p3_{ip}_{vname}", tokens, snap, blob, st0)
                if row is not None:
                    found = row
                    print(f"  P3_RECATCH {ic['id']} subtype={row['recatch_subtype']} "
                          f"NEW=1 win={row['win_mean_pos']:.2f} via {vname}", flush=True)
                    break
                failures.append({"ic_id": ic["id"], "tag": f"p3_{ip}_{vname}", "family": "recatch",
                                 "SUCCESS_NEW": 0, "label": res["label"], "reason": res["reason"]})
                if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                    break

        coverage.append({
            "ic_id": ic["id"], "pool": "recatch_pass3", "subtype": ic.get("subtype"),
            "searched": n_try, "SUCCESS_NEW": int(found is not None),
            "e_pos0": st0["e_pos_mm"], "bilateral0": int(st0["bilateral"]),
            "nL0": st0["nL"], "nR0": st0["nR"], "budget_s": time.time() - t0b,
        })
        print(f"  [{ie+1}/{len(uniq)}] {ic['id']} tries={n_try} found={found is not None}", flush=True)

    recatch_all = append_csv(OUT / "RECATCH_EXPERTS.csv", recatch_new)
    mixed_all = append_csv(OUT / "MIXED_EXPERTS.csv", mixed_new)
    man_old = list(csv.DictReader((OUT / "UNIFIED_MANIFEST.csv").open(encoding="utf-8"))) if (OUT / "UNIFIED_MANIFEST.csv").exists() else []
    manifest = append_csv(OUT / "UNIFIED_MANIFEST.csv", man_old + manifest_add)
    cov_old = list(csv.DictReader((OUT / "IC_COVERAGE.csv").open(encoding="utf-8"))) if (OUT / "IC_COVERAGE.csv").exists() else []
    write_csv(OUT / "IC_COVERAGE.csv", cov_old + coverage)

    write_json(OUT / "RAW" / "pass3_summary.json", {
        "recatch_new": len(recatch_new),
        "recatch_ics": sorted({r["ic_id"] for r in recatch_new}),
        "by_subtype": {
            "single_sided": sorted({r["ic_id"] for r in recatch_new if r.get("recatch_subtype") == "single_sided"}),
            "complete_loss": sorted({r["ic_id"] for r in recatch_new if r.get("recatch_subtype") == "complete_loss"}),
        },
        "wall_s": time.time() - t0w,
    })
    print("PASS3_DONE", f"rec_new={len(recatch_new)}", f"manifest={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
