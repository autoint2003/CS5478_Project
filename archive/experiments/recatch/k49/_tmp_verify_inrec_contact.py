import csv
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.generate_k49_sliding_recatch_dataset import (
    SFT,
    build_k49_table,
    classify_mechanisms,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book48 = load_codebook()
table = build_k49_table(book48, config_id="K49_tok48_wy0p25")
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

rows = list(csv.DictReader(open(
    "results/expert_generation/k49_in_recovery_recatch/IN_RECOVERY_EXPERTS.csv", encoding="utf-8")))
for r in rows[:3]:
    toks = json.loads(r["tokens"])
    print("ic", r["ic_id"], "toks", toks, "n", len(toks))
    blob = pickle.loads(Path(
        f"results/expert_generation/k49_in_recovery_recatch/snapshots/{r['ic_id']}.pkl"
    ).read_bytes())
    res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
    print(" NEW", res["SUCCESS_NEW"], "start", res["cs0"])
    for i, t in enumerate(res["trace"]):
        print(f"  {i:02d} tok={t['tok']:2d} nL={t['nL']} nR={t['nR']} bil={int(t['bilateral'])} "
              f"pos={t['e_pos_mm']:.2f} th={t['e_theta_deg']:.2f}")
    any_uni = any((t["nL"] + t["nR"]) > 0 and not t["bilateral"] for t in res["trace"])
    any_none = any((t["nL"] + t["nR"]) == 0 for t in res["trace"])
    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
    print(" any_uni", any_uni, "any_none", any_none, "mech", mech["observed_recovery"], mech["recatch_subtype"])
    print()
