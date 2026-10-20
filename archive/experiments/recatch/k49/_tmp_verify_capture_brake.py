import csv
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from scripts.generate_k49_dynamic_recatch_experts import analyze_type3_mechanism, run_traced
from scripts.generate_k49_horizontal_endcatch_recatch import analyze_capture_rotation_braking
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
)
from scripts.generate_k49_type3_capture_brake import OUT

rows = list(csv.DictReader((OUT / "FULL_EXPERTS.csv").open(encoding="utf-8")))
print("FULL rows", len(rows))
for r in rows:
    print(r)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book = load_codebook()
table = build_k49_table(book, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

for r in rows:
    toks = json.loads(r["tokens"])
    blob = pickle.loads((OUT / "snapshots" / f"{r['ic_id']}.pkl").read_bytes())
    res = evaluate_full(env, table, book, blob, toks)
    ok, tr = run_traced(env, table, book, blob, toks)
    m3 = analyze_type3_mechanism(tr)
    ph = analyze_capture_rotation_braking(tr)
    fc = m3.get("first_contact_i")
    bil = m3.get("bilateral_i")
    rot_window = None
    if fc is not None:
        end_r = bil if bil is not None else min(len(tr) - 1, fc + 20)
        ths = [float(tr[i]["e_theta_deg"]) for i in range(fc, end_r + 1)]
        rot_window = ths[0] - min(ths) if ths else None
    print("REPRO", r["ic_id"], "NEW", res["SUCCESS_NEW"],
          "hold", res["window"].get("longest_both_ok_s"),
          "win_th", res["window"].get("mean_theta"),
          "win_pos", res["window"].get("mean_pos"))
    print(" m3", {k: m3.get(k) for k in (
        "is_type3", "eccentric", "downward_motion", "beneficial_rotation",
        "delta_theta_deg", "bilateral_i", "post_regrasp_correction", "reason", "axial_max_mm",
    )})
    print(" supported", ph.get("supported_while_rotating"), "rot_window", rot_window, "tokens", toks)

ans = json.loads((OUT / "RAW" / "answers.json").read_text(encoding="utf-8"))
print("answers Q4", ans.get("Q4_n_full_type3"), ans.get("Q1_can_convert_to_bilateral"))
