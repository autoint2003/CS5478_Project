import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.generate_k49_dynamic_recatch_experts import (
    analyze_type3_mechanism,
    find_first_bilateral,
    has_post_manipulation,
    run_traced,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    evaluate_full,
    load_codebook,
    load_trained_k48,
    make_training_env,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book48 = load_codebook()
table = build_k49_table(book48, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)


def analyze(ic, name, seq):
    blob = pickle.loads(Path(
        f"results/expert_generation/k49_in_recovery_recatch/snapshots/{ic}.pkl"
    ).read_bytes())
    res = evaluate_full(env, table, book48, blob, seq)
    ok, tr = run_traced(env, table, book48, blob, seq)
    loss = any(r["nL"] + r["nR"] == 0 for r in tr) or any(
        (r["nL"] + r["nR"]) > 0 and not r["bilateral"] for r in tr
    )
    bil = find_first_bilateral(tr, 0)
    post, _ = has_post_manipulation(tr, bil)
    m3 = analyze_type3_mechanism(tr)
    print(
        ic, name,
        "NEW", res["SUCCESS_NEW"], "s12", res["success_12s"],
        "loss", loss, "post", post,
        "t3", m3.get("is_type3"), "ax", m3.get("axial_max_mm"),
        "dth", m3.get("delta_theta_deg"), "why", m3.get("reason"),
        "win", res["window"].get("mean_pos"), res["window"].get("mean_theta"),
    )


CPP = [32] + [10] * 8 + [12] * 15 + [0] * 5
for name, seq in [
    ("known", CPP),
    ("rel_then_cpp", [30] + [0] * 5 + CPP),
    ("rel_cpp_short", [30] + [0] * 3 + [32] + [10] * 8 + [12] * 10 + [0] * 5),
    ("open_cpp", [30] * 2 + CPP),
]:
    analyze("hard_910101", name, seq)

for name, seq in [
    ("piv_before", [4] * 20 + [12] * 5 + [0] * 3 + [32] + [0] * 8),
    ("rel_after_piv", [4] * 20 + [12] * 5 + [0] * 3 + [32] + [0] * 8),
    ("open_piv", [32] * 2 + [0] * 5 + [4] * 20 + [12] * 5 + [0] * 3),
    ("known", CPP),
    ("rel_cpp", [32] + [0] * 5 + CPP),
]:
    analyze("hard_910220", name, seq)

for ic, pref in [("hrv1_920131", [41]), ("hrv1_920099", [41])]:
    for name, seq in [
        ("simple", pref + [0] * 8),
        ("cpp_hrv", [6] * 20 + [12] * 15 + [0] * 8),
        ("rel_cpp", pref + [0] * 5 + [6] * 20 + [12] * 15 + [0] * 8),
        ("cpp_rel", [6] * 15 + [12] * 5 + [0] * 3 + pref + [0] * 8),
        ("tok10", [10] * 16 + [12] * 5 + [0] * 5),
        ("rel_tok10", pref + [0] * 5 + [10] * 16 + [12] * 5 + [0] * 5),
        ("f_recatch_piv", [4] * 12 + [36] * 5 + [0] * 8 + [4] * 24 + [12] * 5 + [0] * 3),
        ("cpp4", [4] * 22 + [12] * 5 + [0] * 2 + pref + [0] * 8),
    ]:
        analyze(ic, name, seq)
