"""Probe legal pivot-to-horizontal IC generation."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    decompose,
    load_codebook,
    make_training_env,
    run_seq,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
book = load_codebook()
table = build_k49_table(book, config_id=CONFIG_ID)
model, _ = load_trained_k48(SFT, device)
env = make_training_env(device, model=model)

cands = []
for sp in sorted((ROOT / "results/expert_generation/k49_in_recovery_recatch/snapshots").glob("*.pkl")):
    blob = pickle.loads(sp.read_bytes())
    restore_capture(env, blob["capture"])
    m = measure(env.sim, U_TGT, P_TGT)
    cs = contact_summary(env.sim)
    d = decompose(env.sim)
    if cs["bilateral"] and abs(d["e_ax_mm"]) <= 10 and 5 <= m["e_theta_deg"] <= 25:
        cands.append((sp.stem, m["e_theta_deg"], d["e_ax_mm"], sp))
print("center bil mid-th", len(cands), cands[:8])

results = []
for stem, th0, ax0, sp in cands[:8]:
    blob = pickle.loads(sp.read_bytes())
    for piv in (4, 6, 10):
        for n in (20, 40, 60, 80, 100):
            restore_capture(env, blob["capture"])
            toks = [piv] * n
            ok, _ = run_seq(env, table, book, blob, toks)
            if not ok:
                continue
            m = measure(env.sim, U_TGT, P_TGT)
            cs = contact_summary(env.sim)
            d = decompose(env.sim)
            if m["e_theta_deg"] >= 35 and (cs["nL"] + cs["nR"]) > 0 and abs(d["e_ax_mm"]) <= 18:
                results.append({
                    "src": stem, "piv": piv, "n": n,
                    "th": m["e_theta_deg"], "ax": d["e_ax_mm"],
                    "bil": cs["bilateral"], "pos": m["e_pos_mm"],
                })
print("generated", len(results))
for r in results[:30]:
    print(r)

# also score existing high-theta with looser center
hi = []
for dname in ("hard_recovery_v1", "hard_ic_mining"):
    base = ROOT / "results" / ("benchmarks" if dname == "hard_recovery_v1" else "diagnostics") / dname / "snapshots"
    if not base.exists():
        base = ROOT / f"results/diagnostics/{dname}/snapshots"
    if not base.exists() and dname == "hard_recovery_v1":
        base = ROOT / "results/benchmarks/hard_recovery_v1/snapshots"
    for sp in sorted(base.glob("*.pkl")):
        blob = pickle.loads(sp.read_bytes())
        restore_capture(env, blob["capture"])
        m = measure(env.sim, U_TGT, P_TGT)
        cs = contact_summary(env.sim)
        d = decompose(env.sim)
        if m["e_theta_deg"] >= 35 and (cs["nL"] + cs["nR"]) > 0:
            hi.append({
                "id": sp.stem, "th": m["e_theta_deg"], "ax": abs(d["e_ax_mm"]),
                "bil": cs["bilateral"], "pos": m["e_pos_mm"], "path": str(sp),
            })
hi.sort(key=lambda x: (-x["th"], x["ax"]))
print("existing hi-th contact", len(hi))
for r in hi[:20]:
    print(r)
