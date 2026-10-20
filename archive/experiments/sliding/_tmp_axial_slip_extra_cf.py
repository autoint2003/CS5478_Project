"""Extra CFs: pure -wy (no vx) and mild friction multipliers."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture
from envs.simulation.object_dynamics import set_finger_object_sliding_mu
from models.k48_checkpoint import load_trained_k48
from experiments.sliding.diagnose_k49_return_axial_slip import (
    OUT,
    SFT,
    SNAP_IC,
    decode,
    sample_substep,
    step_instrumented,
    summarize_trace,
    write_csv,
    write_json,
)
from training.action_vocab_k49 import SECURE_REVERSE_WRIST, build_k49_table
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)
    pre = pickle.loads((OUT / "SNAPSHOTS" / "hard_910263_prereturn.pkl").read_bytes())
    pre_cap = pre["capture"]
    pre_blob = pre

    a48 = decode(table, book48, SECURE_REVERSE_WRIST).copy()
    a_pure = a48.copy()
    a_pure[0] = 0.0
    a_pure[1] = 0.0
    a_pure[2] = 0.0
    print("a48", a48, "a_pure", a_pure, flush=True)

    results = {}

    def run_action(action, n=19, scale=1.0, tag="X"):
        restore_capture(env, pre_cap)
        rows = [{**sample_substep(env.sim, pre_blob), "label": tag + "_t0", "scale_w": scale}]
        for i in range(n):
            step_instrumented(env.sim, action, pre_blob, rows, f"{tag}_{i}", scale_w=scale)
        s = summarize_trace(rows)
        print(
            tag,
            "d_e_ax", round(s["d_e_ax_mm"], 4),
            "hand", round(s["hand0_deg"], 1), "->", round(s["hand1_deg"], 1),
            "Fn_mean", round(s["Fn_mean"], 2),
            flush=True,
        )
        write_csv(OUT / "RAW" / f"extra_{tag}_substeps.csv", rows)
        results[tag] = s
        return rows, s

    run_action(a48, tag="A_replay")
    run_action(a_pure, tag="CF_pure_wy")

    mu0 = float(env.sim.model.geom_friction[env.sim.ids.object_geom][0])
    print("mu0", mu0, flush=True)
    for mu in (mu0 * 1.1, mu0 * 1.25, mu0 * 1.5):
        restore_capture(env, pre_cap)
        set_finger_object_sliding_mu(env.sim.model, env.sim.ids, float(mu), env.sim.data)
        rows = [{**sample_substep(env.sim, pre_blob), "label": "t0", "scale_w": 1.0}]
        print("  after mu set Fn", rows[0]["Fn_total"], "ncon", rows[0]["n_contacts"], flush=True)
        for i in range(19):
            step_instrumented(env.sim, a48, pre_blob, rows, f"mu{mu:.2f}", scale_w=1.0)
        s = summarize_trace(rows)
        tag = f"E_mild_mu{mu:.2f}"
        print(
            f"  {tag} d_e_ax={s['d_e_ax_mm']:.3f} Fn_mean={s['Fn_mean']:.2f} Fn0={s['Fn0']:.2f}",
            flush=True,
        )
        write_csv(OUT / "RAW" / f"extra_{tag}_substeps.csv", rows)
        results[tag] = {**s, "mu": mu, "mu0": mu0}
        set_finger_object_sliding_mu(env.sim.model, env.sim.ids, mu0, env.sim.data)

    rows, _ = run_action(a48, tag="A_slope")
    angs = np.array([r["hand_to_ref_deg"] for r in rows])
    eax = np.array([r["e_ax_mm"] for r in rows])
    de = np.diff(eax)
    da = np.diff(angs)
    mask = np.abs(da) > 1e-3
    slope = float(np.median(de[mask] / da[mask]))
    total_slope = float((eax[-1] - eax[0]) / (angs[-1] - angs[0]))
    print("median de_ax/dtheta_deg", slope, flush=True)
    print("total de/dtheta", total_slope, flush=True)
    results["slope_mm_per_deg"] = slope
    results["total_slope_mm_per_deg"] = total_slope
    write_json(OUT / "RAW" / "extra_cf_summaries.json", results)
    print("EXTRA_DONE", flush=True)


if __name__ == "__main__":
    main()
