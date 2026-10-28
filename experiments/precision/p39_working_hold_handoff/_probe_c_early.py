"""Probe why Plan-C experts satisfy R1/R2 early."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from envs.outcomes.support_drop import latched
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index
from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_npz_flex
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META, P21, P21_MAN
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book


def main():
    man = json.loads(P21_MAN.read_text(encoding="utf-8"))
    index = build_start_index(man)
    row = next(r for r in man["trajectories"] if r["family"] == "C" and "gbi_002" in r["traj_id"])
    ep = index[row["traj_id"]]
    z = load_npz_flex(P21, row["npz_relpath"])
    toks = np.asarray(z["tokens"], np.int64)
    res = np.asarray(z["residuals"], np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    restore_start(env, ep)
    print("traj", row["traj_id"], "n_motion", n, "phase0", env.sim.fsm.phase)
    print("step theta e_pos wk bil nL nR grip_cmd phase")
    for t in range(min(80, n)):
        m = measure(env.sim, U_TGT, P_TGT)
        pb = pad_bilateral(env.sim)
        wk = m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM
        act = decode_m111_action(
            book, int(toks[t]), res[t] if t < len(res) else np.zeros(7, np.float32), rho=rho
        )
        print(
            f"{t:3d} {m['e_theta_deg']:7.2f} {m['e_pos_mm']:7.2f} {int(wk)} "
            f"{int(pb['bilateral'])} L{pb['nL']}R{pb['nR']} g={act[6]:+.2f} {env.sim.fsm.phase}"
        )
        step_legal(env.sim, act)
        if latched(env.sim):
            print("DROP at", t)
            break

    # also: first time R1 would fire, vs traj end length
    print("--- summary: expert continues long after early WORKING band ---")


if __name__ == "__main__":
    main()
