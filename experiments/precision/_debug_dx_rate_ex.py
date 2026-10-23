#!/usr/bin/env python
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.precision_dx_support_transfer import (
    correct_dx_support_transfer,
)
from data_generation.oracles.precision.precision_dz import correct_dz_inward
from data_generation.oracles.precision.precision_types import PrecisionConfig
from data_generation.oracles.precision.unified_precision_oracle import run_precision
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY

PLAN_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"


def main():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    blob = pickle.loads(PLAN_B.read_bytes())

    for cap in (0.08, 0.10, 0.12):
        cfg = PrecisionConfig(
            evaluate_handoff=False, dx_horizontal_rate_cap=cap, rate_protected_s1=0.14
        )
        st = correct_dx_support_transfer(
            env, {"capture": blob["capture"], "ref": blob["ref"]}, cfg
        )
        print(
            f"cap={cap} ok={st.get('ok')} ex0={st.get('ex0')} ex1={st.get('ex1')} "
            f"n={len(st.get('actions') or [])} phase={st.get('phase')} reason={st.get('reason')}",
            flush=True,
        )
        if st.get("ok"):
            r = correct_dz_inward(
                env,
                {"capture": st["capture"], "ref": blob["ref"]},
                PrecisionConfig(evaluate_handoff=True),
                s1_state=st.get("s1_state"),
            )
            print(
                f"  dz ok={r.get('ok')} code={r.get('failure_code')} reused={r.get('reused_s1')} "
                f"n={len(r.get('actions') or [])} outcomes={r.get('outcomes')}",
                flush=True,
            )

    print("--- full run_precision optimized ---", flush=True)
    res = run_precision(
        env,
        {"capture": blob["capture"], "ref": blob["ref"]},
        "B",
        PrecisionConfig(evaluate_handoff=True, dx_horizontal_rate_cap=0.08),
    )
    to = (res.primitive_logs or {}).get("task_outcomes")
    print(res.status, "n=", len(res.actions or []), "to=", to, "fail=", res.failures, flush=True)
    dx = (res.primitive_logs or {}).get("dx_support_transfer") or {}
    print("dx ex0/ex1", dx.get("ex0"), dx.get("ex1"), "errors", dx.get("errors"), flush=True)


if __name__ == "__main__":
    main()
