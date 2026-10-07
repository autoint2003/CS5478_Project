"""Smoke test for the current 7D recovery mainline. Does not train SAC or the handoff head."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import make_sim
from training.recovery_policy_pretraining_preparation import (
    HIST,
    HandoffHead,
    feat_from,
    blank_hist,
    legal_obs,
    schedules,
)
from envs.observable_obs import ObservableObsState
from training.recovery_runtime import (
    capture_nominal,
    capture_pads,
    continue_long,
    physical_of,
    restore_dyn,
    run_impact,
    step7,
    T_TASK,
)


def main() -> int:
    sim, cfg = make_sim()
    gains = gains_from_cfg(cfg)
    # 1. Nominal grasp/lift is the prefix of the impact episode.
    nominal = capture_nominal(sim)
    pad_ids, pad_mu = capture_pads(sim)
    impact = run_impact(sim, gains, 6.5, -0.006, 0.0, [], [], nominal, None, None)
    snap = impact["snap"]
    if snap is None:
        raise RuntimeError("impact snapshot missing")
    early = impact["meta"]["early"]
    print("impact", early["rh_mm"], "phase_ok", True, flush=True)

    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    if not info["pose_ok"]:
        raise RuntimeError("restore failed")
    print("restore", round(info["obj_z"], 3), flush=True)

    step7(sim, [0.6, 0, 0, 0, 0.8, 0, 1.0], gains)
    print("step7 ok", flush=True)

    hist = ObservableObsState(0.02)
    obs = legal_obs(sim, hist)
    past_obs, past_act, valid = blank_hist()
    feat = feat_from(obs, past_obs, past_act, valid)
    if feat.shape != (27 * (1 + HIST) + 7 * HIST + HIST,):
        raise RuntimeError(f"feat {feat.shape}")
    print("feat", int(feat.shape[0]), flush=True)

    sched = schedules()
    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    for a in sched["wrist_reseat"]:
        step7(sim, a, gains)
    o = physical_pack(sim)
    if int(o["nL"]) == 0 or int(o["nR"]) == 0:
        raise RuntimeError("wrist reseat lost contact")
    wrist_end = __import__("training.recovery_runtime", fromlist=["stamp"]).stamp(sim)
    print("wrist bilateral", int(o["nL"]), int(o["nR"]), flush=True)

    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    air = 0
    for a in sched["recatch_open20"]:
        step7(sim, a, gains)
        oo = physical_pack(sim)
        if int(oo["nL"]) == 0 and int(oo["nR"]) == 0:
            air += 1
    if air < 1:
        raise RuntimeError("recatch had no airborne step")
    print("recatch air_steps", air, flush=True)

    info = restore_dyn(sim, snap, nominal, pad_ids, pad_mu, None, None)
    xs = []
    for i, a in enumerate(sched["gravity_inward"]):
        if i in (19, 43):
            xs.append(float(physical_pack(sim)["rh"][0]))
        step7(sim, a, gains)
    dx = 1e3 * (xs[1] - xs[0])
    if dx > -1.0:
        raise RuntimeError(f"slide dx {dx}")
    print("slide_dx_mm", round(dx, 2), flush=True)

    info = restore_dyn(sim, wrist_end, nominal, pad_ids, pad_mu, None, None)
    rec = continue_long(sim, gains, T_TASK, "wrist")
    y, kind = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
    if y != 1:
        raise RuntimeError(kind)
    print("t12", kind, round(rec["end"]["obj_z"], 3), flush=True)

    data = np.load(ROOT / "results" / "diagnostics" / "raw" / "recovery_policy_pretraining_preparation" / "imitate_steps.npz")
    if data["X"].shape[0] < 20 or data["Y"].shape[1] != 7:
        raise RuntimeError("dataset")
    print("dataset", data["X"].shape, flush=True)

    import torch
    actor = torch.nn.Sequential(
        torch.nn.Linear(int(feat.shape[0]), 128), torch.nn.Tanh(),
        torch.nn.Linear(128, 7), torch.nn.Tanh(),
    )
    head = HandoffHead(int(feat.shape[0]))
    _ = actor(torch.tensor(feat)[None])
    try:
        head.fit()
        raise RuntimeError("handoff fit should refuse")
    except RuntimeError as exc:
        if "Stage A" not in str(exc):
            raise
    print("actor_and_head", type(actor).__name__, head.dim, flush=True)
    print("SMOKE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
