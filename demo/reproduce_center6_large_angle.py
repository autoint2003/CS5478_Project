"""Replay the published CENTER-6 EARLY large-angle sequence. Does not rewrite the historical report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from demo.ballistic_large_angle import (
    BRAKE_S,
    GRIP_S,
    apply_gravity_sequence,
    classify_grip,
    load_early,
    measure,
    rotate_to,
    start_from_early,
    tick_log,
)
from training.demo_ballistic_impact import make_sim

OUT = ROOT / "results" / "diagnostics" / "raw" / "abc_original_revalidation"


def _num(value):
    if isinstance(value, dict):
        return {str(key): _num(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_num(item) for item in value]
    if isinstance(value, np.ndarray):
        return _num(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _slip_summary(events):
    slip = events["slip_end"]
    brake = events["brake_end"]
    ret = events["return_end"]
    return {
        "slip_start_rhx_mm": events.get("slip_start_rhx_mm"),
        "slip_drhx_mm": events.get("slip_drhx_mm"),
        "slip_end_rhx_mm": 1e3 * float(slip["rh"][0]),
        "slip_nL": int(slip["nL"]),
        "slip_nR": int(slip["nR"]),
        "slip_ghx": float(np.asarray(slip["g_h"])[0]),
        "brake_rhx_mm": 1e3 * float(brake["rh"][0]),
        "brake_nL": int(brake["nL"]),
        "brake_nR": int(brake["nR"]),
        "return_deg": events.get("return_deg"),
        "return_rhx_mm": 1e3 * float(ret["rh"][0]),
        "return_nL": int(ret["nL"]),
        "return_nR": int(ret["nR"]),
        "return_err_hand_deg": events["return"]["err_hand_deg"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    snap = load_early()
    sim, _ = make_sim()
    _rh0, early = start_from_early(sim, snap)
    desired = -float(np.sign(early["rh"][0])) or -1.0
    early_row = {
        "t": early["t"],
        "rh_mm": (1e3 * np.asarray(early["rh"])).tolist(),
        "g_h": np.asarray(early["g_h"], float).tolist(),
        "nL": int(early["nL"]),
        "nR": int(early["nR"]),
        "tilt_deg": early["tilt_deg"],
        "desired_drhx_sign": desired,
    }
    print("EARLY", early_row, flush=True)
    plan = {
        "theta_deg": 120.0,
        "tau_slip": -5.0,
        "slip_s": GRIP_S,
        "brake_s": BRAKE_S,
        "desired_drhx_sign": desired,
    }
    sim, _ = make_sim()
    _log, table, events, _rh = apply_gravity_sequence(sim, gains, snap, plan)
    end = _log[-1]
    plus = _slip_summary(events)
    plus.update({
        "table": table,
        "t_end": end["t"],
        "end_rh_mm": (1e3 * np.asarray(end["rh"])).tolist(),
        "end_z": end["obj_z"],
        "end_nL": end["nL"],
        "end_nR": end["nR"],
        "end_gh": np.asarray(end["g_h"], float).tolist(),
    })
    print("PLUS", plus["slip_drhx_mm"], "table", table, "tend", end["t"], "end_rhx", plus["end_rh_mm"][0], flush=True)

    wrong = dict(plan)
    wrong["theta_deg"] = -120.0
    sim, _ = make_sim()
    _log_w, table_w, events_w, _ = apply_gravity_sequence(sim, gains, snap, wrong)
    print("MINUS table", table_w, "slip", events_w.get("slip_drhx_mm"), "n", events_w.get("slip_end", {}).get("nL"), flush=True)

    sim, _ = make_sim()
    rh0, m_early = start_from_early(sim, snap)
    log = [m_early]
    rotate_to(sim, gains, rh0, 120.0, -18.0, log)
    saved = np.array(sim.model.opt.gravity, float).copy()
    sim.model.opt.gravity[:] = 0.0
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    angled = measure(sim, -18.0, rh0)
    n = int(round(GRIP_S / float(sim.model.opt.timestep)))
    i0 = len(log)
    tick_log(sim, gains, 0.0, 0.0, 0.0, -5.0, rh0, log, n, label="SLIP_0G")
    kind = classify_grip(angled, log[-1], log[i0:], desired, float(angled["g_h"][0]))
    sim.model.opt.gravity[:] = saved
    zero_g = {"drhx_mm": kind["drhx_mm"], "kind": kind["kind"], "nL": log[-1]["nL"], "nR": log[-1]["nR"]}
    print("0G", zero_g, flush=True)

    payload = {"early": early_row, "plus_120": plus, "minus_120_table": table_w, "minus_slip_drhx_mm": events_w.get("slip_drhx_mm"), "zero_g": zero_g}
    (OUT / "center6_replay.json").write_text(json.dumps(_num(payload), indent=2), encoding="utf-8")
    print("WROTE", OUT / "center6_replay.json", flush=True)


if __name__ == "__main__":
    main()
