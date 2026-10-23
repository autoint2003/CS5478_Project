#!/usr/bin/env python
"""Phase-2 follow-up: gentle burst orient + secure secondary recatch on gbi_014."""

from __future__ import annotations

import csv
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data_generation.oracles.plan_c.dynamic_catch as dc  # noqa: E402
import data_generation.oracles.plan_c.interception as er  # noqa: E402

sys.modules["data_generation.strategies.ground_dynamic_catch"] = dc
sys.modules["data_generation.strategies.ground_early_recovery"] = er

from controllers.recovery.action_map import TAU_OPEN, TAU_SECURE  # noqa: E402
from data_generation.oracles.plan_b.release_recatch import _action_from  # noqa: E402
from data_generation.oracles.plan_c.catch_stabilization import grasp_geometry  # noqa: E402
from data_generation.oracles.plan_c.future_trajectory import (  # noqa: E402
    hold_action,
    object_axis,
    pad_bilateral,
    sample_kin,
)
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from training.recovery_execution import restore_ballistic  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "gpc", ROOT / "scripts" / "generate_plan_c_experts.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
evaluate_handoff_from_state = mod.evaluate_handoff_from_state

OUT = ROOT / "results/diagnostics/ground_plan_c_postlift_correction"
RAW = OUT / "raw"
pack = pickle.loads(
    (ROOT / "results/diagnostics/ground_plan_c_final_success/raw/gbi_014__elevated_snap.pkl").read_bytes()
)
DT = 0.02


def restore(sim):
    restore_ballistic(sim, pack["elev_snap"])
    restore_drop(sim, None)
    arm(sim)


def gentle_orient(sim, w_scale, burst_s, n_bursts, grip=1.0, retighten_s=0.15):
    acts = []
    for _ in range(n_bursts):
        n = int(round(burst_s / DT))
        for _i in range(n):
            m = measure(sim, U_TGT, P_TGT)
            if m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM:
                return acts
            Rh = sample_kin(sim)["R_hand"]
            u = object_axis(sim)
            u = u / max(float(np.linalg.norm(u)), 1e-12)
            uh = Rh.T @ u
            uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
            w = np.clip(np.cross(uh, np.asarray(U_TGT, float)) * w_scale, -0.6, 0.6)
            a = _action_from(sim, np.zeros(3), w, TAU_SECURE)
            a[6] = grip
            acts.append(a.copy())
            step_legal(sim, a)
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                return acts
        for _i in range(int(round(retighten_s / DT))):
            a = hold_action(1.0)
            acts.append(a.copy())
            step_legal(sim, a)
            if latched(sim) or not pad_bilateral(sim)["bilateral"]:
                return acts
    return acts


def recatch_secure(sim, open_s, re_s, close_s, w_scale, open_grip, chase=0.8, settle_s=0.4):
    acts = []
    for name, dur in (
        ("open", open_s),
        ("re", re_s),
        ("close", close_s),
        ("settle", settle_s),
    ):
        n = int(round(dur / DT))
        for _i in range(n):
            kin = sample_kin(sim)
            Rh, ph, po = kin["R_hand"], kin["p_hand"], kin["p_obj"]
            if name == "open":
                grip = open_grip
                w = np.zeros(3)
                v = np.zeros(3)
            elif name == "re":
                grip = open_grip
                u = object_axis(sim)
                u = u / max(float(np.linalg.norm(u)), 1e-12)
                uh = Rh.T @ u
                uh = uh / max(float(np.linalg.norm(uh)), 1e-12)
                w = np.clip(np.cross(uh, np.asarray(U_TGT, float)) * w_scale, -1.0, 1.0)
                err = po - ph
                err[2] = 0.0
                v = chase * err
            elif name == "close":
                grip = 1.0
                w = np.zeros(3)
                err = po - ph
                err[2] = 0.0
                v = chase * err + np.array([0.0, 0.0, 0.02])
            else:
                grip = 1.0
                w = np.zeros(3)
                v = np.zeros(3)
            a = _action_from(sim, v, w, TAU_OPEN if grip < 0.5 else TAU_SECURE)
            a[6] = float(np.clip(grip, -1.0, 1.0))
            acts.append(a.copy())
            step_legal(sim, a)
            if latched(sim):
                return acts
    return acts


def eval_now(sim, acts, mechanism, tag):
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    lost = latched(sim) or not pb["bilateral"]
    meta = {"SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF"}
    if not lost:
        meta = evaluate_handoff_from_state(sim, pack["ref"], stabilize_n=0)
    return {
        "mechanism": mechanism,
        "tag": tag,
        "theta_end": m["e_theta_deg"],
        "pos_end": m["e_pos_mm"],
        "bilateral_end": int(pb["bilateral"]),
        "SUCCESS_NEW": int(meta.get("SUCCESS_NEW") or 0),
        "reason": meta.get("reason"),
        "n_actions": len(acts),
        "pre_theta": meta.get("pre_theta", m["e_theta_deg"]),
        "pre_pos_mm": meta.get("pre_pos_mm", m["e_pos_mm"]),
        "contact_s_end": grasp_geometry(sim)["contact_s_mean"],
    }, acts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    sim, _ = make_ground_sim()

    for ws in (0.25, 0.4, 0.55):
        for burst in (0.08, 0.12):
            for nb in (8, 15, 25):
                restore(sim)
                acts = gentle_orient(sim, ws, burst, nb)
                row, acts = eval_now(sim, acts, "A2_gentle_burst", f"gentle_w{ws}_b{burst}_n{nb}")
                rows.append(row)
                print(
                    row["tag"],
                    f"th={row['theta_end']:.1f}",
                    f"bil={row['bilateral_end']}",
                    f"SN={row['SUCCESS_NEW']}",
                    flush=True,
                )
                if row["SUCCESS_NEW"]:
                    best = ("A2", acts, row)

    for open_s, re_s, close_s, ws, og, ch in [
        (0.12, 0.5, 0.8, 2.0, -0.5, 1.0),
        (0.18, 0.7, 1.0, 2.5, -0.7, 1.2),
        (0.1, 0.4, 1.2, 2.0, -0.4, 0.9),
        (0.2, 0.8, 1.0, 3.0, -0.8, 1.0),
        (0.15, 0.6, 1.5, 2.2, -0.6, 1.1),
        (0.08, 0.35, 1.0, 1.8, -0.3, 0.8),
        (0.25, 0.9, 1.2, 2.5, -1.0, 1.0),
        (0.2, 0.5, 0.9, 2.0, 0.0, 1.0),
        (0.2, 0.6, 0.8, 2.0, 0.2, 0.9),
        (0.25, 0.7, 1.0, 2.5, 0.35, 1.0),
        (0.3, 1.0, 1.5, 2.0, 0.15, 1.0),
        (0.15, 0.45, 2.0, 2.5, -0.5, 1.2),
    ]:
        restore(sim)
        acts = recatch_secure(sim, open_s, re_s, close_s, ws, og, chase=ch, settle_s=0.5)
        row, acts = eval_now(
            sim, acts, "B2_secure_recatch", f"sec_o{open_s}_r{re_s}_c{close_s}_w{ws}_g{og}"
        )
        rows.append(row)
        print(
            row["tag"],
            f"th={row['theta_end']:.1f}",
            f"bil={row['bilateral_end']}",
            f"SN={row['SUCCESS_NEW']}",
            row["reason"],
            flush=True,
        )
        if row["SUCCESS_NEW"]:
            best = ("B2", acts, row)

    for ws, nb in [(0.35, 20), (0.45, 30), (0.3, 40)]:
        restore(sim)
        acts = gentle_orient(sim, ws, 0.1, nb)
        if pad_bilateral(sim)["bilateral"] and measure(sim, U_TGT, P_TGT)["e_theta_deg"] > 15:
            acts = acts + recatch_secure(
                sim, 0.15, 0.5, 1.0, 2.2, 0.2, chase=1.0, settle_s=0.5
            )
        row, acts = eval_now(sim, acts, "AB_combo", f"combo_w{ws}_n{nb}")
        rows.append(row)
        print(
            row["tag"],
            f"th={row['theta_end']:.1f}",
            f"bil={row['bilateral_end']}",
            f"SN={row['SUCCESS_NEW']}",
            flush=True,
        )
        if row["SUCCESS_NEW"]:
            best = ("AB", acts, row)

    path = OUT / "MATCHED_COMPARISON_PHASE2.csv"
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    surv = [r for r in rows if int(r["bilateral_end"])]
    surv.sort(key=lambda r: float(r["theta_end"]))
    print("survivors", len(surv))
    if surv:
        print("best survivor", surv[0])
    print("any SUCCESS_NEW", any(int(r["SUCCESS_NEW"]) for r in rows))

    # Pareto: lowest theta among survivors; and lowest theta among all
    all_sorted = sorted(rows, key=lambda r: float(r["theta_end"]))
    print("best theta overall", all_sorted[0])

    if best is not None:
        (RAW / "gbi_014__phase2_best_actions.pkl").write_bytes(
            pickle.dumps({"acts": best[1], "meta": best[2]})
        )
        # full-root 3x
        full = list(np.asarray(pack["prefix_actions"], np.float32)) + [
            np.asarray(a, np.float32) for a in best[1]
        ]
        replay = []
        for i in range(3):
            sim_r, _ = make_ground_sim()
            restore_ballistic(sim_r, pack["catch_snap"])
            restore_drop(sim_r, None)
            arm(sim_r)
            lost = False
            for a in full:
                step_legal(sim_r, a)
                if latched(sim_r):
                    lost = True
                    break
            if lost or not pad_bilateral(sim_r)["bilateral"]:
                replay.append({"replay_i": i, "SUCCESS_NEW": 0, "reason": "LOST_BEFORE_HANDOFF"})
                continue
            meta = evaluate_handoff_from_state(sim_r, pack["ref"], stabilize_n=0)
            replay.append(
                {
                    "replay_i": i,
                    "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
                    "reason": meta["reason"],
                    "pre_theta": meta.get("pre_theta"),
                    "pre_pos_mm": meta.get("pre_pos_mm"),
                }
            )
        with (OUT / "EXACT_REPLAY.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(replay[0].keys()))
            w.writeheader()
            w.writerows(replay)
        print("full-root replay", replay)


if __name__ == "__main__":
    main()
