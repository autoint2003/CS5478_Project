"""Verify early-wrist distal correction through SUCCESS_NEW + 3/3 replay."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_pose_feedback import load_root, pose_now, privileged_correct_action
from scripts.run_horizontal_offset_freefall_recatch import eval_full_sn, find_cut
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import build_stabilization, clear_probe_latch


def make_early_wrist_prefix(acts, cut, wx, wy, wz, until=None):
    until = until if until is not None else cut + 3
    out = []
    for i, a0 in enumerate(acts):
        a = np.asarray(a0, np.float32).copy()
        if i <= until:
            a[3] = float(np.clip(wx, -1, 1))
            a[4] = float(np.clip(wy, -1, 1))
            a[5] = float(np.clip(wz, -1, 1))
        out.append(a)
    return out


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, acts, cut = load_root("distal_off01_r8")
    print("orig cut", cut, "n", len(acts))

    candidates = [
        (0.0, 0.14, 0.20),
        (0.0, 0.10, 0.20),
        (0.0, 0.18, 0.15),
        (0.0, 0.20, 0.20),
        (-0.1, 0.14, 0.20),
        (0.05, 0.14, 0.20),
        (0.0, 0.14, 0.25),
        (0.0, 0.12, 0.18),
    ]
    best = None
    for wx, wy, wz in candidates:
        for until in (cut + 1, cut + 2, cut + 3, cut + 5):
            pref = make_early_wrist_prefix(acts, cut, wx, wy, wz, until=until)
            # truncate to original length for stab search; use early portion as catch prefix
            # find first bil on modified traj
            restore_ic(env, ic)
            clear_probe_latch(env.sim)
            bil_i = None
            for i, a in enumerate(pref):
                step_legal(env.sim, a)
                if contact_summary(env.sim)["bilateral"] and bil_i is None:
                    bil_i = i
                if latched(env.sim):
                    break
            if bil_i is None or latched(env.sim):
                continue
            p_bil = pose_now(env.sim)
            # use prefix up to bil+2 then stabilize
            c = min(len(pref), bil_i + 3)
            catch = pref[:c]
            for n_lift, n_seat, wy_stab in [
                (35, 28, 0.35),
                (40, 32, 0.25),
                (30, 24, 0.40),
                (45, 36, 0.30),
            ]:
                p = {
                    "n_lift": n_lift,
                    "lift": -0.95,
                    "n_return": 14,
                    "n_seat": n_seat,
                    "n_fine": 0,
                    "wy": wy_stab,
                    "n_wy": 14,
                    "xy_gain": 0.6,
                    "axis_align": True,
                    "use_fine_down": False,
                    "return_vz": -0.18,
                    "seat_vz": 0.08,
                }
                stab = build_stabilization(env, ic, catch, p)
                if stab is None:
                    continue
                # optional privileged fine pose
                restore_ic(env, ic)
                clear_probe_latch(env.sim)
                for a in catch + stab:
                    step_legal(env.sim, a)
                extra = []
                for _ in range(20):
                    a = privileged_correct_action(
                        env.sim, "combined", {"kp": 1.0, "kz": 0.25, "kw": 0.55, "wy": 0.0}
                    )
                    a[2] = float(np.clip(a[2] - 0.15, -1, 1))
                    step_legal(env.sim, a)
                    extra.append(a.copy())
                    if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                        break
                full = catch + stab + extra
                out = eval_full_sn(env, ic, full, c, nominal_s=float(NOMINAL_S))
                print(
                    f"wx={wx:.2f} wy={wy:.2f} wz={wz:.2f} until={until} bil={bil_i} "
                    f"th_bil={p_bil['e_theta_deg']:.2f} pos_bil={p_bil['e_pos_mm']:.2f} "
                    f"SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
                    f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')}"
                )
                sc = 1000 * int(out["SUCCESS_NEW"]) + 100 * int(out["CATCH_HANDOFF_SUCCESS"])
                th = out.get("SN_mean_theta") or 99
                pos = out.get("SN_mean_pos") or 99
                sc += -th - 0.05 * pos
                if best is None or sc > best[0]:
                    best = (sc, full, c, out, (wx, wy, wz, until))
                if out["SUCCESS_NEW"]:
                    ok = sum(
                        int(
                            eval_full_sn(env, ic, full, c, nominal_s=float(NOMINAL_S))[
                                "SUCCESS_NEW"
                            ]
                        )
                        for _ in range(3)
                    )
                    print("  replay", ok, "/3")
                    if ok == 3:
                        print("VERIFIED", (wx, wy, wz, until), out)
                        np.savez_compressed(
                            ROOT
                            / "results/diagnostics/freefall_recatch_pose_feedback/trajectories"
                            / "distal_off01_r8__SUCCESS_NEW_probe.npz",
                            actions=np.asarray(full, np.float32),
                            cut=c,
                            wx=wx,
                            wy=wy,
                            wz=wz,
                            until=until,
                        )
                        return

    if best:
        print("BEST non-SN", best[4], best[3])


if __name__ == "__main__":
    main()
