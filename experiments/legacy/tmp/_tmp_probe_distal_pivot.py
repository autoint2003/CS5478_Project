"""Probe soft-grip / regrasp / early-wrist authority on distal_off01_r8."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import contact_summary
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT
from scripts.run_freefall_recatch_pose_feedback import (
    load_root,
    pose_now,
    privileged_correct_action,
)
from scripts.run_lateral_offset_falling_recatch import make_training_env, restore_ic
from scripts.run_lateral_recatch_handoff import clear_probe_latch, hand_R


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    ic, acts, cut = load_root("distal_off01_r8")

    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    first_bil = None
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if contact_summary(env.sim)["bilateral"] and first_bil is None:
            first_bil = i
            print("first_bil", i, pose_now(env.sim))
        if first_bil is not None and i >= first_bil + 2:
            break
    print("pivot_start", pose_now(env.sim))
    cap = take_capture(env)

    rows = []
    for grip in [1.0, 0.2, 0.0, -0.4, -0.7]:
        for kw in [0.6, 1.0]:
            for tip in [0.0, 0.25]:
                restore_capture(env, cap)
                clear_probe_latch(env.sim)
                n = 0
                lost = False
                for t in range(45):
                    a = privileged_correct_action(
                        env.sim, "combined", {"kp": 0.5, "kz": 0.15, "kw": kw, "wy": 0.0}
                    )
                    a[6] = grip
                    a[3] = float(np.clip(a[3] + tip, -1, 1))
                    step_legal(env.sim, a)
                    n = t + 1
                    if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                        lost = True
                        break
                # reclose
                if not lost and grip < 0.5:
                    for _ in range(12):
                        a = privileged_correct_action(
                            env.sim, "combined", {"kp": 0.8, "kz": 0.2, "kw": 0.4, "wy": 0.0}
                        )
                        a[6] = 1.0
                        step_legal(env.sim, a)
                        if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                            lost = True
                            break
                p = pose_now(env.sim)
                rows.append((grip, kw, tip, n, p["e_theta_deg"], p["e_pos_mm"], p["bilateral"], lost))
                print(
                    f"grip={grip:5.1f} kw={kw:.1f} tip={tip:.2f} n={n:2d} "
                    f"th={p['e_theta_deg']:6.2f} pos={p['e_pos_mm']:6.2f} "
                    f"bil={int(p['bilateral'])} lost={int(lost)}"
                )

    # Early wrist bias during approach (replace a[3:6] before bil)
    print("\n=== early wrist approach variants ===")
    best = None
    for wx in np.linspace(-0.6, 0.6, 7):
        for wy in np.linspace(-0.4, 0.5, 6):
            for wz in [0.0, 0.2, -0.2]:
                restore_ic(env, ic)
                clear_probe_latch(env.sim)
                new_acts = []
                for i, a0 in enumerate(acts[: max(cut + 5, 15)]):
                    a = np.asarray(a0, np.float32).copy()
                    if i <= cut + 2:
                        a[3] = float(np.clip(wx, -1, 1))
                        a[4] = float(np.clip(wy, -1, 1))
                        a[5] = float(np.clip(wz, -1, 1))
                    step_legal(env.sim, a)
                    new_acts.append(a.copy())
                    if latched(env.sim):
                        break
                if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                    continue
                p = pose_now(env.sim)
                sc = -p["e_theta_deg"] - 0.05 * p["e_pos_mm"]
                if best is None or sc > best[0]:
                    best = (sc, wx, wy, wz, p, list(new_acts))
                if p["e_theta_deg"] < 12:
                    print(f"GOOD early wx={wx:.2f} wy={wy:.2f} wz={wz:.2f} {p}")
    if best:
        print("best early", best[1:4], best[4])

    # Regrasp cycle from after_bil: open briefly, rotate hand, close
    print("\n=== regrasp cycles ===")
    restore_capture(env, cap)
    clear_probe_latch(env.sim)
    for open_n, rot_n, close_n, wy in [
        (6, 12, 16, 0.45),
        (8, 16, 20, 0.55),
        (5, 20, 18, 0.35),
        (10, 24, 24, 0.5),
        (8, 16, 20, -0.45),
        (6, 20, 20, 0.6),
    ]:
        restore_capture(env, cap)
        clear_probe_latch(env.sim)
        lost = False
        for t in range(open_n):
            a = np.zeros(7, np.float32)
            a[6] = -0.8
            a[2] = -0.15
            # keep correcting toward target weakly
            pa = privileged_correct_action(env.sim, "combined", {"kp": 0.3, "kz": 0.1, "kw": 0.2})
            a[3:6] = pa[3:6]
            step_legal(env.sim, a)
            if latched(env.sim):
                lost = True
                break
        if lost:
            print("lost during open")
            continue
        for t in range(rot_n):
            a = privileged_correct_action(
                env.sim, "combined", {"kp": 0.9, "kz": 0.25, "kw": 0.95, "wy": wy}
            )
            a[6] = -0.3
            a[4] = float(np.clip(a[4] + wy, -1, 1))
            step_legal(env.sim, a)
            if latched(env.sim):
                lost = True
                break
        if lost:
            print("lost during rot")
            continue
        for t in range(close_n):
            a = privileged_correct_action(
                env.sim, "combined", {"kp": 1.0, "kz": 0.3, "kw": 0.5, "wy": 0.0}
            )
            a[6] = 1.0
            a[2] = float(np.clip(a[2] - 0.35, -1, 1))
            step_legal(env.sim, a)
            if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                lost = True
                break
        p = pose_now(env.sim)
        print(
            f"regrasp open={open_n} rot={rot_n} close={close_n} wy={wy} "
            f"th={p['e_theta_deg']:.2f} pos={p['e_pos_mm']:.2f} bil={int(p['bilateral'])} lost={int(lost)}"
        )


if __name__ == "__main__":
    main()
