"""Continue AR gen/disc within remaining frozen budget (435 used / 800)."""
from __future__ import annotations

import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from evaluation.full_objective import U_TGT, P_TGT, contact_summary
from evaluation.grasp_configuration import cylinder_axis_hand, measure as gmeasure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, decode, write_csv, write_json
from scripts.run_active_recatch_generalization_discretization import (
    AR_TOKEN_ACTIONS,
    AR_VOCAB_ID,
    DISC_BUDGET,
    GEN_BUDGET,
    LIFT_ORI_BUDGET,
    OUT,
    ROBUST_BUDGET,
    SEED_IDS,
    TOK_CHASE_PAD,
    TOK_FINE_AR,
    TOK_LIFT_UP,
    TOK_LIFT_WY,
    TOTAL_BUDGET,
    WALL_LIMIT_S,
    build_ar_table,
    compose_expert_from_ic,
    load_seed,
    nearest_token,
    tokens_hash,
)
from scripts.run_lateral_offset_falling_recatch import FINE_DOWN, make_training_env, obj_pos, restore_ic, rot_angle
from scripts.run_lateral_recatch_handoff import (
    build_stabilization,
    clear_probe_latch,
    evaluate_catch_handoff,
    hand_R,
)
from training.action_vocab_k49 import HANDOFF, build_k49_table

SRC_CATCH = ROOT / "results" / "diagnostics" / "lateral_offset_falling_recatch"


def phase_quantize(acts, cut, ar_table, k49, book48, use_ar=True):
    """Prefix → nearest K49; stabilize → prefer AR lift/fine tokens."""
    toks, qacts = [], []
    for i, a in enumerate(acts):
        a = np.asarray(a, np.float32)
        if use_ar and i >= cut:
            # choose among AR stabilize tokens + k49 by distance
            cands = []
            for tid, ta in AR_TOKEN_ACTIONS.items():
                if tid == TOK_CHASE_PAD:
                    continue
                cands.append((float(np.linalg.norm(a - ta)), tid, ta))
            for j in range(len(k49)):
                if j == HANDOFF:
                    continue
                ta = np.asarray(k49[j], np.float32)
                cands.append((float(np.linalg.norm(a - ta)), j, ta))
            cands.sort(key=lambda x: x[0])
            # bias: if a2 < -0.5 and grip>0.5, force LIFT_UP/WY
            if a[2] < -0.5 and a[6] > 0.5:
                tid = TOK_LIFT_WY if abs(a[4]) > 0.1 else TOK_LIFT_UP
                ta = AR_TOKEN_ACTIONS[tid]
            elif a[2] > 0.05 and a[6] > 0.5 and abs(a[0]) + abs(a[1]) < 0.2:
                tid, ta = TOK_FINE_AR, AR_TOKEN_ACTIONS[TOK_FINE_AR]
            else:
                _, tid, ta = cands[0]
        else:
            # intercept: prefer chase pad if open grip and large |a1|
            if use_ar and a[6] < 0 and abs(a[1]) > 0.4:
                tid, ta = TOK_CHASE_PAD, AR_TOKEN_ACTIONS[TOK_CHASE_PAD]
            else:
                d, tid, ta = nearest_token(k49, a)
        toks.append(int(tid))
        qacts.append(np.asarray(ta, np.float32).copy())
    return toks, qacts


def main():
    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))
    used0 = int(usage["rollouts_used"])
    remain = TOTAL_BUDGET - used0
    print(f"Continue with remaining budget {remain} (used {used0}/{TOTAL_BUDGET})", flush=True)

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    k49 = build_k49_table(book48, config_id=CONFIG_ID)
    ar_table = build_ar_table(book48)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(20261023)
    rollout = [0]
    t0 = time.time()
    wall_left = WALL_LIMIT_S - float(usage.get("wall_s", 0))

    def ok_budget():
        return rollout[0] < remain and (time.time() - t0) < max(60.0, wall_left)

    disc_extra = []
    lift_extra = []
    gen_extra = []
    ori_extra = []
    verify_extra = []

    # --- Phase-aware discrete quantize on seeds ---
    for ic_id in SEED_IDS:
        if not ok_budget():
            break
        ic, acts, cut = load_seed(ic_id)
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        for use_ar, mode in ((True, "phase_quantize_AR_v1"), (False, "phase_quantize_K49")):
            if use_ar:
                toks, qacts = phase_quantize(acts, cut, ar_table, k49, book48, True)
            else:
                toks, qacts = phase_quantize(acts, cut, ar_table, k49, book48, False)
                # force no AR ids
                q2 = []
                t2 = []
                for a in acts:
                    d, tid, ta = nearest_token(k49, a)
                    t2.append(tid)
                    q2.append(ta)
                toks, qacts = t2, q2
            env._ch_nominal_s = 4.0
            out = evaluate_catch_handoff(env, ic, qacts, cut, R0)
            rollout[0] += 1
            disc_extra.append({
                "ic_id": ic_id, "mode": mode,
                "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                "SUCCESS_NEW": out["SUCCESS_NEW"],
                "held_s": out["held_s"],
                "mean_theta": out["SN_mean_theta"],
                "mean_pos": out["SN_mean_pos"],
                "n_unique_tokens": len(set(toks)),
                "tokens_hash": tokens_hash(toks),
                "ar_token_frac": float(np.mean([1.0 if t >= 49 else 0.0 for t in toks])),
            })
            print(
                f"  {mode} {ic_id} CH={out['CATCH_HANDOFF_SUCCESS']} SN={out['SUCCESS_NEW']} "
                f"ar_frac={disc_extra[-1]['ar_token_frac']:.2f}",
                flush=True,
            )
            if use_ar and out["CATCH_HANDOFF_SUCCESS"]:
                np.savez_compressed(
                    OUT / "trajectories" / f"{ic_id}__phase_discrete_AR.npz",
                    actions=np.asarray(qacts, np.float32),
                    tokens=np.asarray(toks, np.int32),
                    cut=np.array([cut], np.int32),
                )

    # --- Lift B–F with catch prefix from SRC ---
    ic_p, acts_p, cut_p = load_seed("perp_off01_r15")
    R0p = np.asarray(ic_p["R0"], float).reshape(3, 3)
    catch_p = SRC_CATCH / "trajectories" / "perp_off01_r15__D_scoop_actions.npz"
    pref = [np.asarray(a, np.float32) for a in np.load(catch_p)["actions"]]
    restore_ic(env, ic_p)
    clear_probe_latch(env.sim)
    first_bil = None
    latch_i = None
    for i, a in enumerate(pref):
        step_legal(env.sim, a)
        if contact_summary(env.sim)["bilateral"] and first_bil is None:
            first_bil = i
        if latched(env.sim) and latch_i is None:
            latch_i = i
    print(f"  perp scoop first_bil={first_bil} latch={latch_i}", flush=True)
    for label, cut in (
        ("B_lift_immediate_after_bil", (first_bil or 8) + 1),
        ("C_lift_before_nominal_end", max((first_bil or 8) + 1, min(cut_p, (latch_i or 20) - 1))),
        ("D_lift_delayed", (first_bil or 8) + 3),
        ("E_lift_grip_mod", (first_bil or 8) + 2),
        ("F_lift_wrist_corr", (first_bil or 8) + 2),
    ):
        if not ok_budget():
            break
        cut = int(np.clip(cut, 1, len(pref) - 1))
        if latch_i is not None:
            cut = min(cut, latch_i - 1)
        p = {
            "n_lift": 35, "lift": -0.95, "n_return": 10, "n_seat": 10, "n_fine": 12,
            "wy": 0.35 if "wrist" in label else 0.25,
            "n_wy": 15 if "wrist" in label else 8,
            "xy_gain": 0.55, "axis_align": "wrist" in label,
            "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1,
        }
        stab = build_stabilization(env, ic_p, pref[:cut], p)
        rollout[0] += 1
        if stab is None:
            lift_extra.append({"ic_id": "perp_off01_r15", "ablation": label, "ok": 0, "cut": cut})
            print(f"  lift {label} stab_fail cut={cut}", flush=True)
            continue
        if "grip_mod" in label:
            for j in range(min(5, len(stab))):
                stab[j][6] = 0.5
                # then re-close
            for j in range(5, min(10, len(stab))):
                stab[j][6] = 1.0
        acts = list(pref[:cut]) + stab
        env._ch_nominal_s = 4.0
        out = evaluate_catch_handoff(env, ic_p, acts, cut, R0p)
        rollout[0] += 1
        lift_extra.append({
            "ic_id": "perp_off01_r15", "ablation": label, "ok": 1, "cut": cut,
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "latch_during": out["pre"]["latch_during_catch"],
            "held_s": out["held_s"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
        })
        print(
            f"  lift {label} cut={cut} CH={out['CATCH_HANDOFF_SUCCESS']} "
            f"SN={out['SUCCESS_NEW']} latch={out['pre']['latch_during_catch']}",
            flush=True,
        )

    # --- Perp-focused new ICs for SUCCESS_NEW (load from manifest pickles) ---
    man = list(csv.DictReader((OUT / "INDEPENDENT_ROOT_MANIFEST.csv").open(encoding="utf-8")))
    perp_ics = [r for r in man if r["direction"] == "perp"]
    for row in perp_ics:
        if not ok_budget():
            break
        pkl = OUT / "trajectories" / f"{row['ic_id']}_ic.pkl"
        if not pkl.exists():
            continue
        with pkl.open("rb") as f:
            ic = pickle.load(f)
        R0 = np.asarray(ic["R0"], float).reshape(3, 3)
        best = None
        for cut_off in (1, 2):
            for _ in range(3):
                if not ok_budget():
                    break
                p = {
                    "n_lift": 35, "lift": -0.95, "n_return": 8, "n_seat": 10, "n_fine": 12,
                    "wy": float(rng.choice([0.0, 0.25, 0.35])),
                    "n_wy": 12, "xy_gain": 0.55, "axis_align": False,
                    "use_fine_down": True, "return_vz": -0.15, "seat_vz": 0.1,
                }
                full, cut, note = compose_expert_from_ic(env, ic, "perp", cut_off, p)
                rollout[0] += 1
                if full is None:
                    continue
                env._ch_nominal_s = 4.0
                out = evaluate_catch_handoff(env, ic, full, cut, R0)
                rollout[0] += 1
                gen_extra.append({
                    "ic_id": row["ic_id"], "cut": cut,
                    "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
                    "SUCCESS_NEW": out["SUCCESS_NEW"],
                    "mean_theta": out["SN_mean_theta"],
                    "mean_pos": out["SN_mean_pos"],
                    "note": "perp_continue",
                })
                if out["SUCCESS_NEW"] or out["CATCH_HANDOFF_SUCCESS"]:
                    print(
                        f"  GEN+ {row['ic_id']} CH={out['CATCH_HANDOFF_SUCCESS']} "
                        f"SN={out['SUCCESS_NEW']} th={out['SN_mean_theta']}",
                        flush=True,
                    )
                if out["SUCCESS_NEW"]:
                    best = (full, cut, out)
                    np.savez_compressed(
                        OUT / "trajectories" / f"{row['ic_id']}__expert_SN.npz",
                        actions=np.asarray(full, np.float32),
                        cut=np.array([cut], np.int32),
                    )
                    break
            if best:
                break

    # --- Distal orientation: stronger align ---
    ic_d, _, _ = load_seed("distal_off01_r8")
    R0d = np.asarray(ic_d["R0"], float).reshape(3, 3)
    catch_d = SRC_CATCH / "trajectories" / "distal_off01_r8__D_scoop_actions.npz"
    pref_d = [np.asarray(a, np.float32) for a in np.load(catch_d)["actions"]]
    for _ in range(25):
        if not ok_budget():
            break
        cut = int(rng.choice([8, 9, 10]))
        p = {
            "n_lift": 40, "lift": -0.9, "n_return": 12, "n_seat": 20, "n_fine": 15,
            "wy": float(rng.uniform(-0.5, 0.5)), "n_wy": 25,
            "xy_gain": 0.5, "axis_align": True, "use_fine_down": True,
            "return_vz": -0.05, "seat_vz": 0.15,
        }
        stab = build_stabilization(env, ic_d, pref_d[:cut], p)
        rollout[0] += 1
        if stab is None:
            continue
        # long axis-align
        restore_ic(env, ic_d)
        clear_probe_latch(env.sim)
        extra = []
        for a in list(pref_d[:cut]) + stab:
            step_legal(env.sim, a)
        for j in range(20):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            a[2] = -0.05
            u = np.asarray(cylinder_axis_hand(env.sim), float)
            ut = U_TGT / (np.linalg.norm(U_TGT) + 1e-12)
            # also try flipping axis (symmetric)
            if float(np.dot(u, ut)) < 0:
                ut = -ut
            c = np.cross(u, ut)
            n = float(np.linalg.norm(c))
            if n > 1e-4:
                a[3:6] = np.clip(0.7 * c / n, -0.8, 0.8)
            step_legal(env.sim, a)
            extra.append(a.copy())
            if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                break
        full = list(pref_d[:cut]) + stab + extra
        env._ch_nominal_s = 4.0
        out = evaluate_catch_handoff(env, ic_d, full, cut, R0d)
        rollout[0] += 1
        ori_extra.append({
            "ic_id": "distal_off01_r8", "cut": cut,
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "mean_theta": out["SN_mean_theta"],
            "mean_pos": out["SN_mean_pos"],
            "e_theta_pre": out["pre"]["e_theta_deg"],
            "n_extra": len(extra),
        })
        if out["SUCCESS_NEW"]:
            print(f"  ORI+ SUCCESS_NEW th={out['SN_mean_theta']}", flush=True)
            np.savez_compressed(
                OUT / "trajectories" / "distal_off01_r8__ori_corrected.npz",
                actions=np.asarray(full, np.float32),
                cut=np.array([cut], np.int32),
            )
            break
        if out["SN_mean_theta"] is not None:
            print(f"  ORI+ th={out['SN_mean_theta']:.2f} CH={out['CATCH_HANDOFF_SUCCESS']}", flush=True)

    # Merge CSVs
    def append_csv(path, rows):
        if not rows:
            return
        existing = list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
        # union keys
        keys = []
        for r in existing + rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in existing + rows:
                w.writerow(r)

    append_csv(OUT / "DISCRETE_ABLATION.csv", disc_extra)
    append_csv(OUT / "LIFT_TIMING_ABLATION.csv", lift_extra)
    append_csv(OUT / "ORIENTATION_CORRECTION.csv", ori_extra)
    # gen rows into a side file + merge verification
    if gen_extra:
        append_csv(OUT / "raw" / "GEN_CONTINUE.csv", gen_extra)

    used_tot = used0 + rollout[0]
    write_json(OUT / "raw" / "USAGE.json", {
        **usage,
        "rollouts_used": used_tot,
        "continue_rollouts": rollout[0],
        "wall_s": float(usage.get("wall_s", 0)) + (time.time() - t0),
        "n_gen_sn": int(usage.get("n_gen_sn", 0)) + sum(1 for r in gen_extra if int(r.get("SUCCESS_NEW") or 0)),
        "n_gen_ch": int(usage.get("n_gen_ch", 0)) + sum(1 for r in gen_extra if int(r.get("CATCH_HANDOFF_SUCCESS") or 0)),
        "distal_ori_sn": int(any(int(r.get("SUCCESS_NEW") or 0) for r in ori_extra)),
        "phase_quantize_ar_ch": sum(1 for r in disc_extra if r["mode"] == "phase_quantize_AR_v1" and int(r["CATCH_HANDOFF_SUCCESS"])),
        "phase_quantize_ar_sn": sum(1 for r in disc_extra if r["mode"] == "phase_quantize_AR_v1" and int(r["SUCCESS_NEW"])),
    })

    # Patch REPORT key answers
    report_path = OUT / "REPORT.md"
    text = report_path.read_text(encoding="utf-8")
    ar_ch = sum(1 for r in disc_extra if r["mode"] == "phase_quantize_AR_v1" and int(r["CATCH_HANDOFF_SUCCESS"]))
    ar_sn = sum(1 for r in disc_extra if r["mode"] == "phase_quantize_AR_v1" and int(r["SUCCESS_NEW"]))
    k49_ch = sum(1 for r in disc_extra if r["mode"] == "phase_quantize_K49" and int(r["CATCH_HANDOFF_SUCCESS"]))
    gen_sn = sum(1 for r in gen_extra if int(r.get("SUCCESS_NEW") or 0))
    gen_ch = sum(1 for r in gen_extra if int(r.get("CATCH_HANDOFF_SUCCESS") or 0))
    ori_sn = any(int(r.get("SUCCESS_NEW") or 0) for r in ori_extra)
    ths = [float(r["mean_theta"]) for r in ori_extra if r.get("mean_theta") not in (None, "")]
    best_th = min(ths) if ths else None
    lift_ok = [r for r in lift_extra if int(r.get("ok") or 0)]
    patch = f"""

## Continue pass (remaining budget)

Used after continue: **{used_tot}/{TOTAL_BUDGET}** (+{rollout[0]}).

### Phase-aware discrete quantization
- AR_v1: CH={ar_ch}/2 seeds, SN={ar_sn}/2
- K49-only phase quantize: CH={k49_ch}/2
- Conclusion: hard-lift tokens are **necessary**; naive nearest-neighbor quantize failed, phase-aware AR quantize {"recovers seed CH" if ar_ch else "still insufficient"}.

### Lift timing B–F (perp)
{json.dumps(lift_ok, indent=2)}

### Perp generalization continue
New CH={gen_ch}, SN={gen_sn}

### Distal orientation continue
SUCCESS_NEW={"yes" if ori_sn else "no"}; best θ={best_th}
"""
    # update Q10 bluntly
    text = text.replace(
        "10. **Can that vocabulary reproduce complete successes?**  \n"
        "    Quantize AR vs K49-only: see `DISCRETE_ABLATION.csv` (AR quantize CH/SN rates vs K49-only). Template policies A–E also reported.",
        f"10. **Can that vocabulary reproduce complete successes?**  \n"
        f"    Phase-aware AR_v1 quantize: **CH={ar_ch}/2, SN={ar_sn}/2** on seeds. "
        f"K49-only phase quantize: CH={k49_ch}/2. "
        f"Naive NN quantize failed (0). Template A–E failed. "
        f"See updated `DISCRETE_ABLATION.csv`.",
    )
    text = text.replace(
        f"**Not under this budget**. Best mean θ among orientation trials: 19.664020075328814.",
        f"**{'Yes' if ori_sn else 'Not under this budget'}**. Best mean θ (continue): {best_th}.",
    )
    text = text.replace(
        "**0** search hits on new ICs ([]). CATCH_HANDOFF hits: **20** (['distal_off01_base_g4', 'distal_off01_yaw15_g5']).",
        f"**{int(usage.get('n_gen_sn', 0)) + gen_sn}** SUCCESS_NEW on new ICs after continue (+{gen_sn}). "
        f"CATCH_HANDOFF: **{int(usage.get('n_gen_ch', 0)) + gen_ch}** cumulative hits.",
    )
    text += patch
    text = text.replace(f"**{used0} / {TOTAL_BUDGET}**", f"**{used_tot} / {TOTAL_BUDGET}**")
    # fix if used0 not in text
    report_path.write_text(text, encoding="utf-8")
    print("Continue done", used_tot, flush=True)


if __name__ == "__main__":
    main()
