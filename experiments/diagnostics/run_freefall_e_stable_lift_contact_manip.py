"""E stable-lift → existing contact manipulation → SUCCESS_NEW.

Attaches *documented* CPP token schedules (tilt→mid-unload→brake) after frozen
freefall controller E reaches a matched stable-lift snapshot.

Does NOT invent a new 6-stage FSM. See MECHANISM_AUDIT.md.
No SFT/GRPO. Physics / SUCCESS_NEW unchanged. Privileged diagnostic first.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.simulation.capture import take_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import measure as gmeasure, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    aperture,
    axial_summary,
    contacts_axial,
    grasp_aware_action,
)
from scripts.run_freefall_recatch_pose_feedback import pose_now
from scripts.run_horizontal_offset_freefall_recatch import (
    SEARCH_NOMINAL_S,
    eval_full_sn,
    render_expert_video,
)
from scripts.run_lateral_offset_falling_recatch import (
    build_nominal_open_robot,
    make_ic,
    make_training_env,
    obj_pos,
    place_cylinder,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch
from training.k47_codebook import decode_action, load_codebook

SRC_GPA = ROOT / "results/diagnostics/freefall_recatch_grasp_point_alignment"
OUT = ROOT / "results/diagnostics/freefall_e_stable_lift_contact_manip"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261106  # match grasp_point_alignment held-out construction
WALL_S = 45 * 60

# Documented CPP schedules (from wrist_tilt / grasp_recovery_pose_v1 SUCCESS evidence)
CPP_RECIPES = {
    "cpp_hrv1_920031": {
        "source": "wrist_tilt_gravity_pivoting VERIFIED_CONTACT_PRESERVING_PIVOT",
        "phases": [("tilt", [6] * 20), ("unload", [12] * 15), ("brake", [0] * 8)],
    },
    "cpp_hard_910101": {
        "source": "grasp_recovery_pose_v1 VERIFIED_FULL_EXPERT continuation",
        "phases": [("tilt", [10] * 8), ("unload", [12] * 15), ("brake", [0] * 5)],
    },
    "cpp_920031_refined": {
        "source": "grasp_recovery_pose_v1 stage34 refined SUCCESS_NEW",
        "phases": [("tilt", [10] * 20), ("unload", [12] * 10), ("brake", [0] * 8)],
    },
}

DIRECT_THETA = 10.0
DIRECT_POS_MM = 12.0
STABLE_BIL_STEPS = 8


def held_ics(env):
    """Same construction as freefall_recatch_grasp_point_alignment Stage 12."""
    robot_cap, R0, mid0, _ = build_nominal_open_robot(env)
    held_specs = [
        ("perp", 0.01), ("perp", 0.012), ("perp", 0.008), ("perp", 0.015),
        ("distal", 0.01), ("distal", 0.012), ("perp", 0.01), ("perp", 0.01),
        ("distal", 0.008), ("perp", 0.01), ("perp", 0.012), ("distal", 0.01),
    ]
    out = []
    for i, (d, off) in enumerate(held_specs):
        # Prefer snapshot from prior GPA run if present (byte-identical IC)
        prior = SRC_GPA / "snapshots"
        # GPA named files by ic_id after make_ic
        ic = make_ic(env, robot_cap, R0, mid0, d, abs(off), f"h{i}")
        if int(ic.get("n_touch_init", 1)) != 0:
            continue
        if i in (6, 7):
            restore_ic(env, ic)
            pos = np.asarray(ic["release_pos"], float)
            pos[2] += 0.01 if i == 6 else -0.01
            s = math.sqrt(0.5)
            place_cylinder(env.sim, pos, np.array([s, s, 0.0, 0.0], float))
            env.sim.data.qpos[env.sim.ids.finger_jnt] = 0.04
            mujoco.mj_forward(env.sim.model, env.sim.data)
            from envs.simulation.capture import take_capture as _tc
            ic["capture"] = _tc(env)
            ic["release_pos"] = pos.tolist()
            ic["ic_id"] = f"{ic['ic_id']}_hz"
        # Prefer GPA snapshot if id matches
        pkl = prior / f"{ic['ic_id']}.pkl"
        if pkl.exists():
            ic = pickle.loads(pkl.read_bytes())
        out.append(ic)
    return out


def tokens_to_actions(book, phases) -> list[np.ndarray]:
    acts = []
    pre = np.zeros(7, np.float32)
    for _name, toks in phases:
        for tok in toks:
            _r, a = decode_action(book, int(tok), pre)
            a = np.asarray(a, np.float32).copy()
            acts.append(a)
            pre = a
    return acts


def run_e_to_stable_lift(env, ic, gains, max_steps=80):
    """Roll frozen E; snapshot at first stable-lift (sustained bil >= STABLE_BIL_STEPS after lift)."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    align = bil_hold = 0
    first_bil = None
    sustained = 0
    acts = []
    snap = None
    for i in range(max_steps):
        a, meta = grasp_aware_action(
            env.sim, gains, "E", align, bil_hold,
            from_obs=False, first_bil_step=first_bil, step_i=i,
        )
        if meta.get("phase") == "align_xy":
            align += 1
        step_legal(env.sim, a)
        acts.append(a.copy())
        cs = contact_summary(env.sim)
        pose = pose_now(env.sim)
        cons, _, _ = contacts_axial(env.sim)
        ax = axial_summary(cons)
        if cs["bilateral"]:
            if first_bil is None:
                first_bil = i
            bil_hold += 1
            sustained += 1
        else:
            if first_bil is not None:
                bil_hold += 1
            sustained = 0
        age = None if first_bil is None else i - first_bil
        # stable-lift: after E enters post_lift (age>=2) and sustained bil long enough
        if (
            snap is None
            and first_bil is not None
            and age is not None
            and age >= 2
            and sustained >= STABLE_BIL_STEPS
            and cs["bilateral"]
            and not latched(env.sim)
        ):
            snap = {
                "step": i,
                "first_bil": first_bil,
                "pose": pose,
                "ax": ax,
                "aper": aperture(env.sim),
                "obj_z": float(obj_pos(env.sim)[2]),
                "capture": take_capture(env),
                "prefix_actions": [a.copy() for a in acts],
            }
            # continue a bit for baseline continuation buffer collection
        if latched(env.sim):
            break
        if first_bil is not None and i >= first_bil + 45:
            break
    baseline_full = list(acts)
    return snap, baseline_full, first_bil


def run_recovery_from_prefix(env, ic, prefix, recovery_acts, phase_names):
    """Replay prefix then recovery; log per-step contact/pose by phase."""
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    rows = []
    for i, a in enumerate(prefix):
        step_legal(env.sim, np.asarray(a, np.float32))
        rows.append(log_row("e_prefix", i, env, a))
        if latched(env.sim):
            return prefix + recovery_acts, rows, "LATCH_DURING_PREFIX"
    for j, a in enumerate(recovery_acts):
        phase = phase_names[j] if j < len(phase_names) else "recovery"
        step_legal(env.sim, np.asarray(a, np.float32))
        rows.append(log_row(phase, len(prefix) + j, env, a))
        if latched(env.sim):
            return list(prefix) + list(recovery_acts[: j + 1]), rows, "LATCH_DURING_RECOVERY"
        if not contact_summary(env.sim)["bilateral"] and phase.startswith("unload"):
            # mid-unload may briefly lose bil; continue but mark
            rows[-1]["note"] = "bilateral_lost"
    return list(prefix) + list(recovery_acts), rows, None


def log_row(phase, step, env, a):
    cs = contact_summary(env.sim)
    cons, _, _ = contacts_axial(env.sim)
    ax = axial_summary(cons)
    pose = pose_now(env.sim)
    return {
        "phase": phase,
        "step": step,
        "bilateral": int(cs["bilateral"]),
        "nL": cs["nL"],
        "nR": cs["nR"],
        "theta": pose["e_theta_deg"],
        "pos_mm": pose["e_pos_mm"],
        "mean_s_axial": ax["mean_s_axial"],
        "n_near_tail": ax["n_near_tail"],
        "n_near_center": ax["n_near_center"],
        "aper": aperture(env.sim),
        "obj_z": float(obj_pos(env.sim)[2]),
        "a6": float(a[6]),
        "a_wy": float(a[4]),
        "latched": int(latched(env.sim)),
    }


def phase_name_list(phases):
    names = []
    for name, toks in phases:
        names.extend([name] * len(toks))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-videos", action="store_true")
    ap.add_argument("--max-ics", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    for sub in ("raw", "snapshots", "trajectories", "videos", "phase_logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    write_json(OUT / "raw" / "BUDGETS.json", {
        "wall_s": WALL_S, "seed": SEED, "stable_bil_steps": STABLE_BIL_STEPS,
        "direct_theta": DIRECT_THETA, "direct_pos_mm": DIRECT_POS_MM,
        "cpp_recipes": {k: v["source"] for k, v in CPP_RECIPES.items()},
        "note": "Privileged diagnostic; CPP schedules from prior verified pilots",
    })

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    book = load_codebook(CODEBOOK)
    gains = dict(BASE_GAINS)

    ics = held_ics(env)[: int(args.max_ics)]
    print(f"ICs: {len(ics)}", flush=True)

    # Pre-decode recipes
    recipe_acts = {
        name: tokens_to_actions(book, spec["phases"])
        for name, spec in CPP_RECIPES.items()
    }
    recipe_phases = {
        name: phase_name_list(spec["phases"])
        for name, spec in CPP_RECIPES.items()
    }

    per_ic = []
    phase_all = []
    full_verify = []
    n_stable = n_direct_attempt = n_direct_sn = 0
    n_recovery_attempt = n_recovery_sn = n_baseline_sn = 0
    combined_sn_ics = set()

    for ic in ics:
        if time.time() - t0 > WALL_S:
            print("WALL", flush=True)
            break
        ic_id = ic["ic_id"]
        print(f"\n=== {ic_id} ===", flush=True)
        (OUT / "snapshots" / f"{ic_id}.pkl").write_bytes(pickle.dumps(ic))

        snap, baseline_full, first_bil = run_e_to_stable_lift(env, ic, gains)
        row = {
            "ic_id": ic_id,
            "direction": ic.get("direction"),
            "offset_m": ic.get("offset_m"),
            "stable_lift": int(snap is not None),
            "first_bil": first_bil,
            "snap_step": None if snap is None else snap["step"],
            "snap_theta": None if snap is None else snap["pose"]["e_theta_deg"],
            "snap_pos_mm": None if snap is None else snap["pose"]["e_pos_mm"],
            "snap_s_axial": None if snap is None else snap["ax"].get("mean_s_axial"),
            "baseline_SN_search": 0,
            "baseline_CH": 0,
            "direct_attempt": 0,
            "direct_SN": 0,
            "recovery_attempt": 0,
            "recovery_best_recipe": None,
            "recovery_SN_search": 0,
            "recovery_SN_12": 0,
            "recovery_CH": 0,
            "combined_SN": 0,
            "fail_baseline": None,
            "fail_recovery": None,
        }

        # --- Baseline: E continuation ---
        if snap is not None:
            n_stable += 1
            # Use collected baseline_full from same rollout (matched prefix)
            # Ensure baseline includes at least snap prefix
            base_acts = baseline_full
            cut = max(1, (first_bil or 1) + 1)
            out_b = eval_full_sn(env, ic, base_acts, cut, nominal_s=SEARCH_NOMINAL_S)
            row["baseline_SN_search"] = int(out_b["SUCCESS_NEW"])
            row["baseline_CH"] = int(out_b["CATCH_HANDOFF_SUCCESS"])
            row["fail_baseline"] = out_b.get("fail_class")
            row["baseline_theta"] = out_b.get("SN_mean_theta")
            row["baseline_pos"] = out_b.get("SN_mean_pos")
            if out_b["SUCCESS_NEW"]:
                n_baseline_sn += 1
                out12 = eval_full_sn(env, ic, base_acts, cut, nominal_s=float(NOMINAL_S))
                ok = 0
                if out12["SUCCESS_NEW"]:
                    ok = sum(
                        int(eval_full_sn(env, ic, base_acts, cut, nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                        for _ in range(3)
                    )
                full_verify.append({
                    "ic_id": ic_id, "branch": "baseline_E", "recipe": "E_continuation",
                    "SN_12s": out12["SUCCESS_NEW"], "replay_ok": ok,
                    "theta": out12.get("SN_mean_theta"), "pos": out12.get("SN_mean_pos"),
                    "provenance": "privileged_E_continuation",
                })
                if ok == 3:
                    combined_sn_ics.add(ic_id)
                    row["combined_SN"] = 1
                    np.savez_compressed(
                        OUT / "trajectories" / f"{ic_id}__baseline_E__SUCCESS_NEW.npz",
                        actions=np.asarray(base_acts, np.float32),
                    )
                    if not args.skip_videos:
                        render_expert_video(
                            env, ic, base_acts,
                            OUT / "videos" / f"{ic_id}__baseline_E__SUCCESS_NEW.mp4",
                            "baseline_E",
                        )
            print(
                f"  stable@ {snap['step']} th={snap['pose']['e_theta_deg']:.1f} "
                f"pos={snap['pose']['e_pos_mm']:.1f} baseSN={row['baseline_SN_search']} "
                f"baseCH={row['baseline_CH']}",
                flush=True,
            )

            # --- Direct ---
            th_s = float(snap["pose"]["e_theta_deg"])
            pos_s = float(snap["pose"]["e_pos_mm"])
            if th_s <= DIRECT_THETA and pos_s <= DIRECT_POS_MM:
                n_direct_attempt += 1
                row["direct_attempt"] = 1
                pref = snap["prefix_actions"]
                # short secure settle then eval
                restore_ic(env, ic)
                clear_probe_latch(env.sim)
                for a in pref:
                    step_legal(env.sim, np.asarray(a, np.float32))
                extra = []
                for _ in range(6):
                    aa = np.zeros(7, np.float32)
                    aa[6] = 1.0
                    aa[2] = -0.5
                    step_legal(env.sim, aa)
                    extra.append(aa.copy())
                    if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                        break
                full_d = list(pref) + extra
                out_d = eval_full_sn(env, ic, full_d, max(1, (first_bil or 1) + 1),
                                     nominal_s=SEARCH_NOMINAL_S)
                row["direct_SN"] = int(out_d["SUCCESS_NEW"])
                row["direct_CH"] = int(out_d["CATCH_HANDOFF_SUCCESS"])
                row["direct_theta"] = out_d.get("SN_mean_theta")
                row["direct_pos"] = out_d.get("SN_mean_pos")
                if out_d["SUCCESS_NEW"]:
                    n_direct_sn += 1
                    out12 = eval_full_sn(env, ic, full_d, max(1, (first_bil or 1) + 1),
                                         nominal_s=float(NOMINAL_S))
                    ok = 0
                    if out12["SUCCESS_NEW"]:
                        ok = sum(
                            int(eval_full_sn(env, ic, full_d, max(1, (first_bil or 1) + 1),
                                             nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                            for _ in range(3)
                        )
                    full_verify.append({
                        "ic_id": ic_id, "branch": "direct", "recipe": "snap_within_tol",
                        "SN_12s": out12["SUCCESS_NEW"], "replay_ok": ok,
                        "theta": out12.get("SN_mean_theta"), "pos": out12.get("SN_mean_pos"),
                        "provenance": "privileged_direct_handoff",
                    })
                    if ok == 3:
                        combined_sn_ics.add(ic_id)
                        row["combined_SN"] = 1

            # --- Recovery: documented CPP recipes ---
            n_recovery_attempt += 1
            row["recovery_attempt"] = 1
            best = None
            pref = snap["prefix_actions"]
            for rname, racts in recipe_acts.items():
                if time.time() - t0 > WALL_S:
                    break
                full_r, plog, early = run_recovery_from_prefix(
                    env, ic, pref, racts, recipe_phases[rname],
                )
                for pr in plog:
                    pr["ic_id"] = ic_id
                    pr["recipe"] = rname
                phase_all.extend(plog)
                write_csv(OUT / "phase_logs" / f"{ic_id}__{rname}.csv", plog)

                cut = max(1, (first_bil or 1) + 1)
                out_r = eval_full_sn(env, ic, full_r, cut, nominal_s=SEARCH_NOMINAL_S)
                score = (
                    1000 * int(out_r["SUCCESS_NEW"])
                    + 200 * int(out_r["CATCH_HANDOFF_SUCCESS"])
                    - float(out_r.get("SN_mean_theta") or 90)
                    - 0.05 * float(out_r.get("SN_mean_pos") or 90)
                )
                print(
                    f"  {rname}: early={early} SN={out_r['SUCCESS_NEW']} CH={out_r['CATCH_HANDOFF_SUCCESS']} "
                    f"th={out_r.get('SN_mean_theta')} pos={out_r.get('SN_mean_pos')} fail={out_r.get('fail_class')}",
                    flush=True,
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score, "name": rname, "acts": full_r, "out": out_r, "early": early,
                    }
            if best is not None:
                row["recovery_best_recipe"] = best["name"]
                row["recovery_SN_search"] = int(best["out"]["SUCCESS_NEW"])
                row["recovery_CH"] = int(best["out"]["CATCH_HANDOFF_SUCCESS"])
                row["fail_recovery"] = best["out"].get("fail_class")
                row["recovery_theta"] = best["out"].get("SN_mean_theta")
                row["recovery_pos"] = best["out"].get("SN_mean_pos")
                np.savez_compressed(
                    OUT / "trajectories" / f"{ic_id}__recovery_{best['name']}.npz",
                    actions=np.asarray(best["acts"], np.float32),
                )
                if best["out"]["SUCCESS_NEW"]:
                    n_recovery_sn += 1
                    cut = max(1, (first_bil or 1) + 1)
                    out12 = eval_full_sn(env, ic, best["acts"], cut, nominal_s=float(NOMINAL_S))
                    ok = 0
                    if out12["SUCCESS_NEW"]:
                        ok = sum(
                            int(eval_full_sn(env, ic, best["acts"], cut,
                                             nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                            for _ in range(3)
                        )
                    row["recovery_SN_12"] = int(out12["SUCCESS_NEW"])
                    full_verify.append({
                        "ic_id": ic_id, "branch": "recovery", "recipe": best["name"],
                        "SN_12s": out12["SUCCESS_NEW"], "replay_ok": ok,
                        "theta": out12.get("SN_mean_theta"), "pos": out12.get("SN_mean_pos"),
                        "provenance": "privileged_CPP_after_E_stable_lift",
                    })
                    if ok == 3:
                        combined_sn_ics.add(ic_id)
                        row["combined_SN"] = 1
                        np.savez_compressed(
                            OUT / "trajectories" / f"{ic_id}__{best['name']}__SUCCESS_NEW.npz",
                            actions=np.asarray(best["acts"], np.float32),
                        )
                        if not args.skip_videos:
                            render_expert_video(
                                env, ic, best["acts"],
                                OUT / "videos" / f"{ic_id}__{best['name']}__SUCCESS_NEW.mp4",
                                best["name"],
                            )
        else:
            print("  NO stable-lift snapshot", flush=True)
            # still eval baseline partial
            if baseline_full and first_bil is not None:
                out_b = eval_full_sn(
                    env, ic, baseline_full, max(1, first_bil + 1),
                    nominal_s=SEARCH_NOMINAL_S,
                )
                row["baseline_SN_search"] = int(out_b["SUCCESS_NEW"])
                row["baseline_CH"] = int(out_b["CATCH_HANDOFF_SUCCESS"])
                row["fail_baseline"] = out_b.get("fail_class")

        # save snap blob
        if snap is not None:
            blob = {
                "ic_id": ic_id,
                "step": snap["step"],
                "first_bil": snap["first_bil"],
                "pose": snap["pose"],
                "ax": snap["ax"],
                "aper": snap["aper"],
                "obj_z": snap["obj_z"],
            }
            write_json(OUT / "snapshots" / f"{ic_id}__stable_lift_meta.json", blob)
            # keep capture inside pkl with actions
            pickle.dump(
                {"ic": ic, "capture": snap["capture"], "prefix": snap["prefix_actions"],
                 "meta": blob},
                (OUT / "snapshots" / f"{ic_id}__stable_lift.pkl").open("wb"),
            )

        per_ic.append(row)

    write_csv(OUT / "PER_IC_COMPARISON.csv", per_ic)
    write_csv(OUT / "PHASE_CONTACT_TRACES.csv", phase_all if phase_all else [{"note": "empty"}])
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full_verify if full_verify else [{"note": "none"}])

    summary = {
        "n_ics": len(per_ic),
        "n_stable_lift": n_stable,
        "n_baseline_SN_search": n_baseline_sn,
        "n_direct_attempt": n_direct_attempt,
        "n_direct_SN_search": n_direct_sn,
        "n_recovery_attempt": n_recovery_attempt,
        "n_recovery_SN_search": n_recovery_sn,
        "n_combined_SN_roots_triple": len(combined_sn_ics),
        "combined_ic_ids": sorted(combined_sn_ics),
        "wall_s": time.time() - t0,
    }
    write_json(OUT / "raw" / "summary.json", summary)

    # failure tallies
    from collections import Counter
    fail_b = Counter(r.get("fail_baseline") for r in per_ic if r.get("stable_lift"))
    fail_r = Counter(r.get("fail_recovery") for r in per_ic if r.get("recovery_attempt"))
    write_json(OUT / "raw" / "failure_counts.json", {
        "baseline": dict(fail_b), "recovery": dict(fail_r),
    })

    report = f"""# E Stable-Lift → Contact Manipulation → SUCCESS_NEW

**Script:** `scripts/run_freefall_e_stable_lift_contact_manip.py`  
**Wall:** {summary['wall_s']:.1f}s  
**Audit:** `MECHANISM_AUDIT.md` (read first)

## Mechanism status (pre-registered)

- Full user 6-stage slide→reclamp→90°→pivot→slide pipeline as one verified unit: **MISSING**
- Attached recovery: **documented CPP** tilt→mid-unload→brake schedules with prior SUCCESS evidence
  (`cpp_hrv1_920031`, `cpp_hard_910101`, `cpp_920031_refined`)
- Domain: privileged diagnostic (open-loop token schedules after E). Not obs-only policy.

## Counts (frozen E, {summary['n_ics']} held-out ICs)

| Metric | N |
|--------|--:|
| Stable lift snapshots | {summary['n_stable_lift']} |
| Baseline E SUCCESS_NEW (search) | {summary['n_baseline_SN_search']} |
| Direct attempt (snap within 10°/12 mm) | {summary['n_direct_attempt']} |
| Direct SUCCESS_NEW (search) | {summary['n_direct_SN_search']} |
| Recovery attempted | {summary['n_recovery_attempt']} |
| Recovery SUCCESS_NEW (search) | {summary['n_recovery_SN_search']} |
| Combined triple-verified SUCCESS_NEW roots | **{summary['n_combined_SN_roots_triple']}** |

Combined IC ids: `{summary['combined_ic_ids']}`

## Per-IC

See `PER_IC_COMPARISON.csv`. Phase contact logs: `phase_logs/`, `PHASE_CONTACT_TRACES.csv`.

## Failure reasons

Baseline: `{dict(fail_b)}`  
Recovery: `{dict(fail_r)}`

## Legality

- **Privileged diagnostic:** yes (E uses MuJoCo state; CPP is open-loop decoded K48 schedule).
- **39D observation-only:** **not claimed** this round.
- Physics / SUCCESS_NEW / 12 s verification unchanged. No SFT/GRPO.

## Files touched

- `scripts/run_freefall_e_stable_lift_contact_manip.py` (new)
- `results/diagnostics/freefall_e_stable_lift_contact_manip/**` (new)
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    fail_md = f"""# FAILURE_ANALYSIS

## Baseline E after stable lift
{dict(fail_b)}

## Recovery (CPP after E)
{dict(fail_r)}

## Interpretation

If recovery fails with `E_cylinder_orientation_error`, CPP did not finish correcting
orientation on freefall-E states (domain shift from hard-recovery ICs).

If `B_relative_motion_too_high` / latch, unload mid-grip lost the freefall catch.

If `stable_lift=0`, E never retained bilateral long enough for a matched snapshot.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
