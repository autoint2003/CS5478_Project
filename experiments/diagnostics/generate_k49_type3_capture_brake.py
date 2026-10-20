"""Type-3 capture–brake search from verified eccentric-catch intermediate states.

Locked K49_tok48_wy0p25 + SUCCESS_NEW. No SFT/GRPO.
Branch from contact-retaining mechanism candidates in k49_horizontal_endcatch_recatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.ballistic_scene import CYL_H
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import cylinder_axis_world, measure, object_R
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import (
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    describe_k49_config,
)
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env

from scripts.generate_k49_dynamic_recatch_experts import (
    CYL_HALF,
    analyze_type3_mechanism,
    contact_axial_offset,
    finger_object_contacts,
    find_bilateral_after_loss,
    has_post_manipulation,
    rich_state,
    run_traced,
)
from scripts.generate_k49_horizontal_endcatch_recatch import (
    NEAR_MISS_SEEDS,
    analyze_capture_rotation_braking,
    brake_refine_around,
    endcatch_proposals,
    horiz_ic_metrics,
    snapshot_path,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    mapping_checksum,
    run_seq,
    tokens_aligned,
    write_csv,
    write_json,
)

SRC = ROOT / "results" / "expert_generation" / "k49_horizontal_endcatch_recatch"
OUT = ROOT / "results" / "expert_generation" / "k49_type3_capture_brake"
SEED = 20261019


def rebuild_search_queue(book, desire, rng, ic_id, budget=40, max_extra=24):
    """Replay horizontal-endcatch queue construction for exact token sequences."""
    seeds = NEAR_MISS_SEEDS.get(ic_id)
    props = endcatch_proposals(book, desire, rng, n=budget, seeds=seeds)
    return list(props), max_extra


def reconstruct_candidate_sequences(env, table, book48, blob, ic_id, tags_needed, budget=40, max_extra=24):
    """Rebuild queue online, matching generator refine triggers, return tag→tokens."""
    rng = np.random.default_rng(SEED)
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    queue, max_ex = rebuild_search_queue(book48, desire, rng, ic_id, budget=budget, max_extra=max_extra)
    seen = {tuple(s) for s in queue}
    n_extra = 0
    tag_to_seq = {}
    need = set(tags_needed)
    ip = -1
    while ip + 1 < len(queue) and need:
        ip += 1
        seq = queue[ip]
        tag = f"t3_{ip}"
        if tag in need:
            tag_to_seq[tag] = list(seq)
            need.discard(tag)
        # refine trigger identical to generator
        ok, trace = run_traced(env, table, book48, blob, seq)
        m3 = analyze_type3_mechanism(trace)
        if (
            m3.get("eccentric") and m3.get("downward_motion") and m3.get("beneficial_rotation")
            and m3.get("bilateral_i") is None and n_extra < max_ex
        ):
            for v in brake_refine_around(seq):
                k = tuple(v)
                if k in seen:
                    continue
                seen.add(k)
                queue.append(v)
                n_extra += 1
                if n_extra >= max_ex:
                    break
        # stop early if we have all needed and won't need further refine indices
        if not need and ip >= budget - 1 and n_extra >= max_ex:
            break
    # if some tags still missing (refine indices), continue until found
    while need and ip + 1 < len(queue):
        ip += 1
        seq = queue[ip]
        tag = f"t3_{ip}"
        if tag in need:
            tag_to_seq[tag] = list(seq)
            need.discard(tag)
        if (
            n_extra < max_ex
        ):
            ok, trace = run_traced(env, table, book48, blob, seq)
            m3 = analyze_type3_mechanism(trace)
            if (
                m3.get("eccentric") and m3.get("downward_motion") and m3.get("beneficial_rotation")
                and m3.get("bilateral_i") is None
            ):
                for v in brake_refine_around(seq):
                    k = tuple(v)
                    if k in seen:
                        continue
                    seen.add(k)
                    queue.append(v)
                    n_extra += 1
                    if n_extra >= max_ex:
                        break
    return tag_to_seq, len(queue)


def run_traced_with_snapshots(env, table, book48, blob, tokens):
    """Like run_traced but stores controller captures before each step (branch points)."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    trace = []
    snaps = []
    lost_low = 0
    for tok in tokens:
        st = rich_state(env.sim)
        ax, _ = contact_axial_offset(env.sim)
        cons = finger_object_contacts(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        fj = env.sim.ids.finger_jnt
        fq = [float(env.sim.data.qpos[int(j)]) for j in fj]
        fv = [float(env.sim.data.qvel[int(env.sim.model.jnt_dofadr[int(j)])]) for j in fj]
        # capture-only is sufficient and matches end-to-end prefix replay
        cap = take_capture(env)
        snaps.append({"capture": cap})
        trace.append({
            "t": float(env.sim.data.time) - t0,
            "tok": int(tok),
            "nL": st["nL"], "nR": st["nR"], "bilateral": st["bilateral"],
            "e_pos_mm": st["e_pos_mm"], "e_ax_mm": st["e_ax_mm"], "e_lat_mm": st["e_lat_mm"],
            "e_theta_deg": st["e_theta_deg"],
            "v_rel_h": st["v_rel_h"], "w_rel_h": st["w_rel_h"],
            "obj_z": st["obj_z"], "hand_pos": ph.tolist(),
            "p_c_h": np.asarray(st["p_c_h"], float).tolist(),
            "contact_axial": ax,
            "contacts": [{"side": c["side"], "pos": c["pos"].tolist(), "dist": c["dist"]} for c in cons],
            "finger_q": fq, "finger_dq": fv, "fg_cmd": float(env.sim.fsm.fg_cmd),
        })
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        from envs.outcomes.horizon_outcome import recovery_failed
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return False, trace, snaps
    return True, trace, snaps


def restore_branch(env, snap):
    restore_capture(env, snap["capture"])


def pick_branch_indices(trace):
    """Select branch points: first eccentric recontact, peak |w|, max useful Δθ, pre-loss."""
    if not trace:
        return {}
    m3 = analyze_type3_mechanism(trace)
    fc = m3.get("first_contact_i")
    if fc is None:
        # fallback: any contact while theta improving
        contact_any = [i for i, r in enumerate(trace) if int(r["nL"]) + int(r["nR"]) > 0]
        if not contact_any:
            return {}
        fc = contact_any[0]

    # first eccentric recontact
    first_ecc = None
    for i in range(fc, len(trace)):
        ax = trace[i].get("contact_axial")
        any_c = int(trace[i]["nL"]) + int(trace[i]["nR"]) > 0
        if any_c and ax and ax.get("near_end"):
            first_ecc = i
            break
    if first_ecc is None:
        first_ecc = fc

    # window while any contact after first_ecc
    contact_idxs = [
        i for i in range(first_ecc, len(trace))
        if int(trace[i]["nL"]) + int(trace[i]["nR"]) > 0
    ]
    if not contact_idxs:
        contact_idxs = [first_ecc]

    th0 = float(trace[first_ecc]["e_theta_deg"])
    # peak |w_rel|
    peak_w_i = max(contact_idxs, key=lambda i: float(np.linalg.norm(trace[i]["w_rel_h"])))
    # max useful orientation correction while contact remains (min theta)
    best_th_i = min(contact_idxs, key=lambda i: float(trace[i]["e_theta_deg"]))
    # immediately before final contact loss
    last_c = contact_idxs[-1]
    pre_loss = last_c
    # if next step loses contact, pre_loss is last_c; else last contact step
    if last_c + 1 < len(trace):
        pre_loss = last_c

    out = {
        "first_eccentric_recontact": first_ecc,
        "peak_angular_speed": peak_w_i,
        "max_useful_orientation": best_th_i,
        "before_final_contact_loss": pre_loss,
    }
    # dedupe preserving order
    seen, ordered = set(), []
    for name, idx in out.items():
        if idx not in seen:
            seen.add(idx)
            ordered.append((name, idx))
    return {"named": out, "unique": ordered, "m3": m3, "theta_at_first_ecc": th0}


def diagnose_geometry(env, blob, trace_row, branch_name):
    """Contact geometry diagnosis at current env state (caller restores snapshot)."""
    sim = env.sim
    m = measure(sim, U_TGT, P_TGT)
    d = decompose(sim)
    cs = contact_summary(sim)
    cons = finger_object_contacts(sim)
    ax, details = contact_axial_offset(sim)
    Ro = object_R(sim)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    u = cylinder_axis_world(sim)
    # free end positions
    end_p = po + CYL_HALF * u
    end_m = po - CYL_HALF * u
    # finger pads approx
    left = np.asarray(sim.data.xpos[sim.ids.left_body], float)
    right = np.asarray(sim.data.xpos[sim.ids.right_body], float)
    # which finger contacting
    sides = sorted({c["side"] for c in cons})
    contacting = ",".join(sides) if sides else "none"
    # free end relative to opposing finger
    if sides == ["L"]:
        free_end = end_p if np.linalg.norm(end_p - left) > np.linalg.norm(end_m - left) else end_m
        opp = right
        opp_side = "R"
    elif sides == ["R"]:
        free_end = end_p if np.linalg.norm(end_p - right) > np.linalg.norm(end_m - right) else end_m
        opp = left
        opp_side = "L"
    else:
        free_end = end_p
        opp = 0.5 * (left + right)
        opp_side = "both_or_none"
    dist_free_to_opp = float(np.linalg.norm(free_end - opp))

    fj = sim.ids.finger_jnt
    fq = [float(sim.data.qpos[int(j)]) for j in fj]
    fv = [float(sim.data.qvel[int(sim.model.jnt_dofadr[int(j)])]) for j in fj]
    w = np.asarray(m["w_rel_h"], float)
    v = np.asarray(m["v_rel_h"], float)

    # failure mode heuristics
    opening = float(np.mean(fq))
    closing_vel = float(-np.mean(fv))  # positive if closing
    modes = []
    if opening > 0.03 and not cs["bilateral"]:
        modes.append("gripper_too_open")
    if dist_free_to_opp > 0.04 and len(sides) == 1:
        modes.append("cylinder_outside_opposing_finger_reach")
    if float(np.linalg.norm(w)) > 3.0:
        modes.append("excessive_relative_angular_velocity")
    if len(sides) == 1 and float(np.linalg.norm(v)) > 0.15:
        modes.append("escaping_before_grip_force")
    if abs(float(m["e_theta_deg"])) > 25 and len(sides) <= 1:
        modes.append("unfavorable_orientation_for_bilateral")
    if not modes:
        modes.append("timing_or_control_authority_unclear")

    return {
        "branch": branch_name,
        "contacting_fingers": contacting,
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": bool(cs["bilateral"]),
        "axial_max_m": None if not ax else ax["max_abs_m"],
        "axial_mean_m": None if not ax else ax["mean_abs_m"],
        "contact_details": details,
        "u_c_w": u.tolist(),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "v_rel_h": v.tolist(),
        "w_rel_h": w.tolist(),
        "w_norm": float(np.linalg.norm(w)),
        "v_norm": float(np.linalg.norm(v)),
        "finger_q": fq,
        "finger_dq": fv,
        "fg_cmd": float(sim.fsm.fg_cmd),
        "opening_mean": opening,
        "closing_vel_proxy": closing_vel,
        "dist_free_end_to_opp_m": dist_free_to_opp,
        "opp_side": opp_side,
        "hand_pos": np.asarray(sim.data.xpos[sim.ids.hand_body], float).tolist(),
        "p_des": None if sim.fsm.p_des is None else np.asarray(sim.fsm.p_des, float).tolist(),
        "failure_modes": modes,
        "trace_e_theta": None if not trace_row else trace_row.get("e_theta_deg"),
        "trace_tok": None if not trace_row else trace_row.get("tok"),
    }


def capture_brake_proposals(book, desire, rng, n=28):
    """Continuations emphasizing wrist follow, COM track, progressive grip."""
    sec = tokens_aligned(book, desire if desire is not None else np.array([0, 0, -1.0]), "secure", 6) or [
        1, 5, 8, 13, 15, 17, 33
    ]
    chase = list(dict.fromkeys(list(sec[:4]) + [17, 33, 1, 5, 8, 15]))
    seqs = []
    # pure secure / progressive close
    for n0 in (4, 8, 12, 16, 24):
        seqs.append([0] * n0)
        seqs.append([12] * min(6, n0 // 2) + [0] * n0)
        seqs.append([12] * 4 + [0] * 4 + [12] * 4 + [0] * 12)
    # wrist follow then close
    for piv in (4, 6, 10):
        for npiv in (4, 8, 12):
            seqs.append([piv] * npiv + [0] * 16)
            seqs.append([piv] * npiv + [12] * 6 + [0] * 12)
            seqs.append([12] * 3 + [piv] * npiv + [0] * 14)
            seqs.append([piv] * (npiv // 2) + [12] * 4 + [piv] * (npiv // 2) + [0] * 12)
    # translate/track then close
    for t in chase[:5]:
        seqs.append([t] * 6 + [0] * 16)
        seqs.append([t] * 4 + [12] * 4 + [0] * 14)
        seqs.append([t] * 4 + [4] * 6 + [0] * 14)
        seqs.append([12] * 2 + [t] * 6 + [12] * 4 + [0] * 12)
        seqs.append([t] * 3 + [6] * 6 + [12] * 6 + [0] * 10)
    # soft hold + wrist + secure
    seqs.append([12] * 8 + [4] * 8 + [0] * 12)
    seqs.append([12] * 8 + [6] * 8 + [0] * 12)
    seqs.append([12] * 8 + [10] * 8 + [0] * 12)
    seqs.append([0] * 4 + [4] * 8 + [12] * 6 + [0] * 10)
    seqs.append([0] * 4 + [33] * 6 + [0] * 14)
    seqs.append([0] * 4 + [17] * 6 + [0] * 14)
    # reverse wrist settle
    seqs.append([SECURE_REVERSE_WRIST] * 8 + [0] * 12)
    seqs.append([12] * 4 + [SECURE_REVERSE_WRIST] * 8 + [0] * 10)

    seen, uniq = set(), []
    for s in seqs:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:12], uniq[12:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def evaluate_milestones(trace_cont, trace_prefix_tail=None):
    """A–D diagnostic milestones on continuation (+ optional prefix context)."""
    if not trace_cont:
        return {"A": False, "B": False, "C": False, "D": False, "details": {}}
    # A: contact preserved during beneficial rotation in continuation
    thetas = [float(r["e_theta_deg"]) for r in trace_cont]
    contacts = [int(r["nL"]) + int(r["nR"]) > 0 for r in trace_cont]
    dth = thetas[0] - min(thetas) if thetas else 0.0
    A = bool(sum(contacts) >= max(2, int(0.4 * len(contacts))) and dth >= 1.0) or (
        sum(contacts) >= max(2, int(0.5 * len(contacts)))
    )
    # B: angular velocity reduced
    w0 = float(np.linalg.norm(trace_cont[0]["w_rel_h"]))
    w_min = min(float(np.linalg.norm(r["w_rel_h"])) for r in trace_cont)
    B = bool(w_min < w0 - 0.1 or w_min < 0.5 * w0)
    # C: bilateral restored
    bil_idxs = [i for i, r in enumerate(trace_cont) if r["bilateral"]]
    C = bool(bil_idxs)
    # D: bilateral stable interval
    D = False
    hold = 0
    max_hold = 0
    for r in trace_cont:
        if r["bilateral"]:
            hold += 1
            max_hold = max(max_hold, hold)
        else:
            hold = 0
    D = bool(max_hold >= 5)
    return {
        "A": A, "B": B, "C": C, "D": D,
        "details": {
            "contact_frac": float(np.mean(contacts)) if contacts else 0.0,
            "dth_cont": dth,
            "w0": w0, "w_min": w_min,
            "bilateral_i": None if not bil_idxs else bil_idxs[0],
            "max_bilateral_run": max_hold,
            "final_theta": thetas[-1] if thetas else None,
            "final_pos": float(trace_cont[-1]["e_pos_mm"]) if trace_cont else None,
        },
    }


def save_full_expert(env, ic_id, tag, res, mech, mapping_meta, checksum, snap_src, extra):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "tag": tag, "family": "type3_capture_brake",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "snapshot_src": str(snap_src), "obs_dim": 39, "handoff_id": 47,
        **extra,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "extra": extra,
        })
    return meta


def select_candidates():
    """Load 45 supported + diverse 318 contact-retaining high-rot candidates."""
    cands = list(csv.DictReader((SRC / "ENDCATCH_CANDIDATES.csv").open(encoding="utf-8")))
    phases = {
        (r["ic_id"], r["tag"]): r
        for r in csv.DictReader((SRC / "CAPTURE_ROTATION_BRAKING.csv").open(encoding="utf-8"))
    }
    supported = [
        c for c in cands
        if c["eccentric"] == "1" and c["downward"] == "1"
        and c["beneficial_rot"] == "1" and c["supported"] == "1"
    ]
    # diversify 031: unique-ish by tag groups (seeds + refine)
    # keep all 45 but search budget will subsample branches
    extra_318 = []
    for c in cands:
        if c["ic_id"] != "hard_910318":
            continue
        if not (c["eccentric"] == "1" and c["downward"] == "1" and c["beneficial_rot"] == "1"):
            continue
        ph = phases.get((c["ic_id"], c["tag"]), {})
        frac = float(ph["C_contact_frac"]) if ph.get("C_contact_frac") not in (None, "") else 0.0
        if frac >= 0.3 and float(c["delta_theta_deg"]) >= 5.0:
            extra_318.append(c)
    # dedupe similar 318 by delta_theta bins
    extra_318.sort(key=lambda x: -float(x["delta_theta_deg"]))
    picked_318, seen_dth = [], set()
    for c in extra_318:
        key = round(float(c["delta_theta_deg"]), 1)
        if key in seen_dth:
            continue
        seen_dth.add(key)
        picked_318.append(c)
        if len(picked_318) >= 8:
            break
    return supported, picked_318


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-cont", type=int, default=20)
    ap.add_argument("--max-branches-per-cand", type=int, default=3)
    ap.add_argument("--max-cands-031", type=int, default=12)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0wall = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW", "branch_snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    mapping_meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", mapping_meta)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    supported, extra_318 = select_candidates()
    # subsample 031 for budget: prefer early tags (seeds) + some refine
    supported.sort(key=lambda c: int(c["tag"].split("_")[1]))
    cands_031 = supported[: args.max_cands_031]
    if args.smoke:
        cands_031 = cands_031[:3]
        extra_318 = extra_318[:2]
    work = [("hrv1_920031", cands_031), ("hard_910318", extra_318)]
    print(f"WORK 031={len(cands_031)} 318={len(extra_318)}", flush=True)

    branch_rows = []
    diag_rows = []
    candidates_out = []
    full_experts = []
    hybrids = []
    failures = []
    coverage = []
    beyond = []
    n_rollouts = 0
    action_help = Counter()

    for root_id, cand_list in work:
        sp = snapshot_path(root_id)
        if sp is None:
            print("MISSING", root_id, flush=True)
            continue
        blob = pickle.loads(sp.read_bytes())
        shutil.copy2(sp, OUT / "snapshots" / f"{root_id}.pkl")
        tags = [c["tag"] for c in cand_list]
        print(f"\n=== ROOT {root_id} reconstruct {len(tags)} tags ===", flush=True)
        tag_to_seq, qlen = reconstruct_candidate_sequences(
            env, table, book48, blob, root_id, tags, budget=40, max_extra=24
        )
        print(f"  reconstructed {len(tag_to_seq)}/{len(tags)} queue_len~{qlen}", flush=True)
        write_json(OUT / "RAW" / f"{root_id}_tag_tokens.json", {k: v for k, v in tag_to_seq.items()})

        root_stats = {
            "root_episode": root_id, "n_cands": len(cand_list),
            "n_reconstructed": len(tag_to_seq),
            "n_branches": 0, "n_C_bilateral": 0, "n_D_stable": 0,
            "n_full": 0, "n_hybrid": 0, "n_beyond": 0,
        }

        for ci, cand in enumerate(cand_list):
            tag = cand["tag"]
            seq = tag_to_seq.get(tag)
            if seq is None:
                failures.append({"ic_id": root_id, "tag": tag, "reason": "TOKEN_RECONSTRUCT_FAIL"})
                continue
            ok, trace, snaps = run_traced_with_snapshots(env, table, book48, blob, seq)
            n_rollouts += 1
            picks = pick_branch_indices(trace)
            if not picks:
                failures.append({"ic_id": root_id, "tag": tag, "reason": "NO_BRANCH_POINTS"})
                continue

            # unique branches limited
            branches = picks["unique"][: args.max_branches_per_cand]
            # prefer max_useful and before_loss and first_ecc
            prefer = ["max_useful_orientation", "before_final_contact_loss", "first_eccentric_recontact", "peak_angular_speed"]
            named = picks["named"]
            ordered = []
            for name in prefer:
                if name in named:
                    idx = named[name]
                    if not any(i == idx for _, i in ordered):
                        ordered.append((name, idx))
            branches = ordered[: args.max_branches_per_cand]

            any_progress = False
            for bname, bidx in branches:
                prefix = seq[: bidx + 1]  # include token at branch? branch snap is BEFORE token bidx
                # snapshot at step bidx is state BEFORE executing tokens[bidx]
                # continuation should start FROM that state, so prefix for end-to-end is seq[:bidx]
                prefix_tokens = seq[:bidx]
                snap = snaps[bidx]
                stem = f"{root_id}__{tag}__{bname}_i{bidx}"
                # save branch snapshot (capture only for restore_capture path + full snap)
                branch_blob = {
                    "capture": snap["capture"],
                    "ref": blob["ref"],
                    "meta": {
                        "root_ic": root_id, "source_tag": tag, "branch": bname,
                        "branch_idx": bidx, "prefix_tokens": prefix_tokens,
                        "full_prefix_seq": seq,
                    },
                }
                # pickle MjData not always portable — store capture; also keep via take_capture path
                (OUT / "branch_snapshots" / f"{stem}.pkl").write_bytes(pickle.dumps(branch_blob))

                restore_branch(env, snap)
                diag = diagnose_geometry(env, blob, trace[bidx], bname)
                diag_rows.append({
                    "root_ic": root_id, "tag": tag, "branch": bname, "idx": bidx,
                    **{k: diag[k] for k in (
                        "contacting_fingers", "nL", "nR", "bilateral", "e_theta_deg", "e_pos_mm",
                        "w_norm", "v_norm", "opening_mean", "closing_vel_proxy",
                        "dist_free_end_to_opp_m", "opp_side", "fg_cmd",
                    )},
                    "failure_modes": json.dumps(diag["failure_modes"]),
                    "axial_max_mm": None if diag["axial_max_m"] is None else 1e3 * diag["axial_max_m"],
                    "finger_q": json.dumps(diag["finger_q"]),
                    "w_rel_h": json.dumps(diag["w_rel_h"]),
                    "v_rel_h": json.dumps(diag["v_rel_h"]),
                })
                branch_rows.append({
                    "branch_id": stem, "root_ic": root_id, "source_tag": tag,
                    "branch_name": bname, "branch_idx": bidx,
                    "prefix_len": len(prefix_tokens),
                    "e_theta": diag["e_theta_deg"], "e_pos": diag["e_pos_mm"],
                    "w_norm": diag["w_norm"], "contacting": diag["contacting_fingers"],
                    "modes": json.dumps(diag["failure_modes"]),
                    "delta_theta_src": cand.get("delta_theta_deg"),
                    "ok_prefix_src": cand.get("ok_prefix"),
                })
                root_stats["n_branches"] += 1

                # search continuations from branch
                restore_branch(env, snap)
                desire = -decompose(env.sim)["err_h"]
                rng = np.random.default_rng(SEED + bidx + 17 * ci)
                conts = capture_brake_proposals(book48, desire, rng, n=args.budget_cont)
                best_local = None
                no_improve = 0
                got_near_new = False
                for jc, cont in enumerate(conts):
                    n_rollouts += 1
                    # run continuation from branch snapshot
                    restore_branch(env, snap)
                    # build mini-blob for run_traced
                    branch_for_run = {"capture": snap["capture"], "ref": blob["ref"]}
                    ok_c, tr_c = run_traced(env, table, book48, branch_for_run, cont)
                    miles = evaluate_milestones(tr_c)
                    # score
                    score = (
                        10 * int(miles["D"]) + 5 * int(miles["C"]) + 2 * int(miles["B"]) + int(miles["A"])
                        - 0.05 * (miles["details"].get("final_theta") or 50)
                        - 0.02 * (miles["details"].get("final_pos") or 50)
                    )
                    row = {
                        "root_ic": root_id, "tag": tag, "branch": bname, "branch_idx": bidx,
                        "cont_i": jc, "ok_cont": int(ok_c),
                        "A": int(miles["A"]), "B": int(miles["B"]),
                        "C": int(miles["C"]), "D": int(miles["D"]),
                        "score": score,
                        "final_theta": miles["details"].get("final_theta"),
                        "final_pos": miles["details"].get("final_pos"),
                        "w0": miles["details"].get("w0"),
                        "w_min": miles["details"].get("w_min"),
                        "max_bil_run": miles["details"].get("max_bilateral_run"),
                        "contact_frac": miles["details"].get("contact_frac"),
                        "tokens_cont": json.dumps(cont),
                    }
                    candidates_out.append(row)

                    if miles["C"]:
                        root_stats["n_C_bilateral"] += 1
                        any_progress = True
                        # attribute helpful tokens
                        for t in cont[:8]:
                            action_help[int(t)] += 1
                    if miles["D"]:
                        root_stats["n_D_stable"] += 1
                        any_progress = True

                    improved = best_local is None or score > best_local["score"] + 0.01
                    if improved:
                        best_local = row
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= 8 and best_local and best_local["score"] < 5:
                        break  # stop this branch early

                    # If C or D, try full end-to-end reconstruction
                    if miles["C"] or miles["D"]:
                        base = list(prefix_tokens) + list(cont)
                        variants = [
                            ("raw", base + [0] * 3),
                            ("settle8", base + [0] * 8),
                            ("settle20", base + [0] * 20),
                            ("settle40", base + [0] * 40),
                            ("soft_settle", base + [12] * 6 + [0] * 20),
                        ]
                        restore_capture(env, blob["capture"])
                        run_seq(env, table, book48, blob, base)
                        herr = hand_ref_error_deg(env, blob)
                        if herr > 6:
                            for n48 in (8, 12, 20, 28):
                                variants.append((
                                    f"ret48x{n48}",
                                    base + [SECURE_REVERSE_WRIST] * n48 + [0] * 12,
                                ))
                                variants.append((
                                    f"settle12_ret48x{n48}",
                                    base + [0] * 12 + [SECURE_REVERSE_WRIST] * n48 + [0] * 8,
                                ))
                        for vname, ftoks in variants:
                            n_rollouts += 1
                            res = evaluate_full(env, table, book48, blob, ftoks, record_obs=False)
                            ok2, tr2 = run_traced(env, table, book48, blob, ftoks)
                            m3 = analyze_type3_mechanism(tr2)
                            ph = analyze_capture_rotation_braking(tr2)
                            post, pinfo = has_post_manipulation(tr2, m3.get("bilateral_i"))
                            miles_full = evaluate_milestones(tr2)
                            # max orientation improvement during contact window (recapture)
                            rot_window = 0.0
                            fc = m3.get("first_contact_i")
                            bil_i = m3.get("bilateral_i")
                            if fc is not None:
                                end_r = bil_i if bil_i is not None else min(len(tr2) - 1, fc + 20)
                                ths = [float(tr2[i]["e_theta_deg"]) for i in range(fc, end_r + 1)]
                                if ths:
                                    rot_window = float(ths[0] - min(ths))
                            src_ben = float(cand.get("delta_theta_deg") or 0) >= 3.0
                            significant_rot = bool(
                                m3.get("beneficial_rotation")
                                or rot_window >= 3.0
                                or (src_ben and rot_window >= 2.0)
                            )
                            pure_type3 = bool(
                                m3.get("eccentric") and m3.get("downward_motion")
                                and significant_rot
                                and m3.get("bilateral_i") is not None
                                and (not post)
                                and (ph.get("supported_while_rotating") or miles_full["D"] or miles_full["C"])
                            )
                            hold = res["window"].get("longest_both_ok_s")
                            near = (
                                res["success_12s"]
                                and (res["window"].get("mean_theta") or 99) <= 11
                                and (res["window"].get("mean_pos") or 99) <= 14
                            )
                            if near:
                                got_near_new = True

                            if res["SUCCESS_NEW"] and pure_type3 and miles_full["D"]:
                                res = evaluate_full(env, table, book48, blob, ftoks, record_obs=True)
                                mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                                save_full_expert(
                                    env, root_id, f"full_{tag}_{bname}_{vname}", res, mech,
                                    mapping_meta, checksum, sp,
                                    {"expert_type": "TYPE3_DYNAMIC_RECATCH", "branch": bname,
                                     "m3": m3, "milestones": miles_full},
                                )
                                full_experts.append({
                                    "ic_id": root_id, "tag": f"full_{tag}_{bname}_{vname}",
                                    "expert_type": "TYPE3_DYNAMIC_RECATCH", "SUCCESS_NEW": 1,
                                    "win_mean_pos": res["window"].get("mean_pos"),
                                    "win_mean_theta": res["window"].get("mean_theta"),
                                    "hold_both_s": hold,
                                    "tokens": json.dumps(ftoks), "branch": bname,
                                    "mapping_checksum": checksum,
                                })
                                root_stats["n_full"] += 1
                                print(f"  FULL TYPE3 {root_id} {tag} {bname} {vname} "
                                      f"th={res['window'].get('mean_theta'):.1f} "
                                      f"pos={res['window'].get('mean_pos'):.1f}", flush=True)
                                break

                            # SUCCESS_NEW but not pure type3 → hybrid if catch evidence + post or residual
                            if res["SUCCESS_NEW"] and m3.get("eccentric") and m3.get("downward_motion") and miles_full["C"]:
                                hybrids.append({
                                    "ic_id": root_id, "tag": f"hybrid_{tag}_{bname}_{vname}",
                                    "expert_type": "TYPE3_TO_TYPE2_HYBRID",
                                    "SUCCESS_NEW": 1,
                                    "win_mean_pos": res["window"].get("mean_pos"),
                                    "win_mean_theta": res["window"].get("mean_theta"),
                                    "hold_both_s": hold,
                                    "pure_type3": int(pure_type3),
                                    "post_pivot_n": pinfo.get("post_pivot_n"),
                                    "post_slide_n": pinfo.get("post_slide_n"),
                                    "m3_reason": m3.get("reason"),
                                    "tokens": json.dumps(ftoks), "branch": bname,
                                })
                                # also persist trajectory
                                res_r = evaluate_full(env, table, book48, blob, ftoks, record_obs=True)
                                mech = classify_mechanisms(res_r["trace"], res_r["cs0"], res_r["d0"], res_r.get("d_handoff"))
                                save_full_expert(
                                    env, root_id, f"hybrid_{tag}_{bname}_{vname}", res_r, mech,
                                    mapping_meta, checksum, sp,
                                    {"expert_type": "TYPE3_TO_TYPE2_HYBRID", "branch": bname, "m3": m3},
                                )
                                root_stats["n_hybrid"] += 1
                                print(f"  HYBRID NEW {root_id} {tag} {bname} {vname}", flush=True)
                                break

                            if miles_full["D"] and not res["SUCCESS_NEW"] and near:
                                # post-manip for hybrid SUCCESS_NEW
                                for piv_name, piv in [
                                    ("p4", [4] * 16 + [0] * 8),
                                    ("p6", [6] * 16 + [0] * 8),
                                    ("p10", [10] * 14 + [12] * 5 + [0] * 8),
                                    ("slide", [17] * 14 + [0] * 8),
                                ]:
                                    htoks = base + list(piv)
                                    n_rollouts += 1
                                    res_h = evaluate_full(env, table, book48, blob, htoks, record_obs=False)
                                    if res_h["SUCCESS_NEW"]:
                                        hybrids.append({
                                            "ic_id": root_id,
                                            "tag": f"hybrid_{tag}_{bname}_{piv_name}",
                                            "expert_type": "TYPE3_TO_TYPE2_HYBRID",
                                            "SUCCESS_NEW": 1,
                                            "win_mean_pos": res_h["window"].get("mean_pos"),
                                            "win_mean_theta": res_h["window"].get("mean_theta"),
                                            "hold_both_s": res_h["window"].get("longest_both_ok_s"),
                                            "tokens": json.dumps(htoks), "branch": bname,
                                        })
                                        res_r = evaluate_full(env, table, book48, blob, htoks, record_obs=True)
                                        mech = classify_mechanisms(
                                            res_r["trace"], res_r["cs0"], res_r["d0"], res_r.get("d_handoff")
                                        )
                                        save_full_expert(
                                            env, root_id, f"hybrid_{tag}_{bname}_{piv_name}", res_r, mech,
                                            mapping_meta, checksum, sp,
                                            {"expert_type": "TYPE3_TO_TYPE2_HYBRID", "branch": bname},
                                        )
                                        root_stats["n_hybrid"] += 1
                                        print(f"  HYBRID+post {root_id} {tag} {piv_name}", flush=True)
                                        break
                                if root_stats["n_hybrid"] > 0:
                                    break

                            failures.append({
                                "ic_id": root_id, "tag": f"{tag}_{bname}_{vname}",
                                "reason": res["reason"], "s12": int(res["success_12s"]),
                                "win_th": res["window"].get("mean_theta"),
                                "win_pos": res["window"].get("mean_pos"),
                                "hold_both_s": hold,
                                "C": int(miles_full["C"]), "D": int(miles_full["D"]),
                                "pure_type3": int(pure_type3),
                                "m3_reason": m3.get("reason"),
                            })
                        if root_stats["n_full"] > 0 or root_stats["n_hybrid"] > 0:
                            break
                if root_stats["n_full"] > 0:
                    break

                if best_local and best_local["score"] < 1 and not any_progress:
                    # mark difficult branch
                    pass

            if not any_progress:
                # candidate's branches made no C/D progress
                beyond.append({
                    "root_ic": root_id, "tag": tag,
                    "reason": "no_bilateral_from_any_branch",
                    "delta_theta": cand.get("delta_theta_deg"),
                    "modes_sample": diag_rows[-1]["failure_modes"] if diag_rows else "",
                })
                root_stats["n_beyond"] += 1
            if root_stats["n_full"] > 0:
                break

        coverage.append(root_stats)
        print(f"  ROOT DONE {root_stats}", flush=True)

    # outputs
    write_csv(OUT / "BRANCH_SNAPSHOTS.csv", branch_rows)
    write_csv(OUT / "CAPTURE_BRAKE_CANDIDATES.csv", candidates_out)
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "HYBRID_EXPERTS.csv", hybrids)
    write_csv(OUT / "FAILURES.csv", failures[:1500])
    write_csv(OUT / "ROOT_EPISODE_COVERAGE.csv", coverage)
    write_csv(OUT / "BEYOND_RECOVERY.csv", beyond)
    write_csv(OUT / "RAW" / "geometry_diagnosis.csv", diag_rows)

    n_C = sum(1 for r in candidates_out if r.get("C") == 1)
    n_D = sum(1 for r in candidates_out if r.get("D") == 1)
    n_B = sum(1 for r in candidates_out if r.get("B") == 1)
    n_full = len({r["ic_id"] for r in full_experts})
    n_hyb = len({r["ic_id"] for r in hybrids})

    # mode summary
    mode_ctr = Counter()
    for d in diag_rows:
        for m in json.loads(d["failure_modes"]):
            mode_ctr[m] += 1

    # helpful actions for B/C
    help_B = Counter()
    for r in candidates_out:
        if r.get("B") == 1:
            for t in json.loads(r["tokens_cont"])[:6]:
                help_B[int(t)] += 1

    answers = {
        "Q1_can_convert_to_bilateral": n_C > 0,
        "Q1_n_C_milestones": n_C,
        "Q1_n_D_stable": n_D,
        "Q2_tokens_helping_C": action_help.most_common(12),
        "Q2_tokens_helping_B": help_B.most_common(12),
        "Q3_failure_mode_counts": dict(mode_ctr.most_common()),
        "Q3_assessment": None,  # filled below
        "Q4_n_full_type3": n_full,
        "Q4_full_ids": sorted({r["ic_id"] for r in full_experts}),
        "Q5_n_hybrid_ics": n_hyb,
        "Q5_hybrid_ids": sorted({r["ic_id"] for r in hybrids}),
        "Q6_n_beyond": len(beyond),
        "n_rollouts": n_rollouts,
        "wall_s": time.time() - t0wall,
        "n_branches": len(branch_rows),
        "n_B_angvel_reduced": n_B,
    }
    if n_C == 0 and n_D == 0:
        answers["Q3_assessment"] = (
            "missing_control_authority_or_irrecoverable_geometry"
            if mode_ctr.get("cylinder_outside_opposing_finger_reach", 0) > len(diag_rows) // 3
            or mode_ctr.get("excessive_relative_angular_velocity", 0) > len(diag_rows) // 3
            else "timing_resolution_or_search_strategy_insufficient"
        )
    elif n_full == 0 and n_D > 0:
        answers["Q3_assessment"] = "capture_possible_but_pose_hold_or_authority_limits_SUCCESS_NEW"
    else:
        answers["Q3_assessment"] = "search_found_progress" if n_full else "partial_progress_only"

    # refine Q3 with evidence
    if n_C == 0:
        top_modes = [m for m, _ in mode_ctr.most_common(3)]
        if "cylinder_outside_opposing_finger_reach" in top_modes or "excessive_relative_angular_velocity" in top_modes:
            answers["Q3_assessment"] = "missing_control_authority_or_irrecoverable_geometry"
        else:
            answers["Q3_assessment"] = "timing_resolution_within_legal_K49_insufficient_to_brake"

    write_json(OUT / "RAW" / "answers.json", answers)

    # diagnosis markdown
    diag_md = f"""# Contact Geometry Diagnosis

## Summary of branch-point failure modes

| Mode | Count |
|------|------:|
"""
    for m, c in mode_ctr.most_common():
        diag_md += f"| {m} | {c} |\n"
    diag_md += f"""

## Coverage

- Roots: {', '.join(r['root_episode'] for r in coverage)}
- Branches diagnosed: {len(diag_rows)}
- Continuations with C (bilateral restored): {n_C}
- Continuations with D (stable bilateral): {n_D}
- Continuations with B (ω reduced): {n_B}

## Interpretation

Branch points were taken from verified eccentric-catch prefixes (first eccentric recontact,
peak |ω|, max useful orientation while contact remains, pre-loss).

Diagnosis uses measured finger–cylinder contacts, relative COM / angular velocity, finger
opening, and free-end distance to the opposing finger — not token names alone.

See `RAW/geometry_diagnosis.csv` for per-branch rows.
"""
    (OUT / "CONTACT_GEOMETRY_DIAGNOSIS.md").write_text(diag_md, encoding="utf-8")

    report = f"""# Type 3 Capture–Brake from Intermediate States

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**Source:** `k49_horizontal_endcatch_recatch` mechanism candidates  
**No SFT/GRPO. No K49 / SUCCESS_NEW changes.**

## Search

- Roots: `hrv1_920031` (supported mech), `hard_910318` (high-Δθ contact-retaining)
- Branch snapshots: {len(branch_rows)}
- Continuations evaluated: {len(candidates_out)}
- Total rollouts: {n_rollouts}
- Wall time: {answers['wall_s']:.1f}s

## Milestones

| Milestone | Count |
|-----------|------:|
| B ω reduced | {n_B} |
| C bilateral restored | {n_C} |
| D stable bilateral | {n_D} |
| FULL Type3 SUCCESS_NEW | {n_full} |
| Type3→Type2 hybrid | {n_hyb} |

## Q1–Q6

1. **Convert intermediate → stable bilateral with legal K49?** **{answers['Q1_can_convert_to_bilateral']}**  
   (C={n_C}, D={n_D})

2. **Helpful wrist/translation/grip tokens (for C):** {answers['Q2_tokens_helping_C']}  
   **For B (ω reduce):** {answers['Q2_tokens_helping_B']}

3. **Remaining failure class:** **{answers['Q3_assessment']}**  
   Mode counts: {answers['Q3_failure_mode_counts']}

4. **Full Type3 SUCCESS_NEW experts:** **{n_full}** {answers['Q4_full_ids']}

5. **Type3→Type2 hybrids:** **{n_hyb}** {answers['Q5_hybrid_ids']}

6. **Beyond recovery (no C from any branch):** {len(beyond)} — see `BEYOND_RECOVERY.csv`

## Artifacts

`BRANCH_SNAPSHOTS.csv`, `CONTACT_GEOMETRY_DIAGNOSIS.md`, `CAPTURE_BRAKE_CANDIDATES.csv`,  
`FULL_EXPERTS.csv`, `HYBRID_EXPERTS.csv`, `FAILURES.csv`, `ROOT_EPISODE_COVERAGE.csv`,  
`DATASET_READINESS.md`, `branch_snapshots/`, `trajectories/`, `raw_contact_traces/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Type3 Capture–Brake

## Verdict: {"YES" if n_full > 0 else "NOT YET for pure Type3"}

- FULL Type3: {n_full}
- Hybrids (Type3 catch → Type2 correction): {n_hyb}
- Bilateral restored from branch (C): {n_C}

### Merge rule
- Merge only `FULL_EXPERTS.csv` as TYPE3_DYNAMIC_RECATCH.
- Hybrids may merge as TYPE2_RECATCH_THEN_MANIPULATION / mixed, not pure Type3.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
