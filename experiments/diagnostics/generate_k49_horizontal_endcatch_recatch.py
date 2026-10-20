"""Horizontal-cylinder end-catch Type-3 Dynamic Recatch expert generation.

Locked K49_tok48_wy0p25 + SUCCESS_NEW. No SFT/GRPO. No physics/mapping changes.
Focus: release → eccentric end intercept → inertial rotation → brake → HANDOFF.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
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
from evaluation.grasp_geometry import measure
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
    PIVOT_TOKS,
    SLIDE_HINT,
    analyze_type3_mechanism,
    contact_axial_offset,
    eval_ABCD,
    find_bilateral_after_loss,
    has_post_manipulation,
    rich_state,
    run_traced,
    type2_proposals,
)
from scripts.generate_k49_in_recovery_recatch import find_first_bilateral, load_short_handoff_ics
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    adaptive_return_variants,
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

OUT = ROOT / "results" / "expert_generation" / "k49_horizontal_endcatch_recatch"
SEED = 20261019

# Prior eccentric near-misses (DROP / no_bilateral) to refine braking around.
NEAR_MISS_SEEDS = {
    "hrv1_920109": [
        [30, 30, 30, 12, 12] + [5] * 10 + [0] * 14,  # t3_24
        [30, 30, 30, 12, 12] + [5] * 10 + [12] * 4 + [0] * 12,
        [30, 30, 30, 12, 12] + [5] * 6 + [0] * 4 + [12] * 6 + [0] * 10,
        [30, 30] + [5] * 8 + [12] * 3 + [0] * 16,
    ],
    "hrv1_920031": [
        [30, 12, 12] + [8] * 10 + [0] * 14,   # t3_7
        [30, 12, 12] + [15] * 10 + [0] * 14,  # t3_10
        [30, 12, 12] + [8] * 6 + [12] * 4 + [0] * 14,
        [30, 12, 12] + [8] * 8 + [0] * 4 + [12] * 5 + [0] * 10,
        [30, 12, 12] + [15] * 6 + [0] * 6 + [12] * 6 + [0] * 10,
    ],
    # Smoke: eccentric+downward+Δθ≈22° but no_bilateral — brake-focused seeds
    "hard_910318": [
        [30] + [0] * 4 + [33] * 6 + [12] * 5 + [0] * 14,
        [30] + [12] * 2 + [14] * 8 + [12] * 4 + [0] * 14,
        [30] + [0] * 4 + [33] * 4 + [12] * 10 + [0] * 16,
        [30] + [0] * 2 + [33] * 8 + [0] * 20,
        [30] + [0] * 4 + [17] * 6 + [12] * 8 + [0] * 14,
        [36] + [0] * 4 + [33] * 6 + [12] * 8 + [0] * 14,
    ],
    "hard_910185": [
        [30] + [0] * 4 + [33] * 6 + [12] * 5 + [0] * 14,
        [30] + [12] * 2 + [8] * 8 + [12] * 6 + [0] * 14,
        [30] + [0] * 3 + [17] * 8 + [12] * 8 + [0] * 12,
    ],
    "hrv1_920117": [
        [30] + [0] * 6 + [33] * 6 + [12] * 8 + [0] * 14,
        [30] + [0] * 8 + [17] * 6 + [12] * 10 + [0] * 12,
        [36] * 2 + [0] * 4 + [33] * 6 + [12] * 8 + [0] * 12,
    ],
}

# Real disturbance ICs with large orientation error + some contact (mined).
PRIORITY_HORIZONTAL = [
    "hrv1_920117",   # ~51°, bilateral, ax~18mm
    "hard_910185",   # ~37°, bilateral, ax~14mm
    "hard_910318",   # ~36°, bilateral, ax~18mm
    "hard_910144",   # ~43°, bilateral, ax~29mm (more eccentric start)
    "hrv1_920031",   # near-miss seed
    "hrv1_920109",   # near-miss seed
]


def snapshot_path(ic_id: str) -> Path | None:
    cands = [
        ROOT / f"results/expert_generation/k49_in_recovery_recatch/snapshots/{ic_id}.pkl",
        ROOT / f"results/expert_generation/k49_dynamic_recatch_experts/snapshots/{ic_id}.pkl",
        ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/{ic_id}.pkl",
        ROOT / f"results/diagnostics/hard_ic_mining/snapshots/{ic_id}.pkl",
        ROOT / f"results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots/{ic_id}.pkl",
    ]
    for p in cands:
        if p.exists():
            return p
    # leftover alias
    alt = ROOT / f"results/benchmarks/hard_recovery_v1/snapshots/hrv1_leftover_{ic_id}.pkl"
    if alt.exists():
        return alt
    return None


def horiz_ic_metrics(sim):
    m = measure(sim, U_TGT, P_TGT)
    d = decompose(sim)
    cs = contact_summary(sim)
    u = np.asarray(m["u_c_h"], float)
    u = u / (np.linalg.norm(u) + 1e-12)
    ut = np.asarray(U_TGT, float)
    ut = ut / (np.linalg.norm(ut) + 1e-12)
    align = abs(float(np.dot(u, ut)))
    horiz_deg = float(np.degrees(np.arccos(np.clip(align, 0.0, 1.0))))
    ax, _ = contact_axial_offset(sim)
    axial_mean = None if ax is None else ax["mean_abs_m"]
    # center grasp: contact near COM or small COM axial error
    center = bool(
        (axial_mean is not None and axial_mean <= 0.55 * CYL_HALF)
        or abs(float(d["e_ax_mm"])) <= 20.0
    )
    return {
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]),
        "e_lat_mm": float(d["e_lat_mm"]),
        "u_c_h": u.tolist(),
        "align_abs": align,
        "horiz_angle_from_target_deg": horiz_deg,
        "bilateral": bool(cs["bilateral"]),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "v_rel_h": np.asarray(m["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(m["w_rel_h"], float).tolist(),
        "contact_axial_mean_m": axial_mean,
        "contact_axial_max_m": None if ax is None else ax["max_abs_m"],
        "center_grasp": center,
        "approx_horizontal": bool(horiz_deg >= 30.0 or m["e_theta_deg"] >= 30.0),
        "obj_z": float(m["obj_z"]),
    }


def mine_additional_horizontal(env, limit=8):
    """Scan disturbance snapshots for additional large-tilt contact ICs."""
    pools = []
    for d in [
        ROOT / "results/benchmarks/hard_recovery_v1/snapshots",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
    ]:
        if d.exists():
            pools.extend(sorted(d.glob("*.pkl")))
    out, seen = [], set()
    scored = []
    for sp in pools:
        ic = sp.stem.replace("hrv1_leftover_", "")
        if ic in seen or ic in PRIORITY_HORIZONTAL:
            continue
        seen.add(ic)
        try:
            blob = pickle.loads(sp.read_bytes())
            restore_capture(env, blob["capture"])
            h = horiz_ic_metrics(env.sim)
        except Exception:
            continue
        if (h["nL"] + h["nR"]) == 0:
            continue
        if h["e_theta_deg"] < 30.0:
            continue
        scored.append((h["e_theta_deg"] - 0.15 * abs(h["e_ax_mm"]) + (5 if h["bilateral"] else 0),
                       ic, str(sp), h))
    scored.sort(key=lambda x: -x[0])
    for sc, ic, path, h in scored[:limit]:
        out.append({"ic_id": ic, "snapshot": path, "metrics": h, "pool": "mined_horizontal", "score": sc})
    return out


def try_legal_horizontalize(env, table, book48, blob, src_id, max_keep=2):
    """Attempt to raise orientation via legal soft-unload / wrist while keeping contact.

    Snapshots only if physics remains contact-supported (no teleport).
    """
    derived = []
    trials = []
    # soft unload + optional wrist
    for soft_n in (15, 25, 40):
        for wrist in ([], [4] * 20, [6] * 20, [10] * 20, [4] * 40):
            trials.append([12] * soft_n + list(wrist) + [0] * 5)
            trials.append([30] + [12] * soft_n + list(wrist) + [0] * 5)
    for piv in (4, 6, 10):
        for n in (30, 50, 70):
            trials.append([piv] * n + [12] * 5 + [0] * 3)

    for it, seq in enumerate(trials):
        restore_capture(env, blob["capture"])
        ok, _ = run_seq(env, table, book48, blob, seq)
        if not ok:
            continue
        h = horiz_ic_metrics(env.sim)
        if not h["approx_horizontal"]:
            continue
        if (h["nL"] + h["nR"]) == 0:
            continue
        if not h["center_grasp"] and abs(h["e_ax_mm"]) > 22:
            continue
        # freeze as new IC via take_capture (legal path from src)
        cap = take_capture(env)
        new_blob = {
            "capture": cap,
            "ref": blob.get("ref"),
            "meta": {
                "derived_from": src_id,
                "legal_prefix": seq,
                "generation": "legal_horizontalize",
            },
        }
        # keep ref if present
        if "ref" not in new_blob or new_blob["ref"] is None:
            new_blob["ref"] = blob.get("ref")
        derived.append({
            "ic_id": f"{src_id}_hz{it}",
            "blob": new_blob,
            "metrics": h,
            "prefix_tokens": seq,
            "src": src_id,
        })
        if len(derived) >= max_keep:
            break
    return derived


def brake_refine_around(seq):
    """Local variants that prioritize sustaining contact then securing."""
    base = list(seq)
    out = [base]
    # earlier secure close
    for k in (4, 6, 8, 10, 12):
        out.append(base[: max(4, len(base) - 10)] + [0] * k + [12] * 3 + [0] * 8)
    # soft brace then secure
    for soft in (3, 5, 8, 12):
        out.append(base[:8] + [12] * soft + [0] * 18)
        out.append(base[:10] + [12] * soft + [0] * 4 + [12] * 4 + [0] * 10)
    # shorten chase, lengthen brake
    if len(base) >= 12:
        out.append(base[:6] + [12] * 8 + [0] * 16)
        out.append(base[:4] + base[4:8] + [12] * 6 + [0] * 20)
    # both-finger settle with no translation
    out.append(base[: max(5, len(base) // 2)] + [0] * 24)
    return out


def endcatch_proposals(book, desire, rng, n=48, seeds=None):
    """Joint search: release, intercept translation, soft brace, brake-close.

    Emphasizes braking / bilateral capture, not max Δθ alone.
    """
    sec = tokens_aligned(book, desire if desire is not None else np.array([0, 0, -1.0]), "secure", 8) or [
        1, 5, 8, 13, 15, 17, 33
    ]
    # both axial-ish directions from codebook top
    axial_a = list(dict.fromkeys(list(sec[:4]) + [17, 33, 1, 5]))
    axial_b = list(dict.fromkeys([8, 15, 13, 19] + list(sec[4:8])))
    seqs = []
    if seeds:
        for s in seeds:
            for v in brake_refine_around(s):
                seqs.append(v)

    opens = (30, 36, 44, 41, 32, 3)
    for o in opens:
        for n_o in (1, 2, 3, 4, 6, 8):
            for chase in axial_a[:5]:
                # fall → chase end → soft brace → secure
                seqs.append([o] * n_o + [chase] * 8 + [12] * 3 + [0] * 16)
                seqs.append([o] * n_o + [chase] * 10 + [0] * 18)
                seqs.append([o] * n_o + [12] * 2 + [chase] * 8 + [12] * 4 + [0] * 14)
                # delayed chase (more fall)
                seqs.append([o] * n_o + [0] * 2 + [chase] * 8 + [12] * 3 + [0] * 14)
                seqs.append([o] * n_o + [0] * 4 + [chase] * 6 + [12] * 5 + [0] * 14)
                # brace then chase (supportive intercept)
                seqs.append([o] * n_o + [12] * 4 + [chase] * 6 + [0] * 16)
                # long fall then intercept
                seqs.append([o] * n_o + [chase] * 4 + [12] * 2 + [chase] * 4 + [12] * 6 + [0] * 12)
            for chase in axial_b[:4]:
                seqs.append([o] * n_o + [chase] * 8 + [12] * 3 + [0] * 16)
                seqs.append([o] * n_o + [chase] * 6 + [12] * 5 + [0] * 14)
                seqs.append([o] * n_o + [0] * 3 + [chase] * 8 + [12] * 8 + [0] * 12)

    # wrist during soft catch (orientation assist without post-regrasp pivot block)
    for o in (30, 36):
        for piv in (4, 6, 10):
            seqs.append([o] * 2 + [piv] * 6 + [12] * 4 + [17] * 6 + [0] * 12)
            seqs.append([o] * 3 + [12] * 2 + [piv] * 8 + [12] * 6 + [0] * 10)
            seqs.append([o] * 2 + [33] * 6 + [piv] * 6 + [12] * 4 + [0] * 12)
            seqs.append([o] * 4 + [piv] * 4 + [8] * 6 + [12] * 8 + [0] * 12)

    # progressive grip: soft → medium → secure
    for o in (30, 36, 44):
        for chase in (17, 33, 5, 8, 15, 1):
            seqs.append([o] * 3 + [chase] * 5 + [12] * 6 + [0] * 4 + [12] * 4 + [0] * 12)
            seqs.append([o] * 2 + [chase] * 8 + [12] * 8 + [0] * 14)
            seqs.append([o] * 5 + [chase] * 6 + [12] * 10 + [0] * 12)

    seen, uniq = set(), []
    for s in seqs:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head = uniq[:20]
    tail = uniq[20:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def analyze_capture_rotation_braking(trace):
    """Phase A–D diagnostics for end-catch candidates."""
    m3 = analyze_type3_mechanism(trace)
    if not trace:
        return {"phases": {}, "m3": m3, "supported_while_rotating": False}

    fc = m3.get("first_contact_i")
    bil = m3.get("bilateral_i")
    if fc is None:
        return {"phases": {"A_falling": True}, "m3": m3, "supported_while_rotating": False,
                "limit_factor": "no_recontact"}

    pre = max(0, fc - 1)
    # Phase A
    phase_a = {
        "v_rel_h": trace[pre]["v_rel_h"],
        "w_rel_h": trace[pre]["w_rel_h"],
        "e_theta_deg": trace[pre]["e_theta_deg"],
        "obj_z": trace[pre]["obj_z"],
        "obj_dz_from_start": float(trace[pre]["obj_z"]) - float(trace[0]["obj_z"]),
    }
    # Phase B
    ax = trace[fc].get("contact_axial")
    phase_b = {
        "first_contact_i": fc,
        "axial_max_mm": None if not ax else 1e3 * ax.get("max_abs_m"),
        "axial_mean_mm": None if not ax else 1e3 * ax.get("mean_abs_m"),
        "near_end": None if not ax else ax.get("near_end"),
        "v_rel_pre": trace[pre]["v_rel_h"],
        "nL": trace[fc]["nL"], "nR": trace[fc]["nR"],
    }
    # Phase C: between first contact and bilateral (or +15)
    end_c = bil if bil is not None else min(len(trace) - 1, fc + 15)
    contact_flags = []
    thetas = []
    for r in trace[fc: end_c + 1]:
        any_c = int(r["nL"]) + int(r["nR"]) > 0
        contact_flags.append(any_c)
        thetas.append(float(r["e_theta_deg"]))
    supported = bool(sum(contact_flags) >= max(2, int(0.5 * len(contact_flags))))
    rot_acc = float(thetas[0] - thetas[-1]) if thetas else 0.0
    phase_c = {
        "n_steps": end_c - fc + 1,
        "contact_fraction": float(np.mean(contact_flags)) if contact_flags else 0.0,
        "supported_while_rotating": supported,
        "theta_start": None if not thetas else thetas[0],
        "theta_end": None if not thetas else thetas[-1],
        "rotation_delta_deg": rot_acc,
        "beneficial": bool(rot_acc >= 3.0),
        "escaping": bool(np.mean(contact_flags) < 0.3) if contact_flags else True,
    }
    # Phase D
    phase_d = {"bilateral_i": bil, "braked": False}
    if bil is not None:
        w_bil = np.asarray(trace[bil]["w_rel_h"], float)
        w_pre = np.asarray(trace[pre]["w_rel_h"], float)
        phase_d.update({
            "theta_at_bilateral": float(trace[bil]["e_theta_deg"]),
            "pos_at_bilateral": float(trace[bil]["e_pos_mm"]),
            "w_norm_pre": float(np.linalg.norm(w_pre)),
            "w_norm_bil": float(np.linalg.norm(w_bil)),
            "angular_speed_reduced": bool(np.linalg.norm(w_bil) < np.linalg.norm(w_pre) - 0.05),
            "braked": True,
        })
        # retention after bil
        after = trace[bil: min(len(trace), bil + 12)]
        phase_d["bilateral_hold_frac"] = float(np.mean([1.0 if r["bilateral"] else 0.0 for r in after])) if after else 0.0

    # limiting factor heuristic
    limit = "ok"
    if not m3.get("downward_motion"):
        limit = "insufficient_fall"
    elif not m3.get("eccentric"):
        limit = "interception_geometry"
    elif not supported:
        limit = "contact_not_sustained"
    elif bil is None:
        limit = "inability_to_brake_stabilize"
    elif not m3.get("beneficial_rotation"):
        limit = "insufficient_rotation"
    elif m3.get("post_regrasp_correction"):
        limit = "required_post_correction"
    else:
        limit = "control_authority_or_final_pose"

    return {
        "phases": {"A": phase_a, "B": phase_b, "C": phase_c, "D": phase_d},
        "m3": m3,
        "supported_while_rotating": supported,
        "limit_factor": limit,
    }


def save_expert(env, ic_id, tag, res, mech, mapping_meta, checksum, snap, extra):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "tag": tag, "family": "type3_horizontal_endcatch",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "recovery_s": res["recovery_elapsed"],
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "snapshot_src": str(snap), "obs_dim": 39, "handoff_id": 47,
        "expert_type": "TYPE3_DYNAMIC_RECATCH",
        **extra,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "extra": extra,
        })
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget-type3", type=int, default=40)
    ap.add_argument("--budget-type2", type=int, default=16)
    ap.add_argument("--n-extra-mine", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t_wall0 = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    mapping_meta = describe_k49_config(CONFIG_ID, book48)
    checksum = mapping_checksum(table)
    mapping_meta["checksum_sha256_16"] = checksum
    write_json(OUT / "ACTION_MAPPING.json", mapping_meta)
    np.save(OUT / f"{CONFIG_ID}_table.npy", table)

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    short_ics = {e["ic_id"]: e for e in load_short_handoff_ics()}

    # ---------- IC assembly ----------
    ics = []
    coverage_rows = []
    for ic_id in PRIORITY_HORIZONTAL:
        sp = snapshot_path(ic_id)
        if sp is None:
            print("MISSING", ic_id, flush=True)
            continue
        blob = pickle.loads(sp.read_bytes())
        restore_capture(env, blob["capture"])
        h = horiz_ic_metrics(env.sim)
        pref = short_ics.get(ic_id, {}).get("prefix") or [30]
        ics.append({
            "ic_id": ic_id, "snapshot": str(sp), "blob": blob,
            "prefix": pref, "metrics": h, "pool": "priority",
            "near_miss_seed": ic_id in NEAR_MISS_SEEDS,
        })

    # mine more
    for e in mine_additional_horizontal(env, limit=args.n_extra_mine):
        blob = pickle.loads(Path(e["snapshot"]).read_bytes())
        pref = short_ics.get(e["ic_id"], {}).get("prefix") or [30]
        ics.append({
            "ic_id": e["ic_id"], "snapshot": e["snapshot"], "blob": blob,
            "prefix": pref, "metrics": e["metrics"], "pool": "mined_horizontal",
            "near_miss_seed": False,
        })

    # try legal horizontalize from a few center mid-tilt sources if pool is weak
    n_horiz = sum(1 for e in ics if e["metrics"].get("approx_horizontal"))
    if n_horiz < 4:
        for src in ("hard_910101", "hrv1_920099", "hard_910220", "hrv1_920031"):
            sp = snapshot_path(src)
            if sp is None:
                continue
            blob = pickle.loads(sp.read_bytes())
            derived = try_legal_horizontalize(env, table, book48, blob, src, max_keep=2)
            for d in derived:
                out_snap = OUT / "snapshots" / f"{d['ic_id']}.pkl"
                out_snap.write_bytes(pickle.dumps(d["blob"]))
                ics.append({
                    "ic_id": d["ic_id"], "snapshot": str(out_snap), "blob": d["blob"],
                    "prefix": [30], "metrics": d["metrics"], "pool": "legal_derived",
                    "near_miss_seed": False, "derived_from": d["src"],
                })
            if sum(1 for e in ics if e["metrics"].get("approx_horizontal")) >= 6:
                break

    if args.smoke:
        ics = ics[:3]

    write_json(OUT / "RAW" / "selected_ics.json", [
        {k: v for k, v in e.items() if k != "blob"} for e in ics
    ])
    print(f"SELECTED n={len(ics)} horiz={sum(1 for e in ics if e['metrics'].get('approx_horizontal'))}",
          flush=True)

    full_experts = []
    candidates = []
    phase_rows = []
    matched = []
    failures = []
    partials = []

    for ie, ic in enumerate(ics):
        t0b = time.time()
        ic_id = ic["ic_id"]
        blob = ic["blob"]
        snap = Path(ic["snapshot"])
        shutil.copy2(snap, OUT / "snapshots" / f"{ic_id}.pkl")
        h0 = ic["metrics"]
        coverage_rows.append({
            "ic_id": ic_id, "pool": ic["pool"],
            "approx_horizontal": int(h0.get("approx_horizontal") or 0),
            "center_grasp": int(h0.get("center_grasp") or 0),
            "e_theta0": h0["e_theta_deg"], "e_pos0": h0["e_pos_mm"],
            "e_ax0": h0["e_ax_mm"], "horiz_deg": h0["horiz_angle_from_target_deg"],
            "bilateral0": int(h0["bilateral"]),
            "near_miss_seed": int(ic.get("near_miss_seed") or 0),
            "u_c_h": json.dumps(h0["u_c_h"]),
            "v_rel_h": json.dumps(h0["v_rel_h"]),
            "w_rel_h": json.dumps(h0["w_rel_h"]),
            "snapshot": str(snap),
            "root_episode": ic.get("derived_from") or ic_id,
        })
        print(f"\n[{ie+1}/{len(ics)}] {ic_id} th={h0['e_theta_deg']:.1f} "
              f"horiz={h0['horiz_angle_from_target_deg']:.1f} ax={h0['e_ax_mm']:.1f} "
              f"bil={h0['bilateral']} pool={ic['pool']}", flush=True)

        ab = eval_ABCD(env, table, book48, blob, ic["prefix"])
        best_c = None
        best_d = None
        n_try = 0

        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]

        seeds = NEAR_MISS_SEEDS.get(ic_id) or NEAR_MISS_SEEDS.get(
            str(ic.get("derived_from") or ""), None
        )
        props3 = endcatch_proposals(book48, desire, rng, n=args.budget_type3, seeds=seeds)
        # adaptive queue: append brake-refines when mechanism near-miss appears
        queue = list(props3)
        seen_seq = {tuple(s) for s in queue}
        max_extra_refine = 24 if not args.smoke else 4
        n_extra_added = 0
        print(f"  TYPE3_ENDCATCH n={len(queue)}", flush=True)

        ip = -1
        while True:
            ip += 1
            if ip >= len(queue):
                break
            if best_d is not None:
                break
            seq = queue[ip]
            n_try += 1
            ok, trace = run_traced(env, table, book48, blob, seq)
            phases = analyze_capture_rotation_braking(trace)
            m3 = phases["m3"]
            cand = {
                "ic_id": ic_id, "tag": f"t3_{ip}", "ok_prefix": int(ok),
                "eccentric": int(bool(m3.get("eccentric"))),
                "downward": int(bool(m3.get("downward_motion"))),
                "beneficial_rot": int(bool(m3.get("beneficial_rotation"))),
                "supported": int(bool(phases.get("supported_while_rotating"))),
                "bilateral": int(m3.get("bilateral_i") is not None),
                "axial_max_mm": m3.get("axial_max_mm"),
                "delta_theta_deg": m3.get("delta_theta_deg"),
                "theta_pre": m3.get("theta_pre_deg"),
                "theta_bil": m3.get("theta_at_bil_deg"),
                "v_down": m3.get("v_down_hand_z"),
                "obj_dz": m3.get("obj_dz_m"),
                "m3_reason": m3.get("reason"),
                "limit_factor": phases.get("limit_factor"),
                "n_tokens": len(seq),
            }
            candidates.append(cand)
            phase_rows.append({
                "ic_id": ic_id, "tag": f"t3_{ip}",
                "A_v_rel": json.dumps(phases["phases"].get("A", {}).get("v_rel_h")),
                "A_dz": phases["phases"].get("A", {}).get("obj_dz_from_start"),
                "B_axial_max_mm": phases["phases"].get("B", {}).get("axial_max_mm"),
                "B_near_end": phases["phases"].get("B", {}).get("near_end"),
                "C_supported": phases["phases"].get("C", {}).get("supported_while_rotating"),
                "C_contact_frac": phases["phases"].get("C", {}).get("contact_fraction"),
                "C_rot_delta": phases["phases"].get("C", {}).get("rotation_delta_deg"),
                "D_bilateral_i": phases["phases"].get("D", {}).get("bilateral_i"),
                "D_braked": phases["phases"].get("D", {}).get("braked"),
                "D_w_pre": phases["phases"].get("D", {}).get("w_norm_pre"),
                "D_w_bil": phases["phases"].get("D", {}).get("w_norm_bil"),
                "D_theta_bil": phases["phases"].get("D", {}).get("theta_at_bilateral"),
                "limit_factor": phases.get("limit_factor"),
                "is_type3_mech": int(bool(m3.get("is_type3"))),
            })

            # local brake refine around promising no_bilateral / drop-after-mech
            if (
                m3.get("eccentric") and m3.get("downward_motion") and m3.get("beneficial_rotation")
                and m3.get("bilateral_i") is None and n_extra_added < max_extra_refine
            ):
                for v in brake_refine_around(seq):
                    k = tuple(v)
                    if k in seen_seq:
                        continue
                    seen_seq.add(k)
                    queue.append(v)
                    n_extra_added += 1
                    if n_extra_added >= max_extra_refine:
                        break
                print(f"    refine+{n_extra_added} after t3_{ip} dth={m3.get('delta_theta_deg'):.1f}",
                      flush=True)

            if not ok:
                failures.append({"ic_id": ic_id, "tag": f"t3_{ip}", "family": "type3",
                                 "reason": "DROP", "limit": phases.get("limit_factor")})
                # still keep partial mechanism if eccentric+rot
                if m3.get("eccentric") and m3.get("beneficial_rotation") and m3.get("downward_motion"):
                    partials.append({**cand, "status": "PARTIAL_DROP_AFTER_MECH"})
                continue

            variants = [("raw", list(seq) + [0, 0, 0])]
            restore_capture(env, blob["capture"])
            run_seq(env, table, book48, blob, seq)
            herr = hand_ref_error_deg(env, blob)
            if herr > 8:
                for n48 in (8, 12, 16):
                    variants.append((f"ret48x{n48}", list(seq) + [SECURE_REVERSE_WRIST] * n48 + [0, 0, 0]))

            for vname, tokens in variants:
                n_try += 1
                ok2, tr2 = run_traced(env, table, book48, blob, tokens)
                ph2 = analyze_capture_rotation_braking(tr2)
                m32 = ph2["m3"]
                res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)

                mech_ok = bool(
                    m32.get("eccentric") and m32.get("downward_motion")
                    and m32.get("beneficial_rotation") and m32.get("bilateral_i") is not None
                    and (not m32.get("post_regrasp_correction"))
                    and ph2.get("supported_while_rotating")
                )

                if res["SUCCESS_NEW"] and mech_ok:
                    res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
                    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                    extra = {
                        "type3_mech": m32, "phases": ph2["phases"],
                        "root_episode": ic.get("derived_from") or ic_id,
                        "horizontal_ic_metrics0": h0,
                    }
                    save_expert(env, ic_id, f"type3_{ip}_{vname}", res, mech,
                                mapping_meta, checksum, snap, extra)
                    write_json(OUT / "raw_contact_traces" / f"{ic_id}__type3_{ip}_{vname}_phases.json", ph2)
                    row = {
                        "ic_id": ic_id, "tag": f"type3_{ip}_{vname}",
                        "expert_type": "TYPE3_DYNAMIC_RECATCH", "SUCCESS_NEW": 1,
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "theta0": h0["e_theta_deg"],
                        "theta_pre_impact": m32.get("theta_pre_deg"),
                        "theta_at_bilateral": m32.get("theta_at_bil_deg"),
                        "delta_theta_deg": m32.get("delta_theta_deg"),
                        "axial_max_mm": m32.get("axial_max_mm"),
                        "v_down_hand_z": m32.get("v_down_hand_z"),
                        "obj_dz_m": m32.get("obj_dz_m"),
                        "supported": 1,
                        "post_corr": 0,
                        "n_tokens": len(tokens),
                        "config_id": CONFIG_ID, "mapping_checksum": checksum,
                        "tokens": json.dumps(tokens), "reproducible": 1,
                    }
                    full_experts.append(row)
                    best_d = row
                    print(f"  TYPE3 FULL ax={m32.get('axial_max_mm'):.1f} "
                          f"dth={m32.get('delta_theta_deg'):.1f} "
                          f"win_th={row['win_mean_theta']:.1f} win_pos={row['win_mean_pos']:.1f}",
                          flush=True)
                    break

                # partial successes
                status = None
                if mech_ok and not res["SUCCESS_NEW"]:
                    status = "POSE_OR_HOLD_FAIL"
                elif (m32.get("eccentric") and m32.get("beneficial_rotation")
                      and m32.get("downward_motion") and m32.get("bilateral_i") is None):
                    status = "NO_BILATERAL"
                elif (m32.get("eccentric") and m32.get("downward_motion")
                      and m32.get("beneficial_rotation") and m32.get("post_regrasp_correction")):
                    status = "NEEDS_POST_CORR"
                if status:
                    partials.append({
                        **{k: cand.get(k) for k in cand},
                        "tag": f"t3_{ip}_{vname}", "status": status,
                        "s12": int(res["success_12s"]),
                        "win_pos": res["window"].get("mean_pos"),
                        "win_theta": res["window"].get("mean_theta"),
                        "reason": res["reason"],
                    })
                failures.append({
                    "ic_id": ic_id, "tag": f"t3_{ip}_{vname}", "family": "type3",
                    "reason": res["reason"], "s12": int(res["success_12s"]),
                    "win_pos": res["window"].get("mean_pos"),
                    "win_theta": res["window"].get("mean_theta"),
                    "m3_reason": m32.get("reason"),
                    "limit": ph2.get("limit_factor"),
                    "axial_max_mm": m32.get("axial_max_mm"),
                    "delta_theta": m32.get("delta_theta_deg"),
                    "supported": int(bool(ph2.get("supported_while_rotating"))),
                })
            if best_d is not None:
                break

        # Type2 matched baseline (bounded)
        print("  TYPE2_BASELINE", flush=True)
        props2 = type2_proposals(book48, desire, rng, n=args.budget_type2, open_prefix=ic["prefix"])
        for ip, seq in enumerate(props2):
            n_try += 1
            ok, trace = run_traced(env, table, book48, blob, seq)
            if not ok:
                failures.append({"ic_id": ic_id, "tag": f"t2_{ip}", "family": "type2", "reason": "DROP"})
                continue
            bil_i = find_bilateral_after_loss(trace)
            post_ok, post_info = has_post_manipulation(trace, bil_i)
            restore_capture(env, blob["capture"])
            t0_ic = float(env.sim.data.time)
            variants = [("raw", list(seq) + [0, 0, 0])]
            variants += adaptive_return_variants(env, table, book48, blob, t0_ic, seq)
            for vname, tokens in variants:
                n_try += 1
                res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
                if not res["SUCCESS_NEW"]:
                    failures.append({
                        "ic_id": ic_id, "tag": f"t2_{ip}_{vname}", "family": "type2",
                        "reason": res["reason"], "s12": int(res["success_12s"]),
                        "win_pos": res["window"].get("mean_pos"),
                        "win_theta": res["window"].get("mean_theta"),
                    })
                    if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                        break
                    continue
                ok2, tr2 = run_traced(env, table, book48, blob, tokens)
                bil2 = find_bilateral_after_loss(tr2)
                post2, pinfo = has_post_manipulation(tr2, bil2)
                if not (bil2 is not None and post2):
                    continue
                th_b = float(tr2[bil2]["e_theta_deg"])
                th_e = float(tr2[-1]["e_theta_deg"])
                pos_b = float(tr2[bil2]["e_pos_mm"])
                pos_e = float(tr2[-1]["e_pos_mm"])
                if not (th_e < th_b - 2.0 or pos_e < pos_b - 1.5):
                    continue
                best_c = {
                    "ic_id": ic_id, "tag": f"type2_{ip}_{vname}",
                    "SUCCESS_NEW": 1,
                    "win_mean_pos": res["window"].get("mean_pos"),
                    "win_mean_theta": res["window"].get("mean_theta"),
                    "theta_after_recatch": th_b, "theta_after_manip": th_e,
                    "pos_after_recatch": pos_b, "pos_after_manip": pos_e,
                    "post_pivot_n": pinfo.get("post_pivot_n"),
                    "post_slide_n": pinfo.get("post_slide_n"),
                    "n_tokens": len(tokens),
                    "tokens": json.dumps(tokens),
                }
                print(f"  TYPE2 NEW th {th_b:.1f}->{th_e:.1f} pos {pos_b:.1f}->{pos_e:.1f}", flush=True)
                break
            if best_c is not None:
                break

        matched.append({
            "ic_id": ic_id,
            "e_theta0": h0["e_theta_deg"], "e_pos0": h0["e_pos_mm"],
            "horiz_deg": h0["horiz_angle_from_target_deg"],
            "center_grasp": int(h0.get("center_grasp") or 0),
            "A_NEW": ab["A_imm"]["SUCCESS_NEW"], "A_s12": ab["A_imm"]["s12"],
            "A_win_th": ab["A_imm"]["win_theta"], "A_win_pos": ab["A_imm"]["win_pos"],
            "B_NEW": ab["B_simple_release_close"]["SUCCESS_NEW"],
            "B_s12": ab["B_simple_release_close"]["s12"],
            "B_win_th": ab["B_simple_release_close"]["win_theta"],
            "B_win_pos": ab["B_simple_release_close"]["win_pos"],
            "C_TYPE2_NEW": int(best_c is not None),
            "C_win_th": None if not best_c else best_c["win_mean_theta"],
            "C_win_pos": None if not best_c else best_c["win_mean_pos"],
            "D_TYPE3_NEW": int(best_d is not None),
            "D_win_th": None if not best_d else best_d["win_mean_theta"],
            "D_win_pos": None if not best_d else best_d["win_mean_pos"],
            "best": (
                "D_TYPE3" if best_d else ("C_TYPE2" if best_c else (
                    "B" if ab["B_simple_release_close"]["SUCCESS_NEW"] else (
                        "A" if ab["A_imm"]["SUCCESS_NEW"] else "NONE")))
            ),
            "searched": n_try, "budget_s": time.time() - t0b,
        })
        print(f"  DONE t3={best_d is not None} t2={best_c is not None} "
              f"tries={n_try} ({time.time()-t0b:.1f}s)", flush=True)

    # ---------- outputs ----------
    write_csv(OUT / "HORIZONTAL_IC_COVERAGE.csv", coverage_rows)
    write_csv(OUT / "ENDCATCH_CANDIDATES.csv", candidates)
    write_csv(OUT / "CAPTURE_ROTATION_BRAKING.csv", phase_rows)
    write_csv(OUT / "MATCHED_BASELINES.csv", matched)
    write_csv(OUT / "TYPE3_FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "FAILURES.csv", failures[:1200])
    write_csv(OUT / "PARTIAL_MECHANISM.csv", partials)
    write_json(OUT / "RAW" / "type2_matched.json", [
        m for m in matched if m.get("C_TYPE2_NEW")
    ])

    n3 = len({r["ic_id"] for r in full_experts})
    n_endcatch = sum(1 for c in candidates if c.get("eccentric") and c.get("downward") and c.get("beneficial_rot"))
    n_supported = sum(1 for c in candidates if c.get("eccentric") and c.get("supported") and c.get("beneficial_rot"))
    n_bil = sum(1 for c in candidates if c.get("eccentric") and c.get("bilateral") and c.get("beneficial_rot"))

    # limiting factor mode among horizontal ICs' failures
    from collections import Counter
    lim = Counter(f.get("limit") or f.get("m3_reason") for f in failures if f.get("family") == "type3")
    d_wins = sum(1 for m in matched if m["best"] == "D_TYPE3")
    c_wins = sum(1 for m in matched if m["best"] == "C_TYPE2")

    # orientation correction stats from best eccentric+beneficial candidates
    mech_cands = [c for c in candidates if c.get("eccentric") and c.get("downward") and c.get("beneficial_rot")]
    dth_vals = [float(c["delta_theta_deg"]) for c in mech_cands if c.get("delta_theta_deg") is not None]

    answers = {
        "Q1_horizontal_endcatch_possible": n_endcatch > 0,
        "Q1_n_eccentric_beneficial_downward": n_endcatch,
        "Q2_orientation_correction_deg": {
            "n": len(dth_vals),
            "max": None if not dth_vals else float(max(dth_vals)),
            "mean": None if not dth_vals else float(np.mean(dth_vals)),
            "p50": None if not dth_vals else float(np.median(dth_vals)),
        },
        "Q3_supported_while_rotating": n_supported > 0,
        "Q3_n_supported_eccentric_beneficial": n_supported,
        "Q4_brake_to_bilateral": n_bil > 0,
        "Q4_n_bilateral_after_eccentric": n_bil,
        "Q5_full_SUCCESS_NEW_type3": n3 > 0,
        "Q5_n_type3_ics": n3,
        "Q5_type3_ids": sorted({r["ic_id"] for r in full_experts}),
        "Q6_limiting_factor_counts": dict(lim.most_common(12)),
        "Q6_primary_limit": lim.most_common(1)[0][0] if lim else None,
        "Q7_matched_D_wins": d_wins,
        "Q7_matched_C_wins": c_wins,
        "Q7_n_matched": len(matched),
        "n_ics": len(ics),
        "n_approx_horizontal": sum(1 for e in ics if e["metrics"].get("approx_horizontal")),
        "wall_s": time.time() - t_wall0,
    }
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Horizontal-Cylinder End-Catch Dynamic Recatch

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW:** locked 10° / 12 mm / ≥0.5 s hold  
**No SFT/GRPO. No K49 / SUCCESS_NEW changes.**

## IC coverage

- Selected ICs: **{len(ics)}** (approx-horizontal: {answers['n_approx_horizontal']})
- See `HORIZONTAL_IC_COVERAGE.csv` for axis, grasp axial offset, velocities.

## Yield

| Outcome | Count |
|---------|------:|
| VERIFIED_FULL Type3 experts (ICs) | **{n3}** |
| Eccentric∧downward∧beneficial candidates | {n_endcatch} |
| …and supported while rotating | {n_supported} |
| …and bilateral after catch | {n_bil} |

## Q1–Q7

1. **Real horizontal end-catch under K49?** **{answers['Q1_horizontal_endcatch_possible']}**  
   (eccentric∧downward∧beneficial: {n_endcatch})

2. **Orientation correction from impact+constrained rotation:**  
   n={answers['Q2_orientation_correction_deg']['n']},  
   max={answers['Q2_orientation_correction_deg']['max']},  
   mean={answers['Q2_orientation_correction_deg']['mean']},  
   p50={answers['Q2_orientation_correction_deg']['p50']} (deg)

3. **Supported while rotating?** **{answers['Q3_supported_while_rotating']}** ({n_supported} candidates)

4. **Brake → stable bilateral?** **{answers['Q4_brake_to_bilateral']}** ({n_bil} candidates)

5. **Any FULL SUCCESS_NEW Type3 (no post Slide/Pivot)?** **{answers['Q5_full_SUCCESS_NEW_type3']}**  
   ICs: {answers['Q5_type3_ids']}

6. **Primary limiting factor:** **{answers['Q6_primary_limit']}**  
   Counts: {answers['Q6_limiting_factor_counts']}

7. **Matched Type3 vs Type2:** D wins {d_wins}/{len(matched)}; C wins {c_wins}/{len(matched)}  
   (`MATCHED_BASELINES.csv`)

## Artifacts

`HORIZONTAL_IC_COVERAGE.csv`, `ENDCATCH_CANDIDATES.csv`, `CAPTURE_ROTATION_BRAKING.csv`,  
`MATCHED_BASELINES.csv`, `TYPE3_FULL_EXPERTS.csv`, `FAILURES.csv`, `PARTIAL_MECHANISM.csv`,  
`DATASET_READINESS.md`, `trajectories/`, `snapshots/`, `raw_contact_traces/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Horizontal End-Catch Type3

## Verdict: {"YES — merge Type3 rows" if n3 > 0 else "NOT YET"}

- FULL Type3 ICs: **{n3}**
- Mechanism near-misses (eccentric+beneficial+downward): {n_endcatch}
- Supported-while-rotating near-misses: {n_supported}
- Bilateral-after-eccentric near-misses: {n_bil}
- Primary limit: {answers['Q6_primary_limit']}

### Merge rule
Only `TYPE3_FULL_EXPERTS.csv` rows with SUCCESS_NEW=1 and verified phases may enter the unified expert pool.
Do not promote PARTIAL_MECHANISM rows.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
