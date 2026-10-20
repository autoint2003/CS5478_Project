"""Generate Type-2 (Recatch+Manipulation) and Type-3 (Dynamic eccentric Recatch) experts.

Locked K49_tok48_wy0p25 + SUCCESS_NEW. No SFT/GRPO.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.ballistic_scene import CYL_H
from envs.simulation.capture import restore_capture
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

from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    adaptive_return_variants,
    classify_mechanisms,
    decode,
    decompose,
    evaluate_full,
    ic_snapshot_state,
    mapping_checksum,
    run_seq,
    tokens_aligned,
    write_csv,
    write_json,
    jsafe,
)
from scripts.generate_k49_in_recovery_recatch import (
    find_first_bilateral,
    load_short_handoff_ics,
    pose_correction_proposals,
    regrasp_proposals,
    snapshot_for,
)

OUT = ROOT / "results" / "expert_generation" / "k49_dynamic_recatch_experts"
SEED = 20261019
CYL_HALF = 0.5 * float(CYL_H)  # 0.03 m
HARD_ICS = ["hrv1_920099", "hrv1_920131", "hrv1_920116", "hard_910101", "hard_910220"]
PIVOT_TOKS = {4, 6, 10, 32}
SLIDE_HINT = {17, 33, 13, 1, 5, 8, 15, 16, 19, 24}


def rich_state(sim):
    d = decompose(sim)
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    return {
        **{k: d[k] for k in ("e_pos_mm", "e_ax_mm", "e_lat_mm", "e_theta_deg", "err_h", "p_c_h")},
        "bilateral": cs["bilateral"], "nL": cs["nL"], "nR": cs["nR"],
        "v_rel_h": m["v_rel_h"], "w_rel_h": m["w_rel_h"],
        "obj_z": m["obj_z"],
        "u_c_h": m["u_c_h"],
    }


def finger_object_contacts(sim):
    """Return list of contact pos (world) and force-ish normals for finger–object."""
    obj = int(sim.ids.object_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    out = []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        side = "L" if other == left else "R"
        out.append({"pos": np.asarray(c.pos, float).copy(), "side": side, "dist": float(c.dist)})
    return out


def contact_axial_offset(sim):
    """Mean |axial| contact offset from cylinder COM in cylinder frame (m)."""
    cons = finger_object_contacts(sim)
    if not cons:
        return None, []
    Ro = object_R(sim)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    u = cylinder_axis_world(sim)
    details = []
    axials = []
    for c in cons:
        r = c["pos"] - po
        ax = float(np.dot(r, u))
        axials.append(ax)
        details.append({"side": c["side"], "axial_m": ax, "axial_abs_m": abs(ax),
                        "pos": c["pos"].tolist()})
    mean_abs = float(np.mean([abs(a) for a in axials]))
    max_abs = float(np.max([abs(a) for a in axials]))
    return {"mean_abs_m": mean_abs, "max_abs_m": max_abs, "axials_m": axials,
            "near_end": bool(max_abs >= 0.40 * CYL_HALF), "details": details}, details


def run_traced(env, table, book48, blob, tokens):
    """Execute tokens collecting rich per-step diagnostics."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    trace = []
    lost_low = 0
    for tok in tokens:
        st = rich_state(env.sim)
        ax, _ = contact_axial_offset(env.sim)
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
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
        })
        action = decode(table, book48, tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        from envs.outcomes.horizon_outcome import recovery_failed
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return False, trace
    return True, trace


def analyze_type3_mechanism(trace, prefix_len=0):
    """Detect eccentric catch + beneficial rotation during recapture (not post-correct)."""
    if not trace:
        return {"is_type3": False, "reason": "empty"}
    # find first recontact after a none/uni gap
    saw_loss = False
    first_contact_i = None
    for i, r in enumerate(trace):
        any_c = int(r["nL"]) + int(r["nR"]) > 0
        if not any_c or (any_c and not r["bilateral"]):
            if i >= max(0, prefix_len - 1):
                saw_loss = True
        if saw_loss and any_c and first_contact_i is None:
            first_contact_i = i
            break
    if first_contact_i is None:
        # alternate: first contact after open tokens
        for i, r in enumerate(trace):
            if int(r["tok"]) in (30, 36, 39, 44, 41, 32, 3) or True:
                pass
        for i in range(1, len(trace)):
            prev_any = int(trace[i - 1]["nL"]) + int(trace[i - 1]["nR"]) > 0
            cur_any = int(trace[i]["nL"]) + int(trace[i]["nR"]) > 0
            if (not prev_any) and cur_any:
                first_contact_i = i
                saw_loss = True
                break
    if first_contact_i is None:
        return {"is_type3": False, "reason": "no_recontact_after_loss", "saw_loss": saw_loss}

    bil_i = find_first_bilateral(trace, start_idx=first_contact_i)
    # pre-impact relative velocity (step before first contact)
    pre_i = max(0, first_contact_i - 1)
    v_pre = np.asarray(trace[pre_i]["v_rel_h"], float)
    w_pre = np.asarray(trace[pre_i]["w_rel_h"], float)
    # downward relative: object falling vs hand => v_rel_h z negative if hand frame z up-ish;
    # use world-ish: obj_z decreasing and |v_rel|
    v_down = float(-v_pre[2])  # positive if object moving down in hand z
    # also check obj_z drop during release
    z0 = float(trace[0]["obj_z"])
    z_pre = float(trace[pre_i]["obj_z"])
    dz = z_pre - z0

    ax_info = trace[first_contact_i].get("contact_axial")
    eccentric = bool(ax_info and ax_info.get("near_end"))
    axial_max = None if not ax_info else ax_info.get("max_abs_m")

    th_pre = float(trace[pre_i]["e_theta_deg"])
    # orientation during recapture window
    end_i = bil_i if bil_i is not None else min(len(trace) - 1, first_contact_i + 15)
    th_post = float(trace[end_i]["e_theta_deg"])
    d_theta = th_pre - th_post  # positive = improvement
    w_post = np.asarray(trace[end_i]["w_rel_h"], float)
    dw = float(np.linalg.norm(w_post) - np.linalg.norm(w_pre))

    # post-regrasp correction: long pivot/slide AFTER bilateral
    post_corr = False
    post_pivot_n = post_slide_n = 0
    if bil_i is not None:
        for r in trace[bil_i + 1:]:
            if int(r["tok"]) in PIVOT_TOKS:
                post_pivot_n += 1
            if int(r["tok"]) in SLIDE_HINT or int(r["tok"]) == 12:
                # mid unload alone doesn't count; count secure slide hints
                if int(r["tok"]) in SLIDE_HINT:
                    post_slide_n += 1
        if post_pivot_n >= 8 or post_slide_n >= 8:
            post_corr = True

    downward = bool(dz < -0.002 or v_down > 0.05 or float(np.linalg.norm(v_pre)) > 0.08)
    beneficial_rot = bool(d_theta >= 3.0)
    is_type3 = bool(
        saw_loss and eccentric and downward and beneficial_rot and (bil_i is not None) and (not post_corr)
    )
    return {
        "is_type3": is_type3,
        "saw_loss": saw_loss,
        "first_contact_i": first_contact_i,
        "bilateral_i": bil_i,
        "eccentric": eccentric,
        "axial_max_m": axial_max,
        "axial_max_mm": None if axial_max is None else 1e3 * axial_max,
        "cyl_half_m": CYL_HALF,
        "v_rel_pre": v_pre.tolist(),
        "v_down_hand_z": v_down,
        "obj_dz_m": dz,
        "downward_motion": downward,
        "theta_pre_deg": th_pre,
        "theta_at_bil_deg": th_post,
        "delta_theta_deg": d_theta,
        "beneficial_rotation": beneficial_rot,
        "w_pre": w_pre.tolist(),
        "w_post": w_post.tolist(),
        "dw_norm": dw,
        "post_regrasp_correction": post_corr,
        "post_pivot_n": post_pivot_n,
        "post_slide_n": post_slide_n,
        "contact_axial_at_impact": ax_info,
        "reason": (
            "ok" if is_type3 else
            ("not_eccentric" if not eccentric else
             "no_downward" if not downward else
             "no_beneficial_rot" if not beneficial_rot else
             "post_correction" if post_corr else
             "no_bilateral" if bil_i is None else "other")
        ),
    }


def has_post_manipulation(trace, bil_i):
    if bil_i is None:
        return False, {}
    piv = sum(1 for r in trace[bil_i + 1:] if int(r["tok"]) in PIVOT_TOKS)
    sld = sum(1 for r in trace[bil_i + 1:] if int(r["tok"]) in SLIDE_HINT)
    return (piv >= 8 or sld >= 8), {"post_pivot_n": piv, "post_slide_n": sld}


def find_bilateral_after_loss(trace, min_loss_run=1):
    """First sustained bilateral AFTER a contact-loss interval."""
    loss_run = 0
    saw_loss = False
    for i, r in enumerate(trace):
        any_c = int(r["nL"]) + int(r["nR"]) > 0
        lost = (not any_c) or (any_c and not r["bilateral"])
        if lost:
            loss_run += 1
            if loss_run >= min_loss_run:
                saw_loss = True
        else:
            loss_run = 0
        if saw_loss and r["bilateral"]:
            return find_first_bilateral(trace, start_idx=i)
    return None


def classify_existing_row(family, observed, sliding_ver, recatch_sub, n_tokens, tokens=None):
    obs = observed
    if isinstance(obs, str):
        try:
            obs = json.loads(obs)
        except Exception:
            obs = [obs]
    obs = set(obs or [])
    toks = tokens or []
    if isinstance(toks, str):
        try:
            toks = json.loads(toks)
        except Exception:
            toks = []
    # simple release-close: short, open then zeros
    simple = False
    if len(toks) <= 12 and toks:
        opens = sum(1 for t in toks if int(t) in (30, 36, 39, 44, 41, 32, 3))
        zeros = sum(1 for t in toks if int(t) == 0)
        simple = opens <= 3 and zeros >= len(toks) // 2 and len(toks) <= 12

    if simple and ("RECATCH" in str(obs) or family in ("in_recovery_recatch", "recatch")):
        return "SIMPLE_RECATCH"
    if sliding_ver or ("SLIDING" in obs or "PIVOTING" in obs):
        if any("RECATCH" in x for x in obs):
            return "TYPE2_RECATCH_THEN_MANIPULATION"
        return "TYPE1_CONTACT_PRESERVING"
    if any("RECATCH" in x for x in obs):
        return "SIMPLE_RECATCH"
    if family in ("sliding", "consolidated") and ("SLIDING" in obs or "PIVOTING" in obs or "RETURN" in obs):
        return "TYPE1_CONTACT_PRESERVING"
    return "MIXED_OR_UNCERTAIN"


def type3_proposals(book, desire, rng, n=36):
    """Release → fall → intercept (translate) → delayed close. No long post-pivot."""
    sec = tokens_aligned(book, desire if desire is not None else np.array([0, 0, -1.0]), "secure", 6) or [
        1, 5, 13, 17, 33
    ]
    down = tokens_aligned(book, np.array([0.0, 0.0, -1.0]), "secure", 4) or [8, 15, 33]
    seqs = []
    # Priority seeds: short open + axial chase + close (stay recoverable)
    for o in (30, 36, 44, 41, 32):
        for n_o in (1, 2, 3, 5):
            for t in (33, 17, 8, 15, 1, 5):
                seqs.append([o] * n_o + [t] * 8 + [0] * 16)
                seqs.append([o] * n_o + [12] * 2 + [t] * 10 + [0] * 14)
                seqs.append([o] * n_o + [t] * 6 + [12] * 3 + [0] * 12)
    # CPP-like wrist during soft catch (no separate post phase — wrist during open/close)
    for o in (30, 36):
        seqs.append([o] * 3 + [10] * 8 + [12] * 10 + [0] * 8)
        seqs.append([o] * 2 + [32] + [10] * 8 + [12] * 12 + [0] * 5)
        seqs.append([4] * 8 + [o] * 3 + [33] * 8 + [0] * 14)
        seqs.append([6] * 8 + [o] * 3 + [17] * 8 + [0] * 14)
    # Longer fall (higher risk)
    for o in (30, 36, 44):
        for n_o in (6, 8, 10):
            for t in list(dict.fromkeys(list(down[:2]) + list(sec[:2]))):
                seqs.append([o] * n_o + [12] * 2 + [t] * 6 + [0] * 16)
                seqs.append([o] * n_o + [t] * 4 + [0] * 20)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    head, tail = uniq[:15], uniq[15:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def type2_proposals(book, desire, rng, n=36, open_prefix=None):
    """Recatch seed + substantial pivot/slide after contact restore."""
    pref = list(open_prefix) if open_prefix else [30]
    heavy = [
        [4] * 24 + [12] * 5 + [0] * 3,
        [4] * 30 + [12] * 5 + [0] * 3,
        [6] * 24 + [12] * 5 + [0] * 3,
        [10] * 20 + [12] * 5 + [0] * 3,
        [4] * 16 + [6] * 16 + [12] * 5 + [0] * 3,
        [4] * 20 + [12] * 5 + [17] * 16 + [0] * 5,
        [4] * 22 + [12] * 5 + [33] * 20 + [0] * 5,
        [32] + [10] * 8 + [12] * 15 + [0] * 5,
        [6] * 20 + [12] * 15 + [0] * 8,
    ]
    seqs = []
    # Priority: pivot/CPP BEFORE release (works on hard_910220), then release-close
    for piv in heavy[:6]:
        seqs.append(list(piv) + list(pref) + [0] * 8)
        seqs.append(list(piv) + list(pref) + [0] * 12)
    # Priority: short close then manipulation
    for n0 in (3, 5, 8, 12):
        for piv in heavy:
            seqs.append(list(pref) + [0] * n0 + list(piv))
    # mid unload + secure slide after close
    for t in (17, 33, 13, 1, 5):
        seqs.append(list(pref) + [0] * 5 + [12] * 5 + [t] * 16 + [0] * 5)
        seqs.append(list(pref) + [0] * 5 + [4] * 18 + [12] * 5 + [t] * 12 + [0] * 5)
    # F_recatch + CPP
    seqs.append(list(pref) + [0] * 8 + [4] * 24 + [12] * 5 + [0] * 3)
    seqs.append([36] * 5 + [0] * 8 + [4] * 24 + [12] * 5 + [0] * 3)
    seqs.append([4] * 12 + [36] * 5 + [0] * 8 + [4] * 20 + [12] * 5 + [0] * 3)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    # keep priority head
    return uniq[:n]


def save_expert(env, ic_id, tag, family, res, mech, mapping_meta, checksum, snap, extra):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "tag": tag, "family": family,
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "hold_margin_mm": res.get("hold_margin_mm"),
        "recovery_s": res["recovery_elapsed"],
        "e_pos_at_handoff": res.get("e_pos_at_handoff"),
        "e_theta_at_handoff": res.get("e_theta_at_handoff"),
        "observed_recovery": mech.get("observed_recovery"),
        "recatch_subtype": mech.get("recatch_subtype"),
        "sliding_verified": mech.get("sliding_verified"),
        "mechanism_evidence": mech.get("evidence"),
        "snapshot_src": str(snap), "obs_dim": 39, "handoff_id": 47,
        **extra,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "extra": extra,
        })
    return meta


def eval_ABCD(env, table, book48, blob, prefix):
    """Matched comparison A/B/C placeholders filled by caller for C/D."""
    rows = {}
    # A immediate handoff
    res = evaluate_full(env, table, book48, blob, [], record_obs=False)
    rows["A_imm"] = {"SUCCESS_NEW": int(res["SUCCESS_NEW"]), "s12": int(res["success_12s"]),
                     "reason": res["reason"], "win_pos": res["window"].get("mean_pos"),
                     "win_theta": res["window"].get("mean_theta"),
                     "posH": res.get("e_pos_at_handoff"), "thH": res.get("e_theta_at_handoff"),
                     "n_tok": 0}
    # B short release-close (historical style)
    simple = list(prefix) + [0] * 8
    res = evaluate_full(env, table, book48, blob, simple, record_obs=False)
    rows["B_simple_release_close"] = {
        "SUCCESS_NEW": int(res["SUCCESS_NEW"]), "s12": int(res["success_12s"]),
        "reason": res["reason"], "win_pos": res["window"].get("mean_pos"),
        "win_theta": res["window"].get("mean_theta"),
        "posH": res.get("e_pos_at_handoff"), "thH": res.get("e_theta_at_handoff"),
        "n_tok": len(simple), "tokens": json.dumps(simple),
    }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budget-type3", type=int, default=28)
    ap.add_argument("--budget-type2", type=int, default=24)
    ap.add_argument("--n-extra", type=int, default=8)
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

    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    # ---------- 1. Consolidate / classify existing ----------
    consolidated = []
    for src, man_name in [
        ("k49_sliding_recatch_dataset", "UNIFIED_MANIFEST.csv"),
        ("k49_in_recovery_recatch", "UNIFIED_MANIFEST.csv"),
    ]:
        p = ROOT / "results/expert_generation" / src / man_name
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            typ = classify_existing_row(
                r.get("family"), r.get("observed_recovery"),
                int(float(r.get("sliding_verified") or 0)),
                r.get("recatch_subtype"),
                int(float(r.get("n_tokens") or 0)),
                r.get("tokens"),
            )
            consolidated.append({
                "source": src, "ic_id": r["ic_id"], "tag": r.get("tag"),
                "type_class": typ, "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "win_mean_pos": r.get("win_mean_pos"),
                "observed_recovery": r.get("observed_recovery"),
            })
    write_csv(OUT / "RAW" / "EXISTING_CLASSIFICATION.csv", consolidated)
    print("EXISTING", Counter(r["type_class"] for r in consolidated), flush=True)

    # ---------- 2. IC list ----------
    short_ics = {e["ic_id"]: e for e in load_short_handoff_ics()}
    ics = []
    for ic in HARD_ICS:
        snap = ROOT / f"results/expert_generation/k49_in_recovery_recatch/snapshots/{ic}.pkl"
        if not snap.exists():
            snap = snapshot_for(ic)
        pref = short_ics.get(ic, {}).get("prefix") or [30]
        ics.append({"ic_id": ic, "snapshot": str(snap), "prefix": pref, "pool": "hard"})
    # extra high-theta bilateral from short-handoff that may need type2/3
    for e in short_ics.values():
        if e["ic_id"] in HARD_ICS or len(ics) >= 5 + args.n_extra:
            continue
        snap = Path(e["snapshot"])
        blob = pickle.loads(snap.read_bytes())
        st = ic_snapshot_state(env, table, book48, blob)
        if st["e_theta_deg"] >= 12 or st["e_pos_mm"] >= 14:
            ics.append({**e, "pool": "extra_hard_pose", "st": st})
    write_json(OUT / "RAW" / "selected_ics.json", ics)
    print(f"SELECTED n={len(ics)}", flush=True)

    type2_experts, type3_experts = [], []
    eccentric_rows, matched_rows, failures, coverage, manifest = [], [], [], [], []
    budgets = []

    for ie, ic in enumerate(ics):
        t0b = time.time()
        ic_id = ic["ic_id"]
        snap = Path(ic["snapshot"])
        if not snap.exists():
            print("MISSING", ic_id, flush=True)
            continue
        blob = pickle.loads(snap.read_bytes())
        shutil.copy2(snap, OUT / "snapshots" / f"{ic_id}.pkl")
        st0 = ic_snapshot_state(env, table, book48, blob)
        prefix = ic["prefix"]
        print(f"\n[{ie+1}/{len(ics)}] {ic_id} pref={prefix} th={st0['e_theta_deg']:.1f} "
              f"pos={st0['e_pos_mm']:.1f} bil={st0['bilateral']}", flush=True)

        # Matched A/B
        ab = eval_ABCD(env, table, book48, blob, prefix)
        best_c = None
        best_d = None
        n_try = 0

        restore_capture(env, blob["capture"])
        desire = -decompose(env.sim)["err_h"]

        # ---------- Type 3 search ----------
        print("  TYPE3_SEARCH", flush=True)
        props3 = type3_proposals(book48, desire, rng, n=args.budget_type3)
        for ip, pref_seq in enumerate(props3):
            n_try += 1
            ok, trace = run_traced(env, table, book48, blob, pref_seq)
            mech3 = analyze_type3_mechanism(trace)
            eccentric_rows.append({
                "ic_id": ic_id, "tag": f"t3_{ip}", "ok_prefix": int(ok),
                **{k: mech3.get(k) for k in (
                    "is_type3", "eccentric", "axial_max_mm", "downward_motion",
                    "delta_theta_deg", "beneficial_rotation", "post_regrasp_correction",
                    "v_down_hand_z", "obj_dz_m", "reason", "theta_pre_deg", "theta_at_bil_deg",
                )},
            })
            if not ok:
                failures.append({"ic_id": ic_id, "tag": f"t3_{ip}", "family": "type3",
                                 "reason": "DROP", "label": "RECATCH_FAILED"})
                continue
            # evaluate full even if not classic type3 — may still be NEW
            # variants: raw + short return only (no long post pivot — type3 constraint)
            variants = [("raw", list(pref_seq) + [0, 0, 0])]
            restore_capture(env, blob["capture"])
            run_seq(env, table, book48, blob, pref_seq)
            # only short return if needed
            from scripts.generate_k49_sliding_recatch_dataset import hand_ref_error_deg
            herr = hand_ref_error_deg(env, blob)
            if herr > 8:
                for n48 in (8, 16):
                    variants.append((f"ret48x{n48}", list(pref_seq) + [SECURE_REVERSE_WRIST] * n48 + [0, 0, 0]))
            for vname, tokens in variants:
                # reject if tokens contain long post-bilateral pivot — checked via re-trace
                ok2, tr2 = run_traced(env, table, book48, blob, tokens)
                m3 = analyze_type3_mechanism(tr2)
                res = evaluate_full(env, table, book48, blob, tokens, record_obs=False)
                n_try += 1
                if res["SUCCESS_NEW"]:
                    # require type3 physical evidence OR at least eccentric+downward+rot without post-corr
                    if not m3.get("is_type3"):
                        # still may be type2-like; skip type3 label
                        if m3.get("post_regrasp_correction"):
                            failures.append({"ic_id": ic_id, "tag": f"t3_{ip}_{vname}",
                                             "reason": "NEW_BUT_POST_CORR", "label": "NOT_TYPE3"})
                            continue
                        if not (m3.get("eccentric") and m3.get("beneficial_rotation") and m3.get("downward_motion")):
                            failures.append({"ic_id": ic_id, "tag": f"t3_{ip}_{vname}",
                                             "reason": "NEW_NOT_TYPE3_EVIDENCE", "label": "OTHER_NEW"})
                            continue
                    res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
                    mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                    extra = {"expert_type": "TYPE3_DYNAMIC_RECATCH", "type3_mech": m3,
                             "root_episode": ic_id}
                    save_expert(env, ic_id, f"type3_{ip}_{vname}", "type3_dynamic",
                                res, mech, mapping_meta, checksum, snap, extra)
                    write_json(OUT / "raw_contact_traces" / f"{ic_id}__type3_{ip}_{vname}_mech.json", m3)
                    row = {
                        "ic_id": ic_id, "tag": f"type3_{ip}_{vname}",
                        "expert_type": "TYPE3_DYNAMIC_RECATCH", "SUCCESS_NEW": 1,
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "axial_max_mm": m3.get("axial_max_mm"),
                        "delta_theta_deg": m3.get("delta_theta_deg"),
                        "v_down_hand_z": m3.get("v_down_hand_z"),
                        "obj_dz_m": m3.get("obj_dz_m"),
                        "eccentric": int(m3.get("eccentric") or 0),
                        "post_corr": int(m3.get("post_regrasp_correction") or 0),
                        "n_tokens": len(tokens), "config_id": CONFIG_ID,
                        "mapping_checksum": checksum, "tokens": json.dumps(tokens),
                        "reproducible": 1,
                    }
                    type3_experts.append(row)
                    manifest.append(row)
                    best_d = row
                    print(f"  TYPE3 NEW=1 ax={m3.get('axial_max_mm')} dth={m3.get('delta_theta_deg'):.1f} "
                          f"win={row['win_mean_pos']:.2f}", flush=True)
                    break
                else:
                    label = "POSE_ONLY" if res["success_12s"] and (res["window"].get("mean_theta") or 99) <= 10 else (
                        "RECATCH_CONTACT_ONLY" if m3.get("bilateral_i") is not None else "RECATCH_FAILED"
                    )
                    failures.append({
                        "ic_id": ic_id, "tag": f"t3_{ip}_{vname}", "family": "type3",
                        "reason": res["reason"], "label": label,
                        "s12": int(res["success_12s"]),
                        "win_pos": res["window"].get("mean_pos"),
                        "win_theta": res["window"].get("mean_theta"),
                        "type3_reason": m3.get("reason"),
                        "axial_max_mm": m3.get("axial_max_mm"),
                        "delta_theta": m3.get("delta_theta_deg"),
                    })
            if best_d is not None:
                break

        # ---------- Type 2 search ----------
        print("  TYPE2_SEARCH", flush=True)
        props2 = type2_proposals(book48, desire, rng, n=args.budget_type2, open_prefix=prefix)
        for ip, seq in enumerate(props2):
            if best_c is not None and ic_id not in HARD_ICS:
                # still search hard ICs fully
                pass
            n_try += 1
            ok, trace = run_traced(env, table, book48, blob, seq)
            if not ok:
                failures.append({"ic_id": ic_id, "tag": f"t2_{ip}", "family": "type2", "reason": "DROP"})
                continue
            bil_i = find_bilateral_after_loss(trace)
            post_ok, post_info = has_post_manipulation(trace, bil_i)
            had_loss = bil_i is not None
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
                        "had_loss": int(had_loss), "post_manip": int(post_ok),
                    })
                    if res["reason"] in ("DROP_DURING_RECOVERY", "TABLE_DROP"):
                        break
                    continue
                # verify type2: loss + restore + post manipulation after restore
                ok2, tr2 = run_traced(env, table, book48, blob, tokens)
                bil2 = find_bilateral_after_loss(tr2)
                post2, pinfo = has_post_manipulation(tr2, bil2)
                loss2 = bil2 is not None
                if not (loss2 and bil2 is not None and post2):
                    failures.append({"ic_id": ic_id, "tag": f"t2_{ip}_{vname}",
                                     "reason": "NEW_NOT_TYPE2_EVIDENCE",
                                     "had_loss": int(loss2), "post_manip": int(post2)})
                    continue
                th_b = float(tr2[bil2]["e_theta_deg"])
                th_e = float(tr2[-1]["e_theta_deg"])
                pos_b = float(tr2[bil2]["e_pos_mm"])
                pos_e = float(tr2[-1]["e_pos_mm"])
                if not (th_e < th_b - 2.0 or pos_e < pos_b - 1.5):
                    failures.append({"ic_id": ic_id, "tag": f"t2_{ip}_{vname}",
                                     "reason": "POST_MANIP_NO_IMPROVE"})
                    continue
                res = evaluate_full(env, table, book48, blob, tokens, record_obs=True)
                mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
                extra = {
                    "expert_type": "TYPE2_RECATCH_THEN_MANIPULATION",
                    "after_recatch_theta": th_b, "after_recatch_pos": pos_b,
                    "final_theta_before_handoff": th_e, "final_pos_before_handoff": pos_e,
                    "post_manip": pinfo, "root_episode": ic_id,
                }
                save_expert(env, ic_id, f"type2_{ip}_{vname}", "type2_recatch_manip",
                            res, mech, mapping_meta, checksum, snap, extra)
                row = {
                    "ic_id": ic_id, "tag": f"type2_{ip}_{vname}",
                    "expert_type": "TYPE2_RECATCH_THEN_MANIPULATION", "SUCCESS_NEW": 1,
                    "win_mean_pos": res["window"].get("mean_pos"),
                    "win_mean_theta": res["window"].get("mean_theta"),
                    "hold_both_s": res["window"].get("longest_both_ok_s"),
                    "theta_after_recatch": th_b, "theta_after_manip": th_e,
                    "pos_after_recatch": pos_b, "pos_after_manip": pos_e,
                    "post_pivot_n": pinfo.get("post_pivot_n"),
                    "post_slide_n": pinfo.get("post_slide_n"),
                    "n_tokens": len(tokens), "config_id": CONFIG_ID,
                    "mapping_checksum": checksum, "tokens": json.dumps(tokens),
                    "reproducible": 1,
                }
                type2_experts.append(row)
                manifest.append(row)
                best_c = row
                print(f"  TYPE2 NEW=1 th {th_b:.1f}->{th_e:.1f} pos {pos_b:.1f}->{pos_e:.1f} "
                      f"win={row['win_mean_pos']:.2f}", flush=True)
                break
            if best_c is not None:
                break

        # matched row
        matched_rows.append({
            "ic_id": ic_id,
            "e_theta0": st0["e_theta_deg"], "e_pos0": st0["e_pos_mm"],
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
        })
        coverage.append({
            "ic_id": ic_id, "pool": ic.get("pool"), "searched": n_try,
            "type2": int(best_c is not None), "type3": int(best_d is not None),
            "e_theta0": st0["e_theta_deg"], "e_pos0": st0["e_pos_mm"],
            "budget_s": time.time() - t0b,
        })
        budgets.append({"ic_id": ic_id, "n_try": n_try, "s": time.time() - t0b})
        print(f"  DONE t2={best_c is not None} t3={best_d is not None} tries={n_try} "
              f"({time.time()-t0b:.1f}s)", flush=True)

    # ---------- outputs ----------
    write_csv(OUT / "TYPE2_EXPERTS.csv", type2_experts)
    write_csv(OUT / "TYPE3_EXPERTS.csv", type3_experts)
    write_csv(OUT / "ECCENTRIC_CATCH_MECHANISM.csv", eccentric_rows)
    write_csv(OUT / "MATCHED_IC_COMPARISON.csv", matched_rows)
    write_csv(OUT / "FAILURES.csv", failures[:800])
    write_csv(OUT / "ROOT_EPISODE_COVERAGE.csv", coverage)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", manifest)
    write_json(OUT / "RAW" / "budgets.json", {"per_ic": budgets, "wall_s": time.time() - t_wall0})

    n2 = len({r["ic_id"] for r in type2_experts})
    n3 = len({r["ic_id"] for r in type3_experts})
    hard_left = [ic for ic in HARD_ICS
                 if ic not in {r["ic_id"] for r in type2_experts + type3_experts}]
    # Type3 mechanism stats from experts
    t3_stats = []
    for r in type3_experts:
        # load mech file
        p = OUT / "raw_contact_traces" / f"{r['ic_id']}__{r['tag']}_mech.json"
        if p.exists():
            t3_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # matched: does D beat B/C
    d_wins = sum(1 for m in matched_rows if m["best"] == "D_TYPE3")
    c_wins = sum(1 for m in matched_rows if m["best"] == "C_TYPE2")

    answers = {
        "Q1_eccentric_end_catch_possible": any(e.get("eccentric") and e.get("beneficial_rotation")
                                               and e.get("downward_motion") for e in eccentric_rows),
        "Q1_n_eccentric_beneficial": sum(1 for e in eccentric_rows
                                         if e.get("eccentric") and e.get("beneficial_rotation")
                                         and e.get("downward_motion")),
        "Q2_type3_full_expert_exists": n3 > 0,
        "Q3_n_type2_ics": n2,
        "Q3_n_type3_ics": n3,
        "Q3_type2_ids": sorted({r["ic_id"] for r in type2_experts}),
        "Q3_type3_ids": sorted({r["ic_id"] for r in type3_experts}),
        "Q4_type3_stats": [
            {k: s.get(k) for k in (
                "axial_max_mm", "v_rel_pre", "v_down_hand_z", "delta_theta_deg",
                "dw_norm", "theta_pre_deg", "theta_at_bil_deg", "obj_dz_m")}
            for s in t3_stats
        ],
        "Q5_unsolved_hard_ics": hard_left,
        "Q6_matched_D_wins": d_wins,
        "Q6_matched_C_wins": c_wins,
        "Q6_n_matched": len(matched_rows),
        "Q7_ready_unified_SFT": False,  # filled below
        "existing_classification": dict(Counter(r["type_class"] for r in consolidated)),
        "wall_s": time.time() - t_wall0,
    }
    # diversity readiness: need type2 and type3 plus prior type1
    n1 = sum(1 for r in consolidated if r["type_class"] == "TYPE1_CONTACT_PRESERVING")
    answers["Q7_ready_unified_SFT"] = bool(n1 >= 3 and (n2 + n3) >= 3 and (n2 >= 1 or n3 >= 1))
    # honest: want both mechanisms represented
    if n3 == 0:
        answers["Q7_ready_unified_SFT"] = False
        answers["Q7_note"] = (
            f"Type1≈{n1}, Type2 ICs={n2}, Type3 ICs={n3}. "
            "Insufficient Type3 evidence for claiming dynamic-recatch diversity."
        )
    else:
        answers["Q7_note"] = f"Type1≈{n1}, Type2={n2}, Type3={n3} independent ICs."

    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Dynamic Recatch & Recatch-Then-Manipulation Experts

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**SUCCESS_NEW:** locked 10° / 12 mm / ≥0.5 s hold  
**No SFT/GRPO. No K49 changes.**

## Existing consolidation

{answers["existing_classification"]}

Simple Release→Close in-Recovery experts are labeled **SIMPLE_RECATCH**, not Type 2/3.

## Yield

| Class | Distinct ICs | Trajectories |
|-------|-------------:|-------------:|
| TYPE2_RECATCH_THEN_MANIPULATION | **{n2}** | {len(type2_experts)} |
| TYPE3_DYNAMIC_RECATCH | **{n3}** | {len(type3_experts)} |

Hard ICs still unsolved: {hard_left}

## Q1–Q7

1. **Eccentric end-catch with beneficial rotation?** **{answers["Q1_eccentric_end_catch_possible"]}**  
   (candidates with eccentric∧downward∧Δθ>0: {answers["Q1_n_eccentric_beneficial"]})

2. **Type 3 FULL expert without post-regrasp Slide/Pivot?** **{answers["Q2_type3_full_expert_exists"]}**

3. **Independent ICs:** Type2={n2} {answers["Q3_type2_ids"]}; Type3={n3} {answers["Q3_type3_ids"]}

4. **Type 3 measured stats:** see `RAW/answers.json` Q4_type3_stats and `ECCENTRIC_CATCH_MECHANISM.csv`

5. **Unsolved hard ICs:** {hard_left}  
   Typical: orientation >10° after contact restore, or drop after HANDOFF (`FAILURES.csv`).

6. **Matched IC comparison:** Type3 best on {d_wins}/{len(matched_rows)}; Type2 best on {c_wins}/{len(matched_rows)}  
   (`MATCHED_IC_COMPARISON.csv`)

7. **Ready for unified Transformer SFT?** **{answers["Q7_ready_unified_SFT"]}** — {answers.get("Q7_note")}

## Artifacts

`TYPE2_EXPERTS.csv`, `TYPE3_EXPERTS.csv`, `ECCENTRIC_CATCH_MECHANISM.csv`,  
`MATCHED_IC_COMPARISON.csv`, `FAILURES.csv`, `ROOT_EPISODE_COVERAGE.csv`,  
`UNIFIED_MANIFEST.csv`, `DATASET_READINESS.md`, `trajectories/`, `snapshots/`, `raw_contact_traces/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    ready = f"""# Dataset Readiness — Dynamic Recatch Expansion

## Verdict: {"CONDITIONAL YES" if answers["Q7_ready_unified_SFT"] else "NOT YET"}

- Type2 ICs: {n2}
- Type3 ICs: {n3}
- Prior Type1 (from consolidated labels): ~{n1}
- Unsolved hard ICs: {hard_left}

### Merge rule
Append `UNIFIED_MANIFEST.csv` rows with `expert_type` in {{TYPE2_..., TYPE3_...}} into the unified pool.
Keep episode-grouped splits; do not cross-split variants of the same `ic_id`.

### Note
SIMPLE_RECATCH from `k49_in_recovery_recatch` remains useful but is **not** Type 2/3.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
