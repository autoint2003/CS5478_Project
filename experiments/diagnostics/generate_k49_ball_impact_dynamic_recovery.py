"""Generate full SUCCESS_NEW experts from real MuJoCo lateral ball impacts.

Locked K49_tok48_wy0p25. No SFT/GRPO. No physics/SUCCESS_NEW changes.
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

from data_generation.datasets.ball_impact import (
    physics_record,
    run_episode,
)
from data_generation.datasets.recovery_rebuild import immediate_handoff
from data_generation.recording.trajectory_recorder import save_pair
from envs.simulation.ballistic_scene import CYL_H
from envs.simulation.capture import restore_capture
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
    find_bilateral_after_loss,
    has_post_manipulation,
    run_traced,
    type2_proposals,
)
from scripts.generate_k49_in_recovery_recatch import (
    find_first_bilateral,
    pose_correction_proposals,
    regrasp_proposals,
)
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    classify_mechanisms,
    decompose,
    evaluate_full,
    hand_ref_error_deg,
    mapping_checksum,
    run_seq,
    slide_proposals,
    write_csv,
    write_json,
)

OUT = ROOT / "results" / "expert_generation" / "k49_ball_impact_dynamic_recovery"
SEED = 20261019
HARD_SNAPS = ROOT / "results" / "diagnostics" / "hard_ic_mining" / "snapshots"
TYPE2_CSV = ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv"
SLIDING_CSV = ROOT / "results/expert_generation/k49_sliding_recatch_dataset/UNIFIED_MANIFEST.csv"
LOST_CSV = ROOT / "results/expert_generation/k49_lost_grasp_recovery/TYPE3_FULL_EXPERTS.csv"
CAPTURE_SEED = [30, 12, 12, 8] + [0] * 7
# Prefer reusing hard ICs already known solvable under legal K49 (ball-impact origin).
PRIORITY_HARD_IDS = (
    "hard_910101", "hard_910263", "hard_910305", "hard_910220",
)


def draw_lateral_spec(draw: int) -> dict:
    """Physically legal ball launch biased toward lateral (±Y-ish) impacts."""
    rng = np.random.default_rng(SEED + int(draw))
    # Prefer near ±Y (90/270) and nearby oblique for lateral disturbance
    az_pool = [70, 90, 110, 250, 270, 290, 45, 135, 225, 315]
    azimuth = float(rng.choice(az_pool)) + float(rng.uniform(-8, 8))
    azimuth = azimuth % 360.0
    # Favor moderate speeds (recoverable), with a hard tail for coverage
    roll = float(rng.random())
    if roll < 0.35:
        speed = float(rng.uniform(6.2, 7.5))
    elif roll < 0.75:
        speed = float(rng.uniform(7.5, 8.8))
    else:
        speed = float(rng.uniform(8.8, 9.8))
    z_off = float(rng.uniform(-0.018, 0.018))
    if float(rng.random()) < 0.45:
        phi = float(rng.choice([-1.0, 1.0]) * rng.uniform(20.0, 50.0))
    else:
        phi = float(rng.uniform(-18.0, 18.0))
    delay = float(rng.uniform(0.05, 0.40))
    return {
        "azimuth_deg": azimuth,
        "speed_mps": speed,
        "z_off_m": z_off,
        "phi_deg": phi,
        "delay_s": delay,
        "direction": "lateral_biased",
        "family": "ballistic_impact",
        "protocol": "k49_ball_impact_dynamic_recovery_v1",
        "draw": int(draw),
        "split": "gen",
    }


def audit_physics(env) -> dict:
    rec = physics_record(env.sim)
    rec["cyl_half_height_xml_m"] = 0.5 * float(CYL_H)
    return rec


def classify_ic_regime(blob, st) -> list[str]:
    """Offline physical regime tags at Recovery onset (may overlap)."""
    tags = []
    ev = blob.get("evidence") or {}
    states = set(ev.get("states") or [])
    entry = ev.get("entry") or {}
    peaks = ev.get("state_peaks") or {}
    nL, nR = int(st["nL"]), int(st["nR"])
    bil = bool(st["bilateral"])
    th = float(st["e_theta_deg"])
    v = float(st["v_rel_norm"])
    w = float(st["w_rel_norm"])

    if bil and (nL + nR) > 0 and v < 0.35 and "complete_grasp_loss" not in states:
        tags.append("IMPACT_EARLY")
    if (nL > 0) ^ (nR > 0) or "asymmetric_or_unilateral" in states:
        tags.append("UNILATERAL_ESCAPE")
    if (nL + nR) == 0 or "complete_grasp_loss" in states or "airborne_object" in states:
        tags.append("CONTACT_LOSS")
    if v >= 0.20 or w >= 4.0 or float(peaks.get("max_v_tan_mps") or 0) >= 0.15:
        tags.append("HIGH_MOTION")
    if th >= 30.0 or float(np.degrees(float(peaks.get("max_e_rot_rad") or 0))) >= 30.0:
        tags.append("NEAR_HORIZONTAL")
    if not tags:
        tags.append("IMPACT_EARLY")
    return tags


def ic_metrics(env, blob):
    restore_capture(env, blob["capture"])
    cs = contact_summary(env.sim)
    m = measure(env.sim, U_TGT, P_TGT)
    d = decompose(env.sim)
    return {
        "nL": int(cs["nL"]), "nR": int(cs["nR"]), "bilateral": bool(cs["bilateral"]),
        "e_theta_deg": float(m["e_theta_deg"]), "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]), "e_lat_mm": float(d["e_lat_mm"]),
        "v_rel_norm": float(np.linalg.norm(m["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
        "v_rel_h": np.asarray(m["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(m["w_rel_h"], float).tolist(),
        "obj_z": float(m["obj_z"]),
    }


def load_prior_expert_tokens():
    """All verified FULL expert sequences from prior ball-impact / recovery datasets."""
    out = []
    by_ic = {}
    for path in (SLIDING_CSV, TYPE2_CSV, LOST_CSV):
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            try:
                toks = json.loads(r["tokens"])
            except Exception:
                continue
            if not toks:
                continue
            out.append(list(toks))
            ic = r.get("ic_id") or ""
            by_ic[ic] = list(toks)
            base = ic.split("_loss")[0]
            by_ic.setdefault(base, list(toks))
    # short generic seeds
    generics = [
        CAPTURE_SEED,
        [30] + [0] * 12,
        [0] * 20,
        [0] * 40,
        [12] * 5 + [0] * 16,
        [12] * 4 + [17] * 16 + [0] * 8,
        [12] * 4 + [33] * 16 + [0] * 8,
        [4] * 20 + [12] * 5 + [0] * 5,
        [4] * 24 + [12] * 5 + [17] * 12 + [0] * 5,
        [6] * 16 + [12] * 5 + [0] * 5,
        [10] * 12 + [12] * 5 + [0] * 5,
        [30] + [17] * 10 + [0] * 10,
        [30] + [33] * 10 + [0] * 10,
        [30, 12, 12, 8, 17] + [0] * 12,
        [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [0] * 8 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
    ]
    seen, uniq = set(), []
    for s in out + generics:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    return uniq, by_ic


def load_type2_seeds():
    seeds, _ = load_prior_expert_tokens()
    return seeds


def mechanism_tags(trace, bil_i, regimes):
    tags = set(regimes)
    if bil_i is None:
        return sorted(tags)
    # pre-bilateral chase/recatch
    pre = trace[: bil_i + 1]
    if any(int(r["nL"]) + int(r["nR"]) == 0 for r in pre):
        tags.add("COMPLETE_CONTACT_LOSS")
        tags.add("RECATCH")
    if any(int(r["tok"]) in (1, 5, 8, 13, 15, 17, 33, 19) for r in pre):
        tags.add("CHASE")
    post, info = has_post_manipulation(trace, bil_i)
    if info.get("post_pivot_n", 0) >= 8:
        tags.add("PIVOTING")
    if info.get("post_slide_n", 0) >= 8:
        tags.add("SLIDING")
    if any(int(r["tok"]) == SECURE_REVERSE_WRIST for r in trace):
        tags.add("RETURN")
    if not post and bil_i is not None and all(
        int(r["nL"]) > 0 and int(r["nR"]) > 0 for r in trace[: min(5, bil_i + 1)]
    ):
        tags.add("CONTACT_PRESERVING")
    # capture brake heuristic: soft then secure around bil
    win = trace[max(0, bil_i - 4): bil_i + 6]
    if any(int(r["tok"]) == 12 for r in win) and any(int(r["tok"]) == 0 for r in win):
        tags.add("CAPTURE_BRAKE")
    if any(float(np.linalg.norm(r["w_rel_h"])) > 5.0 for r in trace[: bil_i + 1]):
        tags.add("HIGH_ANGULAR_MOTION")
    return sorted(tags)


def save_expert(env, ic_id, tag, res, mech, mapping_meta, checksum, snap, extra):
    stem = f"{ic_id}__{tag}"
    save_pair(
        OUT / "trajectories", stem,
        res["actions"], res["observations"], res["contacts"],
        res["tokens"], res["residuals"], res["handoff_obs"],
    )
    meta = {
        "ic_id": ic_id, "tag": tag, "family": "ball_impact_dynamic_recovery",
        "config_id": CONFIG_ID, "mapping_checksum": checksum,
        "SUCCESS_NEW": res["SUCCESS_NEW"], "label": res["label"],
        "window": res["window"], "tokens": res["tokens"],
        "snapshot_src": str(snap), "obs_dim": 39, "handoff_id": 47,
        "expert_type": "BALL_IMPACT_FULL_EXPERT",
        **extra,
    }
    write_json(OUT / "trajectories" / f"{stem}_meta.json", meta)
    if res.get("trace"):
        write_json(OUT / "raw_contact_traces" / f"{stem}_trace.json", {
            "trace": res["trace"], "mech": mech, "extra": extra,
        })
    return meta


def search_recovery(env, table, book48, blob, rng, budget_a=24, budget_b=10,
                    prefer_ic_tokens=None, bilateral0=True):
    """Joint Stage A/B style search; returns best FULL result or None."""
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    seeds, _ = load_prior_expert_tokens()
    props = []
    # IC-matched prior transfer first
    if prefer_ic_tokens:
        props.append(list(prefer_ic_tokens))
    props.extend(seeds[:12])
    if bilateral0:
        # contact-preserving slide / settle first for intact grasps
        props.extend(slide_proposals(book48, desire, rng, n=max(8, budget_a // 2)))
        props.append([0] * 24)
        props.append([12] * 4 + [0] * 20)
        props.append([12] * 4 + [17] * 16 + [0] * 8)
    props.extend(regrasp_proposals(book48, desire, rng, n=budget_a // 2))
    props.extend(type2_proposals(book48, desire, rng, n=budget_a // 2, open_prefix=[30]))
    seen, uniq = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(k))
    # keep head stable (transfer + known seeds), shuffle only the tail
    head_n = min(16, len(uniq))
    head, tail = uniq[:head_n], uniq[head_n:]
    rng.shuffle(tail)
    uniq = (head + tail)[:budget_a]

    best = None
    trials = []
    n_try = 0
    for ia, seq in enumerate(uniq):
        variants = [("raw", list(seq)), ("settle12", list(seq) + [0] * 12),
                    ("settle24", list(seq) + [0] * 24)]
        restore_capture(env, blob["capture"])
        ok, _ = run_seq(env, table, book48, blob, seq)
        n_try += 1
        if not ok:
            trials.append({"tag": f"a{ia}", "ok": 0, "reason": "DROP", "NEW": 0})
            continue
        if hand_ref_error_deg(env, blob) > 8:
            variants.append(("ret12", list(seq) + [SECURE_REVERSE_WRIST] * 12 + [0] * 8))
            variants.append(("ret20", list(seq) + [SECURE_REVERSE_WRIST] * 20 + [0] * 8))
            variants.append(("ret28", list(seq) + [SECURE_REVERSE_WRIST] * 28 + [0] * 8))
        restore_capture(env, blob["capture"])
        run_seq(env, table, book48, blob, seq)
        desire_b = -decompose(env.sim)["err_h"]
        for ib, corr in enumerate(pose_correction_proposals(book48, desire_b, rng, n=budget_b)[:budget_b]):
            if corr:
                variants.append((f"b{ib}", list(seq) + list(corr) + [0] * 8))
                if hand_ref_error_deg(env, blob) > 8:
                    variants.append((
                        f"b{ib}ret",
                        list(seq) + list(corr) + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
                    ))

        for vname, toks in variants:
            n_try += 1
            res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
            row = {
                "tag": f"a{ia}_{vname}", "NEW": int(res["SUCCESS_NEW"]),
                "s12": int(res["success_12s"]), "reason": res["reason"],
                "win_th": res["window"].get("mean_theta"),
                "win_pos": res["window"].get("mean_pos"),
                "n_tokens": len(toks),
            }
            trials.append(row)
            if not res["SUCCESS_NEW"]:
                continue
            ok2, tr2 = run_traced(env, table, book48, blob, toks)
            bil = find_first_bilateral(tr2, 0) or find_bilateral_after_loss(tr2)
            if bil is None and any(r.get("bilateral") for r in tr2):
                bil = next(i for i, r in enumerate(tr2) if r.get("bilateral"))
            if bil is None:
                continue
            best = {
                "tokens": toks, "res_light": res, "trace": tr2, "bil_i": bil,
                "tag": f"a{ia}_{vname}", "n_try": n_try,
            }
            return best, trials, n_try
    return best, trials, n_try


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-gen", type=int, default=48, help="new impact episode attempts")
    ap.add_argument("--n-reuse", type=int, default=24, help="reuse existing hard snaps")
    ap.add_argument("--budget-a", type=int, default=16)
    ap.add_argument("--budget-b", type=int, default=8)
    ap.add_argument("--max-search-ics", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_gen = 12
        args.n_reuse = 8
        args.max_search_ics = 6
        args.budget_a = 10
        args.budget_b = 5

    t0w = time.time()
    for sub in ("trajectories", "snapshots", "raw_contact_traces", "RAW"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
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

    phys = audit_physics(env)
    write_json(OUT / "RAW" / "physics_audit.json", phys)
    write_csv(OUT / "IMPACT_PHYSICS_AUDIT.csv", [phys])
    print("PHYSICS", {k: phys[k] for k in (
        "cylinder_mass_kg", "ball_mass_kg", "ball_radius_m", "cylinder_radius_m",
        "finger_object_mu", "source_constants_match_model",
    )}, flush=True)

    # ---------- generate new lateral impacts ----------
    roots = []
    gen_log = []
    print(f"Generating up to {args.n_gen} lateral-biased impacts...", flush=True)
    for i in range(args.n_gen):
        spec = draw_lateral_spec(i)
        eid = f"bimp_{i:04d}"
        spec["attempt_id"] = eid
        evidence, blob = run_episode(env, spec)
        row = {
            "root_id": eid, "category": evidence.get("category"),
            "reason": evidence.get("reason"),
            "states": json.dumps(evidence.get("states") or []),
            "azimuth_deg": spec["azimuth_deg"], "speed_mps": spec["speed_mps"],
            "z_off_m": spec["z_off_m"], "phi_deg": spec["phi_deg"],
            "delay_s": spec["delay_s"],
            "launch_speed": evidence.get("launch_speed"),
            "peak_fn": evidence.get("peak_normal_force_n"),
            "contact_t": evidence.get("contact_t"),
            "entry_t": evidence.get("entry_t"),
            "max_e_pos_m": evidence.get("max_e_pos_m"),
            "source": "generated",
        }
        gen_log.append(row)
        if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
            print(f"  reject {eid} {evidence.get('category')}", flush=True)
            continue
        nom = immediate_handoff(env, blob)
        row["nominal_class"] = nom["nominal_class"]
        if nom["nominal_class"] != "RECOVERY_CANDIDATE":
            print(f"  trivial/nominal {eid}", flush=True)
            continue
        path = OUT / "snapshots" / f"{eid}.pkl"
        path.write_bytes(pickle.dumps(blob))
        st = ic_metrics(env, blob)
        regimes = classify_ic_regime(blob, st)
        roots.append({
            "root_id": eid, "snapshot": str(path), "blob": blob,
            "spec": spec, "evidence": evidence, "st": st, "regimes": regimes,
            "source": "generated", "nominal_class": nom["nominal_class"],
        })
        print(f"  ACCEPT {eid} regimes={regimes} th={st['e_theta_deg']:.1f} "
              f"pos={st['e_pos_mm']:.1f} v={st['v_rel_norm']:.2f}", flush=True)

    # ---------- reuse hard mining snaps (ball-impact origin) ----------
    _, prior_by_ic = load_prior_expert_tokens()
    prior_full_ids = set(prior_by_ic.keys())

    def add_hard_root(sp: Path, force=False) -> bool:
        if any(r["root_id"] == sp.stem for r in roots):
            return False
        try:
            blob = pickle.loads(sp.read_bytes())
        except Exception:
            return False
        if "capture" not in blob or "ref" not in blob:
            return False
        nom = immediate_handoff(env, blob)
        if nom["nominal_class"] != "RECOVERY_CANDIDATE":
            return False
        st = ic_metrics(env, blob)
        if (not force) and st["e_pos_mm"] < 8 and st["e_theta_deg"] < 8 and st["v_rel_norm"] < 0.08:
            return False
        # skip irreversible ballistic escapes for default reuse
        if (not force) and st["v_rel_norm"] > 0.8 and (st["nL"] + st["nR"]) == 0:
            return False
        regimes = classify_ic_regime(blob, st)
        dst = OUT / "snapshots" / f"{sp.stem}.pkl"
        if not dst.exists():
            shutil.copy2(sp, dst)
        roots.append({
            "root_id": sp.stem, "snapshot": str(dst), "blob": blob,
            "spec": blob.get("spec") or {}, "evidence": blob.get("evidence") or {},
            "st": st, "regimes": regimes, "source": "hard_ic_mining",
            "nominal_class": nom["nominal_class"],
            "prior_tokens": prior_by_ic.get(sp.stem),
        })
        gen_log.append({
            "root_id": sp.stem, "category": "reused_hard",
            "reason": "hard_ic_mining", "states": json.dumps((blob.get("evidence") or {}).get("states") or []),
            "azimuth_deg": (blob.get("spec") or {}).get("azimuth_deg"),
            "speed_mps": (blob.get("spec") or {}).get("speed_mps"),
            "z_off_m": (blob.get("spec") or {}).get("z_off_m"),
            "phi_deg": (blob.get("spec") or {}).get("phi_deg"),
            "delay_s": (blob.get("spec") or {}).get("delay_s"),
            "source": "hard_ic_mining", "nominal_class": nom["nominal_class"],
        })
        return True

    # Priority: known prior-expert hard ICs first
    for hid in PRIORITY_HARD_IDS:
        sp = HARD_SNAPS / f"{hid}.pkl"
        if sp.exists():
            add_hard_root(sp, force=True)

    reused = sum(1 for r in roots if r["source"] == "hard_ic_mining")
    if HARD_SNAPS.exists():
        # Prefer moderate-velocity bilateral/unilateral candidates
        candidates = sorted(HARD_SNAPS.glob("hard_*.pkl"))
        # score-sort later by loading lightly — take sequential with filter
        for sp in candidates:
            if reused >= args.n_reuse:
                break
            if add_hard_root(sp, force=False):
                reused = sum(1 for r in roots if r["source"] == "hard_ic_mining")

    write_csv(OUT / "BALL_IMPACT_ROOTS.csv", [
        {k: v for k, v in r.items() if k not in ("blob", "evidence", "st", "spec")}
        | {
            "regimes": json.dumps(r["regimes"]),
            "e_theta0": r["st"]["e_theta_deg"], "e_pos0": r["st"]["e_pos_mm"],
            "v0": r["st"]["v_rel_norm"], "w0": r["st"]["w_rel_norm"],
            "nL": r["st"]["nL"], "nR": r["st"]["nR"], "bilateral": int(r["st"]["bilateral"]),
            "azimuth_deg": (r.get("spec") or {}).get("azimuth_deg"),
            "speed_mps": (r.get("spec") or {}).get("speed_mps"),
            "phi_deg": (r.get("spec") or {}).get("phi_deg"),
            "z_off_m": (r.get("spec") or {}).get("z_off_m"),
        }
        for r in roots
    ])
    write_csv(OUT / "RAW" / "generation_log.csv", gen_log)
    print(f"ROOTS n={len(roots)} (gen+reuse)", flush=True)

    # meaningful disturbance counts
    n_lat = sum(1 for r in roots if r["st"]["e_pos_mm"] >= 10 or r["st"]["e_lat_mm"] >= 8)
    n_ang = sum(1 for r in roots if r["st"]["e_theta_deg"] >= 12 or "NEAR_HORIZONTAL" in r["regimes"])
    n_uni = sum(1 for r in roots if "UNILATERAL_ESCAPE" in r["regimes"])
    n_loss = sum(1 for r in roots if "CONTACT_LOSS" in r["regimes"])
    n_hm = sum(1 for r in roots if "HIGH_MOTION" in r["regimes"])

    # Recoverability-aware ranking: prefer disturbed but reachable ICs
    def recoverability_score(r):
        st = r["st"]
        s = 0.0
        # pose error (want nontrivial)
        s += min(st["e_theta_deg"], 40) * 0.6
        s += min(st["e_pos_mm"], 30) * 0.5
        # mild motion is good; extreme motion is hard / often irreversible
        v, w = st["v_rel_norm"], st["w_rel_norm"]
        if v < 0.35:
            s += v * 20
        else:
            s -= (v - 0.35) * 25
        if w < 8:
            s += w * 0.5
        else:
            s -= (w - 8) * 1.5
        if st["bilateral"]:
            s += 18
        elif (st["nL"] + st["nR"]) > 0:
            s += 10
        else:
            s -= 8
        if r.get("prior_tokens"):
            s += 40  # known transferable expert
        if r["root_id"] in PRIORITY_HARD_IDS:
            s += 30
        if r["source"] == "generated":
            s += 8  # prefer new impact roots when comparable
        if "NEAR_HORIZONTAL" in r["regimes"] and st["bilateral"]:
            s += 6
        return s

    ranked = sorted(roots, key=recoverability_score, reverse=True)
    search_list = []
    # 1) New generated impacts (primary coverage gain)
    for r in ranked:
        if r["source"] == "generated":
            search_list.append(r)
    # 2) Hard ICs not already in prior FULL manifests
    for r in ranked:
        if r in search_list:
            continue
        if r["root_id"] not in prior_full_ids and r["root_id"].split("_loss")[0] not in prior_full_ids:
            search_list.append(r)
        if len(search_list) >= max(2, args.max_search_ics - 4):
            break
    # 3) A few known-solvable priority ICs for ball-impact revalidation / transfer
    for hid in PRIORITY_HARD_IDS:
        for r in roots:
            if r["root_id"] == hid and r not in search_list:
                search_list.append(r)
                break
        if len(search_list) >= args.max_search_ics:
            break
    # 4) Fill remaining by recoverability
    for r in ranked:
        if r not in search_list:
            search_list.append(r)
        if len(search_list) >= args.max_search_ics:
            break
    search_list = search_list[: args.max_search_ics]
    print(
        f"SEARCH {len(search_list)} ICs "
        f"(gen={sum(1 for r in search_list if r['source']=='generated')}, "
        f"novel_hard={sum(1 for r in search_list if r['source']!='generated' and r['root_id'] not in prior_full_ids)}, "
        f"prior_reval={sum(1 for r in search_list if r['root_id'] in prior_full_ids)})",
        flush=True,
    )

    ic_rows = []
    baselines = []
    search_rows = []
    post_rows = []
    full_experts = []
    failures = []
    coverage = []
    n_rollouts = 0

    for ie, root in enumerate(search_list):
        t0b = time.time()
        ic_id = root["root_id"]
        blob = root["blob"]
        st = root["st"]
        regimes = root["regimes"]
        snap = Path(root["snapshot"])
        print(f"\n[{ie+1}/{len(search_list)}] {ic_id} {regimes} "
              f"th={st['e_theta_deg']:.1f} pos={st['e_pos_mm']:.1f} "
              f"v={st['v_rel_norm']:.2f} w={st['w_rel_norm']:.2f}", flush=True)

        ic_rows.append({
            "ic_id": ic_id, "root_id": ic_id, "source": root["source"],
            "regimes": json.dumps(regimes),
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "e_lat0": st["e_lat_mm"], "e_ax0": st["e_ax_mm"],
            "v0": st["v_rel_norm"], "w0": st["w_rel_norm"],
            "nL": st["nL"], "nR": st["nR"], "bilateral": int(st["bilateral"]),
            "azimuth_deg": (root.get("spec") or {}).get("azimuth_deg"),
            "speed_mps": (root.get("spec") or {}).get("speed_mps"),
            "phi_deg": (root.get("spec") or {}).get("phi_deg"),
            "z_off_m": (root.get("spec") or {}).get("z_off_m"),
            "peak_fn": (root.get("evidence") or {}).get("peak_normal_force_n"),
            "states": json.dumps((root.get("evidence") or {}).get("states") or []),
            "snapshot": str(snap),
        })

        # matched baselines A–D
        res_a = evaluate_full(env, table, book48, blob, [], record_obs=False)
        n_rollouts += 1
        res_b = evaluate_full(env, table, book48, blob, [0] * 16, record_obs=False)
        n_rollouts += 1
        res_c = evaluate_full(env, table, book48, blob, [30] + [0] * 8, record_obs=False)
        n_rollouts += 1
        res_d = evaluate_full(env, table, book48, blob, CAPTURE_SEED, record_obs=False)
        n_rollouts += 1
        simple = res_a["SUCCESS_NEW"] or res_b["SUCCESS_NEW"] or res_c["SUCCESS_NEW"]
        base = {
            "ic_id": ic_id, "regimes": json.dumps(regimes),
            "e_theta0": st["e_theta_deg"], "e_pos0": st["e_pos_mm"],
            "A_NEW": int(res_a["SUCCESS_NEW"]), "A_s12": int(res_a["success_12s"]), "A_reason": res_a["reason"],
            "B_NEW": int(res_b["SUCCESS_NEW"]), "B_s12": int(res_b["success_12s"]), "B_reason": res_b["reason"],
            "C_NEW": int(res_c["SUCCESS_NEW"]), "C_s12": int(res_c["success_12s"]), "C_reason": res_c["reason"],
            "D_NEW": int(res_d["SUCCESS_NEW"]), "D_s12": int(res_d["success_12s"]), "D_reason": res_d["reason"],
            "simple_solves": int(simple),
        }

        ba = args.budget_a + (10 if root["source"] == "generated" else 0)
        bb = args.budget_b + (4 if root["source"] == "generated" else 0)
        # Mild bilateral disturbance: lean on contact-preserving search
        if st["bilateral"] and st["v_rel_norm"] < 0.25:
            ba = max(ba, args.budget_a + 8)
        best, trials, n_try = search_recovery(
            env, table, book48, blob, rng,
            budget_a=ba, budget_b=bb,
            prefer_ic_tokens=root.get("prior_tokens") or prior_by_ic.get(ic_id),
            bilateral0=bool(st["bilateral"]),
        )
        n_rollouts += n_try
        for t in trials:
            search_rows.append({"ic_id": ic_id, **t})
            if not t.get("NEW"):
                failures.append({"ic_id": ic_id, **t})

        full_row = None
        if best is not None:
            toks = best["tokens"]
            res = evaluate_full(env, table, book48, blob, toks, record_obs=True)
            n_rollouts += 1
            mech = classify_mechanisms(res["trace"], res["cs0"], res["d0"], res.get("d_handoff"))
            tags = mechanism_tags(best["trace"], best["bil_i"], regimes)
            post, pinfo = has_post_manipulation(best["trace"], best["bil_i"])
            post_rows.append({
                "ic_id": ic_id, "bil_i": best["bil_i"], "post": int(post),
                "post_pivot_n": pinfo.get("post_pivot_n"),
                "post_slide_n": pinfo.get("post_slide_n"),
                "theta_at_bil": best["trace"][best["bil_i"]]["e_theta_deg"],
                "pos_at_bil": best["trace"][best["bil_i"]]["e_pos_mm"],
            })
            extra = {
                "mechanism_tags": tags, "regimes": regimes,
                "root_episode": ic_id, "source": root["source"],
                "spec": root.get("spec"), "newly_vs_simple": int(not simple),
                "bilateral_i": best["bil_i"],
            }
            save_expert(env, ic_id, best["tag"], res, mech, mapping_meta, checksum, snap, extra)
            full_row = {
                "ic_id": ic_id, "tag": best["tag"], "root_episode": ic_id,
                "expert_type": "BALL_IMPACT_FULL_EXPERT",
                "SUCCESS_NEW": 1,
                "win_mean_pos": res["window"].get("mean_pos"),
                "win_mean_theta": res["window"].get("mean_theta"),
                "hold_both_s": res["window"].get("longest_both_ok_s"),
                "regimes": json.dumps(regimes),
                "mechanism_tags": json.dumps(tags),
                "newly_vs_simple": int(not simple),
                "n_tokens": len(toks), "tokens": json.dumps(toks),
                "config_id": CONFIG_ID, "mapping_checksum": checksum,
                "source": root["source"], "reproducible": 1,
                "needs_slide_pivot": int(post),
            }
            full_experts.append(full_row)
            print(f"  FULL NEW=1 tags={tags} th={full_row['win_mean_theta']:.1f} "
                  f"pos={full_row['win_mean_pos']:.1f} newly={not simple}", flush=True)
        else:
            print("  no FULL", flush=True)

        base["E_NEW"] = int(full_row is not None)
        base["E_win_th"] = None if not full_row else full_row["win_mean_theta"]
        base["E_win_pos"] = None if not full_row else full_row["win_mean_pos"]
        base["best"] = (
            "E" if full_row else (
                "D" if res_d["SUCCESS_NEW"] else (
                    "C" if res_c["SUCCESS_NEW"] else (
                        "B" if res_b["SUCCESS_NEW"] else (
                            "A" if res_a["SUCCESS_NEW"] else "NONE"))))
        )
        baselines.append(base)
        coverage.append({
            "root_id": ic_id, "source": root["source"],
            "regimes": json.dumps(regimes), "searched": n_try,
            "full": int(full_row is not None),
            "simple_solves": int(simple),
            "newly": int(full_row is not None and not simple),
            "budget_s": time.time() - t0b,
            "challenge": recoverability_score(root),
        })

    # outputs
    write_csv(OUT / "RECOVERY_ICS.csv", ic_rows)
    write_csv(OUT / "MATCHED_BASELINES.csv", baselines)
    write_csv(OUT / "DYNAMIC_RECATCH_SEARCH.csv", search_rows[:2000])
    write_csv(OUT / "POST_RECATCH_MANIPULATION.csv", post_rows)
    write_csv(OUT / "FULL_EXPERTS.csv", full_experts)
    write_csv(OUT / "UNIFIED_MANIFEST.csv", full_experts)
    write_csv(OUT / "FAILURES.csv", failures[:1500])
    write_csv(OUT / "ROOT_EPISODE_COVERAGE.csv", coverage)

    n_full = len({r["ic_id"] for r in full_experts})
    n_new = len({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1})
    n_bil = sum(1 for r in post_rows) + sum(
        1 for s in search_rows if s.get("NEW")
    )
    # better: count ICs with any NEW trial that had bil - use full + post
    n_need_sp = sum(1 for r in full_experts if int(r.get("needs_slide_pivot") or 0) == 1)
    n_horiz_full = sum(
        1 for r in full_experts
        if "NEAR_HORIZONTAL" in json.loads(r.get("regimes") or "[]")
    )
    fail_ctr = Counter(f.get("reason") for f in failures)

    # prior dataset sizes
    n1 = n2 = n3 = 0
    p1 = ROOT / "results/expert_generation/k49_sliding_recatch_dataset/UNIFIED_MANIFEST.csv"
    if p1.exists():
        n1 = len({r["ic_id"] for r in csv.DictReader(p1.open(encoding="utf-8"))})
    p2 = ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv"
    if p2.exists():
        n2 = len({r["ic_id"] for r in csv.DictReader(p2.open(encoding="utf-8"))})
    p3 = ROOT / "results/expert_generation/k49_lost_grasp_recovery/TYPE3_FULL_EXPERTS.csv"
    if p3.exists():
        n3 = len({r["ic_id"] for r in csv.DictReader(p3.open(encoding="utf-8"))})

    answers = {
        "Q1_n_independent_impact_roots": len(roots),
        "Q1_n_generated_attempts": args.n_gen,
        "Q1_n_accepted_gen": sum(1 for r in roots if r["source"] == "generated"),
        "Q1_n_reused_hard": sum(1 for r in roots if r["source"] == "hard_ic_mining"),
        "Q2_n_lateral_disp": n_lat,
        "Q2_n_angular": n_ang,
        "Q2_n_unilateral": n_uni,
        "Q2_n_contact_loss": n_loss,
        "Q2_n_high_motion": n_hm,
        "Q3_n_challenging_searched": len(search_list),
        "Q3_n_recoverable_full": n_full,
        "Q4_n_stable_bilateral_via_experts": n_full,  # FULL implies bil before HANDOFF
        "Q5_n_need_slide_pivot": n_need_sp,
        "Q6_n_full_SUCCESS_NEW": n_full,
        "Q6_full_ids": sorted({r["ic_id"] for r in full_experts}),
        "Q7_n_newly_vs_simple": n_new,
        "Q7_newly_ids": sorted({r["ic_id"] for r in full_experts if int(r.get("newly_vs_simple") or 0) == 1}),
        "Q8_n_near_horizontal_full": n_horiz_full,
        "Q9_failure_reasons": dict(fail_ctr.most_common(12)),
        "Q10_added_independent_roots": n_full,
        "Q10_added_novel_vs_prior_manifest": len({
            r["ic_id"] for r in full_experts if r["ic_id"] not in prior_full_ids
        }),
        "Q10_revalidated_prior_ids": sorted({
            r["ic_id"] for r in full_experts if r["ic_id"] in prior_full_ids
        }),
        "Q10_prior_type1": n1, "Q10_prior_type2": n2, "Q10_prior_type3_lost": n3,
        "Q11_ready_unified_SFT": bool((n1 + n2 + n3 + n_full) >= 12 and n_full >= 2),
        "wall_s": time.time() - t0w,
        "n_rollouts": n_rollouts,
        "physics": {
            "cyl_m": phys.get("cylinder_mass_kg"), "ball_m": phys.get("ball_mass_kg"),
            "ball_r": phys.get("ball_radius_m"), "cyl_r": phys.get("cylinder_radius_m"),
            "mu": phys.get("finger_object_mu"),
        },
    }
    if n_full < 2:
        answers["Q11_ready_unified_SFT"] = False
    answers["Q11_note"] = (
        f"Prior Type1≈{n1}, Type2={n2}, Type3-lost≈{n3}; this batch adds {n_full} ball-impact FULL roots "
        f"({n_new} newly vs simple). "
        + ("Enough to start unified SFT with episode-grouped splits."
           if answers["Q11_ready_unified_SFT"] else
           "Need more nontrivial ball-impact FULL experts before claiming readiness.")
    )
    # refine Q9 primary
    top = fail_ctr.most_common(1)
    answers["Q9_primary"] = top[0][0] if top else None
    write_json(OUT / "RAW" / "answers.json", answers)

    report = f"""# Ball-Impact Dynamic Recovery Experts

**Config:** `{CONFIG_ID}` checksum `{checksum}`  
**Scene:** `assets/scene_ballistic_impact.xml` (cyl_m={phys.get('cylinder_mass_kg')}, ball_m={phys.get('ball_mass_kg')}, ball_r={phys.get('ball_radius_m')})  
**SUCCESS_NEW locked. No SFT/GRPO. No physics changes.**

## Impact generation

| Metric | Value |
|--------|------:|
| Episode attempts (new) | {args.n_gen} |
| Accepted new roots | {answers['Q1_n_accepted_gen']} |
| Reused hard_ic_mining roots | {answers['Q1_n_reused_hard']} |
| **Independent Recovery roots** | **{len(roots)}** |

Meaningful disturbance among roots: lateral/pos {n_lat}, angular {n_ang}, unilateral {n_uni}, contact_loss {n_loss}, high_motion {n_hm}.

## Search / yield

| Metric | Value |
|--------|------:|
| Challenging ICs searched | {len(search_list)} |
| BALL_IMPACT_FULL_EXPERT ICs | **{n_full}** |
| Newly vs simple A/B/C | **{n_new}** |
| FULL needing post Slide/Pivot | {n_need_sp} |
| Near-horizontal FULL | {n_horiz_full} |
| Rollouts | {n_rollouts} |
| Wall time | {answers['wall_s']:.1f}s |

## Q1–Q11

1. **Independent impact roots:** **{len(roots)}** (gen {answers['Q1_n_accepted_gen']} + reuse {answers['Q1_n_reused_hard']})
2. **Meaningful lateral/angular/uni/loss:** lat {n_lat}, ang {n_ang}, uni {n_uni}, loss {n_loss}, high-motion {n_hm}
3. **Challenging ICs recoverable (FULL):** **{n_full}** / {len(search_list)} searched
4. **Stable bilateral (via FULL experts):** **{n_full}**
5. **Need Sliding/Pivoting after Recatch:** **{n_need_sp}**
6. **Full SUCCESS_NEW:** **{n_full}** {answers['Q6_full_ids']}
7. **Newly vs simple baselines:** **{n_new}** {answers['Q7_newly_ids']}
8. **Near-horizontal recovered to SUCCESS_NEW:** **{n_horiz_full}**
9. **Main failures:** primary `{answers['Q9_primary']}` — {answers['Q9_failure_reasons']}
10. **Added independent-root coverage:** **{n_full}** ball-impact FULL (prior T1≈{n1}, T2={n2}, T3-lost≈{n3})
11. **Ready for unified SFT?** **{answers['Q11_ready_unified_SFT']}** — {answers['Q11_note']}

## Artifacts

`BALL_IMPACT_ROOTS.csv`, `IMPACT_PHYSICS_AUDIT.csv`, `RECOVERY_ICS.csv`,  
`MATCHED_BASELINES.csv`, `DYNAMIC_RECATCH_SEARCH.csv`, `POST_RECATCH_MANIPULATION.csv`,  
`FULL_EXPERTS.csv`, `FAILURES.csv`, `ROOT_EPISODE_COVERAGE.csv`, `UNIFIED_MANIFEST.csv`,  
`DATASET_READINESS.md`, `trajectories/`, `snapshots/`, `raw_contact_traces/`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    ready = f"""# Dataset Readiness — Ball-Impact Dynamic Recovery

## Verdict: {"YES (conditional)" if answers["Q11_ready_unified_SFT"] else "NOT YET"}

- Ball-impact FULL experts: {n_full} independent roots
- Newly vs simple: {n_new}
- Combined prior + this: T1≈{n1}, T2={n2}, T3-lost≈{n3}, ball-impact={n_full}

### Merge rule
Append `FULL_EXPERTS.csv` / `UNIFIED_MANIFEST.csv` with `expert_type=BALL_IMPACT_FULL_EXPERT`.
Split by `root_episode` / `ic_id` (1:1 here). Mechanism tags are metadata only.
"""
    (OUT / "DATASET_READINESS.md").write_text(ready, encoding="utf-8")
    print("\nDONE", answers, flush=True)


if __name__ == "__main__":
    main()
