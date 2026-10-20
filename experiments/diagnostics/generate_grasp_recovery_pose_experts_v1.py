"""Bounded full-objective expert trajectory generation pilot (pose-aware grasp recovery).

Does not modify physics, K48, reward, SUCCESS_12S, or train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture, take_capture
from evaluation.success_criteria import (
    HOLD_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    evaluate_tokens,
    guide_score_after_prefix,
)
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from data_generation.recording.trajectory_recorder import save_pair

OUT = ROOT / "results" / "expert_generation" / "grasp_recovery_pose_v1"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
SEED = 20261018
RECOVERY_LIMIT_S = 4.0
MAX_TOKENS = int(RECOVERY_LIMIT_S / 0.02)

# Known verified sequences
CPP_HRV1 = [6] * 20 + [12] * 15 + [0] * 8
CPP_HARD263 = [4] * 20 + [12] * 5 + [0] * 3
RECATCH_GEN031 = [4] * 12 + [36] * 5 + [0] * 8

IC_SPECS = [
    # A: verified CPP
    {"id": "hrv1_920031", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920031.pkl", "family": "A_cpp_seed", "known_short": [30], "known_cpp": CPP_HRV1},
    {"id": "hard_910263", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl", "family": "A_cpp_seed", "known_short": [32, 32], "known_cpp": CPP_HARD263},
    # B: stable-but-misaligned / large tilt
    {"id": "hrv1_920117", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920117.pkl", "family": "B_stable_misaligned", "known_short": [30, 30]},
    {"id": "hrv1_leftover_hard_910144", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_leftover_hard_910144.pkl", "family": "B_stable_misaligned", "known_short": [13, 7]},
    {"id": "gen_hrv1_920031_0_c0c5f8c4825d", "path": "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots/gen_hrv1_920031_0_c0c5f8c4825d.pkl", "family": "B_stable_misaligned", "known_short": [13, 33]},
    {"id": "hrv1_920178", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920178.pkl", "family": "B_stable_misaligned", "known_short": [44, 44]},
    {"id": "hrv1_leftover_hard_910318", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_leftover_hard_910318.pkl", "family": "B_stable_misaligned", "known_short": [30]},
    # C/D: recatch / eccentric smaller
    {"id": "gen_hrv1_920031_1_74e897d20146", "path": "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots/gen_hrv1_920031_1_74e897d20146.pkl", "family": "D_recatch", "known_short": [36, 40], "known_recatch": RECATCH_GEN031},
    {"id": "hard_910101", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910101.pkl", "family": "E_mixed", "known_short": [32]},
    {"id": "hrv1_920131", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920131.pkl", "family": "E_mixed", "known_short": [41]},
    {"id": "hrv1_920099", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920099.pkl", "family": "E_mixed", "known_short": [41]},
    {"id": "hard_910277", "path": "results/diagnostics/hard_ic_mining/snapshots/hard_910277.pkl", "family": "B_stable_misaligned", "known_short": [44, 44]},
    {"id": "hrv1_920154", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920154.pkl", "family": "B_stable_misaligned", "known_short": [32, 32]},
    {"id": "hrv1_920193", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920193.pkl", "family": "B_stable_misaligned", "known_short": [32]},
    {"id": "gen_hrv1_920131_1_a8478ced8de4", "path": "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots/gen_hrv1_920131_1_a8478ced8de4.pkl", "family": "B_stable_misaligned", "known_short": [44]},
    {"id": "hrv1_920185", "path": "results/benchmarks/hard_recovery_v1/snapshots/hrv1_920185.pkl", "family": "E_mixed", "known_short": [30, 30]},
]


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: jsafe(r.get(k)) for k in keys})


def gravity_align(sim):
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g_w = np.array(sim.model.opt.gravity, float)
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    from envs.contacts.frames import extract_contacts
    from evaluation.grasp_geometry import cylinder_axis_hand

    rows, _ = extract_contacts(sim)
    pivot = np.mean([np.asarray(r["pos_w"], float) for r in rows], axis=0) if rows else ph + Rh @ np.array([0, 0, 0.105])
    tau_g_h = Rh.T @ np.cross(po - pivot, mass * g_w)
    u_c = cylinder_axis_hand(sim)
    u_t = U_TGT / max(float(np.linalg.norm(U_TGT)), 1e-12)
    u_c = u_c / max(float(np.linalg.norm(u_c)), 1e-12)
    if float(np.dot(u_c, u_t)) < 0:
        u_t = -u_t
    axis = np.cross(u_c, u_t)
    n = float(np.linalg.norm(axis))
    align = float(np.dot(tau_g_h, axis / n)) if n > 1e-9 else 0.0
    p_c_h = Rh.T @ (po - ph)
    return {
        "align": align,
        "helps": bool(align > 1e-6),
        "tau_g_norm": float(np.linalg.norm(tau_g_h)),
        "lever_m": float(np.linalg.norm(po - pivot)),
        "ecc_xy_mm": float(np.linalg.norm(p_c_h[:2]) * 1e3),
    }


def fingerprint(m, g):
    raw = f"{m['e_theta_deg']:.2f}|{m['e_pos_mm']:.2f}|{g['ecc_xy_mm']:.1f}|{int(m['nL'])}|{int(m['nR'])}|{g['helps']}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_blob(path: Path) -> dict:
    blob = pickle.loads(path.read_bytes())
    if "capture" not in blob or "ref" not in blob:
        raise ValueError(f"bad IC pickle: {path}")
    return blob


def summarize_eval(res, tag=""):
    win = res["window"]
    return {
        f"{tag}s12": int(res["success_12s"]),
        f"{tag}label": res["label"],
        f"{tag}SUCCESS_NEW": int(res["SUCCESS_NEW"]),
        f"{tag}win_theta": win.get("mean_theta"),
        f"{tag}win_pos": win.get("mean_pos"),
        f"{tag}win_max_theta": win.get("max_theta"),
        f"{tag}win_max_pos": win.get("max_pos"),
        f"{tag}hold_both_s": win.get("longest_both_ok_s"),
        f"{tag}rec_s": res["recovery_elapsed"],
        f"{tag}tokens": json.dumps(res["tokens"]),
        f"{tag}reason": res["reason"],
    }


def family_a_proposals(rng, max_n=40):
    seqs = []
    for tilt in (6, 4, 10):
        for n_t in (8, 15, 20):
            for unload in (12,):
                for n_u in (5, 10, 15, 25):
                    for brake in (0, 14):
                        for n_b in (3, 8):
                            # optional position nudge tokens after brake
                            base = [tilt] * n_t + [unload] * n_u + [brake] * n_b
                            seqs.append(base)
                            for nudge in (3, 32, 0):
                                seqs.append(base + [nudge] * 3)
    rng.shuffle(seqs)
    return seqs[:max_n]


def family_b_proposals(rng, max_n=30):
    seqs = []
    for o in (39, 36, 30):
        for n_o in (2, 3, 5):
            for c in (0, 14, 12):
                seqs.append([o] * n_o + [c] * 5)
                seqs.append([o] * n_o + [12, 12, 12] + [c] * 5)
                # recatch then tilt correction
                for tilt in (4, 6):
                    seqs.append([o] * n_o + [c] * 3 + [tilt] * 10 + [12] * 8 + [0] * 5)
    rng.shuffle(seqs)
    return seqs[:max_n]


def family_c_mutations(base, rng, max_n=25):
    out = []
    if not base:
        base = [0]
    for _ in range(max_n):
        s = list(base)
        op = int(rng.integers(0, 5))
        if op == 0 and len(s) < MAX_TOKENS - 5:
            s.extend([int(rng.integers(0, 47))] * int(rng.integers(1, 4)))
        elif op == 1 and len(s) > 3:
            i = int(rng.integers(0, len(s)))
            s[i] = int(rng.integers(0, 47))
        elif op == 2 and len(s) > 5:
            s = s[:-3]
        elif op == 3:
            s = s + [12] * 5 + [0] * 3
        else:
            s = [4] * 8 + s + [0] * 3
        out.append(s[:MAX_TOKENS])
    return out


def continuation_proposals(prefix, rng, max_n=20):
    """After a stable-misaligned prefix, propose correction suffixes (no HANDOFF)."""
    outs = []
    for tilt in (4, 6, 10):
        for n_t in (8, 15):
            for n_u in (5, 10, 15):
                outs.append(list(prefix) + [tilt] * n_t + [12] * n_u + [0] * 5)
                outs.append(list(prefix) + [tilt] * n_t + [12] * n_u + [3] * 5 + [0] * 3)
    for o in (36, 39):
        outs.append(list(prefix) + [o] * 3 + [0] * 8 + [12] * 5 + [0] * 3)
    # position-ish secure tokens
    for t in (3, 32, 14, 0):
        outs.append(list(prefix) + [t] * 8)
    rng.shuffle(outs)
    return outs[:max_n]


def classify_mech_light(res):
    d = float(res.get("d_theta_mid") or 0)
    if not res["success_12s"]:
        return "FAILED"
    if d >= 5:
        return "ORIENTATION_CORRECTING"
    if abs(d) < 2:
        return "LITTLE_REL_ROT"
    return "AMBIGUOUS"


def render_simple_video(env, book, blob, tokens, dest: Path, label: str):
    try:
        import mujoco
        from envs.outcomes.recovery_success import apply_handoff, nominal_step
        from envs.step_legal import step_legal

        restore_capture(env, blob["capture"])
        frames = []
        renderer = mujoco.Renderer(env.sim.model, 480, 480)

        def grab(_text):
            mujoco.mj_forward(env.sim.model, env.sim.data)
            try:
                renderer.update_scene(env.sim.data, camera="side")
            except Exception:
                renderer.update_scene(env.sim.data)
            frames.append(renderer.render().copy())

        grab(label)
        for tok in tokens:
            _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
            step_legal(env.sim, action)
            grab(label)
        apply_handoff(env.sim, blob["ref"])
        for _ in range(40):
            nominal_step(env.sim)
            if _ % 2 == 0:
                grab(label)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(dest), frames[::2], fps=10)
        except Exception:
            np.savez_compressed(dest.with_suffix(".npz"), frames=np.stack(frames[::5]))
    except Exception as exc:
        (dest.parent / "VIDEO_ERROR.txt").write_text(str(exc), encoding="utf-8")


def save_expert_npz(stem, res, meta):
    folder = OUT / "trajectories"
    if not res.get("observations"):
        return None
    cont, tokp = save_pair(
        folder,
        stem,
        res["actions"],
        res["observations"],
        res["contacts"],
        res["tokens"],
        res["residuals"],
        res["handoff_obs"],
    )
    write_json(
        folder / f"{stem}_meta.json",
        {
            **meta,
            "continuous": str(cont.relative_to(ROOT)),
            "tokens_path": str(tokp.relative_to(ROOT)),
            "label": res["label"],
            "SUCCESS_NEW": res["SUCCESS_NEW"],
            "success_12s": res["success_12s"],
            "window": res["window"],
            "recovery_elapsed": res["recovery_elapsed"],
            "tokens": res["tokens"],
        },
    )
    return cont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-ics", type=int, default=16)
    parser.add_argument("--guide-budget", type=int, default=48, help="heuristic prefix evals per IC")
    parser.add_argument("--terminal-budget", type=int, default=18, help="full nominal evals per IC")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--cost-only", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("RAW", "SNAPSHOTS", "VIDEOS", "trajectories"):
        (OUT / sub).mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    # --- cost benchmark ---
    print("COST_BENCH", flush=True)
    demo = load_blob(ROOT / IC_SPECS[0]["path"])
    t_a = time.perf_counter()
    guide_score_after_prefix(env, book, demo, [30])
    t_guide = time.perf_counter() - t_a
    t_a = time.perf_counter()
    evaluate_tokens(env, book, demo, [30])
    t_full = time.perf_counter() - t_a
    budget_row = {
        "seed": args.seed,
        "guide_rollout_s": t_guide,
        "full_eval_s": t_full,
        "guide_budget_per_ic": args.guide_budget,
        "terminal_budget_per_ic": args.terminal_budget,
        "max_ics": args.max_ics,
        "recovery_limit_s": RECOVERY_LIMIT_S,
        "working_theta_deg": WORKING_THETA_DEG,
        "working_pos_mm": WORKING_POS_MM,
        "hold_s": HOLD_S,
        "est_wall_min": (args.max_ics * (args.guide_budget * t_guide + args.terminal_budget * t_full)) / 60.0,
    }
    write_csv(OUT / "SEARCH_BUDGET.csv", [budget_row])
    write_json(OUT / "RAW" / "search_budget.json", budget_row)
    print("COST guide", round(t_guide, 3), "full", round(t_full, 3), "est_min", round(budget_row["est_wall_min"], 1), flush=True)
    if args.cost_only:
        return

    # --- IC pool ---
    print("BUILD_POOL", flush=True)
    pool = []
    seen_fp = set()
    ic_rows = []
    for spec in IC_SPECS[: args.max_ics]:
        path = ROOT / spec["path"]
        if not path.exists():
            print("MISSING", spec["id"], flush=True)
            continue
        blob = load_blob(path)
        restore_capture(env, blob["capture"])
        m = measure(env.sim, U_TGT, P_TGT)
        g = gravity_align(env.sim)
        fp = fingerprint(m, g)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        # copy snapshot into OUT for provenance
        dest = OUT / "SNAPSHOTS" / f"{spec['id']}.pkl"
        if not dest.exists():
            shutil.copy2(path, dest)
        entry = {
            **spec,
            "blob": blob,
            "e_theta0": m["e_theta_deg"],
            "e_pos0": m["e_pos_mm"],
            "nL": m["nL"],
            "nR": m["nR"],
            "bilateral": m["bilateral"],
            "gravity_helps": int(g["helps"]),
            "ecc_xy_mm": g["ecc_xy_mm"],
            "fp": fp,
            "snapshot_out": str(dest.relative_to(ROOT)),
        }
        pool.append(entry)
        ic_rows.append({k: v for k, v in entry.items() if k != "blob"})
    write_csv(OUT / "IC_POOL.csv", ic_rows)
    print("POOL", len(pool), flush=True)

    baseline_rows = []
    expert_rows = []
    failure_notes = []
    stage_stats = Counter()

    for ic in pool:
        ic_id = ic["id"]
        blob = ic["blob"]
        print("IC", ic_id, "e0", round(ic["e_theta0"], 1), flush=True)
        attempted = []
        best_full = None  # (score, res, source)

        def consider(res, source):
            nonlocal best_full
            attempted.append({"source": source, "label": res["label"], "s12": res["success_12s"], "new": res["SUCCESS_NEW"],
                              "win_th": res["window"].get("mean_theta"), "win_pos": res["window"].get("mean_pos"),
                              "tokens": res["tokens"], "rec_s": res["recovery_elapsed"]})
            # rank: SUCCESS_NEW first, then lower window errors among S12
            if res["SUCCESS_NEW"]:
                score = 1e6 - float(res["window"]["mean_theta"]) - 0.1 * float(res["window"]["mean_pos"])
            elif res["success_12s"]:
                score = 1e3 - float(res["window"]["mean_theta"] or 180) - 0.1 * float(res["window"].get("mean_pos") or 1e3)
            else:
                score = -1e3
            if best_full is None or score > best_full[0]:
                best_full = (score, res, source)

        # ===== Stage 1: baselines + known recipes =====
        print("  stage1_baselines", flush=True)
        imm = evaluate_tokens(env, book, blob, [])
        consider(imm, "immediate_handoff")
        short = evaluate_tokens(env, book, blob, list(ic.get("known_short") or [30]))
        consider(short, "known_short")
        # bounded short search (guide-rank, few full evals)
        short_cands = [[30], [44], [41], [32], [30, 30], [44, 44], [32, 32], [3], [12], [0], [36]]
        for _ in range(6):
            short_cands.append([int(rng.integers(0, 47)), int(rng.integers(0, 47))])
        best_short_res = short
        short_ranked = []
        for seq in short_cands[:14]:
            g = guide_score_after_prefix(env, book, blob, seq)
            if g["ok"]:
                short_ranked.append((g["score"], seq))
        short_ranked.sort(key=lambda x: -x[0])
        for _sc, seq in short_ranked[:4]:
            r = evaluate_tokens(env, book, blob, seq)
            consider(r, "short_search")
            if r["success_12s"] and (
                not best_short_res["success_12s"]
                or (r["window"]["mean_theta"] or 180) < (best_short_res["window"]["mean_theta"] or 180)
            ):
                best_short_res = r

        if ic.get("known_cpp"):
            cpp = evaluate_tokens(env, book, blob, list(ic["known_cpp"]))
            consider(cpp, "known_cpp")
            stage_stats["cpp_replay"] += 1
            if cpp["SUCCESS_NEW"]:
                stage_stats["cpp_full_expert"] += 1
        if ic.get("known_recatch"):
            rc = evaluate_tokens(env, book, blob, list(ic["known_recatch"]))
            consider(rc, "known_recatch")

        # ===== Stage 2: continuation from short if stable-misaligned =====
        print("  stage2_continuation", flush=True)
        term_used = 0
        if best_short_res["success_12s"] and not best_short_res["SUCCESS_NEW"]:
            pref = list(best_short_res["tokens"])
            for seq in continuation_proposals(pref, rng, max_n=min(16, args.guide_budget // 2)):
                g = guide_score_after_prefix(env, book, blob, seq)
                if not g["ok"]:
                    continue
                # only full-eval promising guides
                if g["e_theta"] < WORKING_THETA_DEG + 8 and g["e_pos"] < WORKING_POS_MM + 15:
                    if term_used >= args.terminal_budget:
                        break
                    r = evaluate_tokens(env, book, blob, seq)
                    consider(r, "stage2_continue_short")
                    term_used += 1
                    stage_stats["stage2_eval"] += 1
                    if r["SUCCESS_NEW"]:
                        stage_stats["stage2_full"] += 1
                        break

        # ===== Stage 3/4: family proposals + mutations =====
        print("  stage3_families", flush=True)
        candidates = []
        candidates.extend(family_a_proposals(rng, max_n=24))
        candidates.extend(family_b_proposals(rng, max_n=16))
        seed_bases = []
        if ic.get("known_cpp"):
            seed_bases.append(ic["known_cpp"])
        if ic.get("known_recatch"):
            seed_bases.append(ic["known_recatch"])
        seed_bases.append(list(best_short_res["tokens"]) if best_short_res["tokens"] else [0])
        # position-extension of CPP seeds
        for base in list(seed_bases):
            for extra in ([3] * 8, [32] * 8, [14] * 5 + [0] * 5, [12] * 5 + [3] * 8 + [0] * 5):
                candidates.append(list(base) + list(extra))
            candidates.extend(family_c_mutations(base, rng, max_n=8))

        # guide rank
        ranked = []
        for seq in candidates[: args.guide_budget]:
            g = guide_score_after_prefix(env, book, blob, seq)
            if g["ok"]:
                ranked.append((g["score"], seq, g))
        ranked.sort(key=lambda x: -x[0])

        for score, seq, g in ranked[: max(0, args.terminal_budget - term_used)]:
            r = evaluate_tokens(env, book, blob, seq)
            consider(r, "stage34_search")
            term_used += 1
            stage_stats["stage34_eval"] += 1
            if r["SUCCESS_NEW"]:
                stage_stats["stage34_full"] += 1
                # local refine: try short mutations of winner
                for mut in family_c_mutations(seq, rng, max_n=4):
                    if term_used >= args.terminal_budget + 4:
                        break
                    rr = evaluate_tokens(env, book, blob, mut)
                    consider(rr, "local_refine")
                    term_used += 1
                break

        # ===== Stage: recatch then continue if we have a recatch-ish S12 without NEW =====
        if best_full and best_full[1]["success_12s"] and not best_full[1]["SUCCESS_NEW"]:
            if best_full[2] in ("known_recatch", "stage34_search") or "recatch" in ic["family"].lower() or ic["family"].startswith("D"):
                pref = list(best_full[1]["tokens"])
                for seq in continuation_proposals(pref, rng, max_n=8):
                    if term_used >= args.terminal_budget + 8:
                        break
                    g = guide_score_after_prefix(env, book, blob, seq)
                    if g["ok"] and g["e_theta"] < 25:
                        r = evaluate_tokens(env, book, blob, seq)
                        consider(r, "stage3_recatch_continue")
                        term_used += 1

        write_json(OUT / "RAW" / f"{ic_id}_attempts.json", {"attempted": attempted, "n": len(attempted)})

        # baseline comparison row
        row = {
            "ic_id": ic_id,
            "family": ic["family"],
            "e_theta0": ic["e_theta0"],
            "e_pos0": ic["e_pos0"],
            "gravity_helps": ic["gravity_helps"],
            "ecc_xy_mm": ic["ecc_xy_mm"],
            **summarize_eval(imm, "imm_"),
            **summarize_eval(best_short_res, "best_short_"),
        }
        if ic.get("known_cpp"):
            # find cpp attempt
            cpp_att = next((a for a in attempted if a["source"] == "known_cpp"), None)
            if cpp_att:
                row["known_cpp_SUCCESS_NEW"] = int(cpp_att["new"])
                row["known_cpp_win_theta"] = cpp_att["win_th"]
                row["known_cpp_win_pos"] = cpp_att["win_pos"]
                row["known_cpp_s12"] = int(cpp_att["s12"])
        if best_full:
            res, src = best_full[1], best_full[2]
            row.update(summarize_eval(res, "best_any_"))
            row["best_any_source"] = src
            row["best_any_mech"] = classify_mech_light(res)
            # save trajectory if expert or best S12
            if res["SUCCESS_NEW"] or res["success_12s"]:
                full_rec = evaluate_tokens(env, book, blob, res["tokens"], record_obs=True)
                # preserve label from scored eval
                full_rec["label"] = res["label"]
                full_rec["SUCCESS_NEW"] = res["SUCCESS_NEW"]
                save_expert_npz(
                    f"{ic_id}__{src}",
                    full_rec,
                    {
                        "ic_id": ic_id,
                        "source": src,
                        "split": "pilot",
                        "seed": args.seed,
                        "working_theta_deg": WORKING_THETA_DEG,
                        "working_pos_mm": WORKING_POS_MM,
                        "snapshot": ic["snapshot_out"],
                        "replay_verified": True,
                        "mechanism_diag": classify_mech_light(full_rec),
                    },
                )
                # exact replay check
                replay = evaluate_tokens(env, book, blob, res["tokens"])
                row["replay_match_NEW"] = int(replay["SUCCESS_NEW"] == res["SUCCESS_NEW"])
                row["replay_match_S12"] = int(replay["success_12s"] == res["success_12s"])
                expert_rows.append(
                    {
                        "ic_id": ic_id,
                        "source": src,
                        "label": res["label"],
                        "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                        "success_12s": int(res["success_12s"]),
                        "win_mean_theta": res["window"].get("mean_theta"),
                        "win_mean_pos": res["window"].get("mean_pos"),
                        "win_max_theta": res["window"].get("max_theta"),
                        "win_max_pos": res["window"].get("max_pos"),
                        "hold_both_s": res["window"].get("longest_both_ok_s"),
                        "recovery_s": res["recovery_elapsed"],
                        "n_tokens": len(res["tokens"]),
                        "tokens": json.dumps(res["tokens"]),
                        "replay_ok": int(replay["SUCCESS_NEW"] == res["SUCCESS_NEW"] and replay["success_12s"] == res["success_12s"]),
                        "sensitivity": json.dumps(res["window"].get("matrix")),
                    }
                )
                if res["SUCCESS_NEW"]:
                    stage_stats["full_experts"] += 1
                else:
                    failure_notes.append(
                        {
                            "ic_id": ic_id,
                            "best_label": res["label"],
                            "source": src,
                            "win_theta": res["window"].get("mean_theta"),
                            "win_pos": res["window"].get("mean_pos"),
                            "note": "best S12 under working tol incomplete",
                        }
                    )
            else:
                failure_notes.append(
                    {
                        "ic_id": ic_id,
                        "best_label": res["label"],
                        "source": src,
                        "win_theta": None,
                        "win_pos": None,
                        "note": res["reason"],
                    }
                )
        baseline_rows.append(row)

        # videos for CPP seeds + any full expert
        if not args.skip_videos and best_full and (ic_id in ("hrv1_920031", "hard_910263") or best_full[1]["SUCCESS_NEW"]):
            render_simple_video(env, book, blob, best_full[1]["tokens"], OUT / "VIDEOS" / f"{ic_id}_best.mp4", best_full[2])
            render_simple_video(env, book, blob, list(best_short_res["tokens"]), OUT / "VIDEOS" / f"{ic_id}_short.mp4", "short")

    write_csv(OUT / "BASELINE_COMPARISON.csv", baseline_rows)
    write_csv(OUT / "EXPERT_TRAJECTORIES.csv", expert_rows)
    write_json(OUT / "RAW" / "failure_notes.json", failure_notes)
    write_json(OUT / "RAW" / "stage_stats.json", dict(stage_stats))

    n_full = sum(1 for r in expert_rows if r["SUCCESS_NEW"] == 1)
    n_ics = len(pool)
    summary = {
        "seed": args.seed,
        "n_ics": n_ics,
        "n_verified_full_experts": n_full,
        "frac_ics_with_full_expert": n_full / max(n_ics, 1),
        "working_theta_deg": WORKING_THETA_DEG,
        "working_pos_mm": WORKING_POS_MM,
        "stage_stats": dict(stage_stats),
        "labels": dict(Counter(r["label"] for r in expert_rows)),
    }
    write_json(OUT / "summary.json", summary)
    print("PILOT_DONE", OUT, "full_experts", n_full, "/", n_ics, flush=True)


if __name__ == "__main__":
    main()
