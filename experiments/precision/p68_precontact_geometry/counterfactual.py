"""Phase 3–5: M111 token counterfactuals + timing + expressivity."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p68_precontact_geometry.paths import (
    MAX_EP_STEPS,
    MULTI_STEP_DEPTH,
    MULTI_STEP_STATES,
    MULTI_STEP_WIDTH,
    N_CF_SEEDS,
    N_PHYS_TOKENS,
    N_SNAPSHOTS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    STATE_DIR,
    T_MAX,
    TOP_K_TOKENS,
)
from models.m111_checkpoint import MOTION_K, decode_m111_action


def _lookup():
    return {s.traj_id: s for s in build_p54_dataset(seed=SEED)["schedule"] if s.skill == "C"}


@torch.no_grad()
def actor_topk_tokens(actor, obs_seq, prev_seq, device, *, k: int) -> list[tuple[int, float]]:
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, res, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    dist = Categorical(logits=logit)
    probs = dist.probs.detach().cpu().numpy()
    idx = np.argsort(-probs)[:k]
    resid = res[-1].detach().cpu().numpy().astype(np.float32)
    return [(int(i), float(probs[i]), resid) for i in idx]


def physical_candidate_tokens(book: np.ndarray, geo: dict, *, n: int) -> list[int]:
    """Select M111 prototypes whose linear velocity aligns with hand→object approach + grip."""
    approach = np.asarray(geo.get("approach_unit") or [0, 0, 1], float)
    # book columns: vx,vy,vz,wx,wy,wz,grip (normalized)
    scores = []
    for i in range(book.shape[0]):
        v = np.asarray(book[i, :3], float)
        vn = np.linalg.norm(v) + 1e-9
        align = float(np.dot(v / vn, approach))
        # prefer moderate grip openness diversity
        grip = float(book[i, 6])
        scores.append((align - 0.15 * abs(grip), i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:n]]


def continue_from_state(
    env,
    actor,
    critic,
    start,
    state,
    device,
    *,
    force_tokens: list[int] | None = None,
    seed: int,
) -> dict:
    """Restore state; optionally force first len(force_tokens) actions (Actor residual), then Actor."""
    seed_all(seed, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    geo0 = relative_geometry(env.env.sim)
    done = False
    info: dict = {}
    ever_sw = False
    ever_bil = False
    max_min_pad = 0
    near = False
    r3 = False
    fi = 0
    steps = 0
    while not done:
        if force_tokens is not None and fi < len(force_tokens):
            # use Actor residual at this state for fair decode
            _tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok = int(force_tokens[fi])
            fi += 1
        else:
            tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        r, done, info = env.step(tok, resid)
        steps += 1
        geo = relative_geometry(env.env.sim)
        ever_sw = ever_sw or geo["correct_sidewall"]
        ever_bil = ever_bil or geo["bilateral"]
        max_min_pad = max(max_min_pad, geo["min_pad"])
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"] and ok_s > 0:
            near = True
        if preds["dwell_complete"] and preds["frame_ok"]:
            r3 = True
    geo1 = relative_geometry(env.env.sim) if not done else relative_geometry(env.env.sim)
    return {
        "e_pos0": geo0["e_pos_mm"],
        "theta0": geo0["theta_deg"],
        "e_pos_final": float((info.get("metrics") or {}).get("e_pos_mm") or geo1["e_pos_mm"]),
        "theta_final": float((info.get("metrics") or {}).get("theta") or geo1["theta_deg"]),
        "delta_e_pos": float((info.get("metrics") or {}).get("e_pos_mm") or geo1["e_pos_mm"]) - geo0["e_pos_mm"],
        "correct_sidewall": int(ever_sw),
        "bilateral": int(ever_bil),
        "max_min_pad": max_min_pad,
        "near_ready": int(near),
        "r3_ready": int(r3 or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "n_steps": steps,
        "forced_n": 0 if force_tokens is None else len(force_tokens),
    }


def select_snapshots(bank: list, *, n: int = N_SNAPSHOTS) -> list:
    """Diversify across timing windows (round-robin) from failing episodes."""
    rng = np.random.default_rng(SEED + 68)
    by_kind = defaultdict(list)
    for s in bank:
        if s["stages"].get("SUCCESS_RELAXED") or s["stages"].get("r3_ready"):
            if rng.random() > 0.15:
                continue
        by_kind[s["kind"]].append(s)
    order = [
        "early_approach",
        "pre_contact",
        "immediate_pre_contact",
        "at_first_contact",
        "post_first_contact",
    ]
    for kind in order:
        rng.shuffle(by_kind[kind])
    out = []
    # round-robin so Phase-5 timing comparison is valid
    while len(out) < n:
        progressed = False
        for kind in order:
            if by_kind[kind]:
                out.append(by_kind[kind].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out[:n]


def run_counterfactuals(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    bank_blob = json.loads((OUT / "SNAP_BANK.json").read_text(encoding="utf-8"))
    bank = bank_blob["snaps"]
    selected = select_snapshots(bank, n=N_SNAPSHOTS)
    lookup = _lookup()
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    # persist selected states
    manifest = []
    for i, s in enumerate(selected):
        # already on disk from mining
        path = Path(s["state_path"])
        manifest.append(
            {
                "idx": i,
                "path": str(path),
                "traj_id": s["traj_id"],
                "kind": s["kind"],
                "t": s["t"],
                "first_contact_t": s["first_contact_t"],
                "failure_classes": s["failure_classes"],
                "pre_e_pos": s["pre_e_pos"],
            }
        )
    (OUT / "COUNTERFACTUAL_DESIGN.json").write_text(
        json.dumps(
            {
                "n_snapshots": len(manifest),
                "top_k": TOP_K_TOKENS,
                "n_phys": N_PHYS_TOKENS,
                "n_cf_seeds": N_CF_SEEDS,
                "snapshots": manifest,
                "protocol": (
                    "Force 1 M111 token at restored state using Actor residual decode; "
                    "then unchanged Actor continuation. Matched seeds."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    by_kind = defaultdict(list)
    sim_steps = 0

    for mi, meta in enumerate(manifest):
        start = lookup[meta["traj_id"]]
        state = pickle.loads(Path(meta["path"]).read_bytes())
        # restoration check
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        q0 = np.asarray(env.env.sim.data.qpos, float).copy()
        restore_full_state(env, state, start)
        ok_restore = bool(np.allclose(q0, np.asarray(env.env.sim.data.qpos, float), atol=1e-7))

        restore_full_state(env, state, start)
        geo = relative_geometry(env.env.sim)
        topk = actor_topk_tokens(actor, env.obs_seq, env.prev_seq, device_t, k=TOP_K_TOKENS)
        phys = physical_candidate_tokens(book, geo, n=N_PHYS_TOKENS)
        baseline_tok = topk[0][0]
        candidates = []
        seen = set()
        for tok, p, resid in topk:
            if tok not in seen:
                candidates.append({"token": tok, "source": "topk", "prob": p})
                seen.add(tok)
        for tok in phys:
            if tok not in seen:
                candidates.append({"token": tok, "source": "physical_align", "prob": None})
                seen.add(tok)
        # always include baseline explicitly
        candidates = [{"token": baseline_tok, "source": "baseline_actor", "prob": topk[0][1]}] + [
            c for c in candidates if c["token"] != baseline_tok
        ]

        # baseline multi-seed
        base_stats = []
        for si in range(N_CF_SEEDS):
            out = continue_from_state(
                env, actor, critic, start, state, device_t,
                force_tokens=None,
                seed=SEED + 9000 + mi * 50 + si,
            )
            sim_steps += out["n_steps"]
            base_stats.append(out)

        def agg(stats):
            keys = (
                "delta_e_pos", "correct_sidewall", "bilateral", "max_min_pad",
                "near_ready", "r3_ready", "handoff", "SUCCESS_RELAXED",
            )
            return {k: float(np.mean([s[k] for s in stats])) for k in keys}

        base_agg = agg(base_stats)
        token_results = []
        for cand in candidates:
            stats = []
            for si in range(N_CF_SEEDS):
                out = continue_from_state(
                    env, actor, critic, start, state, device_t,
                    force_tokens=[cand["token"]],
                    seed=SEED + 9000 + mi * 50 + si,  # matched seeds
                )
                sim_steps += out["n_steps"]
                stats.append(out)
            a = agg(stats)
            a["token"] = cand["token"]
            a["source"] = cand["source"]
            a["prob"] = cand["prob"]
            a["delta_r3_vs_base"] = a["r3_ready"] - base_agg["r3_ready"]
            a["delta_sw_vs_base"] = a["correct_sidewall"] - base_agg["correct_sidewall"]
            a["delta_epos_vs_base"] = a["delta_e_pos"] - base_agg["delta_e_pos"]
            token_results.append(a)

        best = max(token_results, key=lambda x: (x["delta_r3_vs_base"], x["delta_sw_vs_base"], -x["delta_e_pos"]))
        row = {
            "idx": mi,
            "kind": meta["kind"],
            "traj_id": meta["traj_id"],
            "t": meta["t"],
            "restore_ok": ok_restore,
            "geo0_e_pos": geo["e_pos_mm"],
            "geo0_theta": geo["theta_deg"],
            "baseline": base_agg,
            "baseline_token": baseline_tok,
            "n_candidates": len(token_results),
            "best": best,
            "any_r3_improve": int(any(t["delta_r3_vs_base"] > 0.05 for t in token_results)),
            "any_sw_improve": int(any(t["delta_sw_vs_base"] > 0.05 for t in token_results)),
            "any_epos_improve": int(any(t["delta_epos_vs_base"] < -1.0 for t in token_results)),
            "token_results": token_results,
        }
        rows.append(row)
        by_kind[meta["kind"]].append(row)
        print(
            f"[cf] {meta['kind']} t={meta['t']} base_r3={base_agg['r3_ready']:.2f} "
            f"best_tok={best['token']} d_r3={best['delta_r3_vs_base']:.2f} "
            f"d_sw={best['delta_sw_vs_base']:.2f} improve_r3={row['any_r3_improve']}",
            flush=True,
        )

    # Multi-step bounded search on few states
    multi = []
    for row in rows[:MULTI_STEP_STATES]:
        meta = manifest[row["idx"]]
        start = lookup[meta["traj_id"]]
        state = pickle.loads(Path(meta["path"]).read_bytes())
        restore_full_state(env, state, start)
        # beam: sequences of length <= depth from topk
        topk = actor_topk_tokens(actor, env.obs_seq, env.prev_seq, device_t, k=MULTI_STEP_WIDTH)
        beam = [[]]
        best_seq = None
        best_score = -1e9
        for depth in range(1, MULTI_STEP_DEPTH + 1):
            new_beam = []
            for pref in beam:
                restore_full_state(env, state, start)
                # expand
                for tok, _p, _r in topk:
                    seq = pref + [tok]
                    stats = []
                    for si in range(N_CF_SEEDS):
                        out = continue_from_state(
                            env, actor, critic, start, state, device_t,
                            force_tokens=seq,
                            seed=SEED + 9100 + row["idx"] * 20 + si,
                        )
                        sim_steps += out["n_steps"]
                        stats.append(out)
                    score = float(np.mean([s["r3_ready"] for s in stats])) + 0.25 * float(
                        np.mean([s["correct_sidewall"] for s in stats])
                    )
                    new_beam.append((score, seq, stats))
                    if score > best_score:
                        best_score = score
                        best_seq = seq
            new_beam.sort(key=lambda x: -x[0])
            beam = [s for _, s, _ in new_beam[:MULTI_STEP_WIDTH]]
        multi.append(
            {
                "idx": row["idx"],
                "kind": row["kind"],
                "best_seq": best_seq,
                "best_score": best_score,
                "baseline_r3": row["baseline"]["r3_ready"],
                "single_best_r3": row["best"]["r3_ready"],
            }
        )
        print(f"[multi] idx={row['idx']} seq={best_seq} score={best_score:.2f}", flush=True)

    # Timing summary
    timing = {}
    for kind, rs in by_kind.items():
        timing[kind] = {
            "n": len(rs),
            "baseline_r3": float(np.mean([r["baseline"]["r3_ready"] for r in rs])),
            "best_r3": float(np.mean([r["best"]["r3_ready"] for r in rs])),
            "frac_any_r3_improve": float(np.mean([r["any_r3_improve"] for r in rs])),
            "frac_any_sw_improve": float(np.mean([r["any_sw_improve"] for r in rs])),
            "frac_any_epos_improve": float(np.mean([r["any_epos_improve"] for r in rs])),
            "baseline_sw": float(np.mean([r["baseline"]["correct_sidewall"] for r in rs])),
            "best_sw": float(np.mean([r["best"]["correct_sidewall"] for r in rs])),
        }

    # Expressivity verdict helpers
    n_r3 = sum(r["any_r3_improve"] for r in rows)
    n_sw = sum(r["any_sw_improve"] for r in rows)
    n_epos = sum(r["any_epos_improve"] for r in rows)
    multi_helps = sum(
        1
        for m in multi
        if m["best_score"] > m["baseline_r3"] + 0.05 and m["best_score"] > m["single_best_r3"] + 0.05
    )

    report = {
        "n_snapshots": len(rows),
        "sim_steps": sim_steps,
        "timing": timing,
        "overall": {
            "frac_any_r3_improve": n_r3 / max(1, len(rows)),
            "frac_any_sw_improve": n_sw / max(1, len(rows)),
            "frac_any_epos_improve": n_epos / max(1, len(rows)),
            "baseline_r3_mean": float(np.mean([r["baseline"]["r3_ready"] for r in rows])),
            "best_r3_mean": float(np.mean([r["best"]["r3_ready"] for r in rows])),
        },
        "multi_step": multi,
        "multi_step_helps_n": multi_helps,
        "rows": rows,
    }
    (OUT / "COUNTERFACTUAL_RESULTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_counterfactuals()
