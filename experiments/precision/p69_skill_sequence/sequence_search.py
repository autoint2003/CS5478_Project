"""Phase 1–2: Sequence-level counterfactuals with IC discovery/validation split."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.paths import WINDOW
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.paths import (
    BEAM_WIDTH,
    MAX_EP_STEPS,
    N_CF_SEEDS,
    N_DISCOVERY_ICS,
    N_VALIDATION_ICS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    SEQ_LENS,
    STATE_DIR,
    N_DISC_STATES,
    N_VAL_STATES,
    T_MAX,
    TOP_K,
)
from experiments.precision.p69_skill_sequence.skill_families import (
    family_of_token,
    family_sequence,
    n_family_switches,
)


def _train_c_all() -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ]


def make_ic_split() -> dict:
    ics = _train_c_all()
    rng = np.random.default_rng(SEED + 69)
    ids = [s.traj_id for s in ics]
    rng.shuffle(ids)
    disc = ids[:N_DISCOVERY_ICS]
    val = ids[N_DISCOVERY_ICS : N_DISCOVERY_ICS + N_VALIDATION_ICS]
    held = ids[N_DISCOVERY_ICS + N_VALIDATION_ICS : N_DISCOVERY_ICS + N_VALIDATION_ICS + 8]
    split = {
        "discovery_ics": disc,
        "validation_ics": val,
        "heldout_train_ics": held,
        "seed": SEED + 69,
        "note": "No IC overlap across discovery / validation / held-out train.",
    }
    (OUT / "DISCOVERY_VALIDATION_SPLIT.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


@torch.no_grad()
def topk_tokens(actor, obs_seq, prev_seq, device, *, k: int) -> list[int]:
    obs = torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device, dtype=torch.float32)
    logits, _, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    probs = Categorical(logits=logit).probs.detach().cpu().numpy()
    return [int(i) for i in np.argsort(-probs)[:k]]


def continue_seq(env, actor, critic, start, state, device, *, seq: list[int] | None, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    geo0 = relative_geometry(env.env.sim)
    e0 = float(geo0["e_pos_mm"])
    th0 = float(geo0["theta_deg"])
    done = False
    info: dict = {}
    fi = 0
    ever_sw = ever_bil = near = r3 = False
    max_pad = 0
    while not done:
        if seq is not None and fi < len(seq):
            _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok = int(seq[fi])
            fi += 1
        else:
            tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        _r, done, info = env.step(tok, resid)
        g = relative_geometry(env.env.sim)
        ever_sw = ever_sw or g["correct_sidewall"]
        ever_bil = ever_bil or g["bilateral"]
        max_pad = max(max_pad, g["min_pad"])
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["frame_ok"] and ok_s > 0:
            near = True
        if preds["dwell_complete"] and preds["frame_ok"]:
            r3 = True
    e1 = float((info.get("metrics") or {}).get("e_pos_mm") or relative_geometry(env.env.sim)["e_pos_mm"])
    return {
        "e_pos0": e0,
        "theta0": th0,
        "delta_e_pos": e1 - e0,
        "correct_sidewall": int(ever_sw),
        "bilateral": int(ever_bil),
        "min_pad_ge4": int(max_pad >= 4),
        "near_ready": int(near),
        "r3_ready": int(r3 or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
    }


def mine_precontact_states(env, actor, critic, starts, device, *, per_ic: int, target: int, tag: str) -> list:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for st in starts:
        if len(out) >= target:
            break
        got = 0
        for i in range(per_ic * 2):
            if got >= per_ic or len(out) >= target:
                break
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=91000 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ensure_inner_capture_buffers(env)
            done = False
            info = {}
            last_no_contact_state = None
            last_meta = None
            while not done:
                geo = relative_geometry(env.env.sim)
                if (
                    not geo["has_pad_contact"]
                    and env.t <= 14
                    and env.t + 16 <= env.max_steps
                ):
                    ensure_inner_capture_buffers(env)
                    last_no_contact_state = take_full_state(
                        env,
                        meta={
                            "traj_id": st.traj_id,
                            "t": int(env.t),
                            "e_pos_mm": geo["e_pos_mm"],
                            "tag": tag,
                        },
                    )
                    last_meta = {
                        "t": int(env.t),
                        "remaining": int(env.max_steps - env.t),
                        "e_pos_mm": float(geo["e_pos_mm"]),
                        "theta_deg": float(geo["theta_deg"]),
                    }
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                _r, done, info = env.step(tok, resid)
                # After first contact forms, commit the last no-contact snapshot (immediate pre-contact)
                if last_no_contact_state is not None and relative_geometry(env.env.sim)["has_pad_contact"]:
                    sid = f"{tag}_{len(out):04d}_{st.traj_id[-20:]}_t{last_meta['t']}"
                    path = STATE_DIR / f"{sid}.pkl"
                    path.write_bytes(pickle.dumps(last_no_contact_state))
                    out.append(
                        {
                            "id": sid,
                            "path": str(path),
                            "traj_id": st.traj_id,
                            "t": last_meta["t"],
                            "remaining": last_meta["remaining"],
                            "e_pos_mm": last_meta["e_pos_mm"],
                            "theta_deg": last_meta["theta_deg"],
                            "seed": seed,
                            "split": tag,
                        }
                    )
                    got += 1
                    break
            print(f"[mine-{tag}] {st.traj_id[-28:]} got={got} bank={len(out)}", flush=True)
    return out


def _proxy_after_seq(env, actor, critic, start, state, device, *, seq: list[int]) -> float:
    """Cheap beam guide: negative e_pos + sidewall after forced sequence (no Actor resume)."""
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    for tok in seq:
        _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        env.step(int(tok), resid)
        if env.t >= env.max_steps:
            break
    g = relative_geometry(env.env.sim)
    return -float(g["e_pos_mm"]) + 15.0 * float(g["correct_sidewall"]) + 5.0 * float(g["bilateral"])


def beam_search_sequences(
    env, actor, critic, book, start, state, device, *, lengths: tuple[int, ...], width: int, seeds: int
) -> list[dict]:
    """Bounded beam: expand with geometry proxy; fully score only finalists + persist."""
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    roots = topk_tokens(actor, env.obs_seq, env.prev_seq, device, k=TOP_K)
    candidates_final = []
    for L in lengths:
        beam = [[]]
        for depth in range(L):
            expanded = []
            for pref in beam:
                restore_full_state(env, state, start)
                ensure_inner_capture_buffers(env)
                for t_forced in pref:
                    _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                    env.step(int(t_forced), resid)
                    if env.t >= env.max_steps:
                        break
                toks = topk_tokens(actor, env.obs_seq, env.prev_seq, device, k=width)
                if pref:
                    toks = list(dict.fromkeys(toks + [pref[-1]]))
                for tok in toks[:width]:
                    seq = pref + [int(tok)]
                    proxy = _proxy_after_seq(env, actor, critic, start, state, device, seq=seq)
                    expanded.append((proxy, seq))
            expanded.sort(key=lambda x: -x[0])
            beam = [s for _, s in expanded[:width]]
        # Full closed-loop score for final beam + persist
        finals = list(beam)
        if roots:
            finals.append([roots[0]] * L)
        # high-prob Actor sequence (re-sample greedy at each step from original state)
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        greedy = []
        for _ in range(L):
            toks = topk_tokens(actor, env.obs_seq, env.prev_seq, device, k=1)
            greedy.append(toks[0])
            _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            env.step(int(toks[0]), resid)
        finals.append(greedy)

        seen = set()
        for seq in finals:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            stats = [
                continue_seq(
                    env, actor, critic, start, state, device,
                    seq=seq, seed=SEED + 7000 + si + 17 * len(seq) + seq[0],
                )
                for si in range(seeds)
            ]
            fams = family_sequence(book, seq)
            src = "persist_top1" if roots and seq == [roots[0]] * L else ("greedy_actor" if seq == greedy else "beam")
            candidates_final.append(
                {
                    "seq": seq,
                    "length": L,
                    "score": float(np.mean([s["r3_ready"] for s in stats]))
                    + 0.2 * float(np.mean([s["correct_sidewall"] for s in stats])),
                    "families": fams,
                    "n_switches": n_family_switches(fams),
                    "mean_r3": float(np.mean([s["r3_ready"] for s in stats])),
                    "mean_sw": float(np.mean([s["correct_sidewall"] for s in stats])),
                    "mean_delta_e": float(np.mean([s["delta_e_pos"] for s in stats])),
                    "mean_succ": float(np.mean([s["SUCCESS_RELAXED"] for s in stats])),
                    "source": src,
                }
            )
    candidates_final.sort(key=lambda x: -x["score"])
    return candidates_final


def build_deployable_rule(discovery_rows: list, book) -> dict:
    """Freeze a small hypothesis from discovery only.

    Rule (obs-available):
      IF C episode AND no tactile contact (obs/min_pad from metrics after reset path)
         AND policy_step < 8:
         execute fixed sequence H for up to |H| steps, else Actor.

    Sequence H = mode of best beam sequences of length 4 among discovery states
    where best beam beat baseline; fallback = persist top Actor token ×4 is
    NOT fixed — instead we store the most common concrete token sequence.
    """
    # Collect best seq length 4 from each discovery state
    seqs = []
    for row in discovery_rows:
        cands = [c for c in row["candidates"] if c["length"] == 4]
        if not cands:
            continue
        best = cands[0]
        if best["mean_r3"] > row["baseline"]["r3_ready"] + 0.05:
            seqs.append(tuple(best["seq"]))
    if not seqs:
        # fallback: most common persist length-4 from all bests
        for row in discovery_rows:
            if row["candidates"]:
                seqs.append(tuple(row["candidates"][0]["seq"][:4]))
    # mode
    cnt = Counter(seqs)
    best_seq = list(cnt.most_common(1)[0][0]) if cnt else [0, 0, 0, 0]
    fams = family_sequence(book, best_seq)
    rule = {
        "name": "fixed_precontact_seq_L4",
        "activation": {
            "skill": "C",
            "max_policy_step": 7,
            "require_no_pad_contact": True,
            "uses_only_obs_available_signals": True,
            "note": "no_pad_contact approximated via runtime metrics.min_pad==0 / bilateral false (also in tactile obs)",
        },
        "sequence": best_seq,
        "families": fams,
        "n_switches": n_family_switches(fams),
        "max_duration_steps": len(best_seq),
        "termination": "after sequence exhausted OR contact appears OR max_duration",
        "fallback": "unchanged P64 Actor",
        "discovery_support_n": len(seqs),
        "sequence_mode_count": int(cnt.most_common(1)[0][1]) if cnt else 0,
    }
    return rule


def run_phase12(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    split = make_ic_split()
    lookup = {s.traj_id: s for s in _train_c_all()}
    disc_starts = [lookup[t] for t in split["discovery_ics"] if t in lookup]
    val_starts = [lookup[t] for t in split["validation_ics"] if t in lookup]

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    print("=== Mine discovery states ===", flush=True)
    disc_states = mine_precontact_states(
        env, actor, critic, disc_starts, device_t, per_ic=4, target=N_DISC_STATES, tag="disc"
    )
    print("=== Mine validation states ===", flush=True)
    val_states = mine_precontact_states(
        env, actor, critic, val_starts, device_t, per_ic=3, target=N_VAL_STATES, tag="val"
    )

    def evaluate_bank(states: list, *, search: bool) -> list:
        rows = []
        for si, stmeta in enumerate(states):
            start = lookup[stmeta["traj_id"]]
            state = pickle.loads(Path(stmeta["path"]).read_bytes())
            # baseline
            base_stats = [
                continue_seq(env, actor, critic, start, state, device_t, seq=None, seed=SEED + 8000 + j)
                for j in range(N_CF_SEEDS)
            ]
            baseline = {
                "r3_ready": float(np.mean([s["r3_ready"] for s in base_stats])),
                "correct_sidewall": float(np.mean([s["correct_sidewall"] for s in base_stats])),
                "delta_e_pos": float(np.mean([s["delta_e_pos"] for s in base_stats])),
                "SUCCESS_RELAXED": float(np.mean([s["SUCCESS_RELAXED"] for s in base_stats])),
            }
            cands = []
            if search:
                cands = beam_search_sequences(
                    env, actor, critic, book, start, state, device_t,
                    lengths=SEQ_LENS, width=BEAM_WIDTH, seeds=N_CF_SEEDS,
                )
            rows.append(
                {
                    **stmeta,
                    "baseline": baseline,
                    "candidates": cands[:12],
                    "oracle_best": cands[0] if cands else None,
                }
            )
            ob = cands[0] if cands else None
            print(
                f"[search] {stmeta['split']} {si+1}/{len(states)} "
                f"base_r3={baseline['r3_ready']:.2f} "
                f"oracle_r3={None if not ob else ob['mean_r3']:.2f} "
                f"seq={None if not ob else ob['seq']}",
                flush=True,
            )
        return rows

    print("=== Discovery oracle search ===", flush=True)
    disc_rows = evaluate_bank(disc_states, search=True)
    rule = build_deployable_rule(disc_rows, book)
    (OUT / "FIXED_SEQUENCE_HYPOTHESIS.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    print("=== Validation: oracle upper bound + fixed rule ===", flush=True)
    val_rows = evaluate_bank(val_states, search=True)
    # Apply fixed rule (same seq) to validation
    fixed_results = []
    for stmeta, row in zip(val_states, val_rows):
        start = lookup[stmeta["traj_id"]]
        state = pickle.loads(Path(stmeta["path"]).read_bytes())
        stats = [
            continue_seq(
                env, actor, critic, start, state, device_t,
                seq=list(rule["sequence"]), seed=SEED + 8200 + j,
            )
            for j in range(N_CF_SEEDS)
        ]
        fixed_results.append(
            {
                "id": stmeta["id"],
                "baseline": row["baseline"],
                "oracle": row["oracle_best"],
                "fixed_rule": {
                    "r3_ready": float(np.mean([s["r3_ready"] for s in stats])),
                    "correct_sidewall": float(np.mean([s["correct_sidewall"] for s in stats])),
                    "delta_e_pos": float(np.mean([s["delta_e_pos"] for s in stats])),
                    "SUCCESS_RELAXED": float(np.mean([s["SUCCESS_RELAXED"] for s in stats])),
                    "near_ready": float(np.mean([s["near_ready"] for s in stats])),
                },
            }
        )

    # Aggregate
    def mean_key(rows, path):
        vals = []
        for r in rows:
            cur = r
            for p in path:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None and not isinstance(cur, dict):
                vals.append(float(cur))
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_discovery_states": len(disc_rows),
        "n_validation_states": len(val_rows),
        "discovery_baseline_r3": mean_key(disc_rows, ["baseline", "r3_ready"]),
        "discovery_oracle_r3": mean_key(disc_rows, ["oracle_best", "mean_r3"]),
        "validation_baseline_r3": mean_key(fixed_results, ["baseline", "r3_ready"]),
        "validation_oracle_r3": mean_key(
            [{"oracle_best": f["oracle"]} for f in fixed_results if f["oracle"]],
            ["oracle_best", "mean_r3"],
        ),
        "validation_fixed_rule_r3": mean_key(fixed_results, ["fixed_rule", "r3_ready"]),
        "validation_fixed_rule_sw": mean_key(fixed_results, ["fixed_rule", "correct_sidewall"]),
        "validation_baseline_sw": mean_key(fixed_results, ["baseline", "correct_sidewall"]),
        "rule": rule,
    }

    # Persistence analysis on discovery oracle successes
    persist_rows = []
    for row in disc_rows:
        ob = row.get("oracle_best")
        if not ob:
            continue
        persist_rows.append(
            {
                "seq": ob["seq"],
                "families": ob["families"],
                "n_switches": ob["n_switches"],
                "source": ob["source"],
                "mean_r3": ob["mean_r3"],
                "beat_baseline": ob["mean_r3"] > row["baseline"]["r3_ready"] + 0.05,
            }
        )
    sw_succ = [r["n_switches"] for r in persist_rows if r["beat_baseline"]]
    sw_all = [r["n_switches"] for r in persist_rows]
    fam_counter = Counter()
    for r in persist_rows:
        if r["beat_baseline"]:
            fam_counter[tuple(r["families"])] += 1
    persist_analysis = {
        "n_oracle_beats": int(sum(1 for r in persist_rows if r["beat_baseline"])),
        "mean_switches_success": float(np.mean(sw_succ)) if sw_succ else None,
        "mean_switches_all": float(np.mean(sw_all)) if sw_all else None,
        "frac_zero_switch_success": float(np.mean([s == 0 for s in sw_succ])) if sw_succ else None,
        "frac_persist_source": float(np.mean([r["source"] == "persist_top1" for r in persist_rows if r["beat_baseline"]])) if sw_succ else None,
        "top_family_patterns": [{"families": list(k), "n": v} for k, v in fam_counter.most_common(8)],
    }

    # Dominant mechanism guess
    if persist_analysis["frac_zero_switch_success"] is not None and persist_analysis["frac_zero_switch_success"] >= 0.5:
        mechanism = "A_stay_in_family"
    elif persist_analysis["mean_switches_success"] is not None and persist_analysis["mean_switches_success"] >= 1.0:
        mechanism = "B_ordered_transition"
    else:
        mechanism = "C_token_selection_within_skill"

    report = {
        "summary": summary,
        "persistence": persist_analysis,
        "dominant_mechanism_hypothesis": mechanism,
        "discovery_rows_slim": [
            {
                "id": r["id"],
                "e_pos_mm": r["e_pos_mm"],
                "baseline": r["baseline"],
                "oracle_best": r["oracle_best"],
            }
            for r in disc_rows
        ],
        "validation_fixed": fixed_results,
    }
    (OUT / "SEQUENCE_SEARCH_RESULTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "SKILL_PERSISTENCE_ANALYSIS.json").write_text(
        json.dumps({"persistence": persist_analysis, "mechanism": mechanism}, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    run_phase12()
