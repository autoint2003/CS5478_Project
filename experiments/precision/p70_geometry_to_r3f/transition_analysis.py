"""Phase 3 — Per-token contribution of frozen sequence [56,82,50,71]."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.skill_families import family_of_token, family_sequence
from experiments.precision.p70_geometry_to_r3f.paths import (
    MAX_EP_STEPS,
    N_TRANSITION_STATES,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_OUT,
    P69_SEQ,
    P69_SPLIT,
    SEED,
    STATE_DIR,
    T_MAX,
)


def _lookup():
    blocked = blocked_eval_ids()
    return {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }


def _mine_or_reuse_states(env, actor, critic, device, n: int) -> list[dict]:
    """Reuse P69 disc states if present; else mine immediate pre-contact on discovery ICs."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p69_states = sorted((P69_OUT / "states").glob("disc_*.pkl")) if (P69_OUT / "states").exists() else []
    out = []
    lookup = _lookup()
    if p69_states:
        for path in p69_states[:n]:
            state = pickle.loads(path.read_bytes())
            meta = state.get("meta") or {}
            tid = meta.get("traj_id")
            if tid not in lookup:
                continue
            sid = path.stem
            dst = STATE_DIR / f"{sid}.pkl"
            if not dst.exists():
                dst.write_bytes(path.read_bytes())
            out.append({"id": sid, "path": str(dst), "traj_id": tid, "e_pos_mm": meta.get("e_pos_mm")})
        if out:
            return out[:n]

    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    starts = [lookup[t] for t in split["discovery_ics"] if t in lookup]
    for st in starts:
        if len(out) >= n:
            break
        for i in range(4):
            if len(out) >= n:
                break
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=98000 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ensure_inner_capture_buffers(env)
            last = None
            last_meta = None
            done = False
            while not done:
                g = relative_geometry(env.env.sim)
                if not g["has_pad_contact"] and env.t <= 14:
                    ensure_inner_capture_buffers(env)
                    last = take_full_state(env, meta={"traj_id": st.traj_id, "t": int(env.t), "e_pos_mm": g["e_pos_mm"]})
                    last_meta = {"t": int(env.t), "e_pos_mm": float(g["e_pos_mm"])}
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                _r, done, _info = env.step(tok, resid)
                if last is not None and relative_geometry(env.env.sim)["has_pad_contact"]:
                    sid = f"p70_{len(out):04d}_{st.traj_id[-16:]}_t{last_meta['t']}"
                    path = STATE_DIR / f"{sid}.pkl"
                    path.write_bytes(pickle.dumps(last))
                    out.append({"id": sid, "path": str(path), "traj_id": st.traj_id, "e_pos_mm": last_meta["e_pos_mm"]})
                    break
    return out


def decode_token(book: np.ndarray, tok: int) -> dict:
    a = np.asarray(book[int(tok)], float).reshape(7)
    return {
        "token": int(tok),
        "family": family_of_token(book, tok),
        "lin_norm": float(np.linalg.norm(a[:3])),
        "ang_norm": float(np.linalg.norm(a[3:6])),
        "grip": float(a[6]),
        "residual": a.tolist(),
    }


def analyze_sequence_on_state(env, actor, critic, book, start, state, device, seq: list[int]) -> dict:
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    g0 = relative_geometry(env.env.sim)
    obj0 = np.asarray(g0["obj_pos"], float)
    steps = []
    for i, tok in enumerate(seq):
        g_before = relative_geometry(env.env.sim)
        obj_b = np.asarray(g_before["obj_pos"], float)
        _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        env.step(int(tok), resid)
        g_after = relative_geometry(env.env.sim)
        obj_a = np.asarray(g_after["obj_pos"], float)
        steps.append(
            {
                "i": i,
                "token": int(tok),
                "family": family_of_token(book, tok),
                "decode": decode_token(book, tok),
                "e_before": float(g_before["e_pos_mm"]),
                "e_after": float(g_after["e_pos_mm"]),
                "delta_e": float(g_after["e_pos_mm"] - g_before["e_pos_mm"]),
                "theta_before": float(g_before["theta_deg"]),
                "theta_after": float(g_after["theta_deg"]),
                "delta_theta": float(g_after["theta_deg"] - g_before["theta_deg"]),
                "contact_before": bool(g_before["has_pad_contact"]),
                "contact_after": bool(g_after["has_pad_contact"]),
                "first_contact_this_step": bool(not g_before["has_pad_contact"] and g_after["has_pad_contact"]),
                "sidewall_after": bool(g_after["correct_sidewall"]),
                "min_pad_after": int(g_after["min_pad"]),
                "obj_disp_m": float(np.linalg.norm(obj_a - obj_b)),
                "obj_disp_from_start_m": float(np.linalg.norm(obj_a - obj0)),
            }
        )
        if env.t >= env.max_steps:
            break
    return {
        "e0": float(g0["e_pos_mm"]),
        "theta0": float(g0["theta_deg"]),
        "steps": steps,
        "families": family_sequence(book, seq),
    }


def run_phase3(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = _lookup()
    states = _mine_or_reuse_states(env, actor, critic, device_t, N_TRANSITION_STATES)
    print(f"[p3] transition states={len(states)} seq={P69_SEQ}", flush=True)

    token_meta = [decode_token(book, t) for t in P69_SEQ]
    rows = []
    for sm in states:
        start = lookup[sm["traj_id"]]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        row = analyze_sequence_on_state(env, actor, critic, book, start, state, device_t, P69_SEQ)
        rows.append({"id": sm["id"], "traj_id": sm["traj_id"], **row})
        print(
            f"[p3] {sm['id']} e0={row['e0']:.1f} "
            + " | ".join(f"{s['family']}:{s['delta_e']:+.1f}" for s in row["steps"]),
            flush=True,
        )

    # Aggregate per index
    per_idx = []
    for i in range(len(P69_SEQ)):
        deltas_e = [r["steps"][i]["delta_e"] for r in rows if i < len(r["steps"])]
        deltas_th = [r["steps"][i]["delta_theta"] for r in rows if i < len(r["steps"])]
        fc = [r["steps"][i]["first_contact_this_step"] for r in rows if i < len(r["steps"])]
        sw = [r["steps"][i]["sidewall_after"] for r in rows if i < len(r["steps"])]
        pads = [r["steps"][i]["min_pad_after"] for r in rows if i < len(r["steps"])]
        disp = [r["steps"][i]["obj_disp_m"] for r in rows if i < len(r["steps"])]
        e_after = [r["steps"][i]["e_after"] for r in rows if i < len(r["steps"])]
        per_idx.append(
            {
                "i": i,
                "token": P69_SEQ[i],
                "family": token_meta[i]["family"],
                "decode": token_meta[i],
                "mean_delta_e": float(np.mean(deltas_e)) if deltas_e else None,
                "mean_delta_theta": float(np.mean(deltas_th)) if deltas_th else None,
                "frac_first_contact": float(np.mean(fc)) if fc else None,
                "frac_sidewall": float(np.mean(sw)) if sw else None,
                "mean_min_pad": float(np.mean(pads)) if pads else None,
                "mean_obj_disp_m": float(np.mean(disp)) if disp else None,
                "mean_e_after": float(np.mean(e_after)) if e_after else None,
                "frac_e_after_le25": float(np.mean([e <= 25 for e in e_after])) if e_after else None,
                "frac_e_after_le9": float(np.mean([e <= 9 for e in e_after])) if e_after else None,
            }
        )

    # Failure mode hypotheses
    # 1. APPROACH too early: index0 improves e but still high
    # 2. GRIP before aligned: index1 (GRIP) when e_before still large
    grip_e_before = []
    approach_to_grip_loss = []
    for r in rows:
        if len(r["steps"]) < 2:
            continue
        s0, s1 = r["steps"][0], r["steps"][1]
        grip_e_before.append(s1["e_before"])
        if s0["family"] == "APPROACH" and s1["family"] == "GRIP":
            approach_to_grip_loss.append(s1["delta_e"])

    fail_modes = {
        "mean_e_before_GRIP": float(np.mean(grip_e_before)) if grip_e_before else None,
        "frac_GRIP_while_e_gt25": float(np.mean([e > 25 for e in grip_e_before])) if grip_e_before else None,
        "frac_GRIP_while_e_gt9": float(np.mean([e > 9 for e in grip_e_before])) if grip_e_before else None,
        "mean_delta_e_on_APPROACH_to_GRIP": float(np.mean(approach_to_grip_loss)) if approach_to_grip_loss else None,
        "frac_e_worsens_on_GRIP": float(np.mean([d > 0 for d in approach_to_grip_loss])) if approach_to_grip_loss else None,
        "final_mean_e": float(np.mean([r["steps"][-1]["e_after"] for r in rows if r["steps"]])) if rows else None,
        "hypotheses": [],
    }
    if fail_modes["frac_GRIP_while_e_gt25"] and fail_modes["frac_GRIP_while_e_gt25"] >= 0.5:
        fail_modes["hypotheses"].append("2_GRIP_before_alignment_sufficient")
    if fail_modes["frac_e_worsens_on_GRIP"] and fail_modes["frac_e_worsens_on_GRIP"] >= 0.4:
        fail_modes["hypotheses"].append("3_alignment_lost_during_APPROACH_to_GRIP")
    if per_idx and per_idx[0]["mean_e_after"] and per_idx[0]["mean_e_after"] > 50:
        fail_modes["hypotheses"].append("1_APPROACH_terminates_too_early_or_weak")
    if fail_modes["final_mean_e"] and fail_modes["final_mean_e"] > 25:
        fail_modes["hypotheses"].append("4_sequence_duration_too_short_for_threshold_cross")
    # diversity: large variance of e0
    e0s = [r["e0"] for r in rows]
    if e0s and float(np.std(e0s)) > 40:
        fail_modes["hypotheses"].append("5_sequence_inappropriate_for_some_ICs")

    report = {
        "sequence": P69_SEQ,
        "token_meta": token_meta,
        "n_states": len(rows),
        "per_index": per_idx,
        "fail_modes": fail_modes,
        "rows_slim": [
            {
                "id": r["id"],
                "e0": r["e0"],
                "steps": [
                    {
                        "tok": s["token"],
                        "fam": s["family"],
                        "de": s["delta_e"],
                        "dth": s["delta_theta"],
                        "fc": s["first_contact_this_step"],
                        "e_after": s["e_after"],
                        "pad": s["min_pad_after"],
                    }
                    for s in r["steps"]
                ],
            }
            for r in rows
        ],
    }
    (OUT / "SEQUENCE_TRANSITION_ANALYSIS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_phase3()
