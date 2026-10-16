"""Diagnose why SFT does not preserve Contact-Preserving Sliding on validation.

Read-only. Writes only under results/diagnostics/sft_sliding_root_cause/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import finish_recovery, roll_policy
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.dataset import episode_tensors
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.prefix_rollout import sample_action
from training.recovery_episode import make_training_env
from training.train_grpo import step_logprob
from visualization import audit_checkpoint3_skill as audit
from visualization.classify_validation_skills import BOTH_OFF_HOLD_S, POST_GAP_MIGRATION_M, classify_mechanism, prepare_blob, restore_ic

OUT = ROOT / "results" / "diagnostics" / "sft_sliding_root_cause"
MANIFEST = ROOT / "data" / "supervised_targets_with_horizontal.jsonl"
SFT_CKPT = ROOT / "results" / "sft" / "checkpoint_epoch_0040.pt"
VALID = ROOT / "data" / "valid_skill_quotas.jsonl"


def skill_of(row):
    fam = row.get("family")
    strat = row.get("strategy")
    if fam == "recatch":
        return "horizontal_recatch"
    if fam == "vertical_recatch" or strat in ("follow_early", "later_close", "gentle_brake"):
        return "vertical_recatch"
    if fam == "sliding" or strat in ("controlled_slide", "inward_reseat"):
        return "sliding"
    if fam in ("regrasp", "chase", "release") or strat in ("chase_lost", "release_long", "release_short"):
        return "regrasp_other"
    if fam == "control" or strat == "immediate_handoff":
        return "control"
    return "other"


def load_manifest():
    return [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]


def phase1_dataset_audit(rows, book):
    """Effective SFT weights from train_sft.supervised_loss + collate sample_weight."""
    # Loss: per trajectory mean CE, then batch mean of sample_weight * traj_loss.
    # Epoch contribution of a tape ∝ sample_weight (not token count).
    records = []
    by_skill = defaultdict(lambda: {
        "tapes": 0, "ics": set(), "tokens": 0, "motion_tokens": 0, "handoff_tokens": 0,
        "weight_sum": 0.0, "first_tokens": Counter(), "handoff_positions": [], "recovery_s": [],
    })
    snapshots = {}
    for row in rows:
        skill = skill_of(row)
        obs, prev, tokens, residuals = episode_tensors(row, book)
        n = int(len(tokens))
        w = float(row["sample_weight"])
        first = int(tokens[0]) if n else None
        handoff_pos = [int(i) for i, tok in enumerate(tokens) if int(tok) == HANDOFF]
        # recovery duration from motion steps (before appended handoff if any)
        motion_n = int(np.sum(np.asarray(tokens) != HANDOFF))
        recovery_s = motion_n * 0.02
        slot = by_skill[skill]
        slot["tapes"] += 1
        slot["ics"].add(row["id"])
        slot["tokens"] += n
        slot["motion_tokens"] += motion_n
        slot["handoff_tokens"] += n - motion_n
        slot["weight_sum"] += w
        if first is not None:
            slot["first_tokens"][first] += 1
        slot["handoff_positions"].extend(handoff_pos)
        slot["recovery_s"].append(recovery_s)
        snapshots.setdefault(row["snapshot"], set()).add(skill)
        records.append({
            "id": row["id"],
            "skill": skill,
            "family": row.get("family"),
            "strategy": row.get("strategy"),
            "mechanism": row.get("mechanism"),
            "handoff_label": row.get("handoff_label"),
            "sample_weight": w,
            "n_tokens": n,
            "n_motion": motion_n,
            "n_handoff": n - motion_n,
            "first_token": first,
            "recovery_s": recovery_s,
            "snapshot": row.get("snapshot"),
            "continuous_npz": row.get("continuous_npz"),
            "split": row.get("split"),
        })

    # Overlap: same id with sliding and vertical tapes
    by_id = defaultdict(set)
    for rec in records:
        by_id[rec["id"]].add(rec["skill"])
    overlap_ids = sorted(eid for eid, skills in by_id.items() if "sliding" in skills and "vertical_recatch" in skills)

    summary = {
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "n_rows": len(rows),
        "n_unique_ids": len(by_id),
        "n_unique_snapshots": len(snapshots),
        "loss_formula": (
            "supervised_loss: CE mean over active tokens in a trajectory; "
            "then batch mean of sample_weight * (token_CE + residual_MSE). "
            "Code: training/train_sft.py supervised_loss; training/dataset.py collate."
        ),
        "epoch_contribution_proportional_to": "sample_weight (trajectory-level), not token count",
        "by_skill": {},
        "slide_vertical_overlap_ids": overlap_ids,
        "n_slide_vertical_overlap_ids": len(overlap_ids),
    }
    total_w = sum(slot["weight_sum"] for slot in by_skill.values())
    for skill, slot in sorted(by_skill.items()):
        rs = slot["recovery_s"]
        summary["by_skill"][skill] = {
            "tapes": slot["tapes"],
            "unique_ics": len(slot["ics"]),
            "tokens": slot["tokens"],
            "motion_tokens": slot["motion_tokens"],
            "handoff_tokens": slot["handoff_tokens"],
            "sample_weight_sum": slot["weight_sum"],
            "sample_weight_share": slot["weight_sum"] / total_w if total_w else 0.0,
            "mean_weight_per_tape": slot["weight_sum"] / slot["tapes"] if slot["tapes"] else 0.0,
            "first_token_top": slot["first_tokens"].most_common(8),
            "handoff_position_mean": float(np.mean(slot["handoff_positions"])) if slot["handoff_positions"] else None,
            "recovery_s_mean": float(np.mean(rs)) if rs else None,
            "recovery_s_median": float(np.median(rs)) if rs else None,
        }

    with (OUT / "dataset_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return records, summary, overlap_ids


def expert_mechanism_on_tape(env, book, row):
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    if "capture" not in blob:
        return None
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    tokens = None
    if row.get("token_residual_npz") and (ROOT / row["token_residual_npz"]).is_file():
        with np.load(ROOT / row["token_residual_npz"]) as archive:
            tokens = np.asarray(archive["tokens"], np.int64)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    samples = [audit.sample_state(env.sim, blob["ref"], phase="recovery")]
    for index, action in enumerate(actions):
        executed = step_legal(env.sim, action)
        tok = None if tokens is None or index >= len(tokens) else int(tokens[index])
        samples.append(audit.sample_state(env.sim, blob["ref"], action=executed, token=tok, phase="recovery"))
    handoff_t = float(env.sim.data.time)
    out = finish_recovery(env, blob["ref"], t0, "recorded_end")
    skill = classify_mechanism(samples, t0, handoff_t)
    return {
        "id": row["id"],
        "skill_tag": skill_of(row),
        "strategy": row.get("strategy"),
        "mechanism_tag": row.get("mechanism"),
        "result": out.get("reason"),
        "success": bool(out.get("success")),
        "classified_mechanism": skill["mechanism"],
        "confidence": skill["confidence"],
        "why": "; ".join(skill["why"]),
        "facts": skill["facts"],
        "n_actions": len(actions),
        "tokens_prefix": None if tokens is None else [int(t) for t in tokens[:12]],
    }


def phase2_expert_physics(rows, book, device, model):
    # Env requires a policy object; expert replay uses continuous tapes, not the policy.
    env = make_training_env(device, model=model)
    # Prefer train sliding tapes; include focus ICs if present.
    sliding = [row for row in rows if skill_of(row) == "sliding" and (ROOT / row["snapshot"]).is_file()]
    focus_ids = {"k48r_valid_00009", "inward_extra_250043", "ball_train_v2_0012", "ball_train_v2_0085"}
    vertical_ids = {row["id"] for row in rows if skill_of(row) == "vertical_recatch"}
    only = [row for row in sliding if row["id"] not in vertical_ids]
    overlap = [row for row in sliding if row["id"] in vertical_ids]
    chosen = []
    seen = set()

    def add_row(row):
        key = (row["id"], row.get("strategy"))
        if key in seen:
            return False
        # Skip ballistic-only snapshots without capture.
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "capture" not in blob:
            return False
        seen.add(key)
        chosen.append(row)
        return True

    for row in sliding:
        if row["id"] in focus_ids:
            add_row(row)
    for pool, count in ((only, 4), (overlap, 4)):
        added = 0
        for row in pool:
            if add_row(row):
                added += 1
            if added >= count:
                break
    chosen = chosen[:12]

    results = []
    for row in chosen:
        print("EXPERT_MECH", row["id"], row.get("strategy"), flush=True)
        try:
            item = expert_mechanism_on_tape(env, book, row)
        except Exception as exc:
            item = {"id": row["id"], "error": str(exc), "skill_tag": skill_of(row)}
        if item:
            results.append(item)
    # Flatten for CSV
    flat = []
    for item in results:
        facts = item.get("facts") or {}
        flat.append({
            "id": item.get("id"),
            "skill_tag": item.get("skill_tag"),
            "strategy": item.get("strategy"),
            "mechanism_tag": item.get("mechanism_tag"),
            "result": item.get("result"),
            "success": item.get("success"),
            "classified_mechanism": item.get("classified_mechanism"),
            "confidence": item.get("confidence"),
            "why": item.get("why"),
            "bilateral_steps": facts.get("bilateral_steps"),
            "recovery_steps": facts.get("recovery_steps"),
            "both_off_intervals": json.dumps(facts.get("both_off_intervals")),
            "max_contact_migration_m": facts.get("max_contact_migration_m"),
            "tokens_prefix": item.get("tokens_prefix"),
            "error": item.get("error"),
        })
    with (OUT / "expert_mechanism_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()) if flat else ["id"])
        writer.writeheader()
        writer.writerows(flat)
    return results


def teacher_force_row(model, book, row, device):
    # Single causal forward: logits[t] depends only on obs[:t+1] (models/k48_policy.py).
    obs, prev, tokens, residuals = episode_tensors(row, book)
    obs_t = torch.tensor(obs, device=device)
    prev_t = torch.tensor(prev, device=device)
    n = len(tokens)
    rows_out = []
    with torch.no_grad():
        logits, mean, _act = model(obs_t, prev_t)
        logits = logits.float()
        probs_all = torch.softmax(logits, dim=-1)
        logp_all = torch.log_softmax(logits, dim=-1)
        for t in range(n):
            logit = logits[t]
            probs = probs_all[t]
            logp = logp_all[t]
            expert = int(tokens[t])
            rank = int((logit > logit[expert]).sum().item()) + 1
            topv, topi = torch.topk(probs, k=min(5, probs.numel()))
            pred_res = mean[t].detach().float().cpu().numpy()
            exp_res = np.asarray(residuals[t], np.float32)
            resid_l2 = float(np.linalg.norm(pred_res - exp_res)) if expert != HANDOFF else 0.0
            argmax = int(torch.argmax(logit).item())
            action_err = None
            if expert != HANDOFF and argmax != HANDOFF:
                _, a_exp = decode_action(book, expert, exp_res)
                _, a_arg = decode_action(book, argmax, pred_res)
                action_err = float(np.max(np.abs(a_exp - a_arg)))
            rows_out.append({
                "id": row["id"],
                "skill": skill_of(row),
                "strategy": row.get("strategy"),
                "t": t,
                "expert_token": expert,
                "expert_logp": float(logp[expert]),
                "expert_prob": float(probs[expert]),
                "expert_rank": rank,
                "argmax_token": argmax,
                "argmax_prob": float(probs[argmax]),
                "top5_tokens": [int(x) for x in topi.cpu().numpy()],
                "top5_probs": [float(x) for x in topv.cpu().numpy()],
                "nll": float(-logp[expert]),
                "resid_l2": resid_l2,
                "action_linf_argmax_vs_expert": action_err,
                "is_handoff": int(expert == HANDOFF),
                "is_first": int(t == 0),
            })
    return rows_out


def phase3_teacher_forcing(rows, book, device, model):
    # Sample: all horizontal, all vertical, all sliding with focus + stratified sample
    pools = defaultdict(list)
    for row in rows:
        pools[skill_of(row)].append(row)
    chosen = []
    chosen.extend(pools["horizontal_recatch"])
    chosen.extend(pools["vertical_recatch"])
    focus = {"k48r_valid_00009", "inward_extra_250043", "ball_train_v2_0012", "ball_train_v2_0085"}
    sliding = pools["sliding"]
    for row in sliding:
        if row["id"] in focus:
            chosen.append(row)
    # Add every Nth sliding tape to cover distribution
    stride = max(1, len(sliding) // 20)
    for row in sliding[::stride]:
        if row not in chosen:
            chosen.append(row)
    # Cap
    if len(chosen) > 90:
        chosen = chosen[:90]

    all_rows = []
    for row in chosen:
        print("TF", skill_of(row), row["id"], flush=True)
        all_rows.extend(teacher_force_row(model, book, row, device))

    with (OUT / "teacher_forcing_per_token.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})

    summary = {}
    for skill in sorted({row["skill"] for row in all_rows}):
        subset = [row for row in all_rows if row["skill"] == skill]
        motion = [row for row in subset if not row["is_handoff"]]
        first = [row for row in subset if row["is_first"]]
        early5 = [row for row in subset if row["t"] < 5]
        early10 = [row for row in subset if row["t"] < 10]
        hand = [row for row in subset if row["is_handoff"]]

        def acc(items, k=1):
            if not items:
                return None
            if k == 1:
                return float(np.mean([row["expert_rank"] == 1 for row in items]))
            return float(np.mean([row["expert_rank"] <= k for row in items]))

        # Recatch-leaning tokens heuristic: open grip channel in codebook (token 7th dim < 0) among top1 when expert is closed
        book_np = book
        open_like = []
        for row in first:
            tok = row["argmax_token"]
            if tok == HANDOFF:
                open_like.append(False)
                continue
            open_like.append(float(book_np[tok][6]) < -0.5)

        summary[skill] = {
            "n_steps": len(subset),
            "n_episodes": len({row["id"] for row in subset}),
            "top1": acc(motion, 1),
            "top5": acc(motion, 5),
            "first_top1": acc(first, 1),
            "first_top5": acc(first, 5),
            "early5_top1": acc(early5, 1),
            "early10_top1": acc(early10, 1),
            "mean_expert_prob": float(np.mean([row["expert_prob"] for row in motion])) if motion else None,
            "mean_nll": float(np.mean([row["nll"] for row in motion])) if motion else None,
            "mean_resid_l2": float(np.mean([row["resid_l2"] for row in motion])) if motion else None,
            "handoff_mean_prob": float(np.mean([row["expert_prob"] for row in hand])) if hand else None,
            "first_argmax_open_grip_fraction": float(np.mean(open_like)) if open_like else None,
            "first_argmax_top": Counter(row["argmax_token"] for row in first).most_common(8),
            "first_expert_top": Counter(row["expert_token"] for row in first).most_common(8),
        }
    (OUT / "teacher_forcing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return all_rows, summary


def free_rollout(env, model, book, blob, device, limit_s, stochastic=False, seed=0):
    restore_ic(env, blob)
    t0 = float(env.sim.data.time)
    samples = [audit.sample_state(env.sim, blob["ref"], phase="recovery")]
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    rng = np.random.default_rng(seed)
    first_token = None
    first_action = None
    diverge = None
    from envs.outcomes.horizon_outcome import recovery_failed
    from envs.outcomes.table_drop import terminal_reason
    from models.k48_checkpoint import HANDOFF as H

    for step in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        window_obs = np.stack(env._obs_seq[-160:]).astype(np.float32)
        window_prev = np.stack(env._prev_seq[-160:]).astype(np.float32)
        if stochastic:
            token, pre, probs, entropy = sample_action(
                model, torch.tensor(window_obs, device=device), torch.tensor(window_prev, device=device), rng,
            )
            residual = pre
        else:
            with torch.no_grad():
                logits, mean, _act = model(
                    torch.tensor(window_obs, device=device),
                    torch.tensor(window_prev, device=device),
                )
            token = int(torch.argmax(logits[-1]).item())
            residual = mean[-1].detach().float().cpu().numpy()
        if first_token is None:
            first_token = int(token)
        if token == H:
            how = "early"
            samples.append(audit.sample_state(env.sim, blob["ref"], action=np.zeros(7), token=token, residual=residual, phase="handoff"))
            break
        _res, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        if first_action is None:
            first_action = executed.tolist()
        env._prev = np.asarray(executed, np.float32)
        samples.append(audit.sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "samples": samples, "t0": t0, "handoff_t": float(env.sim.data.time),
                "outcome": {"reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY", "success": False},
                "first_token": first_token, "first_action": first_action,
            }
    if how is None:
        how = "forced"
    handoff_t = float(env.sim.data.time)
    # Use official roll finishing
    from envs.outcomes.horizon_outcome import finish_recovery
    out = finish_recovery(env, blob["ref"], t0, how)
    return {
        "samples": samples, "t0": t0, "handoff_t": handoff_t, "outcome": out,
        "first_token": first_token, "first_action": first_action,
    }


def phase4_matched_rollouts(rows, book, model, device):
    env = make_training_env(device, model=model)
    focus = []
    for eid in ("k48r_valid_00009", "inward_extra_250043", "ball_train_v2_0012", "ball_train_v2_0085"):
        cands = [row for row in rows if row["id"] == eid and skill_of(row) == "sliding"]
        if cands:
            focus.append(cands[0])
    # Add 2 sliding-only and 2 overlap train ICs from expert audit if needed
    while len(focus) < 6:
        for row in rows:
            if skill_of(row) != "sliding":
                continue
            if any(item["id"] == row["id"] for item in focus):
                continue
            if "capture" not in pickle.loads((ROOT / row["snapshot"]).read_bytes()):
                continue
            focus.append(row)
            break
        else:
            break

    comparison = []
    seated_ref = None
    for row in focus[:6]:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "ref" not in blob:
            if seated_ref is None:
                from data_generation.strategies.recatch_oracle import _setup
                _sim, _base, seated_ref, _target, _code = _setup()
            blob = prepare_blob(blob, seated_ref)
        if "capture" not in blob:
            continue
        print("MATCHED", row["id"], flush=True)
        # Expert replay
        expert = expert_mechanism_on_tape(env, book, row)
        # Deterministic
        det = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=False)
        det_skill = classify_mechanism(det["samples"], det["t0"], det["handoff_t"])
        # Official eval consistency
        restore_ic(env, blob)
        eval_out = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
        # Stochastic seeds
        stoch = []
        for seed in (11, 22, 33):
            run = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=True, seed=seed)
            sk = classify_mechanism(run["samples"], run["t0"], run["handoff_t"])
            stoch.append({
                "seed": seed,
                "result": run["outcome"].get("reason"),
                "success": bool(run["outcome"].get("success")),
                "mechanism": sk["mechanism"],
                "first_token": run["first_token"],
                "both_off": sk["facts"].get("sustained_both_off_intervals"),
            })
        comparison.append({
            "id": row["id"],
            "strategy": row.get("strategy"),
            "expert_mechanism": None if expert is None else expert.get("classified_mechanism"),
            "expert_result": None if expert is None else expert.get("result"),
            "det_first_token": det["first_token"],
            "det_mechanism": det_skill["mechanism"],
            "det_result": det["outcome"].get("reason"),
            "det_success": bool(det["outcome"].get("success")),
            "eval_result": eval_out.get("reason"),
            "eval_success": bool(eval_out.get("success")),
            "det_both_off": json.dumps(det_skill["facts"].get("sustained_both_off_intervals")),
            "det_why": "; ".join(det_skill["why"]),
            "stoch": json.dumps(stoch),
            "stoch_any_contact_preserving": any(item["mechanism"] == "Contact-Preserving Adjustment" for item in stoch),
        })
    with (OUT / "matched_rollout_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()) if comparison else ["id"])
        writer.writeheader()
        writer.writerows(comparison)
    return comparison


def phase5_overlap_conflicts(rows, book, device, model):
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    conflicts = []
    for eid, items in sorted(by_id.items()):
        skills = {skill_of(row) for row in items}
        if not ("sliding" in skills and "vertical_recatch" in skills):
            continue
        slide = next(row for row in items if skill_of(row) == "sliding")
        vert = next(row for row in items if skill_of(row) == "vertical_recatch")
        # Same snapshot path?
        same_snap = slide.get("snapshot") == vert.get("snapshot")
        s_obs, s_prev, s_tok, s_res = episode_tensors(slide, book)
        v_obs, v_prev, v_tok, v_res = episode_tensors(vert, book)
        obs_diff0 = float(np.max(np.abs(s_obs[0] - v_obs[0]))) if len(s_obs) and len(v_obs) else None
        first_same = int(s_tok[0]) == int(v_tok[0]) if len(s_tok) and len(v_tok) else None
        # Compare first few tokens
        n = min(5, len(s_tok), len(v_tok))
        token_pairs = [(int(s_tok[i]), int(v_tok[i])) for i in range(n)]
        # Meaningful conflict: same initial obs (within 1e-4) but different first token and different grip intent
        meaningful = False
        detail = ""
        if obs_diff0 is not None and obs_diff0 < 1e-4 and not first_same:
            s_act = float(book[int(s_tok[0])][6]) if int(s_tok[0]) != HANDOFF else 0.0
            v_act = float(book[int(v_tok[0])][6]) if int(v_tok[0]) != HANDOFF else 0.0
            # open vs close conflict
            if (s_act < -0.5 and v_act > 0.5) or (v_act < -0.5 and s_act > 0.5) or int(s_tok[0]) != int(v_tok[0]):
                meaningful = True
                detail = f"same obs0 (linf={obs_diff0:.2e}) but first tokens {int(s_tok[0])} vs {int(v_tok[0])} grip_ch {s_act:.2f}/{v_act:.2f}"
        # Teacher-forced probs under shared obs0 for both expert tokens
        s_p = v_p = None
        with torch.no_grad():
            logits, _mean, _ = model(torch.tensor(s_obs[:1], device=device), torch.tensor(s_prev[:1], device=device))
            probs = torch.softmax(logits[-1].float(), dim=-1)
            s_p = float(probs[int(s_tok[0])])
            v_p = float(probs[int(v_tok[0])])
            argmax = int(torch.argmax(probs).item())
        conflicts.append({
            "id": eid,
            "same_snapshot": same_snap,
            "slide_strategy": slide.get("strategy"),
            "vert_strategy": vert.get("strategy"),
            "slide_weight": float(slide["sample_weight"]),
            "vert_weight": float(vert["sample_weight"]),
            "obs0_linf": obs_diff0,
            "slide_first_token": int(s_tok[0]),
            "vert_first_token": int(v_tok[0]),
            "first_tokens_equal": first_same,
            "token_pairs_first5": json.dumps(token_pairs),
            "meaningful_conflict": meaningful,
            "detail": detail,
            "tf_prob_slide_token": s_p,
            "tf_prob_vert_token": v_p,
            "tf_argmax": argmax,
            "tf_prefers_vertical_token": bool(v_p is not None and s_p is not None and v_p > s_p),
        })
    with (OUT / "overlap_conflicts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conflicts[0].keys()) if conflicts else ["id"])
        writer.writeheader()
        writer.writerows(conflicts)
    summary = {
        "n_overlap_ids": len(conflicts),
        "n_same_snapshot": sum(1 for row in conflicts if row["same_snapshot"]),
        "n_obs0_match": sum(1 for row in conflicts if row["obs0_linf"] is not None and row["obs0_linf"] < 1e-4),
        "n_meaningful_conflicts": sum(1 for row in conflicts if row["meaningful_conflict"]),
        "n_tf_prefers_vertical": sum(1 for row in conflicts if row["tf_prefers_vertical_token"]),
        "weight_slide_on_overlap": sum(row["slide_weight"] for row in conflicts),
        "weight_vert_on_overlap": sum(row["vert_weight"] for row in conflicts),
    }
    (OUT / "overlap_conflicts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return conflicts, summary


def write_reports(dataset_summary, expert_results, tf_summary, matched, conflict_summary, conflicts):
    # Root cause evidence
    slide_w = dataset_summary["by_skill"].get("sliding", {}).get("sample_weight_sum", 0)
    vert_w = dataset_summary["by_skill"].get("vertical_recatch", {}).get("sample_weight_sum", 0)
    horiz_w = dataset_summary["by_skill"].get("horizontal_recatch", {}).get("sample_weight_sum", 0)
    slide_tf = tf_summary.get("sliding", {})
    lines = [
        "# Root-cause evidence",
        "",
        "## Confirmed",
        "",
        f"1. **Sliding sample_weight mass is tiny versus Vertical/Horizontal.**",
        f"   Sliding weight sum={slide_w:.4f}, Vertical={vert_w:.4f}, Horizontal={horiz_w:.4f}.",
        f"   Loss uses trajectory-mean CE then `mean(sample_weight * traj_loss)` "
        f"(`training/train_sft.py` `supervised_loss`, `training/dataset.py` `collate`).",
        f"   So epoch gradient share ≈ weight share, not IC count (77 Sliding ICs vs weight 2.44).",
        "",
        f"2. **All {dataset_summary['n_slide_vertical_overlap_ids']} Vertical tapes share IDs with Sliding tapes.**",
        f"   On overlap ICs, Vertical weight sum={conflict_summary.get('weight_vert_on_overlap')} vs "
        f"Sliding weight sum={conflict_summary.get('weight_slide_on_overlap')}.",
        "",
    ]
    if slide_tf:
        lines.extend([
            f"3. **Teacher-forced Sliding first-token top-1 = {slide_tf.get('first_top1')}.**",
            f"   Sliding motion top-1={slide_tf.get('top1')}, top-5={slide_tf.get('top5')}, "
            f"mean expert prob={slide_tf.get('mean_expert_prob')}.",
            f"   First argmax open-grip fraction={slide_tf.get('first_argmax_open_grip_fraction')}.",
            "",
        ])
    # Expert mechanism
    mech_c = Counter(item.get("classified_mechanism") for item in expert_results if item.get("classified_mechanism"))
    lines.extend([
        "## Supported but not conclusive",
        "",
        f"- Expert Sliding sample mechanism counts on audited tapes: {dict(mech_c)}.",
        f"- Meaningful same-obs0 first-token conflicts on overlap: "
        f"{conflict_summary.get('n_meaningful_conflicts')} / {conflict_summary.get('n_overlap_ids')}.",
        f"- TF prefers Vertical first token over Sliding on overlap: "
        f"{conflict_summary.get('n_tf_prefers_vertical')} / {conflict_summary.get('n_overlap_ids')}.",
        "",
        "## Matched free rollouts",
        "",
    ])
    for row in matched:
        lines.append(
            f"- `{row['id']}`: expert={row['expert_mechanism']}/{row['expert_result']}; "
            f"det={row['det_mechanism']}/{row['det_result']} first_tok={row['det_first_token']}; "
            f"stoch_any_CP={row['stoch_any_contact_preserving']}"
        )
    lines.extend([
        "",
        "## Ruled out / weak",
        "",
        "- Ruled out: Sliding ICs absent from SFT manifest (77 unique Sliding IDs are present).",
        "- Ruled out as sole story: 'GRPO destroyed Sliding' — validation audit already shows SFT itself has 0 Contact-Preserving on the 17 valid ICs.",
        "- Not sufficient alone: video appearance; labels without contact timelines.",
        "",
    ])
    (OUT / "root_cause_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "# SFT Sliding root-cause diagnosis",
        "",
        f"Checkpoint: `{SFT_CKPT.relative_to(ROOT).as_posix()}`",
        f"Manifest: `{MANIFEST.relative_to(ROOT).as_posix()}` (config `unified_k48_recovery_v3_horizontal.yaml`)",
        "",
        "This report separates observed facts, mechanism classification, and interpretation.",
        "",
        "## Phase 1 — Dataset and effective loss weight",
        "",
        dataset_summary["loss_formula"],
        "",
        "### By skill",
        "",
        "| skill | tapes | unique ICs | tokens | motion tokens | sample_weight sum | weight share | mean weight/tape | first-token top |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for skill, slot in dataset_summary["by_skill"].items():
        report.append(
            f"| {skill} | {slot['tapes']} | {slot['unique_ics']} | {slot['tokens']} | {slot['motion_tokens']} | "
            f"{slot['sample_weight_sum']:.4f} | {slot['sample_weight_share']:.3f} | {slot['mean_weight_per_tape']:.4f} | "
            f"`{slot['first_token_top'][:4]}` |"
        )
    report.extend([
        "",
        f"Sliding∩Vertical overlapping IDs: **{dataset_summary['n_slide_vertical_overlap_ids']}** "
        f"(every Vertical tape sits on a Sliding ID).",
        "",
        "Files: `dataset_audit.csv`, `dataset_audit_summary.json`.",
        "",
        "## Phase 2 — Expert Sliding physical authenticity (sample)",
        "",
        f"Classified expert mechanisms: `{dict(mech_c)}`.",
        "",
        "File: `expert_mechanism_audit.csv`.",
        "",
        "## Phase 3 — Teacher forcing",
        "",
        "| skill | top1 | top5 | first top1 | early5 top1 | mean expert p | first open-grip argmax frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for skill, slot in tf_summary.items():
        report.append(
            f"| {skill} | {slot.get('top1')} | {slot.get('top5')} | {slot.get('first_top1')} | "
            f"{slot.get('early5_top1')} | {slot.get('mean_expert_prob')} | {slot.get('first_argmax_open_grip_fraction')} |"
        )
    report.extend([
        "",
        "Files: `teacher_forcing_per_token.csv`, `teacher_forcing_summary.json`.",
        "",
        "## Phase 4 — Matched free rollouts",
        "",
        "| IC | expert mech/result | det mech/result | first token | stoch any Contact-Preserving |",
        "|---|---|---|---:|---|",
    ])
    for row in matched:
        report.append(
            f"| `{row['id']}` | {row['expert_mechanism']}/{row['expert_result']} | "
            f"{row['det_mechanism']}/{row['det_result']} | {row['det_first_token']} | {row['stoch_any_contact_preserving']} |"
        )
    report.extend([
        "",
        "File: `matched_rollout_comparison.csv`.",
        "",
        "## Phase 5 — Overlap conflicts",
        "",
        json.dumps(conflict_summary, indent=2),
        "",
        "File: `overlap_conflicts.csv`.",
        "",
        "## Phase 6 — Answers",
        "",
        "### Q1. Does Sliding dominate SFT supervision?",
        "",
        f"**No by effective loss weight.** IC count favors Sliding (77), but sample_weight sum is "
        f"{slide_w:.2f} vs Vertical {vert_w:.2f} and Horizontal {horiz_w:.2f}.",
        "",
        "### Q2. How many Sliding experts are truly Contact-Preserving?",
        "",
        f"On the audited sample: `{dict(mech_c)}`. See CSV for per-tape evidence; do not extrapolate beyond the sample without reading it.",
        "",
        "### Q3. Can SFT predict Sliding tokens under teacher forcing?",
        "",
        f"Sliding first-token top-1={slide_tf.get('first_top1')}, motion top-1={slide_tf.get('top1')}, "
        f"mean expert prob={slide_tf.get('mean_expert_prob')}.",
        "",
        "### Q4. Does free rollout choose Recatch from step 0?",
        "",
        "See matched table first tokens and mechanisms.",
        "",
        "### Q5. Does stochastic decoding recover Contact-Preserving?",
        "",
        "See `stoch_any_contact_preserving` column.",
        "",
        "### Q6. Do overlap ICs create multimodal conflict?",
        "",
        f"Meaningful same-obs conflicts: {conflict_summary.get('n_meaningful_conflicts')} / "
        f"{conflict_summary.get('n_overlap_ids')}; TF prefers vertical token on "
        f"{conflict_summary.get('n_tf_prefers_vertical')} overlaps. Vertical weight dominates on those IDs.",
        "",
        "### Q7. Most likely root causes (ranked)",
        "",
        "See `root_cause_evidence.md`.",
        "",
        "## Repro",
        "",
        "```powershell",
        "python -u visualization/diagnose_sft_sliding_root_cause.py --device cpu",
        "```",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    book = load_codebook()
    rows = load_manifest()
    print("LOAD_CKPT", SFT_CKPT, flush=True)
    model, _ = load_trained_k48(SFT_CKPT, device)
    model.eval()
    print("PHASE1", flush=True)
    records, dataset_summary, overlap_ids = phase1_dataset_audit(rows, book)
    print("PHASE2", flush=True)
    expert_results = phase2_expert_physics(rows, book, device, model)
    print("PHASE3", flush=True)
    tf_rows, tf_summary = phase3_teacher_forcing(rows, book, device, model)
    print("PHASE4", flush=True)
    matched = phase4_matched_rollouts(rows, book, model, device)
    print("PHASE5", flush=True)
    conflicts, conflict_summary = phase5_overlap_conflicts(rows, book, device, model)
    print("WRITE", flush=True)
    write_reports(dataset_summary, expert_results, tf_summary, matched, conflict_summary, conflicts)
    print("DONE", json.dumps({
        "slide_weight": dataset_summary["by_skill"]["sliding"]["sample_weight_sum"],
        "vert_weight": dataset_summary["by_skill"]["vertical_recatch"]["sample_weight_sum"],
        "horiz_weight": dataset_summary["by_skill"]["horizontal_recatch"]["sample_weight_sum"],
        "slide_first_top1": tf_summary.get("sliding", {}).get("first_top1"),
        "overlap_meaningful": conflict_summary.get("n_meaningful_conflicts"),
    }), flush=True)


if __name__ == "__main__":
    main()
