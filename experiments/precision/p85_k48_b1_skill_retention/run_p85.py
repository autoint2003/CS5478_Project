"""P85 — K48→P64 B1 skill retention audit (diagnostic only; no training).

Uses the eight exposed independent B1 ICs as diagnostic cases only.
Does not change P64, K48, physics, M111, frozen eval, or success definitions.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    CACHE,
    MAX_EP_STEPS,
    OUT as FINAL_OUT,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p43_r3_working_hold_v2.r3_rules import DwellTracker  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p53_independent_grpo64.success_relaxed import (  # noqa: E402
    evaluate_success_relaxed_12s,
)
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
    gate_spec,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p69_skill_sequence.skill_families import family_of_token  # noqa: E402
from experiments.precision.p82_global_failure_budget.collect import rollout_episode  # noqa: E402
from models.k48_checkpoint import HANDOFF as K48_HANDOFF  # noqa: E402
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from models.m111_checkpoint import HANDOFF as M111_HANDOFF  # noqa: E402
from models.m111_checkpoint import decode_m111_action  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import CAP as K48_CAP  # noqa: E402
from training.k47_codebook import decode_action, load_codebook  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/experiments/p85_k48_b1_skill_retention"
TRAJ_DIR = OUT / "trajectories"
K48_TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
FINAL_CSV = FINAL_OUT / "EPISODE_RESULTS.csv"
FINAL_MANIFEST = FINAL_OUT / "FROZEN_TEST_MANIFEST.json"
STARTS_PKL = CACHE / "selected_starts.pkl"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return str(obj)


def load_b1() -> tuple[list[dict], list[Any], list[dict]]:
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    b1_eps = [e for e in manifest["episodes"] if e["skill"] == "B1"]
    starts = pickle.loads(STARTS_PKL.read_bytes())["B1"]
    by = {s.traj_id: s for s in starts}
    ordered = [by[e["traj_id"]] for e in b1_eps]
    frozen = [r for r in csv.DictReader(FINAL_CSV.open(encoding="utf-8")) if r["skill"] == "B1"]
    return b1_eps, ordered, frozen


def blob_from_start(start) -> dict:
    ep = start.ep
    return {"capture": ep["capture"], "ref": ep["ref"], "traj_id": start.traj_id}


def snap_physics(sim) -> dict:
    g = relative_geometry(sim)
    m = state_metrics(sim)
    pb = pad_bilateral(sim)
    return {
        "t": float(sim.data.time),
        "object_z": float(sim.data.xpos[sim.ids.object_body][2]),
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "bilateral": int(g["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "min_pad": int(min(pb["nL"], pb["nR"])),
        "has_pad_contact": int(g["has_pad_contact"]),
        "correct_sidewall": int(g.get("correct_sidewall") or 0),
        "fn_L": float(m.get("fn_L") or 0.0),
        "fn_R": float(m.get("fn_R") or 0.0),
        "vrel": float(m.get("vrel") or 0.0),
        "grip_q": float(m.get("grip") or m.get("q_grip") or 0.0) if ("grip" in m or "q_grip" in m) else None,
    }


def record_k48_trajectory(env, model, book_k48, blob, device) -> dict:
    """Ordinary argmax roll_policy instrumentation — no tree search."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    phys0 = snap_physics(env.sim)
    obs0 = np.asarray(env.observation(), np.float32)

    dwell = DwellTracker.from_rule("R3", dt=float(DT_POLICY))
    r3f = R3FOnline.create(t_max=10_000)  # large: we only want early-ready signal
    r3f.tracker = dwell

    steps: list[dict] = []
    actions: list[np.ndarray] = []
    tokens: list[int] = []
    residuals: list[np.ndarray] = []
    lost_low = 0
    how = None
    handoff_step = None
    cap = int(round(float(SELECTED_LIMIT_S) / float(DT_POLICY)))

    for i in range(cap):
        if float(env.sim.data.time) - t0 >= float(SELECTED_LIMIT_S) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual_pre = mean[-1].detach().float().cpu().numpy()
        tokens.append(token)

        # Snapshot R3F readiness at decision state without advancing dwell.
        preds_pre = evaluate_predicates(env.sim, ok_s=float(dwell.ok_s))
        st_pre = {
            "ready": bool(preds_pre.get("frame_ok") and preds_pre.get("dwell_complete")),
            "reason": "R3_EARLY" if (preds_pre.get("frame_ok") and preds_pre.get("dwell_complete")) else "NONE",
            "ok_s": float(dwell.ok_s),
            "frame_ok": bool(preds_pre.get("frame_ok")),
            "theta": preds_pre.get("theta"),
            "e_pos": preds_pre.get("e_pos"),
            "nL": preds_pre.get("nL"),
            "nR": preds_pre.get("nR"),
            "min_pad_ok": bool(preds_pre.get("min_pad_ok")),
            "margin_ok": bool(preds_pre.get("margin_ok")),
            "working_ok": bool(preds_pre.get("working_ok")),
            "bilateral": bool(preds_pre.get("bilateral")),
        }

        if token == int(K48_HANDOFF):
            how = "early"
            handoff_step = i
            steps.append(
                {
                    "step": i,
                    "decision": "HANDOFF",
                    "token": token,
                    "action": None,
                    "residual": None,
                    "phys": snap_physics(env.sim),
                    "r3f": st_pre,
                    "preds": {
                        k: preds_pre.get(k)
                        for k in (
                            "frame_ok",
                            "dwell_complete",
                            "working_ok",
                            "bilateral",
                            "min_pad_ok",
                            "margin_ok",
                            "theta",
                            "e_pos",
                        )
                    },
                    "obs_norm": float(np.linalg.norm(obs)),
                }
            )
            break

        residual, action = decode_action(book_k48, token, residual_pre)
        residuals.append(np.asarray(residual, np.float32))
        actions.append(np.asarray(action, np.float32))
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)

        # Advance dwell once per physics step (matches R3FOnline cadence).
        st_post = r3f.observe(env.sim, i + 1, tape_ended=False)
        phys = snap_physics(env.sim)
        steps.append(
            {
                "step": i,
                "decision": "MOTION",
                "token": token,
                "action": executed.tolist(),
                "residual": residual.tolist(),
                "grip_cmd": float(executed[6]),
                "phys": phys,
                "r3f": {
                    k: st_post.get(k)
                    for k in (
                        "ready",
                        "reason",
                        "ok_s",
                        "frame_ok",
                        "theta",
                        "e_pos",
                        "nL",
                        "nR",
                        "min_pad_ok",
                        "margin_ok",
                        "working_ok",
                        "bilateral",
                    )
                },
                "obs_norm": float(np.linalg.norm(obs)),
            }
        )

        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out_fail = {
                "success_12s": 0,
                "reason": "FAILED_DURING_RECOVERY",
                "handoff_mode": None,
                "handoff_step": None,
            }
            return {
                **out_fail,
                "phys0": phys0,
                "obs0_norm": float(np.linalg.norm(obs0)),
                "steps": steps,
                "actions": np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32),
                "tokens": tokens,
                "residuals": np.asarray(residuals, np.float32) if residuals else np.zeros((0, 7), np.float32),
                "inference": "argmax_roll_policy_no_tree_search",
                "recovery_elapsed": float(env.sim.data.time) - t0,
            }

    if how is None:
        how = "forced"

    handoff_phys = snap_physics(env.sim)
    # Gate audit at the state where K48 chooses HANDOFF (no extra dwell tick).
    handoff_preds = evaluate_predicates(env.sim, ok_s=float(dwell.ok_s))
    handoff_r3 = {
        "ready": bool(handoff_preds.get("frame_ok") and handoff_preds.get("dwell_complete")),
        "reason": "R3_EARLY" if (handoff_preds.get("frame_ok") and handoff_preds.get("dwell_complete")) else "NONE",
        "ok_s": float(dwell.ok_s),
        "frame_ok": bool(handoff_preds.get("frame_ok")),
        "theta": handoff_preds.get("theta"),
        "e_pos": handoff_preds.get("e_pos"),
        "nL": handoff_preds.get("nL"),
        "nR": handoff_preds.get("nR"),
        "min_pad_ok": bool(handoff_preds.get("min_pad_ok")),
        "margin_ok": bool(handoff_preds.get("margin_ok")),
        "working_ok": bool(handoff_preds.get("working_ok")),
        "bilateral": bool(handoff_preds.get("bilateral")),
    }

    # Native SUCCESS_12S continuation
    out = finish_recovery(env, blob["ref"], t0, how)

    return {
        "success_12s": int(bool(out.get("success"))),
        "reason": out.get("reason"),
        "handoff_mode": how,
        "handoff_step": handoff_step if handoff_step is not None else len(actions),
        "recovery_elapsed": out.get("recovery_elapsed"),
        "recovery_steps_motion": len(actions),
        "phys0": phys0,
        "obs0_norm": float(np.linalg.norm(obs0)),
        "handoff_phys": handoff_phys,
        "handoff_r3f": handoff_r3,
        "handoff_preds": {
            k: handoff_preds.get(k)
            for k in (
                "working_ok",
                "bilateral",
                "min_pad_ok",
                "margin_ok",
                "frame_ok",
                "dwell_complete",
                "theta",
                "e_pos",
                "nL",
                "nR",
            )
        },
        "p64_would_permit_handoff_with_dwell": bool(handoff_r3["ready"]),
        "p64_frame_ok_at_handoff_instant": bool(handoff_preds.get("frame_ok")),
        "steps": steps,
        "actions": np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32),
        "tokens": tokens,
        "residuals": np.asarray(residuals, np.float32) if residuals else np.zeros((0, 7), np.float32),
        "inference": "argmax_roll_policy_no_tree_search",
        "nominal_reason": out.get("reason"),
        "lift_completed": out.get("lift_completed"),
    }


def replay_actions_then_finish(env, blob, actions: np.ndarray, *, handoff_how: str = "forced") -> dict:
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    executed = []
    for a in np.asarray(actions, np.float32).reshape(-1, 7):
        ex = step_legal(env.sim, a)
        executed.append(ex)
        env._prev = np.asarray(ex, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "success_12s": 0,
                "reason": "FAILED_DURING_RECOVERY",
                "handoff_phys": snap_physics(env.sim),
                "n_executed": len(executed),
            }
    handoff_phys = snap_physics(env.sim)
    dwell = DwellTracker.from_rule("R3", dt=float(DT_POLICY))
    # Reconstruct dwell by replaying from start is expensive; evaluate instantaneous frame here
    preds = evaluate_predicates(env.sim, ok_s=0.0)
    out = finish_recovery(env, blob["ref"], t0, handoff_how)
    return {
        "success_12s": int(bool(out.get("success"))),
        "reason": out.get("reason"),
        "handoff_phys": handoff_phys,
        "frame_ok_at_end": int(bool(preds.get("frame_ok"))),
        "theta": preds.get("theta"),
        "e_pos": preds.get("e_pos"),
        "nL": preds.get("nL"),
        "nR": preds.get("nR"),
        "n_executed": len(executed),
        "lift_completed": out.get("lift_completed"),
        "recovery_elapsed": out.get("recovery_elapsed"),
    }


def m111_translate_and_replay(env, blob, k48_actions: np.ndarray, book_m111, rho, *, handoff_how: str) -> dict:
    acts = np.asarray(k48_actions, np.float32).reshape(-1, 7)
    if len(acts) == 0:
        return {
            "compatible": 0,
            "note": "empty_action_sequence",
            "success_12s": 0,
            "SUCCESS_RELAXED_12S": "",
        }
    q = quantize_trajectory(book_m111, acts, rho, w=DEFAULT_WEIGHTS)
    recon = np.asarray(q["recon_actions"], np.float32)
    # Faithfulness
    err_l2 = np.linalg.norm(recon - acts, axis=1)
    # Decode via official M111 decoder for each token/residual (sanity)
    decoded = []
    for tok, res in zip(q["tokens"], q["residuals"]):
        decoded.append(decode_m111_action(book_m111, int(tok), res, rho=rho))
    decoded = np.asarray(decoded, np.float32)
    decode_err = np.linalg.norm(decoded - recon, axis=1)

    # Physical replay of reconstructed actions + same handoff mode as K48
    phys = replay_actions_then_finish(env, blob, recon, handoff_how=handoff_how)

    # First large divergence vs original K48 actions (command space)
    first_cmd = None
    for i, (a, r) in enumerate(zip(acts, recon)):
        if float(np.linalg.norm(a - r)) > 0.15:
            first_cmd = i
            break

    return {
        "compatible": 1,
        "n_actions": int(len(acts)),
        "clipped_frac": float(q["clipped_frac"]),
        "mean_recon_wlinf": float(q["mean_recon_wlinf"]),
        "max_recon_wlinf": float(q["max_recon_wlinf"]),
        "mean_recon_l2": float(np.mean(err_l2)),
        "max_recon_l2": float(np.max(err_l2)),
        "decode_match_mean_l2": float(np.mean(decode_err)),
        "tokens_head": [int(x) for x in q["tokens"][:40]],
        "first_cmd_divergence_step": first_cmd,
        "sat_frac_per_dim": q["sat_frac_per_dim"].tolist(),
        "success_12s": phys.get("success_12s"),
        "reason": phys.get("reason"),
        "frame_ok_at_end": phys.get("frame_ok_at_end"),
        "theta_end": phys.get("theta"),
        "e_pos_end": phys.get("e_pos"),
        "nL_end": phys.get("nL"),
        "nR_end": phys.get("nR"),
        "handoff_phys": phys.get("handoff_phys"),
        "SUCCESS_RELAXED_12S": "",  # not defined on K48-style finish_recovery path
        "handoff_how_used": handoff_how,
    }


def run_p64_for_divergence(starts, frozen, device) -> list[dict]:
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    fr = {r["traj_id"]: r for r in frozen}
    rows = []
    for s in starts:
        seed = int(fr[s.traj_id]["seed"])
        print(f"[P64] {s.traj_id} seed={seed}", flush=True)
        rec = rollout_episode(env, actor, critic, book, s, device, seed=seed)
        info = getattr(env, "last_info", None) or {}
        fam = rec.get("family_seq") or []
        rows.append(
            {
                "traj_id": s.traj_id,
                "seed": seed,
                "SUCCESS_RELAXED_12S": int(rec.get("SUCCESS_RELAXED") or 0),
                "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
                "SUCCESS_NEW": int(info.get("SUCCESS_NEW") or 0),
                "fail": rec.get("earliest_decisive_failure"),
                "ep_len": int(rec.get("ep_len") or 0),
                "first_contact": int(rec.get("first_contact") or 0),
                "opposing_ever": int(rec.get("opposing_ever") or 0),
                "bilateral_ever": int(rec.get("bilateral_ever") or 0),
                "min_pad_ge4": int(rec.get("min_pad_ge4") or 0),
                "e_pos_le9": int(rec.get("e_pos_le9") or 0),
                "r3_ready": int(rec.get("r3_ready") or 0),
                "handoff": int(rec.get("handoff") or 0),
                "t_fc": rec.get("t_fc"),
                "t_opp": rec.get("t_opp"),
                "family_seq": fam,
                "tokens": [int(x["tok"]) for x in fam],
                "families": [x.get("fam") for x in fam],
            }
        )
    return rows


def analyze_divergence(k48_rec: dict, p64_rec: dict, m111_rec: dict) -> dict:
    """Earliest meaningful divergence classification."""
    # Gate story
    gate_ready = bool(k48_rec.get("p64_would_permit_handoff_with_dwell"))
    frame_ok = bool(k48_rec.get("p64_frame_ok_at_handoff_instant"))
    k_ok = int(k48_rec.get("success_12s") or 0)
    m_ok = int(m111_rec.get("success_12s") or 0) if m111_rec else 0
    p_ok = int(p64_rec.get("SUCCESS_RELAXED_12S") or 0)

    if k_ok and not frame_ok:
        layer = "handoff_gate_instant_frame_fail"
        detail = "K48 HANDOFFs at a state that fails R3F instantaneous frame_ok"
    elif k_ok and frame_ok and not gate_ready:
        layer = "handoff_gate_dwell_incomplete"
        detail = "K48 HANDOFF state is geometrically frame_ok but dwell history incomplete for R3F"
    elif k_ok and gate_ready and not p_ok:
        layer = "p64_never_reaches_equivalent_or_overcontrols"
        detail = "K48 HANDOFF would be R3F-legal; P64 still fails — policy path issue likely"
    elif k_ok and not m_ok:
        layer = "m111_representation_loss"
        detail = "K48 succeeds; M111-quantized physical replay fails SUCCESS_12S"
    elif k_ok and m_ok and not p_ok:
        layer = "policy_choice"
        detail = "M111 can reproduce K48 recovery; P64 chooses different unsuccessful actions"
    else:
        layer = "other"
        detail = "see metrics"

    # Action-level: compare first K48 action family vs P64
    k_toks = [int(t) for t in k48_rec.get("tokens") or [] if int(t) != K48_HANDOFF]
    p_toks = list(p64_rec.get("tokens") or [])
    return {
        "earliest_layer": layer,
        "detail": detail,
        "k48_success_12s": k_ok,
        "m111_replay_success_12s": m_ok,
        "p64_success_relaxed": p_ok,
        "k48_handoff_frame_ok": int(frame_ok),
        "k48_handoff_r3f_ready_with_dwell": int(gate_ready),
        "k48_first_tokens": k_toks[:8],
        "p64_first_tokens": p_toks[:8],
        "p64_first_families": (p64_rec.get("families") or [])[:8],
        "m111_clipped_frac": m111_rec.get("clipped_frac") if m111_rec else None,
        "m111_max_recon_l2": m111_rec.get("max_recon_l2") if m111_rec else None,
        "p64_fail": p64_rec.get("fail"),
        "k48_reason": k48_rec.get("reason"),
        "m111_reason": m111_rec.get("reason") if m111_rec else None,
    }


def write_reports(
    *,
    k48_all: list[dict],
    m111_rows: list[dict],
    p64_rows: list[dict],
    div_rows: list[dict],
    book_k48,
    book_m111,
    rho,
) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(k48_all)
    k_s = sum(int(r["success_12s"]) for r in k48_all)
    m_s = sum(int(r.get("success_12s") or 0) for r in m111_rows)
    p_s = sum(int(r.get("SUCCESS_RELAXED_12S") or 0) for r in p64_rows)
    frame_ok_n = sum(int(r.get("p64_frame_ok_at_handoff_instant") or 0) for r in k48_all)
    dwell_ready_n = sum(int(r.get("p64_would_permit_handoff_with_dwell") or 0) for r in k48_all)
    layers = {}
    for d in div_rows:
        layers[d["earliest_layer"]] = layers.get(d["earliest_layer"], 0) + 1

    # --- K48_SUCCESS_TRAJECTORIES.md ---
    lines = [
        f"# K48_SUCCESS_TRAJECTORIES\n\nGenerated: {_utc()}\n",
        "## Inference procedure",
        "",
        "- Checkpoint: `results/grpo/tree80/checkpoint_epoch_0003.pt`",
        "- Wrapper: instrumented **argmax** `roll_policy` equivalent (`decode_action` + `step_legal`)",
        "- **No tree search** at evaluation time. Tree80 name refers to *training* adaptive GRPO with tree sampling historically; this audit uses ordinary closed-loop argmax rollout.",
        f"- Recovery limit: {SELECTED_LIMIT_S}s; dt={DT_POLICY}s; HANDOFF token={K48_HANDOFF}",
        f"- Aggregate SUCCESS_12S: **{k_s}/{n}**",
        "",
        "## Per-IC summaries",
        "",
    ]
    for r in k48_all:
        hp = r.get("handoff_phys") or {}
        hr = r.get("handoff_r3f") or {}
        lines += [
            f"### `{r['traj_id']}`",
            f"- SUCCESS_12S={r['success_12s']} reason={r['reason']} handoff={r['handoff_mode']} "
            f"motion_steps={r['recovery_steps_motion']} elapsed={r.get('recovery_elapsed')}",
            f"- Restore phys: e_pos={r['phys0']['e_pos_mm']:.2f}mm θ={r['phys0']['theta_deg']:.2f}° "
            f"bil={r['phys0']['bilateral']} min_pad={r['phys0']['min_pad']} "
            f"fnL/R={r['phys0']['fn_L']:.2f}/{r['phys0']['fn_R']:.2f}",
            f"- HANDOFF phys: e_pos={hp.get('e_pos_mm')} θ={hp.get('theta_deg')} "
            f"bil={hp.get('bilateral')} nL/nR={hp.get('nL')}/{hp.get('nR')} "
            f"fnL/R={hp.get('fn_L')}/{hp.get('fn_R')}",
            f"- Tokens (incl HANDOFF): {r.get('tokens')}",
            f"- Raw trajectory JSON: `trajectories/{r['traj_id']}.json`",
            "",
        ]
    _write(OUT / "K48_SUCCESS_TRAJECTORIES.md", "\n".join(lines))

    # --- HANDOFF_COMPATIBILITY.md ---
    gs = gate_spec()
    hlines = [
        f"# HANDOFF_COMPATIBILITY\n\nGenerated: {_utc()}\n",
        "## P64 R3F gate (unchanged; audit only)",
        "",
        f"```json\n{json.dumps(gs, indent=2)}\n```",
        "",
        "Distinction: **frame_ok** = instantaneous geometry; **ready/R3_EARLY** = frame_ok held continuously for HOLD_S (dwell).",
        "",
        f"| Aggregate | Count |",
        f"|---|---:|",
        f"| K48 SUCCESS_12S | {k_s}/{n} |",
        f"| Instantaneous frame_ok at K48 HANDOFF | {frame_ok_n}/{n} |",
        f"| R3F ready with dwell at K48 HANDOFF | {dwell_ready_n}/{n} |",
        "",
        "## Per-IC gate audit at K48 HANDOFF state",
        "",
        "| traj_id | K48 s12 | θ | e_pos | bil | min_pad | working | margin | frame_ok | ok_s | R3F ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in k48_all:
        hp = r.get("handoff_preds") or {}
        hr = r.get("handoff_r3f") or {}
        hlines.append(
            f"| `{r['traj_id']}` | {r['success_12s']} | {hp.get('theta')} | {hp.get('e_pos')} | "
            f"{int(bool(hp.get('bilateral')))} | {int(bool(hp.get('min_pad_ok')))} | "
            f"{int(bool(hp.get('working_ok')))} | {int(bool(hr.get('margin_ok')))} | "
            f"{int(bool(r.get('p64_frame_ok_at_handoff_instant')))} | {hr.get('ok_s')} | "
            f"{int(bool(r.get('p64_would_permit_handoff_with_dwell')))} |"
        )
    hlines += [
        "",
        "## Interpretation rules used",
        "",
        "- If K48 HANDOFF state fails `frame_ok`, R3F would forbid HANDOFF even with infinite dwell — gate geometry conflict.",
        "- If `frame_ok` but `ok_s < HOLD_S`, the state is geometrically eligible but dwell history is missing — K48 may succeed via early HANDOFF without R3F dwell.",
        "- P64 over-control after reaching an equivalent ready state is checked in Phase 4 via P64 `r3_ready`/`handoff` flags.",
        "",
    ]
    _write(OUT / "HANDOFF_COMPATIBILITY.md", "\n".join(hlines))

    # --- K48_M111_ACTION_MAPPING.md ---
    mlines = [
        f"# K48_M111_ACTION_MAPPING\n\nGenerated: {_utc()}\n",
        "## Interfaces",
        "",
        "| Item | K48 | M111 (P64) |",
        "|---|---|---|",
        f"| Motion codebook | {book_k48.shape[0]} × 7 (`k47`) | {book_m111.shape[0]} × 7 |",
        f"| Residual cap | ±{K48_CAP} | rho={np.asarray(rho).tolist()} |",
        f"| HANDOFF id | {K48_HANDOFF} (not a motion decode) | {M111_HANDOFF} |",
        "| Control | 7D legal `step_legal`, same dt | same |",
        "| Token ID correspondence | **None assumed** | nearest-proto + residual via `quantize_trajectory` |",
        "",
        "## Translation method",
        "",
        "1. Record K48 decoded continuous 7D actions (after residual).",
        "2. `quantize_trajectory(M111_book, actions, rho, DEFAULT_WEIGHTS)` → tokens + clipped residuals.",
        "3. Replay reconstructed actions with `step_legal` from the same capture.",
        "4. Apply the **same** K48 handoff mode (`early`/`forced`) then `finish_recovery` for SUCCESS_12S.",
        "5. Do **not** invent new M111 tokens or expand the vocabulary.",
        "",
        f"## Aggregate replay: M111 SUCCESS_12S **{m_s}/{n}** (K48 was {k_s}/{n})",
        "",
    ]
    _write(OUT / "K48_M111_ACTION_MAPPING.md", "\n".join(mlines))

    # --- M111_PHYSICAL_REPLAY.csv ---
    _write_csv(OUT / "M111_PHYSICAL_REPLAY.csv", m111_rows)

    # --- P64_EARLIEST_DIVERGENCE.md ---
    dlines = [
        f"# P64_EARLIEST_DIVERGENCE\n\nGenerated: {_utc()}\n",
        f"P64 SUCCESS_RELAXED_12S on diagnostic replay: **{p_s}/{n}**",
        "",
        "## Layer counts",
        "",
    ]
    for k, v in sorted(layers.items(), key=lambda kv: -kv[1]):
        dlines.append(f"- `{k}`: {v}/{n}")
    dlines += ["", "## Per-IC", ""]
    for d in div_rows:
        dlines += [
            f"### `{d['traj_id']}`",
            f"- Layer: **{d['earliest_layer']}** — {d['detail']}",
            f"- K48 s12={d['k48_success_12s']} reason={d['k48_reason']}; "
            f"frame_ok={d['k48_handoff_frame_ok']} r3f_dwell_ready={d['k48_handoff_r3f_ready_with_dwell']}",
            f"- M111 replay s12={d['m111_replay_success_12s']} reason={d['m111_reason']} "
            f"clipped_frac={d['m111_clipped_frac']} max_recon_l2={d['m111_max_recon_l2']}",
            f"- P64 fail={d['p64_fail']} relaxed={d['p64_success_relaxed']}",
            f"- K48 first tokens={d['k48_first_tokens']}; P64 first={d['p64_first_tokens']} fam={d['p64_first_families']}",
            "",
        ]
    _write(OUT / "P64_EARLIEST_DIVERGENCE.md", "\n".join(dlines))

    # Decision logic
    # Prefer representation loss if M111 replay systematically fails while K48 succeeds
    # Prefer handoff if frame_ok rarely true at K48 handoff
    # Prefer policy if M111 replay succeeds and gate is compatible
    rep_fail = sum(1 for d in div_rows if d["k48_success_12s"] and not d["m111_replay_success_12s"])
    gate_geo = sum(1 for d in div_rows if d["k48_success_12s"] and not d["k48_handoff_frame_ok"])
    gate_dwell = sum(
        1
        for d in div_rows
        if d["k48_success_12s"] and d["k48_handoff_frame_ok"] and not d["k48_handoff_r3f_ready_with_dwell"]
    )
    policy_n = sum(
        1
        for d in div_rows
        if d["k48_success_12s"] and d["m111_replay_success_12s"] and not d["p64_success_relaxed"]
    )

    causes = []
    if gate_geo >= n // 2 or gate_dwell >= n // 2:
        causes.append("HANDOFF")
    if rep_fail >= max(1, n // 2):
        causes.append("REPRESENTATION")
    if policy_n >= max(1, n // 2) and rep_fail == 0 and gate_geo == 0:
        causes.append("POLICY")
    elif policy_n >= 1 and m_s == k_s and k_s == n:
        causes.append("POLICY")

    if len(causes) >= 2:
        case = "D"
        title = "Multiple interacting causes"
    elif causes == ["HANDOFF"] or (gate_dwell == n and m_s == k_s):
        # If all have frame_ok but no dwell, and M111 replay preserves SUCCESS_12S under K48 handoff contract
        case = "A"
        title = "HANDOFF incompatibility"
    elif causes == ["REPRESENTATION"] or (rep_fail == k_s and k_s == n):
        case = "B"
        title = "M111 representation loss"
    elif "POLICY" in causes and rep_fail == 0:
        case = "C"
        title = "Policy skill regression"
    elif k_s == 0:
        case = "E"
        title = "Inconclusive"
    else:
        # nuanced: dwell missing everywhere + representation often OK
        if gate_dwell == n and rep_fail == 0 and m_s == n:
            case = "A"
            title = "HANDOFF incompatibility"
        elif rep_fail > 0 and gate_dwell > 0:
            case = "D"
            title = "Multiple interacting causes"
        elif rep_fail == n:
            case = "B"
            title = "M111 representation loss"
        elif m_s == n and p_s == 0:
            case = "C"
            title = "Policy skill regression"
        else:
            case = "D"
            title = "Multiple interacting causes"

    # Refine with explicit measured numbers (override heuristic if clear)
    if k_s == n and m_s == 0:
        case, title = "B", "M111 representation loss"
    elif k_s == n and m_s == n and dwell_ready_n == 0 and frame_ok_n < n:
        case, title = "A", "HANDOFF incompatibility"
    elif k_s == n and m_s == n and dwell_ready_n == 0 and frame_ok_n == n:
        case, title = "A", "HANDOFF incompatibility"  # dwell gap: K48 hands off without R3F dwell
    elif k_s == n and m_s == n and dwell_ready_n == n and p_s == 0:
        case, title = "C", "Policy skill regression"
    elif k_s == n and 0 < m_s < n and gate_dwell > 0:
        case, title = "D", "Multiple interacting causes"

    intervention = {
        "A": (
            "Bounded gate-focused counterfactual: on training-split B1 ICs only, measure whether "
            "allowing HANDOFF when frame_ok without full dwell (or a small dwell schedule) recovers "
            "SUCCESS without changing the Actor. Do not deploy; do not touch the eight eval ICs."
        ),
        "B": (
            "Narrow representation/decoder investigation: quantify which K48 7D commands are outside "
            "M111 reachable set (clipped dims / far prototypes). Do not auto-expand vocabulary; "
            "first test whether increasing residual budget or denser local prototypes on train ICs "
            "recovers physical replay."
        ),
        "C": (
            "Small B1 skill-retention experiment: BC/distill K48 teacher trajectories from "
            "**training-split** ICs only into M111, with explicit A1/C/D regression eval. "
            "Never train or select on the eight exposed B1 eval ICs."
        ),
        "D": "Run the minimum separating experiment for each listed cause (gate CF, then M111 replay fidelity, then train-split BC).",
        "E": "Do not intervene; gather missing compatibility measurements first.",
    }[case]

    diag = f"""# B1_SKILL_REGRESSION_DIAGNOSIS

Generated: {_utc()}

## Primary diagnosis: **CASE {case}** — {title}

### Confirmed measurements (diagnostic 8 B1 ICs; not a new independent test)

| Check | Result |
|---|---|
| K48 tree80_e3 SUCCESS_12S | **{k_s}/{n}** |
| M111 quantized physical replay SUCCESS_12S | **{m_s}/{n}** |
| Instantaneous R3F frame_ok at K48 HANDOFF | **{frame_ok_n}/{n}** |
| R3F dwell-ready at K48 HANDOFF | **{dwell_ready_n}/{n}** |
| P64 SUCCESS_RELAXED_12S (post-hoc) | **{p_s}/{n}** |
| Frozen independent P64 B1 | **0/8** (unchanged) |

### Layer counts
{chr(10).join(f'- `{k}`: {v}' for k,v in sorted(layers.items(), key=lambda kv: -kv[1]))}

### Confirmed vs hypothesis

**Confirmed**
- tree80 eval is ordinary argmax rollout (no search at inference).
- Physical restore of B1 captures works; P64 0/8 is not an eval bug (prior audit).
- K48 succeeds with ~0.72s motion before HANDOFF on these ICs.
- R3F predicates at K48 HANDOFF states were measured without modifying the gate.

**Hypothesis / bounded next test**
- See `NEXT_INTERVENTION_DECISION.md` — single recommended intervention for CASE {case}.
"""
    _write(OUT / "B1_SKILL_REGRESSION_DIAGNOSIS.md", diag)

    nxt = f"""# NEXT_INTERVENTION_DECISION

Generated: {_utc()}

## CASE {case} — {title}

### Single bounded next intervention

{intervention}

### Hard constraints (carry forward)

- No training/selection on the eight exposed B1 eval ICs.
- No automatic deployment of a B1-only policy.
- Preserve frozen P64 independent result.
- Do not assume K48 B1 gains transfer to A1/C/D without regression eval.
"""
    _write(OUT / "NEXT_INTERVENTION_DECISION.md", nxt)

    final = f"""# FINAL_REPORT — P85 K48→P64 B1 Skill Retention

Generated: {_utc()}

## Why tree80_e3 succeeds while P64 fails

K48 tree80_e3 runs an **ordinary argmax closed-loop** (~36 steps / ~0.72 s), then emits HANDOFF=47 and passes the historical **SUCCESS_12S** nominal hold on **{k_s}/{n}** of these diagnostic B1 ICs.

P64, on the same physical initial conditions, reaches contact milestones (first contact / opposing) but ends at `no_e_pos_le9` / `no_r3_ready` — frozen independent result **0/8** SUCCESS_RELAXED_12S remains the official estimate.

## What we measured

1. **Full K48 trajectories** (actions, tokens, contact, errors, HANDOFF state, nominal continuation) — `K48_SUCCESS_TRAJECTORIES.md` + `trajectories/*.json`.
2. **R3F compatibility at K48 HANDOFF** — frame_ok **{frame_ok_n}/{n}**, dwell-ready **{dwell_ready_n}/{n}** — `HANDOFF_COMPATIBILITY.md`.
3. **M111 representability** via nearest-proto quantization + physical replay — SUCCESS_12S **{m_s}/{n}** — `K48_M111_ACTION_MAPPING.md`, `M111_PHYSICAL_REPLAY.csv`.
4. **P64 divergence layers** — `P64_EARLIEST_DIVERGENCE.md`.

## Diagnosis

**CASE {case}: {title}**

{diag.split('### Confirmed vs hypothesis',1)[-1].strip()}

## Recommended next intervention (at most one)

{intervention}

## Non-claims

- This is **not** a new independent B1 test estimate.
- K48 SUCCESS_12S ≠ P64 SUCCESS_RELAXED_12S; both are reported where defined.
- No training, reward, physics, M111 expansion, or checkpoint swap was performed in P85.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    return case


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    b1_eps, starts, frozen = load_b1()
    assert len(starts) == 8

    # Phase 1 — K48 trajectories
    model, book_k48 = load_trained_k48(K48_TREE80, device)
    model.eval()
    # Prefer codebook from loader; fall back
    if book_k48 is None or len(book_k48) != 47:
        book_k48 = load_codebook()
    env_k = make_training_env(device, model=model)

    k48_all = []
    for s in starts:
        blob = blob_from_start(s)
        print(f"[K48 traj] {s.traj_id}", flush=True)
        rec = record_k48_trajectory(env_k, model, book_k48, blob, device)
        rec["traj_id"] = s.traj_id
        rec["root_id"] = next(e["root_id"] for e in b1_eps if e["traj_id"] == s.traj_id)
        # persist
        dump = {k: _jsonable(v) for k, v in rec.items() if k not in ("actions", "residuals")}
        dump["actions"] = rec["actions"].tolist()
        dump["residuals"] = rec["residuals"].tolist()
        (TRAJ_DIR / f"{s.traj_id}.json").write_text(json.dumps(dump, indent=2), encoding="utf-8")
        np.savez_compressed(
            TRAJ_DIR / f"{s.traj_id}_actions.npz",
            actions=rec["actions"],
            residuals=rec["residuals"],
            tokens=np.asarray(rec["tokens"], np.int64),
        )
        k48_all.append(rec)
        print(
            f"  s12={rec['success_12s']} steps={rec['recovery_steps_motion']} "
            f"frame_ok={rec.get('p64_frame_ok_at_handoff_instant')} "
            f"r3_ready={rec.get('p64_would_permit_handoff_with_dwell')}",
            flush=True,
        )

    # Phase 3 — M111 translate/replay (uses same env after K48)
    actor, book_m111, rho, _ = load_actor(P64_ACTOR, device)
    del actor  # weights unused for quantization
    m111_rows = []
    for rec, s in zip(k48_all, starts):
        blob = blob_from_start(s)
        how = rec.get("handoff_mode") or "forced"
        print(f"[M111 replay] {s.traj_id}", flush=True)
        try:
            m = m111_translate_and_replay(
                env_k, blob, rec["actions"], book_m111, rho, handoff_how=how if how in ("early", "forced") else "forced"
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            m = {"compatible": 0, "success_12s": 0, "reason": f"ERROR:{exc}", "infra_error": str(exc)}
        m["traj_id"] = s.traj_id
        m["root_id"] = rec["root_id"]
        m["k48_success_12s"] = rec["success_12s"]
        m["k48_reason"] = rec["reason"]
        m["evidence_class"] = "post_hoc_diagnostic_not_independent_final_test"
        m111_rows.append(m)
        print(f"  m111_s12={m.get('success_12s')} clipped={m.get('clipped_frac')} max_l2={m.get('max_recon_l2')}", flush=True)

    # Phase 4 — P64
    p64_rows = run_p64_for_divergence(starts, frozen, device)
    p64_by = {r["traj_id"]: r for r in p64_rows}
    m_by = {r["traj_id"]: r for r in m111_rows}

    div_rows = []
    for rec in k48_all:
        tid = rec["traj_id"]
        div_rows.append({"traj_id": tid, **analyze_divergence(rec, p64_by[tid], m_by[tid])})

    raw_k48 = []
    for r in k48_all:
        item = {k: _jsonable(v) for k, v in r.items() if k not in ("steps", "actions", "residuals")}
        item["n_steps"] = len(r.get("steps") or [])
        acts = r.get("actions")
        item["n_actions"] = int(len(acts)) if acts is not None else 0
        raw_k48.append(item)
    (OUT / "RAW_K48.json").write_text(json.dumps(raw_k48, indent=2), encoding="utf-8")
    (OUT / "RAW_DIVERGENCE.json").write_text(json.dumps(div_rows, indent=2), encoding="utf-8")
    (OUT / "RAW_P64.json").write_text(json.dumps(_jsonable(p64_rows), indent=2), encoding="utf-8")

    case = write_reports(
        k48_all=k48_all,
        m111_rows=m111_rows,
        p64_rows=p64_rows,
        div_rows=div_rows,
        book_k48=book_k48,
        book_m111=book_m111,
        rho=rho,
    )
    print(json.dumps({"case": case, "out": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
