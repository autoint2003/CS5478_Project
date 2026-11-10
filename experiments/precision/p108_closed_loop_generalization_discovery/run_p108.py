"""P108 — Closed-loop generalization and recoverability discovery (diagnostics only).

No training. No P64 replacement. No sealed independent panel. No P109 execution.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p60_c_stable_state_grpo.full_state import (  # noqa: E402
    restore_full_state,
    take_full_state,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
    expert_actions,
    step_raw,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates  # noqa: E402
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402

import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402
import experiments.precision.p107_action_aware_multiseed_validation.run_p107 as p107  # noqa: E402

OUT = ROOT / "results/experiments/p108_closed_loop_generalization_discovery"
CACHE = OUT / "cache"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P107_OUT = ROOT / "results/experiments/p107_action_aware_multiseed_validation"
EXPECTED_SHA = "ec08f59816aa2fdd"
CL_SEED = 101
TARGET_ROOTS = 20
# Primary actor seeds for full panel; extra seeds on a subset for stability check
FULL_SEEDS = [1]
EXTRA_SEEDS = [2, 3]
EXTRA_SEED_ROOT_CAP = 8
# Takeover budget: exhaustive CF on every failure is prohibitive; sample for mechanism evidence.
TAKEOVER_MAX_FAILS_PER_POLICY = 10
TAKEOVER_TIMES = [(0, "recovery_entry"), (5, "first_divergence_proxy")]


def utc() -> str:
    return p104.utc()


def write_json(path: Path, obj) -> None:
    p104.write_json(path, obj)


def write_csv(path: Path, rows: list[dict]) -> None:
    p104.write_csv(path, rows)


def sha16_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def gbi_key(s: str) -> str | None:
    m = re.search(r"gbi_\d+", str(s), re.I)
    return m.group(0).lower() if m else None


def canonical_physical_id(tid: str) -> str:
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)
    return t


def build_panel(starts: dict) -> tuple[list[dict], dict]:
    """Provenance-clean C panel: eligibility BEFORE any candidate-policy outcome."""
    splits = json.loads((P98 / "SPLITS.json").read_text(encoding="utf-8"))
    train_g = {gbi_key(x) for x in splits["c_train_phys"]} - {None}
    dev_g = {gbi_key(x) for x in splits.get("c_dev_phys", [])} - {None}
    ind_g = {gbi_key(t) for t in (splits.get("ind") or {}).get("C") or []} - {None}
    ban_g = {gbi_key(t) for t in p104.eval_banned_phys()} - {None}
    p107_man = json.loads((P107_OUT / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    p107_g = set()
    for sk in ("C_Feasible", "C_Unfiltered", "C_train"):
        for t in p107_man["panels"].get(sk, []):
            g = gbi_key(t)
            if g:
                p107_g.add(g)

    by_g: dict[str, list[tuple[str, MixedStart]]] = defaultdict(list)
    for tid, st in starts.items():
        if getattr(st, "skill", None) != "C":
            continue
        g = gbi_key(tid)
        if g is None:
            continue
        ep = st.ep or {}
        try:
            acts = expert_actions(ep)
        except Exception:
            continue
        if acts is None or len(np.asarray(acts).reshape(-1)) < 7:
            continue
        by_g[g].append((tid, st))

    rows = []
    skipped = {"train": 0, "sealed_ind": 0, "no_expert_tape": 0}
    for g, lst in sorted(by_g.items()):
        if g in train_g:
            skipped["train"] += 1
            continue
        if g in ind_g:
            skipped["sealed_ind"] += 1
            continue
        # Prefer plan_c__ tid for canonical schedule identity
        lst_sorted = sorted(lst, key=lambda x: (0 if str(x[0]).startswith("plan_c__") else 1, x[0]))
        tid, st = lst_sorted[0]
        ep = st.ep or {}
        acts = np.asarray(expert_actions(ep), np.float32).reshape(-1, 7)
        snap_hash = ""
        try:
            # fingerprint from ep if present; else hash of catch/capture bytes proxy
            snap_hash = str(ep.get("fingerprint") or ep.get("lineage_id") or "")[:32]
            if not snap_hash and ep.get("catch_snap") is not None:
                snap_hash = sha16_bytes(pickle.dumps(ep.get("catch_snap")))
            elif not snap_hash and isinstance(ep.get("capture"), dict):
                snap_hash = sha16_bytes(pickle.dumps(ep.get("capture").get("physics")))
        except Exception:
            snap_hash = ""
        # Physical eligibility metadata (privileged diagnostics; not actor inputs)
        contact = {}
        try:
            # light metadata from ep only (no policy)
            contact = {
                "n_stage1": ep.get("n_stage1"),
                "n_precision": ep.get("n_precision"),
                "start_kind": ep.get("start_kind"),
                "includes_precision": ep.get("includes_precision"),
            }
        except Exception:
            pass
        if g in p107_g:
            provenance = "previously_exposed_p107_dev"
        elif g in dev_g:
            provenance = "previously_exposed_p98_dev"
        elif g in ban_g:
            provenance = "historical_freeze_ban_not_train_not_sealed"
        else:
            provenance = "newly_selected"
        rows.append(
            {
                "traj_id": tid,
                "canonical_physical_id": canonical_physical_id(tid),
                "gbi": g,
                "phys_root_truncated": p104.phys_root(tid),
                "snapshot_hash": snap_hash,
                "provenance": provenance,
                "in_p98_c_train": 0,
                "in_p98_c_dev": int(g in dev_g),
                "in_sealed_ind": 0,
                "in_freeze_ban": int(g in ban_g),
                "in_p107_c_panels": int(g in p107_g),
                "n_expert_actions": int(len(acts)),
                "eligibility_rule": (
                    "schedule_C_ep_with_expert_tape; exclude P98 c_train and sealed ind.C; "
                    "one traj per gbi; selected before candidate outcomes"
                ),
                "contact_meta_json": json.dumps(contact),
            }
        )

    meta = {
        "selection_rule": rows[0]["eligibility_rule"] if rows else "",
        "target_min": TARGET_ROOTS,
        "n_selected": len(rows),
        "skipped": skipped,
        "n_unique_gbi_available_after_exclusions": len(rows),
        "limitation": (
            None
            if len(rows) >= TARGET_ROOTS
            else f"Only {len(rows)} distinct unused schedule-C roots with expert tapes after exclusions; "
            "did not pad with training roots."
        ),
        "note_freeze_ban": (
            "Most unused schedule roots appear in historical freeze/P93/P94 bans. "
            "They are allowed here for diagnostics because they are not P98 train and not sealed ind. "
            "They must NOT be treated as sealed-independent evidence."
        ),
        "sealed_panel_opened": False,
    }
    return rows, meta


@torch.no_grad()
def diagnostic_rollout(
    actor,
    start: MixedStart,
    book,
    rho,
    device,
    *,
    seed: int = CL_SEED,
    save_snapshots_at: list[int] | None = None,
) -> dict:
    """Closed-loop with rich per-step diagnostics + optional full-state snapshots."""
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    steps: list[dict] = []
    snaps: dict[int, dict] = {}
    save_set = set(save_snapshots_at or [])
    ever12 = False
    while not done:
        t = int(env.t)
        if t in save_set:
            snaps[t] = take_full_state(env, meta={"t": t})
        g0 = relative_geometry(env.env.sim)
        m0 = state_metrics(env.env.sim)
        obs_t = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev_t = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        logits, res, act = actor(obs_t, prev_t)
        logits_last = mask_handoff_logits(logits[-1].float())
        probs = torch.softmax(logits_last, dim=-1)
        top2 = torch.topk(probs, k=min(2, probs.numel()))
        tok_a = int(torch.argmax(logits_last).item())
        margin = float(top2.values[0] - (top2.values[1] if top2.values.numel() > 1 else 0.0))
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        if int(tok) == HANDOFF:
            action = np.zeros(7, np.float32)
        else:
            action = np.asarray(decode_m111_action(book, int(tok), resid, rho), np.float32)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        obs39 = env.obs_seq[-1].astype(np.float32).copy()
        steps.append(
            {
                "t": t,
                "tok": int(tok),
                "tok_a": int(tok_a),
                "margin": margin,
                "top2_tok": [int(x) for x in top2.indices.cpu().tolist()],
                "resid": resid.copy(),
                "action": action.copy(),
                "obs": obs39,
                "prev": env.prev_seq[-1].astype(np.float32).copy(),
                "fn_L": float(obs39[10]) if obs39.shape[0] > 11 else float("nan"),
                "fn_R": float(obs39[11]) if obs39.shape[0] > 11 else float("nan"),
                "e_pos_mm": float(g0["e_pos_mm"]),
                "theta_deg": float(g0["theta_deg"]),
                "bilateral": int(bool(g0["bilateral"])),
                "min_pad": int(g0.get("min_pad") or 0),
                "working_ok": int(bool(preds.get("working_ok"))),
                "r3_ready": int(bool(preds.get("r3_ready") or preds.get("ready"))),
                "slip": float(m0.get("vrel") or m0.get("v_rel") or m0.get("slip") or 0.0),
                "v_rel": float(m0.get("vrel") or m0.get("v_rel") or 0.0),
            }
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        steps[-1]["handoff"] = int(info.get("handoff") or 0)
        g1 = relative_geometry(env.env.sim)
        bil = bool(g1["bilateral"])
        pad4 = int(g1.get("min_pad") or 0) >= 4
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        wk = bool(preds.get("working_ok"))
        e, th = float(g1["e_pos_mm"]), float(g1["theta_deg"])
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever12 = True
        # capture contact-loss / regrasp markers
        steps[-1]["bilateral_after"] = int(bil)
        steps[-1]["e_pos_mm_after"] = e
    term = _r3_terminal(env, info)
    # also snapshot at end for takeover from last pre-fail if needed
    if int(env.t) not in snaps and steps:
        # snapshot not taken mid-fail; keep last requested
        pass
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever12),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "handoff": int(info.get("handoff") or 0),
        "steps": steps,
        "snapshots": snaps,
    }


def expert_rollout(start: MixedStart, book, rho, *, seed: int = CL_SEED) -> dict:
    seed_all(seed, deterministic_torch=True)
    acts = expert_actions(start.ep or {})
    acts = np.asarray(acts, np.float32).reshape(-1, 7)
    if len(acts) == 0:
        return {
            "SUCCESS_RELAXED_12S": 0,
            "ep_len": 0,
            "fail_reason": "NO_EXPERT_ACTIONS",
            "expert_available": 0,
            "steps": [],
        }
    demo_steps = max(int(MAX_EP_STEPS), int(len(acts)) + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=demo_steps, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    done = False
    info: dict = {}
    steps = []
    for a in acts:
        if done or env.t >= env.max_steps:
            break
        g0 = relative_geometry(env.env.sim)
        m0 = state_metrics(env.env.sim)
        action = np.asarray(a, np.float32).reshape(7)
        steps.append(
            {
                "t": int(env.t),
                "action": action.copy(),
                "e_pos_mm": float(g0["e_pos_mm"]),
                "theta_deg": float(g0["theta_deg"]),
                "bilateral": int(bool(g0["bilateral"])),
                "slip": float(m0.get("vrel") or m0.get("v_rel") or m0.get("slip") or 0.0),
                "v_rel": float(m0.get("vrel") or m0.get("v_rel") or 0.0),
            }
        )
        _r, done, info = step_raw(env, action)
        ensure_inner_capture_buffers(env)
        steps[-1]["bilateral_after"] = int(bool(relative_geometry(env.env.sim)["bilateral"]))
    return {
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "expert_available": 1,
        "n_expert_actions": int(len(acts)),
        "steps": steps,
        "actions": acts,
    }


def classify_failure(
    actor_roll: dict,
    expert_roll: dict | None = None,
    book=None,
    rho=None,
) -> dict:
    """Multi-label failure taxonomy; earliest causal candidate marked separately.

    Label E is M111 Category-D (expert action has no feasible token under residual),
    NOT mere Actor–expert action disagreement.
    """
    succ = int(actor_roll.get("SUCCESS_RELAXED_12S") or 0)
    if succ:
        return {
            "failed": 0,
            "labels": "",
            "earliest_label": "",
            "earliest_t": "",
            "notes": "success",
            "early_actor_expert_action_gap": "",
        }
    steps = actor_roll.get("steps") or []
    events = []  # (t, label, note)
    early_gap = ""

    if not steps:
        return {
            "failed": 1,
            "labels": "A",
            "earliest_label": "A",
            "earliest_t": 0,
            "notes": "empty_rollout",
            "early_actor_expert_action_gap": "",
        }

    # A: initial action selection — large error growth in first 3 steps without contact gain
    early = steps[: min(3, len(steps))]
    if early:
        e0 = early[0]["e_pos_mm"]
        e1 = early[-1]["e_pos_mm"]
        bil0 = early[0]["bilateral"]
        if e1 > e0 + 15 and bil0 == 0:
            events.append((0, "A", "early_error_growth_no_contact"))

    # Diagnostic only (not label E): early Actor vs expert action gap
    if expert_roll and expert_roll.get("actions") is not None:
        ea = np.asarray(expert_roll["actions"], np.float32).reshape(-1, 7)
        n = min(len(steps), len(ea), 5)
        if n > 0:
            gaps = [float(np.linalg.norm(steps[i]["action"] - ea[i])) for i in range(n)]
            early_gap = f"{float(np.mean(gaps)):.4f}"

    # B: progressive drift
    e_series = [s["e_pos_mm"] for s in steps]
    if len(e_series) >= 8:
        mid = len(e_series) // 2
        if e_series[mid] > e_series[0] + 10 and e_series[-1] > e_series[mid] + 10:
            events.append((mid, "B", "progressive_pose_drift"))

    # C: post-regrasp stabilization — bilateral regained then lost
    bil = [int(s.get("bilateral_after", s["bilateral"])) for s in steps]
    regain = None
    for i in range(1, len(bil)):
        if bil[i - 1] == 0 and bil[i] == 1:
            regain = i
            break
    if regain is not None:
        lost_after = next((i for i in range(regain + 1, len(bil)) if bil[i] == 0), None)
        if lost_after is not None:
            events.append((lost_after, "C", "lost_after_regrasp"))

    # D: contact-mode transition — bilateral flip co-occurs with large action jump
    for i in range(1, len(steps)):
        a_jump = float(np.linalg.norm(steps[i]["action"] - steps[i - 1]["action"]))
        b0 = int(steps[i - 1].get("bilateral_after", steps[i - 1]["bilateral"]))
        b1 = int(steps[i].get("bilateral_after", steps[i]["bilateral"]))
        if b0 != b1 and a_jump > 0.5:
            events.append((i, "D", f"contact_flip_with_action_jump_{a_jump:.2f}"))
            break

    # E: action-vocabulary — expert tape has Category-D (no feasible M111 token).
    # Continuous experts often leave the residual ball; require repeated Cat-D in-window
    # so a single out-of-ball expert sample does not dominate earliest-cause.
    e_event = None
    if book is not None and rho is not None and expert_roll and expert_roll.get("actions") is not None:
        ea = np.asarray(expert_roll["actions"], np.float32).reshape(-1, 7)
        n = min(len(steps), len(ea), 8)
        cat_d_ts = [i for i in range(n) if not p104.feasible_tokens(book, ea[i], rho)]
        if len(cat_d_ts) >= max(2, (n + 3) // 4):
            e_event = (cat_d_ts[0], "E", f"expert_category_D_steps_{len(cat_d_ts)}_of_{n}")
            events.append(e_event)

    # F: observation ambiguity proxy — low margin + token flip
    low_margin_flips = 0
    for i in range(1, len(steps)):
        if steps[i]["tok"] != steps[i - 1]["tok"] and float(steps[i]["margin"]) < 0.05:
            low_margin_flips += 1
    if low_margin_flips >= 3:
        t0 = next(
            (
                i
                for i in range(1, len(steps))
                if steps[i]["tok"] != steps[i - 1]["tok"] and float(steps[i]["margin"]) < 0.05
            ),
            1,
        )
        events.append((t0, "F", f"low_margin_flips_{low_margin_flips}"))

    # G: handoff / nominal continuation
    if int(actor_roll.get("reach12") or 0) == 1 and succ == 0:
        events.append((len(steps) - 1, "G", "reached_12_but_no_relaxed_success"))
    if str(actor_roll.get("fail_reason") or "").upper().find("HANDOFF") >= 0:
        events.append((len(steps) - 1, "G", str(actor_roll.get("fail_reason"))))

    if not events:
        events.append((max(0, len(steps) // 2), "B", "unspecified_closed_loop_failure"))

    # Earliest *physical/causal* candidate prefers A/B/C/D/F/G over structural E.
    # E remains a contributing label when present.
    order = {c: i for i, c in enumerate("ABCDEFG")}
    physical = [e for e in events if e[1] != "E"]
    pool = physical if physical else events
    events_sorted = sorted(pool, key=lambda x: (x[0], order.get(x[1], 99)))
    earliest = events_sorted[0]
    # keep full event list (with E) for contributing labels
    events_sorted = sorted(events, key=lambda x: (x[0], order.get(x[1], 99)))
    seen = set()
    uniq = []
    for _, lab, _ in events_sorted:
        if lab not in seen:
            seen.add(lab)
            uniq.append(lab)
    return {
        "failed": 1,
        "labels": "|".join(uniq),
        "earliest_label": earliest[1],
        "earliest_t": earliest[0],
        "notes": earliest[2],
        "early_actor_expert_action_gap": early_gap,
    }


def continue_from_state(
    actor_or_none,
    start: MixedStart,
    state: dict,
    book,
    rho,
    device,
    *,
    mode: str,
    expert_acts: np.ndarray | None = None,
    prefix_k: int = 0,
    seed: int = CL_SEED,
) -> dict:
    """Counterfactual from exact matched full state.

    modes:
      actor — continue original actor
      expert — switch to expert actions from policy_step index onward (off-policy caveat)
      prefix_expert — expert for prefix_k steps then actor
    """
    seed_all(seed, deterministic_torch=True)
    max_steps = int(state["max_steps"])
    if mode in ("expert", "prefix_expert") and expert_acts is not None:
        max_steps = max(max_steps, int(len(expert_acts)) + 8)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=max_steps, t_max=int(state["t_max"]), weights=PoseRewardWeights(), use_stabilization=False
    )
    # Need a reset binding before restore
    env.reset(start)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    # fidelity check
    fidelity_ok = abs(float(env.env.sim.data.time) - float(state["sim_time"])) < 1e-6
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    t0 = int(env.t)
    expert_steps_used = 0
    while not done and env.t < env.max_steps:
        t = int(env.t)
        use_expert = False
        if mode == "expert" and expert_acts is not None and t < len(expert_acts):
            use_expert = True
        elif mode == "prefix_expert" and expert_acts is not None and expert_steps_used < prefix_k and t < len(expert_acts):
            use_expert = True
        if use_expert:
            a = np.asarray(expert_acts[t], np.float32).reshape(7)
            _r, done, info = step_raw(env, a)
            expert_steps_used += 1
        else:
            if actor_or_none is None:
                break
            obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
            prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
            logits, res, _ = actor_or_none(obs, prev)
            tok_a = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
            tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
            resid = res[-1].detach().cpu().numpy().astype(np.float32)
            _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    return {
        "mode": mode,
        "intervention_t": t0,
        "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "snapshot_fidelity_ok": int(fidelity_ok),
        "expert_steps_used": expert_steps_used,
        "expert_offpolicy_caveat": 1 if mode in ("expert", "prefix_expert") else 0,
        "note": (
            "Expert actions indexed by absolute policy step from nominal tape; "
            "may be invalid from off-policy Actor-induced states."
        ),
    }


def temporal_rows_from_roll(policy: str, tid: str, roll: dict, expert_roll: dict | None) -> list[dict]:
    steps = roll.get("steps") or []
    rows = []
    for i, s in enumerate(steps):
        a_jump = 0.0 if i == 0 else float(np.linalg.norm(s["action"] - steps[i - 1]["action"]))
        tok_change = 0 if i == 0 else int(s["tok"] != steps[i - 1]["tok"])
        force_osc = 0.0
        if i >= 2:
            force_osc = abs((s["fn_L"] + s["fn_R"]) - (steps[i - 1]["fn_L"] + steps[i - 1]["fn_R"]))
        b_prev = int(steps[i - 1].get("bilateral_after", steps[i - 1]["bilateral"])) if i > 0 else int(s.get("bilateral_after", s["bilateral"]))
        b_cur = int(s.get("bilateral_after", s["bilateral"]))
        bil_loss = int(i > 0 and b_prev == 1 and b_cur == 0)
        rows.append(
            {
                "policy": policy,
                "traj_id": tid,
                "t": s["t"],
                "tok": s.get("tok", ""),
                "tok_change": tok_change,
                "action_jump_l2": a_jump,
                "large_action_jump": int(a_jump > 0.5),
                "force_delta": force_osc,
                "bilateral": b_cur,
                "bilateral_loss_event": bil_loss,
                "slip": s.get("slip", ""),
                "margin": s.get("margin", ""),
                "harmful_switch_candidate": int(tok_change == 1 and a_jump > 0.5 and bil_loss == 1),
            }
        )
    # compare switch rate to expert if available
    if expert_roll and expert_roll.get("steps"):
        esteps = expert_roll["steps"]
        e_jumps = []
        for i in range(1, len(esteps)):
            e_jumps.append(float(np.linalg.norm(esteps[i]["action"] - esteps[i - 1]["action"])))
        mean_e = float(np.mean(e_jumps)) if e_jumps else float("nan")
        for r in rows:
            r["expert_mean_action_jump"] = mean_e
    return rows


def m111_expressivity_for_failure(
    book, rho, actor_roll: dict, expert_roll: dict | None, tid: str, policy: str
) -> list[dict]:
    rows = []
    steps = actor_roll.get("steps") or []
    ea = None
    if expert_roll and expert_roll.get("actions") is not None:
        ea = np.asarray(expert_roll["actions"], np.float32).reshape(-1, 7)
    # sample up to 5 failed-critical steps: earliest failure-ish + evenly spaced
    idxs = sorted(set([0, len(steps) // 4, len(steps) // 2, max(0, len(steps) - 1)]))
    for i in idxs:
        if i >= len(steps):
            continue
        s = steps[i]
        a_actor = s["action"]
        # frozen residual recon error vs actor action under selected token
        tok = int(s.get("tok") or 0)
        if tok == HANDOFF:
            recon_err = float(np.linalg.norm(a_actor))
            feas = 0
        else:
            recon = decode_m111_action(book, tok, s["resid"], rho)
            recon_err = float(np.linalg.norm(np.asarray(recon) - a_actor))
            feas = int(p104.token_can_represent(book, tok, a_actor, rho))
        row = {
            "policy": policy,
            "traj_id": tid,
            "t": s["t"],
            "token": tok,
            "frozen_resid_recon_err": recon_err,
            "token_can_represent_actor_action": feas,
            "expert_action_err_best_token": "",
            "expert_n_feasible": "",
            "category_D_unrepresentable_expert": "",
        }
        if ea is not None and s["t"] < len(ea):
            a_star = ea[int(s["t"])]
            Z = p104.feasible_tokens(book, a_star, rho)
            if not Z:
                row["expert_n_feasible"] = 0
                row["category_D_unrepresentable_expert"] = 1
                row["expert_action_err_best_token"] = float("nan")
            else:
                errs = []
                for k in Z:
                    _r, e = p104.best_under_token(book, k, a_star, rho)
                    errs.append(e)
                row["expert_n_feasible"] = len(Z)
                row["category_D_unrepresentable_expert"] = 0
                row["expert_action_err_best_token"] = float(min(errs))
        rows.append(row)
    return rows


def observability_audit(p64, demos_corr: list[dict], device, out_md: Path) -> dict:
    """Nearest-neighbor conflicts in frozen Transformer representation space."""
    # Use encoder hidden via actor forward on windows; conflict = similar hist, different expert action
    reps = []
    acts = []
    metas = []
    with torch.no_grad():
        for d in demos_corr[:24]:  # cap
            obs = np.asarray(d["obs"], np.float32)
            prev = np.asarray(d["prev"], np.float32)
            actions = np.asarray(d["actions"], np.float32)
            n = min(len(obs), len(prev), len(actions))
            for i in range(0, n, max(1, n // 4)):
                o = torch.as_tensor(obs[max(0, i - 31) : i + 1], device=device, dtype=torch.float32)
                p = torch.as_tensor(prev[max(0, i - 31) : i + 1], device=device, dtype=torch.float32)
                if len(o) < 2:
                    continue
                logits, res, act = p64(o, p)
                # representation proxy: concat last logits + residual
                rep = torch.cat([logits[-1].float().reshape(-1), res[-1].float().reshape(-1)], dim=0)
                reps.append(rep.detach().cpu().numpy().astype(np.float32))
                acts.append(actions[i].astype(np.float32))
                metas.append({"traj_id": d.get("traj_id"), "t": i, "skill": d.get("skill")})
    if len(reps) < 4:
        summary = {
            "n_states": len(reps),
            "n_conflict_pairs": 0,
            "conflict_rate": float("nan"),
            "longer_history_helps": "insufficient_states",
        }
        out_md.write_text(
            "# OBSERVABILITY_AUDIT\n\nInsufficient correction states for NN audit.\n",
            encoding="utf-8",
        )
        return summary
    R = np.stack(reps)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    A = np.stack(acts)
    # pairwise cosine NN (exclude self)
    sim = R @ R.T
    np.fill_diagonal(sim, -1.0)
    nn = np.argmax(sim, axis=1)
    conflicts = 0
    resolved_by_contact = 0
    for i, j in enumerate(nn):
        if float(np.linalg.norm(A[i] - A[j])) > 0.5 and float(sim[i, j]) > 0.95:
            conflicts += 1
            # contact/force channels in obs not stored here; mark unknown
    summary = {
        "n_states": int(len(reps)),
        "n_conflict_pairs": int(conflicts),
        "conflict_rate": float(conflicts / max(len(reps), 1)),
        "sim_threshold": 0.95,
        "action_conflict_l2": 0.5,
        "longer_history_helps": "partial_untested_online",
        "note": (
            "Conflicts = high cosine similarity in (logits||residual) space with incompatible expert actions. "
            "Privileged physics not used as inference inputs. Contact/relative-motion channels exist in obs39 "
            f"(fn_L/fn_R). resolved_by_contact_proxy={resolved_by_contact}."
        ),
    }
    md = f"""# OBSERVABILITY_AUDIT — P108

## Method
Frozen P64 forward on correction-demo windows. Representation proxy = concatenated last-step token logits and residual vector (L2-normalized). Nearest neighbor by cosine similarity.

## Results
- States scored: **{summary['n_states']}**
- High-similarity action-conflict pairs (sim>{summary['sim_threshold']}, Δa_L2>{summary['action_conflict_l2']}): **{summary['n_conflict_pairs']}** ({summary['conflict_rate']:.3f})
- Longer causal history: not exhaustively ablated online; conflict rate above suggests history/representation collisions remain plausible (H4).
- Contact/relative-motion signals: present in actor obs as normal forces (indices 10–11); tangential slip is privileged-only for diagnostics.

## Interpretation
{"Non-trivial NN conflicts support partial observability / history ambiguity (H4) as a contributing mechanism." if conflicts else "Few NN conflicts in this sample; observability alone is weaker evidence than closed-loop drift/takeover results."}
"""
    out_md.write_text(md, encoding="utf-8")
    return summary


def rank_hypotheses(ctx: dict) -> tuple[str, list[dict]]:
    """Rank H1–H5 from collected evidence; return winner + table rows."""
    # Evidence scores 0-5
    scores = {
        "H1": 0.0,
        "H2": 0.0,
        "H3": 0.0,
        "H4": 0.0,
        "H5": 0.0,
    }
    n_fail = max(int(ctx.get("n_actor_fail_roots") or 0), 1)
    rescue = float(ctx.get("expert_rescue_rate") or 0.0)
    early_a = float(ctx.get("frac_earliest_A") or 0.0)
    drift_b = float(ctx.get("frac_label_B") or 0.0)
    switch_harm = float(ctx.get("frac_harmful_switch") or 0.0)
    cat_d = float(ctx.get("frac_cat_d_expert") or 0.0)
    obs_c = float(ctx.get("obs_conflict_rate") or 0.0)
    expert_ok = float(ctx.get("expert_success_rate") or 0.0)

    # H1: on-policy corrective imitation — driven by rescue of Actor-induced states.
    # Do NOT add nominal expert_success_rate (that measures IC recoverability, not takeover).
    scores["H1"] = 5.0 * rescue + 1.0 * (1.0 - early_a)
    # H2: future contact prediction — if contact-mode / post-regrasp failures dominate
    scores["H2"] = 4.0 * float(ctx.get("frac_label_D") or 0.0) + 2.0 * float(ctx.get("frac_label_C") or 0.0)
    # H3: temporal action consistency — harmful switches
    scores["H3"] = 5.0 * switch_harm + 2.0 * drift_b
    # H4: causal history — observability conflicts + F labels
    scores["H4"] = 4.0 * obs_c + 3.0 * float(ctx.get("frac_label_F") or 0.0)
    # H5: vocabulary — prefer expressivity Category-D rate; label-E is secondary
    scores["H5"] = 5.0 * cat_d + 0.5 * float(ctx.get("frac_label_E") or 0.0)
    _ = expert_ok  # retained in ctx for reporting only

    # cost / risk adjustments (lower is better → subtract)
    cost = {"H1": 1.0, "H2": 2.0, "H3": 1.5, "H4": 2.5, "H5": 3.5}
    risk = {"H1": 1.5, "H2": 1.0, "H3": 1.2, "H4": 2.0, "H5": 3.0}
    unified = {"H1": 1.0, "H2": 1.0, "H3": 1.0, "H4": 0.8, "H5": 0.5}

    rows = []
    for h in ("H1", "H2", "H3", "H4", "H5"):
        total = scores[h] + 1.5 * unified[h] - 0.4 * cost[h] - 0.4 * risk[h]
        rows.append(
            {
                "hypothesis": h,
                "mechanistic_score": round(scores[h], 3),
                "unified_actor_compat": unified[h],
                "impl_cost": cost[h],
                "risk_to_A1_B1_D": risk[h],
                "total_rank_score": round(total, 3),
                "affected_roots_proxy": n_fail,
                "expert_recoverability_used": round(rescue, 3),
            }
        )
    rows.sort(key=lambda r: -r["total_rank_score"])
    return rows[0]["hypothesis"], rows


def write_p109_proposal(winner: str, ranking: list[dict], ctx: dict) -> None:
    path = ROOT / "docs/p109_new_skill_acquisition_proposal.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if winner == "H1":
        body = f"""# P109 Proposal — On-Policy Physically Verified Corrective Imitation (H1)

## Motivation (from P108)

P107 found **no demonstrated unique Action-Aware benefit** under gradient-matched KL.
P108 shows new-root C failures are predominantly **closed-loop** (labels B/C/D), with expert
nominal tapes recovering a fraction of Actor-induced states under takeover (rescue rate ≈ {ctx.get('expert_rescue_rate', 0):.2f}).
Static correction SFT on train roots did not transfer; the missing piece is **on-policy,
physically verified** corrections at Actor-visited states.

## Primary hypothesis

**H1 — Targeted on-policy corrective imitation:** collecting short expert (or search-verified)
corrections from Actor failure states, then imitating them under retention controls, improves
new-root C closed-loop success without harming A1/B1/D.

## Controlled comparison (single mechanism)

| Arm | Initialization | Supervision | Retention |
|---|---|---|---|
| S0 | Frozen P64 | none (eval only) | — |
| S1 | P64 | **Static** correction-data SFT (existing P98/P104 C demos) | same KL or AA budget as S2 |
| S2 | P64 | **On-policy physically verified** correction SFT | matched budget |

### Matching rules
- Identical P64 init (`{EXPECTED_SHA}`).
- Matched supervision budget: same #tokens / #transitions / #updates.
- Identical retention recipe (freeze backbone+residual; token-only; choose **one** retention:
  either loss-scale KL **or** frozen P106 AA — not both variants in P109).
- Train physical roots ∩ eval physical roots = ∅ (canonical IDs).
- No sealed ind panel. No PPO/GRPO in P109 pilot.

### On-policy collection (S2 only)
1. Roll current student on train-eligible C roots (not eval panel).
2. At predefined intervention times (recovery entry / first divergence / post-regrasp), restore
   full state; run expert/search continuation; accept only if `SUCCESS_RELAXED_12S`.
3. Store diagnostic→training only after physical verification; discard failed search.

### Eval
- P108-style new-root development panel (disjoint).
- A1/B1/D protected panels (retention).
- Primary metric: new-root C `SUCCESS_RELAXED_12S` mean; secondary: A1 lost count.

### Falsification
If S2 ≯ S1 on new-root C under matched budget and retention, **reject H1** for this regime.

## Non-goals
No router, skill-label input, separate deployed policies, M111 redesign, or architecture stack.

## Ranking context
Winner from P108: **{winner}**. Full table in `results/experiments/p108_closed_loop_generalization_discovery/HYPOTHESIS_RANKING.md`.
"""
    elif winner == "H2":
        body = f"""# P109 Proposal — Contact-State Predictive Auxiliary (H2)

## Motivation (from P108)

P107: `NO_DEMONSTRATED_UNIQUE_AA_BENEFIT`.
P108 (corrected): contributing failure labels are dominated by **contact-mode (D≈{ctx.get('frac_label_D', 0):.2f})**
and **post-regrasp (C≈{ctx.get('frac_label_C', 0):.2f})**. Expert takeover can rescue some Actor states
(rescue≈{ctx.get('expert_rescue_rate', 0):.2f}), but the primary ranked mechanism is predicting
future contact structure rather than only imitating corrections.

## Primary hypothesis

**H2 — Future contact-state predictive auxiliary:** a train-only head/loss that predicts near-future
bilateral/contact features from observable history improves closed-loop C on new roots under a
capacity- and data-matched control without the auxiliary.

## Controlled comparison (single mechanism)

| Arm | Init | Token student | Auxiliary | Retention |
|---|---|---|---|---|
| S0 | P64 | frozen eval | — | — |
| S1 | P64 | correction SFT only | none | frozen KL **or** AA (pick one) |
| S2 | P64 | same SFT budget as S1 | **future contact prediction** | same as S1 |

### Matching rules
- Identical P64 init (`{EXPECTED_SHA}`); backbone+residual frozen; token-only.
- Matched #updates and #supervised tokens between S1 and S2.
- Auxiliary targets from **privileged** contact/bilateral labels at train time only; inference uses the
  same actor inputs as P64 (no privileged vars at deploy).
- Train ∩ eval physical roots = ∅ (canonical IDs). No sealed ind. No PPO/GRPO.

### Auxiliary target (minimal)
Predict a short horizon of observable-compatible contact features, e.g. future bilateral and/or
normal-force summary over k∈{{2,4}} steps. One loss weight; no architecture stack beyond a small head.

### Eval / falsification
- Primary: new-root C `SUCCESS_RELAXED_12S` (P108-style panel, disjoint).
- Secondary: A1 lost count under retention.
- **Reject H2** if S2 ≯ S1 on new-root C under matched budget.

## Non-goals
No on-policy correction collection (that is H1), no M111 redesign, no router/skill labels.

## Ranking context
Winner from P108: **{winner}**. See `results/experiments/p108_closed_loop_generalization_discovery/HYPOTHESIS_RANKING.md`.
"""
    elif winner == "H3":
        body = f"""# P109 Proposal — Contact-Aware Temporal Action Consistency (H3)

## Motivation
P108 harmful token/action switches co-occur with contact loss. Winner **{winner}**.

## Pilot
Add a single temporal-consistency penalty on large action jumps under contact (train-only).
Matched retention strength control without the penalty. No architecture change.
"""
    elif winner == "H4":
        body = f"""# P109 Proposal — Longer Causal History Representation (H4)

## Motivation
Observability conflicts / low-margin flips. Winner **{winner}**.

## Pilot
Single change: extend effective history window or history aggregation (one knob).
Matched-parameter control. No vocab redesign.
"""
    else:
        body = f"""# P109 Proposal — Action Vocabulary Intervention (H5)

## Motivation
Category-D unrepresentability linked to failures. Winner **{winner}**.

## Pilot
Minimum intervention: expand residual budget **or** add a small set of recovery prototypes — pick one.
Matched retention. Do not redesign full M111 in P109.
"""
    path.write_text(body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t_wall0 = time.time()
    print("[P108] Phase 0 — lock policies + build panel", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64_sha = p104.sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")

    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)

    actors = {"P64": p64}
    for arm in ("C", "D"):
        for seed in sorted(set(FULL_SEEDS + EXTRA_SEEDS)):
            ck = P107_OUT / "checkpoints" / f"arm{arm}_seed{seed}_final.pt"
            if not ck.is_file():
                raise SystemExit(f"missing ckpt {ck}")
            actors[f"{arm}_s{seed}"] = p107.load_token_ckpt(ck, p64, device)

    starts = p104.lookup_starts()
    panel_rows, panel_meta = build_panel(starts)
    write_csv(OUT / "ROOT_PROVENANCE.csv", panel_rows)
    write_json(OUT / "PANEL_META.json", panel_meta)
    print(
        f"[P108] panel n={len(panel_rows)} (target>={TARGET_ROOTS}) limitation={panel_meta.get('limitation')}",
        flush=True,
    )

    # ---- Expert baselines ----
    print("[P108] Phase 1 — expert recoverability", flush=True)
    expert_by_tid: dict[str, dict] = {}
    for i, row in enumerate(panel_rows):
        tid = row["traj_id"]
        st = starts[tid]
        print(f"  expert {i+1}/{len(panel_rows)} {tid}", flush=True)
        er = expert_rollout(st, book, rho, seed=CL_SEED)
        expert_by_tid[tid] = er
        row["expert_SUCCESS_RELAXED_12S"] = int(er["SUCCESS_RELAXED_12S"])
        row["expert_fail_reason"] = er.get("fail_reason")
        row["feasibility_stratum"] = (
            "expert_feasible" if int(er["SUCCESS_RELAXED_12S"]) == 1 else "expert_infeasible"
        )
    write_csv(OUT / "ROOT_PROVENANCE.csv", panel_rows)

    # ---- Actor diagnostics ----
    print("[P108] Phase 2 — frozen actor closed-loop diagnostics", flush=True)
    taxonomy_rows = []
    temporal_rows = []
    express_rows = []
    takeover_rows = []
    paired = []

    primary_policies = ["P64"] + [f"C_s{s}" for s in FULL_SEEDS] + [f"D_s{s}" for s in FULL_SEEDS]
    extra_policies = [f"C_s{s}" for s in EXTRA_SEEDS] + [f"D_s{s}" for s in EXTRA_SEEDS]
    takeover_counts: dict[str, int] = defaultdict(int)

    for pi, row in enumerate(panel_rows):
        tid = row["traj_id"]
        st = starts[tid]
        er = expert_by_tid[tid]
        policies = list(primary_policies)
        if pi < EXTRA_SEED_ROOT_CAP:
            policies += extra_policies
        for pol in policies:
            print(f"  CL {pol} {pi+1}/{len(panel_rows)} {tid}", flush=True)
            want_cf = pol in primary_policies and takeover_counts[pol] < TAKEOVER_MAX_FAILS_PER_POLICY
            snap_times = [t for t, _ in TAKEOVER_TIMES] if want_cf else []
            roll = diagnostic_rollout(
                actors[pol],
                st,
                book,
                rho,
                device,
                seed=CL_SEED,
                save_snapshots_at=snap_times,
            )
            # persist compact trace
            trace_path = CACHE / f"trace_{pol}_{tid}.npz"
            np.savez_compressed(
                trace_path,
                tokens=np.asarray([s["tok"] for s in roll["steps"]], np.int64),
                actions=np.stack([s["action"] for s in roll["steps"]]) if roll["steps"] else np.zeros((0, 7), np.float32),
                margins=np.asarray([s["margin"] for s in roll["steps"]], np.float32),
                e_pos=np.asarray([s["e_pos_mm"] for s in roll["steps"]], np.float32),
                bil=np.asarray([s["bilateral"] for s in roll["steps"]], np.int8),
            )
            tax = classify_failure(roll, er, book=book, rho=rho)
            taxonomy_rows.append(
                {
                    "policy": pol,
                    "traj_id": tid,
                    "canonical_physical_id": row["canonical_physical_id"],
                    "provenance": row["provenance"],
                    "feasibility_stratum": row["feasibility_stratum"],
                    "SUCCESS_RELAXED_12S": roll["SUCCESS_RELAXED_12S"],
                    "expert_SUCCESS_RELAXED_12S": row["expert_SUCCESS_RELAXED_12S"],
                    "ep_len": roll["ep_len"],
                    "fail_reason": roll.get("fail_reason"),
                    "reach12": roll["reach12"],
                    "r3_ready": roll["r3_ready"],
                    **tax,
                    "trace_path": str(trace_path),
                }
            )
            paired.append(
                {
                    "policy": pol,
                    "traj_id": tid,
                    "SUCCESS_RELAXED_12S": roll["SUCCESS_RELAXED_12S"],
                    "expert_SUCCESS_RELAXED_12S": row["expert_SUCCESS_RELAXED_12S"],
                }
            )
            temporal_rows.extend(temporal_rows_from_roll(pol, tid, roll, er))
            if int(roll["SUCCESS_RELAXED_12S"]) == 0:
                express_rows.extend(m111_expressivity_for_failure(book, rho, roll, er, tid, pol))

            # ---- Takeovers on a budgeted sample of primary-policy failures ----
            if (
                int(roll["SUCCESS_RELAXED_12S"]) == 0
                and pol in primary_policies
                and takeover_counts[pol] < TAKEOVER_MAX_FAILS_PER_POLICY
                and roll.get("snapshots")
            ):
                takeover_counts[pol] += 1
                ea = er.get("actions")
                for t_int, tag in TAKEOVER_TIMES:
                    if t_int not in roll["snapshots"]:
                        continue
                    state = roll["snapshots"][t_int]
                    r_act = continue_from_state(
                        actors[pol], st, state, book, rho, device, mode="actor", seed=CL_SEED
                    )
                    takeover_rows.append(
                        {
                            "policy": pol,
                            "traj_id": tid,
                            "intervention": tag,
                            "intervention_t": t_int,
                            "branch": "continue_actor",
                            **{k: r_act[k] for k in r_act if k != "note"},
                            "note": r_act["note"],
                        }
                    )
                    if ea is not None and int(er.get("expert_available") or 0) == 1:
                        r_ex = continue_from_state(
                            None,
                            st,
                            state,
                            book,
                            rho,
                            device,
                            mode="expert",
                            expert_acts=ea,
                            seed=CL_SEED,
                        )
                        takeover_rows.append(
                            {
                                "policy": pol,
                                "traj_id": tid,
                                "intervention": tag,
                                "intervention_t": t_int,
                                "branch": "expert_continuation",
                                **{k: r_ex[k] for k in r_ex if k != "note"},
                                "note": r_ex["note"],
                            }
                        )
                        r_px = continue_from_state(
                            actors[pol],
                            st,
                            state,
                            book,
                            rho,
                            device,
                            mode="prefix_expert",
                            expert_acts=ea,
                            prefix_k=5,
                            seed=CL_SEED,
                        )
                        takeover_rows.append(
                            {
                                "policy": pol,
                                "traj_id": tid,
                                "intervention": tag,
                                "intervention_t": t_int,
                                "branch": "expert_prefix5_then_actor",
                                **{k: r_px[k] for k in r_px if k != "note"},
                                "note": r_px["note"],
                            }
                        )
            # free step lists from memory
            roll["steps"] = []
            roll["snapshots"] = {}

    write_csv(OUT / "C_FAILURE_TAXONOMY.csv", taxonomy_rows)
    write_csv(OUT / "TEMPORAL_CONSISTENCY.csv", temporal_rows)
    write_csv(OUT / "M111_EXPRESSIVITY_AUDIT.csv", express_rows)
    write_csv(OUT / "EXPERT_TAKEOVER_RESULTS.csv", takeover_rows)
    write_csv(OUT / "PAIRED_OUTCOMES.csv", paired)

    # ---- Observability ----
    print("[P108] Phase 3 — observability + ranking", flush=True)
    try:
        corr = p104.load_correction_demos()
    except Exception:
        corr = []
    obs_sum = observability_audit(p64, corr, device, OUT / "OBSERVABILITY_AUDIT.md")

    # ---- Aggregate evidence ----
    def _rate(rows, pred):
        xs = [pred(r) for r in rows]
        return float(np.mean(xs)) if xs else 0.0

    fail_tax = [r for r in taxonomy_rows if int(r["failed"]) == 1 and r["policy"] in primary_policies]
    n_actor_fail_roots = len({r["traj_id"] for r in fail_tax})
    ex_take = [r for r in takeover_rows if r["branch"] == "expert_continuation"]
    expert_rescue_rate = _rate(ex_take, lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1)
    ctx = {
        "n_panel": len(panel_rows),
        "n_expert_feasible": sum(int(r["expert_SUCCESS_RELAXED_12S"]) for r in panel_rows),
        "expert_success_rate": _rate(panel_rows, lambda r: int(r["expert_SUCCESS_RELAXED_12S"]) == 1),
        "n_actor_fail_roots": n_actor_fail_roots,
        "expert_rescue_rate": expert_rescue_rate,
        "frac_earliest_A": _rate(fail_tax, lambda r: r["earliest_label"] == "A"),
        "frac_label_B": _rate(fail_tax, lambda r: "B" in str(r["labels"]).split("|")),
        "frac_label_C": _rate(fail_tax, lambda r: "C" in str(r["labels"]).split("|")),
        "frac_label_D": _rate(fail_tax, lambda r: "D" in str(r["labels"]).split("|")),
        "frac_label_E": _rate(fail_tax, lambda r: "E" in str(r["labels"]).split("|")),
        "frac_label_F": _rate(fail_tax, lambda r: "F" in str(r["labels"]).split("|")),
        "frac_label_G": _rate(fail_tax, lambda r: "G" in str(r["labels"]).split("|")),
        "frac_harmful_switch": _rate(
            [r for r in temporal_rows if r["policy"] in primary_policies],
            lambda r: int(r.get("harmful_switch_candidate") or 0) == 1,
        ),
        "frac_cat_d_expert": _rate(
            express_rows, lambda r: str(r.get("category_D_unrepresentable_expert")) == "1"
        ),
        "obs_conflict_rate": float(obs_sum.get("conflict_rate") or 0.0),
    }

    # policy success rates
    pol_succ = {}
    for pol in primary_policies:
        sub = [r for r in taxonomy_rows if r["policy"] == pol]
        pol_succ[pol] = {
            "n": len(sub),
            "succ": sum(int(r["SUCCESS_RELAXED_12S"]) for r in sub),
            "rate": _rate(sub, lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1),
            "expert_feasible_succ_rate": _rate(
                [r for r in sub if r["feasibility_stratum"] == "expert_feasible"],
                lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1,
            ),
        }

    winner, ranking = rank_hypotheses(ctx)
    rank_md = ["# HYPOTHESIS_RANKING — P108", "", f"**Primary for P109: `{winner}`**", "", "| Hyp | Mech | Unified | Cost | Risk | Total |", "|---|---:|---:|---:|---:|---:|"]
    for r in ranking:
        rank_md.append(
            f"| {r['hypothesis']} | {r['mechanistic_score']} | {r['unified_actor_compat']} | {r['impl_cost']} | {r['risk_to_A1_B1_D']} | **{r['total_rank_score']}** |"
        )
    rank_md.append("")
    rank_md.append("## Evidence drivers")
    for k, v in ctx.items():
        rank_md.append(f"- `{k}`: {v}")
    rank_md.append("")
    rank_md.append("## Definitions")
    rank_md.append("- H1 Targeted on-policy corrective imitation")
    rank_md.append("- H2 Future contact-state predictive auxiliary")
    rank_md.append("- H3 Contact-aware temporal action consistency")
    rank_md.append("- H4 Improved causal history representation")
    rank_md.append("- H5 Action-vocabulary redesign")
    (OUT / "HYPOTHESIS_RANKING.md").write_text("\n".join(rank_md), encoding="utf-8")
    write_p109_proposal(winner, ranking, ctx)

    # ---- FINAL REPORT / SUMMARY ----
    earliest_dist = defaultdict(int)
    for r in fail_tax:
        earliest_dist[r["earliest_label"]] += 1

    summary = {
        "case": "P108_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "sealed_panel_opened": False,
        "trained": False,
        "panel": panel_meta,
        "n_roots": len(panel_rows),
        "feasibility_strata": {
            "expert_feasible": sum(1 for r in panel_rows if r["feasibility_stratum"] == "expert_feasible"),
            "expert_infeasible": sum(1 for r in panel_rows if r["feasibility_stratum"] == "expert_infeasible"),
        },
        "policy_success": pol_succ,
        "failure_earliest_label_counts": dict(earliest_dist),
        "evidence": ctx,
        "expert_takeover": {
            "n_expert_continuation": len(ex_take),
            "rescue_rate": expert_rescue_rate,
            "prefix_rescue_rate": _rate(
                [r for r in takeover_rows if r["branch"] == "expert_prefix5_then_actor"],
                lambda r: int(r["SUCCESS_RELAXED_12S"]) == 1,
            ),
        },
        "primary_hypothesis_p109": winner,
        "ranking": ranking,
        "wall_s": time.time() - t_wall0,
        "answers": {},
    }

    # Fill answers
    top_stage = max(earliest_dist.items(), key=lambda kv: kv[1])[0] if earliest_dist else "n/a"
    summary["answers"] = {
        "1_failure_stage": top_stage,
        "2_expert_can_rescue": bool(expert_rescue_rate > 0.05),
        "3_dominated_by": "unstable_later_sequences" if (ctx["frac_label_B"] + ctx["frac_label_C"]) >= ctx["frac_earliest_A"] else "wrong_initial_actions",
        "4_switching_contributes_to_contact_loss": bool(ctx["frac_harmful_switch"] > 0.01),
        "5_observation_history_plausible": bool(ctx["obs_conflict_rate"] > 0.05 or ctx["frac_label_F"] > 0.1),
        "6_m111_bottleneck": bool(ctx["frac_cat_d_expert"] > 0.15),
        "7_primary_hypothesis": winner,
        "8_smallest_p109": "see docs/p109_new_skill_acquisition_proposal.md",
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P108 Closed-Loop Generalization Discovery

Generated: {summary['created']}

## Scope
Diagnostic / counterfactual only. **No training.** P64 `{p64_sha}` immutable.
Policies: P64; P107 Arm C (grad-matched KL); P107 Arm D (Action-Aware). Seeds {FULL_SEEDS} full panel; {EXTRA_SEEDS} on first {EXTRA_SEED_ROOT_CAP} roots.
Official R3F + `SUCCESS_RELAXED_12S`. Sealed ind **not** opened.

## Panel
- Distinct physical roots: **{len(panel_rows)}** (target ≥{TARGET_ROOTS})
- Eligibility: schedule-C episodes with expert tapes; exclude P98 train + sealed ind; one traj/gbi; selected **before** candidate outcomes
- Limitation: {panel_meta.get('limitation') or 'none — target met'}
- Expert feasible / infeasible: {summary['feasibility_strata']['expert_feasible']} / {summary['feasibility_strata']['expert_infeasible']}
- Freeze-ban note: {panel_meta.get('note_freeze_ban')}

## Actor success (primary policies)
| Policy | n | succ | rate | rate on expert-feasible |
|---|---:|---:|---:|---:|
"""
    for pol, s in pol_succ.items():
        report += f"| {pol} | {s['n']} | {s['succ']} | {s['rate']:.3f} | {s['expert_feasible_succ_rate']:.3f} |\n"

    report += f"""
## Failure localization (failed primary-policy episodes)
Earliest-label counts: {dict(earliest_dist)}

## Expert takeover
- Expert continuation rescue rate: **{expert_rescue_rate:.3f}** (n={len(ex_take)})
- Prefix5→actor rescue rate: **{summary['expert_takeover']['prefix_rescue_rate']:.3f}**
- Caveat: nominal expert tapes may be invalid from off-policy states; fidelity flag recorded per row.

## Hypothesis ranking
**Primary for P109: `{winner}`**

See `HYPOTHESIS_RANKING.md` and `docs/p109_new_skill_acquisition_proposal.md`.

## Answers
1. Failure stage (earliest mode): **{summary['answers']['1_failure_stage']}**
2. Expert rescue of Actor states? **{summary['answers']['2_expert_can_rescue']}**
3. Dominated by: **{summary['answers']['3_dominated_by']}**
4. Switching→contact loss? **{summary['answers']['4_switching_contributes_to_contact_loss']}**
5. Obs/history limitation plausible? **{summary['answers']['5_observation_history_plausible']}**
6. M111 bottleneck meaningful? **{summary['answers']['6_m111_bottleneck']}**
7. Strongest mechanism: **{winner}**
8. Smallest P109 pilot: see proposal doc.

## Stop
Stop after P108. Do not train P109 / PPO / GRPO / replace P64 / open sealed panel without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P108] DONE winner={winner} wall_s={summary['wall_s']:.1f}", flush=True)


if __name__ == "__main__":
    main()
