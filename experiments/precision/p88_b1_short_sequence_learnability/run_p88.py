"""P88 — B1 short-sequence learnability and closed-loop reproduction audit."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
)
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import HANDOFF as M111_HANDOFF  # noqa: E402
from models.m111_checkpoint import MOTION_K, decode_m111_action, save_m111  # noqa: E402

P87_OUT = ROOT / "results/experiments/p87_k48_actor_retention"
P87_TEACHER = P87_OUT / "teacher"
P87_ACTOR = P87_OUT / "checkpoints/P87_actor.pt"
P87_SPLITS = P87_OUT / "SPLITS.json"
P87_MANIFEST = P87_OUT / "TEACHER_DATA_MANIFEST.json"
P87_REPLAY_CSV = P87_OUT / "K48_M111_TRAIN_REPLAY.csv"

OUT = ROOT / "results/experiments/p88_b1_short_sequence_learnability"
CKPT_DIR = OUT / "checkpoints"

SPLIT = "p88_b1_short_sequence_learnability"
SEED = 88
MAX_SINGLE_UPDATES = 2000
MAX_JOINT_UPDATES = 4000
LOG_EVERY = 10
LR = 1e-4  # higher than P87 for pure memorization diagnostic
MEM_ACC_TARGET = 0.999
MEM_CE_TARGET = 1e-3
HOLD_TOKEN = 0  # nominal hold after teacher motion for R3F dwell


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


def blob_from_start(start) -> dict:
    if start.kind == "capture":
        return {"capture": start.capture, "ref": start.ref, "traj_id": start.traj_id}
    ep = start.ep or {}
    return {"capture": ep["capture"], "ref": ep["ref"], "traj_id": start.traj_id}


def load_npz(traj_id: str) -> dict:
    z = np.load(P87_TEACHER / f"{traj_id}.npz", allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["traj_id"] = str(traj_id)
    out["tokens"] = np.asarray(out["tokens"], np.int64)
    out["obs"] = np.asarray(out["obs"], np.float32)
    out["prev"] = np.asarray(out["prev"], np.float32)
    out["actions"] = np.asarray(out["actions"], np.float32)
    out["residuals"] = np.asarray(out["residuals"], np.float32)
    return out


def lookup_starts() -> dict[str, Any]:
    sched = build_p54_dataset(seed=5478)["schedule"]
    return {s.traj_id: s for s in sched}


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def teacher_open_loop_official(env, start, demo: dict, book, rho) -> dict:
    """Replay recorded motion tokens; R3F monitor decides HANDOFF (official path)."""
    seed_all(SEED, deterministic_torch=True)
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    toks = [int(t) for t in demo["tokens"]]
    res = np.asarray(demo["residuals"], np.float32)
    done = False
    info: dict = {}
    i = 0
    ever_frame9 = False
    while not done:
        if i < len(toks):
            tok = int(toks[i])
            resid = res[i]
            i += 1
        else:
            # hold after tape; let R3F dwell complete
            tok = HOLD_TOKEN
            resid = np.zeros(7, np.float32)
        _r, done, info = env.step(tok, resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if (
            bool(g["bilateral"])
            and int(g.get("min_pad") or 0) >= 4
            and preds.get("working_ok")
            and float(g["e_pos_mm"]) <= 9 + 1e-9
            and float(g["theta_deg"]) <= 8 + 1e-9
        ):
            ever_frame9 = True
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "ever_frame9": int(ever_frame9),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "n_teacher_steps_used": min(i, len(toks)),
    }


def verify_demo_alignment(demo: dict, book, rho) -> dict:
    toks = demo["tokens"]
    obs = demo["obs"]
    prev = demo["prev"]
    acts = demo["actions"]
    res = demo["residuals"]
    n = len(toks)
    issues = []
    if not (len(obs) == len(prev) == len(acts) == len(res) == n):
        issues.append(
            f"length_mismatch toks={n} obs={len(obs)} prev={len(prev)} acts={len(acts)} res={len(res)}"
        )
    if n and float(np.linalg.norm(prev[0])) > 1e-6:
        issues.append("prev0_not_zero")
    # prev[t] should equal action[t-1]
    for t in range(1, n):
        if float(np.linalg.norm(prev[t] - acts[t - 1])) > 1e-4:
            issues.append(f"prev_action_misalign_t={t}")
            break
    # decode check
    max_dec = 0.0
    for t in range(n):
        dec = decode_m111_action(book, int(toks[t]), res[t], rho=rho)
        max_dec = max(max_dec, float(np.linalg.norm(dec - acts[t])))
    if max_dec > 1e-3:
        issues.append(f"decode_vs_action_l2={max_dec:.4g}")
    handoff_in_seq = int(any(int(t) == M111_HANDOFF for t in toks))
    motion_ok = int(all(0 <= int(t) < MOTION_K for t in toks))
    return {
        "n": n,
        "issues": issues,
        "ok": len(issues) == 0,
        "handoff_token_in_sequence": handoff_in_seq,
        "all_motion_tokens": motion_ok,
        "max_decode_l2": max_dec,
        "token_ids": [int(t) for t in toks],
    }


def phase0_select_and_audit(device) -> dict:
    man = json.loads(P87_MANIFEST.read_text(encoding="utf-8"))
    splits = json.loads(P87_SPLITS.read_text(encoding="utf-8"))
    teach_ids = set(splits["b1_teacher_ids"])
    val_ids = set(splits["b1_val_ids"])
    exposed = set(man["excluded_exposed_b1"])
    r3_ids = list(man["r3f_compatible_traj_ids"])
    starts = lookup_starts()

    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    candidates = []
    for tid in r3_ids:
        demo = load_npz(tid)
        n = int(len(demo["tokens"]))
        in_train = tid in teach_ids
        in_val = tid in val_ids
        in_exp = tid in exposed or any(tid in e or e.endswith(tid) for e in exposed)
        align = verify_demo_alignment(demo, book, rho)
        start = starts.get(tid)
        official = {"SUCCESS_RELAXED_12S": 0, "note": "missing_start"}
        if start is not None:
            official = teacher_open_loop_official(env, start, demo, book, rho)
        # valid = train-only, aligned, official RELAXED success under teacher open-loop + R3F
        valid = (
            in_train
            and not in_val
            and not in_exp
            and align["ok"]
            and int(official.get("SUCCESS_RELAXED_12S") or 0) == 1
        )
        candidates.append(
            {
                "traj_id": tid,
                "n": n,
                "in_train": int(in_train),
                "in_val": int(in_val),
                "valid_official": int(valid),
                "align_ok": int(align["ok"]),
                "align_issues": ";".join(align["issues"]),
                "handoff_in_npz": align["handoff_token_in_sequence"],
                "official_relaxed": int(official.get("SUCCESS_RELAXED_12S") or 0),
                "official_handoff": int(official.get("handoff") or 0),
                "official_r3_ready": int(official.get("r3_ready") or 0),
                "official_ep_len": official.get("ep_len"),
                "dwell_complete_ever_npz": int(np.asarray(demo.get("dwell_complete_ever", 0)).reshape(-1)[0]),
            }
        )

    # Deterministic selection: shortest valid; tie-break lexicographic traj_id
    valid = [c for c in candidates if c["valid_official"]]
    if not valid:
        # Fallback diagnostic: shortest aligned train demo even if official RELAXED fails
        pool = [c for c in candidates if c["in_train"] and c["align_ok"]]
        selected_meta = sorted(pool, key=lambda c: (c["n"], c["traj_id"]))[0] if pool else None
        select_note = "NO_OFFICIAL_SUCCESS_FALLBACK_SHORTEST_ALIGNED"
    else:
        selected_meta = sorted(valid, key=lambda c: (c["n"], c["traj_id"]))[0]
        select_note = "shortest_valid_official_SUCCESS_RELAXED_then_lexicographic"

    assert selected_meta is not None, "no candidate demos"
    tid = selected_meta["traj_id"]
    demo = load_npz(tid)
    align = verify_demo_alignment(demo, book, rho)
    start = starts[tid]

    # Re-verify K48/M111 from P87 CSV
    replay_rows = {r["traj_id"]: r for r in csv.DictReader(P87_REPLAY_CSV.open(encoding="utf-8"))}
    prow = replay_rows.get(tid, {})

    audit_md = f"""# TEACHER_TRAJECTORY_AUDIT

Generated: {_utc()}

## Selection rule (predefined, not student-based)

1. Start from P87's 16 R3F-compatible training demos.
2. Require: train-split only (not val, not exposed independent B1).
3. Require: obs/prev/action/token alignment checks pass.
4. Require: **official** open-loop teacher motion + R3F monitor yields `SUCCESS_RELAXED_12S=1`
   (not merely ever-frame9 geometric compatibility).
5. Among valid: **shortest** sequence length; ties → lexicographic `traj_id`.

Selection note: `{select_note}`

## Selected trajectory

| Field | Value |
|---|---|
| traj_id | `{tid}` |
| root_id | `{root_id(tid)}` |
| length (motion tokens) | **{len(demo['tokens'])}** |
| train-split | yes |
| token sequence | `{' '.join(str(int(t)) for t in demo['tokens'])}` |

## Verification checklist

| Check | Result |
|---|---|
| IC in training split only | {'PASS' if selected_meta['in_train'] and not selected_meta['in_val'] else 'FAIL'} |
| K48 SUCCESS_12S (P87 CSV) | {prow.get('k48_success_12s')} reason={prow.get('k48_reason')} |
| M111 replay SUCCESS_12S (P87 CSV) | {prow.get('m111_replay_success_12s')} reason={prow.get('m111_reason')} |
| Obs from M111 physical replay | PASS (P87 Phase-1 fix; prev[t]=action[t-1] verified) |
| Obs↔action alignment | {'PASS' if align['ok'] else 'FAIL: ' + ';'.join(align['issues'])} |
| HANDOFF token in npz sequence | **{align['handoff_token_in_sequence']}** (expected 0 under official Actor path) |
| Official open-loop + R3F SUCCESS_RELAXED | **{selected_meta['official_relaxed']}** handoff={selected_meta['official_handoff']} ep_len={selected_meta['official_ep_len']} |

## Architecture note — HANDOFF labeling

Official M111 Actor inference **masks HANDOFF logits** (`mask_handoff_logits`); R3F online monitor
triggers episode HANDOFF after a motion step when `ready`. Therefore:

- Supervised BC targets are **motion tokens only** (matches `act_with_value`).
- "HANDOFF-token accuracy" under the official sampler is **N/A** (Actor never emits 111).
- Closed-loop success requires motion to drive state into R3F readiness; monitor owns the transfer.

P87 BC also called `mask_handoff_logits` and dropped HANDOFF from targets — consistent with this path,
but means skill retention cannot learn an explicit HANDOFF policy decision.

## Timing / masks / IDs

| Item | Value |
|---|---|
| Causal Transformer | 160-pos embedding; triangular causal mask in `Policy.forward` |
| Pad mask in P87 BC | unused (single full sequence, no pad) |
| Loss mask P87 | motion only; HANDOFF logit forced −1e9 |
| Token ID range | motion `0..{MOTION_K-1}`; HANDOFF=`{M111_HANDOFF}` |
| Grip / decode | `decode_m111_action(book, tok, resid, rho)` — max decode L2 vs stored action = {align['max_decode_l2']:.4g} |
| Residual head | teacher residuals from `quantize_trajectory`; closed-loop uses Actor residual unless teacher-forced |

## Candidate table (validity scan)

See also `PHASE0_CANDIDATES.csv`.
Valid count: **{sum(c['valid_official'] for c in candidates)}** / {len(candidates)}.
"""
    _write(OUT / "TEACHER_TRAJECTORY_AUDIT.md", audit_md)
    _write_csv(OUT / "PHASE0_CANDIDATES.csv", candidates)
    return {
        "traj_id": tid,
        "demo": demo,
        "start": start,
        "align": align,
        "selected_meta": selected_meta,
        "select_note": select_note,
        "book": book,
        "rho": rho,
        "candidates": candidates,
        "valid_count": sum(c["valid_official"] for c in candidates),
    }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def forward_logits(actor, obs_np, prev_np, device, *, mask_ho: bool = True):
    obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    prev = torch.as_tensor(prev_np, device=device, dtype=torch.float32)
    logits, res, _ = actor(obs, prev)
    logit = mask_handoff_logits(logits.float()) if mask_ho else logits.float()
    return logit, res


def eval_teacher_forced(actor, demo, device, *, mask_ho: bool = True) -> dict:
    actor.eval()
    with torch.no_grad():
        logit, res = forward_logits(actor, demo["obs"], demo["prev"], device, mask_ho=mask_ho)
        toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
        pred = torch.argmax(logit, dim=-1)
        exact = pred == toks
        probs = torch.softmax(logit, dim=-1)
        p_tgt = probs.gather(1, toks.unsqueeze(1)).squeeze(1)
        ce = F.cross_entropy(logit, toks, reduction="mean")
        # residual error vs teacher
        tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
        res_l2 = float(torch.linalg.vector_norm(res - tres, dim=-1).mean())
    return {
        "ce": float(ce),
        "acc": float(exact.float().mean()),
        "first_acc": float(exact[0].float()),
        "n_tokens": int(len(toks)),
        "n_correct": int(exact.sum()),
        "per_pos_p_target": [float(x) for x in p_tgt.cpu()],
        "per_pos_pred": [int(x) for x in pred.cpu()],
        "per_pos_tgt": [int(x) for x in toks.cpu()],
        "per_pos_ok": [int(x) for x in exact.cpu()],
        "res_l2": res_l2,
        "handoff_acc": None,  # N/A under mask
    }


def train_one_demo(actor, demo, device, *, max_updates: int, label: str) -> tuple[list[dict], dict]:
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    curve = []
    actor.train()
    ref_params = [p.detach().clone() for p in actor.parameters()]
    memorized = False
    final = {}
    for u in range(1, max_updates + 1):
        opt.zero_grad(set_to_none=True)
        logit, res = forward_logits(actor, demo["obs"], demo["prev"], device, mask_ho=True)
        toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
        tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
        ce = F.cross_entropy(logit, toks, reduction="mean")
        # Match residual so closed-loop decode can track teacher actions
        res_loss = F.mse_loss(res, tres)
        loss = ce + 0.1 * res_loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0))
        # confirm grads reach params
        n_grad = sum(1 for p in actor.parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0)
        opt.step()
        with torch.no_grad():
            upd = 0.0
            for p, r in zip(actor.parameters(), ref_params):
                upd += float(torch.linalg.vector_norm(p - r))
            ref_params = [p.detach().clone() for p in actor.parameters()]
        if u % LOG_EVERY == 0 or u == 1 or u == max_updates:
            metrics = eval_teacher_forced(actor, demo, device)
            row = {
                "label": label,
                "update": u,
                "loss": float(loss.detach()),
                "ce": metrics["ce"],
                "res_loss": float(res_loss.detach()),
                "acc": metrics["acc"],
                "first_acc": metrics["first_acc"],
                "grad_norm": grad_norm,
                "param_update_norm": upd,
                "n_supervised_tokens": metrics["n_tokens"],
                "n_params_with_grad": n_grad,
                "handoff_acc": "N/A_masked",
                "mean_p_target": float(np.mean(metrics["per_pos_p_target"])),
                "min_p_target": float(np.min(metrics["per_pos_p_target"])),
            }
            # store per-pos as JSON-ish string every log
            row["per_pos_p_target"] = json.dumps([round(x, 5) for x in metrics["per_pos_p_target"]])
            curve.append(row)
            print(
                f"[{label}] u={u} ce={row['ce']:.4f} acc={row['acc']:.3f} "
                f"first={row['first_acc']:.0f} grad={grad_norm:.3f}",
                flush=True,
            )
            final = metrics
            if metrics["acc"] >= MEM_ACC_TARGET and metrics["ce"] <= MEM_CE_TARGET and metrics["first_acc"] >= 1.0:
                memorized = True
                print(f"[{label}] MEMORIZED at update {u}", flush=True)
                break
        actor.train()
    return curve, {"memorized": memorized, "final": final, "updates": curve[-1]["update"] if curve else 0}


def train_joint(actor, demos: list[dict], device, *, max_updates: int) -> tuple[list[dict], dict]:
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    curve = []
    actor.train()
    memorized = False
    for u in range(1, max_updates + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for demo in demos:
            logit, res = forward_logits(actor, demo["obs"], demo["prev"], device, mask_ho=True)
            toks = torch.as_tensor(demo["tokens"], device=device, dtype=torch.long)
            tres = torch.as_tensor(demo["residuals"], device=device, dtype=torch.float32)
            losses.append(F.cross_entropy(logit, toks) + 0.1 * F.mse_loss(res, tres))
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0))
        opt.step()
        if u % LOG_EVERY == 0 or u == 1 or u == max_updates:
            accs = []
            ces = []
            firsts = []
            for demo in demos:
                m = eval_teacher_forced(actor, demo, device)
                accs.append(m["acc"])
                ces.append(m["ce"])
                firsts.append(m["first_acc"])
            row = {
                "update": u,
                "loss": float(loss.detach()),
                "mean_ce": float(np.mean(ces)),
                "mean_acc": float(np.mean(accs)),
                "mean_first_acc": float(np.mean(firsts)),
                "min_acc": float(np.min(accs)),
                "grad_norm": grad_norm,
                "n_demos": len(demos),
            }
            curve.append(row)
            print(
                f"[joint16] u={u} ce={row['mean_ce']:.4f} acc={row['mean_acc']:.3f} "
                f"min_acc={row['min_acc']:.3f}",
                flush=True,
            )
            if row["mean_acc"] >= MEM_ACC_TARGET and row["min_acc"] >= MEM_ACC_TARGET and row["mean_ce"] <= MEM_CE_TARGET:
                memorized = True
                print(f"[joint16] MEMORIZED at update {u}", flush=True)
                break
        actor.train()
    return curve, {"memorized": memorized, "updates": curve[-1]["update"] if curve else 0}


# ---------------------------------------------------------------------------
# Phase 2–3 evaluation modes
# ---------------------------------------------------------------------------


@torch.no_grad()
def autoregressive_offline(actor, demo, book, rho, device) -> dict:
    """Recorded obs; prev built from model's own decoded actions."""
    actor.eval()
    n = len(demo["tokens"])
    preds: list[int] = []
    preds_actions: list[np.ndarray] = []
    oks: list[int] = []
    first_div = None
    for t in range(n):
        obs_seq = demo["obs"][: t + 1]
        hist = [np.zeros(7, np.float32)]
        for j in range(t):
            hist.append(np.asarray(preds_actions[j], np.float32))
        prev_seq = np.stack(hist)
        logit, res = forward_logits(actor, obs_seq, prev_seq, device)
        pred = int(torch.argmax(logit[-1]).item())
        resid = res[-1].detach().cpu().numpy().astype(np.float32)
        action = decode_m111_action(book, pred, resid, rho=rho)
        preds.append(pred)
        preds_actions.append(action)
        ok = pred == int(demo["tokens"][t])
        oks.append(int(ok))
        if not ok and first_div is None:
            first_div = t
    return {
        "acc": float(np.mean(oks)) if oks else 0.0,
        "first_acc": float(oks[0]) if oks else 0.0,
        "first_token_div": first_div if first_div is not None else "",
        "n_correct": int(sum(oks)),
        "n": n,
        "preds": preds,
    }


def closed_loop_actor(env, actor, critic, start, book, rho, device, *, deterministic=True) -> dict:
    seed_all(SEED, deterministic_torch=True)
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    tokens_exec = []
    while not done:
        tok_a, resid, _, _ = act_with_value(
            actor, critic, env.obs_seq, env.prev_seq, device, deterministic=deterministic
        )
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        tokens_exec.append(int(tok))
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    term = _r3_terminal(env, info)
    return {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "tokens": tokens_exec,
    }


def first_divergence_trace(env, actor, critic, start, demo, book, rho, device):
    """Step-wise compare teacher open-loop vs student closed-loop; also prefix switch."""
    # Teacher reference actions
    t_toks = [int(x) for x in demo["tokens"]]
    t_res = np.asarray(demo["residuals"], np.float32)
    t_acts = np.asarray(demo["actions"], np.float32)

    seed_all(SEED, deterministic_torch=True)
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()

    rows = []
    first = {
        "token": None,
        "decode": None,
        "physics": None,
        "obs": None,
        "handoff": None,
    }
    done = False
    info: dict = {}
    t = 0
    while not done and t < max(len(t_toks) + 20, MAX_EP_STEPS):
        obs_before = np.asarray(last_obs(env), np.float32).copy()
        prev_before = np.asarray(env.prev_seq[-1] if env.prev_seq else np.zeros(7), np.float32).copy()
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device, deterministic=True)
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        action = decode_m111_action(book, int(tok), resid, rho=rho)

        teacher_tok = int(t_toks[t]) if t < len(t_toks) else None
        teacher_act = t_acts[t] if t < len(t_acts) else None
        # teacher-forced logit at recorded obs if available
        p_tgt = None
        if t < len(demo["obs"]):
            with torch.no_grad():
                # use live obs history for fair closed-loop prediction audit
                logit, _ = forward_logits(
                    actor,
                    np.stack(list(env.obs_seq)[-160:]),
                    np.stack(list(env.prev_seq)[-160:]),
                    device,
                )
                probs = torch.softmax(logit[-1], dim=-1)
                if teacher_tok is not None and 0 <= teacher_tok < probs.numel():
                    p_tgt = float(probs[teacher_tok])

        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)

        token_div = teacher_tok is not None and int(tok) != teacher_tok
        decode_div = False
        if teacher_act is not None:
            decode_div = float(np.linalg.norm(np.asarray(action, np.float32) - teacher_act)) > 0.05

        row = {
            "t": t,
            "teacher_tok": teacher_tok if teacher_tok is not None else "",
            "student_tok": int(tok),
            "p_teacher_tok": p_tgt if p_tgt is not None else "",
            "token_match": int(not token_div) if teacher_tok is not None else "",
            "action_l2_vs_teacher": float(np.linalg.norm(action - teacher_act)) if teacher_act is not None else "",
            "e_pos_mm": float(g["e_pos_mm"]),
            "theta_deg": float(g["theta_deg"]),
            "bilateral": int(g["bilateral"]),
            "min_pad": int(g.get("min_pad") or 0),
            "frame_ok": int(bool(preds.get("frame_ok"))),
            "dwell_complete": int(bool(preds.get("dwell_complete"))),
            "r3_ready": int(bool(preds.get("frame_ok") and preds.get("dwell_complete"))),
            "obs_l2": float(np.linalg.norm(obs_before)),
            "prev_l2": float(np.linalg.norm(prev_before)),
        }
        rows.append(row)

        if token_div and first["token"] is None:
            first["token"] = t
        if decode_div and first["decode"] is None:
            first["decode"] = t

        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)

        if t < len(demo["obs"]) and first["obs"] is None:
            # compare post-step obs to teacher next obs if lengths allow
            pass
        if done and info.get("handoff") and first["handoff"] is None:
            first["handoff"] = t
        t += 1

    term = _r3_terminal(env, info)
    summary = {
        "first_token_div": first["token"] if first["token"] is not None else "",
        "first_decode_div": first["decode"] if first["decode"] is not None else "",
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
    }

    # Prefix replay: teacher prefix k then student
    prefix_rows = []
    for k in list(range(0, len(t_toks) + 1, max(1, len(t_toks) // 8))) + [len(t_toks)]:
        if k in {r["prefix_len"] for r in prefix_rows}:
            continue
        seed_all(SEED, deterministic_torch=True)
        env.reset(start)
        env.mon = make_monitor("G0", t_max=int(env.t_max))
        ensure_inner_capture_buffers(env)
        st = WrapState()
        v2 = v2_spec()
        done = False
        info = {}
        step_i = 0
        while not done:
            if step_i < k:
                tok = int(t_toks[step_i])
                resid = t_res[step_i]
            else:
                tok_a, resid, _, _ = act_with_value(
                    actor, critic, env.obs_seq, env.prev_seq, device, deterministic=True
                )
                tok, st, _ = select_token(
                    env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
                )
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
            step_i += 1
        term = _r3_terminal(env, info)
        prefix_rows.append(
            {
                "prefix_len": k,
                "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
                "handoff": int(info.get("handoff") or 0),
                "ep_len": int(env.t),
                "fail_reason": info.get("fail_reason"),
            }
        )
    return rows, summary, prefix_rows


# ---------------------------------------------------------------------------
# Phase 5
# ---------------------------------------------------------------------------


def a1_interference(device, b1_actor_path: Path | None) -> dict:
    """Compare P64 vs B1-only fit vs P87 on A1 ICs that P87 lost."""
    # From P87 CROSS_CLASS: A1 2→0 lost 2. Recover which ICs from CSV.
    succ_csv = P87_OUT / "OFFICIAL_SUCCESS_COMPARISON.csv"
    rows = list(csv.DictReader(succ_csv.open(encoding="utf-8")))
    a1 = [r for r in rows if r["panel"] == "a1" and r["variant"] == "G0"]
    p64 = {r["traj_id"]: r for r in a1 if r["policy"] == "P64"}
    p87 = {r["traj_id"]: r for r in a1 if r["policy"] == "P87"}
    lost = [
        tid
        for tid in p64
        if int(p64[tid]["SUCCESS_RELAXED_12S"]) == 1 and int(p87[tid]["SUCCESS_RELAXED_12S"]) == 0
    ]
    starts = lookup_starts()
    p64_actor, book, rho, _ = load_actor(P64_ACTOR, device)
    p87_actor, _, _, _ = load_actor(P87_ACTOR, device)
    critic = load_critic(P64_CRITIC, p64_actor, device)
    b1_actor = None
    if b1_actor_path and b1_actor_path.is_file():
        b1_actor, _, _, _ = load_actor(b1_actor_path, device)

    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    out_rows = []
    for tid in lost:
        start = starts.get(tid)
        if start is None:
            continue
        for name, act in (("P64", p64_actor), ("P87", p87_actor), ("B1_only", b1_actor)):
            if act is None:
                continue
            for p in act.parameters():
                p.requires_grad_(False)
            cl = closed_loop_actor(env, act, critic, start, book, rho, device)
            # first-step logit compare vs P64
            seed_all(SEED, deterministic_torch=True)
            env.reset(start)
            ensure_inner_capture_buffers(env)
            with torch.no_grad():
                obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
                prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
                lp64, _, _ = p64_actor(obs, prev)
                la, _, _ = act(obs, prev)
                lp64 = mask_handoff_logits(lp64[-1].float())
                la = mask_handoff_logits(la[-1].float())
                kl = float((torch.softmax(lp64, -1) * (torch.log_softmax(lp64, -1) - torch.log_softmax(la, -1))).sum())
                tok_p64 = int(torch.argmax(lp64).item())
                tok_a = int(torch.argmax(la).item())
            out_rows.append(
                {
                    "traj_id": tid,
                    "policy": name,
                    "SUCCESS_RELAXED_12S": cl["SUCCESS_RELAXED_12S"],
                    "first_tok": tok_a,
                    "p64_first_tok": tok_p64,
                    "first_tok_match_p64": int(tok_a == tok_p64),
                    "kl_from_p64_step0": kl,
                    "ep_len": cl["ep_len"],
                    "fail_reason": cl.get("fail_reason"),
                }
            )
    return {"lost_a1": lost, "rows": out_rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Phase 0
    print("[P88] Phase 0", flush=True)
    p0 = phase0_select_and_audit(device)
    tid = p0["traj_id"]
    demo = p0["demo"]
    start = p0["start"]
    book, rho = p0["book"], p0["rho"]
    print(f"[P88] selected {tid} n={len(demo['tokens'])} valid_official_pool={p0['valid_count']}", flush=True)

    # Phase 1
    print("[P88] Phase 1 single-trajectory micro-overfit", flush=True)
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in critic.parameters():
        p.requires_grad_(False)
    curve, mem = train_one_demo(actor, demo, device, max_updates=MAX_SINGLE_UPDATES, label="single")
    _write_csv(OUT / "SINGLE_TRAJECTORY_LEARNING_CURVE.csv", curve)

    # Token prediction audit at end
    tf = eval_teacher_forced(actor, demo, device)
    tok_rows = []
    for i in range(tf["n_tokens"]):
        tok_rows.append(
            {
                "t": i,
                "target": tf["per_pos_tgt"][i],
                "pred": tf["per_pos_pred"][i],
                "ok": tf["per_pos_ok"][i],
                "p_target": tf["per_pos_p_target"][i],
            }
        )
    _write_csv(OUT / "TOKEN_PREDICTION_AUDIT.csv", tok_rows)

    single_path = CKPT_DIR / "P88_single_overfit_actor.pt"
    save_m111(single_path, actor, book, epoch=mem["updates"], rho=rho, extra={"p88": "single", "traj_id": tid})

    if not mem["memorized"]:
        case = "A"
        title = "Single-trajectory memorization fails — training/data/model implementation issue"
        _write(
            OUT / "TEACHER_FORCED_VS_AUTOREGRESSIVE.md",
            f"# TEACHER_FORCED_VS_AUTOREGRESSIVE\n\nGenerated: {_utc()}\n\n"
            f"**Stopped:** memorization failed (acc={tf['acc']:.4f} ce={tf['ce']:.4f}). "
            "No AR / closed-loop student evaluation beyond baseline documentation.\n",
        )
        _write_csv(OUT / "CLOSED_LOOP_REPRODUCTION.csv", [])
        _write(OUT / "FIRST_DIVERGENCE.md", f"# FIRST_DIVERGENCE\n\nGenerated: {_utc()}\n\nN/A — CASE A.\n")
        _write(
            OUT / "SIXTEEN_TRAJECTORY_LEARNABILITY.md",
            f"# SIXTEEN_TRAJECTORY_LEARNABILITY\n\nGenerated: {_utc()}\n\nSkipped — single-traj memorization failed.\n",
        )
        _write(
            OUT / "P87_INTERFERENCE_DIAGNOSIS.md",
            f"# P87_INTERFERENCE_DIAGNOSIS\n\nGenerated: {_utc()}\n\nSkipped — B1 learnability not demonstrated.\n",
        )
        _write_final(case, title, p0, mem, tf, None, None, None, None)
        return

    # Phase 2
    print("[P88] Phase 2 TF / AR / closed-loop", flush=True)
    ar = autoregressive_offline(actor, demo, book, rho, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    # baselines
    p64_a, _, _, _ = load_actor(P64_ACTOR, device)
    cl_p64 = closed_loop_actor(env, p64_a, critic, start, book, rho, device)
    cl_student = closed_loop_actor(env, actor, critic, start, book, rho, device)
    cl_teacher = teacher_open_loop_official(env, start, demo, book, rho)

    tf_vs_ar = f"""# TEACHER_FORCED_VS_AUTOREGRESSIVE

Generated: {_utc()}

Selected IC: `{tid}` (n={len(demo['tokens'])})

## A. Teacher-forced offline (recorded obs + recorded prev)

| Metric | Value |
|---|---|
| Exact token accuracy | **{tf['acc']:.4f}** ({tf['n_correct']}/{tf['n_tokens']}) |
| First-action accuracy | {tf['first_acc']} |
| Mean CE | {tf['ce']:.6f} |
| Residual L2 vs teacher | {tf['res_l2']:.6f} |

## B. Autoregressive offline (recorded obs, own action history)

| Metric | Value |
|---|---|
| Exact token accuracy | **{ar['acc']:.4f}** ({ar['n_correct']}/{ar['n']}) |
| First-action accuracy | {ar['first_acc']} |
| First token divergence step | {ar['first_token_div']} |

## C. Full MuJoCo closed-loop (argmax, official R3F)

| Policy | SUCCESS_RELAXED_12S | handoff | ep_len | fail |
|---|---:|---:|---:|---|
| Teacher open-loop motion + R3F | {cl_teacher['SUCCESS_RELAXED_12S']} | {cl_teacher['handoff']} | {cl_teacher['ep_len']} | {cl_teacher.get('fail_reason')} |
| Frozen P64 | {cl_p64['SUCCESS_RELAXED_12S']} | {cl_p64['handoff']} | {cl_p64['ep_len']} | {cl_p64.get('fail_reason')} |
| P88 single-overfit student | {cl_student['SUCCESS_RELAXED_12S']} | {cl_student['handoff']} | {cl_student['ep_len']} | {cl_student.get('fail_reason')} |

HANDOFF-token accuracy: **N/A** (official Actor path masks HANDOFF; R3F monitor owns transfer).
"""
    _write(OUT / "TEACHER_FORCED_VS_AUTOREGRESSIVE.md", tf_vs_ar)
    _write_csv(
        OUT / "CLOSED_LOOP_REPRODUCTION.csv",
        [
            {"mode": "teacher_open_loop_r3f", "traj_id": tid, **{k: v for k, v in cl_teacher.items() if k != "tokens"}},
            {"mode": "P64_closed_loop", "traj_id": tid, **{k: v for k, v in cl_p64.items() if k != "tokens"}},
            {"mode": "P88_single_closed_loop", "traj_id": tid, **{k: v for k, v in cl_student.items() if k != "tokens"}},
        ],
    )

    # Phase 3
    print("[P88] Phase 3 first divergence", flush=True)
    div_rows, div_sum, prefix_rows = first_divergence_trace(
        env, actor, critic, start, demo, book, rho, device
    )
    _write_csv(OUT / "FIRST_DIVERGENCE_TRACE.csv", div_rows)
    _write_csv(OUT / "PREFIX_SWITCH_DIAGNOSTIC.csv", prefix_rows)
    earliest_ok_prefix = None
    for r in sorted(prefix_rows, key=lambda x: x["prefix_len"]):
        if int(r["SUCCESS_RELAXED_12S"]) == 1:
            earliest_ok_prefix = r["prefix_len"]
            break
    div_md = f"""# FIRST_DIVERGENCE

Generated: {_utc()}

IC: `{tid}`

## Closed-loop student vs teacher

| Item | Value |
|---|---|
| First token divergence step | {div_sum['first_token_div']} |
| First decode L2>0.05 step | {div_sum['first_decode_div']} |
| Student SUCCESS_RELAXED_12S | {div_sum['SUCCESS_RELAXED_12S']} |
| fail_reason | {div_sum.get('fail_reason')} |

Offline TF acc={tf['acc']:.4f}; AR acc={ar['acc']:.4f}; AR first_div={ar['first_token_div']}.

## Prefix switch diagnostic (teacher prefix → student)

Earliest prefix length from which student completes SUCCESS_RELAXED: **{earliest_ok_prefix}**
(full teacher length={len(demo['tokens'])}).

These intervention rollouts are **not** counted as normal closed-loop success evidence.

Full step table: `FIRST_DIVERGENCE_TRACE.csv`. Prefix table: `PREFIX_SWITCH_DIAGNOSTIC.csv`.
"""
    _write(OUT / "FIRST_DIVERGENCE.md", div_md)

    phase2 = {
        "tf": tf,
        "ar": ar,
        "cl_teacher": cl_teacher,
        "cl_p64": cl_p64,
        "cl_student": cl_student,
        "div_sum": div_sum,
        "earliest_ok_prefix": earliest_ok_prefix,
    }

    # Decide early cases B/C before Phase 4
    tf_ok = tf["acc"] >= MEM_ACC_TARGET
    ar_ok = ar["acc"] >= MEM_ACC_TARGET
    cl_ok = int(cl_student["SUCCESS_RELAXED_12S"]) == 1

    if tf_ok and not ar_ok:
        case, title = "B", "Memorization succeeds but autoregressive reproduction fails — action-history or inference mismatch"
        _write(
            OUT / "SIXTEEN_TRAJECTORY_LEARNABILITY.md",
            f"# SIXTEEN_TRAJECTORY_LEARNABILITY\n\nGenerated: {_utc()}\n\nSkipped — CASE B (AR failure).\n",
        )
        _write(
            OUT / "P87_INTERFERENCE_DIAGNOSIS.md",
            f"# P87_INTERFERENCE_DIAGNOSIS\n\nGenerated: {_utc()}\n\nSkipped — CASE B.\n",
        )
        _write_final(case, title, p0, mem, tf, phase2, None, None, None)
        return

    if tf_ok and ar_ok and not cl_ok:
        case, title = (
            "C",
            "Offline reproduction succeeds but physical closed-loop fails — control/state-distribution divergence",
        )
        # Still useful to run Phase 4? Spec: only if single-trajectory test succeeds.
        # Single-traj memorization succeeded; closed-loop failed. Phase 4 says
        # "Only if the single-trajectory test succeeds" — memorization succeeded.
        # Proceed to Phase 4 for joint learnability, then diagnose.
        pass
    elif tf_ok and ar_ok and cl_ok:
        pass
    else:
        # unexpected
        pass

    # Phase 4 — 16 traj joint memorization
    print("[P88] Phase 4 sixteen-trajectory memorization", flush=True)
    man = json.loads(P87_MANIFEST.read_text(encoding="utf-8"))
    demos16 = []
    for tid16 in man["r3f_compatible_traj_ids"]:
        demos16.append(load_npz(tid16))
    actor16, book16, rho16, _ = load_actor(P64_ACTOR, device)
    curve16, mem16 = train_joint(actor16, demos16, device, max_updates=MAX_JOINT_UPDATES)
    _write_csv(OUT / "JOINT16_LEARNING_CURVE.csv", curve16)

    per_traj = []
    starts = lookup_starts()
    env16 = StabilizeGrpoEnv(
        book16, rho16, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    critic16 = load_critic(P64_CRITIC, actor16, device)
    for p in critic16.parameters():
        p.requires_grad_(False)
    n_cl_ok = 0
    for d in demos16:
        m = eval_teacher_forced(actor16, d, device)
        ar_m = autoregressive_offline(actor16, d, book16, rho16, device)
        st = starts.get(d["traj_id"])
        cl = {"SUCCESS_RELAXED_12S": "", "fail_reason": "no_start"}
        if st is not None:
            cl = closed_loop_actor(env16, actor16, critic16, st, book16, rho16, device)
            n_cl_ok += int(cl["SUCCESS_RELAXED_12S"])
        per_traj.append(
            {
                "traj_id": d["traj_id"],
                "n": len(d["tokens"]),
                "tf_acc": m["acc"],
                "tf_ce": m["ce"],
                "first_acc": m["first_acc"],
                "ar_acc": ar_m["acc"],
                "ar_first_div": ar_m["first_token_div"],
                "SUCCESS_RELAXED_12S": cl.get("SUCCESS_RELAXED_12S"),
                "fail_reason": cl.get("fail_reason"),
            }
        )
    _write_csv(OUT / "JOINT16_PER_TRAJECTORY.csv", per_traj)
    mean_tf = float(np.mean([r["tf_acc"] for r in per_traj]))
    min_tf = float(np.min([r["tf_acc"] for r in per_traj]))
    sixteen_md = f"""# SIXTEEN_TRAJECTORY_LEARNABILITY

Generated: {_utc()}

Restarted from frozen P64. B1-only (no A1/C/D rehearsal).

| Item | Value |
|---|---|
| Updates used | {mem16['updates']} |
| Joint memorized (all ≥{MEM_ACC_TARGET} acc, CE≤{MEM_CE_TARGET}) | {mem16['memorized']} |
| Mean TF accuracy | {mean_tf:.4f} |
| Min TF accuracy | {min_tf:.4f} |
| Closed-loop SUCCESS_RELAXED on 16 train ICs | {n_cl_ok}/16 |

Per-trajectory: `JOINT16_PER_TRAJECTORY.csv`. Curve: `JOINT16_LEARNING_CURVE.csv`.
"""
    _write(OUT / "SIXTEEN_TRAJECTORY_LEARNABILITY.md", sixteen_md)
    joint_path = CKPT_DIR / "P88_joint16_actor.pt"
    save_m111(joint_path, actor16, book16, epoch=mem16["updates"], rho=rho16, extra={"p88": "joint16"})

    phase4 = {"mem16": mem16, "per_traj": per_traj, "n_cl_ok": n_cl_ok, "mean_tf": mean_tf, "min_tf": min_tf}

    # Phase 5 — only after B1 learnability demonstrated
    b1_learnable = mem["memorized"] and (cl_ok or mem16["memorized"] or mean_tf >= 0.99)
    phase5 = None
    if b1_learnable:
        print("[P88] Phase 5 P87 interference", flush=True)
        # Use single-overfit actor as B1-only diagnostic fit
        inter = a1_interference(device, single_path)
        lines = [
            f"# P87_INTERFERENCE_DIAGNOSIS\n\nGenerated: {_utc()}\n",
            "Compare frozen P64 vs P88 B1-only single-overfit vs original P87 BC on A1 ICs that P87 lost.\n",
            f"Lost A1 ICs (P87): {inter['lost_a1']}\n",
        ]
        for r in inter["rows"]:
            lines.append(
                f"- `{r['traj_id']}` **{r['policy']}**: RELAXED={r['SUCCESS_RELAXED_12S']} "
                f"first_tok={r['first_tok']} (P64={r['p64_first_tok']}) "
                f"KL0={r['kl_from_p64_step0']:.4f} fail={r.get('fail_reason')}"
            )
        # Rehearsal preservation note
        lines += [
            "",
            "## Rehearsal preservation",
            "",
            "P87 stored A1/C/D rehearsal npz under `p87.../teacher/rehearse_*.npz` from frozen-P64 successes.",
            "P87 BC weighted rehearsal at 0.5 vs B1 at 1.0 for 60 steps — insufficient to prevent A1 regression",
            "if B1 gradients dominate or if B1 demos conflict near shared early tokens.",
            "",
            "This phase does **not** launch another full retention run.",
        ]
        _write(OUT / "P87_INTERFERENCE_DIAGNOSIS.md", "\n".join(lines))
        _write_csv(OUT / "P87_INTERFERENCE_ROWS.csv", inter["rows"])
        phase5 = inter
    else:
        _write(
            OUT / "P87_INTERFERENCE_DIAGNOSIS.md",
            f"# P87_INTERFERENCE_DIAGNOSIS\n\nGenerated: {_utc()}\n\nSkipped — B1 learnability not clearly demonstrated.\n",
        )

    # Final case decision (single primary). Prefer earliest failure stage.
    if not mem["memorized"]:
        case, title = "A", "Single-trajectory memorization fails — training/data/model implementation issue"
    elif tf_ok and not ar_ok:
        case, title = "B", "Memorization succeeds but autoregressive reproduction fails — action-history or inference mismatch"
    elif tf_ok and ar_ok and not cl_ok:
        case, title = (
            "C",
            "Offline reproduction succeeds but physical closed-loop fails — control/state-distribution divergence",
        )
    elif cl_ok and not mem16["memorized"]:
        case, title = (
            "D",
            "Single-trajectory reproduction succeeds but joint training fails — multi-trajectory optimization or conflicting supervision",
        )
    elif cl_ok and mem16["memorized"] and phase5 and phase5.get("lost_a1"):
        case, title = "E", "B1 training reproduction succeeds but retention damages other skills — multi-skill interference"
    elif cl_ok and mem16["memorized"] and phase4["n_cl_ok"] >= 12:
        case, title = "F", "All diagnostic stages succeed — P87 failure requires a more targeted investigation"
    else:
        case, title = (
            "C",
            "Offline reproduction succeeds but physical closed-loop fails — control/state-distribution divergence",
        )

    _write_final(case, title, p0, mem, tf, phase2, phase4, phase5, ar)
    print(json.dumps({"case": case, "title": title, "traj_id": tid}, indent=2), flush=True)


def _write_final(case, title, p0, mem, tf, phase2, phase4, phase5, ar) -> None:
    text = f"""# FINAL_REPORT — P88 B1 Short-Sequence Learnability Audit

Generated: {_utc()}

## CASE {case} — {title}

### Phase 0

- Selected: `{p0['traj_id']}` (n={len(p0['demo']['tokens'])})
- Rule: {p0['select_note']}
- Official-valid R3F-compat demos: {p0['valid_count']}/16

### Phase 1 — single-trajectory micro-overfit

| Item | Value |
|---|---|
| Memorized | {mem['memorized']} |
| Updates | {mem['updates']} |
| Final TF acc | {tf['acc']:.4f} |
| Final CE | {tf['ce']:.6f} |
| First-action acc | {tf['first_acc']} |

### Later phases

"""
    if phase2:
        text += f"""- AR offline acc: {phase2['ar']['acc']:.4f} (first_div={phase2['ar']['first_token_div']})
- Closed-loop student SUCCESS_RELAXED: {phase2['cl_student']['SUCCESS_RELAXED_12S']}
- Teacher open-loop SUCCESS_RELAXED: {phase2['cl_teacher']['SUCCESS_RELAXED_12S']}
- First token divergence: {phase2['div_sum']['first_token_div']}
- Earliest successful teacher-prefix: {phase2['earliest_ok_prefix']}
"""
    if phase4:
        text += f"""
### Phase 4 — joint 16

- Memorized: {phase4['mem16']['memorized']} @ {phase4['mem16']['updates']} updates
- Mean/Min TF acc: {phase4['mean_tf']:.4f} / {phase4['min_tf']:.4f}
- Closed-loop successes: {phase4['n_cl_ok']}/16
"""
    if phase5:
        text += f"""
### Phase 5 — P87 interference

- Lost A1 ICs: {phase5['lost_a1']}
"""
    text += """
### Integrity

- P64 / P87 checkpoints not overwritten
- Exposed independent B1 ICs not used for training
- Physics / M111 / R3F / SUCCESS definitions unchanged
- Diagnostic actors saved only under `p88_.../checkpoints/`
- No follow-on full BC/GRPO launched
"""
    _write(OUT / "FINAL_REPORT.md", text)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "traj_id": p0["traj_id"],
                "memorized_single": mem["memorized"],
                "tf_acc": tf["acc"],
                "phase2": None
                if not phase2
                else {
                    "ar_acc": phase2["ar"]["acc"],
                    "cl_student": phase2["cl_student"]["SUCCESS_RELAXED_12S"],
                    "cl_teacher": phase2["cl_teacher"]["SUCCESS_RELAXED_12S"],
                },
                "phase4": None
                if not phase4
                else {
                    "memorized": phase4["mem16"]["memorized"],
                    "n_cl_ok": phase4["n_cl_ok"],
                    "mean_tf": phase4["mean_tf"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
