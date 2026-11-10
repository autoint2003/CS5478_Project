"""P89 — B1 residual causality and physical execution parity (diagnostic only; no training)."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
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
from models.m111_checkpoint import MOTION_K, decode_m111_action  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget, clip_residual, hybrid_decode  # noqa: E402

P88_OUT = ROOT / "results/experiments/p88_b1_short_sequence_learnability"
P88_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = P88_OUT / "PHASE0_CANDIDATES.csv"
P88_SINGLE = P88_OUT / "checkpoints/P88_single_overfit_actor.pt"
P88_AUDIT_IC = "ball_train_v2_0047"  # P88 selected diagnostic IC

OUT = ROOT / "results/experiments/p89_b1_residual_causality"
SEED = 89
DIM_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip")


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


def load_demo(traj_id: str) -> dict:
    z = np.load(P88_TEACHER / f"{traj_id}.npz", allow_pickle=True)
    return {
        "traj_id": traj_id,
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


def lookup_starts() -> dict[str, Any]:
    sched = build_p54_dataset(seed=5478)["schedule"]
    return {s.traj_id: s for s in sched}


def verified_demo_ids() -> tuple[list[str], list[str]]:
    rows = list(csv.DictReader(P88_CAND.open(encoding="utf-8")))
    ok = [r["traj_id"] for r in rows if int(r["valid_official"]) == 1]
    bad = [r["traj_id"] for r in rows if int(r["valid_official"]) == 0]
    return ok, bad


def rho_arr(rho) -> np.ndarray:
    return ResidualBudget.from_spec(rho).as_array().astype(np.float32)


def decompose_action(book, tok: int, resid: np.ndarray, rho) -> dict:
    """Full decode chain in normalized action units [-1,1]^7."""
    r = rho_arr(rho)
    proto = np.asarray(book[int(tok)], np.float32)
    resid_raw = np.asarray(resid, np.float32).reshape(7)
    resid_clipped = clip_residual(resid_raw, rho).astype(np.float32)
    pre_clip = proto + resid_clipped
    post_clip = np.clip(pre_clip, -1.0, 1.0).astype(np.float32)
    via_api = decode_m111_action(book, int(tok), resid_raw, rho=rho)
    return {
        "proto": proto,
        "resid_raw": resid_raw,
        "resid_clipped": resid_clipped,
        "pre_clip": pre_clip.astype(np.float32),
        "post_clip": post_clip,
        "via_api": via_api,
        "api_match": float(np.linalg.norm(via_api - post_clip)) < 1e-6,
        "rho": r,
    }


def student_step_pred(actor, critic, env, device, book, rho):
    tok_a, resid, _, _ = act_with_value(
        actor, critic, env.obs_seq, env.prev_seq, device, deterministic=True
    )
    st = getattr(env, "_p89_wrap", None)
    if st is None:
        env._p89_wrap = WrapState()
        env._p89_v2 = v2_spec()
        st = env._p89_wrap
    tok, st, _ = select_token(
        env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=env._p89_v2, rule=None
    )
    env._p89_wrap = st
    action = decode_m111_action(book, int(tok), resid, rho=rho)
    return int(tok), np.asarray(resid, np.float32), action, int(tok_a)


# ---------------------------------------------------------------------------
# Phase 1 — first-action decomposition
# ---------------------------------------------------------------------------


def phase1_decompose(actor, critic, start, demo, book, rho, device) -> list[dict]:
    seed_all(SEED, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    env._p89_wrap = WrapState()
    env._p89_v2 = v2_spec()

    t_tok = int(demo["tokens"][0])
    t_res = np.asarray(demo["residuals"][0], np.float32)
    t_act = np.asarray(demo["actions"][0], np.float32)
    s_tok, s_res, s_act, s_tok_raw = student_step_pred(actor, critic, env, device, book, rho)

    td = decompose_action(book, t_tok, t_res, rho)
    sd = decompose_action(book, s_tok, s_res, rho)
    # Also: student token + teacher residual
    mix = decompose_action(book, s_tok, t_res, rho)
    # teacher token + student residual
    mix2 = decompose_action(book, t_tok, s_res, rho)

    rows = []
    for dim_i, name in enumerate(DIM_NAMES):
        rows.append(
            {
                "t": 0,
                "dim": name,
                "dim_i": dim_i,
                "teacher_token": t_tok,
                "student_token": s_tok,
                "student_token_raw_actor": s_tok_raw,
                "token_match": int(t_tok == s_tok),
                "codebook_proto": float(td["proto"][dim_i]),
                "teacher_residual_raw": float(td["resid_raw"][dim_i]),
                "student_residual_raw": float(sd["resid_raw"][dim_i]),
                "rho": float(td["rho"][dim_i]),
                "teacher_residual_clipped": float(td["resid_clipped"][dim_i]),
                "student_residual_clipped": float(sd["resid_clipped"][dim_i]),
                "teacher_pre_clip_action": float(td["pre_clip"][dim_i]),
                "student_pre_clip_action": float(sd["pre_clip"][dim_i]),
                "teacher_post_clip_action": float(td["post_clip"][dim_i]),
                "student_post_clip_action": float(sd["post_clip"][dim_i]),
                "recorded_teacher_action": float(t_act[dim_i]),
                "stu_tok_tea_res_action": float(mix["post_clip"][dim_i]),
                "tea_tok_stu_res_action": float(mix2["post_clip"][dim_i]),
                "residual_delta": float(sd["resid_raw"][dim_i] - td["resid_raw"][dim_i]),
                "action_delta_stu_vs_tea": float(sd["post_clip"][dim_i] - td["post_clip"][dim_i]),
                "units": "normalized_action_[-1,1]_and_residual_same_frame",
            }
        )

    # summary vectors
    summary = {
        "teacher_token": t_tok,
        "student_token": s_tok,
        "token_match": int(t_tok == s_tok),
        "resid_l2": float(np.linalg.norm(sd["resid_raw"] - td["resid_raw"])),
        "action_l2_decoded": float(np.linalg.norm(sd["post_clip"] - td["post_clip"])),
        "action_l2_vs_recorded": float(np.linalg.norm(sd["post_clip"] - t_act)),
        "recorded_vs_tea_decode_l2": float(np.linalg.norm(td["post_clip"] - t_act)),
        "mix_stu_tok_tea_res_l2_vs_tea": float(np.linalg.norm(mix["post_clip"] - td["post_clip"])),
        "mix_tea_tok_stu_res_l2_vs_tea": float(np.linalg.norm(mix2["post_clip"] - td["post_clip"])),
        "obs0_l2_vs_recorded": float(np.linalg.norm(np.asarray(last_obs(env), np.float32) - demo["obs"][0])),
        "prev0_l2_vs_recorded": float(
            np.linalg.norm(np.asarray(env.prev_seq[-1], np.float32) - demo["prev"][0])
        ),
    }
    (OUT / "ACTION_DECOMPOSITION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return rows, summary, env


# ---------------------------------------------------------------------------
# Phase 2 — execution parity
# ---------------------------------------------------------------------------


def phase2_parity(start, demo, book, rho) -> dict:
    """Teacher token+residual must reproduce recorded actions and physical open-loop success."""
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    seed_all(SEED, deterministic_torch=True)
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)

    n = len(demo["tokens"])
    step_rows = []
    max_cmd_l2 = 0.0
    max_vs_recorded = 0.0
    for t in range(n):
        tok = int(demo["tokens"][t])
        res = np.asarray(demo["residuals"][t], np.float32)
        rec = np.asarray(demo["actions"][t], np.float32)
        dec = decode_m111_action(book, tok, res, rho=rho)
        # Execute via env.step (same path as closed-loop)
        action_before = dec.copy()
        # step_legal clips again internally; capture by decoding then stepping
        _r, done, info = env.step(tok, res)
        ensure_inner_capture_buffers(env)
        # Use env.env._prev (just executed). Note: on terminal done, prev_seq is NOT
        # appended by StabilizeGrpoEnv — comparing prev_seq[-1] would be a false mismatch.
        executed = np.asarray(env.env._prev, np.float32)
        cmd_l2 = float(np.linalg.norm(dec - rec))
        exec_l2 = float(np.linalg.norm(executed - rec))
        max_cmd_l2 = max(max_cmd_l2, cmd_l2)
        max_vs_recorded = max(max_vs_recorded, exec_l2)
        step_rows.append(
            {
                "t": t,
                "token": tok,
                "decode_vs_recorded_l2": cmd_l2,
                "executed_cmd_vs_recorded_l2": exec_l2,
                "done": int(done),
            }
        )
        if done:
            break

    # Continue hold until R3F if needed
    while not done and env.t < env.max_steps:
        _r, done, info = env.step(0, np.zeros(7, np.float32))
        ensure_inner_capture_buffers(env)

    term = _r3_terminal(env, info)
    parity_ok = max_cmd_l2 < 1e-4 and max_vs_recorded < 1e-4
    # Slightly looser for float: 1e-3
    parity_ok_loose = max_cmd_l2 < 1e-3 and max_vs_recorded < 1e-3

    md = f"""# EXECUTION_PARITY

Generated: {_utc()}

IC: `{demo['traj_id']}`

## Claim

Teacher token + teacher residual through `decode_m111_action` → `env.step` must match the
recorded M111 teacher action trajectory used in P87/P88.

## Results

| Check | Value |
|---|---|
| Steps compared | {len(step_rows)} / {n} |
| Max decode vs recorded L2 | **{max_cmd_l2:.6g}** |
| Max executed command (`env._prev`) vs recorded L2 | **{max_vs_recorded:.6g}** |
| Exact parity (L2 < 1e-4) | {parity_ok} |
| Loose parity (L2 < 1e-3) | {parity_ok_loose} |
| Open-loop + R3F SUCCESS_RELAXED | {int(term.get('SUCCESS_RELAXED') or info.get('SUCCESS_RELAXED_12S') or 0)} |
| handoff | {int(info.get('handoff') or 0)} |
| ep_len | {int(env.t)} |

## Decoder / residual semantics (audited)

- Residual frame: **same 7D normalized action space** as prototypes (`dx,dy,dz,droll,dpitch,dyaw,grip`), not SI.
- Policy residual head: `res_cap * tanh(...)` with `res_cap = max(rho)` (ρ≈0.35).
- Deployed decode: `clip(proto + clip(r, ±rho), ±1)`.
- `step_legal` re-clips action to ±1 then maps via `map_action` to Cartesian/grip commands.
- HANDOFF is monitor-driven; Actor HANDOFF logit masked at inference.
- Caveat: on the terminal handoff step, `prev_seq` is not appended (`if not done`); parity uses `env._prev`.

## Verdict

{"**PASS** — proceed to residual counterfactuals." if parity_ok_loose else "**FAIL** — stop; diagnose execution pipeline (CASE C)."}
"""
    _write(OUT / "EXECUTION_PARITY.md", md)
    _write_csv(OUT / "EXECUTION_PARITY_STEPS.csv", step_rows)
    return {
        "parity_ok": parity_ok_loose,
        "max_cmd_l2": max_cmd_l2,
        "max_exec_l2": max_vs_recorded,
        "success": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "steps": step_rows,
    }


# ---------------------------------------------------------------------------
# Phase 3–4 — counterfactual rollouts
# ---------------------------------------------------------------------------


def rollout_cf(
    actor,
    critic,
    start,
    demo,
    book,
    rho,
    device,
    *,
    mode: str,
    k_res: int = 0,
    k_full: int = 0,
    alpha: float | None = None,
) -> dict:
    """Closed-loop student with optional residual / full-action interventions on prefix.

    modes:
      A / student: no intervention
      res_k: student token + teacher residual for first k_res steps
      full_k: teacher token+residual (full teacher action) for first k_full steps
      interp_k: student token + (1-a)*stu_res + a*tea_res for first k_res steps
    """
    seed_all(SEED, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    env._p89_wrap = WrapState()
    env._p89_v2 = v2_spec()

    t_toks = [int(x) for x in demo["tokens"]]
    t_res = np.asarray(demo["residuals"], np.float32)
    t_acts = np.asarray(demo["actions"], np.float32)

    done = False
    info: dict = {}
    trace = []
    token_mismatch_in_intervened = 0
    intervened_steps = 0

    while not done:
        t = int(env.t)
        s_tok, s_res, s_act, _ = student_step_pred(actor, critic, env, device, book, rho)
        tea_tok = int(t_toks[t]) if t < len(t_toks) else None
        tea_r = t_res[t] if t < len(t_res) else None

        use_full = k_full > 0 and t < k_full and tea_tok is not None
        use_res = (not use_full) and k_res > 0 and t < k_res and tea_r is not None
        use_interp = (not use_full) and alpha is not None and t < k_res and tea_r is not None

        if use_full:
            tok = tea_tok
            resid = tea_r
            intervened_steps += 1
            if int(s_tok) != int(tea_tok):
                token_mismatch_in_intervened += 1
        elif use_interp:
            tok = s_tok
            a = float(alpha)
            resid = (1.0 - a) * s_res + a * tea_r
            intervened_steps += 1
            if tea_tok is not None and int(s_tok) != int(tea_tok):
                token_mismatch_in_intervened += 1
        elif use_res:
            tok = s_tok
            resid = tea_r
            intervened_steps += 1
            if tea_tok is not None and int(s_tok) != int(tea_tok):
                token_mismatch_in_intervened += 1
        else:
            tok = s_tok
            resid = s_res

        action = decode_m111_action(book, int(tok), resid, rho=rho)
        g0 = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds0 = evaluate_predicates(env.env.sim, ok_s=ok_s)

        trace.append(
            {
                "t": t,
                "mode": mode,
                "student_token": int(s_tok),
                "teacher_token": tea_tok if tea_tok is not None else "",
                "executed_token": int(tok),
                "token_match_teacher": int(tea_tok is not None and int(s_tok) == int(tea_tok)),
                "intervened": int(use_full or use_res or use_interp),
                "resid_l2_vs_teacher": float(np.linalg.norm(resid - tea_r)) if tea_r is not None else "",
                "action_l2_vs_teacher": float(np.linalg.norm(action - t_acts[t])) if t < len(t_acts) else "",
                "e_pos_mm": float(g0["e_pos_mm"]),
                "theta_deg": float(g0["theta_deg"]),
                "bilateral": int(g0["bilateral"]),
                "min_pad": int(g0.get("min_pad") or 0),
                "frame_ok": int(bool(preds0.get("frame_ok"))),
            }
        )

        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)

    term = _r3_terminal(env, info)
    # contact summary
    bil_ever = any(int(r["bilateral"]) for r in trace)
    frame_ever = any(int(r["frame_ok"]) for r in trace)
    min_e = min((float(r["e_pos_mm"]) for r in trace), default=-1)

    return {
        "mode": mode,
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "intervened_steps": intervened_steps,
        "token_mismatch_in_intervened": token_mismatch_in_intervened,
        "bilateral_ever": int(bil_ever),
        "frame_ok_ever": int(frame_ever),
        "min_e_pos_mm": float(min_e),
        "trace": trace,
    }


def first_four_analysis(cf_rows: list[dict], traces: dict[str, list]) -> str:
    lines = [
        f"# FIRST_FOUR_STEPS_ANALYSIS\n\nGenerated: {_utc()}\n",
        "Diagnostic IC closed-loop counterfactuals (identical seed / R3F / controller).\n",
        "## Outcome table\n",
        "| Mode | SUCCESS_RELAXED | handoff | ep_len | tok_mismatch_in_interv | min_e_pos | frame_ok_ever |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cf_rows:
        lines.append(
            f"| {r['mode']} | {r['SUCCESS_RELAXED_12S']} | {r['handoff']} | {r['ep_len']} | "
            f"{r['token_mismatch_in_intervened']} | {r['min_e_pos_mm']:.3f} | {r['frame_ok_ever']} |"
        )

    lines += ["\n## First 4 steps — action L2 vs teacher (where available)\n"]
    for mode, tr in traces.items():
        lines.append(f"### {mode}\n")
        lines.append("| t | stu_tok | tea_tok | match | intervened | action_l2 | e_pos | bil | frame |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in tr[:4]:
            lines.append(
                f"| {row['t']} | {row['student_token']} | {row['teacher_token']} | "
                f"{row['token_match_teacher']} | {row['intervened']} | {row['action_l2_vs_teacher']} | "
                f"{row['e_pos_mm']:.3f} | {row['bilateral']} | {row['frame_ok']} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    assert P88_SINGLE.is_file(), f"missing {P88_SINGLE}"
    starts = lookup_starts()
    verified, unverified = verified_demo_ids()
    demo = load_demo(P88_AUDIT_IC)
    start = starts[P88_AUDIT_IC]

    actor, book, rho, _ = load_actor(P88_SINGLE, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in list(actor.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)
    actor.eval()
    critic.eval()

    # Phase 1
    print("[P89] Phase 1 decompose", flush=True)
    dec_rows, summary, _ = phase1_decompose(actor, critic, start, demo, book, rho, device)
    _write_csv(OUT / "ACTION_DECOMPOSITION.csv", dec_rows)
    print(
        f"  tok match={summary['token_match']} resid_l2={summary['resid_l2']:.4f} "
        f"action_l2={summary['action_l2_decoded']:.4f}",
        flush=True,
    )

    # Phase 2
    print("[P89] Phase 2 execution parity", flush=True)
    parity = phase2_parity(start, demo, book, rho)
    print(f"  parity_ok={parity['parity_ok']} max_l2={parity['max_cmd_l2']:.6g}", flush=True)
    if not parity["parity_ok"]:
        _write(
            OUT / "FINAL_REPORT.md",
            f"""# FINAL_REPORT — P89

Generated: {_utc()}

## CASE C — Action parity fails due to decoder, normalization, frame, clipping, or timing mismatch

Max decode vs recorded L2 = {parity['max_cmd_l2']:.6g}.
Residual counterfactuals were **not** run.
""",
        )
        _write_csv(OUT / "RESIDUAL_PREFIX_COUNTERFACTUAL.csv", [])
        _write(OUT / "FIRST_FOUR_STEPS_ANALYSIS.md", "# FIRST_FOUR_STEPS_ANALYSIS\n\nSkipped — parity failed.\n")
        _write_csv(OUT / "VERIFIED_DEMO_REPLICATION.csv", [])
        print(json.dumps({"case": "C"}), flush=True)
        return

    # Phase 3
    print("[P89] Phase 3 residual prefix CF", flush=True)
    specs = [
        ("A_student", dict(k_res=0, k_full=0, alpha=None)),
        ("B_res1", dict(k_res=1, k_full=0, alpha=None)),
        ("C_res2", dict(k_res=2, k_full=0, alpha=None)),
        ("D_res4", dict(k_res=4, k_full=0, alpha=None)),
        ("E_full4", dict(k_res=0, k_full=4, alpha=None)),
    ]
    cf_rows = []
    traces = {}
    for mode, kw in specs:
        print(f"  CF {mode}", flush=True)
        out = rollout_cf(actor, critic, start, demo, book, rho, device, mode=mode, **kw)
        traces[mode] = out["trace"]
        cf_rows.append({k: v for k, v in out.items() if k != "trace"})
        _write_csv(OUT / f"TRACE_{mode}.csv", out["trace"])

    # Phase 4 — interpolation if residual-only (B/C/D) rescues
    res_rescue = any(
        int(r["SUCCESS_RELAXED_12S"]) == 1 and r["mode"] in ("B_res1", "C_res2", "D_res4") for r in cf_rows
    )
    interp_rows = []
    if res_rescue:
        print("[P89] Phase 4 residual interpolation", flush=True)
        # use earliest k that rescued
        k_star = 4
        for mode, k in (("B_res1", 1), ("C_res2", 2), ("D_res4", 4)):
            hit = next(r for r in cf_rows if r["mode"] == mode)
            if int(hit["SUCCESS_RELAXED_12S"]) == 1:
                k_star = k
                break
        for a in (0.0, 0.5, 1.0):
            mode = f"interp_k{k_star}_a{a}"
            print(f"  CF {mode}", flush=True)
            out = rollout_cf(
                actor, critic, start, demo, book, rho, device, mode=mode, k_res=k_star, k_full=0, alpha=a
            )
            traces[mode] = out["trace"]
            row = {k: v for k, v in out.items() if k != "trace"}
            row["alpha"] = a
            row["k"] = k_star
            interp_rows.append(row)
            cf_rows.append(row)
            _write_csv(OUT / f"TRACE_{mode}.csv", out["trace"])
    else:
        print("[P89] Phase 4 skipped (no residual-only rescue on diagnostic IC)", flush=True)

    _write_csv(OUT / "RESIDUAL_PREFIX_COUNTERFACTUAL.csv", cf_rows)
    _write(OUT / "FIRST_FOUR_STEPS_ANALYSIS.md", first_four_analysis(cf_rows, traces))

    # Phase 5 — replicate decisive CF on all verified demos
    print("[P89] Phase 5 verified-demo replication", flush=True)
    # Decisive protocol: A_student, D_res4, E_full4 (and B_res1 if useful)
    repl = []
    for tid in verified:
        d = load_demo(tid)
        st = starts[tid]
        for mode, kw in (
            ("A_student", dict(k_res=0, k_full=0)),
            ("D_res4", dict(k_res=4, k_full=0)),
            ("E_full4", dict(k_res=0, k_full=4)),
        ):
            out = rollout_cf(actor, critic, st, d, book, rho, device, mode=mode, **kw)
            repl.append(
                {
                    "traj_id": tid,
                    "category": "verified_official",
                    "mode": mode,
                    "SUCCESS_RELAXED_12S": out["SUCCESS_RELAXED_12S"],
                    "handoff": out["handoff"],
                    "ep_len": out["ep_len"],
                    "fail_reason": out.get("fail_reason"),
                    "token_mismatch_in_intervened": out["token_mismatch_in_intervened"],
                    "min_e_pos_mm": out["min_e_pos_mm"],
                    "frame_ok_ever": out["frame_ok_ever"],
                    "bilateral_ever": out["bilateral_ever"],
                }
            )
            print(f"  {tid} {mode} s={out['SUCCESS_RELAXED_12S']}", flush=True)

    for tid in unverified:
        repl.append(
            {
                "traj_id": tid,
                "category": "unverified_official_teacher",
                "mode": "SKIPPED",
                "SUCCESS_RELAXED_12S": "",
                "note": "P88 Phase0 valid_official=0; held separate",
            }
        )
    _write_csv(OUT / "VERIFIED_DEMO_REPLICATION.csv", repl)

    # Phase 6 — attribution
    a = next(r for r in cf_rows if r["mode"] == "A_student")
    b = next(r for r in cf_rows if r["mode"] == "B_res1")
    c = next(r for r in cf_rows if r["mode"] == "C_res2")
    d = next(r for r in cf_rows if r["mode"] == "D_res4")
    e = next(r for r in cf_rows if r["mode"] == "E_full4")
    res_any = any(int(x["SUCCESS_RELAXED_12S"]) == 1 for x in (b, c, d))
    full_ok = int(e["SUCCESS_RELAXED_12S"]) == 1
    base_ok = int(a["SUCCESS_RELAXED_12S"]) == 1

    # replication aggregates
    def rate(mode: str) -> tuple[int, int]:
        xs = [r for r in repl if r["mode"] == mode and r["category"] == "verified_official"]
        return sum(int(r["SUCCESS_RELAXED_12S"]) for r in xs), len(xs)

    nA, nT = rate("A_student")
    nD, _ = rate("D_res4")
    nE, _ = rate("E_full4")

    # token match on diagnostic during res interventions
    tok_ok = all(int(x["token_mismatch_in_intervened"]) == 0 for x in (b, c, d))

    if not parity["parity_ok"]:
        case, title = "C", "Action parity fails due to decoder, normalization, frame, clipping, or timing mismatch"
    elif res_any and tok_ok:
        case, title = "A", "Residual-only correction rescues closed-loop execution"
    elif res_any and not tok_ok:
        # residual helped but tokens also differed — still primarily residual if A fails and res works
        case, title = "A", "Residual-only correction rescues closed-loop execution"
        title += " (note: some intervened steps had token mismatch; see CSV)"
    elif full_ok and not res_any:
        case, title = "B", "Full teacher-action prefix rescues execution, but residual-only correction does not"
    elif (int(d["min_e_pos_mm"]) < int(a["min_e_pos_mm"]) or int(d["frame_ok_ever"]) > int(a["frame_ok_ever"])) and not res_any and not full_ok:
        case, title = "D", "Residual correction improves physical alignment but does not achieve success"
    elif not res_any and not full_ok:
        case, title = "E", "No reproducible causal effect or insufficient evidence"
    else:
        case, title = "E", "No reproducible causal effect or insufficient evidence"

    # Prefer A if residual rescues diagnostic even if replication mixed
    if res_any:
        case, title = "A", "Residual-only correction rescues closed-loop execution"
    elif full_ok and not res_any:
        case, title = "B", "Full teacher-action prefix rescues execution, but residual-only correction does not"

    final = f"""# FINAL_REPORT — P89 B1 Residual Causality

Generated: {_utc()}

## CASE {case} — {title}

### Phase 1 — first-action decomposition (`{P88_AUDIT_IC}`)

| Item | Value |
|---|---|
| Teacher token | {summary['teacher_token']} |
| Student token | {summary['student_token']} |
| Token match | {summary['token_match']} |
| Residual L2 (stu−tea) | {summary['resid_l2']:.6f} |
| Decoded action L2 | {summary['action_l2_decoded']:.6f} |
| Recorded vs teacher-decode L2 | {summary['recorded_vs_tea_decode_l2']:.6g} |
| stu_tok+tea_res action L2 vs teacher | {summary['mix_stu_tok_tea_res_l2_vs_tea']:.6g} |
| tea_tok+stu_res action L2 vs teacher | {summary['mix_tea_tok_stu_res_l2_vs_tea']:.6g} |
| Obs0 / prev0 match recorded | obs_l2={summary['obs0_l2_vs_recorded']:.6g}, prev_l2={summary['prev0_l2_vs_recorded']:.6g} |

Residuals and actions are in **normalized action units** ∈ [−1,1]^7 (same frame as codebook), not SI.

### Phase 2 — execution parity

PASS (max decode L2={parity['max_cmd_l2']:.6g}). Teacher tok+res reproduces recorded actions.

### Phase 3 — diagnostic IC counterfactuals

| Mode | SUCCESS_RELAXED |
|---|---:|
| A student | {a['SUCCESS_RELAXED_12S']} |
| B student tok + tea res ×1 | {b['SUCCESS_RELAXED_12S']} |
| C student tok + tea res ×2 | {c['SUCCESS_RELAXED_12S']} |
| D student tok + tea res ×4 | {d['SUCCESS_RELAXED_12S']} |
| E full teacher action ×4 | {e['SUCCESS_RELAXED_12S']} |

### Phase 4 — interpolation

{"Ran; see RESIDUAL_PREFIX_COUNTERFACTUAL.csv." if interp_rows else "Skipped (no residual-only rescue) or N/A."}

### Phase 5 — verified training demos (not heldout)

| Mode | SUCCESS_RELAXED |
|---|---|
| A student | {nA}/{nT} |
| D res4 | {nD}/{nT} |
| E full4 | {nE}/{nT} |

Unverified teacher demos ({len(unverified)}): listed as SKIPPED in `VERIFIED_DEMO_REPLICATION.csv`.

### Integrity

- No Actor/Critic/BC/GRPO training
- P64 / P87 / P88 checkpoints not overwritten
- Official R3F unchanged
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "diagnostic_ic": P88_AUDIT_IC,
                "summary_t0": summary,
                "parity_ok": parity["parity_ok"],
                "cf_diagnostic": {r["mode"]: r["SUCCESS_RELAXED_12S"] for r in cf_rows if "alpha" not in r or r.get("alpha") is None},
                "replication": {"A": nA, "D_res4": nD, "E_full4": nE, "n_verified": nT},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"case": case, "title": title, "A": nA, "D": nD, "E": nE, "n": nT}, indent=2), flush=True)


if __name__ == "__main__":
    main()
