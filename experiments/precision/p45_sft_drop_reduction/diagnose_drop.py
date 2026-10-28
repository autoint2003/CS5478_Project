"""P45 Phase 1 — Diagnose P44 DROP_BEFORE failures vs expert tapes.

For each held-out IC in P44 eval:
  - Parallel expert token sequence vs closed-loop policy
  - First token mismatch under expert-forced history (prediction error)
  - First bilateral loss / latch under policy rollout
  - Classify: action_prediction_error | covariate_shift | contact_sensitivity

Does not modify HANDOFF / nominal / physics / SUCCESS criteria.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM, WORKING_THETA_DEG  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paths import (  # noqa: E402
    OUT,
    P44_CKPT,
    P44_EVAL,
    SEED,
    T_MAX,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW as K_WIN  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def snap_geo(sim) -> dict[str, Any]:
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    return {
        "theta": float(m["e_theta_deg"]),
        "e_pos": float(m["e_pos_mm"]),
        "obj_z": float(m["obj_z"]),
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "min_pad": int(min(pb["nL"], pb["nR"])),
        "asym": int(abs(int(pb["nL"]) - int(pb["nR"]))),
        "working": int(
            m["e_theta_deg"] <= WORKING_THETA_DEG and m["e_pos_mm"] <= WORKING_POS_MM
        ),
    }


def policy_tok(model, book, rho, obs_seq, prev_seq, device):
    tok, _r, action, logits = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K_WIN)
    if action is None or tok == HANDOFF:
        with torch.no_grad():
            lg, resid, _ = model(
                torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
            )
            lg = lg[-1].clone()
            lg[HANDOFF] = -1e9
            tok = int(torch.argmax(lg).item())
            rr = resid[-1].float().cpu().numpy().astype(np.float32)
            action = decode_m111_action(book, tok, rr, rho=rho)
            logits = lg.detach().float().cpu().numpy()
    else:
        logits = np.asarray(logits, np.float32)
    # entropy of top distribution
    x = logits.astype(np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return int(tok), np.asarray(action, np.float32), ent


def diagnose_one(
    env,
    job,
    model,
    book,
    rho,
    device,
    *,
    p44_row: dict | None,
    max_steps: int = 400,
) -> dict[str, Any]:
    skill = job["skill"]
    if skill in ("A1", "B1", "C"):
        ep = job["ep"]
        ref = ep["ref"]

        def restore():
            restore_start(env, ep)

    else:
        cap, ref = job["capture"], job["ref"]

        def restore():
            restore_capture(env, cap)

    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    n_exp = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    n_exp = min(n_exp, max_steps)

    # --- Pass 1: expert-forced history; model predicts at each expert state ---
    restore()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    first_pred_err = None
    pred_match = 0
    pred_total = 0
    for t in range(n_exp):
        if latched(env.sim):
            break
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        exp_tok = int(toks[t])
        if exp_tok == HANDOFF:
            break
        pred, _a, ent = policy_tok(model, book, rho, obs_seq, prev_seq, device)
        pred_total += 1
        if pred == exp_tok:
            pred_match += 1
        elif first_pred_err is None:
            g = snap_geo(env.sim)
            first_pred_err = {
                "step": t,
                "expert_tok": exp_tok,
                "policy_tok": pred,
                "entropy": ent,
                **g,
            }
        # step expert
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        action = decode_m111_action(book, exp_tok, rr, rho=rho)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)

    # --- Pass 2: closed-loop policy until drop / R3 / max ---
    restore()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    mon = R3FOnline.create(t_max=T_MAX)
    first_bil_loss = None
    drop_step = None
    handoff_step = None
    handoff_reason = "NONE"
    token_switches = 0
    last_tok = None
    geo_at_drop = None
    first_div_closed = None  # first step policy tok != expert tok while still bilateral
    pre_drop_window: list[dict] = []

    for t in range(min(max_steps, T_MAX + 1)):
        if latched(env.sim):
            drop_step = t
            geo_at_drop = snap_geo(env.sim)
            handoff_reason = "DROP_BEFORE"
            break
        st = mon.observe(env.sim, t, tape_ended=False)
        g = snap_geo(env.sim)
        if first_bil_loss is None and not g["bilateral"]:
            first_bil_loss = {"step": t, **g}
        if st["ready"]:
            handoff_step = t
            handoff_reason = st["reason"]
            break
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        pred, action, ent = policy_tok(model, book, rho, obs_seq, prev_seq, device)
        if last_tok is not None and pred != last_tok:
            token_switches += 1
        last_tok = pred
        exp_tok = int(toks[t]) if t < n_exp else None
        if (
            first_div_closed is None
            and exp_tok is not None
            and exp_tok != HANDOFF
            and pred != exp_tok
            and g["bilateral"]
        ):
            first_div_closed = {"step": t, "expert_tok": exp_tok, "policy_tok": pred, "entropy": ent, **g}
        pre_drop_window.append(
            {"step": t, "tok": pred, "exp": exp_tok, "ent": ent, **g}
        )
        if len(pre_drop_window) > 30:
            pre_drop_window.pop(0)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)

    # Classify mechanism
    mechanism = "unknown"
    if handoff_reason != "DROP_BEFORE" and handoff_step is not None:
        mechanism = "reached_handoff"
    elif first_pred_err is not None and first_pred_err["step"] <= 5:
        mechanism = "action_prediction_error_early"
    elif first_pred_err is not None and (
        first_div_closed is None or first_pred_err["step"] <= (first_div_closed["step"] + 2)
    ):
        # On expert state, model already wrong near divergence
        mechanism = "action_prediction_error"
    elif first_div_closed is not None and first_bil_loss is not None:
        if first_div_closed["step"] < first_bil_loss["step"]:
            mechanism = "covariate_shift_then_contact_loss"
        else:
            mechanism = "contact_sensitivity"
    elif first_bil_loss is not None and drop_step is not None:
        mechanism = "contact_sensitivity"
    elif drop_step is not None:
        mechanism = "drop_without_clear_contact_trace"

    # Early contact quality on expert-forced pass start
    restore()
    g0 = snap_geo(env.sim)

    return {
        "traj_id": job["traj_id"],
        "skill": skill,
        "split": job.get("split"),
        "p44_fail": (p44_row or {}).get("fail_category"),
        "p44_reason": (p44_row or {}).get("handoff_reason"),
        "p44_12S": (p44_row or {}).get("SUCCESS_12S"),
        "init_theta": g0["theta"],
        "init_e_pos": g0["e_pos"],
        "init_bilateral": g0["bilateral"],
        "init_nL": g0["nL"],
        "init_nR": g0["nR"],
        "n_expert": n_exp,
        "pred_match_rate": (pred_match / pred_total) if pred_total else None,
        "pred_total": pred_total,
        "first_pred_err_step": None if first_pred_err is None else first_pred_err["step"],
        "first_pred_err_exp": None if first_pred_err is None else first_pred_err["expert_tok"],
        "first_pred_err_pol": None if first_pred_err is None else first_pred_err["policy_tok"],
        "first_pred_err_ent": None if first_pred_err is None else first_pred_err["entropy"],
        "first_pred_err_min_pad": None if first_pred_err is None else first_pred_err["min_pad"],
        "first_div_closed_step": None if first_div_closed is None else first_div_closed["step"],
        "first_bil_loss_step": None if first_bil_loss is None else first_bil_loss["step"],
        "first_bil_loss_nL": None if first_bil_loss is None else first_bil_loss["nL"],
        "first_bil_loss_nR": None if first_bil_loss is None else first_bil_loss["nR"],
        "drop_step": drop_step,
        "handoff_step": handoff_step,
        "handoff_reason": handoff_reason,
        "token_switches_before_end": token_switches,
        "drop_theta": None if geo_at_drop is None else geo_at_drop["theta"],
        "drop_e_pos": None if geo_at_drop is None else geo_at_drop["e_pos"],
        "drop_nL": None if geo_at_drop is None else geo_at_drop["nL"],
        "drop_nR": None if geo_at_drop is None else geo_at_drop["nR"],
        "mechanism": mechanism,
        "pre_drop_tail": pre_drop_window[-8:] if handoff_reason == "DROP_BEFORE" else [],
    }


def write_report(rows: list[dict]) -> None:
    lines = [
        "# FAILURE_DIAGNOSIS — P45 Phase 1",
        "",
        f"Generated: {utc()}",
        f"Checkpoint: `{P44_CKPT}`",
        "",
        "## Scope",
        "",
        "Held-out ICs from P44 `SFT_EVAL_ROLLOUTS.csv`, re-instrumented with",
        "expert-forced prediction match vs closed-loop drop timing.",
        "",
        "## Mechanism counts (all diagnosed)",
        "",
    ]
    by_mech = Counter(r["mechanism"] for r in rows)
    by_skill_mech = Counter((r["skill"], r["mechanism"]) for r in rows)
    lines.append("| Mechanism | n |")
    lines.append("|---|---:|")
    for k, v in by_mech.most_common():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "### Per skill", ""]
    lines.append("| Skill | mechanism | n |")
    lines.append("|---|---|---:|")
    for (sk, m), v in sorted(by_skill_mech.items()):
        lines.append(f"| {sk} | `{m}` | {v} |")

    drops = [r for r in rows if r["handoff_reason"] == "DROP_BEFORE"]
    succ = [r for r in rows if str(r.get("p44_fail")) == "SUCCESS" or r["handoff_reason"] == "R3_EARLY"]
    lines += [
        "",
        "## DROP_BEFORE timing",
        "",
        f"n_drop={len(drops)}",
    ]
    if drops:
        ds = [int(r["drop_step"]) for r in drops if r["drop_step"] is not None]
        bl = [int(r["first_bil_loss_step"]) for r in drops if r["first_bil_loss_step"] is not None]
        pe = [int(r["first_pred_err_step"]) for r in drops if r["first_pred_err_step"] is not None]
        lines.append(
            f"- drop_step median={float(np.median(ds)):.1f} mean={float(np.mean(ds)):.1f} "
            f"min={min(ds)} max={max(ds)}"
        )
        if bl:
            lines.append(
                f"- first_bilateral_loss median={float(np.median(bl)):.1f} "
                f"(before drop in {sum(1 for r in drops if r['first_bil_loss_step'] is not None and r['drop_step'] is not None and r['first_bil_loss_step'] < r['drop_step'])}/{len(drops)})"
            )
        if pe:
            lines.append(
                f"- first_pred_err_on_expert_state median={float(np.median(pe)):.1f} "
                f"early(≤5)={sum(1 for x in pe if x <= 5)}/{len(pe)}"
            )
        pm = [float(r["pred_match_rate"]) for r in drops if r["pred_match_rate"] is not None]
        pms = [float(r["pred_match_rate"]) for r in succ if r["pred_match_rate"] is not None]
        if pm:
            lines.append(f"- expert-state token match rate (DROP): mean={float(np.mean(pm)):.3f}")
        if pms:
            lines.append(f"- expert-state token match rate (SUCCESS/R3): mean={float(np.mean(pms)):.3f}")

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    n_ape = sum(1 for r in drops if "action_prediction" in r["mechanism"])
    n_cov = sum(1 for r in drops if "covariate" in r["mechanism"])
    n_cs = sum(1 for r in drops if r["mechanism"] == "contact_sensitivity")
    lines.append(
        f"- Among DROP_BEFORE: action_prediction*={n_ape}, covariate_shift*={n_cov}, "
        f"contact_sensitivity={n_cs}."
    )
    lines.append(
        "- If action_prediction_error dominates early steps → need more/better BC on early contact tokens "
        "(P45-A expert enrichment) and/or DAgger on those states (P45-B)."
    )
    lines.append(
        "- If covariate_shift dominates → DAgger on policy-visited pre-drop states is the primary lever."
    )
    lines.append(
        "- Skills may differ: report per-skill mechanism table above; do not assume one cause for A/B/C/D."
    )
    lines += [
        "",
        "## Minimal data plan (pending Phase 2)",
        "",
        "1. **P45-A:** upsample early-horizon chunks from P44 expert cuts (steps 0–min(80,cut)) for A/B especially.",
        "2. **P45-B:** on **train** ICs only, collect captures N steps before DROP_BEFORE / first bil loss; "
        "attempt verified expert continuation (capture→oracle/tape segment) with SUCCESS_12S gate.",
        "3. Hold out P44 eval ICs from training labels.",
        "",
        "## Artifacts",
        "",
        "- `DIAGNOSIS_ROWS.csv`",
        "- `FAILURE_DIAGNOSIS.md`",
        "",
    ]
    (OUT / "FAILURE_DIAGNOSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(*, limit: int = 0, max_steps: int = 400) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    p44_rows = list(csv.DictReader(P44_EVAL.open(encoding="utf-8")))
    hold = [r for r in p44_rows if r.get("tag") == "holdout"]
    by_id = {(r["skill"], r["traj_id"]): r for r in hold}

    jobs = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}
    # diagnose all holdout ICs from P44
    targets = []
    for r in hold:
        key = (r["skill"], r["traj_id"])
        j = jobs.get(key)
        if j is None:
            cands = [
                jj
                for (sk, tid), jj in jobs.items()
                if sk == r["skill"] and (tid == r["traj_id"] or r["traj_id"] in tid or tid in r["traj_id"])
            ]
            j = cands[0] if cands else None
        if j is None:
            continue
        targets.append((j, r))
    if limit > 0:
        targets = targets[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P44_CKPT, device)
    model.eval()
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2

    env = make_env()
    env._obs_seq = []
    env._prev_seq = []

    rows = []
    print(f"P45 diagnose n={len(targets)} max_steps={max_steps}", flush=True)
    for i, (job, p44) in enumerate(targets):
        row = diagnose_one(env, job, model, book, rho, device, p44_row=p44, max_steps=max_steps)
        rows.append(row)
        write_csv(OUT / "DIAGNOSIS_ROWS.csv", rows)
        print(
            f"  [{i+1}/{len(targets)}] {job['skill']} {job['traj_id'][:40]} "
            f"mech={row['mechanism']} drop={row['drop_step']} "
            f"pred_err={row['first_pred_err_step']} bil_loss={row['first_bil_loss_step']} "
            f"match={row['pred_match_rate']}",
            flush=True,
        )

    write_report(rows)
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()
    run(limit=args.limit, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
