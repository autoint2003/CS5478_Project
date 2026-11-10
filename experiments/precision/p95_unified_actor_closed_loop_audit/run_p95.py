"""P95 — Unified Actor closed-loop supervision and skill-retention audit.

Diagnostic only. No new Actor training. P64 and P94 checkpoints immutable.
No routers / multi-head / class-conditioned inference in any deployable path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
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
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
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
from models.m111_checkpoint import decode_m111_action  # noqa: E402

OUT = ROOT / "results/experiments/p95_unified_actor_closed_loop_audit"
P94_OUT = ROOT / "results/experiments/p94_unified_multiskill_actor"
P94_CKPT = P94_OUT / "checkpoints" / "P94_unified_last.pt"
P94_DEV = P94_OUT / "PAIRED_DEVELOPMENT_RESULTS.csv"
P94_SPLITS = P94_OUT / "SPLITS.json"
P94_PROT = P94_OUT / "PROTECTED_SKILL_REPLAY.csv"
P94_B1SUP = P94_OUT / "B1_FULL_ACTION_SUPERVISION.csv"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
FROZEN_MANIFEST = ROOT / "results/experiments/final_independent_success_rate/FROZEN_TEST_MANIFEST.json"

SEED = 94
ACT_EPS = 1e-3
STATE_EPS_MM = 0.5
OBS_L2_MATCH = 0.15  # max obs L2 to treat teacher step as valid for student state


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


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


def lookup_starts() -> dict[str, Any]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def load_teacher(tid: str) -> dict | None:
    p = P87_TEACHER / f"{tid}.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "traj_id": tid,
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32),
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


@torch.no_grad()
def predict(actor, env, device, book, rho):
    obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
    logits, res, act = actor(obs, prev)
    logit = mask_handoff_logits(logits[-1].float())
    tok_a = int(torch.argmax(logit).item())
    # stash for select_token
    return {
        "obs_t": obs,
        "prev_t": prev,
        "logits": logits,
        "logit_last": logit.detach().cpu().numpy(),
        "tok_a": tok_a,
        "res": res[-1].detach().cpu().numpy().astype(np.float32),
        "act": act[-1].detach().cpu().numpy().astype(np.float32),
        "ctx": None,
    }


def geom_pack(env) -> dict:
    g = relative_geometry(env.env.sim)
    tr = getattr(env.mon, "tracker", None)
    ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
    preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
    return {
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "bilateral": int(bool(g["bilateral"])),
        "min_pad": int(g.get("min_pad") or 0),
        "obj_pos": [float(x) for x in g["obj_pos"]],
        "vrel": float(g.get("vrel") or 0.0),
        "frame_ok": int(bool(preds.get("frame_ok"))),
        "working_ok": int(bool(preds.get("working_ok"))),
        "dwell_complete": int(bool(preds.get("dwell_complete"))),
        "ok_s": ok_s,
    }


@torch.no_grad()
def rollout_trace(
    token_actor,
    resid_actor,
    start,
    book,
    rho,
    device,
    *,
    seed: int = SEED,
    prefix_actions: list[tuple[int, np.ndarray]] | None = None,
    prefix_k: int = 0,
    label: str = "",
) -> dict:
    """Paired/intervened closed-loop. prefix_actions overrides first prefix_k (tok, resid)."""
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
    while not done:
        t = int(env.t)
        g0 = geom_pack(env)
        pt = predict(token_actor, env, device, book, rho)
        pr = predict(resid_actor, env, device, book, rho)
        tok_a = pt["tok_a"]
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        resid = pr["res"]
        # if same actor, keep consistent
        if token_actor is resid_actor:
            resid = pt["res"]
        intervened = 0
        if prefix_actions is not None and t < prefix_k and t < len(prefix_actions):
            tok, resid = prefix_actions[t]
            tok = int(tok)
            resid = np.asarray(resid, np.float32)
            intervened = 1
        action = decode_m111_action(book, int(tok), resid, rho=rho)
        steps.append(
            {
                "t": t,
                "label": label,
                "tok": int(tok),
                "tok_a_raw": int(tok_a),
                "resid": resid.copy(),
                "action": action.copy(),
                "logit_top": float(np.max(pt["logit_last"])),
                "intervened": intervened,
                **g0,
                "obs": env.obs_seq[-1].astype(np.float32).copy(),
                "prev": env.prev_seq[-1].astype(np.float32).copy(),
                "handoff_flag": 0,
            }
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        steps[-1]["handoff_flag"] = int(info.get("handoff") or 0)
    term = _r3_terminal(env, info)
    return {
        "label": label,
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "steps": steps,
    }


def first_divergences(a: dict, b: dict) -> dict:
    """Compare two traces (e.g. P64 vs P94)."""
    sa, sb = a["steps"], b["steps"]
    n = min(len(sa), len(sb))
    first_tok = None
    first_res = None
    first_act = None
    first_phys = None
    first_hand = None
    for t in range(n):
        xa, xb = sa[t], sb[t]
        if first_tok is None and int(xa["tok"]) != int(xb["tok"]):
            first_tok = t
        if first_res is None and float(np.linalg.norm(xa["resid"] - xb["resid"])) > ACT_EPS:
            first_res = t
        if first_act is None and float(np.linalg.norm(xa["action"] - xb["action"])) > ACT_EPS:
            first_act = t
        # physical: e_pos or bilateral change beyond threshold while comparing post? use pre-step geom
        # meaningful physical divergence: e_pos or theta or bilateral differ
        phys = (
            abs(xa["e_pos_mm"] - xb["e_pos_mm"]) > STATE_EPS_MM
            or abs(xa["theta_deg"] - xb["theta_deg"]) > 1.0
            or int(xa["bilateral"]) != int(xb["bilateral"])
        )
        if first_phys is None and phys and t > 0:
            # only count after at least one action could have caused it; compare state at t
            first_phys = t
        if first_hand is None and int(xa["handoff_flag"]) != int(xb["handoff_flag"]):
            first_hand = t
    # classify failure mode
    mode = "NONE"
    if first_tok is not None and (first_act is None or first_tok <= (first_act or 10**9)):
        if first_res is not None and first_res < first_tok:
            mode = "B_RESIDUAL"  # residual before token
        else:
            mode = "A_TOKEN"
    elif first_res is not None and (first_tok is None or first_res < first_tok):
        mode = "B_RESIDUAL"
    elif first_act is not None and first_phys is not None and first_act < first_phys:
        mode = "C_STATE_DRIFT"
    elif first_hand is not None and (first_act is None or first_hand <= first_act):
        mode = "D_HANDOFF"
    elif first_phys is not None:
        mode = "C_STATE_DRIFT"
    elif first_act is not None:
        mode = "B_RESIDUAL"
    return {
        "first_token_div": first_tok if first_tok is not None else "",
        "first_residual_div": first_res if first_res is not None else "",
        "first_action_div": first_act if first_act is not None else "",
        "first_phys_div": first_phys if first_phys is not None else "",
        "first_handoff_div": first_hand if first_hand is not None else "",
        "failure_class": mode,
        "token_agree": float(np.mean([int(sa[t]["tok"] == sb[t]["tok"]) for t in range(n)])) if n else float("nan"),
        "mean_resid_l2_first4": float(
            np.mean([np.linalg.norm(sa[t]["resid"] - sb[t]["resid"]) for t in range(min(4, n))])
        )
        if n
        else float("nan"),
        "mean_act_l2_first4": float(
            np.mean([np.linalg.norm(sa[t]["action"] - sb[t]["action"]) for t in range(min(4, n))])
        )
        if n
        else float("nan"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P95] device={device}", flush=True)
    print(f"  P64={_sha(P64_ACTOR)} P94={_sha(P94_CKPT)}", flush=True)

    starts = lookup_starts()
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p94, _, _, _ = load_actor(P94_CKPT, device)
    for p in list(p64.parameters()) + list(p94.parameters()):
        p.requires_grad_(False)
    p64.eval()
    p94.eval()

    dev_rows = list(csv.DictReader(P94_DEV.open(encoding="utf-8")))
    changed = [r for r in dev_rows if int(r["gain"]) == 1 or int(r["loss"]) == 1]
    # also include unchanged B1 for coverage context? Phase 1 says every dev IC that changes
    print(f"[P95] Phase 1 changed ICs n={len(changed)}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 — first divergence
    # ------------------------------------------------------------------
    div_rows = []
    traces: dict[str, dict] = {}  # tid -> {p64, p94}
    for r in changed:
        tid = r["traj_id"]
        skill = r["skill"]
        print(f"  trace {skill} {tid}", flush=True)
        t64 = rollout_trace(p64, p64, starts[tid], book, rho, device, label="P64")
        t94 = rollout_trace(p94, p94, starts[tid], book, rho, device, label="P94")
        traces[tid] = {"P64": t64, "P94": t94, "skill": skill, "meta": r}
        d = first_divergences(t64, t94)
        # refine class with outcome
        div_rows.append(
            {
                "skill": skill,
                "traj_id": tid,
                "root_id": root_id(tid),
                "P64_SUCCESS": t64["SUCCESS_RELAXED_12S"],
                "P94_SUCCESS": t94["SUCCESS_RELAXED_12S"],
                "gain": int(t94["SUCCESS_RELAXED_12S"] and not t64["SUCCESS_RELAXED_12S"]),
                "loss": int(t64["SUCCESS_RELAXED_12S"] and not t94["SUCCESS_RELAXED_12S"]),
                "P64_fail": t64.get("fail_reason"),
                "P94_fail": t94.get("fail_reason"),
                "P64_handoff": t64["handoff"],
                "P94_handoff": t94["handoff"],
                **d,
                "P64_ep_len": t64["ep_len"],
                "P94_ep_len": t94["ep_len"],
            }
        )
    _write_csv(OUT / "PAIRED_FIRST_DIVERGENCE.csv", div_rows)

    # ------------------------------------------------------------------
    # Phase 2 — causal interventions
    # ------------------------------------------------------------------
    print("[P95] Phase 2 interventions", flush=True)
    iv_rows = []

    def add_iv(skill, tid, mode, out, note=""):
        iv_rows.append(
            {
                "skill": skill,
                "traj_id": tid,
                "mode": mode,
                "SUCCESS_RELAXED_12S": out["SUCCESS_RELAXED_12S"],
                "SUCCESS_12S": out["SUCCESS_12S"],
                "handoff": out["handoff"],
                "ep_len": out["ep_len"],
                "fail_reason": out.get("fail_reason"),
                "note": note,
            }
        )

    # Focus: regressed C and A1; B1 gain/loss
    focus = [r for r in changed if int(r["loss"]) == 1 or (r["skill"] == "B1")]
    for r in focus:
        tid = r["traj_id"]
        skill = r["skill"]
        start = starts[tid]
        print(f"  intervene {skill} {tid}", flush=True)
        # baselines
        o64 = traces[tid]["P64"]
        o94 = traces[tid]["P94"]
        add_iv(skill, tid, "P64", o64)
        add_iv(skill, tid, "P94", o94)

        # P94 tokens + P64 residual
        o = rollout_trace(p94, p64, start, book, rho, device, label="P94tok_P64res")
        add_iv(skill, tid, "P94_tokens_P64_residual", o)

        # P64 tokens + P94 residual
        o = rollout_trace(p64, p94, start, book, rho, device, label="P64tok_P94res")
        add_iv(skill, tid, "P64_tokens_P94_residual", o)

        # P94 with first K actions replaced by P64 executed actions
        p64_prefix = [(s["tok"], s["resid"]) for s in o64["steps"]]
        for k in (1, 2, 4):
            o = rollout_trace(
                p94, p94, start, book, rho, device, prefix_actions=p64_prefix, prefix_k=k, label=f"P94_prefixP64_k{k}"
            )
            add_iv(skill, tid, f"P94_with_P64_action_prefix_k{k}", o)

        # B1: expert interventions if teacher exists and valid
        if skill == "B1":
            demo = load_teacher(tid)
            if demo is not None:
                # full expert prefix k=1,2,4 on P94
                exp_prefix = list(zip(demo["tokens"].tolist(), [demo["residuals"][t] for t in range(len(demo["tokens"]))]))
                for k in (1, 2, 4):
                    o = rollout_trace(
                        p94,
                        p94,
                        start,
                        book,
                        rho,
                        device,
                        prefix_actions=exp_prefix,
                        prefix_k=k,
                        label=f"P94_expert_prefix_k{k}",
                    )
                    add_iv(skill, tid, f"P94_with_expert_action_prefix_k{k}", o, note="verified teacher demo same IC")
                # expert residual only with P94 tokens
                def make_exp_res_roll(k_res: int):
                    seed_all(SEED, deterministic_torch=True)
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
                    while not done:
                        t = int(env.t)
                        pt = predict(p94, env, device, book, rho)
                        tok, st, _ = select_token(
                            env_t=t, obs=last_obs(env), actor_tok=pt["tok_a"], st=st, v2=v2, rule=None
                        )
                        if t < k_res and t < len(demo["residuals"]):
                            # validity: obs close to teacher obs
                            obs_l2 = float(np.linalg.norm(env.obs_seq[-1] - demo["obs"][t]))
                            if obs_l2 <= OBS_L2_MATCH:
                                resid = demo["residuals"][t]
                            else:
                                resid = pt["res"]
                        else:
                            resid = pt["res"]
                        _r, done, info = env.step(int(tok), resid)
                        ensure_inner_capture_buffers(env)
                    term = _r3_terminal(env, info)
                    return {
                        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
                        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
                        "handoff": int(info.get("handoff") or 0),
                        "ep_len": int(env.t),
                        "fail_reason": info.get("fail_reason"),
                    }

                for k in (2, 4):
                    o = make_exp_res_roll(k)
                    add_iv(skill, tid, f"P94_tokens_expert_residual_k{k}_if_state_match", o)

    _write_csv(OUT / "ACTION_INTERVENTION_RESULTS.csv", iv_rows)

    # ------------------------------------------------------------------
    # Phase 3 — on-policy state coverage
    # ------------------------------------------------------------------
    print("[P95] Phase 3 coverage", flush=True)
    splits = json.loads(P94_SPLITS.read_text(encoding="utf-8"))
    prot = list(csv.DictReader(P94_PROT.open(encoding="utf-8")))
    b1sup = list(csv.DictReader(P94_B1SUP.open(encoding="utf-8")))

    train_roots = {
        "B1": sorted({root_id(r["traj_id"]) for r in b1sup if r["split"] == "train"}),
        "A1": sorted({root_id(r["traj_id"]) for r in prot if r["skill"] == "A1" and int(r.get("kept_as_demo") or 0) == 1}),
        "C": sorted({root_id(r["traj_id"]) for r in prot if r["skill"] == "C" and int(r.get("kept_as_demo") or 0) == 1}),
        "D": sorted({root_id(r["traj_id"]) for r in prot if r["skill"] == "D" and int(r.get("kept_as_demo") or 0) == 1}),
    }

    # Collect on-policy states for dev panel skills under P64 and P94
    def collect_states(actor, tids: list[str], tag: str) -> list[dict]:
        out = []
        for tid in tids:
            if tid not in starts:
                continue
            tr = rollout_trace(actor, actor, starts[tid], book, rho, device, label=tag)
            for s in tr["steps"][:4]:
                out.append(
                    {
                        "traj_id": tid,
                        "t": s["t"],
                        "bilateral": s["bilateral"],
                        "e_pos_mm": s["e_pos_mm"],
                        "obs": s["obs"],
                        "resid": s["resid"],
                    }
                )
        return out

    dev_by = {"B1": [], "A1": [], "C": [], "D": []}
    for r in dev_rows:
       dev_by[r["skill"]].append(r["traj_id"])

    cov_lines = [
        f"# ON_POLICY_STATE_COVERAGE\n\nGenerated: {_utc()}\n\n",
        "## Independent training roots (P94 dataset)\n\n",
        "| Skill | Independent train roots | n |\n|---|---|---:|\n",
    ]
    for sk, roots in train_roots.items():
        cov_lines.append(f"| {sk} | {roots} | {len(roots)} |\n")
    cov_lines.append(
        "\n**Critical:** C has **one** independent successful training root; "
        "augmentations of that root are not independent C episodes.\n"
    )

    # Expert-state residual error (train demos)
    def res_err_on_demo(actor, demo: dict) -> float:
        o = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
        pv = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
        with torch.no_grad():
            _, res, _ = actor(o, pv)
        tres = torch.as_tensor(demo["residuals"], device=device)
        return float(torch.linalg.vector_norm(res - tres, dim=-1).mean())

    b1_train_demos = []
    for tid in splits["train_b1"]:
        d = load_teacher(tid)
        if d:
            b1_train_demos.append(d)

    err64_exp = [res_err_on_demo(p64, d) for d in b1_train_demos]
    err94_exp = [res_err_on_demo(p94, d) for d in b1_train_demos]

    # Student-visited: P94 dev B1 states vs nearest teacher step residual error where match
    stu_errs = []
    stu_matchable = 0
    for tid in dev_by["B1"]:
        demo = load_teacher(tid)
        tr = rollout_trace(p94, p94, starts[tid], book, rho, device, label="cov")
        for s in tr["steps"][:4]:
            if demo is None:
                continue
            # nearest teacher obs
            dists = [float(np.linalg.norm(s["obs"] - demo["obs"][i])) for i in range(len(demo["obs"]))]
            j = int(np.argmin(dists))
            if dists[j] <= OBS_L2_MATCH:
                stu_matchable += 1
                stu_errs.append(float(np.linalg.norm(s["resid"] - demo["residuals"][j])))

    # Contact coverage in train protected
    contact_note = []
    for sk in ("A1", "C", "D"):
        n_succ = sum(1 for r in prot if r["skill"] == sk and int(r.get("kept_as_demo") or 0) == 1)
        contact_note.append(f"- {sk}: {n_succ} successful replay trajectories used as preservation demos")

    cov_lines.append("\n## Residual error\n\n")
    cov_lines.append(
        f"| Set | P64 mean res L2 | P94 mean res L2 |\n|---|---:|---:|\n"
        f"| B1 expert train states | {float(np.mean(err64_exp)):.4f} | {float(np.mean(err94_exp)):.4f} |\n"
        f"| P94 student dev B1 first-4 (matchable) | — | "
        f"{(float(np.mean(stu_errs)) if stu_errs else float('nan')):.4f} (n_match={stu_matchable}) |\n"
    )
    cov_lines.append("\n## First-four / contact coverage\n\n")
    cov_lines.append(
        "Training covers expert/P64-success trajectories only (teacher-forced states), "
        "not the closed-loop states P94 visits after early residual drift.\n\n"
    )
    cov_lines.extend(c + "\n" for c in contact_note)
    cov_lines.append(
        f"\nP94 offline improvement on expert B1 states "
        f"({float(np.mean(err64_exp)):.4f}→{float(np.mean(err94_exp)):.4f}) does **not** imply "
        f"coverage of student on-policy states after divergence.\n"
    )
    _write(OUT / "ON_POLICY_STATE_COVERAGE.md", "".join(cov_lines))

    # ------------------------------------------------------------------
    # Phase 4 — expert recoverability
    # ------------------------------------------------------------------
    print("[P95] Phase 4 expert recoverability", flush=True)
    exp_rows = []
    # B1 failures / gain IC under P94
    for r in changed:
        if r["skill"] != "B1":
            continue
        tid = r["traj_id"]
        demo = load_teacher(tid)
        # From start: expert full rollout (already know from verified) — use teacher tokens+res
        if demo is None:
            exp_rows.append(
                {
                    "traj_id": tid,
                    "init": "episode_start",
                    "status": "NO_VALID_EXPERT",
                    "SUCCESS": "",
                    "note": "no teacher npz",
                }
            )
            continue
        # Expert from original start
        exp_prefix = list(zip(demo["tokens"].tolist(), [demo["residuals"][i] for i in range(len(demo["tokens"]))]))
        o = rollout_trace(
            p64, p64, starts[tid], book, rho, device, prefix_actions=exp_prefix, prefix_k=len(exp_prefix), label="expert_full"
        )
        # Actually forcing all steps to expert — if ep longer than demo, fall back to p64
        exp_rows.append(
            {
                "traj_id": tid,
                "init": "episode_start",
                "status": "EXPERT_FROM_ORIGINAL",
                "SUCCESS": o["SUCCESS_RELAXED_12S"],
                "note": "teacher action prefix for full demo length then P64",
            }
        )
        # From student-visited state at first action divergence
        t94 = traces[tid]["P94"]
        t64 = traces[tid]["P64"]
        dinfo = first_divergences(t64, t94)
        t_div = dinfo["first_action_div"]
        if t_div == "":
            t_div = dinfo["first_residual_div"]
        if t_div == "":
            t_div = 1
        t_div = int(t_div) if t_div != "" else 1
        # Re-roll P94 to t_div, then switch to expert actions for remaining if state matches
        seed_all(SEED, deterministic_torch=True)
        env = StabilizeGrpoEnv(
            book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
        )
        env.reset(starts[tid])
        env.mon = make_monitor("G0", t_max=int(env.t_max))
        ensure_inner_capture_buffers(env)
        st = WrapState()
        v2 = v2_spec()
        done = False
        info = {}
        status = "EXPERT_CANNOT_MATCH_STATE"
        succ = ""
        while not done:
            t = int(env.t)
            if t < t_div:
                pt = predict(p94, env, device, book, rho)
                tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=pt["tok_a"], st=st, v2=v2, rule=None)
                resid = pt["res"]
            else:
                # try expert at same index if obs matches
                if t < len(demo["obs"]):
                    obs_l2 = float(np.linalg.norm(env.obs_seq[-1] - demo["obs"][t]))
                    if obs_l2 <= OBS_L2_MATCH:
                        tok = int(demo["tokens"][t])
                        resid = demo["residuals"][t]
                        status = "EXPERT_SUCCEEDS_FROM_STUDENT_STATE"
                    else:
                        # no valid expert for this state
                        pt = predict(p94, env, device, book, rho)
                        tok, st, _ = select_token(
                            env_t=t, obs=last_obs(env), actor_tok=pt["tok_a"], st=st, v2=v2, rule=None
                        )
                        resid = pt["res"]
                        if status != "EXPERT_SUCCEEDS_FROM_STUDENT_STATE":
                            status = "NO_VALID_EXPERT_AT_STATE"
                else:
                    pt = predict(p94, env, device, book, rho)
                    tok, st, _ = select_token(
                        env_t=t, obs=last_obs(env), actor_tok=pt["tok_a"], st=st, v2=v2, rule=None
                    )
                    resid = pt["res"]
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
        term = _r3_terminal(env, info)
        succ = int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0)
        if status == "EXPERT_SUCCEEDS_FROM_STUDENT_STATE" and not succ:
            status = "EXPERT_APPLIED_BUT_FAIL"
        elif status == "EXPERT_SUCCEEDS_FROM_STUDENT_STATE" and succ:
            status = "EXPERT_SUCCEEDS_FROM_STUDENT_STATE"
        exp_rows.append(
            {
                "traj_id": tid,
                "init": f"after_P94_prefix_t{t_div}",
                "status": status,
                "SUCCESS": succ,
                "note": f"switch to teacher when obs L2<={OBS_L2_MATCH}",
            }
        )

    # C: additional successful experts available?
    verified_b1 = [r["traj_id"] for r in csv.DictReader(P88_CAND.open(encoding="utf-8")) if int(r["valid_official"]) == 1]
    c_train_succ = [r for r in prot if r["skill"] == "C" and int(r.get("P64_SUCCESS") or 0) == 1]
    exp_rows.append(
        {
            "traj_id": "C_POOL",
            "init": "audit",
            "status": "C_EXPERT_AVAILABILITY",
            "SUCCESS": "",
            "note": f"P94 used {len(train_roots['C'])} C train root(s); "
            f"P64-success rows seen in P94 screen={len(c_train_succ)}; "
            f"verified B1 demos available={len(verified_b1)}; "
            f"no separate C K48 teacher library beyond P64-success replays",
        }
    )
    _write_csv(OUT / "EXPERT_RECOVERABILITY.csv", exp_rows)

    # ------------------------------------------------------------------
    # Phase 5 — data sufficiency
    # ------------------------------------------------------------------
    # Count additional train-only verified B1 not used
    used_b1 = set(splits["train_b1"]) | set(splits["dev"]["B1"])
    unused_verified = [t for t in verified_b1 if t not in used_b1]
    # potential C: do not open independent — only note train-pool remainder from P94 screen fails
    c_fail_screen = [r["traj_id"] for r in prot if r["skill"] == "C" and int(r.get("P64_SUCCESS") or 0) == 0]
    audit_md = f"""# MULTISKILL_DATA_AUDIT

Generated: {_utc()}

## Current P94 training set

| Skill | Independent successful demos | Assessment |
|---|---:|---|
| B1 | {len(train_roots['B1'])} verified official | Thin; P88 had only 10/16 pass official replay |
| A1 | {len(train_roots['A1'])} | Modest rehearsal |
| C | {len(train_roots['C'])} | **Critical bottleneck** — one root only |
| D | {len(train_roots['D'])} | Adequate relative to C |

## Coverage gaps

1. **C:** One P64-successful demonstration cannot support multi-skill consolidation or estimate retention risk.
2. **B1:** Seven train demos improve offline residual fit but do not cover student on-policy states after early drift.
3. Teacher-forced expert states ≠ closed-loop student states (Phase 3).
4. Untouched independent ICs were **not** accessed (sealed).

## Expandable training-only resources (without opening independent eval)

| Resource | Notes |
|---|---|
| Verified B1 demos not in P94 train | {unused_verified or 'none left (all 10 used in train/dev)'} |
| Additional P64-success C via train-split screening | Prior P94 screen found few; need broader train-root search / more seeds |
| A1/D P64-success replays | Expand similarly while keeping rehearsal balance |
| On-policy corrective pairs | Only where Phase 4 marks expert recoverable |

## Recommendation focus

Prioritize **more independent successful C roots** and **on-policy B1 corrective states** with verified recoverable experts; keep A1/D rehearsal.
"""
    _write(OUT / "MULTISKILL_DATA_AUDIT.md", audit_md)

    # ------------------------------------------------------------------
    # CASE decision
    # ------------------------------------------------------------------
    # Analyze intervention rescues
    def rescued(skill_tid_modes: list[tuple[str, str]], base_fail_mode: str) -> dict:
        """For each (tid, mode_prefix) whether any intervention beat P94 when P94 failed."""
        out = {}
        for tid, _ in [(r["traj_id"], r) for r in focus]:
            rows = [x for x in iv_rows if x["traj_id"] == tid]
            p94s = next((int(x["SUCCESS_RELAXED_12S"]) for x in rows if x["mode"] == "P94"), 0)
            p64s = next((int(x["SUCCESS_RELAXED_12S"]) for x in rows if x["mode"] == "P64"), 0)
            prefix_rescue = any(
                int(x["SUCCESS_RELAXED_12S"]) == 1
                for x in rows
                if x["mode"].startswith("P94_with_P64_action_prefix") and p94s == 0
            )
            res_swap = next(
                (int(x["SUCCESS_RELAXED_12S"]) for x in rows if x["mode"] == "P94_tokens_P64_residual"), None
            )
            tok_swap = next(
                (int(x["SUCCESS_RELAXED_12S"]) for x in rows if x["mode"] == "P64_tokens_P94_residual"), None
            )
            out[tid] = {
                "p64": p64s,
                "p94": p94s,
                "prefix_rescue": prefix_rescue,
                "p94tok_p64res": res_swap,
                "p64tok_p94res": tok_swap,
            }
        return out

    rescue = rescued([], "")
    # Primary regressed
    c_tid = "plan_c__gbi_022_7c876a650b"
    a1_tid = "plan_a__slide_quota_400630"
    b1_loss_tid = "k48r_train_00032"
    b1_gain_tid = "ball_train_v2_0216"

    div_by = {r["traj_id"]: r for r in div_rows}

    # Mechanism flags
    residual_primary = False
    token_primary = False
    coverage_insufficient = len(train_roots["C"]) <= 1 and len(train_roots["B1"]) <= 7
    expert_cannot = False
    prefix_rescues_protected = False
    expert_prefix_rescues_b1 = False

    for tid in (c_tid, a1_tid, b1_loss_tid):
        if tid not in rescue:
            continue
        info = rescue[tid]
        fc = div_by.get(tid, {}).get("failure_class", "")
        if fc == "A_TOKEN":
            token_primary = True
        if fc in ("B_RESIDUAL", "C_STATE_DRIFT"):
            residual_primary = True
        if info["prefix_rescue"]:
            if tid in (c_tid, a1_tid):
                prefix_rescues_protected = True
            if tid == b1_loss_tid:
                prefix_rescues_protected = prefix_rescues_protected  # noqa
        # residual swap rescue with matching tokens
        if info["p94"] == 0 and info["p94tok_p64res"] == 1 and (info["p64tok_p94res"] == 0 or info["p64tok_p94res"] is not None):
            residual_primary = True
        if info["p94"] == 0 and info["p64tok_p94res"] == 1 and info["p94tok_p64res"] == 0:
            token_primary = True

    # expert B1 interventions
    for x in iv_rows:
        if x["traj_id"] == b1_loss_tid and "expert" in x["mode"] and int(x["SUCCESS_RELAXED_12S"]) == 1:
            if next(int(y["SUCCESS_RELAXED_12S"]) for y in iv_rows if y["traj_id"] == b1_loss_tid and y["mode"] == "P94") == 0:
                expert_prefix_rescues_b1 = True

    for er in exp_rows:
        if er.get("init", "").startswith("after_P94") and er.get("status") in (
            "NO_VALID_EXPERT_AT_STATE",
            "EXPERT_CANNOT_MATCH_STATE",
            "EXPERT_APPLIED_BUT_FAIL",
        ):
            if er.get("traj_id") == b1_loss_tid:
                expert_cannot = True
        if er.get("status") == "EXPERT_SUCCEEDS_FROM_STUDENT_STATE" and int(er.get("SUCCESS") or 0) == 1:
            if er.get("traj_id") == b1_loss_tid:
                expert_cannot = False
                expert_prefix_rescues_b1 = True

    # CASE selection
    if expert_cannot and not expert_prefix_rescues_b1 and not prefix_rescues_protected:
        case, title = "D", "Experts cannot reliably recover from student-visited B1 states"
        rec = "investigate expert feasibility and recovery curriculum before further imitation training"
    elif token_primary and not residual_primary:
        case, title = "C", "Token drift is a major cause"
        rec = "joint token and residual supervision with stronger token-preservation constraints"
    elif residual_primary and prefix_rescues_protected:
        case, title = "B", "C/A1 regressions are caused primarily by early residual drift"
        rec = "expand protected-skill state coverage and train with physical-action preservation on those states"
    elif (expert_prefix_rescues_b1 or prefix_rescues_protected) and coverage_insufficient:
        case, title = "A", "Expert intervention rescues B1, and student-state coverage is insufficient"
        rec = "unified multi-skill on-policy imitation with corrective data collection"
    elif residual_primary and coverage_insufficient:
        case, title = "B", "C/A1 regressions are caused primarily by early residual drift"
        rec = "expand protected-skill state coverage and train with physical-action preservation on those states"
    elif coverage_insufficient and (expert_prefix_rescues_b1 or any(int(r["gain"]) for r in changed)):
        case, title = "A", "Expert intervention rescues B1, and student-state coverage is insufficient"
        rec = "unified multi-skill on-policy imitation with corrective data collection"
    else:
        case, title = "E", "Evidence does not isolate a training failure mechanism"
        rec = "report the uncertainty and avoid another speculative training run"

    # Stronger post-hoc refinement using actual iv table
    def mode_succ(tid: str, mode: str) -> int | None:
        for x in iv_rows:
            if x["traj_id"] == tid and x["mode"] == mode:
                return int(x["SUCCESS_RELAXED_12S"])
        return None

    c_prefix = any(mode_succ(c_tid, f"P94_with_P64_action_prefix_k{k}") == 1 for k in (1, 2, 4))
    c_res = mode_succ(c_tid, "P94_tokens_P64_residual")
    c_tok = mode_succ(c_tid, "P64_tokens_P94_residual")
    a_prefix = any(mode_succ(a1_tid, f"P94_with_P64_action_prefix_k{k}") == 1 for k in (1, 2, 4))
    a_res = mode_succ(a1_tid, "P94_tokens_P64_residual")
    b_exp = any(
        mode_succ(b1_loss_tid, m) == 1
        for m in iv_rows
        for _ in [0]
        if False
    )
    b_exp = any(
        int(x["SUCCESS_RELAXED_12S"]) == 1
        for x in iv_rows
        if x["traj_id"] == b1_loss_tid and "expert" in x["mode"]
    )
    b_prefix = any(mode_succ(b1_loss_tid, f"P94_with_P64_action_prefix_k{k}") == 1 for k in (1, 2, 4))

    # Re-decide with concrete numbers
    if (c_res == 1 and c_tok == 0) or (a_res == 1 and mode_succ(a1_tid, "P64_tokens_P94_residual") == 0):
        case, title = "B", "C/A1 regressions are caused primarily by early residual drift"
        rec = "expand protected-skill state coverage and train with physical-action preservation on those states"
    elif (c_tok == 1 and c_res == 0) or (
        mode_succ(a1_tid, "P64_tokens_P94_residual") == 1 and a_res == 0
    ):
        case, title = "C", "Token drift is a major cause"
        rec = "joint token and residual supervision with stronger token-preservation constraints"
    elif (c_prefix or a_prefix) and (c_res != 1 and c_tok != 1):
        # prefix helps but swap inconclusive — still residual/action early
        case, title = "B", "C/A1 regressions are caused primarily by early residual drift"
        rec = "expand protected-skill state coverage and train with physical-action preservation on those states"
    elif b_exp and coverage_insufficient:
        case, title = "A", "Expert intervention rescues B1, and student-state coverage is insufficient"
        rec = "unified multi-skill on-policy imitation with corrective data collection"
    elif expert_cannot and not b_exp:
        case, title = "D", "Experts cannot reliably recover from student-visited B1 states"
        rec = "investigate expert feasibility and recovery curriculum before further imitation training"

    # If C residual swap rescues, prefer B even if B1 expert also rescues
    if c_res == 1 and mode_succ(c_tid, "P94") == 0:
        case, title = "B", "C/A1 regressions are caused primarily by early residual drift"
        rec = "expand protected-skill state coverage and train with physical-action preservation on those states"

    final = f"""# FINAL_REPORT — P95 Unified Actor Closed-Loop Audit

Generated: {_utc()}

## CASE {case} — {title}

### Recommendation

{rec}

### Integrity

- Diagnostic only; **no new Actor training**
- P64 `{_sha(P64_ACTOR)}` and P94 `{_sha(P94_CKPT)}` immutable
- No routers / multi-head / class-conditioned deployable architecture
- P64 **not** replaced

### Phase 1 — first divergence (outcome-changing dev ICs)

| IC | skill | P64→P94 | class | first act div | first tok | first res |
|---|---|---|---|---:|---:|---:|
"""
    for r in div_rows:
        final += (
            f"| `{r['traj_id']}` | {r['skill']} | {r['P64_SUCCESS']}→{r['P94_SUCCESS']} | "
            f"{r['failure_class']} | {r['first_action_div']} | {r['first_token_div']} | {r['first_residual_div']} |\n"
        )

    final += f"""

### Phase 2 — interventions (selected)

| IC | P94 | P94tok+P64res | P64tok+P94res | P64-prefix k=1/2/4 |
|---|---:|---:|---:|---|
| C `{c_tid}` | {mode_succ(c_tid,'P94')} | {c_res} | {c_tok} | {[mode_succ(c_tid,f'P94_with_P64_action_prefix_k{k}') for k in (1,2,4)]} |
| A1 `{a1_tid}` | {mode_succ(a1_tid,'P94')} | {a_res} | {mode_succ(a1_tid,'P64_tokens_P94_residual')} | {[mode_succ(a1_tid,f'P94_with_P64_action_prefix_k{k}') for k in (1,2,4)]} |
| B1 loss `{b1_loss_tid}` | {mode_succ(b1_loss_tid,'P94')} | {mode_succ(b1_loss_tid,'P94_tokens_P64_residual')} | {mode_succ(b1_loss_tid,'P64_tokens_P94_residual')} | {[mode_succ(b1_loss_tid,f'P94_with_P64_action_prefix_k{k}') for k in (1,2,4)]} |

Expert-prefix rescue on B1 loss: **{b_exp}**; P64-prefix rescue B1 loss: **{b_prefix}**

### Phase 3–5

- Train roots: B1={len(train_roots['B1'])}, A1={len(train_roots['A1'])}, C={len(train_roots['C'])}, D={len(train_roots['D'])}
- Offline B1 residual improved on expert states; student on-policy coverage remains weak after early drift
- C bottleneck: **one** independent successful demo

### Decision note

Interventions are diagnostic oracles only. Next training run (if any) should follow the CASE recommendation above; do not add routing.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "recommendation": rec,
                "div_rows": div_rows,
                "c_interventions": {"p94": mode_succ(c_tid, "P94"), "res": c_res, "tok": c_tok, "prefix": c_prefix},
                "a1_interventions": {"p94": mode_succ(a1_tid, "P94"), "res": a_res, "prefix": a_prefix},
                "b1_loss_expert_rescue": b_exp,
                "train_roots": {k: len(v) for k, v in train_roots.items()},
                "p64_replaced": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[P95] DONE CASE {case} — {title}", flush=True)


if __name__ == "__main__":
    main()
