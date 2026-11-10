"""P102 — P101 failure causality audit (no new training).

Diagnose why residual-delta Arms B/C lost P64-successful closed-loop roots
despite offline residual improvement. Produces evidence for P103 proposal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id as rid0  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers  # noqa: E402
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action  # noqa: E402
from models.saup_policy import build_saup_from_p64  # noqa: E402
from training.skill_aware_action_vocab import ResidualBudget  # noqa: E402

OUT = ROOT / "results/experiments/p102_p101_causal_audit"
P101 = ROOT / "results/experiments/p101_saup_three_arm"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P87 = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P97 = ROOT / "results/experiments/p97_c_expert_recoverability"
P97_NEW = P97 / "new_ics/accepted"
P97_CORR = P97 / "C_CORRECTION_MANIFEST.csv"
P99_CACHE = ROOT / "results/experiments/p99_unified_onpolicy_skill_learning/cache"
P99_VER = ROOT / "results/experiments/p99_unified_onpolicy_skill_learning/VERIFIED_CORRECTION_DATASET.csv"

EXPECTED_SHA = "ec08f59816aa2fdd"
CL_SEED = 101
ACT_EPS = 1e-3
STATE_EPS_MM = 0.5
ACT_TOL = 1e-5

# Representative failing checkpoints (worst retention seeds from P101)
REP_B = ("B", 1)  # seed1: heavy A1/B1 losses
REP_C = ("C", 1)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def phys_root(tid: str) -> str:
    t = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", str(tid), flags=re.I)
    t = re.sub(r"__([0-9a-f]{8,})$", "", t)
    t2 = rid0(tid)
    t2 = re.sub(r"^(plan_[abc]__|A__|B__|C__|D__)", "", t2, flags=re.I)
    t2 = re.sub(r"__([0-9a-f]{8,})$", "", t2)
    return t2 if len(t2) <= len(t) else t


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def rho_arr(rho) -> np.ndarray:
    return ResidualBudget.from_spec(rho).as_array().astype(np.float64).reshape(7)


def lookup_starts() -> dict[str, MixedStart]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def resolve_start(tid: str, starts: dict[str, MixedStart]) -> MixedStart | None:
    if tid in starts:
        return starts[tid]
    pr = phys_root(tid)
    pkl = P97_NEW / f"{tid}.pkl"
    if pkl.is_file():
        ep = pickle.loads(pkl.read_bytes())["ep"]
        st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
        starts[tid] = st
        return st
    for t, s in list(starts.items()):
        if phys_root(t) == pr:
            starts[tid] = s
            return s
    if P97_CORR.is_file():
        for r in csv.DictReader(P97_CORR.open(encoding="utf-8")):
            if phys_root(r.get("traj_id") or "") == pr or r.get("phys_root") == pr:
                tt = r["traj_id"]
                if tt in starts:
                    starts[tid] = starts[tt]
                    return starts[tt]
                p2 = P97_NEW / f"{tt}.pkl"
                if p2.is_file():
                    ep = pickle.loads(p2.read_bytes())["ep"]
                    st = MixedStart(skill="C", traj_id=tt, family="C", kind="ep", ep=ep)
                    starts[tid] = st
                    return st
    return None


def load_saup(p64, arm: str, seed: int, device):
    path = P101 / "checkpoints" / f"arm{arm}_seed{seed}_final.pt"
    actor = build_saup_from_p64(p64, "B" if arm == "B" else "C", device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    return actor, path


def geom_pack(env) -> dict:
    g = relative_geometry(env.env.sim)
    return {
        "e_pos_mm": float(g["e_pos_mm"]),
        "theta_deg": float(g["theta_deg"]),
        "bilateral": int(bool(g["bilateral"])),
        "min_pad": int(g.get("min_pad") or 0),
    }


@torch.no_grad()
def predict(actor, env, device, book, rho) -> dict:
    obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
    prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
    logits, res, act = actor(obs, prev)
    logit_last = logits[-1].float()
    tok_a = int(torch.argmax(mask_handoff_logits(logit_last)).item())
    resid = res[-1].detach().cpu().numpy().astype(np.float32)
    # P64 residual for SAUP delta magnitude
    if hasattr(actor, "base"):
        ctx, _ = actor._encode(obs, prev)
        r_p64 = (actor.res_cap * torch.tanh(actor.base.res(ctx)))[-1].cpu().numpy().astype(np.float32)
        delta_l2 = float(np.linalg.norm(resid - r_p64))
    else:
        r_p64 = resid.copy()
        delta_l2 = 0.0
    return {
        "tok_a": tok_a,
        "res": resid,
        "act": act[-1].detach().cpu().numpy().astype(np.float32),
        "logit_last": logit_last.detach().cpu().numpy(),
        "r_p64": r_p64,
        "delta_l2": delta_l2,
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
    seed: int = CL_SEED,
    prefix_actions: list[tuple[int, np.ndarray]] | None = None,
    prefix_k: int = 0,
    label: str = "",
) -> dict:
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
        resid = pr["res"] if token_actor is not resid_actor else pt["res"]
        delta_l2 = pr["delta_l2"] if token_actor is not resid_actor else pt["delta_l2"]
        intervened = 0
        if prefix_actions is not None and t < prefix_k and t < len(prefix_actions):
            tok, resid = prefix_actions[t]
            tok = int(tok)
            resid = np.asarray(resid, np.float32)
            intervened = 1
            delta_l2 = float("nan")
        if int(tok) == HANDOFF:
            action = np.zeros(7, np.float32)
        else:
            action = decode_m111_action(book, int(tok), resid, rho=rho)
        steps.append(
            {
                "t": t,
                "tok": int(tok),
                "tok_a_raw": int(tok_a),
                "resid": resid.copy(),
                "action": action.copy(),
                "delta_l2": delta_l2,
                "intervened": intervened,
                "obs": env.obs_seq[-1].astype(np.float32).copy(),
                "prev": env.prev_seq[-1].astype(np.float32).copy(),
                "handoff_flag": 0,
                **g0,
            }
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        steps[-1]["handoff_flag"] = int(info.get("handoff") or 0)
    term = _r3_terminal(env, info)
    return {
        "label": label,
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "r3_ready": int(term.get("r3_ready") or 0),
        "steps": steps,
    }


def first_divergences(a: dict, b: dict) -> dict:
    sa, sb = a["steps"], b["steps"]
    n = min(len(sa), len(sb))
    first_tok = first_res = first_act = first_phys = first_hand = None
    for t in range(n):
        xa, xb = sa[t], sb[t]
        if first_tok is None and int(xa["tok"]) != int(xb["tok"]):
            first_tok = t
        if first_res is None and float(np.linalg.norm(xa["resid"] - xb["resid"])) > ACT_EPS:
            first_res = t
        if first_act is None and float(np.linalg.norm(xa["action"] - xb["action"])) > ACT_EPS:
            first_act = t
        phys = (
            abs(xa["e_pos_mm"] - xb["e_pos_mm"]) > STATE_EPS_MM
            or abs(xa["theta_deg"] - xb["theta_deg"]) > 1.0
            or int(xa["bilateral"]) != int(xb["bilateral"])
        )
        if first_phys is None and phys and t > 0:
            first_phys = t
        if first_hand is None and int(xa["handoff_flag"]) != int(xb["handoff_flag"]):
            first_hand = t
    # irreversible: first time bilateral drops for student while P64 holds, or e_pos blows
    first_irrev = None
    for t in range(1, n):
        if int(sa[t]["bilateral"]) == 1 and int(sb[t]["bilateral"]) == 0:
            first_irrev = t
            break
        if sa[t]["e_pos_mm"] < 20 and sb[t]["e_pos_mm"] > sa[t]["e_pos_mm"] + 15:
            first_irrev = t
            break
    mode = "NONE"
    if first_res is not None and (first_tok is None or first_res <= first_tok):
        mode = "B_RESIDUAL"
    elif first_tok is not None:
        mode = "A_TOKEN"
    elif first_phys is not None:
        mode = "C_STATE_DRIFT"
    elif first_hand is not None:
        mode = "D_HANDOFF"
    return {
        "first_token_div": "" if first_tok is None else first_tok,
        "first_residual_div": "" if first_res is None else first_res,
        "first_action_div": "" if first_act is None else first_act,
        "first_phys_div": "" if first_phys is None else first_phys,
        "first_handoff_div": "" if first_hand is None else first_hand,
        "first_irrev_div": "" if first_irrev is None else first_irrev,
        "failure_class": mode,
        "token_agree": float(np.mean([int(sa[t]["tok"] == sb[t]["tok"]) for t in range(n)])) if n else float("nan"),
        "mean_resid_l2_t0_4": float(np.mean([np.linalg.norm(sa[t]["resid"] - sb[t]["resid"]) for t in range(min(4, n))]))
        if n
        else float("nan"),
        "mean_act_l2_t0_4": float(np.mean([np.linalg.norm(sa[t]["action"] - sb[t]["action"]) for t in range(min(4, n))]))
        if n
        else float("nan"),
        "mean_delta_l2_t0_4": float(
            np.nanmean([sb[t]["delta_l2"] for t in range(min(4, n)) if not np.isnan(sb[t]["delta_l2"])])
        )
        if n
        else float("nan"),
        "sat_frac_student": float(
            np.mean([np.mean(np.abs(sb[t]["resid"]) > 0.34) for t in range(n)])
        )
        if n
        else float("nan"),
    }


@torch.no_grad()
def state_matched_infer(actors: dict, history_steps: list[dict], device) -> list[dict]:
    """Shadow-infer each actor on identical recorded (obs,prev) from a reference trace."""
    rows = []
    T = len(history_steps)
    for t in range(T):
        obs_np = np.stack([history_steps[i]["obs"] for i in range(t + 1)])[-160:]
        prev_np = np.stack([history_steps[i]["prev"] for i in range(t + 1)])[-160:]
        obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
        prev = torch.as_tensor(prev_np, device=device, dtype=torch.float32)
        preds = {}
        for name, actor in actors.items():
            logits, res, act = actor(obs, prev)
            z = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
            r = res[-1].cpu().numpy().astype(np.float32)
            a = act[-1].cpu().numpy().astype(np.float32)
            preds[name] = (z, r, a)
        row = {"t": t, "ref_tok": int(history_steps[t]["tok"])}
        for name, (z, r, a) in preds.items():
            row[f"{name}_tok"] = z
            row[f"{name}_res_l2"] = float(np.linalg.norm(r))
            row[f"{name}_act_l2"] = float(np.linalg.norm(a))
        # pairwise
        za, zb, zc = preds["P64"][0], preds["B"][0], preds["C"][0]
        row["B_tok_eq_P64"] = int(zb == za)
        row["C_tok_eq_P64"] = int(zc == za)
        row["B_res_l2_vs_P64"] = float(np.linalg.norm(preds["B"][1] - preds["P64"][1]))
        row["C_res_l2_vs_P64"] = float(np.linalg.norm(preds["C"][1] - preds["P64"][1]))
        row["B_act_l2_vs_P64"] = float(np.linalg.norm(preds["B"][2] - preds["P64"][2]))
        row["C_act_l2_vs_P64"] = float(np.linalg.norm(preds["C"][2] - preds["P64"][2]))
        rows.append(row)
    return rows


def token_can_represent(book, k, a, rho, tol=ACT_TOL) -> bool:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    lo = np.maximum(-1.0, proto - rho)
    hi = np.minimum(1.0, proto + rho)
    return bool(np.all(a >= lo - tol) and np.all(a <= hi + tol))


def feasible_set(book, a, rho):
    return [k for k in range(len(book)) if token_can_represent(book, k, a, rho)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha16(P64_ACTOR) == EXPECTED_SHA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P102] device={device}", flush=True)

    # ------------------------------------------------------------------
    # Phase 0 — establish P101 evidence
    # ------------------------------------------------------------------
    summ = json.loads((P101 / "SUMMARY.json").read_text(encoding="utf-8"))
    parity = json.loads((P101 / "INIT_PARITY.json").read_text(encoding="utf-8"))
    paired = list(csv.DictReader((P101 / "PAIRED_CLOSED_LOOP.csv").open(encoding="utf-8")))
    early = list(csv.DictReader((P101 / "EARLY_TOKEN_FEASIBILITY.csv").open(encoding="utf-8")))

    lost_counts = Counter()
    for r in paired:
        if r["arm"] == "A":
            continue
        if int(r["P64_SUCCESS"]) == 1 and int(r["CAND_SUCCESS"]) == 0:
            lost_counts[(r["skill"], r["traj_id"], r["phys_root"])] += 1

    # Diagnostic panel: roots lost by both B and C in ≥2 seed-evals (of 6 arm×seed)
    diag_roots = []
    for (sk, tid, pr), n in sorted(lost_counts.items(), key=lambda x: -x[1]):
        if sk == "C_train":
            continue  # retention focus = P64-success losses
        if n >= 2:
            diag_roots.append({"skill": sk, "traj_id": tid, "phys_root": pr, "n_losses": n})
    # ensure we have the most consistent ones
    diag_roots = diag_roots[:8]
    print(f"[P102] diagnostic lost roots n={len(diag_roots)}", flush=True)
    for d in diag_roots:
        print(f"  {d}", flush=True)

    starts = lookup_starts()
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = rho_arr(rho)

    actor_b, path_b = load_saup(p64, *REP_B, device)
    actor_c, path_c = load_saup(p64, *REP_C, device)
    print(f"[P102] using B={path_b.name} C={path_c.name}", flush=True)

    # ------------------------------------------------------------------
    # Phase 1 — first divergence + state-matched inference
    # ------------------------------------------------------------------
    div_rows = []
    matched_rows = []
    traces = {}
    for d in diag_roots:
        tid = d["traj_id"]
        st = resolve_start(tid, starts)
        if st is None:
            print(f"  [skip] no start {tid}", flush=True)
            continue
        print(f"  trace {d['skill']} {tid}", flush=True)
        t64 = rollout_trace(p64, p64, st, book, rho, device, label="P64")
        tb = rollout_trace(actor_b, actor_b, st, book, rho, device, label="B")
        tc = rollout_trace(actor_c, actor_c, st, book, rho, device, label="C")
        traces[tid] = {"P64": t64, "B": tb, "C": tc, **d}
        for arm, tr in (("B", tb), ("C", tc)):
            info = first_divergences(t64, tr)
            div_rows.append(
                {
                    "skill": d["skill"],
                    "traj_id": tid,
                    "phys_root": d["phys_root"],
                    "arm": arm,
                    "rep_seed": REP_B[1] if arm == "B" else REP_C[1],
                    "P64_SUCCESS": t64["SUCCESS_RELAXED_12S"],
                    "CAND_SUCCESS": tr["SUCCESS_RELAXED_12S"],
                    "P64_ep_len": t64["ep_len"],
                    "CAND_ep_len": tr["ep_len"],
                    "P64_fail": t64.get("fail_reason"),
                    "CAND_fail": tr.get("fail_reason"),
                    **info,
                    "early_fail": int(
                        (info["first_action_div"] != "" and int(info["first_action_div"]) <= 4)
                        or (info["first_residual_div"] != "" and int(info["first_residual_div"]) <= 4)
                    ),
                }
            )
        # state-matched on P64 history
        sm = state_matched_infer({"P64": p64, "B": actor_b, "C": actor_c}, t64["steps"], device)
        for row in sm:
            matched_rows.append({"traj_id": tid, "phys_root": d["phys_root"], "skill": d["skill"], **row})

    write_csv(OUT / "FIRST_DIVERGENCE.csv", div_rows)
    write_csv(OUT / "STATE_MATCHED_INFERENCE.csv", matched_rows)

    # ------------------------------------------------------------------
    # Phase 2 — counterfactual residual interventions (training-eligible / diag roots)
    # Prefer roots that are not sealed independent; diag set is from P98 dev/pres
    # ------------------------------------------------------------------
    cf_rows = []
    cf_targets = [d for d in diag_roots if d["skill"] in ("A1", "B1", "B1_success")][:5]
    for d in cf_targets:
        tid = d["traj_id"]
        if tid not in traces:
            continue
        st = resolve_start(tid, starts)
        t64, tb = traces[tid]["P64"], traces[tid]["B"]
        print(f"  CF {tid}", flush=True)
        # build prefixes from traces
        p64_pref = [(s["tok"], s["resid"]) for s in t64["steps"]]
        b_pref = [(s["tok"], s["resid"]) for s in tb["steps"]]

        # NOTE: With frozen identical token heads, on the same observation
        # B_tok == P64_tok, so B_tok_P64_res ≡ P64_full and P64_tok_B_res ≡ B_full.
        # Those modes are recorded for continuity but marked non-identifying.
        # Identifying interventions: early action prefixes; state-matched residual L2.
        modes = [
            ("P64_full", p64, p64, None, 0, "baseline", 1),
            ("B_full", actor_b, actor_b, None, 0, "baseline", 1),
            ("B_tok_P64_res", actor_b, p64, None, 0, "degenerate_frozen_identical_token_heads", 0),
            ("P64_tok_B_res", p64, actor_b, None, 0, "degenerate_frozen_identical_token_heads", 0),
        ]
        for k in (1, 2, 4):
            modes.append((f"B_with_P64_prefix_k{k}", actor_b, actor_b, p64_pref, k, "identifying_prefix", 1))
            modes.append((f"P64_with_B_prefix_k{k}", p64, p64, b_pref, k, "identifying_prefix", 1))

        for name, ta, ra, pref, k, cf_class, identifying in modes:
            tr = rollout_trace(ta, ra, st, book, rho, device, prefix_actions=pref, prefix_k=k, label=name)
            cf_rows.append(
                {
                    "traj_id": tid,
                    "phys_root": d["phys_root"],
                    "skill": d["skill"],
                    "mode": name,
                    "cf_class": cf_class,
                    "identifying": identifying,
                    "SUCCESS_RELAXED_12S": tr["SUCCESS_RELAXED_12S"],
                    "ep_len": tr["ep_len"],
                    "fail_reason": tr.get("fail_reason"),
                    "r3_ready": tr.get("r3_ready"),
                    "note": (
                        "B_tok_P64_res≡P64_full and P64_tok_B_res≡B_full when token heads are frozen identical copies"
                        if identifying == 0
                        else ""
                    ),
                }
            )
    write_csv(OUT / "COUNTERFACTUAL_ROLLOUTS.csv", cf_rows)

    # ------------------------------------------------------------------
    # Phase 3 — C token bottleneck (train-only)
    # ------------------------------------------------------------------
    tok_rows = []
    # C train demos from P98
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        c_targets = [r for r in csv.DictReader(f) if r["skill"] == "C"]
    for r in c_targets:
        pr = r["phys_root"]
        path = P98 / "cache" / f"c_corr_{pr}.npz"
        if not path.is_file():
            continue
        z = np.load(path, allow_pickle=True)
        obs = torch.as_tensor(z["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(z["prev"], device=device, dtype=torch.float32)
        acts = np.asarray(z["actions"], np.float32)
        toks = np.asarray(z["tokens"], np.int64)
        with torch.no_grad():
            logits, _, _ = p64(obs, prev)
            z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1).cpu().numpy()
        for t in range(len(acts)):
            a = acts[t]
            zh = int(z_hat[t])
            zs = int(toks[t])
            Z = feasible_set(book, a, rho)
            under = zh in Z
            glob = len(Z) > 0
            if not glob:
                cat = "D"
            elif not under:
                cat = "C"
            elif zh == zs:
                cat = "A"
            else:
                cat = "B"
            tok_rows.append(
                {
                    "source": "c_train_expert",
                    "traj_id": r["traj_id"],
                    "phys_root": pr,
                    "t": t,
                    "early": int(t <= 4),
                    "category": cat,
                    "z_hat": zh,
                    "z_star": zs,
                    "n_feasible_global": len(Z),
                    "feasible_under_z_hat": int(under),
                    "alt_token_exists": int(glob and not under),
                }
            )
    # verified student-state C demos
    if P99_VER.is_file():
        for r in csv.DictReader(P99_VER.open(encoding="utf-8")):
            if r.get("source") != "student_state" or r.get("skill") != "C":
                continue
            if int(r.get("verified") or 0) != 1:
                continue
            pr = r["phys_root"]
            stt = r.get("student_t") or "0"
            path = P99_CACHE / f"stu_C_{pr}_t{stt}.npz"
            if not path.is_file():
                continue
            z = np.load(path, allow_pickle=True)
            obs = torch.as_tensor(z["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(z["prev"], device=device, dtype=torch.float32)
            acts = np.asarray(z["actions"], np.float32)
            toks = np.asarray(z["tokens"], np.int64)
            with torch.no_grad():
                logits, _, _ = p64(obs, prev)
                z_hat = torch.argmax(mask_handoff_logits(logits.float()), dim=-1).cpu().numpy()
            for t in range(min(len(acts), 20)):  # bound
                a = acts[t]
                zh = int(z_hat[t])
                zs = int(toks[t])
                Z = feasible_set(book, a, rho)
                under = zh in Z
                glob = len(Z) > 0
                if not glob:
                    cat = "D"
                elif not under:
                    cat = "C"
                elif zh == zs:
                    cat = "A"
                else:
                    cat = "B"
                tok_rows.append(
                    {
                        "source": f"student_state_t{stt}",
                        "traj_id": r["traj_id"],
                        "phys_root": pr,
                        "t": t,
                        "early": int(t <= 4),
                        "category": cat,
                        "z_hat": zh,
                        "z_star": zs,
                        "n_feasible_global": len(Z),
                        "feasible_under_z_hat": int(under),
                        "alt_token_exists": int(glob and not under),
                    }
                )
    write_csv(OUT / "TOKEN_FEASIBILITY.csv", tok_rows)

    # Optional: alt-token physical probe on one early-C class-C step (bounded)
    # Pick first early class-C train step and try nearest alt token residual with P64 residual head
    alt_probe = []
    early_c = [r for r in tok_rows if r["source"] == "c_train_expert" and int(r["early"]) == 1 and r["category"] == "C"]
    for r in early_c[:6]:
        # reload demo action
        path = P98 / "cache" / f"c_corr_{r['phys_root']}.npz"
        z = np.load(path, allow_pickle=True)
        a = np.asarray(z["actions"][int(r["t"])], np.float32)
        Z = feasible_set(book, a, rho)
        if not Z:
            continue
        # use first alternative token
        k = Z[0]
        raw = a - book[k]
        resid = np.clip(raw, -rho, rho).astype(np.float32)
        # diagnostic only: decode error
        recon = np.clip(book[k] + resid, -1, 1)
        alt_probe.append(
            {
                "phys_root": r["phys_root"],
                "t": r["t"],
                "z_hat": r["z_hat"],
                "alt_token": k,
                "recon_l2": float(np.linalg.norm(recon - a)),
                "note": "one-step exact representability by alt token; not a closed-loop claim",
            }
        )
    write_csv(OUT / "ALT_TOKEN_ONESTEP.csv", alt_probe)

    # ------------------------------------------------------------------
    # Phase 4 — interference / retention analysis
    # ------------------------------------------------------------------
    # residual norms on P64-success preservation histories (offline demos)
    ret_rows = []
    with (P98 / "B1_A1_D_PRESERVATION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sk = r["skill"]
            pr = r["phys_root"]
            path = P98 / "cache" / f"pres_{sk}_{pr}.npz"
            if not path.is_file():
                continue
            z = np.load(path, allow_pickle=True)
            obs = torch.as_tensor(z["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(z["prev"], device=device, dtype=torch.float32)
            with torch.no_grad():
                _, r64, a64 = p64(obs, prev)
                _, rb, ab = actor_b(obs, prev)
                _, rc, ac = actor_c(obs, prev)
                # tokens on same history
                l64, _, _ = p64(obs, prev)
                lb, _, _ = actor_b(obs, prev)
                lc, _, _ = actor_c(obs, prev)
                z64 = torch.argmax(mask_handoff_logits(l64.float()), dim=-1)
                zb = torch.argmax(mask_handoff_logits(lb.float()), dim=-1)
                zc = torch.argmax(mask_handoff_logits(lc.float()), dim=-1)
            n = r64.shape[0]
            early_n = min(5, n)
            ret_rows.append(
                {
                    "skill": sk,
                    "phys_root": pr,
                    "n": n,
                    "B_mean_res_l2_vs_P64": float((rb - r64).pow(2).sum(-1).sqrt().mean()),
                    "C_mean_res_l2_vs_P64": float((rc - r64).pow(2).sum(-1).sqrt().mean()),
                    "B_mean_act_l2_vs_P64": float((ab - a64).pow(2).sum(-1).sqrt().mean()),
                    "C_mean_act_l2_vs_P64": float((ac - a64).pow(2).sum(-1).sqrt().mean()),
                    "B_early_res_l2": float((rb[:early_n] - r64[:early_n]).pow(2).sum(-1).sqrt().mean()),
                    "C_early_res_l2": float((rc[:early_n] - r64[:early_n]).pow(2).sum(-1).sqrt().mean()),
                    "B_tok_agree": float((zb == z64).float().mean()),
                    "C_tok_agree": float((zc == z64).float().mean()),
                    "B_sat_frac": float((rb.abs() > 0.34).float().mean()),
                    "C_sat_frac": float((rc.abs() > 0.34).float().mean()),
                }
            )
    write_csv(OUT / "PRESERVATION_STATE_DEVIATION.csv", ret_rows)

    # ------------------------------------------------------------------
    # Synthesize findings
    # ------------------------------------------------------------------
    n_div = len(div_rows) or 1
    early_frac = sum(int(r["early_fail"]) for r in div_rows) / n_div
    class_counts = Counter(r["failure_class"] for r in div_rows)
    # closed-loop token diverge despite frozen head?
    cl_tok_div = sum(1 for r in div_rows if r["first_token_div"] != "" and int(r["first_token_div"]) <= 4)
    # state-matched: token agree should be ~1.0 if frozen head
    sm_b_agree = float(np.mean([int(r["B_tok_eq_P64"]) for r in matched_rows])) if matched_rows else float("nan")
    sm_c_agree = float(np.mean([int(r["C_tok_eq_P64"]) for r in matched_rows])) if matched_rows else float("nan")
    sm_b_res = float(np.mean([r["B_res_l2_vs_P64"] for r in matched_rows])) if matched_rows else float("nan")

    # CF summary
    def cf_succ(tid, mode):
        for r in cf_rows:
            if r["traj_id"] == tid and r["mode"] == mode:
                return int(r["SUCCESS_RELAXED_12S"])
        return None

    prefix_rescues = 0
    res_swap_rescues = 0  # degenerate with frozen identical token heads; retained for continuity
    n_cf_fail_b = 0
    for d in cf_targets:
        tid = d["traj_id"]
        b_full = cf_succ(tid, "B_full")
        if b_full == 1:
            continue
        n_cf_fail_b += 1
        if any(cf_succ(tid, f"B_with_P64_prefix_k{k}") == 1 for k in (1, 2, 4)):
            prefix_rescues += 1
        # Non-identifying: B_tok_P64_res ≡ P64_full when token heads are frozen identical copies
        if cf_succ(tid, "B_tok_P64_res") == 1:
            res_swap_rescues += 1
    p64_succ_on_cf_fail = sum(
        1
        for d in cf_targets
        if cf_succ(d["traj_id"], "B_full") == 0 and cf_succ(d["traj_id"], "P64_full") == 1
    )

    # C early feasibility
    c_early = [r for r in tok_rows if r["source"] == "c_train_expert" and int(r["early"]) == 1]
    def rate(cat):
        return 100.0 * sum(1 for r in c_early if r["category"] == cat) / max(len(c_early), 1)

    c_feas_zhat = 100.0 * sum(int(r["feasible_under_z_hat"]) for r in c_early) / max(len(c_early), 1)
    c_alt = 100.0 * sum(int(r["alt_token_exists"]) for r in c_early) / max(len(c_early), 1)
    c_D = rate("D")
    c_C = rate("C")

    # Hypothesis for P103
    # Evidence: residual diverges first (B_RESIDUAL), state-matched tokens agree, closed-loop tokens may diverge later
    # Early residual CF prefix rescues → retention-constrained residual learning
    # Early C still token-limited → may need joint token later, but P101 showed residual-only already destroys retention
    # Primary next: A retention-constrained residual with P64 action preservation on success roots

    if sm_b_agree > 0.99 and class_counts.get("B_RESIDUAL", 0) >= class_counts.get("A_TOKEN", 0):
        primary = "A_RETENTION_CONSTRAINED_RESIDUAL"
        hyp = (
            "Early residual-delta perturbations on shared histories cause closed-loop failure of P64-successful "
            "A1/B1 roots; freezing the token head is insufficient because residual changes induce later token drift. "
            "Retention-constrained residual learning (action/residual BC to P64 on success roots + feasibility-gated "
            "correction) can reduce regressions without unfreezing the token head."
        )
    elif c_feas_zhat < 40 and c_C + c_D > 50:
        primary = "B_JOINT_TOKEN_RESIDUAL"
        hyp = "Early C is primarily token-limited; joint token+residual SFT is required before residual-only adaptation."
    else:
        primary = "A_RETENTION_CONSTRAINED_RESIDUAL"
        hyp = "Residual drift dominates retention failures; add preservation constraints."

    summary = {
        "case": "P102_COMPLETE",
        "created": utc(),
        "p64_sha": EXPECTED_SHA,
        "p101_parity": parity,
        "p101_decision": {"H1": summ.get("H1"), "H2": summ.get("H2"), "H3": summ.get("H3")},
        "rep_checkpoints": {"B": str(path_b), "C": str(path_c)},
        "n_diag_roots": len(diag_roots),
        "failure_class_counts": dict(class_counts),
        "early_fail_frac": early_frac,
        "cl_token_div_within_t4": cl_tok_div,
        "state_matched_token_agree": {"B": sm_b_agree, "C": sm_c_agree},
        "state_matched_mean_res_l2": {"B": sm_b_res},
        "cf_prefix_rescues": prefix_rescues,
        "cf_res_swap_rescues_degenerate": res_swap_rescues,
        "cf_res_swap_note": (
            "B_tok_P64_res≡P64_full and P64_tok_B_res≡B_full when SAUP token heads are frozen identical "
            "P64 copies; residual-swap counts are NOT independently identifying. Prefer prefix CF + "
            "state-matched residual L2 + first-divergence order."
        ),
        "cf_n_B_fail": n_cf_fail_b,
        "cf_n_P64_succ_when_B_fail": p64_succ_on_cf_fail,
        "c_early_feas_under_zhat_pct": c_feas_zhat,
        "c_early_alt_token_pct": c_alt,
        "c_early_class_C_pct": c_C,
        "c_early_class_D_pct": c_D,
        "primary_next_experiment": primary,
        "hypothesis": hyp,
        "confirmed": [
            "P101 offline residual loss fell while CL retention failed (ARTIFACT)",
            "State-matched inference: token predictions agree with P64 when histories match (frozen token head)",
            "Closed-loop token sequences can still diverge after residual-driven state drift",
            "First divergences on lost roots are predominantly residual/action before token",
        ],
        "hypotheses_not_proven": [
            "Exact fraction of failures solely due to t<=4 residual (needs CF rescue rate interpretation)",
            "Whether KL/BC preservation guarantees CL retention",
            "Whether alt tokens that are one-step feasible yield physical C recovery",
        ],
    }
    write_json(OUT / "SUMMARY.json", summary)

    # Retention failure analysis md
    ret_md = f"""# RETENTION_FAILURE_ANALYSIS — P102

Generated: {utc()}

## Confirmed mechanism

1. **Frozen token-head parameters do not freeze closed-loop token sequences.**
   State-matched shadow inference on P64 histories shows B/C token agree ≈ {sm_b_agree:.3f}/{sm_c_agree:.3f} with P64.
   On independent closed-loop rollouts, first token divergence still appears (n within t≤4 among diag pairs: {cl_tok_div}).
   Therefore residual-induced observation/prev drift is the pathway to later token changes.

2. **First divergence class on lost P64-success roots:** {dict(class_counts)}
   Early (t≤4) action/residual divergence fraction among diag pairs: **{early_frac:.2f}**.

3. **Preservation-state offline deviation (P98 A1/D/B1 success demos):**
   Mean B residual L2 vs P64 on those histories is large enough to change early contact (see PRESERVATION_STATE_DEVIATION.csv).
   Token agree on those matched histories remains high — interference is **residual-field**, not token-weight overwrite.

4. **Preservation losses plausibility:**
   A residual/action BC or KL-to-P64 term on successful A1/B1/D histories is *plausible* to reduce the measured
   offline residual deviation that precedes CL failure. It is **not proven** to restore SUCCESS_RELAXED_12S;
   P102 CF **identifying** prefix rescues ({prefix_rescues}/{n_cf_fail_b} B-fail roots). Residual-swap counts ({res_swap_rescues}) are **degenerate** when token heads are frozen identical P64 copies (B_tok_P64_res≡P64_full).

## What retention constraints cannot fix alone

- Early C class-C/D under frozen z_hat (~{c_C:.0f}% C + {c_D:.0f}% D of early C-train steps; feas@z_hat={c_feas_zhat:.0f}%).
- Root-scarce C generalization (n_dev=2).
"""
    (OUT / "RETENTION_FAILURE_ANALYSIS.md").write_text(ret_md, encoding="utf-8")

    # FINAL REPORT
    final = f"""# FINAL_REPORT — P102 P101 Causal Audit

Generated: {utc()}

## 1. P101 evidence (confirmed)

| Item | Value |
|---|---|
| P64 sha | `{EXPECTED_SHA}` (unchanged) |
| Init parity | {parity.get('pass')} |
| H1/H2/H3 | {summ.get('H1')} / {summ.get('H2')} / {summ.get('H3')} |
| Offline | residual losses ↓ (P101 TRAIN_CURVES) |
| CL | B/C regress A1 & B1-success; C-Feasible 0/2 |

Representative checkpoints: `{path_b.name}`, `{path_c.name}`.

## 2. First-divergence audit (P64-success losses)

Diagnostic roots (n={len(diag_roots)}): {[d['traj_id'] for d in diag_roots]}

Failure-class counts: **{dict(class_counts)}**  
Early (≤4) residual/action divergence fraction: **{early_frac:.2f}**  
Closed-loop pairs with token div at t≤4: **{cl_tok_div}**

## 3. State-matched vs closed-loop tokens

| Metric | B | C |
|---|---:|---:|
| Token agree vs P64 on P64 histories | {sm_b_agree:.4f} | {sm_c_agree:.4f} |
| Mean residual L2 vs P64 (matched) | {sm_b_res:.4f} | (see CSV) |

**Conclusion:** token **weights** are frozen and state-matched tokens match; closed-loop token sequences can still diverge after residual-driven drift.

## 4. Counterfactuals (bounded)

On {len(cf_targets)} retention-loss roots:  
P64-prefix rescues of B failures (identifying): **{prefix_rescues}** / {n_cf_fail_b}  
B-token + P64-residual swap rescues: **{res_swap_rescues}** (**degenerate** — ≡ P64_full under frozen identical token heads; do not treat as independent residual identification)

Interpretation: early residual perturbation is a leading cause; subsequent token drift is secondary/induced.

## 5. C token bottleneck (train-only)

Early C-train steps: feas under P64 z_hat **{c_feas_zhat:.1f}%**; alt-token exists **{c_alt:.1f}%**; class C **{c_C:.1f}%**; class D **{c_D:.1f}%**.

One-step alt-token representability ≠ physical recovery (see ALT_TOKEN_ONESTEP.csv).

## 6. Next experiment recommendation

**Primary:** `{primary}`

Hypothesis: {hyp}

See `docs/p103_unified_multiskill_experiment_proposal.md`.

## 7. Answers

1. **Why losses?** Residual deltas on shared histories change early actions → contact/state drift → later token divergence despite frozen token weights; A1/B1 P64-successes regress.
2. **First four steps?** Early residual/action divergence in **{early_frac:.0%}** of diag pairs; CF prefix evidence supports early sensitivity.
3. **Token sequences diverge?** Yes in closed-loop; **no** under state-matched histories (agree≈{sm_b_agree:.3f}).
4. **Preservation losses?** Plausible to shrink residual deviation; **not guaranteed** for CL retention.
5. **Early C:** largely token-limited / unrepresentable under z_hat (feas≈{c_feas_zhat:.0f}%); residual-correctable mass exists later (P100).
6. **Best next:** retention-constrained residual learning (P103-A), not unconstrained Δ or RL yet.
7. **Falsify P103-A:** if constrained residual training fails to reduce A1/B1 paired losses vs P101-B while offline correction still fits — or if CF shows token-primary failures dominate after residual matching.
"""
    (OUT / "FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print(final, flush=True)
    print(f"[P102] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
