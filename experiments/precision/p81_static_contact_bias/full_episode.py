"""Phase 5 — bounded full-episode diagnostic: baseline vs single FC tok_56 substitution."""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids, sample_stratified
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal
from experiments.precision.p81_static_contact_bias.paths import (
    CACHE,
    MAX_EP_STEPS,
    N_FULL_EP_PER_CLASS,
    N_FULL_EP_SEEDS,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    SEED,
    T_MAX,
    TOKEN_56,
)


def _rollout(env, actor, critic, start, device, *, seed: int, inject_56_at_fc: bool) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info = {}
    injected = False
    ever_opp = ever_loss = False
    had = False
    fc_done = False
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        # Single predefined substitution at first Actor-owned FC decision
        if (
            inject_56_at_fc
            and not injected
            and st.saw_contact
            and dbg.get("forced") is None
            and relative_geometry(env.env.sim)["has_pad_contact"]
        ):
            tok = TOKEN_56
            injected = True
            fc_done = True
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"]:
            had = True
        elif had and not g["has_pad_contact"]:
            ever_loss = True
        if g["bilateral"] and g["correct_sidewall"]:
            ever_opp = True
    term = _r3_terminal(env, info)
    return {
        "skill": start.skill,
        "traj_id": start.traj_id,
        "seed": seed,
        "inject": int(inject_56_at_fc),
        "injected_applied": int(injected),
        "SUCCESS_RELAXED": term["SUCCESS_RELAXED"],
        "r3_ready": term["r3_ready"],
        "handoff": term["handoff"],
        "opposing": int(ever_opp),
        "contact_loss": int(ever_loss),
        "saw_fc": int(fc_done or st.saw_contact),
    }


def run_full_episode(*, device: str = "cuda", force: bool = False, skip_if_weak_cf: bool = True, cf_report: dict | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / "full_ep_rows.pkl"

    # Gate: only run if matched CF shows directional benefit
    if skip_if_weak_cf and cf_report is not None:
        delta = (cf_report.get("paired_delta_succ_ic") or {}).get("mean")
        if delta is not None and delta <= 0:
            report = {
                "skipped": True,
                "reason": "Matched CF did not show positive mean ΔSUCCESS for tok_56; skip full-episode transfer per protocol.",
                "cf_delta": delta,
            }
            (OUT / "FULL_EPISODE_TRANSFER.md").write_text(
                "# Full-Episode Transfer (P81 Phase 5)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
                encoding="utf-8",
            )
            return report

    if cache_path.exists() and not force:
        rows = pickle.loads(cache_path.read_bytes())
        return _summarize_full(rows)

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    blocked = blocked_eval_ids()
    pool = build_p54_dataset(seed=5478)["schedule"]
    # Stratified sample — avoid final-test blocked
    starts = sample_stratified(pool, n_ics=N_FULL_EP_PER_CLASS * 4, seed=SEED + 9, blocked=blocked)
    # Ensure per-class counts
    by = defaultdict(list)
    for s in starts:
        by[s.skill].append(s)
    selected = []
    for sk in ("A1", "B1", "C", "D"):
        selected.extend(by.get(sk, [])[:N_FULL_EP_PER_CLASS])
    # Fallback fill C if needed
    if len([s for s in selected if s.skill == "C"]) < N_FULL_EP_PER_CLASS:
        more = [s for s in pool if s.skill == "C" and s.traj_id not in blocked and s not in selected]
        selected.extend(more[: N_FULL_EP_PER_CLASS - len([s for s in selected if s.skill == "C"])])

    rows = []
    for ii, start in enumerate(selected):
        for k in range(N_FULL_EP_SEEDS):
            seed = rollout_seed(split="train", traj_id=start.traj_id, rollout_i=82000 + 10 * ii + k)
            a = _rollout(env, actor, critic, start, device_t, seed=seed, inject_56_at_fc=False)
            b = _rollout(env, actor, critic, start, device_t, seed=seed, inject_56_at_fc=True)
            rows.append({"baseline": a, "inject56": b, "d_succ": b["SUCCESS_RELAXED"] - a["SUCCESS_RELAXED"], "d_r3": b["r3_ready"] - a["r3_ready"]})
        print(
            f"[full] {ii+1}/{len(selected)} {start.skill} {start.traj_id[-16:]} "
            f"d_succ={rows[-1]['d_succ']:+d}",
            flush=True,
        )

    cache_path.write_bytes(pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL))
    return _summarize_full(rows)


def _summarize_full(rows: list[dict]) -> dict:
    def rate(key, cond="baseline"):
        return float(np.mean([r[cond][key] for r in rows]))

    by_skill = defaultdict(list)
    for r in rows:
        by_skill[r["baseline"]["skill"]].append(r)

    per_class = {}
    for sk, rs in by_skill.items():
        per_class[sk] = {
            "n": len(rs),
            "baseline_succ": float(np.mean([r["baseline"]["SUCCESS_RELAXED"] for r in rs])),
            "inject56_succ": float(np.mean([r["inject56"]["SUCCESS_RELAXED"] for r in rs])),
            "mean_d_succ": float(np.mean([r["d_succ"] for r in rs])),
            "baseline_r3": float(np.mean([r["baseline"]["r3_ready"] for r in rs])),
            "inject56_r3": float(np.mean([r["inject56"]["r3_ready"] for r in rs])),
        }

    # MSP = mean over classes of class success (equal weight)
    skills = [sk for sk in ("A1", "B1", "C", "D") if sk in per_class]
    msp_base = float(np.mean([per_class[sk]["baseline_succ"] for sk in skills])) if skills else None
    msp_inj = float(np.mean([per_class[sk]["inject56_succ"] for sk in skills])) if skills else None

    # IC paired bootstrap on overall episode Δ (treat each row as IC×seed unit but cluster by traj)
    by_ic = defaultdict(list)
    for r in rows:
        by_ic[r["baseline"]["traj_id"]].append(r["d_succ"])
    d_ic = {tid: float(np.mean(vs)) for tid, vs in by_ic.items()}
    rng = np.random.default_rng(SEED + 11)
    arr = np.asarray(list(d_ic.values()), float)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(800)] if len(arr) >= 3 else []
    ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else None

    abd_reg = {}
    for sk in ("A1", "B1", "D"):
        if sk in per_class:
            abd_reg[sk] = per_class[sk]["mean_d_succ"]

    report = {
        "n_episodes_per_condition": len(rows),
        "n_ics": len(by_ic),
        "msp_baseline": msp_base,
        "msp_inject56": msp_inj,
        "msp_delta": None if msp_base is None else float(msp_inj - msp_base),
        "overall_succ_baseline": rate("SUCCESS_RELAXED", "baseline"),
        "overall_succ_inject56": rate("SUCCESS_RELAXED", "inject56"),
        "paired_delta_succ_ic": {"mean": float(arr.mean()) if len(arr) else None, "ci95": ci, "n_ics": len(arr)},
        "per_class": per_class,
        "abd_deltas": abd_reg,
        "abd_any_regression": any(v < -0.05 for v in abd_reg.values()) if abd_reg else None,
        "secondary": {
            "r3_baseline": rate("r3_ready", "baseline"),
            "r3_inject56": rate("r3_ready", "inject56"),
            "contact_loss_baseline": rate("contact_loss", "baseline"),
            "contact_loss_inject56": rate("contact_loss", "inject56"),
            "opposing_baseline": rate("opposing", "baseline"),
            "opposing_inject56": rate("opposing", "inject56"),
            "handoff_baseline": rate("handoff", "baseline"),
            "handoff_inject56": rate("handoff", "inject56"),
        },
        "p73_context": (
            "P73 found multi-step FC sequence overlays failed full-episode generalization. "
            "This test is a stricter single-step substitution only."
        ),
        "not_a_deployed_overlay": True,
    }
    (OUT / "FULL_EPISODE_TRANSFER.md").write_text(
        "# Full-Episode Transfer (P81 Phase 5)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"msp_delta": report["msp_delta"], "per_class": {k: v["mean_d_succ"] for k, v in per_class.items()}, "ci": ci}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_full_episode(skip_if_weak_cf=False)
