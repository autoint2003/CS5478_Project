"""Phase 3 — P64 Critic diagnostics on C curriculum states (no Expert-return training)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae, explained_variance
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p66_c_curriculum_ppo.paths import (
    CRITIC_CALIB_CONTINUATIONS,
    MAX_EP_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def _lookup() -> dict:
    return {s.traj_id: s for s in build_p54_dataset(seed=SEED)["schedule"] if s.skill == "C"}


def run_critic_diagnostics() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((OUT / "CURRICULUM_SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    lookup = _lookup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    critic = load_critic(P64_STEP4_CRITIC, actor, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    vals_all, rets_all = [], []
    by_stage: dict[str, list] = {}
    succ_rets, fail_rets = [], []
    rows = []

    # Stratified sample: up to 8 states per stage for diagnostics wall-time
    by_stage_ents: dict[str, list] = {}
    for ent in man["states"]:
        by_stage_ents.setdefault(ent["stage"], []).append(ent)
    sampled = []
    for stage, ents in by_stage_ents.items():
        sampled.extend(ents[:8])
    print(f"[critic] sampling {len(sampled)}/{len(man['states'])} states", flush=True)

    for ent in sampled:
        start = lookup[ent["traj_id"]]
        state = pickle.loads(Path(ent["path"]).read_bytes())
        stage_vals, stage_rets, stage_succ = [], [], []
        for i in range(CRITIC_CALIB_CONTINUATIONS):
            seed_all(SEED + 12000 + i + hash(ent["id"]) % 5000, deterministic_torch=True)
            restore_full_state(env, state, start)
            ep_v, ep_r, ep_done = [], [], []
            done = False
            info = {}
            # initial value at restored state
            while not done:
                tok, resid, _, val = act_with_value(
                    actor, critic, env.obs_seq, env.prev_seq, device
                )
                ep_v.append(float(val))
                r, done, info = env.step(tok, resid)
                ep_r.append(float(r))
                ep_done.append(1.0 if done else 0.0)
            adv, ret = compute_gae(
                np.asarray(ep_r, np.float32),
                np.asarray(ep_v + [0.0], np.float32),
                np.asarray(ep_done, np.float32),
                1.0,
                0.95,
            )
            vals_all.extend(ep_v)
            rets_all.extend(ret.tolist())
            stage_vals.append(float(ep_v[0]) if ep_v else 0.0)
            mc = float(np.sum(ep_r))
            stage_rets.append(mc)
            succ = int(info.get("SUCCESS_RELAXED_12S") or 0)
            stage_succ.append(succ)
            if succ:
                succ_rets.append(mc)
            else:
                fail_rets.append(mc)
            _ = adv

        pred_mean = float(np.mean(stage_vals)) if stage_vals else 0.0
        ret_mean = float(np.mean(stage_rets)) if stage_rets else 0.0
        row = {
            "id": ent["id"],
            "stage": ent["stage"],
            "pred_v0_mean": pred_mean,
            "mc_return_mean": ret_mean,
            "bias_v0_minus_mc": pred_mean - ret_mean,
            "p_success": float(np.mean(stage_succ)) if stage_succ else 0.0,
        }
        rows.append(row)
        by_stage.setdefault(ent["stage"], []).append(row)
        print(
            f"[critic] {ent['stage']} V0={pred_mean:.3f} MC={ret_mean:.3f} "
            f"bias={pred_mean-ret_mean:.3f} p_succ={row['p_success']:.2f}",
            flush=True,
        )

    y_pred = np.asarray(vals_all, np.float32)
    y_true = np.asarray(rets_all, np.float32)
    ev = float(explained_variance(y_pred, y_true)) if len(y_pred) else float("nan")
    bias = float(np.mean(y_pred - y_true)) if len(y_pred) else float("nan")
    mc_err = float(np.mean(np.abs(y_pred - y_true))) if len(y_pred) else float("nan")

    stage_summary = {}
    for stage, rs in by_stage.items():
        stage_summary[stage] = {
            "n": len(rs),
            "pred_v0_mean": float(np.mean([r["pred_v0_mean"] for r in rs])),
            "mc_return_mean": float(np.mean([r["mc_return_mean"] for r in rs])),
            "bias_mean": float(np.mean([r["bias_v0_minus_mc"] for r in rs])),
            "p_success_mean": float(np.mean([r["p_success"] for r in rs])),
        }

    report = {
        "policy": str(P64_STEP4_ACTOR),
        "critic": str(P64_STEP4_CRITIC),
        "n_states_manifest": len(man["states"]),
        "n_states_sampled": len(sampled),
        "continuations_per_state": CRITIC_CALIB_CONTINUATIONS,
        "explained_variance": ev,
        "value_bias_mean": bias,
        "mc_abs_error_mean": mc_err,
        "n_value_samples": int(len(y_pred)),
        "by_stage": stage_summary,
        "succ_return_mean": float(np.mean(succ_rets)) if succ_rets else None,
        "fail_return_mean": float(np.mean(fail_rets)) if fail_rets else None,
        "rows": rows,
        "note": "Values/returns are on-policy P64 continuations from restored states; not Expert returns.",
    }
    (OUT / "CRITIC_C_DIAGNOSTICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = ["# Critic C Diagnostics (P66 Phase 3)\n"]
    md.append(f"- Explained variance: **{ev:.4f}**")
    md.append(f"- Value bias (V−MC): **{bias:.4f}**")
    md.append(f"- Mean |V−MC|: **{mc_err:.4f}**")
    md.append(f"- Samples: {len(y_pred)} across {len(man['states'])} states × {CRITIC_CALIB_CONTINUATIONS}\n")
    md.append("## Predicted value by recovery stage\n")
    for stage, s in stage_summary.items():
        md.append(
            f"- **{stage}**: V0={s['pred_v0_mean']:.3f}, MC={s['mc_return_mean']:.3f}, "
            f"bias={s['bias_mean']:.3f}, p_succ={s['p_success_mean']:.2f}"
        )
    md.append("\n## Successful vs failed continuation returns\n")
    md.append(f"- Success mean return: {report['succ_return_mean']}")
    md.append(f"- Fail mean return: {report['fail_return_mean']}")
    md.append(f"\n_{report['note']}_\n")
    (OUT / "CRITIC_C_DIAGNOSTICS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return report


if __name__ == "__main__":
    run_critic_diagnostics()
