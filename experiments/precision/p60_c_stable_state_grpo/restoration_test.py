"""Phase 3 — Deterministic restore consistency: save → continue → restore → replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.continuation import (
    load_state_pkl,
    rollout_continuation,
)
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p60_c_stable_state_grpo.paths import (
    MAX_EP_STEPS,
    OUT,
    SEED,
    T_MAX,
)


def _qpos(env) -> np.ndarray:
    return np.asarray(env.env.sim.data.qpos, float).copy()


def _traj_sig(env, n_steps: int, actions: list[tuple[int, np.ndarray]]) -> dict:
    qpos_trace = []
    rewards = []
    infos = []
    for tok, resid in actions[:n_steps]:
        r, done, info = env.step(int(tok), resid)
        qpos_trace.append(_qpos(env))
        rewards.append(float(r))
        infos.append(
            {
                "drop": int(info.get("drop") or 0),
                "handoff": int(info.get("handoff") or 0),
                "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
                "t": int(info.get("t") or 0),
            }
        )
        if done:
            break
    return {
        "qpos_trace": qpos_trace,
        "rewards": rewards,
        "infos": infos,
        "final_qpos": qpos_trace[-1] if qpos_trace else _qpos(env),
        "n": len(rewards),
    }


def run_restoration_test(*, device: str = "cuda", n_states: int = 4, n_actions: int = 12) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "STATE_SELECTION_MANIFEST.json"
    if not man_path.is_file():
        raise FileNotFoundError(man_path)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    ckpt = Path(man["init_checkpoint"]["path"])
    states = man["states"][:n_states]
    ds = build_p54_dataset(seed=SEED)
    lookup = {s.traj_id: s for s in ds["schedule"] if s.skill == "C"}

    device_t = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )
    actor, book, rho, _ = load_policy_actor(ckpt, device_t)
    actor.eval()
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    results = []
    all_ok = True
    for row in states:
        st = lookup[row["ic_id"]]
        state = load_state_pkl(Path(row["path"]))
        # Build fixed action sequence from policy at restored state (deterministic seed)
        torch.manual_seed(SEED + 7000 + int(row["state_id"][1:]))
        restore_full_state(env, state, st)
        actions = []
        for _ in range(n_actions):
            tok, resid, _, _ = sample_step(
                actor, env.obs_seq, env.prev_seq, device_t, temperature=1.0
            )
            actions.append((int(tok), np.asarray(resid, np.float32).copy()))
            r, done, info = env.step(int(tok), resid)
            if done:
                break
        # Reference continuation from fresh restore with same fixed actions
        ref = rollout_continuation(
            env,
            actor,
            st,
            state,
            device_t,
            temperature=1.0,
            gamma_return=1.0,
            fixed_actions=actions,
        )
        # Replay after restore again
        rep = rollout_continuation(
            env,
            actor,
            st,
            state,
            device_t,
            temperature=1.0,
            gamma_return=1.0,
            fixed_actions=actions,
        )
        # Compare
        q_err = 0.0
        # Compare via stepping with observations equality on rewards/tokens/outcomes
        reward_match = np.allclose(ref["rewards"], rep["rewards"], atol=1e-5, rtol=1e-5)
        out_match = (
            ref["SUCCESS_RELAXED_12S"] == rep["SUCCESS_RELAXED_12S"]
            and ref["drop"] == rep["drop"]
            and ref["handoff"] == rep["handoff"]
            and ref["n_steps"] == rep["n_steps"]
        )
        # Physical: restore twice and compare qpos after identical action
        restore_full_state(env, state, st)
        q0 = _qpos(env)
        restore_full_state(env, state, st)
        q1 = _qpos(env)
        q_restore = float(np.max(np.abs(q0 - q1)))
        # After one action
        restore_full_state(env, state, st)
        if actions:
            env.step(actions[0][0], actions[0][1])
            qa = _qpos(env)
            restore_full_state(env, state, st)
            env.step(actions[0][0], actions[0][1])
            qb = _qpos(env)
            q_err = float(np.max(np.abs(qa - qb)))
        else:
            q_err = 0.0

        # Obs window restore
        restore_full_state(env, state, st)
        obs_a = np.stack(env.obs_seq).astype(np.float64)
        restore_full_state(env, state, st)
        obs_b = np.stack(env.obs_seq).astype(np.float64)
        obs_err = float(np.max(np.abs(obs_a - obs_b)))

        ok = reward_match and out_match and q_restore < 1e-9 and q_err < 1e-7 and obs_err < 1e-5
        # Remaining time / R3F / obs history must match capture (fail hard on mismatch)
        restore_full_state(env, state, st)
        rem_ok = (
            env.t == int(state["policy_step"])
            and env.max_steps == int(state["max_steps"])
            and env.t_max == int(state["t_max"])
            and env.mon is not None
            and int(env.mon.t_max) == int(state["r3f"]["t_max"])
            and len(env.obs_seq) == len(state["obs_seq"])
            and len(env.prev_seq) == len(state["prev_seq"])
        )
        if not rem_ok:
            ok = False

        row_out = {
            "state_id": row["state_id"],
            "ic_id": row["ic_id"],
            "ok": bool(ok and rem_ok),
            "reward_match": bool(reward_match),
            "outcome_match": bool(out_match),
            "qpos_restore_err": q_restore,
            "qpos_after_action_err": q_err,
            "obs_restore_err": obs_err,
            "remaining_timer_ok": bool(rem_ok),
            "n_actions_tested": len(actions),
            "ref_success": ref["SUCCESS_RELAXED_12S"],
            "ref_handoff": ref["handoff"],
            "ref_n_steps": ref["n_steps"],
        }
        results.append(row_out)
        all_ok = all_ok and row_out["ok"]
        print(f"[restore] {row['state_id']} ok={row_out['ok']} q_err={q_err:.2e}", flush=True)

    # Extra: capture mid-rollout, continue fixed, restore, replay (live capture path)
    live_ok = True
    if states:
        st0 = lookup[states[0]["ic_id"]]
        env.reset(st0)
        torch.manual_seed(SEED + 777)
        # Run a few steps then capture
        for _ in range(min(15, MAX_EP_STEPS // 4)):
            tok, resid, _, _ = sample_step(
                actor, env.obs_seq, env.prev_seq, device_t, temperature=1.0
            )
            _r, done, _info = env.step(tok, resid)
            if done:
                break
        if not done and env.t + 3 < env.max_steps:
            live = take_full_state(env)
            acts = []
            # record actions while continuing
            for _ in range(8):
                tok, resid, _, _ = sample_step(
                    actor, env.obs_seq, env.prev_seq, device_t, temperature=1.0
                )
                acts.append((int(tok), np.asarray(resid, np.float32).copy()))
                _r, done, _info = env.step(tok, resid)
                if done:
                    break
            a = rollout_continuation(
                env, actor, st0, live, device_t, temperature=1.0, gamma_return=1.0, fixed_actions=acts
            )
            b = rollout_continuation(
                env, actor, st0, live, device_t, temperature=1.0, gamma_return=1.0, fixed_actions=acts
            )
            live_ok = (
                np.allclose(a["rewards"], b["rewards"], atol=1e-5)
                and a["SUCCESS_RELAXED_12S"] == b["SUCCESS_RELAXED_12S"]
                and a["n_steps"] == b["n_steps"]
            )
            print(f"[restore] live_capture_replay ok={live_ok}", flush=True)

    report = {
        "all_ok": bool(all_ok and live_ok),
        "live_capture_ok": bool(live_ok),
        "n_states_tested": len(results),
        "results": results,
        "atol_notes": "qpos restore 1e-9; after-action 1e-7; obs 1e-5; rewards 1e-5",
    }
    (OUT / "STATE_RESTORATION_TEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# STATE_RESTORATION_TEST",
        "",
        f"**Pass: {report['all_ok']}** ({len(results)} selected states + live capture).",
        "",
        "Protocol: restore full state → fixed action sequence → restore → replay same actions;",
        "compare qpos, observation windows, rewards, and terminal outcomes.",
        "Remaining policy step / max_steps / t_max preserved (timer not reset).",
        "",
        "| state | ok | q_restore | q_action | obs | rem_ok |",
        "|-------|:--:|----------:|---------:|----:|:------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['state_id']} | {r['ok']} | {r['qpos_restore_err']:.2e} | "
            f"{r['qpos_after_action_err']:.2e} | {r['obs_restore_err']:.2e} | {r['remaining_timer_ok']} |"
        )
    lines += ["", f"Live capture replay: **{live_ok}**", ""]
    (OUT / "STATE_RESTORATION_TEST.md").write_text("\n".join(lines), encoding="utf-8")
    if not report["all_ok"]:
        raise RuntimeError("State restoration consistency failed; aborting before GRPO.")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps(run_restoration_test(device=args.device), indent=2))
