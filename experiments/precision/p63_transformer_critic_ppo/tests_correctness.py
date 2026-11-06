"""Critical numerical / causal / isolation tests before expensive training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.policy_sample import (
    mask_handoff_logits,
    sample_step,
)
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
    load_critic,
    save_pair,
)
from experiments.precision.p63_transformer_critic_ppo.critic import (
    TransformerCritic,
    critic_parameters_disjoint,
)
from experiments.precision.p63_transformer_critic_ppo.gae import compute_gae
from experiments.precision.p63_transformer_critic_ppo.paths import OUT, P62_EPOCH3, SEED, WINDOW
from experiments.precision.p63_transformer_critic_ppo.ppo import mean_ratio_vs_old, ppo_update, make_optimizers
from experiments.precision.p63_transformer_critic_ppo.rollout import collect_ic_group, make_env
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value


def _report(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def test_actor_parity(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    actor2, _, _, _ = load_actor(P62_EPOCH3, device)
    ds = build_p54_dataset(seed=SEED)
    st = ds["schedule"][0]
    env = make_env(book, rho)
    env.reset(st)
    # identical deterministic actions
    a1 = sample_step(actor, env.obs_seq, env.prev_seq, device, deterministic=True)
    a2 = sample_step(actor2, env.obs_seq, env.prev_seq, device, deterministic=True)
    ok = a1[0] == a2[0] and np.allclose(a1[1], a2[1]) and abs(a1[2] - a2[2]) < 1e-5
    return _report("actor_checkpoint_parity", ok, f"tok={a1[0]} vs {a2[0]}")


def test_identical_sampling_before_ppo(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    ds = build_p54_dataset(seed=SEED)
    env = make_env(book, rho)
    env.reset(ds["schedule"][0])
    torch.manual_seed(0)
    t1, r1, lp1, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
    torch.manual_seed(0)
    t2, r2, lp2, meta = sample_step(actor, env.obs_seq, env.prev_seq, device)
    ok = t1 == t2 and np.allclose(r1, r2) and abs(lp1 - lp2) < 1e-5
    return _report("identical_action_sampling", ok, f"tok={t1} logp_delta={lp1-lp2}")


def test_causal_value(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    # Zero-init value head would make all V≈0; randomize so causal effects are visible.
    torch.nn.init.normal_(critic.value.weight, std=0.1)
    torch.nn.init.normal_(critic.value.bias, std=0.1)
    T = 12
    obs = torch.randn(T, 39, device=device)
    prev = torch.randn(T, 7, device=device)
    v0 = critic(obs, prev).detach().clone()
    ctx0 = critic.encode(obs, prev).detach().clone()
    obs2 = obs.clone()
    obs2[-1] = torch.randn(39, device=device)  # change only last (future for earlier t)
    v1 = critic(obs2, prev)
    ctx1 = critic.encode(obs2, prev)
    # Earlier timesteps must be unchanged under causal mask
    ok = bool(torch.allclose(v0[:-1], v1[:-1], atol=1e-5, rtol=1e-5))
    ok_ctx = bool(torch.allclose(ctx0[:-1], ctx1[:-1], atol=1e-5, rtol=1e-5))
    # Changing a past observation should change later context/values
    obs3 = obs.clone()
    obs3[0] = torch.randn(39, device=device)
    v2 = critic(obs3, prev)
    ctx2 = critic.encode(obs3, prev)
    changed_later = not bool(torch.allclose(ctx0[1:], ctx2[1:], atol=1e-5))
    changed_v = not bool(torch.allclose(v0[1:], v2[1:], atol=1e-5))
    return _report(
        "causal_history_value",
        ok and ok_ctx and changed_later and changed_v,
        f"prefix_match={ok}/{ok_ctx} past_affects_future={changed_later}/{changed_v}",
    )


def test_gae_unit() -> dict:
    rewards = np.array([1.0, 1.0, 1.0], np.float32)
    values = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    dones = np.array([0.0, 0.0, 1.0], np.float32)
    adv, ret = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
    # With V=0, gamma=1, lam=1: adv = remaining return sum
    ok = np.allclose(ret, [3.0, 2.0, 1.0]) and np.allclose(adv, [3.0, 2.0, 1.0])
    # Terminal: bootstrap ignored even if values[-1] nonzero
    values_b = np.array([0.0, 0.0, 0.0, 99.0], np.float32)
    adv2, ret2 = compute_gae(rewards, values_b, dones, gamma=1.0, lam=1.0)
    ok2 = np.allclose(ret2, [3.0, 2.0, 1.0])
    # Timeout-style nonterminal last step would bootstrap — artificial cut
    dones_cut = np.array([0.0, 0.0, 0.0], np.float32)
    adv3, ret3 = compute_gae(rewards, values_b, dones_cut, gamma=1.0, lam=1.0)
    ok3 = abs(float(ret3[-1]) - (1.0 + 99.0)) < 1e-5
    return _report("gae_terminal_vs_bootstrap", ok and ok2 and ok3, f"ret={ret.tolist()} boot_ret={ret3[-1]}")


def test_padding_mask(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    o1 = torch.randn(5, 39, device=device)
    p1 = torch.randn(5, 7, device=device)
    o2 = torch.randn(3, 39, device=device)
    p2 = torch.randn(3, 7, device=device)
    # Batched padded
    o = torch.zeros(2, 5, 39, device=device)
    p = torch.zeros(2, 5, 7, device=device)
    o[0] = o1
    p[0] = p1
    o[1, :3] = o2
    p[1, :3] = p2
    pad = torch.tensor([[False] * 5, [False, False, False, True, True]], device=device)
    v_batch = critic(o, p, pad_mask=pad)
    v_short = critic(o2, p2)
    ok = bool(torch.allclose(v_batch[1, :3], v_short, atol=1e-4, rtol=1e-4))
    return _report("padding_sequence_mask", ok, f"max_abs={(v_batch[1,:3]-v_short).abs().max().item():.2e}")


def test_no_traj_leakage(device) -> dict:
    """Changing another batch row must not change this row's values."""
    actor, _, _, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    o = torch.randn(2, 8, 39, device=device)
    p = torch.randn(2, 8, 7, device=device)
    v0 = critic(o, p).detach().clone()
    o2 = o.clone()
    o2[1] = torch.randn(8, 39, device=device)
    v1 = critic(o2, p)
    ok = bool(torch.allclose(v0[0], v1[0], atol=1e-5))
    return _report("no_inter_trajectory_leakage", ok)


def test_param_isolation(device) -> dict:
    actor, _, _, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    ok = critic_parameters_disjoint(actor, critic)
    # mutate critic; actor unchanged
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    for p in critic.parameters():
        p.data.add_(0.1)
    after_ok = all(torch.equal(before[k], v) for k, v in actor.state_dict().items())
    return _report("actor_critic_param_isolation", ok and after_ok)


def test_ratio_one_before_update(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    ds = build_p54_dataset(seed=SEED)
    env = make_env(book, rho)
    batch, _ = collect_ic_group(env, actor, critic, ds["schedule"][0], device, n_traj=2)
    ratio = mean_ratio_vs_old(actor, batch, device)
    ok = abs(ratio - 1.0) < 1e-4
    return _report("ppo_ratio_equals_1_before_updates", ok, f"ratio={ratio:.8f}")


def test_save_resume(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    # perturb critic slightly so round-trip is nontrivial
    with torch.no_grad():
        critic.value.bias.add_(0.123)
    tmp = OUT / "_tmp_ckpt"
    tmp.mkdir(parents=True, exist_ok=True)
    ap, cp = tmp / "actor.pt", tmp / "critic.pt"
    save_pair(ap, cp, actor, critic, book, rho, epoch=1, extra={"test": True})
    actor2, _, _, _ = load_actor(ap, device)
    critic2 = load_critic(cp, actor2, device)
    ok_a = all(
        torch.allclose(a, b) for a, b in zip(actor.parameters(), actor2.parameters())
    )
    ok_c = all(
        torch.allclose(a, b) for a, b in zip(critic.parameters(), critic2.parameters())
    )
    return _report("checkpoint_save_resume", ok_a and ok_c)


def test_ppo_step_runs(device) -> dict:
    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    ds = build_p54_dataset(seed=SEED)
    env = make_env(book, rho)
    batch, _ = collect_ic_group(env, actor, critic, ds["schedule"][0], device, n_traj=2)
    opt_a, opt_c = make_optimizers(actor, critic)
    metrics = ppo_update(actor, critic, batch, opt_a, opt_c)
    ok = "pg" in metrics and np.isfinite(metrics["pg"]) and np.isfinite(metrics["vf"])
    return _report("ppo_update_smoke", ok, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}))


def run_all(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    results = []
    for fn in (
        test_actor_parity,
        test_identical_sampling_before_ppo,
        test_causal_value,
        test_padding_mask,
        test_no_traj_leakage,
        test_param_isolation,
        test_ratio_one_before_update,
        test_save_resume,
        test_ppo_step_runs,
    ):
        print(f"=== {fn.__name__} ===", flush=True)
        results.append(fn(device_t))
        print(results[-1], flush=True)
    results.append(test_gae_unit())
    print(results[-1], flush=True)
    all_ok = all(r["ok"] for r in results)
    blob = {"all_ok": all_ok, "tests": results}
    (OUT / "PPO_CORRECTNESS_TESTS.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    lines = [
        "# PPO_CORRECTNESS_TESTS",
        "",
        f"**all_ok = {all_ok}**",
        "",
        "| Test | OK | Detail |",
        "|------|:--:|--------|",
    ]
    for r in results:
        lines.append(f"| `{r['name']}` | {'PASS' if r['ok'] else 'FAIL'} | {r['detail']} |")
    lines.append("")
    (OUT / "PPO_CORRECTNESS_TESTS.md").write_text("\n".join(lines), encoding="utf-8")
    return blob


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps(run_all(device=args.device), indent=2))
