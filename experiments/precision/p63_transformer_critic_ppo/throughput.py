"""Rollout throughput and critic memory/latency benchmark vs GRPO-style cost notes."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
    init_critic_from_actor,
    load_actor,
)
from experiments.precision.p63_transformer_critic_ppo.paths import OUT, P62_EPOCH3, SEED, WINDOW
from experiments.precision.p63_transformer_critic_ppo.rollout import collect_ic_group, make_env
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value


def bench_critic_latency(critic, device, lengths=(1, 20, 80, 160), repeats: int = 50) -> list[dict]:
    rows = []
    for T in lengths:
        obs = torch.randn(T, 39, device=device)
        prev = torch.randn(T, 7, device=device)
        # warmup
        for _ in range(5):
            _ = critic(obs, prev)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = critic(obs, prev)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / repeats
        mem = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            _ = critic(obs, prev)
            torch.cuda.synchronize()
            mem = int(torch.cuda.max_memory_allocated())
        rows.append({"T": T, "latency_ms": dt * 1000.0, "peak_alloc_bytes": mem})
    return rows


def bench_batched_prefixes(critic, device, n: int = 32, T: int = 80) -> dict:
    obs = torch.randn(n, T, 39, device=device)
    prev = torch.randn(n, T, 7, device=device)
    for _ in range(3):
        _ = critic(obs, prev)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = critic(obs, prev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    mem = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    return {"batch": n, "T": T, "latency_ms": dt * 1000.0, "peak_alloc_bytes": mem}


def run_benchmark(*, device: str = "cuda", n_ics: int = 4, n_traj: int = 8) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P62_EPOCH3, device_t)
    critic = init_critic_from_actor(actor).to(device_t)
    actor.eval()
    critic.eval()

    lat = bench_critic_latency(critic, device_t)
    batched = bench_batched_prefixes(critic, device_t)

    ds = build_p54_dataset(seed=SEED)
    env = make_env(book, rho)
    t0 = time.perf_counter()
    steps = 0
    for st in ds["schedule"][:n_ics]:
        _b, s = collect_ic_group(env, actor, critic, st, device_t, n_traj=n_traj)
        steps += s["n_steps"]
    wall = time.perf_counter() - t0
    # GRPO-64 reference: 64 traj/IC vs PPO pilot 8 traj/IC → 8× fewer rollouts per IC
    blob = {
        "device": str(device_t),
        "critic_latency": lat,
        "critic_batched": batched,
        "rollout": {
            "n_ics": n_ics,
            "n_traj_per_ic": n_traj,
            "n_steps": steps,
            "wall_s": wall,
            "steps_per_s": steps / max(wall, 1e-9),
            "note": (
                "Independent GRPO-64 uses 64 traj/IC; this PPO pilot uses "
                f"{n_traj}/IC → ~{64/n_traj:.0f}× fewer sim episodes per IC collection."
            ),
        },
        "window": WINDOW,
    }
    (OUT / "THROUGHPUT_BENCHMARK.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    lines = [
        "# THROUGHPUT_BENCHMARK",
        "",
        f"Device: `{device_t}`",
        "",
        "## Critic full-history latency",
        "",
        "| T | ms | peak bytes |",
        "|--:|---:|-----------:|",
    ]
    for r in lat:
        lines.append(f"| {r['T']} | {r['latency_ms']:.3f} | {r['peak_alloc_bytes']} |")
    lines += [
        "",
        f"Batched prefixes: {batched}",
        "",
        "## Rollout",
        "",
        f"- ICs×traj: {n_ics}×{n_traj}",
        f"- steps={steps}, wall={wall:.1f}s, {steps/max(wall,1e-9):.1f} steps/s",
        f"- {blob['rollout']['note']}",
        "",
    ]
    (OUT / "THROUGHPUT_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    return blob


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps(run_benchmark(device=args.device), indent=2))
