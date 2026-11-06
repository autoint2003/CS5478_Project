"""Re-run core PPO correctness + C-grasp env wiring checks."""

from __future__ import annotations

import json
import shutil

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo import tests_correctness as core
from experiments.precision.p63_transformer_ppo_c_grasp.paths import OUT, P63_CORE_OUT
from experiments.precision.p63_transformer_ppo_c_grasp.tests_contact import run_contact_tests


def test_c_env_wiring(device) -> dict:
    from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
    from experiments.precision.p63_transformer_critic_ppo.checkpointing import (
        init_critic_from_actor,
        load_actor,
    )
    from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
    from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
    from experiments.precision.p63_transformer_ppo_c_grasp.paths import P62_EPOCH3, SEED

    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    critic = init_critic_from_actor(actor).to(device)
    ds = build_p54_dataset(seed=SEED)
    st = next(s for s in ds["schedule"] if s.skill == "C")
    env = make_env(book, rho, enable_c_grasp_reward=True)
    env.reset(st)
    tok, resid, logp, val = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
    r, done, info = env.step(tok, resid)
    ok = (
        np.isfinite(r)
        and "r_raw" in info
        and "r_c_grasp" in info
        and "SUCCESS_RELAXED_12S" in info
        and abs(float(info["r_raw"]) + float(info["r_c_grasp"]) - float(r)) < 1e-6
    )
    return {"name": "c_grasp_env_wiring", "ok": ok, "detail": f"r={r} done={done}"}


def run_all(device: str = "cpu") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Point core OUT writes... core writes to its own OUT; we copy afterward.
    core_report = core.run_all(device=device)
    contact = run_contact_tests(device=device)
    extra = [test_c_env_wiring(device)]
    rows = list(core_report.get("tests") or []) + list(contact.get("tests") or []) + extra
    all_ok = all(r["ok"] for r in rows) and bool(core_report.get("all_ok")) and bool(contact.get("all_ok"))
    report = {
        "all_ok": all_ok,
        "core_all_ok": core_report.get("all_ok"),
        "contact_all_ok": contact.get("all_ok"),
        "tests": rows,
        "reused_core": str(P63_CORE_OUT / "PPO_CORRECTNESS_TESTS.json"),
    }
    (OUT / "PPO_CORRECTNESS_TESTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Copy core md for convenience then append
    src_md = P63_CORE_OUT / "PPO_CORRECTNESS_TESTS.md"
    lines = [
        "# PPO_CORRECTNESS_TESTS (P63 C-grasp)",
        "",
        f"**all_ok={all_ok}** (core={core_report.get('all_ok')}, contact={contact.get('all_ok')})",
        "",
        "Core PPO tests reused from `p63_transformer_critic_ppo` (not reimplemented).",
        "",
        "| Test | OK | Detail |",
        "|------|:--:|--------|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['ok']} | {r.get('detail','')} |")
    if src_md.is_file():
        lines += ["", "---", "", "## Upstream core report", "", src_md.read_text(encoding="utf-8")]
    (OUT / "PPO_CORRECTNESS_TESTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(run_all(device=args.device), indent=2)[:4000])
