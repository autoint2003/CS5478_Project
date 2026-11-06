"""Phase 4 — Feasibility probe: 16 policy continuations per selected state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.paths import GAMMA_RETURN
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.continuation import (
    load_state_pkl,
    rollout_continuation,
)
from experiments.precision.p60_c_stable_state_grpo.feasibility_gate import (
    evaluate_probe_viability,
    evaluate_state_feasibility,
)
from experiments.precision.p60_c_stable_state_grpo.paths import (
    MAX_EP_STEPS,
    OUT,
    PROBE_CONTINUATIONS,
    SEED,
    T_MAX,
    TEMPERATURE,
)


def _handoff_ready_proxy(info: dict) -> int:
    """Actual HANDOFF already implies ready; also count R3 early if present."""
    if int(info.get("handoff") or 0):
        return 1
    hd = info.get("handoff_diag") or {}
    if hd.get("dwell_ready"):
        return 1
    return 0


def run_feasibility_probe(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((OUT / "STATE_SELECTION_MANIFEST.json").read_text(encoding="utf-8"))
    ckpt = Path(man["init_checkpoint"]["path"])
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

    per_state = []
    n_zero = 0
    n_progress = 0
    for si, row in enumerate(man["states"]):
        st = lookup[row["ic_id"]]
        state = load_state_pkl(Path(row["path"]))
        rem = int(state["remaining_steps"])
        succ = hand = ready = lost = drop = trunc = 0
        returns = []
        epos = []
        for i in range(PROBE_CONTINUATIONS):
            torch.manual_seed(SEED + 8000 + si * 100 + i)
            m = rollout_continuation(
                env,
                actor,
                st,
                state,
                device_t,
                temperature=TEMPERATURE,
                gamma_return=GAMMA_RETURN,
            )
            info = m.get("info") or {}
            succ += int(m["SUCCESS_RELAXED_12S"])
            hand += int(m["handoff"])
            ready += _handoff_ready_proxy(info)
            lost += int(m.get("lost_contact") or 0)
            drop += int(m["drop"])
            trunc += int(m.get("truncated") or 0)
            returns.append(float(m["return"]))
            mm = (info.get("metrics") or {})
            epos.append(float(mm.get("e_pos_mm") or row.get("e_pos_mm") or 0.0))

        n = PROBE_CONTINUATIONS
        ret_std = float(np.std(returns))
        epos_delta = float(np.mean(epos)) - float(row.get("e_pos_mm") or np.mean(epos))
        gate = evaluate_state_feasibility(
            n_success=succ,
            n_handoff=hand,
            n_probe=n,
            return_std=ret_std,
            mean_epos_delta=epos_delta,
        )
        zero = succ == 0
        if zero:
            n_zero += 1
            if gate["diagnostics"]["dense_reward_variance_signal"]:
                n_progress += 1  # diagnostic counter only
        rec = {
            "state_id": row["state_id"],
            "ic_id": row["ic_id"],
            "failure_pattern": row.get("failure_pattern") or row.get("primary_capture_label"),
            "primary_capture_label": row.get("primary_capture_label"),
            "remaining_steps": rem,
            "p_success_relaxed": succ / n,
            "p_handoff": hand / n,
            "p_handoff_ready": ready / n,
            "p_contact_loss": lost / n,
            "p_drop": drop / n,
            "p_truncated": trunc / n,
            "return_mean": float(np.mean(returns)),
            "return_std": ret_std,
            "zero_success": int(zero),
            "n_success": succ,
            "n_handoff": hand,
            "retain_for_train": gate["retain_for_train"],
            "task_level_feasible": gate["task_level_feasible"],
            "diagnostics": gate["diagnostics"],
            "gate": gate["gate"],
        }
        per_state.append(rec)
        print(
            f"[probe] {row['state_id']} succ={succ}/{n} hand={hand}/{n} "
            f"lost={lost}/{n} ret_std={ret_std:.3f} keep={rec['retain_for_train']} "
            f"task_ok={rec['task_level_feasible']}",
            flush=True,
        )

    agg = evaluate_probe_viability(per_state)
    out = {
        "n_states": len(per_state),
        "n_continuations": PROBE_CONTINUATIONS,
        "mean_p_success": float(np.mean([r["p_success_relaxed"] for r in per_state])),
        "mean_p_handoff": float(np.mean([r["p_handoff"] for r in per_state])),
        "mean_p_handoff_ready": float(np.mean([r["p_handoff_ready"] for r in per_state])),
        "mean_p_contact_loss": float(np.mean([r["p_contact_loss"] for r in per_state])),
        "n_zero_success_states": n_zero,
        "n_zero_with_dense_var_diag": n_progress,
        "n_retain": agg["n_retain"],
        "viable_for_training": agg["viable_for_training"],
        "hard_stop_zero_handoff_and_success": agg["hard_stop_zero_handoff_and_success"],
        "stop_reason": agg["stop_reason"],
        "gate_policy": "HANDOFF_or_SUCCESS_required; return_std diagnostic only",
        "per_state": per_state,
    }
    (OUT / "FEASIBILITY_PROBE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = [
        "# FEASIBILITY_PROBE",
        "",
        f"Continuations per state: **{PROBE_CONTINUATIONS}**. States: **{len(per_state)}**.",
        f"Mean SUCCESS_RELAXED: **{out['mean_p_success']:.3f}**. "
        f"Mean HANDOFF: **{out['mean_p_handoff']:.3f}**. "
        f"Mean contact-loss: **{out['mean_p_contact_loss']:.3f}**.",
        f"Zero-success states: {n_zero} (dense-var diagnostic count={n_progress}). "
        f"Retain: {agg['n_retain']}.",
        f"**Viable for GRPO training: {agg['viable_for_training']}** "
        f"(hard_stop={agg['hard_stop_zero_handoff_and_success']})",
        "",
        "Gate: require ≥1 HANDOFF or ≥1 SUCCESS among probe rolls; return_std is diagnostic only.",
        "",
    ]
    if agg.get("stop_reason"):
        lines += [f"> STOP: {agg['stop_reason']}", ""]
    lines += [
        "| state | capture | p_succ | p_hand | p_ready | p_lost | rem | keep | task |",
        "|-------|---------|-------:|-------:|--------:|-------:|----:|:----:|:----:|",
    ]
    for r in per_state:
        lines.append(
            f"| {r['state_id']} | {r.get('failure_pattern')} | {r['p_success_relaxed']:.3f} | "
            f"{r['p_handoff']:.3f} | {r['p_handoff_ready']:.3f} | {r['p_contact_loss']:.3f} | "
            f"{r['remaining_steps']} | {r['retain_for_train']} | {r['task_level_feasible']} |"
        )
    lines.append("")
    (OUT / "FEASIBILITY_PROBE.md").write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps({k: v for k, v in run_feasibility_probe(device=args.device).items() if k != "per_state"}, indent=2))
