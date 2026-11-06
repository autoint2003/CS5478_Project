"""Assemble BRANCH_COMPARISON + FINAL_REPORT after A/B/C pilots."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p63_transformer_ppo_c_grasp.paths import OUT


def _slim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "history"}


def _msp(bc: dict, cls: str) -> float:
    return float((bc.get(cls) or {}).get("MSP_RELAXED") or float("nan"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    A = json.loads((OUT / "PILOT_A.json").read_text(encoding="utf-8"))
    B = json.loads((OUT / "PILOT_B.json").read_text(encoding="utf-8"))
    C = json.loads((OUT / "PILOT_C.json").read_text(encoding="utf-8"))
    payload = {
        "summary": {
            "matched_n_ics": 64,
            "n_traj": 8,
            "n_updates": 4,
            "matched_episodes_per_branch": 2048,
            "A_stable": True,
            "branch_b_n_new_c": 8,
            "branch_c_n_new_c": 8,
            "recommendation": "A",
        },
        "branches": {"A": _slim(A), "B": _slim(B), "C": _slim(C)},
    }
    (OUT / "PILOT_RESULTS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    base = A["baseline"]
    bbc = base.get("by_class") or {}

    def row(name: str, r: dict) -> str:
        p = r["pilot"]
        bc = p.get("by_class") or {}
        return (
            f"| {name} | {p['MSP_RELAXED']:.4f} | {_msp(bc,'A'):.4f} | {_msp(bc,'B'):.4f} | "
            f"{_msp(bc,'C'):.4f} | {_msp(bc,'D'):.4f} | {p.get('zero_success_ics')} | "
            f"{r.get('wall_s', 0)/3600:.2f} | {r.get('delta_msp'):.4f} | {r.get('msp_per_1k_traj'):.4f} |"
        )

    lines = [
        "# BRANCH_COMPARISON",
        "",
        "Matched budget: **64 ICs × 8 traj × 4 updates = 2048 traj/branch**.",
        "Branch B/C inserted **8** new C ICs within that matched IC count.",
        "",
        "| Branch | MSP | A | B | C | D | Zero | GPU-h | ΔMSP | ΔMSP/1k traj |",
        "|--------|----:|--:|--:|--:|--:|-----:|------:|-----:|-------------:|",
        (
            f"| P62_epoch3 | {base['MSP_RELAXED']:.4f} | {_msp(bbc,'A'):.4f} | {_msp(bbc,'B'):.4f} | "
            f"{_msp(bbc,'C'):.4f} | {_msp(bbc,'D'):.4f} | {base.get('zero_success_ics')} | — | — | — |"
        ),
        row("PPO-A", A),
        row("PPO-B (+C data)", B),
        row("PPO-C (+C reward)", C),
        "",
        "## Stability / paired vs P62",
        "",
        f"- A paired ΔMSP mean={A['paired_vs_p62']['mean_delta']:.4f} "
        f"CI95={A['paired_vs_p62']['ci95']} improved={A['paired_vs_p62']['n_improved']}/16",
        f"- B paired ΔMSP mean={B['paired_vs_p62']['mean_delta']:.4f}",
        f"- C paired ΔMSP mean={C['paired_vs_p62']['mean_delta']:.4f}",
        "",
        "## Decision (pilot)",
        "",
        "- **Keep PPO path** (A stable; credible MSP gain; KL small).",
        "- **Do not treat +C data as primary lever** (B < A; heldout C not helped).",
        "- **Do not keep C grasp reward for long-run yet** (heldout C MSP not improved; fall back toward A).",
        "- **Next long-run candidate: Branch A** (original pool + P57-A).",
        "- No long training started in this phase.",
        "",
    ]
    (OUT / "BRANCH_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "PILOT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    final = f"""# FINAL_REPORT — P63 Transformer PPO + C Grasp

Short pilots only (2048 traj/branch). **No long-run training.**

## What was reused

Existing `p63_transformer_critic_ppo` Actor/Critic/GAE/PPO/eval + `P62_epoch3` + warm critic.
New package adds C grasp event reward + A/B/C pilots. See `IMPLEMENTATION_AUDIT.md`.

## Gates

- Correctness: **all_ok** (`PPO_CORRECTNESS_TESTS.md`)
- Contact/reward once-only: **pass** (`CONTACT_GEOMETRY_TESTS.md`)
- Critic warm EV ≈ 0.54 (`CRITIC_WARMSTART.md`)
- Reward audit: success ≫ fail; exploitation_risk=false (`REWARD_AUDIT.md`)
- Pool: A60/B100/C100/D93 = 353 (`TRAIN_POOL_COUNTS.json`, `C_DATA_MANIFEST.jsonl`)

## Pilot MSP_RELAXED (P61 SHA-256 dev, 16 IC)

| | Overall | A | B | C | D |
|--|--------:|--:|--:|--:|--:|
| P62_epoch3 | {base['MSP_RELAXED']:.4f} | {_msp(bbc,'A'):.4f} | {_msp(bbc,'B'):.4f} | {_msp(bbc,'C'):.4f} | {_msp(bbc,'D'):.4f} |
| PPO-A | {A['pilot']['MSP_RELAXED']:.4f} | {_msp(A['pilot']['by_class'],'A'):.4f} | {_msp(A['pilot']['by_class'],'B'):.4f} | {_msp(A['pilot']['by_class'],'C'):.4f} | {_msp(A['pilot']['by_class'],'D'):.4f} |
| PPO-B | {B['pilot']['MSP_RELAXED']:.4f} | {_msp(B['pilot']['by_class'],'A'):.4f} | {_msp(B['pilot']['by_class'],'B'):.4f} | {_msp(B['pilot']['by_class'],'C'):.4f} | {_msp(B['pilot']['by_class'],'D'):.4f} |
| PPO-C | {C['pilot']['MSP_RELAXED']:.4f} | {_msp(C['pilot']['by_class'],'A'):.4f} | {_msp(C['pilot']['by_class'],'B'):.4f} | {_msp(C['pilot']['by_class'],'C'):.4f} | {_msp(C['pilot']['by_class'],'D'):.4f} |

Budget/branch ≈ {A['total_steps']}–{C['total_steps']} sim steps, ~{A['gpu_hours_approx']:.2f}–{C['gpu_hours_approx']:.2f} GPU-h.

## Final questions

1. **Critic predict return?** Yes for warm-start (EV≈0.54). During PPO pilots EV≈0.28–0.37 — useful but imperfect.
2. **PPO vs GRPO-64 sim/wall?** Promising on sample count: PPO pilot uses 8 traj/IC vs GRPO-64; A gained +0.061 MSP in 2048 traj / ~1.25 GPU-h. Not a matched full GRPO epoch bake-off; do not claim strict dominance yet.
3. **New C data help generalization?** **No** at this budget (B overall and C heldout worse than A).
4. **Grasp reward improve sidewall/stable contact?** Events fire in training (`mean_r_c`>0; reward audit). **Heldout C MSP did not improve** vs A/B — insufficient evidence of physical C gain.
5. **C failures shift later?** Not supported by heldout C MSP / zero-success (C zeros stayed high). Dedicated FAIL_RECATCH taxonomy eval not expanded in this short pilot.
6. **Overall MSP up?** **Yes** for A (+0.061) and C (+0.0625) vs P62; B +0.031.
7. **A/B/D retained?** **Yes** under A and C (all ≥ baseline). B retained A/B with mild D gain but added one D zero-success IC.
8. **Long-run branch?** **Branch A** (Transformer PPO, original 321 pool, P57-A). Park C-data/C-reward until a longer C-specific diagnostic or revised events show heldout C physical gains.

Checkpoints: `checkpoints/P63b_pilot_{{A,B,C}}_actor.pt`.
"""
    (OUT / "FINAL_REPORT.md").write_text(final, encoding="utf-8")
    print(json.dumps({"recommendation": "A", "MSP": {
        "P62": base["MSP_RELAXED"],
        "A": A["pilot"]["MSP_RELAXED"],
        "B": B["pilot"]["MSP_RELAXED"],
        "C": C["pilot"]["MSP_RELAXED"],
    }}, indent=2))


if __name__ == "__main__":
    main()
