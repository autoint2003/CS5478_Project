"""Re-run short analysis/gates from cached panels; compare vs on-disk decisions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids, is_blocked_eval
from experiments.precision.p82_global_failure_budget.analyze import failure_budget
from experiments.precision.p83_recoverable_failure_mass.analyze import (
    exclusive_attribution,
    failure_mass,
    msp_scenarios,
    reconstruct_funnel,
    recoverability_evidence,
    statistical_sensitivity,
)
from experiments.precision.p83_recoverable_failure_mass.decide import decide
from experiments.precision.p83_recoverable_failure_mass.panel import _panel_report

ROOT = Path(__file__).resolve().parents[3] / "results" / "experiments"
OUT = ROOT / "final_consolidation"


def main() -> dict:
    diff: dict = {}

    # ---- P83 ----
    p83_rows = pickle.loads((ROOT / "p83_recoverable_failure_mass/cache/panel_32.pkl").read_bytes())
    old_dec = json.loads((ROOT / "p83_recoverable_failure_mass/P84_DECISION.json").read_text(encoding="utf-8"))
    panel_report = _panel_report(p83_rows, reused=True, target=32)
    n_forced = sum(1 for r in p83_rows if int(r.get("forced_steps") or 0) > 0)
    v2_rate = n_forced / len(p83_rows) if p83_rows else 0.0
    panel_report["v2_activation"] = {
        "episodes_with_forced_gt0": n_forced,
        "n_episodes": len(p83_rows),
        "activation_rate": v2_rate,
        "attribution": "Actor-only" if v2_rate < 0.05 else "P64+V2 mixed",
    }
    funnel = reconstruct_funnel(p83_rows)
    attr = exclusive_attribution(p83_rows)
    mass = failure_mass(p83_rows, attr)
    recov = recoverability_evidence(mass)
    scenarios = msp_scenarios(p83_rows, mass, recov)
    sens = statistical_sensitivity(p83_rows, mass, scenarios)
    new_dec = decide(panel_report, funnel, attr, mass, recov, scenarios, sens, stage=1)
    cmp = mass["comparisons"]
    diff["p83"] = {
        "v2_activation_rate": v2_rate,
        "v2_forced_episodes": n_forced,
        "n_episodes": len(p83_rows),
        "msp_equal_weight": panel_report.get("msp_equal_weight"),
        "old_case": old_dec.get("case"),
        "new_case": new_dec.get("case"),
        "old_p84": old_dec.get("p84"),
        "new_p84": new_dec.get("p84"),
        "case_changed": old_dec.get("case") != new_dec.get("case"),
        "p84_changed": old_dec.get("p84") != new_dec.get("p84"),
        "mass": {
            "C_cf": cmp["A_C_contact_formation_mass"],
            "nonC_sum_legacy": cmp["B_combined_A1B1D_any_failure_mass"],
            "nonC_mean": cmp["B_mean_A1B1D_any_failure_mass"],
            "C_any": cmp["A_C_any_failure_mass"],
            "per_nonC": cmp["C_individual_nonC_any_failure_mass"],
        },
        "need_stage2_old": old_dec.get("need_stage2_to_48"),
        "need_stage2_new": new_dec.get("need_stage2_to_48"),
        "new_reason_head": (new_dec.get("reason") or "")[:280],
    }
    (ROOT / "p83_recoverable_failure_mass/P84_DECISION_RERUN.json").write_text(
        json.dumps(new_dec, indent=2, default=str), encoding="utf-8"
    )

    # ---- P69 gate from stored deltas ----
    p69 = json.loads(
        (ROOT / "p69_skill_sequence/NO_TRAIN_INTERVENTION_RESULTS.json").read_text(encoding="utf-8")
    )
    delta = p69.get("delta") or {}
    succ_up = (delta.get("SUCCESS_RELAXED") or 0) > 0.05
    r3_up = (delta.get("r3_ready") or 0) > 0.05 or (delta.get("handoff") or 0) > 0.05
    geom_up = (delta.get("correct_sidewall") or 0) > 0.08 or (delta.get("min_pad_ge4") or 0) > 0.05
    e_improved = (delta.get("pre_contact_e_pos_mm") or 0) < -5.0
    e_only = e_improved and not r3_up and not geom_up
    old_r3 = (delta.get("r3_ready") or 0) > 0.01 or (delta.get("handoff") or 0) > 0.01
    old_geom = (delta.get("correct_sidewall") or 0) > 0.05 or (delta.get("min_pad_ge4") or 0) > 0.03
    old_e_only = e_improved and not old_r3 and not old_geom
    old_proceed = bool((old_r3 or old_geom) and not old_e_only)
    new_proceed = bool(succ_up and (r3_up or geom_up) and not e_only)
    diff["p69_gate"] = {
        "delta": delta,
        "old_proceed_to_ppo_file": p69.get("proceed_to_ppo"),
        "old_soft_gate_recomputed": old_proceed,
        "new_tight_gate": new_proceed,
        "gate_flags": {"succ_up": succ_up, "r3_up": r3_up, "geom_up": geom_up, "e_only": e_only},
        "changed": bool(p69.get("proceed_to_ppo")) != new_proceed,
    }

    # ---- P82 ----
    p82_rows = pickle.loads(
        (ROOT / "p82_global_failure_budget/cache/baseline_episodes.pkl").read_bytes()
    )
    old_budget = json.loads(
        (ROOT / "p82_global_failure_budget/CROSS_CLASS_FAILURE_BUDGET.json").read_text(encoding="utf-8")
    )
    new_budget = failure_budget(p82_rows)
    n_forced82 = sum(1 for r in p82_rows if int(r.get("forced_steps") or 0) > 0)
    diff["p82"] = {
        "n": len(p82_rows),
        "v2_activation_rate": n_forced82 / len(p82_rows) if p82_rows else 0.0,
        "v2_forced_episodes": n_forced82,
        "old_msp": old_budget.get("msp_equal_class_weight"),
        "new_msp": new_budget.get("msp_equal_class_weight"),
        "msp_changed": old_budget.get("msp_equal_class_weight") != new_budget.get("msp_equal_class_weight"),
    }
    meta_path = ROOT / "p82_global_failure_budget/cache/collection_meta.json"
    old_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    new_meta = dict(old_meta)
    new_meta["v2_activation"] = {
        "episodes_with_forced_gt0": n_forced82,
        "n_episodes": len(p82_rows),
        "activation_rate": n_forced82 / len(p82_rows) if p82_rows else 0.0,
        "note": "From cached episodes; near-zero => ≈ Actor-only.",
    }
    new_meta["blocked_eval_uses_root_id"] = True
    meta_path.write_text(json.dumps(new_meta, indent=2), encoding="utf-8")
    diff["p82"]["meta_updated"] = True

    # ---- blocked_eval impact ----
    blocked = blocked_eval_ids()
    pool = build_p54_dataset(seed=5478)["schedule"]
    traj_only = sum(1 for s in pool if s.traj_id in blocked)
    root_also = sum(1 for s in pool if is_blocked_eval(s.traj_id, blocked))
    diff["blocked_eval"] = {
        "blocked_set_size": len(blocked),
        "pool_size": len(pool),
        "excluded_traj_only": traj_only,
        "excluded_with_root": root_also,
        "extra_excluded_by_root": root_also - traj_only,
    }

    # ---- P55 obs coverage ----
    man_path = ROOT / "p55_expert_takeover/EXPERT_REPLAY_MANIFEST.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        n = n_obs = 0
        for s in man.get("samples") or []:
            if not s.get("m111_valid"):
                continue
            p = Path(s["sample_path"])
            if not p.is_file():
                continue
            n += 1
            z = np.load(p)
            if "obs" in z.files:
                n_obs += 1
        diff["p55_bc"] = {
            "m111_valid_samples_checked": n,
            "with_obs": n_obs,
            "needs_build_replay_rebuild": n_obs == 0,
        }
    else:
        diff["p55_bc"] = {"manifest_missing": True}

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "SHORT_RERUN_DIFF.json"
    out_path.write_text(json.dumps(diff, indent=2, default=str), encoding="utf-8")
    print(json.dumps(diff, indent=2, default=str))
    print("WROTE", out_path)
    return diff


if __name__ == "__main__":
    main()
