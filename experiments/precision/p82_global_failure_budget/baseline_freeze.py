"""Phase 1 — freeze and verify baseline configuration."""

from __future__ import annotations

import hashlib
import json

from experiments.precision.p71_contact_safe_alignment.controllers import make_specs
from experiments.precision.p73_first_contact_bilateral.controller import v2_spec
from experiments.precision.p82_global_failure_budget.paths import OUT, P64_ACTOR, P64_CRITIC


def _md5(path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "md5": hashlib.md5(data).hexdigest(), "nbytes": len(data), "exists": True}


def freeze_baseline() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    v2 = v2_spec()
    # Serialize Spec dataclass fields
    v2_cfg = {k: getattr(v2, k) for k in v2.__dataclass_fields__}  # type: ignore[attr-defined]
    # Make JSON-safe
    for k, v in list(v2_cfg.items()):
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
            try:
                v2_cfg[k] = list(v)
            except TypeError:
                v2_cfg[k] = str(v)

    report = {
        "baseline": {
            "actor": _md5(P64_ACTOR),
            "critic": _md5(P64_CRITIC),
            "overlay": "P70 V2 (unchanged)",
            "vocab": "original M111",
            "reward": "P57-A PoseRewardWeights, use_stabilization=False",
            "r3f": "original R3F / evaluate_predicates",
            "terminal_success": "SUCCESS_RELAXED_12S (env info) + original R3-ready definition",
            "physics": "unchanged MuJoCo",
        },
        "v2_controller": v2_cfg,
        "available_specs": list(make_specs([]).keys()),
        "constraints": {
            "no_actor_training": True,
            "no_critic_training": True,
            "no_overlays_beyond_v2": True,
            "no_final_test_for_selection": True,
            "fixed_token_interventions_stopped": True,
        },
        "data_policy": (
            "Prefer existing valid full-episode baseline rollouts; "
            "collect additional baseline-only episodes only as needed for A/B/C/D failure budget."
        ),
    }
    (OUT / "BASELINE_FREEZE.md").write_text(
        "# Baseline Freeze (P82 Phase 1)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "BASELINE_FREEZE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"actor_md5": report["baseline"]["actor"]["md5"], "critic_md5": report["baseline"]["critic"]["md5"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    freeze_baseline()
