"""Phase 6 — representation integration feasibility (design only, no training)."""

from __future__ import annotations

import json

from experiments.precision.p78_action_relevant_representation.features import ENG_NAMES
from experiments.precision.p78_action_relevant_representation.paths import ENG_HIST, OUT


def write_integration(action_relevant: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "implemented_now": False,
        "action_relevant_supported": action_relevant,
        "if_proceed_later": {
            "feature_computation": (
                f"Maintain rolling buffer of last {ENG_HIST} deployable 27D frames; "
                f"compute engineered vector ({len(ENG_NAMES)} dims) causally at each decision."
            ),
            "normalization": "Fit mean/std on train-IC rollouts only; freeze for eval/deploy.",
            "history_alignment": "Same WINDOW semantics as Actor Transformer; eng uses short ENG_HIST only.",
            "inference_latency": "O(ENG_HIST) on CPU — negligible vs physics/policy forward.",
            "missing_signal_handling": "If estimate_valid=0, zero e_hat-derived eng terms; keep contact/aperture terms.",
            "actor_input_compatibility": {
                "option_1_concat_input": (
                    "Change input dim 39+7 → 39+7+16; requires NEW Actor weights (not compatible with P64 ckpt)."
                ),
                "option_2_auxiliary_encoder": (
                    "Keep 39D Actor trunk frozen; small adapter on eng features — still needs training run."
                ),
                "option_3_no_change": "Use eng only for offline diagnostics / Critic targets — no Actor input change.",
            },
            "checkpoint_compatibility": "P64 Actor checkpoint NOT loadable if input dim changes without adapter.",
            "input_dim_change": True if action_relevant else False,
            "new_actor_training_required": True if action_relevant else False,
            "m111_unchanged": True,
            "no_new_sensors": True,
            "preferred_minimal_path": (
                "If Case A: small matched Actor experiment with concat eng (option 1) OR frozen-trunk adapter "
                "(option 2), IC-disjoint eval, no sensing. Do not auto-start."
            ),
        },
        "do_not": [
            "Silently alter production 39D Actor input",
            "Overwrite P64 checkpoint",
            "Add Ft/shear or privileged sidewall to Actor",
            "Expand M111",
        ],
    }
    (OUT / "REPRESENTATION_INTEGRATION.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (OUT / "REPRESENTATION_INTEGRATION.md").write_text(
        "# Representation Integration Feasibility (P78 Phase 6)\n\n"
        f"**Action relevance supported:** {action_relevant}\n\n"
        "Design-only — no policy training in P78.\n\n```json\n"
        + json.dumps(plan, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return plan
