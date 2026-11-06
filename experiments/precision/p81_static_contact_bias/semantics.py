"""Verify physical meaning of tok_56 via M111 decoder (not index heuristics alone)."""

from __future__ import annotations

import json

import numpy as np
import torch

from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p69_skill_sequence.skill_families import family_of_token
from experiments.precision.p71_contact_safe_alignment.align_tokens import decode_token
from experiments.precision.p81_static_contact_bias.paths import OUT, P64_ACTOR, TOKEN_56
from models.m111_checkpoint import MOTION_K, decode_m111_action


def write_token56_semantics(*, device: str = "cpu") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    assert 0 <= TOKEN_56 < MOTION_K
    dec = decode_token(book, TOKEN_56)
    zero_res = np.zeros(7, np.float32)
    phys = decode_m111_action(book, TOKEN_56, zero_res, rho=rho)

    # Neighbors by codebook L2 for context
    dists = []
    for i in range(MOTION_K):
        if i == TOKEN_56:
            continue
        d = float(np.linalg.norm(np.asarray(book[i], float) - np.asarray(book[TOKEN_56], float)))
        dists.append((d, i, family_of_token(book, i)))
    dists.sort()
    nearest = [{"token": t, "l2": d, "family": f} for d, t, f in dists[:8]]

    # Compare to other P79 candidates
    others = {}
    for name, tok in (("tok_50", 50), ("tok_82", 82), ("tok_40", 40)):
        others[name] = decode_token(book, tok)

    report = {
        "token": TOKEN_56,
        "motion_k": int(MOTION_K),
        "codebook_decode": dec,
        "physical_action_zero_residual": phys.tolist(),
        "components": {
            "linear_xyz": phys[:3].tolist(),
            "angular_rpy_proxy": phys[3:6].tolist(),
            "gripper": float(phys[6]),
            "lin_norm": float(np.linalg.norm(phys[:3])),
            "ang_norm": float(np.linalg.norm(phys[3:6])),
        },
        "family_heuristic": dec["family"],
        "nearest_codebook_neighbors": nearest,
        "other_candidates": others,
        "interpretation": {
            "primary_motion": (
                "linear-dominant" if dec["lin_norm"] >= dec["ang_norm"] and dec["lin_norm"] >= abs(dec["grip"])
                else "angular-dominant" if dec["ang_norm"] >= abs(dec["grip"])
                else "gripper-dominant"
            ),
            "grip_sign": "closing" if phys[6] > 0.05 else ("opening" if phys[6] < -0.05 else "near_zero_grip"),
            "note": (
                "Semantics come from M111 codebook vector + decode_m111_action, "
                "not from the integer index or informal prior labels."
            ),
        },
        "source": str(P64_ACTOR),
    }
    (OUT / "TOKEN56_SEMANTICS.md").write_text(
        "# Token 56 Semantics (P81)\n\n"
        "Decoded via `decode_m111_action` / codebook prototype.\n\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (OUT / "TOKEN56_SEMANTICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"family": dec["family"], "lin": dec["lin_norm"], "ang": dec["ang_norm"], "grip": dec["grip"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    write_token56_semantics()
