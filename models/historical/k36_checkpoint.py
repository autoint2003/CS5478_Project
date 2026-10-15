"""Load a K=36 continual checkpoint or a later build_policy checkpoint."""

from __future__ import annotations

import numpy as np
import torch

from models.historical.k36_policy import FuturePolicy
from models.k48_policy import build_policy


def _reject_head_mismatch(model, state):
    """Raise unless every checkpoint tensor matches a parameter of this head.

    A K=36 logit matrix is not copied into a wider head, and unknown keys are
    not dropped.
    """
    own = model.state_dict()
    problems = []
    for key, value in state.items():
        if key not in own:
            problems.append(f"unexpected {key}")
        elif tuple(own[key].shape) != tuple(value.shape):
            problems.append(f"{key} checkpoint {tuple(value.shape)} model {tuple(own[key].shape)}")
    missing = [key for key in own if key not in state]
    if missing:
        problems.append("missing " + ", ".join(missing[:6]))
    if problems:
        raise RuntimeError("incompatible token head: " + "; ".join(problems[:8]))


def load_models(init_path, ref_path, device):
    saved_ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_proto = np.asarray(saved_ref["proto"], np.float32)
    ref_model = build_policy(ref_proto, device, float(saved_ref["res_cap"]))
    ref_model.load_state_dict(saved_ref["model"])
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    saved_init = torch.load(init_path, map_location="cpu", weights_only=False)
    proto = np.asarray(saved_init["proto"], np.float32)
    if proto.ndim != 2 or proto.shape[1] != 7:
        raise RuntimeError(f"codebook must be (K, 7), got {proto.shape}")
    if abs(float(saved_ref["res_cap"]) - 0.25) > 1e-9 or abs(float(saved_init["res_cap"]) - 0.25) > 1e-9:
        raise RuntimeError("residual cap is not 0.25")
    state = saved_init["model"]
    legacy = "logit35_w" in state or "logits_old.weight" in state
    if legacy:
        if proto.shape != (36, 7):
            raise RuntimeError(
                f"FuturePolicy checkpoint is a 36-token head and cannot be remapped onto K={proto.shape[0]}"
            )
        model = FuturePolicy(proto).to(device)
        _reject_head_mismatch(model, state)
        model.load_state_dict(state)
        for parameter in model.fut.parameters():
            parameter.requires_grad_(False)
        return model, ref_model
    model = build_policy(proto, device, 0.25)
    _reject_head_mismatch(model, state)
    model.load_state_dict(state)
    return model, ref_model
