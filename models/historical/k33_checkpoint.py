"""K=33 lateral-chase checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.k48_policy import build_policy

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "results" / "diagnostics" / "raw" / "action_representation" / "ckpt_lateral_token.pt"


def load_policy(device):
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    proto = np.asarray(saved["proto"], np.float32)
    cap = float(saved["res_cap"])
    if proto.shape != (33, 7) or abs(cap - 0.25) > 1e-9:
        raise RuntimeError(f"checkpoint is K={proto.shape} cap={cap}, expected K=33 and 0.25")
    if abs(float(proto[32, 0]) - 1.0) > 1e-3 or abs(float(proto[32, 6]) + 1.0) > 1e-3:
        raise RuntimeError("prototype 32 is not the lateral-chase open command")
    model = build_policy(proto, device, cap)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, proto, cap
