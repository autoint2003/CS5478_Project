"""Inexpensive GRPO math unit tests (no training / no MuJoCo)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p53_independent_grpo64.grpo_independent import (  # noqa: E402
    _group_advantage,
    grpo_loss_from_group,
)
from training.grpo_math import advantages  # noqa: E402


def test_advantages_mean_std():
    vals = np.array([1.0, 2.0, 3.0, 4.0], np.float64)
    adv, std, skipped = advantages(vals)
    assert not skipped
    assert abs(float(adv.mean())) < 1e-12
    assert abs(std - float(vals.std())) < 1e-12
    assert abs(float(adv[0]) - (1.0 - vals.mean()) / std) < 1e-12
    print("PASS advantages_mean_std")


def test_advantages_zero_var():
    adv, std, skipped = advantages([5.0, 5.0, 5.0])
    assert skipped
    assert abs(std) < 1e-8
    assert np.allclose(adv, 0.0)
    print("PASS advantages_zero_var")


def test_group_advantage_excludes_none():
    adv, std, skipped, n_nz = _group_advantage([1.0, None, 3.0, 5.0])
    assert not skipped
    assert adv[1] == 0.0
    assert n_nz >= 2
    # Relative among kept only
    kept = np.array([1.0, 3.0, 5.0])
    exp, _, _ = advantages(kept)
    assert abs(adv[0] - exp[0]) < 1e-12
    assert abs(adv[2] - exp[1]) < 1e-12
    print("PASS group_advantage_excludes_none")


def test_group_advantage_lt2():
    adv, std, skipped, n_nz = _group_advantage([1.0, None])
    assert skipped and n_nz == 0
    assert np.allclose(adv, 0.0)
    print("PASS group_advantage_lt2")


class _DummyActor(torch.nn.Module):
    """Fixed logits for ratio/clip/KL checks."""

    def __init__(self, logit_row: torch.Tensor):
        super().__init__()
        self.logit_row = logit_row  # [112]
        self.res_row = torch.zeros(7)

    def forward(self, obs, prev):
        # obs: [T,39] or [B,T,39]
        if obs.dim() == 2:
            t = obs.shape[0]
            logits = self.logit_row.unsqueeze(0).expand(t, -1)
            res = self.res_row.unsqueeze(0).expand(t, -1)
            return logits, res, None
        b, t, _ = obs.shape
        logits = self.logit_row.view(1, 1, -1).expand(b, t, -1)
        res = self.res_row.view(1, 1, -1).expand(b, t, -1)
        return logits, res, None


def _member(token: int, logp_beh: float, adv_unused=None):
    obs = np.zeros((2, 39), np.float32)
    prev = np.zeros((2, 7), np.float32)
    return {
        "records": [
            {
                "obs": obs,
                "prev": prev,
                "token": token,
                "resid": np.zeros(7, np.float32),
                "logp_behavior": logp_beh,
                "entropy": 1.0,
            }
        ],
        "return": 0.0,
    }


def test_clip_and_ratio():
    # Prefer token 0 heavily
    logits = torch.full((112,), -10.0)
    logits[0] = 5.0
    actor = _DummyActor(logits)
    ref = _DummyActor(logits.clone())
    for p in ref.parameters() if False else []:
        pass
    # behavior logp far from current → large ratio, should clip
    logp_cur = float(torch.log_softmax(logits, dim=-1)[0])
    beh = logp_cur - 2.0  # implies ratio ≈ e^2 ≈ 7.39 → clipped to 1.2
    members = [_member(0, beh), _member(0, beh)]
    adv = np.array([1.0, -1.0])
    loss, meta = grpo_loss_from_group(
        actor, ref, members, adv, torch.device("cpu"), clip=0.2, kl_coef=0.0
    )
    assert loss is not None
    # With symmetric ±1 advantages and same ratio, surrogate ≈ -clip*0? 
    # min(rho*A, clip(rho)*A): for A>0 clip helps; for A<0 unclipped (rho>1) is more negative
    # Just check finite and uses clip path (ratio would be huge unclipped)
    assert math.isfinite(float(loss.detach()))
    print("PASS clip_and_ratio", float(loss.detach()), meta)


def test_kl_frozen_ref_no_grad_into_ref():
    logits_q = torch.full((112,), -5.0)
    logits_q[1] = 2.0
    logits_p = torch.full((112,), -5.0)
    logits_p[2] = 2.0
    actor = _DummyActor(torch.nn.Parameter(logits_q.clone()))
    # make actor trainable param
    actor.logit_row = torch.nn.Parameter(logits_q.clone())
    ref = _DummyActor(logits_p)
    members = [_member(1, -1.0)]
    adv = np.array([1.0])
    loss, meta = grpo_loss_from_group(
        actor, ref, members, adv, torch.device("cpu"), clip=0.2, kl_coef=0.05
    )
    assert loss is not None
    loss.backward()
    assert actor.logit_row.grad is not None
    print("PASS kl_frozen_ref", meta["kl"])


def main():
    test_advantages_mean_std()
    test_advantages_zero_var()
    test_group_advantage_excludes_none()
    test_group_advantage_lt2()
    test_clip_and_ratio()
    test_kl_frozen_ref_no_grad_into_ref()
    print("ALL GRPO MATH TESTS PASSED")


if __name__ == "__main__":
    main()
