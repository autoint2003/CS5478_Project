"""Argmax rollout step through the training environment."""

from __future__ import annotations

import numpy as np
import torch

import models.airborne_v3 as v3

def policy_step(env):
    obs = np.asarray(env.observation(), np.float32)
    env._obs_seq.append(obs)
    env._prev_seq.append(env._prev.copy())
    with torch.no_grad():
        logits, _res, act = env.model(
            torch.tensor(np.stack(env._obs_seq[-160:]), device=env.device),
            torch.tensor(np.stack(env._prev_seq[-160:]), device=env.device),
        )
    token = int(torch.argmax(logits[-1]).item())
    clipped = np.clip(act[-1].float().cpu().numpy(), -1.0, 1.0).astype(np.float32)
    v3.step_v3(env.sim, clipped)
    env._prev = clipped
    return token, clipped
