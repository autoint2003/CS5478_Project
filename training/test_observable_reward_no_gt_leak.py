"""Observable reward must not consume GT object state."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_SUBSTR = (
    "e_x",
    "object_qpos",
    "object_qvel",
    "v_rel",
    "w_rel",
    "physical_pack",
    "obs_from_sim",
    "recovered",
)


def test_terms_signature_no_gt():
    from envs.observable_reward import ObservableTactileReward

    names = list(inspect.signature(ObservableTactileReward.terms).parameters)
    assert names == ["self", "tactile", "a_used"]


def test_source_no_gt_in_observable_module():
    src = (ROOT / "envs" / "observable_reward.py").read_text(encoding="utf-8")
    assert "physical_pack" not in src
    assert "obs_from_sim" not in src
    assert "data.qpos" not in src
    assert "data.qvel" not in src
    assert "e_x_GT" not in src
    assert 'o["e_x"]' not in src
    assert "v_rel" not in src
    assert "object_body" not in src


def test_poison_gt_does_not_change_terms():
    from envs.observable_reward import ObservableTactileReward

    tac = {
        "e_hat_x": 0.005,
        "estimate_valid": True,
        "bilateral_valid": True,
        "valid_L": True,
        "valid_R": True,
        "contact_present_L": True,
        "contact_present_R": True,
        "e_x_GT": 99.0,
        "v_rel": 99.0,
    }
    rw = ObservableTactileReward(dt_policy=0.02)
    rw.seed_from_reading(tac)
    a = np.zeros(4)
    for _ in range(5):
        out1 = rw.terms(tac, a)
    tac2 = dict(tac)
    tac2["e_x_GT"] = -99.0
    tac2["v_rel"] = 1e6
    rw2 = ObservableTactileReward(dt_policy=0.02)
    rw2.seed_from_reading(tac)
    for _ in range(5):
        out2 = rw2.terms(tac2, a)
    assert out1["r"] == out2["r"]
    assert out1["r_progress"] == out2["r_progress"]


def test_invalid_does_not_fill_zero():
    from envs.observable_reward import ObservableTactileReward

    rw = ObservableTactileReward(dt_policy=0.02)
    tac = {
        "e_hat_x": float("nan"),
        "estimate_valid": False,
        "bilateral_valid": False,
        "valid_L": False,
        "valid_R": False,
        "contact_present_L": True,
        "contact_present_R": False,
    }
    rw.seed_from_reading(tac)
    out = rw.terms(tac, np.zeros(4))
    assert out["r_progress"] == 0.0
    assert out["progress_valid"] is False


def test_progress_emitted_once_per_100ms():
    from envs.observable_reward import ObservableTactileReward, phi_hat

    rw = ObservableTactileReward(dt_policy=0.02, t_progress=0.10)
    assert rw.n_prog == 5
    tac = {
        "e_hat_x": 0.0075,
        "estimate_valid": True,
        "bilateral_valid": True,
        "valid_L": True,
        "valid_R": True,
        "contact_present_L": True,
        "contact_present_R": True,
    }
    rw.seed_from_reading(tac)
    n_emit = 0
    last = 0.0
    for i in range(5):
        tac["e_hat_x"] = 0.0075 - 0.001 * (i + 1)
        out = rw.terms(tac, np.zeros(4))
        if out["progress_valid"]:
            n_emit += 1
            last = out["r_progress"]
    assert n_emit == 1
    expect = phi_hat(0.0025) - phi_hat(0.0075)
    assert abs(last - expect) < 1e-9


def test_oscillation_net_near_zero():
    from envs.observable_reward import ObservableTactileReward

    rw = ObservableTactileReward(dt_policy=0.02, t_progress=0.10)
    seq = [0.007, 0.002, 0.007, 0.002, 0.007]
    tac = {
        "estimate_valid": True,
        "bilateral_valid": True,
        "valid_L": True,
        "valid_R": True,
        "contact_present_L": True,
        "contact_present_R": True,
        "e_hat_x": seq[0],
    }
    rw.seed_from_reading(tac)
    acc = 0.0
    # 4 intervals of 5 steps
    k = 0
    for e in seq[1:]:
        for _ in range(5):
            k += 1
            tac["e_hat_x"] = e
            acc += rw.terms(tac, np.zeros(4))["r_progress"]
    # net |e| 7mm -> 7mm should be ~0 progress
    assert abs(acc) < 1e-9
