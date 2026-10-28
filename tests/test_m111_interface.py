"""Unit tests for M111 112-way interface (P20)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    assert_m111_checkpoint_payload,
    build_m111,
    decode_m111_action,
    load_m111_book,
    save_m111,
)
from training.m111_dataset import (  # noqa: E402
    assert_no_target_in_inputs,
    build_samples,
    collate_m111,
    load_p19_manifest,
)
from training.skill_aware_action_vocab import hybrid_decode  # noqa: E402


def test_constants():
    assert MOTION_K == 111
    assert HANDOFF == 111
    assert LOGIT_K == 112


def test_book_and_build():
    book, rho, cb = load_m111_book()
    assert book.shape == (111, 7)
    assert int(cb.handoff_id) == 111
    model = build_m111(book, torch.device("cpu"), rho=rho)
    assert model.logits.out_features == 112
    assert model.handoff_index == 111


def test_token_ids_in_range():
    book, rho, _ = load_m111_book()
    for tok in range(MOTION_K):
        a = decode_m111_action(book, tok, np.zeros(7, np.float32), rho=rho)
        assert a.shape == (7,)
        assert np.all(a >= -1.0 - 1e-6) and np.all(a <= 1.0 + 1e-6)
    with pytest.raises(RuntimeError, match="HANDOFF"):
        decode_m111_action(book, HANDOFF, np.zeros(7, np.float32), rho=rho)
    with pytest.raises(RuntimeError):
        decode_m111_action(book, 112, np.zeros(7, np.float32), rho=rho)
    with pytest.raises(RuntimeError):
        decode_m111_action(book, -1, np.zeros(7, np.float32), rho=rho)


def test_decode_matches_hybrid():
    book, rho, _ = load_m111_book()
    rng = np.random.default_rng(0)
    for _ in range(20):
        tok = int(rng.integers(0, MOTION_K))
        res = rng.normal(0, 0.2, size=7).astype(np.float32)
        a = decode_m111_action(book, tok, res, rho=rho)
        b = hybrid_decode(book[tok], res, rho)
        assert np.allclose(a, b, atol=1e-6)


def test_handoff_zeros_action():
    book, rho, _ = load_m111_book()
    model = build_m111(book, torch.device("cpu"), rho=rho)
    with torch.no_grad():
        model.logits.bias.zero_()
        model.logits.bias[HANDOFF] = 1e6
        logits, res, act = model(torch.zeros(1, 3, 39), torch.zeros(1, 3, 7))
    assert int(torch.argmax(logits[0, -1]).item()) == HANDOFF
    assert float(act[0, -1].abs().max()) == 0.0


def test_reject_k48_checkpoint(tmp_path):
    book, rho, _ = load_m111_book()
    # Fake a K48-shaped payload
    fake = {
        "vocab": "K48",
        "proto": np.zeros((47, 7), np.float32),
        "handoff": 47,
        "model": {"logits.weight": torch.zeros(48, 64)},
    }
    with pytest.raises(RuntimeError, match="incompatible|not M111|refusing"):
        assert_m111_checkpoint_payload(fake, "fake_k48.pt")


def test_p19_tokens_in_range():
    man = load_p19_manifest()
    samples = build_samples(man, split="train", scheme="natural")
    assert len(samples) > 0
    for s in samples[:20]:
        assert int(s["tokens"].min()) >= 0
        assert int(s["tokens"].max()) <= HANDOFF
        assert int(s["tokens"][-1]) == HANDOFF


def test_collate_no_target_leakage():
    man = load_p19_manifest()
    samples = build_samples(man, split="val", scheme="natural")[:4]
    obs, prev, tokens, residuals, mask, weight = collate_m111(samples, torch.device("cpu"))
    assert_no_target_in_inputs(obs, prev, tokens)
    assert obs.shape[-1] == 39
    assert prev.shape[-1] == 7
    assert tokens.shape == obs.shape[:2]
    # Pads masked
    for i, s in enumerate(samples):
        n = s["observations"].shape[0]
        assert bool(mask[i, :n].all())
        if n < mask.shape[1]:
            assert not bool(mask[i, n:].any())
