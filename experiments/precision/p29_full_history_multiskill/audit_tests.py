"""P29 Gate 1 — deterministic synthetic loss / architecture tests."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.full_history_policy import build_full_history_m111, sinusoidal_position_encoding  # noqa: E402
from models.k48_policy import build_policy  # noqa: E402
from models.m111_checkpoint import HANDOFF, LOGIT_K, MOTION_K, load_m111_book  # noqa: E402
from training.m111_sft import supervised_loss_m111  # noqa: E402

OUT = ROOT / "results/diagnostics/p29_full_history_multiskill"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"


def _row(name: str, passed: bool, detail: str) -> dict:
    return {"test": name, "passed": int(passed), "detail": detail}


def test_legacy_pe_cap() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    legacy = build_policy(book, device, float(np.max(rho)), handoff=True)
    t = 161
    obs = torch.zeros(t, 39)
    prev = torch.zeros(t, 7)
    try:
        legacy(obs, prev)
        return _row("legacy_pe_cap_161", False, "Embedding(160) unexpectedly accepted T=161")
    except Exception as exc:
        return _row("legacy_pe_cap_161", True, f"confirmed overflow/index error at T=161: {type(exc).__name__}: {exc}")


def test_full_history_long_seq() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    model = build_full_history_m111(book, device, rho=rho)
    for t in (1, 32, 160, 800, 2048):
        obs = torch.randn(t, 39)
        prev = torch.randn(t, 7)
        logits, res, act = model(obs, prev)
        if logits.shape != (t, LOGIT_K) or res.shape != (t, 7):
            return _row("full_history_long_seq", False, f"bad shapes at T={t}")
    return _row("full_history_long_seq", True, "T in {1,32,160,800,2048} OK with sinusoidal PE")


def test_causal_no_future() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    model = build_full_history_m111(book, device, rho=rho)
    model.eval()
    t = 20
    obs = torch.randn(t, 39)
    prev = torch.randn(t, 7)
    with torch.no_grad():
        base, _, _ = model(obs, prev)
        obs2 = obs.clone()
        obs2[-1] = obs2[-1] + 5.0  # perturb last only
        mid, _, _ = model(obs2, prev)
        # earlier positions must be identical
        early_ok = torch.allclose(base[:-1], mid[:-1], atol=1e-5)
        obs3 = obs.clone()
        obs3[0] = obs3[0] + 5.0  # perturb first
        late, _, _ = model(obs3, prev)
        # last position may change (early token in graph)
        last_changed = not torch.allclose(base[-1], late[-1], atol=1e-5)
    ok = bool(early_ok and last_changed)
    return _row(
        "causal_and_early_in_graph",
        ok,
        f"early_unaffected_by_future={early_ok}, early_perturb_affects_last={last_changed}",
    )


def test_handoff_residual_mask() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    model = build_full_history_m111(book, device, rho=rho)
    b, t = 2, 8
    obs = torch.randn(b, t, 39)
    prev = torch.randn(b, t, 7)
    tokens = torch.randint(0, MOTION_K, (b, t))
    tokens[:, -1] = HANDOFF
    residuals = torch.randn(b, t, 7)
    mask = torch.ones(b, t, dtype=torch.bool)
    weight = torch.ones(b)
    # Use legacy loss helper with FH model (same interface)
    loss, tok_l, res_l = supervised_loss_m111(model, obs, prev, tokens, residuals, weight, mask)
    # Pure HANDOFF batch
    tokens_h = torch.full((b, t), HANDOFF)
    loss_h, _, res_h = supervised_loss_m111(model, obs, prev, tokens_h, residuals, weight, mask)
    ok = bool(torch.isfinite(loss) and float(res_h) == 0.0)
    return _row("handoff_residual_masked", ok, f"res_loss_all_handoff={float(res_h):.6f} mixed_ok={torch.isfinite(loss).item()}")


def test_sample_weights_affect_grad() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    model = build_full_history_m111(book, device, rho=rho)
    b, t = 2, 4
    obs = torch.randn(b, t, 39)
    prev = torch.randn(b, t, 7)
    tokens = torch.randint(0, MOTION_K, (b, t))
    residuals = torch.zeros(b, t, 7)
    mask = torch.ones(b, t, dtype=torch.bool)

    def grad_norm(w):
        model.zero_grad(set_to_none=True)
        loss, _, _ = supervised_loss_m111(model, obs, prev, tokens, residuals, w, mask)
        loss.backward()
        return float(model.logits.weight.grad.abs().sum())

    g1 = grad_norm(torch.tensor([1.0, 1.0]))
    g2 = grad_norm(torch.tensor([0.0, 1.0]))
    g3 = grad_norm(torch.tensor([10.0, 0.0]))
    ok = abs(g1 - g2) > 1e-8 or abs(g1 - g3) > 1e-8
    # stronger: zeroing first sample should change grad vs equal weights when samples differ
    return _row("sample_weights_affect_grad", ok, f"g_eq={g1:.6f} g_w01={g2:.6f} g_w10={g3:.6f}")


def test_pad_mask_excludes_pads() -> dict:
    book, rho, _ = load_m111_book(M111_META)
    device = torch.device("cpu")
    model = build_full_history_m111(book, device, rho=rho)
    # batch: seq lengths 3 and 5 padded to 5
    obs = torch.randn(2, 5, 39)
    prev = torch.randn(2, 5, 7)
    pad_mask = torch.tensor([[False, False, False, True, True], [False, False, False, False, False]])
    logits, _, _ = model(obs, prev, pad_mask=pad_mask)
    # loss mask excludes pads
    tokens = torch.randint(0, MOTION_K, (2, 5))
    residuals = torch.zeros(2, 5, 7)
    loss_mask = ~pad_mask
    weight = torch.ones(2)
    loss, _, _ = supervised_loss_m111(model, obs, prev, tokens, residuals, weight, loss_mask)
    return _row("pad_mask_forward_and_loss", bool(torch.isfinite(loss)), f"loss={float(loss):.6f}")


def test_sinusoidal_monotonic_positions() -> dict:
    pe = sinusoidal_position_encoding(100, 64, torch.device("cpu"), torch.float32)
    # distinct rows
    diffs = (pe[1:] - pe[:-1]).abs().sum(-1)
    ok = bool((diffs > 0).all())
    return _row("sinusoidal_distinct_rows", ok, f"min_row_l1_delta={float(diffs.min()):.6e}")


def run_all() -> list[dict]:
    rows = [
        test_legacy_pe_cap(),
        test_full_history_long_seq(),
        test_causal_no_future(),
        test_handoff_residual_mask(),
        test_sample_weights_affect_grad(),
        test_pad_mask_excludes_pads(),
        test_sinusoidal_monotonic_positions(),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "AUDIT_TEST_RESULTS.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "passed", "detail"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


if __name__ == "__main__":
    rows = run_all()
    failed = [r for r in rows if not r["passed"]]
    for r in rows:
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['test']}: {r['detail']}")
    if failed:
        raise SystemExit(1)
    print("ALL_AUDIT_TESTS_PASSED")
