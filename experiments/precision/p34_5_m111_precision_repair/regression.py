"""Phase 4 — protect existing skill compatibility."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env, replay_actions
from experiments.precision.p34_5_m111_precision_repair.io_util import load_csv, utc, write_csv, write_md
from experiments.precision.p34_5_m111_precision_repair.paths import (
    HANDOFF_M111,
    M111_META,
    M111_REPLAY,
    MOTION_K,
    OUT,
    ROOT,
)
from experiments.precision.p34_5_m111_precision_repair.repairs import load_book, recon_with_book
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    snapshot_default_dynamics,
)
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, hybrid_decode, ResidualBudget


def run_regression(book_ext: np.ndarray | None, *, n_ok_spot: int = 12) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    book0, rho, _ = load_book()
    rows = []

    # 1) ID identity
    if book_ext is None:
        book_ext = book0
    n = min(MOTION_K, int(book_ext.shape[0]))
    max_diff = float(np.max(np.abs(book_ext[:n] - book0[:n])))
    rows.append(
        {
            "check": "proto_0_110_identical",
            "ok": int(max_diff < 1e-8),
            "detail": f"max_abs_diff={max_diff}",
        }
    )

    # 2) decode identity for every original token (zero residual)
    rho_a = ResidualBudget.from_spec(rho).as_array()
    decode_ok = True
    for t in range(MOTION_K):
        a0 = hybrid_decode(book0[t], np.zeros(7, np.float32), rho)
        a1 = hybrid_decode(book_ext[t], np.zeros(7, np.float32), rho)
        if float(np.max(np.abs(a0 - a1))) > 1e-8:
            decode_ok = False
            break
    rows.append(
        {
            "check": "decode_tokens_0_110_identical",
            "ok": int(decode_ok),
            "detail": "zero-residual hybrid_decode",
        }
    )

    # 3) HANDOFF id mapping documented
    handoff_new = int(book_ext.shape[0])
    rows.append(
        {
            "check": "handoff_id_mapping",
            "ok": 1,
            "detail": f"old_HANDOFF={HANDOFF_M111}; new_HANDOFF={handoff_new}; motion_0_110 preserved",
        }
    )

    # 4) Physical replay spot-check on previously M111-ok P34 roots with BOTH books
    ok_rows = [r for r in load_csv(M111_REPLAY) if str(r.get("ok")).lower() in ("1", "true")][
        :n_ok_spot
    ]
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    n_match = 0
    for r in ok_rows:
        p = Path(r["ic_path"])
        if not p.is_absolute():
            p = ROOT / p
        blob = pickle.loads(p.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        h0 = replay_actions(
            env, blob["capture"], blob["ref"], recon_with_book(book0, rho, acts)
        )
        h1 = replay_actions(
            env, blob["capture"], blob["ref"], recon_with_book(book_ext, rho, acts)
        )
        match = bool(h0.get("SUCCESS_NEW")) == bool(h1.get("SUCCESS_NEW"))
        n_match += int(match)
        rows.append(
            {
                "check": f"replay_parity::{r.get('tag')}",
                "ok": int(match and bool(h0.get("SUCCESS_NEW"))),
                "detail": f"V0_NEW={h0.get('SUCCESS_NEW')} Vext_NEW={h1.get('SUCCESS_NEW')}",
            }
        )
    restore_default_dynamics(env.sim, defaults)

    rows.append(
        {
            "check": "ok_cohort_parity_summary",
            "ok": int(n_match == len(ok_rows)),
            "detail": f"matched_outcome {n_match}/{len(ok_rows)}",
        }
    )

    write_csv(OUT / "REGRESSION_CHECKS.csv", rows)
    n_pass = sum(1 for r in rows if r["ok"])
    text = f"""# REGRESSION_AUDIT — P34.5

Generated: {utc()}

## Scope

- Original M111 prototypes 0..110 must remain byte-identical under any extension.
- HANDOFF stays a distinct token; numeric ID may move to `M_ext`.
- Spot-check previously successful P34 M111 roots for outcome parity.
- No checkpoint migration / retraining performed.

## Results

| Check group | Pass |
|---|---:|
| Proto identity 0..110 | {rows[0]['ok']} (diff={rows[0]['detail']}) |
| Decode identity | {rows[1]['ok']} |
| HANDOFF mapping documented | {rows[2]['ok']} ({rows[2]['detail']}) |
| Replay parity on ok spot | {n_match}/{len(ok_rows)} |

Overall atomic checks recorded: {n_pass}/{len(rows)} (see REGRESSION_CHECKS.csv).

## Future checkpoint migration (not done here)

If an extended vocabulary is adopted, M111 SFT heads must grow from 112 → (M_ext+1) logits.
Existing checkpoints are **not** compatible without a documented head-expansion migration.
"""
    write_md(OUT / "REGRESSION_AUDIT.md", text)
    print(text, flush=True)
    return {"rows": rows, "n_match": n_match, "n_ok_spot": len(ok_rows)}
