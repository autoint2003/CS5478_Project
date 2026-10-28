"""P29 Gate 5–6: train + eval. Requires GATE_1_4_PASSED.json from prior audit run."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_d2_eval_eps  # noqa: E402
from experiments.precision.p29_full_history_multiskill.run_p29 import (  # noqa: E402
    M111_META,
    OUT,
    P23,
    P27,
    phase5_train,
    phase6_eval,
    phase8_report,
)
from models.m111_checkpoint import load_m111_book  # noqa: E402


def main() -> None:
    gate_path = OUT / "GATE_1_4_PASSED.json"
    if not gate_path.is_file():
        raise RuntimeError("Refuse training: GATE_1_4_PASSED.json missing. Run Gates 1–4 first.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("passed"):
        raise RuntimeError("Refuse training: Gates 1–4 did not pass.")
    audit_csv = OUT / "AUDIT_TEST_RESULTS.csv"
    if not audit_csv.is_file():
        raise RuntimeError("Refuse training: AUDIT_TEST_RESULTS.csv missing.")
    parity = OUT / "FULL_HISTORY_PARITY.csv"
    if not parity.is_file():
        raise RuntimeError("Refuse training: FULL_HISTORY_PARITY.csv missing.")

    print("=== P29 Gate 5–6: audit artifacts verified; starting training ===", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book, rho, _ = load_m111_book(M111_META)
    man = json.loads((P27 / "UNIFIED_MANIFEST.json").read_text(encoding="utf-8"))
    p21 = json.loads(
        (ROOT / "results/diagnostics/p21_full_stage1_experts/SFT_FULL_HORIZON_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    starts = build_start_index(p21)
    val_abc = [r for r in p21["trajectories"] if r["split"] == "val"]
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    d2 = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    d2_eps = load_d2_eval_eps({"d2": d2})
    baseline_rows = list(csv.DictReader((OUT / "BASELINE_REPRODUCTION.csv").open(encoding="utf-8")))
    for r in baseline_rows:
        for k in (
            "SUCCESS_NEW",
            "SUCCESS_12S",
            "SUCCESS_NEW_policy_handoff",
            "forced_timeout",
            "explicit_handoff",
            "n_motion",
            "max_token_run",
        ):
            if k in r and r[k] != "":
                try:
                    r[k] = int(float(r[k]))
                except ValueError:
                    pass

    distill = phase5_train(device, book, rho, man)
    eval_out = phase6_eval(device, book, rho, starts, val_abc, d2_eps, distill["ckpts"], baseline_rows)
    phase8_report(t0, eval_out["best"], eval_out["agg"])


if __name__ == "__main__":
    main()
