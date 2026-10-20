#!/usr/bin/env python
"""Frozen K49 representability audit for continuous Plan C experts.

Does not modify the vocabulary. Quantization is nearest-neighbor on motion
tokens 0–46 and 48; HANDOFF (47) is never a continuous nearest-neighbor target.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from training.action_vocab_k49 import (  # noqa: E402
    DEFAULT_K49_CONFIG_ID,
    HANDOFF,
    K49_TABLE_ROWS,
    SECURE_REVERSE_WRIST,
    build_k49_table,
    describe_k49_config,
)
from training.recovery_execution import restore_ballistic  # noqa: E402

# Active frozen config used by Plan C / ground work
CONFIG_ID = "K49_tok48_wy0p25"
RECORDS = ROOT / "data" / "unified_experts" / "records"
OUT = ROOT / "results" / "consolidation"
PLAN_C = ["plan_c__gbi_010", "plan_c__gbi_022"]


def nearest_token(table: np.ndarray, action: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Nearest among motion tokens; exclude HANDOFF row 47."""
    a = np.asarray(action, np.float32).reshape(7)
    best_i, best_d = None, None
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return int(best_i), float(best_d), table[best_i].copy()


def dim_errors(table: np.ndarray, actions: np.ndarray, tokens: np.ndarray) -> dict:
    errs = np.abs(table[tokens] - actions)
    names = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
    return {
        names[i]: {
            "mean": float(errs[:, i].mean()),
            "max": float(errs[:, i].max()),
            "p95": float(np.percentile(errs[:, i], 95)),
        }
        for i in range(7)
    }


def high_risk_phases(actions: np.ndarray, tokens: np.ndarray, dists: np.ndarray) -> list[dict]:
    """Heuristic phase tags by grip / speed magnitude."""
    phases = []
    n = len(actions)
    # Simple: first 25% intercept/match, mid close/brake, last lift
    cuts = [0, n // 4, n // 2, (3 * n) // 4, n]
    labels = ["early_intercept_match", "close_capture", "brake", "slow_lift_tail"]
    for lab, a, b in zip(labels, cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        phases.append({
            "phase": lab,
            "t0": a,
            "t1": b,
            "mean_l2": float(dists[a:b].mean()),
            "max_l2": float(dists[a:b].max()),
            "mean_abs_grip_err": float(np.abs(
                build_k49_table(config_id=CONFIG_ID)[tokens[a:b], 6] - actions[a:b, 6]
            ).mean()),
        })
    return phases


def physical_token_replay(expert_pkl: Path, tokens: np.ndarray) -> dict:
    data = pickle.loads(expert_pkl.read_bytes())
    snap = data["catch_snap"]
    ref = data["ref"]
    table = build_k49_table(config_id=CONFIG_ID)
    import importlib.util

    path = ROOT / "scripts" / "run_ground_plan_c_final_success.py"
    spec = importlib.util.spec_from_file_location("run_ground_plan_c_final_success", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    sim, _ = make_ground_sim()
    restore_ballistic(sim, snap)
    restore_drop(sim, None)
    arm(sim)
    diverge_t = None
    for t, tok in enumerate(tokens):
        if int(tok) == HANDOFF:
            break
        a = table[int(tok)].copy()
        step_legal(sim, a)
        if latched(sim):
            diverge_t = t
            return {
                "SUCCESS_NEW": 0,
                "reason": "GROUND_DROP_DURING_TOKEN_REPLAY",
                "diverge_step": diverge_t,
            }
    meta = mod.evaluate_handoff_from_state(sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": int(meta["SUCCESS_NEW"]),
        "reason": meta["reason"],
        "pre_pos_mm": meta.get("pre_pos_mm"),
        "pre_theta": meta.get("pre_theta"),
        "diverge_step": diverge_t,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_k49_table(config_id=CONFIG_ID)
    assert table.shape == (49, 7), table.shape
    assert HANDOFF == 47
    assert SECURE_REVERSE_WRIST == 48
    cfg = describe_k49_config(CONFIG_ID)

    vocab_report = {
        "n_tokens": int(table.shape[0]),
        "ids": "0..48",
        "HANDOFF": HANDOFF,
        "token_48": SECURE_REVERSE_WRIST,
        "config_id": CONFIG_ID,
        "default_config_id": DEFAULT_K49_CONFIG_ID,
        "tok48_action": cfg["token_48_action"] if "token_48_action" in cfg else cfg.get("token_4_action"),
        "description": cfg["description"],
        "vocabulary_modified": False,
    }
    # describe keys
    vocab_report["tok48_wy"] = float(table[48, 4])
    vocab_report["tok4_wy"] = float(table[4, 4])

    quant_rows = []
    phys_rows = []
    details = {}

    for eid in PLAN_C:
        pkl = RECORDS / f"{eid}.pkl"
        if not pkl.exists():
            quant_rows.append({"expert_id": eid, "status": "MISSING"})
            continue
        data = pickle.loads(pkl.read_bytes())
        actions = np.asarray(data["actions_7d"], np.float32)
        tokens, dists, approx = [], [], []
        for a in actions:
            ti, d, ap = nearest_token(table, a)
            tokens.append(ti)
            dists.append(d)
            approx.append(ap)
        tokens = np.asarray(tokens, np.int32)
        dists = np.asarray(dists, float)
        approx = np.stack(approx, axis=0)
        dim_err = dim_errors(table, actions, tokens)
        phases = high_risk_phases(actions, tokens, dists)
        # Sustained high error: runs of L2 > median*3 for >= 5 steps
        thr = float(np.median(dists) * 3 + 1e-6)
        sustained = []
        run_s = None
        for i, d in enumerate(dists):
            if d > thr:
                if run_s is None:
                    run_s = i
            elif run_s is not None:
                if i - run_s >= 5:
                    sustained.append({"start": run_s, "end": i - 1, "max_l2": float(dists[run_s:i].max())})
                run_s = None
        if run_s is not None and len(dists) - run_s >= 5:
            sustained.append({"start": run_s, "end": len(dists) - 1, "max_l2": float(dists[run_s:].max())})

        quant_rows.append({
            "expert_id": eid,
            "root_id": data.get("physical_root_id") or eid.split("__")[-1],
            "n_steps": int(len(actions)),
            "mean_l2": float(dists.mean()),
            "max_l2": float(dists.max()),
            "p95_l2": float(np.percentile(dists, 95)),
            "n_sustained_high_err_runs": len(sustained),
            "status": "QUANTIZED",
        })
        print(f"Physical token replay {eid}...", flush=True)
        phys = physical_token_replay(pkl, tokens)
        phys["expert_id"] = eid
        phys_rows.append(phys)
        details[eid] = {
            "dim_errors": dim_err,
            "phases": phases,
            "sustained_high_error": sustained,
            "token_histogram": {int(k): int(v) for k, v in zip(*np.unique(tokens, return_counts=True))},
            "physical_replay": phys,
        }
        # Save token sequence alongside record
        np.savez_compressed(
            OUT / f"{eid}_k49_quantize.npz",
            tokens=tokens,
            dists=dists,
            actions=actions,
            approx=approx,
        )

    # Write reports
    with (OUT / "K49_QUANTIZATION.csv").open("w", newline="", encoding="utf-8") as f:
        if quant_rows:
            w = csv.DictWriter(f, fieldnames=list(quant_rows[0].keys()))
            w.writeheader()
            w.writerows(quant_rows)
    with (OUT / "K49_PHYSICAL_REPLAY.csv").open("w", newline="", encoding="utf-8") as f:
        if phys_rows:
            w = csv.DictWriter(f, fieldnames=list(phys_rows[0].keys()))
            w.writeheader()
            w.writerows(phys_rows)

    report = [
        "# K49 Representability Report",
        "",
        f"UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Frozen config: `{CONFIG_ID}` (IDs 0–48 unchanged; HANDOFF=47; tok48 slow reverse wrist).",
        "",
        "## Vocabulary check",
        "",
        "```json",
        json.dumps(vocab_report, indent=2),
        "```",
        "",
        "## Quantization (continuous → nearest legal token)",
        "",
        "| Expert | Steps | Mean L2 | Max L2 | P95 L2 | Sustained runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in quant_rows:
        if r.get("status") != "QUANTIZED":
            report.append(f"| {r['expert_id']} | — | — | — | — | MISSING |")
            continue
        report.append(
            f"| {r['expert_id']} | {r['n_steps']} | {r['mean_l2']:.4f} | {r['max_l2']:.4f} | "
            f"{r['p95_l2']:.4f} | {r['n_sustained_high_err_runs']} |"
        )
    report += [
        "",
        "## Physical token replay vs continuous SUCCESS_NEW",
        "",
        "| Expert | SUCCESS_NEW | Reason | Diverge step |",
        "|---|---:|---|---|",
    ]
    for r in phys_rows:
        report.append(
            f"| {r['expert_id']} | {r.get('SUCCESS_NEW')} | {r.get('reason')} | {r.get('diverge_step')} |"
        )
    report += [
        "",
        "## Interpretation",
        "",
        "Action-space nearest-neighbor error does **not** imply physical success.",
        "If tokenized replay fails while continuous replay passes, the frozen K49 vocabulary",
        "cannot represent that continuous Plan C expert without an offline token search",
        "(kept separate; vocabulary unchanged).",
        "",
        "HANDOFF is applied after the tokenized motion prefix, identical to continuous experts.",
        "",
    ]
    (OUT / "K49_REPRESENTABILITY.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "K49_REPRESENTABILITY.json").write_text(
        json.dumps({"vocab": vocab_report, "quantization": quant_rows, "physical": phys_rows, "details": details}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'K49_REPRESENTABILITY.md'}")


if __name__ == "__main__":
    main()
