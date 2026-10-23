#!/usr/bin/env python
"""Re-quantize Stage-2 pilot demos with frozen K49 residual (±0.25) and re-replay.

Does not re-run the Precision Oracle. Updates tokens/, outcomes/, ledgers, REPORT.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402

OUT = ROOT / "results/data_generation/stage2_abc_pilot"
K49_CONFIG = "K49_tok48_wy0p25"


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def nearest_motion_token(table, action):
    a = np.asarray(action, np.float32).reshape(7)
    best_i, best_d = 0, float("inf")
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < best_d:
            best_i, best_d = i, d
    return int(best_i), float(best_d)


def quantize_with_residual(actions, table):
    acts = np.asarray(actions, np.float32)
    if acts.size == 0:
        return {
            "tokens": np.zeros((0,), np.int64),
            "residuals": np.zeros((0, 7), np.float32),
            "recon_actions": np.zeros((0, 7), np.float32),
            "proto_l2": np.zeros((0,), np.float32),
            "recon_l2": np.zeros((0,), np.float32),
            "max_proto_l2": 0.0,
            "max_recon_l2": 0.0,
            "max_recon_linf": 0.0,
            "residual_clipped_frac": 0.0,
        }
    tokens, residuals, recons, proto_l2, recon_l2 = [], [], [], [], []
    clipped = 0
    for a in acts:
        ti, d = nearest_motion_token(table, a)
        raw_res = a - table[ti]
        res = np.clip(raw_res, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw_res - res))) > 1e-8:
            clipped += 1
        recon = np.clip(table[ti] + res, -1.0, 1.0).astype(np.float32)
        tokens.append(ti)
        residuals.append(res)
        recons.append(recon)
        proto_l2.append(d)
        recon_l2.append(float(np.linalg.norm(recon - a)))
    toks = np.asarray(tokens, np.int64)
    res_a = np.asarray(residuals, np.float32)
    recon_a = np.asarray(recons, np.float32)
    pl2 = np.asarray(proto_l2, np.float32)
    rl2 = np.asarray(recon_l2, np.float32)
    rlinf = np.max(np.abs(recon_a - acts), axis=1)
    return {
        "tokens": toks,
        "residuals": res_a,
        "recon_actions": recon_a,
        "proto_l2": pl2,
        "recon_l2": rl2,
        "max_proto_l2": float(pl2.max()),
        "mean_proto_l2": float(pl2.mean()),
        "max_recon_l2": float(rl2.max()),
        "mean_recon_l2": float(rl2.mean()),
        "max_recon_linf": float(rlinf.max()),
        "residual_clipped_frac": float(clipped / len(acts)),
    }


def replay_recon(env, start_cap, recon_actions, ref):
    restore_capture(env, start_cap)
    for a in recon_actions:
        step_legal(env.sim, np.asarray(a, np.float32))
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "SUCCESS_AXIS_V1": bool(h.get("SUCCESS_AXIS_V1")),
        "reason": h.get("std_reason") or h.get("reason"),
    }


def main():
    env_cls = type("E", (), {})
    env = env_cls()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    table = build_k49_table(config_id=K49_CONFIG)

    outcomes_dir = OUT / "outcomes"
    updated = 0
    for path in sorted(outcomes_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        cont = row.get("continuous_demo")
        if not cont or not cont.get("path"):
            continue
        cont_npz = OUT / cont["path"]
        if not cont_npz.exists():
            continue
        blob = np.load(cont_npz)
        actions = blob["actions"]
        ic_pkl = OUT / "ics" / f"{row['sample_id']}.pkl"
        if not ic_pkl.exists():
            raise FileNotFoundError(f"missing IC pickle: {ic_pkl}")
        ic = pickle.loads(ic_pkl.read_bytes())

        q = quantize_with_residual(actions, table)
        tok_path = OUT / "tokens" / f"{row['sample_id']}__k49.npz"
        np.savez_compressed(
            tok_path,
            tokens=q["tokens"],
            residuals=q["residuals"],
            recon_actions=q["recon_actions"],
            proto_l2=q["proto_l2"],
            recon_l2=q["recon_l2"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(actions, np.float32),
            cap=np.asarray([CAP], np.float32),
        )
        rep = replay_recon(env, ic["capture"], q["recon_actions"], ic["ref"])
        verified = bool(rep["SUCCESS_NEW"])
        # Gap: continuous OK but token+residual still fails, or residual saturates heavily
        cont_ok = bool(cont.get("verified"))
        repr_gap = cont_ok and (
            (not verified) or q["residual_clipped_frac"] > 0.25 or q["max_recon_l2"] > 0.15
        )
        row["k49_demo"] = {
            "path": str(tok_path.relative_to(OUT)),
            "n_tokens": int(len(q["tokens"])),
            "max_l2": q["max_proto_l2"],
            "mean_l2": q["mean_proto_l2"],
            "max_recon_l2": q["max_recon_l2"],
            "mean_recon_l2": q["mean_recon_l2"],
            "max_recon_linf": q["max_recon_linf"],
            "residual_clipped_frac": q["residual_clipped_frac"],
            "encoding": "nearest_motion_token_plus_residual_cap_0p25",
            "replay_SUCCESS_NEW": rep["SUCCESS_NEW"],
            "replay_SUCCESS_12S": rep["SUCCESS_12S"],
            "replay_reason": rep["reason"],
            "verified": verified,
            "config_id": K49_CONFIG,
        }
        row["repr_gap"] = repr_gap
        path.write_text(json.dumps(jsafe(row), indent=2), encoding="utf-8")
        updated += 1
        print(
            f"{row['sample_id']}: k49={verified} recon_l2={q['max_recon_l2']:.3f} "
            f"clip_frac={q['residual_clipped_frac']:.2f} gap={repr_gap}",
            flush=True,
        )

    # Rebuild ledgers + summary stats
    outcomes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(outcomes_dir.glob("*.json"))]
    man_path = OUT / "manifest.jsonl"
    out_path = OUT / "oracle_outcomes.jsonl"
    if man_path.exists():
        man_path.unlink()
    if out_path.exists():
        out_path.unlink()
    with man_path.open("w", encoding="utf-8") as fm, out_path.open("w", encoding="utf-8") as fo:
        for row in outcomes:
            fo.write(json.dumps(jsafe(row), ensure_ascii=False) + "\n")
            fm.write(
                json.dumps(
                    jsafe(
                        {
                            "sample_id": row["sample_id"],
                            "plan": row["plan"],
                            "category": row["category"],
                            "outcome_json": f"outcomes/{row['sample_id']}.json",
                            "continuous_demo": (row.get("continuous_demo") or {}).get("path"),
                            "k49_demo": (row.get("k49_demo") or {}).get("path"),
                            "SUCCESS_NEW_continuous": (row.get("continuous_demo") or {}).get("verified"),
                            "SUCCESS_NEW_k49": (row.get("k49_demo") or {}).get("verified"),
                            "repr_gap": row.get("repr_gap"),
                        }
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )

    def stats(rows):
        cats = Counter(r["category"] for r in rows)
        lens = [r["precision"]["n_actions"] for r in rows]
        return {
            "n_eligible": len(rows),
            "direct_SUCCESS_NEW": sum(1 for r in rows if r["direct"]["SUCCESS_NEW"]),
            "precision_SUCCESS_NEW": sum(1 for r in rows if r["precision"]["SUCCESS_NEW"]),
            "direct_fail_precision_success": sum(1 for r in rows if r["category"] == "PRECISION_NEEDED"),
            "oracle_failure": sum(1 for r in rows if r["category"] == "ORACLE_FAILURE"),
            "regression": sum(1 for r in rows if r["category"] == "REGRESSION"),
            "categories": dict(cats),
            "continuous_verified": sum(1 for r in rows if (r.get("continuous_demo") or {}).get("verified")),
            "k49_verified": sum(1 for r in rows if (r.get("k49_demo") or {}).get("verified")),
            "repr_gaps": sum(1 for r in rows if r.get("repr_gap")),
            "action_length": {
                "median": float(np.median(lens)) if lens else 0,
                "mean": float(np.mean(lens)) if lens else 0,
                "p90": float(np.percentile(lens, 90)) if lens else 0,
                "max": int(max(lens) if lens else 0),
            },
        }

    overall = stats(outcomes)
    per = {p: stats([r for r in outcomes if r["plan"] == p]) for p in "ABC"}
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    summary["overall"] = overall
    summary["per_plan"] = per
    summary["k49_encoding"] = "nearest_motion_token_plus_residual_cap_0p25"
    summary["k49_fixup_note"] = "Initial pilot used proto-only replay; residuals restored without re-running Oracle."
    (OUT / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    # Patch REPORT section on K49
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    marker = "## 4. Dataset layout"
    insert = f"""## 3b. K49 residual fixup

Encoding: nearest motion token among {{0–46, 48}} + residual clipped to ±{CAP}, then reconstruct `proto+residual`.
HANDOFF token 47 is applied as controller switch after the motion prefix (not NN-targeted).

Updated overall K49 stats:
```
{json.dumps(jsafe(overall), indent=2)}
```

Per-plan K49 verified: A={per['A']['k49_verified']}/{per['A']['n_eligible']}, B={per['B']['k49_verified']}/{per['B']['n_eligible']}, C={per['C']['k49_verified']}/{per['C']['n_eligible']}.
Repr gaps remaining: {overall['repr_gaps']} (continuous OK but token+residual fails or residual saturates).

Oracle failures (6× Plan A): dtheta primitive fail or post-assess HANDOFF LOST_GRASP — retained under `failures/`; not labeled physically impossible.

"""
    if "## 3b. K49 residual fixup" not in report:
        if marker in report:
            report = report.replace(marker, insert + marker)
        else:
            report += "\n" + insert
        (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Updated {updated} demos. k49_verified={overall['k49_verified']} gaps={overall['repr_gaps']}", flush=True)


if __name__ == "__main__":
    main()
