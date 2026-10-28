"""Direct-nominal HANDOFF at expert-terminal states (after replaying motion tokens)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p36_rule_handoff_d_sft.audit_handoff_contract import (  # noqa: E402
    apply_handoff_contract,
    evaluate_with,
    nominal_step_contract,
    snap_metrics,
)
from experiments.precision.p36_rule_handoff_d_sft.geometry import geometric_target_reached  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402

OUT = ROOT / "results/diagnostics/p36_rule_handoff_d_sft/handoff_contract"
P35 = ROOT / "results/diagnostics/p35_precision_sft"
P35_MAN = P35 / "PACKAGED_MANIFEST.csv"
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)


def main(limit: int = 30):
    OUT.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    rows = [r for r in csv.DictReader(P35_MAN.open(encoding="utf-8")) if r["class"] == "P1"]
    # Prefer nontrivial first, then init-sat
    rows_nt = []
    rows_sat = []
    for r in rows:
        blob = pickle.loads(Path(r["ic_path"]).read_bytes())
        restore_capture(env, blob["capture"])
        ok, _ = geometric_target_reached(env.sim)
        (rows_sat if ok else rows_nt).append(r)
    ordered = rows_nt + rows_sat
    if limit > 0:
        ordered = ordered[:limit]

    results = []
    for i, r in enumerate(ordered):
        z = np.load(P35 / r["npz_relpath"], allow_pickle=False)
        toks = np.asarray(z["tokens"], np.int64)
        resid = np.asarray(z["residuals"], np.float32)
        blob = pickle.loads(Path(r["ic_path"]).read_bytes())

        # Replay expert motion only
        restore_capture(env, blob["capture"])
        n_motion = len(toks) - (1 if int(toks[-1]) == HANDOFF else 0)
        for t in range(n_motion):
            action = decode_m111_action(book, int(toks[t]), resid[t], rho=rho)
            step_legal(env.sim, action)
        geo_ok, geo = geometric_target_reached(env.sim)
        pre = snap_metrics(env.sim, blob["ref"])

        # Baseline official eval from this terminal (re-restore+replay is expensive;
        # capture terminal once)
        from envs.simulation.capture import take_capture

        term_cap = take_capture(env)
        term_blob = {"capture": term_cap, "ref": blob["ref"]}

        restore_capture(env, term_cap)
        h_base = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)

        restore_capture(env, term_cap)
        h_fix = evaluate_with(
            env.sim,
            blob["ref"],
            apply_fn=apply_handoff_contract,
            step_fn=nominal_step_contract,
            stabilize_n=8,
        )

        row = {
            "traj_id": r["traj_id"],
            "split": r["split"],
            "n_motion": n_motion,
            "term_geo_ok": int(geo_ok),
            "term_dtheta": geo["dtheta_deg"],
            "term_dx": geo["dx_mm"],
            "term_dz": geo["dz_mm"],
            "term_hand_to_ref_deg": pre["hand_to_ref_deg"],
            "term_ctrl7": pre["ctrl7"],
            "term_bilateral": int(pre["bilateral"]),
            "base_12S": int(bool(h_base.get("SUCCESS_12S"))),
            "base_NEW": int(bool(h_base.get("SUCCESS_NEW"))),
            "base_reason": h_base.get("reason"),
            "fix_12S": int(bool(h_fix.get("SUCCESS_12S"))),
            "fix_NEW": int(bool(h_fix.get("SUCCESS_NEW"))),
            "fix_reason": h_fix.get("reason"),
            "fix_drop_t": h_fix.get("drop_t_rel"),
            "fix_first_loss_t": h_fix.get("first_contact_loss_t_rel"),
            "fix_min_touch": h_fix.get("min_touch_after"),
        }
        results.append(row)
        print(
            f"[{i+1}/{len(ordered)}] {r['traj_id'][:44]} geo={geo_ok} "
            f"base={row['base_NEW']}/{row['base_12S']} fix={row['fix_NEW']}/{row['fix_12S']} "
            f"rerr={pre['hand_to_ref_deg']:.1f} ctrl={pre['ctrl7']}",
            flush=True,
        )

    path = OUT / "EXPERT_TERMINAL_HANDOFF.csv"
    keys = list(results[0].keys()) if results else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in results:
            w.writerow(row)

    n = len(results)
    summary = {
        "n": n,
        "term_geo_ok": sum(r["term_geo_ok"] for r in results),
        "base_NEW": sum(r["base_NEW"] for r in results),
        "base_12S": sum(r["base_12S"] for r in results),
        "fix_NEW": sum(r["fix_NEW"] for r in results),
        "fix_12S": sum(r["fix_12S"] for r in results),
        "fix_better_new": sum(1 for r in results if r["fix_NEW"] > r["base_NEW"]),
        "fix_worse_new": sum(1 for r in results if r["fix_NEW"] < r["base_NEW"]),
        "fix_better_12": sum(1 for r in results if r["fix_12S"] > r["base_12S"]),
        "fix_worse_12": sum(1 for r in results if r["fix_12S"] < r["base_12S"]),
    }
    (OUT / "EXPERT_TERMINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(lim)
