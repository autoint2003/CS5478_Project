#!/usr/bin/env python
"""Re-quantize mining continuous demos with residual K49; refresh SFT/RL manifests + REPORT."""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402

BASE = ROOT / "results/data_generation/stage2_abc_mining/v1"
K49_CONFIG = "K49_tok48_wy0p25"


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def nearest(table, a):
    a = np.asarray(a, np.float32).reshape(7)
    bi, bd = 0, 1e9
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < bd:
            bi, bd = i, d
    return bi, bd, table[bi].copy()


def quantize(acts, table):
    acts = np.asarray(acts, np.float32)
    if acts.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, 7), np.float32), 0.0, 0.0
    toks, res, rl2, clipped = [], [], [], 0
    for a in acts:
        ti, _, ap = nearest(table, a)
        raw = a - ap
        r = np.clip(raw, -CAP, CAP).astype(np.float32)
        if float(np.max(np.abs(raw - r))) > 1e-8:
            clipped += 1
        recon = np.clip(ap + r, -1, 1).astype(np.float32)
        toks.append(ti)
        res.append(r)
        rl2.append(float(np.linalg.norm(recon - a)))
    return (
        np.asarray(toks, np.int64),
        np.asarray(res, np.float32),
        float(max(rl2)),
        float(clipped / len(acts)),
    )


def replay(env, cap, toks, res, table, ref):
    restore_capture(env, cap)
    for i, t in enumerate(toks):
        a = table[int(t)].copy()
        if i < len(res):
            a = np.clip(a + np.clip(res[i], -CAP, CAP), -1, 1).astype(np.float32)
        step_legal(env.sim, a)
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return bool(h.get("SUCCESS_NEW")), h.get("std_reason") or h.get("reason")


def main():
    env_cls = type("E", (), {})
    env = env_cls()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    table = build_k49_table(config_id=K49_CONFIG)

    outcomes = []
    for p in sorted((BASE / "outcomes").glob("*.json")):
        row = json.loads(p.read_text(encoding="utf-8"))
        cont = row.get("continuous_demo") or {}
        if cont.get("path") and cont.get("n_actions", 0) > 0 and row.get("ic_pkl"):
            acts = np.load(BASE / cont["path"])["actions"]
            ic = pickle.loads((BASE / row["ic_pkl"]).read_bytes())
            toks, res, max_rl2, clip_frac = quantize(acts, table)
            tok_path = BASE / "tokens" / f"{row['sample_id']}__k49.npz"
            np.savez_compressed(
                tok_path,
                tokens=toks,
                residuals=res,
                handoff_token=np.asarray([HANDOFF], np.int64),
                continuous_actions=np.asarray(acts, np.float32),
                cap=np.asarray([CAP], np.float32),
            )
            ok, reason = replay(env, ic["capture"], toks, res, table, ic["ref"])
            gap = bool(cont.get("verified")) and ((not ok) or clip_frac > 0.25 or max_rl2 > 0.15)
            row["k49_demo"] = {
                "path": str(tok_path.relative_to(BASE)),
                "n_tokens": int(len(toks)),
                "max_recon_l2": max_rl2,
                "residual_clipped_frac": clip_frac,
                "encoding": "nearest_motion_token_plus_residual_cap_0p25",
                "replay_SUCCESS_NEW": ok,
                "verified": ok,
                "nontrivial_representability": ok,
                "replay_reason": reason,
                "config_id": K49_CONFIG,
            }
            row["repr_gap"] = gap
            p.write_text(json.dumps(jsafe(row), indent=2), encoding="utf-8")
            print(f"{row['sample_id']}: k49={ok} recon_l2={max_rl2:.3f} gap={gap}", flush=True)
        outcomes.append(row)

    # Dedup RL/SFT by capture_fp preferring genuine_stage1_plan_b and NONTRIVIAL
    by_fp = {}
    for r in outcomes:
        fp = r.get("capture_fp") or r["sample_id"]
        prev = by_fp.get(fp)
        if prev is None:
            by_fp[fp] = r
            continue
        # prefer genuine B nontrivial over duplicate post_b1
        score = (
            int(r.get("genuine_stage1_plan_b") or 0) * 10
            + int(r.get("sft_category") == "NONTRIVIAL_PRECISION") * 5
            + int((r.get("k49_demo") or {}).get("nontrivial_representability") or 0)
        )
        pscore = (
            int(prev.get("genuine_stage1_plan_b") or 0) * 10
            + int(prev.get("sft_category") == "NONTRIVIAL_PRECISION") * 5
            + int((prev.get("k49_demo") or {}).get("nontrivial_representability") or 0)
        )
        if score > pscore:
            by_fp[fp] = r
    deduped = list(by_fp.values())

    # Rebuild SFT
    sft_path = BASE / "sft/sft_manifest.jsonl"
    sft_path.write_text("", encoding="utf-8")
    n_sft = 0
    for r in deduped:
        cat = r.get("sft_category")
        if cat not in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION", "INTERMEDIATE_CURRICULUM"):
            continue
        cont = r.get("continuous_demo") or {}
        k49 = r.get("k49_demo") or {}
        if cat != "DIRECT_HANDOFF" and not cont.get("verified"):
            continue
        row = {
            "sample_id": r["sample_id"],
            "sft_category": cat,
            "plan_provenance": r["plan_provenance"],
            "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
            "ic_pkl": r.get("ic_pkl"),
            "continuous_path": cont.get("path"),
            "continuous_verified": cont.get("verified"),
            "continuous_n_actions": cont.get("n_actions"),
            "k49_path": k49.get("path"),
            "k49_verified": k49.get("verified"),
            "k49_nontrivial_representability": k49.get("nontrivial_representability"),
            "repr_gap": r.get("repr_gap"),
        }
        with sft_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsafe(row)) + "\n")
        n_sft += 1

    # Rebuild RL + splits
    rl_rows = []
    for r in deduped:
        if r.get("status") not in (None, "OK") and r.get("status") != "OK":
            if not r.get("ic_pkl"):
                continue
        if not r.get("ic_pkl"):
            continue
        if r.get("status") == "GATE_FAIL" or r.get("status") == "LOAD_FAIL":
            continue
        rl_rows.append(
            {
                "sample_id": r["sample_id"],
                "plan_provenance": r["plan_provenance"],
                "source_kind": r.get("source_kind"),
                "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                "ic_pkl": r.get("ic_pkl"),
                "direct_SUCCESS_NEW": (r.get("direct") or {}).get("SUCCESS_NEW"),
                "oracle_SUCCESS_NEW": (r.get("precision") or {}).get("SUCCESS_NEW"),
                "sft_category": r.get("sft_category"),
                "difficulty": (
                    "easy_direct"
                    if (r.get("direct") or {}).get("SUCCESS_NEW")
                    else (
                        "nontrivial_oracle_ok"
                        if (r.get("precision") or {}).get("SUCCESS_NEW")
                        else "hard_oracle_fail"
                    )
                ),
                "e_pos_mm": (r.get("errors_entry") or {}).get("e_pos_mm"),
                "dynamics": r.get("dynamics"),
                "source_episode": r.get("ic_id"),
                "intermediate_curriculum": r.get("intermediate_curriculum", False),
            }
        )
    episodes = sorted({r["source_episode"] for r in rl_rows})
    rng = np.random.default_rng(5478)
    order = list(episodes)
    rng.shuffle(order)
    n = len(order)
    n_hold = max(1, int(0.15 * n)) if n >= 7 else max(0, n // 5)
    n_val = max(1, int(0.15 * n)) if n >= 7 else max(0, n // 5)
    hold, val = set(order[:n_hold]), set(order[n_hold : n_hold + n_val])
    splits = {"train": [], "val": [], "holdout": []}
    rl_path = BASE / "rl/rl_ic_manifest.jsonl"
    rl_path.write_text("", encoding="utf-8")
    for r in rl_rows:
        ep = r["source_episode"]
        split = "holdout" if ep in hold else ("val" if ep in val else "train")
        r["split"] = split
        splits[split].append(r["sample_id"])
        with rl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsafe(r)) + "\n")
    (BASE / "rl/splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # Summary + REPORT enrichment
    def stats(rows):
        return {
            "n": len(rows),
            "genuine_stage1_plan_b": sum(1 for r in rows if r.get("genuine_stage1_plan_b")),
            "DIRECT_HANDOFF": sum(1 for r in rows if r.get("sft_category") == "DIRECT_HANDOFF"),
            "NONTRIVIAL_PRECISION": sum(1 for r in rows if r.get("sft_category") == "NONTRIVIAL_PRECISION"),
            "FAILED_ORACLE": sum(1 for r in rows if r.get("sft_category") == "FAILED_ORACLE"),
            "INTERMEDIATE_CURRICULUM": sum(1 for r in rows if r.get("sft_category") == "INTERMEDIATE_CURRICULUM"),
            "continuous_verified": sum(1 for r in rows if (r.get("continuous_demo") or {}).get("verified")),
            "k49_verified": sum(1 for r in rows if (r.get("k49_demo") or {}).get("verified")),
            "k49_nontrivial_ok": sum(1 for r in rows if (r.get("k49_demo") or {}).get("nontrivial_representability")),
            "repr_gaps": sum(1 for r in rows if r.get("repr_gap")),
            "distinct_episodes": len({r.get("ic_id") for r in rows}),
        }

    per = {p: stats([r for r in deduped if r.get("plan_provenance") == p]) for p in sorted({r.get("plan_provenance") for r in deduped})}
    fail_phases = Counter()
    for r in deduped:
        if r.get("sft_category") == "FAILED_ORACLE":
            for ph in (r.get("precision") or {}).get("failures") or ["unknown"]:
                fail_phases[str(ph)[:80]] += 1

    summary = {
        "n_outcomes_raw": len(outcomes),
        "n_deduped": len(deduped),
        "sft_categories": dict(Counter(r.get("sft_category") for r in deduped)),
        "source_kinds": dict(Counter(r.get("source_kind") for r in deduped)),
        "per_plan_provenance": per,
        "genuine_plan_b_nontrivial": sum(
            1
            for r in deduped
            if r.get("genuine_stage1_plan_b") and r.get("sft_category") == "NONTRIVIAL_PRECISION"
        ),
        "genuine_plan_b_nontrivial_k49_ok": sum(
            1
            for r in deduped
            if r.get("genuine_stage1_plan_b")
            and r.get("sft_category") == "NONTRIVIAL_PRECISION"
            and (r.get("k49_demo") or {}).get("nontrivial_representability")
        ),
        "hard_ic_nontrivial_k49_ok": sum(
            1
            for r in deduped
            if r.get("plan_provenance") == "HARD_IC"
            and r.get("sft_category") == "NONTRIVIAL_PRECISION"
            and (r.get("k49_demo") or {}).get("nontrivial_representability")
        ),
        "intermediate_curriculum": sum(1 for r in deduped if r.get("sft_category") == "INTERMEDIATE_CURRICULUM"),
        "common_failure_tags": fail_phases.most_common(15),
        "export": {"n_sft": n_sft, "n_rl": len(rl_rows), "splits": {k: len(v) for k, v in splits.items()}},
        "k49_fixup": "residual_cap_0p25_requantize",
    }
    man = BASE / "manifests/ic_manifest_v1.jsonl"
    man.write_text("", encoding="utf-8")
    for r in deduped:
        with man.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    jsafe(
                        {
                            "sample_id": r["sample_id"],
                            "ic_id": r.get("ic_id"),
                            "plan_provenance": r.get("plan_provenance"),
                            "source_kind": r.get("source_kind"),
                            "genuine_stage1_plan_b": r.get("genuine_stage1_plan_b"),
                            "sft_category": r.get("sft_category"),
                            "ic_pkl": r.get("ic_pkl"),
                            "outcome_json": f"outcomes/{r['sample_id']}.json",
                            "capture_fp": r.get("capture_fp"),
                        }
                    )
                )
                + "\n"
            )

    (BASE / "summary.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")

    lines = [
        "# Stage-2 A/B/C Nontrivial Mining — v1",
        "",
        "Precision Oracle v2.3 **frozen**. Pilot `ic_manifest_frozen.jsonl` **preserved**.",
        f"K49 encoding: nearest motion token + residual ±{CAP}; continuous vs K49 success tracked separately.",
        "",
        "## Headline counts (deduped by capture fingerprint)",
        "",
        f"- Outcomes (raw / deduped): {len(outcomes)} / {len(deduped)}",
        f"- Genuine Plan-B NONTRIVIAL (continuous): **{summary['genuine_plan_b_nontrivial']}**",
        f"- Genuine Plan-B NONTRIVIAL with K49 replay OK: **{summary['genuine_plan_b_nontrivial_k49_ok']}**",
        f"- HARD_IC NONTRIVIAL with K49 replay OK: **{summary['hard_ic_nontrivial_k49_ok']}** (not labeled Plan B)",
        f"- Intermediate curriculum: {summary['intermediate_curriculum']}",
        f"- SFT rows: {n_sft} | RL ICs: {len(rl_rows)} | splits: {summary['export']['splits']}",
        "",
        "## Per plan_provenance",
        "",
        "| Prov | n | GenB | DIRECT | NONTRIVIAL | FAILED | INTER | Cont OK | K49 OK | K49 NT | Repr gaps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p, s in per.items():
        lines.append(
            f"| {p} | {s['n']} | {s['genuine_stage1_plan_b']} | {s['DIRECT_HANDOFF']} | {s['NONTRIVIAL_PRECISION']} | "
            f"{s['FAILED_ORACLE']} | {s['INTERMEDIATE_CURRICULUM']} | {s['continuous_verified']} | "
            f"{s['k49_verified']} | {s['k49_nontrivial_ok']} | {s['repr_gaps']} |"
        )
    lines += [
        "",
        "## Phase notes",
        "",
        "### Phase 1",
        "Screened 47 PRECISION_CANDIDATE states (1 Plan-B diagnostic stable + 45 unique hard_ic + 1 unassigned Stage-1).",
        "hard_ic kept as `plan_provenance=HARD_IC` — not classified as Plan B.",
        "",
        "### Phase 2",
        "Ran Plan-B B1 Recovery tokens on known 12S-not-NEW ICs and FAILED_ORACLE hard ICs; extracted stable pre-HANDOFF captures.",
        f"Genuine Plan-B nontrivial continuous demos reached {summary['genuine_plan_b_nontrivial']} (target was 8).",
        "Shortfall retained as explicit supply gap; HARD_IC nontrivial demos available for corrective SFT with separate provenance.",
        "K49 representation gaps on long DualLoop tapes (max_recon_l2 > 0.5) are logged, not hidden — codebook not modified.",
        "",
        "### Common FAILED_ORACLE tags",
        "",
    ]
    for tag, n in summary["common_failure_tags"][:10]:
        lines.append(f"- `{tag}`: {n}")
    lines += [
        "",
        "## Exports",
        "",
        "- `v1/sft/sft_manifest.jsonl`",
        "- `v1/rl/rl_ic_manifest.jsonl` + `splits.json` (split by source episode)",
        "- `v1/manifests/ic_manifest_v1.jsonl`",
        "",
        "No bulk SFT/RL training launched.",
    ]
    text = "\n".join(lines) + "\n"
    (BASE / "REPORT.md").write_text(text, encoding="utf-8")
    (BASE.parent / "REPORT.md").write_text(text, encoding="utf-8")
    print(json.dumps(jsafe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
