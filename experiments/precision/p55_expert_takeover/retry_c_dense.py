"""Run expert continuation on dense C candidates; merge into EXPERT_CONTINUATION_RESULTS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p55_expert_takeover.expert_continue import continue_expert_from_candidate
from experiments.precision.p55_expert_takeover.ic_resolve import resolve_starts
from experiments.precision.p55_expert_takeover.paths import OUT


def main(*, c_catch_rollouts: int = 48) -> dict:
    dense_path = OUT / "TAKEOVER_CANDIDATES_C_DENSE.json"
    blob = json.loads(dense_path.read_text(encoding="utf-8"))
    candidates = blob["candidates"]
    starts = resolve_starts(sorted({c["ic_id"] for c in candidates}))
    env = make_env()

    results = []
    rescued: set[str] = set()
    for i, cand in enumerate(candidates):
        start = starts.get(cand["ic_id"])
        if start is None:
            continue
        print(
            f"[{i+1}/{len(candidates)}] dense-C {cand['candidate_id']} "
            f"e={cand['metrics'].get('e_pos_mm'):.1f} bil={cand['metrics'].get('bilateral')}",
            flush=True,
        )
        row = continue_expert_from_candidate(
            env, start, cand, c_catch_rollouts=c_catch_rollouts
        )
        acts = row.pop("_actions", [])
        if row.get("SUCCESS_RELAXED_12S"):
            rescued.add(cand["ic_id"])
            ap = OUT / "replay" / f"expert_actions_{cand['candidate_id'].replace(':','_').replace('/','_')}.npy"
            ap.parent.mkdir(parents=True, exist_ok=True)
            if acts:
                np.save(ap, np.stack([np.asarray(a, np.float32) for a in acts]))
                row["expert_actions_path"] = str(ap)
        results.append(row)
        print(
            f"  -> accepted={row['accepted']} relaxed={row['SUCCESS_RELAXED_12S']} "
            f"status={row['status']} reason={row.get('eval_reason')}",
            flush=True,
        )

    # Merge into main continuation results
    main_path = OUT / "EXPERT_CONTINUATION_RESULTS.json"
    main = json.loads(main_path.read_text(encoding="utf-8"))
    # Drop prior dense_c rows if re-run
    kept = [r for r in main["results"] if "::dense_c" not in str(r.get("candidate_id"))]
    kept.extend(results)
    rescued_all = {r["ic_id"] for r in kept if r.get("SUCCESS_RELAXED_12S")}
    by_skill: dict = {}
    for sk in ("A1", "B1", "C", "D"):
        sub = [r for r in kept if r.get("skill") == sk]
        ics = {r["ic_id"] for r in sub}
        res = {r["ic_id"] for r in sub if r.get("SUCCESS_RELAXED_12S")}
        by_skill[sk] = {
            "n_candidates": len(sub),
            "n_ics": len(ics),
            "n_accepted": sum(1 for r in sub if r.get("accepted")),
            "n_success_relaxed": sum(1 for r in sub if r.get("SUCCESS_RELAXED_12S")),
            "n_ics_rescued": len(res),
            "rescue_rate_ics": len(res) / max(len(ics), 1),
        }
    out = {
        "n_candidates": len(kept),
        "n_ics": len({r["ic_id"] for r in kept}),
        "n_ics_rescued": len(rescued_all),
        "rescued_ics": sorted(rescued_all),
        "by_skill": by_skill,
        "c_dense_pass": {
            "n_candidates": len(results),
            "n_ics_rescued": len(rescued),
            "rescued_ics": sorted(rescued),
            "c_catch_rollouts": c_catch_rollouts,
        },
        "results": kept,
    }
    main_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "EXPERT_CONTINUATION_C_DENSE.json").write_text(
        json.dumps({"results": results, "rescued_ics": sorted(rescued)}, indent=2),
        encoding="utf-8",
    )
    print("C dense rescued", len(rescued), "/", len({c['ic_id'] for c in candidates}))
    print("total rescued", out["n_ics_rescued"], "/", out["n_ics"])
    return out


if __name__ == "__main__":
    main()
