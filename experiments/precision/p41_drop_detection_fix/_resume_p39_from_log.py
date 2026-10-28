"""Resume P39 revalidation: parse full_run.log rows, finish remaining jobs, write aggregates."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p39_working_hold_handoff.expert_validate import (  # noqa: E402
    RULE_IDS,
    aggregate,
    collect_jobs,
    replay_expert_rule,
    r2_reliable,
    write_csv,
)
from experiments.precision.p39_working_hold_handoff.rules import metric_meta  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from experiments.precision.p41_drop_detection_fix.paths import OUT  # noqa: E402
from models.m111_checkpoint import load_m111_book  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402

LOG = OUT / "full_run.log"
DEST = OUT / "p39_reeval" / "expert_validation"
LINE_RE = re.compile(
    r"\[\s*(\d+)/930\]\s+(R[012])\s+(\S+)\s+(\S+)\s+hand=(\d+)\s+step=(\S+)\s+NEW=(\d+)\s+cat=(\S+)"
)


def _read_log_text() -> str:
    raw = LOG.read_bytes()
    if raw[:2] == b"\xff\xfe":
        return raw.decode("utf-16")
    if raw[:2] == b"\xfe\xff":
        return raw.decode("utf-16-be")
    for enc in ("utf-8", "utf-16-le", "gbk", "cp936"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_log() -> list[dict]:
    rows = []
    if not LOG.is_file():
        return rows
    for line in _read_log_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        idx, rule, skill, traj_stub, hand, step, new, cat = m.groups()
        step_v = None if step in ("None", "—") else int(float(step))
        rows.append(
            {
                "log_idx": int(idx),
                "rule": rule,
                "skill": skill,
                "traj_stub": traj_stub,
                "handoff": int(hand),
                "handoff_step": step_v,
                "SUCCESS_NEW": int(new),
                "fail_category": cat,
                # placeholders filled on resume match
                "SUCCESS_12S": 1 if cat == "SUCCESS" else (1 if "HOLD_FAIL" in cat else 0),
                "drop_before_handoff": 0,
                "never_satisfied": int(hand) == 0 and "NEVER" in cat,
                "eval_reason": cat.replace("POST_HANDOFF:", "") if cat.startswith("POST_HANDOFF:") else (
                    "SUCCESS_12S" if cat == "SUCCESS" else cat
                ),
            }
        )
    # keep latest by log_idx
    rows.sort(key=lambda r: r["log_idx"])
    return rows


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "RULE_METRIC.json").write_text(json.dumps(metric_meta(), indent=2), encoding="utf-8")

    parsed = parse_log()
    print(f"parsed log rows={len(parsed)} last_idx={parsed[-1]['log_idx'] if parsed else None}", flush=True)

    jobs = collect_jobs()
    # Build expected order: for each job, R0/R1/R2
    expected = []
    for job in jobs:
        for rule in RULE_IDS:
            expected.append((job, rule))
    assert len(expected) == 930

    # Match parsed stubs to full traj_ids by sequential order
    n_done = len(parsed)
    # Re-run from n_done to get full fidelity rows for remainder; for completed,
    # re-execute is expensive — instead re-run ALL remaining only with full replay_expert_rule,
    # and for completed use a second pass that maps stub→traj_id from expected order.

    full_rows: list[dict] = []
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()

    for i, (job, rule) in enumerate(expected):
        skill = job["skill"]
        traj_id = job["traj_id"]
        if i < n_done:
            pr = parsed[i]
            # sanity: skill and rule and stub prefix
            stub = pr["traj_stub"]
            if pr["rule"] != rule or pr["skill"] != skill:
                print(f"WARN order mismatch at {i}: log={pr} exp={rule}/{skill}", flush=True)
            if not traj_id.startswith(stub) and stub not in traj_id:
                # still accept sequential alignment
                pass
            # Reconstruct minimal row; SUCCESS_12S heuristic from cat
            cat = pr["fail_category"]
            s12 = 1 if cat in ("SUCCESS", "HOLD_FAIL_12S_OK") else 0
            if cat == "SUCCESS":
                snew = 1
            else:
                snew = int(pr["SUCCESS_NEW"])
            full_rows.append(
                {
                    "traj_id": traj_id,
                    "skill": skill,
                    "family": job["family"],
                    "split": job["split"],
                    "source": job["source"],
                    "lineage_id": job.get("lineage_id"),
                    "rule": rule,
                    "handoff": pr["handoff"],
                    "handoff_step": pr["handoff_step"],
                    "SUCCESS_12S": s12,
                    "SUCCESS_NEW": snew,
                    "fail_category": cat,
                    "eval_reason": pr["eval_reason"],
                    "drop_before_handoff": 0,
                    "never_satisfied": int(pr["handoff"] == 0),
                    "from_log": 1,
                }
            )
            continue

        # Fresh replay for remaining
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]

            def restore(ep=ep):
                restore_start(env, ep)

            ref = ep["ref"]
        else:
            cap, ref = job["capture"], job["ref"]

            def restore(cap=cap):
                restore_capture(env, cap)

        restore()
        m0 = measure(env.sim, U_TGT, P_TGT)
        res = replay_expert_rule(
            env,
            restore_fn=restore,
            ref=ref,
            tokens=job["tokens"],
            residuals=job["residuals"],
            book=book,
            rho=rho,
            rule_id=rule,
        )
        row = {
            "traj_id": traj_id,
            "skill": skill,
            "family": job["family"],
            "split": job["split"],
            "source": job["source"],
            "lineage_id": job.get("lineage_id"),
            "pre_theta": float(m0["e_theta_deg"]),
            "pre_e_pos": float(m0["e_pos_mm"]),
            "from_log": 0,
            **res,
        }
        full_rows.append(row)
        print(
            f"  [{i+1}/930] {rule} {skill} {traj_id[:36]} "
            f"hand={row['handoff']} NEW={row['SUCCESS_NEW']} cat={row['fail_category']}",
            flush=True,
        )

    write_csv(DEST / "EXPERT_ROLLOUTS.csv", full_rows)
    agg = aggregate(full_rows)
    write_csv(DEST / "AGGREGATE.csv", agg)
    # per-root
    by_key: dict = {}
    for r in full_rows:
        key = (r["skill"], r["traj_id"])
        by_key.setdefault(key, {"skill": r["skill"], "traj_id": r["traj_id"], "split": r.get("split")})
        rid = r["rule"]
        by_key[key][f"{rid}_handoff"] = r["handoff"]
        by_key[key][f"{rid}_step"] = r.get("handoff_step")
        by_key[key][f"{rid}_NEW"] = r["SUCCESS_NEW"]
        by_key[key][f"{rid}_12S"] = r["SUCCESS_12S"]
        by_key[key][f"{rid}_cat"] = r["fail_category"]
    write_csv(DEST / "PER_ROOT_COMPARE.csv", list(by_key.values()))
    reliable, flags = r2_reliable(agg)
    verdict = {"r2_reliable": reliable, "flags": flags, "n_rows": len(full_rows), "n_from_log": n_done}
    (DEST / "R2_RELIABILITY.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("=== Expert resume done ===", verdict, flush=True)
    return {"rows": full_rows, "agg": agg, "verdict": verdict}


if __name__ == "__main__":
    main()
