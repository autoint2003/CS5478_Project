"""A1-focused ablation: contact-rich HANDOFF variants (no geometric margin).

Priority: maximize A1 SUCCESS_NEW / eliminate LOST_GRASP.
D is out of scope here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p43_r3_working_hold_v2.paths import OUT  # noqa: E402
from evaluation.success_criteria import HOLD_S  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

# R2 baseline + contact-only variants (WORKING thresholds unchanged).
VARIANTS = {
    "R2": {"min_pad": 0, "terminal_fallback": False},
    "Rc3": {"min_pad": 3, "terminal_fallback": False},
    "Rc4": {"min_pad": 4, "terminal_fallback": False},
    "Rc4F": {"min_pad": 4, "terminal_fallback": True},  # if never ready → handoff at expert end
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


class ContactDwell:
    def __init__(self, min_pad: int, dt: float = float(DT_POLICY), dwell_s: float = float(HOLD_S)):
        self.min_pad = int(min_pad)
        self.dt = float(dt)
        self.dwell_s = float(dwell_s)
        self.ok_s = 0.0
        self.n_resets = 0
        self._active = False
        self.ever_partial = False

    def observe(self, sim, step: int) -> dict[str, Any]:
        if latched(sim):
            if self._active:
                self.n_resets += 1
            self._active = False
            self.ok_s = 0.0
            return {"ready": False, "frame_ok": False, "latched": True, "ok_s": 0.0}

        wk = working_geometry(sim)
        pb = pad_bilateral(sim)
        nL, nR = int(pb["nL"]), int(pb["nR"])
        bilateral = bool(pb["bilateral"])
        working_ok = bool(wk["working_ok"])
        pad_ok = True if self.min_pad <= 0 else (min(nL, nR) >= self.min_pad)
        frame_ok = working_ok and bilateral and pad_ok

        if frame_ok:
            if not self._active:
                self._active = True
            self.ok_s += self.dt
            self.ever_partial = True
        else:
            if self._active:
                self.n_resets += 1
            self._active = False
            self.ok_s = 0.0

        return {
            "ready": bool(self.ok_s + 1e-9 >= self.dwell_s),
            "frame_ok": frame_ok,
            "working_ok": working_ok,
            "bilateral": bilateral,
            "latched": False,
            "ok_s": self.ok_s,
            "theta": float(wk["dtheta_deg"]),
            "e_pos": float(wk["e_pos_mm"]),
            "nL": nL,
            "nR": nR,
        }


def replay(
    env,
    *,
    restore_fn: Callable,
    ref,
    tokens,
    residuals,
    book,
    rho,
    variant: str,
) -> dict[str, Any]:
    cfg = VARIANTS[variant]
    restore_fn()
    tracker = ContactDwell(min_pad=cfg["min_pad"])
    toks = np.asarray(tokens, np.int64)
    res = np.asarray(residuals, np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)

    handoff = False
    handoff_step = None
    used_fallback = False
    last: dict[str, Any] = {}
    drop = False

    for t in range(n + 1):
        if latched(env.sim):
            drop = True
            break
        st = tracker.observe(env.sim, t)
        last = st
        if st["ready"]:
            handoff = True
            handoff_step = t
            break
        if t >= n:
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))

    if not handoff and not drop and cfg["terminal_fallback"]:
        # Expert finished still holding — HANDOFF at terminal (P42: often SUCCESS).
        handoff = True
        handoff_step = n
        used_fallback = True
        last = tracker.observe(env.sim, n)

    out: dict[str, Any] = {
        "rule": variant,
        "handoff": int(handoff),
        "handoff_step": handoff_step,
        "used_fallback": int(used_fallback),
        "never_satisfied": int(not handoff and not drop),
        "drop_before_handoff": int(drop and not handoff),
        "n_dwell_resets": tracker.n_resets,
        "hand_nL": last.get("nL") if handoff else None,
        "hand_nR": last.get("nR") if handoff else None,
        "hand_theta": last.get("theta") if handoff else None,
        "hand_e_pos": last.get("e_pos") if handoff else None,
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "eval_reason": None,
        "fail_category": None,
    }

    if drop and not handoff:
        out["eval_reason"] = "DROP_BEFORE_HANDOFF"
        out["fail_category"] = "DROP_BEFORE_HANDOFF"
        return out
    if not handoff:
        out["eval_reason"] = "NEVER_SATISFIED_RULE"
        out["fail_category"] = "NEVER_SATISFIED_RULE"
        return out

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    reason = hand.get("reason")
    out.update({"SUCCESS_12S": s12, "SUCCESS_NEW": snew, "eval_reason": reason})
    if snew:
        out["fail_category"] = "SUCCESS"
    elif s12 and not snew:
        out["fail_category"] = "HOLD_FAIL_12S_OK"
    else:
        out["fail_category"] = f"POST_HANDOFF:{reason}"
    return out


def run(*, limit: int = 0) -> dict:
    dest = OUT / "a1_contact_ablation"
    dest.mkdir(parents=True, exist_ok=True)
    book, rho, _ = load_m111_book(M111_META)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    jobs = [j for j in collect_jobs() if j["skill"] == "A1"]
    if limit > 0:
        jobs = jobs[:limit]

    rows: list[dict] = []
    variants = list(VARIANTS.keys())
    total = len(jobs) * len(variants)
    k = 0
    print(f"A1 contact ablation jobs={len(jobs)} variants={variants}", flush=True)

    for job in jobs:
        ep = job["ep"]

        def restore(ep=ep):
            restore_start(env, ep)

        for variant in variants:
            k += 1
            ev = replay(
                env,
                restore_fn=restore,
                ref=ep["ref"],
                tokens=job["tokens"],
                residuals=job["residuals"],
                book=book,
                rho=rho,
                variant=variant,
            )
            row = {"traj_id": job["traj_id"], "skill": "A1", "split": job["split"], **ev}
            rows.append(row)
            write_csv(dest / "ROLLOUTS.csv", rows)
            print(
                f"  [{k}/{total}] {variant} {job['traj_id'][:40]} "
                f"hand={ev['handoff']} fb={ev['used_fallback']} NEW={ev['SUCCESS_NEW']} "
                f"cat={ev['fail_category']} step={ev['handoff_step']}",
                flush=True,
            )

    # Aggregate
    agg = []
    for v in variants:
        sub = [r for r in rows if r["rule"] == v]
        n = len(sub)
        snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
        s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
        hand = sum(int(r["handoff"]) for r in sub)
        lost = sum(1 for r in sub if "LOST_GRASP" in str(r["fail_category"]))
        never = sum(int(r["never_satisfied"]) for r in sub)
        fb = sum(int(r["used_fallback"]) for r in sub)
        agg.append(
            {
                "rule": v,
                "n": n,
                "SUCCESS_NEW": snew,
                "NEW_rate": snew / n if n else None,
                "SUCCESS_12S": s12,
                "handoff": hand,
                "LOST_GRASP": lost,
                "never_satisfied": never,
                "fallback_used": fb,
                "fail_cats": dict(Counter(r["fail_category"] for r in sub)),
                "NEW_given_handoff": f"{snew}/{hand}",
            }
        )
    write_csv(dest / "AGGREGATE.csv", agg)
    (dest / "AGGREGATE.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = [
        "# A1 Contact-Only HANDOFF Ablation",
        "",
        f"Generated: {utc()}",
        "",
        "Priority: A1 SUCCESS_NEW high, LOST_GRASP ≈ 0. No geometric margin.",
        "",
        "| Rule | NEW/all | NEW|hand | LOST_GRASP | never | fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a in agg:
        lines.append(
            f"| {a['rule']} | {a['SUCCESS_NEW']}/{a['n']} | {a['NEW_given_handoff']} | "
            f"{a['LOST_GRASP']} | {a['never_satisfied']} | {a['fallback_used']} |"
        )
    lines += [
        "",
        "- **R2**: WORKING+bilateral 0.5s",
        "- **Rc3/Rc4**: R2 + min(nL,nR)≥3/4",
        "- **Rc4F**: Rc4, else HANDOFF at expert terminal",
        "",
        f"fail_cats: {json.dumps({a['rule']: a['fail_cats'] for a in agg})}",
        "",
    ]
    (dest / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return {"agg": agg, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run(limit=args.limit)


if __name__ == "__main__":
    main()
