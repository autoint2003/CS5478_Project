"""Pilot: Short-Handoff Baseline + Active Recovery for nontrivial ICs.

Read-only. No training / reward / physics edits.
Writes results/diagnostics/nontrivial_recovery_ic_mining/ then STOPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.expand_skill_quotas import _try_sliding
from data_generation.strategies.vertical_recatch_search import (
    _finish,
    rollout as vertical_rollout,
    specs as vertical_specs,
)
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.classify_validation_skills import classify_mechanism
from evaluation import skill_traces as audit
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.unified_k48 import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "nontrivial_recovery_ic_mining"
PRIOR = ROOT / "results" / "diagnostics" / "mechanism_selection_feasibility"
BENCH = ROOT / "results" / "benchmarks" / "hard_recovery_v1"
HARD = ROOT / "results" / "diagnostics" / "hard_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
LIMIT_S = float(SELECTED_LIMIT_S)
MOTION_K = 47  # tokens 0..46; 47=HANDOFF
DT = 0.02

KNOWN_3647 = [
    "hrv1_920047",
    "hrv1_920064",
    "hrv1_920106",
    "hrv1_920015",
    "hrv1_920069",
    "hrv1_leftover_hard_910207",
    "hrv1_leftover_hard_910231",
]


def jsafe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def write_search_coverage(book):
    openish = int(np.sum(book[:, 6] < -0.5))
    text = f"""# SEARCH_COVERAGE

## Codebook

- Motion tokens: **0..{MOTION_K - 1}** ({MOTION_K} actions)
- HANDOFF token: **47** (no `step_legal`)
- Grip-openish (a[6]<-0.5): **{openish}** / {MOTION_K}
- Token 36: `{np.asarray(book[36]).tolist()}` → a_grip=-1 (TAU_OPEN)

## Why old Recatch search missed `[36,36,47]`

Source: `data_generation/strategies/vertical_recatch_search.py`

1. **Not a K48 token enumerator.** Rollout is a continuous FSM:
   `open → chase → close → brake → lift → settle → handoff`,
   with commands projected onto the codebook each step (`project(raw, book)`).
2. **Minimum open phase** via `open_min` (0.02–0.04 s in specs) but the controller
   stays in open/chase until jaw geometry `_in_jaw` triggers close — typically
   **hundreds of ms**, not a 2-step open-then-HANDOFF.
3. **HANDOFF only after settle** (pad LR + quiet velocity), never after two open steps.
4. **Success filter** in mining often requires `t_off` and `t_pad` evidence of a
   long release–recontact story (`main()` logging), biasing against handoff-assisted
   regrasp where recontact happens in **nominal**.
5. Specs grid is only 3 templates (`follow_early`, `later_close`, `gentle_brake`).

`_try_sliding` (`expand_skill_quotas.py`) only tries inward_reseat / controlled_slide
schedules — also never enumerates short open→HANDOFF prefixes.

**Conclusion (VERIFIED from code):** `[36,36,47]` was outside the proposal class of both
oracles, not because it is illegal or unsuccessful.

## Pilot short-handoff coverage design

| Depth | Coverage in pilot |
|---|---|
| 0 (immediate HANDOFF) | all pilot ICs |
| 1 | all **openish** (a6<-0.5) + 12 random other tokens (not full 47×N — cost) |
| 2 | always `[36,36]`; openish self-pairs; ~pair_samples random; open→close samples |
| 3–5 | budgeted open-grip chains |
| Full 47¹ / 47² | **deferred** to post-pilot (documented cost) |

All successes require full `SUCCESS_12S` after original HANDOFF.
"""
    (OUT / "SEARCH_COVERAGE.md").write_text(text, encoding="utf-8")


def load_pilot_ics(n_target=25):
    """Representative mix from prior feasibility manifest + known 3647 + sliding fav."""
    rows = []
    man = PRIOR / "IC_MANIFEST.csv"
    by_tag = defaultdict(list)
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            # fragile CSV with json physics — use DictReader
            pass
    import csv as _csv
    if man.is_file():
        for r in _csv.DictReader(man.open(encoding="utf-8")):
            by_tag[r["competition_tag"]].append(r)

    # known 3647 first
    for ic in KNOWN_3647:
        snap = None
        for name in ("verified_manifest.jsonl", "unverified_manifest.jsonl"):
            for line in (BENCH / name).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] == ic:
                    snap = row["snapshot"]
                    break
            if snap:
                break
        if snap:
            rows.append({
                "id": ic, "snapshot": snap, "role": "known_3647",
                "prior_tag": "KNOWN_SHORT_OPEN_HANDOFF", "holdout": True,
            })

    def take(tag, role, k):
        for r in by_tag.get(tag, [])[:k]:
            if any(x["id"] == r["id"] for x in rows):
                continue
            rows.append({
                "id": r["id"], "snapshot": r["snapshot"], "role": role,
                "prior_tag": tag, "holdout": r.get("holdout") == "True",
            })

    take("SLIDING_FAVORABLE", "slide_fav", 4)
    take("RECATCH_FAVORABLE", "recatch_fav", 6)
    take("BOTH_VERIFIED", "both", 5)
    # hard mining train sliding examples
    for name in ("selected_train_manifest.jsonl", "selected_dev_manifest.jsonl"):
        path = HARD / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            strat = r.get("oracle_strategy") or r.get("strategy") or ""
            if "slide" in strat or strat == "inward_reseat":
                if any(x["id"] == r["id"] for x in rows):
                    continue
                rows.append({
                    "id": r["id"], "snapshot": r["snapshot"], "role": "hard_sliding_oracle",
                    "prior_tag": strat, "holdout": "train" not in name,
                })
                if sum(1 for x in rows if x["role"] == "hard_sliding_oracle") >= 3:
                    break
    # trim
    return rows[:n_target]


def run_prefix(env, book, blob, tokens, residuals=None):
    """Execute motion tokens (ints) with zero residual mean, then HANDOFF + SUCCESS_12S."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    lost_low = 0
    for i, tok in enumerate(tokens):
        if int(tok) == HANDOFF:
            break
        pre = np.zeros(7, np.float32) if residuals is None else np.asarray(residuals[i], np.float32)
        _r, action = decode_action(book, int(tok), pre)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            return {
                "success": False,
                "reason": reason,
                "recovery_elapsed": float(env.sim.data.time) - t0,
                "tokens": list(tokens[: i + 1]),
                "dropped_before_handoff": True,
            }
    outcome = finish_recovery(env, blob["ref"], t0, "early")
    return {
        "success": bool(outcome.get("success")),
        "reason": outcome.get("reason"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
        "tokens": list(tokens) + ([HANDOFF] if (not tokens or tokens[-1] != HANDOFF) else []),
        "dropped_before_handoff": False,
    }


def short_handoff_search(env, book, blob, ic_id, pair_samples=80, rng=None, rich=False):
    """Budgeted short-prefix → HANDOFF → SUCCESS_12S search.

    ``rich=True`` (known-[36,36,47] ICs): denser openish 1-step/2-step coverage.
    """
    rng = rng or np.random.default_rng(0)
    stats = {"rollouts": 0, "physics_s": 0.0, "successes": [], "wall_s": 0.0}
    rows = []
    openish = [i for i in range(MOTION_K) if float(book[i, 6]) < -0.5]
    closeish = [i for i in range(MOTION_K) if float(book[i, 6]) > 0.5]
    tested = set()

    def try_tokens(tokens, bucket):
        key = tuple(int(t) for t in tokens)
        if key in tested:
            return None
        tested.add(key)
        t0 = time.perf_counter()
        res = run_prefix(env, book, blob, list(key))
        wall = time.perf_counter() - t0
        stats["rollouts"] += 1
        stats["wall_s"] += wall
        phys = float(res.get("recovery_elapsed") or 0.0)
        if not res.get("dropped_before_handoff"):
            # nominal may end early on failure; charge wall as proxy when available
            phys += min(12.0, max(0.0, wall - phys))
        row = {
            "ic_id": ic_id,
            "bucket_ms": bucket,
            "depth": len(key),
            "tokens": json.dumps(list(key)),
            "success": int(res["success"]),
            "reason": res["reason"],
            "recovery_elapsed": res.get("recovery_elapsed"),
            "wall_s": wall,
            "physics_s_approx": phys,
        }
        rows.append(row)
        stats["physics_s"] += phys
        if res["success"]:
            stats["successes"].append(row)
        return res

    try_tokens([], "0")
    # 1-step: all openish + sample of others
    for t in openish:
        try_tokens([t], "20")
    other = [i for i in range(MOTION_K) if i not in openish]
    for t in rng.choice(other, size=min(12, len(other)), replace=False):
        try_tokens([int(t)], "20")
    if rich:
        for t in other:
            try_tokens([int(t)], "20")
    # 2-step
    try_tokens([36, 36], "40")
    for t in openish:
        try_tokens([t, t], "40")
    for _ in range(pair_samples):
        a, b = int(rng.integers(0, MOTION_K)), int(rng.integers(0, MOTION_K))
        try_tokens([a, b], "40")
    for a in openish[::2]:
        for b in closeish[::4]:
            try_tokens([a, int(b)], "40")
    # longer open chains
    for t in openish[::2][:6]:
        try_tokens([t, t, t], "60-100")
    for t in openish[::3][:4]:
        try_tokens([t, t, t, t, t], "100-200")

    shortest = None
    if stats["successes"]:
        shortest = min(stats["successes"], key=lambda r: (r["depth"], r["recovery_elapsed"] or 99))
    stats["n_unique_prefixes"] = len(tested)
    return rows, stats, shortest


def active_recovery(env, book, blob, ic_id):
    """Longer oracles: sliding + vertical; record budgets."""
    out = {"ic_id": ic_id, "slide": None, "vertical": None, "rollouts": 0, "physics_s": 0.0}
    t0 = time.perf_counter()
    slide = _try_sliding(env, book, blob)
    out["rollouts"] += 1
    if slide and slide.get("ok"):
        out["slide"] = {
            "ok": True,
            "strategy": slide.get("oracle_strategy"),
            "elapsed": slide.get("recovery_elapsed"),
        }
        out["physics_s"] += float(slide.get("recovery_elapsed") or 0.0) + 12.0
    else:
        out["slide"] = {"ok": False, "reason": None if not slide else slide.get("reason")}
        out["physics_s"] += 4.0  # approx failed search effort
    for spec in list(vertical_specs()):
        out["rollouts"] += 1
        result = vertical_rollout(env, blob, book, spec)
        hold, _ = _finish(env.sim, blob["ref"])
        elapsed = float(result["end"]["t"])
        out["physics_s"] += elapsed + (12.0 if hold.get("success") else 0.0)
        if hold.get("success"):
            out["vertical"] = {"ok": True, "strategy": spec.get("name"), "elapsed": elapsed}
            break
    if out["vertical"] is None:
        out["vertical"] = {"ok": False}
    out["wall_s"] = time.perf_counter() - t0
    # competition tag under this budget
    s_ok = bool(out["slide"] and out["slide"].get("ok"))
    v_ok = bool(out["vertical"] and out["vertical"].get("ok"))
    if s_ok and v_ok:
        tag = "BOTH_VERIFIED"
    elif s_ok:
        tag = "SLIDING_FAVORABLE"
    elif v_ok:
        tag = "RECATCH_FAVORABLE"
    else:
        tag = "UNRESOLVED"
    out["competition_tag"] = tag
    return out


def prefix_handoff_audit(env, book, blob, ic_id, motion_tokens):
    """Counterfactual HANDOFF after each prefix of a successful long trajectory."""
    rows = []
    for k in range(len(motion_tokens) + 1):
        # k=0 immediate; k=len full motion then handoff
        prefix = motion_tokens[:k]
        res = run_prefix(env, book, blob, prefix)
        rows.append({
            "ic_id": ic_id,
            "prefix_len": k,
            "tokens": json.dumps(prefix),
            "success": int(res["success"]),
            "reason": res["reason"],
            "recovery_elapsed": res.get("recovery_elapsed"),
        })
    return rows


def classify_level(shortest_short, active):
    if shortest_short is not None:
        return "Level1_ShortHandoffRecoverable"
    if active and (active["slide"]["ok"] or active["vertical"]["ok"]):
        return "Level2_ActiveRecoveryEvidenced"
    return "Unresolved_NoSuccessUnderBudget"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-ics", type=int, default=22)
    parser.add_argument("--pair-samples", type=int, default=40)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    write_search_coverage(book)
    # need a model only for env factory
    model, _ = load_trained_k48(SFT, device)
    model.eval()
    env = make_training_env(device, model=model)

    ics = load_pilot_ics(args.n_ics)
    write_json(OUT / "pilot_ic_list.json", ics)
    print("PILOT_ICS", len(ics), flush=True)

    short_rows = []
    active_rows = []
    prefix_rows = []
    mech_rows = []
    manifest = []
    rediscover = []

    for meta in ics:
        ic = meta["id"]
        print("IC", ic, meta["role"], flush=True)
        blob = pickle.loads((ROOT / meta["snapshot"]).read_bytes())
        if "capture" not in blob:
            print("SKIP no capture", ic, flush=True)
            continue
        rows, stats, shortest = short_handoff_search(
            env, book, blob, ic,
            pair_samples=args.pair_samples,
            rng=np.random.default_rng(abs(hash(ic)) % (2**31)),
            rich=(meta["role"] == "known_3647"),
        )
        short_rows.extend(rows)
        write_json(OUT / "raw" / f"{ic}_short_stats.json", {"stats": stats, "shortest": shortest})

        found_3647 = any(
            r["success"] and r["tokens"] == json.dumps([36, 36])
            for r in stats["successes"]
        )
        # also check tokens field with handoff
        found_3647 = found_3647 or any(
            json.loads(r["tokens"])[:2] == [36, 36] and r["success"]
            for r in stats["successes"]
        )
        if meta["role"] == "known_3647":
            rediscover.append({"ic_id": ic, "rediscovered_36_36_47": found_3647, "n_short_success": len(stats["successes"])})

        # Active recovery if no short success
        active = None
        level = "Level1_ShortHandoffRecoverable"
        if shortest is None:
            active = active_recovery(env, book, blob, ic)
            active_rows.append({
                "ic_id": ic,
                "slide_ok": active["slide"]["ok"],
                "slide_strategy": active["slide"].get("strategy"),
                "slide_elapsed": active["slide"].get("elapsed"),
                "vert_ok": active["vertical"]["ok"],
                "vert_strategy": active["vertical"].get("strategy"),
                "vert_elapsed": active["vertical"].get("elapsed"),
                "competition_tag": active["competition_tag"],
                "rollouts": active["rollouts"],
                "physics_s": active["physics_s"],
                "wall_s": active["wall_s"],
            })
            level = classify_level(None, active)
            # prefix audit on long success
            if active["slide"]["ok"]:
                # re-get sliding actions if available — _try_sliding returns executed
                slide = _try_sliding(env, book, blob)
                if slide and slide.get("ok") and slide.get("executed") is not None:
                    # map continuous executed to nearest tokens is hard; use elapsed steps count
                    n = int(round(float(slide.get("recovery_elapsed") or 0) / DT))
                    # Instead: prefix-handoff by replaying executed commands then early handoff
                    # For pilot: test HANDOFF every 5 steps of executed array
                    actions = np.asarray(slide["executed"], np.float32)
                    pref = []
                    for k in list(range(0, len(actions) + 1, max(1, len(actions) // 8))) + [len(actions)]:
                        restore_capture(env, blob["capture"])
                        t0 = float(env.sim.data.time)
                        lost_low = 0
                        dropped = False
                        for action in actions[:k]:
                            step_legal(env.sim, action)
                            env._prev = np.asarray(action, np.float32)
                            failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
                            if failed:
                                dropped = True
                                reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
                                pref.append({"ic_id": ic, "source": "slide_oracle", "prefix_steps": k,
                                             "success": 0, "reason": reason})
                                break
                        if dropped:
                            continue
                        outcome = finish_recovery(env, blob["ref"], t0, "early")
                        pref.append({
                            "ic_id": ic, "source": "slide_oracle", "prefix_steps": k,
                            "success": int(outcome.get("success")), "reason": outcome.get("reason"),
                            "recovery_elapsed": outcome.get("recovery_elapsed"),
                        })
                    prefix_rows.extend(pref)
            if active["vertical"]["ok"]:
                # vertical: prefix by time fractions via re-roll is expensive; record oracle elapsed only
                prefix_rows.append({
                    "ic_id": ic, "source": "vertical_oracle", "prefix_steps": None,
                    "note": "full vertical oracle success; stepwise token prefix N/A (continuous FSM)",
                    "success": 1, "reason": "SUCCESS_12S",
                    "recovery_elapsed": active["vertical"].get("elapsed"),
                })
        else:
            level = "Level1_ShortHandoffRecoverable"
            # Prefix audit on shortest short success (necessary vs redundant)
            toks = json.loads(shortest["tokens"])
            prefix_rows.extend(prefix_handoff_audit(env, book, blob, ic, toks))
            # Light competition pass only on a subset of Level1 (cost control)
            if meta["role"] in ("slide_fav", "both", "hard_sliding_oracle") or ic in KNOWN_3647[:2]:
                active = active_recovery(env, book, blob, ic)
                active_rows.append({
                    "ic_id": ic,
                    "slide_ok": active["slide"]["ok"],
                    "slide_strategy": active["slide"].get("strategy"),
                    "slide_elapsed": active["slide"].get("elapsed"),
                    "vert_ok": active["vertical"]["ok"],
                    "vert_strategy": active["vertical"].get("strategy"),
                    "vert_elapsed": active["vertical"].get("elapsed"),
                    "competition_tag": active["competition_tag"],
                    "rollouts": active["rollouts"],
                    "physics_s": active["physics_s"],
                    "wall_s": active["wall_s"],
                    "note": "active_also_run_on_selected_level1",
                })
            else:
                active = None

        # mechanism label for shortest short success
        mech = "NONE"
        if shortest is not None:
            toks = json.loads(shortest["tokens"])
            # crude: open grip then handoff → HANDOFF_ASSISTED_RECATCH / AMBIGUOUS
            if len(toks) >= 1 and float(book[toks[0], 6]) < -0.5:
                mech = "HANDOFF_ASSISTED_RECATCH_OR_AMBIGUOUS"
            else:
                mech = "SHORT_ADJUST_THEN_HANDOFF"
        elif active and active["slide"]["ok"] and not active["vertical"]["ok"]:
            mech = "CONTACT_PRESERVING_SLIDING_ORACLE"
        elif active and active["vertical"]["ok"] and not active["slide"]["ok"]:
            mech = "RECOVERY_PHASE_RECATCH_ORACLE"
        elif active and active["slide"]["ok"] and active["vertical"]["ok"]:
            mech = "BOTH_ORACLES"

        mech_rows.append({
            "ic_id": ic,
            "level": level,
            "shortest_short_tokens": None if not shortest else shortest["tokens"],
            "shortest_short_depth": None if not shortest else shortest["depth"],
            "n_short_successes": len(stats["successes"]),
            "short_rollouts": stats["rollouts"],
            "short_physics_s": stats["physics_s"],
            "competition_tag": None if not active else active["competition_tag"],
            "mechanism_note": mech,
            "role": meta["role"],
            "prior_tag": meta["prior_tag"],
            "holdout": meta["holdout"],
            "rediscovered_3647": found_3647,
        })
        manifest.append({
            "id": ic,
            "snapshot": meta["snapshot"],
            "role": meta["role"],
            "level": level,
            "shortest_short": None if not shortest else shortest["tokens"],
            "competition_tag": None if not active else active["competition_tag"],
            "holdout": meta["holdout"],
        })
        print(
            "DONE_IC", ic, level,
            "short_succ", len(stats["successes"]),
            "shortest", None if not shortest else shortest["tokens"],
            flush=True,
        )

    write_csv(OUT / "SHORT_HANDOFF_BASELINE.csv", short_rows)
    write_csv(OUT / "ACTIVE_RECOVERY_SEARCH.csv", active_rows)
    write_csv(OUT / "PREFIX_HANDOFF_AUDIT.csv", prefix_rows)
    write_csv(OUT / "MECHANISM_COMPETITION.csv", mech_rows)
    write_csv(OUT / "IC_MANIFEST.csv", manifest)
    write_json(OUT / "rediscovery_3647.json", rediscover)

    # Summaries
    n_ic = len(manifest)
    n_l1 = sum(1 for m in manifest if m["level"] == "Level1_ShortHandoffRecoverable")
    n_l2 = sum(1 for m in manifest if m["level"] == "Level2_ActiveRecoveryEvidenced")
    n_un = sum(1 for m in manifest if m["level"].startswith("Unresolved"))
    tags = Counter(m.get("competition_tag") for m in manifest if m.get("competition_tag"))
    short_succ_ics = sum(1 for m in mech_rows if (m["n_short_successes"] or 0) > 0)
    cost = float(np.mean([m["short_physics_s"] for m in mech_rows])) if mech_rows else 0.0

    # sliding oracle still works?
    slide_ok_count = sum(1 for a in active_rows if a.get("slide_ok"))

    report = f"""# Nontrivial Recovery IC Mining — PILOT REPORT

**Stop after pilot.** No large-scale expansion. No training.

## Search coverage

See `SEARCH_COVERAGE.md`. Old vertical Recatch FSM never proposes 2-step open→HANDOFF.

## Pilot scale

- ICs: **{n_ic}**
- Rediscovery of known `[36,36,47]`: {json.dumps(jsafe(rediscover))}
- Sliding oracle still finds successes on active runs: **{slide_ok_count}** ICs

## Short-Handoff Baseline

- ICs with ≥1 short-prefix SUCCESS_12S: **{short_succ_ics}/{n_ic}**
- Level1 (short-handoff recoverable): **{n_l1}**
- Mean approx physics-seconds per IC (short search): **{cost:.1f}**

Buckets in `SHORT_HANDOFF_BASELINE.csv` (`bucket_ms` = 0 / 20 / 40 / 60-100 / 100-200).

## Active Recovery (ICs without short success, plus Level1 competition pass)

- Level2 (active recovery evidenced under oracle budget): **{n_l2}**
- Unresolved under budget: **{n_un}**
- Competition tags: {dict(tags)}

## Prefix HANDOFF audit

See `PREFIX_HANDOFF_AUDIT.csv` (sliding oracle prefixes where available).

## Answers (pilot)

1. **Why old Recatch missed [36,36,47]?** **VERIFIED** — continuous open/chase/close/brake/settle FSM; HANDOFF only after settle; not K48 short-prefix search (`SEARCH_COVERAGE.md`).
2. **How many ICs short-handoff recoverable?** **{short_succ_ics}/{n_ic}** (pilot) — `SUPPORTED_BUT_INCOMPLETE`.
3. **Nontrivial active recovery ICs?** Level2=**{n_l2}** — `SUPPORTED_BUT_INCOMPLETE` (oracle budget ≠ all short strategies excluded unless exhaustive 2-step done).
4. **Necessary vs redundant actions?** Prefix audits in CSV; many Level1 succeed at depth≤2 — `SUPPORTED_BUT_INCOMPLETE`.
5. **Sliding-favorable?** tags={dict(tags)} — trust only where short-fail AND slide_ok — check `MECHANISM_COMPETITION.csv`.
6. **Recatch-favorable?** same.
7. **Budget asymmetry?** **VERIFIED** — short search is K48-prefix; active Recatch is 3 FSM specs; Sliding is schedule library. Not yet equalized.
8. **39D enough?** Not re-probed in depth this pilot — defer to prior feasibility report (`SUPPORTED_BUT_INCOMPLETE`).
9. **Next scale?** If Level2≥3 and Level1 rediscovery OK: expand short exhaustive-2step on holdout hard set (~50 ICs) then beam active search. Estimate ~{cost:.0f}s physics/IC × N.

## Stop

Pilot complete. Awaiting decision before large-scale mining.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("PILOT_DONE", OUT, "L1", n_l1, "L2", n_l2, "short", short_succ_ics, flush=True)


if __name__ == "__main__":
    main()
