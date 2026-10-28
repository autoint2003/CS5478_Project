"""P12: audit B2/C2 Precision reality + scale D2 off-target generation.

No A2 campaign. Official dataset not modified. STRICT admit only.
Writes results/diagnostics/p12_precision_centric_expansion/
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "diagnostics" / "p12_precision_centric_expansion"
OFFICIAL = ROOT / "data" / "experts_v2_official"
P11_NEW = ROOT / "results" / "diagnostics" / "p11_targeted_ic_generation" / "NEW_VERIFIED_EXPERTS.csv"

# Bounded D2 scale (round-robin recipes × bases)
BUDGET_D2 = 36
SEED = 12
BASES_PER_RECIPE = 5

D2_RECIPES = [
    {"target_skill": "dz_inward", "dtheta_deg": 0.0, "dx_mm": 1.5, "dz_mm": -16.0, "want": "DZ"},
    {"target_skill": "dz_outward", "dtheta_deg": 0.0, "dx_mm": 1.5, "dz_mm": 16.0, "want": "DZ"},
    {"target_skill": "dx_pos", "dtheta_deg": 0.0, "dx_mm": 12.0, "dz_mm": 1.0, "want": "DX"},
    {"target_skill": "dx_neg", "dtheta_deg": 0.0, "dx_mm": -12.0, "dz_mm": 1.0, "want": "DX"},
    {"target_skill": "dx_then_dz", "dtheta_deg": 0.0, "dx_mm": 11.0, "dz_mm": -10.0, "want": "DX→DZ"},
    {"target_skill": "dz_then_dx", "dtheta_deg": 0.0, "dx_mm": 3.0, "dz_mm": -18.0, "want": "DZ→DX"},
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def _transition_from_skills(skills: list[str]) -> str:
    """Map coarse executed skills to coverage labels. dz_inward/outward stay DZ, not DX."""
    if not skills:
        return "empty"
    norm: list[str] = []
    for s in skills:
        sl = str(s).lower()
        if sl in ("dz", "dz_inward", "dz_outward"):
            if not norm or norm[-1] != "dz":
                norm.append("dz")
        elif sl in ("dx", "support_transfer"):
            # support_transfer is the DX support-transfer primitive
            if sl == "support_transfer":
                if not norm or norm[-1] != "dx":
                    norm.append("dx")
            else:
                if not norm or norm[-1] != "dx":
                    norm.append("dx")
        elif "dtheta" in sl:
            if not norm or norm[-1] != "dtheta":
                norm.append("dtheta")
        else:
            if not norm or norm[-1] != sl:
                norm.append(sl)
    return "→".join(norm) if norm else "empty"


def _load_heavy(branch_id: str, row: dict) -> dict | None:
    aid = row.get("attempt_id")
    sid = row.get("source_ic_id")
    candidates = []
    if aid:
        candidates.append(OFFICIAL / branch_id / f"{aid}.pkl")
    if sid:
        candidates.extend(sorted((OFFICIAL / branch_id).glob(f"*{sid}*.pkl")))
    for p in candidates:
        if p.is_file():
            try:
                return pickle.loads(p.read_bytes())
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Part 1 — Official audit (esp. B2/C2)
# ---------------------------------------------------------------------------
def audit_official() -> tuple[list[dict], dict]:
    from data_generation.expert_pipeline.branches import BRANCH_TO_CLASS
    from data_generation.expert_pipeline.live_stage1 import extract_precision_skills

    man = OFFICIAL / "manifest.jsonl"
    rows_out: list[dict] = []
    summary: dict[str, Any] = {
        "by_class": Counter(),
        "skill_sequence_manifest": Counter(),
        "b2_executed_vs_skip": Counter(),
        "c2_executed_vs_skip": Counter(),
        "d2_transitions": Counter(),
        "recovery_direct": Counter(),
        "precision_real_transitions": Counter(),
    }
    if not man.is_file():
        return rows_out, summary

    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bid = r.get("branch_id") or ""
        cls = BRANCH_TO_CLASS.get(bid, bid)
        summary["by_class"][cls] += 1
        sk_man = r.get("skill_sequence") or "unknown"
        summary["skill_sequence_manifest"][sk_man] += 1

        heavy = _load_heavy(bid, r) if cls in ("B2", "C2", "D2", "A2") else None
        phases = list((heavy or {}).get("phases") or [])
        skipped = bool((heavy or {}).get("precision_skipped"))
        if heavy is None and sk_man == "precision_skipped_direct_handoff":
            skipped = True
        n_act = (heavy or {}).get("n_actions")
        if n_act is None:
            n_act = r.get("n_actions")
        pl = (heavy or {}).get("primitive_logs") or {}
        fsm_exec = pl.get("fsm_executed_skills")
        skills = extract_precision_skills(phases) if phases else []
        if not skills and fsm_exec:
            skills = extract_precision_skills([str(x) for x in fsm_exec])
        transition = (
            "precision_skipped"
            if skipped or sk_man == "precision_skipped_direct_handoff"
            else (_transition_from_skills(skills) if skills else sk_man)
        )
        executed_precision = bool(
            cls.endswith("2")
            and not skipped
            and sk_man != "precision_skipped_direct_handoff"
            and (int(n_act or 0) > 0 or bool(skills))
        )

        row = {
            "class_label": cls,
            "branch_id": bid,
            "source_ic_id": r.get("source_ic_id"),
            "attempt_id": r.get("attempt_id"),
            "manifest_skill_sequence": sk_man,
            "precision_skipped": skipped,
            "n_actions": n_act,
            "phases_head": "|".join(phases[:12]),
            "skills_from_phases": "|".join(skills),
            "fsm_executed": "|".join(str(x) for x in (fsm_exec or [])),
            "transition_normalized": transition,
            "executed_precision": executed_precision,
            "SUCCESS_12S": r.get("SUCCESS_12S"),
            "SUCCESS_NEW": r.get("SUCCESS_NEW") if "SUCCESS_NEW" in r else True,
        }
        rows_out.append(row)

        if cls in ("A1", "B1", "C1"):
            summary["recovery_direct"][cls] += 1
        if cls == "B2":
            summary["b2_executed_vs_skip"]["executed" if executed_precision else "skipped_or_empty"] += 1
        if cls == "C2":
            summary["c2_executed_vs_skip"]["executed" if executed_precision else "skipped_or_empty"] += 1
        if cls == "D2":
            summary["d2_transitions"][transition] += 1
        if executed_precision:
            summary["precision_real_transitions"][transition] += 1

    # Fold P11 diagnostic D2 (independent_new) into coverage view (not official)
    p11_extra = []
    if P11_NEW.is_file() and P11_NEW.stat().st_size:
        for r in csv.DictReader(P11_NEW.open(encoding="utf-8")):
            if r.get("class_label") == "D2" and str(r.get("independent_new")).lower() in ("true", "1"):
                p11_extra.append(r)
                summary.setdefault("p11_d2_diagnostic", Counter())[r.get("skill_sequence") or "unknown"] += 1
    summary["p11_d2_n"] = len(p11_extra)
    return rows_out, summary


# ---------------------------------------------------------------------------
# Part 2 — Scale D2 (round-robin)
# ---------------------------------------------------------------------------
def scale_d2(env) -> tuple[list[dict], list[dict], list[dict]]:
    from data_generation.expert_pipeline.live_stage1 import (
        extract_precision_skills,
        generate_d2_from_synthetic_ic,
    )
    from data_generation.oracles.precision.d2_eligibility import check_d2_eligible
    from data_generation.oracles.precision.fsm import classify_case
    from data_generation.oracles.precision.residuals import measure_errors
    from data_generation.oracles.precision.state import PrecisionConfig
    from envs.simulation.capture import restore_capture
    from experiments.precision.precision_required_pilot_v2.d2_synthetic import construct_d2_ic
    from experiments.precision.precision_required_pilot_v2.run_pilot import discover_d2_bases

    rng = np.random.default_rng(SEED)
    bases = discover_d2_bases(rng)
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    funnel: list[dict] = []
    verified: list[dict] = []
    skills: list[dict] = []

    # Round-robin: zip recipes with rotating bases
    pairs: list[tuple[dict, dict]] = []
    for i, recipe in enumerate(D2_RECIPES):
        # offset base list per recipe for diversity
        rotated = bases[i:] + bases[:i]
        for b in rotated[:BASES_PER_RECIPE]:
            pairs.append((recipe, b))
    pairs = pairs[:BUDGET_D2]

    print(f"[D2 scale] pairs={len(pairs)} recipes={len(D2_RECIPES)} bases_avail={len(bases)}", flush=True)
    for n, (recipe, base) in enumerate(pairs, 1):
        skill = recipe["target_skill"]
        base_id = base["sample_id"]
        sid = f"P12_D2__{skill}__{base_id[:36]}__{n}"
        t0 = time.time()
        print(f"  [{n}/{len(pairs)}] want={recipe['want']} {skill} base={base_id[:48]}", flush=True)
        try:
            raw = pickle.loads(base["path"].read_bytes())
            base_blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": base_id}
            built = construct_d2_ic(
                env,
                base_blob,
                target_skill=skill,
                dtheta_deg=recipe["dtheta_deg"],
                dx_mm=recipe["dx_mm"],
                dz_mm=recipe["dz_mm"],
                settle_n=14,
            )
            row_f: dict[str, Any] = {
                "family": "D2_p12_scale",
                "sample_id": sid,
                "base_id": base_id,
                "target_skill": skill,
                "want_coverage": recipe["want"],
                "requested_dx_mm": recipe["dx_mm"],
                "requested_dz_mm": recipe["dz_mm"],
                "construction_ok": bool(built.get("ok")),
                "elapsed_s": round(time.time() - t0, 2),
            }
            if not built.get("ok"):
                row_f["reject"] = f"CONSTRUCT:{built.get('reason')}"
                funnel.append(row_f)
                print(f"    construct fail {built.get('reason')}", flush=True)
                continue
            ic = {"capture": built["capture"], "ref": built["ref"], "sample_id": sid}
            elig = check_d2_eligible(env, ic, cfg=cfg)
            row_f["SUCCESS_12S"] = bool(elig.SUCCESS_12S)
            row_f["PRECISION_REQUIRED"] = elig.precision_required
            row_f["d2_status"] = elig.status
            if not elig.eligible:
                row_f["reject"] = f"D2_GATE:{elig.status}:{elig.reason}"
                funnel.append(row_f)
                print(f"    gate {elig.status}", flush=True)
                continue
            restore_capture(env, ic["capture"])
            err = measure_errors(env.sim)
            plan = classify_case(err, cfg)
            row_f["fsm_case"] = plan.case
            row_f["entry_dx_mm"] = err.get("dx_mm")
            row_f["entry_dz_mm"] = err.get("dz_mm")
            live = generate_d2_from_synthetic_ic(
                env,
                ic,
                raw_scenario_id=sid,
                construction_meta={
                    **(built.get("construction") or {}),
                    "p12": True,
                    "want_coverage": recipe["want"],
                },
                precision_config=cfg,
            )
            sk = extract_precision_skills(list(live.precision_phases or []))
            trans = _transition_from_skills(sk)
            row_f.update(
                {
                    "PRECISION_EXECUTED": bool(live.n_precision_actions),
                    "SUCCESS_NEW_STRICT": bool(live.SUCCESS_NEW),
                    "precision_status": live.precision_status,
                    "skills": "|".join(sk),
                    "transition_normalized": trans,
                    "accepted": bool(live.accepted),
                    "planned_fsm": plan.case + ":" + "|".join(plan.planned_skills),
                }
            )
            funnel.append(row_f)
            skills.append(
                {
                    "family": "D2_p12",
                    "sample_id": sid,
                    "want": recipe["want"],
                    "fsm_case": plan.case,
                    "planned": "|".join(plan.planned_skills),
                    "executed": "|".join(sk),
                    "transition_normalized": trans,
                    "accepted": live.accepted,
                }
            )
            if live.accepted:
                verified.append(
                    {
                        "class_label": "D2",
                        "sample_id": sid,
                        "parent_lineage": base_id,
                        "skill_sequence": trans,
                        "skills_raw": "|".join(sk),
                        "SUCCESS_NEW": True,
                        "SUCCESS_12S": live.SUCCESS_12S_final,
                        "n_precision_actions": live.n_precision_actions,
                        "admit_success_key": "SUCCESS_NEW",
                        "source_family": "D2_p12_scale",
                        "synthetic": True,
                        "independent_new": True,
                        "fsm_case": plan.case,
                        "want_coverage": recipe["want"],
                        "capture_fp": live.capture_fp,
                    }
                )
                print(f"    ACCEPTED trans={trans} case={plan.case}", flush=True)
            else:
                print(f"    fail {live.reject_reason or live.precision_status}", flush=True)
        except Exception as exc:
            funnel.append(
                {
                    "family": "D2_p12_scale",
                    "sample_id": sid,
                    "reject": f"EXCEPTION:{type(exc).__name__}:{exc}",
                }
            )
            print(f"    EXC {exc}", flush=True)
    return funnel, verified, skills


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_reports(
    audit_rows: list[dict],
    summary: dict,
    funnel: list[dict],
    verified: list[dict],
    skills: list[dict],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT / "OFFICIAL_PRECISION_AUDIT.csv", audit_rows)
    _write_csv(OUT / "D2_SCALE_FUNNEL.csv", funnel)
    _write_csv(OUT / "NEW_VERIFIED_D2.csv", verified)
    _write_csv(OUT / "SKILL_EXECUTION_COVERAGE.csv", skills)

    b2_exec = summary["b2_executed_vs_skip"].get("executed", 0)
    b2_skip = summary["b2_executed_vs_skip"].get("skipped_or_empty", 0)
    c2_exec = summary["c2_executed_vs_skip"].get("executed", 0)
    c2_skip = summary["c2_executed_vs_skip"].get("skipped_or_empty", 0)
    real_prec = dict(summary["precision_real_transitions"])
    d2_off = dict(summary["d2_transitions"])
    p11_n = summary.get("p11_d2_n", 0)

    d2_12 = sum(1 for r in funnel if r.get("SUCCESS_12S") in (True, "True"))
    d2_need = sum(1 for r in funnel if r.get("PRECISION_REQUIRED") in (True, "True"))
    d2_ok = sum(1 for r in funnel if r.get("accepted") in (True, "True"))
    by_want = Counter(r.get("want_coverage") for r in verified)
    by_trans = Counter(r.get("skill_sequence") for r in verified)

    # Combined D2 coverage (official + P11 diag + P12)
    combined_d2_trans = Counter(d2_off)
    if P11_NEW.is_file() and P11_NEW.stat().st_size:
        for r in csv.DictReader(P11_NEW.open(encoding="utf-8")):
            if r.get("class_label") == "D2" and str(r.get("independent_new")).lower() in ("true", "1"):
                combined_d2_trans[r.get("skill_sequence") or "unknown"] += 1
    for t, n in by_trans.items():
        combined_d2_trans[t] += n

    gaps = []
    if b2_exec < 10:
        gaps.append(f"B2 real Precision executions only {b2_exec}/{b2_exec+b2_skip} (mostly skip)")
    if c2_exec < 8:
        gaps.append(f"C2 real Precision executions only {c2_exec}/{c2_exec+c2_skip}")
    if combined_d2_trans.get("dz", 0) + sum(1 for k in combined_d2_trans if str(k).startswith("dz→")) < 5:
        gaps.append("DZ-first / DZ-only D2 coverage still thin vs DX→DZ")
    if not any("dz→dx" in str(k) for k in combined_d2_trans):
        # check normalized
        if combined_d2_trans.get("dz→dx", 0) == 0 and not any(
            str(k).startswith("dz→dx") for k in combined_d2_trans
        ):
            gaps.append("No accepted dz→dx normalized transitions yet in combined pool")
    # refine dz→dx check
    has_dz_dx = any(
        "dz" in str(k).split("→")[0:1]
        and "dx" in str(k)
        and str(k).find("dz") < str(k).find("dx")
        for k in combined_d2_trans
        if "→" in str(k)
    )
    if not has_dz_dx:
        gaps.append("Verified DZ→DX still missing or unlabeled in combined D2/B2 executed set")
    if summary["by_class"].get("D2", 0) + p11_n + len(verified) < 20:
        gaps.append(
            f"Independent D2 Precision pool still <20 "
            f"(official {summary['by_class'].get('D2', 0)} + P11 {p11_n} + P12 {len(verified)})"
        )

    _write_md(
        OUT / "B2_C2_PRECISION_AUDIT.md",
        f"""# B2 / C2 Precision Audit (P12)

## Finding
Branch labels `*_PRECISION` do **not** imply executed Precision skills.

| Branch | Real Precision executed | Skipped / empty |
|---|---:|---:|
| B2 | {b2_exec} | {b2_skip} |
| C2 | {c2_exec} | {c2_skip} |

Most B2/C2 experts are `precision_skipped_direct_handoff`: Stage-1 recovery succeeded and the Oracle skipped corrections because the exit was already near handoff-ready.

## Expansion policy
- **Do not** expand B2/C2 merely to raise class counts.
- Expand only when Stage-1 exits are 12S-stable **and** `PRECISION_REQUIRED`, then record actual executed skills.
- Preserve genuine B/C recovery + lift/handoff in those tapes.

See `OFFICIAL_PRECISION_AUDIT.csv`.
""",
    )

    _write_md(
        OUT / "D2_SCALE_RESULTS.md",
        f"""# D2 Scale Results (P12)

Scaled P11 stable off-target procedure with **round-robin** recipes (DX / DZ / DX→DZ / DZ→DX intents).

## Funnel
- Attempts: {len(funnel)}
- Construction→eligible 12S+need: ~{d2_need} (12S flag rows {d2_12})
- STRICT accepted: **{d2_ok}**

## Accepted by intended coverage
{json.dumps(dict(by_want), indent=2)}

## Accepted transitions (normalized)
{json.dumps(dict(by_trans), indent=2)}

Normalization: `dz_inward`/`dz_outward` → **dz** (not dx); `support_transfer` → **dx**.

Official dataset not modified; candidates in `NEW_VERIFIED_D2.csv`.
""",
    )

    ready = (
        b2_exec + c2_exec >= 8
        and (summary["by_class"].get("D2", 0) + p11_n + len(verified)) >= 12
        and len(gaps) <= 2
    )

    _write_md(
        OUT / "TRAINING_READINESS_SKILL_GAPS.md",
        f"""# Training Readiness — Skill Gaps (P12)

**Seven-class quotas abandoned.** Evaluation is skill- and sequence-centric for SFT → GRPO.

## Coverage snapshot

### Recovery (A/B/C direct)
| Class | n official |
|---|---:|
| A1 | {summary['by_class'].get('A1', 0)} |
| B1 | {summary['by_class'].get('B1', 0)} |
| C1 | {summary['by_class'].get('C1', 0)} |

Keep as verified recovery demos. No forced balance.

### Precision reality
| Pool | Real executed Precision | Notes |
|---|---:|---|
| Official B2 | {b2_exec} | {b2_skip} skipped |
| Official C2 | {c2_exec} | {c2_skip} skipped |
| Official D2 | {summary['by_class'].get('D2', 0)} | transitions {dict(d2_off)} |
| P11 diagnostic D2 | {p11_n} | independent_new |
| P12 new D2 | {len(verified)} | this pilot |

### Official real Precision transitions
{json.dumps(real_prec, indent=2)}

### Combined D2 transitions (official + P11 + P12)
{json.dumps(dict(combined_d2_trans), indent=2)}

## Remaining **training-relevant** gaps
"""
        + ("\n".join(f"- {g}" for g in gaps) if gaps else "- None critical identified in this pass.")
        + f"""

## A2
Optional only — official n={summary['by_class'].get('A2', 0)}. **No generation campaign.**

## Verdict
{"**Approaching** skill-diverse SFT readiness for Precision/GRPO if D2 continues to scale and B2/C2 expansion targets Precision-required exits only." if ready else "**Not yet SFT-ready** for multi-skill Precision — close gaps above first."}

STRICT admission, dual θ metrics, physics, M111 unchanged.
""",
    )

    _write_md(
        OUT / "UPDATED_DATASET_COVERAGE.md",
        f"""# Updated Dataset Coverage (P12 — Precision-centric)

Generated: {_utc()}

## Policy change vs P9–P11
- **Stop A2 campaigns** (optional natural accepts only)
- **Primary:** scale D2 stable off-target Precision
- **Secondary:** B2/C2 only where Precision is actually required/executed
- **Keep** A1/B1/C1 recovery experts as-is
- Success metric = skill diversity + physical validity, **not** 7×20 class quotas

## Official class counts (reference only)
{json.dumps(dict(summary['by_class']), indent=2)}

## Manifest skill_sequence (official)
{json.dumps(dict(summary['skill_sequence_manifest']), indent=2)}

## P12 D2 scale
- Attempts {len(funnel)}, STRICT accepts {len(verified)}
- See `D2_SCALE_RESULTS.md`, `NEW_VERIFIED_D2.csv`

## Skill gaps
See `TRAINING_READINESS_SKILL_GAPS.md`.
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P12 Precision-Centric Expansion

## Done
1. Audited official B2/C2: **{b2_exec}/{b2_exec+b2_skip}** B2 and **{c2_exec}/{c2_exec+c2_skip}** C2 have real Precision execution (rest skipped).
2. Scaled D2 off-target generation (round-robin): **{len(verified)}** new STRICT accepts (diagnostic).
3. Replaced class-quota readiness with skill-gap report.
4. **No A2 campaign.**

## Training-relevant gaps
"""
        + ("\n".join(f"- {g}" for g in gaps) if gaps else "- None listed.")
        + f"""

## Scale next
1. Continue D2 round-robin until ~20+ independent STRICT D2 with DX-only / DZ-only / DX→DZ / DZ→DX represented
2. Mine B/C Stage-1 exits that are Precision-required (not skip-majority branches)
3. Do not spend compute on A2 or HARD_IC

## Integrity
Official `experts_v2_official` untouched. STRICT admit. Physics/M111 frozen. No hard-IC tuning.
""",
    )

    (OUT / "raw_p12_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "official_by_class": dict(summary["by_class"]),
                "b2_executed": b2_exec,
                "b2_skipped": b2_skip,
                "c2_executed": c2_exec,
                "c2_skipped": c2_skip,
                "p12_d2_accepted": len(verified),
                "p12_d2_attempts": len(funnel),
                "p11_d2_diagnostic": p11_n,
                "gaps": gaps,
                "a2_campaign": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== P12 audit official B2/C2/D2 ===", flush=True)
    audit_rows, summary = audit_official()
    print(
        "B2",
        dict(summary["b2_executed_vs_skip"]),
        "C2",
        dict(summary["c2_executed_vs_skip"]),
        "D2",
        dict(summary["d2_transitions"]),
        flush=True,
    )

    print("=== P12 scale D2 ===", flush=True)
    env = make_env()
    funnel, verified, skills = scale_d2(env)

    print("=== Writing reports ===", flush=True)
    write_reports(audit_rows, summary, funnel, verified, skills)
    print(f"Done → {OUT} new_D2={len(verified)}", flush=True)


if __name__ == "__main__":
    main()
