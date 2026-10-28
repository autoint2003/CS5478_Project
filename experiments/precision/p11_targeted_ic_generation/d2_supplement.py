"""Extra D2 recipes beyond the first dz_inward-heavy budget burn."""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.live_stage1 import (  # noqa: E402
    extract_precision_skills,
    generate_d2_from_synthetic_ic,
)
from data_generation.oracles.precision.d2_eligibility import check_d2_eligible  # noqa: E402
from data_generation.oracles.precision.fsm import classify_case  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from experiments.precision.p11_targeted_ic_generation.run_p11 import (  # noqa: E402
    D2_RECIPES,
    OUT,
    make_env,
)
from experiments.precision.precision_required_pilot_v2.d2_synthetic import construct_d2_ic  # noqa: E402
from experiments.precision.precision_required_pilot_v2.run_pilot import discover_d2_bases  # noqa: E402


def _append(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = OUT / name
    old: list[dict] = []
    if path.is_file() and path.stat().st_size:
        old = list(csv.DictReader(path.open(encoding="utf-8")))
    keys: list[str] = []
    seen: set[str] = set()
    for r in old + rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in old + rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> None:
    env = make_env()
    rng = np.random.default_rng(99)
    bases = discover_d2_bases(rng)[:3]
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    recipes = [r for r in D2_RECIPES if r["target_skill"] != "dz_inward"]
    funnel: list[dict] = []
    verified: list[dict] = []
    skills: list[dict] = []
    n = 0
    for recipe in recipes:
        for base in bases[:2]:
            n += 1
            skill = recipe["target_skill"]
            base_id = base["sample_id"]
            sid = f"D2_SUP__{skill}__{base_id[:36]}__{n}"
            print(f"[{n}] {skill} {base_id[:48]}", flush=True)
            try:
                raw = pickle.loads(base["path"].read_bytes())
                base_blob = {
                    "capture": raw["capture"],
                    "ref": raw["ref"],
                    "sample_id": base_id,
                }
                built = construct_d2_ic(
                    env,
                    base_blob,
                    target_skill=skill,
                    dtheta_deg=recipe["dtheta_deg"],
                    dx_mm=recipe["dx_mm"],
                    dz_mm=recipe["dz_mm"],
                    settle_n=14,
                )
                if not built.get("ok"):
                    print("  construct fail", built.get("reason"), flush=True)
                    funnel.append(
                        {
                            "family": "D2_supplement",
                            "sample_id": sid,
                            "target_skill": skill,
                            "construction_ok": False,
                            "reject": built.get("reason"),
                        }
                    )
                    continue
                ic = {"capture": built["capture"], "ref": built["ref"], "sample_id": sid}
                elig = check_d2_eligible(env, ic, cfg=cfg)
                if not elig.eligible:
                    print("  gate", elig.status, flush=True)
                    funnel.append(
                        {
                            "family": "D2_supplement",
                            "sample_id": sid,
                            "target_skill": skill,
                            "SUCCESS_12S": elig.SUCCESS_12S,
                            "PRECISION_REQUIRED": elig.precision_required,
                            "reject": f"D2_GATE:{elig.status}",
                        }
                    )
                    continue
                restore_capture(env, ic["capture"])
                err = measure_errors(env.sim)
                plan = classify_case(err, cfg)
                live = generate_d2_from_synthetic_ic(
                    env,
                    ic,
                    raw_scenario_id=sid,
                    construction_meta=built.get("construction") or {},
                    precision_config=cfg,
                )
                sk = extract_precision_skills(list(live.precision_phases or []))
                print(
                    f"  {live.precision_status} case={plan.case} skills={sk} ok={live.accepted}",
                    flush=True,
                )
                funnel.append(
                    {
                        "family": "D2_supplement",
                        "sample_id": sid,
                        "target_skill": skill,
                        "SUCCESS_12S": True,
                        "PRECISION_REQUIRED": True,
                        "SUCCESS_NEW_STRICT": bool(live.SUCCESS_NEW),
                        "accepted": bool(live.accepted),
                        "skills": "|".join(sk),
                        "fsm_case": plan.case,
                    }
                )
                if live.accepted:
                    verified.append(
                        {
                            "class_label": "D2",
                            "sample_id": sid,
                            "parent_lineage": base_id,
                            "skill_sequence": "→".join(sk),
                            "SUCCESS_NEW": True,
                            "source_family": "D2_supplement",
                            "synthetic": True,
                            "fsm_case": plan.case,
                            "admit_success_key": "SUCCESS_NEW",
                            "independent_new": True,
                        }
                    )
                    skills.append(
                        {
                            "family": "D2",
                            "sample_id": sid,
                            "executed": "|".join(sk),
                            "accepted": True,
                            "transition": "→".join(sk),
                            "fsm_case": plan.case,
                        }
                    )
            except Exception as exc:
                print(f"  EXC {exc}", flush=True)
                funnel.append(
                    {
                        "family": "D2_supplement",
                        "sample_id": sid,
                        "reject": f"EXCEPTION:{exc}",
                    }
                )
    _append("IC_SCREENING_FUNNEL.csv", funnel)
    _append("NEW_VERIFIED_EXPERTS.csv", verified)
    _append("SKILL_EXECUTION_COVERAGE.csv", skills)
    print(f"supplement funnel={len(funnel)} verified={len(verified)}", flush=True)


if __name__ == "__main__":
    main()
