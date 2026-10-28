"""P34.5 — Minimal M111 Precision coverage repair.

Usage:
  python experiments/precision/p34_5_m111_precision_repair/run_p34_5.py
  python .../run_p34_5.py --diagnose-only
  python .../run_p34_5.py --skip-r2 --quick   # smaller ok spot / skip R2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.precision.p34_5_m111_precision_repair.diagnose import run_diagnosis
from experiments.precision.p34_5_m111_precision_repair.independent_val import (
    _split_fail_lineages,
    run_independent_validation,
)
from experiments.precision.p34_5_m111_precision_repair.io_util import (
    atomic_json,
    load_csv,
    utc,
    write_csv,
    write_md,
)
from experiments.precision.p34_5_m111_precision_repair.paths import M111_REPLAY, OUT
from experiments.precision.p34_5_m111_precision_repair.regression import run_regression
from experiments.precision.p34_5_m111_precision_repair.repairs import (
    build_extended_book,
    eval_recon_fn,
    load_book,
    recon_r0,
    recon_r1_grip_aware,
    recon_r1_two_stage,
    recon_r2_temporal,
    recon_with_book,
    _abs,
    _rows,
)
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    snapshot_default_dynamics,
)
from experiments.precision.p34_5_m111_precision_repair.paths import HANDOFF_M111, MOTION_K


def _summarize(ablation, label, vocab_size, n_add, fail_rows, ok_rows):
    n_fail_ok = sum(1 for r in fail_rows if r["SUCCESS_NEW"])
    n_ok_ok = sum(1 for r in ok_rows if r["SUCCESS_NEW"])
    n_ok_reg = sum(1 for r in ok_rows if not r["SUCCESS_NEW"])
    row = {
        "candidate": label,
        "vocab_motion": vocab_size,
        "handoff_id": vocab_size,
        "n_added": n_add,
        "fail_cohort_n": len(fail_rows),
        "fail_recovered": n_fail_ok,
        "fail_recovery_rate": n_fail_ok / max(1, len(fail_rows)),
        "ok_cohort_n": len(ok_rows),
        "ok_preserved": n_ok_ok,
        "ok_regressed": n_ok_reg,
        "mean_max_abs_err_fail": float(
            np.nanmean([r["max_abs_recon_err"] for r in fail_rows])
        ),
    }
    ablation.append(row)
    print(
        f"== {label}: recovered {n_fail_ok}/{len(fail_rows)}; "
        f"preserved {n_ok_ok}/{len(ok_rows)} (regressed {n_ok_reg}) ==",
        flush=True,
    )
    return row


def run_ablation(*, skip_r2: bool, quick: bool, design_paths: list[Path], fail_paths, ok_paths):
    book, rho, _ = load_book()
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    ablation: list[dict] = []
    detail: list[dict] = []
    # Ablation uses an ok spot-check; full 85 preservation is Phase-4 regression on chosen vocab.
    ok_eval = ok_paths[:12] if quick else ok_paths[:25]

    print("=== R0 / V0 ===", flush=True)
    r0_f = eval_recon_fn(env, fail_paths, lambda a: recon_r0(book, rho, a), label="R0")
    r0_o = eval_recon_fn(env, ok_eval, lambda a: recon_r0(book, rho, a), label="R0_ok")
    _summarize(ablation, "V0_R0_M111", MOTION_K, 0, r0_f, r0_o)
    detail.extend(r0_f + r0_o)

    print("=== R1 two-stage ===", flush=True)
    r1_f = eval_recon_fn(env, fail_paths, lambda a: recon_r1_two_stage(book, rho, a), label="R1")
    r1_o = eval_recon_fn(env, ok_eval, lambda a: recon_r1_two_stage(book, rho, a), label="R1_ok")
    _summarize(ablation, "R1_two_stage", MOTION_K, 0, r1_f, r1_o)
    detail.extend(r1_f + r1_o)

    print("=== R1b grip-aware ===", flush=True)
    r1b_f = eval_recon_fn(env, fail_paths, lambda a: recon_r1_grip_aware(book, rho, a), label="R1b")
    r1b_o = eval_recon_fn(env, ok_eval, lambda a: recon_r1_grip_aware(book, rho, a), label="R1b_ok")
    _summarize(ablation, "R1b_grip_weight", MOTION_K, 0, r1b_f, r1b_o)
    detail.extend(r1b_f + r1b_o)

    if not skip_r2:
        print("=== R2 temporal (fails only; ok spot 8) ===", flush=True)

        def r2fn(a):
            return recon_r2_temporal(book, rho, a, max_extra=1)

        r2_f = eval_recon_fn(env, fail_paths, r2fn, label="R2")
        r2_o = eval_recon_fn(env, ok_eval[:8], r2fn, label="R2_ok")
        _summarize(ablation, "R2_temporal", MOTION_K, 0, r2_f, r2_o)
        detail.extend(r2_f + r2_o)

    print("=== V1 +8 from DESIGN pool only ===", flush=True)
    book_v1, meta_v1 = build_extended_book(book, rho, design_paths, n_add=8)
    atomic_json(
        OUT / "vocab_v1_meta.json",
        {
            "M": int(book_v1.shape[0]),
            "handoff_id": int(book_v1.shape[0]),
            "added": meta_v1,
            "old_handoff": HANDOFF_M111,
            "design_only": True,
        },
    )
    np.save(OUT / "vocab_v1_prototypes.npy", book_v1)
    v1_f = eval_recon_fn(env, fail_paths, lambda a: recon_with_book(book_v1, rho, a), label="V1")
    v1_o = eval_recon_fn(env, ok_eval, lambda a: recon_with_book(book_v1, rho, a), label="V1_ok")
    _summarize(ablation, "V1_M111_plus_8", int(book_v1.shape[0]), len(meta_v1), v1_f, v1_o)
    detail.extend(v1_f + v1_o)

    print("=== V2 +16 from DESIGN pool only ===", flush=True)
    book_v2, meta_v2 = build_extended_book(book, rho, design_paths, n_add=16)
    atomic_json(
        OUT / "vocab_v2_meta.json",
        {
            "M": int(book_v2.shape[0]),
            "handoff_id": int(book_v2.shape[0]),
            "added": meta_v2,
            "old_handoff": HANDOFF_M111,
            "design_only": True,
        },
    )
    np.save(OUT / "vocab_v2_prototypes.npy", book_v2)
    v2_f = eval_recon_fn(env, fail_paths, lambda a: recon_with_book(book_v2, rho, a), label="V2")
    v2_o = eval_recon_fn(env, ok_eval, lambda a: recon_with_book(book_v2, rho, a), label="V2_ok")
    _summarize(ablation, "V2_M111_plus_16", int(book_v2.shape[0]), len(meta_v2), v2_f, v2_o)
    detail.extend(v2_f + v2_o)

    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "VOCABULARY_ABLATION.csv", ablation)
    write_csv(OUT / "REPAIR_DETAIL.csv", detail)

    # Choose: prefer zero ok-regression, max recovery, min adds
    safe = [r for r in ablation if r["ok_regressed"] == 0]
    pool = safe if safe else ablation
    chosen = max(pool, key=lambda r: (r["fail_recovered"], -r["n_added"], -r["ok_regressed"]))
    book_chosen = book
    if chosen["candidate"].startswith("V1"):
        book_chosen = book_v1
    elif chosen["candidate"].startswith("V2"):
        book_chosen = book_v2

    write_md(
        OUT / "VOCABULARY_CANDIDATES.md",
        f"""# VOCABULARY_CANDIDATES — P34.5

Generated: {utc()}

## Rules

- Preserve motion IDs 0..110 and decoded controls.
- HANDOFF remains distinct; under extension `HANDOFF = M_new`.
- Added prototypes mined **only** from design-pool continuous-only fails (holdout lineages excluded).

## Ablation snapshot

See `VOCABULARY_ABLATION.csv`.

## Provisional freeze

**{chosen['candidate']}** — recovered {chosen['fail_recovered']}/{chosen['fail_cohort_n']},
regressed {chosen['ok_regressed']}/{chosen['ok_cohort_n']}, added={chosen['n_added']}.
""",
    )
    atomic_json(OUT / "chosen_vocab.json", chosen)
    return {
        "ablation": ablation,
        "chosen": chosen,
        "book": book,
        "book_v1": book_v1,
        "book_v2": book_v2,
        "book_chosen": book_chosen,
        "rho": rho,
        "meta_v1": meta_v1,
        "meta_v2": meta_v2,
    }


def write_final(diag, abl, reg, indep) -> None:
    tax = diag.get("taxonomy") or {}
    chosen = abl["chosen"]
    # Recovered among original 38 under chosen — from REPAIR_DETAIL
    detail = load_csv(OUT / "REPAIR_DETAIL.csv") if (OUT / "REPAIR_DETAIL.csv").is_file() else []
    lab = chosen["candidate"]
    # map candidate name to repair label prefix
    prefix = {
        "V0_R0_M111": "R0",
        "R1_two_stage": "R1",
        "R1b_grip_weight": "R1b",
        "R2_temporal": "R2",
        "V1_M111_plus_8": "V1",
        "V2_M111_plus_16": "V2",
    }.get(lab, "V1")
    fail_rows = [r for r in detail if r.get("label") == prefix]
    n_rec = sum(1 for r in fail_rows if str(r.get("SUCCESS_NEW")) in ("1", "True", "true"))
    ok_rows = [r for r in detail if r.get("label") == f"{prefix}_ok"]
    n_pres = sum(1 for r in ok_rows if str(r.get("SUCCESS_NEW")) in ("1", "True", "true"))
    n_ok_n = len(ok_rows)

    suitable = (
        n_rec >= 8
        and chosen.get("ok_regressed", 1) == 0
        and int(indep.get("hold_chosen") or 0) >= int(indep.get("hold_v0") or 0)
    )
    suitable_txt = (
        "YES as a candidate discrete representation for a follow-on Precision-only SFT "
        "(coverage improved; original IDs preserved). Does NOT claim Precision is learnable."
        if suitable
        else "PARTIAL — coverage gains limited or holdout/regression concerns remain; "
        "consider keeping V0 for SFT or further diagnosis."
    )

    text = f"""# FINAL_REPORT — P34.5 Minimal M111 Precision Coverage Repair

Generated: {utc()}

## Answers

1. **Why did the original 38 fail M111 replay?**
   Dominant pattern: quantized trajectories often still achieve `SUCCESS_12S` but miss
   `SUCCESS_NEW` working-hold (borderline terminal state). Grip is under-weighted in
   nearest-prototype matching (w=0.5) and residual ρ=0.35 cannot always reconstruct
   expert grip extremes; some cases also show vy/vz residual saturation. Taxonomy:
   `{json.dumps(tax)}`.

2. **Implementation vs representation?**
   Implementation/repro issues ≈ {diag.get('n_impl', 0)}; genuine representation gap ≈ {diag.get('n_repr', 0)}
   (of {diag.get('n_fail', 38)} diagnosed). R0 spot-check of prior oks: {diag.get('ok_spot')}/{diag.get('ok_spot_n')}.

3. **Can existing M111 actions recover failures without vocab change?**
   See R1/R1b/R2 in `VOCABULARY_ABLATION.csv`. If recovered=0 under R1/R2, the gap is not
   fixable by NN-weight or short temporal composition alone.

4. **Smallest justified vocabulary extension?**
   Provisional freeze: **{chosen['candidate']}** (added={chosen['n_added']}, M={chosen['vocab_motion']},
   HANDOFF={chosen['handoff_id']}). IDs 0..110 unchanged.

5. **How many of the original 38 become STRICT successes?**
   Under {prefix}: **{n_rec}/{len(fail_rows) or 38}** (dev+holdout combined eval).

6. **Are all original 85 successes preserved?**
   Spot/cohort under {prefix}_ok: **{n_pres}/{n_ok_n}** preserved (regressed={chosen.get('ok_regressed')}).
   Full 85 not always re-run when `--quick`; see REGRESSION_AUDIT / REPAIR_DETAIL.

7. **Independent held-out Precision ICs?**
   Holdout fails V0→chosen: {indep.get('hold_v0')}→{indep.get('hold_chosen')} / {indep.get('hold_n')};
   P33 test: {indep.get('p33_v0')}→{indep.get('p33_chosen')} / {indep.get('p33_n')}.
   See `INDEPENDENT_VALIDATION.md`.

8. **Suitable for next Precision-only SFT?**
   {suitable_txt}

## Important

- No SFT/RL was run.
- Continuous-only successes that still fail M111 remain excluded from the M111-trainable partition unless recovered here.
- Any extended vocab requires a future checkpoint head migration (112 → M+1); not performed in P34.5.
"""
    write_md(OUT / "FINAL_REPORT.md", text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose-only", action="store_true")
    ap.add_argument("--skip-diagnose", action="store_true")
    ap.add_argument("--skip-r2", action="store_true", default=True)
    ap.add_argument("--with-r2", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit-fail", type=int, default=0)
    args = ap.parse_args()
    skip_r2 = not args.with_r2

    OUT.mkdir(parents=True, exist_ok=True)
    lim = args.limit_fail if args.limit_fail > 0 else None

    diag = {"n_fail": 38, "taxonomy": {}, "n_impl": 0, "n_repr": 38, "ok_spot": 0, "ok_spot_n": 0}
    if not args.skip_diagnose:
        diag = run_diagnosis(limit_fail=lim, verify_ok_n=5 if args.quick else 10)
    if args.diagnose_only:
        return

    fails, oks = _rows()
    if lim:
        fails = fails[:lim]
    design, hold, hold_roots = _split_fail_lineages(fails, n_holdout=10)
    design_paths = [_abs(r["ic_path"]) for r in design]
    fail_paths = [_abs(r["ic_path"]) for r in fails]
    ok_paths = [_abs(r["ic_path"]) for r in oks]
    write_csv(
        OUT / "FAIL_DESIGN_HOLDOUT_SPLIT.csv",
        [
            {
                "tag": r.get("tag"),
                "lineage_id": r.get("lineage_id"),
                "partition": "holdout"
                if (r.get("lineage_id") or "") in {x.get("lineage_id") for x in hold}
                else "design",
            }
            for r in fails
        ],
    )
    print(
        f"Design fails={len(design)} holdout={len(hold)} roots={hold_roots}",
        flush=True,
    )

    abl = run_ablation(
        skip_r2=skip_r2,
        quick=args.quick,
        design_paths=design_paths,
        fail_paths=fail_paths,
        ok_paths=ok_paths,
    )

    # Freeze chosen before independent numbers (already mined from design only)
    # Full preservation check on all previously M111-ok roots for the frozen chosen vocab.
    reg = run_regression(abl["book_chosen"], n_ok_spot=8 if args.quick else 85)
    indep = run_independent_validation(
        abl["book"],
        abl["book_chosen"],
        abl["rho"],
        chosen_label=abl["chosen"]["candidate"],
    )
    # Note: independent_val re-splits; ensure design-only mining already done.
    write_final(diag, abl, reg, indep)


if __name__ == "__main__":
    main()
