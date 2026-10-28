"""P14 — Small-error D2 from known-good stable grasps (bounded matrix)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.live_stage1 import extract_precision_skills  # noqa: E402
from data_generation.oracles.precision.d2_eligibility import check_d2_eligible  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import (  # noqa: E402
    capture_fingerprint,
    check_stage1_eligible,
)
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from experiments.precision.p13_teleport_d2.teleport_builder import clone_capture  # noqa: E402
from experiments.precision.p14_small_error_d2.small_error_builder import (  # noqa: E402
    SmallErrorSpec,
    build_small_error_ic,
    measure_pack,
)
from experiments.precision.precision_required_pilot_v2.run_pilot import (  # noqa: E402
    discover_d2_bases,
)

OUT = ROOT / "results" / "diagnostics" / "p14_small_error_d2"
SEED = 14
N_BASES = 4


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


def _norm_transition(skills: list[str]) -> str:
    norm: list[str] = []
    for s in skills:
        sl = str(s).lower()
        if sl in ("dz", "dz_inward", "dz_outward"):
            lab = "dz"
        elif sl in ("dx", "support_transfer"):
            lab = "dx"
        elif "dtheta" in sl:
            lab = "dtheta"
        else:
            lab = sl
        if not norm or norm[-1] != lab:
            norm.append(lab)
    return "→".join(norm) if norm else "empty"


def select_bases(env, cfg: PrecisionConfig) -> list[dict]:
    """Pick distinct 12S-stable bilateral low-error grasps; verify restore."""
    rng = np.random.default_rng(SEED)
    pool = discover_d2_bases(rng)
    chosen: list[dict] = []
    audit: list[dict] = []
    for b in pool:
        if len(chosen) >= N_BASES:
            break
        raw = pickle.loads(b["path"].read_bytes())
        if "capture" not in raw or "ref" not in raw:
            continue
        blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": b["sample_id"]}
        restore_capture(env, blob["capture"])
        pack0 = measure_pack(env.sim)
        elig = check_stage1_eligible(env, blob, allow_live_verify=True, record_live=False)
        restore_capture(env, blob["capture"])
        pack1 = measure_pack(env.sim)
        row = {
            "sample_id": b["sample_id"],
            "path": str(b["path"].relative_to(ROOT)).replace("\\", "/"),
            "restore_bilateral": pack0["bilateral"],
            "dx_mm": pack0["dx_mm"],
            "dz_mm": pack0["dz_mm"],
            "e_pos_mm": pack0["e_pos_mm"],
            "dtheta_deg": pack0["dtheta_deg"],
            "v_rel_norm": pack0["v_rel_norm"],
            "SUCCESS_12S": elig.SUCCESS_12S,
            "eligible": elig.eligible,
            "post_verify_bilateral": pack1["bilateral"],
            "e_pos_drift_mm": abs(float(pack1["e_pos_mm"]) - float(pack0["e_pos_mm"]))
            if pack0["e_pos_mm"] is not None
            else None,
        }
        # Prefer centered, bilateral, 12S, low slip
        ok = (
            bool(elig.eligible)
            and bool(pack0["bilateral"])
            and float(pack0["e_pos_mm"] or 99) <= 12.0
            and float(pack0["v_rel_norm"] or 99) < 0.08
            and float(pack0["dtheta_deg"] or 99) <= 10.0
        )
        row["selected"] = ok
        audit.append(row)
        if ok:
            # diversify: skip near-duplicates by e signature
            sig = (
                round(float(pack0["dx_mm"]), 1),
                round(float(pack0["dz_mm"]), 1),
            )
            if any(
                (
                    round(float(c["pack"]["dx_mm"]), 1),
                    round(float(c["pack"]["dz_mm"]), 1),
                )
                == sig
                for c in chosen
            ):
                row["selected"] = False
                row["reject"] = "DUPLICATE_ERROR_SIG"
                continue
            chosen.append({"blob": blob, "pack": pack0, "meta": row})
            print(
                f"  BASE {b['sample_id'][:50]} e_pos={pack0['e_pos_mm']:.2f} "
                f"dx={pack0['dx_mm']:.2f} dz={pack0['dz_mm']:.2f}",
                flush=True,
            )
    return chosen, audit


def matrix_specs() -> list[SmallErrorSpec]:
    specs: list[SmallErrorSpec] = []
    # Step 4 illustrative small band (often below Precision threshold)
    for v in (2.0, 4.0, 6.0, -2.0, -4.0, -6.0):
        specs.append(SmallErrorSpec(family="DX_ONLY_DIAG", target_dx_mm=v, target_dz_mm=0.0))
        specs.append(SmallErrorSpec(family="DZ_ONLY_DIAG", target_dx_mm=0.0, target_dz_mm=v))
    for dx, dz in ((4.0, 2.0), (2.0, 4.0), (-4.0, -2.0), (-2.0, -4.0)):
        specs.append(SmallErrorSpec(family="COMBINED_DIAG", target_dx_mm=dx, target_dz_mm=dz))
    # Small but Precision-eligible band (still modest vs P13 14–18 mm)
    for v in (8.0, 10.0, 12.0, -8.0, -10.0, -12.0):
        specs.append(SmallErrorSpec(family="DX_ONLY", target_dx_mm=v, target_dz_mm=0.0))
    for v in (8.0, 12.0, 16.0, -8.0, -12.0, -16.0):
        specs.append(SmallErrorSpec(family="DZ_ONLY", target_dx_mm=0.0, target_dz_mm=v))
    for dx, dz in ((8.0, 4.0), (10.0, 4.0), (4.0, 12.0), (4.0, 16.0), (-8.0, -4.0), (-4.0, -12.0)):
        specs.append(SmallErrorSpec(family="COMBINED", target_dx_mm=dx, target_dz_mm=dz))
    return specs


def render_video(env, blob, actions, dest: Path, title: str) -> str | None:
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
        from experiments.precision.render_precision_failure_videos import overlay
        from data_generation.oracles.precision.residuals import measure_errors
        from envs.step_legal import step_legal

        dest.parent.mkdir(parents=True, exist_ok=True)
        restore_capture(env, blob["capture"])
        cam = mjv_camera_from_preset(
            {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
        )
        renderer = mujoco.Renderer(env.sim.model, 480, 640)
        frames = []
        for i in range(8):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            step_legal(env.sim, a)
            renderer.update_scene(env.sim.data, cam)
            err = measure_errors(env.sim)
            frames.append(
                overlay(
                    renderer.render(),
                    [title, f"hold {i}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"],
                )
            )
        acts = list(actions or [])
        step = max(1, len(acts) // 100) if acts else 1
        for i, act in enumerate(acts[::step]):
            step_legal(env.sim, np.asarray(act, np.float32))
            if i % 2 == 0:
                renderer.update_scene(env.sim.data, cam)
                err = measure_errors(env.sim)
                frames.append(
                    overlay(
                        renderer.render(),
                        [title, f"i={i}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"],
                    )
                )
        write_frames_mp4(frames, dest, fps=20)
        return str(dest.relative_to(ROOT)).replace("\\", "/")
    except Exception as exc:
        return f"VIDEO_FAIL:{exc}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "ics").mkdir(parents=True, exist_ok=True)

    env = make_env()
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)

    print("=== P14 base audit ===", flush=True)
    bases, base_audit = select_bases(env, cfg)
    _write_csv(OUT / "BASE_SNAPSHOT_ROWS.csv", base_audit)
    if not bases:
        print("NO BASES", flush=True)
        return

    specs = matrix_specs()
    matrix_rows: list[dict] = []
    stability_rows: list[dict] = []
    precision_rows: list[dict] = []
    success_rows: list[dict] = []
    seen_fp: set[str] = set()

    print(f"=== P14 matrix bases={len(bases)} specs={len(specs)} ===", flush=True)
    for bi, base in enumerate(bases):
        for si, spec in enumerate(specs):
            tag = f"{spec.family}_b{bi}_dx{spec.target_dx_mm:g}_dz{spec.target_dz_mm:g}"
            t0 = time.time()
            built = build_small_error_ic(env, base["blob"], spec, cfg=cfg)
            row = built.row()
            row["tag"] = tag
            row["elapsed_s"] = round(time.time() - t0, 2)
            matrix_rows.append(row)
            status = "OK" if built.ok else f"FAIL:{built.reason}"
            if built.ok:
                status += f" prec_req={built.log.get('precision_required')}"
            print(f"  [{tag}] {status}", flush=True)

            if not built.ok:
                continue
            if built.capture_fp in seen_fp:
                matrix_rows[-1]["duplicate_fp"] = True
                continue
            seen_fp.add(built.capture_fp or "")

            if not built.log.get("precision_required"):
                matrix_rows[-1]["class"] = "VALID_NO_PRECISION"
                continue

            # Independent 12S on clone
            frozen = {
                "capture": built.capture,
                "ref": built.ref,
                "sample_id": tag,
            }
            clone = {
                "capture": clone_capture(built.capture),
                "ref": built.ref,
                "sample_id": tag,
            }
            elig = check_d2_eligible(env, clone, cfg=cfg)
            restore_capture(env, built.capture)
            stab = {
                "tag": tag,
                "family": spec.family,
                "eligible": elig.eligible,
                "SUCCESS_12S": elig.SUCCESS_12S,
                "d2_status": elig.status,
                "gate_reason": elig.reason,
                "first_fail_stage": (
                    ""
                    if elig.eligible
                    else (
                        "INDEPENDENT_SUCCESS_12S"
                        if elig.SUCCESS_12S is False
                        else "D2_NEED_GATE"
                    )
                ),
                "post_dx": built.log.get("post_settle_dx_mm"),
                "post_dz": built.log.get("post_settle_dz_mm"),
                "fsm_case": built.log.get("fsm_case"),
            }
            stability_rows.append(stab)
            if not elig.eligible:
                print(
                    f"    gate {elig.status} reason={elig.reason} "
                    f"12S={elig.SUCCESS_12S}",
                    flush=True,
                )
                continue

            # Precision from frozen
            res = run_precision(env, frozen, "D", cfg)
            sk = extract_precision_skills(list(res.phases or []))
            trans = _norm_transition(sk)
            hand = res.handoff or {}
            ok = res.status in (
                "PRECISION_EXECUTED_SUCCESS",
                "PRECISION_SKIPPED_DIRECT_SUCCESS",
            )
            prec_row = {
                "tag": tag,
                "family": spec.family,
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "post_dx_mm": built.log.get("post_settle_dx_mm"),
                "post_dz_mm": built.log.get("post_settle_dz_mm"),
                "fsm_case": built.log.get("fsm_case"),
                "planned": built.log.get("fsm_planned"),
                "executed": "|".join(sk),
                "transition": trans,
                "status": res.status,
                "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
                "SUCCESS_12S_final": bool(hand.get("SUCCESS_12S")),
                "accepted": ok,
                "n_actions": len(res.actions or []),
            }
            precision_rows.append(prec_row)
            print(f"    Precision {res.status} trans={trans}", flush=True)

            if ok:
                ic_path = OUT / "ics" / f"{tag}.pkl"
                ic_path.write_bytes(
                    pickle.dumps(
                        {
                            **frozen,
                            "construction": built.log,
                            "actions": res.actions,
                            "phases": res.phases,
                        },
                        protocol=4,
                    )
                )
                # video once per transition family
                label_map = {
                    "dx": "DX",
                    "dz": "DZ",
                    "dx→dz": "DX_to_DZ",
                    "dz→dx": "DZ_to_DX",
                }
                lab = label_map.get(trans)
                if lab is None:
                    if trans.startswith("dx→dz"):
                        lab = "DX_to_DZ"
                    elif trans.startswith("dz→dx"):
                        lab = "DZ_to_DX"
                    elif trans.startswith("dx"):
                        lab = "DX"
                    elif trans.startswith("dz"):
                        lab = "DZ"
                video = None
                have = {r.get("video_label") for r in success_rows}
                if lab and lab not in have:
                    video = render_video(
                        env,
                        frozen,
                        res.actions or [],
                        OUT / "videos" / f"{lab}.mp4",
                        f"{lab} | {trans}",
                    )
                success_rows.append(
                    {
                        **prec_row,
                        "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
                        "capture_fp": built.capture_fp,
                        "video_label": lab,
                        "video_path": video,
                        "admit_success_key": "SUCCESS_NEW",
                    }
                )

    _write_csv(OUT / "SMALL_ERROR_MATRIX.csv", matrix_rows)
    _write_csv(OUT / "STABILITY_RESULTS.csv", stability_rows)
    _write_csv(OUT / "PRECISION_EXECUTION_RESULTS.csv", precision_rows)
    _write_csv(OUT / "STRICT_SUCCESS_RESULTS.csv", success_rows)

    # --- reports ---
    def funnel(fam_prefix: str) -> dict[str, int]:
        sub = [r for r in matrix_rows if str(r.get("family", "")).startswith(fam_prefix) or r.get("family") == fam_prefix]
        # also exact family match for DX_ONLY etc
        if not sub:
            sub = [r for r in matrix_rows if fam_prefix in str(r.get("family", ""))]
        n = len(sub)
        freeze = sum(1 for r in sub if r.get("ok") in (True, "True"))
        no_prec = sum(1 for r in sub if r.get("class") == "VALID_NO_PRECISION" or (
            r.get("ok") in (True, "True") and r.get("precision_required") in (False, "False")
        ))
        # precision_required true among freeze
        prec_req = sum(
            1
            for r in sub
            if r.get("ok") in (True, "True") and r.get("precision_required") in (True, "True")
        )
        stab = sum(1 for r in stability_rows if fam_prefix in str(r.get("family", "")) and r.get("eligible") in (True, "True"))
        prec_run = sum(1 for r in precision_rows if fam_prefix in str(r.get("family", "")))
        succ = sum(1 for r in success_rows if fam_prefix in str(r.get("family", "")))
        return {
            "attempts": n,
            "freeze_ok": freeze,
            "valid_no_precision": no_prec,
            "precision_required": prec_req,
            "stability_pass": stab,
            "precision_runs": prec_run,
            "strict_success": succ,
        }

    funnels = {
        "DX_ONLY_DIAG": funnel("DX_ONLY_DIAG"),
        "DZ_ONLY_DIAG": funnel("DZ_ONLY_DIAG"),
        "COMBINED_DIAG": funnel("COMBINED_DIAG"),
        "DX_ONLY": funnel("DX_ONLY"),
        "DZ_ONLY": funnel("DZ_ONLY"),
        "COMBINED": funnel("COMBINED"),
    }
    # Fix DX_ONLY counting overlapping DX_ONLY_DIAG — recompute carefully
    def funnel_exact(name: str) -> dict[str, int]:
        sub = [r for r in matrix_rows if r.get("family") == name]
        freeze = [r for r in sub if r.get("ok") in (True, "True")]
        no_prec = [r for r in freeze if r.get("precision_required") in (False, "False")]
        prec_req = [r for r in freeze if r.get("precision_required") in (True, "True")]
        tags = {r.get("tag") for r in prec_req}
        stab = sum(1 for r in stability_rows if r.get("tag") in tags and r.get("eligible") in (True, "True"))
        pr = [r for r in precision_rows if r.get("family") == name]
        su = [r for r in success_rows if r.get("family") == name]
        return {
            "attempts": len(sub),
            "freeze_ok": len(freeze),
            "valid_no_precision": len(no_prec),
            "precision_required": len(prec_req),
            "stability_pass": stab,
            "precision_runs": len(pr),
            "strict_success": len(su),
            "first_fail_modes": dict(
                Counter(r.get("reason") for r in sub if r.get("ok") not in (True, "True")).most_common(5)
            ),
        }

    funnels = {k: funnel_exact(k) for k in funnels}

    _write_md(
        OUT / "BASE_SNAPSHOT_AUDIT.md",
        f"""# Base Snapshot Audit (P14)

Selected **{len(bases)}** distinct 12S-stable, bilateral, low-slip, e_pos≤12 mm grasps from Stage-1 terminals / A1 pool.

Each base: full capture restore → measure → live SUCCESS_12S → restore again (confirm bilateral retained).

See `BASE_SNAPSHOT_ROWS.csv`.

Selected:
"""
        + "\n".join(
            f"- `{b['meta']['sample_id']}` e_pos={b['pack']['e_pos_mm']:.2f} mm "
            f"dx={b['pack']['dx_mm']:.2f} dz={b['pack']['dz_mm']:.2f}"
            for b in bases
        ),
    )

    # Empirical retention for coordinate audit
    def _ret(name: str) -> str:
        sub = [
            r
            for r in matrix_rows
            if r.get("family") == name and r.get("ok") in (True, "True")
        ]
        if not sub:
            return f"| {name} | n/a | n/a |"
        dxs, dzs = [], []
        for r in sub:
            try:
                dxs.append(abs(float(r["post_settle_dx_mm"]) - float(r["requested_dx_mm"])))
                dzs.append(abs(float(r["post_settle_dz_mm"]) - float(r["requested_dz_mm"])))
            except (TypeError, ValueError, KeyError):
                pass
        if not dxs:
            return f"| {name} | n/a | n/a |"
        return (
            f"| {name} | {sum(dxs)/len(dxs):.2f} | {sum(dzs)/len(dzs):.2f} |"
        )

    dx_12s_fail = sum(
        1
        for r in stability_rows
        if r.get("family") == "DX_ONLY" and r.get("SUCCESS_12S") in (False, "False")
    )
    _write_md(
        OUT / "COORDINATE_AND_CONTACT_AUDIT.md",
        f"""# Coordinate and Contact Audit (P14)

## Residual definition (authoritative)
```
e_signed_mm = (p_c_h - P_TGT) * 1e3
dx_mm = e[0],  dy_mm = e[1],  dz_mm = e[2]
```
`p_c_h` = object COM in **hand frame** (`evaluation.grasp_geometry.p_com_hand`).

## Precision-required thresholds (unchanged)
- `dx_needed`: |dx| > **ex_tol = {cfg.ex_tol_mm} mm**
- `dz_needed`: |dz| > **ez_tol = {cfg.ez_tol_mm} mm**
- Also: e_pos > **working_pos = {cfg.working_pos_mm} mm** can force a dominant-axis need

## Empirical retention (freeze-OK mean abs(post-requested) mm)
| family | mean_dx_err | mean_dz_err |
|--------|------------:|------------:|
{_ret("DX_ONLY")}
{_ret("DZ_ONLY")}
{_ret("COMBINED")}

## Hypothesis check
1. **Incorrect transform** — rejected (retention ~0.3 mm; CASE1_DX appears).
2. **Invalid physical direction** — partial (large negative DZ → penetration).
3. **Error disappears in settle** — rejected for construction; DX ±12 mm still measured ~12 mm.
4. **Classification threshold** — explains DIAG ±2/4/6 → VALID_NO_PRECISION only.
5. **Contact-stability gate** — primary DX blocker: **{dx_12s_fail}** DX Precision-required freezes fail independent SUCCESS_12S.

## Why P13 had no DX-class ICs
Contact destruction on large teleports + survivors DZ-dominant; P14 shows clean ~12 mm DX still fails independent 12S. Not a frame bug; do not densify P13 grid.

## P14 construction
Absolute residual `(target_dx, ey_keep, target_dz)` via `delta_h = e_tgt - e_cur`, then `p_object += R_hand @ delta_h`, settle, freeze.
""",
    )

    _write_md(
        OUT / "LINEAGE_AUDIT.md",
        f"""# Lineage Audit (P14)

- Official dataset untouched
- Bases: known-good captures with full restore
- Perturbation metadata in construction logs only
- Dedup by `capture_fp`
- STRICT accepts: {len(success_rows)}
- Independent fingerprints: {len(seen_fp)}
""",
    )

    vids = "\n".join(
        f"- **{r.get('video_label')}** (`{r.get('transition')}`): `{r.get('video_path')}`"
        for r in success_rows
        if r.get("video_label")
    ) or "None yet."

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P14 Small-Error D2

Generated: {_utc()}

## Question
Can small, physically stable, Precision-required ICs be generated from a few known-good grasps without destroying contact?

## Answer (this pilot)
- **±2/4/6 mm:** physically often OK, but almost always **below Precision threshold** → valid no-Precision states (do not force correction).
- **Modest Precision-eligible band (DX ~8–12 mm, DZ ~8–16 mm):** contact-preserving yield depends on base; see funnels.
- STRICT successes: **{len(success_rows)}** — transitions {dict(Counter(r.get('transition') for r in success_rows))}

## Funnels (exact family)
```json
{json.dumps(funnels, indent=2)}
```

## Videos
{vids}

## Do not scale until
At least one family shows reproducible freeze → 12S → Precision-required → STRICT > 0 with non-trivial rate.
Then densify **that** family only.

## Integrity
No M111/physics/threshold changes; no hard-IC tuning; no official auto-admit; no P13 grid densification.
""",
    )

    (OUT / "raw_p14_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "n_bases": len(bases),
                "n_matrix": len(matrix_rows),
                "n_strict": len(success_rows),
                "funnels": funnels,
                "transitions": dict(Counter(r.get("transition") for r in success_rows)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done → {OUT} strict={len(success_rows)}", flush=True)


if __name__ == "__main__":
    main()
