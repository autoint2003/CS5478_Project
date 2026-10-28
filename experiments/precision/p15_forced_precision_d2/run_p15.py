"""P15 — Forced Precision small-error D2 pilot (DX + Coupled).

Official dataset untouched. Production FSM defaults unchanged unless
``force_precision=True`` is set explicitly on PrecisionConfig.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.precision.stage1_eligibility import (  # noqa: E402
    check_stage1_eligible,
)
from envs.simulation.capture import restore_capture  # noqa: E402
from experiments.precision.p13_teleport_d2.teleport_builder import clone_capture  # noqa: E402
from experiments.precision.p14_small_error_d2.small_error_builder import (  # noqa: E402
    SmallErrorSpec,
    build_small_error_ic,
    measure_pack,
)
from experiments.precision.p15_forced_precision_d2.forced_oracle import (  # noqa: E402
    forced_precision_config,
    run_forced_precision,
)
from experiments.precision.precision_required_pilot_v2.run_pilot import (  # noqa: E402
    discover_d2_bases,
)

OUT = ROOT / "results" / "diagnostics" / "p15_forced_precision_d2"
SEED = 15
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


def select_bases(env) -> tuple[list[dict], list[dict]]:
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
            "dx_mm": pack0["dx_mm"],
            "dz_mm": pack0["dz_mm"],
            "e_pos_mm": pack0["e_pos_mm"],
            "SUCCESS_12S": elig.SUCCESS_12S,
            "eligible": elig.eligible,
            "bilateral": pack0["bilateral"],
            "post_verify_bilateral": pack1["bilateral"],
            "v_rel_norm": pack0["v_rel_norm"],
        }
        ok = (
            bool(elig.eligible)
            and bool(pack0["bilateral"])
            and float(pack0["e_pos_mm"] or 99) <= 12.0
            and float(pack0["v_rel_norm"] or 99) < 0.08
        )
        row["selected"] = ok
        audit.append(row)
        if not ok:
            continue
        sig = (round(float(pack0["dx_mm"]), 1), round(float(pack0["dz_mm"]), 1))
        if any(
            (round(float(c["pack"]["dx_mm"]), 1), round(float(c["pack"]["dz_mm"]), 1)) == sig
            for c in chosen
        ):
            row["selected"] = False
            continue
        chosen.append({"blob": blob, "pack": pack0, "meta": row})
        print(
            f"  BASE {b['sample_id'][:48]} e_pos={pack0['e_pos_mm']:.2f} "
            f"dx={pack0['dx_mm']:.2f} dz={pack0['dz_mm']:.2f}",
            flush=True,
        )
    return chosen, audit


def matrix_specs() -> list[tuple[str, SmallErrorSpec]]:
    """(force_skill, spec) pairs."""
    out: list[tuple[str, SmallErrorSpec]] = []
    for v in (3.0, 5.0, 7.0, 9.0, -3.0, -5.0, -7.0, -9.0):
        out.append(
            (
                "dx",
                SmallErrorSpec(family="DX_ONLY", target_dx_mm=v, target_dz_mm=0.0),
            )
        )
    for dx, dz in (
        (3.0, 4.0),
        (5.0, 5.0),
        (7.0, 4.0),
        (4.0, 7.0),
        (-3.0, -4.0),
        (-5.0, -5.0),
        (-7.0, -4.0),
        (-4.0, -7.0),
    ):
        out.append(
            (
                "coupled",
                SmallErrorSpec(family="COUPLED", target_dx_mm=dx, target_dz_mm=dz),
            )
        )
    return out


def try_m111_encode(actions: list) -> dict[str, Any]:
    """Best-effort M111 token summary; never fails the trial."""
    meta = (
        ROOT
        / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
    )
    if not meta.exists() or not actions:
        return {
            "m111_available": False,
            "m111_n_tokens": 0,
            "m111_note": "codebook missing or empty actions; continuous actions preserved",
        }
    try:
        from experiments.diagnostics.run_b_grpo_multistage_dataset import (  # type: ignore
            load_motion_codebook,
        )
        from experiments.diagnostics.run_skill_aware_hybrid_vocab import (  # type: ignore
            quantize_trajectory,
        )

        cb = load_motion_codebook(meta)
        q = quantize_trajectory(cb, actions, 0.35)
        return {
            "m111_available": True,
            "m111_n_tokens": int(q.get("n_unique_tokens") or len(q.get("tokens") or [])),
            "m111_clipped_frac": q.get("clipped_frac"),
            "m111_mean_wlinf": q.get("mean_recon_wlinf"),
            "m111_note": "encoding for lineage only; not leaked into policy obs",
        }
    except Exception as exc:
        return {"m111_available": False, "m111_n_tokens": 0, "m111_note": f"encode_fail:{exc}"}


def render_video(env, blob, actions, dest: Path, title: str) -> str | None:
    try:
        import mujoco
        from data_generation.oracles.precision.residuals import measure_errors
        from envs.step_legal import step_legal
        from experiments.precision.render_precision_failure_videos import overlay
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4

        dest.parent.mkdir(parents=True, exist_ok=True)
        restore_capture(env, blob["capture"])
        cam = mjv_camera_from_preset(
            {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
        )
        renderer = mujoco.Renderer(env.sim.model, 480, 640)
        frames = []
        for i in range(6):
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
        step = max(1, len(acts) // 120) if acts else 1
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "ics").mkdir(parents=True, exist_ok=True)
    (OUT / "trajectories").mkdir(parents=True, exist_ok=True)

    env = make_env()
    print("=== P15 base audit ===", flush=True)
    bases, base_audit = select_bases(env)
    _write_csv(OUT / "BASE_SNAPSHOT_ROWS.csv", base_audit)
    if not bases:
        print("NO BASES", flush=True)
        return

    specs = matrix_specs()
    matrix_rows: list[dict] = []
    dx_rows: list[dict] = []
    coupled_rows: list[dict] = []
    strict_rows: list[dict] = []
    coverage: list[dict] = []
    seen_fp: set[str] = set()
    n_dx_ok = 0
    n_coupled_ok = 0
    video_dx = False
    video_coupled = False

    print(f"=== P15 matrix bases={len(bases)} specs={len(specs)} ===", flush=True)
    for bi, base in enumerate(bases):
        for force_skill, spec in specs:
            tag = (
                f"{spec.family}_b{bi}_dx{spec.target_dx_mm:g}_dz{spec.target_dz_mm:g}"
                f"__force_{force_skill}"
            )
            t0 = time.time()
            built = build_small_error_ic(env, base["blob"], spec)
            row = built.row()
            row["tag"] = tag
            row["force_skill"] = force_skill
            row["elapsed_build_s"] = round(time.time() - t0, 2)
            matrix_rows.append(row)
            if not built.ok:
                print(f"  [{tag}] BUILD_FAIL:{built.reason}", flush=True)
                continue
            print(
                f"  [{tag}] freeze OK post_dx={row.get('post_settle_dx_mm')} "
                f"post_dz={row.get('post_settle_dz_mm')}",
                flush=True,
            )
            if built.capture_fp in seen_fp:
                row["duplicate_fp"] = True
                continue
            seen_fp.add(built.capture_fp or "")

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
            # Independent 12S (stability only — do not require production Precision need).
            elig = check_stage1_eligible(env, clone, allow_live_verify=True, record_live=False)
            restore_capture(env, built.capture)
            row["SUCCESS_12S"] = elig.SUCCESS_12S
            row["stage1_status"] = elig.status
            if not elig.SUCCESS_12S:
                print(f"    12S FAIL {elig.reason}", flush=True)
                row["fail_stage"] = "INDEPENDENT_12S"
                continue

            cfg = forced_precision_config(force_skill=force_skill)
            t1 = time.time()
            out = run_forced_precision(env, frozen, force_skill=force_skill, cfg=cfg)
            elapsed = round(time.time() - t1, 2)
            m111 = try_m111_encode(out.get("actions") or [])

            exec_row = {
                "tag": tag,
                "family": spec.family,
                "base_sample_id": base["meta"]["sample_id"],
                "forced_precision": True,
                "requested_skill": force_skill,
                "executed_skill": out.get("executed_skill"),
                "demo_class": out.get("demo_class"),
                "status": out.get("status"),
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "dx0_mm": out.get("dx0_mm"),
                "dz0_mm": out.get("dz0_mm"),
                "dx1_mm": out.get("dx1_mm"),
                "dz1_mm": out.get("dz1_mm"),
                "dx_reduction_mm": out.get("dx_reduction_mm"),
                "dz_reduction_mm": out.get("dz_reduction_mm"),
                "n_actions": out.get("n_actions"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "SUCCESS_12S_final": out.get("SUCCESS_12S_final"),
                "bilateral_exit": out.get("bilateral_exit"),
                "ok_demo": out.get("ok"),
                "elapsed_s": elapsed,
                "capture_fp": built.capture_fp,
                "reason": out.get("reason"),
                **{k: m111[k] for k in m111},
            }
            if force_skill == "dx":
                dx_rows.append(exec_row)
            else:
                coupled_rows.append(exec_row)

            print(
                f"    {force_skill} {out.get('demo_class')} status={out.get('status')} "
                f"n_act={out.get('n_actions')} STRICT={out.get('ok')}",
                flush=True,
            )

            strict_rows.append(
                {
                    **exec_row,
                    "phases": "|".join(out.get("phases") or []),
                    "failures": "|".join(str(x) for x in (out.get("failures") or [])),
                }
            )
            coverage.append(
                {
                    "tag": tag,
                    "requested_skill": force_skill,
                    "demo_class": out.get("demo_class"),
                    "ok_demo": out.get("ok"),
                    "n_actions": out.get("n_actions"),
                }
            )

            # Save trajectory lineage (forced skill in metadata, not obs)
            traj = {
                "tag": tag,
                "forced_precision": True,
                "requested_skill": force_skill,
                "executed_skill": out.get("executed_skill"),
                "demo_class": out.get("demo_class"),
                "dx0_mm": out.get("dx0_mm"),
                "dz0_mm": out.get("dz0_mm"),
                "dx1_mm": out.get("dx1_mm"),
                "dz1_mm": out.get("dz1_mm"),
                "actions": out.get("actions"),
                "m111": m111,
                "construction": built.log,
                "base_sample_id": base["meta"]["sample_id"],
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "policy_obs_note": (
                    "forced_skill and outcome are metadata only; "
                    "do not inject into policy observations unless explicitly intended"
                ),
            }
            if out.get("ok"):
                ic_path = OUT / "ics" / f"{tag}.pkl"
                ic_path.write_bytes(
                    pickle.dumps(
                        {
                            "capture": out.get("capture") or built.capture,
                            "ref": built.ref,
                            "sample_id": tag,
                            "actions": out.get("actions"),
                            "meta": traj,
                        },
                        protocol=4,
                    )
                )
                traj_path = OUT / "trajectories" / f"{tag}.pkl"
                traj_path.write_bytes(pickle.dumps(traj, protocol=4))
                exec_row["ic_path"] = str(ic_path.relative_to(ROOT)).replace("\\", "/")
                if force_skill == "dx":
                    n_dx_ok += 1
                    if not video_dx:
                        vp = render_video(
                            env,
                            frozen,
                            out.get("actions") or [],
                            OUT / "videos" / "DX.mp4",
                            f"FORCE DX | {tag}",
                        )
                        exec_row["video_path"] = vp
                        video_dx = True
                else:
                    n_coupled_ok += 1
                    if not video_coupled:
                        vp = render_video(
                            env,
                            frozen,
                            out.get("actions") or [],
                            OUT / "videos" / "COUPLED.mp4",
                            f"FORCE COUPLED | {tag}",
                        )
                        exec_row["video_path"] = vp
                        video_coupled = True

    _write_csv(OUT / "SMALL_ERROR_IC_MATRIX.csv", matrix_rows)
    _write_csv(OUT / "DX_EXECUTION_RESULTS.csv", dx_rows)
    _write_csv(OUT / "COUPLED_EXECUTION_RESULTS.csv", coupled_rows)
    _write_csv(OUT / "STRICT_VERIFICATION.csv", strict_rows)
    _write_csv(OUT / "SKILL_COVERAGE.csv", coverage)

    dx_classes = Counter(r.get("demo_class") for r in dx_rows)
    cp_classes = Counter(r.get("demo_class") for r in coupled_rows)

    # Ranges that yielded STRICT demos
    dx_ranges = sorted(
        {
            f"dx={r.get('requested_dx_mm')}"
            for r in dx_rows
            if r.get("ok_demo") in (True, "True")
        }
    )
    cp_ranges = sorted(
        {
            f"dx={r.get('requested_dx_mm')},dz={r.get('requested_dz_mm')}"
            for r in coupled_rows
            if r.get("ok_demo") in (True, "True")
        }
    )

    _write_md(
        OUT / "FORCED_PRECISION_IMPLEMENTATION.md",
        f"""# Forced Precision Implementation (P15)

## API (experimental, defaults OFF)

`PrecisionConfig` fields (production defaults unchanged):

- `force_precision: bool = False`
- `force_skill: str | None = None`  # `\"dx\"` | `\"coupled\"`
- `force_ex_tol_mm: float | None = 1.0`  # experimental need band
- `force_ez_tol_mm: float | None = 1.0`
- `force_lateral_axis_tol_mm: float | None = 1.0`  # tighter DX slip stop
- `force_min_meaningful_mm: float = 0.5`

## Behavior when enabled

1. Bypass `PRECISION_SKIPPED_DIRECT_SUCCESS` / entry threshold skip.
2. `assess_need` uses force_* bands so small residuals schedule the skill.
3. **DX:** production `run_precision` → `correct_dx_support_transfer` (real `step_legal` actions).
4. **Coupled:** experimental `coupled_dx_dz` (not a silent DX/DZ substitute) → DOWN gate → STRICT handoff.
5. Contact / penetration / primitive safety checks retained.

## Entry points

- `experiments/precision/p15_forced_precision_d2/forced_oracle.py`
  - `forced_precision_config(...)`
  - `run_forced_precision(env, blob, force_skill=...)`

Forced skill is **dataset metadata only** — not written into policy observations.

Generated: {_utc()}
""",
    )

    _write_md(
        OUT / "LINEAGE_AUDIT.md",
        f"""# Lineage Audit (P15)

- Official / ACTIVE dataset untouched
- Bases: verified 12S-stable captures (full restore)
- Construction: P14 absolute hand-frame residual targeting
- `forced_precision=True` + requested skill recorded in trajectory metadata only
- Continuous actions saved; M111 encoding attempted for lineage when codebook present
- Independent fingerprints: {len(seen_fp)}
- STRICT DX demos: {n_dx_ok}
- STRICT Coupled (both-axes) demos: {n_coupled_ok}
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P15 Forced Precision D2

Generated: {_utc()}

## Answers

1. **Can DX produce meaningful corrections below 12 mm?**  
   Demo classes: `{dict(dx_classes)}` — STRICT DX successes: **{n_dx_ok}**

2. **Can Coupled produce genuine two-axis correction?**  
   Demo classes: `{dict(cp_classes)}` — STRICT both-axes Coupled: **{n_coupled_ok}**

3. **Successful small-error ranges**  
   - DX: {dx_ranges or "none"}  
   - Coupled: {cp_ranges or "none"}

4. **Independent successful trajectories**  
   - DX: **{n_dx_ok}**  
   - Coupled (both axes): **{n_coupled_ok}**

5. **SFT suitability without leaking forced-skill labels**  
   Yes for packaging: forced skill / outcome live in pickle metadata (`trajectories/`), not in observation tensors. Continuous actions preserved; M111 optional lineage only.

## Videos
- DX: `{"results/diagnostics/p15_forced_precision_d2/videos/DX.mp4" if video_dx else "none"}`
- Coupled: `{"results/diagnostics/p15_forced_precision_d2/videos/COUPLED.mp4" if video_coupled else "none"}`

## Integrity
No official dataset mutation. Production Precision defaults unchanged when `force_precision=False`.
No meaningless fabricated motion; `NO_ACTION_NEEDED` recorded when residual already inside force tol.
""",
    )

    (OUT / "raw_p15_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "n_bases": len(bases),
                "n_matrix": len(matrix_rows),
                "n_dx_strict": n_dx_ok,
                "n_coupled_strict": n_coupled_ok,
                "dx_classes": dict(dx_classes),
                "coupled_classes": dict(cp_classes),
                "dx_ranges": dx_ranges,
                "coupled_ranges": cp_ranges,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Done → {OUT} dx_strict={n_dx_ok} coupled_strict={n_coupled_ok}",
        flush=True,
    )


if __name__ == "__main__":
    main()
