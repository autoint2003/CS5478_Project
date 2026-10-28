"""P16 — Expand DX/DZ demos; reassess + limited Coupled generation.

Official corpus untouched. Production FSM defaults unchanged.
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

OUT = ROOT / "results" / "diagnostics" / "p16_precision_skill_expansion"
P15 = ROOT / "results" / "diagnostics" / "p15_forced_precision_d2"
# P15-compatible DX base (b3)
DX_BASE_ID = "A_STABLE_DIRECT_HANDOFF__A_STAGE1__ball_train_v2_0168"
DX_BASE_PATH = (
    ROOT
    / "data/experts_v3_stage1_verified/A_STABLE_DIRECT_HANDOFF"
    / f"{DX_BASE_ID}.pkl"
)
SEED = 16
MIN_MM = 0.5


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


def load_base(path: Path, sample_id: str) -> dict:
    raw = pickle.loads(path.read_bytes())
    return {"capture": raw["capture"], "ref": raw["ref"], "sample_id": sample_id}


def action_diversity(actions: list) -> dict[str, Any]:
    if not actions:
        return {"n_actions": 0, "n_unique_rounded": 0, "mean_linf_step": 0.0}
    arr = np.asarray(actions, dtype=np.float64)
    # Quantize to 0.05 bins for token-like diversity (continuous; not M111)
    q = np.round(arr / 0.05) * 0.05
    uniq = len({tuple(row.tolist()) for row in q})
    diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1, ord=np.inf) if len(arr) > 1 else [0.0]
    return {
        "n_actions": int(len(actions)),
        "n_unique_rounded": int(uniq),
        "mean_linf_step": float(np.mean(diffs)),
        "action_std_mean": float(np.mean(np.std(arr, axis=0))),
    }


def try_m111(actions: list) -> dict[str, Any]:
    meta = (
        ROOT
        / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
    )
    if not meta.exists() or not actions:
        return {"m111_n_tokens": 0, "m111_ok": False}
    try:
        from experiments.diagnostics.run_b_grpo_multistage_dataset import load_motion_codebook
        from experiments.diagnostics.run_skill_aware_hybrid_vocab import quantize_trajectory

        cb = load_motion_codebook(meta)
        q = quantize_trajectory(cb, actions, 0.35)
        return {
            "m111_ok": True,
            "m111_n_tokens": int(q.get("n_unique_tokens") or len(q.get("tokens") or [])),
            "m111_clipped_frac": q.get("clipped_frac"),
        }
    except Exception as exc:
        return {"m111_ok": False, "m111_n_tokens": 0, "m111_note": str(exc)}


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
        for i in range(5):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            step_legal(env.sim, a)
            renderer.update_scene(env.sim.data, cam)
            err = measure_errors(env.sim)
            frames.append(
                overlay(renderer.render(), [title, f"h{i}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"])
            )
        acts = list(actions or [])
        step = max(1, len(acts) // 100) if acts else 1
        for i, act in enumerate(acts[::step]):
            step_legal(env.sim, np.asarray(act, np.float32))
            if i % 2 == 0:
                renderer.update_scene(env.sim.data, cam)
                err = measure_errors(env.sim)
                frames.append(
                    overlay(renderer.render(), [title, f"i={i}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"])
                )
        write_frames_mp4(frames, dest, fps=20)
        return str(dest.relative_to(ROOT)).replace("\\", "/")
    except Exception as exc:
        return f"VIDEO_FAIL:{exc}"


def verify_base(env, blob) -> dict:
    restore_capture(env, blob["capture"])
    pack = measure_pack(env.sim)
    elig = check_stage1_eligible(env, blob, allow_live_verify=True, record_live=False)
    restore_capture(env, blob["capture"])
    return {
        "SUCCESS_12S": elig.SUCCESS_12S,
        "eligible": elig.eligible,
        "dx_mm": pack["dx_mm"],
        "dz_mm": pack["dz_mm"],
        "e_pos_mm": pack["e_pos_mm"],
        "bilateral": pack["bilateral"],
    }


def run_skill_matrix(
    env,
    bases: list[dict],
    specs: list[tuple[str, SmallErrorSpec]],
    *,
    rows_out: list[dict],
    accepted: list[dict],
    coverage: list[dict],
    video_flags: dict[str, bool],
) -> None:
    for bi, base in enumerate(bases):
        for force_skill, spec in specs:
            tag = (
                f"{spec.family}_b{bi}_{base['sample_id'][-8:]}"
                f"_dx{spec.target_dx_mm:g}_dz{spec.target_dz_mm:g}__{force_skill}"
            )
            built = build_small_error_ic(env, base, spec)
            row = {
                "tag": tag,
                "force_skill": force_skill,
                "base_sample_id": base["sample_id"],
                "requested_dx_mm": spec.target_dx_mm,
                "requested_dz_mm": spec.target_dz_mm,
                "build_ok": built.ok,
                "build_reason": built.reason,
            }
            if built.ok:
                row.update(
                    {
                        "post_dx_mm": built.log.get("post_settle_dx_mm"),
                        "post_dz_mm": built.log.get("post_settle_dz_mm"),
                        "post_e_pos_mm": built.log.get("post_settle_e_pos_mm"),
                        "capture_fp": built.capture_fp,
                    }
                )
            if not built.ok:
                print(f"  [{tag}] BUILD_FAIL {built.reason}", flush=True)
                rows_out.append(row)
                continue

            frozen = {"capture": built.capture, "ref": built.ref, "sample_id": tag}
            clone = {
                "capture": clone_capture(built.capture),
                "ref": built.ref,
                "sample_id": tag,
            }
            elig = check_stage1_eligible(env, clone, allow_live_verify=True, record_live=False)
            restore_capture(env, built.capture)
            row["SUCCESS_12S"] = elig.SUCCESS_12S
            if not elig.SUCCESS_12S:
                print(f"  [{tag}] 12S FAIL", flush=True)
                row["fail_stage"] = "INDEPENDENT_12S"
                rows_out.append(row)
                continue

            cfg = forced_precision_config(force_skill=force_skill)
            t0 = time.time()
            out = run_forced_precision(env, frozen, force_skill=force_skill, cfg=cfg)
            div = action_diversity(out.get("actions") or [])
            m111 = try_m111(out.get("actions") or [])
            row.update(
                {
                    "demo_class": out.get("demo_class"),
                    "status": out.get("status"),
                    "dx0_mm": out.get("dx0_mm"),
                    "dz0_mm": out.get("dz0_mm"),
                    "dx1_mm": out.get("dx1_mm"),
                    "dz1_mm": out.get("dz1_mm"),
                    "e_pos0_mm": out.get("e_pos0_mm"),
                    "e_pos1_mm": out.get("e_pos1_mm"),
                    "dx_reduction_mm": out.get("dx_reduction_mm"),
                    "dz_reduction_mm": out.get("dz_reduction_mm"),
                    "n_actions": out.get("n_actions"),
                    "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                    "SUCCESS_12S_final": out.get("SUCCESS_12S_final"),
                    "bilateral_exit": out.get("bilateral_exit"),
                    "ok_demo": out.get("ok"),
                    "dz_mode": out.get("dz_mode"),
                    "both_axis_improved": out.get("both_axis_improved"),
                    "e_pos_improved": out.get("e_pos_improved"),
                    "primary_improvement": out.get("primary_improvement"),
                    "reason": out.get("reason"),
                    "elapsed_s": round(time.time() - t0, 2),
                    **div,
                    **m111,
                    "same_base_lineage": base["sample_id"],
                    "forced_precision": True,
                    "policy_obs_note": "forced_skill is metadata only",
                }
            )
            rows_out.append(row)
            coverage.append(
                {
                    "tag": tag,
                    "skill": force_skill,
                    "demo_class": out.get("demo_class"),
                    "ok_demo": out.get("ok"),
                    "n_actions": out.get("n_actions"),
                    "base_sample_id": base["sample_id"],
                }
            )
            print(
                f"  [{tag}] {out.get('demo_class')} STRICT={out.get('ok')} n={out.get('n_actions')}",
                flush=True,
            )
            if out.get("ok"):
                ic_path = OUT / "ics" / f"{tag}.pkl"
                ic_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "capture": out.get("capture") or built.capture,
                    "ref": built.ref,
                    "sample_id": tag,
                    "actions": out.get("actions"),
                    "meta": {k: row[k] for k in row if k != "actions"},
                    "base_sample_id": base["sample_id"],
                    "forced_precision": True,
                    "requested_skill": force_skill,
                }
                ic_path.write_bytes(pickle.dumps(payload, protocol=4))
                acc = {
                    **row,
                    "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
                    "skill": force_skill,
                }
                # videos once per skill
                lab = {"dx": "DX", "dz": "DZ", "coupled": "COUPLED"}.get(force_skill)
                if lab and not video_flags.get(lab):
                    vp = render_video(
                        env, frozen, out.get("actions") or [], OUT / "videos" / f"{lab}.mp4", f"{lab} | {tag}"
                    )
                    acc["video_path"] = vp
                    video_flags[lab] = True
                accepted.append(acc)


def reassess_p15_coupled() -> list[dict]:
    """Task-relevant metrics on P15 Coupled CSV (+ pickles when present)."""
    csv_path = P15 / "COUPLED_EXECUTION_RESULTS.csv"
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    out: list[dict] = []
    for r in rows:
        try:
            dx0, dz0 = float(r["dx0_mm"]), float(r["dz0_mm"])
            dx1, dz1 = float(r["dx1_mm"]), float(r["dz1_mm"])
        except (TypeError, ValueError, KeyError):
            continue
        e0 = float(np.sqrt(dx0 * dx0 + dz0 * dz0))  # xz proxy if e_pos missing
        e1 = float(np.sqrt(dx1 * dx1 + dz1 * dz1))
        # Prefer pickle e_pos if construction has it — still use xz for consistency with CSV
        both = (abs(dx0) - abs(dx1)) >= MIN_MM and (abs(dz0) - abs(dz1)) >= MIN_MM
        e_imp = (e0 - e1) >= MIN_MM
        n_act = int(float(r.get("n_actions") or 0))
        strict = str(r.get("SUCCESS_NEW")).lower() == "true"
        bilat = str(r.get("bilateral_exit")).lower() in ("true", "1", "")
        # Enrich from pickle if available
        pkl = P15 / "rejected_coupled_partial" / f"{r['tag']}.pkl"
        if not pkl.exists():
            pkl = P15 / "trajectories" / f"{r['tag']}.pkl"
        n_from_pkl = None
        if pkl.exists():
            try:
                raw = pickle.loads(pkl.read_bytes())
                acts = raw.get("actions") or (raw.get("meta") or {}).get("actions")
                if acts is not None:
                    n_from_pkl = len(acts)
            except Exception:
                pass
        if n_from_pkl is not None:
            n_act = n_from_pkl
        primary = "none"
        rdx, rdz = abs(dx0) - abs(dx1), abs(dz0) - abs(dz1)
        if rdx >= rdz and rdx >= MIN_MM:
            primary = "dx"
        elif rdz >= MIN_MM:
            primary = "dz"
        if both:
            primary = "both"
        # Unsuccessful trade-off: STRICT but e_pos not improved and not both-axis
        unsuccessful_tradeoff = strict and n_act > 0 and (not e_imp) and (not both)
        expert_accept = strict and n_act > 0 and (e_imp or both) and not unsuccessful_tradeoff
        out.append(
            {
                "tag": r.get("tag"),
                "base_sample_id": r.get("base_sample_id"),
                "requested_dx_mm": r.get("requested_dx_mm"),
                "requested_dz_mm": r.get("requested_dz_mm"),
                "dx0_mm": dx0,
                "dz0_mm": dz0,
                "dx1_mm": dx1,
                "dz1_mm": dz1,
                "e_pos0_xz_mm": e0,
                "e_pos1_xz_mm": e1,
                "both_axis_improved": both,
                "e_pos_improved": e_imp,
                "bilateral_exit": bilat,
                "SUCCESS_NEW": strict,
                "n_actions": n_act,
                "primary_improvement": primary,
                "unsuccessful_tradeoff": unsuccessful_tradeoff,
                "expert_accept_reassess": expert_accept,
                "status": r.get("status"),
                "demo_class_p15": r.get("demo_class"),
                "source": "p15_coupled_csv",
            }
        )
    return out


def discover_dz_bases(env, n: int = 3) -> list[dict]:
    """Reuse P15-style bases that stay bilateral for small DZ offsets."""
    from experiments.precision.precision_required_pilot_v2.run_pilot import discover_d2_bases

    rng = np.random.default_rng(SEED)
    pool = discover_d2_bases(rng)
    chosen: list[dict] = []
    for b in pool:
        if len(chosen) >= n:
            break
        raw = pickle.loads(b["path"].read_bytes())
        if "capture" not in raw:
            continue
        blob = {"capture": raw["capture"], "ref": raw["ref"], "sample_id": b["sample_id"]}
        info = verify_base(env, blob)
        if not (info["SUCCESS_12S"] and info["bilateral"] and float(info["e_pos_mm"]) <= 12):
            continue
        # Prefer bases with small |dz| so we can inject signed DZ cleanly
        if abs(float(info["dz_mm"])) > 8:
            continue
        chosen.append(blob)
        print(
            f"  DZ_BASE {b['sample_id'][:48]} e={info['e_pos_mm']:.2f} dz={info['dz_mm']:.2f}",
            flush=True,
        )
    # Always include DX-compatible base as well
    if DX_BASE_PATH.exists():
        dxb = load_base(DX_BASE_PATH, DX_BASE_ID)
        if not any(c["sample_id"] == DX_BASE_ID for c in chosen):
            chosen.insert(0, dxb)
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "ics").mkdir(parents=True, exist_ok=True)

    env = make_env()
    video_flags: dict[str, bool] = {}
    accepted: list[dict] = []
    coverage: list[dict] = []

    # ---- Part 1: DX expansion on compatible base ----
    print("=== P16 DX expansion ===", flush=True)
    if not DX_BASE_PATH.exists():
        raise FileNotFoundError(DX_BASE_PATH)
    dx_base = load_base(DX_BASE_PATH, DX_BASE_ID)
    info = verify_base(env, dx_base)
    print(f"  DX base 12S={info['SUCCESS_12S']} e={info['e_pos_mm']:.2f}", flush=True)
    dx_rows: list[dict] = []
    dx_specs: list[tuple[str, SmallErrorSpec]] = []
    for v in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -2.0, -3.0, -4.0, -5.0, -6.0):
        dx_specs.append(("dx", SmallErrorSpec(family="DX_EXPAND", target_dx_mm=v, target_dz_mm=0.0)))
    run_skill_matrix(
        env,
        [dx_base],
        dx_specs,
        rows_out=dx_rows,
        accepted=accepted,
        coverage=coverage,
        video_flags=video_flags,
    )
    _write_csv(OUT / "DX_EXPANSION.csv", dx_rows)

    # ---- Part 2: DZ expansion ----
    print("=== P16 DZ expansion ===", flush=True)
    dz_bases = discover_dz_bases(env, n=3)
    dz_rows: list[dict] = []
    dz_specs: list[tuple[str, SmallErrorSpec]] = []
    # Positive dz → inward; negative → outward (opening-axis convention)
    for v in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, -4.0, -6.0, -8.0, -10.0):
        dz_specs.append(("dz", SmallErrorSpec(family="DZ_EXPAND", target_dx_mm=0.0, target_dz_mm=v)))
    run_skill_matrix(
        env,
        dz_bases[:2],  # bounded
        dz_specs,
        rows_out=dz_rows,
        accepted=accepted,
        coverage=coverage,
        video_flags=video_flags,
    )
    _write_csv(OUT / "DZ_EXPANSION.csv", dz_rows)

    # ---- Part 3: Reassess P15 Coupled ----
    print("=== P16 Coupled reassessment ===", flush=True)
    reass = reassess_p15_coupled()
    _write_csv(OUT / "COUPLED_REASSESSMENT.csv", reass)
    n_epos = sum(1 for r in reass if r.get("e_pos_improved"))
    n_strict = sum(1 for r in reass if r.get("SUCCESS_NEW"))
    n_expert = sum(1 for r in reass if r.get("expert_accept_reassess"))
    n_both = sum(1 for r in reass if r.get("both_axis_improved"))
    print(
        f"  reassess n={len(reass)} e_pos_imp={n_epos} both={n_both} STRICT={n_strict} expert={n_expert}",
        flush=True,
    )

    # ---- Part 4: Limited Coupled generation ----
    print("=== P16 Coupled limited generation ===", flush=True)
    coupled_rows: list[dict] = []
    coup_specs: list[tuple[str, SmallErrorSpec]] = []
    for dx, dz in ((3.0, 4.0), (5.0, 5.0), (4.0, 7.0), (-3.0, -4.0), (-5.0, -5.0)):
        coup_specs.append(
            ("coupled", SmallErrorSpec(family="COUPLED_P16", target_dx_mm=dx, target_dz_mm=dz))
        )
    # Use DX-compatible base + one DZ base
    coup_bases = [dx_base]
    if len(dz_bases) > 1:
        coup_bases.append(dz_bases[1])
    run_skill_matrix(
        env,
        coup_bases,
        coup_specs,
        rows_out=coupled_rows,
        accepted=accepted,
        coverage=coverage,
        video_flags=video_flags,
    )
    # Append coupled gen summary into reassessment-style file section via separate write
    _write_csv(OUT / "COUPLED_P16_GENERATION.csv", coupled_rows)

    _write_csv(OUT / "PRECISION_SKILL_COVERAGE.csv", coverage)
    _write_csv(OUT / "STRICT_ACCEPTED_TRAJECTORIES.csv", accepted)

    # ---- Reports ----
    dx_ok = [r for r in dx_rows if r.get("ok_demo") in (True, "True")]
    dz_ok = [r for r in dz_rows if r.get("ok_demo") in (True, "True")]
    cp_ok = [r for r in coupled_rows if r.get("ok_demo") in (True, "True")]
    dx_mags = sorted({abs(float(r["requested_dx_mm"])) for r in dx_ok})
    dx_signs = sorted({np.sign(float(r["requested_dx_mm"])) for r in dx_ok})
    dz_dirs = Counter(r.get("demo_class") for r in dz_ok)
    uniq_bases_dx = {r["base_sample_id"] for r in dx_ok}
    uniq_bases_all = {r.get("base_sample_id") or r.get("same_base_lineage") for r in accepted}

    mean_unique = (
        float(np.mean([float(r.get("n_unique_rounded") or 0) for r in dx_ok])) if dx_ok else 0.0
    )

    _write_md(
        OUT / "LINEAGE_AUDIT.md",
        f"""# Lineage Audit (P16)

- Official dataset untouched
- Forced skill = generation metadata only (`policy_obs_note`)
- DX expansion: single verified compatible base `{DX_BASE_ID}` — variants are **same lineage**, different error offsets
- Unique base lineages in STRICT accepts: {len(uniq_bases_all)}
- Train/eval split guidance: group by `base_sample_id`; do not treat same-base offsets as independent ICs
- Diagnostic Coupled rollouts not promoted to official corpus
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P16 Precision Skill Expansion

Generated: {_utc()}

## Answers

1. **Useful DX range?**  
   STRICT DX accepts: **{len(dx_ok)}** from **{len(uniq_bases_dx)}** unique base lineage(s).  
   Magnitudes: {dx_mags}; signs: {list(dx_signs)}.  
   Mean unique rounded action tokens/traj: **{mean_unique:.1f}**.  
   Same-base offsets are **not** independent grasp lineages.

2. **DZ both directions?**  
   STRICT DZ accepts: **{len(dz_ok)}**; classes: `{dict(dz_dirs)}`.  
   Inward/outward coverage depends on signed residual + axial mode.

3. **P15 Coupled e_pos improved?**  
   Of {len(reass)} rollouts: **{n_epos}** improved xz/e_pos proxy; **{n_both}** both-axis; **{n_strict}** STRICT SUCCESS_NEW.

4. **Coupled STRICT success?**  
   P15 reassess expert-accept (STRICT + meaningful e_pos/both improvement): **{n_expert}**.  
   P16 new Coupled STRICT accepts: **{len(cp_ok)}**.

5. **Missing before SFT?**  
   - More **independent base lineages** for DX (currently one compatible base)  
   - Broader **DZ** coverage if inward/outward still sparse  
   - Coupled still weak for both-axis improvement; use only STRICT+e_pos-improved as optional demos  
   - Explicit train/eval split by base lineage; M111 retention check on accepts  
   - Do not admit diagnostic forced rollouts into official corpus until packaging review

## Videos
- DX: `{"yes" if video_flags.get("DX") else "none"}`
- DZ: `{"yes" if video_flags.get("DZ") else "none"}`
- Coupled: `{"yes" if video_flags.get("COUPLED") else "none"}`

## Integrity
No M111/physics/production-FSM/official-success changes. Forced Precision remains experimental.
""",
    )

    summary = {
        "generated_utc": _utc(),
        "n_dx_strict": len(dx_ok),
        "n_dz_strict": len(dz_ok),
        "n_coupled_p16_strict": len(cp_ok),
        "dx_unique_bases": len(uniq_bases_dx),
        "dx_magnitudes": dx_mags,
        "p15_coupled_e_pos_improved": n_epos,
        "p15_coupled_both_axis": n_both,
        "p15_coupled_strict": n_strict,
        "p15_coupled_expert_accept": n_expert,
        "n_accepted_total": len(accepted),
    }
    (OUT / "raw_p16_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done → {OUT} accepted={len(accepted)}", flush=True)


if __name__ == "__main__":
    main()
