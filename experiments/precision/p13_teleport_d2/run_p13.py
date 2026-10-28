"""P13 bounded teleport-D2 pilot.

TELEPORT → SETTLE → VALIDATE → FREEZE → independent 12S (clone) → Precision.
Official dataset untouched. STRICT admit. No RL / physics / M111 changes.
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

from data_generation.expert_pipeline.live_stage1 import (  # noqa: E402
    extract_precision_skills,
    generate_d2_from_synthetic_ic,
)
from data_generation.oracles.precision.d2_eligibility import check_d2_eligible  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from experiments.precision.p13_teleport_d2.teleport_builder import (  # noqa: E402
    TeleportSpec,
    build_teleport_ic,
    clone_capture,
    frozen_ic_blob,
)
from experiments.precision.precision_required_pilot_v2.run_pilot import (  # noqa: E402
    discover_d2_bases,
)

OUT = ROOT / "results" / "diagnostics" / "p13_teleport_d2"
SEED = 13
MAX_BASES = 3
MAX_PRECISION_RUNS = 24  # cap expensive Precision after gate


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


def structured_grid() -> list[TeleportSpec]:
    """Coarse signed grid covering DX / DZ / DX-DZ / DZ-DX families."""
    specs: list[TeleportSpec] = []
    # D2-DX: large |dx|, small |dz|
    for dx in (10.0, 14.0, -10.0, -14.0):
        for dz in (0.0, 2.0, -2.0):
            specs.append(TeleportSpec(family="D2-DX", dx_mm=dx, dz_mm=dz, s_axial_mm=0.0))
    # D2-DZ: large |dz|, small |dx|
    for dz in (14.0, 18.0, -14.0, -18.0):
        for dx in (0.0, 2.0, -2.0):
            specs.append(TeleportSpec(family="D2-DZ", dx_mm=dx, dz_mm=dz, s_axial_mm=0.0))
    # D2-DX-DZ: both nonzero, |dx| > |dz|
    for dx, dz in ((12.0, -8.0), (-12.0, 8.0), (14.0, -6.0), (-14.0, -6.0)):
        specs.append(TeleportSpec(family="D2-DX-DZ", dx_mm=dx, dz_mm=dz, s_axial_mm=2.0))
    # D2-DZ-DX: both nonzero, |dz| > |dx|
    for dx, dz in ((4.0, -16.0), (-4.0, 16.0), (3.0, -18.0), (-3.0, 18.0)):
        specs.append(TeleportSpec(family="D2-DZ-DX", dx_mm=dx, dz_mm=dz, s_axial_mm=-2.0))
    # mild orientation variants on a few cells
    for fam, dx, dz, th in (
        ("D2-DX", 12.0, 1.0, 8.0),
        ("D2-DZ", 1.0, -16.0, -6.0),
    ):
        specs.append(TeleportSpec(family=fam, dx_mm=dx, dz_mm=dz, dtheta_deg=th))
    return specs


def render_accepted_video(env, blob: dict, actions: list, dest: Path, title: str) -> str | None:
    """Best-effort short video of frozen IC + Precision actions."""
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
        for i in range(10):
            a = np.zeros(7, np.float32)
            a[6] = 1.0
            step_legal(env.sim, a)
            renderer.update_scene(env.sim.data, cam)
            err = measure_errors(env.sim)
            frames.append(
                overlay(
                    renderer.render(),
                    [title, f"hold t={i*0.02:.2f}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"],
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
                        [title, f"prec i={i}", f"dx={err['dx_mm']:.1f} dz={err['dz_mm']:.1f}"],
                    )
                )
        if not frames:
            return None
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
    rng = np.random.default_rng(SEED)
    bases = discover_d2_bases(rng)[:MAX_BASES]
    specs = structured_grid()

    construction_log: list[dict] = []
    stability_rows: list[dict] = []
    skill_rows: list[dict] = []
    accepted: list[dict] = []
    seen_lineage: set[str] = set()
    n_prec = 0

    print(f"[P13] bases={len(bases)} grid={len(specs)}", flush=True)

    for bi, base in enumerate(bases):
        raw = pickle.loads(base["path"].read_bytes())
        base_blob = {
            "capture": raw["capture"],
            "ref": raw["ref"],
            "sample_id": base["sample_id"],
        }
        for si, spec in enumerate(specs):
            t0 = time.time()
            tag = f"{spec.family}_b{bi}_s{si}"
            print(
                f"  [{tag}] cmd dx={spec.dx_mm} dz={spec.dz_mm} th={spec.dtheta_deg}",
                flush=True,
            )
            built = build_teleport_ic(env, base_blob, spec, cfg=cfg)
            row = built.to_log_row()
            row["tag"] = tag
            row["elapsed_s"] = round(time.time() - t0, 2)
            construction_log.append(row)

            if not built.ok:
                print(f"    build fail @{built.stage}: {built.reason}", flush=True)
                continue

            # Duplicate settled niche?
            if built.lineage_key in seen_lineage:
                row["duplicate_lineage"] = True
                construction_log[-1]["duplicate_lineage"] = True
                print(f"    duplicate lineage {built.lineage_key}", flush=True)
                continue
            seen_lineage.add(built.lineage_key or "")

            if not built.precision_required:
                construction_log[-1]["reject"] = "SETTLED_NOT_PRECISION_REQUIRED"
                print("    settled but not Precision-required", flush=True)
                continue

            # Independent 12S on CLONE — do not advance frozen IC
            frozen = frozen_ic_blob(built)
            clone_blob = {
                "capture": clone_capture(frozen["capture"]),
                "ref": frozen["ref"],
                "sample_id": frozen["sample_id"],
            }
            elig = check_d2_eligible(env, clone_blob, cfg=cfg)
            # Restore frozen IC after gate (gate restores capture but be explicit)
            restore_capture(env, frozen["capture"])
            stab = {
                "tag": tag,
                "sample_id": frozen["sample_id"],
                "lineage_key": built.lineage_key,
                "capture_fp": built.capture_fp,
                "family": spec.family,
                "SUCCESS_12S": elig.SUCCESS_12S,
                "SUCCESS_STRICT_10DEG": elig.SUCCESS_STRICT_10DEG,
                "SUCCESS_RELAXED_15DEG": elig.SUCCESS_RELAXED_15DEG,
                "d2_status": elig.status,
                "precision_required_gate": elig.precision_required,
                "eligible": elig.eligible,
                "settled_dx_mm": built.settled_errors.get("dx_mm"),
                "settled_dz_mm": built.settled_errors.get("dz_mm"),
                "settled_dtheta_deg": built.settled_errors.get("dtheta_deg"),
                "fsm_case_at_freeze": built.fsm_case,
                "verification": "clone_capture_independent_12s",
            }
            stability_rows.append(stab)
            if not elig.eligible:
                print(f"    12S gate {elig.status}", flush=True)
                continue

            if n_prec >= MAX_PRECISION_RUNS:
                print("    skip Precision (budget)", flush=True)
                continue
            n_prec += 1

            # Save frozen IC pickle (construction metadata separate from actions)
            ic_path = OUT / "ics" / f"{tag}.pkl"
            ic_path.write_bytes(
                pickle.dumps(
                    {
                        **frozen,
                        "construction_only": True,
                        "note": "Expert must start at capture; do not teleport after freeze",
                    },
                    protocol=4,
                )
            )

            live = generate_d2_from_synthetic_ic(
                env,
                {
                    "capture": frozen["capture"],
                    "ref": frozen["ref"],
                    "sample_id": frozen["sample_id"],
                },
                raw_scenario_id=frozen["sample_id"],
                construction_meta={
                    **(built.construction or {}),
                    "lineage_key": built.lineage_key,
                    "capture_fp": built.capture_fp,
                    "p13_teleport": True,
                },
                precision_config=cfg,
            )
            sk = extract_precision_skills(list(live.precision_phases or []))
            trans = _norm_transition(sk)
            skill_rows.append(
                {
                    "tag": tag,
                    "family": spec.family,
                    "lineage_key": built.lineage_key,
                    "planned_fsm": built.fsm_case,
                    "planned_skills": "|".join(built.fsm_planned or []),
                    "executed_skills": "|".join(sk),
                    "transition_normalized": trans,
                    "precision_status": live.precision_status,
                    "accepted_strict": bool(live.accepted),
                    "SUCCESS_NEW": live.SUCCESS_NEW,
                    "SUCCESS_12S_final": live.SUCCESS_12S_final,
                    "n_precision_actions": live.n_precision_actions,
                }
            )
            print(
                f"    Precision {live.precision_status} trans={trans} ok={live.accepted}",
                flush=True,
            )
            if live.accepted:
                video = None
                # one video per normalized transition family (first win)
                want_label = {
                    "dx": "DX-only",
                    "dz": "DZ-only",
                    "dx→dz": "DX→DZ",
                    "dz→dx": "DZ→DX",
                }.get(trans)
                # also allow dx→dz→… shortened
                if want_label is None:
                    if trans.startswith("dx→dz"):
                        want_label = "DX→DZ"
                    elif trans.startswith("dz→dx"):
                        want_label = "DZ→DX"
                    elif trans == "dx" or trans.startswith("dx→"):
                        want_label = "DX-only" if "dz" not in trans else None
                    elif trans == "dz" or (trans.startswith("dz") and "dx" not in trans):
                        want_label = "DZ-only"
                existing = {a.get("video_label") for a in accepted}
                if want_label and want_label not in existing:
                    vpath = OUT / "videos" / f"{want_label.replace('→', '_to_')}.mp4"
                    video = render_accepted_video(
                        env,
                        frozen,
                        live.precision_actions or [],
                        vpath,
                        f"{want_label} | {trans}",
                    )
                accepted.append(
                    {
                        "tag": tag,
                        "sample_id": frozen["sample_id"],
                        "family": spec.family,
                        "lineage_key": built.lineage_key,
                        "capture_fp": built.capture_fp,
                        "cmd_dx_mm": spec.dx_mm,
                        "cmd_dz_mm": spec.dz_mm,
                        "settled_dx_mm": built.settled_errors.get("dx_mm"),
                        "settled_dz_mm": built.settled_errors.get("dz_mm"),
                        "transition_normalized": trans,
                        "executed_skills": "|".join(sk),
                        "planned_fsm": built.fsm_case,
                        "SUCCESS_NEW": True,
                        "admit_success_key": "SUCCESS_NEW",
                        "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
                        "video_label": want_label,
                        "video_path": video,
                        "independent_new": True,
                        "synthetic": True,
                    }
                )

    # ---- reports ----
    _write_csv(OUT / "IC_CONSTRUCTION_LOG.csv", construction_log)
    _write_csv(OUT / "STABILITY_GATE_RESULTS.csv", stability_rows)
    _write_csv(OUT / "PRECISION_SKILL_COVERAGE.csv", skill_rows)
    _write_csv(OUT / "STRICT_ACCEPTED_D2.csv", accepted)

    n_build_ok = sum(1 for r in construction_log if r.get("ok") in (True, "True"))
    n_12 = sum(1 for r in stability_rows if r.get("eligible") in (True, "True"))
    by_fam_acc = Counter(r.get("family") for r in accepted)
    by_trans = Counter(r.get("transition_normalized") for r in accepted)
    build_fail = Counter(r.get("reason") for r in construction_log if not r.get("ok"))

    yield_by_family = []
    for fam in ("D2-DX", "D2-DZ", "D2-DX-DZ", "D2-DZ-DX"):
        tried = sum(1 for r in construction_log if r.get("family") == fam)
        okb = sum(1 for r in construction_log if r.get("family") == fam and r.get("ok") in (True, "True"))
        acc = sum(1 for r in accepted if r.get("family") == fam)
        yield_by_family.append(f"- **{fam}**: tried={tried}, freeze_ok={okb}, STRICT_accept={acc}")

    _write_md(
        OUT / "TELEPORT_IC_BUILDER.md",
        f"""# Teleport IC Builder (P13)

## Boundary
`TELEPORT_CONSTRUCTION → SETTLE → VALIDATE → FREEZE_IC → PRECISION_EXPERT`

After **FREEZE_IC**:
- No teleport / qpos / qvel rewrite
- No privileged forces or pose reset
- Expert actions only via `step_legal` / Precision FSM

Offsets are **hand-frame** DX/DY/DZ (mm) plus optional **object-axis** `s_axial_mm`.
Not world-frame. See `teleport_builder.TeleportSpec`.

## Implementation
`experiments/precision/p13_teleport_d2/teleport_builder.py`

Rejects: severe penetration (>4 mm), lost bilateral after settle, object too low, high slip.

## Pilot construction fails
{json.dumps(dict(build_fail), indent=2)}
""",
    )

    _write_md(
        OUT / "LINEAGE_AND_FINGERPRINT_AUDIT.md",
        f"""# Lineage and Fingerprint Audit (P13)

- Official `data/experts_v2_official` **not modified**
- Each freeze gets `capture_fp` + `lineage_key` (quantized settled dx/dz/θ)
- Duplicate `lineage_key` skipped (not independent)
- Unique lineages attempted after freeze: {len(seen_lineage)}
- STRICT accepted: {len(accepted)}
- 12S verification uses **cloned** capture (`verification=clone_capture_independent_12s`)

Do not admit to official until capture_fp uniqueness vs existing D2 is checked.
""",
    )

    videos = [a for a in accepted if a.get("video_path") and not str(a.get("video_path")).startswith("VIDEO_FAIL")]
    _write_md(
        OUT / "VIDEO_INDEX.md",
        "# VIDEO_INDEX — P13\n\n"
        + (
            "\n".join(
                f"- **{a.get('video_label')}** (`{a.get('transition_normalized')}`): `{a.get('video_path')}`"
                for a in accepted
                if a.get("video_label")
            )
            or "No videos (no STRICT accepts or render failed)."
        )
        + f"\n\nRendered ok: {len(videos)}\n",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P13 Teleport-Based D2 IC Generation

Generated: {_utc()}

## Pipeline
Explicit teleport builder with FREEZE boundary; independent clone 12S gate; existing Precision FSM; STRICT admit.

## Pilot counts
| Stage | n |
|---|---:|
| Construction attempts | {len(construction_log)} |
| Freeze OK | {n_build_ok} |
| Independent 12S eligible | {n_12} |
| Precision runs | {n_prec} |
| STRICT accepted | {len(accepted)} |

## Yield by commanded family
{chr(10).join(yield_by_family)}

## Accepted transitions (executed, normalized)
{json.dumps(dict(by_trans), indent=2)}

`dz_inward`/`dz_outward` count as **dz**, not dx. `support_transfer` → **dx**.

## Highest-yield families
Prefer families with freeze_ok ∧ STRICT_accept > 0 in the list above; densify those signed-offset ranges next.
Stop expanding cells that systematically lose bilateral contact or fail 12S.

## Stop conditions respected
No production physics/Oracle/M111 change; no RL; no STRICT weakening; no official auto-admit.
""",
    )

    (OUT / "raw_p13_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "n_construction": len(construction_log),
                "n_freeze_ok": n_build_ok,
                "n_12s_eligible": n_12,
                "n_precision": n_prec,
                "n_accepted": len(accepted),
                "by_family_accepted": dict(by_fam_acc),
                "by_transition": dict(by_trans),
                "build_fail_reasons": dict(build_fail),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done → {OUT} accepted={len(accepted)}", flush=True)


if __name__ == "__main__":
    main()
