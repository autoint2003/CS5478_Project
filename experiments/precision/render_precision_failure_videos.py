#!/usr/bin/env python
"""Render failure + paired Direct/Precision videos for Precision-required failures.

Initial hold uses a cloned capture so it cannot mutate the Oracle start state.
Official Oracle: use_verified_lateral_fallback=False. Outcomes are not staged.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.mechanics.open_axis import gates  # noqa: E402
from data_generation.oracles.precision.mechanics.support_verification import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

AUDIT = ROOT / "results/diagnostics/precision_candidate_audit"
OUT = AUDIT / "VIDEOS"
FPS = 20
HOLD_S = 1.0
DT = 0.02


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def cfg() -> PrecisionConfig:
    return PrecisionConfig(
        evaluate_handoff=True,
        use_support_transfer_dx=True,
        use_verified_lateral_fallback=False,
    )


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = np.asarray(img, np.uint8).copy()
    bar = min(out.shape[0], 14 + 14 * len(lines))
    out[:bar] = (12, 12, 12)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 2
        for line in lines:
            dr.text((4, y), line[:110], fill=(255, 255, 255))
            y += 13
        return np.asarray(im)
    except Exception:
        return out


def hud(env, *, ic_id: str, phase: str, t: float, fail: str | None = None) -> list[str]:
    err = measure_errors(env.sim)
    g = gates(env.sim)
    mode = classify_contact_mode(env.sim)
    e = np.asarray(err.get("e_signed_mm") or [0, 0, 0], float)
    lines = [
        f"IC={ic_id}",
        f"t={t:.2f}s  phase={phase}" + (f"  FAIL={fail}" if fail else ""),
        f"dθ={float(err.get('dtheta_deg') or 0):.1f}  dx={e[0]:.1f} dy={e[1]:.1f} dz={e[2]:.1f}  e_pos={float(err.get('e_pos_mm') or 0):.1f}",
        f"DOWN={bool(g.get('fingertips_down'))} UP={bool(g.get('fingertips_up'))} α={float(g.get('alpha_up_deg') or 0):.1f}",
        f"bil={bool(err.get('bilateral'))} FnL={float(err.get('fn_L') or 0):.1f} FnR={float(err.get('fn_R') or 0):.1f} "
        f"FnzL={float(mode.get('Fn_z_L') or 0):.1f} FnzR={float(mode.get('Fn_z_R') or 0):.1f} S1={mode.get('support_side')}",
    ]
    return lines


def render_pair(env, cam, renderer, lines: list[str]) -> np.ndarray:
    po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
    ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
    cam.lookat[:] = 0.55 * po + 0.45 * ph
    renderer.update_scene(env.sim.data, cam)
    return overlay(np.asarray(renderer.render(), np.uint8), lines)


def hold_frames(env, blob, cam, renderer, ic_id: str, hold_s: float = HOLD_S) -> list[np.ndarray]:
    """Hold on a cloned capture; caller must restore Oracle start afterwards."""
    frames = []
    clone = take_capture(env)  # already at blob if restored
    restore_capture(env, clone)
    n = max(1, int(round(hold_s / DT)))
    t0 = float(env.sim.data.time)
    for i in range(n):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(env.sim, a)
        if i % 2 == 0 or i == n - 1:
            t = float(env.sim.data.time) - t0
            img = render_pair(env, cam, renderer, hud(env, ic_id=ic_id, phase="INITIAL_HOLD_CLONE", t=t))
            frames.append(img)
    restore_capture(env, blob["capture"])  # critical: Oracle start unchanged
    return frames


def replay_actions_video(
    env,
    blob,
    actions: np.ndarray,
    cam,
    renderer,
    ic_id: str,
    *,
    phase_prefix: str,
    fail_reason: str | None,
    max_frames: int = 400,
) -> list[np.ndarray]:
    frames = []
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    n = len(actions)
    stride = max(1, n // max_frames) if n else 1
    lost_at = None
    for i, a in enumerate(actions):
        step_legal(env.sim, np.asarray(a, np.float32))
        err = measure_errors(env.sim)
        if lost_at is None and not err.get("bilateral"):
            lost_at = i
        if i % stride == 0 or i == n - 1 or (lost_at is not None and i == lost_at):
            t = float(env.sim.data.time) - t0
            phase = phase_prefix
            fail = None
            if lost_at is not None and i >= lost_at:
                phase = f"{phase_prefix}|CONTACT_LOSS"
                fail = fail_reason or "contact_loss"
            elif i == n - 1 and fail_reason:
                fail = fail_reason
            frames.append(render_pair(env, cam, renderer, hud(env, ic_id=ic_id, phase=phase, t=t, fail=fail)))
    # short outcome settle
    for _ in range(int(0.6 / DT)):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(env.sim, a)
    t = float(env.sim.data.time) - t0
    frames.append(
        render_pair(
            env,
            cam,
            renderer,
            hud(env, ic_id=ic_id, phase="OUTCOME", t=t, fail=fail_reason),
        )
    )
    return frames


def render_failure_case(env, row: dict, cam, renderer) -> dict[str, Any]:
    sid = row["sample_id"]
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    prec = row.get("precision") or {}
    fail = prec.get("first_failed_primitive") or (prec.get("failures") or ["unknown"])[0]
    dest = OUT / "failures" / f"{sid}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Re-run Oracle to obtain live actions if needed
    actions = None
    act_rel = prec.get("actions_path")
    if act_rel and (ROOT / act_rel).is_file():
        actions = np.load(ROOT / act_rel)["actions"]
    restore_capture(env, blob["capture"])
    res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, str(row["plan_family"]), cfg())
    if actions is None or len(actions) == 0:
        actions = np.asarray(res.actions or [], np.float32)
    if len(actions) == 0:
        # Still render hold + static fail frame
        restore_capture(env, blob["capture"])
        frames = hold_frames(env, blob, cam, renderer, sid)
        frames.append(
            render_pair(
                env,
                cam,
                renderer,
                hud(env, ic_id=sid, phase=str(res.status), t=0.0, fail=str(fail)),
            )
        )
        for _ in range(FPS):
            frames.append(frames[-1])
        path = write_frames_mp4(frames, dest, fps=FPS)
        return {"sample_id": sid, "video": path, "n_actions": 0, "status": res.status, "fail": fail}

    restore_capture(env, blob["capture"])
    frames = hold_frames(env, blob, cam, renderer, sid)
    frames.extend(
        replay_actions_video(
            env,
            blob,
            actions,
            cam,
            renderer,
            sid,
            phase_prefix="PRECISION_ORACLE",
            fail_reason=str(fail),
        )
    )
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {
        "sample_id": sid,
        "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
        "n_actions": int(len(actions)),
        "status": res.status,
        "fail": fail,
        "absolute": path,
    }


def render_paired(env, row: dict, cam, renderer) -> dict[str, Any]:
    sid = row["sample_id"]
    blob = pickle.loads((ROOT / row["ic_path"]).read_bytes())
    out_a = OUT / "paired" / f"{sid}__DIRECT.mp4"
    out_b = OUT / "paired" / f"{sid}__PRECISION.mp4"
    out_a.parent.mkdir(parents=True, exist_ok=True)

    # Video A: direct baseline — hold clone + handoff probe visualization as hold after apply?
    # We visualize: hold, then run probe (state restored after). Show pre-handoff residuals
    # then a short nominal-like grip hold annotated with direct outcome.
    restore_capture(env, blob["capture"])
    frames_a = hold_frames(env, blob, cam, renderer, sid)
    direct = probe_direct_handoff(env, blob["capture"], blob["ref"])
    restore_capture(env, blob["capture"])
    for _ in range(FPS):
        frames_a.append(
            render_pair(
                env,
                cam,
                renderer,
                hud(
                    env,
                    ic_id=sid,
                    phase=f"DIRECT_BASELINE NEW={bool(direct.get('SUCCESS_NEW'))} 12S={bool(direct.get('SUCCESS_12S'))}",
                    t=0.0,
                    fail=None if direct.get("SUCCESS_NEW") else str(direct.get("reason")),
                ),
            )
        )
    path_a = write_frames_mp4(frames_a, out_a, fps=FPS)

    # Video B: official Precision
    restore_capture(env, blob["capture"])
    res = run_precision(env, {"capture": blob["capture"], "ref": blob["ref"]}, str(row["plan_family"]), cfg())
    actions = np.asarray(res.actions or [], np.float32)
    restore_capture(env, blob["capture"])
    frames_b = hold_frames(env, blob, cam, renderer, sid)
    fail = None
    if not (res.handoff or {}).get("SUCCESS_NEW"):
        fail = (res.failures or [res.status] or ["fail"])[0]
    if len(actions):
        frames_b.extend(
            replay_actions_video(
                env,
                blob,
                actions,
                cam,
                renderer,
                sid,
                phase_prefix="PRECISION",
                fail_reason=str(fail) if fail else None,
            )
        )
    else:
        for _ in range(FPS):
            frames_b.append(
                render_pair(
                    env,
                    cam,
                    renderer,
                    hud(env, ic_id=sid, phase=str(res.status), t=0.0, fail=str(fail) if fail else None),
                )
            )
    path_b = write_frames_mp4(frames_b, out_b, fps=FPS)
    return {
        "sample_id": sid,
        "direct_video": str(Path(path_a).relative_to(ROOT)).replace("\\", "/") if path_a else None,
        "precision_video": str(Path(path_b).relative_to(ROOT)).replace("\\", "/") if path_b else None,
        "direct_SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
        "direct_SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
        "precision_SUCCESS_NEW": bool((res.handoff or {}).get("SUCCESS_NEW")),
        "precision_status": res.status,
    }


def load_failed_rows() -> list[dict]:
    rows = []
    path = AUDIT / "CLASSIFICATION.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("class") != "PRECISION_REQUIRED_BUT_FAILED":
            continue
        # Skip success-without-skill subclass for failure videos (not a physical Oracle fail)
        if row.get("subclass") == "success_without_correction_skill":
            continue
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max failure videos (0=all)")
    ap.add_argument("--paired", type=int, default=8, help="Number of paired comparison videos")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    failed = load_failed_rows()
    if args.limit and args.limit > 0:
        failed = failed[: args.limit]

    env = make_env()
    cam = mjv_camera_from_preset({"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20})
    renderer = mujoco.Renderer(env.sim.model, 480, 640)

    index = {"generated_utc": datetime.now(timezone.utc).isoformat(), "failures": [], "paired": []}
    print(f"rendering {len(failed)} failure videos...", flush=True)
    for i, row in enumerate(failed):
        print(f"[fail {i+1}/{len(failed)}] {row['sample_id']}", flush=True)
        try:
            meta = render_failure_case(env, row, cam, renderer)
            index["failures"].append(meta)
        except Exception as exc:
            index["failures"].append({"sample_id": row["sample_id"], "error": str(exc)})
            print(f"  ERROR {exc}", flush=True)

    # Paired: mix of both_fail and direct_fail_precision cases
    paired_src = failed[: max(0, int(args.paired))]
    print(f"rendering {len(paired_src)} paired videos...", flush=True)
    for i, row in enumerate(paired_src):
        print(f"[pair {i+1}/{len(paired_src)}] {row['sample_id']}", flush=True)
        try:
            index["paired"].append(render_paired(env, row, cam, renderer))
        except Exception as exc:
            index["paired"].append({"sample_id": row["sample_id"], "error": str(exc)})

    renderer.close()
    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'INDEX.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
