#!/usr/bin/env python
"""Rebuild mechanism-focused Recovery vs Nominal videos under stage2_abc_final_v2/videos.

Includes visual QA (scene brightness / dark fraction) and contact sheets.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.corrections.contact_primitives import _wrist
from data_generation.oracles.precision.state_machine import measure_errors
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from models.transformer_policy import DT_POLICY
from visualization.camera import mjv_camera_from_preset
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results/data_generation/stage2_abc_final_v2"
VID = OUT / "videos"
V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
V2 = ROOT / "results/data_generation/stage2_abc_mining/v2"
FINAL_V1 = ROOT / "results/data_generation/stage2_abc_final"
SAMPLE_S = 0.04
DT = 0.02
H, W = 480, 640


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    # Brighten scene for offscreen rendering
    try:
        env.sim.model.vis.headlight.ambient[:] = [0.4, 0.4, 0.4]
        env.sim.model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        env.sim.model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
    except Exception:
        pass
    return env


def resolve_ic(sample_id: str) -> Path:
    for base in (OUT, FINAL_V1, V2, V1):
        p = base / "ics" / f"{sample_id}.pkl"
        if p.exists():
            return p
    raise FileNotFoundError(sample_id)


def resolve_actions(sample_id: str) -> np.ndarray:
    for base in (OUT, FINAL_V1, V2, V1):
        for p in (base / "continuous").glob(f"{sample_id}*"):
            z = np.load(p)
            if "actions" in z:
                return np.asarray(z["actions"], np.float32)
        op = base / "outcomes" / f"{sample_id}.json"
        if op.exists():
            o = json.loads(op.read_text(encoding="utf-8"))
            rel = (o.get("continuous_demo") or {}).get("path") or o.get("continuous_npz")
            if rel and (base / rel).exists():
                return np.asarray(np.load(base / rel)["actions"], np.float32)
    return np.zeros((0, 7), np.float32)


def resolve_lift_actions(sample_id: str) -> np.ndarray:
    p = OUT / "lift" / f"{sample_id}__lift.npz"
    if p.exists():
        return np.asarray(np.load(p)["actions"], np.float32)
    return np.zeros((0, 7), np.float32)


def resolve_outcome(sample_id: str) -> dict:
    for base in (OUT, FINAL_V1, V2, V1):
        p = base / "outcomes" / f"{sample_id}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def overlay(frame, lines: list[str]) -> np.ndarray:
    img = np.asarray(frame, np.uint8).copy()
    bar = min(110, max(72, 16 * (len(lines) + 1)))
    img[-bar:] = (12, 12, 12)
    try:
        from PIL import Image, ImageDraw, ImageFont

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        y = img.shape[0] - bar + 6
        for ln in lines:
            dr.text((8, y), ln, fill=(255, 255, 255))
            y += 14
        return np.asarray(im)
    except Exception:
        return img


def shoot(sim, cam, renderer):
    obj = np.array(sim.data.xpos[sim.ids.object_body], float)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    cam.lookat[:] = 0.5 * (obj + hand)
    # Keep distance in a visible band
    cam.distance = float(np.clip(0.45 + 0.3 * abs(obj[2] - 0.5), 0.40, 0.85))
    renderer.update_scene(sim.data, cam)
    return np.asarray(renderer.render(), np.uint8).copy()


def run_branch(env, cam, renderer, cap, ref, prelude, recovery, *, plan, sid, branch, phase_hint=""):
    restore_capture(env, cap)
    # If prelude (shared lift) provided, first restore catch and play prelude then this is wrong —
    # For C we pass post-lift cap as IC and empty prelude; lift shown separately.
    t0 = float(env.sim.data.time)
    frames = []
    next_t = t0
    handoff_done = False
    succ12 = succ_new = False
    phase = phase_hint or ("recovery" if len(recovery) else "direct")

    def stamp(ph):
        nonlocal next_t, succ12, succ_new
        now = float(env.sim.data.time)
        if now + 1e-9 < next_t:
            return
        err = measure_errors(env.sim)
        img = shoot(env.sim, cam, renderer)
        lines = [
            f"{branch} | Plan {plan}",
            f"{sid[:42]}",
            f"phase={ph}  t={now - t0:.2f}s  HANDOFF={'Y' if handoff_done else 'N'}",
            f"e_pos={err['e_pos_mm']:.1f}mm  dth={err['dtheta_deg']:.1f}deg  "
            f"ex={err['e_signed_mm'][0]:.1f} ez={err['e_signed_mm'][2]:.1f}",
            f"contact={'BIL' if err['bilateral'] else 'NO'}  "
            f"fnL/R={err['fn_L']:.1f}/{err['fn_R']:.1f}  obj_z={env.sim.data.xpos[env.sim.ids.object_body][2]:.3f}",
            f"SUCCESS_12S={'Y' if succ12 else 'N'}  SUCCESS_NEW={'Y' if succ_new else 'N'}",
        ]
        frames.append(overlay(img, lines))
        next_t = now + SAMPLE_S

    for a in prelude:
        step_legal(env.sim, np.asarray(a, np.float32))
        stamp("shared_lift")
    for a in recovery:
        step_legal(env.sim, np.asarray(a, np.float32))
        stamp("recovery")
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
        stamp("stabilize")
    apply_handoff(env.sim, ref)
    handoff_done = True
    stamp("HANDOFF")
    for _ in range(int(round(NOMINAL_S / DT))):
        nominal_step(env.sim)
        stamp("nominal_12s")
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    succ12 = bool(h.get("SUCCESS_12S"))
    succ_new = bool(h.get("SUCCESS_NEW"))
    stamp("FINAL")
    return frames, {
        "SUCCESS_12S": succ12,
        "SUCCESS_NEW": succ_new,
        "reason": h.get("std_reason") or h.get("reason"),
        "n_recovery": int(len(recovery)),
        "n_prelude": int(len(prelude)),
        "sim_time_s": float(env.sim.data.time) - t0,
    }


def pad_pair(left, right):
    n = max(len(left), len(right))
    while len(left) < n:
        left.append(left[-1])
    while len(right) < n:
        right.append(right[-1])
    return [np.hstack([a, b]) for a, b in zip(left, right)]


def scene_stats(frame: np.ndarray) -> dict:
    arr = np.asarray(frame)
    h = arr.shape[0]
    # exclude bottom overlay band
    scene = arr[: int(h * 0.75), :, :3]
    mean = float(scene.mean())
    dark = float((scene.max(axis=2) < 12).mean())
    return {"mean": mean, "dark_frac": dark, "shape": list(arr.shape)}


def qa_video(path: Path, min_frames: int = 20) -> dict:
    tmp = path.parent / f"_qa_{path.stem}"
    tmp.mkdir(exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    cmd = [
        exe,
        "-y",
        "-i",
        str(path),
        "-vf",
        r"select=not(mod(n\,20))",
        "-vsync",
        "vfr",
        "-frames:v",
        "12",
        str(tmp / "f_%03d.png"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    pngs = sorted(tmp.glob("*.png"))
    from PIL import Image

    stats = []
    for p in pngs:
        im = np.asarray(Image.open(p))
        stats.append(scene_stats(im))
    means = [s["mean"] for s in stats] or [0.0]
    darks = [s["dark_frac"] for s in stats] or [1.0]
    # also estimate frame count via ffprobe-ish: decode all as count
    n_est = len(pngs) * 20
    ok = (
        path.is_file()
        and path.stat().st_size > 50_000
        and len(pngs) >= 3
        and float(np.mean(means)) >= 25.0
        and float(np.mean(darks)) <= 0.55
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "ok": ok,
        "ffmpeg_rc": r.returncode,
        "n_sample_frames": len(pngs),
        "est_frames": n_est,
        "bytes": path.stat().st_size if path.exists() else 0,
        "mean_brightness": round(float(np.mean(means)), 2),
        "mean_dark_frac": round(float(np.mean(darks)), 3),
        "samples": stats[:4],
        "fail_reasons": ([] if ok else [
            x
            for x, cond in [
                ("small_file", path.stat().st_size <= 50_000),
                ("too_few_samples", len(pngs) < 3),
                ("too_dark", float(np.mean(means)) < 25.0),
                ("mostly_black", float(np.mean(darks)) > 0.55),
            ]
            if cond
        ]),
    }


def contact_sheet(video_path: Path, dest: Path, n: int = 6):
    tmp = video_path.parent / f"_cs_{video_path.stem}"
    tmp.mkdir(exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    # evenly spaced
    subprocess.run(
        [
            exe,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            rf"select=not(mod(n\,max(1\,n/{n}))),scale=320:-1",
            "-vsync",
            "vfr",
            "-frames:v",
            str(n),
            str(tmp / "c_%02d.png"),
        ],
        capture_output=True,
    )
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in sorted(tmp.glob("*.png"))]
    if not imgs:
        return None
    w, h = imgs[0].size
    sheet = Image.new("RGB", (w * len(imgs), h), (0, 0, 0))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * w, 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest


def render_case(
    *,
    tag: str,
    sample_id: str,
    plan: str,
    mechanism: str,
    shared_lift: bool = False,
    catch_id: str | None = None,
):
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    if shared_lift and catch_id:
        # Start both branches from catch, play same lift, then diverge at post-lift
        catch_pack = pickle.loads(resolve_ic(catch_id).read_bytes())
        lift_pack = pickle.loads(resolve_ic(sample_id).read_bytes())
        cap_catch, ref = catch_pack["capture"], catch_pack["ref"]
        cap_lift = lift_pack["capture"]
        lift_acts = resolve_lift_actions(sample_id)
        rec_acts = resolve_actions(sample_id)
        # Camera from catch
        restore_capture(env, cap_catch)
        look = 0.5 * (
            np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        cam = mjv_camera_from_preset(
            {"lookat": look, "distance": 0.65, "azimuth": 140.0, "elevation": -25.0}
        )
        renderer = mujoco.Renderer(env.sim.model, H, W)
        left, left_out = run_branch(
            env, cam, renderer, cap_catch, ref, lift_acts, np.zeros((0, 7), np.float32),
            plan=plan, sid=sample_id, branch="DIRECT (post-lift HANDOFF)",
        )
        right, right_out = run_branch(
            env, cam, renderer, cap_catch, ref, lift_acts, rec_acts,
            plan=plan, sid=sample_id, branch="RECOVERY (post-lift Precision)",
        )
    else:
        pack = pickle.loads(resolve_ic(sample_id).read_bytes())
        cap, ref = pack["capture"], pack["ref"]
        acts = resolve_actions(sample_id)
        restore_capture(env, cap)
        look = 0.5 * (
            np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
            + np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
        )
        cam = mjv_camera_from_preset(
            {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -22.0}
        )
        renderer = mujoco.Renderer(env.sim.model, H, W)
        left, left_out = run_branch(
            env, cam, renderer, cap, ref, [], np.zeros((0, 7), np.float32),
            plan=plan, sid=sample_id, branch="DIRECT NOMINAL",
        )
        right, right_out = run_branch(
            env, cam, renderer, cap, ref, [], acts,
            plan=plan, sid=sample_id, branch="RECOVERY",
        )

    frames = pad_pair(left, right)
    # live QA on a few rendered frames before encode
    live = [scene_stats(f) for f in frames[:: max(1, len(frames) // 8)][:8]]
    live_mean = float(np.mean([s["mean"] for s in live]))
    if live_mean < 20:
        raise RuntimeError(f"live render too dark mean={live_mean:.1f} for {sample_id}")

    dest = VID / f"{tag}__{sample_id}__vs_nominal.mp4"
    write_frames_mp4(frames, dest, fps=25)
    qa = qa_video(dest)
    cs = contact_sheet(dest, VID / "contact_sheets" / f"{tag}__{sample_id}.jpg")
    meta = {
        "tag": tag,
        "mechanism": mechanism,
        "sample_id": sample_id,
        "plan": plan,
        "video": str(dest.relative_to(ROOT)),
        "contact_sheet": str(cs.relative_to(ROOT)) if cs else None,
        "n_frames": len(frames),
        "direct": left_out,
        "recovery": right_out,
        "qa": qa,
        "outcome": resolve_outcome(sample_id),
    }
    (VID / f"{tag}__{sample_id}__meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(
        "WROTE",
        dest.name,
        "qa",
        qa["ok"],
        "bright",
        qa["mean_brightness"],
        "direct_NEW",
        left_out["SUCCESS_NEW"],
        "rec_NEW",
        right_out["SUCCESS_NEW"],
        flush=True,
    )
    if not qa["ok"]:
        raise RuntimeError(f"QA failed {dest.name}: {qa['fail_reasons']}")
    return meta


CASES = [
    {
        "tag": "mech_dtheta",
        "mechanism": "dtheta_pivot",
        "sample_id": "PILOT_A__ball_train_v2_0070__962cc37bb1af2e85",
        "plan": "A",
    },
    {
        "tag": "mech_dx",
        "mechanism": "dx_verified_lateral",
        "sample_id": "PILOT_A__ball_train_0017__f423e91b56091488",
        "plan": "A",
    },
    {
        "tag": "mech_dz_inward",
        "mechanism": "dz_inward",
        "sample_id": "B__hard_910220__0412621ee978627d",
        "plan": "B",
    },
    {
        "tag": "mech_combined_dx_dz",
        "mechanism": "dx_then_dz_inward",
        "sample_id": "B__hard_910220__0412621ee978627d",
        "plan": "B",
    },
    {
        "tag": "mech_dx_k49_B",
        "mechanism": "dx_verified_lateral",
        "sample_id": "B__hard_910021__post_b1_v2__8ccbfb0c3eb750e8",
        "plan": "B",
    },
]


def pick_c_cases():
    cases = []
    # successful post-lift direct
    for p in sorted((OUT / "outcomes").glob("C__postlift__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if o.get("sft_category") == "DIRECT_HANDOFF":
            cases.append(
                {
                    "tag": "C_postlift_direct",
                    "mechanism": "catch_lift_direct_handoff",
                    "sample_id": o["sample_id"],
                    "plan": "C",
                    "shared_lift": True,
                    "catch_id": o.get("catch_sample_id"),
                }
            )
            break
    # diagnostic fail
    for p in sorted((OUT / "outcomes").glob("C__postlift__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if o.get("sft_category") == "FAILED_ORACLE":
            cases.append(
                {
                    "tag": "C_postlift_dz_fail",
                    "mechanism": "dz_inward_contact_loss",
                    "sample_id": o["sample_id"],
                    "plan": "C",
                    "shared_lift": True,
                    "catch_id": o.get("catch_sample_id"),
                }
            )
            break
    return cases


def main():
    VID.mkdir(parents=True, exist_ok=True)
    (VID / "contact_sheets").mkdir(exist_ok=True)
    index = []
    qa_rows = []
    for c in CASES + pick_c_cases():
        try:
            meta = render_case(**c)
            index.append(meta)
            qa_rows.append(meta["qa"])
        except Exception as e:
            print("FAIL", c.get("tag"), c.get("sample_id"), e, flush=True)
            qa_rows.append({"tag": c.get("tag"), "sample_id": c.get("sample_id"), "ok": False, "error": str(e)})

    (VID / "INDEX.json").write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
    qa_report = {
        "n": len(qa_rows),
        "n_ok": sum(1 for r in qa_rows if r.get("ok")),
        "rows": qa_rows,
        "all_passed": all(r.get("ok") for r in qa_rows) and len(qa_rows) > 0,
        "dz_outward_note": "No verified successful correct_dz_outward tape in A/B/C mining; not fabricated.",
    }
    (OUT / "VIDEO_QA_REPORT.md").write_text(
        "# Video QA Report\n\n"
        + f"Delivered: {qa_report['n_ok']}/{qa_report['n']} passed brightness/dark-pixel gate.\n\n"
        + "```json\n"
        + json.dumps(qa_report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (VID / "QA.json").write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    print("Done videos", qa_report["n_ok"], "/", qa_report["n"], flush=True)


if __name__ == "__main__":
    main()
