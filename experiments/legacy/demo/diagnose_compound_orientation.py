"""Replay the compound-orientation visual audit.

The historical runner was not stored in git. This file follows the command
written in results/archive/demo_results/diagnostics/COMPOUND_ORIENTATION_GRAVITY_ROLLING_VISUAL_AUDIT.md:
one 6.5 m/s, -6 mm impact snapshot, then the inward, outward, and secure clips.

No training. Physics, friction, and the recovery action map are unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from controllers.recovery_actions import W_MAX, grip_tau
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import physical_pack
from envs.ballistic_impact import TAU_SEC, make_sim, z_tgt_of
from training.recovery_runtime import capture_nominal, restore_ballistic, run_impact
from training.replay_core import tick_vw

OUT = ROOT / "results" / "diagnostics" / "raw" / "compound_orientation_gravity_rolling_visual_audit"
VIDEO = OUT / "comparison_inward_outward_secure.mp4"

# Report: frames every 0.01 s, 264 frames (t = 0 ... 2.63 s), playback 20 fps.
FRAME_DT = 0.01
N_FRAMES = 264
PLAY_FPS = 20
SUPPORT_S = 0.40
RELAX_S = 0.48
TIGHT_S = 0.16
# 20 policy steps of 0.02 s. wx/wy are recovery7 actions; W_MAX is 4 rad/s.
SUPPORT_STEPS = 20
WX = 0.28
WY_IN = 0.50
WY_OUT = -0.50
GRIP_SECURE = 1.0
GRIP_RELAX = -0.48

CLOSE = {"distance": 0.155, "azimuth": 42.0, "elevation": -8.0}
RENDER_W, RENDER_H = 640, 480


def phase_at(t: float, secure: bool) -> str:
    if t < SUPPORT_S - 1e-9:
        return "support+pitch"
    if t < SUPPORT_S + RELAX_S - 1e-9:
        return "secure" if secure else "relax"
    if t < SUPPORT_S + RELAX_S + TIGHT_S - 1e-9:
        return "retighten"
    return "handoff"


def command_at(t: float, wy: float, secure: bool) -> tuple[float, float, float | None]:
    """Return body-frame wx action, wy action, and grip action.

    Grip None means the nominal lift/hold controller (handoff).
    """
    if t < SUPPORT_S - 1e-9:
        return WX, wy, GRIP_SECURE
    if t < SUPPORT_S + RELAX_S - 1e-9:
        return 0.0, 0.0, GRIP_SECURE if secure else GRIP_RELAX
    if t < SUPPORT_S + RELAX_S + TIGHT_S - 1e-9:
        return 0.0, 0.0, GRIP_SECURE
    return 0.0, 0.0, None


def step_once(sim, gains, wx, wy, grip, lift: bool) -> None:
    if grip is None:
        z_tgt = z_tgt_of(sim)
        if float(sim.fsm.p_des[2]) >= z_tgt - 1e-9:
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU_SEC, gains)
        else:
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
        return
    v = np.zeros(3)
    if lift and float(sim.fsm.p_des[2]) < z_tgt_of(sim) - 1e-9:
        v[2] = float(sim.cfg["fsm"].get("lift_speed", 0.08))
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    w = rd @ np.array([float(wx) * W_MAX, float(wy) * W_MAX, 0.0], float)
    tick_vw(sim, v, w, grip_tau(grip), gains)


def sample(sim, t_clip: float, phase: str, grip) -> dict:
    o = physical_pack(sim)
    rh = np.asarray(o["rh"], float)
    g_h = np.asarray(o["g_h"], float)
    hand = np.array(sim.data.xpos[sim.ids.hand_body], float)
    return {
        "t": float(t_clip),
        "phase": phase,
        "rh": rh.copy(),
        "g_h": g_h.copy(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "obj_z": float(o["obj_z"]),
        "hand_p": hand.copy(),
        "grip": float("nan") if grip is None else float(grip),
        "tau": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
    }


def rollout(sim, gains, snap, wy: float, secure: bool, lift: bool, record) -> list[dict]:
    restore_ballistic(sim, snap)
    dt = float(sim.model.opt.timestep)
    rows = []
    t = 0.0
    hand0 = np.array(sim.data.xpos[sim.ids.hand_body], float)
    for frame in range(N_FRAMES):
        phase = phase_at(t, secure)
        _wx, _wy, grip = command_at(t, wy, secure)
        row = sample(sim, t, phase, grip)
        row["hand_disp"] = float(np.linalg.norm(row["hand_p"] - hand0))
        rows.append(row)
        if record is not None:
            record(sim, row, frame)
        if frame + 1 == N_FRAMES:
            break
        n = int(round(FRAME_DT / dt))
        for _ in range(n):
            wx, wy_c, grip_c = command_at(t, wy, secure)
            step_once(sim, gains, wx, wy_c, grip_c, lift)
            t += dt
    return rows


def capture_impact(sim, gains):
    nominal = capture_nominal(sim)
    impact = run_impact(sim, gains, 6.5, -0.006, 0.0, [], [], nominal, None, None)
    snap = impact["snap"]
    if snap is None:
        raise RuntimeError(f"impact snapshot missing: {impact['meta']}")
    return impact


def summarize(name: str, rows: list[dict]) -> None:
    want = (0.0, 0.40, 0.88, 2.63)
    print(f"== {name} ==", flush=True)
    by_t = {}
    for row in rows:
        key = round(row["t"], 2)
        by_t[key] = row
    for t in want:
        row = by_t[round(t, 2)]
        rh = row["rh"] * 1e3
        print(
            f"  t={t:.2f} {row['phase']:14s} x={rh[0]:7.2f} mm  "
            f"g_h.x={row['g_h'][0]:+6.2f}  nL/R={row['nL']}/{row['nR']}  "
            f"z={row['obj_z']:.3f}  hand_disp={row['hand_disp']*1e3:.1f} mm  "
            f"grip={row['grip']:+.2f} tau={row['tau']:+.2f}",
            flush=True,
        )


def metrics_only() -> None:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim, _ = make_sim()
    print("running 6.5 m/s -6 mm impact", flush=True)
    impact = capture_impact(sim, gains)
    print("capture", impact["meta"]["early"], "z_tgt", z_tgt_of(sim), flush=True)
    snap = impact["snap"]
    for lift in (False, True):
        tag = "lift" if lift else "hold"
        summarize(f"inward/{tag}", rollout(sim, gains, snap, WY_IN, False, lift, None))
        summarize(f"outward/{tag}", rollout(sim, gains, snap, WY_OUT, False, lift, None))
        summarize(f"secure/{tag}", rollout(sim, gains, snap, WY_IN, True, lift, None))


def _font(size: int):
    from PIL import ImageFont

    for path in (
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def burn(img: np.ndarray, row: dict, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    font = _font(18)
    x_mm = float(row["rh"][0]) * 1e3
    grip = "nominal" if not np.isfinite(row["grip"]) else f"{row['grip']:+.2f}"
    lines = [
        title,
        f"t={row['t']:.2f}s  {row['phase']}",
        f"hand x={x_mm:.1f} mm",
        f"g_h.x={row['g_h'][0]:+.2f} m/s^2",
        f"grip={grip}",
        f"contact L/R={row['nL']}/{row['nR']}",
    ]
    width = max(dr.textlength(line, font=font) for line in lines) + 16
    dr.rectangle((8, 8, 16 + width, 16 + 22 * len(lines)), fill=(0, 0, 0, 160))
    y = 12
    for line in lines:
        dr.text((14, y), line, fill=(255, 255, 255, 255), font=font)
        y += 22
    return np.asarray(im)


def encode_mp4(frames: list[np.ndarray], dest: Path, fps: int) -> None:
    import subprocess

    from PIL import Image

    tmp = dest.parent / f"_frames_{dest.stem}"
    if tmp.exists():
        for path in tmp.glob("*.png"):
            path.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(tmp / f"frame_{index:06d}.png")
    candidates = []
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.append("ffmpeg")
    pattern = str(tmp / "frame_%06d.png")
    cmd_tail = [
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    for exe in candidates:
        try:
            subprocess.run([exe, *cmd_tail], check=True, capture_output=True)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("ffmpeg failed to encode the comparison")
    for path in tmp.glob("*.png"):
        path.unlink()
    tmp.rmdir()


def render_comparison() -> None:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    sim, _ = make_sim()
    print("running 6.5 m/s -6 mm impact", flush=True)
    impact = capture_impact(sim, gains)
    print("capture", impact["meta"]["early"], flush=True)
    snap = impact["snap"]
    renderer = mujoco.Renderer(sim.model, RENDER_H, RENDER_W)
    cam = mujoco.MjvCamera()
    panels = []
    logs = []
    specs = (
        ("inward slope", WY_IN, False),
        ("outward slope", WY_OUT, False),
        ("secure grip", WY_IN, True),
    )
    # Position command stays put during the slope. The pitch itself moves the
    # hand about 24 mm, which is the motion reported for 0.00-0.40 s.
    for name, wy, secure in specs:
        frames = []

        def record(sim, row, frame, frames=frames, name=name):
            if frame % 40 == 0:
                print(f"  {name} frame {frame}", flush=True)
            look = np.array(sim.data.xpos[sim.ids.object_body], float)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = look
            cam.distance = CLOSE["distance"]
            cam.azimuth = CLOSE["azimuth"]
            cam.elevation = CLOSE["elevation"]
            renderer.update_scene(sim.data, cam)
            frames.append(burn(renderer.render(), row, name))

        print("rollout", name, flush=True)
        rows = rollout(sim, gains, snap, wy, secure, False, record)
        summarize(name, rows)
        panels.append(frames)
        logs.append(rows)
    combo = [np.concatenate([a, b, c], axis=1) for a, b, c in zip(*panels)]
    OUT.mkdir(parents=True, exist_ok=True)
    print("encode", VIDEO, len(combo), combo[0].shape, flush=True)
    encode_mp4(combo, VIDEO, PLAY_FPS)
    print("wrote", VIDEO, "bytes", VIDEO.stat().st_size, flush=True)
    del logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if args.metrics_only:
        metrics_only()
    else:
        render_comparison()


if __name__ == "__main__":
    main()
