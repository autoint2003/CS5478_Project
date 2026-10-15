"""Write rendered frames to an MP4. Does not simulate."""

from __future__ import annotations

import subprocess
from pathlib import Path


def encode_mp4(frame_dir: Path, dest: Path, fps: int = 25) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%06d.png")
    cmd_tail = ["-y", "-framerate", str(fps), "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)]
    candidates = []
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.append("ffmpeg")
    for exe in candidates:
        try:
            subprocess.run([exe, *cmd_tail], check=True, capture_output=True)
            if dest.is_file() and dest.stat().st_size > 1000:
                return str(dest)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
    return None


def write_frames_mp4(frames, dest: Path, fps: int = 25) -> str | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"_frames_{dest.stem}"
    if tmp.exists():
        for path in tmp.glob("*.png"):
            path.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        for index, frame in enumerate(frames):
            imageio.imwrite(str(tmp / f"frame_{index:06d}.png"), frame)
    except Exception:
        import matplotlib.image as mpimg

        for index, frame in enumerate(frames):
            mpimg.imsave(str(tmp / f"frame_{index:06d}.png"), frame)
    return encode_mp4(tmp, dest, fps=fps)
