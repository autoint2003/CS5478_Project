#!/usr/bin/env python
"""Render cold-upright DZ-only inward precondition reject video (seq4)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.endpoint_only_axial_roll import gates  # noqa: E402
from data_generation.oracles.precision.force_slip_control import DT  # noqa: E402
from data_generation.oracles.precision.precision_dz import (  # noqa: E402
    check_dz_inward_entry,
    correct_dz_inward,
)
from data_generation.oracles.precision.precision_goals import measure_errors  # noqa: E402
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

OUT = ROOT / "results/diagnostics/final_down_handoff_sequences/videos"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
FPS = 20
HOLD_S = 4.0


def overlay(img, lines):
    out = np.asarray(img, np.uint8).copy()
    h = min(16 + 16 * len(lines), out.shape[0])
    out[:h] = (22, 18, 18)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 4
        for i, line in enumerate(lines):
            fill = (255, 120, 120) if i == 0 or "REJECT" in line or "FAIL" in line else (240, 240, 240)
            dr.text((6, y), str(line)[:110], fill=fill)
            y += 15
        return np.asarray(im)
    except Exception:
        return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    st = pickle.loads((STATES / "plan_b__hard_910220__stable.pkl").read_bytes())
    restore_capture(env, st["capture"])
    cfg = PrecisionConfig()
    entry = check_dz_inward_entry(env.sim, cfg)
    rz = correct_dz_inward(env, {"capture": st["capture"], "ref": st["ref"]}, cfg, s1_state=None)
    restore_capture(env, st["capture"])

    import mujoco

    rw = mujoco.Renderer(env.sim.model, 480, 640)
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.62, "azimuth": 125, "elevation": -20}
    )
    frames = []
    t0 = float(env.sim.data.time)
    n_hold = int(HOLD_S / DT)

    for i in range(n_hold):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
        if i % 3 != 0 and i != n_hold - 1:
            continue
        ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
        po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
        cam.azimuth = 125.0 + 25.0 * np.sin(2 * np.pi * i / n_hold)
        cam.lookat[:] = 0.5 * po + 0.5 * ph + np.array([0, 0, 0.02])
        err = measure_errors(env.sim)
        mode = classify_contact_mode(env.sim)
        g = gates(env.sim)
        orient = "DOWN" if g["fingertips_down"] else ("UP" if g["fingertips_up"] else f"a={g['alpha_up_deg']:.0f}")
        lines = [
            "SEQ4 DZ-only inward | COLD UPRIGHT PRECONDITION REJECT",
            f"FAIL  lateral_ex_too_large: |ex|={abs(err['dx_mm']):.2f} mm > dz_safe_ex={entry['dz_safe_ex_mm']:.1f} mm",
            f"t={float(env.sim.data.time)-t0:.2f}s  orient={orient}  mode={mode['mode']}",
            f"ex={err['dx_mm']:.1f} ez={err['dz_mm']:.1f} th={err['dtheta_deg']:.1f} epos={err['e_pos_mm']:.1f}",
            f"vrel={err['v_rel_norm']:.3f} fnL/R={err['fn_L']:.1f}/{err['fn_R']:.1f}  bilateral={err['bilateral']}",
            "gate not reached | scheduler must run dx (S1/S2) before inward dz",
            f"branch={rz.get('branch')}  n_actions={len(rz.get('actions') or [])}",
        ]
        rw.update_scene(env.sim.data, cam)
        scn = rw.scene
        u = cylinder_axis_world(env.sim)
        add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
        add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.25, 0.95, 0.3, 1.0], 0.003)
        frames.append(overlay(np.asarray(rw.render(), np.uint8), lines))

    rw.close()
    dest = OUT / "04_dz_only_inward_cold_upright_REJECT.mp4"
    path = write_frames_mp4(frames, dest, fps=FPS)

    sheet = dest.with_suffix("").as_posix() + "_sheet.png"
    try:
        from PIL import Image

        idxs = [0, len(frames) // 2, len(frames) - 1]
        imgs = [Image.fromarray(frames[i]) for i in idxs]
        w, h = imgs[0].size
        sheet_im = Image.new("RGB", (w * 3, h))
        for i, im in enumerate(imgs):
            sheet_im.paste(im, (i * w, 0))
        sheet_im.save(sheet)
    except Exception:
        sheet = None

    meta = {
        "path": str(path),
        "sheet": sheet,
        "n_frames": len(frames),
        "reason": rz.get("reason"),
        "branch": rz.get("branch"),
        "n_actions": len(rz.get("actions") or []),
        "entry_check": {k: entry[k] for k in entry if k != "errors"},
        "errors_entry": {
            k: entry["errors"].get(k)
            for k in ("dx_mm", "dz_mm", "dtheta_deg", "e_pos_mm", "bilateral", "v_rel_norm")
        },
    }
    (OUT / "04_INDEX.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=2, default=str))
    print(f"\nVideo: {path}")


if __name__ == "__main__":
    main()
