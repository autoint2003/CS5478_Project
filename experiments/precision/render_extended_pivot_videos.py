#!/usr/bin/env python
"""Extended dual-view explanatory videos for saved position-preserving pivot experiments.

Replays exact reconstructed action tapes from saved tags/metrics.
Does not modify Oracle code, physics, or prior diagnostic JSON/pkl outputs.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.action_map import grip_action  # noqa: E402
from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import (  # noqa: E402
    _action_vw,
    _wrist,
    pivot_segment_metrics,
    primitive_axial_tail_slide,
    relative_pose,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_hand, cylinder_axis_world, hand_R  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import load_codebook  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PRIOR = ROOT / "results" / "diagnostics" / "position_preserving_pivot"
STATE_A = (
    ROOT
    / "results/diagnostics/abc_precision_architecture/stable_states/plan_a__ball_train_v2_0070__stable.pkl"
)
STATE_B = (
    ROOT
    / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
)
OUT = ROOT / "results" / "diagnostics" / "position_preserving_pivot_extended"
FPS = 25
DT = 0.02
PIVOT_SPEED = 0.25  # 0.25× → each sim step occupies DT/0.25 wall-video seconds

EXPECTED = {
    "A_raw": {"theta0": 14.505, "theta1": 0.261, "delta_p_rel_mm": 2.975, "tol_th": 0.15, "tol_dp": 0.15},
    "B_raw": {"theta0": 8.192, "theta1": 4.013, "delta_p_rel_mm": 1.963, "tol_th": 0.15, "tol_dp": 0.15},
    "B_post_tail": {"d_axis_deg": 12.690, "delta_p_rel_mm": 11.267, "tol_axis": 0.2, "tol_dp": 0.2},
}


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def parse_ppivot_tag(tag: str) -> dict:
    # ppiv_v{vx}_{vz}_w{wx}_{wy}_g{grip}_d{dur}
    assert tag.startswith("ppiv_v"), tag
    rest = tag[len("ppiv_v") :]
    # split carefully: v0.12_0.08_w0.0_1.0_g-0.3_d0.25
    pre_w, after_w = rest.split("_w", 1)
    vx_s, vz_s = pre_w.split("_", 1)
    pre_g, after_g = after_w.split("_g", 1)
    wx_s, wy_s = pre_g.split("_", 1)
    grip_s, dur_s = after_g.split("_d", 1)
    return {
        "vx": float(vx_s),
        "vz": float(vz_s),
        "wx": float(wx_s),
        "wy": float(wy_s),
        "grip": float(grip_s),
        "dur": float(dur_s),
    }


def actions_from_ppivot_tag(tag: str) -> tuple[list, list]:
    p = parse_ppivot_tag(tag)
    a_p = _action_vw([p["vx"], 0.0, p["vz"]], [p["wx"], p["wy"], 0.0], p["grip"])
    a_b = _wrist(0.0, 0.0, 1.0)
    n = max(1, int(round(p["dur"] / DT)))
    return [a_p] * n, [a_b] * 8


def actions_legacy(wx, wy, grip, dur) -> tuple[list, list]:
    a_p = _wrist(wx, wy, grip)
    a_b = _wrist(0.0, 0.0, 1.0)
    n = max(1, int(round(dur / DT)))
    return [a_p] * n, [a_b] * 8


def advance_tail(env, book, state) -> dict:
    blob = {"capture": state["capture"], "ref": state["ref"]}
    full = primitive_axial_tail_slide(env, blob, book, budget=40)
    if not full.get("physically_demonstrated") or not full.get("actions"):
        raise RuntimeError("tail slide failed; cannot rebuild post_tail start state")
    restore_capture(env, blob["capture"])
    for a in full["actions"]:
        step_legal(env.sim, np.asarray(a, np.float32))
    return {"capture": take_capture(env), "ref": state["ref"]}


def overlay_text(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = np.asarray(img, np.uint8).copy()
    bar = 18 + 16 * len(lines)
    out[:bar] = (12, 12, 12)
    try:
        from PIL import Image, ImageDraw

        im = Image.fromarray(out)
        dr = ImageDraw.Draw(im)
        y = 4
        for line in lines:
            dr.text((6, y), line, fill=(255, 255, 255))
            y += 16
        return np.asarray(im)
    except Exception:
        return out


def set_world_cam(cam, sim):
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    cam.lookat[:] = 0.55 * po + 0.45 * ph
    cam.distance = 0.52
    cam.azimuth = 138.0
    cam.elevation = -18.0


def set_hand_cam(cam, sim):
    """Fixed pose in the hand frame → gripper appears stationary."""
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    # look near grasp region; camera sits on +hand-y / +hand-z
    look_h = np.array([0.00, 0.00, 0.09], float)
    pos_h = np.array([0.28, -0.32, 0.18], float)
    look = ph + Rh @ look_h
    pos = ph + Rh @ pos_h
    vec = pos - look
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        dist = 0.4
        vec = np.array([0.3, -0.3, 0.2])
    elev = float(np.degrees(np.arcsin(np.clip(vec[2] / dist, -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(vec[1], vec[0])))
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev


def draw_annotations(scn, sim, refs, *, show_both_axes: bool):
    """Display-only markers. refs: p_rel0, u_hand0, u_world0."""
    scn.ngeom = 0
    Rh = hand_R(sim)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    u_now = cylinder_axis_world(sim)
    p_rel0 = np.asarray(refs["p_rel0"], float)
    u0_h = np.asarray(refs["u_hand0"], float)
    # Current COM (cyan), initial COM ref in current hand frame (magenta ghost)
    add_sphere(scn, po, [0.15, 0.95, 1.0, 1.0], 0.007)
    p_ref = ph + Rh @ p_rel0
    add_sphere(scn, p_ref, [1.0, 0.2, 0.85, 1.0], 0.006)
    # Current cylinder axis (green)
    add_capsule(scn, po - 0.05 * u_now, po + 0.05 * u_now, [0.2, 0.95, 0.25, 1.0], 0.003)
    # Initial axis expressed through current COM via hand-frame reference (orange)
    u0_w = Rh @ u0_h
    u0_w = u0_w / max(float(np.linalg.norm(u0_w)), 1e-12)
    add_capsule(scn, po - 0.05 * u0_w, po + 0.05 * u0_w, [1.0, 0.55, 0.1, 1.0], 0.0025)
    if show_both_axes:
        # Also draw initial axis at the ghost COM for clarity at AFTER
        add_capsule(scn, p_ref - 0.04 * u0_w, p_ref + 0.04 * u0_w, [1.0, 0.35, 0.05, 0.85], 0.002)


def metrics_now(sim, refs) -> dict:
    pr = relative_pose(sim)
    p0 = np.asarray(refs["p_rel0"], float)
    p1 = np.asarray(pr["p_rel"], float)
    dp = 1e3 * (p1 - p0)
    return {
        "theta": pr["e_theta_deg"],
        "delta_p_rel_mm": float(np.linalg.norm(dp)),
        "delta_p_components_mm": dp.tolist(),
        "bilateral": pr["bilateral"],
        "t": float(sim.data.time),
    }


def render_pair(renderer_w, renderer_h, cam_w, cam_h, sim, refs, phase, t_sim0, lines_extra=None):
    show_both = phase in ("AFTER", "BRAKE")
    set_world_cam(cam_w, sim)
    set_hand_cam(cam_h, sim)
    m = metrics_now(sim, refs)
    t_rel = float(sim.data.time) - float(t_sim0)
    bil = "BILAT" if m["bilateral"] else "NO-CONTACT"
    lines = [
        f"{phase}  t_sim={t_rel:.3f}s  θ={m['theta']:.2f}°  Δp_rel={m['delta_p_rel_mm']:.2f}mm  {bil}",
        f"Δp_xyz_mm=[{m['delta_p_components_mm'][0]:+.2f},{m['delta_p_components_mm'][1]:+.2f},{m['delta_p_components_mm'][2]:+.2f}]  "
        f"cyan=COM now  magenta=COM0@hand  green=axis now  orange=axis0",
    ]
    if lines_extra:
        lines.extend(lines_extra)

    renderer_w.update_scene(sim.data, cam_w)
    draw_annotations(renderer_w.scene, sim, refs, show_both_axes=show_both)
    img_w = overlay_text(np.asarray(renderer_w.render(), np.uint8), ["WORLD  " + lines[0], lines[1]] + (lines[2:] if len(lines) > 2 else []))

    renderer_h.update_scene(sim.data, cam_h)
    draw_annotations(renderer_h.scene, sim, refs, show_both_axes=show_both)
    img_h = overlay_text(np.asarray(renderer_h.render(), np.uint8), ["HAND-REL  " + lines[0], lines[1]] + (lines[2:] if len(lines) > 2 else []))

    # side-by-side world | hand
    return np.concatenate([img_w, img_h], axis=1), m


def append_hold(frames, render_fn, seconds: float):
    n = max(1, int(round(seconds * FPS)))
    img, m = render_fn()
    for _ in range(n):
        frames.append(img)
    return m


def append_actions(frames, env, actions, refs, t_sim0, phase, renderer_w, renderer_h, cam_w, cam_h, speed: float, title: str):
    """Replay actions; each sim step duplicated for 1/speed video stretch."""
    dup = max(1, int(round((DT / speed) * FPS)))
    last_m = None
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        img, last_m = render_pair(
            renderer_w,
            renderer_h,
            cam_w,
            cam_h,
            env.sim,
            refs,
            phase,
            t_sim0,
            lines_extra=[f"{title}  playback={speed:.2f}×  (presentation; physics unchanged)"],
        )
        for _ in range(dup):
            frames.append(img)
    return last_m


def verify(label: str, met: dict) -> dict:
    exp = EXPECTED.get(label)
    ok = True
    detail = {"label": label, "measured": {}, "expected": exp, "ok": True, "notes": []}
    if exp is None:
        detail["notes"].append("no numeric gate")
        return detail
    if "theta0" in exp:
        detail["measured"]["theta0"] = met["theta0"]
        detail["measured"]["theta1"] = met["theta1"]
        detail["measured"]["delta_p_rel_mm"] = met["delta_p_rel_mm"]
        if abs(met["theta0"] - exp["theta0"]) > exp["tol_th"]:
            ok = False
            detail["notes"].append("theta0 mismatch")
        if abs(met["theta1"] - exp["theta1"]) > exp["tol_th"]:
            ok = False
            detail["notes"].append("theta1 mismatch")
        if abs(met["delta_p_rel_mm"] - exp["delta_p_rel_mm"]) > exp["tol_dp"]:
            ok = False
            detail["notes"].append("delta_p mismatch")
    if "d_axis_deg" in exp:
        detail["measured"]["d_axis_deg"] = met["d_axis_deg"]
        detail["measured"]["delta_p_rel_mm"] = met["delta_p_rel_mm"]
        if abs(met["d_axis_deg"] - exp["d_axis_deg"]) > exp["tol_axis"]:
            ok = False
            detail["notes"].append("d_axis mismatch")
        if abs(met["delta_p_rel_mm"] - exp["delta_p_rel_mm"]) > exp["tol_dp"]:
            ok = False
            detail["notes"].append("delta_p mismatch")
    detail["ok"] = ok
    return detail


def build_case(env, book, search: dict, audit: dict) -> dict:
    state_a = pickle.loads(STATE_A.read_bytes())
    state_b = pickle.loads(STATE_B.read_bytes())
    cases = {}

    # A_raw success
    tag = search["A_raw"]["best"]["tag"]
    piv, brk = actions_from_ppivot_tag(tag)
    blob = {"capture": state_a["capture"], "ref": state_a["ref"]}
    met = pivot_segment_metrics(env, blob, piv + brk, record_series=False)
    cases["A_raw"] = {
        "blob": blob,
        "pivot": piv,
        "brake": brk,
        "title": "A_raw position-preserving SUCCESS",
        "class": "position_preserving_relative_pivot",
        "metrics": met,
        "verify": verify(
            "A_raw",
            {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        ),
    }

    # B_raw success
    tag = search["B_raw"]["best"]["tag"]
    piv, brk = actions_from_ppivot_tag(tag)
    blob = {"capture": state_b["capture"], "ref": state_b["ref"]}
    met = pivot_segment_metrics(env, blob, piv + brk, record_series=False)
    cases["B_raw"] = {
        "blob": blob,
        "pivot": piv,
        "brake": brk,
        "title": "B_raw position-preserving SUCCESS",
        "class": "position_preserving_relative_pivot",
        "metrics": met,
        "verify": verify(
            "B_raw",
            {"theta0": met["theta0"], "theta1": met["theta1"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        ),
    }

    # A_post_tail legacy rigid
    blob_at = advance_tail(env, book, state_a)
    piv, brk = actions_legacy(0.0, 1.0, -0.48, 0.40)
    met = pivot_segment_metrics(env, blob_at, piv + brk, record_series=False)
    cases["A_post_tail"] = {
        "blob": blob_at,
        "pivot": piv,
        "brake": brk,
        "title": "A_post_tail rigid co-rotation (NEG)",
        "class": audit["A_post_tail"]["classification"],
        "metrics": met,
        "verify": {
            "label": "A_post_tail",
            "ok": abs(met["d_axis_deg"] - audit["A_post_tail"]["d_axis_deg"]) < 0.25
            and abs(met["delta_p_rel_mm"] - audit["A_post_tail"]["delta_p_rel_mm"]) < 0.25,
            "measured": {"d_axis_deg": met["d_axis_deg"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
            "expected": {
                "d_axis_deg": audit["A_post_tail"]["d_axis_deg"],
                "delta_p_rel_mm": audit["A_post_tail"]["delta_p_rel_mm"],
            },
        },
    }

    # B_post_tail legacy coupled
    blob_bt = advance_tail(env, book, state_b)
    piv, brk = actions_legacy(0.0, 3.0, 1.0, 0.30)
    met = pivot_segment_metrics(env, blob_bt, piv + brk, record_series=False)
    cases["B_post_tail"] = {
        "blob": blob_bt,
        "pivot": piv,
        "brake": brk,
        "title": "B_post_tail coupled rot-trans (NEG)",
        "class": "coupled_rotation_translation",
        "metrics": met,
        "verify": verify(
            "B_post_tail",
            {"d_axis_deg": met["d_axis_deg"], "delta_p_rel_mm": met["delta_p_rel_mm"]},
        ),
    }
    return cases


def render_case(env, case: dict, dest: Path) -> dict:
    restore_capture(env, case["blob"]["capture"])
    refs = {
        "p_rel0": relative_pose(env.sim)["p_rel"],
        "u_hand0": cylinder_axis_hand(env.sim).tolist(),
        "u_world0": cylinder_axis_world(env.sim).tolist(),
    }
    t0 = float(env.sim.data.time)
    renderer_w = mujoco.Renderer(env.sim.model, 480, 640)
    renderer_h = mujoco.Renderer(env.sim.model, 480, 640)
    cam_w = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    cam_h = mjv_camera_from_preset({"lookat": [0, 0, 0.5], "distance": 0.5, "azimuth": 140, "elevation": -18})
    frames = []
    title = case["title"]
    cls = case["class"]

    def still(phase, extra=None):
        return lambda: render_pair(
            renderer_w,
            renderer_h,
            cam_w,
            cam_h,
            env.sim,
            refs,
            phase,
            t0,
            lines_extra=[f"{title} | {cls}", *(extra or []), "PRESENTATION HOLD (no extra sim steps)"],
        )

    # Presentation holds (frozen frames — not extra simulated control).
    # Sized so short 0.25–0.40 s pivot tapes still yield ~15–20 s videos.
    append_hold(frames, still("BEFORE"), 4.0)
    append_hold(
        frames,
        still("BEFORE", ["Show initial axis (orange) + COM0 (magenta) vs COM (cyan)"]),
        3.0,
    )
    append_actions(
        frames, env, case["pivot"], refs, t0, "PIVOT", renderer_w, renderer_h, cam_w, cam_h, PIVOT_SPEED, title
    )
    n_before_brake = len(frames)
    append_actions(
        frames, env, case["brake"], refs, t0, "BRAKE", renderer_w, renderer_h, cam_w, cam_h, PIVOT_SPEED, title
    )
    brake_video_s = (len(frames) - n_before_brake) / FPS
    pad = max(0.0, 4.0 - brake_video_s)
    if pad > 0:
        append_hold(frames, still("BRAKE", [f"brake tape done; pad {pad:.1f}s presentation"]), pad)
    append_hold(
        frames,
        still("AFTER", ["Final axes: green=now orange=initial | COM drift = cyan vs magenta"]),
        4.0,
    )

    renderer_w.close()
    renderer_h.close()
    path = write_frames_mp4(frames, dest, fps=FPS)
    return {
        "path": path,
        "n_frames": len(frames),
        "duration_s": len(frames) / FPS,
        "title": title,
        "class": cls,
    }


def side_by_side(cases_frames: dict, dest: Path, order: list[str]):
    """Normalized-progress comparison; each strip already rendered as frame lists is heavy —
    instead re-encode by sampling existing frame PNGs if present, else rebuild lightly.
    """
    # Load frames from _frames_* directories produced by write_frames_mp4
    import imageio.v2 as imageio

    frame_dirs = {}
    lengths = {}
    for key in order:
        d = OUT / "videos" / f"_frames_{key}_extended"
        pngs = sorted(d.glob("frame_*.png"))
        if not pngs:
            raise FileNotFoundError(d)
        frame_dirs[key] = pngs
        lengths[key] = len(pngs)
    n = max(lengths.values())
    out_frames = []
    for i in range(n):
        strips = []
        for key in order:
            pngs = frame_dirs[key]
            # normalized progress
            j = int(round(i * (len(pngs) - 1) / max(n - 1, 1)))
            img = np.asarray(imageio.imread(pngs[j]), np.uint8)
            # downscale each dual-view to half width for 3-up
            h, w = img.shape[:2]
            img_s = img[:, ::2, :][::2, :, :]  # rough 1/2
            # progress banner
            prog = i / max(n - 1, 1)
            t_note = f"{key} norm={prog:.2f} frame={j}/{len(pngs)-1}"
            img_s = overlay_text(img_s, [t_note, "normalized progress ≠ synchronized physical time"])
            strips.append(img_s)
        # match heights
        hmin = min(s.shape[0] for s in strips)
        strips = [s[:hmin] for s in strips]
        wmin = min(s.shape[1] for s in strips)
        strips = [s[:, :wmin] for s in strips]
        out_frames.append(np.concatenate(strips, axis=1))
    return write_frames_mp4(out_frames, dest, fps=FPS), len(out_frames) / FPS


def write_readme(meta: dict):
    lines = [
        "# Extended Position-Preserving Pivot Videos",
        "",
        "Presentation-only extension of trajectories from `position_preserving_pivot/`.",
        "No new Oracle search. Physics / actions / SUCCESS_NEW unchanged.",
        "",
        "## Camera conventions",
        "",
        "- **Left (WORLD):** free camera on hand–object midpoint; wrist motion visible.",
        "- **Right (HAND-REL):** camera pose fixed in the hand frame each frame so the gripper",
        "  appears stationary. Relative cylinder rotation/translation is visible here.",
        "  Rigid co-rotation ≈ stationary cylinder in HAND-REL; genuine relative pivot moves the cylinder.",
        "- Scale fixed (no auto-zoom).",
        "",
        "## Markers",
        "",
        "- Cyan sphere: current object COM",
        "- Magenta sphere: initial COM expressed in the *current* hand frame (ghost)",
        "- Green capsule: current cylinder axis",
        "- Orange capsule: initial axis (hand-frame reference) through current COM",
        "",
        "## Presentation timeline (not extra control)",
        "",
        "1. 4 s BEFORE — frozen presentation hold",
        "2. 3 s BEFORE — axis/COM callout (still frozen)",
        "3. PIVOT tape at 0.25×",
        "4. BRAKE tape at 0.25×, padded to ≥4 s presentation",
        "5. 4 s AFTER — frozen presentation hold",
        "",
        "## Videos",
        "",
    ]
    for k, v in meta["videos"].items():
        lines.append(f"- `{v.get('filename')}` — {v.get('duration_s'):.1f}s — {v.get('title')} — verify_ok={v.get('verify_ok')}")
    lines += [
        "",
        f"- Comparison: `{meta['comparison']['filename']}` ({meta['comparison']['duration_s']:.1f}s)",
        "  Shows A_raw | B_raw | B_post_tail with **normalized progress** (timestamps in overlays).",
        "",
        "## Reproduction verification",
        "",
    ]
    for k, v in meta["verification"].items():
        lines.append(f"- **{k}**: ok={v.get('ok')} measured={v.get('measured')} expected={v.get('expected')} notes={v.get('notes')}")
    lines += [
        "",
        "## Limitations",
        "",
        "- Pad contact patches are not mesh-highlighted; bilateral status is text-only.",
        "- HAND-REL uses MjvCamera spherical params derived from a hand-frame offset; small",
        "  discretization differences vs a true body-fixed OpenGL camera are possible.",
        "- `A_post_tail` / `B_post_tail` start states are rebuilt by replaying the same",
        "  `primitive_axial_tail_slide` used in the original audit (deterministic given seedless search).",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    search = json.loads((PRIOR / "search.json").read_text(encoding="utf-8"))
    audit = json.loads((PRIOR / "audit.json").read_text(encoding="utf-8"))
    # grip tag uses grip_action(-5) ≈ -0.3 — ensure import retained
    _ = grip_action

    env = make_env()
    book = load_codebook()
    print("Building / verifying cases...", flush=True)
    cases = build_case(env, book, search, audit)
    for k, c in cases.items():
        print(
            f"  {k}: verify={c['verify'].get('ok')} "
            f"θ {c['metrics']['theta0']:.3f}->{c['metrics']['theta1']:.3f} "
            f"Δp={c['metrics']['delta_p_rel_mm']:.3f} d_axis={c['metrics']['d_axis_deg']:.3f}",
            flush=True,
        )
        if not c["verify"].get("ok"):
            print(f"  WARNING verification failed: {c['verify']}", flush=True)

    meta = {"videos": {}, "verification": {}, "comparison": {}}
    for key in ("A_raw", "B_raw", "A_post_tail", "B_post_tail"):
        dest = OUT / "videos" / f"{key}_extended.mp4"
        print(f"Rendering {key}...", flush=True)
        info = render_case(env, cases[key], dest)
        meta["videos"][key] = {
            "filename": dest.name,
            "path": str(dest),
            "duration_s": info["duration_s"],
            "n_frames": info["n_frames"],
            "title": info["title"],
            "class": info["class"],
            "verify_ok": cases[key]["verify"].get("ok"),
        }
        meta["verification"][key] = cases[key]["verify"]

    print("Building side-by-side comparison...", flush=True)
    from PIL import Image

    order = ["A_raw", "B_raw", "B_post_tail"]
    frame_map = {k: OUT / "videos" / f"_frames_{k}_extended" for k in order}
    dest_cmp = OUT / "videos" / "compare_Araw_Braw_Bposttail.mp4"
    pngs = {k: sorted(frame_map[k].glob("frame_*.png")) for k in order}
    for k in order:
        if not pngs[k]:
            raise FileNotFoundError(frame_map[k])
    n = max(len(pngs[k]) for k in order)
    out_frames = []
    for i in range(n):
        strips = []
        for key in order:
            j = int(round(i * (len(pngs[key]) - 1) / max(n - 1, 1)))
            img = np.asarray(Image.open(pngs[key][j]).convert("RGB"), np.uint8)
            img = img[::2, ::2, :]
            prog = i / max(n - 1, 1)
            img = overlay_text(
                img,
                [
                    f"{key}  norm_progress={prog:.2f}  src_frame={j}/{len(pngs[key])-1}",
                    "normalized progress — not synchronized physical time",
                ],
            )
            strips.append(img)
        hmin = min(s.shape[0] for s in strips)
        wmin = min(s.shape[1] for s in strips)
        out_frames.append(np.concatenate([s[:hmin, :wmin] for s in strips], axis=1))
    path = write_frames_mp4(out_frames, dest_cmp, fps=FPS)
    meta["comparison"] = {
        "filename": dest_cmp.name,
        "path": str(path),
        "duration_s": len(out_frames) / FPS,
        "order": order,
    }

    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(meta)
    print(json.dumps(meta, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)
    return 0 if all(cases[k]["verify"].get("ok") for k in cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
