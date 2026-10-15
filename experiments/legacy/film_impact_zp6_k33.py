"""Film the K=33 lateral-chase policy on impact_zp6 from the live nominal episode.

Physics, geometry, the observation, the codebook, the residual cap, and the
v3 action map are the ones that produced RETAINED_TO_12. This script only
reads that checkpoint and writes a video plus a contact trace.
"""

from __future__ import annotations


import csv
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.tracking import TRACK
from envs.simulation.support_contacts import restore_supports, supports_of
import models.airborne_v3 as v3
from controllers.recovery.geometry_switch import hand_pose
from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.observations.airborne_obs import relative_state
from envs.outcomes.physical_recovery import physical_pack
from models.k48_policy import build_policy
from training.recovery_episode import (
    T_TASK,
    Z_ABORT,
    ball_substep,
    empty_ball,
    make_training_env,
)


from models.historical.k33_checkpoint import CKPT, load_policy

VID = ROOT / "results" / "videos"
SPEC = {"v_mps": 6.5, "z_off_m": 0.006, "phi_deg": 0.0}
W, H = 960, 540
FPS = 30.0
FONT = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 22)
FONT_B = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf", 26)




def cameras(sim):
    """Fixed third-person cameras. Azimuths were chosen so one view shows the
    ball beside the cylinder and the other shows the finger jaws."""
    look = np.array(sim.data.xpos[sim.ids.hand_body], float) + np.array([0.0, 0.0, 0.05])

    def make(azimuth, distance, elevation):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(sim.model, cam)
        cam.lookat[:] = look
        cam.distance = float(distance)
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        return cam

    return make(180.0, 0.95, -20.0), make(100.0, 0.82, -16.0)


def paint(rgb, hud):
    image = Image.fromarray(rgb).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rectangle((0, 0, W, 132), fill=(0, 0, 0, 170))
    token = hud["token"]
    lateral = token == 32
    def num(value, spec):
        return "  --  " if value is None else format(value, spec)

    lines = [
        f"t = {hud['t']:7.3f} s    {hud['mode']}",
        f"e_pos {hud['e_pos']*1000:6.1f} mm    bilateral {'YES' if hud['bilateral'] else 'NO'}    nL {hud['nL']}  nR {hud['nR']}",
        f"token {('--' if token is None else token)}{'   LATERAL CHASE' if lateral else ''}",
        f"vx {num(hud['vx'], '+6.2f')}  vy {num(hud['vy'], '+6.2f')}  vz {num(hud['vz'], '+6.2f')} m/s   grip {num(hud['grip'], '+5.2f')}",
        f"entry {hud['entry']}    stable exit {hud['exit']}    step cap {hud['cap']}",
    ]
    y = 4
    for i, line in enumerate(lines):
        fill = (255, 220, 60, 255) if i == 2 and lateral else (240, 240, 240, 255)
        font = FONT_B if i == 2 and lateral else FONT
        draw.text((12, y), line, font=font, fill=fill)
        y += 26 if i == 2 and lateral else 24
    if lateral:
        draw.rectangle((0, H - 48, W, H), fill=(180, 120, 0, 210))
        draw.text((12, H - 42), "TOKEN 32 ACTIVE  lateral chase", font=FONT_B, fill=(20, 12, 0, 255))
    out = Image.alpha_composite(image, layer).convert("RGB")
    return np.asarray(out)


def blank_cmd():
    return {"vx": float("nan"), "vy": float("nan"), "vz": float("nan"), "grip": float("nan")}


class Film:
    def __init__(self, sim, writers, csv_file, preview):
        self.sim = sim
        self.writers = writers
        self.preview = preview
        sim.model.vis.global_.offwidth = max(int(sim.model.vis.global_.offwidth), W)
        sim.model.vis.global_.offheight = max(int(sim.model.vis.global_.offheight), H)
        self.renderer = mujoco.Renderer(sim.model, H, W)
        self.cams = None
        self.frame = 0
        self.hud = {
            "t": 0.0, "mode": "NOMINAL", "e_pos": 0.0, "bilateral": False,
            "nL": 0, "nR": 0, "token": None, "entry": "--", "exit": "--", "cap": "--",
            **blank_cmd(),
        }
        self.action = None
        self.residual = None
        self.v_hand = np.zeros(3)
        self.token = None
        self.raw = None
        self._swept = False
        self.events = []
        self.policy_log = []
        self.t_loss = None
        self.t_recontact = None
        self.csv = csv.writer(csv_file)
        self.csv.writerow([
            "frame", "time", "nL", "nR",
            "p_rel_x", "p_rel_y", "p_rel_z",
            "v_rel_x", "v_rel_y", "v_rel_z",
            "token", "a0", "a1", "a2", "a3", "a4", "a5", "a6",
            "grip", "mode", "obj_z", "e_pos", "bilateral",
            "res0", "res1", "res2", "res3", "res4", "res5", "res6",
        ])
        self._switch = None

    def bind(self, switch):
        self._switch = switch

    def set_pose_cameras(self):
        if self.cams is None:
            self.cams = cameras(self.sim)

    def mark(self, key, text):
        self.hud[key] = text

    def grab(self):
        self.set_pose_cameras()
        sim = self.sim
        row_pack = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        nL, nR = int(row_pack["nL"]), int(row_pack["nR"])
        bilateral = nL > 0 and nR > 0
        if self.hud["mode"] == "RECOVERY":
            if (not bilateral) and self.t_loss is None:
                self.t_loss = float(sim.data.time)
            if self.t_loss is not None and bilateral and self.t_recontact is None:
                self.t_recontact = float(sim.data.time)
        e_pos = 0.0
        if self._switch is not None and self._switch.p0 is not None:
            p, _r = hand_pose(sim)
            e_pos = float(np.linalg.norm(p - self._switch.p0))
        act = self.action
        res = self.residual
        self.hud.update({
            "t": float(sim.data.time),
            "e_pos": e_pos,
            "bilateral": bilateral,
            "nL": nL,
            "nR": nR,
            "token": self.token,
            "vx": None if act is None else float(self.v_hand[0]),
            "vy": None if act is None else float(self.v_hand[1]),
            "vz": None if act is None else float(self.v_hand[2]),
            "grip": None if act is None else float(act[6]),
        })
        p_rel = np.asarray(rel["p_rel_h"], float)
        v_rel = np.asarray(rel["v_rel_h"], float)
        a = np.zeros(7) if act is None else np.asarray(act, float)
        r = np.zeros(7) if res is None else np.asarray(res, float)
        self.csv.writerow([
            self.frame, f"{float(sim.data.time):.6f}", nL, nR,
            *(f"{x:.6f}" for x in p_rel),
            *(f"{x:.6f}" for x in v_rel),
            "" if self.token is None else int(self.token),
            *(f"{x:.6f}" for x in a),
            f"{a[6]:.6f}", self.hud["mode"], f"{float(row_pack['obj_z']):.6f}",
            f"{e_pos:.6f}", int(bilateral),
            *(f"{x:.6f}" for x in r),
        ])
        if self.preview and self.hud["mode"] == "RECOVERY" and not self._swept:
            self._swept = True
            look = np.array(sim.data.xpos[sim.ids.hand_body], float)
            for az in range(0, 360, 45):
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(sim.model, cam)
                cam.lookat[:] = look
                cam.distance = 0.72
                cam.azimuth = float(az)
                cam.elevation = -18.0
                self.renderer.update_scene(sim.data, cam)
                Image.fromarray(self.renderer.render()).save(VID / f"sweep_{az:03d}.png")
        save_preview = self.preview and (
            self.frame < 3 or self.frame % 400 == 0 or self.hud["mode"] == "RECOVERY"
        )
        if not self.preview:
            for cam, writer in zip(self.cams, self.writers):
                self.renderer.update_scene(sim.data, cam)
                frame = paint(self.renderer.render(), self.hud)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        elif save_preview:
            self.renderer.update_scene(sim.data, self.cams[0])
            Image.fromarray(paint(self.renderer.render(), self.hud)).save(VID / f"preview_{self.frame:05d}.png")
            self.renderer.update_scene(sim.data, self.cams[1])
            Image.fromarray(paint(self.renderer.render(), self.hud)).save(VID / f"preview_side_{self.frame:05d}.png")
        self.frame += 1
        if self.frame % 1000 == 0:
            print(f"frame {self.frame} t {float(sim.data.time):.2f} {self.hud['mode']}", flush=True)


def policy_step(env, film):
    obs = env.observation()
    env._obs_seq.append(np.asarray(obs, np.float32))
    env._prev_seq.append(env._prev.copy())
    obs_in = env._obs_seq[-160:]
    prev_in = env._prev_seq[-160:]
    with torch.no_grad():
        logits, res, act = env.model(
            torch.tensor(np.stack(obs_in), device=env.device),
            torch.tensor(np.stack(prev_in), device=env.device),
        )
    logits_l = logits[-1].float().cpu().numpy()
    res_l = res[-1].float().cpu().numpy()
    raw = act[-1].float().cpu().numpy()
    token = int(np.argmax(logits_l))
    clipped = np.clip(raw, -1.0, 1.0).astype(np.float32)
    rh = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(env.sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(clipped, rd, rh, AIRBORNE_V3)
    film.token = token
    film.residual = res_l
    film.action = clipped
    film.v_hand = np.asarray(cmd["v_hand"], float)
    film.raw = raw
    film.policy_log.append({
        "t": float(env.sim.data.time),
        "step": len(film.policy_log),
        "token": token,
        "raw": [round(float(x), 4) for x in raw],
        "action": [round(float(x), 4) for x in clipped],
        "residual": [round(float(x), 4) for x in res_l],
        "v_hand": [round(float(x), 4) for x in cmd["v_hand"]],
        "clip_beyond_box": bool(np.any(np.abs(raw) > 1.0 + 1e-5)),
        "residual_outside_cap": bool(np.any(np.abs(res_l) > 0.25 + 1e-5)),
        "tau_cmd": round(float(cmd["tau"]), 4),
        "tau_sent": round(float(np.clip(cmd["tau"], env.sim.ids.ctrl_low[7], env.sim.ids.ctrl_high[7])), 4),
    })
    v3.step_v3(env.sim, clipped)
    env._prev = clipped
    return clipped


def fmt_t(value):
    return "--" if value is None else f"{value:.3f}s"


def main():
    preview = len(sys.argv) > 1 and sys.argv[1] == "preview"
    VID.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_training_env()
    table = supports_of(env.sim)
    model, _proto, cap = load_policy(device)
    env.model = model.to(env.device)
    env.model.eval()
    restore_supports(env.sim, table)
    writers = []
    paths = [
        VID / "impact_zp6_k33_lateral_chase.mp4",
        VID / "impact_zp6_k33_lateral_chase_side.mp4",
    ]
    if not preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for path in paths:
            writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"could not open {path}")
            writers.append(writer)
    trace_path = VID / ("preview_trace.csv" if preview else "impact_zp6_k33_lateral_chase_trace.csv")
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        film = Film(env.sim, writers, handle, preview)
        orig = mujoco.mj_step
        recording = {"on": False}

        def hooked(model_m, data):
            orig(model_m, data)
            if recording["on"]:
                film.grab()

        mujoco.mj_step = hooked
        try:
            env.begin_nominal()
            film.bind(env.switch)
            film.grab()
            ball = empty_ball()
            recording["on"] = True
            switch = env.switch
            while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
                for _ in range(v3.N_SUB):
                    ball_substep(
                        env.sim, env.gains, env.nominal, ball,
                        SPEC["v_mps"], SPEC["z_off_m"], SPEC["phi_deg"], None,
                    )
                    if float(env.sim.data.time) >= T_TASK - 1e-9:
                        break
                row = switch.measure(env.sim)
                if ball["released"]:
                    switch.update_entry(env.sim)
                if ball["released"] and row["z"] < Z_ABORT:
                    break
                if (not ball["released"]) and float(env.sim.data.time) > 8.0:
                    break
            if not switch.entered:
                raise RuntimeError("geometry entry did not fire")
            film.mark("entry", fmt_t(switch.t_entry))
            env.start_recovery()
            film.hud["mode"] = "RECOVERY"
            kind = "time_cap"
            while float(env.sim.data.time) < T_TASK - 1e-9 and len(film.policy_log) < 300:
                policy_step(env, film)
                row = env.switch.measure(env.sim)
                if row["z"] < Z_ABORT:
                    kind = "fall_abort"
                    break
                if env.switch.update_exit(env.sim):
                    kind = "geometry"
                    break
                if preview and len(film.policy_log) >= 2:
                    kind = "preview_stop"
                    break
            capped = kind == "time_cap" and len(film.policy_log) >= 300
            if capped:
                film.mark("cap", fmt_t(float(env.sim.data.time)))
            if env.switch.exited:
                film.mark("exit", fmt_t(env.switch.t_exit))
            if kind == "fall_abort":
                t12 = "DROP"
            elif preview:
                t12 = "PREVIEW"
            else:
                film.hud["mode"] = "NOMINAL"
                film.token = None
                film.action = None
                film.residual = None
                t12, _t = env.finish_nominal(TRACK)
            end_pack = physical_pack(env.sim)
            end_rel = relative_state(env.sim.model, env.sim.data, env.sim.ids)
        finally:
            mujoco.mj_step = orig
            recording["on"] = False
            for writer in writers:
                writer.release()
            film.renderer.close()
    summary = {
        "checkpoint": str(CKPT),
        "res_cap": cap,
        "spec": SPEC,
        "t_impact": None if ball["t_impact"] is None else float(ball["t_impact"]),
        "t_entry": None if switch.t_entry is None else float(switch.t_entry),
        "t_exit": None if env.switch.t_exit is None else float(env.switch.t_exit),
        "stable_exit": bool(env.switch.exited),
        "recovery_end": kind,
        "policy_steps": len(film.policy_log),
        "hit_step_cap": bool(capped) if not preview else False,
        "t12": t12,
        "t_end": float(env.sim.data.time),
        "nL": int(end_pack["nL"]),
        "nR": int(end_pack["nR"]),
        "obj_z": float(end_pack["obj_z"]),
        "p_rel": [round(float(x), 6) for x in end_rel["p_rel_h"]],
        "v_rel": [round(float(x), 6) for x in end_rel["v_rel_h"]],
        "v_rel_speed": float(np.linalg.norm(end_rel["v_rel_h"])),
        "frames": film.frame,
        "t_loss": film.t_loss,
        "t_recontact": film.t_recontact,
        "chase": film.policy_log[:4],
    }
    out = VID / ("preview_summary.txt" if preview else "impact_zp6_k33_summary.txt")
    lines = [f"{k}={v}" for k, v in summary.items()]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
