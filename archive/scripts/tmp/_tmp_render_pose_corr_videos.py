"""Re-render pose-correction videos starting at RECATCh onset (no idle preamble)."""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import measure
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, decode
from training.action_vocab_k49 import build_k49_table
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SRC = ROOT / "results" / "expert_generation" / "lateral_force_grasp_loss_recovery"
OUT = ROOT / "results" / "expert_generation" / "lateral_force_full_pose_recovery"
VID = OUT / "videos"
HOLD_S = 2.5

ICS = [
    "lforce_0000__SLIDING_BILATERAL__0",
    "lforce_0001__SLIDING_BILATERAL__0",
    "lforce_0009__SLIDING_BILATERAL__0",
    "lforce_0026__SLIDING_BILATERAL__0",
    "lforce_0036__SLIDING_BILATERAL__0",
]


def render_one(env, table, book48, ic_id: str) -> Path:
    meta = json.loads((OUT / "trajectories" / f"{ic_id}__POSECORR_pos_meta.json").read_text(encoding="utf-8"))
    tokens = [int(t) for t in meta["tokens"]]
    snap = SRC / "snapshots" / f"{ic_id}.pkl"
    blob = pickle.loads(snap.read_bytes())

    restore_capture(env, blob["capture"])
    env.sim.data.xfrc_applied[:] = 0.0
    restore_table_drop(env.sim, blob["capture"].get("table_drop"))

    sim = env.sim
    look = 0.5 * (
        np.asarray(sim.data.xpos[sim.ids.object_body], float)
        + np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    )
    cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.52, "azimuth": 135.0, "elevation": -18.0}
    )
    renderer = mujoco.Renderer(sim.model, 480, 640)
    frames = []
    t0 = float(sim.data.time)

    def grab(label: str):
        cam.lookat[:] = 0.5 * (
            np.asarray(sim.data.xpos[sim.ids.object_body], float)
            + np.asarray(sim.data.xpos[sim.ids.hand_body], float)
        )
        renderer.update_scene(sim.data, cam)
        m = measure(sim, U_TGT, P_TGT)
        cs = contact_summary(sim)
        text = (
            f"{ic_id} | {label} | t={float(sim.data.time)-t0:.2f}s | "
            f"e_pos={m['e_pos_mm']:.1f}mm e_th={m['e_theta_deg']:.1f}deg | "
            f"bil={int(cs['bilateral'])} F=0"
        )
        frames.append(caption(renderer.render(), text))

    # Start video at RECATCh onset (post-force IC), then first recovery step.
    grab("RECATCh_ONSET")
    for i, tok in enumerate(tokens):
        action = np.asarray(decode(table, book48, int(tok)), np.float32)
        step_legal(sim, action)
        env._prev = action.copy()
        # denser early recovery, then every 2 steps
        if i < 40 or i % 2 == 0 or i == len(tokens) - 1:
            grab("RECATCh")
        if latched(sim):
            grab("TABLE_DROP")
            break

    if not latched(sim):
        apply_handoff(sim, blob["ref"])
        grab("HANDOFF")
        t_h = float(sim.data.time)
        step = 0
        while float(sim.data.time) - t_h < HOLD_S - 1e-9:
            nominal_step(sim)
            step += 1
            if step % 2 == 0:
                grab("NOMINAL")
            if latched(sim):
                grab("DROP")
                break
    grab("END")
    renderer.close()

    VID.mkdir(parents=True, exist_ok=True)
    dest = VID / f"{ic_id}__POSECORR_pos.mp4"
    # remove old
    if dest.exists():
        dest.unlink()
    path = write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print(f"ok {dest.name} frames={len(frames)} start=RECATCh_ONSET", flush=True)
    return Path(path) if path else dest


def main():
    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    for ic_id in ICS:
        render_one(env, table, book48, ic_id)
    print("done", VID, flush=True)


if __name__ == "__main__":
    main()
