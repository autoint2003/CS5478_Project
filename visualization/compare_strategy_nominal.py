"""Side-by-side strategy versus nominal videos.

Each clip starts when the hand descends to grasp the cylinder, follows the
real ball impact, then splits. The left panel replays the saved recovery
tape. The right panel is the official immediate nominal handoff. Both panels
keep simulating for 12 s after their own handoff.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.airborne_v3 as v3
from data_generation.ball_impact_dataset import LAUNCH_WINDOW_S, aim_ball, cylinder_contacts, ready_to_launch
from data_generation.rebuild_k48_recovery import _open_env
from data_generation.strategies.recatch_oracle import _aim, _flight_step, _pose, _setup
from envs.ballistic_impact import z_tgt_of
from envs.initialization.grasp_seat import hold_step, nominal_physics_step, seat_horizontal
from envs.simulation.grasp_quality import body_twist
from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.physical_recovery import Z_AIR, physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.simulation.capture import restore_capture
from envs.simulation.impact_ball import apply_ball_free_state
from envs.simulation.scene_state import TABLE_TOP, park_impact_ball
from envs.ballistic_impact import HAND_RISE_M
from envs.outcomes.table_drop import latched
from envs.step_legal import step_legal
from training.recovery_episode import T_TASK, Z_ABORT, fresh_nominal_state, nominal_substep
from training.recovery_execution import make_snap, restore_ballistic
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "videos" / "strategy_vs_nominal"
SAMPLE_S = 0.04
CASES = (
    ("controlled_sliding", "ball_train_v2_0070", "controlled_slide", "data/controlled_sliding/manifest.jsonl"),
    ("inward_sliding", "inward_extra_250005", "inward_reseat", "data/inward_sliding/manifest.jsonl"),
    ("vertical_recatch", "ball_train_v2_0168", "follow_early", "data/vertical_recatch/manifest.jsonl"),
    ("recatch", "ref_s6_p10_a0_e0", "reference", "data/recatch/manifest.jsonl"),
)


def load_row(manifest, episode, strategy):
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == episode and row["strategy"] == strategy:
            return row
    raise RuntimeError(f"missing {episode} {strategy}")


def load_blob(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def target_qpos(blob):
    physics = (blob.get("capture") or {}).get("physics") or blob
    return np.asarray(physics["qpos"], float)


def open_camera(sim):
    look = np.array(sim.data.xpos[sim.ids.object_body], float)
    cam = mjv_camera_from_preset({"lookat": look, "distance": 0.55, "azimuth": 132.0, "elevation": -22.0})
    renderer = mujoco.Renderer(sim.model, 480, 640)
    return cam, renderer


def shoot(sim, cam, renderer):
    cam.lookat[:] = np.array(sim.data.xpos[sim.ids.object_body], float)
    renderer.update_scene(sim.data, cam)
    return np.asarray(renderer.render(), np.uint8).copy()


def pair(left, right, left_text, right_text):
    return np.hstack([caption(left, left_text), caption(right, right_text)])


def pad(frames, count):
    if not frames:
        raise RuntimeError("empty frame list")
    if len(frames) >= count:
        return frames[:count]
    return frames + [frames[-1]] * (count - len(frames))


def replay_sliding_prefix(env, spec, grab):
    """Nominal descend, grasp, lift, and ball impact up to geometry entry."""
    env.begin_nominal()
    st = fresh_nominal_state()
    switch = env.switch
    launched = False
    t_ready = None
    t_launch = None
    dropped = False
    frames = []
    next_frame = None
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
            now = float(env.sim.data.time)
            if (not launched) and ready_to_launch(env.sim, st):
                if t_ready is None:
                    t_ready = now
                if now + 1e-9 >= t_ready + float(spec["delay_s"]):
                    launch = aim_ball(
                        env.sim, spec["azimuth_deg"], spec["speed_mps"], spec["z_off_m"], spec["phi_deg"],
                    )
                    env.sim.set_guide_weld(False)
                    apply_ball_free_state(env.sim, launch["p_launch"], launch["v0"])
                    launched = True
                    t_launch = float(env.sim.data.time)
            if launched and float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
                dropped = True
                break
        if dropped:
            break
        phase = str(env.sim.fsm.phase)
        if phase == "descend" and next_frame is None:
            next_frame = float(env.sim.data.time)
        if next_frame is not None and float(env.sim.data.time) + 1e-9 >= next_frame:
            frames.append(grab(env.sim))
            next_frame += SAMPLE_S
        switch.measure(env.sim)
        if launched and switch.p0 is not None:
            switch.update_entry(env.sim)
        if launched and t_launch is not None and float(env.sim.data.time) > t_launch + LAUNCH_WINDOW_S:
            break
        if (not launched) and float(env.sim.data.time) > 9.0:
            break
    if next_frame is not None:
        frames.append(grab(env.sim))
    return frames, bool(switch.entered), bool(dropped)


def qpos_error(sim, target):
    live = np.asarray(sim.data.qpos, float)
    if live.shape != target.shape:
        return None
    return float(np.max(np.abs(live - target)))


def play_recovery(sim, actions, ref, grab, title):
    frames = []
    t0 = float(sim.data.time)
    next_frame = t0
    for index, action in enumerate(actions):
        step_legal(sim, action)
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            frames.append(grab(sim))
            frames[-1] = (frames[-1], f"{title} recover {now - t0:.2f}s")
            next_frame += SAMPLE_S
    handoff = apply_handoff(sim, ref)
    if abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    t_hand = float(sim.data.time)
    next_frame = t_hand
    while float(sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        nominal_step(sim)
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            hold = now - t_hand
            frames.append(grab(sim))
            frames[-1] = (frames[-1], f"{title} hold {hold:.2f}/12s")
            next_frame += SAMPLE_S
    frames.append((grab(sim), f"{title} hold {float(sim.data.time) - t_hand:.2f}/12s"))
    return frames


def play_nominal(sim, ref, grab):
    frames = []
    handoff = apply_handoff(sim, ref)
    if abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    t_hand = float(sim.data.time)
    next_frame = t_hand
    while float(sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        nominal_step(sim)
        now = float(sim.data.time)
        if now + 1e-9 >= next_frame:
            frames.append((grab(sim), f"nominal hold {now - t_hand:.2f}/12s"))
            next_frame += SAMPLE_S
    frames.append((grab(sim), f"nominal hold {float(sim.data.time) - t_hand:.2f}/12s"))
    return frames


def sliding_case(env, row, blob):
    spec = blob["spec"]
    target = target_qpos(blob)
    cam, renderer = open_camera(env.sim)

    def grab(sim):
        return shoot(sim, cam, renderer)

    prefix, entered, dropped = replay_sliding_prefix(env, spec, grab)
    err = qpos_error(env.sim, target)
    print("PREFIX", row["id"], "frames", len(prefix), "entered", entered, "dropped", dropped, "qpos", err, flush=True)
    restore_capture(env, blob["capture"])
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    left = play_recovery(env.sim, actions, blob["ref"], grab, row["strategy"])
    restore_capture(env, blob["capture"])
    right = play_nominal(env.sim, blob["ref"], grab)
    renderer.close()
    return prefix, left, right, err


def record_recatch_prefix(sim, gains, cell, grab):
    sim.model.opt.timestep = 0.002
    sim.reset()
    seat_horizontal(sim)
    hand_z0 = None
    risen_at = None
    frames = []
    next_frame = None
    for _ in range(int(8.0 / 0.002)):
        if risen_at is not None and float(sim.fsm.p_des[2]) >= float(z_tgt_of(sim)) - 1e-4:
            hold_step(sim, gains)
        else:
            nominal_physics_step(sim)
        phase = str(sim.fsm.phase)
        if phase == "descend" and next_frame is None:
            next_frame = float(sim.data.time)
        if next_frame is not None and float(sim.data.time) + 1e-9 >= next_frame:
            frames.append(grab(sim))
            next_frame += SAMPLE_S
        hz = float(sim.data.xpos[sim.ids.hand_body][2])
        if sim.fsm.phase == "lift" and hand_z0 is None:
            hand_z0 = hz
        pack = physical_pack(sim)
        risen = (
            hand_z0 is not None
            and (hz - hand_z0) >= HAND_RISE_M + 0.03
            and int(pack["nL"]) > 0
            and int(pack["nR"]) > 0
            and float(pack["obj_z"]) >= Z_AIR
            and float(sim.data.ctrl[7]) <= -17.0
        )
        if risen and risen_at is None:
            risen_at = float(sim.data.time)
        vz = float(body_twist(sim.model, sim.data, sim.ids.object_body)[0][2])
        if risen_at is not None and float(sim.data.time) >= risen_at + 0.30 and abs(vz) < 0.02:
            break
    else:
        return frames, None
    frames.append(grab(sim))
    seated = make_snap(sim)
    restore_ballistic(sim, seated)
    sim.set_guide_weld(False)
    launch = _aim(sim, cell["speed"], cell["pitch"], cell["axial"], cell["ecc"])
    before = _pose(sim)
    apply_ball_free_state(sim, launch["p_launch"], launch["v0"])
    t0 = float(sim.data.time)
    contact_t = None
    unstable = None
    unstable_snap = None
    for _ in range(int(0.45 / 0.002)):
        _flight_step(sim)
        now = float(sim.data.time)
        if next_frame is None or now + 1e-9 >= next_frame:
            frames.append(grab(sim))
            next_frame = now + SAMPLE_S
        hits, _other = cylinder_contacts(sim)
        if hits and contact_t is None:
            contact_t = now
        if contact_t is None:
            if now > t0 + 0.30:
                break
            continue
        pose = _pose(sim)
        age = now - contact_t
        slipped = pose["p_rel"][2] > before["p_rel"][2] + 0.004 or pose["v_obj"][2] < -0.15
        lost_side = (pose["nL"] == 0) != (pose["nR"] == 0) or (pose["nL"] + pose["nR"] == 0)
        if age > 0.012 and (slipped or lost_side or pose["tilt_deg"] > before["tilt_deg"] + 8.0):
            if unstable is None or pose["p_rel"][2] > unstable["p_rel"][2]:
                unstable = pose
                unstable_snap = make_snap(sim)
        if latched(sim) or pose["bottom"] < TABLE_TOP + 0.02 or age > 0.22:
            break
    if unstable_snap is None:
        park_impact_ball(sim)
        end = make_snap(sim)
    else:
        restore_ballistic(sim, unstable_snap)
        park_impact_ball(sim)
        end = make_snap(sim)
    frames.append(grab(sim))
    return frames, end


def recatch_case(row, blob):
    sim, base, ref, _target, _book = _setup()
    from envs.ballistic_impact import world
    _sim2, gains, _nominal, *_rest = world()
    del _sim2
    cam, renderer = open_camera(sim)
    cell = {
        "speed": row["speed"], "pitch": row["pitch"], "axial": row["axial"], "ecc": row["ecc"],
        "mass": row.get("mass"), "friction": row.get("friction"),
    }

    def grab(active):
        return shoot(active, cam, renderer)

    prefix, live_snap = record_recatch_prefix(sim, gains, cell, grab)
    err = None if live_snap is None else qpos_error_arrays(live_snap, blob)
    print("PREFIX", row["id"], "frames", len(prefix), "qpos", err, flush=True)
    restore_ballistic(sim, blob)
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    left = play_recovery(sim, actions, ref, grab, row["strategy"])
    restore_ballistic(sim, blob)
    right = play_nominal(sim, ref, grab)
    renderer.close()
    return prefix, left, right, err


def qpos_error_arrays(snap, blob):
    live = np.asarray(snap["qpos"], float)
    target = np.asarray(blob["qpos"], float)
    if live.shape != target.shape:
        return None
    return float(np.max(np.abs(live - target)))


def stitch(prefix, left, right, strategy):
    shared = [pair(frame, frame, f"{strategy} descend/grasp", "nominal descend/grasp") for frame in prefix]
    count = max(len(left), len(right))
    left = pad(left, count)
    right = pad(right, count)
    split = [pair(image_l, image_r, text_l, text_r) for (image_l, text_l), (image_r, text_r) in zip(left, right)]
    return shared + split


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = set(sys.argv[1:])
    env = None
    for family, episode, strategy, manifest in CASES:
        if chosen and family not in chosen:
            continue
        row = load_row(ROOT / manifest, episode, strategy)
        blob = load_blob(ROOT / row["snapshot"])
        print("CASE", family, episode, flush=True)
        if family == "recatch":
            prefix, left, right, err = recatch_case(row, blob)
        else:
            if env is None:
                env = _open_env()
            prefix, left, right, err = sliding_case(env, row, blob)
        frames = stitch(prefix, left, right, strategy)
        dest = OUT / f"{family}__{episode}__vs_nominal.mp4"
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
        print("WROTE", dest, "frames", len(frames), "qpos_err", err, flush=True)


if __name__ == "__main__":
    main()
