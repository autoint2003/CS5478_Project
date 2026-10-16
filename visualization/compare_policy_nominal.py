"""Side-by-side closed-loop policy versus immediate nominal handoff.

Both panels start from the same recovery snapshot. The left panel runs the
deterministic K48 policy (argmax token, residual mean). The right panel hands
off to the nominal controller immediately. Labels match horizon_outcome.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.ballistic_impact import z_tgt_of
from envs.outcomes.horizon_outcome import (
    LOST_STEPS,
    NOMINAL_S,
    _bilateral,
    _dropped,
    _fail_recovery,
    _monitor,
    _row,
    recovery_failed,
)
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.recovery_execution import _escaped, restore_ballistic
from training.reward_v2 import NominalHoldTrace
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SAMPLE_S = 0.04
OUT = ROOT / "results" / "videos" / "policy_vs_nominal"


def load_rows(manifest):
    rows = []
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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


def prepare_blob(blob, seated_ref):
    if "ref" in blob and ("capture" in blob or "qpos" in blob):
        if "ref" in blob:
            return blob
    blob = dict(blob)
    if "ref" not in blob:
        if seated_ref is None:
            raise RuntimeError("horizontal snapshot needs the seated grasp reference")
        blob["ref"] = seated_ref
    return blob


def restore_ic(env, blob):
    if "capture" in blob:
        restore_capture(env, blob["capture"])
        return
    from models.airborne_v3 import DT_POLICY
    from envs.observations.observable_obs import ObservableObsState
    from envs.outcomes.table_drop import restore_state

    restore_ballistic(env.sim, blob)
    restore_state(env.sim, None)
    env._hist = ObservableObsState(DT_POLICY)
    env._obs_seq = []
    env._prev_seq = []
    env._prev = np.zeros(7, np.float32)


def _maybe_frame(frames, grab, sim, next_frame, text):
    now = float(sim.data.time)
    if now + 1e-9 >= next_frame:
        frames.append((grab(sim), text))
        return next_frame + SAMPLE_S
    return next_frame


def play_policy(env, model, book, blob, device, grab, limit_s):
    frames = []
    t0 = float(env.sim.data.time)
    next_frame = t0
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        pack = physical_pack(env.sim)
        next_frame = _maybe_frame(
            frames, grab, env.sim, next_frame,
            f"policy recover {float(env.sim.data.time) - t0:.2f}s tok={token} n={int(pack['nL'])}/{int(pack['nR'])}",
        )
        if token == HANDOFF:
            how = "early"
            break
        _residual, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out = _fail_recovery(env.sim, blob["ref"], t0)
            frames.append((grab(env.sim), f"policy {out['reason']} {out['recovery_elapsed']:.2f}s"))
            return frames, out["reason"], float(out["recovery_elapsed"]), 0.0
    if how is None:
        how = "forced"
    reason, recovery_elapsed, nominal_elapsed = nominal_hold_framed(
        env, blob["ref"], t0, how, grab, frames, "policy",
    )
    return frames, reason, recovery_elapsed, nominal_elapsed


def nominal_hold_framed(env, ref, t0, how, grab, frames, title):
    """Same rules as nominal_verify, with video frames."""
    sim = env.sim
    handoff = apply_handoff(sim, ref)
    if abs(float(handoff["time_delta"])) > 1e-9 or abs(float(handoff["qpos_max_abs"])) > 1e-12:
        raise RuntimeError("handoff edited the physical state")
    t_hand = float(sim.data.time)
    next_frame = t_hand
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    hold_trace = NominalHoldTrace(NOMINAL_S)
    if latched(sim):
        reason = "TABLE_DROP"
    while reason is None and float(sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        t_prev = float(sim.data.time)
        nominal_step(sim)
        row = _monitor(sim)
        end_row = row
        target = float(z_tgt_of(sim))
        bilateral = _bilateral(row)
        command_reached = float(sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(sim)
        hold_trace.observe(float(sim.data.time) - t_prev, bilateral, dropped, command_reached)
        if bilateral and command_reached:
            lift_done = True
        if latched(sim):
            reason = "TABLE_DROP"
            break
        if dropped:
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
        next_frame = _maybe_frame(
            frames, grab, sim, next_frame,
            f"{title} hold {float(sim.data.time) - t_hand:.2f}/12s",
        )
    nominal_elapsed = float(sim.data.time) - t_hand
    recovery_elapsed = t_hand - t0
    if reason is None:
        if nominal_elapsed < NOMINAL_S - 1e-6:
            reason = "PHYSICAL_VERIFICATION_FAILED"
        elif end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    frames.append((grab(sim), f"{title} {reason}"))
    return reason, recovery_elapsed, nominal_elapsed


def play_nominal(env, ref, grab):
    frames = []
    t0 = float(env.sim.data.time)
    reason, _rec, _nom = nominal_hold_framed(env, ref, t0, "immediate", grab, frames, "nominal")
    return frames, reason


def stitch(left, right):
    count = max(len(left), len(right))
    left = pad(left, count)
    right = pad(right, count)
    return [pair(image_l, image_r, text_l, text_r) for (image_l, text_l), (image_r, text_r) in zip(left, right)]


def main():
    parser = argparse.ArgumentParser(description="Policy vs nominal videos on a recovery list.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/valid_skill_quotas.jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-s", type=float, default=SELECTED_LIMIT_S)
    parser.add_argument("--id", default="", help="Optional single episode id.")
    args = parser.parse_args()

    out = Path(args.out) if args.out else OUT / Path(args.checkpoint).stem
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, book = load_trained_k48(ROOT / args.checkpoint, device)
    model.eval()
    codebook = load_codebook()
    if not np.array_equal(book, codebook):
        raise RuntimeError("checkpoint codebook differs from the configured codebook")
    env = make_training_env(device, model=model)
    rows = load_rows(ROOT / args.manifest)
    if args.id:
        rows = [row for row in rows if row["id"] == args.id]
        if not rows:
            raise RuntimeError(f"missing id {args.id}")
    seated_ref = None
    summary = []
    for row in rows:
        with (ROOT / row["snapshot"]).open("rb") as handle:
            blob = pickle.load(handle)
        if "ref" not in blob:
            if seated_ref is None:
                from data_generation.strategies.recatch_oracle import _setup

                _sim, _base, seated_ref, _target, _code = _setup()
        blob = prepare_blob(blob, seated_ref)
        cats = ",".join(row.get("categories") or [])
        print("CASE", row["id"], cats, flush=True)
        cam, renderer = open_camera(env.sim)

        def grab(sim):
            return shoot(sim, cam, renderer)

        restore_ic(env, blob)
        left, left_label, left_rec, left_nom = play_policy(
            env, model, book, blob, device, grab, float(args.limit_s),
        )
        restore_ic(env, blob)
        right, right_label = play_nominal(env, blob["ref"], grab)
        renderer.close()
        frames = stitch(left, right)
        dest = out / f"{row['id']}__policy_vs_nominal.mp4"
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
        entry = {
            "id": row["id"],
            "categories": row.get("categories") or [],
            "policy_label": left_label,
            "nominal_label": right_label,
            "policy_recovery_s": left_rec,
            "policy_nominal_s": left_nom,
            "frames": len(frames),
            "video": str(dest.relative_to(ROOT)).replace("\\", "/"),
        }
        summary.append(entry)
        print(
            "WROTE", dest.name, "policy", left_label, "nominal", right_label,
            "frames", len(frames), flush=True,
        )
    report = {
        "checkpoint": args.checkpoint.replace("\\", "/"),
        "manifest": args.manifest.replace("\\", "/"),
        "episodes": len(summary),
        "policy_success": sum(1 for row in summary if row["policy_label"] == "SUCCESS_12S"),
        "nominal_success": sum(1 for row in summary if row["nominal_label"] == "SUCCESS_12S"),
        "rows": summary,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DONE", out, "policy", report["policy_success"], "nominal", report["nominal_success"], flush=True)


if __name__ == "__main__":
    main()
