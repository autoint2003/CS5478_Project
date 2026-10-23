#!/usr/bin/env python
"""Diagnostic video: continuous teacher vs frozen K49 replay (same IC)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.stage2_mining.render_stage2_recovery_vs_nominal import (  # noqa: E402
    FINAL,
    OUT,
    make_env,
    overlay_text,
    pad_pair,
    resolve_actions,
    resolve_ic,
    run_branch,
    shoot,
)
from training.action_vocab_k49 import build_k49_table, CAP, HANDOFF
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal

V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
K49_CONFIG = "K49_tok48_wy0p25"


def decode_k49_actions(acts, table):
    """Nearest-token + residual (±CAP) decode matching training path."""
    rows = np.asarray(table, np.float32)
    out = []
    for a in np.asarray(acts, np.float32):
        d = np.linalg.norm(rows[:, :7] - a.reshape(1, 7), axis=1)
        tok = int(np.argmin(d))
        proto = rows[tok, :7]
        resid = np.clip(a - proto, -CAP, CAP)
        out.append(proto + resid)
    return np.asarray(out, np.float32)


def main():
    sid = "B__hard_910166__post_b1__7b45f78c0ce7796e"
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    pack = pickle.loads(resolve_ic(sid).read_bytes())
    cap, ref = pack["capture"], pack["ref"]
    acts = resolve_actions(sid)
    table = build_k49_table(config_id=K49_CONFIG)
    # Prefer stored tokens if present
    k49_acts = None
    for base in (FINAL, V1):
        for p in (base / "tokens").glob(f"{sid}*"):
            z = np.load(p)
            if "continuous_actions" in z:
                # re-decode from tokens+residuals if available
                if "tokens" in z and "residuals" in z:
                    toks = np.asarray(z["tokens"], np.int64)
                    res = np.asarray(z["residuals"], np.float32)
                    rows = np.asarray(table, np.float32)
                    k49_acts = np.asarray([rows[int(t), :7] + res[i] for i, t in enumerate(toks)], np.float32)
                break
    if k49_acts is None:
        k49_acts = decode_k49_actions(acts, table)

    restore_capture(env, cap)
    look = 0.5 * (
        np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
        + np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    )
    import experiments.stage2_mining.render_stage2_recovery_vs_nominal as R

    R.cam = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -20.0}
    )
    R.renderer = mujoco.Renderer(env.sim.model, 360, 480)

    left, left_out = run_branch(env, cap, ref, acts, label_plan="B", sid=sid, branch_name="CONTINUOUS TEACHER")
    right, right_out = run_branch(env, cap, ref, k49_acts, label_plan="B", sid=sid, branch_name="K49 REPLAY")
    frames = pad_pair(left, right)
    dest = OUT / f"B_k49_vs_cont_diag__{sid}__vs.mp4"
    write_frames_mp4(frames, dest, fps=25)
    meta = {
        "sample_id": sid,
        "tag": "B_k49_vs_continuous_diag",
        "video": str(dest.relative_to(ROOT)),
        "continuous": left_out,
        "k49": right_out,
        "note": "Diagnostic only — not a Recovery vs Nominal pair. Quantization-induced contact failure.",
    }
    (OUT / f"B_k49_vs_cont_diag__{sid}__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", dest, meta["continuous"]["SUCCESS_NEW"], meta["k49"]["SUCCESS_NEW"])


if __name__ == "__main__":
    main()
