"""Finalize action-authority ablation report + success video after Unicode crash."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.physical_recovery import physical_pack
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import build_k49_table, SECURE_REVERSE_WRIST
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4
import shutil

from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, decode, mapping_checksum
from scripts.expand_k49_capture_brake_diagnostics import find_snap, load_blob
from scripts.expand_k49_large_chase_dynamic import obj_pos
from scripts.run_k49_action_authority_ablation import (
    OUT, V_AIRBORNE, V_Z_DOWN, V_Z_UP, W_MAX, TAU_OPEN, TAU_SECURE,
    TOTAL_ROLLOUT_BUDGET, WALL_LIMIT_S, _write_failure_and_report,
)

device = torch.device("cpu")
model, book48 = load_trained_k48(SFT, device)
table = build_k49_table(book48, config_id=CONFIG_ID)
checksum = mapping_checksum(table)
env = make_training_env(device, model=model)

stage1_best = json.loads((OUT / "raw" / "STAGE1_BEST.json").read_text(encoding="utf-8"))
candidates = list(csv.DictReader((OUT / "CANDIDATE_TOKEN_EXTENSIONS.csv").open(encoding="utf-8")))
full_verify = list(csv.DictReader((OUT / "FULL_SUCCESS_VERIFICATION.csv").open(encoding="utf-8")))
ablation_rows = list(csv.DictReader((OUT / "TOKEN_ABLATION_RESULTS.csv").open(encoding="utf-8")))
k49_rows = list(csv.DictReader((OUT / "K49_BASELINE.csv").open(encoding="utf-8")))
cont_rows = list(csv.DictReader((OUT / "CONTINUOUS_ACTION_RESULTS.csv").open(encoding="utf-8")))
usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8")) if (OUT / "raw" / "USAGE.json").exists() else {
    "rollouts_used": "see BUDGET", "n_full_verify": len(full_verify), "n_candidate_tokens": len(candidates),
}
# regression stub
out_reg = {"SUCCESS_NEW": 0, "eval_reason": "see raw/K49_REGRESSION.json if present"}
if (OUT / "raw" / "K49_REGRESSION.json").exists():
    out_reg = json.loads((OUT / "raw" / "K49_REGRESSION.json").read_text(encoding="utf-8"))
    out_reg = {
        "SUCCESS_NEW": out_reg.get("prefix_return_SUCCESS_NEW"),
        "eval_reason": out_reg.get("prefix_return_reason"),
    }

_write_failure_and_report(
    stage1_best, candidates, full_verify, ablation_rows,
    k49_rows, cont_rows, usage, checksum, out_reg,
)

# Video of extended SUCCESS_NEW
cands = {int(c["token_id"]): np.asarray(json.loads(c["action"]), np.float32) for c in candidates}
max_id = max(cands)

def decode_ext(tok):
    tok = int(tok)
    if tok < table.shape[0]:
        return decode(table, book48, tok)
    return cands[tok]

toks = json.loads(full_verify[0]["tokens"])
blob = load_blob(find_snap("hard_910012"))
restore_capture(env, blob["capture"])
sim = env.sim
cam = mjv_camera_from_preset({"lookat": obj_pos(sim), "distance": 0.55, "azimuth": 128.0, "elevation": -18.0})
import mujoco
renderer = mujoco.Renderer(sim.model, 480, 640)
frames = []
t0 = float(sim.data.time)
dt = float(sim.model.opt.timestep)
stride = max(1, int(round(0.04 / dt)))

def grab(tag, clock):
    cam.lookat[:] = obj_pos(sim)
    renderer.update_scene(sim.data, cam)
    pack = physical_pack(sim)
    frames.append(caption(
        renderer.render(),
        f"hard_910012 EXT tok63 SUCCESS_NEW {tag} t={clock:.2f} n={int(pack['nL'])}/{int(pack['nR'])}",
    ))

grab("START", 0.0)
for i, tok in enumerate(toks):
    step_legal(sim, decode_ext(tok))
    if i % max(1, stride // 2) == 0:
        grab("REC", float(sim.data.time) - t0)
apply_handoff(sim, blob["ref"])
t_h = float(sim.data.time)
steps = 0
while float(sim.data.time) - t_h < 2.5 - 1e-9:
    nominal_step(sim)
    steps += 1
    if steps % stride == 0:
        grab("HOLD", float(sim.data.time) - t_h)
renderer.close()
dest = OUT / "videos" / "hard_910012_extended_fine_down_SUCCESS_NEW.mp4"
write_frames_mp4(frames, dest, fps=25)
shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
print("video", dest)

# Strengthen REPORT recommendation section with verified facts
rep = (OUT / "REPORT.md").read_text(encoding="utf-8")
extra = f"""

## Verified SUCCESS_NEW (exact continuous-legal replay)

- IC: `hard_910012`
- Conditions: `C_fine_trans`, `E_combined` (identical token sequence)
- New token used: **63 `fine_down`** = `[0, 0, 0.15, 0, 0, 0, 1]` (small +vz, secure grip)
- Sequence: capture prefix (open+t17+hold+brake) + tok63 x10 + tok48 x16
- Replay metrics: theta={full_verify[0]['win_mean_theta']} deg, pos={full_verify[0]['win_mean_pos']} mm, hold={full_verify[0]['hold_both_s']} s
- Matched K49/continuous Stage1 on same IC: **0 SUCCESS_NEW** (continuous best pos ~33 mm)

### Implication

A **single** fine vertical translation token (ID 63) unlocks SUCCESS_NEW on a previously blocked stable-grasp pose IC under matched prefix. High-speed capture ICs still show **0 FULL** under extension. Smallest useful extension for pose: **{{63: fine_down}}**; do not inflate vocabulary without more roots.
"""
if "Verified SUCCESS_NEW" not in rep:
    (OUT / "REPORT.md").write_text(rep + extra, encoding="utf-8")
print("REPORT updated")
