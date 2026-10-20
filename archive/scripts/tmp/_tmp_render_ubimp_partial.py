import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import build_k49_table
from training.recovery_episode import make_training_env
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT
from scripts.expand_k49_large_chase_dynamic import OUT, load_blob, render_expert_video, replay_motion

snap = ROOT / "results/expert_generation/k49_unified_dataset_expansion/snapshots/ubimp_0040.pkl"
device = torch.device("cpu")
model, book = load_trained_k48(SFT, device)
table = build_k49_table(book, config_id=CONFIG_ID)
env = make_training_env(device, model=model)
blob = load_blob(snap)
seeds = [
    [17] * 18 + [30] * 3 + [0] * 8 + [12] * 8,
    [33] * 18 + [30] * 3 + [0] * 8 + [12] * 8,
    [1] * 20 + [30] * 3 + [0] * 8 + [12] * 8,
]
best_toks = seeds[0]
best_m = None
for s in seeds:
    m, _ = replay_motion(env, table, book, blob, s)
    if best_m is None or float(m.get("hand_max_disp_m") or 0) > float(best_m.get("hand_max_disp_m") or 0):
        best_m = m
        best_toks = s
dH = float(best_m["hand_max_disp_m"])
dest = OUT / "videos" / f"partial_LC_ubimp_0040_dH{dH:.3f}.mp4"
render_expert_video(
    env, blob, best_toks, table, book, dest,
    {"ic_id": "ubimp_0040", "hand_max_disp_m": dH, "SUCCESS_NEW": 0},
)
print("wrote", dest, "dH", dH)
