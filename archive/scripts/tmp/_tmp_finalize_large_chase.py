"""Finalize large-chase deliverables: best partial videos + report polish."""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.k48_checkpoint import load_trained_k48
from training.action_vocab_k49 import build_k49_table, describe_k49_config
from training.recovery_episode import make_training_env
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, mapping_checksum, write_csv
from scripts.expand_k49_large_chase_dynamic import (
    OUT,
    HARD,
    load_blob,
    render_expert_video,
    replay_motion,
)

audit = pickle.loads((OUT / "RAW" / "PHASE_A_AUDIT.pkl").read_bytes())
thr = audit["thr_pack"]
device = torch.device("cpu")
model, book48 = load_trained_k48(SFT, device)
table = build_k49_table(book48, config_id=CONFIG_ID)
env = make_training_env(device, model=model)

# Best LARGE_CHASE partials by hand displacement among non-NEW rows
inter = list(csv.DictReader((OUT / "INTERCEPTION_SEARCH.csv").open(encoding="utf-8")))
lc = [r for r in inter if int(r.get("large_chase") or 0) == 1]
lc.sort(key=lambda r: -float(r.get("hand_max_disp_m") or 0))
# unique ICs
seen = set()
best = []
for r in lc:
    if r["ic_id"] in seen:
        continue
    seen.add(r["ic_id"])
    best.append(r)
    if len(best) >= 5:
        break
write_csv(OUT / "BEST_LARGE_CHASE_PARTIALS.csv", best)
print("best partials", [(b["ic_id"], b.get("hand_max_disp_m"), b.get("reason")) for b in best])

# Reconstruct tokens is not stored in INTERCEPTION rows — re-search top IC with best known chase seed
chase_seed = [17] * 16 + [30] * 3 + [0] * 6 + [12] * 8 + [48] * 20 + [0] * 4
vid_dir = OUT / "videos"
for b in best[:3]:
    snap = None
    for d in [OUT / "snapshots", HARD / "snapshots",
              ROOT / "results/expert_generation/k49_ball_impact_dynamic_recovery/snapshots",
              ROOT / "results/diagnostics/hard_ic_mining/snapshots"]:
        p = d / f"{b['ic_id']}.pkl"
        if p.exists():
            snap = p
            break
    if snap is None:
        print("no snap", b["ic_id"])
        continue
    blob = load_blob(snap)
    # try several chase seeds; pick largest disp
    seeds = [
        [17] * 18 + [30] * 3 + [0] * 8 + [12] * 8,
        [33] * 18 + [30] * 3 + [0] * 8 + [12] * 8,
        [1] * 20 + [30] * 3 + [0] * 8 + [12] * 8,
        [17] * 14 + [30] * 3 + [0] * 6 + [12] * 6 + [48] * 16,
    ]
    best_toks = seeds[0]
    best_m = None
    for s in seeds:
        m, _ = replay_motion(env, table, book48, blob, s)
        if best_m is None or float(m.get("hand_max_disp_m") or 0) > float(best_m.get("hand_max_disp_m") or 0):
            best_m = m
            best_toks = s
    overlay = {
        "ic_id": b["ic_id"],
        "hand_max_disp_m": best_m.get("hand_max_disp_m") if best_m else b.get("hand_max_disp_m"),
        "SUCCESS_NEW": 0,
    }
    dest = vid_dir / f"partial_LC_{b['ic_id']}_dH{float(overlay['hand_max_disp_m']):.3f}.mp4"
    render_expert_video(env, blob, best_toks, table, book48, dest, overlay)
    print("wrote", dest)

# Ensure UPDATED_UNIFIED_MANIFEST exists (copy of baseline if no new)
if not (OUT / "UPDATED_UNIFIED_MANIFEST.csv").exists() or (OUT / "UPDATED_UNIFIED_MANIFEST.csv").stat().st_size < 10:
    import shutil
    shutil.copy2(HARD / "UNIFIED_MANIFEST.csv", OUT / "UPDATED_UNIFIED_MANIFEST.csv")

# Empty FULL_EXPERTS header if empty
if (OUT / "FULL_EXPERTS.csv").stat().st_size < 5:
    (OUT / "FULL_EXPERTS.csv").write_text(
        "ic_id,root_episode,tag,expert_type,SUCCESS_NEW,large_chase,high_motion,complete_contact_loss\n",
        encoding="utf-8",
    )

# Annotate existing LARGE_CHASE success identity
succ = list(csv.DictReader((OUT / "EXISTING_SUCCESS_MOTION_AUDIT.csv").open(encoding="utf-8")))
lc_exist = [r for r in succ if int(r.get("large_chase") or 0) == 1]
print("existing LC", [(r["ic_id"], r["hand_max_disp_m"], r["disp_to_recatch_m"], r.get("source_dataset")) for r in lc_exist])

# Append limitation note to REPORT
note = f"""

## Physical / search limitations (no LARGE_CHASE FULL found)

- Prior 7 dynamic SUCCESS_NEW are **all local corrections** (mean hand max_disp 1.54 cm; max 3.3 cm).
- Only **{len(lc_exist)}/{len(succ)}** existing SUCCESS_NEW meet LARGE_CHASE (`{[r['ic_id'] for r in lc_exist]}`).
- Among 42 HIGH_MOTION ICs: **27 class A** (no plausible interception within horizon); **15 class B–E**.
- Phase B used **3845** rollouts across **34** ICs (B/C/D preferred). Sustained translation produced
  **{len(lc)}** LARGE_CHASE *partial* rows (hand travel ≥ thr) but **0** SUCCESS_NEW — dominant terminal: TABLE_DROP.
- Hand can pursue (measured displacements up to ~{max(float(r.get('hand_max_disp_m') or 0) for r in lc):.3f} m on partials) but
  legal K49 capture/brake/stabilization after large pursuit did not achieve SUCCESS_NEW in budget.
- Short-prefix transfer of the 7 prior experts onto HM ICs also failed (0 NEW).

Neither HIGH_MOTION nor LARGE_CHASE diagnostics change SUCCESS_NEW.
"""
rep = (OUT / "REPORT.md").read_text(encoding="utf-8")
if "Physical / search limitations" not in rep:
    (OUT / "REPORT.md").write_text(rep + note, encoding="utf-8")
print("done")
