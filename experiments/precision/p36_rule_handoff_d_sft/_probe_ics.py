"""One-shot IC geometry probe for P36."""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p36_rule_handoff_d_sft.geometry import (  # noqa: E402
    geometric_target_reached,
)

rows = [
    r
    for r in csv.DictReader(
        (ROOT / "results/diagnostics/p35_precision_sft/PACKAGED_MANIFEST.csv").open(encoding="utf-8")
    )
    if r["class"] == "P1"
]
env = make_env()
print("n_p1", len(rows))
oks = []
for i, r in enumerate(rows):
    blob = pickle.loads(Path(r["ic_path"]).read_bytes())
    restore_capture(env, blob["capture"])
    ok, g = geometric_target_reached(env.sim)
    oks.append(ok)
    if i < 15 or (not ok and sum(1 for x in oks if not x) <= 15):
        h = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)
        print(
            f"{r['traj_id'][:52]:52} split={r['split']:5} ok={int(ok)} "
            f"th={g['dtheta_deg']:.2f} dx={g['dx_mm']:.2f} dz={g['dz_mm']:.2f} "
            f"NEW={int(bool(h.get('SUCCESS_NEW')))} 12S={int(bool(h.get('SUCCESS_12S')))} "
            f"reason={h.get('reason')}"
        )
print("init_sat", sum(oks), "/", len(oks), "nontrivial", len(oks) - sum(oks))
by_split = {}
for r, ok in zip(rows, oks):
    by_split.setdefault(r["split"], [0, 0])
    by_split[r["split"]][0] += 1
    by_split[r["split"]][1] += int(ok)
print("by_split total/init_sat", by_split)
