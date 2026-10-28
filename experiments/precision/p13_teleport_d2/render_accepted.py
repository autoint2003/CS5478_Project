"""Render video(s) for STRICT-accepted P13 D2 ICs."""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.live_stage1 import extract_precision_skills  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from experiments.precision.p13_teleport_d2.run_p13 import (  # noqa: E402
    _norm_transition,
    make_env,
    render_accepted_video,
)

OUT = ROOT / "results" / "diagnostics" / "p13_teleport_d2"


def main() -> None:
    path = OUT / "STRICT_ACCEPTED_D2.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print("no accepts", flush=True)
        return
    env = make_env()
    cfg = PrecisionConfig(use_verified_lateral_fallback=False, use_explicit_fsm=True)
    for row in rows:
        tag = row["tag"]
        ic = pickle.loads((OUT / "ics" / f"{tag}.pkl").read_bytes())
        res = run_precision(
            env,
            {"capture": ic["capture"], "ref": ic["ref"], "sample_id": ic["sample_id"]},
            "D",
            cfg,
        )
        sk = extract_precision_skills(list(res.phases or []))
        trans = _norm_transition(sk)
        label = row.get("video_label") or "DZ-only"
        safe = label.replace("→", "_to_").replace(" ", "_")
        vpath = OUT / "videos" / f"{safe}.mp4"
        rel = render_accepted_video(env, ic, res.actions or [], vpath, f"{label} | {trans}")
        row["video_path"] = rel
        row["video_label"] = label
        print(tag, res.status, trans, rel, flush=True)
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
