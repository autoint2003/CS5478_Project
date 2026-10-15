"""Smoke test for GRPO progress reporting. It does not roll out or train."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.unified_k48 import (
    epoch_progress,
    format_seconds,
    open_grpo_progress,
    progress_postfix,
    progress_timing,
    summarize_group,
    write_progress_row,
)


def _member(label, reward, handoff):
    return {
        "label": label,
        "return": reward,
        "truncated": False,
        "reward_log": {"handoff": handoff, "return": reward},
    }


def test_timing_and_epochs():
    recent = deque(maxlen=20)
    now = datetime(2026, 10, 13, 3, 0, 0)
    last = None
    for step, duration in enumerate(range(1, 26), start=1):
        elapsed = float(step * 2 + (5.0 if step % 312 == 0 else 0.0))
        last = progress_timing(step, 936, 312, 3, float(duration), recent, elapsed, now)
    assert len(recent) == 20
    assert abs(last["avg_update_s"] - 15.5) < 1e-9
    assert abs(last["eta_s"] - 15.5 * (936 - 25)) < 1e-9
    assert last["eta_local"] == (now + timedelta(seconds=last["eta_s"])).strftime("%Y-%m-%d %H:%M:%S")
    assert last["update_s"] == 25.0
    boundaries = {
        312: "1/3",
        313: "2/3",
        624: "2/3",
        625: "3/3",
        936: "3/3",
    }
    for update, label in boundaries.items():
        assert epoch_progress(update, 312, 3)[2] == label
    saved = progress_timing(312, 936, 312, 3, 2.0, deque([2.0], maxlen=20), 7.0, now)
    assert saved["update_s"] == 2.0
    assert saved["elapsed_s"] == 7.0
    assert saved["epoch_label"] == "1/3"
    final = progress_timing(936, 936, 312, 3, 2.0, deque([2.0], maxlen=20), 100.0, now)
    assert final["percent"] == 100.0
    assert final["eta_s"] == 0.0
    assert final["eta_local"] == now.strftime("%Y-%m-%d %H:%M:%S")
    assert format_seconds(3661) == "1:01:01"
    return {"avg20": last["avg_update_s"], "eta_s": last["eta_s"], "final_eta": final["eta_s"]}


def test_log_and_bar():
    members = [
        _member("SUCCESS_12S", 1.8, "early"),
        _member("SUCCESS_12S", 1.9, "forced"),
        _member("SUCCESS_12S", 1.7, "early"),
        _member("DROP_DURING_RECOVERY", -0.99, None),
    ]
    summary = summarize_group(members, [0.5, 0.2, 0.1, -0.8], False, 0.01, 0.2)
    assert summary["success"] == 3 and summary["group"] == 4
    assert summary["handoff"]["early"] == 2 and summary["handoff"]["none"] == 1
    rows = []
    for version in ("v1", "v2"):
        timing = progress_timing(936, 936, 312, 3, 1.5, deque(maxlen=20), 120.0, datetime(2026, 10, 13, 4, 0, 0))
        rows.append({
            "update": 936,
            "episode_id": "ball_train_v2_0001",
            "reward_version": version,
            "checkpoint": "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_%s/checkpoint_u0936.pt" % version,
            **summary,
            **timing,
        })
    assert set(rows[0]) == set(rows[1])
    shared = set(rows[0]) - {"reward_version", "checkpoint"}
    assert all(rows[0][key] == rows[1][key] for key in shared)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "training_log.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                write_progress_row(handle, row)
        loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["reward_version"] == "v1" and loaded[1]["reward_version"] == "v2"
    assert loaded[0]["percent"] == 100.0 and loaded[0]["eta_s"] == 0.0
    buffer = io.StringIO()
    bar = open_grpo_progress(2, file=buffer)
    try:
        from tqdm import tqdm
        for update in (1, 2):
            timing = progress_timing(update, 2, 312, 1, 1.25, deque(maxlen=20), 1.25 * update, datetime(2026, 10, 13, 4, 0, 0))
            tqdm.write("GRPO diagnostic", file=buffer)
            bar.set_postfix_str(progress_postfix(timing, 3, 4), refresh=False)
            bar.update(1)
        tqdm.write("GRPO_DONE 2/2 elapsed=0:00:03 checkpoint=checkpoint_u0002.pt", file=buffer)
    finally:
        bar.close()
    text = buffer.getvalue()
    assert "2/2" in text and "100" in text
    assert "epoch=1/1" in text and "success=3/4" in text
    assert "GRPO diagnostic" in text and "checkpoint_u0002.pt" in text
    return {"logged": len(loaded), "bar_has_100": "100" in text}


def main():
    timing = test_timing_and_epochs()
    logged = test_log_and_bar()
    print({"timing": timing, "log": logged})


if __name__ == "__main__":
    main()
