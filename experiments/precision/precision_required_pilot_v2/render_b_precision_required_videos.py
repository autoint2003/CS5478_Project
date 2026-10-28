#!/usr/bin/env python
"""Render Precision Oracle videos for the 7 B Precision-required pilot ICs.

Reconstructs Stage-1 exit (hard IC + known B1 tokens) then runs/replays Oracle.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.stage1_runner import replay_plan_b_tokens_to_exit  # noqa: E402
from data_generation.oracles.plan_b.branch_compare import load_vocab  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.state_machine import run_precision  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from experiments.precision.render_precision_failure_videos import (  # noqa: E402
    hold_frames,
    hud,
    render_pair,
    replay_actions_video,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

PILOT = ROOT / "results/data_generation/precision_required_pilot_v2"
OUT = PILOT / "VIDEOS" / "B_precision_required"
HARD = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
B1_OUT = ROOT / "results/data_generation/stage2_abc_mining/v2/outcomes"
FPS = 20


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def cfg() -> PrecisionConfig:
    return PrecisionConfig(use_verified_lateral_fallback=False)


def load_seven() -> list[dict]:
    fun = list(csv.DictReader((PILOT / "FULL_FUNNEL.csv").open(encoding="utf-8")))
    rows = [
        r
        for r in fun
        if r.get("branch") == "B"
        and str(r.get("stage1_12s")) in ("1", "True", "true")
        and str(r.get("precision_required")) in ("1", "True", "true")
    ]
    # de-dup by raw_id (prefer supplement known_b1)
    by = {}
    for r in rows:
        by[r["raw_id"]] = r
    return list(by.values())


def hard_id_from_raw(raw_id: str) -> str:
    # B_RAW__hard_910026__known_b1
    for part in raw_id.split("__"):
        if part.startswith("hard_"):
            return part
    raise KeyError(raw_id)


def b1_tokens(hard_id: str) -> list[int]:
    hits = sorted(B1_OUT.glob(f"B__{hard_id}__post_b1_v2__*.json"))
    if not hits:
        raise FileNotFoundError(hard_id)
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    toks = list(d.get("b1_tokens") or [])
    if not toks:
        raise ValueError(f"no b1_tokens for {hard_id}")
    return toks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    table, book48 = load_vocab()
    cam = mjv_camera_from_preset(
        {"lookat": [0, 0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    renderer = mujoco.Renderer(env.sim.model, 420, 560)
    rows = load_seven()
    print(f"rendering {len(rows)} B Precision-required Oracle videos → {OUT}", flush=True)
    index = []

    for i, row in enumerate(rows, 1):
        raw_id = row["raw_id"]
        hid = hard_id_from_raw(raw_id)
        snap_path = HARD / f"{hid}.pkl"
        print(f"[{i}/{len(rows)}] {raw_id}", flush=True)
        if not snap_path.is_file():
            index.append({"raw_id": raw_id, "error": f"missing {snap_path}"})
            continue
        blob = pickle.loads(snap_path.read_bytes())
        toks = b1_tokens(hid)
        pref = min(3, len(toks))
        exit_obj = replay_plan_b_tokens_to_exit(
            env,
            {"capture": blob["capture"], "ref": blob["ref"], "sample_id": raw_id},
            toks,
            plan_id="B",
            source_ic_id=raw_id,
            require_active_recatch=True,
            prefix_len=pref,
            precision_config=cfg(),
            table=table,
            book48=book48,
        )
        if not exit_obj.stage1_success_12s or not exit_obj.precision_required:
            index.append(
                {
                    "raw_id": raw_id,
                    "error": "stage1_exit_not_precision_required_on_rerun",
                    "classification": exit_obj.classification,
                    "stage1_12s": exit_obj.stage1_success_12s,
                    "precision_required": exit_obj.precision_required,
                    "reason": exit_obj.stage1_failure_reason,
                }
            )
            print(f"  skip: {exit_obj.classification} {exit_obj.stage1_failure_reason}", flush=True)
            continue

        exit_blob = {"capture": exit_obj.stage1_exit_capture, "ref": exit_obj.stage1_exit_ref}
        fp = capture_fingerprint(exit_blob["capture"])
        restore_capture(env, exit_blob["capture"])
        res = run_precision(env, {"capture": exit_blob["capture"], "ref": exit_blob["ref"], "sample_id": raw_id}, "B", cfg())
        actions = np.asarray(res.actions or [], np.float32)
        skills = "|".join(
            p
            for p in (res.phases or [])
            if any(k in str(p).lower() for k in ("dtheta", "dx", "dz", "support"))
        )
        fail = None
        if res.status != "PRECISION_EXECUTED_SUCCESS":
            fails = list(res.failures or [])
            fail = fails[0] if fails else res.status

        dest = OUT / f"{hid}__{res.status}.mp4"
        restore_capture(env, exit_blob["capture"])
        frames = hold_frames(env, exit_blob, cam, renderer, hid, hold_s=0.8)
        if len(actions) == 0:
            frames.append(
                render_pair(
                    env,
                    cam,
                    renderer,
                    hud(env, ic_id=hid, phase=str(res.status), t=0.0, fail=str(fail)),
                )
            )
            for _ in range(FPS):
                frames.append(frames[-1])
        else:
            frames.extend(
                replay_actions_video(
                    env,
                    exit_blob,
                    actions,
                    cam,
                    renderer,
                    hid,
                    phase_prefix=f"ORACLE|{res.status}",
                    fail_reason=str(fail) if fail else None,
                    max_frames=500,
                )
            )
        # title card outcome
        restore_capture(env, exit_blob["capture"])
        outcome_lines = hud(env, ic_id=hid, phase="SUMMARY", t=0.0, fail=fail)
        outcome_lines.insert(
            1,
            f"status={res.status}  n_act={len(actions)}  skills={skills[:80]}",
        )
        outcome_lines.insert(2, f"pilot_ok={row.get('ok')}  funnel_status={row.get('precision_status')}")
        for _ in range(FPS * 2):
            frames.append(render_pair(env, cam, renderer, outcome_lines))

        path = write_frames_mp4(frames, dest, fps=FPS)
        meta = {
            "raw_id": raw_id,
            "hard_id": hid,
            "capture_fp": fp,
            "b1_tokens": toks,
            "prefix_len": pref,
            "oracle_status": res.status,
            "n_actions": int(len(actions)),
            "phases": list(res.phases or []),
            "failures": list(res.failures or []),
            "pilot_funnel_ok": row.get("ok"),
            "pilot_funnel_status": row.get("precision_status"),
            "pilot_skills": row.get("skills"),
            "video": str(Path(path).relative_to(ROOT)).replace("\\", "/") if path else None,
            "entry_errors": exit_obj.precision_entry_errors,
        }
        (OUT / f"{hid}__meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        # save actions for replay
        np.savez_compressed(OUT / f"{hid}__actions.npz", actions=actions)
        index.append(meta)
        print(f"  → {dest.name} status={res.status} n_act={len(actions)}", flush=True)

    index_path = OUT / "INDEX.json"
    index_path.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n": len(index),
                "videos": index,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    # markdown index
    lines = [
        "# B Precision-required Oracle videos",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| hard_id | status | n_actions | video |",
        "|---|---|---:|---|",
    ]
    for m in index:
        if m.get("error"):
            lines.append(f"| {m.get('raw_id')} | ERROR | | {m.get('error')} |")
        else:
            lines.append(
                f"| {m.get('hard_id')} | `{m.get('oracle_status')}` | {m.get('n_actions')} | `{m.get('video')}` |"
            )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    renderer.close()
    print(f"done → {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
