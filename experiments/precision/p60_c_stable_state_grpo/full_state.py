"""Full StabilizeGrpoEnv continuation state: MuJoCo snapshot + R3F + policy windows + recatch."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from envs.simulation.capture import restore_snapshot, take_snapshot
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import (
    export_r3f,
    import_r3f,
)
from experiments.precision.p57_post_recatch_stabilization.criteria import RecatchTracker
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.state_labels import (  # noqa: F401
    classify_capture_and_future,
    classify_failure_pattern,
)


def export_recatch(tr: RecatchTracker) -> dict[str, Any]:
    return {
        "bil_run": int(tr.bil_run),
        "recatch_step": tr.recatch_step,
        "post_recatch": bool(tr.post_recatch),
        "stable_hold": bool(tr.stable_hold),
        "lost_after_recatch": bool(tr.lost_after_recatch),
        "hold_run": int(tr.hold_run),
        "hold_vrels": [float(x) for x in tr.hold_vrels],
        "steps_since_recatch": int(tr.steps_since_recatch),
    }


def import_recatch(blob: dict[str, Any]) -> RecatchTracker:
    tr = RecatchTracker()
    tr.bil_run = int(blob["bil_run"])
    tr.recatch_step = blob["recatch_step"]
    tr.post_recatch = bool(blob["post_recatch"])
    tr.stable_hold = bool(blob["stable_hold"])
    tr.lost_after_recatch = bool(blob["lost_after_recatch"])
    tr.hold_run = int(blob["hold_run"])
    tr.hold_vrels = [float(x) for x in blob.get("hold_vrels") or []]
    tr.steps_since_recatch = int(blob["steps_since_recatch"])
    return tr


def _start_blob(start: MixedStart) -> dict[str, Any]:
    return {
        "skill": start.skill,
        "traj_id": start.traj_id,
        "family": start.family,
        "kind": start.kind,
        # ep/capture/ref kept live on the MixedStart object; restored via caller binding
    }


def take_full_state(
    env: StabilizeGrpoEnv,
    *,
    failure_pattern: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture complete continuation state at the current policy step (pre-next-action)."""
    assert env.start is not None and env.mon is not None
    m = state_metrics(env.env.sim)
    return {
        "version": 1,
        "start_meta": _start_blob(env.start),
        "policy_step": int(env.t),
        "max_steps": int(env.max_steps),
        "t_max": int(env.t_max),
        "remaining_steps": int(env.max_steps) - int(env.t),
        "prev_tok": None if env.prev_tok is None else int(env.prev_tok),
        "snap": take_snapshot(env.env),
        "r3f": export_r3f(env.mon),
        "recatch": export_recatch(env.recatch),
        "diag": dict(env.diag),
        "obs_seq": [np.asarray(x, np.float32).copy() for x in env.obs_seq],
        "prev_seq": [np.asarray(x, np.float32).copy() for x in env.prev_seq],
        "_prev_m": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer, bool, np.bool_)) else v) for k, v in (env._prev_m or m).items() if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_))},
        "comp_sums": dict(env.comp_sums),
        "stab_sums": dict(env.stab_sums),
        "metrics": {
            k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer, bool, np.bool_)) else v)
            for k, v in m.items()
            if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_))
        },
        "sim_time": float(env.env.sim.data.time),
        "failure_pattern": failure_pattern,
        "meta": meta or {},
    }


def restore_full_state(env: StabilizeGrpoEnv, state: dict[str, Any], start: MixedStart) -> None:
    """Restore full continuation state. Preserves remaining horizon (env.t / max_steps / t_max)."""
    restore_snapshot(env.env, state["snap"])
    env.start = start
    env.env.ref = start.ref if start.kind == "capture" else (start.ep or {}).get("ref")
    env.t = int(state["policy_step"])
    env.max_steps = int(state["max_steps"])
    env.t_max = int(state["t_max"])
    env.prev_tok = state["prev_tok"]
    env.mon = import_r3f(copy.deepcopy(state["r3f"]))
    env.recatch = import_recatch(copy.deepcopy(state["recatch"]))
    env.diag = dict(state["diag"])
    env.obs_seq = [np.asarray(x, np.float32).copy() for x in state["obs_seq"]]
    env.prev_seq = [np.asarray(x, np.float32).copy() for x in state["prev_seq"]]
    env._prev_m = dict(state["_prev_m"])
    env.comp_sums = {k: float(v) for k, v in (state.get("comp_sums") or {}).items()}
    if not env.comp_sums:
        env.comp_sums = {
            k: 0.0 for k in ("pose", "contact", "slip", "imbalance", "time", "switch", "terminal")
        }
    env.stab_sums = {k: float(v) for k, v in (state.get("stab_sums") or {}).items()}
    if not env.stab_sums:
        env.stab_sums = {
            k: 0.0 for k in ("hold", "calm", "post_slip", "lost_contact", "overgrip", "stab_total")
        }
    env.last_info = {}

