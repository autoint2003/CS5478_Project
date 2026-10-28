"""Plan A Stage-1 exit helper.

Physical control laws stay in ``inward_reseat`` / ``controlled_sliding`` /
``recovery_rebuild``. This module only:
  1. Replays a legal Stage-1 action prefix onto an IC capture.
  2. Captures the exact post-prefix state.
  3. Finalizes via the common Stage-1 exit contract (SUCCESS_12S + A1/A2).

``legal_verify`` / ``_verify_prefix`` already mean the unchanged 12-second
nominal SUCCESS_12S check starting after the recovered actions (HANDOFF from
the post-prefix state). We do not redefine that contract here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data_generation.oracles.stage1_exit import Stage1Exit, finalize_stage1_exit
from envs.outcomes.horizon_outcome import recovery_failed
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal


def replay_plan_a_actions_to_exit(
    env,
    blob: dict,
    actions: list | np.ndarray,
    *,
    plan_id: str = "A",
    source_ic_id: str | None = None,
    precision_config=None,
) -> Stage1Exit:
    """Restore IC → execute Stage-1 actions → capture → SUCCESS_12S finalize.

    Does not alter actions to force a pass. Does not run Precision.
    """
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    executed: list[np.ndarray] = []
    lost = 0
    for action in np.asarray(actions, np.float32):
        step_legal(env.sim, action)
        executed.append(np.asarray(action, np.float32).copy())
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if failed:
            cap = take_capture(env)
            return finalize_stage1_exit(
                env,
                plan_id=plan_id,
                source_ic_id=source_ic_id or blob.get("sample_id"),
                raw_ic=blob,
                stage1_actions=executed,
                stage1_exit_capture=cap,
                stage1_exit_ref=blob["ref"],
                stage1_duration=float(env.sim.data.time) - t0,
                precision_config=precision_config,
                allow_live_verify=True,
                extra={"abort": "DROP_DURING_RECOVERY"},
            )
    cap = take_capture(env)
    return finalize_stage1_exit(
        env,
        plan_id=plan_id,
        source_ic_id=source_ic_id or blob.get("sample_id"),
        raw_ic=blob,
        stage1_actions=executed,
        stage1_exit_capture=cap,
        stage1_exit_ref=blob["ref"],
        stage1_duration=float(env.sim.data.time) - t0,
        precision_config=precision_config,
        allow_live_verify=True,
        extra={"runner": "plan_a.stage1_runner.replay_plan_a_actions_to_exit"},
    )


def plan_a_verify_prefix_is_success_12s(outcome: dict[str, Any]) -> bool:
    """Map ``_verify_prefix`` / ``legal_verify`` success onto SUCCESS_12S semantics."""
    return bool(outcome.get("success")) and str(outcome.get("reason") or "") == "SUCCESS_12S"
