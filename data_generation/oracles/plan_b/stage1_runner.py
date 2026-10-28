"""Plan B Stage-1 exit helper (Recovery tokens → capture before HANDOFF).

Active-recatch chronology is enforced by ``branch_compare.detect_active_recatch``.
SUCCESS_12S is evaluated from the captured exit via the common Stage-1 contract,
not by embedding Precision.
"""

from __future__ import annotations

from typing import Any

from data_generation.oracles.plan_b.branch_compare import (
    decode_tok,
    detect_active_recatch,
    load_vocab,
    run_tokens,
)
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint
from data_generation.oracles.stage1_exit import Stage1Exit, finalize_stage1_exit
from envs.simulation.capture import restore_capture, take_capture
from evaluation.success_criteria import contact_summary


def _decode_actions(table, book48, tokens: list[int]) -> list:
    return [decode_tok(table, book48, int(t)) for t in tokens]


def _failed_exit(
    env,
    blob: dict,
    *,
    plan_id: str,
    source_ic_id: str | None,
    actions: list,
    cap,
    duration: float,
    reason: str,
    extra: dict,
) -> Stage1Exit:
    restore_capture(env, cap)
    cs = contact_summary(env.sim)
    fp = capture_fingerprint(cap)
    return Stage1Exit(
        plan_id=str(plan_id),
        source_ic_id=source_ic_id or blob.get("sample_id"),
        raw_ic=blob,
        stage1_actions=actions,
        stage1_exit_capture=cap,
        stage1_exit_ref=blob["ref"],
        capture_fingerprint=fp,
        stage1_success_12s=False,
        stage1_verification_evidence={"rejected_before_12s": True, "reason": reason},
        stage1_failure_reason=reason,
        stage1_action_count=len(actions),
        stage1_duration=duration,
        contact_summary_at_exit={
            "nL": int(cs.get("nL") or 0),
            "nR": int(cs.get("nR") or 0),
            "bilateral": bool(cs.get("bilateral")),
            "fnL": float(cs.get("fnL") or 0.0),
            "fnR": float(cs.get("fnR") or 0.0),
        },
        precision_required=None,
        precision_entry_errors=None,
        classification="STAGE1_FAILED",
        extra=extra,
    )


def replay_plan_b_tokens_to_exit(
    env,
    blob: dict,
    tokens: list[int],
    *,
    plan_id: str = "B",
    source_ic_id: str | None = None,
    require_active_recatch: bool = True,
    prefix_len: int | None = None,
    precision_config=None,
    table=None,
    book48=None,
) -> Stage1Exit:
    """Restore IC → Recovery tokens → capture (pre-HANDOFF) → SUCCESS_12S finalize."""
    if table is None or book48 is None:
        table, book48 = load_vocab()
    actions = _decode_actions(table, book48, tokens)
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    trace: list[dict] = []
    ok, fail = run_tokens(env, blob, table, book48, tokens, trace=trace)
    det = detect_active_recatch(trace)
    cap = take_capture(env)
    duration = float(env.sim.data.time) - t0
    extra: dict[str, Any] = {
        "runner": "plan_b.stage1_runner.replay_plan_b_tokens_to_exit",
        "ok_recovery": bool(ok),
        "fail_reason": fail,
        "recatch_detect": det,
        "prefix_len": prefix_len,
        "tokens": list(tokens),
    }
    if not ok:
        return _failed_exit(
            env,
            blob,
            plan_id=plan_id,
            source_ic_id=source_ic_id,
            actions=actions,
            cap=cap,
            duration=duration,
            reason=fail or "DROP_DURING_RECOVERY",
            extra=extra,
        )
    if require_active_recatch and not det.get("active_recatch"):
        return _failed_exit(
            env,
            blob,
            plan_id=plan_id,
            source_ic_id=source_ic_id,
            actions=actions,
            cap=cap,
            duration=duration,
            reason=str(det.get("reason") or "NO_ACTIVE_RECATCH"),
            extra=extra,
        )
    return finalize_stage1_exit(
        env,
        plan_id=plan_id,
        source_ic_id=source_ic_id or blob.get("sample_id"),
        raw_ic=blob,
        stage1_actions=actions,
        stage1_exit_capture=cap,
        stage1_exit_ref=blob["ref"],
        stage1_duration=duration,
        precision_config=precision_config,
        allow_live_verify=True,
        extra=extra,
    )
