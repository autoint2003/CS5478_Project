"""Plan B active-recatch chronology: release → contact-free → bilateral before HANDOFF."""

from __future__ import annotations

from data_generation.oracles.plan_b.branch_compare import (
    BILATERAL_PERSIST_STEPS,
    classify_recatch_controller,
    detect_active_recatch,
)


def _row(*, nL: int, nR: int, bilateral: bool | None = None, t: float = 0.0, tok=None, phase="post"):
    bil = bool(nL > 0 and nR > 0) if bilateral is None else bool(bilateral)
    return {"nL": nL, "nR": nR, "bilateral": bil, "t": t, "tok": tok, "phase": phase,
            "obj_z": 0.5, "e_pos_mm": 20.0, "e_theta_deg": 15.0, "ground_drop": False}


def _bil_run(n: int, t0: float = 0.0):
    return [_row(nL=1, nR=1, t=t0 + 0.02 * i) for i in range(n)]


def _off_run(n: int, t0: float = 0.0):
    return [_row(nL=0, nR=0, t=t0 + 0.02 * i) for i in range(n)]


def test_valid_release_then_reacquisition():
    trace = _bil_run(2) + _off_run(2) + _bil_run(BILATERAL_PERSIST_STEPS)
    det = detect_active_recatch(trace)
    assert det["genuine_release"] is True
    assert det["active_recatch"] is True
    assert det["off_i"] == 2
    assert det["bil_i"] == 4
    assert classify_recatch_controller(trace, prefix_len=2, post_handoff_bilateral_t=None) == "recovery"


def test_release_without_reacquisition_invalid():
    trace = _bil_run(2) + _off_run(3)
    det = detect_active_recatch(trace)
    assert det["genuine_release"] is True
    assert det["active_recatch"] is False
    assert det["reason"] == "RELEASE_WITHOUT_REACQUISITION"


def test_bilateral_before_release_only_invalid():
    # Bilateral only before release; after release never reacquires
    trace = _bil_run(4) + _off_run(2)
    det = detect_active_recatch(trace)
    assert det["active_recatch"] is False
    # Old bug would have found bil_i early independently of off_i
    assert det["bil_i"] is None


def test_reacquisition_only_credited_after_contact_free():
    # Bilateral samples exist before release; must not count as active recatch alone
    trace = _bil_run(5) + _off_run(1) + _bil_run(BILATERAL_PERSIST_STEPS)
    det = detect_active_recatch(trace)
    assert det["active_recatch"] is True
    assert det["bil_i"] is not None and det["bil_i"] > det["contact_free_end_i"]


def test_no_full_release_not_active_recatch():
    trace = _bil_run(6)  # retained bilateral
    det = detect_active_recatch(trace)
    assert det["genuine_release"] is False
    assert det["active_recatch"] is False
    assert det["retained_bilateral"] is True
    assert classify_recatch_controller(trace, 0, None) == "retained_or_partial"


def test_post_handoff_only_is_nominal_not_recovery():
    # Recovery trace: release without reacquisition; bil only "after handoff"
    trace = _bil_run(1) + _off_run(2)
    det = detect_active_recatch(trace)
    assert det["active_recatch"] is False
    assert classify_recatch_controller(trace, 1, post_handoff_bilateral_t=1.0) == "nominal"


def test_reacquisition_on_final_recovery_sample():
    # Post-action sampling: last samples are the reacquisition (final Recovery actions)
    trace = (
        [_row(nL=1, nR=1, t=0.0, tok=None, phase="pre")]
        + _off_run(2, t0=0.02)
        + [_row(nL=1, nR=1, t=0.06 + 0.02 * i, tok=0, phase="post") for i in range(BILATERAL_PERSIST_STEPS)]
    )
    det = detect_active_recatch(trace)
    assert det["active_recatch"] is True
    assert det["bil_i"] == len(trace) - BILATERAL_PERSIST_STEPS


def test_transient_bilateral_shorter_than_persist_invalid():
    persist = BILATERAL_PERSIST_STEPS
    trace = _bil_run(1) + _off_run(2) + _bil_run(persist - 1)
    det = detect_active_recatch(trace)
    assert det["genuine_release"] is True
    assert det["active_recatch"] is False
    assert det["reason"] == "RELEASE_WITHOUT_REACQUISITION"


def test_partial_contact_not_full_release():
    # One finger still contacting — not both-off
    trace = [_row(nL=1, nR=0), _row(nL=0, nR=1)] + _bil_run(3)
    det = detect_active_recatch(trace)
    assert det["genuine_release"] is False
    assert det["active_recatch"] is False
