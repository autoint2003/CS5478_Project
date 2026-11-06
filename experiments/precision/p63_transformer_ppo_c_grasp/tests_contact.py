"""Contact geometry + C grasp reward event tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np

from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import (
    CGraspEventTracker,
    is_bad_contact,
    is_correct_sidewall,
)
from experiments.precision.p63_transformer_ppo_c_grasp.paths import DT, OUT, R_CORRECT_RECATCH, SUSTAIN_S


def _audit(**kw):
    base = {
        "grasp_kind": "no_pad_contact",
        "opposing_sidewall": 0,
        "bilateral": False,
        "n_end_face": 0,
    }
    base.update(kw)
    return base


def test_sidewall_not_mere_bilateral() -> dict:
    # Bilateral alone is NOT correct
    a = _audit(grasp_kind="sidewall_partial", opposing_sidewall=0, bilateral=True, n_end_face=0)
    b = _audit(
        grasp_kind="opposing_sidewall", opposing_sidewall=1, bilateral=True, n_end_face=0
    )
    c = _audit(
        grasp_kind="end_face_pinch", opposing_sidewall=0, bilateral=True, n_end_face=2
    )
    ok = (not is_correct_sidewall(a)) and is_correct_sidewall(b) and (not is_correct_sidewall(c))
    ok = ok and is_bad_contact(c) and (not is_bad_contact(b))
    return {
        "name": "sidewall_geometry_not_mere_bilateral",
        "ok": ok,
        "detail": f"partial={is_correct_sidewall(a)} oppose={is_correct_sidewall(b)} face={is_correct_sidewall(c)}",
    }


def test_events_once_only() -> dict:
    tr = CGraspEventTracker()
    sim = MagicMock()
    import experiments.precision.p63_transformer_ppo_c_grasp.contact_events as ce

    need = int(round(SUSTAIN_S / DT))

    def fake_classify(_sim):
        # Keep correct sidewall for entire sustain window so lost-after does not fire.
        return _audit(
            grasp_kind="opposing_sidewall",
            opposing_sidewall=1,
            bilateral=True,
            n_end_face=0,
        )

    old = ce.classify_contacts
    ce.classify_contacts = fake_classify
    try:
        d1, f1 = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=False, dropped=False)
        d2, f2 = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=False, dropped=False)
        sustain_hit = False
        for _ in range(need + 2):
            d_s, f_s = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=False, dropped=False)
            if f_s.get("event_sustain"):
                sustain_hit = True
                d_sustain = d_s
        d_r3, f_r3 = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=True, dropped=False)
        d_r3b, f_r3b = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=True, dropped=False)
        # Second recatch attempt must not re-award
        d_again, f_again = tr.step(sim, metrics={"vrel": 0.1}, r3f_early_ready=False, dropped=False)
    finally:
        ce.classify_contacts = old

    ok = (
        abs(d1 - R_CORRECT_RECATCH) < 1e-9
        and f1["event_recatch"] == 1
        and f2["event_recatch"] == 0
        and tr.awarded_recatch
        and tr.awarded_sustain
        and sustain_hit
        and abs(d_sustain - 2.0) < 1e-9
        and abs(d_r3 - 1.0) < 1e-9
        and f_r3["event_r3f"] == 1
        and f_r3b["event_r3f"] == 0
        and f_again["event_recatch"] == 0
        and abs(tr.pos_total - 5.0) < 1e-9
        and "lost_after_recatch" not in tr.log
    )
    return {
        "name": "reward_events_once_only",
        "ok": ok,
        "detail": f"pos={tr.pos_total} events={tr.log} d1={d1} d_r3={d_r3}",
    }


def test_success_flag_independent_of_reward(device: str = "cpu") -> dict:
    """C grasp reward must not mutate SUCCESS_RELAXED_12S computation."""
    from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
    from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
    from experiments.precision.p63_transformer_ppo_c_grasp.env_c_grasp import make_env
    from experiments.precision.p63_transformer_ppo_c_grasp.paths import P62_EPOCH3, SEED

    actor, book, rho, _ = load_actor(P62_EPOCH3, device)
    ds = build_p54_dataset(seed=SEED)
    # Prefer a C start if present
    starts = [s for s in ds["schedule"] if s.skill == "C"] or ds["schedule"]
    st = starts[0]
    env0 = make_env(book, rho, enable_c_grasp_reward=False)
    env1 = make_env(book, rho, enable_c_grasp_reward=True)
    env0.reset(st)
    env1.reset(st)
    # Take a few hold steps then stop — compare success flags on identical actions
    infos0, infos1 = [], []
    for _ in range(5):
        r0, d0, i0 = env0.step(0, np.zeros(7, np.float32))
        r1, d1, i1 = env1.step(0, np.zeros(7, np.float32))
        infos0.append(i0)
        infos1.append(i1)
        if d0 or d1:
            break
    # Success keys equal whenever both terminated same way; always compare last
    k = "SUCCESS_RELAXED_12S"
    ok = infos0[-1].get(k) == infos1[-1].get(k)
    ok = ok and infos0[-1].get("SUCCESS_12S") == infos1[-1].get("SUCCESS_12S")
    # When reward enabled on C, r may differ but success path identical for same physics
    return {
        "name": "reward_does_not_change_success_relaxed",
        "ok": ok,
        "detail": f"s0={infos0[-1].get(k)} s1={infos1[-1].get(k)} r_c={infos1[-1].get('r_c_grasp')}",
    }


def run_contact_tests(device: str = "cpu") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        test_sidewall_not_mere_bilateral(),
        test_events_once_only(),
        test_success_flag_independent_of_reward(device),
    ]
    all_ok = all(r["ok"] for r in rows)
    report = {"all_ok": all_ok, "tests": rows}
    (OUT / "CONTACT_GEOMETRY_TESTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# CONTACT_GEOMETRY_TESTS",
        "",
        f"**all_ok={all_ok}**",
        "",
        "| Test | OK | Detail |",
        "|------|:--:|--------|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['ok']} | {r.get('detail','')} |")
    (OUT / "CONTACT_GEOMETRY_TESTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_contact_tests(), indent=2))
