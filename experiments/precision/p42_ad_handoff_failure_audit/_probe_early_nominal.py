"""One-shot probe: why early R2 HANDOFF loses grasp under nominal (diagnosis only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from controllers.recovery.geometry_switch import rot_angle  # noqa: E402
from envs.ballistic_impact import z_tgt_of  # noqa: E402
from envs.outcomes.horizon_outcome import LOST_STEPS  # noqa: E402
from envs.outcomes.nominal_pose import hand_R  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p39_working_hold_handoff.rules import DwellTracker  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402


def probe(env, cap, ref, tag: str, *, stabilize_n: int = 8) -> None:
    from data_generation.oracles.precision.corrections.contact_primitives import _wrist
    from envs.step_legal import step_legal as _step
    from envs.outcomes.support_drop import latched

    restore_capture(env, cap)
    for _ in range(int(stabilize_n)):
        _step(env.sim, _wrist(0.0, 0.0, 1.0))
        if latched(env.sim):
            break

    m = measure(env.sim, U_TGT, P_TGT)
    pb = pad_bilateral(env.sim)
    hand = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    Rd = hand_R(env.sim)
    Rt = np.array(ref["r_des"], float).reshape(3, 3)
    ang = float(np.degrees(rot_angle(Rt, Rd)))
    zt = float(z_tgt_of(env.sim))
    obj_z = float(m["obj_z"])
    hand_z = float(hand[2])
    info = apply_handoff(env.sim, ref)

    hist = []
    lost = 0
    reason = None
    t0 = float(env.sim.data.time)
    # Full 12 s nominal path (same abort as evaluator)
    while float(env.sim.data.time) - t0 < 12.0 - 1e-9:
        nominal_step(env.sim)
        pb2 = pad_bilateral(env.sim)
        m2 = measure(env.sim, U_TGT, P_TGT)
        bil = bool(pb2["bilateral"])
        lost = lost + 1 if not bil else 0
        i = len(hist)
        row = {
            "i": i,
            "t": float(env.sim.data.time) - t0,
            "bil": int(bil),
            "nL": int(pb2["nL"]),
            "nR": int(pb2["nR"]),
            "theta": float(m2["e_theta_deg"]),
            "epos": float(m2["e_pos_mm"]),
            "obj_z": float(m2["obj_z"]),
            "pdes_z": float(env.sim.fsm.p_des[2]),
            "lost": lost,
        }
        hist.append(row)
        if latched(env.sim):
            reason = "GROUND_DROP"
            break
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break

    first_loss = next((h["i"] for h in hist if not h["bil"]), None)
    min_n = min((h["nL"] + h["nR"] for h in hist), default=0)
    print(f"--- {tag} ---")
    print(
        f"  pre: theta={m['e_theta_deg']:.2f} e_pos={m['e_pos_mm']:.2f} "
        f"bil={pb['bilateral']} nL={pb['nL']} nR={pb['nR']} "
        f"obj_z={obj_z:.3f} hand_z={hand_z:.3f} z_tgt={zt:.3f} "
        f"lift_gap={zt - hand_z:.3f} r_err_deg={ang:.2f} grip={info['grip_ctrl']}"
    )
    print(
        f"  outcome={reason or 'RAN_12S'} first_bilateral_loss_step={first_loss} "
        f"n_steps={len(hist)} min_nL+nR={min_n} final_t={hist[-1]['t'] if hist else None}"
    )
    if first_loss is not None:
        h = hist[first_loss]
        print(
            f"  at_loss: t={h['t']:.3f} nL={h['nL']} nR={h['nR']} theta={h['theta']:.2f} "
            f"e_pos={h['epos']:.2f} obj_z={h['obj_z']:.3f} pdes_z={h['pdes_z']:.3f}"
        )
        # contact just before loss
        if first_loss > 0:
            b = hist[first_loss - 1]
            print(
                f"  pre_loss: t={b['t']:.3f} nL={b['nL']} nR={b['nR']} theta={b['theta']:.2f} "
                f"e_pos={b['epos']:.2f} obj_z={b['obj_z']:.3f} pdes_z={b['pdes_z']:.3f}"
            )
    # samples at ~0, mid-lift, near end or fail
    idxs = [0]
    if len(hist) > 1:
        idxs += [len(hist) // 4, len(hist) // 2, len(hist) - 1]
    for i in idxs:
        h = hist[i]
        print(
            f"  nom[{h['i']}]: t={h['t']:.2f} bil={h['bil']} nL={h['nL']} nR={h['nR']} "
            f"th={h['theta']:.1f} ep={h['epos']:.1f} oz={h['obj_z']:.3f} pz={h['pdes_z']:.3f}"
        )


def main() -> None:
    print("LOST_STEPS", LOST_STEPS)
    book, rho, _ = load_m111_book(M111_META)
    want = "plan_a__ball_train_0017"
    job = next(
        j
        for j in collect_jobs()
        if j["skill"] == "A1" and (j["traj_id"] == want or want in j["traj_id"])
    )
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    ep = job["ep"]
    ref = ep["ref"]
    restore_start(env, ep)
    toks = np.asarray(job["tokens"], np.int64)
    res = np.asarray(job["residuals"], np.float32)
    n = len(toks) - (1 if len(toks) and int(toks[-1]) == HANDOFF else 0)
    tr = DwellTracker.from_rule("R2", dt=float(DT_POLICY))
    r2_cap = None
    r2_t = None
    for t in range(n + 1):
        st = tr.observe(env.sim, t)
        if r2_t is None and st["ready"]:
            r2_t = t
            r2_cap = take_capture(env)
        if t >= n:
            break
        tok = int(toks[t])
        if tok == HANDOFF:
            continue
        rr = res[t] if t < len(res) else np.zeros(7, np.float32)
        step_legal(env.sim, decode_m111_action(book, tok, rr, rho=rho))
    term_cap = take_capture(env)
    print("traj", job["traj_id"], "r2_step", r2_t, "term_n", n)
    probe(env, r2_cap, ref, "R2")
    probe(env, term_cap, ref, "TERM")


if __name__ == "__main__":
    main()
