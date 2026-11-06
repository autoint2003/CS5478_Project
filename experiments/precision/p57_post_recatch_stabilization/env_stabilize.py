"""GRPO env with optional post-recatch stabilization reward (P57-B)."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics
from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart
from experiments.precision.p53_independent_grpo64.env_grpo import GrpoRecoveryEnv
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p53_independent_grpo64.success_relaxed import evaluate_success_relaxed_12s
from experiments.precision.p57_post_recatch_stabilization.criteria import RecatchTracker
from experiments.precision.p57_post_recatch_stabilization.reward_stabilize import (
    StabilizationWeights,
    combined_step_reward,
    terminal_reward,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action
from envs.step_legal import step_legal


class StabilizeGrpoEnv(GrpoRecoveryEnv):
    """Same as GrpoRecoveryEnv; if use_stabilization, add post-recatch shaping."""

    def __init__(
        self,
        book,
        rho,
        *,
        max_steps: int = 80,
        t_max: int = 3100,
        weights: PoseRewardWeights | None = None,
        stab_weights: StabilizationWeights | None = None,
        use_stabilization: bool = False,
    ):
        super().__init__(book, rho, max_steps=max_steps, t_max=t_max, weights=weights)
        self.stab_weights = stab_weights or StabilizationWeights()
        self.use_stabilization = bool(use_stabilization)
        self.recatch = RecatchTracker()
        self.stab_sums = {
            k: 0.0 for k in ("hold", "calm", "post_slip", "lost_contact", "overgrip", "stab_total")
        }
        self.diag: dict[str, Any] = {}

    def reset(self, start: MixedStart) -> np.ndarray:
        obs = super().reset(start)
        self.recatch = RecatchTracker()
        self.stab_sums = {k: 0.0 for k in self.stab_sums}
        self.diag = {
            "recatch": 0,
            "recatch_step": None,
            "stable_hold": 0,
            "lost_after_recatch": 0,
            "post_recatch_steps": 0,
        }
        return obs

    def step(self, token: int, resid: np.ndarray) -> tuple[float, bool, dict]:
        assert self.start is not None and self.mon is not None and self._prev_m is not None
        m0 = self._prev_m
        tok = int(token)
        if tok == HANDOFF:
            tok = 0
            resid = np.zeros(7, np.float32)
        switched = self.prev_tok is not None and tok != int(self.prev_tok)
        self.prev_tok = tok

        action = decode_m111_action(self.book, tok, resid, rho=self.rho)
        step_legal(self.env.sim, np.asarray(action, np.float32))
        self.env._prev = np.asarray(action, np.float32).reshape(7)
        self.t += 1

        m1 = state_metrics(self.env.sim)
        tr = self.recatch.update(self.t, m1)
        if tr["recatch"]:
            self.diag["recatch"] = 1
            self.diag["recatch_step"] = tr["recatch_step"]
            self.diag["post_recatch_steps"] = self.recatch.steps_since_recatch
        if tr["stable_hold"]:
            self.diag["stable_hold"] = 1
        if tr.get("lost_after_recatch_latched") or tr.get("lost_after_recatch"):
            self.diag["lost_after_recatch"] = 1

        r_step, bd, bds = combined_step_reward(
            m0,
            m1,
            token_switched=switched,
            post_recatch=bool(tr["recatch"]),
            just_lost_bilateral=bool(tr.get("lost_after_recatch")),
            base_weights=self.pose_weights,
            stab_weights=self.stab_weights,
            use_stabilization=self.use_stabilization,
        )
        for k in ("pose", "contact", "slip", "imbalance", "time", "switch"):
            self.comp_sums[k] += getattr(bd, k)
        for k in ("hold", "calm", "post_slip", "lost_contact", "overgrip"):
            self.stab_sums[k] += getattr(bds, k)
        self.stab_sums["stab_total"] += bds.total

        drop = bool(m1["latched"])
        hand = False
        hand_reason = None
        hand_diag: dict[str, Any] = {}
        st = self.mon.observe(self.env.sim, self.t, tape_ended=False)
        if st["ready"]:
            hand = True
            hand_reason = st["reason"]
            hand_diag = {
                "handoff_time": float(self.env.sim.data.time),
                "handoff_step": int(self.t),
                "e_pos_mm": float(m1["e_pos_mm"]),
                "e_theta_deg": float(m1["theta"]),
                "fn_L": float(m1.get("fn_L") or 0.0),
                "fn_R": float(m1.get("fn_R") or 0.0),
                "min_pad": int(m1.get("min_pad") or 0),
                "bilateral": int(m1.get("bilateral") or 0),
                "dwell_ready": bool(st.get("ready")),
                "r3f_reason": hand_reason,
            }

        truncated = self.t >= self.max_steps and not drop and not hand
        done = drop or hand or truncated

        r_term = 0.0
        s12 = snew = srel = False
        lost_after = False
        timeout_fail = False
        fail_reason = None
        ref = self.start.ref if self.start.kind == "capture" else (self.start.ep or {}).get("ref")
        if done:
            if drop:
                fail_reason = "DROP_BEFORE"
                r_term, bd_t = terminal_reward(
                    drop=True,
                    success_relaxed=False,
                    success_12s=False,
                    success_new=False,
                    lost_after_hand=False,
                    timeout_fail=False,
                    weights=self.pose_weights,
                )
            elif truncated and not hand:
                timeout_fail = True
                fail_reason = "TIMEOUT"
                r_term, bd_t = terminal_reward(
                    drop=False,
                    success_relaxed=False,
                    success_12s=False,
                    success_new=False,
                    lost_after_hand=False,
                    timeout_fail=True,
                    weights=self.pose_weights,
                )
            else:
                ev = evaluate_handoff_with_axis(self.env.sim, ref, stabilize_n=8)
                rel = evaluate_success_relaxed_12s(ev)
                s12 = bool(rel["SUCCESS_12S"])
                snew = bool(rel["SUCCESS_NEW"])
                srel = bool(rel["SUCCESS_RELAXED_12S"])
                reason = str(ev.get("reason") or "")
                lost_after = (not s12) and (
                    "LOST" in reason.upper() or "DROP" in reason.upper() or "GROUND" in reason.upper()
                )
                timeout_fail = (not s12) and not lost_after
                fail_reason = None if srel else (reason or ("POSE_FAIL" if s12 else "NOMINAL_FAIL"))
                r_term, bd_t = terminal_reward(
                    drop=False,
                    success_relaxed=srel,
                    success_12s=s12,
                    success_new=snew,
                    lost_after_hand=lost_after,
                    timeout_fail=timeout_fail,
                    weights=self.pose_weights,
                )
                hand_diag["nominal_reason"] = reason
                hand_diag["pre_pos_mm"] = ev.get("pre_pos_mm")
                hand_diag["pre_theta"] = ev.get("pre_theta")
            self.comp_sums["terminal"] += bd_t.terminal
            r_step += r_term

        if not done:
            obs = self._observe()
            self.obs_seq.append(obs)
            self.prev_seq.append(self.env._prev.copy())
        self._prev_m = m1

        info = {
            "skill": self.start.skill,
            "traj_id": self.start.traj_id,
            "metrics": m1,
            "drop": int(drop),
            "handoff": int(hand),
            "handoff_reason": hand_reason
            if hand
            else ("DROP_BEFORE" if drop else ("TIMEOUT" if truncated else None)),
            "handoff_diag": hand_diag,
            "truncated": int(truncated),
            "SUCCESS_12S": int(s12),
            "SUCCESS_NEW": int(snew),
            "SUCCESS_RELAXED_12S": int(srel),
            "fail_reason": fail_reason,
            "t": self.t,
            "reward_breakdown": {
                "step_dense": float(bd.total),
                "terminal": float(r_term),
                "components": dict(self.comp_sums),
                "stabilization": dict(self.stab_sums),
            },
            "c_diag": dict(self.diag),
            "use_stabilization": int(self.use_stabilization),
        }
        self.last_info = info
        return float(r_step), bool(done), info
