"""B recovery env for P50 PPO: M111 discrete actions + R3F terminal + SUCCESS_12S."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral
from data_generation.oracles.precision.residuals import measure_errors
from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline
from experiments.precision.p50_ppo_reward_redesign.reward import (
    RewardWeights,
    step_reward,
    terminal_reward,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action
from models.transformer_policy import DT_POLICY


def state_metrics(sim) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    pb = pad_bilateral(sim)
    err = measure_errors(sim)
    return {
        "e_pos_mm": float(m["e_pos_mm"]),
        "theta": float(m["e_theta_deg"]),
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "min_pad": int(min(pb["nL"], pb["nR"])),
        "asym": int(abs(int(pb["nL"]) - int(pb["nR"]))),
        "vrel": float(err.get("v_rel_norm") or 0.0),
        "fn_L": float(err.get("fn_L") or 0.0),
        "fn_R": float(err.get("fn_R") or 0.0),
        "obj_z": float(m.get("obj_z") or 0.0),
        "latched": int(latched(sim)),
    }


@dataclass
class BStart:
    ic_id: str
    family: str
    capture: dict
    ref: Any


class BRecoveryEnv:
    """Policy acts until DROP / R3F ready / max_steps; SUCCESS_12S from nominal evaluate only."""

    def __init__(
        self,
        book,
        rho,
        *,
        max_steps: int = 100,
        t_max: int = 3100,
        weights: RewardWeights | None = None,
    ):
        self.book = book
        self.rho = rho
        self.max_steps = int(max_steps)
        self.t_max = int(t_max)
        self.weights = weights or RewardWeights()
        self.env = make_env()
        self.obs_seq: list = []
        self.prev_seq: list = []
        self.t = 0
        self.prev_tok = None
        self.start: BStart | None = None
        self.mon: R3FOnline | None = None
        self._prev_m: dict | None = None
        self.last_info: dict = {}
        self.comp_sums = {
            k: 0.0 for k in ("progress", "contact", "slip", "imbalance", "time", "switch", "terminal")
        }

    def _observe(self) -> np.ndarray:
        return np.asarray(
            observe_airborne(
                self.env.sim.model, self.env.sim.data, self.env.sim.ids, self.env.sim.fsm, self.env._hist
            ),
            np.float32,
        )

    def reset(self, start: BStart) -> np.ndarray:
        self.start = start
        self.t = 0
        self.prev_tok = None
        self.comp_sums = {k: 0.0 for k in self.comp_sums}
        restore_capture(self.env, start.capture)
        self.env._hist = ObservableObsState(DT_POLICY)
        self.env._prev = np.zeros(7, np.float32)
        self.mon = R3FOnline.create(t_max=self.t_max)
        self.obs_seq = []
        self.prev_seq = []
        obs = self._observe()
        self.obs_seq.append(obs)
        self.prev_seq.append(self.env._prev.copy())
        self._prev_m = state_metrics(self.env.sim)
        return obs

    def step(self, token: int, resid: np.ndarray) -> tuple[float, bool, dict]:
        assert self.start is not None and self.mon is not None and self._prev_m is not None
        m0 = self._prev_m
        tok = int(token)
        # R3F decides handoff — never decode HANDOFF as a control vector
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
        r_step, bd = step_reward(m0, m1, token_switched=switched, weights=self.weights)
        for k in ("progress", "contact", "slip", "imbalance", "time", "switch"):
            self.comp_sums[k] += getattr(bd, k)

        drop = bool(m1["latched"])
        hand = False
        hand_reason = None
        st = self.mon.observe(self.env.sim, self.t, tape_ended=False)
        if st["ready"]:
            hand = True
            hand_reason = st["reason"]

        truncated = self.t >= self.max_steps and not drop and not hand
        done = drop or hand or truncated

        r_term = 0.0
        s12 = snew = False
        lost_after = False
        timeout_fail = False
        if done:
            if drop:
                r_term, bd_t = terminal_reward(
                    drop=True,
                    success_12s=False,
                    success_new=False,
                    lost_after_hand=False,
                    timeout_fail=False,
                    weights=self.weights,
                )
            elif truncated and not hand:
                # Did not reach R3F — timeout failure without nominal 12s (not a handoff)
                timeout_fail = True
                r_term, bd_t = terminal_reward(
                    drop=False,
                    success_12s=False,
                    success_new=False,
                    lost_after_hand=False,
                    timeout_fail=True,
                    weights=self.weights,
                )
            else:
                # R3F ready → nominal continuation inside evaluator (not policy actions)
                ev = evaluate_handoff_with_axis(self.env.sim, self.start.ref, stabilize_n=8)
                s12 = bool(ev.get("SUCCESS_12S"))
                snew = bool(ev.get("SUCCESS_NEW"))
                reason = str(ev.get("reason") or "")
                lost_after = (not s12) and ("LOST" in reason.upper() or "GRASP" in reason.upper())
                timeout_fail = (not s12) and not lost_after
                r_term, bd_t = terminal_reward(
                    drop=False,
                    success_12s=s12,
                    success_new=snew,
                    lost_after_hand=lost_after,
                    timeout_fail=timeout_fail,
                    weights=self.weights,
                )
            self.comp_sums["terminal"] += bd_t.terminal
            r_step += r_term

        if not done:
            obs = self._observe()
            self.obs_seq.append(obs)
            self.prev_seq.append(self.env._prev.copy())
        self._prev_m = m1

        info = {
            "metrics": m1,
            "drop": int(drop),
            "handoff": int(hand),
            "handoff_reason": hand_reason if hand else ("DROP_BEFORE" if drop else ("TIMEOUT" if truncated else None)),
            "truncated": int(truncated),
            "true_terminal": int(drop or hand or (truncated and True)),
            "SUCCESS_12S": int(s12),
            "SUCCESS_NEW": int(snew),
            "t": self.t,
            "reward_breakdown": {
                "step_dense": float(bd.total),
                "terminal": float(r_term),
                "components": dict(self.comp_sums),
            },
        }
        self.last_info = info
        return float(r_step), bool(done), info
