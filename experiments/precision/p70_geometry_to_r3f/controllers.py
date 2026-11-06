"""Closed-loop sequence variants (deterministic overlays on Actor).

All decisions use Actor-available observations only.
Privileged e_pos is never a controller input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.precision.p70_geometry_to_r3f.obs_proxy import alignment_proxy_deployable, obs_features
from experiments.precision.p70_geometry_to_r3f.paths import APPROACH_TOKS, GRIP_TOK, P69_SEQ


@dataclass
class ControllerState:
    phase: str = "idle"  # idle | approach | grip | post | done
    approach_steps: int = 0
    grip_steps: int = 0
    post_steps: int = 0
    forced_tokens: list[int] = field(default_factory=list)
    activated: bool = False
    eligible_seen: bool = False
    abort_reason: str | None = None


@dataclass
class VariantSpec:
    name: str
    kind: str  # none | fixed | adaptive_approach | contact_aware
    max_activate_step: int = 7
    require_no_contact: bool = True
    max_approach: int = 6
    min_approach: int = 1
    grip_steps: int = 1
    post_approach: int = 2
    ready_score_thresh: float = -0.05
    sequence: list[int] = field(default_factory=lambda: list(P69_SEQ))


VARIANTS: dict[str, VariantSpec] = {
    "V0_baseline": VariantSpec(name="V0_baseline", kind="none"),
    "V1_fixed": VariantSpec(name="V1_fixed", kind="fixed", sequence=list(P69_SEQ)),
    "V2_adaptive_approach": VariantSpec(
        name="V2_adaptive_approach",
        kind="adaptive_approach",
        max_approach=8,
        min_approach=2,
        grip_steps=1,
        post_approach=2,
        ready_score_thresh=-0.05,
    ),
    "V3_contact_aware": VariantSpec(
        name="V3_contact_aware",
        kind="contact_aware",
        max_approach=6,
        min_approach=1,
        grip_steps=1,
        post_approach=2,
    ),
}


def eligibility(env_t: int, feat: dict, spec: VariantSpec) -> bool:
    if spec.kind == "none":
        return False
    if int(env_t) > int(spec.max_activate_step):
        return False
    if spec.require_no_contact and feat.get("any_contact_obs"):
        return False
    return True


def select_token(
    *,
    env_t: int,
    obs,
    actor_tok: int,
    st: ControllerState,
    spec: VariantSpec,
) -> tuple[int, ControllerState, dict[str, Any]]:
    """Return (token, updated_state, debug)."""
    feat = obs_features(obs)
    proxy = alignment_proxy_deployable(obs)
    dbg: dict[str, Any] = {"feat": feat, "proxy_ready": proxy["ready_score"], "phase": st.phase}

    if spec.kind == "none":
        return int(actor_tok), st, dbg

    elig = eligibility(env_t, feat, spec)
    if elig:
        st.eligible_seen = True

    # Fixed P69 sequence
    if spec.kind == "fixed":
        if st.phase == "idle" and elig:
            st.phase = "fixed"
            st.activated = True
            st.forced_tokens = []
        if st.phase == "fixed":
            i = len(st.forced_tokens)
            if i < len(spec.sequence) and not feat["any_contact_obs"]:
                tok = int(spec.sequence[i])
                st.forced_tokens.append(tok)
                dbg["forced"] = tok
                return tok, st, dbg
            st.phase = "done"
            st.abort_reason = "seq_done_or_contact"
        return int(actor_tok), st, dbg

    # Shared structure: APPROACH* → GRIP → APPROACH*
    if spec.kind in ("adaptive_approach", "contact_aware"):
        if st.phase == "idle":
            if elig:
                st.phase = "approach"
                st.activated = True
            else:
                return int(actor_tok), st, dbg

        if st.phase == "approach":
            # Abort to Actor if unexpected early contact before min_approach (contact-aware)
            if feat["any_contact_obs"]:
                if spec.kind == "contact_aware":
                    st.phase = "done"
                    st.abort_reason = "contact_during_approach_fallback_actor"
                    return int(actor_tok), st, dbg
                # adaptive: if contact, skip grip sequence → Actor
                st.phase = "done"
                st.abort_reason = "contact_during_approach"
                return int(actor_tok), st, dbg

            switch = False
            if st.approach_steps >= spec.max_approach:
                switch = True
                dbg["switch_reason"] = "max_approach"
            elif st.approach_steps >= spec.min_approach:
                if spec.kind == "adaptive_approach":
                    if proxy["ready_score"] >= spec.ready_score_thresh:
                        switch = True
                        dbg["switch_reason"] = "ready_score"
                # contact_aware: only switch on max/min schedule unless contact (handled above)
                elif spec.kind == "contact_aware" and st.approach_steps >= max(spec.min_approach, 3):
                    # hold APPROACH until max; grip only when still no contact and enough approach
                    if st.approach_steps >= 4:
                        switch = True
                        dbg["switch_reason"] = "scheduled_pre_grip"

            if switch:
                st.phase = "grip"
            else:
                tok = int(APPROACH_TOKS[st.approach_steps % len(APPROACH_TOKS)])
                st.approach_steps += 1
                st.forced_tokens.append(tok)
                dbg["forced"] = tok
                return tok, st, dbg

        if st.phase == "grip":
            if feat["any_contact_obs"] and spec.kind == "contact_aware":
                st.phase = "done"
                st.abort_reason = "contact_at_grip_fallback"
                return int(actor_tok), st, dbg
            if st.grip_steps < spec.grip_steps:
                tok = int(GRIP_TOK)
                st.grip_steps += 1
                st.forced_tokens.append(tok)
                dbg["forced"] = tok
                return tok, st, dbg
            st.phase = "post"

        if st.phase == "post":
            if feat["any_contact_obs"]:
                st.phase = "done"
                st.abort_reason = "contact_in_post"
                return int(actor_tok), st, dbg
            if st.post_steps < spec.post_approach:
                tok = int(APPROACH_TOKS[(st.post_steps + 1) % len(APPROACH_TOKS)])
                st.post_steps += 1
                st.forced_tokens.append(tok)
                dbg["forced"] = tok
                return tok, st, dbg
            st.phase = "done"
            st.abort_reason = "sequence_complete"

        return int(actor_tok), st, dbg

    return int(actor_tok), st, dbg
