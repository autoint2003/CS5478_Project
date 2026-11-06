"""P73 event-conditioned post-contact overlay on frozen P70/P71 V2.

Pre-contact V2 is never modified.
Activate only after Actor-observable pad contact.
Stop on: obs bilateral | max_steps | contact loss → resume V2/Actor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.precision.p70_geometry_to_r3f.obs_proxy import obs_features
from experiments.precision.p71_contact_safe_alignment.controllers import (
    CtrlState,
    Spec,
    make_specs,
    select_token as select_v2_token,
)


def v2_spec() -> Spec:
    return make_specs([])["V2"]


@dataclass
class EventRule:
    """Bounded observation-only first-contact intervention."""

    name: str = "none"
    sequence: list[int] = field(default_factory=list)
    max_steps: int = 5
    # Terminate when both pads report contact (deployable proxy for bilateral)
    stop_on_obs_bilateral: bool = True
    abort_on_contact_loss: bool = True
    # Optional: only activate if still unilateral (obs) — skip if already bilateral
    require_unilateral_start: bool = True


@dataclass
class WrapState:
    v2: CtrlState = field(default_factory=CtrlState)
    post_i: int = 0
    post_active: bool = False
    post_done: bool = False
    saw_contact: bool = False
    forced: list[int] = field(default_factory=list)
    stop_reason: str | None = None


def select_token(
    *,
    env_t: int,
    obs,
    actor_tok: int,
    st: WrapState,
    v2: Spec,
    rule: EventRule | None,
) -> tuple[int, WrapState, dict[str, Any]]:
    feat = obs_features(obs)
    dbg: dict[str, Any] = {
        "contact": feat["any_contact_obs"],
        "obs_bilateral": bool(feat["contact_L"] and feat["contact_R"]),
    }

    if feat["any_contact_obs"]:
        st.saw_contact = True

    # Pre-contact: unchanged V2
    if not st.saw_contact:
        tok, st.v2, d2 = select_v2_token(
            env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
        )
        dbg["phase"] = "precontact_v2"
        if d2.get("forced") is not None:
            dbg["forced"] = d2["forced"]
        return int(tok), st, dbg

    # No rule / finished → V2 (usually Actor after contact)
    if rule is None or rule.name in ("none", "baseline") or st.post_done or not rule.sequence:
        tok, st.v2, _d2 = select_v2_token(
            env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
        )
        dbg["phase"] = "post_v2_or_actor"
        return int(tok), st, dbg

    # Contact-loss abort
    if rule.abort_on_contact_loss and st.post_active and not feat["any_contact_obs"]:
        st.post_done = True
        st.post_active = False
        st.stop_reason = "contact_loss"
        dbg["phase"] = "abort_contact_loss"
        return int(actor_tok), st, dbg

    # Obs bilateral stop (deployable termination)
    if rule.stop_on_obs_bilateral and st.post_active and feat["contact_L"] and feat["contact_R"]:
        st.post_done = True
        st.post_active = False
        st.stop_reason = "obs_bilateral"
        dbg["phase"] = "stop_obs_bilateral"
        tok, st.v2, _ = select_v2_token(
            env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
        )
        return int(tok), st, dbg

    # Activate
    if not st.post_active:
        if rule.require_unilateral_start and feat["contact_L"] and feat["contact_R"]:
            # Already bilateral at first contact — skip overlay
            st.post_done = True
            st.stop_reason = "already_bilateral"
            dbg["phase"] = "skip_already_bilateral"
            tok, st.v2, _ = select_v2_token(
                env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
            )
            return int(tok), st, dbg
        st.post_active = True
        st.post_i = 0

    if st.post_i < rule.max_steps:
        tok = int(rule.sequence[st.post_i % len(rule.sequence)])
        st.post_i += 1
        st.forced.append(tok)
        dbg["forced"] = tok
        dbg["phase"] = f"p73_{rule.name}"
        if st.post_i >= rule.max_steps:
            st.post_done = True
            st.post_active = False
            st.stop_reason = "max_steps"
        return tok, st, dbg

    st.post_done = True
    st.post_active = False
    st.stop_reason = st.stop_reason or "max_steps"
    dbg["phase"] = "resume_v2"
    return int(actor_tok), st, dbg
