"""Frozen P70/P71 V2 pre-contact controller + optional post-contact overlay.

Pre-contact V2 mechanism is NEVER modified.
Post-contact overlay activates only after obs reports pad contact.
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
class PostRefinement:
    """Bounded observation-only post-contact intervention."""

    name: str = "none"
    kind: str = "none"  # none | continue_seq | contact_preserve | near_ready_hold
    sequence: list[int] = field(default_factory=list)
    max_steps: int = 5
    require_contact: bool = True
    abort_on_contact_loss: bool = True
    # near-ready: only when obs shows bilateral-ish (both contact pads)
    require_bilateral_obs: bool = False


@dataclass
class WrapState:
    v2: CtrlState = field(default_factory=CtrlState)
    post_i: int = 0
    post_active: bool = False
    post_done: bool = False
    saw_contact: bool = False
    forced_post: list[int] = field(default_factory=list)


def select_token(
    *,
    env_t: int,
    obs,
    actor_tok: int,
    st: WrapState,
    v2: Spec,
    post: PostRefinement | None,
) -> tuple[int, WrapState, dict[str, Any]]:
    feat = obs_features(obs)
    dbg: dict[str, Any] = {"contact": feat["any_contact_obs"]}

    if feat["any_contact_obs"]:
        st.saw_contact = True

    # Pre-contact: unchanged V2
    if not st.saw_contact:
        tok, st.v2, d2 = select_v2_token(
            env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
        )
        dbg.update({"phase": "precontact_v2", "v2_phase": st.v2.phase, **{k: d2.get(k) for k in ("forced",)}})
        if d2.get("forced") is not None:
            dbg["forced"] = d2["forced"]
        return int(tok), st, dbg

    # Post-contact overlay
    if post is None or post.kind == "none" or st.post_done:
        tok, st.v2, d2 = select_v2_token(
            env_t=env_t, obs=obs, actor_tok=actor_tok, st=st.v2, spec=v2
        )
        # After contact V2 is usually done → Actor
        dbg["phase"] = "post_v2_or_actor"
        return int(tok), st, dbg

    if post.abort_on_contact_loss and st.post_active and not feat["any_contact_obs"]:
        st.post_done = True
        st.post_active = False
        dbg["phase"] = "post_abort_contact_loss"
        return int(actor_tok), st, dbg

    if post.require_bilateral_obs and not (feat["contact_L"] and feat["contact_R"]):
        # Not yet bilateral — let Actor/V2 act
        dbg["phase"] = "wait_bilateral_obs"
        return int(actor_tok), st, dbg

    if not st.post_active:
        st.post_active = True
        st.post_i = 0

    # Bound by max_steps only (do not invent extra cycles beyond the sequence length
    # unless sequence is shorter — then cycle within the bound).
    if st.post_i < post.max_steps:
        if post.sequence:
            tok = int(post.sequence[st.post_i % len(post.sequence)])
        else:
            tok = int(actor_tok)
        st.post_i += 1
        st.forced_post.append(tok)
        dbg["forced"] = tok
        dbg["phase"] = f"post_{post.kind}"
        if st.post_i >= post.max_steps:
            st.post_done = True
            st.post_active = False
        return tok, st, dbg

    st.post_done = True
    st.post_active = False
    dbg["phase"] = "post_resume_actor"
    return int(actor_tok), st, dbg
