"""P71 controllers: V2 reference + V3/V4/V5 refinements.

All decisions use Actor-available observations only (no privileged e_pos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from experiments.precision.p70_geometry_to_r3f.obs_proxy import alignment_proxy_deployable, obs_features
from experiments.precision.p71_contact_safe_alignment.paths import (
    GRIP_TOK,
    TOKEN_56,
    V2_GRIP_STEPS,
    V2_MAX_ACTIVATE,
    V2_MAX_APPROACH,
    V2_MIN_APPROACH,
    V2_POST,
    V2_READY,
)


@dataclass
class CtrlState:
    phase: str = "idle"  # idle|align|approach|grip|post|done
    align_steps: int = 0
    approach_steps: int = 0
    grip_steps: int = 0
    post_steps: int = 0
    forced: list[int] = field(default_factory=list)
    activated: bool = False
    abort_reason: str | None = None
    switch_reason: str | None = None


@dataclass
class Spec:
    name: str
    kind: str  # v2 | v3_ext | v4_safe | v5_adapt
    max_activate_step: int = V2_MAX_ACTIVATE
    require_no_contact: bool = True
    # V2 core
    max_approach: int = V2_MAX_APPROACH
    min_approach: int = V2_MIN_APPROACH
    grip_steps: int = V2_GRIP_STEPS
    post_approach: int = V2_POST
    ready_score_thresh: float = V2_READY
    # V3/V4 extended alignment before V2 approach/grip
    max_align: int = 0
    min_align: int = 0
    align_pool: list[int] = field(default_factory=list)
    # V4: abort align on rising contact risk
    contact_safe: bool = False
    # V5: use ready_score / aperture history for transition
    adaptive_transition: bool = False
    approach_pool: list[int] = field(default_factory=lambda: [TOKEN_56, 50, 71])


def make_specs(align_pool: list[int]) -> dict[str, Spec]:
    pool = list(align_pool) if align_pool else [50, 71, 40]
    return {
        "V2": Spec(
            name="V2_p70_ref",
            kind="v2",
            max_align=0,
            approach_pool=[TOKEN_56, 50, 71],
        ),
        "V3": Spec(
            name="V3_extended_align",
            kind="v3_ext",
            max_align=4,
            min_align=1,
            align_pool=pool,
            approach_pool=[50, 71] + [t for t in pool if t not in (50, 71)][:2],  # avoid 56 in approach if possible
        ),
        "V4": Spec(
            name="V4_contact_safe",
            kind="v4_safe",
            max_align=4,
            min_align=1,
            align_pool=pool,
            contact_safe=True,
            approach_pool=[50, 71] + [t for t in pool if t not in (50, 71)][:2],
        ),
        "V5": Spec(
            name="V5_adaptive_transition",
            kind="v5_adapt",
            max_align=3,
            min_align=1,
            align_pool=pool,
            contact_safe=True,
            adaptive_transition=True,
            approach_pool=[50, 71] + pool[:2],
            ready_score_thresh=-0.02,
            max_approach=8,
            min_approach=2,
        ),
    }


def _elig(t: int, feat: dict, spec: Spec) -> bool:
    if int(t) > int(spec.max_activate_step):
        return False
    if spec.require_no_contact and feat.get("any_contact_obs"):
        return False
    return True


def _imminent_contact(feat: dict, proxy: dict) -> bool:
    """Deployable contact-risk proxy (no privileged geometry).

    Avoid treating any nonzero e_hat as contact — estimate_valid can be noisy
    and would abort alignment on the first tactile flicker.
    """
    if feat.get("any_contact_obs"):
        return True
    # Fast approach while closing fingers → likely imminent pad contact
    if float(feat.get("hand_speed") or 0) > 0.35 and float(feat.get("ap_dot") or 0) < -0.05:
        return True
    return False


def _pick_align(spec: Spec, step: int) -> int:
    pool = spec.align_pool or [50, 71]
    return int(pool[step % len(pool)])


def _pick_approach(spec: Spec, step: int) -> int:
    pool = spec.approach_pool or [TOKEN_56, 50, 71]
    return int(pool[step % len(pool)])


def select_token(
    *,
    env_t: int,
    obs,
    actor_tok: int,
    st: CtrlState,
    spec: Spec,
) -> tuple[int, CtrlState, dict[str, Any]]:
    feat = obs_features(obs)
    proxy = alignment_proxy_deployable(obs)
    dbg: dict[str, Any] = {"phase": st.phase, "ready": proxy["ready_score"], "feat_contact": feat["any_contact_obs"]}

    if not _elig(env_t, feat, spec) and st.phase == "idle":
        return int(actor_tok), st, dbg

    # Activate
    if st.phase == "idle":
        if not _elig(env_t, feat, spec):
            return int(actor_tok), st, dbg
        st.activated = True
        if spec.max_align > 0:
            st.phase = "align"
        else:
            st.phase = "approach"

    # --- ALIGN phase (V3/V4/V5) ---
    if st.phase == "align":
        if feat["any_contact_obs"] or (spec.contact_safe and _imminent_contact(feat, proxy)):
            # Fall through to V2 approach/grip logic or Actor
            st.switch_reason = "contact_safe_abort_align"
            st.phase = "approach" if not feat["any_contact_obs"] else "done"
            if st.phase == "done":
                st.abort_reason = "contact_during_align"
                return int(actor_tok), st, dbg
        else:
            end_align = False
            if st.align_steps >= spec.max_align:
                end_align = True
                st.switch_reason = "max_align"
            elif st.align_steps >= spec.min_align and spec.adaptive_transition:
                if proxy["ready_score"] >= spec.ready_score_thresh:
                    end_align = True
                    st.switch_reason = "ready_after_align"
            if end_align:
                st.phase = "approach"
            else:
                tok = _pick_align(spec, st.align_steps)
                st.align_steps += 1
                st.forced.append(tok)
                dbg["forced"] = tok
                return tok, st, dbg

    # --- APPROACH (V2 core, shared) ---
    if st.phase == "approach":
        if feat["any_contact_obs"]:
            st.phase = "done"
            st.abort_reason = "contact_during_approach"
            return int(actor_tok), st, dbg
        switch = False
        if st.approach_steps >= spec.max_approach:
            switch = True
            st.switch_reason = st.switch_reason or "max_approach"
        elif st.approach_steps >= spec.min_approach:
            if proxy["ready_score"] >= spec.ready_score_thresh:
                switch = True
                st.switch_reason = st.switch_reason or "ready_score"
            elif spec.adaptive_transition and st.approach_steps >= spec.min_approach + 1:
                # slower hand → grip
                if float(feat.get("hand_speed") or 0) < 0.12:
                    switch = True
                    st.switch_reason = st.switch_reason or "slow_hand"
        if switch:
            st.phase = "grip"
        else:
            tok = _pick_approach(spec, st.approach_steps)
            st.approach_steps += 1
            st.forced.append(tok)
            dbg["forced"] = tok
            return tok, st, dbg

    if st.phase == "grip":
        if feat["any_contact_obs"]:
            st.phase = "done"
            st.abort_reason = "contact_at_grip"
            return int(actor_tok), st, dbg
        if st.grip_steps < spec.grip_steps:
            tok = int(GRIP_TOK)
            st.grip_steps += 1
            st.forced.append(tok)
            dbg["forced"] = tok
            return tok, st, dbg
        st.phase = "post"

    if st.phase == "post":
        if feat["any_contact_obs"]:
            st.phase = "done"
            st.abort_reason = "contact_in_post"
            return int(actor_tok), st, dbg
        if st.post_steps < spec.post_approach:
            tok = _pick_approach(spec, st.post_steps + 1)
            st.post_steps += 1
            st.forced.append(tok)
            dbg["forced"] = tok
            return tok, st, dbg
        st.phase = "done"
        st.abort_reason = "sequence_complete"

    return int(actor_tok), st, dbg
