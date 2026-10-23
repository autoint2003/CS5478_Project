"""Historical Plan A mechanism audit entry. Implementation lives in evaluation."""

from evaluation.plan_a_mechanism import *  # noqa: F403
from evaluation.plan_a_mechanism import (  # noqa: F401
    ExpertSpec,
    analyze_mechanism,
    audit_action_legality,
    decode_token,
    load_vocab,
    make_negative_actions,
    rollout_traced,
    sample_kinematics,
    transfer_state_to_ground,
)
