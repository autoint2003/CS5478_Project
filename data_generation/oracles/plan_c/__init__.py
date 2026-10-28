"""Plan C offline oracle: surface-aware dynamic recatch on GROUND_SCENE.

Privileged contact-surface geometry is for offline search/ranking only.
Runtime policy inputs remain the frozen 39D observation and legal 7D actions.
"""

from data_generation.oracles.plan_c.contact_surface import (
    classify_contacts,
    score_catch_out,
    surface_score,
)
from data_generation.oracles.plan_c.dynamic_catch import (
    CatchParams,
    catch_param_grid,
    dynamic_catch_sequence,
    propose_catch_interceptions,
)
from data_generation.oracles.plan_c.expert_search import (
    build_physical_root,
    diverse_impact_configs,
    exact_replay_3of3,
    lift_and_validate,
    pack_expert,
    physical_root_fingerprint,
    run_stage1_catch_and_lift,
    search_catch,
)
from data_generation.oracles.plan_c.catch_stabilization import (
    fast_world_lift_sequence,
    slow_lift_sequence,
)

__all__ = [
    "CatchParams",
    "build_physical_root",
    "catch_param_grid",
    "classify_contacts",
    "diverse_impact_configs",
    "dynamic_catch_sequence",
    "exact_replay_3of3",
    "fast_world_lift_sequence",
    "lift_and_validate",
    "pack_expert",
    "physical_root_fingerprint",
    "propose_catch_interceptions",
    "run_stage1_catch_and_lift",
    "score_catch_out",
    "search_catch",
    "slow_lift_sequence",
    "surface_score",
]
