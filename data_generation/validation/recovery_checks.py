"""Official physical checks used when a generated tape is accepted."""

from envs.outcomes.horizon_outcome import finish_recovery, nominal_verify, recovery_failed
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, terminal_reason

__all__ = [
    "apply_handoff",
    "finish_recovery",
    "latched",
    "nominal_step",
    "nominal_verify",
    "recovery_failed",
    "terminal_reason",
]
