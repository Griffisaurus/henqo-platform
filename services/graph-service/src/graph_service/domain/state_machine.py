"""
State machine enforcement for IR entity lifecycle transitions.

Legal transitions and approval-gate rules from state-machine-workflow-spec.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class IllegalTransitionError(Exception):
    def __init__(self, entity_type: str, from_state: str, to_state: str) -> None:
        self.entity_type = entity_type
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"{entity_type}: transition {from_state!r} → {to_state!r} is not permitted"
        )


class MissingApprovalError(Exception):
    def __init__(self, entity_type: str, from_state: str, to_state: str) -> None:
        self.entity_type = entity_type
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"{entity_type}: transition {from_state!r} → {to_state!r} requires approver_id"
        )


# ---------------------------------------------------------------------------
# Transition registry
# ---------------------------------------------------------------------------

class _Transition(NamedTuple):
    requires_approval: bool = False


# Maps (entity_type, from_state, to_state) → _Transition
_TRANSITIONS: dict[tuple[str, str, str], _Transition] = {}


def _reg(entity_type: str, transitions: list[tuple[str, str] | tuple[str, str, bool]]) -> None:
    for t in transitions:
        from_s, to_s = t[0], t[1]
        requires_approval = t[2] if len(t) == 3 else False  # type: ignore[misc]
        _TRANSITIONS[(entity_type, from_s, to_s)] = _Transition(
            requires_approval=requires_approval
        )


# Characteristic
_reg("Characteristic", [
    ("unverified",           "surrogate_estimated"),
    ("unverified",           "simulation_validated"),
    ("unverified",           "unresolved"),
    ("surrogate_estimated",  "simulation_validated"),
    ("surrogate_estimated",  "unresolved"),
    ("simulation_validated", "inspection_confirmed"),
    ("simulation_validated", "surrogate_estimated"),
    ("inspection_confirmed", "released",              True),   # approval-gated
    ("inspection_confirmed", "simulation_validated"),
    ("released",             "superseded"),
    ("unresolved",           "surrogate_estimated"),
    ("unresolved",           "simulation_validated"),
])

# Prediction
_reg("Prediction", [
    ("created",   "used"),
    ("created",   "blocked"),
    ("created",   "abstained"),
    ("created",   "superseded"),
    ("blocked",   "created"),
    ("used",      "superseded"),
    ("abstained", "superseded"),
])

# SimulationCase
_reg("SimulationCase", [
    ("queued",    "running"),
    ("queued",    "failed"),
    ("running",   "completed"),
    ("running",   "failed"),
    ("completed", "validated"),
    ("completed", "invalidated"),
    ("validated", "invalidated"),
    ("failed",    "queued"),
])

# ComponentRevision
_reg("ComponentRevision", [
    ("in_design",                      "manufacturing_review_requested"),
    ("manufacturing_review_requested", "manufacturing_reviewed"),
    ("manufacturing_review_requested", "in_design"),           # recall
    ("manufacturing_reviewed",         "released",             True),  # approval-gated
    ("manufacturing_reviewed",         "in_design"),           # revision restart
    ("released",                       "obsolete"),
    ("in_design",                      "obsolete"),
])

# AssemblyRevision (mirrors ComponentRevision)
_reg("AssemblyRevision", [
    ("in_design",                      "manufacturing_review_requested"),
    ("manufacturing_review_requested", "manufacturing_reviewed"),
    ("manufacturing_review_requested", "in_design"),
    ("manufacturing_reviewed",         "released",             True),
    ("manufacturing_reviewed",         "in_design"),
    ("released",                       "obsolete"),
    ("in_design",                      "obsolete"),
])

# ReleaseManifest
_reg("ReleaseManifest", [
    ("draft",              "pending_signatures"),
    ("pending_signatures", "active",              True),   # approval-gated (dual sign-off)
    ("pending_signatures", "draft"),              # rejection → back to draft
    ("active",             "superseded"),
])

# SurrogateModel (from surrogate-model-lifecycle-spec.md)
_reg("SurrogateModel", [
    ("training",       "benchmarking"),
    ("training",       "failed"),
    ("benchmarking",   "staged"),
    ("benchmarking",   "failed"),
    ("staged",         "production",   True),   # approval-gated
    ("staged",         "deprecated"),
    ("production",     "deprecated"),
    ("deprecated",     "retired"),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_transition(
    entity_type: str,
    current_state: str,
    new_state: str,
    approver_id: str | None = None,
) -> None:
    """
    Validate a state transition.

    Raises IllegalTransitionError if the transition is not in the state machine.
    Raises MissingApprovalError if the transition requires an approver but none provided.
    """
    key = (entity_type, current_state, new_state)
    transition = _TRANSITIONS.get(key)

    if transition is None:
        raise IllegalTransitionError(entity_type, current_state, new_state)

    if transition.requires_approval and not approver_id:
        raise MissingApprovalError(entity_type, current_state, new_state)


def legal_next_states(entity_type: str, current_state: str) -> list[str]:
    """Return all states reachable from current_state for entity_type."""
    return [
        to_s
        for (et, from_s, to_s) in _TRANSITIONS
        if et == entity_type and from_s == current_state
    ]


def is_approval_gated(entity_type: str, from_state: str, to_state: str) -> bool:
    t = _TRANSITIONS.get((entity_type, from_state, to_state))
    return t is not None and t.requires_approval
